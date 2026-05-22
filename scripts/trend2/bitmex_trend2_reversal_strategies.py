import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from numba import njit

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hftbacktest import (  # noqa: E402
    ALL_ASSETS,
    BUY_EVENT,
    SELL_EVENT,
    BacktestAsset,
    GTX,
    LIMIT,
    HashMapMarketDepthBacktest,
    Recorder,
)
from hftbacktest.order import IOC  # noqa: E402

from bitmex_single_market_mm_backtest import (  # noqa: E402
    BITMEX_EXCHANGE,
    BITMEX_LOT_SIZE,
    BITMEX_ORDER_ENTRY_LATENCY_NS,
    BITMEX_ORDER_QTY,
    BITMEX_ORDER_RESPONSE_LATENCY_NS,
    BITMEX_TICK_SIZE,
    CSV_DIR,
    NPZ_DIR,
    RESULT_DIR,
    convert_bitmex,
    download_file,
    end_close_ts_ns,
    tardis_key,
)


STRATEGY_MICROPRICE_SNAPBACK = 1
STRATEGY_IMPACT_EXHAUSTION = 2
STRATEGY_QUEUE_REFILL_REVERTER = 3
STRATEGY_WIDE_SPREAD_SNAPBACK = 4

STRATEGY_SLUGS = {
    STRATEGY_MICROPRICE_SNAPBACK: "microprice_snapback",
    STRATEGY_IMPACT_EXHAUSTION: "impact_exhaustion",
    STRATEGY_QUEUE_REFILL_REVERTER: "queue_refill_reverter",
    STRATEGY_WIDE_SPREAD_SNAPBACK: "wide_spread_snapback",
}

BITMEX_SYMBOL = "XBTUSDT"
BITMEX_CONTRACT_SIZE = 0.000001
BITMEX_IS_INVERSE = False

ORDER_UPDATE_INTERVAL_NS = 20_000_000
HISTORY_RECORD_INTERVAL_NS = 20_000_000
SIGNAL_HISTORY_LEN = 8192
ORDER_QTY = BITMEX_ORDER_QTY
MAX_POSITION_CONTRACTS = 300.0

MAKER_FEE_RATE = -0.0002
TAKER_FEE_RATE = 0.0001
EXCHANGE_MODEL = "no_partial"
RESULT_TAG = ""
EXECUTION_MODE_PASSIVE_MAKER = 0
EXECUTION_MODE_TAKER_TAKER = 1
EXECUTION_MODE = EXECUTION_MODE_TAKER_TAKER
INVERT_SIGNAL = False

SLOW_WINDOW_NS = 350_000_000
FAST_WINDOW_NS = 100_000_000
MIN_HOLD_NS = 250_000_000
MAX_HOLD_NS = 3_000_000_000
COOLDOWN_NS = 250_000_000
ENTRY_TTL_NS = 500_000_000
EXIT_TTL_NS = 600_000_000
MIN_AMEND_TICKS = 1.0

DEVIATION_BPS = 0.018
SHOCK_MOVE_BPS = 0.45
FAST_RELIEF_BPS = 0.25
MAX_FLOW_CONTINUATION = 0.35
MAX_MICRO_CONTINUATION_BPS = 0.08
MIN_REFILL_RATIO = 0.95
SPREAD_SHOCK_BPS = 0.08
MAX_SPREAD_BPS = 6.0
TARGET_PROFIT_BPS = 3.0
STOP_LOSS_BPS = 2.0
TAKER_SLIPPAGE_BPS = 0.8
FLOW_DECAY = 0.90
FLOW_MIN_QTY = 200.0


@njit
def ceil_to_tick(px, tick_size):
    return math.ceil(px / tick_size) * tick_size


@njit
def floor_to_tick(px, tick_size):
    return math.floor(px / tick_size) * tick_size


@njit
def ratio_minus_one_bps(numerator, denominator):
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return 0.0
    return (numerator / denominator - 1.0) * 10_000.0


@njit
def current_mid(depth):
    return (depth.best_bid + depth.best_ask) / 2.0


@njit
def spread_bps(depth):
    return ratio_minus_one_bps(depth.best_ask, depth.best_bid)


@njit
def microprice(depth):
    total = depth.best_bid_qty + depth.best_ask_qty
    if total <= 0:
        return current_mid(depth)
    return (depth.best_ask * depth.best_bid_qty + depth.best_bid * depth.best_ask_qty) / total


@njit
def microprice_bps(depth):
    return ratio_minus_one_bps(microprice(depth), current_mid(depth))


@njit
def queue_imbalance(depth):
    total = depth.best_bid_qty + depth.best_ask_qty
    if total <= 0:
        return 0.0
    return (depth.best_bid_qty - depth.best_ask_qty) / total


@njit
def bitmex_base_from_contracts(contracts, price):
    if BITMEX_IS_INVERSE:
        if price <= 0:
            return 0.0
        return contracts * BITMEX_CONTRACT_SIZE / price
    return contracts * BITMEX_CONTRACT_SIZE


@njit
def bitmex_equity_usdt(hbt):
    depth = hbt.depth(0)
    mid = current_mid(depth)
    state = hbt.state_values(0)
    if BITMEX_IS_INVERSE:
        equity_btc = state.balance + state.position * BITMEX_CONTRACT_SIZE / mid - state.fee
        return equity_btc * mid
    return state.balance + state.position * mid * BITMEX_CONTRACT_SIZE - state.fee


@njit
def update_market_flow(hbt, flow):
    trades = hbt.last_trades(0)
    buy_qty = 0.0
    sell_qty = 0.0
    buy_count = 0.0
    sell_count = 0.0
    for trade in trades:
        if (trade.ev & BUY_EVENT) == BUY_EVENT:
            buy_qty += trade.qty
            buy_count += 1.0
        elif (trade.ev & SELL_EVENT) == SELL_EVENT:
            sell_qty += trade.qty
            sell_count += 1.0
    hbt.clear_last_trades(0)
    flow[0] = flow[0] * FLOW_DECAY + buy_qty
    flow[1] = flow[1] * FLOW_DECAY + sell_qty
    flow[2] += buy_qty
    flow[3] += sell_qty
    flow[4] += buy_count
    flow[5] += sell_count


@njit
def flow_score(flow):
    total = flow[0] + flow[1]
    if total < FLOW_MIN_QTY:
        return 0.0
    return (flow[0] - flow[1]) / total


@njit
def record_history(hbt, hist_ts, hist_mid, hist_fair, hist_bid_qty, hist_ask_qty, write_idx, count):
    depth = hbt.depth(0)
    if depth.best_bid <= 0 or depth.best_ask <= 0:
        return write_idx, count
    if count > 0:
        prev_idx = (write_idx - 1) % SIGNAL_HISTORY_LEN
        if hbt.current_timestamp - hist_ts[prev_idx] < HISTORY_RECORD_INTERVAL_NS:
            return write_idx, count
    hist_ts[write_idx] = hbt.current_timestamp
    hist_mid[write_idx] = current_mid(depth)
    hist_fair[write_idx] = microprice(depth)
    hist_bid_qty[write_idx] = depth.best_bid_qty
    hist_ask_qty[write_idx] = depth.best_ask_qty
    write_idx = (write_idx + 1) % SIGNAL_HISTORY_LEN
    count = min(count + 1, SIGNAL_HISTORY_LEN)
    return write_idx, count


@njit
def history_idx_at_or_before(hist_ts, write_idx, count, target_ts):
    for offset in range(count):
        idx = (write_idx - 1 - offset) % SIGNAL_HISTORY_LEN
        if hist_ts[idx] <= target_ts:
            return idx
    return -1


@njit
def recent_move_bps(hist_ts, hist_mid, write_idx, count, window_ns):
    if count <= 1:
        return 0.0
    cur_idx = (write_idx - 1) % SIGNAL_HISTORY_LEN
    past_idx = history_idx_at_or_before(hist_ts, write_idx, count, hist_ts[cur_idx] - window_ns)
    if past_idx < 0 or hist_mid[past_idx] <= 0:
        return 0.0
    return ratio_minus_one_bps(hist_mid[cur_idx], hist_mid[past_idx])


@njit
def refill_ratio(hist_ts, hist_qty, write_idx, count, window_ns):
    if count <= 1:
        return 1.0
    cur_idx = (write_idx - 1) % SIGNAL_HISTORY_LEN
    past_idx = history_idx_at_or_before(hist_ts, write_idx, count, hist_ts[cur_idx] - window_ns)
    if past_idx < 0 or hist_qty[past_idx] <= 0:
        return 1.0
    return hist_qty[cur_idx] / hist_qty[past_idx]


@njit
def strategy_signal(strategy_mode, hbt, flow, hist_ts, hist_mid, hist_fair, hist_bid_qty, hist_ask_qty, write_idx, count):
    depth = hbt.depth(0)
    if depth.best_bid <= 0 or depth.best_ask <= 0:
        return 0, 0.0

    spr = spread_bps(depth)
    if spr <= 0 or spr > MAX_SPREAD_BPS:
        return 0, 0.0

    mid = current_mid(depth)
    fair = microprice(depth)
    dev = ratio_minus_one_bps(mid, fair)
    micro = microprice_bps(depth)
    qimb = queue_imbalance(depth)
    fs = flow_score(flow)
    slow_move = recent_move_bps(hist_ts, hist_mid, write_idx, count, SLOW_WINDOW_NS)
    fast_move = recent_move_bps(hist_ts, hist_mid, write_idx, count, FAST_WINDOW_NS)
    bid_refill = refill_ratio(hist_ts, hist_bid_qty, write_idx, count, FAST_WINDOW_NS)
    ask_refill = refill_ratio(hist_ts, hist_ask_qty, write_idx, count, FAST_WINDOW_NS)

    # Mid is below the book fair price after a short downside shock: buy snapback.
    if strategy_mode == STRATEGY_MICROPRICE_SNAPBACK:
        alpha = -dev - 0.6 * slow_move + 0.5 * micro + 0.25 * qimb
        if dev <= -DEVIATION_BPS and slow_move <= -SHOCK_MOVE_BPS:
            if micro >= -MAX_MICRO_CONTINUATION_BPS and fs >= -MAX_FLOW_CONTINUATION:
                return 1, alpha
        if dev >= DEVIATION_BPS and slow_move >= SHOCK_MOVE_BPS:
            if micro <= MAX_MICRO_CONTINUATION_BPS and fs <= MAX_FLOW_CONTINUATION:
                return -1, alpha
        return 0, alpha

    # Larger move, but the newest slice and trade flow stop confirming continuation.
    if strategy_mode == STRATEGY_IMPACT_EXHAUSTION:
        alpha = -slow_move - 0.6 * fast_move - 0.35 * fs + 0.25 * micro
        if slow_move <= -SHOCK_MOVE_BPS and fast_move >= -FAST_RELIEF_BPS and fs >= -MAX_FLOW_CONTINUATION:
            return 1, alpha
        if slow_move >= SHOCK_MOVE_BPS and fast_move <= FAST_RELIEF_BPS and fs <= MAX_FLOW_CONTINUATION:
            return -1, alpha
        return 0, alpha

    # After a shock, the impacted side refills instead of disappearing.
    if strategy_mode == STRATEGY_QUEUE_REFILL_REVERTER:
        alpha = -slow_move + 0.5 * micro + 0.5 * qimb
        if slow_move <= -SHOCK_MOVE_BPS and bid_refill >= MIN_REFILL_RATIO and qimb >= -0.15:
            if fs >= -MAX_FLOW_CONTINUATION:
                return 1, alpha
        if slow_move >= SHOCK_MOVE_BPS and ask_refill >= MIN_REFILL_RATIO and qimb <= 0.15:
            if fs <= MAX_FLOW_CONTINUATION:
                return -1, alpha
        return 0, alpha

    # Spread widens after a micro shock; trade only when fair price points back.
    alpha = -slow_move - dev + 0.4 * micro + 0.2 * qimb
    if spr >= SPREAD_SHOCK_BPS:
        if fast_move <= -FAST_RELIEF_BPS and dev <= 0.0 and micro >= -MAX_MICRO_CONTINUATION_BPS:
            if fs >= -MAX_FLOW_CONTINUATION:
                return 1, alpha
        if fast_move >= FAST_RELIEF_BPS and dev >= 0.0 and micro <= MAX_MICRO_CONTINUATION_BPS:
            if fs <= MAX_FLOW_CONTINUATION:
                return -1, alpha
    return 0, alpha


@njit
def cancel_order(hbt, order_id):
    order = hbt.orders(0).get(order_id)
    if order is not None and order.cancellable:
        hbt.cancel(0, order_id, False)
        return True
    return False


@njit
def cancel_all_orders(hbt):
    orders = hbt.orders(0)
    values = orders.values()
    while values.has_next():
        order = values.get()
        if order.cancellable:
            hbt.cancel(0, order.order_id, False)


@njit
def can_add_position(pos, side):
    if side > 0:
        return pos + ORDER_QTY <= MAX_POSITION_CONTRACTS
    return pos - ORDER_QTY >= -MAX_POSITION_CONTRACTS


@njit
def manage_passive_entry(hbt, signal, entry_order_id, live_since, metrics):
    depth = hbt.depth(0)
    existing = hbt.orders(0).get(entry_order_id)
    pos = hbt.position(0)
    if abs(pos) > 0 or signal == 0 or not can_add_position(pos, signal):
        if existing is not None and existing.cancellable:
            hbt.cancel(0, entry_order_id, False)
            metrics[17] += 1
        return 0

    if live_since > 0 and hbt.current_timestamp - live_since >= ENTRY_TTL_NS:
        if existing is not None and existing.cancellable:
            hbt.cancel(0, entry_order_id, False)
            metrics[17] += 1
        return 0

    px = floor_to_tick(depth.best_bid, BITMEX_TICK_SIZE) if signal > 0 else ceil_to_tick(depth.best_ask, BITMEX_TICK_SIZE)
    if existing is not None:
        if existing.cancellable and abs(existing.price - px) >= MIN_AMEND_TICKS * BITMEX_TICK_SIZE:
            hbt.modify(0, entry_order_id, px, ORDER_QTY, False)
            metrics[18] += 1
            return hbt.current_timestamp
        return live_since

    if signal > 0:
        hbt.submit_buy_order(0, entry_order_id, px, ORDER_QTY, GTX, LIMIT, False)
    else:
        hbt.submit_sell_order(0, entry_order_id, px, ORDER_QTY, GTX, LIMIT, False)
    metrics[16] += 1
    return hbt.current_timestamp


@njit
def maker_exit_price(pos, entry_px, depth):
    if entry_px <= 0:
        return 0.0
    if pos > 0:
        raw = entry_px * (1.0 + TARGET_PROFIT_BPS / 10_000.0)
        return ceil_to_tick(max(raw, depth.best_ask), BITMEX_TICK_SIZE)
    raw = entry_px * (1.0 - TARGET_PROFIT_BPS / 10_000.0)
    return floor_to_tick(min(raw, depth.best_bid), BITMEX_TICK_SIZE)


@njit
def manage_maker_exit(hbt, exit_order_id, live_since, entry_px, metrics):
    depth = hbt.depth(0)
    pos = hbt.position(0)
    existing = hbt.orders(0).get(exit_order_id)
    if abs(pos) <= 0:
        if existing is not None and existing.cancellable:
            hbt.cancel(0, exit_order_id, False)
            metrics[28] += 1
        return 0

    if live_since > 0 and hbt.current_timestamp - live_since >= EXIT_TTL_NS:
        if existing is not None and existing.cancellable:
            hbt.cancel(0, exit_order_id, False)
            metrics[30] += 1
        return 0

    px = maker_exit_price(pos, entry_px, depth)
    qty = abs(pos)
    if px <= 0 or qty < BITMEX_LOT_SIZE:
        return live_since

    if existing is not None:
        price_changed = abs(existing.price - px) >= MIN_AMEND_TICKS * BITMEX_TICK_SIZE
        qty_changed = existing.qty != qty
        if existing.cancellable and (price_changed or qty_changed):
            hbt.modify(0, exit_order_id, px, qty, False)
            metrics[27] += 1
            return hbt.current_timestamp
        return live_since

    if pos > 0:
        hbt.submit_sell_order(0, exit_order_id, px, qty, GTX, LIMIT, False)
    else:
        hbt.submit_buy_order(0, exit_order_id, px, qty, GTX, LIMIT, False)
    metrics[26] += 1
    return hbt.current_timestamp


@njit
def submit_ioc_exit(hbt, side, qty, order_id):
    depth = hbt.depth(0)
    if depth.best_bid <= 0 or depth.best_ask <= 0 or qty < BITMEX_LOT_SIZE:
        return order_id, False
    if side > 0:
        px = ceil_to_tick(depth.best_ask * (1.0 + TAKER_SLIPPAGE_BPS / 10_000.0), BITMEX_TICK_SIZE)
        hbt.submit_buy_order(0, order_id, px, qty, IOC, LIMIT, True)
    else:
        px = floor_to_tick(depth.best_bid * (1.0 - TAKER_SLIPPAGE_BPS / 10_000.0), BITMEX_TICK_SIZE)
        hbt.submit_sell_order(0, order_id, px, qty, IOC, LIMIT, True)
    return order_id + 1, True


@njit
def should_taker_exit(pos, signal, alpha, entry_px, entry_ts, now_ts, depth):
    if abs(pos) <= 0 or entry_px <= 0:
        return False
    mid = current_mid(depth)
    if mid <= 0:
        return False
    if pos > 0:
        pnl_bps = ratio_minus_one_bps(mid, entry_px)
        if now_ts - entry_ts >= MIN_HOLD_NS and (signal < 0 or alpha < -0.5):
            return True
    else:
        pnl_bps = ratio_minus_one_bps(entry_px, mid)
        if now_ts - entry_ts >= MIN_HOLD_NS and (signal > 0 or alpha > 0.5):
            return True
    if now_ts - entry_ts >= MIN_HOLD_NS and pnl_bps >= TARGET_PROFIT_BPS:
        return True
    if pnl_bps <= -STOP_LOSS_BPS:
        return True
    return now_ts - entry_ts >= MAX_HOLD_NS


@njit
def mark_trade_metrics(hbt, metrics, last_pos, last_trades, last_trading_value, entry_px, entry_ts):
    state = hbt.state_values(0)
    if state.num_trades <= last_trades:
        return last_pos, last_trades, last_trading_value, entry_px, entry_ts

    delta_pos = state.position - last_pos
    delta_value = state.trading_value - last_trading_value
    trade_count = state.num_trades - last_trades
    depth = hbt.depth(0)
    mid = current_mid(depth) if depth.best_bid > 0 and depth.best_ask > 0 else 0.0
    exec_px = mid
    if abs(delta_pos) > 0 and delta_value > 0:
        if BITMEX_IS_INVERSE:
            exec_px = abs(delta_pos) * BITMEX_CONTRACT_SIZE / delta_value
        else:
            exec_px = delta_value / (abs(delta_pos) * BITMEX_CONTRACT_SIZE)

    metrics[0] += trade_count
    if delta_pos > 0:
        metrics[1] += trade_count
    elif delta_pos < 0:
        metrics[2] += trade_count
    metrics[3] += abs(bitmex_base_from_contracts(delta_pos, exec_px))

    if abs(last_pos) <= 0 and abs(state.position) > 0:
        metrics[4] += 1
        entry_px = exec_px
        entry_ts = hbt.current_timestamp
    elif abs(last_pos) > 0 and abs(state.position) <= 0:
        metrics[5] += 1
        entry_px = 0.0
        entry_ts = 0
    elif last_pos * state.position < 0:
        metrics[5] += 1
        metrics[4] += 1
        entry_px = exec_px
        entry_ts = hbt.current_timestamp
    return state.position, state.num_trades, state.trading_value, entry_px, entry_ts


@njit
def update_risk_metrics(hbt, metrics):
    depth = hbt.depth(0)
    if depth.best_bid <= 0 or depth.best_ask <= 0:
        return
    mid = current_mid(depth)
    pos_contracts = abs(hbt.position(0))
    metrics[8] = max(metrics[8], pos_contracts)
    metrics[9] = max(metrics[9], abs(bitmex_base_from_contracts(pos_contracts, mid)))
    equity = bitmex_equity_usdt(hbt)
    if not math.isfinite(equity):
        return
    if metrics[11] == 0.0 and metrics[12] == 0.0:
        metrics[11] = equity
        metrics[12] = equity
    metrics[11] = max(metrics[11], equity)
    metrics[12] = min(metrics[12], equity)


@njit
def force_flatten(hbt, next_order_id, metrics):
    cancel_all_orders(hbt)
    hbt.elapse(1_000_000_000)
    pos = hbt.position(0)
    if abs(pos) <= 0:
        return next_order_id
    before = bitmex_equity_usdt(hbt)
    if pos > 0:
        next_order_id, _ = submit_ioc_exit(hbt, -1, abs(pos), next_order_id)
    else:
        next_order_id, _ = submit_ioc_exit(hbt, 1, abs(pos), next_order_id)
    hbt.elapse(1_000_000_000)
    metrics[10] = bitmex_equity_usdt(hbt) - before
    return next_order_id


@njit
def run_strategy(hbt, recorder, metrics, end_close_ts_ns, strategy_mode):
    hist_ts = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.int64)
    hist_mid = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    hist_fair = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    hist_bid_qty = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    hist_ask_qty = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    flow = np.zeros(6, dtype=np.float64)
    write_idx = 0
    count = 0

    next_order_id = 100_001
    entry_order_id = 50_001
    exit_order_id = 60_001
    entry_live_since = 0
    exit_live_since = 0
    next_action_ts = 0
    last_record_ts = 0
    entry_px = 0.0
    entry_ts = 0

    last_pos = hbt.position(0)
    last_trades = hbt.state_values(0).num_trades
    last_trading_value = hbt.state_values(0).trading_value

    while hbt.elapse(ORDER_UPDATE_INTERVAL_NS) == 0:
        if hbt.current_timestamp >= end_close_ts_ns:
            break
        hbt.clear_inactive_orders(ALL_ASSETS)
        update_market_flow(hbt, flow)
        write_idx, count = record_history(
            hbt, hist_ts, hist_mid, hist_fair, hist_bid_qty, hist_ask_qty, write_idx, count
        )

        depth = hbt.depth(0)
        if depth.best_bid <= 0 or depth.best_ask <= 0:
            if cancel_order(hbt, entry_order_id):
                metrics[17] += 1
            if cancel_order(hbt, exit_order_id):
                metrics[28] += 1
        else:
            sig, alpha = strategy_signal(
                strategy_mode, hbt, flow, hist_ts, hist_mid, hist_fair, hist_bid_qty, hist_ask_qty, write_idx, count
            )
            metrics[20] += abs(alpha)
            metrics[21] += 1
            metrics[22] = flow[2]
            metrics[23] = flow[3]
            metrics[24] = flow[4]
            metrics[25] = flow[5]
            if sig > 0:
                metrics[6] += 1
            elif sig < 0:
                metrics[7] += 1
            if INVERT_SIGNAL:
                sig = -sig

            pos = hbt.position(0)
            if EXECUTION_MODE == EXECUTION_MODE_TAKER_TAKER:
                if abs(pos) > 0 and should_taker_exit(pos, sig, alpha, entry_px, entry_ts, hbt.current_timestamp, depth):
                    if pos > 0:
                        next_order_id, ok = submit_ioc_exit(hbt, -1, abs(pos), next_order_id)
                    else:
                        next_order_id, ok = submit_ioc_exit(hbt, 1, abs(pos), next_order_id)
                    if ok:
                        metrics[14] += 1
                        metrics[29] += 1
                        next_action_ts = hbt.current_timestamp + COOLDOWN_NS
                elif abs(pos) <= 0 and sig != 0 and hbt.current_timestamp >= next_action_ts:
                    if can_add_position(pos, sig):
                        next_order_id, ok = submit_ioc_exit(hbt, sig, ORDER_QTY, next_order_id)
                        if ok:
                            metrics[13] += 1
                            next_action_ts = hbt.current_timestamp + COOLDOWN_NS
            else:
                if abs(pos) > 0 and should_taker_exit(pos, sig, alpha, entry_px, entry_ts, hbt.current_timestamp, depth):
                    cancel_all_orders(hbt)
                    exit_live_since = 0
                    if pos > 0:
                        next_order_id, ok = submit_ioc_exit(hbt, -1, abs(pos), next_order_id)
                    else:
                        next_order_id, ok = submit_ioc_exit(hbt, 1, abs(pos), next_order_id)
                    if ok:
                        metrics[14] += 1
                        metrics[29] += 1
                        next_action_ts = hbt.current_timestamp + COOLDOWN_NS
                elif abs(pos) > 0:
                    if cancel_order(hbt, entry_order_id):
                        metrics[17] += 1
                    exit_live_since = manage_maker_exit(hbt, exit_order_id, exit_live_since, entry_px, metrics)
                elif hbt.current_timestamp >= next_action_ts:
                    if cancel_order(hbt, exit_order_id):
                        metrics[28] += 1
                    exit_live_since = 0
                    entry_live_since = manage_passive_entry(hbt, sig, entry_order_id, entry_live_since, metrics)

        last_pos, last_trades, last_trading_value, entry_px, entry_ts = mark_trade_metrics(
            hbt, metrics, last_pos, last_trades, last_trading_value, entry_px, entry_ts
        )
        update_risk_metrics(hbt, metrics)
        if hbt.current_timestamp - last_record_ts >= 1_000_000_000:
            recorder.record(hbt)
            last_record_ts = hbt.current_timestamp

    next_order_id = force_flatten(hbt, next_order_id, metrics)
    last_pos, last_trades, last_trading_value, entry_px, entry_ts = mark_trade_metrics(
        hbt, metrics, last_pos, last_trades, last_trading_value, entry_px, entry_ts
    )
    recorder.record(hbt)
    return True


def strategy_slug(strategy_mode: int) -> str:
    return STRATEGY_SLUGS[strategy_mode]


def parse_strategy(value: str) -> int:
    normalized = value.lower().replace("-", "_")
    for mode, slug in STRATEGY_SLUGS.items():
        if normalized == slug:
            return mode
    raise ValueError(f"unknown strategy: {value}")


def build_asset(bitmex_npz: Path):
    asset = BacktestAsset().data([str(bitmex_npz)])
    if BITMEX_IS_INVERSE:
        asset = asset.inverse_asset(BITMEX_CONTRACT_SIZE)
    else:
        asset = asset.linear_asset(BITMEX_CONTRACT_SIZE)
    asset = (
        asset.constant_order_latency(BITMEX_ORDER_ENTRY_LATENCY_NS, BITMEX_ORDER_RESPONSE_LATENCY_NS)
        .risk_adverse_queue_model()
        .trading_value_fee_model(MAKER_FEE_RATE, TAKER_FEE_RATE)
        .tick_size(BITMEX_TICK_SIZE)
        .lot_size(BITMEX_LOT_SIZE)
        .last_trades_capacity(10_000)
    )
    if EXCHANGE_MODEL == "partial":
        return asset.partial_fill_exchange()
    if EXCHANGE_MODEL == "strict_no_partial":
        return asset.strict_no_partial_fill_exchange()
    return asset.no_partial_fill_exchange()


def run_backtest(bitmex_npz: Path, yyyymmdd: str, strategy_mode: int) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    hbt = HashMapMarketDepthBacktest([build_asset(bitmex_npz)])
    recorder = Recorder(1, 100_000)
    metrics = np.zeros(32, dtype=np.float64)
    ok = run_strategy(hbt, recorder.recorder, metrics, end_close_ts_ns(yyyymmdd), strategy_mode)
    if not ok:
        raise RuntimeError("strategy returned false")

    tag = f"{RESULT_TAG}_{strategy_slug(strategy_mode)}" if RESULT_TAG else f"{strategy_slug(strategy_mode)}_{EXCHANGE_MODEL}"
    out = RESULT_DIR / f"bitmex_{BITMEX_SYMBOL.lower()}_trend2_reversal_{tag}_{yyyymmdd}.npz"
    np.savez_compressed(out, **{"0": recorder.get(0), "metrics": metrics})
    return out


def signed_money(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.4f} USDT"


def config_dict() -> dict:
    return {
        "order_qty_contracts": ORDER_QTY,
        "max_position_contracts": MAX_POSITION_CONTRACTS,
        "execution_mode": "taker_taker" if EXECUTION_MODE == EXECUTION_MODE_TAKER_TAKER else "passive_maker",
        "invert_signal": INVERT_SIGNAL,
        "update_interval_ms": ORDER_UPDATE_INTERVAL_NS / 1_000_000.0,
        "history_interval_ms": HISTORY_RECORD_INTERVAL_NS / 1_000_000.0,
        "min_hold_ms": MIN_HOLD_NS / 1_000_000.0,
        "max_hold_ms": MAX_HOLD_NS / 1_000_000.0,
        "cooldown_ms": COOLDOWN_NS / 1_000_000.0,
        "entry_ttl_ms": ENTRY_TTL_NS / 1_000_000.0,
        "exit_ttl_ms": EXIT_TTL_NS / 1_000_000.0,
        "maker_fee_rate": MAKER_FEE_RATE,
        "taker_fee_rate": TAKER_FEE_RATE,
        "exchange_model": EXCHANGE_MODEL,
        "slow_window_ms": SLOW_WINDOW_NS / 1_000_000.0,
        "fast_window_ms": FAST_WINDOW_NS / 1_000_000.0,
        "deviation_bps": DEVIATION_BPS,
        "shock_move_bps": SHOCK_MOVE_BPS,
        "fast_relief_bps": FAST_RELIEF_BPS,
        "target_profit_bps": TARGET_PROFIT_BPS,
        "stop_loss_bps": STOP_LOSS_BPS,
        "spread_shock_bps": SPREAD_SHOCK_BPS,
        "max_spread_bps": MAX_SPREAD_BPS,
    }


def write_summary(result_npz: Path, yyyymmdd: str, strategy_mode: int) -> dict:
    data = np.load(result_npz)
    records = data["0"]
    metrics = data["metrics"]
    final = records[-1]
    price = float(final["price"])
    final_contracts = float(final["position"])
    if BITMEX_IS_INVERSE:
        final_base = final_contracts * BITMEX_CONTRACT_SIZE / price if price > 0 else 0.0
        fee_usdt = float(final["fee"]) * price
        equity_btc = float(final["balance"]) + final_base - float(final["fee"])
        equity_usdt = equity_btc * price
    else:
        final_base = final_contracts * BITMEX_CONTRACT_SIZE
        fee_usdt = float(final["fee"])
        equity_usdt = float(final["balance"]) + final_contracts * price * BITMEX_CONTRACT_SIZE - float(final["fee"])
    gross = equity_usdt + fee_usdt
    avg_abs_alpha = metrics[20] / metrics[21] if metrics[21] > 0 else 0.0
    summary = {
        "date": yyyymmdd,
        "symbol": BITMEX_SYMBOL,
        "strategy": strategy_slug(strategy_mode),
        **config_dict(),
        "total_pnl_usdt": equity_usdt,
        "gross_pnl_before_fee_usdt": gross,
        "fee_usdt": fee_usdt,
        "fills": int(metrics[0]),
        "buy_fills": int(metrics[1]),
        "sell_fills": int(metrics[2]),
        "filled_base_btc": float(metrics[3]),
        "entries": int(metrics[4]),
        "exits": int(metrics[5]),
        "long_signal_samples": int(metrics[6]),
        "short_signal_samples": int(metrics[7]),
        "max_position_contracts_seen": float(metrics[8]),
        "max_position_base_seen": float(metrics[9]),
        "force_close_pnl_usdt": float(metrics[10]),
        "max_equity_usdt": float(metrics[11]),
        "min_equity_usdt": float(metrics[12]),
        "taker_entry_orders": int(metrics[13]),
        "taker_exit_orders": int(metrics[14]),
        "passive_entry_place": int(metrics[16]),
        "passive_entry_cancel": int(metrics[17]),
        "passive_entry_modify": int(metrics[18]),
        "maker_exit_place": int(metrics[26]),
        "maker_exit_modify": int(metrics[27]),
        "maker_exit_cancel": int(metrics[28]),
        "emergency_exit_orders": int(metrics[29]),
        "maker_exit_ttl_cancel": int(metrics[30]),
        "avg_abs_alpha": float(avg_abs_alpha),
        "market_buy_qty_seen": float(metrics[22]),
        "market_sell_qty_seen": float(metrics[23]),
        "market_buy_trade_count_seen": int(metrics[24]),
        "market_sell_trade_count_seen": int(metrics[25]),
        "final_position_contracts": final_contracts,
        "final_position_base": final_base,
        "records": int(len(records)),
        "num_trades": int(final["num_trades"]),
    }
    result_npz.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    result_npz.with_suffix(".report.md").write_text(render_report(summary))
    print(render_console_summary(summary))
    return summary


def render_console_summary(summary: dict) -> str:
    return "\n".join(
        [
            "",
            "========== BitMEX trend2 reversal 回测 ==========",
            f"策略: {summary['strategy']}",
            f"日期: {summary['date']}",
            f"PnL: {signed_money(summary['total_pnl_usdt'])}",
            f"gross 不含手续费: {signed_money(summary['gross_pnl_before_fee_usdt'])}",
            f"fee: {signed_money(summary['fee_usdt'])}",
            f"fills: {summary['fills']}，entries/exits: {summary['entries']}/{summary['exits']}",
            f"信号样本: long={summary['long_signal_samples']}, short={summary['short_signal_samples']}",
            f"最大仓位: {summary['max_position_contracts_seen']:,.0f} contracts",
            f"taker orders: entry={summary['taker_entry_orders']}, exit={summary['taker_exit_orders']}",
            f"被动 entry: place={summary['passive_entry_place']}, modify={summary['passive_entry_modify']}, cancel={summary['passive_entry_cancel']}",
            f"maker exit: place={summary['maker_exit_place']}, modify={summary['maker_exit_modify']}, cancel={summary['maker_exit_cancel']}, emergency={summary['emergency_exit_orders']}",
            "=================================================",
            "",
        ]
    )


def render_report(summary: dict) -> str:
    return f"""# BitMEX Trend2 Reversal Strategy Report

## Result

- strategy: `{summary['strategy']}`
- date: `{summary['date']}`
- symbol: `{summary['symbol']}`
- exchange model: `{summary['exchange_model']}`
- total PnL: `{signed_money(summary['total_pnl_usdt'])}`
- gross before fee: `{signed_money(summary['gross_pnl_before_fee_usdt'])}`
- fee: `{signed_money(summary['fee_usdt'])}`
- fills: `{summary['fills']}`
- entries: `{summary['entries']}`
- exits: `{summary['exits']}`
- filled base: `{summary['filled_base_btc']:,.8f} BTC`
- max position: `{summary['max_position_contracts_seen']:,.0f} contracts`
- force close PnL: `{signed_money(summary['force_close_pnl_usdt'])}`

## Signal

- long signal samples: `{summary['long_signal_samples']}`
- short signal samples: `{summary['short_signal_samples']}`
- avg abs alpha: `{summary['avg_abs_alpha']:,.6f}`

## Execution

- passive entry place: `{summary['passive_entry_place']}`
- passive entry modify: `{summary['passive_entry_modify']}`
- passive entry cancel: `{summary['passive_entry_cancel']}`
- maker exit place: `{summary['maker_exit_place']}`
- maker exit modify: `{summary['maker_exit_modify']}`
- maker exit cancel: `{summary['maker_exit_cancel']}`
- emergency exit orders: `{summary['emergency_exit_orders']}`
- taker entry orders: `{summary['taker_entry_orders']}`
- taker exit orders: `{summary['taker_exit_orders']}`
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BitMEX second-level fair-price deviation reversal backtests.")
    parser.add_argument("--strategy", default="all", choices=["all", *STRATEGY_SLUGS.values()])
    parser.add_argument("--symbol", default=BITMEX_SYMBOL)
    parser.add_argument("--dates", nargs="+", default=["20260512", "20260513", "20260514"])
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--buffer-rows", type=int, default=None)
    parser.add_argument("--result-tag", default="")
    parser.add_argument("--exchange-model", choices=("no_partial", "strict_no_partial", "partial"), default=EXCHANGE_MODEL)
    parser.add_argument("--order-qty-contracts", type=float, default=ORDER_QTY)
    parser.add_argument("--max-position-contracts", type=float, default=MAX_POSITION_CONTRACTS)
    parser.add_argument("--maker-fee-rate", type=float, default=MAKER_FEE_RATE)
    parser.add_argument("--taker-fee-rate", type=float, default=TAKER_FEE_RATE)
    parser.add_argument("--execution-mode", choices=("passive_maker", "taker_taker"), default="taker_taker")
    parser.add_argument("--invert-signal", action="store_true")
    parser.add_argument("--update-interval-ms", type=float, default=ORDER_UPDATE_INTERVAL_NS / 1_000_000.0)
    parser.add_argument("--history-interval-ms", type=float, default=HISTORY_RECORD_INTERVAL_NS / 1_000_000.0)
    parser.add_argument("--min-hold-ms", type=float, default=MIN_HOLD_NS / 1_000_000.0)
    parser.add_argument("--max-hold-ms", type=float, default=MAX_HOLD_NS / 1_000_000.0)
    parser.add_argument("--cooldown-ms", type=float, default=COOLDOWN_NS / 1_000_000.0)
    parser.add_argument("--slow-window-ms", type=float, default=SLOW_WINDOW_NS / 1_000_000.0)
    parser.add_argument("--fast-window-ms", type=float, default=FAST_WINDOW_NS / 1_000_000.0)
    parser.add_argument("--deviation-bps", type=float, default=DEVIATION_BPS)
    parser.add_argument("--shock-move-bps", type=float, default=SHOCK_MOVE_BPS)
    parser.add_argument("--fast-relief-bps", type=float, default=FAST_RELIEF_BPS)
    parser.add_argument("--target-profit-bps", type=float, default=TARGET_PROFIT_BPS)
    parser.add_argument("--stop-loss-bps", type=float, default=STOP_LOSS_BPS)
    parser.add_argument("--spread-shock-bps", type=float, default=SPREAD_SHOCK_BPS)
    parser.add_argument("--max-spread-bps", type=float, default=MAX_SPREAD_BPS)
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    global BITMEX_SYMBOL, ORDER_QTY, MAX_POSITION_CONTRACTS, MAKER_FEE_RATE, TAKER_FEE_RATE, EXCHANGE_MODEL, RESULT_TAG
    global EXECUTION_MODE, INVERT_SIGNAL, ORDER_UPDATE_INTERVAL_NS, HISTORY_RECORD_INTERVAL_NS
    global MIN_HOLD_NS, MAX_HOLD_NS, COOLDOWN_NS, SLOW_WINDOW_NS, FAST_WINDOW_NS
    global DEVIATION_BPS, SHOCK_MOVE_BPS, TARGET_PROFIT_BPS, STOP_LOSS_BPS
    global FAST_RELIEF_BPS, SPREAD_SHOCK_BPS, MAX_SPREAD_BPS
    BITMEX_SYMBOL = args.symbol
    ORDER_QTY = args.order_qty_contracts
    MAX_POSITION_CONTRACTS = args.max_position_contracts
    MAKER_FEE_RATE = args.maker_fee_rate
    TAKER_FEE_RATE = args.taker_fee_rate
    EXCHANGE_MODEL = args.exchange_model
    RESULT_TAG = args.result_tag
    EXECUTION_MODE = EXECUTION_MODE_TAKER_TAKER if args.execution_mode == "taker_taker" else EXECUTION_MODE_PASSIVE_MAKER
    INVERT_SIGNAL = args.invert_signal
    ORDER_UPDATE_INTERVAL_NS = int(args.update_interval_ms * 1_000_000)
    HISTORY_RECORD_INTERVAL_NS = int(args.history_interval_ms * 1_000_000)
    MIN_HOLD_NS = int(args.min_hold_ms * 1_000_000)
    MAX_HOLD_NS = int(args.max_hold_ms * 1_000_000)
    COOLDOWN_NS = int(args.cooldown_ms * 1_000_000)
    SLOW_WINDOW_NS = int(args.slow_window_ms * 1_000_000)
    FAST_WINDOW_NS = int(args.fast_window_ms * 1_000_000)
    DEVIATION_BPS = args.deviation_bps
    SHOCK_MOVE_BPS = args.shock_move_bps
    FAST_RELIEF_BPS = args.fast_relief_bps
    TARGET_PROFIT_BPS = args.target_profit_bps
    STOP_LOSS_BPS = args.stop_loss_bps
    SPREAD_SHOCK_BPS = args.spread_shock_bps
    MAX_SPREAD_BPS = args.max_spread_bps


def write_aggregate(rows: list[dict], tag: str) -> Path:
    out = RESULT_DIR / f"bitmex_{BITMEX_SYMBOL.lower()}_trend2_reversal_{tag}.aggregate.csv"
    keys = [
        "strategy",
        "date",
        "total_pnl_usdt",
        "gross_pnl_before_fee_usdt",
        "fee_usdt",
        "fills",
        "entries",
        "exits",
        "long_signal_samples",
        "short_signal_samples",
        "filled_base_btc",
        "max_position_contracts_seen",
        "taker_entry_orders",
        "taker_exit_orders",
        "passive_entry_place",
        "maker_exit_place",
        "emergency_exit_orders",
    ]
    with out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in keys} for row in rows])
    return out


def main() -> None:
    args = parse_args()
    apply_args(args)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    strategies = list(STRATEGY_SLUGS.keys()) if args.strategy == "all" else [parse_strategy(args.strategy)]
    key = None if args.skip_download else tardis_key()
    rows = []
    for yyyymmdd in args.dates:
        if not args.skip_download:
            download_file(BITMEX_EXCHANGE, "trades", BITMEX_SYMBOL, yyyymmdd, key)
            download_file(BITMEX_EXCHANGE, "incremental_book_L2", BITMEX_SYMBOL, yyyymmdd, key)
        bitmex_npz = convert_bitmex(BITMEX_SYMBOL, yyyymmdd, args.buffer_rows)
        for strategy_mode in strategies:
            rows.append(write_summary(run_backtest(bitmex_npz, yyyymmdd, strategy_mode), yyyymmdd, strategy_mode))
    tag = args.result_tag or f"{args.dates[0]}_{args.dates[-1]}_{EXCHANGE_MODEL}"
    aggregate = write_aggregate(rows, tag)
    total_pnl = sum(row["total_pnl_usdt"] for row in rows)
    total_fills = sum(row["fills"] for row in rows)
    print(f"aggregate={aggregate} total_pnl={total_pnl:.6f} fills={total_fills}")


if __name__ == "__main__":
    main()
