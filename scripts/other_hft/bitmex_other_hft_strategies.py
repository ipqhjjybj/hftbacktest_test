import argparse
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
from hftbacktest.order import IOC, MARKET  # noqa: E402

from bitmex_single_market_mm_backtest import (  # noqa: E402
    BITMEX_EXCHANGE,
    BITMEX_LOT_SIZE,
    BITMEX_ORDER_ENTRY_LATENCY_NS,
    BITMEX_ORDER_QTY,
    BITMEX_ORDER_RESPONSE_LATENCY_NS,
    BITMEX_SYMBOL,
    BITMEX_TICK_SIZE,
    CSV_DIR,
    DEFAULT_DATES,
    NPZ_DIR,
    RESULT_DIR,
    convert_bitmex,
    download_file,
    end_close_ts_ns,
    tardis_key,
)


STRATEGY_ORDER_FLOW_MOMENTUM = 1
STRATEGY_QUEUE_IMBALANCE_BREAKOUT = 2
STRATEGY_LIQUIDITY_FADE = 3
STRATEGY_MEAN_REVERSION_SCALPER = 4
STRATEGY_PASSIVE_ENTRY_ACTIVE_EXIT = 5

STRATEGY_SLUGS = {
    STRATEGY_ORDER_FLOW_MOMENTUM: "order_flow_momentum",
    STRATEGY_QUEUE_IMBALANCE_BREAKOUT: "queue_imbalance_breakout",
    STRATEGY_LIQUIDITY_FADE: "liquidity_fade",
    STRATEGY_MEAN_REVERSION_SCALPER: "mean_reversion_scalper",
    STRATEGY_PASSIVE_ENTRY_ACTIVE_EXIT: "passive_entry_active_exit",
}

STRATEGY_MODE = STRATEGY_ORDER_FLOW_MOMENTUM

BITMEX_CONTRACT_SIZE = 0.000001
BITMEX_IS_INVERSE = False

ORDER_UPDATE_INTERVAL_NS = 20_000_000
SIGNAL_HISTORY_LEN = 8192
ORDER_QTY = BITMEX_ORDER_QTY
MAX_POSITION_CONTRACTS = 300.0
MAX_HOLD_NS = 2_000_000_000
COOLDOWN_NS = 300_000_000

MAKER_FEE_RATE = -0.0002
TAKER_FEE_RATE = 0.0001
EXCHANGE_MODEL = "no_partial"
RESULT_TAG = ""

FLOW_DECAY = 0.92
FLOW_SIGNAL_THRESHOLD = 0.35
FLOW_MIN_QTY = 300.0
FLOW_MICRO_CONFIRM_BPS = 0.05

QUEUE_IMBALANCE_THRESHOLD = 0.68
QUEUE_MICRO_THRESHOLD_BPS = 0.15
QUEUE_MIN_BBO_QTY = 50.0

LIQUIDITY_FADE_LOOKBACK_NS = 300_000_000
LIQUIDITY_FADE_DROP_RATIO = 0.55
LIQUIDITY_FADE_MIN_START_QTY = 200.0

MEAN_REVERSION_MOVE_BPS = 1.2
MEAN_REVERSION_WINDOW_NS = 600_000_000
MEAN_REVERSION_FLOW_CAP = 0.20

TAKER_SLIPPAGE_BPS = 1.0
TARGET_PROFIT_BPS = 1.2
STOP_LOSS_BPS = 2.0

PASSIVE_ENTRY_TTL_NS = 1_500_000_000
PASSIVE_MIN_AMEND_TICKS = 2.0
PASSIVE_ALPHA_THRESHOLD = 0.30
PASSIVE_EXIT_REVERSE_THRESHOLD = 0.10


@njit
def ceil_to_tick(px, tick_size):
    return math.ceil(px / tick_size) * tick_size


@njit
def floor_to_tick(px, tick_size):
    return math.floor(px / tick_size) * tick_size


@njit
def round_to_tick(px, tick_size):
    return round(px / tick_size) * tick_size


@njit
def clamp(value, low, high):
    return min(max(value, low), high)


@njit
def ratio_minus_one_bps(numerator, denominator):
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return 0.0
    return (numerator / denominator - 1.0) * 10_000.0


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
    mid = (depth.best_bid + depth.best_ask) / 2.0
    state = hbt.state_values(0)
    if BITMEX_IS_INVERSE:
        equity_btc = state.balance + state.position * BITMEX_CONTRACT_SIZE / mid - state.fee
        return equity_btc * mid
    return state.balance + state.position * mid * BITMEX_CONTRACT_SIZE - state.fee


@njit
def current_mid(depth):
    return (depth.best_bid + depth.best_ask) / 2.0


@njit
def microprice_bps(depth):
    mid = current_mid(depth)
    total = depth.best_bid_qty + depth.best_ask_qty
    if total <= 0 or mid <= 0:
        return 0.0
    micro = (depth.best_ask * depth.best_bid_qty + depth.best_bid * depth.best_ask_qty) / total
    return ratio_minus_one_bps(micro, mid)


@njit
def queue_imbalance(depth):
    total = depth.best_bid_qty + depth.best_ask_qty
    if total <= 0:
        return 0.0
    return depth.best_bid_qty / total


@njit
def signed_queue_imbalance(depth):
    return queue_imbalance(depth) * 2.0 - 1.0


@njit
def update_market_trades(hbt, flow):
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
def record_depth_history(hbt, hist_ts, hist_mid, hist_bid_qty, hist_ask_qty, write_idx, count):
    depth = hbt.depth(0)
    if depth.best_bid <= 0 or depth.best_ask <= 0:
        return write_idx, count
    hist_ts[write_idx] = hbt.current_timestamp
    hist_mid[write_idx] = current_mid(depth)
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
def liquidity_fade_signal(hbt, hist_ts, hist_bid_qty, hist_ask_qty, write_idx, count):
    if count <= 1:
        return 0
    depth = hbt.depth(0)
    past_idx = history_idx_at_or_before(hist_ts, write_idx, count, hbt.current_timestamp - LIQUIDITY_FADE_LOOKBACK_NS)
    if past_idx < 0:
        return 0

    past_bid = hist_bid_qty[past_idx]
    past_ask = hist_ask_qty[past_idx]
    if past_ask >= LIQUIDITY_FADE_MIN_START_QTY and depth.best_ask_qty <= past_ask * LIQUIDITY_FADE_DROP_RATIO:
        if past_bid <= 0 or depth.best_bid_qty >= past_bid * 0.75:
            return 1
    if past_bid >= LIQUIDITY_FADE_MIN_START_QTY and depth.best_bid_qty <= past_bid * LIQUIDITY_FADE_DROP_RATIO:
        if past_ask <= 0 or depth.best_ask_qty >= past_ask * 0.75:
            return -1
    return 0


@njit
def strategy_signal(hbt, flow, hist_ts, hist_mid, hist_bid_qty, hist_ask_qty, write_idx, count):
    depth = hbt.depth(0)
    if depth.best_bid <= 0 or depth.best_ask <= 0:
        return 0, 0.0

    fs = flow_score(flow)
    micro = microprice_bps(depth)
    qimb = queue_imbalance(depth)
    sqimb = qimb * 2.0 - 1.0

    if STRATEGY_MODE == STRATEGY_ORDER_FLOW_MOMENTUM:
        alpha = fs + micro / 2.0
        if fs >= FLOW_SIGNAL_THRESHOLD and micro >= -FLOW_MICRO_CONFIRM_BPS:
            return 1, alpha
        if fs <= -FLOW_SIGNAL_THRESHOLD and micro <= FLOW_MICRO_CONFIRM_BPS:
            return -1, alpha
        return 0, alpha

    if STRATEGY_MODE == STRATEGY_QUEUE_IMBALANCE_BREAKOUT:
        alpha = sqimb + micro / 2.0
        if depth.best_bid_qty < QUEUE_MIN_BBO_QTY or depth.best_ask_qty < QUEUE_MIN_BBO_QTY:
            return 0, alpha
        if qimb >= QUEUE_IMBALANCE_THRESHOLD and micro >= QUEUE_MICRO_THRESHOLD_BPS:
            return 1, alpha
        if qimb <= 1.0 - QUEUE_IMBALANCE_THRESHOLD and micro <= -QUEUE_MICRO_THRESHOLD_BPS:
            return -1, alpha
        return 0, alpha

    if STRATEGY_MODE == STRATEGY_LIQUIDITY_FADE:
        sig = liquidity_fade_signal(hbt, hist_ts, hist_bid_qty, hist_ask_qty, write_idx, count)
        alpha = float(sig) + fs * 0.25 + micro * 0.05
        return sig, alpha

    if STRATEGY_MODE == STRATEGY_MEAN_REVERSION_SCALPER:
        move = recent_move_bps(hist_ts, hist_mid, write_idx, count, MEAN_REVERSION_WINDOW_NS)
        alpha = -move / max(MEAN_REVERSION_MOVE_BPS, 1e-9)
        if move >= MEAN_REVERSION_MOVE_BPS and fs <= MEAN_REVERSION_FLOW_CAP:
            return -1, alpha
        if move <= -MEAN_REVERSION_MOVE_BPS and fs >= -MEAN_REVERSION_FLOW_CAP:
            return 1, alpha
        return 0, alpha

    alpha = 0.55 * fs + 0.35 * sqimb + 0.10 * micro
    if alpha >= PASSIVE_ALPHA_THRESHOLD:
        return 1, alpha
    if alpha <= -PASSIVE_ALPHA_THRESHOLD:
        return -1, alpha
    return 0, alpha


@njit
def can_add_position(pos, side):
    if side > 0:
        return pos + ORDER_QTY <= MAX_POSITION_CONTRACTS
    return pos - ORDER_QTY >= -MAX_POSITION_CONTRACTS


@njit
def submit_ioc_limit(hbt, side, qty, order_id):
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
def manage_passive_entry(hbt, signal, entry_order_id, live_since, metrics):
    depth = hbt.depth(0)
    existing = hbt.orders(0).get(entry_order_id)
    pos = hbt.position(0)

    if abs(pos) > 0:
        if existing is not None and existing.cancellable:
            hbt.cancel(0, entry_order_id, False)
            metrics[17] += 1
        return 0

    if signal == 0 or not can_add_position(pos, signal):
        if existing is not None and existing.cancellable:
            hbt.cancel(0, entry_order_id, False)
            metrics[17] += 1
        return 0

    if live_since > 0 and hbt.current_timestamp - live_since >= PASSIVE_ENTRY_TTL_NS:
        if existing is not None and existing.cancellable:
            hbt.cancel(0, entry_order_id, False)
            metrics[17] += 1
        return 0

    if signal > 0:
        px = floor_to_tick(depth.best_bid, BITMEX_TICK_SIZE)
    else:
        px = ceil_to_tick(depth.best_ask, BITMEX_TICK_SIZE)

    if existing is not None:
        price_changed = abs(existing.price - px) >= PASSIVE_MIN_AMEND_TICKS * BITMEX_TICK_SIZE
        if existing.cancellable and price_changed:
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
def should_exit_position(pos, signal, alpha, entry_px, entry_ts, now_ts, depth):
    if abs(pos) <= 0:
        return False
    mid = current_mid(depth)
    if entry_px <= 0 or mid <= 0:
        return False

    pnl_bps = 0.0
    if pos > 0:
        pnl_bps = ratio_minus_one_bps(mid, entry_px)
        if signal < 0 or alpha <= -PASSIVE_EXIT_REVERSE_THRESHOLD:
            return True
    else:
        pnl_bps = ratio_minus_one_bps(entry_px, mid)
        if signal > 0 or alpha >= PASSIVE_EXIT_REVERSE_THRESHOLD:
            return True

    if pnl_bps >= TARGET_PROFIT_BPS:
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
        next_order_id, _ = submit_ioc_limit(hbt, -1, abs(pos), next_order_id)
    else:
        next_order_id, _ = submit_ioc_limit(hbt, 1, abs(pos), next_order_id)
    hbt.elapse(1_000_000_000)
    metrics[10] = bitmex_equity_usdt(hbt) - before
    return next_order_id


@njit
def run_strategy(hbt, recorder, metrics, end_close_ts_ns):
    hist_ts = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.int64)
    hist_mid = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    hist_bid_qty = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    hist_ask_qty = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    write_idx = 0
    count = 0

    flow = np.zeros(6, dtype=np.float64)
    next_order_id = 100_001
    entry_order_id = 50_001
    entry_live_since = 0
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

        update_market_trades(hbt, flow)
        metrics[22] = flow[2]
        metrics[23] = flow[3]
        metrics[24] = flow[4]
        metrics[25] = flow[5]
        write_idx, count = record_depth_history(
            hbt, hist_ts, hist_mid, hist_bid_qty, hist_ask_qty, write_idx, count
        )

        depth = hbt.depth(0)
        if depth.best_bid <= 0 or depth.best_ask <= 0:
            if cancel_order(hbt, entry_order_id):
                metrics[17] += 1
        else:
            sig, alpha = strategy_signal(
                hbt, flow, hist_ts, hist_mid, hist_bid_qty, hist_ask_qty, write_idx, count
            )
            metrics[20] += abs(alpha)
            metrics[21] += 1
            if sig > 0:
                metrics[6] += 1
            elif sig < 0:
                metrics[7] += 1

            pos = hbt.position(0)
            if STRATEGY_MODE == STRATEGY_PASSIVE_ENTRY_ACTIVE_EXIT:
                if should_exit_position(pos, sig, alpha, entry_px, entry_ts, hbt.current_timestamp, depth):
                    if pos > 0:
                        next_order_id, ok = submit_ioc_limit(hbt, -1, abs(pos), next_order_id)
                    else:
                        next_order_id, ok = submit_ioc_limit(hbt, 1, abs(pos), next_order_id)
                    if ok:
                        metrics[14] += 1
                        next_action_ts = hbt.current_timestamp + COOLDOWN_NS
                elif hbt.current_timestamp >= next_action_ts:
                    entry_live_since = manage_passive_entry(hbt, sig, entry_order_id, entry_live_since, metrics)
            else:
                if should_exit_position(pos, sig, alpha, entry_px, entry_ts, hbt.current_timestamp, depth):
                    if pos > 0:
                        next_order_id, ok = submit_ioc_limit(hbt, -1, abs(pos), next_order_id)
                    else:
                        next_order_id, ok = submit_ioc_limit(hbt, 1, abs(pos), next_order_id)
                    if ok:
                        metrics[14] += 1
                        next_action_ts = hbt.current_timestamp + COOLDOWN_NS
                elif abs(pos) <= 0 and sig != 0 and hbt.current_timestamp >= next_action_ts:
                    if can_add_position(pos, sig):
                        next_order_id, ok = submit_ioc_limit(hbt, sig, ORDER_QTY, next_order_id)
                        if ok:
                            metrics[13] += 1
                            next_action_ts = hbt.current_timestamp + COOLDOWN_NS

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
    raise ValueError(f"unknown strategy: {value}; choices={list(STRATEGY_SLUGS.values())}")


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


def run_backtest(bitmex_npz: Path, yyyymmdd: str) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    hbt = HashMapMarketDepthBacktest([build_asset(bitmex_npz)])
    recorder = Recorder(1, 100_000)
    metrics = np.zeros(32, dtype=np.float64)
    ok = run_strategy(hbt, recorder.recorder, metrics, end_close_ts_ns(yyyymmdd))
    if not ok:
        raise RuntimeError("strategy returned false")

    tag = RESULT_TAG or f"{strategy_slug(STRATEGY_MODE)}_{EXCHANGE_MODEL}"
    out = RESULT_DIR / f"bitmex_{BITMEX_SYMBOL.lower()}_other_hft_{tag}_{yyyymmdd}.npz"
    np.savez_compressed(out, **{"0": recorder.get(0), "metrics": metrics})
    write_summary(out, yyyymmdd)
    return out


def signed_money(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.4f} USDT"


def config_dict() -> dict:
    return {
        "order_qty_contracts": ORDER_QTY,
        "max_position_contracts": MAX_POSITION_CONTRACTS,
        "max_hold_ms": MAX_HOLD_NS / 1_000_000.0,
        "cooldown_ms": COOLDOWN_NS / 1_000_000.0,
        "maker_fee_rate": MAKER_FEE_RATE,
        "taker_fee_rate": TAKER_FEE_RATE,
        "exchange_model": EXCHANGE_MODEL,
        "flow_decay": FLOW_DECAY,
        "flow_signal_threshold": FLOW_SIGNAL_THRESHOLD,
        "flow_min_qty": FLOW_MIN_QTY,
        "queue_imbalance_threshold": QUEUE_IMBALANCE_THRESHOLD,
        "liquidity_fade_lookback_ms": LIQUIDITY_FADE_LOOKBACK_NS / 1_000_000.0,
        "liquidity_fade_drop_ratio": LIQUIDITY_FADE_DROP_RATIO,
        "mean_reversion_move_bps": MEAN_REVERSION_MOVE_BPS,
        "mean_reversion_window_ms": MEAN_REVERSION_WINDOW_NS / 1_000_000.0,
        "taker_slippage_bps": TAKER_SLIPPAGE_BPS,
        "target_profit_bps": TARGET_PROFIT_BPS,
        "stop_loss_bps": STOP_LOSS_BPS,
        "passive_entry_ttl_ms": PASSIVE_ENTRY_TTL_NS / 1_000_000.0,
        "passive_alpha_threshold": PASSIVE_ALPHA_THRESHOLD,
    }


def write_summary(result_npz: Path, yyyymmdd: str) -> None:
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
        "strategy": strategy_slug(STRATEGY_MODE),
        "asset_type": "inverse" if BITMEX_IS_INVERSE else "linear",
        "contract_size": BITMEX_CONTRACT_SIZE,
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
    print(json.dumps(summary, indent=2, sort_keys=True))


def render_console_summary(summary: dict) -> str:
    return "\n".join(
        [
            "",
            "========== BitMEX other_hft 回测 ==========",
            f"策略: {summary['strategy']}",
            f"日期: {summary['date']}",
            f"PnL: {signed_money(summary['total_pnl_usdt'])}",
            f"gross 不含手续费: {signed_money(summary['gross_pnl_before_fee_usdt'])}",
            f"fee: {signed_money(summary['fee_usdt'])}",
            f"fills: {summary['fills']}，买={summary['buy_fills']}，卖={summary['sell_fills']}",
            f"entries/exits: {summary['entries']}/{summary['exits']}",
            f"最大仓位: {summary['max_position_contracts_seen']:,.0f} contracts",
            f"主动单: entry={summary['taker_entry_orders']}, exit={summary['taker_exit_orders']}",
            f"被动 entry: place={summary['passive_entry_place']}, modify={summary['passive_entry_modify']}, cancel={summary['passive_entry_cancel']}",
            "===========================================",
            "",
        ]
    )


def render_report(summary: dict) -> str:
    return f"""# BitMEX Other HFT Strategy Report

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

## Signal Activity

- long signal samples: `{summary['long_signal_samples']}`
- short signal samples: `{summary['short_signal_samples']}`
- avg abs alpha: `{summary['avg_abs_alpha']:,.6f}`

## Execution

- taker entry orders: `{summary['taker_entry_orders']}`
- taker exit orders: `{summary['taker_exit_orders']}`
- passive entry place: `{summary['passive_entry_place']}`
- passive entry modify: `{summary['passive_entry_modify']}`
- passive entry cancel: `{summary['passive_entry_cancel']}`
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Educational non-ladder BitMEX HFT strategy backtests.")
    parser.add_argument("--strategy", default=STRATEGY_SLUGS[STRATEGY_MODE], choices=list(STRATEGY_SLUGS.values()))
    parser.add_argument("--symbol", default=BITMEX_SYMBOL)
    parser.add_argument("--asset-type", choices=("linear", "inverse"), default=None)
    parser.add_argument("--contract-size", type=float, default=None)
    parser.add_argument("--dates", nargs="+", default=list(DEFAULT_DATES))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--buffer-rows", type=int, default=None)
    parser.add_argument("--result-tag", default="")
    parser.add_argument("--exchange-model", choices=("no_partial", "strict_no_partial", "partial"), default=EXCHANGE_MODEL)
    parser.add_argument("--order-qty-contracts", type=float, default=ORDER_QTY)
    parser.add_argument("--max-position-contracts", type=float, default=MAX_POSITION_CONTRACTS)
    parser.add_argument("--maker-fee-rate", type=float, default=MAKER_FEE_RATE)
    parser.add_argument("--taker-fee-rate", type=float, default=TAKER_FEE_RATE)
    parser.add_argument("--max-hold-ms", type=float, default=MAX_HOLD_NS / 1_000_000.0)
    parser.add_argument("--cooldown-ms", type=float, default=COOLDOWN_NS / 1_000_000.0)
    parser.add_argument("--flow-threshold", type=float, default=FLOW_SIGNAL_THRESHOLD)
    parser.add_argument("--flow-min-qty", type=float, default=FLOW_MIN_QTY)
    parser.add_argument("--queue-threshold", type=float, default=QUEUE_IMBALANCE_THRESHOLD)
    parser.add_argument("--fade-lookback-ms", type=float, default=LIQUIDITY_FADE_LOOKBACK_NS / 1_000_000.0)
    parser.add_argument("--fade-drop-ratio", type=float, default=LIQUIDITY_FADE_DROP_RATIO)
    parser.add_argument("--reversion-move-bps", type=float, default=MEAN_REVERSION_MOVE_BPS)
    parser.add_argument("--target-profit-bps", type=float, default=TARGET_PROFIT_BPS)
    parser.add_argument("--stop-loss-bps", type=float, default=STOP_LOSS_BPS)
    parser.add_argument("--taker-slippage-bps", type=float, default=TAKER_SLIPPAGE_BPS)
    parser.add_argument("--passive-alpha-threshold", type=float, default=PASSIVE_ALPHA_THRESHOLD)
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    global STRATEGY_MODE, BITMEX_SYMBOL, BITMEX_IS_INVERSE, BITMEX_CONTRACT_SIZE
    global ORDER_QTY, MAX_POSITION_CONTRACTS, MAKER_FEE_RATE, TAKER_FEE_RATE, EXCHANGE_MODEL, RESULT_TAG
    global MAX_HOLD_NS, COOLDOWN_NS, FLOW_SIGNAL_THRESHOLD, FLOW_MIN_QTY, QUEUE_IMBALANCE_THRESHOLD
    global LIQUIDITY_FADE_LOOKBACK_NS, LIQUIDITY_FADE_DROP_RATIO, MEAN_REVERSION_MOVE_BPS
    global TARGET_PROFIT_BPS, STOP_LOSS_BPS, TAKER_SLIPPAGE_BPS, PASSIVE_ALPHA_THRESHOLD

    STRATEGY_MODE = parse_strategy(args.strategy)
    BITMEX_SYMBOL = args.symbol
    asset_type = args.asset_type or ("inverse" if args.symbol.upper() == "XBTUSD" else "linear")
    BITMEX_IS_INVERSE = asset_type == "inverse"
    if args.contract_size is not None:
        BITMEX_CONTRACT_SIZE = args.contract_size
    elif BITMEX_IS_INVERSE:
        BITMEX_CONTRACT_SIZE = 1.0
    else:
        BITMEX_CONTRACT_SIZE = 0.000001

    ORDER_QTY = args.order_qty_contracts
    MAX_POSITION_CONTRACTS = args.max_position_contracts
    MAKER_FEE_RATE = args.maker_fee_rate
    TAKER_FEE_RATE = args.taker_fee_rate
    EXCHANGE_MODEL = args.exchange_model
    RESULT_TAG = args.result_tag
    MAX_HOLD_NS = int(args.max_hold_ms * 1_000_000)
    COOLDOWN_NS = int(args.cooldown_ms * 1_000_000)
    FLOW_SIGNAL_THRESHOLD = args.flow_threshold
    FLOW_MIN_QTY = args.flow_min_qty
    QUEUE_IMBALANCE_THRESHOLD = args.queue_threshold
    LIQUIDITY_FADE_LOOKBACK_NS = int(args.fade_lookback_ms * 1_000_000)
    LIQUIDITY_FADE_DROP_RATIO = args.fade_drop_ratio
    MEAN_REVERSION_MOVE_BPS = args.reversion_move_bps
    TARGET_PROFIT_BPS = args.target_profit_bps
    STOP_LOSS_BPS = args.stop_loss_bps
    TAKER_SLIPPAGE_BPS = args.taker_slippage_bps
    PASSIVE_ALPHA_THRESHOLD = args.passive_alpha_threshold


def main() -> None:
    args = parse_args()
    apply_args(args)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    key = None if args.skip_download else tardis_key()
    outputs = []
    for yyyymmdd in args.dates:
        if not args.skip_download:
            download_file(BITMEX_EXCHANGE, "trades", BITMEX_SYMBOL, yyyymmdd, key)
            download_file(BITMEX_EXCHANGE, "incremental_book_L2", BITMEX_SYMBOL, yyyymmdd, key)
        bitmex_npz = convert_bitmex(BITMEX_SYMBOL, yyyymmdd, args.buffer_rows)
        outputs.append(run_backtest(bitmex_npz, yyyymmdd))
    print("all_results=" + ",".join(str(path) for path in outputs))


def main_with_strategy(strategy: str) -> None:
    if "--strategy" not in sys.argv:
        sys.argv.extend(["--strategy", strategy])
    main()


if __name__ == "__main__":
    main()
