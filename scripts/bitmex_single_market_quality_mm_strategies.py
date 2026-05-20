import argparse
import json
import math
from pathlib import Path

import numpy as np
from numba import njit

from hftbacktest import (
    ALL_ASSETS,
    BacktestAsset,
    GTX,
    LIMIT,
    HashMapMarketDepthBacktest,
    Recorder,
)
from hftbacktest.order import IOC, MARKET

from bitmex_single_market_mm_backtest import (
    BITMEX_CONTRACT_SIZE,
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


STRATEGY_SELECTIVE_MAKER = 1
STRATEGY_REBATE_GATED_MAKER = 2
STRATEGY_COOLDOWN_INVENTORY_MAKER = 3

STRATEGY_SLUGS = {
    STRATEGY_SELECTIVE_MAKER: "selective_maker",
    STRATEGY_REBATE_GATED_MAKER: "rebate_gated_maker",
    STRATEGY_COOLDOWN_INVENTORY_MAKER: "cooldown_inventory_maker",
}

STRATEGY_MODE = STRATEGY_SELECTIVE_MAKER

ORDER_UPDATE_INTERVAL_NS = 10_000_000
ORDER_INFLIGHT_NS = 120_000_000
REST_MIN_INTERVAL_NS = 700_000_000
ORDER_TTL_NS = 5_000_000_000
MIN_AMEND_TICKS = 5.0

SIGNAL_HISTORY_LEN = 4096
MOMENTUM_WINDOW_NS = 250_000_000
VOL_WINDOW_NS = 1_000_000_000

ORDER_QTY = BITMEX_ORDER_QTY
SOFT_POSITION_CONTRACTS = 500.0
MAX_POSITION_CONTRACTS = 1_000.0
MAKER_FEE_RATE = -0.0002
TAKER_FEE_RATE = 0.0001

# selective_maker: only quotes when short-term book state is not hostile.
SELECTIVE_HALF_SPREAD_BPS = 4.0
SELECTIVE_MAX_VOL_BPS = 2.0
SELECTIVE_MAX_ADVERSE_MOMENTUM_BPS = 0.35
SELECTIVE_MAX_ADVERSE_MICROPRICE_BPS = 0.25
SELECTIVE_MAX_OPPOSING_IMBALANCE = 0.25

# rebate_gated_maker: quote only when expected edge clears a threshold.
REBATE_BASE_HALF_SPREAD_BPS = 2.5
REBATE_EDGE_THRESHOLD_BPS = 1.2
REBATE_VOL_PENALTY_MULT = 0.6
REBATE_MOMENTUM_PENALTY_MULT = 1.2
REBATE_MICROPRICE_PENALTY_MULT = 1.0
REBATE_MAX_ADVERSE_BPS = 4.0

# cooldown_inventory_maker: after a fill, do not immediately refresh the same side.
COOLDOWN_HALF_SPREAD_BPS = 3.5
COOLDOWN_AFTER_FILL_NS = 3_000_000_000
COOLDOWN_VOL_RISK_OFF_BPS = 4.0
COOLDOWN_INVENTORY_HARD_RATIO = 0.75
COOLDOWN_INVENTORY_SKEW_BPS = 5.0

EXCHANGE_MODEL = "live_l2"
LIVE_L2_TRADE_THROUGH_PROBABILITY = 0.145
LIVE_L2_MIN_ORDER_AGE_NS = 0
RESULT_TAG = ""


@njit
def ceil_to_tick(px, tick_size):
    return math.ceil(px / tick_size) * tick_size


@njit
def floor_to_tick(px, tick_size):
    return math.floor(px / tick_size) * tick_size


@njit
def clamp(value, low, high):
    return min(max(value, low), high)


@njit
def ratio_minus_one_bps(numerator, denominator):
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return math.nan
    return (numerator / denominator - 1.0) * 10_000.0


@njit
def bitmex_equity_usdt(hbt):
    depth = hbt.depth(0)
    mid = (depth.best_bid + depth.best_ask) / 2.0
    state = hbt.state_values(0)
    return state.balance + state.position * mid * BITMEX_CONTRACT_SIZE - state.fee


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
def record_signal(
    hbt,
    signal_ts,
    signal_bid,
    signal_ask,
    signal_bid_qty,
    signal_ask_qty,
    write_idx,
    count,
):
    depth = hbt.depth(0)
    if depth.best_bid <= 0 or depth.best_ask <= 0:
        return write_idx, count
    signal_ts[write_idx] = hbt.current_timestamp
    signal_bid[write_idx] = depth.best_bid
    signal_ask[write_idx] = depth.best_ask
    signal_bid_qty[write_idx] = depth.best_bid_qty
    signal_ask_qty[write_idx] = depth.best_ask_qty
    write_idx = (write_idx + 1) % SIGNAL_HISTORY_LEN
    count = min(count + 1, SIGNAL_HISTORY_LEN)
    return write_idx, count


@njit
def signal_at_or_before(signal_ts, write_idx, count, target_ts):
    for offset in range(count):
        idx = (write_idx - 1 - offset) % SIGNAL_HISTORY_LEN
        if signal_ts[idx] <= target_ts:
            return idx
    return -1


@njit
def mid_from_history(signal_bid, signal_ask, idx):
    if idx < 0:
        return 0.0
    return (signal_bid[idx] + signal_ask[idx]) / 2.0


@njit
def latest_idx(write_idx):
    return (write_idx - 1) % SIGNAL_HISTORY_LEN


@njit
def recent_move_bps(signal_ts, signal_bid, signal_ask, write_idx, count, window_ns):
    if count <= 1:
        return 0.0
    cur_idx = latest_idx(write_idx)
    past_idx = signal_at_or_before(signal_ts, write_idx, count, signal_ts[cur_idx] - window_ns)
    if past_idx < 0:
        return 0.0
    move = ratio_minus_one_bps(
        mid_from_history(signal_bid, signal_ask, cur_idx),
        mid_from_history(signal_bid, signal_ask, past_idx),
    )
    if not math.isfinite(move):
        return 0.0
    return move


@njit
def recent_vol_bps(signal_ts, signal_bid, signal_ask, write_idx, count):
    return abs(recent_move_bps(signal_ts, signal_bid, signal_ask, write_idx, count, VOL_WINDOW_NS))


@njit
def microprice_bps(depth):
    mid = (depth.best_bid + depth.best_ask) / 2.0
    total_qty = depth.best_bid_qty + depth.best_ask_qty
    if total_qty <= 0 or mid <= 0:
        return 0.0
    micro = (depth.best_ask * depth.best_bid_qty + depth.best_bid * depth.best_ask_qty) / total_qty
    return ratio_minus_one_bps(micro, mid)


@njit
def imbalance(depth):
    total_qty = depth.best_bid_qty + depth.best_ask_qty
    if total_qty <= 0:
        return 0.0
    return (depth.best_bid_qty - depth.best_ask_qty) / total_qty


@njit
def inventory_ratio(pos):
    if SOFT_POSITION_CONTRACTS <= 0:
        return 0.0
    return clamp(pos / SOFT_POSITION_CONTRACTS, -1.0, 1.0)


@njit
def side_adverse_signal(side, momentum_bps, micro_bps):
    if side > 0:
        return max(0.0, -momentum_bps), max(0.0, -micro_bps)
    return max(0.0, momentum_bps), max(0.0, micro_bps)


@njit
def price_from_half_spread(side, mid, half_spread_bps, inv_ratio, depth):
    reservation = mid * (1.0 - inv_ratio * COOLDOWN_INVENTORY_SKEW_BPS / 10_000.0)
    if side > 0:
        raw_px = reservation * (1.0 - half_spread_bps / 10_000.0)
        return floor_to_tick(min(raw_px, depth.best_bid), BITMEX_TICK_SIZE)
    raw_px = reservation * (1.0 + half_spread_bps / 10_000.0)
    return ceil_to_tick(max(raw_px, depth.best_ask), BITMEX_TICK_SIZE)


@njit
def selective_maker_price(side, depth, pos, signal_ts, signal_bid, signal_ask, write_idx, count):
    mid = (depth.best_bid + depth.best_ask) / 2.0
    momentum_bps = recent_move_bps(signal_ts, signal_bid, signal_ask, write_idx, count, MOMENTUM_WINDOW_NS)
    vol_bps = recent_vol_bps(signal_ts, signal_bid, signal_ask, write_idx, count)
    micro_bps = microprice_bps(depth)
    imb = imbalance(depth)
    adverse_momentum, adverse_micro = side_adverse_signal(side, momentum_bps, micro_bps)
    opposing_imbalance = -imb if side > 0 else imb

    if vol_bps > SELECTIVE_MAX_VOL_BPS:
        return 0.0, 1.0
    if adverse_momentum > SELECTIVE_MAX_ADVERSE_MOMENTUM_BPS:
        return 0.0, 2.0
    if adverse_micro > SELECTIVE_MAX_ADVERSE_MICROPRICE_BPS:
        return 0.0, 3.0
    if opposing_imbalance > SELECTIVE_MAX_OPPOSING_IMBALANCE:
        return 0.0, 4.0
    if side > 0 and pos + ORDER_QTY > MAX_POSITION_CONTRACTS:
        return 0.0, 5.0
    if side < 0 and pos - ORDER_QTY < -MAX_POSITION_CONTRACTS:
        return 0.0, 5.0

    half_spread = SELECTIVE_HALF_SPREAD_BPS + vol_bps * 0.5
    return price_from_half_spread(side, mid, half_spread, inventory_ratio(pos), depth), 0.0


@njit
def rebate_gated_price(side, depth, pos, signal_ts, signal_bid, signal_ask, write_idx, count):
    mid = (depth.best_bid + depth.best_ask) / 2.0
    momentum_bps = recent_move_bps(signal_ts, signal_bid, signal_ask, write_idx, count, MOMENTUM_WINDOW_NS)
    vol_bps = recent_vol_bps(signal_ts, signal_bid, signal_ask, write_idx, count)
    micro_bps = microprice_bps(depth)
    adverse_momentum, adverse_micro = side_adverse_signal(side, momentum_bps, micro_bps)

    adverse_est = (
        vol_bps * REBATE_VOL_PENALTY_MULT
        + adverse_momentum * REBATE_MOMENTUM_PENALTY_MULT
        + adverse_micro * REBATE_MICROPRICE_PENALTY_MULT
    )
    if adverse_est > REBATE_MAX_ADVERSE_BPS:
        return 0.0, 1.0
    if side > 0 and pos + ORDER_QTY > MAX_POSITION_CONTRACTS:
        return 0.0, 5.0
    if side < 0 and pos - ORDER_QTY < -MAX_POSITION_CONTRACTS:
        return 0.0, 5.0

    maker_rebate_bps = max(0.0, -MAKER_FEE_RATE * 10_000.0)
    half_spread = REBATE_BASE_HALF_SPREAD_BPS + adverse_est * 0.5
    expected_edge = half_spread + maker_rebate_bps - adverse_est
    if expected_edge < REBATE_EDGE_THRESHOLD_BPS:
        return 0.0, 6.0
    return price_from_half_spread(side, mid, half_spread, inventory_ratio(pos), depth), 0.0


@njit
def cooldown_inventory_price(
    side,
    depth,
    pos,
    side_cooldown_until,
    signal_ts,
    signal_bid,
    signal_ask,
    write_idx,
    count,
    now_ts,
):
    if now_ts < side_cooldown_until:
        return 0.0, 7.0
    inv = inventory_ratio(pos)
    if inv >= COOLDOWN_INVENTORY_HARD_RATIO and side > 0:
        return 0.0, 8.0
    if inv <= -COOLDOWN_INVENTORY_HARD_RATIO and side < 0:
        return 0.0, 8.0
    if side > 0 and pos + ORDER_QTY > MAX_POSITION_CONTRACTS:
        return 0.0, 5.0
    if side < 0 and pos - ORDER_QTY < -MAX_POSITION_CONTRACTS:
        return 0.0, 5.0

    vol_bps = recent_vol_bps(signal_ts, signal_bid, signal_ask, write_idx, count)
    if vol_bps > COOLDOWN_VOL_RISK_OFF_BPS:
        if pos > 0 and side < 0:
            pass
        elif pos < 0 and side > 0:
            pass
        else:
            return 0.0, 1.0

    mid = (depth.best_bid + depth.best_ask) / 2.0
    half_spread = COOLDOWN_HALF_SPREAD_BPS + vol_bps * 0.5 + abs(inv) * 1.5
    return price_from_half_spread(side, mid, half_spread, inv, depth), 0.0


@njit
def target_price(side, hbt, buy_cooldown_until, sell_cooldown_until, signal_ts, signal_bid, signal_ask, write_idx, count):
    depth = hbt.depth(0)
    if depth.best_bid <= 0 or depth.best_ask <= 0:
        return 0.0, 9.0
    pos = hbt.position(0)
    if STRATEGY_MODE == STRATEGY_SELECTIVE_MAKER:
        return selective_maker_price(side, depth, pos, signal_ts, signal_bid, signal_ask, write_idx, count)
    if STRATEGY_MODE == STRATEGY_REBATE_GATED_MAKER:
        return rebate_gated_price(side, depth, pos, signal_ts, signal_bid, signal_ask, write_idx, count)
    cooldown_until = buy_cooldown_until if side > 0 else sell_cooldown_until
    return cooldown_inventory_price(
        side,
        depth,
        pos,
        cooldown_until,
        signal_ts,
        signal_bid,
        signal_ask,
        write_idx,
        count,
        hbt.current_timestamp,
    )


@njit
def rest_ready(hbt, next_rest_allowed_ts, metrics, side):
    if hbt.current_timestamp < next_rest_allowed_ts:
        if side > 0:
            metrics[20] += 1
        else:
            metrics[21] += 1
        return False
    return True


@njit
def manage_side(
    hbt,
    side,
    order_id,
    inflight_until,
    next_rest_allowed_ts,
    live_since,
    buy_cooldown_until,
    sell_cooldown_until,
    signal_ts,
    signal_bid,
    signal_ask,
    write_idx,
    count,
    metrics,
):
    if hbt.current_timestamp < inflight_until:
        return inflight_until, next_rest_allowed_ts, live_since

    existing = hbt.orders(0).get(order_id)
    if existing is not None and live_since > 0 and hbt.current_timestamp - live_since >= ORDER_TTL_NS:
        if existing.cancellable and rest_ready(hbt, next_rest_allowed_ts, metrics, side):
            hbt.cancel(0, order_id, False)
            if side > 0:
                metrics[10] += 1
            else:
                metrics[11] += 1
            return hbt.current_timestamp + ORDER_INFLIGHT_NS, hbt.current_timestamp + REST_MIN_INTERVAL_NS, 0
        return inflight_until, next_rest_allowed_ts, live_since

    px, reason = target_price(
        side,
        hbt,
        buy_cooldown_until,
        sell_cooldown_until,
        signal_ts,
        signal_bid,
        signal_ask,
        write_idx,
        count,
    )
    if px <= 0:
        if reason > 0:
            reason_idx = int(reason)
            if 0 <= reason_idx < 10:
                metrics[30 + reason_idx] += 1
        if existing is not None and existing.cancellable and rest_ready(hbt, next_rest_allowed_ts, metrics, side):
            hbt.cancel(0, order_id, False)
            if side > 0:
                metrics[12] += 1
            else:
                metrics[13] += 1
            return hbt.current_timestamp + ORDER_INFLIGHT_NS, hbt.current_timestamp + REST_MIN_INTERVAL_NS, 0
        return inflight_until, next_rest_allowed_ts, live_since

    if existing is not None:
        price_changed = abs(existing.price - px) >= MIN_AMEND_TICKS * BITMEX_TICK_SIZE
        if existing.cancellable and price_changed and rest_ready(hbt, next_rest_allowed_ts, metrics, side):
            hbt.modify(0, order_id, px, ORDER_QTY, False)
            if side > 0:
                metrics[14] += 1
            else:
                metrics[15] += 1
            return hbt.current_timestamp + ORDER_INFLIGHT_NS, hbt.current_timestamp + REST_MIN_INTERVAL_NS, hbt.current_timestamp
        return inflight_until, next_rest_allowed_ts, live_since

    if rest_ready(hbt, next_rest_allowed_ts, metrics, side):
        if side > 0:
            hbt.submit_buy_order(0, order_id, px, ORDER_QTY, GTX, LIMIT, False)
            metrics[16] += 1
        else:
            hbt.submit_sell_order(0, order_id, px, ORDER_QTY, GTX, LIMIT, False)
            metrics[17] += 1
        return hbt.current_timestamp + ORDER_INFLIGHT_NS, hbt.current_timestamp + REST_MIN_INTERVAL_NS, hbt.current_timestamp

    return inflight_until, next_rest_allowed_ts, live_since


@njit
def update_risk_metrics(hbt, metrics):
    depth = hbt.depth(0)
    if depth.best_bid <= 0 or depth.best_ask <= 0:
        return
    mid = (depth.best_bid + depth.best_ask) / 2.0
    pos = abs(hbt.position(0))
    metrics[4] = max(metrics[4], pos)
    metrics[5] = max(metrics[5], pos * BITMEX_CONTRACT_SIZE)
    equity = bitmex_equity_usdt(hbt)
    if not math.isfinite(equity):
        return
    if metrics[6] == 0.0 and metrics[7] == 0.0:
        metrics[6] = equity
        metrics[7] = equity
    metrics[6] = max(metrics[6], equity)
    metrics[7] = min(metrics[7], equity)


@njit
def force_flatten(hbt, metrics):
    cancel_all_orders(hbt)
    hbt.elapse(1_000_000_000)
    pos = hbt.position(0)
    if abs(pos) <= 0:
        return
    before = bitmex_equity_usdt(hbt)
    depth = hbt.depth(0)
    if pos > 0:
        hbt.submit_sell_order(0, 91_001, depth.best_bid, abs(pos), IOC, MARKET, True)
    else:
        hbt.submit_buy_order(0, 91_002, depth.best_ask, abs(pos), IOC, MARKET, True)
    hbt.elapse(1_000_000_000)
    metrics[8] = bitmex_equity_usdt(hbt) - before


@njit
def run_strategy(hbt, recorder, metrics, end_close_ts):
    bid_id = 11_001
    ask_id = 21_001
    bid_inflight_until = 0
    ask_inflight_until = 0
    bid_live_since = 0
    ask_live_since = 0
    next_rest_allowed_ts = 0
    buy_cooldown_until = 0
    sell_cooldown_until = 0
    last_pos = hbt.position(0)
    last_trades = hbt.state_values(0).num_trades
    last_record_ts = 0

    signal_ts = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.int64)
    signal_bid = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    signal_ask = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    signal_bid_qty = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    signal_ask_qty = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    write_idx = 0
    count = 0

    while hbt.elapse(ORDER_UPDATE_INTERVAL_NS) == 0:
        if hbt.current_timestamp >= end_close_ts:
            break
        hbt.clear_inactive_orders(ALL_ASSETS)
        write_idx, count = record_signal(
            hbt,
            signal_ts,
            signal_bid,
            signal_ask,
            signal_bid_qty,
            signal_ask_qty,
            write_idx,
            count,
        )

        bid_inflight_until, next_rest_allowed_ts, bid_live_since = manage_side(
            hbt,
            1,
            bid_id,
            bid_inflight_until,
            next_rest_allowed_ts,
            bid_live_since,
            buy_cooldown_until,
            sell_cooldown_until,
            signal_ts,
            signal_bid,
            signal_ask,
            write_idx,
            count,
            metrics,
        )
        ask_inflight_until, next_rest_allowed_ts, ask_live_since = manage_side(
            hbt,
            -1,
            ask_id,
            ask_inflight_until,
            next_rest_allowed_ts,
            ask_live_since,
            buy_cooldown_until,
            sell_cooldown_until,
            signal_ts,
            signal_bid,
            signal_ask,
            write_idx,
            count,
            metrics,
        )

        state = hbt.state_values(0)
        if state.num_trades > last_trades:
            delta_pos = state.position - last_pos
            trade_count = state.num_trades - last_trades
            metrics[0] += trade_count
            if delta_pos > 0:
                metrics[1] += trade_count
                buy_cooldown_until = hbt.current_timestamp + COOLDOWN_AFTER_FILL_NS
            elif delta_pos < 0:
                metrics[2] += trade_count
                sell_cooldown_until = hbt.current_timestamp + COOLDOWN_AFTER_FILL_NS
            metrics[3] += abs(delta_pos) * BITMEX_CONTRACT_SIZE
            last_pos = state.position
            last_trades = state.num_trades

        update_risk_metrics(hbt, metrics)
        if hbt.current_timestamp - last_record_ts >= 1_000_000_000:
            recorder.record(hbt)
            last_record_ts = hbt.current_timestamp

    force_flatten(hbt, metrics)
    recorder.record(hbt)
    return True


def build_asset(bitmex_npz: Path):
    asset = (
        BacktestAsset()
        .data([str(bitmex_npz)])
        .linear_asset(BITMEX_CONTRACT_SIZE)
        .constant_order_latency(BITMEX_ORDER_ENTRY_LATENCY_NS, BITMEX_ORDER_RESPONSE_LATENCY_NS)
        .risk_adverse_queue_model()
        .trading_value_fee_model(MAKER_FEE_RATE, TAKER_FEE_RATE)
        .tick_size(BITMEX_TICK_SIZE)
        .lot_size(BITMEX_LOT_SIZE)
        .last_trades_capacity(10_000)
    )
    if EXCHANGE_MODEL == "live_l2":
        return asset.live_l2_no_partial_fill_exchange(
            LIVE_L2_TRADE_THROUGH_PROBABILITY,
            LIVE_L2_MIN_ORDER_AGE_NS,
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
    metrics = np.zeros(48, dtype=np.float64)
    ok = run_strategy(hbt, recorder.recorder, metrics, end_close_ts_ns(yyyymmdd))
    if not ok:
        raise RuntimeError("strategy returned false")

    slug = STRATEGY_SLUGS[STRATEGY_MODE]
    tag = RESULT_TAG or f"{slug}_{EXCHANGE_MODEL}"
    out = RESULT_DIR / f"bitmex_xbtusdt_quality_mm_{tag}_{yyyymmdd}.npz"
    np.savez_compressed(out, **{"0": recorder.get(0), "metrics": metrics})
    write_summary(out, yyyymmdd)
    return out


def signed_money(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.4f} USDT"


def write_summary(result_npz: Path, yyyymmdd: str) -> None:
    data = np.load(result_npz)
    records = data["0"]
    metrics = data["metrics"]
    final = records[-1]
    final_price = float(final["price"])
    final_pos = float(final["position"])
    equity = float(final["balance"]) + final_pos * final_price * BITMEX_CONTRACT_SIZE - float(final["fee"])
    total_fee = float(final["fee"])
    gross = equity + total_fee
    summary = {
        "date": yyyymmdd,
        "symbol": BITMEX_SYMBOL,
        "strategy": STRATEGY_SLUGS[STRATEGY_MODE],
        "exchange_model": EXCHANGE_MODEL,
        "maker_fee_rate": MAKER_FEE_RATE,
        "taker_fee_rate": TAKER_FEE_RATE,
        "total_pnl_usdt": equity,
        "gross_pnl_before_fee_usdt": gross,
        "fee_usdt": total_fee,
        "maker_rebate_usdt": -total_fee,
        "fills": int(metrics[0]),
        "buy_fills": int(metrics[1]),
        "sell_fills": int(metrics[2]),
        "filled_base_btc": float(metrics[3]),
        "max_position_contracts_seen": float(metrics[4]),
        "max_position_btc_seen": float(metrics[5]),
        "max_equity_usdt": float(metrics[6]),
        "min_equity_usdt": float(metrics[7]),
        "force_close_pnl_usdt": float(metrics[8]),
        "ttl_cancel_bid": int(metrics[10]),
        "ttl_cancel_ask": int(metrics[11]),
        "gate_cancel_bid": int(metrics[12]),
        "gate_cancel_ask": int(metrics[13]),
        "modify_bid": int(metrics[14]),
        "modify_ask": int(metrics[15]),
        "place_bid": int(metrics[16]),
        "place_ask": int(metrics[17]),
        "rest_skip_bid": int(metrics[20]),
        "rest_skip_ask": int(metrics[21]),
        "gate_vol": int(metrics[31]),
        "gate_momentum": int(metrics[32]),
        "gate_microprice": int(metrics[33]),
        "gate_imbalance": int(metrics[34]),
        "gate_position": int(metrics[35]),
        "gate_edge": int(metrics[36]),
        "gate_cooldown": int(metrics[37]),
        "gate_inventory": int(metrics[38]),
        "gate_bad_book": int(metrics[39]),
        "final_position_contracts": final_pos,
        "records": int(len(records)),
    }
    result_npz.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    result_npz.with_suffix(".report.md").write_text(render_report(summary))
    print(render_console_summary(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


def render_console_summary(summary: dict) -> str:
    return "\n".join(
        [
            "",
            "========== BitMEX quality maker 回测 ==========",
            f"策略: {summary['strategy']}",
            f"日期: {summary['date']}",
            f"PnL: {signed_money(summary['total_pnl_usdt'])}",
            f"gross 不含手续费: {signed_money(summary['gross_pnl_before_fee_usdt'])}",
            f"手续费/返佣贡献: {signed_money(summary['maker_rebate_usdt'])}",
            f"fills: {summary['fills']}，买={summary['buy_fills']}，卖={summary['sell_fills']}",
            f"最大仓位: {summary['max_position_contracts_seen']:,.0f} contracts",
            f"gate: vol={summary['gate_vol']}, momentum={summary['gate_momentum']}, micro={summary['gate_microprice']}, edge={summary['gate_edge']}, cooldown={summary['gate_cooldown']}",
            "==============================================",
            "",
        ]
    )


def render_report(summary: dict) -> str:
    return f"""# BitMEX Quality Maker Strategy Report

## Result

- strategy: `{summary['strategy']}`
- date: `{summary['date']}`
- exchange model: `{summary['exchange_model']}`
- total PnL: `{signed_money(summary['total_pnl_usdt'])}`
- gross before fee: `{signed_money(summary['gross_pnl_before_fee_usdt'])}`
- fee/rebate contribution: `{signed_money(summary['maker_rebate_usdt'])}`
- fills: `{summary['fills']}`
- buy fills: `{summary['buy_fills']}`
- sell fills: `{summary['sell_fills']}`
- filled base: `{summary['filled_base_btc']:,.8f} BTC`
- max position: `{summary['max_position_contracts_seen']:,.0f} contracts`

## Gates

- volatility gate: `{summary['gate_vol']}`
- momentum gate: `{summary['gate_momentum']}`
- microprice gate: `{summary['gate_microprice']}`
- imbalance gate: `{summary['gate_imbalance']}`
- position gate: `{summary['gate_position']}`
- expected-edge gate: `{summary['gate_edge']}`
- cooldown gate: `{summary['gate_cooldown']}`
- inventory gate: `{summary['gate_inventory']}`
- bad-book gate: `{summary['gate_bad_book']}`
"""


def parse_strategy(value: str) -> int:
    normalized = value.lower().replace("-", "_")
    for mode, slug in STRATEGY_SLUGS.items():
        if normalized == slug:
            return mode
    raise ValueError(f"unknown strategy: {value}; choices={list(STRATEGY_SLUGS.values())}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Educational BitMEX single-market quality-maker strategies.")
    parser.add_argument("--strategy", default=STRATEGY_SLUGS[STRATEGY_MODE], choices=list(STRATEGY_SLUGS.values()))
    parser.add_argument("--dates", nargs="+", default=list(DEFAULT_DATES))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--buffer-rows", type=int, default=None)
    parser.add_argument("--maker-fee-rate", type=float, default=MAKER_FEE_RATE)
    parser.add_argument("--taker-fee-rate", type=float, default=TAKER_FEE_RATE)
    parser.add_argument("--exchange-model", choices=("live_l2", "no_partial", "strict_no_partial", "partial"), default=EXCHANGE_MODEL)
    parser.add_argument("--live-l2-trade-through-probability", type=float, default=LIVE_L2_TRADE_THROUGH_PROBABILITY)
    parser.add_argument("--result-tag", default="")
    return parser.parse_args()


def main() -> None:
    global STRATEGY_MODE
    global MAKER_FEE_RATE
    global TAKER_FEE_RATE
    global EXCHANGE_MODEL
    global LIVE_L2_TRADE_THROUGH_PROBABILITY
    global RESULT_TAG

    args = parse_args()
    STRATEGY_MODE = parse_strategy(args.strategy)
    MAKER_FEE_RATE = args.maker_fee_rate
    TAKER_FEE_RATE = args.taker_fee_rate
    EXCHANGE_MODEL = args.exchange_model
    LIVE_L2_TRADE_THROUGH_PROBABILITY = args.live_l2_trade_through_probability
    RESULT_TAG = args.result_tag

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


if __name__ == "__main__":
    main()
