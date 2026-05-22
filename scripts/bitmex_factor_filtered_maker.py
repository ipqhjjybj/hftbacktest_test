import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from numba import njit

from hftbacktest import (
    ALL_ASSETS,
    BUY_EVENT,
    SELL_EVENT,
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
    NPZ_DIR,
    RESULT_DIR,
    convert_bitmex,
    download_file,
    end_close_ts_ns,
    tardis_key,
)


ORDER_UPDATE_INTERVAL_NS = 10_000_000
ORDER_INFLIGHT_NS = 80_000_000
REST_MIN_INTERVAL_NS = 700_000_000
ORDER_TTL_NS = 5_000_000_000
MIN_AMEND_TICKS = 5.0

SIGNAL_HISTORY_LEN = 8192
FLOW_DECAY = 0.92

ORDER_QTY = BITMEX_ORDER_QTY
BASE_HALF_SPREAD_BPS = 3.0
MIN_HALF_SPREAD_TICKS = 1.0
VOL_SPREAD_MULTIPLIER = 0.3
INVENTORY_SKEW_BPS_AT_SOFT_LIMIT = 4.0
INVENTORY_SPREAD_BPS_AT_SOFT_LIMIT = 1.5
SOFT_POSITION_CONTRACTS = 500.0
MAX_POSITION_CONTRACTS = 1_000.0

MAKER_FEE_RATE = 0.0
TAKER_FEE_RATE = 0.0001
EXCHANGE_MODEL = "no_partial"
RESULT_TAG = ""

FACTOR_IDS = {
    "spread_bps": 0,
    "queue_imbalance": 1,
    "microprice_bps": 2,
    "trade_flow_imbalance": 3,
    "trade_flow_ewm_imbalance": 4,
    "momentum_100ms_bps": 5,
    "momentum_250ms_bps": 6,
    "momentum_1000ms_bps": 7,
    "vol_1000ms_bps": 8,
}


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
        return 0.0
    return (numerator / denominator - 1.0) * 10_000.0


@njit
def bitmex_equity_usdt(hbt):
    depth = hbt.depth(0)
    mid = (depth.best_bid + depth.best_ask) / 2.0
    state = hbt.state_values(0)
    return state.balance + state.position * mid * BITMEX_CONTRACT_SIZE - state.fee


@njit
def cancel_all_orders(hbt):
    orders = hbt.orders(0)
    values = orders.values()
    while values.has_next():
        order = values.get()
        if order.cancellable:
            hbt.cancel(0, order.order_id, False)


@njit
def record_signal(hbt, signal_ts, signal_bid, signal_ask, write_idx, count):
    depth = hbt.depth(0)
    if depth.best_bid <= 0 or depth.best_ask <= 0:
        return write_idx, count
    signal_ts[write_idx] = hbt.current_timestamp
    signal_bid[write_idx] = depth.best_bid
    signal_ask[write_idx] = depth.best_ask
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
def recent_move_bps(signal_ts, signal_bid, signal_ask, write_idx, count, window_ns):
    if count <= 1:
        return 0.0
    cur_idx = (write_idx - 1) % SIGNAL_HISTORY_LEN
    past_idx = signal_at_or_before(signal_ts, write_idx, count, signal_ts[cur_idx] - window_ns)
    if past_idx < 0:
        return 0.0
    return ratio_minus_one_bps(mid_from_history(signal_bid, signal_ask, cur_idx), mid_from_history(signal_bid, signal_ask, past_idx))


@njit
def microprice_bps(depth):
    mid = (depth.best_bid + depth.best_ask) / 2.0
    total_qty = depth.best_bid_qty + depth.best_ask_qty
    if total_qty <= 0 or mid <= 0:
        return 0.0
    micro = (depth.best_ask * depth.best_bid_qty + depth.best_bid * depth.best_ask_qty) / total_qty
    return ratio_minus_one_bps(micro, mid)


@njit
def queue_imbalance(depth):
    total_qty = depth.best_bid_qty + depth.best_ask_qty
    if total_qty <= 0:
        return 0.0
    return (depth.best_bid_qty - depth.best_ask_qty) / total_qty


@njit
def update_trade_flow(hbt, flow):
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
    flow[2] = buy_qty
    flow[3] = sell_qty
    flow[4] += buy_qty
    flow[5] += sell_qty
    flow[6] += buy_count
    flow[7] += sell_count


@njit
def imbalance(buy_qty, sell_qty):
    total = buy_qty + sell_qty
    if total <= 0:
        return 0.0
    return (buy_qty - sell_qty) / total


@njit
def inventory_ratio(pos):
    if SOFT_POSITION_CONTRACTS <= 0:
        return 0.0
    return clamp(pos / SOFT_POSITION_CONTRACTS, -1.0, 1.0)


@njit
def factor_value(factor_id, depth, signal_ts, signal_bid, signal_ask, write_idx, count, flow):
    if factor_id == 0:
        return ratio_minus_one_bps(depth.best_ask, depth.best_bid)
    if factor_id == 1:
        return queue_imbalance(depth)
    if factor_id == 2:
        return microprice_bps(depth)
    if factor_id == 3:
        return imbalance(flow[2], flow[3])
    if factor_id == 4:
        return imbalance(flow[0], flow[1])
    if factor_id == 5:
        return recent_move_bps(signal_ts, signal_bid, signal_ask, write_idx, count, 100_000_000)
    if factor_id == 6:
        return recent_move_bps(signal_ts, signal_bid, signal_ask, write_idx, count, 250_000_000)
    if factor_id == 7:
        return recent_move_bps(signal_ts, signal_bid, signal_ask, write_idx, count, 1_000_000_000)
    if factor_id == 8:
        return abs(recent_move_bps(signal_ts, signal_bid, signal_ask, write_idx, count, 1_000_000_000))
    return 0.0


@njit
def match_rules(side, rule_factor_ids, rule_sides, rule_mins, rule_maxs, rule_leg_counts, depth, signal_ts, signal_bid, signal_ask, write_idx, count, flow):
    for i in range(len(rule_sides)):
        if rule_sides[i] != side:
            continue
        matched = True
        for leg in range(rule_leg_counts[i]):
            factor_id = rule_factor_ids[i, leg]
            value = factor_value(factor_id, depth, signal_ts, signal_bid, signal_ask, write_idx, count, flow)
            if value < rule_mins[i, leg] or value > rule_maxs[i, leg]:
                matched = False
                break
        if matched:
            return True
    return False


@njit
def is_reduce_side(side, pos):
    return (side > 0 and pos < 0) or (side < 0 and pos > 0)


@njit
def should_quote(side, pos, rule_factor_ids, rule_sides, rule_mins, rule_maxs, rule_leg_counts, depth, signal_ts, signal_bid, signal_ask, write_idx, count, flow):
    if ORDER_QTY < BITMEX_LOT_SIZE:
        return False, 1
    if side > 0 and pos + ORDER_QTY > MAX_POSITION_CONTRACTS:
        return False, 2
    if side < 0 and pos - ORDER_QTY < -MAX_POSITION_CONTRACTS:
        return False, 2
    matched = match_rules(
        side,
        rule_factor_ids,
        rule_sides,
        rule_mins,
        rule_maxs,
        rule_leg_counts,
        depth,
        signal_ts,
        signal_bid,
        signal_ask,
        write_idx,
        count,
        flow,
    )
    if not matched and not is_reduce_side(side, pos):
        return False, 3
    return True, 0


@njit
def target_price(side, depth, pos):
    mid = (depth.best_bid + depth.best_ask) / 2.0
    inv = inventory_ratio(pos)
    min_half_spread_bps = MIN_HALF_SPREAD_TICKS * BITMEX_TICK_SIZE / mid * 10_000.0
    vol_bps = 0.0
    half_spread = max(
        BASE_HALF_SPREAD_BPS + VOL_SPREAD_MULTIPLIER * vol_bps + abs(inv) * INVENTORY_SPREAD_BPS_AT_SOFT_LIMIT,
        min_half_spread_bps,
    )
    anchor = mid * (1.0 - inv * INVENTORY_SKEW_BPS_AT_SOFT_LIMIT / 10_000.0)
    if side > 0:
        raw = anchor * (1.0 - half_spread / 10_000.0)
        return floor_to_tick(min(raw, depth.best_bid), BITMEX_TICK_SIZE)
    raw = anchor * (1.0 + half_spread / 10_000.0)
    return ceil_to_tick(max(raw, depth.best_ask), BITMEX_TICK_SIZE)


@njit
def manage_side(hbt, side, order_id, target_px, should_place, inflight_until, next_rest_allowed_ts, live_since, metrics):
    if hbt.current_timestamp < inflight_until:
        return inflight_until, next_rest_allowed_ts, live_since

    existing = hbt.orders(0).get(order_id)
    rest_skip_metric = 18 if side > 0 else 19
    ttl_metric = 10 if side > 0 else 11
    filter_cancel_metric = 12 if side > 0 else 13
    modify_metric = 14 if side > 0 else 15
    place_metric = 16 if side > 0 else 17
    suppress_metric = 22 if side > 0 else 23

    if not should_place or target_px <= 0:
        metrics[suppress_metric] += 1
        if existing is not None and existing.cancellable:
            if hbt.current_timestamp < next_rest_allowed_ts:
                metrics[rest_skip_metric] += 1
                return inflight_until, next_rest_allowed_ts, live_since
            hbt.cancel(0, order_id, False)
            metrics[filter_cancel_metric] += 1
            return hbt.current_timestamp + ORDER_INFLIGHT_NS, hbt.current_timestamp + REST_MIN_INTERVAL_NS, 0
        return inflight_until, next_rest_allowed_ts, live_since

    if existing is not None and live_since > 0 and hbt.current_timestamp - live_since >= ORDER_TTL_NS:
        if existing.cancellable:
            if hbt.current_timestamp < next_rest_allowed_ts:
                metrics[rest_skip_metric] += 1
                return inflight_until, next_rest_allowed_ts, live_since
            hbt.cancel(0, order_id, False)
            metrics[ttl_metric] += 1
            return hbt.current_timestamp + ORDER_INFLIGHT_NS, hbt.current_timestamp + REST_MIN_INTERVAL_NS, 0
        return inflight_until, next_rest_allowed_ts, live_since

    if existing is not None:
        price_changed = abs(existing.price - target_px) >= MIN_AMEND_TICKS * BITMEX_TICK_SIZE
        if existing.cancellable and (price_changed or existing.qty != ORDER_QTY):
            if hbt.current_timestamp < next_rest_allowed_ts:
                metrics[rest_skip_metric] += 1
                return inflight_until, next_rest_allowed_ts, live_since
            hbt.modify(0, order_id, target_px, ORDER_QTY, False)
            metrics[modify_metric] += 1
            return hbt.current_timestamp + ORDER_INFLIGHT_NS, hbt.current_timestamp + REST_MIN_INTERVAL_NS, hbt.current_timestamp
        return inflight_until, next_rest_allowed_ts, live_since

    if hbt.current_timestamp < next_rest_allowed_ts:
        metrics[rest_skip_metric] += 1
        return inflight_until, next_rest_allowed_ts, live_since

    if side > 0:
        hbt.submit_buy_order(0, order_id, target_px, ORDER_QTY, GTX, LIMIT, False)
    else:
        hbt.submit_sell_order(0, order_id, target_px, ORDER_QTY, GTX, LIMIT, False)
    metrics[place_metric] += 1
    return hbt.current_timestamp + ORDER_INFLIGHT_NS, hbt.current_timestamp + REST_MIN_INTERVAL_NS, hbt.current_timestamp


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
def run_strategy(hbt, recorder, metrics, end_close_ts, rule_factor_ids, rule_sides, rule_mins, rule_maxs, rule_leg_counts):
    bid_id = 11_001
    ask_id = 21_001
    bid_inflight_until = 0
    ask_inflight_until = 0
    bid_live_since = 0
    ask_live_since = 0
    next_rest_allowed_ts = 0
    last_record_ts = 0
    last_pos = hbt.position(0)
    last_trades = hbt.state_values(0).num_trades
    last_trading_value = hbt.state_values(0).trading_value

    flow = np.zeros(8, dtype=np.float64)
    signal_ts = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.int64)
    signal_bid = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    signal_ask = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    write_idx = 0
    count = 0

    while hbt.elapse(ORDER_UPDATE_INTERVAL_NS) == 0:
        if hbt.current_timestamp >= end_close_ts:
            break
        hbt.clear_inactive_orders(ALL_ASSETS)
        update_trade_flow(hbt, flow)
        metrics[24] = flow[4]
        metrics[25] = flow[5]
        metrics[26] = flow[6]
        metrics[27] = flow[7]
        write_idx, count = record_signal(hbt, signal_ts, signal_bid, signal_ask, write_idx, count)

        depth = hbt.depth(0)
        if depth.best_bid <= 0 or depth.best_ask <= 0:
            cancel_all_orders(hbt)
        else:
            pos = hbt.position(0)
            bid_ok, bid_reason = should_quote(
                1,
                pos,
                rule_factor_ids,
                rule_sides,
                rule_mins,
                rule_maxs,
                rule_leg_counts,
                depth,
                signal_ts,
                signal_bid,
                signal_ask,
                write_idx,
                count,
                flow,
            )
            ask_ok, ask_reason = should_quote(
                -1,
                pos,
                rule_factor_ids,
                rule_sides,
                rule_mins,
                rule_maxs,
                rule_leg_counts,
                depth,
                signal_ts,
                signal_bid,
                signal_ask,
                write_idx,
                count,
                flow,
            )
            if bid_reason == 3:
                metrics[20] += 1
            if ask_reason == 3:
                metrics[21] += 1

            bid_px = target_price(1, depth, pos)
            bid_inflight_until, next_rest_allowed_ts, bid_live_since = manage_side(
                hbt,
                1,
                bid_id,
                bid_px,
                bid_ok,
                bid_inflight_until,
                next_rest_allowed_ts,
                bid_live_since,
                metrics,
            )
            ask_px = target_price(-1, depth, pos)
            ask_inflight_until, next_rest_allowed_ts, ask_live_since = manage_side(
                hbt,
                -1,
                ask_id,
                ask_px,
                ask_ok,
                ask_inflight_until,
                next_rest_allowed_ts,
                ask_live_since,
                metrics,
            )

        state = hbt.state_values(0)
        if state.num_trades > last_trades:
            depth = hbt.depth(0)
            if depth.best_bid > 0 and depth.best_ask > 0:
                mid = (depth.best_bid + depth.best_ask) / 2.0
                delta_contracts = state.position - last_pos
                delta_value = state.trading_value - last_trading_value
                exec_px = 0.0
                if abs(delta_contracts) > 0 and delta_value > 0:
                    exec_px = delta_value / (abs(delta_contracts) * BITMEX_CONTRACT_SIZE)
                fill_count = state.num_trades - last_trades
                metrics[0] += fill_count
                metrics[3] += abs(delta_contracts) * BITMEX_CONTRACT_SIZE
                if delta_contracts > 0:
                    metrics[1] += fill_count
                    if exec_px > 0:
                        metrics[28] += mid - exec_px
                        metrics[29] += 1
                elif delta_contracts < 0:
                    metrics[2] += fill_count
                    if exec_px > 0:
                        metrics[28] += exec_px - mid
                        metrics[29] += 1
            last_pos = state.position
            last_trades = state.num_trades
            last_trading_value = state.trading_value

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
    if EXCHANGE_MODEL == "partial":
        return asset.partial_fill_exchange()
    if EXCHANGE_MODEL == "strict_no_partial":
        return asset.strict_no_partial_fill_exchange()
    return asset.no_partial_fill_exchange()


def load_rules(
    path: Path,
    min_expected_edge_bps: float,
    min_edge_if_filled_bps: float,
    min_fill_prob: float,
    min_fill_samples: int,
    max_rules_per_side: int,
    min_factor_matches: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    rows = []
    with path.open() as file:
        for row in csv.DictReader(file):
            factors: list[str]
            mins: list[float]
            maxs: list[float]
            if "factor1" in row and row.get("factor1") and row.get("factor2"):
                factors = [row["factor1"], row["factor2"]]
                mins = [float(row["factor1_min"]), float(row["factor2_min"])]
                maxs = [float(row["factor1_max"]), float(row["factor2_max"])]
            else:
                factor = row["factor"]
                factors = [factor]
                mins = [float(row["factor_min"])]
                maxs = [float(row["factor_max"])]
            if len(factors) < min_factor_matches:
                continue
            if any(factor not in FACTOR_IDS for factor in factors):
                continue
            expected = float(row["expected_edge_bps"])
            edge = float(row["edge_if_filled_bps"])
            fill_prob = float(row["fill_prob"])
            fill_samples = int(float(row["fill_samples"]))
            if expected < min_expected_edge_bps:
                continue
            if edge < min_edge_if_filled_bps:
                continue
            if fill_prob < min_fill_prob:
                continue
            if fill_samples < min_fill_samples:
                continue
            rows.append(
                {
                    "factor": "&".join(factors),
                    "factors": factors,
                    "factor_ids": [FACTOR_IDS[factor] for factor in factors],
                    "factor_mins": mins,
                    "factor_maxs": maxs,
                    "factor_count": len(factors),
                    "side": row["side"],
                    "side_id": 1 if row["side"] == "bid" else -1,
                    "factor_min": mins[0],
                    "factor_max": maxs[0],
                    "factor1": factors[0],
                    "factor1_min": mins[0],
                    "factor1_max": maxs[0],
                    "factor2": factors[1] if len(factors) > 1 else "",
                    "factor2_min": mins[1] if len(factors) > 1 else float("nan"),
                    "factor2_max": maxs[1] if len(factors) > 1 else float("nan"),
                    "expected_edge_bps": expected,
                    "edge_if_filled_bps": edge,
                    "fill_prob": fill_prob,
                    "fill_samples": fill_samples,
                }
            )
    selected = []
    for side in ("bid", "ask"):
        side_rows = [row for row in rows if row["side"] == side]
        side_rows.sort(key=lambda row: row["expected_edge_bps"], reverse=True)
        selected.extend(side_rows[:max_rules_per_side])
    selected.sort(key=lambda row: row["expected_edge_bps"], reverse=True)
    if not selected:
        raise ValueError(f"no rules selected from {path}")
    max_legs = max(len(row["factor_ids"]) for row in selected)
    rule_factor_ids = np.full((len(selected), max_legs), -1, dtype=np.int64)
    rule_mins = np.full((len(selected), max_legs), np.nan, dtype=np.float64)
    rule_maxs = np.full((len(selected), max_legs), np.nan, dtype=np.float64)
    rule_leg_counts = np.zeros(len(selected), dtype=np.int64)
    for row_idx, row in enumerate(selected):
        factor_ids = row["factor_ids"]
        mins = row["factor_mins"]
        maxs = row["factor_maxs"]
        rule_leg_counts[row_idx] = len(factor_ids)
        for leg_idx, factor_id in enumerate(factor_ids):
            rule_factor_ids[row_idx, leg_idx] = factor_id
            rule_mins[row_idx, leg_idx] = mins[leg_idx]
            rule_maxs[row_idx, leg_idx] = maxs[leg_idx]
    return (
        rule_factor_ids,
        np.array([row["side_id"] for row in selected], dtype=np.int64),
        rule_mins,
        rule_maxs,
        rule_leg_counts,
        selected,
    )


def rule_factor_coverage(selected_rules: list[dict]) -> dict:
    bid_factors = sorted({row["factor"] for row in selected_rules if row["side"] == "bid"})
    ask_factors = sorted({row["factor"] for row in selected_rules if row["side"] == "ask"})
    bid_atomic_factors = sorted({factor for row in selected_rules if row["side"] == "bid" for factor in row["factors"]})
    ask_atomic_factors = sorted({factor for row in selected_rules if row["side"] == "ask" for factor in row["factors"]})
    bid_rule_factors = [len(row["factors"]) for row in selected_rules if row["side"] == "bid"]
    ask_rule_factors = [len(row["factors"]) for row in selected_rules if row["side"] == "ask"]
    return {
        "bid": bid_factors,
        "ask": ask_factors,
        "bid_count": len(bid_factors),
        "ask_count": len(ask_factors),
        "bid_atomic_factors": bid_atomic_factors,
        "ask_atomic_factors": ask_atomic_factors,
        "bid_atomic_count": len(bid_atomic_factors),
        "ask_atomic_count": len(ask_atomic_factors),
        "bid_max_rule_factors": max(bid_rule_factors) if bid_rule_factors else 0,
        "ask_max_rule_factors": max(ask_rule_factors) if ask_rule_factors else 0,
    }


def run_backtest(bitmex_npz: Path, yyyymmdd: str, rules, min_factor_matches: int) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rule_factor_ids, rule_sides, rule_mins, rule_maxs, rule_leg_counts, selected_rules = rules
    hbt = HashMapMarketDepthBacktest([build_asset(bitmex_npz)])
    recorder = Recorder(1, 100_000)
    metrics = np.zeros(40, dtype=np.float64)
    ok = run_strategy(
        hbt,
        recorder.recorder,
        metrics,
        end_close_ts_ns(yyyymmdd),
        rule_factor_ids,
        rule_sides,
        rule_mins,
        rule_maxs,
        rule_leg_counts,
    )
    if not ok:
        raise RuntimeError("strategy returned false")

    tag = RESULT_TAG or f"factor_filtered_{EXCHANGE_MODEL}"
    out = RESULT_DIR / f"bitmex_xbtusdt_factor_filtered_maker_{tag}_{yyyymmdd}.npz"
    np.savez_compressed(out, **{"0": recorder.get(0), "metrics": metrics})
    write_summary(out, yyyymmdd, selected_rules, min_factor_matches)
    return out


def signed_money(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.4f} USDT"


def write_summary(result_npz: Path, yyyymmdd: str, selected_rules: list[dict], min_factor_matches: int) -> dict:
    data = np.load(result_npz)
    records = data["0"]
    metrics = data["metrics"]
    final = records[-1]
    final_price = float(final["price"])
    final_pos = float(final["position"])
    equity = float(final["balance"]) + final_pos * final_price * BITMEX_CONTRACT_SIZE - float(final["fee"])
    total_fee = float(final["fee"])
    gross = equity + total_fee
    avg_capture = metrics[28] / metrics[29] if metrics[29] > 0 else 0.0
    coverage = rule_factor_coverage(selected_rules)
    summary = {
        "date": yyyymmdd,
        "symbol": BITMEX_SYMBOL,
        "strategy": "factor_filtered_maker",
        "exchange_model": EXCHANGE_MODEL,
        "maker_fee_rate": MAKER_FEE_RATE,
        "taker_fee_rate": TAKER_FEE_RATE,
        "order_qty_contracts": ORDER_QTY,
        "base_half_spread_bps": BASE_HALF_SPREAD_BPS,
        "max_position_contracts": MAX_POSITION_CONTRACTS,
        "soft_position_contracts": SOFT_POSITION_CONTRACTS,
        "order_ttl_ms": ORDER_TTL_NS / 1_000_000.0,
        "rest_min_interval_ms": REST_MIN_INTERVAL_NS / 1_000_000.0,
        "selected_rules": len(selected_rules),
        "min_factor_matches": min_factor_matches,
        "selected_bid_rule_count": coverage["bid_count"],
        "selected_ask_rule_count": coverage["ask_count"],
        "selected_bid_factor_count": coverage["bid_atomic_count"],
        "selected_ask_factor_count": coverage["ask_atomic_count"],
        "selected_bid_factors": ",".join(coverage["bid_atomic_factors"]),
        "selected_ask_factors": ",".join(coverage["ask_atomic_factors"]),
        "selected_bid_rules": ";".join(coverage["bid"]),
        "selected_ask_rules": ";".join(coverage["ask"]),
        "selected_bid_max_rule_factors": coverage["bid_max_rule_factors"],
        "selected_ask_max_rule_factors": coverage["ask_max_rule_factors"],
        "bid_factor_match_possible": coverage["bid_max_rule_factors"] >= min_factor_matches,
        "ask_factor_match_possible": coverage["ask_max_rule_factors"] >= min_factor_matches,
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
        "filter_cancel_bid": int(metrics[12]),
        "filter_cancel_ask": int(metrics[13]),
        "modify_bid": int(metrics[14]),
        "modify_ask": int(metrics[15]),
        "place_bid": int(metrics[16]),
        "place_ask": int(metrics[17]),
        "rest_skip_bid": int(metrics[18]),
        "rest_skip_ask": int(metrics[19]),
        "factor_gate_bid": int(metrics[20]),
        "factor_gate_ask": int(metrics[21]),
        "suppress_bid": int(metrics[22]),
        "suppress_ask": int(metrics[23]),
        "market_buy_qty_seen": float(metrics[24]),
        "market_sell_qty_seen": float(metrics[25]),
        "market_buy_trade_count_seen": int(metrics[26]),
        "market_sell_trade_count_seen": int(metrics[27]),
        "avg_spread_capture_usdt_per_btc": float(avg_capture),
        "spread_capture_events": int(metrics[29]),
        "final_position_contracts": final_pos,
        "final_position_btc": final_pos * BITMEX_CONTRACT_SIZE,
        "records": int(len(records)),
        "num_trades": int(final["num_trades"]),
    }
    result_npz.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    result_npz.with_suffix(".rules.json").write_text(json.dumps(selected_rules, indent=2, sort_keys=True))
    print(render_console_summary(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def render_console_summary(summary: dict) -> str:
    return "\n".join(
        [
            "",
            "========== BitMEX factor-filtered maker 回测 ==========",
            f"日期: {summary['date']}",
            f"PnL: {signed_money(summary['total_pnl_usdt'])}",
            f"gross 不含手续费: {signed_money(summary['gross_pnl_before_fee_usdt'])}",
            f"手续费/返佣贡献: {signed_money(summary['maker_rebate_usdt'])}",
            f"fills: {summary['fills']}，买={summary['buy_fills']}，卖={summary['sell_fills']}",
            f"最大仓位: {summary['max_position_contracts_seen']:,.0f} contracts",
            f"factor gate: bid={summary['factor_gate_bid']}, ask={summary['factor_gate_ask']}",
            f"filter cancel: bid={summary['filter_cancel_bid']}, ask={summary['filter_cancel_ask']}",
            f"avg capture: {summary['avg_spread_capture_usdt_per_btc']:,.4f} USDT/BTC",
            "=======================================================",
            "",
        ]
    )


def write_aggregate(summaries: list[dict]) -> Path:
    tag = RESULT_TAG or f"factor_filtered_{EXCHANGE_MODEL}"
    out = RESULT_DIR / f"bitmex_xbtusdt_factor_filtered_maker_{tag}.aggregate.csv"
    fields = [
        "date",
        "total_pnl_usdt",
        "gross_pnl_before_fee_usdt",
        "maker_rebate_usdt",
        "fills",
        "buy_fills",
        "sell_fills",
        "filled_base_btc",
        "max_position_contracts_seen",
        "factor_gate_bid",
        "factor_gate_ask",
        "filter_cancel_bid",
        "filter_cancel_ask",
        "avg_spread_capture_usdt_per_btc",
        "force_close_pnl_usdt",
        "final_position_contracts",
    ]
    with out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary[field] for field in fields})
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest BitMEX XBTUSDT factor-filtered maker.")
    parser.add_argument("--dates", nargs="+", default=["20260516", "20260517", "20260518"])
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--buffer-rows", type=int, default=None)
    parser.add_argument("--result-tag", default="")
    parser.add_argument("--exchange-model", choices=("no_partial", "strict_no_partial", "partial"), default=EXCHANGE_MODEL)
    parser.add_argument("--order-qty-contracts", type=float, default=ORDER_QTY)
    parser.add_argument("--maker-fee-rate", type=float, default=MAKER_FEE_RATE)
    parser.add_argument("--taker-fee-rate", type=float, default=TAKER_FEE_RATE)
    parser.add_argument("--base-half-spread-bps", type=float, default=BASE_HALF_SPREAD_BPS)
    parser.add_argument("--max-position-contracts", type=float, default=MAX_POSITION_CONTRACTS)
    parser.add_argument("--soft-position-contracts", type=float, default=SOFT_POSITION_CONTRACTS)
    parser.add_argument("--order-ttl-ms", type=float, default=ORDER_TTL_NS / 1_000_000.0)
    parser.add_argument("--rest-min-interval-ms", type=float, default=REST_MIN_INTERVAL_NS / 1_000_000.0)
    parser.add_argument(
        "--rules-csv",
        default="results/factor_research/bitmex_xbtusdt_maker_fill_edge_20260512_20260518_maker0.maker_fill_combo_rules.csv",
    )
    parser.add_argument("--min-expected-edge-bps", type=float, default=0.02)
    parser.add_argument("--min-edge-if-filled-bps", type=float, default=0.25)
    parser.add_argument("--min-fill-prob", type=float, default=0.02)
    parser.add_argument("--min-fill-samples", type=int, default=1_000)
    parser.add_argument("--max-rules-per-side", type=int, default=8)
    parser.add_argument(
        "--min-factor-matches",
        type=int,
        default=1,
        help="Require selected rules to contain at least this many factor legs. Combo rule files encode AND conditions inside each rule.",
    )
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    global RESULT_TAG, EXCHANGE_MODEL, ORDER_QTY, MAKER_FEE_RATE, TAKER_FEE_RATE
    global BASE_HALF_SPREAD_BPS, MAX_POSITION_CONTRACTS, SOFT_POSITION_CONTRACTS
    global ORDER_TTL_NS, REST_MIN_INTERVAL_NS

    RESULT_TAG = args.result_tag
    EXCHANGE_MODEL = args.exchange_model
    ORDER_QTY = args.order_qty_contracts
    MAKER_FEE_RATE = args.maker_fee_rate
    TAKER_FEE_RATE = args.taker_fee_rate
    BASE_HALF_SPREAD_BPS = args.base_half_spread_bps
    MAX_POSITION_CONTRACTS = args.max_position_contracts
    SOFT_POSITION_CONTRACTS = args.soft_position_contracts
    ORDER_TTL_NS = int(args.order_ttl_ms * 1_000_000)
    REST_MIN_INTERVAL_NS = int(args.rest_min_interval_ms * 1_000_000)


def main() -> None:
    args = parse_args()
    apply_args(args)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    rules = load_rules(
        Path(args.rules_csv),
        args.min_expected_edge_bps,
        args.min_edge_if_filled_bps,
        args.min_fill_prob,
        args.min_fill_samples,
        args.max_rules_per_side,
        args.min_factor_matches,
    )
    selected_rules = rules[5]
    print(f"selected_rules={len(selected_rules)} from {args.rules_csv}")
    coverage = rule_factor_coverage(selected_rules)
    print(
        "rule_factor_coverage "
        f"bid_rules={coverage['bid_count']} max_legs={coverage['bid_max_rule_factors']} atomic={coverage['bid_atomic_factors']} "
        f"ask_rules={coverage['ask_count']} max_legs={coverage['ask_max_rule_factors']} atomic={coverage['ask_atomic_factors']} "
        f"required={args.min_factor_matches}"
    )
    if coverage["bid_max_rule_factors"] < args.min_factor_matches:
        print(
            "WARNING: bid side cannot satisfy --min-factor-matches; "
            "new bid quotes will be suppressed unless reducing an existing short position."
        )
    if coverage["ask_max_rule_factors"] < args.min_factor_matches:
        print(
            "WARNING: ask side cannot satisfy --min-factor-matches; "
            "new ask quotes will be suppressed unless reducing an existing long position."
        )
    for rule in selected_rules:
        legs = []
        for factor, lo, hi in zip(rule["factors"], rule["factor_mins"], rule["factor_maxs"]):
            legs.append(f"{factor}=[{lo:.6g},{hi:.6g}]")
        print(
            "rule side={side} legs={legs} expected={expected_edge_bps:.6f} "
            "edge={edge_if_filled_bps:.6f} fill_prob={fill_prob:.6f}".format(
                legs=" AND ".join(legs),
                **rule,
            )
        )

    key = None if args.skip_download else tardis_key()
    summaries = []
    outputs = []
    for yyyymmdd in args.dates:
        if not args.skip_download:
            download_file(BITMEX_EXCHANGE, "trades", BITMEX_SYMBOL, yyyymmdd, key)
            download_file(BITMEX_EXCHANGE, "incremental_book_L2", BITMEX_SYMBOL, yyyymmdd, key)
        bitmex_npz = convert_bitmex(BITMEX_SYMBOL, yyyymmdd, args.buffer_rows)
        result = run_backtest(bitmex_npz, yyyymmdd, rules, args.min_factor_matches)
        outputs.append(result)
        summary_path = result.with_suffix(".summary.json")
        summaries.append(json.loads(summary_path.read_text()))

    aggregate = write_aggregate(summaries)
    total_pnl = sum(item["total_pnl_usdt"] for item in summaries)
    total_gross = sum(item["gross_pnl_before_fee_usdt"] for item in summaries)
    total_rebate = sum(item["maker_rebate_usdt"] for item in summaries)
    total_fills = sum(item["fills"] for item in summaries)
    print(
        f"aggregate={aggregate} total_pnl={total_pnl:.6f} gross={total_gross:.6f} "
        f"rebate={total_rebate:.6f} fills={total_fills}"
    )
    print("all_results=" + ",".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
