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


MAX_LEVELS = 10
SIGNAL_HISTORY_LEN = 4096

ORDER_UPDATE_INTERVAL_NS = 10_000_000
ORDER_INFLIGHT_NS = 80_000_000
REST_MIN_INTERVAL_NS = 700_000_000
ORDER_TTL_NS = 5_000_000_000
MIN_AMEND_TICKS = 5.0

ORDER_QTY = BITMEX_ORDER_QTY
LADDER_LEVELS = 3
BASE_HALF_SPREAD_BPS = 3.0
LADDER_SPACING_BPS = 2.0
LADDER_MIN_SPACING_TICKS = 1.0
VOL_WINDOW_NS = 1_000_000_000
MOMENTUM_WINDOW_NS = 250_000_000
VOL_SPREAD_MULTIPLIER = 0.5

SOFT_POSITION_CONTRACTS = 500.0
MAX_POSITION_CONTRACTS = 1_000.0
INVENTORY_SKEW_BPS_AT_SOFT_LIMIT = 4.0

FLOW_DECAY = 0.92
FLOW_MIN_QTY = 300.0
QUEUE_MIN_CONSUME_RATIO = 0.08
QUEUE_GOOD_CONSUME_RATIO = 0.30
QUEUE_MAX_ADVERSE_SCORE = 2.0
QUEUE_WIDEN_BPS = 2.5
QUEUE_LEVEL_CUT_THRESHOLD = 0.05
QUEUE_DEPTH_NORMALIZER = 1_000.0

MAKER_FEE_RATE = -0.0002
TAKER_FEE_RATE = 0.0001
EXCHANGE_MODEL = "no_partial"
RESULT_TAG = ""
TOXIC_FILL_MID_MOVE_BPS = 1.5


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
def current_mid(depth):
    return (depth.best_bid + depth.best_ask) / 2.0


@njit
def bitmex_equity_usdt(hbt):
    depth = hbt.depth(0)
    mid = current_mid(depth)
    state = hbt.state_values(0)
    return state.balance + state.position * mid * BITMEX_CONTRACT_SIZE - state.fee


@njit
def fair_microprice(depth):
    mid = current_mid(depth)
    total = depth.best_bid_qty + depth.best_ask_qty
    if total <= 0:
        return mid
    return (depth.best_ask * depth.best_bid_qty + depth.best_bid * depth.best_ask_qty) / total


@njit
def microprice_bps(depth):
    mid = current_mid(depth)
    if mid <= 0:
        return 0.0
    return ratio_minus_one_bps(fair_microprice(depth), mid)


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
def recent_vol_bps(signal_ts, signal_bid, signal_ask, write_idx, count):
    return abs(recent_move_bps(signal_ts, signal_bid, signal_ask, write_idx, count, VOL_WINDOW_NS))


@njit
def inventory_ratio(pos):
    if SOFT_POSITION_CONTRACTS <= 0:
        return 0.0
    return clamp(pos / SOFT_POSITION_CONTRACTS, -1.0, 1.0)


@njit
def queue_consume_ratio(side, depth, flow):
    if side > 0:
        queue_qty = max(depth.best_bid_qty + ORDER_QTY, QUEUE_DEPTH_NORMALIZER)
        return flow[1] / queue_qty
    queue_qty = max(depth.best_ask_qty + ORDER_QTY, QUEUE_DEPTH_NORMALIZER)
    return flow[0] / queue_qty


@njit
def adverse_score(side, flow_sig, momentum_bps, micro_bps):
    score = 0.0
    if side > 0:
        score += max(0.0, -flow_sig)
        score += max(0.0, -momentum_bps)
        score += max(0.0, -micro_bps)
    else:
        score += max(0.0, flow_sig)
        score += max(0.0, momentum_bps)
        score += max(0.0, micro_bps)
    return score


@njit
def active_levels_for_side(side, pos, consume_ratio, adverse):
    levels = LADDER_LEVELS
    if levels < 1:
        levels = 1
    if levels > MAX_LEVELS:
        levels = MAX_LEVELS

    inv = inventory_ratio(pos)
    if side > 0 and inv > 0:
        levels -= int(math.ceil(inv * (levels - 1)))
    elif side < 0 and inv < 0:
        levels -= int(math.ceil(-inv * (levels - 1)))

    if consume_ratio < QUEUE_LEVEL_CUT_THRESHOLD:
        levels -= int(math.ceil((QUEUE_LEVEL_CUT_THRESHOLD - consume_ratio) / max(QUEUE_LEVEL_CUT_THRESHOLD, 1e-9) * (LADDER_LEVELS - 1)))
    if adverse > QUEUE_MAX_ADVERSE_SCORE * 0.5:
        levels -= 1
    return max(1, levels)


@njit
def should_quote_level(side, level_idx, pos, consume_ratio, adverse):
    if ORDER_QTY < BITMEX_LOT_SIZE:
        return False, 1
    projected = level_idx + 1
    if side > 0 and pos + ORDER_QTY * projected > MAX_POSITION_CONTRACTS:
        return False, 2
    if side < 0 and pos - ORDER_QTY * projected < -MAX_POSITION_CONTRACTS:
        return False, 2
    if consume_ratio < QUEUE_MIN_CONSUME_RATIO and level_idx == 0:
        return False, 3
    if adverse > QUEUE_MAX_ADVERSE_SCORE:
        return False, 4
    levels = active_levels_for_side(side, pos, consume_ratio, adverse)
    if level_idx >= levels:
        return False, 5
    return True, 0


@njit
def quote_price(side, level_idx, depth, pos, consume_ratio, adverse, vol_bps):
    mid = current_mid(depth)
    inv = inventory_ratio(pos)
    min_spacing_bps = LADDER_MIN_SPACING_TICKS * BITMEX_TICK_SIZE / mid * 10_000.0
    spacing_bps = max(LADDER_SPACING_BPS + VOL_SPREAD_MULTIPLIER * vol_bps, min_spacing_bps)
    half_bps = max(BASE_HALF_SPREAD_BPS, spacing_bps * (level_idx + 1))

    queue_shortfall = max(0.0, QUEUE_GOOD_CONSUME_RATIO - consume_ratio) / max(QUEUE_GOOD_CONSUME_RATIO, 1e-9)
    half_bps += QUEUE_WIDEN_BPS * min(queue_shortfall, 1.0)
    half_bps += min(adverse, QUEUE_MAX_ADVERSE_SCORE) * 0.8

    anchor = fair_microprice(depth) * (1.0 - inv * INVENTORY_SKEW_BPS_AT_SOFT_LIMIT / 10_000.0)
    if side > 0:
        return floor_to_tick(min(anchor * (1.0 - half_bps / 10_000.0), depth.best_bid), BITMEX_TICK_SIZE)
    return ceil_to_tick(max(anchor * (1.0 + half_bps / 10_000.0), depth.best_ask), BITMEX_TICK_SIZE)


@njit
def cancel_all_orders(hbt):
    orders = hbt.orders(0)
    values = orders.values()
    while values.has_next():
        order = values.get()
        if order.cancellable:
            hbt.cancel(0, order.order_id, False)


@njit
def manage_quote(hbt, side, order_id, target_px, should_quote, inflight_until, next_rest_allowed_ts, live_since, metrics):
    if hbt.current_timestamp < inflight_until:
        return inflight_until, next_rest_allowed_ts, live_since

    existing = hbt.orders(0).get(order_id)
    rest_metric = 18 if side > 0 else 19
    ttl_metric = 10 if side > 0 else 11
    cancel_metric = 12 if side > 0 else 13
    modify_metric = 14 if side > 0 else 15
    place_metric = 16 if side > 0 else 17
    suppress_metric = 20 if side > 0 else 21

    if not should_quote or target_px <= 0:
        metrics[suppress_metric] += 1
        if existing is not None and existing.cancellable:
            if hbt.current_timestamp < next_rest_allowed_ts:
                metrics[rest_metric] += 1
                return inflight_until, next_rest_allowed_ts, live_since
            hbt.cancel(0, order_id, False)
            metrics[cancel_metric] += 1
            return hbt.current_timestamp + ORDER_INFLIGHT_NS, hbt.current_timestamp + REST_MIN_INTERVAL_NS, 0
        return inflight_until, next_rest_allowed_ts, live_since

    if existing is not None and live_since > 0 and hbt.current_timestamp - live_since >= ORDER_TTL_NS:
        if existing.cancellable:
            if hbt.current_timestamp < next_rest_allowed_ts:
                metrics[rest_metric] += 1
                return inflight_until, next_rest_allowed_ts, live_since
            hbt.cancel(0, order_id, False)
            metrics[ttl_metric] += 1
            return hbt.current_timestamp + ORDER_INFLIGHT_NS, hbt.current_timestamp + REST_MIN_INTERVAL_NS, 0
        return inflight_until, next_rest_allowed_ts, live_since

    if existing is not None:
        price_changed = abs(existing.price - target_px) >= MIN_AMEND_TICKS * BITMEX_TICK_SIZE
        if existing.cancellable and (price_changed or existing.qty != ORDER_QTY):
            if hbt.current_timestamp < next_rest_allowed_ts:
                metrics[rest_metric] += 1
                return inflight_until, next_rest_allowed_ts, live_since
            hbt.modify(0, order_id, target_px, ORDER_QTY, False)
            metrics[modify_metric] += 1
            return hbt.current_timestamp + ORDER_INFLIGHT_NS, hbt.current_timestamp + REST_MIN_INTERVAL_NS, hbt.current_timestamp
        return inflight_until, next_rest_allowed_ts, live_since

    if hbt.current_timestamp < next_rest_allowed_ts:
        metrics[rest_metric] += 1
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
    mid = current_mid(depth)
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
    bid_ids = np.zeros(MAX_LEVELS, dtype=np.int64)
    ask_ids = np.zeros(MAX_LEVELS, dtype=np.int64)
    bid_inflight = np.zeros(MAX_LEVELS, dtype=np.int64)
    ask_inflight = np.zeros(MAX_LEVELS, dtype=np.int64)
    bid_live_since = np.zeros(MAX_LEVELS, dtype=np.int64)
    ask_live_since = np.zeros(MAX_LEVELS, dtype=np.int64)
    for idx in range(MAX_LEVELS):
        bid_ids[idx] = 11_001 + idx
        ask_ids[idx] = 21_001 + idx

    signal_ts = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.int64)
    signal_bid = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    signal_ask = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    flow = np.zeros(6, dtype=np.float64)
    write_idx = 0
    count = 0
    next_rest_allowed_ts = 0
    last_record_ts = 0
    last_pos = hbt.position(0)
    last_trades = hbt.state_values(0).num_trades
    last_trading_value = hbt.state_values(0).trading_value

    while hbt.elapse(ORDER_UPDATE_INTERVAL_NS) == 0:
        if hbt.current_timestamp >= end_close_ts:
            break
        hbt.clear_inactive_orders(ALL_ASSETS)
        update_trade_flow(hbt, flow)
        metrics[24] = flow[2]
        metrics[25] = flow[3]
        metrics[26] = flow[4]
        metrics[27] = flow[5]
        write_idx, count = record_signal(hbt, signal_ts, signal_bid, signal_ask, write_idx, count)

        depth = hbt.depth(0)
        if depth.best_bid > 0 and depth.best_ask > 0:
            pos = hbt.position(0)
            momentum_bps = recent_move_bps(signal_ts, signal_bid, signal_ask, write_idx, count, MOMENTUM_WINDOW_NS)
            vol_bps = recent_vol_bps(signal_ts, signal_bid, signal_ask, write_idx, count)
            micro_bps = microprice_bps(depth)
            fs = flow_score(flow)
            bid_consume = queue_consume_ratio(1, depth, flow)
            ask_consume = queue_consume_ratio(-1, depth, flow)
            bid_adverse = adverse_score(1, fs, momentum_bps, micro_bps)
            ask_adverse = adverse_score(-1, fs, momentum_bps, micro_bps)
            metrics[32] += bid_consume
            metrics[33] += ask_consume
            metrics[34] += bid_adverse
            metrics[35] += ask_adverse
            metrics[36] += 1

            loop_levels = LADDER_LEVELS
            if loop_levels < 1:
                loop_levels = 1
            if loop_levels > MAX_LEVELS:
                loop_levels = MAX_LEVELS

            for idx in range(loop_levels):
                bid_ok, bid_reason = should_quote_level(1, idx, pos, bid_consume, bid_adverse)
                if bid_reason == 3:
                    metrics[37] += 1
                elif bid_reason == 4:
                    metrics[38] += 1
                bid_px = quote_price(1, idx, depth, pos, bid_consume, bid_adverse, vol_bps)
                bid_inflight[idx], next_rest_allowed_ts, bid_live_since[idx] = manage_quote(
                    hbt, 1, bid_ids[idx], bid_px, bid_ok, bid_inflight[idx], next_rest_allowed_ts, bid_live_since[idx], metrics
                )

                ask_ok, ask_reason = should_quote_level(-1, idx, pos, ask_consume, ask_adverse)
                if ask_reason == 3:
                    metrics[39] += 1
                elif ask_reason == 4:
                    metrics[40] += 1
                ask_px = quote_price(-1, idx, depth, pos, ask_consume, ask_adverse, vol_bps)
                ask_inflight[idx], next_rest_allowed_ts, ask_live_since[idx] = manage_quote(
                    hbt, -1, ask_ids[idx], ask_px, ask_ok, ask_inflight[idx], next_rest_allowed_ts, ask_live_since[idx], metrics
                )

        state = hbt.state_values(0)
        if state.num_trades > last_trades:
            depth = hbt.depth(0)
            if depth.best_bid > 0 and depth.best_ask > 0:
                mid = current_mid(depth)
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
                        if ratio_minus_one_bps(exec_px, mid) >= TOXIC_FILL_MID_MOVE_BPS:
                            metrics[30] += 1
                elif delta_contracts < 0:
                    metrics[2] += fill_count
                    if exec_px > 0:
                        metrics[28] += exec_px - mid
                        metrics[29] += 1
                        if ratio_minus_one_bps(mid, exec_px) >= TOXIC_FILL_MID_MOVE_BPS:
                            metrics[30] += 1
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


def run_backtest(bitmex_npz: Path, yyyymmdd: str) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    hbt = HashMapMarketDepthBacktest([build_asset(bitmex_npz)])
    recorder = Recorder(1, 100_000)
    metrics = np.zeros(48, dtype=np.float64)
    ok = run_strategy(hbt, recorder.recorder, metrics, end_close_ts_ns(yyyymmdd))
    if not ok:
        raise RuntimeError("strategy returned false")

    tag = RESULT_TAG or f"queue_aware_ladder_{EXCHANGE_MODEL}"
    out = RESULT_DIR / f"bitmex_xbtusdt_queue_aware_ladder_{tag}_{yyyymmdd}.npz"
    np.savez_compressed(out, **{"0": recorder.get(0), "metrics": metrics})
    write_summary(out, yyyymmdd)
    return out


def signed_money(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.4f} USDT"


def write_summary(result_npz: Path, yyyymmdd: str) -> dict:
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
    samples = metrics[36] if metrics[36] > 0 else 1.0
    summary = {
        "date": yyyymmdd,
        "symbol": BITMEX_SYMBOL,
        "strategy": "queue_aware_ladder",
        "exchange_model": EXCHANGE_MODEL,
        "maker_fee_rate": MAKER_FEE_RATE,
        "taker_fee_rate": TAKER_FEE_RATE,
        "levels": LADDER_LEVELS,
        "order_qty_contracts": ORDER_QTY,
        "base_half_spread_bps": BASE_HALF_SPREAD_BPS,
        "ladder_spacing_bps": LADDER_SPACING_BPS,
        "queue_min_consume_ratio": QUEUE_MIN_CONSUME_RATIO,
        "queue_good_consume_ratio": QUEUE_GOOD_CONSUME_RATIO,
        "queue_max_adverse_score": QUEUE_MAX_ADVERSE_SCORE,
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
        "target_cancel_bid": int(metrics[12]),
        "target_cancel_ask": int(metrics[13]),
        "modify_bid": int(metrics[14]),
        "modify_ask": int(metrics[15]),
        "place_bid": int(metrics[16]),
        "place_ask": int(metrics[17]),
        "rest_skip_bid": int(metrics[18]),
        "rest_skip_ask": int(metrics[19]),
        "suppress_bid": int(metrics[20]),
        "suppress_ask": int(metrics[21]),
        "market_buy_qty_seen": float(metrics[24]),
        "market_sell_qty_seen": float(metrics[25]),
        "market_buy_trade_count_seen": int(metrics[26]),
        "market_sell_trade_count_seen": int(metrics[27]),
        "avg_spread_capture_usdt_per_btc": float(avg_capture),
        "spread_capture_events": int(metrics[29]),
        "toxic_fill_events": int(metrics[30]),
        "avg_bid_consume_ratio": float(metrics[32] / samples),
        "avg_ask_consume_ratio": float(metrics[33] / samples),
        "avg_bid_adverse_score": float(metrics[34] / samples),
        "avg_ask_adverse_score": float(metrics[35] / samples),
        "queue_gate_bid": int(metrics[37]),
        "adverse_gate_bid": int(metrics[38]),
        "queue_gate_ask": int(metrics[39]),
        "adverse_gate_ask": int(metrics[40]),
        "final_position_contracts": final_pos,
        "final_position_btc": final_pos * BITMEX_CONTRACT_SIZE,
        "records": int(len(records)),
        "num_trades": int(final["num_trades"]),
    }
    result_npz.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    result_npz.with_suffix(".report.md").write_text(render_report(summary))
    print(render_console_summary(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def render_console_summary(summary: dict) -> str:
    return "\n".join(
        [
            "",
            "========== BitMEX queue-aware ladder 回测 ==========",
            f"日期: {summary['date']}",
            f"PnL: {signed_money(summary['total_pnl_usdt'])}",
            f"gross 不含手续费: {signed_money(summary['gross_pnl_before_fee_usdt'])}",
            f"手续费/返佣贡献: {signed_money(summary['maker_rebate_usdt'])}",
            f"fills: {summary['fills']}，买={summary['buy_fills']}，卖={summary['sell_fills']}",
            f"最大仓位: {summary['max_position_contracts_seen']:,.0f} contracts",
            f"avg consume: bid={summary['avg_bid_consume_ratio']:,.4f}, ask={summary['avg_ask_consume_ratio']:,.4f}",
            f"queue gate: bid={summary['queue_gate_bid']}, ask={summary['queue_gate_ask']}",
            f"avg capture: {summary['avg_spread_capture_usdt_per_btc']:,.4f} USDT/BTC",
            "====================================================",
            "",
        ]
    )


def render_report(summary: dict) -> str:
    return f"""# BitMEX XBTUSDT Queue-Aware Ladder 回测报告

## Result

- date: `{summary['date']}`
- exchange model: `{summary['exchange_model']}`
- total PnL: `{signed_money(summary['total_pnl_usdt'])}`
- gross before fee: `{signed_money(summary['gross_pnl_before_fee_usdt'])}`
- maker rebate: `{signed_money(summary['maker_rebate_usdt'])}`
- fills: `{summary['fills']}`
- filled base: `{summary['filled_base_btc']:,.8f} BTC`
- max position: `{summary['max_position_contracts_seen']:,.0f} contracts`
- avg bid consume ratio: `{summary['avg_bid_consume_ratio']:,.6f}`
- avg ask consume ratio: `{summary['avg_ask_consume_ratio']:,.6f}`
- avg spread capture: `{summary['avg_spread_capture_usdt_per_btc']:,.4f} USDT/BTC`
"""


def write_aggregate(summaries: list[dict]) -> Path:
    tag = RESULT_TAG or f"queue_aware_ladder_{EXCHANGE_MODEL}"
    out = RESULT_DIR / f"bitmex_xbtusdt_queue_aware_ladder_{tag}.aggregate.csv"
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
        "toxic_fill_events",
        "avg_bid_consume_ratio",
        "avg_ask_consume_ratio",
        "queue_gate_bid",
        "queue_gate_ask",
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
    parser = argparse.ArgumentParser(description="Backtest BitMEX XBTUSDT queue-aware ladder maker.")
    parser.add_argument("--dates", nargs="+", default=["20260516", "20260517", "20260518"])
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--buffer-rows", type=int, default=None)
    parser.add_argument("--result-tag", default="")
    parser.add_argument("--exchange-model", choices=("no_partial", "strict_no_partial", "partial"), default=EXCHANGE_MODEL)
    parser.add_argument("--levels", type=int, default=LADDER_LEVELS)
    parser.add_argument("--order-qty-contracts", type=float, default=ORDER_QTY)
    parser.add_argument("--maker-fee-rate", type=float, default=MAKER_FEE_RATE)
    parser.add_argument("--taker-fee-rate", type=float, default=TAKER_FEE_RATE)
    parser.add_argument("--base-half-spread-bps", type=float, default=BASE_HALF_SPREAD_BPS)
    parser.add_argument("--grid-spacing-bps", type=float, default=LADDER_SPACING_BPS)
    parser.add_argument("--max-position-contracts", type=float, default=MAX_POSITION_CONTRACTS)
    parser.add_argument("--soft-position-contracts", type=float, default=SOFT_POSITION_CONTRACTS)
    parser.add_argument("--queue-min-consume-ratio", type=float, default=QUEUE_MIN_CONSUME_RATIO)
    parser.add_argument("--queue-good-consume-ratio", type=float, default=QUEUE_GOOD_CONSUME_RATIO)
    parser.add_argument("--queue-max-adverse-score", type=float, default=QUEUE_MAX_ADVERSE_SCORE)
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    global RESULT_TAG, EXCHANGE_MODEL, LADDER_LEVELS, ORDER_QTY, MAKER_FEE_RATE, TAKER_FEE_RATE
    global BASE_HALF_SPREAD_BPS, LADDER_SPACING_BPS, MAX_POSITION_CONTRACTS, SOFT_POSITION_CONTRACTS
    global QUEUE_MIN_CONSUME_RATIO, QUEUE_GOOD_CONSUME_RATIO, QUEUE_MAX_ADVERSE_SCORE

    RESULT_TAG = args.result_tag
    EXCHANGE_MODEL = args.exchange_model
    LADDER_LEVELS = args.levels
    ORDER_QTY = args.order_qty_contracts
    MAKER_FEE_RATE = args.maker_fee_rate
    TAKER_FEE_RATE = args.taker_fee_rate
    BASE_HALF_SPREAD_BPS = args.base_half_spread_bps
    LADDER_SPACING_BPS = args.grid_spacing_bps
    MAX_POSITION_CONTRACTS = args.max_position_contracts
    SOFT_POSITION_CONTRACTS = args.soft_position_contracts
    QUEUE_MIN_CONSUME_RATIO = args.queue_min_consume_ratio
    QUEUE_GOOD_CONSUME_RATIO = args.queue_good_consume_ratio
    QUEUE_MAX_ADVERSE_SCORE = args.queue_max_adverse_score


def main() -> None:
    args = parse_args()
    apply_args(args)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    key = None if args.skip_download else tardis_key()
    outputs = []
    summaries = []
    for yyyymmdd in args.dates:
        if not args.skip_download:
            download_file(BITMEX_EXCHANGE, "trades", BITMEX_SYMBOL, yyyymmdd, key)
            download_file(BITMEX_EXCHANGE, "incremental_book_L2", BITMEX_SYMBOL, yyyymmdd, key)
        bitmex_npz = convert_bitmex(BITMEX_SYMBOL, yyyymmdd, args.buffer_rows)
        result = run_backtest(bitmex_npz, yyyymmdd)
        outputs.append(result)
        summaries.append(json.loads(result.with_suffix(".summary.json").read_text()))

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
