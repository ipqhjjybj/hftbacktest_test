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


STRATEGY_FIXED_SPREAD = 1
STRATEGY_AVELLANEDA_STOIKOV = 2
STRATEGY_LADDER_GRID = 3

STRATEGY_NAMES = {
    STRATEGY_FIXED_SPREAD: "fixed_spread_mm",
    STRATEGY_AVELLANEDA_STOIKOV: "avellaneda_stoikov_mm",
    STRATEGY_LADDER_GRID: "ladder_grid_mm",
}

STRATEGY_SLUGS = {
    STRATEGY_FIXED_SPREAD: "fixed_spread",
    STRATEGY_AVELLANEDA_STOIKOV: "as",
    STRATEGY_LADDER_GRID: "ladder",
}

ORDER_UPDATE_INTERVAL_NS = 10_000_000
BITMEX_COMMAND_INFLIGHT_NS = 80_000_000
BITMEX_REST_MIN_INTERVAL_NS = 350_000_000

MAX_LEVELS = 8
SIGNAL_HISTORY_LEN = 4096

BASE_HALF_SPREAD_BPS = 3.0
MIN_HALF_SPREAD_TICKS = 1.0
MAX_POSITION_CONTRACTS = 1_000.0
SOFT_POSITION_CONTRACTS = 500.0
ORDER_TTL_NS = 200_000_000
ORDER_QTY = BITMEX_ORDER_QTY
MAKER_FEE_RATE = 0.0
TAKER_FEE_RATE = 0.0

VOL_WINDOW_NS = 1_000_000_000
VOL_SPREAD_MULTIPLIER = 0.5
TOXIC_FILL_MID_MOVE_BPS = 1.5

AS_RISK_AVERSION = 0.20
AS_HORIZON_NS = 2_000_000_000
AS_LIQUIDITY_SPREAD_BPS = 1.0
AS_INVENTORY_SPREAD_BPS_AT_SOFT_LIMIT = 1.5

LADDER_LEVELS = 3
LADDER_SPACING_BPS = 2.0
LADDER_MIN_SPACING_TICKS = 1.0
LADDER_INVENTORY_SKEW_BPS_AT_SOFT_LIMIT = 4.0

RESULT_TAG = ""
CURRENT_STRATEGY_MODE = STRATEGY_FIXED_SPREAD


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
def bitmex_base_from_contracts(contracts):
    return contracts * BITMEX_CONTRACT_SIZE


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
def current_mid_from_history(signal_bid, signal_ask, idx):
    if idx < 0:
        return 0.0
    return (signal_bid[idx] + signal_ask[idx]) / 2.0


@njit
def recent_move_bps(signal_ts, signal_bid, signal_ask, write_idx, count):
    if count <= 1 or VOL_WINDOW_NS <= 0:
        return 0.0
    current_idx = (write_idx - 1) % SIGNAL_HISTORY_LEN
    past_idx = signal_at_or_before(signal_ts, write_idx, count, signal_ts[current_idx] - VOL_WINDOW_NS)
    if past_idx < 0:
        return 0.0
    current_mid = current_mid_from_history(signal_bid, signal_ask, current_idx)
    past_mid = current_mid_from_history(signal_bid, signal_ask, past_idx)
    move_bps = ratio_minus_one_bps(current_mid, past_mid)
    if not math.isfinite(move_bps):
        return 0.0
    return move_bps


@njit
def recent_volatility_bps(signal_ts, signal_bid, signal_ask, write_idx, count):
    return abs(recent_move_bps(signal_ts, signal_bid, signal_ask, write_idx, count))


@njit
def current_fair_price(depth):
    bid = depth.best_bid
    ask = depth.best_ask
    mid = (bid + ask) / 2.0
    total_qty = depth.best_bid_qty + depth.best_ask_qty
    if total_qty > 0:
        return (ask * depth.best_bid_qty + bid * depth.best_ask_qty) / total_qty
    return mid


@njit
def inventory_ratio(pos):
    if SOFT_POSITION_CONTRACTS <= 0:
        return 0.0
    return clamp(pos / SOFT_POSITION_CONTRACTS, -1.0, 1.0)


@njit
def active_ladder_levels(side, inv_ratio):
    levels = LADDER_LEVELS
    if levels < 1:
        levels = 1
    if levels > MAX_LEVELS:
        levels = MAX_LEVELS

    if side > 0 and inv_ratio > 0:
        reduce = int(math.ceil(inv_ratio * (levels - 1)))
        return max(1, levels - reduce)
    if side < 0 and inv_ratio < 0:
        reduce = int(math.ceil(-inv_ratio * (levels - 1)))
        return max(1, levels - reduce)
    return levels


@njit
def quote_price_for_level(
    strategy_mode,
    side,
    level_idx,
    fair,
    mid,
    best_bid,
    best_ask,
    pos,
    vol_bps,
):
    inv = inventory_ratio(pos)
    min_half_spread_bps = MIN_HALF_SPREAD_TICKS * BITMEX_TICK_SIZE / mid * 10_000.0
    half_spread_bps = max(BASE_HALF_SPREAD_BPS, min_half_spread_bps)
    anchor = fair

    if strategy_mode == STRATEGY_AVELLANEDA_STOIKOV:
        horizon_seconds = AS_HORIZON_NS / 1_000_000_000.0
        variance_bps2 = vol_bps * vol_bps
        inventory_shift_bps = inv * AS_RISK_AVERSION * variance_bps2 * horizon_seconds / 100.0
        inventory_spread_bps = abs(inv) * AS_INVENTORY_SPREAD_BPS_AT_SOFT_LIMIT
        anchor = fair * (1.0 - inventory_shift_bps / 10_000.0)
        half_spread_bps = max(
            BASE_HALF_SPREAD_BPS
            + AS_LIQUIDITY_SPREAD_BPS
            + VOL_SPREAD_MULTIPLIER * vol_bps
            + inventory_spread_bps,
            min_half_spread_bps,
        )
    elif strategy_mode == STRATEGY_LADDER_GRID:
        min_spacing_bps = LADDER_MIN_SPACING_TICKS * BITMEX_TICK_SIZE / mid * 10_000.0
        spacing_bps = max(
            LADDER_SPACING_BPS + VOL_SPREAD_MULTIPLIER * vol_bps,
            min_spacing_bps,
        )
        anchor = fair * (1.0 - inv * LADDER_INVENTORY_SKEW_BPS_AT_SOFT_LIMIT / 10_000.0)
        half_spread_bps = spacing_bps * (level_idx + 1)

    if side > 0:
        return floor_to_tick(min(anchor * (1.0 - half_spread_bps / 10_000.0), best_bid), BITMEX_TICK_SIZE)
    return ceil_to_tick(max(anchor * (1.0 + half_spread_bps / 10_000.0), best_ask), BITMEX_TICK_SIZE)


@njit
def should_quote_level(strategy_mode, side, level_idx, pos):
    if ORDER_QTY < BITMEX_LOT_SIZE:
        return False

    projected_levels = level_idx + 1
    if side > 0:
        if pos + ORDER_QTY * projected_levels > MAX_POSITION_CONTRACTS:
            return False
    else:
        if pos - ORDER_QTY * projected_levels < -MAX_POSITION_CONTRACTS:
            return False

    if strategy_mode == STRATEGY_LADDER_GRID:
        inv = inventory_ratio(pos)
        allowed = active_ladder_levels(side, inv)
        if level_idx >= allowed:
            return False
    else:
        if level_idx > 0:
            return False

    return True


@njit
def manage_quote(
    hbt,
    side,
    order_id,
    target_px,
    should_quote,
    inflight_until,
    next_rest_allowed_ts,
    live_since,
    metrics,
):
    if hbt.current_timestamp < inflight_until:
        return inflight_until, next_rest_allowed_ts, live_since

    existing = hbt.orders(0).get(order_id)
    pacing_metric = 20 if side > 0 else 21
    ttl_metric = 6 if side > 0 else 7
    inventory_cancel_metric = 8 if side > 0 else 9
    place_metric = 10 if side > 0 else 11
    modify_metric = 12 if side > 0 else 13
    suppress_metric = 22 if side > 0 else 23

    if not should_quote or target_px <= 0:
        if existing is not None and existing.cancellable:
            if hbt.current_timestamp < next_rest_allowed_ts:
                metrics[pacing_metric] += 1
                return inflight_until, next_rest_allowed_ts, live_since
            hbt.cancel(0, order_id, False)
            metrics[inventory_cancel_metric] += 1
            return (
                hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS,
                hbt.current_timestamp + BITMEX_REST_MIN_INTERVAL_NS,
                0,
            )
        metrics[suppress_metric] += 1
        return inflight_until, next_rest_allowed_ts, live_since

    if existing is not None and live_since > 0 and hbt.current_timestamp - live_since >= ORDER_TTL_NS:
        if existing.cancellable:
            if hbt.current_timestamp < next_rest_allowed_ts:
                metrics[pacing_metric] += 1
                return inflight_until, next_rest_allowed_ts, live_since
            hbt.cancel(0, order_id, False)
            metrics[ttl_metric] += 1
            return (
                hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS,
                hbt.current_timestamp + BITMEX_REST_MIN_INTERVAL_NS,
                0,
            )
        return inflight_until, next_rest_allowed_ts, live_since

    if existing is not None:
        if existing.cancellable and (existing.price != target_px or existing.qty != ORDER_QTY):
            if hbt.current_timestamp < next_rest_allowed_ts:
                metrics[pacing_metric] += 1
                return inflight_until, next_rest_allowed_ts, live_since
            hbt.modify(0, order_id, target_px, ORDER_QTY, False)
            metrics[modify_metric] += 1
            return (
                hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS,
                hbt.current_timestamp + BITMEX_REST_MIN_INTERVAL_NS,
                hbt.current_timestamp,
            )
        return inflight_until, next_rest_allowed_ts, live_since

    if hbt.current_timestamp < next_rest_allowed_ts:
        metrics[pacing_metric] += 1
        return inflight_until, next_rest_allowed_ts, live_since

    if side > 0:
        hbt.submit_buy_order(0, order_id, target_px, ORDER_QTY, GTX, LIMIT, False)
    else:
        hbt.submit_sell_order(0, order_id, target_px, ORDER_QTY, GTX, LIMIT, False)
    metrics[place_metric] += 1
    return (
        hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS,
        hbt.current_timestamp + BITMEX_REST_MIN_INTERVAL_NS,
        hbt.current_timestamp,
    )


@njit
def update_risk_metrics(hbt, metrics):
    depth = hbt.depth(0)
    if depth.best_bid <= 0 or depth.best_ask <= 0:
        return
    pos_contracts = abs(hbt.position(0))
    pos_base = abs(bitmex_base_from_contracts(pos_contracts))
    metrics[1] = max(metrics[1], pos_base)
    metrics[2] = max(metrics[2], pos_contracts)
    equity = bitmex_equity_usdt(hbt)
    if not math.isfinite(equity):
        return
    if (
        (metrics[16] == 0.0 and metrics[17] == 0.0)
        or not math.isfinite(metrics[16])
        or not math.isfinite(metrics[17])
    ):
        metrics[16] = equity
        metrics[17] = equity
    metrics[16] = max(metrics[16], equity)
    metrics[17] = min(metrics[17], equity)


@njit
def force_flatten(hbt, metrics):
    depth = hbt.depth(0)
    if depth.best_bid <= 0 or depth.best_ask <= 0:
        return
    cancel_all_orders(hbt)
    before = bitmex_equity_usdt(hbt)
    pos = hbt.position(0)
    if pos > 0:
        hbt.submit_sell_order(0, 90_001, depth.best_bid, abs(pos), IOC, MARKET, True)
    elif pos < 0:
        hbt.submit_buy_order(0, 90_002, depth.best_ask, abs(pos), IOC, MARKET, True)
    after = bitmex_equity_usdt(hbt)
    metrics[14] = after - before
    update_risk_metrics(hbt, metrics)


@njit
def run_strategy(hbt, recorder, metrics, end_close_ts_ns, strategy_mode):
    loop_levels = 1
    if strategy_mode == STRATEGY_LADDER_GRID:
        loop_levels = LADDER_LEVELS
        if loop_levels < 1:
            loop_levels = 1
        if loop_levels > MAX_LEVELS:
            loop_levels = MAX_LEVELS

    bid_order_ids = np.zeros(MAX_LEVELS, dtype=np.int64)
    ask_order_ids = np.zeros(MAX_LEVELS, dtype=np.int64)
    bid_inflight_until = np.zeros(MAX_LEVELS, dtype=np.int64)
    ask_inflight_until = np.zeros(MAX_LEVELS, dtype=np.int64)
    bid_live_since = np.zeros(MAX_LEVELS, dtype=np.int64)
    ask_live_since = np.zeros(MAX_LEVELS, dtype=np.int64)

    for idx in range(MAX_LEVELS):
        bid_order_ids[idx] = 10_001 + idx
        ask_order_ids[idx] = 20_001 + idx

    next_bid_rest_allowed_ts = 0
    next_ask_rest_allowed_ts = 0
    last_record_ts = 0
    last_pos = hbt.position(0)
    last_trades = hbt.state_values(0).num_trades
    last_trading_value = hbt.state_values(0).trading_value

    signal_ts = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.int64)
    signal_bid = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    signal_ask = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    signal_bid_qty = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    signal_ask_qty = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    write_idx = 0
    count = 0

    while hbt.elapse(ORDER_UPDATE_INTERVAL_NS) == 0:
        if hbt.current_timestamp >= end_close_ts_ns:
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

        depth = hbt.depth(0)
        if depth.best_bid <= 0 or depth.best_ask <= 0:
            for idx in range(loop_levels):
                if cancel_order(hbt, bid_order_ids[idx]):
                    metrics[24] += 1
                if cancel_order(hbt, ask_order_ids[idx]):
                    metrics[25] += 1
        else:
            mid = (depth.best_bid + depth.best_ask) / 2.0
            fair = current_fair_price(depth)
            pos = hbt.position(0)
            vol_bps = recent_volatility_bps(signal_ts, signal_bid, signal_ask, write_idx, count)

            for idx in range(loop_levels):
                should_bid = should_quote_level(strategy_mode, 1, idx, pos)
                bid_px = quote_price_for_level(
                    strategy_mode,
                    1,
                    idx,
                    fair,
                    mid,
                    depth.best_bid,
                    depth.best_ask,
                    pos,
                    vol_bps,
                )
                bid_inflight_until[idx], next_rest_allowed_ts, bid_live_since[idx] = manage_quote(
                    hbt,
                    1,
                    bid_order_ids[idx],
                    bid_px,
                    should_bid,
                    bid_inflight_until[idx],
                    next_bid_rest_allowed_ts,
                    bid_live_since[idx],
                    metrics,
                )
                next_bid_rest_allowed_ts = next_rest_allowed_ts

                should_ask = should_quote_level(strategy_mode, -1, idx, pos)
                ask_px = quote_price_for_level(
                    strategy_mode,
                    -1,
                    idx,
                    fair,
                    mid,
                    depth.best_bid,
                    depth.best_ask,
                    pos,
                    vol_bps,
                )
                ask_inflight_until[idx], next_rest_allowed_ts, ask_live_since[idx] = manage_quote(
                    hbt,
                    -1,
                    ask_order_ids[idx],
                    ask_px,
                    should_ask,
                    ask_inflight_until[idx],
                    next_ask_rest_allowed_ts,
                    ask_live_since[idx],
                    metrics,
                )
                next_ask_rest_allowed_ts = next_rest_allowed_ts

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
                fill_base = abs(bitmex_base_from_contracts(delta_contracts))
                fill_count = state.num_trades - last_trades
                metrics[0] += fill_base
                metrics[3] += fill_count
                if delta_contracts > 0:
                    metrics[4] += fill_count
                    if exec_px > 0:
                        metrics[18] += mid - exec_px
                        metrics[19] += 1
                        adverse_bps = ratio_minus_one_bps(exec_px, mid)
                        if adverse_bps >= TOXIC_FILL_MID_MOVE_BPS:
                            metrics[15] += 1
                elif delta_contracts < 0:
                    metrics[5] += fill_count
                    if exec_px > 0:
                        metrics[18] += exec_px - mid
                        metrics[19] += 1
                        adverse_bps = ratio_minus_one_bps(mid, exec_px)
                        if adverse_bps >= TOXIC_FILL_MID_MOVE_BPS:
                            metrics[15] += 1
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


def strategy_name(strategy_mode: int) -> str:
    return STRATEGY_NAMES[strategy_mode]


def strategy_slug(strategy_mode: int) -> str:
    return STRATEGY_SLUGS[strategy_mode]


def run_backtest(bitmex_npz: Path, yyyymmdd: str, strategy_mode: int) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    asset = (
        BacktestAsset()
        .data([str(bitmex_npz)])
        .linear_asset(BITMEX_CONTRACT_SIZE)
        .constant_order_latency(BITMEX_ORDER_ENTRY_LATENCY_NS, BITMEX_ORDER_RESPONSE_LATENCY_NS)
        .risk_adverse_queue_model()
        .no_partial_fill_exchange()
        .trading_value_fee_model(MAKER_FEE_RATE, TAKER_FEE_RATE)
        .tick_size(BITMEX_TICK_SIZE)
        .lot_size(BITMEX_LOT_SIZE)
        .last_trades_capacity(10_000)
    )
    hbt = HashMapMarketDepthBacktest([asset])
    recorder = Recorder(1, 100_000)
    metrics = np.zeros(40, dtype=np.float64)
    ok = run_strategy(hbt, recorder.recorder, metrics, end_close_ts_ns(yyyymmdd), strategy_mode)
    if not ok:
        raise RuntimeError("strategy returned false")

    tag = RESULT_TAG or default_result_tag(strategy_mode)
    out = RESULT_DIR / f"bitmex_xbtusdt_single_market_{strategy_slug(strategy_mode)}_mm_{tag}_{yyyymmdd}.npz"
    np.savez_compressed(out, **{"0": recorder.get(0), "metrics": metrics})
    write_summary(out, metrics, yyyymmdd, strategy_mode)
    return out


def signed_money(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.4f} USDT"


def signed_base(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.8f} BTC"


def default_result_tag(strategy_mode: int) -> str:
    ttl_ms = int(ORDER_TTL_NS / 1_000_000)
    if strategy_mode == STRATEGY_LADDER_GRID:
        return f"lv{LADDER_LEVELS}_sp{LADDER_SPACING_BPS:g}_ttl{ttl_ms}".replace(".", "p")
    if strategy_mode == STRATEGY_AVELLANEDA_STOIKOV:
        return f"hs{BASE_HALF_SPREAD_BPS:g}_ra{AS_RISK_AVERSION:g}_ttl{ttl_ms}".replace(".", "p")
    return f"hs{BASE_HALF_SPREAD_BPS:g}_ttl{ttl_ms}".replace(".", "p")


def strategy_config(strategy_mode: int) -> dict:
    base = {
        "base_half_spread_bps": BASE_HALF_SPREAD_BPS,
        "min_half_spread_ticks": MIN_HALF_SPREAD_TICKS,
        "max_position_contracts": MAX_POSITION_CONTRACTS,
        "soft_position_contracts": SOFT_POSITION_CONTRACTS,
        "order_qty_contracts": ORDER_QTY,
        "order_qty_base": ORDER_QTY * BITMEX_CONTRACT_SIZE,
        "maker_fee_rate": MAKER_FEE_RATE,
        "taker_fee_rate": TAKER_FEE_RATE,
        "order_ttl_ms": ORDER_TTL_NS / 1_000_000.0,
        "order_update_interval_ms": ORDER_UPDATE_INTERVAL_NS / 1_000_000.0,
        "command_inflight_ms": BITMEX_COMMAND_INFLIGHT_NS / 1_000_000.0,
        "rest_min_interval_ms": BITMEX_REST_MIN_INTERVAL_NS / 1_000_000.0,
        "order_entry_latency_ms": BITMEX_ORDER_ENTRY_LATENCY_NS / 1_000_000.0,
        "order_response_latency_ms": BITMEX_ORDER_RESPONSE_LATENCY_NS / 1_000_000.0,
        "vol_window_ms": VOL_WINDOW_NS / 1_000_000.0,
        "vol_spread_multiplier": VOL_SPREAD_MULTIPLIER,
        "toxic_fill_mid_move_bps": TOXIC_FILL_MID_MOVE_BPS,
        "tick_size": BITMEX_TICK_SIZE,
        "lot_size": BITMEX_LOT_SIZE,
        "queue_model": "risk_adverse_queue_model",
    }
    if strategy_mode == STRATEGY_AVELLANEDA_STOIKOV:
        base.update(
            {
                "risk_aversion": AS_RISK_AVERSION,
                "horizon_ms": AS_HORIZON_NS / 1_000_000.0,
                "liquidity_spread_bps": AS_LIQUIDITY_SPREAD_BPS,
                "inventory_spread_bps_at_soft_limit": AS_INVENTORY_SPREAD_BPS_AT_SOFT_LIMIT,
            }
        )
    elif strategy_mode == STRATEGY_LADDER_GRID:
        base.update(
            {
                "levels": LADDER_LEVELS,
                "grid_spacing_bps": LADDER_SPACING_BPS,
                "grid_min_spacing_ticks": LADDER_MIN_SPACING_TICKS,
                "inventory_skew_bps_at_soft_limit": LADDER_INVENTORY_SKEW_BPS_AT_SOFT_LIMIT,
            }
        )
    return base


def write_summary(result_npz: Path, metrics: np.ndarray, yyyymmdd: str, strategy_mode: int) -> None:
    data = np.load(result_npz)
    records = data["0"]
    final = records[-1]
    price = float(final["price"])
    final_contracts = float(final["position"])
    final_base = final_contracts * BITMEX_CONTRACT_SIZE
    equity_usdt = float(final["balance"]) + final_contracts * price * BITMEX_CONTRACT_SIZE - float(final["fee"])
    total_pnl_usdt = equity_usdt
    avg_capture = metrics[18] / metrics[19] if metrics[19] > 0 else 0.0
    pnl_status = "profit" if total_pnl_usdt > 0 else "loss" if total_pnl_usdt < 0 else "flat"
    pnl_status_zh = "赚钱" if total_pnl_usdt > 0 else "亏钱" if total_pnl_usdt < 0 else "不赚不亏"

    summary = {
        "date": yyyymmdd,
        "symbol": BITMEX_SYMBOL,
        "strategy": strategy_name(strategy_mode),
        "result_tag": RESULT_TAG,
        **strategy_config(strategy_mode),
        "pnl_status": pnl_status,
        "pnl_status_zh": pnl_status_zh,
        "total_pnl_usdt": total_pnl_usdt,
        "total_fee_usdt": float(final["fee"]),
        "total_filled_base": float(metrics[0]),
        "max_position_base": float(metrics[1]),
        "max_position_contracts_seen": float(metrics[2]),
        "maker_fills": int(metrics[3]),
        "buy_fills": int(metrics[4]),
        "sell_fills": int(metrics[5]),
        "bid_ttl_cancel_events": int(metrics[6]),
        "ask_ttl_cancel_events": int(metrics[7]),
        "bid_inventory_cancel_events": int(metrics[8]),
        "ask_inventory_cancel_events": int(metrics[9]),
        "bid_place_events": int(metrics[10]),
        "ask_place_events": int(metrics[11]),
        "bid_modify_events": int(metrics[12]),
        "ask_modify_events": int(metrics[13]),
        "force_close_pnl_usdt": float(metrics[14]),
        "toxic_fill_events": int(metrics[15]),
        "max_equity_usdt": float(metrics[16]),
        "min_equity_usdt": float(metrics[17]),
        "avg_spread_capture_usdt_per_btc": float(avg_capture),
        "spread_capture_events": int(metrics[19]),
        "bid_rest_pacing_skip_events": int(metrics[20]),
        "ask_rest_pacing_skip_events": int(metrics[21]),
        "bid_target_suppress_events": int(metrics[22]),
        "ask_target_suppress_events": int(metrics[23]),
        "bid_stale_cancel_events": int(metrics[24]),
        "ask_stale_cancel_events": int(metrics[25]),
        "final_position_contracts": final_contracts,
        "final_position_base": final_base,
        "final_flat": abs(final_contracts) < 1e-9,
        "equity_usdt": equity_usdt,
        "start_timestamp_ns": int(records[0]["timestamp"]),
        "end_timestamp_ns": int(final["timestamp"]),
        "records": int(len(records)),
        "num_trades": int(final["num_trades"]),
        "trading_value_usdt": float(final["trading_value"]),
    }
    summary_path = result_npz.with_suffix(".summary.json")
    report_path = result_npz.with_suffix(".report.md")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    report_path.write_text(render_report(summary))
    print(render_console_summary(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


def render_console_summary(summary: dict) -> str:
    return "\n".join(
        [
            "",
            f"========== BitMEX 单市场 {summary['strategy']} 回测 ==========",
            f"结论: {summary['pnl_status_zh']} ({signed_money(summary['total_pnl_usdt'])})",
            f"日期: {summary['date']}",
            f"市场: BitMEX {summary['symbol']}",
            f"最终仓位归零: {'是' if summary['final_flat'] else '否'}",
            f"总成交 base: {summary['total_filled_base']:,.8f} BTC",
            f"maker 成交次数: {summary['maker_fills']}，买={summary['buy_fills']}，卖={summary['sell_fills']}",
            f"最大仓位: {summary['max_position_contracts_seen']:,.0f} contracts / {summary['max_position_base']:,.8f} BTC",
            f"平均 spread capture: {summary['avg_spread_capture_usdt_per_btc']:,.4f} USDT/BTC",
            f"toxic fill: {summary['toxic_fill_events']}",
            f"REST pacing skip: bid={summary['bid_rest_pacing_skip_events']}, ask={summary['ask_rest_pacing_skip_events']}",
            f"TTL cancel: bid={summary['bid_ttl_cancel_events']}, ask={summary['ask_ttl_cancel_events']}",
            f"库存/目标 cancel: bid={summary['bid_inventory_cancel_events']}, ask={summary['ask_inventory_cancel_events']}",
            f"日终强平 PnL: {signed_money(summary['force_close_pnl_usdt'])}",
            f"最终仓位: {summary['final_position_contracts']:,.0f} contracts / {signed_base(summary['final_position_base'])}",
            "======================================================",
            "",
        ]
    )


def render_report(summary: dict) -> str:
    config_lines = []
    for key, value in summary.items():
        if key in {
            "date",
            "symbol",
            "strategy",
            "result_tag",
            "pnl_status",
            "pnl_status_zh",
            "total_pnl_usdt",
            "total_fee_usdt",
            "total_filled_base",
            "max_position_base",
            "max_position_contracts_seen",
            "maker_fills",
            "buy_fills",
            "sell_fills",
            "final_position_contracts",
            "final_position_base",
            "final_flat",
            "equity_usdt",
            "start_timestamp_ns",
            "end_timestamp_ns",
            "records",
            "num_trades",
            "trading_value_usdt",
        }:
            continue
        if key.endswith("_events") or key.endswith("_usdt") or key.startswith("avg_"):
            continue
        config_lines.append(f"- {key}: `{value}`")
    config_text = "\n".join(config_lines)
    return f"""# BitMEX {summary['symbol']} {summary['strategy']} 回测报告

本次回测结果为 **{summary['pnl_status_zh']}**，总 PnL 为 **{signed_money(summary['total_pnl_usdt'])}**。

## 参数

- 日期: `{summary['date']}`
- 市场: `BitMEX {summary['symbol']}`
{config_text}

## 结果

- 结论: **{summary['pnl_status_zh']}**
- 总 PnL: **{signed_money(summary['total_pnl_usdt'])}**
- 总成交 base: `{summary['total_filled_base']:,.8f} BTC`
- maker 成交次数: `{summary['maker_fills']}`
- 买成交: `{summary['buy_fills']}`
- 卖成交: `{summary['sell_fills']}`
- 平均 spread capture: `{summary['avg_spread_capture_usdt_per_btc']:,.4f} USDT/BTC`
- toxic fill: `{summary['toxic_fill_events']}`
- 最大仓位: `{summary['max_position_contracts_seen']:,.0f} contracts`, `{summary['max_position_base']:,.8f} BTC`
- 日终强平 PnL: `{signed_money(summary['force_close_pnl_usdt'])}`
- 最终仓位: `{summary['final_position_contracts']:,.0f} contracts`, `{signed_base(summary['final_position_base'])}`
"""


def parse_args(strategy_mode: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Backtest BitMEX XBTUSDT {strategy_name(strategy_mode)} single-market strategy."
    )
    parser.add_argument("--dates", nargs="+", default=list(DEFAULT_DATES), help="YYYYMMDD dates to run.")
    parser.add_argument("--skip-download", action="store_true", help="Use existing CSV/NPZ files only.")
    parser.add_argument("--buffer-rows", type=int, default=None, help="Override tardis conversion buffer rows.")
    parser.add_argument("--result-tag", default="", help="Optional tag used in output filenames.")
    parser.add_argument("--base-half-spread-bps", type=float, default=BASE_HALF_SPREAD_BPS)
    parser.add_argument("--min-half-spread-ticks", type=float, default=MIN_HALF_SPREAD_TICKS)
    parser.add_argument("--order-qty-contracts", type=float, default=ORDER_QTY)
    parser.add_argument("--maker-fee-rate", type=float, default=MAKER_FEE_RATE)
    parser.add_argument("--taker-fee-rate", type=float, default=TAKER_FEE_RATE)
    parser.add_argument("--order-ttl-ms", type=float, default=ORDER_TTL_NS / 1_000_000.0)
    parser.add_argument("--rest-min-interval-ms", type=float, default=BITMEX_REST_MIN_INTERVAL_NS / 1_000_000.0)
    parser.add_argument("--max-position-contracts", type=float, default=MAX_POSITION_CONTRACTS)
    parser.add_argument("--soft-position-contracts", type=float, default=SOFT_POSITION_CONTRACTS)
    parser.add_argument("--vol-window-ms", type=float, default=VOL_WINDOW_NS / 1_000_000.0)
    parser.add_argument("--vol-spread-multiplier", type=float, default=VOL_SPREAD_MULTIPLIER)
    parser.add_argument("--toxic-fill-mid-move-bps", type=float, default=TOXIC_FILL_MID_MOVE_BPS)

    if strategy_mode == STRATEGY_AVELLANEDA_STOIKOV:
        parser.add_argument("--risk-aversion", type=float, default=AS_RISK_AVERSION)
        parser.add_argument("--horizon-ms", type=float, default=AS_HORIZON_NS / 1_000_000.0)
        parser.add_argument("--liquidity-spread-bps", type=float, default=AS_LIQUIDITY_SPREAD_BPS)
        parser.add_argument(
            "--inventory-spread-bps",
            type=float,
            default=AS_INVENTORY_SPREAD_BPS_AT_SOFT_LIMIT,
            help="Extra half-spread at the soft inventory limit.",
        )
    elif strategy_mode == STRATEGY_LADDER_GRID:
        parser.add_argument("--levels", type=int, default=LADDER_LEVELS)
        parser.add_argument("--grid-spacing-bps", type=float, default=LADDER_SPACING_BPS)
        parser.add_argument("--grid-min-spacing-ticks", type=float, default=LADDER_MIN_SPACING_TICKS)
        parser.add_argument("--inventory-skew-bps", type=float, default=LADDER_INVENTORY_SKEW_BPS_AT_SOFT_LIMIT)
    return parser.parse_args()


def apply_args(args: argparse.Namespace, strategy_mode: int) -> None:
    global BASE_HALF_SPREAD_BPS
    global MIN_HALF_SPREAD_TICKS
    global ORDER_QTY
    global MAKER_FEE_RATE
    global TAKER_FEE_RATE
    global ORDER_TTL_NS
    global BITMEX_REST_MIN_INTERVAL_NS
    global MAX_POSITION_CONTRACTS
    global SOFT_POSITION_CONTRACTS
    global VOL_WINDOW_NS
    global VOL_SPREAD_MULTIPLIER
    global TOXIC_FILL_MID_MOVE_BPS
    global AS_RISK_AVERSION
    global AS_HORIZON_NS
    global AS_LIQUIDITY_SPREAD_BPS
    global AS_INVENTORY_SPREAD_BPS_AT_SOFT_LIMIT
    global LADDER_LEVELS
    global LADDER_SPACING_BPS
    global LADDER_MIN_SPACING_TICKS
    global LADDER_INVENTORY_SKEW_BPS_AT_SOFT_LIMIT
    global RESULT_TAG
    global CURRENT_STRATEGY_MODE

    BASE_HALF_SPREAD_BPS = args.base_half_spread_bps
    MIN_HALF_SPREAD_TICKS = args.min_half_spread_ticks
    ORDER_QTY = args.order_qty_contracts
    MAKER_FEE_RATE = args.maker_fee_rate
    TAKER_FEE_RATE = args.taker_fee_rate
    ORDER_TTL_NS = int(args.order_ttl_ms * 1_000_000)
    BITMEX_REST_MIN_INTERVAL_NS = int(args.rest_min_interval_ms * 1_000_000)
    MAX_POSITION_CONTRACTS = args.max_position_contracts
    SOFT_POSITION_CONTRACTS = args.soft_position_contracts
    VOL_WINDOW_NS = int(args.vol_window_ms * 1_000_000)
    VOL_SPREAD_MULTIPLIER = args.vol_spread_multiplier
    TOXIC_FILL_MID_MOVE_BPS = args.toxic_fill_mid_move_bps
    RESULT_TAG = args.result_tag
    CURRENT_STRATEGY_MODE = strategy_mode

    if strategy_mode == STRATEGY_AVELLANEDA_STOIKOV:
        AS_RISK_AVERSION = args.risk_aversion
        AS_HORIZON_NS = int(args.horizon_ms * 1_000_000)
        AS_LIQUIDITY_SPREAD_BPS = args.liquidity_spread_bps
        AS_INVENTORY_SPREAD_BPS_AT_SOFT_LIMIT = args.inventory_spread_bps
    elif strategy_mode == STRATEGY_LADDER_GRID:
        LADDER_LEVELS = max(1, min(MAX_LEVELS, args.levels))
        LADDER_SPACING_BPS = args.grid_spacing_bps
        LADDER_MIN_SPACING_TICKS = args.grid_min_spacing_ticks
        LADDER_INVENTORY_SKEW_BPS_AT_SOFT_LIMIT = args.inventory_skew_bps


def main(strategy_mode: int) -> None:
    args = parse_args(strategy_mode)
    apply_args(args, strategy_mode)

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    key = None if args.skip_download else tardis_key()
    outputs: list[Path] = []
    for yyyymmdd in args.dates:
        if not args.skip_download:
            download_file(BITMEX_EXCHANGE, "trades", BITMEX_SYMBOL, yyyymmdd, key)
            download_file(BITMEX_EXCHANGE, "incremental_book_L2", BITMEX_SYMBOL, yyyymmdd, key)
        bitmex_npz = convert_bitmex(BITMEX_SYMBOL, yyyymmdd, args.buffer_rows)
        outputs.append(run_backtest(bitmex_npz, yyyymmdd, strategy_mode))
    print("all_results=" + ",".join(str(path) for path in outputs))
