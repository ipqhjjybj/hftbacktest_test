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


ORDER_UPDATE_INTERVAL_NS = 10_000_000
ORDER_COMMAND_INFLIGHT_NS = 120_000_000
REST_MIN_INTERVAL_NS = 700_000_000
ORDER_TTL_NS = 2_000_000_000
MIN_AMEND_TICKS = 2.0

SIGNAL_HISTORY_LEN = 8192
OFI_WINDOW_NS = 250_000_000
MOMENTUM_WINDOW_NS = 250_000_000
VOL_WINDOW_NS = 1_000_000_000
STALE_BOOK_NS = 3_000_000_000

ORDER_QTY = BITMEX_ORDER_QTY
SOFT_POSITION_CONTRACTS = 500.0
MAX_POSITION_CONTRACTS = 1_000.0

BASE_HALF_SPREAD_BPS = 3.0
MIN_HALF_SPREAD_TICKS = 1.0
VOL_SPREAD_MULTIPLIER = 0.5
INVENTORY_SKEW_BPS_AT_SOFT_LIMIT = 4.0
INVENTORY_SPREAD_BPS_AT_SOFT_LIMIT = 1.5

MICROPRICE_ALPHA_MULT = 0.7
OFI_ALPHA_BPS = 1.2
MOMENTUM_ALPHA_MULT = 0.25
MAX_ALPHA_BPS = 3.0
ALPHA_CANCEL_BPS = 1.0
MAX_VOL_BPS = 6.0

MAKER_FEE_RATE = -0.0002
TAKER_FEE_RATE = 0.0007
QUEUE_MODEL = "risk_adverse"
QUEUE_POWER = 3.0
EXCHANGE_MODEL = "no_partial"
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
def command_inflight_until(ts):
    round_trip = BITMEX_ORDER_ENTRY_LATENCY_NS + BITMEX_ORDER_RESPONSE_LATENCY_NS
    return ts + max(ORDER_COMMAND_INFLIGHT_NS, round_trip)


@njit
def order_live_since(ts):
    return ts + BITMEX_ORDER_ENTRY_LATENCY_NS


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
def latest_idx(write_idx):
    return (write_idx - 1) % SIGNAL_HISTORY_LEN


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
    if count <= 1 or window_ns <= 0:
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
    value = ratio_minus_one_bps(micro, mid)
    if not math.isfinite(value):
        return 0.0
    return value


@njit
def event_ofi(bid, ask, bid_qty, ask_qty, prev_bid, prev_ask, prev_bid_qty, prev_ask_qty):
    value = 0.0
    if bid >= prev_bid:
        value += bid_qty
    if bid <= prev_bid:
        value -= prev_bid_qty
    if ask <= prev_ask:
        value -= ask_qty
    if ask >= prev_ask:
        value += prev_ask_qty
    return value


@njit
def rolling_ofi_norm(signal_ts, signal_bid, signal_ask, signal_bid_qty, signal_ask_qty, write_idx, count):
    if count <= 2 or OFI_WINDOW_NS <= 0:
        return 0.0

    cur_idx = latest_idx(write_idx)
    cutoff = signal_ts[cur_idx] - OFI_WINDOW_NS
    ofi_sum = 0.0
    depth_sum = 0.0
    steps = 0

    for offset in range(count - 1):
        idx = (write_idx - 1 - offset) % SIGNAL_HISTORY_LEN
        prev_idx = (idx - 1) % SIGNAL_HISTORY_LEN
        if signal_ts[idx] < cutoff or signal_ts[prev_idx] <= 0:
            break
        ofi_sum += event_ofi(
            signal_bid[idx],
            signal_ask[idx],
            signal_bid_qty[idx],
            signal_ask_qty[idx],
            signal_bid[prev_idx],
            signal_ask[prev_idx],
            signal_bid_qty[prev_idx],
            signal_ask_qty[prev_idx],
        )
        depth_sum += signal_bid_qty[idx] + signal_ask_qty[idx]
        steps += 1

    if steps <= 0 or depth_sum <= 0:
        return 0.0
    return clamp(ofi_sum / (depth_sum / steps), -2.0, 2.0)


@njit
def inventory_ratio(pos):
    if SOFT_POSITION_CONTRACTS <= 0:
        return 0.0
    return clamp(pos / SOFT_POSITION_CONTRACTS, -1.0, 1.0)


@njit
def fair_alpha_bps(depth, signal_ts, signal_bid, signal_ask, signal_bid_qty, signal_ask_qty, write_idx, count):
    micro_bps = microprice_bps(depth)
    ofi_norm = rolling_ofi_norm(
        signal_ts,
        signal_bid,
        signal_ask,
        signal_bid_qty,
        signal_ask_qty,
        write_idx,
        count,
    )
    momentum_bps = recent_move_bps(signal_ts, signal_bid, signal_ask, write_idx, count, MOMENTUM_WINDOW_NS)
    alpha = (
        MICROPRICE_ALPHA_MULT * micro_bps
        + OFI_ALPHA_BPS * ofi_norm
        + MOMENTUM_ALPHA_MULT * momentum_bps
    )
    return clamp(alpha, -MAX_ALPHA_BPS, MAX_ALPHA_BPS)


@njit
def side_is_allowed(side, pos, alpha_bps, vol_bps):
    if side > 0 and pos + ORDER_QTY > MAX_POSITION_CONTRACTS:
        return False, 1.0
    if side < 0 and pos - ORDER_QTY < -MAX_POSITION_CONTRACTS:
        return False, 1.0
    if vol_bps > MAX_VOL_BPS:
        return False, 2.0
    if side > 0 and alpha_bps < -ALPHA_CANCEL_BPS:
        return False, 3.0
    if side < 0 and alpha_bps > ALPHA_CANCEL_BPS:
        return False, 3.0
    return True, 0.0


@njit
def target_price(side, depth, pos, alpha_bps, vol_bps):
    mid = (depth.best_bid + depth.best_ask) / 2.0
    inv = inventory_ratio(pos)
    fair = mid * (1.0 + alpha_bps / 10_000.0)
    reservation = fair * (1.0 - inv * INVENTORY_SKEW_BPS_AT_SOFT_LIMIT / 10_000.0)
    min_half_spread = MIN_HALF_SPREAD_TICKS * BITMEX_TICK_SIZE / mid * 10_000.0
    half_spread = max(
        BASE_HALF_SPREAD_BPS
        + VOL_SPREAD_MULTIPLIER * vol_bps
        + abs(inv) * INVENTORY_SPREAD_BPS_AT_SOFT_LIMIT,
        min_half_spread,
    )
    if side > 0:
        return floor_to_tick(min(reservation * (1.0 - half_spread / 10_000.0), depth.best_bid), BITMEX_TICK_SIZE)
    return ceil_to_tick(max(reservation * (1.0 + half_spread / 10_000.0), depth.best_ask), BITMEX_TICK_SIZE)


@njit
def rest_ready(hbt, next_rest_allowed_ts, metrics, side):
    if hbt.current_timestamp < next_rest_allowed_ts:
        if side > 0:
            metrics[18] += 1
        else:
            metrics[19] += 1
        return False
    return True


@njit
def manage_side(hbt, side, order_id, inflight_until, next_rest_allowed_ts, live_since, px, allowed, reason, metrics):
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
            return command_inflight_until(hbt.current_timestamp), hbt.current_timestamp + REST_MIN_INTERVAL_NS, 0
        return inflight_until, next_rest_allowed_ts, live_since

    if not allowed or px <= 0:
        if reason > 0:
            idx = int(reason)
            if idx == 1:
                metrics[20] += 1
            elif idx == 2:
                metrics[21] += 1
            elif idx == 3:
                metrics[22] += 1
            elif idx == 4:
                metrics[23] += 1
        if existing is not None and existing.cancellable and rest_ready(hbt, next_rest_allowed_ts, metrics, side):
            hbt.cancel(0, order_id, False)
            if side > 0:
                metrics[12] += 1
            else:
                metrics[13] += 1
            return command_inflight_until(hbt.current_timestamp), hbt.current_timestamp + REST_MIN_INTERVAL_NS, 0
        return inflight_until, next_rest_allowed_ts, live_since

    if existing is not None:
        price_changed = abs(existing.price - px) >= MIN_AMEND_TICKS * BITMEX_TICK_SIZE
        qty_changed = existing.qty != ORDER_QTY
        if existing.cancellable and (price_changed or qty_changed) and rest_ready(hbt, next_rest_allowed_ts, metrics, side):
            hbt.modify(0, order_id, px, ORDER_QTY, False)
            if side > 0:
                metrics[14] += 1
            else:
                metrics[15] += 1
            return (
                command_inflight_until(hbt.current_timestamp),
                hbt.current_timestamp + REST_MIN_INTERVAL_NS,
                order_live_since(hbt.current_timestamp),
            )
        return inflight_until, next_rest_allowed_ts, live_since

    if rest_ready(hbt, next_rest_allowed_ts, metrics, side):
        if side > 0:
            hbt.submit_buy_order(0, order_id, px, ORDER_QTY, GTX, LIMIT, False)
            metrics[16] += 1
        else:
            hbt.submit_sell_order(0, order_id, px, ORDER_QTY, GTX, LIMIT, False)
            metrics[17] += 1
        return (
            command_inflight_until(hbt.current_timestamp),
            hbt.current_timestamp + REST_MIN_INTERVAL_NS,
            order_live_since(hbt.current_timestamp),
        )
    return inflight_until, next_rest_allowed_ts, live_since


@njit
def update_risk_metrics(hbt, metrics):
    depth = hbt.depth(0)
    if depth.best_bid <= 0 or depth.best_ask <= 0:
        return
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
    metrics[28] += abs(inventory_ratio(hbt.position(0)))
    metrics[29] += 1


@njit
def force_flatten(hbt, metrics):
    cancel_all_orders(hbt)
    hbt.elapse(1_000_000_000)
    pos = hbt.position(0)
    if abs(pos) <= 0:
        return
    depth = hbt.depth(0)
    if depth.best_bid <= 0 or depth.best_ask <= 0:
        return
    before = bitmex_equity_usdt(hbt)
    if pos > 0:
        hbt.submit_sell_order(0, 91_001, depth.best_bid, abs(pos), IOC, MARKET, True)
    else:
        hbt.submit_buy_order(0, 91_002, depth.best_ask, abs(pos), IOC, MARKET, True)
    hbt.elapse(1_000_000_000)
    metrics[8] = bitmex_equity_usdt(hbt) - before


@njit
def run_strategy(hbt, recorder, metrics, end_close_ts):
    bid_id = 12_001
    ask_id = 22_001
    bid_inflight_until = 0
    ask_inflight_until = 0
    bid_live_since = 0
    ask_live_since = 0
    next_rest_allowed_ts = 0
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

        depth = hbt.depth(0)
        bid_px = 0.0
        ask_px = 0.0
        bid_allowed = False
        ask_allowed = False
        bid_reason = 4.0
        ask_reason = 4.0

        if depth.best_bid > 0 and depth.best_ask > 0 and count > 1:
            cur_idx = latest_idx(write_idx)
            if hbt.current_timestamp - signal_ts[cur_idx] <= STALE_BOOK_NS:
                pos = hbt.position(0)
                alpha_bps = fair_alpha_bps(
                    depth,
                    signal_ts,
                    signal_bid,
                    signal_ask,
                    signal_bid_qty,
                    signal_ask_qty,
                    write_idx,
                    count,
                )
                vol_bps = recent_vol_bps(signal_ts, signal_bid, signal_ask, write_idx, count)
                bid_allowed, bid_reason = side_is_allowed(1, pos, alpha_bps, vol_bps)
                ask_allowed, ask_reason = side_is_allowed(-1, pos, alpha_bps, vol_bps)
                bid_px = target_price(1, depth, pos, alpha_bps, vol_bps)
                ask_px = target_price(-1, depth, pos, alpha_bps, vol_bps)
                metrics[24] += alpha_bps
                metrics[25] += 1
                metrics[26] = max(metrics[26], abs(alpha_bps))
                metrics[27] = max(metrics[27], vol_bps)

        bid_inflight_until, next_rest_allowed_ts, bid_live_since = manage_side(
            hbt,
            1,
            bid_id,
            bid_inflight_until,
            next_rest_allowed_ts,
            bid_live_since,
            bid_px,
            bid_allowed,
            bid_reason,
            metrics,
        )
        ask_inflight_until, next_rest_allowed_ts, ask_live_since = manage_side(
            hbt,
            -1,
            ask_id,
            ask_inflight_until,
            next_rest_allowed_ts,
            ask_live_since,
            ask_px,
            ask_allowed,
            ask_reason,
            metrics,
        )

        state = hbt.state_values(0)
        if state.num_trades > last_trades:
            delta_pos = state.position - last_pos
            trade_count = state.num_trades - last_trades
            metrics[0] += trade_count
            if delta_pos > 0:
                metrics[1] += trade_count
            elif delta_pos < 0:
                metrics[2] += trade_count
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
        .trading_value_fee_model(MAKER_FEE_RATE, TAKER_FEE_RATE)
        .tick_size(BITMEX_TICK_SIZE)
        .lot_size(BITMEX_LOT_SIZE)
        .last_trades_capacity(10_000)
    )

    if QUEUE_MODEL == "power_prob3":
        asset = asset.power_prob_queue_model3(QUEUE_POWER)
    elif QUEUE_MODEL == "power_prob":
        asset = asset.power_prob_queue_model(QUEUE_POWER)
    else:
        asset = asset.risk_adverse_queue_model()

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
    metrics = np.zeros(64, dtype=np.float64)
    ok = run_strategy(hbt, recorder.recorder, metrics, end_close_ts_ns(yyyymmdd))
    if not ok:
        raise RuntimeError("strategy returned false")

    tag = RESULT_TAG or default_result_tag()
    out = RESULT_DIR / f"bitmex_xbtusdt_inventory_imbalance_mm_{tag}_{yyyymmdd}.npz"
    np.savez_compressed(out, **{"0": recorder.get(0), "metrics": metrics})
    write_summary(out, yyyymmdd)
    return out


def signed_money(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.4f} USDT"


def default_result_tag() -> str:
    ttl_ms = int(ORDER_TTL_NS / 1_000_000)
    spread = str(BASE_HALF_SPREAD_BPS).replace(".", "p")
    return f"{QUEUE_MODEL}_{EXCHANGE_MODEL}_hs{spread}_ttl{ttl_ms}"


def config_summary() -> dict:
    return {
        "base_half_spread_bps": BASE_HALF_SPREAD_BPS,
        "min_half_spread_ticks": MIN_HALF_SPREAD_TICKS,
        "vol_spread_multiplier": VOL_SPREAD_MULTIPLIER,
        "inventory_skew_bps_at_soft_limit": INVENTORY_SKEW_BPS_AT_SOFT_LIMIT,
        "inventory_spread_bps_at_soft_limit": INVENTORY_SPREAD_BPS_AT_SOFT_LIMIT,
        "microprice_alpha_mult": MICROPRICE_ALPHA_MULT,
        "ofi_alpha_bps": OFI_ALPHA_BPS,
        "momentum_alpha_mult": MOMENTUM_ALPHA_MULT,
        "max_alpha_bps": MAX_ALPHA_BPS,
        "alpha_cancel_bps": ALPHA_CANCEL_BPS,
        "max_vol_bps": MAX_VOL_BPS,
        "order_qty_contracts": ORDER_QTY,
        "soft_position_contracts": SOFT_POSITION_CONTRACTS,
        "max_position_contracts": MAX_POSITION_CONTRACTS,
        "maker_fee_rate": MAKER_FEE_RATE,
        "taker_fee_rate": TAKER_FEE_RATE,
        "queue_model": QUEUE_MODEL,
        "queue_power": QUEUE_POWER,
        "exchange_model": EXCHANGE_MODEL,
        "live_l2_trade_through_probability": LIVE_L2_TRADE_THROUGH_PROBABILITY,
        "live_l2_min_order_age_ms": LIVE_L2_MIN_ORDER_AGE_NS / 1_000_000.0,
        "order_entry_latency_ms": BITMEX_ORDER_ENTRY_LATENCY_NS / 1_000_000.0,
        "order_response_latency_ms": BITMEX_ORDER_RESPONSE_LATENCY_NS / 1_000_000.0,
        "order_ttl_ms": ORDER_TTL_NS / 1_000_000.0,
        "rest_min_interval_ms": REST_MIN_INTERVAL_NS / 1_000_000.0,
        "ofi_window_ms": OFI_WINDOW_NS / 1_000_000.0,
        "momentum_window_ms": MOMENTUM_WINDOW_NS / 1_000_000.0,
        "vol_window_ms": VOL_WINDOW_NS / 1_000_000.0,
    }


def write_summary(result_npz: Path, yyyymmdd: str) -> None:
    data = np.load(result_npz)
    records = data["0"]
    metrics = data["metrics"]
    final = records[-1]
    final_price = float(final["price"])
    final_pos = float(final["position"])
    equity = float(final["balance"]) + final_pos * final_price * BITMEX_CONTRACT_SIZE - float(final["fee"])
    total_fee = float(final["fee"])
    avg_alpha = metrics[24] / metrics[25] if metrics[25] > 0 else 0.0
    avg_abs_inventory_ratio = metrics[28] / metrics[29] if metrics[29] > 0 else 0.0

    summary = {
        "date": yyyymmdd,
        "symbol": BITMEX_SYMBOL,
        "strategy": "inventory_imbalance_mm",
        **config_summary(),
        "total_pnl_usdt": equity,
        "gross_pnl_before_fee_usdt": equity + total_fee,
        "fee_usdt": total_fee,
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
        "rest_skip_bid": int(metrics[18]),
        "rest_skip_ask": int(metrics[19]),
        "gate_position": int(metrics[20]),
        "gate_volatility": int(metrics[21]),
        "gate_alpha_toxic": int(metrics[22]),
        "gate_bad_book_or_stale": int(metrics[23]),
        "avg_alpha_bps": float(avg_alpha),
        "max_abs_alpha_bps": float(metrics[26]),
        "max_vol_bps": float(metrics[27]),
        "avg_abs_inventory_ratio": float(avg_abs_inventory_ratio),
        "final_position_contracts": final_pos,
        "final_position_btc": final_pos * BITMEX_CONTRACT_SIZE,
        "records": int(len(records)),
        "result_npz": str(result_npz),
    }
    result_npz.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    result_npz.with_suffix(".report.md").write_text(render_report(summary))
    print(render_console_summary(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


def render_console_summary(summary: dict) -> str:
    return "\n".join(
        [
            "",
            "========== BitMEX inventory + imbalance MM 回测 ==========",
            f"日期: {summary['date']}",
            f"PnL: {signed_money(summary['total_pnl_usdt'])}",
            f"gross 不含手续费: {signed_money(summary['gross_pnl_before_fee_usdt'])}",
            f"手续费: {signed_money(summary['fee_usdt'])}",
            f"fills: {summary['fills']}，买={summary['buy_fills']}，卖={summary['sell_fills']}",
            f"最大仓位: {summary['max_position_contracts_seen']:,.0f} contracts",
            f"alpha: avg={summary['avg_alpha_bps']:.4f} bps, max_abs={summary['max_abs_alpha_bps']:.4f} bps",
            f"gate: pos={summary['gate_position']}, vol={summary['gate_volatility']}, alpha={summary['gate_alpha_toxic']}, stale={summary['gate_bad_book_or_stale']}",
            "========================================================",
            "",
        ]
    )


def render_report(summary: dict) -> str:
    return f"""# BitMEX Inventory + Imbalance MM Report

## Result

- date: `{summary['date']}`
- symbol: `{summary['symbol']}`
- total PnL: `{signed_money(summary['total_pnl_usdt'])}`
- gross before fee: `{signed_money(summary['gross_pnl_before_fee_usdt'])}`
- fee: `{signed_money(summary['fee_usdt'])}`
- fills: `{summary['fills']}`
- buy fills: `{summary['buy_fills']}`
- sell fills: `{summary['sell_fills']}`
- filled base: `{summary['filled_base_btc']:,.8f} BTC`
- max position: `{summary['max_position_contracts_seen']:,.0f} contracts`
- final position: `{summary['final_position_contracts']:,.0f} contracts`

## Model

- fair: `mid + microprice alpha + rolling OFI alpha + short momentum alpha`
- inventory: reservation price shifts by `inventory_skew_bps_at_soft_limit`
- spread: base spread plus volatility and inventory penalties
- queue model: `{summary['queue_model']}`
- exchange model: `{summary['exchange_model']}`
- maker/taker fee: `{summary['maker_fee_rate']}`, `{summary['taker_fee_rate']}`
- order latency: `{summary['order_entry_latency_ms']} ms entry`, `{summary['order_response_latency_ms']} ms response`

## Gates

- position gate: `{summary['gate_position']}`
- volatility gate: `{summary['gate_volatility']}`
- alpha toxicity gate: `{summary['gate_alpha_toxic']}`
- stale/bad book gate: `{summary['gate_bad_book_or_stale']}`
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BitMEX XBTUSDT inventory-constrained maker with microprice/OFI alpha."
    )
    parser.add_argument("--dates", nargs="+", default=list(DEFAULT_DATES), help="YYYYMMDD dates to run.")
    parser.add_argument("--skip-download", action="store_true", help="Use existing CSV/NPZ files only.")
    parser.add_argument("--buffer-rows", type=int, default=None, help="Override tardis conversion buffer rows.")
    parser.add_argument("--result-tag", default="", help="Optional tag used in output filenames.")

    parser.add_argument("--maker-fee-rate", type=float, default=MAKER_FEE_RATE)
    parser.add_argument("--taker-fee-rate", type=float, default=TAKER_FEE_RATE)
    parser.add_argument("--queue-model", choices=("risk_adverse", "power_prob", "power_prob3"), default=QUEUE_MODEL)
    parser.add_argument("--queue-power", type=float, default=QUEUE_POWER)
    parser.add_argument("--exchange-model", choices=("live_l2", "no_partial", "strict_no_partial", "partial"), default=EXCHANGE_MODEL)
    parser.add_argument("--live-l2-trade-through-probability", type=float, default=LIVE_L2_TRADE_THROUGH_PROBABILITY)
    parser.add_argument("--live-l2-min-order-age-ms", type=float, default=LIVE_L2_MIN_ORDER_AGE_NS / 1_000_000.0)

    parser.add_argument("--order-entry-latency-ms", type=float, default=BITMEX_ORDER_ENTRY_LATENCY_NS / 1_000_000.0)
    parser.add_argument("--order-response-latency-ms", type=float, default=BITMEX_ORDER_RESPONSE_LATENCY_NS / 1_000_000.0)
    parser.add_argument("--command-inflight-ms", type=float, default=ORDER_COMMAND_INFLIGHT_NS / 1_000_000.0)
    parser.add_argument("--rest-min-interval-ms", type=float, default=REST_MIN_INTERVAL_NS / 1_000_000.0)
    parser.add_argument("--order-ttl-ms", type=float, default=ORDER_TTL_NS / 1_000_000.0)
    parser.add_argument("--min-amend-ticks", type=float, default=MIN_AMEND_TICKS)

    parser.add_argument("--order-qty-contracts", type=float, default=ORDER_QTY)
    parser.add_argument("--soft-position-contracts", type=float, default=SOFT_POSITION_CONTRACTS)
    parser.add_argument("--max-position-contracts", type=float, default=MAX_POSITION_CONTRACTS)

    parser.add_argument("--base-half-spread-bps", type=float, default=BASE_HALF_SPREAD_BPS)
    parser.add_argument("--min-half-spread-ticks", type=float, default=MIN_HALF_SPREAD_TICKS)
    parser.add_argument("--vol-spread-multiplier", type=float, default=VOL_SPREAD_MULTIPLIER)
    parser.add_argument("--inventory-skew-bps", type=float, default=INVENTORY_SKEW_BPS_AT_SOFT_LIMIT)
    parser.add_argument("--inventory-spread-bps", type=float, default=INVENTORY_SPREAD_BPS_AT_SOFT_LIMIT)

    parser.add_argument("--microprice-alpha-mult", type=float, default=MICROPRICE_ALPHA_MULT)
    parser.add_argument("--ofi-alpha-bps", type=float, default=OFI_ALPHA_BPS)
    parser.add_argument("--momentum-alpha-mult", type=float, default=MOMENTUM_ALPHA_MULT)
    parser.add_argument("--max-alpha-bps", type=float, default=MAX_ALPHA_BPS)
    parser.add_argument("--alpha-cancel-bps", type=float, default=ALPHA_CANCEL_BPS)
    parser.add_argument("--max-vol-bps", type=float, default=MAX_VOL_BPS)
    parser.add_argument("--ofi-window-ms", type=float, default=OFI_WINDOW_NS / 1_000_000.0)
    parser.add_argument("--momentum-window-ms", type=float, default=MOMENTUM_WINDOW_NS / 1_000_000.0)
    parser.add_argument("--vol-window-ms", type=float, default=VOL_WINDOW_NS / 1_000_000.0)
    parser.add_argument("--stale-book-ms", type=float, default=STALE_BOOK_NS / 1_000_000.0)
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    global MAKER_FEE_RATE
    global TAKER_FEE_RATE
    global QUEUE_MODEL
    global QUEUE_POWER
    global EXCHANGE_MODEL
    global LIVE_L2_TRADE_THROUGH_PROBABILITY
    global LIVE_L2_MIN_ORDER_AGE_NS
    global BITMEX_ORDER_ENTRY_LATENCY_NS
    global BITMEX_ORDER_RESPONSE_LATENCY_NS
    global ORDER_COMMAND_INFLIGHT_NS
    global REST_MIN_INTERVAL_NS
    global ORDER_TTL_NS
    global MIN_AMEND_TICKS
    global ORDER_QTY
    global SOFT_POSITION_CONTRACTS
    global MAX_POSITION_CONTRACTS
    global BASE_HALF_SPREAD_BPS
    global MIN_HALF_SPREAD_TICKS
    global VOL_SPREAD_MULTIPLIER
    global INVENTORY_SKEW_BPS_AT_SOFT_LIMIT
    global INVENTORY_SPREAD_BPS_AT_SOFT_LIMIT
    global MICROPRICE_ALPHA_MULT
    global OFI_ALPHA_BPS
    global MOMENTUM_ALPHA_MULT
    global MAX_ALPHA_BPS
    global ALPHA_CANCEL_BPS
    global MAX_VOL_BPS
    global OFI_WINDOW_NS
    global MOMENTUM_WINDOW_NS
    global VOL_WINDOW_NS
    global STALE_BOOK_NS
    global RESULT_TAG

    MAKER_FEE_RATE = args.maker_fee_rate
    TAKER_FEE_RATE = args.taker_fee_rate
    QUEUE_MODEL = args.queue_model
    QUEUE_POWER = args.queue_power
    EXCHANGE_MODEL = args.exchange_model
    LIVE_L2_TRADE_THROUGH_PROBABILITY = args.live_l2_trade_through_probability
    LIVE_L2_MIN_ORDER_AGE_NS = int(args.live_l2_min_order_age_ms * 1_000_000)
    BITMEX_ORDER_ENTRY_LATENCY_NS = int(args.order_entry_latency_ms * 1_000_000)
    BITMEX_ORDER_RESPONSE_LATENCY_NS = int(args.order_response_latency_ms * 1_000_000)
    ORDER_COMMAND_INFLIGHT_NS = int(args.command_inflight_ms * 1_000_000)
    REST_MIN_INTERVAL_NS = int(args.rest_min_interval_ms * 1_000_000)
    ORDER_TTL_NS = int(args.order_ttl_ms * 1_000_000)
    MIN_AMEND_TICKS = args.min_amend_ticks
    ORDER_QTY = args.order_qty_contracts
    SOFT_POSITION_CONTRACTS = args.soft_position_contracts
    MAX_POSITION_CONTRACTS = args.max_position_contracts
    BASE_HALF_SPREAD_BPS = args.base_half_spread_bps
    MIN_HALF_SPREAD_TICKS = args.min_half_spread_ticks
    VOL_SPREAD_MULTIPLIER = args.vol_spread_multiplier
    INVENTORY_SKEW_BPS_AT_SOFT_LIMIT = args.inventory_skew_bps
    INVENTORY_SPREAD_BPS_AT_SOFT_LIMIT = args.inventory_spread_bps
    MICROPRICE_ALPHA_MULT = args.microprice_alpha_mult
    OFI_ALPHA_BPS = args.ofi_alpha_bps
    MOMENTUM_ALPHA_MULT = args.momentum_alpha_mult
    MAX_ALPHA_BPS = args.max_alpha_bps
    ALPHA_CANCEL_BPS = args.alpha_cancel_bps
    MAX_VOL_BPS = args.max_vol_bps
    OFI_WINDOW_NS = int(args.ofi_window_ms * 1_000_000)
    MOMENTUM_WINDOW_NS = int(args.momentum_window_ms * 1_000_000)
    VOL_WINDOW_NS = int(args.vol_window_ms * 1_000_000)
    STALE_BOOK_NS = int(args.stale_book_ms * 1_000_000)
    RESULT_TAG = args.result_tag

    if LIVE_L2_TRADE_THROUGH_PROBABILITY < 0 or LIVE_L2_TRADE_THROUGH_PROBABILITY > 1:
        raise ValueError("--live-l2-trade-through-probability must be between 0 and 1")
    if LIVE_L2_MIN_ORDER_AGE_NS < 0:
        raise ValueError("--live-l2-min-order-age-ms must be >= 0")
    if ORDER_QTY < BITMEX_LOT_SIZE:
        raise ValueError(f"--order-qty-contracts must be >= lot size {BITMEX_LOT_SIZE}")
    if SOFT_POSITION_CONTRACTS <= 0:
        raise ValueError("--soft-position-contracts must be > 0")
    if MAX_POSITION_CONTRACTS < ORDER_QTY:
        raise ValueError("--max-position-contracts must be >= --order-qty-contracts")


def main() -> None:
    args = parse_args()
    apply_args(args)

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
        outputs.append(run_backtest(bitmex_npz, yyyymmdd))
    print("all_results=" + ",".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
