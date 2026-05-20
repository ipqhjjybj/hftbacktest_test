import argparse
import gzip
import json
import math
import os
import urllib.request
from datetime import datetime, timedelta, timezone
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
from hftbacktest.data.utils import tardis


DEFAULT_DATES = ("20260512", "20260513", "20260514")

BITMEX_EXCHANGE = "bitmex"
BITMEX_SYMBOL = "XBTUSDT"

BITMEX_TICK_SIZE = 0.5
BITMEX_LOT_SIZE = 100.0
# XBTUSDT linear sizing for this backtest:
# 100 contracts = 0.0001 BTC, so 1 contract = 0.000001 BTC.
BITMEX_CONTRACT_SIZE = 0.000001
BITMEX_ORDER_QTY = 100.0

ORDER_UPDATE_INTERVAL_NS = 10_000_000
BITMEX_COMMAND_INFLIGHT_NS = 80_000_000
BITMEX_REST_MIN_INTERVAL_NS = 700_000_000
BITMEX_ORDER_ENTRY_LATENCY_NS = 80_000_000
BITMEX_ORDER_RESPONSE_LATENCY_NS = 40_000_000
LIVE_LIKE_EXECUTION = True

# Single-market MM controls.
BASE_HALF_SPREAD_BPS = 3.0
MIN_HALF_SPREAD_TICKS = 1.0
MAX_POSITION_CONTRACTS = 1_000.0
SOFT_POSITION_CONTRACTS = 500.0
INVENTORY_SKEW_BPS_AT_SOFT_LIMIT = 4.0
ORDER_TTL_NS = 5_000_000_000
MIN_AMEND_TICKS = 5.0
STALE_MARKET_NS = 10_000_000_000
REST_RATE_LIMIT_WINDOW_NS = 60_000_000_000
REST_RATE_LIMIT_MAX_REQUESTS = 120
RATE_LIMIT_COOLDOWN_NS = 5_000_000_000

# Quote toxicity filters.
SIGNAL_HISTORY_LEN = 4096
SHORT_MOMENTUM_WINDOW_NS = 100_000_000
MOMENTUM_CANCEL_BPS = 0.8
MICROPRICE_CANCEL_BPS = 0.8
VOL_WINDOW_NS = 1_000_000_000
VOL_SPREAD_MULTIPLIER = 0.5
TOXIC_FILL_MID_MOVE_BPS = 1.5

CSV_DIR = Path("data/tardis_csv")
NPZ_DIR = Path("data/npz")
RESULT_DIR = Path("results")
RESULT_TAG = ""
MAKER_FEE_RATE = 0.0
TAKER_FEE_RATE = 0.0
LIVE_FILL_HAIRCUT = 1.0
LIVE_ADVERSE_BPS = 0.0
EXCHANGE_MODEL = "no_partial"
LIVE_L2_TRADE_THROUGH_PROBABILITY = 0.145
LIVE_L2_MIN_ORDER_AGE_NS = 0
MAX_FILL_ATTR_ROWS = 200_000
FILL_ATTR_COLS = 13


def load_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def tardis_key() -> str:
    load_env()
    key = os.environ.get("TARDIS_API_KEY") or os.environ.get("TARDIS_KEY")
    if not key:
        raise RuntimeError("TARDIS_API_KEY or TARDIS_KEY is required")
    return key


def dataset_url(exchange: str, data_type: str, symbol: str, yyyymmdd: str) -> str:
    return (
        f"https://datasets.tardis.dev/v1/{exchange}/{data_type}/"
        f"{yyyymmdd[:4]}/{yyyymmdd[4:6]}/{yyyymmdd[6:]}/{symbol}.csv.gz"
    )


def csv_path(exchange: str, data_type: str, symbol: str, yyyymmdd: str) -> Path:
    return CSV_DIR / f"{exchange}_{data_type}_{symbol}_{yyyymmdd}.csv.gz"


def npz_path(exchange: str, symbol: str, yyyymmdd: str) -> Path:
    return NPZ_DIR / f"{exchange}_{symbol}_{yyyymmdd}.npz"


def end_close_ts_ns(yyyymmdd: str) -> int:
    dt = datetime.strptime(yyyymmdd, "%Y%m%d").replace(tzinfo=timezone.utc)
    end = dt + timedelta(days=1) - timedelta(seconds=60)
    return int(end.timestamp() * 1_000_000_000)


def download_file(exchange: str, data_type: str, symbol: str, yyyymmdd: str, key: str) -> Path:
    out = csv_path(exchange, data_type, symbol, yyyymmdd)
    if out.exists() and out.stat().st_size > 0:
        print(f"exists {out}")
        return out

    tmp = out.with_suffix(out.suffix + ".tmp")
    url = dataset_url(exchange, data_type, symbol, yyyymmdd)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    print(f"download {url}")
    with urllib.request.urlopen(req, timeout=120) as response:
        with tmp.open("wb") as file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
    tmp.replace(out)
    return out


def count_gzip_rows(path: Path) -> int:
    with gzip.open(path, "rt") as file:
        return max(0, sum(1 for _ in file) - 1)


def convert_bitmex(symbol: str, yyyymmdd: str, buffer_rows: int | None = None) -> Path:
    out = npz_path(BITMEX_EXCHANGE, symbol, yyyymmdd)
    if out.exists() and out.stat().st_size > 0:
        print(f"exists {out}")
        return out

    trade_file = csv_path(BITMEX_EXCHANGE, "trades", symbol, yyyymmdd)
    depth_file = csv_path(BITMEX_EXCHANGE, "incremental_book_L2", symbol, yyyymmdd)
    if buffer_rows is None:
        rows = count_gzip_rows(trade_file) + count_gzip_rows(depth_file)
        buffer_rows = max(1_000_000, int(rows * 1.35) + 1_000_000)

    print(f"convert {BITMEX_EXCHANGE} {symbol}, buffer_rows={buffer_rows:,}")
    tardis.convert(
        [str(trade_file), str(depth_file)],
        output_filename=str(out),
        buffer_size=buffer_rows,
        snapshot_mode="process",
    )
    return out


@njit
def ceil_to_tick(px, tick_size):
    return math.ceil(px / tick_size) * tick_size


@njit
def floor_to_tick(px, tick_size):
    return math.floor(px / tick_size) * tick_size


@njit
def ratio_minus_one_bps(numerator, denominator):
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return math.nan
    return (numerator / denominator - 1.0) * 10_000.0


@njit
def bitmex_base_from_contracts(contracts, price):
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
def rest_pacing_ready(hbt, rest_state, next_rest_allowed_ts, metrics, pacing_metric):
    if hbt.current_timestamp < rest_state[2]:
        metrics[pacing_metric] += 1
        return False, next_rest_allowed_ts
    if hbt.current_timestamp < next_rest_allowed_ts:
        metrics[pacing_metric] += 1
        return False, next_rest_allowed_ts

    if REST_RATE_LIMIT_MAX_REQUESTS > 0 and REST_RATE_LIMIT_WINDOW_NS > 0:
        if rest_state[0] <= 0 or hbt.current_timestamp - rest_state[0] >= REST_RATE_LIMIT_WINDOW_NS:
            rest_state[0] = hbt.current_timestamp
            rest_state[1] = 0
        if rest_state[1] >= REST_RATE_LIMIT_MAX_REQUESTS:
            rest_state[2] = hbt.current_timestamp + RATE_LIMIT_COOLDOWN_NS
            metrics[34] += 1
            return False, next_rest_allowed_ts
        rest_state[1] += 1

    return True, hbt.current_timestamp + BITMEX_REST_MIN_INTERVAL_NS


@njit
def command_inflight_until(timestamp):
    if LIVE_LIKE_EXECUTION:
        round_trip_ns = BITMEX_ORDER_ENTRY_LATENCY_NS + BITMEX_ORDER_RESPONSE_LATENCY_NS
        if round_trip_ns > BITMEX_COMMAND_INFLIGHT_NS:
            return timestamp + round_trip_ns
    return timestamp + BITMEX_COMMAND_INFLIGHT_NS


@njit
def order_live_since(timestamp):
    if LIVE_LIKE_EXECUTION:
        return timestamp + BITMEX_ORDER_ENTRY_LATENCY_NS
    return timestamp


@njit
def audit_order_lifecycle(hbt, bid_order_id, ask_order_id, metrics):
    bid_order = hbt.orders(0).get(bid_order_id)
    if bid_order is not None:
        if bid_order.req == 1:
            metrics[46] += 1
        elif bid_order.req == 4:
            metrics[44] += 1
        elif bid_order.req == 7:
            metrics[48] += 1
        elif bid_order.req != 0:
            metrics[50] += 1
        elif bid_order.cancellable:
            metrics[38] += 1

        if bid_order.status == 2:
            metrics[40] += 1
        elif bid_order.status == 6 or bid_order.req == 6:
            metrics[42] += 1

    ask_order = hbt.orders(0).get(ask_order_id)
    if ask_order is not None:
        if ask_order.req == 1:
            metrics[47] += 1
        elif ask_order.req == 4:
            metrics[45] += 1
        elif ask_order.req == 7:
            metrics[49] += 1
        elif ask_order.req != 0:
            metrics[51] += 1
        elif ask_order.cancellable:
            metrics[39] += 1

        if ask_order.status == 2:
            metrics[41] += 1
        elif ask_order.status == 6 or ask_order.req == 6:
            metrics[43] += 1


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
def quote_point_changed(depth, last_bid, last_ask, last_bid_qty, last_ask_qty):
    return (
        depth.best_bid != last_bid
        or depth.best_ask != last_ask
        or depth.best_bid_qty != last_bid_qty
        or depth.best_ask_qty != last_ask_qty
    )


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
def toxic_signal(
    side,
    signal_ts,
    signal_bid,
    signal_ask,
    signal_bid_qty,
    signal_ask_qty,
    write_idx,
    count,
):
    if count <= 1:
        return False, 0.0, 0.0
    current_idx = (write_idx - 1) % SIGNAL_HISTORY_LEN

    if MOMENTUM_CANCEL_BPS > 0 and SHORT_MOMENTUM_WINDOW_NS > 0:
        past_idx = signal_at_or_before(
            signal_ts,
            write_idx,
            count,
            signal_ts[current_idx] - SHORT_MOMENTUM_WINDOW_NS,
        )
        if past_idx >= 0:
            current_mid = current_mid_from_history(signal_bid, signal_ask, current_idx)
            past_mid = current_mid_from_history(signal_bid, signal_ask, past_idx)
            move_bps = ratio_minus_one_bps(current_mid, past_mid)
            adverse_bps = -move_bps if side > 0 else move_bps
            if adverse_bps >= MOMENTUM_CANCEL_BPS:
                return True, move_bps, 1.0

    if MICROPRICE_CANCEL_BPS > 0:
        total_qty = signal_bid_qty[current_idx] + signal_ask_qty[current_idx]
        if total_qty > 0:
            bid = signal_bid[current_idx]
            ask = signal_ask[current_idx]
            mid = (bid + ask) / 2.0
            microprice = (ask * signal_bid_qty[current_idx] + bid * signal_ask_qty[current_idx]) / total_qty
            micro_bps = ratio_minus_one_bps(microprice, mid)
            adverse_bps = -micro_bps if side > 0 else micro_bps
            if adverse_bps >= MICROPRICE_CANCEL_BPS:
                return True, micro_bps, 2.0

    return False, 0.0, 0.0


@njit
def volatility_penalty_bps(signal_ts, signal_bid, signal_ask, write_idx, count):
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
    return abs(move_bps) * VOL_SPREAD_MULTIPLIER


@njit
def manage_bid(
    hbt,
    order_id,
    inflight_until,
    next_rest_allowed_ts,
    live_since,
    anchor_mid,
    signal_ts,
    signal_bid,
    signal_ask,
    signal_bid_qty,
    signal_ask_qty,
    write_idx,
    count,
    rest_state,
    metrics,
):
    if hbt.current_timestamp < inflight_until:
        return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid

    depth = hbt.depth(0)
    if depth.best_bid <= 0 or depth.best_ask <= 0:
        existing = hbt.orders(0).get(order_id)
        if existing is not None and existing.cancellable:
            ready, next_rest_allowed_ts = rest_pacing_ready(hbt, rest_state, next_rest_allowed_ts, metrics, 32)
            if not ready:
                return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid
            hbt.cancel(0, order_id, False)
            metrics[16] += 1
            return (
                order_id,
                command_inflight_until(hbt.current_timestamp),
                next_rest_allowed_ts,
                0,
                anchor_mid,
            )
        return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid

    existing = hbt.orders(0).get(order_id)
    if existing is not None and live_since > 0 and hbt.current_timestamp - live_since >= ORDER_TTL_NS:
        if existing.cancellable:
            ready, next_rest_allowed_ts = rest_pacing_ready(hbt, rest_state, next_rest_allowed_ts, metrics, 32)
            if not ready:
                return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid
            hbt.cancel(0, order_id, False)
            metrics[8] += 1
            return (
                order_id,
                command_inflight_until(hbt.current_timestamp),
                next_rest_allowed_ts,
                0,
                anchor_mid,
            )
        return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid

    is_toxic, value_bps, reason = toxic_signal(
        1,
        signal_ts,
        signal_bid,
        signal_ask,
        signal_bid_qty,
        signal_ask_qty,
        write_idx,
        count,
    )
    if is_toxic:
        metrics[10] += 1
        if reason == 1.0:
            metrics[18] += 1
        elif reason == 2.0:
            metrics[19] += 1
        if existing is not None and existing.cancellable:
            ready, next_rest_allowed_ts = rest_pacing_ready(hbt, rest_state, next_rest_allowed_ts, metrics, 32)
            if not ready:
                return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid
            hbt.cancel(0, order_id, False)
            metrics[6] += 1
            return (
                order_id,
                command_inflight_until(hbt.current_timestamp),
                next_rest_allowed_ts,
                0,
                anchor_mid,
            )
        metrics[12] += 1
        return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid

    pos = hbt.position(0)
    if pos + BITMEX_ORDER_QTY > MAX_POSITION_CONTRACTS:
        if existing is not None and existing.cancellable:
            ready, next_rest_allowed_ts = rest_pacing_ready(hbt, rest_state, next_rest_allowed_ts, metrics, 32)
            if not ready:
                return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid
            hbt.cancel(0, order_id, False)
            metrics[14] += 1
            return (
                order_id,
                command_inflight_until(hbt.current_timestamp),
                next_rest_allowed_ts,
                0,
                anchor_mid,
            )
        metrics[20] += 1
        return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid

    bid = depth.best_bid
    ask = depth.best_ask
    mid = (bid + ask) / 2.0
    total_qty = depth.best_bid_qty + depth.best_ask_qty
    fair = mid
    if total_qty > 0:
        fair = (ask * depth.best_bid_qty + bid * depth.best_ask_qty) / total_qty

    inv_ratio = pos / SOFT_POSITION_CONTRACTS
    if inv_ratio > 1.0:
        inv_ratio = 1.0
    elif inv_ratio < -1.0:
        inv_ratio = -1.0
    reservation = fair * (1.0 - inv_ratio * INVENTORY_SKEW_BPS_AT_SOFT_LIMIT / 10_000.0)
    min_half_spread_bps = MIN_HALF_SPREAD_TICKS * BITMEX_TICK_SIZE / mid * 10_000.0
    half_spread_bps = max(BASE_HALF_SPREAD_BPS + volatility_penalty_bps(signal_ts, signal_bid, signal_ask, write_idx, count), min_half_spread_bps)
    raw_bid = min(reservation * (1.0 - half_spread_bps / 10_000.0), depth.best_bid)
    bid_px = floor_to_tick(raw_bid, BITMEX_TICK_SIZE)
    if bid_px <= 0:
        return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid

    if existing is not None:
        price_changed = abs(existing.price - bid_px) >= MIN_AMEND_TICKS * BITMEX_TICK_SIZE
        qty_changed = existing.qty != BITMEX_ORDER_QTY
        if existing.cancellable and (price_changed or qty_changed):
            ready, next_rest_allowed_ts = rest_pacing_ready(hbt, rest_state, next_rest_allowed_ts, metrics, 32)
            if not ready:
                return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid
            hbt.modify(0, order_id, bid_px, BITMEX_ORDER_QTY, False)
            metrics[22] += 1
            return (
                order_id,
                command_inflight_until(hbt.current_timestamp),
                next_rest_allowed_ts,
                order_live_since(hbt.current_timestamp),
                mid,
            )
        return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid

    ready, next_rest_allowed_ts = rest_pacing_ready(hbt, rest_state, next_rest_allowed_ts, metrics, 32)
    if not ready:
        return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid
    hbt.submit_buy_order(0, order_id, bid_px, BITMEX_ORDER_QTY, GTX, LIMIT, False)
    metrics[24] += 1
    return (
        order_id,
        command_inflight_until(hbt.current_timestamp),
        next_rest_allowed_ts,
        order_live_since(hbt.current_timestamp),
        mid,
    )


@njit
def manage_ask(
    hbt,
    order_id,
    inflight_until,
    next_rest_allowed_ts,
    live_since,
    anchor_mid,
    signal_ts,
    signal_bid,
    signal_ask,
    signal_bid_qty,
    signal_ask_qty,
    write_idx,
    count,
    rest_state,
    metrics,
):
    if hbt.current_timestamp < inflight_until:
        return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid

    depth = hbt.depth(0)
    if depth.best_bid <= 0 or depth.best_ask <= 0:
        existing = hbt.orders(0).get(order_id)
        if existing is not None and existing.cancellable:
            ready, next_rest_allowed_ts = rest_pacing_ready(hbt, rest_state, next_rest_allowed_ts, metrics, 33)
            if not ready:
                return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid
            hbt.cancel(0, order_id, False)
            metrics[17] += 1
            return (
                order_id,
                command_inflight_until(hbt.current_timestamp),
                next_rest_allowed_ts,
                0,
                anchor_mid,
            )
        return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid

    existing = hbt.orders(0).get(order_id)
    if existing is not None and live_since > 0 and hbt.current_timestamp - live_since >= ORDER_TTL_NS:
        if existing.cancellable:
            ready, next_rest_allowed_ts = rest_pacing_ready(hbt, rest_state, next_rest_allowed_ts, metrics, 33)
            if not ready:
                return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid
            hbt.cancel(0, order_id, False)
            metrics[9] += 1
            return (
                order_id,
                command_inflight_until(hbt.current_timestamp),
                next_rest_allowed_ts,
                0,
                anchor_mid,
            )
        return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid

    is_toxic, value_bps, reason = toxic_signal(
        -1,
        signal_ts,
        signal_bid,
        signal_ask,
        signal_bid_qty,
        signal_ask_qty,
        write_idx,
        count,
    )
    if is_toxic:
        metrics[11] += 1
        if reason == 1.0:
            metrics[18] += 1
        elif reason == 2.0:
            metrics[19] += 1
        if existing is not None and existing.cancellable:
            ready, next_rest_allowed_ts = rest_pacing_ready(hbt, rest_state, next_rest_allowed_ts, metrics, 33)
            if not ready:
                return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid
            hbt.cancel(0, order_id, False)
            metrics[7] += 1
            return (
                order_id,
                command_inflight_until(hbt.current_timestamp),
                next_rest_allowed_ts,
                0,
                anchor_mid,
            )
        metrics[13] += 1
        return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid

    pos = hbt.position(0)
    if pos - BITMEX_ORDER_QTY < -MAX_POSITION_CONTRACTS:
        if existing is not None and existing.cancellable:
            ready, next_rest_allowed_ts = rest_pacing_ready(hbt, rest_state, next_rest_allowed_ts, metrics, 33)
            if not ready:
                return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid
            hbt.cancel(0, order_id, False)
            metrics[15] += 1
            return (
                order_id,
                command_inflight_until(hbt.current_timestamp),
                next_rest_allowed_ts,
                0,
                anchor_mid,
            )
        metrics[21] += 1
        return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid

    bid = depth.best_bid
    ask = depth.best_ask
    mid = (bid + ask) / 2.0
    total_qty = depth.best_bid_qty + depth.best_ask_qty
    fair = mid
    if total_qty > 0:
        fair = (ask * depth.best_bid_qty + bid * depth.best_ask_qty) / total_qty

    inv_ratio = pos / SOFT_POSITION_CONTRACTS
    if inv_ratio > 1.0:
        inv_ratio = 1.0
    elif inv_ratio < -1.0:
        inv_ratio = -1.0
    reservation = fair * (1.0 - inv_ratio * INVENTORY_SKEW_BPS_AT_SOFT_LIMIT / 10_000.0)
    min_half_spread_bps = MIN_HALF_SPREAD_TICKS * BITMEX_TICK_SIZE / mid * 10_000.0
    half_spread_bps = max(BASE_HALF_SPREAD_BPS + volatility_penalty_bps(signal_ts, signal_bid, signal_ask, write_idx, count), min_half_spread_bps)
    raw_ask = max(reservation * (1.0 + half_spread_bps / 10_000.0), depth.best_ask)
    ask_px = ceil_to_tick(raw_ask, BITMEX_TICK_SIZE)
    if ask_px <= 0:
        return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid

    if existing is not None:
        price_changed = abs(existing.price - ask_px) >= MIN_AMEND_TICKS * BITMEX_TICK_SIZE
        qty_changed = existing.qty != BITMEX_ORDER_QTY
        if existing.cancellable and (price_changed or qty_changed):
            ready, next_rest_allowed_ts = rest_pacing_ready(hbt, rest_state, next_rest_allowed_ts, metrics, 33)
            if not ready:
                return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid
            hbt.modify(0, order_id, ask_px, BITMEX_ORDER_QTY, False)
            metrics[23] += 1
            return (
                order_id,
                command_inflight_until(hbt.current_timestamp),
                next_rest_allowed_ts,
                order_live_since(hbt.current_timestamp),
                mid,
            )
        return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid

    ready, next_rest_allowed_ts = rest_pacing_ready(hbt, rest_state, next_rest_allowed_ts, metrics, 33)
    if not ready:
        return order_id, inflight_until, next_rest_allowed_ts, live_since, anchor_mid
    hbt.submit_sell_order(0, order_id, ask_px, BITMEX_ORDER_QTY, GTX, LIMIT, False)
    metrics[25] += 1
    return (
        order_id,
        command_inflight_until(hbt.current_timestamp),
        next_rest_allowed_ts,
        order_live_since(hbt.current_timestamp),
        mid,
    )


@njit
def update_risk_metrics(hbt, metrics):
    depth = hbt.depth(0)
    if depth.best_bid <= 0 or depth.best_ask <= 0:
        return
    mid = (depth.best_bid + depth.best_ask) / 2.0
    pos_contracts = abs(hbt.position(0))
    pos_base = abs(bitmex_base_from_contracts(pos_contracts, mid))
    metrics[1] = max(metrics[1], pos_base)
    metrics[2] = max(metrics[2], pos_contracts)
    equity = bitmex_equity_usdt(hbt)
    if not math.isfinite(equity):
        return
    if (
        (metrics[28] == 0.0 and metrics[29] == 0.0)
        or not math.isfinite(metrics[28])
        or not math.isfinite(metrics[29])
    ):
        metrics[28] = equity
        metrics[29] = equity
    metrics[28] = max(metrics[28], equity)
    metrics[29] = min(metrics[29], equity)


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
    metrics[26] = after - before
    update_risk_metrics(hbt, metrics)


@njit
def run_strategy(hbt, recorder, metrics, fill_attr, fill_attr_count, end_close_ts_ns):
    bid_order_id = 10_001
    ask_order_id = 20_001
    bid_inflight_until = 0
    ask_inflight_until = 0
    next_rest_allowed_ts = 0
    bid_live_since = 0
    ask_live_since = 0
    bid_anchor_mid = 0.0
    ask_anchor_mid = 0.0
    last_record_ts = 0
    last_pos = hbt.position(0)
    last_trades = hbt.state_values(0).num_trades
    last_trading_value = hbt.state_values(0).trading_value
    last_fee = hbt.state_values(0).fee
    last_market_update_ts = 0
    last_bid = 0.0
    last_ask = 0.0
    last_bid_qty = 0.0
    last_ask_qty = 0.0
    market_is_stale = False
    rest_state = np.zeros(3, dtype=np.int64)

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

        audit_order_lifecycle(hbt, bid_order_id, ask_order_id, metrics)
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
        if depth.best_bid > 0 and depth.best_ask > 0:
            if quote_point_changed(depth, last_bid, last_ask, last_bid_qty, last_ask_qty):
                last_market_update_ts = hbt.current_timestamp
                last_bid = depth.best_bid
                last_ask = depth.best_ask
                last_bid_qty = depth.best_bid_qty
                last_ask_qty = depth.best_ask_qty
                market_is_stale = False

            if (
                STALE_MARKET_NS > 0
                and last_market_update_ts > 0
                and hbt.current_timestamp - last_market_update_ts >= STALE_MARKET_NS
            ):
                bid_order = hbt.orders(0).get(bid_order_id)
                if (
                    hbt.current_timestamp >= bid_inflight_until
                    and bid_order is not None
                    and bid_order.cancellable
                ):
                    ready, next_rest_allowed_ts = rest_pacing_ready(
                        hbt, rest_state, next_rest_allowed_ts, metrics, 32
                    )
                    if ready:
                        hbt.cancel(0, bid_order_id, False)
                        metrics[16] += 1
                        bid_inflight_until = command_inflight_until(hbt.current_timestamp)
                        bid_live_since = 0
                ask_order = hbt.orders(0).get(ask_order_id)
                if (
                    hbt.current_timestamp >= ask_inflight_until
                    and ask_order is not None
                    and ask_order.cancellable
                ):
                    ready, next_rest_allowed_ts = rest_pacing_ready(
                        hbt, rest_state, next_rest_allowed_ts, metrics, 33
                    )
                    if ready:
                        hbt.cancel(0, ask_order_id, False)
                        metrics[17] += 1
                        ask_inflight_until = command_inflight_until(hbt.current_timestamp)
                        ask_live_since = 0
                if not market_is_stale:
                    metrics[35] += 1
                market_is_stale = True
                update_risk_metrics(hbt, metrics)
                if hbt.current_timestamp - last_record_ts >= 1_000_000_000:
                    recorder.record(hbt)
                    last_record_ts = hbt.current_timestamp
                continue

        bid_order_id, bid_inflight_until, next_rest_allowed_ts, bid_live_since, bid_anchor_mid = manage_bid(
            hbt,
            bid_order_id,
            bid_inflight_until,
            next_rest_allowed_ts,
            bid_live_since,
            bid_anchor_mid,
            signal_ts,
            signal_bid,
            signal_ask,
            signal_bid_qty,
            signal_ask_qty,
            write_idx,
            count,
            rest_state,
            metrics,
        )
        ask_order_id, ask_inflight_until, next_rest_allowed_ts, ask_live_since, ask_anchor_mid = manage_ask(
            hbt,
            ask_order_id,
            ask_inflight_until,
            next_rest_allowed_ts,
            ask_live_since,
            ask_anchor_mid,
            signal_ts,
            signal_bid,
            signal_ask,
            signal_bid_qty,
            signal_ask_qty,
            write_idx,
            count,
            rest_state,
            metrics,
        )

        state = hbt.state_values(0)
        if state.num_trades > last_trades:
            depth = hbt.depth(0)
            mid = (depth.best_bid + depth.best_ask) / 2.0
            delta_contracts = state.position - last_pos
            delta_value = state.trading_value - last_trading_value
            fee_delta = state.fee - last_fee
            exec_px = 0.0
            if abs(delta_contracts) > 0 and delta_value > 0:
                exec_px = delta_value / (abs(delta_contracts) * BITMEX_CONTRACT_SIZE)
            fill_base = abs(bitmex_base_from_contracts(delta_contracts, mid))
            metrics[0] += fill_base
            metrics[3] += state.num_trades - last_trades
            if delta_contracts > 0:
                metrics[4] += state.num_trades - last_trades
                if bid_anchor_mid > 0 and exec_px > 0:
                    capture = bid_anchor_mid - exec_px
                    metrics[30] += capture
                    metrics[31] += 1
                    adverse_bps = -ratio_minus_one_bps(mid, bid_anchor_mid)
                    if adverse_bps >= TOXIC_FILL_MID_MOVE_BPS:
                        metrics[27] += 1
                    if fill_attr_count[0] < fill_attr.shape[0] and exec_px > 0:
                        idx = fill_attr_count[0]
                        spread_capture = mid - exec_px
                        fill_attr[idx, 0] = hbt.current_timestamp
                        fill_attr[idx, 1] = 1.0
                        fill_attr[idx, 2] = abs(delta_contracts)
                        fill_attr[idx, 3] = abs(bitmex_base_from_contracts(delta_contracts, mid))
                        fill_attr[idx, 4] = exec_px
                        fill_attr[idx, 5] = mid
                        fill_attr[idx, 6] = spread_capture
                        fill_attr[idx, 7] = spread_capture / mid * 10_000.0
                        fill_attr[idx, 8] = delta_value
                        fill_attr[idx, 9] = fee_delta
                        fill_attr[idx, 10] = -fee_delta
                        fill_attr[idx, 11] = state.position
                        fill_attr[idx, 12] = state.num_trades - last_trades
                        fill_attr_count[0] += 1
            elif delta_contracts < 0:
                metrics[5] += state.num_trades - last_trades
                if ask_anchor_mid > 0 and exec_px > 0:
                    capture = exec_px - ask_anchor_mid
                    metrics[30] += capture
                    metrics[31] += 1
                    adverse_bps = ratio_minus_one_bps(mid, ask_anchor_mid)
                    if adverse_bps >= TOXIC_FILL_MID_MOVE_BPS:
                        metrics[27] += 1
                    if fill_attr_count[0] < fill_attr.shape[0] and exec_px > 0:
                        idx = fill_attr_count[0]
                        spread_capture = exec_px - mid
                        fill_attr[idx, 0] = hbt.current_timestamp
                        fill_attr[idx, 1] = -1.0
                        fill_attr[idx, 2] = abs(delta_contracts)
                        fill_attr[idx, 3] = abs(bitmex_base_from_contracts(delta_contracts, mid))
                        fill_attr[idx, 4] = exec_px
                        fill_attr[idx, 5] = mid
                        fill_attr[idx, 6] = spread_capture
                        fill_attr[idx, 7] = spread_capture / mid * 10_000.0
                        fill_attr[idx, 8] = delta_value
                        fill_attr[idx, 9] = fee_delta
                        fill_attr[idx, 10] = -fee_delta
                        fill_attr[idx, 11] = state.position
                        fill_attr[idx, 12] = state.num_trades - last_trades
                        fill_attr_count[0] += 1
            last_pos = state.position
            last_trades = state.num_trades
            last_trading_value = state.trading_value
            last_fee = state.fee

        update_risk_metrics(hbt, metrics)
        if hbt.current_timestamp - last_record_ts >= 1_000_000_000:
            recorder.record(hbt)
            last_record_ts = hbt.current_timestamp

    force_flatten(hbt, metrics)
    recorder.record(hbt)
    return True


def run_backtest(bitmex_npz: Path, yyyymmdd: str) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
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
    if EXCHANGE_MODEL == "no_partial":
        asset = asset.no_partial_fill_exchange()
    elif EXCHANGE_MODEL == "strict_no_partial":
        asset = asset.strict_no_partial_fill_exchange()
    elif EXCHANGE_MODEL == "live_l2":
        asset = asset.live_l2_no_partial_fill_exchange(
            LIVE_L2_TRADE_THROUGH_PROBABILITY,
            LIVE_L2_MIN_ORDER_AGE_NS,
        )
    elif EXCHANGE_MODEL == "partial":
        asset = asset.partial_fill_exchange()
    else:
        raise ValueError(f"unsupported exchange model: {EXCHANGE_MODEL}")
    hbt = HashMapMarketDepthBacktest([asset])
    recorder = Recorder(1, 100_000)
    metrics = np.zeros(60, dtype=np.float64)
    fill_attr = np.zeros((MAX_FILL_ATTR_ROWS, FILL_ATTR_COLS), dtype=np.float64)
    fill_attr_count = np.zeros(1, dtype=np.int64)
    ok = run_strategy(
        hbt,
        recorder.recorder,
        metrics,
        fill_attr,
        fill_attr_count,
        end_close_ts_ns(yyyymmdd),
    )
    if not ok:
        raise RuntimeError("strategy returned false")

    ttl_ms = int(ORDER_TTL_NS / 1_000_000)
    half_spread_tag = str(BASE_HALF_SPREAD_BPS).replace(".", "p")
    param_tag = RESULT_TAG or f"hs{half_spread_tag}_ttl{ttl_ms}"
    out = RESULT_DIR / f"bitmex_xbtusdt_single_market_mm_{param_tag}_{yyyymmdd}.npz"
    fill_attr = fill_attr[: int(fill_attr_count[0])]
    np.savez_compressed(out, **{"0": recorder.get(0), "metrics": metrics, "fill_attribution": fill_attr})
    write_summary(out, metrics, yyyymmdd)
    return out


def signed_money(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.4f} USDT"


def signed_base(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.8f} BTC"


def metric_value(metrics: np.ndarray, idx: int) -> float:
    return float(metrics[idx]) if idx < len(metrics) else 0.0


def metric_int(metrics: np.ndarray, idx: int) -> int:
    return int(metric_value(metrics, idx))


def write_fill_attribution_csv(result_npz: Path, records: np.ndarray, fill_attr: np.ndarray, yyyymmdd: str) -> None:
    out = result_npz.with_suffix(".fill_attribution.csv")
    header = [
        "fill_id",
        "date",
        "symbol",
        "ts_ns",
        "side",
        "exec_qty_contracts",
        "exec_qty_btc",
        "exec_price",
        "fill_mid",
        "spread_capture_usdt_per_btc",
        "spread_capture_bps",
        "notional_usdt",
        "fee_delta_usdt",
        "maker_rebate_usdt",
        "position_after_contracts",
        "trade_count_delta",
        "mid_1s",
        "mid_move_1s_bps",
        "adverse_1s_bps",
        "markout_1s_usdt_per_btc",
        "markout_1s_usdt",
        "mid_3s",
        "mid_move_3s_bps",
        "adverse_3s_bps",
        "markout_3s_usdt_per_btc",
        "markout_3s_usdt",
        "mid_10s",
        "mid_move_10s_bps",
        "adverse_10s_bps",
        "markout_10s_usdt_per_btc",
        "markout_10s_usdt",
    ]
    timestamps = records["timestamp"].astype(np.int64)
    mids = records["price"].astype(np.float64)

    def horizon_values(ts_ns: int, side: float, exec_price: float, fill_mid: float, qty_btc: float, horizon_s: int):
        idx = int(np.searchsorted(timestamps, ts_ns + horizon_s * 1_000_000_000, side="left"))
        if idx >= len(timestamps):
            return "", "", "", "", ""
        horizon_mid = float(mids[idx])
        mid_move_bps = ratio_minus_one_bps(horizon_mid, fill_mid)
        adverse_bps = -mid_move_bps if side > 0 else mid_move_bps
        markout_per_btc = horizon_mid - exec_price if side > 0 else exec_price - horizon_mid
        markout_usdt = markout_per_btc * qty_btc
        return (
            f"{horizon_mid:.8f}",
            f"{mid_move_bps:.8f}",
            f"{adverse_bps:.8f}",
            f"{markout_per_btc:.8f}",
            f"{markout_usdt:.12f}",
        )

    lines = [",".join(header)]
    for idx, row in enumerate(fill_attr):
        ts_ns = int(row[0])
        side = float(row[1])
        side_text = "Buy" if side > 0 else "Sell"
        qty_btc = float(row[3])
        exec_price = float(row[4])
        fill_mid = float(row[5])
        values = [
            f"bt-{yyyymmdd}-{idx + 1}",
            yyyymmdd,
            BITMEX_SYMBOL,
            str(ts_ns),
            side_text,
            f"{row[2]:.8f}",
            f"{qty_btc:.12f}",
            f"{exec_price:.8f}",
            f"{fill_mid:.8f}",
            f"{row[6]:.8f}",
            f"{row[7]:.8f}",
            f"{row[8]:.12f}",
            f"{row[9]:.12f}",
            f"{row[10]:.12f}",
            f"{row[11]:.8f}",
            f"{row[12]:.0f}",
        ]
        for horizon_s in (1, 3, 10):
            values.extend(horizon_values(ts_ns, side, exec_price, fill_mid, qty_btc, horizon_s))
        lines.append(",".join(values))
    out.write_text("\n".join(lines) + "\n")


def write_summary(result_npz: Path, metrics: np.ndarray | None = None, yyyymmdd: str = "") -> None:
    data = np.load(result_npz)
    records = data["0"]
    if metrics is None:
        metrics = data["metrics"] if "metrics" in data.files else np.zeros(40, dtype=np.float64)
    fill_attr = data["fill_attribution"] if "fill_attribution" in data.files else np.zeros((0, FILL_ATTR_COLS), dtype=np.float64)

    final = records[-1]
    price = float(final["price"])
    final_contracts = float(final["position"])
    final_base = final_contracts * BITMEX_CONTRACT_SIZE
    equity_usdt = float(final["balance"]) + final_contracts * price * BITMEX_CONTRACT_SIZE - float(final["fee"])
    total_pnl_usdt = equity_usdt
    total_fee_usdt = float(final["fee"])
    raw_gross_pnl_usdt = total_pnl_usdt + total_fee_usdt
    raw_trading_value_usdt = float(final["trading_value"])
    live_adjusted_trading_value_usdt = raw_trading_value_usdt * LIVE_FILL_HAIRCUT
    live_adjusted_gross_pnl_usdt = raw_gross_pnl_usdt * LIVE_FILL_HAIRCUT
    live_adjusted_fee_usdt = total_fee_usdt * LIVE_FILL_HAIRCUT
    live_adverse_cost_usdt = live_adjusted_trading_value_usdt * LIVE_ADVERSE_BPS / 10_000.0
    live_adjusted_total_pnl_usdt = live_adjusted_gross_pnl_usdt - live_adjusted_fee_usdt - live_adverse_cost_usdt
    avg_capture = metrics[30] / metrics[31] if metrics[31] > 0 else 0.0
    final_flat = abs(final_contracts) < 1e-9
    pnl_status = "profit" if total_pnl_usdt > 0 else "loss" if total_pnl_usdt < 0 else "flat"
    pnl_status_zh = "赚钱" if total_pnl_usdt > 0 else "亏钱" if total_pnl_usdt < 0 else "不赚不亏"

    summary = {
        "date": yyyymmdd,
        "symbol": BITMEX_SYMBOL,
        "strategy": "single_market_mm",
        "result_tag": RESULT_TAG,
        "base_half_spread_bps": BASE_HALF_SPREAD_BPS,
        "min_half_spread_ticks": MIN_HALF_SPREAD_TICKS,
        "max_position_contracts": MAX_POSITION_CONTRACTS,
        "soft_position_contracts": SOFT_POSITION_CONTRACTS,
        "inventory_skew_bps_at_soft_limit": INVENTORY_SKEW_BPS_AT_SOFT_LIMIT,
        "order_qty_contracts": BITMEX_ORDER_QTY,
        "order_qty_base": BITMEX_ORDER_QTY * BITMEX_CONTRACT_SIZE,
        "maker_fee_rate": MAKER_FEE_RATE,
        "taker_fee_rate": TAKER_FEE_RATE,
        "min_amend_ticks": MIN_AMEND_TICKS,
        "stale_market_ms": STALE_MARKET_NS / 1_000_000.0,
        "rest_rate_limit_window_ms": REST_RATE_LIMIT_WINDOW_NS / 1_000_000.0,
        "rest_rate_limit_max_requests": REST_RATE_LIMIT_MAX_REQUESTS,
        "rate_limit_cooldown_ms": RATE_LIMIT_COOLDOWN_NS / 1_000_000.0,
        "order_ttl_ms": ORDER_TTL_NS / 1_000_000.0,
        "order_update_interval_ms": ORDER_UPDATE_INTERVAL_NS / 1_000_000.0,
        "command_inflight_ms": BITMEX_COMMAND_INFLIGHT_NS / 1_000_000.0,
        "rest_min_interval_ms": BITMEX_REST_MIN_INTERVAL_NS / 1_000_000.0,
        "order_entry_latency_ms": BITMEX_ORDER_ENTRY_LATENCY_NS / 1_000_000.0,
        "order_response_latency_ms": BITMEX_ORDER_RESPONSE_LATENCY_NS / 1_000_000.0,
        "short_momentum_window_ms": SHORT_MOMENTUM_WINDOW_NS / 1_000_000.0,
        "momentum_cancel_bps": MOMENTUM_CANCEL_BPS,
        "microprice_cancel_bps": MICROPRICE_CANCEL_BPS,
        "vol_window_ms": VOL_WINDOW_NS / 1_000_000.0,
        "vol_spread_multiplier": VOL_SPREAD_MULTIPLIER,
        "toxic_fill_mid_move_bps": TOXIC_FILL_MID_MOVE_BPS,
        "tick_size": BITMEX_TICK_SIZE,
        "lot_size": BITMEX_LOT_SIZE,
        "queue_model": "risk_adverse_queue_model",
        "exchange_model": EXCHANGE_MODEL,
        "live_l2_trade_through_probability": LIVE_L2_TRADE_THROUGH_PROBABILITY,
        "live_l2_min_order_age_ms": LIVE_L2_MIN_ORDER_AGE_NS / 1_000_000.0,
        "execution_mode": "live_like" if LIVE_LIKE_EXECUTION else "legacy",
        "live_fill_haircut": LIVE_FILL_HAIRCUT,
        "live_adverse_bps": LIVE_ADVERSE_BPS,
        "live_adjustment_note": (
            "Diagnostic only: adjusted metrics scale raw fills, gross PnL, and fees by live_fill_haircut, "
            "then subtract live_adverse_bps on adjusted trading value; "
            "inventory path is not re-simulated."
        ),
        "pnl_status": pnl_status,
        "pnl_status_zh": pnl_status_zh,
        "raw_total_pnl_usdt": total_pnl_usdt,
        "raw_gross_pnl_before_fees_usdt": raw_gross_pnl_usdt,
        "raw_total_fee_usdt": total_fee_usdt,
        "raw_total_filled_base": float(metrics[0]),
        "raw_maker_fills": int(metrics[3]),
        "fill_attribution_rows": int(len(fill_attr)),
        "raw_trading_value_usdt": raw_trading_value_usdt,
        "live_adjusted_total_pnl_usdt": live_adjusted_total_pnl_usdt,
        "live_adjusted_gross_pnl_before_fees_usdt": live_adjusted_gross_pnl_usdt,
        "live_adjusted_total_fee_usdt": live_adjusted_fee_usdt,
        "live_adjusted_total_filled_base": float(metrics[0]) * LIVE_FILL_HAIRCUT,
        "live_adjusted_maker_fills": float(metrics[3]) * LIVE_FILL_HAIRCUT,
        "live_adjusted_buy_fills": float(metrics[4]) * LIVE_FILL_HAIRCUT,
        "live_adjusted_sell_fills": float(metrics[5]) * LIVE_FILL_HAIRCUT,
        "live_adjusted_trading_value_usdt": live_adjusted_trading_value_usdt,
        "live_adverse_cost_usdt": live_adverse_cost_usdt,
        "total_pnl_usdt": total_pnl_usdt,
        "total_fee_usdt": total_fee_usdt,
        "total_filled_base": float(metrics[0]),
        "max_position_base": float(metrics[1]),
        "max_position_contracts_seen": float(metrics[2]),
        "maker_fills": int(metrics[3]),
        "buy_fills": int(metrics[4]),
        "sell_fills": int(metrics[5]),
        "bid_toxic_cancel_events": int(metrics[6]),
        "ask_toxic_cancel_events": int(metrics[7]),
        "bid_ttl_cancel_events": int(metrics[8]),
        "ask_ttl_cancel_events": int(metrics[9]),
        "bid_toxic_signal_events": int(metrics[10]),
        "ask_toxic_signal_events": int(metrics[11]),
        "bid_toxic_suppress_events": int(metrics[12]),
        "ask_toxic_suppress_events": int(metrics[13]),
        "bid_inventory_cancel_events": int(metrics[14]),
        "ask_inventory_cancel_events": int(metrics[15]),
        "bid_stale_cancel_events": int(metrics[16]),
        "ask_stale_cancel_events": int(metrics[17]),
        "momentum_signal_events": int(metrics[18]),
        "microprice_signal_events": int(metrics[19]),
        "bid_inventory_suppress_events": int(metrics[20]),
        "ask_inventory_suppress_events": int(metrics[21]),
        "bid_modify_events": int(metrics[22]),
        "ask_modify_events": int(metrics[23]),
        "bid_place_events": int(metrics[24]),
        "ask_place_events": int(metrics[25]),
        "force_close_pnl_usdt": float(metrics[26]),
        "toxic_fill_events": int(metrics[27]),
        "max_equity_usdt": float(metrics[28]),
        "min_equity_usdt": float(metrics[29]),
        "avg_spread_capture_usdt_per_btc": float(avg_capture),
        "spread_capture_events": int(metrics[31]),
        "bid_rest_pacing_skip_events": int(metrics[32]),
        "ask_rest_pacing_skip_events": int(metrics[33]),
        "rate_limit_events": int(metrics[34]),
        "stale_market_events": int(metrics[35]),
        "bid_resting_samples": metric_int(metrics, 38),
        "ask_resting_samples": metric_int(metrics, 39),
        "bid_post_only_expire_events": metric_int(metrics, 40),
        "ask_post_only_expire_events": metric_int(metrics, 41),
        "bid_reject_events": metric_int(metrics, 42),
        "ask_reject_events": metric_int(metrics, 43),
        "bid_cancel_pending_samples": metric_int(metrics, 44),
        "ask_cancel_pending_samples": metric_int(metrics, 45),
        "bid_new_pending_samples": metric_int(metrics, 46),
        "ask_new_pending_samples": metric_int(metrics, 47),
        "bid_modify_pending_samples": metric_int(metrics, 48),
        "ask_modify_pending_samples": metric_int(metrics, 49),
        "bid_other_pending_samples": metric_int(metrics, 50),
        "ask_other_pending_samples": metric_int(metrics, 51),
        "final_position_contracts": final_contracts,
        "final_position_base": final_base,
        "final_flat": final_flat,
        "equity_usdt": equity_usdt,
        "start_timestamp_ns": int(records[0]["timestamp"]),
        "end_timestamp_ns": int(final["timestamp"]),
        "records": int(len(records)),
        "num_trades": int(final["num_trades"]),
        "trading_value_usdt": raw_trading_value_usdt,
    }
    summary_path = result_npz.with_suffix(".summary.json")
    report_path = result_npz.with_suffix(".report.md")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    report_path.write_text(render_report(summary))
    write_fill_attribution_csv(result_npz, records, fill_attr, yyyymmdd)
    print(render_console_summary(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


def render_console_summary(summary: dict) -> str:
    lines = [
        "",
        "========== BitMEX 单市场做市回测 ==========",
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
        f"rate limit events: {summary['rate_limit_events']}",
        f"execution mode: {summary['execution_mode']}",
        f"post-only expire: bid={summary['bid_post_only_expire_events']}, ask={summary['ask_post_only_expire_events']}",
        f"pending samples: new bid={summary['bid_new_pending_samples']}, new ask={summary['ask_new_pending_samples']}, cancel bid={summary['bid_cancel_pending_samples']}, cancel ask={summary['ask_cancel_pending_samples']}",
        f"stale market events: {summary['stale_market_events']}",
        f"toxic cancel: bid={summary['bid_toxic_cancel_events']}, ask={summary['ask_toxic_cancel_events']}",
        f"TTL cancel: bid={summary['bid_ttl_cancel_events']}, ask={summary['ask_ttl_cancel_events']}",
        f"库存 cancel: bid={summary['bid_inventory_cancel_events']}, ask={summary['ask_inventory_cancel_events']}",
        f"日终强平 PnL: {signed_money(summary['force_close_pnl_usdt'])}",
        f"最终仓位: {summary['final_position_contracts']:,.0f} contracts / {signed_base(summary['final_position_base'])}",
    ]
    if summary["live_fill_haircut"] < 0.999999:
        lines.extend(
            [
                "---------- 实盘 fill 折扣校准 ----------",
                f"fill haircut: {summary['live_fill_haircut']:.4f}",
                f"校准后成交次数: {summary['live_adjusted_maker_fills']:,.2f}",
                f"校准后总成交 base: {summary['live_adjusted_total_filled_base']:,.8f} BTC",
                f"校准后 gross PnL(不含手续费): {signed_money(summary['live_adjusted_gross_pnl_before_fees_usdt'])}",
                f"校准后手续费/返佣: {signed_money(-summary['live_adjusted_total_fee_usdt'])}",
                f"校准后 adverse cost: {signed_money(-summary['live_adverse_cost_usdt'])} ({summary['live_adverse_bps']:.4f} bps)",
                f"校准后近似总 PnL: {signed_money(summary['live_adjusted_total_pnl_usdt'])}",
            ]
        )
    lines.extend(["==========================================", ""])
    return "\n".join(lines)


def render_report(summary: dict) -> str:
    return f"""# BitMEX {summary['symbol']} 单市场做市回测报告

本次回测结果为 **{summary['pnl_status_zh']}**，总 PnL 为 **{signed_money(summary['total_pnl_usdt'])}**。

## 参数

- 日期: `{summary['date']}`
- 市场: `BitMEX {summary['symbol']}`
- 每边基础 half-spread: `{summary['base_half_spread_bps']} bps`
- 最小 half-spread: `{summary['min_half_spread_ticks']} ticks`
- 每单数量: `{summary['order_qty_contracts']} contracts`
- maker fee rate: `{summary['maker_fee_rate']}`
- taker fee rate: `{summary['taker_fee_rate']}`
- 最大仓位: `{summary['max_position_contracts']} contracts`
- soft 仓位: `{summary['soft_position_contracts']} contracts`
- soft 仓位处库存 skew: `{summary['inventory_skew_bps_at_soft_limit']} bps`
- quote TTL: `{summary['order_ttl_ms']} ms`
- 最小 amend 价格变化: `{summary['min_amend_ticks']} ticks`
- stale market: `{summary['stale_market_ms']} ms`
- REST rate window: `{summary['rest_rate_limit_window_ms']} ms`
- REST rate max requests: `{summary['rest_rate_limit_max_requests']}`
- rate limit cooldown: `{summary['rate_limit_cooldown_ms']} ms`
- 更新间隔: `{summary['order_update_interval_ms']} ms`
- REST 最小间隔: `{summary['rest_min_interval_ms']} ms`
- 订单延迟: `{summary['order_entry_latency_ms']} ms entry`, `{summary['order_response_latency_ms']} ms response`
- 短动量窗口: `{summary['short_momentum_window_ms']} ms`
- 短动量撤单阈值: `{summary['momentum_cancel_bps']} bps`
- microprice 撤单阈值: `{summary['microprice_cancel_bps']} bps`
- 波动加宽窗口: `{summary['vol_window_ms']} ms`
- 波动加宽系数: `{summary['vol_spread_multiplier']}`
- toxic fill 阈值: `{summary['toxic_fill_mid_move_bps']} bps`
- queue model: `{summary['queue_model']}`
- exchange model: `{summary['exchange_model']}`
- live_l2 trade-through probability: `{summary['live_l2_trade_through_probability']}`
- live_l2 min order age: `{summary['live_l2_min_order_age_ms']} ms`
- execution mode: `{summary['execution_mode']}`
- live fill haircut: `{summary['live_fill_haircut']}`
- live adverse bps: `{summary['live_adverse_bps']}`

## 结果

- 结论: **{summary['pnl_status_zh']}**
- 总 PnL: **{signed_money(summary['total_pnl_usdt'])}**
- gross PnL(不含手续费): `{signed_money(summary['raw_gross_pnl_before_fees_usdt'])}`
- 手续费/返佣贡献: `{signed_money(-summary['raw_total_fee_usdt'])}`
- 总成交 base: `{summary['total_filled_base']:,.8f} BTC`
- maker 成交次数: `{summary['maker_fills']}`
- 买成交: `{summary['buy_fills']}`
- 卖成交: `{summary['sell_fills']}`
- 平均 spread capture: `{summary['avg_spread_capture_usdt_per_btc']:,.4f} USDT/BTC`
- toxic fill: `{summary['toxic_fill_events']}`
- bid REST pacing skip: `{summary['bid_rest_pacing_skip_events']}`
- ask REST pacing skip: `{summary['ask_rest_pacing_skip_events']}`
- rate limit events: `{summary['rate_limit_events']}`
- stale market events: `{summary['stale_market_events']}`
- bid resting samples: `{summary['bid_resting_samples']}`
- ask resting samples: `{summary['ask_resting_samples']}`
- bid post-only expire events: `{summary['bid_post_only_expire_events']}`
- ask post-only expire events: `{summary['ask_post_only_expire_events']}`
- bid rejected events: `{summary['bid_reject_events']}`
- ask rejected events: `{summary['ask_reject_events']}`
- bid new pending samples: `{summary['bid_new_pending_samples']}`
- ask new pending samples: `{summary['ask_new_pending_samples']}`
- bid cancel pending samples: `{summary['bid_cancel_pending_samples']}`
- ask cancel pending samples: `{summary['ask_cancel_pending_samples']}`
- bid modify pending samples: `{summary['bid_modify_pending_samples']}`
- ask modify pending samples: `{summary['ask_modify_pending_samples']}`
- bid toxic cancel: `{summary['bid_toxic_cancel_events']}`
- ask toxic cancel: `{summary['ask_toxic_cancel_events']}`
- bid TTL cancel: `{summary['bid_ttl_cancel_events']}`
- ask TTL cancel: `{summary['ask_ttl_cancel_events']}`
- 最大仓位: `{summary['max_position_contracts_seen']:,.0f} contracts`, `{summary['max_position_base']:,.8f} BTC`
- 日终强平 PnL: `{signed_money(summary['force_close_pnl_usdt'])}`
- 最终仓位: `{summary['final_position_contracts']:,.0f} contracts`, `{signed_base(summary['final_position_base'])}`

## 实盘 fill 折扣校准

这个部分只是诊断口径：把原始回测成交数、gross PnL、手续费按 `live_fill_haircut` 等比例缩放，
再按校准后成交额扣掉 `live_adverse_bps` 的成交质量惩罚，没有重新模拟库存路径。
它用来估计 queue model 过于乐观时，策略表现大概会被打到什么水平。

- fill haircut: `{summary['live_fill_haircut']}`
- adverse bps: `{summary['live_adverse_bps']}`
- 校准后 maker 成交次数: `{summary['live_adjusted_maker_fills']:,.2f}`
- 校准后买成交: `{summary['live_adjusted_buy_fills']:,.2f}`
- 校准后卖成交: `{summary['live_adjusted_sell_fills']:,.2f}`
- 校准后总成交 base: `{summary['live_adjusted_total_filled_base']:,.8f} BTC`
- 校准后成交额: `{summary['live_adjusted_trading_value_usdt']:,.4f} USDT`
- 校准后 gross PnL(不含手续费): `{signed_money(summary['live_adjusted_gross_pnl_before_fees_usdt'])}`
- 校准后手续费/返佣贡献: `{signed_money(-summary['live_adjusted_total_fee_usdt'])}`
- 校准后 adverse cost: `{signed_money(-summary['live_adverse_cost_usdt'])}`
- 校准后近似总 PnL: **{signed_money(summary['live_adjusted_total_pnl_usdt'])}**
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest a single-market BitMEX XBTUSDT market making strategy.")
    parser.add_argument("--dates", nargs="+", default=list(DEFAULT_DATES), help="YYYYMMDD dates to run.")
    parser.add_argument("--skip-download", action="store_true", help="Use existing CSV/NPZ files only.")
    parser.add_argument("--buffer-rows", type=int, default=None, help="Override tardis conversion buffer rows.")
    parser.add_argument("--base-half-spread-bps", type=float, default=BASE_HALF_SPREAD_BPS)
    parser.add_argument("--maker-fee-rate", type=float, default=MAKER_FEE_RATE)
    parser.add_argument("--taker-fee-rate", type=float, default=TAKER_FEE_RATE)
    parser.add_argument("--min-amend-ticks", type=float, default=MIN_AMEND_TICKS)
    parser.add_argument("--stale-market-ms", type=float, default=STALE_MARKET_NS / 1_000_000.0)
    parser.add_argument(
        "--rest-rate-limit-window-ms",
        type=float,
        default=REST_RATE_LIMIT_WINDOW_NS / 1_000_000.0,
    )
    parser.add_argument("--rest-rate-limit-max-requests", type=int, default=REST_RATE_LIMIT_MAX_REQUESTS)
    parser.add_argument(
        "--rate-limit-cooldown-ms",
        type=float,
        default=RATE_LIMIT_COOLDOWN_NS / 1_000_000.0,
    )
    parser.add_argument("--order-ttl-ms", type=float, default=ORDER_TTL_NS / 1_000_000.0)
    parser.add_argument("--rest-min-interval-ms", type=float, default=BITMEX_REST_MIN_INTERVAL_NS / 1_000_000.0)
    parser.add_argument("--command-inflight-ms", type=float, default=BITMEX_COMMAND_INFLIGHT_NS / 1_000_000.0)
    parser.add_argument("--order-entry-latency-ms", type=float, default=BITMEX_ORDER_ENTRY_LATENCY_NS / 1_000_000.0)
    parser.add_argument("--order-response-latency-ms", type=float, default=BITMEX_ORDER_RESPONSE_LATENCY_NS / 1_000_000.0)
    parser.add_argument(
        "--execution-mode",
        choices=("live_like", "legacy"),
        default="live_like" if LIVE_LIKE_EXECUTION else "legacy",
        help=(
            "live_like starts TTL after exchange entry latency, waits for entry+response before "
            "reusing a slot, audits pending/resting/post-only-expired order states, and applies "
            "REST pacing to stale-market cancels. legacy keeps the earlier strategy-layer timing."
        ),
    )
    parser.add_argument("--max-position-contracts", type=float, default=MAX_POSITION_CONTRACTS)
    parser.add_argument("--soft-position-contracts", type=float, default=SOFT_POSITION_CONTRACTS)
    parser.add_argument("--inventory-skew-bps", type=float, default=INVENTORY_SKEW_BPS_AT_SOFT_LIMIT)
    parser.add_argument("--momentum-cancel-bps", type=float, default=MOMENTUM_CANCEL_BPS)
    parser.add_argument("--microprice-cancel-bps", type=float, default=MICROPRICE_CANCEL_BPS)
    parser.add_argument("--vol-spread-multiplier", type=float, default=VOL_SPREAD_MULTIPLIER)
    parser.add_argument(
        "--live-fill-haircut",
        type=float,
        default=LIVE_FILL_HAIRCUT,
        help=(
            "Scale raw simulated fills/PnL/fees for live calibration. "
            "Example: 0.25 means live fills are about 25%% of backtest fills."
        ),
    )
    parser.add_argument(
        "--live-adverse-bps",
        type=float,
        default=LIVE_ADVERSE_BPS,
        help=(
            "Extra adverse-selection cost in bps on live-adjusted trading value. "
            "Use this to stress-test worse live fill quality."
        ),
    )
    parser.add_argument(
        "--exchange-model",
        choices=("no_partial", "strict_no_partial", "live_l2", "partial"),
        default=EXCHANGE_MODEL,
        help=(
            "Exchange simulation model. live_l2 keeps same-price queue fills but only accepts a "
            "configured fraction of trade-through maker fills."
        ),
    )
    parser.add_argument(
        "--live-l2-trade-through-probability",
        type=float,
        default=LIVE_L2_TRADE_THROUGH_PROBABILITY,
        help="Deterministic acceptance probability for live_l2 trade-through maker fills.",
    )
    parser.add_argument(
        "--live-l2-min-order-age-ms",
        type=float,
        default=LIVE_L2_MIN_ORDER_AGE_NS / 1_000_000.0,
        help="Minimum exchange-resting age before live_l2 may accept a trade-through maker fill.",
    )
    parser.add_argument("--result-tag", default="", help="Optional tag used in output filenames.")
    return parser.parse_args()


def main() -> None:
    global BASE_HALF_SPREAD_BPS
    global ORDER_TTL_NS
    global BITMEX_REST_MIN_INTERVAL_NS
    global BITMEX_COMMAND_INFLIGHT_NS
    global BITMEX_ORDER_ENTRY_LATENCY_NS
    global BITMEX_ORDER_RESPONSE_LATENCY_NS
    global LIVE_LIKE_EXECUTION
    global MAX_POSITION_CONTRACTS
    global SOFT_POSITION_CONTRACTS
    global INVENTORY_SKEW_BPS_AT_SOFT_LIMIT
    global MOMENTUM_CANCEL_BPS
    global MICROPRICE_CANCEL_BPS
    global VOL_SPREAD_MULTIPLIER
    global RESULT_TAG
    global MAKER_FEE_RATE
    global TAKER_FEE_RATE
    global MIN_AMEND_TICKS
    global STALE_MARKET_NS
    global REST_RATE_LIMIT_WINDOW_NS
    global REST_RATE_LIMIT_MAX_REQUESTS
    global RATE_LIMIT_COOLDOWN_NS
    global LIVE_FILL_HAIRCUT
    global LIVE_ADVERSE_BPS
    global EXCHANGE_MODEL
    global LIVE_L2_TRADE_THROUGH_PROBABILITY
    global LIVE_L2_MIN_ORDER_AGE_NS

    args = parse_args()
    BASE_HALF_SPREAD_BPS = args.base_half_spread_bps
    MAKER_FEE_RATE = args.maker_fee_rate
    TAKER_FEE_RATE = args.taker_fee_rate
    MIN_AMEND_TICKS = args.min_amend_ticks
    STALE_MARKET_NS = int(args.stale_market_ms * 1_000_000)
    REST_RATE_LIMIT_WINDOW_NS = int(args.rest_rate_limit_window_ms * 1_000_000)
    REST_RATE_LIMIT_MAX_REQUESTS = args.rest_rate_limit_max_requests
    RATE_LIMIT_COOLDOWN_NS = int(args.rate_limit_cooldown_ms * 1_000_000)
    ORDER_TTL_NS = int(args.order_ttl_ms * 1_000_000)
    BITMEX_REST_MIN_INTERVAL_NS = int(args.rest_min_interval_ms * 1_000_000)
    BITMEX_COMMAND_INFLIGHT_NS = int(args.command_inflight_ms * 1_000_000)
    BITMEX_ORDER_ENTRY_LATENCY_NS = int(args.order_entry_latency_ms * 1_000_000)
    BITMEX_ORDER_RESPONSE_LATENCY_NS = int(args.order_response_latency_ms * 1_000_000)
    LIVE_LIKE_EXECUTION = args.execution_mode == "live_like"
    MAX_POSITION_CONTRACTS = args.max_position_contracts
    SOFT_POSITION_CONTRACTS = args.soft_position_contracts
    INVENTORY_SKEW_BPS_AT_SOFT_LIMIT = args.inventory_skew_bps
    MOMENTUM_CANCEL_BPS = args.momentum_cancel_bps
    MICROPRICE_CANCEL_BPS = args.microprice_cancel_bps
    VOL_SPREAD_MULTIPLIER = args.vol_spread_multiplier
    RESULT_TAG = args.result_tag
    if args.live_fill_haircut < 0 or args.live_fill_haircut > 1:
        raise ValueError("--live-fill-haircut must be between 0 and 1")
    if args.live_adverse_bps < 0:
        raise ValueError("--live-adverse-bps must be >= 0")
    LIVE_FILL_HAIRCUT = args.live_fill_haircut
    LIVE_ADVERSE_BPS = args.live_adverse_bps
    EXCHANGE_MODEL = args.exchange_model
    if args.live_l2_trade_through_probability < 0 or args.live_l2_trade_through_probability > 1:
        raise ValueError("--live-l2-trade-through-probability must be between 0 and 1")
    if args.live_l2_min_order_age_ms < 0:
        raise ValueError("--live-l2-min-order-age-ms must be >= 0")
    LIVE_L2_TRADE_THROUGH_PROBABILITY = args.live_l2_trade_through_probability
    LIVE_L2_MIN_ORDER_AGE_NS = int(args.live_l2_min_order_age_ms * 1_000_000)

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
