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
BITMEX_SYMBOL = "XBTUSD"

BITMEX_TICK_SIZE = 0.5
BITMEX_LOT_SIZE = 100.0
BITMEX_CONTRACT_SIZE = 1.0
BITMEX_ORDER_QTY = 100.0

ORDER_UPDATE_INTERVAL_NS = 10_000_000
BITMEX_COMMAND_INFLIGHT_NS = 80_000_000
BITMEX_ORDER_ENTRY_LATENCY_NS = 80_000_000
BITMEX_ORDER_RESPONSE_LATENCY_NS = 40_000_000

# Single-market MM controls.
BASE_HALF_SPREAD_BPS = 2.5
MIN_HALF_SPREAD_TICKS = 1.0
MAX_POSITION_CONTRACTS = 1_000.0
SOFT_POSITION_CONTRACTS = 500.0
INVENTORY_SKEW_BPS_AT_SOFT_LIMIT = 4.0
ORDER_TTL_NS = 500_000_000

# Quote toxicity filters.
SIGNAL_HISTORY_LEN = 4096
SHORT_MOMENTUM_WINDOW_NS = 100_000_000
MOMENTUM_CANCEL_BPS = 1.0
MICROPRICE_CANCEL_BPS = 0.5
VOL_WINDOW_NS = 1_000_000_000
VOL_SPREAD_MULTIPLIER = 0.5
TOXIC_FILL_MID_MOVE_BPS = 1.5

CSV_DIR = Path("data/tardis_csv")
NPZ_DIR = Path("data/npz")
RESULT_DIR = Path("results")
RESULT_TAG = ""


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
    if price <= 0:
        return 0.0
    return contracts * BITMEX_CONTRACT_SIZE / price


@njit
def bitmex_equity_usdt(hbt):
    depth = hbt.depth(0)
    mid = (depth.best_bid + depth.best_ask) / 2.0
    state = hbt.state_values(0)
    equity_btc = -state.balance - state.position * BITMEX_CONTRACT_SIZE / mid - state.fee
    return equity_btc * mid


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
    live_since,
    anchor_mid,
    signal_ts,
    signal_bid,
    signal_ask,
    signal_bid_qty,
    signal_ask_qty,
    write_idx,
    count,
    metrics,
):
    if hbt.current_timestamp < inflight_until:
        return order_id, inflight_until, live_since, anchor_mid

    depth = hbt.depth(0)
    if depth.best_bid <= 0 or depth.best_ask <= 0:
        if cancel_order(hbt, order_id):
            metrics[16] += 1
            return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, 0, anchor_mid
        return order_id, inflight_until, live_since, anchor_mid

    existing = hbt.orders(0).get(order_id)
    if existing is not None and live_since > 0 and hbt.current_timestamp - live_since >= ORDER_TTL_NS:
        if existing.cancellable:
            hbt.cancel(0, order_id, False)
            metrics[8] += 1
            return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, 0, anchor_mid
        return order_id, inflight_until, live_since, anchor_mid

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
            hbt.cancel(0, order_id, False)
            metrics[6] += 1
            return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, 0, anchor_mid
        metrics[12] += 1
        return order_id, inflight_until, live_since, anchor_mid

    pos = hbt.position(0)
    if pos + BITMEX_ORDER_QTY > MAX_POSITION_CONTRACTS:
        if existing is not None and existing.cancellable:
            hbt.cancel(0, order_id, False)
            metrics[14] += 1
            return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, 0, anchor_mid
        metrics[20] += 1
        return order_id, inflight_until, live_since, anchor_mid

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
        return order_id, inflight_until, live_since, anchor_mid

    if existing is not None:
        if existing.cancellable and (existing.price != bid_px or existing.qty != BITMEX_ORDER_QTY):
            hbt.modify(0, order_id, bid_px, BITMEX_ORDER_QTY, False)
            metrics[22] += 1
            return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, hbt.current_timestamp, mid
        return order_id, inflight_until, live_since, anchor_mid

    hbt.submit_buy_order(0, order_id, bid_px, BITMEX_ORDER_QTY, GTX, LIMIT, False)
    metrics[24] += 1
    return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, hbt.current_timestamp, mid


@njit
def manage_ask(
    hbt,
    order_id,
    inflight_until,
    live_since,
    anchor_mid,
    signal_ts,
    signal_bid,
    signal_ask,
    signal_bid_qty,
    signal_ask_qty,
    write_idx,
    count,
    metrics,
):
    if hbt.current_timestamp < inflight_until:
        return order_id, inflight_until, live_since, anchor_mid

    depth = hbt.depth(0)
    if depth.best_bid <= 0 or depth.best_ask <= 0:
        if cancel_order(hbt, order_id):
            metrics[17] += 1
            return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, 0, anchor_mid
        return order_id, inflight_until, live_since, anchor_mid

    existing = hbt.orders(0).get(order_id)
    if existing is not None and live_since > 0 and hbt.current_timestamp - live_since >= ORDER_TTL_NS:
        if existing.cancellable:
            hbt.cancel(0, order_id, False)
            metrics[9] += 1
            return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, 0, anchor_mid
        return order_id, inflight_until, live_since, anchor_mid

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
            hbt.cancel(0, order_id, False)
            metrics[7] += 1
            return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, 0, anchor_mid
        metrics[13] += 1
        return order_id, inflight_until, live_since, anchor_mid

    pos = hbt.position(0)
    if pos - BITMEX_ORDER_QTY < -MAX_POSITION_CONTRACTS:
        if existing is not None and existing.cancellable:
            hbt.cancel(0, order_id, False)
            metrics[15] += 1
            return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, 0, anchor_mid
        metrics[21] += 1
        return order_id, inflight_until, live_since, anchor_mid

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
        return order_id, inflight_until, live_since, anchor_mid

    if existing is not None:
        if existing.cancellable and (existing.price != ask_px or existing.qty != BITMEX_ORDER_QTY):
            hbt.modify(0, order_id, ask_px, BITMEX_ORDER_QTY, False)
            metrics[23] += 1
            return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, hbt.current_timestamp, mid
        return order_id, inflight_until, live_since, anchor_mid

    hbt.submit_sell_order(0, order_id, ask_px, BITMEX_ORDER_QTY, GTX, LIMIT, False)
    metrics[25] += 1
    return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, hbt.current_timestamp, mid


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
def run_strategy(hbt, recorder, metrics, end_close_ts_ns):
    bid_order_id = 10_001
    ask_order_id = 20_001
    bid_inflight_until = 0
    ask_inflight_until = 0
    bid_live_since = 0
    ask_live_since = 0
    bid_anchor_mid = 0.0
    ask_anchor_mid = 0.0
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

        bid_order_id, bid_inflight_until, bid_live_since, bid_anchor_mid = manage_bid(
            hbt,
            bid_order_id,
            bid_inflight_until,
            bid_live_since,
            bid_anchor_mid,
            signal_ts,
            signal_bid,
            signal_ask,
            signal_bid_qty,
            signal_ask_qty,
            write_idx,
            count,
            metrics,
        )
        ask_order_id, ask_inflight_until, ask_live_since, ask_anchor_mid = manage_ask(
            hbt,
            ask_order_id,
            ask_inflight_until,
            ask_live_since,
            ask_anchor_mid,
            signal_ts,
            signal_bid,
            signal_ask,
            signal_bid_qty,
            signal_ask_qty,
            write_idx,
            count,
            metrics,
        )

        state = hbt.state_values(0)
        if state.num_trades > last_trades:
            depth = hbt.depth(0)
            mid = (depth.best_bid + depth.best_ask) / 2.0
            delta_contracts = state.position - last_pos
            delta_value = state.trading_value - last_trading_value
            exec_px = 0.0
            if abs(delta_contracts) > 0 and delta_value > 0:
                exec_px = abs(delta_contracts) * BITMEX_CONTRACT_SIZE / delta_value
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
            elif delta_contracts < 0:
                metrics[5] += state.num_trades - last_trades
                if ask_anchor_mid > 0 and exec_px > 0:
                    capture = exec_px - ask_anchor_mid
                    metrics[30] += capture
                    metrics[31] += 1
                    adverse_bps = ratio_minus_one_bps(mid, ask_anchor_mid)
                    if adverse_bps >= TOXIC_FILL_MID_MOVE_BPS:
                        metrics[27] += 1
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


def run_backtest(bitmex_npz: Path, yyyymmdd: str) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    asset = (
        BacktestAsset()
        .data([str(bitmex_npz)])
        .inverse_asset(BITMEX_CONTRACT_SIZE)
        .constant_order_latency(BITMEX_ORDER_ENTRY_LATENCY_NS, BITMEX_ORDER_RESPONSE_LATENCY_NS)
        .risk_adverse_queue_model()
        .no_partial_fill_exchange()
        .trading_value_fee_model(0.0, 0.0)
        .tick_size(BITMEX_TICK_SIZE)
        .lot_size(BITMEX_LOT_SIZE)
        .last_trades_capacity(10_000)
    )
    hbt = HashMapMarketDepthBacktest([asset])
    recorder = Recorder(1, 100_000)
    metrics = np.zeros(40, dtype=np.float64)
    ok = run_strategy(hbt, recorder.recorder, metrics, end_close_ts_ns(yyyymmdd))
    if not ok:
        raise RuntimeError("strategy returned false")

    ttl_ms = int(ORDER_TTL_NS / 1_000_000)
    half_spread_tag = str(BASE_HALF_SPREAD_BPS).replace(".", "p")
    param_tag = RESULT_TAG or f"hs{half_spread_tag}_ttl{ttl_ms}"
    out = RESULT_DIR / f"bitmex_xbtusd_single_market_mm_{param_tag}_{yyyymmdd}.npz"
    np.savez_compressed(out, **{"0": recorder.get(0), "metrics": metrics})
    write_summary(out, metrics, yyyymmdd)
    return out


def signed_money(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.4f} USDT"


def signed_base(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.8f} BTC"


def write_summary(result_npz: Path, metrics: np.ndarray | None = None, yyyymmdd: str = "") -> None:
    data = np.load(result_npz)
    records = data["0"]
    if metrics is None:
        metrics = data["metrics"] if "metrics" in data.files else np.zeros(40, dtype=np.float64)

    final = records[-1]
    price = float(final["price"])
    final_contracts = float(final["position"])
    final_base = final_contracts * BITMEX_CONTRACT_SIZE / price
    equity_btc = -float(final["balance"]) - final_contracts * BITMEX_CONTRACT_SIZE / price - float(final["fee"])
    total_pnl_usdt = equity_btc * price
    total_fee_usdt = float(final["fee"]) * price
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
        "order_ttl_ms": ORDER_TTL_NS / 1_000_000.0,
        "order_update_interval_ms": ORDER_UPDATE_INTERVAL_NS / 1_000_000.0,
        "command_inflight_ms": BITMEX_COMMAND_INFLIGHT_NS / 1_000_000.0,
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
        "pnl_status": pnl_status,
        "pnl_status_zh": pnl_status_zh,
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
        "final_position_contracts": final_contracts,
        "final_position_base": final_base,
        "final_flat": final_flat,
        "equity_btc": equity_btc,
        "start_timestamp_ns": int(records[0]["timestamp"]),
        "end_timestamp_ns": int(final["timestamp"]),
        "records": int(len(records)),
        "num_trades": int(final["num_trades"]),
        "trading_value_btc": float(final["trading_value"]),
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
            f"toxic cancel: bid={summary['bid_toxic_cancel_events']}, ask={summary['ask_toxic_cancel_events']}",
            f"TTL cancel: bid={summary['bid_ttl_cancel_events']}, ask={summary['ask_ttl_cancel_events']}",
            f"库存 cancel: bid={summary['bid_inventory_cancel_events']}, ask={summary['ask_inventory_cancel_events']}",
            f"日终强平 PnL: {signed_money(summary['force_close_pnl_usdt'])}",
            f"最终仓位: {summary['final_position_contracts']:,.0f} contracts / {signed_base(summary['final_position_base'])}",
            "==========================================",
            "",
        ]
    )


def render_report(summary: dict) -> str:
    return f"""# BitMEX XBTUSD 单市场做市回测报告

本次回测结果为 **{summary['pnl_status_zh']}**，总 PnL 为 **{signed_money(summary['total_pnl_usdt'])}**。

## 参数

- 日期: `{summary['date']}`
- 市场: `BitMEX {summary['symbol']}`
- 每边基础 half-spread: `{summary['base_half_spread_bps']} bps`
- 最小 half-spread: `{summary['min_half_spread_ticks']} ticks`
- 每单数量: `{summary['order_qty_contracts']} contracts`
- 最大仓位: `{summary['max_position_contracts']} contracts`
- soft 仓位: `{summary['soft_position_contracts']} contracts`
- soft 仓位处库存 skew: `{summary['inventory_skew_bps_at_soft_limit']} bps`
- quote TTL: `{summary['order_ttl_ms']} ms`
- 更新间隔: `{summary['order_update_interval_ms']} ms`
- 订单延迟: `{summary['order_entry_latency_ms']} ms entry`, `{summary['order_response_latency_ms']} ms response`
- 短动量窗口: `{summary['short_momentum_window_ms']} ms`
- 短动量撤单阈值: `{summary['momentum_cancel_bps']} bps`
- microprice 撤单阈值: `{summary['microprice_cancel_bps']} bps`
- 波动加宽窗口: `{summary['vol_window_ms']} ms`
- 波动加宽系数: `{summary['vol_spread_multiplier']}`
- toxic fill 阈值: `{summary['toxic_fill_mid_move_bps']} bps`
- queue model: `{summary['queue_model']}`

## 结果

- 结论: **{summary['pnl_status_zh']}**
- 总 PnL: **{signed_money(summary['total_pnl_usdt'])}**
- 总成交 base: `{summary['total_filled_base']:,.8f} BTC`
- maker 成交次数: `{summary['maker_fills']}`
- 买成交: `{summary['buy_fills']}`
- 卖成交: `{summary['sell_fills']}`
- 平均 spread capture: `{summary['avg_spread_capture_usdt_per_btc']:,.4f} USDT/BTC`
- toxic fill: `{summary['toxic_fill_events']}`
- bid toxic cancel: `{summary['bid_toxic_cancel_events']}`
- ask toxic cancel: `{summary['ask_toxic_cancel_events']}`
- bid TTL cancel: `{summary['bid_ttl_cancel_events']}`
- ask TTL cancel: `{summary['ask_ttl_cancel_events']}`
- 最大仓位: `{summary['max_position_contracts_seen']:,.0f} contracts`, `{summary['max_position_base']:,.8f} BTC`
- 日终强平 PnL: `{signed_money(summary['force_close_pnl_usdt'])}`
- 最终仓位: `{summary['final_position_contracts']:,.0f} contracts`, `{signed_base(summary['final_position_base'])}`
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest a single-market BitMEX XBTUSD market making strategy.")
    parser.add_argument("--dates", nargs="+", default=list(DEFAULT_DATES), help="YYYYMMDD dates to run.")
    parser.add_argument("--skip-download", action="store_true", help="Use existing CSV/NPZ files only.")
    parser.add_argument("--buffer-rows", type=int, default=None, help="Override tardis conversion buffer rows.")
    parser.add_argument("--base-half-spread-bps", type=float, default=BASE_HALF_SPREAD_BPS)
    parser.add_argument("--order-ttl-ms", type=float, default=ORDER_TTL_NS / 1_000_000.0)
    parser.add_argument("--max-position-contracts", type=float, default=MAX_POSITION_CONTRACTS)
    parser.add_argument("--soft-position-contracts", type=float, default=SOFT_POSITION_CONTRACTS)
    parser.add_argument("--inventory-skew-bps", type=float, default=INVENTORY_SKEW_BPS_AT_SOFT_LIMIT)
    parser.add_argument("--momentum-cancel-bps", type=float, default=MOMENTUM_CANCEL_BPS)
    parser.add_argument("--microprice-cancel-bps", type=float, default=MICROPRICE_CANCEL_BPS)
    parser.add_argument("--vol-spread-multiplier", type=float, default=VOL_SPREAD_MULTIPLIER)
    parser.add_argument("--result-tag", default="", help="Optional tag used in output filenames.")
    return parser.parse_args()


def main() -> None:
    global BASE_HALF_SPREAD_BPS
    global ORDER_TTL_NS
    global MAX_POSITION_CONTRACTS
    global SOFT_POSITION_CONTRACTS
    global INVENTORY_SKEW_BPS_AT_SOFT_LIMIT
    global MOMENTUM_CANCEL_BPS
    global MICROPRICE_CANCEL_BPS
    global VOL_SPREAD_MULTIPLIER
    global RESULT_TAG

    args = parse_args()
    BASE_HALF_SPREAD_BPS = args.base_half_spread_bps
    ORDER_TTL_NS = int(args.order_ttl_ms * 1_000_000)
    MAX_POSITION_CONTRACTS = args.max_position_contracts
    SOFT_POSITION_CONTRACTS = args.soft_position_contracts
    INVENTORY_SKEW_BPS_AT_SOFT_LIMIT = args.inventory_skew_bps
    MOMENTUM_CANCEL_BPS = args.momentum_cancel_bps
    MICROPRICE_CANCEL_BPS = args.microprice_cancel_bps
    VOL_SPREAD_MULTIPLIER = args.vol_spread_multiplier
    RESULT_TAG = args.result_tag

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
