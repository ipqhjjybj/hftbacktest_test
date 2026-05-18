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
    BUY_EVENT,
    DEPTH_EVENT,
    GTX,
    LIMIT,
    SELL_EVENT,
    TRADE_EVENT,
    HashMapMarketDepthBacktest,
    Recorder,
    event_dtype,
)
from hftbacktest.order import IOC, MARKET
from hftbacktest.data.validation import correct_event_order, correct_local_timestamp, validate_event_order
from hftbacktest.data.utils import tardis


DATE = "20260513"
DEFAULT_DATES = ("20260512", "20260513", "20260514")

BITMEX_EXCHANGE = "bitmex"
BITMEX_SYMBOL = "XBTUSD"
GATE_EXCHANGE = "gate-io-futures"
GATE_SYMBOL = "BTC_USDT"

OPEN_LONG_SPREAD_RATIO = 0.00035
CLOSE_SPREAD_RATIO = 0.00016
MAX_POSITION_BASE = 0.0001
ORDER_UPDATE_INTERVAL_NS = 10_000_000
END_CLOSE_TS_NS = 1_778_716_740_000_000_000
GATE_TAKER_SLIPPAGE_BPS = 6.0
BITMEX_COMMAND_INFLIGHT_NS = 80_000_000
BITMEX_FILL_REPORT_DELAY_NS = 10_000_000
LOCAL_GATE_SEND_DELAY_NS = 27_000
GATE_HEDGE_INFLIGHT_NS = 20_000_000
GATE_HEDGE_SEND_DELAY_NS = BITMEX_FILL_REPORT_DELAY_NS + LOCAL_GATE_SEND_DELAY_NS
BITMEX_ORDER_ENTRY_LATENCY_NS = 80_000_000
BITMEX_ORDER_RESPONSE_LATENCY_NS = 80_000_000
GATE_ORDER_ENTRY_LATENCY_NS = 20_000_000
GATE_ORDER_RESPONSE_LATENCY_NS = 20_000_000
GATE_SHORT_MOMENTUM_WINDOW_NS = 100_000_000
GATE_MOMENTUM_CANCEL_BPS = 1.0
GATE_MICROPRICE_CANCEL_BPS = 0.5
TOXIC_FILL_GATE_ANCHOR_MOVE_BPS = 1.5
TOXIC_FILL_COOLDOWN_NS = 1_000_000_000
GATE_SIGNAL_HISTORY_LEN = 4096

BITMEX_TICK_SIZE = 0.5
BITMEX_LOT_SIZE = 100.0
BITMEX_CONTRACT_SIZE = 1.0
BITMEX_ORDER_QTY = 100.0

GATE_TICK_SIZE = 0.1
GATE_LOT_SIZE = 1.0
GATE_CONTRACT_SIZE = 0.0001

CSV_DIR = Path("data/tardis_csv")
NPZ_DIR = Path("data/npz")
RESULT_DIR = Path("results")


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


def convert_pair(exchange: str, symbol: str, yyyymmdd: str, buffer_rows: int | None = None) -> Path:
    out = npz_path(exchange, symbol, yyyymmdd)
    if out.exists() and out.stat().st_size > 0:
        print(f"exists {out}")
        return out

    if exchange == GATE_EXCHANGE and symbol == GATE_SYMBOL:
        return convert_gate_book_ticker(symbol, yyyymmdd)

    trade_file = csv_path(exchange, "trades", symbol, yyyymmdd)
    depth_file = csv_path(exchange, "incremental_book_L2", symbol, yyyymmdd)
    if buffer_rows is None:
        rows = count_gzip_rows(trade_file) + count_gzip_rows(depth_file)
        buffer_rows = max(1_000_000, int(rows * 1.35) + 1_000_000)

    print(f"convert {exchange} {symbol}, buffer_rows={buffer_rows:,}")
    tardis.convert(
        [str(trade_file), str(depth_file)],
        output_filename=str(out),
        buffer_size=buffer_rows,
        snapshot_mode="process",
    )
    return out


def convert_gate_book_ticker(symbol: str, yyyymmdd: str) -> Path:
    out = npz_path(GATE_EXCHANGE, symbol, yyyymmdd)
    if out.exists() and out.stat().st_size > 0:
        print(f"exists {out}")
        return out

    trade_file = csv_path(GATE_EXCHANGE, "trades", symbol, yyyymmdd)
    ticker_file = csv_path(GATE_EXCHANGE, "book_ticker", symbol, yyyymmdd)
    trade_rows = count_gzip_rows(trade_file)
    ticker_rows = count_gzip_rows(ticker_file)
    max_rows = trade_rows + ticker_rows * 4 + 16
    print(f"convert {GATE_EXCHANGE} {symbol} book_ticker, max_rows={max_rows:,}")

    tmp = np.empty(max_rows, event_dtype)
    rn = 0
    with gzip.open(trade_file, "rt") as file:
        header = file.readline()
        for line in file:
            fields = line.rstrip("\n").split(",")
            side = fields[5]
            tmp[rn]["ev"] = TRADE_EVENT | (BUY_EVENT if side == "buy" else SELL_EVENT)
            tmp[rn]["exch_ts"] = int(fields[2]) * 1000
            tmp[rn]["local_ts"] = int(fields[3]) * 1000
            tmp[rn]["px"] = float(fields[6])
            tmp[rn]["qty"] = float(fields[7])
            tmp[rn]["order_id"] = 0
            tmp[rn]["ival"] = 0
            tmp[rn]["fval"] = 0.0
            rn += 1

    prev_bid_px = 0.0
    prev_ask_px = 0.0
    prev_bid_qty = -1.0
    prev_ask_qty = -1.0
    with gzip.open(ticker_file, "rt") as file:
        header = file.readline()
        for line in file:
            fields = line.rstrip("\n").split(",")
            exch_ts = int(fields[2]) * 1000
            local_ts = int(fields[3]) * 1000
            ask_qty = float(fields[4])
            ask_px = float(fields[5])
            bid_px = float(fields[6])
            bid_qty = float(fields[7])

            if prev_bid_px > 0.0 and prev_bid_px != bid_px:
                tmp[rn]["ev"] = DEPTH_EVENT | BUY_EVENT
                tmp[rn]["exch_ts"] = exch_ts
                tmp[rn]["local_ts"] = local_ts
                tmp[rn]["px"] = prev_bid_px
                tmp[rn]["qty"] = 0.0
                tmp[rn]["order_id"] = 0
                tmp[rn]["ival"] = 0
                tmp[rn]["fval"] = 0.0
                rn += 1
            if prev_ask_px > 0.0 and prev_ask_px != ask_px:
                tmp[rn]["ev"] = DEPTH_EVENT | SELL_EVENT
                tmp[rn]["exch_ts"] = exch_ts
                tmp[rn]["local_ts"] = local_ts
                tmp[rn]["px"] = prev_ask_px
                tmp[rn]["qty"] = 0.0
                tmp[rn]["order_id"] = 0
                tmp[rn]["ival"] = 0
                tmp[rn]["fval"] = 0.0
                rn += 1
            if bid_px > 0.0 and (prev_bid_px != bid_px or prev_bid_qty != bid_qty):
                tmp[rn]["ev"] = DEPTH_EVENT | BUY_EVENT
                tmp[rn]["exch_ts"] = exch_ts
                tmp[rn]["local_ts"] = local_ts
                tmp[rn]["px"] = bid_px
                tmp[rn]["qty"] = bid_qty
                tmp[rn]["order_id"] = 0
                tmp[rn]["ival"] = 0
                tmp[rn]["fval"] = 0.0
                rn += 1
            if ask_px > 0.0 and (prev_ask_px != ask_px or prev_ask_qty != ask_qty):
                tmp[rn]["ev"] = DEPTH_EVENT | SELL_EVENT
                tmp[rn]["exch_ts"] = exch_ts
                tmp[rn]["local_ts"] = local_ts
                tmp[rn]["px"] = ask_px
                tmp[rn]["qty"] = ask_qty
                tmp[rn]["order_id"] = 0
                tmp[rn]["ival"] = 0
                tmp[rn]["fval"] = 0.0
                rn += 1

            prev_bid_px = bid_px
            prev_ask_px = ask_px
            prev_bid_qty = bid_qty
            prev_ask_qty = ask_qty

    tmp = tmp[:rn]
    print("Correcting the latency")
    tmp = correct_local_timestamp(tmp, 0)
    print("Correcting the event order")
    data = correct_event_order(
        tmp,
        np.argsort(tmp["exch_ts"], kind="mergesort"),
        np.argsort(tmp["local_ts"], kind="mergesort"),
    )
    validate_event_order(data)
    print(f"Saving to {out}")
    np.savez_compressed(out, data=data)
    return out


@njit
def floor_to_lot(qty, lot_size):
    return math.floor(qty / lot_size) * lot_size


@njit
def ceil_to_lot(qty, lot_size):
    return math.ceil(qty / lot_size) * lot_size


@njit
def ceil_to_tick(px, tick_size):
    return math.ceil(px / tick_size) * tick_size


@njit
def floor_to_tick(px, tick_size):
    return math.floor(px / tick_size) * tick_size


@njit
def round_to_tick(px, tick_size):
    if tick_size <= 0:
        return px
    return math.floor(px / tick_size + 0.5) * tick_size


@njit
def current_bitmex_base(hbt):
    depth = hbt.depth(0)
    mid = (depth.best_bid + depth.best_ask) / 2.0
    if mid <= 0:
        return 0.0
    return hbt.position(0) * BITMEX_CONTRACT_SIZE / mid


@njit
def current_gate_base(hbt):
    return hbt.position(1) * GATE_CONTRACT_SIZE


@njit
def bitmex_base_from_contracts(contracts, price):
    if price <= 0:
        return 0.0
    return contracts * BITMEX_CONTRACT_SIZE / price


@njit
def combined_equity_usdt(hbt):
    bitmex_depth = hbt.depth(0)
    gate_depth = hbt.depth(1)
    bitmex_mid = (bitmex_depth.best_bid + bitmex_depth.best_ask) / 2.0
    gate_mid = (gate_depth.best_bid + gate_depth.best_ask) / 2.0
    bitmex_state = hbt.state_values(0)
    gate_state = hbt.state_values(1)
    bitmex_equity_btc = (
        -bitmex_state.balance
        - bitmex_state.position * BITMEX_CONTRACT_SIZE / bitmex_mid
        - bitmex_state.fee
    )
    gate_equity_usdt = (
        gate_state.balance
        + gate_state.position * gate_mid * GATE_CONTRACT_SIZE
        - gate_state.fee
    )
    return bitmex_equity_btc * bitmex_mid + gate_equity_usdt


@njit
def dynamic_spreads(hbt):
    return OPEN_LONG_SPREAD_RATIO, CLOSE_SPREAD_RATIO


@njit
def ratio_minus_one_bps(numerator, denominator):
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return math.nan
    return (numerator / denominator - 1.0) * 10_000.0


@njit
def record_gate_signal(
    hbt,
    gate_signal_ts,
    gate_signal_bid,
    gate_signal_ask,
    gate_signal_bid_qty,
    gate_signal_ask_qty,
    gate_signal_write_idx,
    gate_signal_count,
):
    gate_depth = hbt.depth(1)
    if gate_depth.best_bid <= 0 or gate_depth.best_ask <= 0:
        return gate_signal_write_idx, gate_signal_count
    gate_signal_ts[gate_signal_write_idx] = hbt.current_timestamp
    gate_signal_bid[gate_signal_write_idx] = gate_depth.best_bid
    gate_signal_ask[gate_signal_write_idx] = gate_depth.best_ask
    gate_signal_bid_qty[gate_signal_write_idx] = gate_depth.best_bid_qty
    gate_signal_ask_qty[gate_signal_write_idx] = gate_depth.best_ask_qty
    gate_signal_write_idx = (gate_signal_write_idx + 1) % GATE_SIGNAL_HISTORY_LEN
    gate_signal_count = min(gate_signal_count + 1, GATE_SIGNAL_HISTORY_LEN)
    return gate_signal_write_idx, gate_signal_count


@njit
def gate_signal_at_or_before(
    gate_signal_ts,
    gate_signal_write_idx,
    gate_signal_count,
    target_ts,
):
    for offset in range(gate_signal_count):
        idx = (gate_signal_write_idx - 1 - offset) % GATE_SIGNAL_HISTORY_LEN
        if gate_signal_ts[idx] <= target_ts:
            return idx
    return -1


@njit
def gate_toxic_quote_signal(
    side,
    gate_signal_ts,
    gate_signal_bid,
    gate_signal_ask,
    gate_signal_bid_qty,
    gate_signal_ask_qty,
    gate_signal_write_idx,
    gate_signal_count,
):
    if gate_signal_count <= 1:
        return False, 0.0, 0.0
    current_idx = (gate_signal_write_idx - 1) % GATE_SIGNAL_HISTORY_LEN

    if GATE_MOMENTUM_CANCEL_BPS > 0 and GATE_SHORT_MOMENTUM_WINDOW_NS > 0:
        target_ts = gate_signal_ts[current_idx] - GATE_SHORT_MOMENTUM_WINDOW_NS
        past_idx = gate_signal_at_or_before(
            gate_signal_ts,
            gate_signal_write_idx,
            gate_signal_count,
            target_ts,
        )
        if past_idx >= 0:
            if side > 0:
                move_bps = ratio_minus_one_bps(gate_signal_bid[current_idx], gate_signal_bid[past_idx])
                adverse_bps = -move_bps
            else:
                move_bps = ratio_minus_one_bps(gate_signal_ask[current_idx], gate_signal_ask[past_idx])
                adverse_bps = move_bps
            if adverse_bps >= GATE_MOMENTUM_CANCEL_BPS:
                return True, move_bps, 1.0

    if GATE_MICROPRICE_CANCEL_BPS > 0:
        total_qty = gate_signal_bid_qty[current_idx] + gate_signal_ask_qty[current_idx]
        if total_qty > 0:
            bid = gate_signal_bid[current_idx]
            ask = gate_signal_ask[current_idx]
            mid = (bid + ask) / 2.0
            microprice = (
                ask * gate_signal_bid_qty[current_idx] + bid * gate_signal_ask_qty[current_idx]
            ) / total_qty
            micro_bps = ratio_minus_one_bps(microprice, mid)
            if side > 0:
                adverse_bps = -micro_bps
            else:
                adverse_bps = micro_bps
            if adverse_bps >= GATE_MICROPRICE_CANCEL_BPS:
                return True, micro_bps, 2.0

    return False, 0.0, 0.0


@njit
def cancel_all_side(hbt, asset_no, side):
    orders = hbt.orders(asset_no)
    values = orders.values()
    while values.has_next():
        order = values.get()
        if order.side == side and order.cancellable:
            hbt.cancel(asset_no, order.order_id, False)


@njit
def cancel_all_orders(hbt, asset_no):
    orders = hbt.orders(asset_no)
    values = orders.values()
    while values.has_next():
        order = values.get()
        if order.cancellable:
            hbt.cancel(asset_no, order.order_id, False)


@njit
def manage_bitmex_bid(
    hbt,
    order_id,
    inflight_until,
    quote_gate_anchor,
    toxic_enabled,
    toxic_bid_cooldown_until,
    gate_signal_ts,
    gate_signal_bid,
    gate_signal_ask,
    gate_signal_bid_qty,
    gate_signal_ask_qty,
    gate_signal_write_idx,
    gate_signal_count,
    metrics,
):
    if hbt.current_timestamp < inflight_until:
        return order_id, inflight_until, quote_gate_anchor

    bitmex_depth = hbt.depth(0)
    gate_depth = hbt.depth(1)

    if bitmex_depth.best_bid <= 0 or bitmex_depth.best_ask <= 0:
        cancel_all_side(hbt, 0, 1)
        return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, quote_gate_anchor
    if gate_depth.best_bid <= 0 or gate_depth.best_ask <= 0:
        cancel_all_side(hbt, 0, 1)
        return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, quote_gate_anchor

    base_pos = current_bitmex_base(hbt)
    bitmex_mid = (bitmex_depth.best_bid + bitmex_depth.best_ask) / 2.0
    order_base = bitmex_base_from_contracts(BITMEX_ORDER_QTY, bitmex_mid)
    effective_max_base = max(MAX_POSITION_BASE, order_base)

    bid_spread, ask_spread = dynamic_spreads(hbt)
    raw_price = min(gate_depth.best_bid * (1.0 - bid_spread), bitmex_depth.best_bid)
    bid_price = floor_to_tick(raw_price, BITMEX_TICK_SIZE)
    bid_qty = BITMEX_ORDER_QTY

    existing = hbt.orders(0).get(order_id)
    if toxic_enabled:
        if hbt.current_timestamp < toxic_bid_cooldown_until:
            if existing is not None and existing.cancellable:
                hbt.cancel(0, order_id, False)
                metrics[25] += 1
                return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, quote_gate_anchor
            metrics[29] += 1
            return order_id, inflight_until, quote_gate_anchor
        signal, signal_bps, signal_code = gate_toxic_quote_signal(
            1.0,
            gate_signal_ts,
            gate_signal_bid,
            gate_signal_ask,
            gate_signal_bid_qty,
            gate_signal_ask_qty,
            gate_signal_write_idx,
            gate_signal_count,
        )
        if signal:
            metrics[22] += 1
            if signal_code == 1.0:
                metrics[32] += 1
            else:
                metrics[34] += 1
            if existing is not None and existing.cancellable:
                hbt.cancel(0, order_id, False)
                metrics[23] += 1
                return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, quote_gate_anchor
            return order_id, inflight_until, quote_gate_anchor

    should_quote = (
        (base_pos < 0 or base_pos + order_base <= effective_max_base)
        and bid_qty >= BITMEX_LOT_SIZE
        and bid_price > 0
    )
    if not should_quote:
        if existing is not None and existing.cancellable:
            hbt.cancel(0, order_id, False)
            return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, quote_gate_anchor
        return order_id, inflight_until, quote_gate_anchor

    if existing is not None:
        if existing.cancellable and (existing.price != bid_price or existing.qty != bid_qty):
            hbt.modify(0, order_id, bid_price, bid_qty, False)
            return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, gate_depth.best_bid
        return order_id, inflight_until, quote_gate_anchor

    hbt.submit_buy_order(0, order_id, bid_price, bid_qty, GTX, LIMIT, False)
    return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, gate_depth.best_bid


@njit
def manage_bitmex_ask(
    hbt,
    order_id,
    inflight_until,
    quote_gate_anchor,
    toxic_enabled,
    toxic_ask_cooldown_until,
    gate_signal_ts,
    gate_signal_bid,
    gate_signal_ask,
    gate_signal_bid_qty,
    gate_signal_ask_qty,
    gate_signal_write_idx,
    gate_signal_count,
    metrics,
):
    if hbt.current_timestamp < inflight_until:
        return order_id, inflight_until, quote_gate_anchor

    bitmex_depth = hbt.depth(0)
    gate_depth = hbt.depth(1)

    if bitmex_depth.best_bid <= 0 or bitmex_depth.best_ask <= 0:
        cancel_all_side(hbt, 0, -1)
        return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, quote_gate_anchor
    if gate_depth.best_bid <= 0 or gate_depth.best_ask <= 0:
        cancel_all_side(hbt, 0, -1)
        return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, quote_gate_anchor

    base_pos = current_bitmex_base(hbt)
    bitmex_mid = (bitmex_depth.best_bid + bitmex_depth.best_ask) / 2.0
    order_base = bitmex_base_from_contracts(BITMEX_ORDER_QTY, bitmex_mid)
    effective_max_base = max(MAX_POSITION_BASE, order_base)

    bid_spread, ask_spread = dynamic_spreads(hbt)
    raw_price = max(gate_depth.best_ask * (1.0 - ask_spread), bitmex_depth.best_ask)
    ask_price = ceil_to_tick(raw_price, BITMEX_TICK_SIZE)
    ask_qty = BITMEX_ORDER_QTY

    existing = hbt.orders(0).get(order_id)
    if toxic_enabled:
        if hbt.current_timestamp < toxic_ask_cooldown_until:
            if existing is not None and existing.cancellable:
                hbt.cancel(0, order_id, False)
                metrics[26] += 1
                return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, quote_gate_anchor
            metrics[30] += 1
            return order_id, inflight_until, quote_gate_anchor
        signal, signal_bps, signal_code = gate_toxic_quote_signal(
            -1.0,
            gate_signal_ts,
            gate_signal_bid,
            gate_signal_ask,
            gate_signal_bid_qty,
            gate_signal_ask_qty,
            gate_signal_write_idx,
            gate_signal_count,
        )
        if signal:
            metrics[22] += 1
            if signal_code == 1.0:
                metrics[33] += 1
            else:
                metrics[35] += 1
            if existing is not None and existing.cancellable:
                hbt.cancel(0, order_id, False)
                metrics[24] += 1
                return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, quote_gate_anchor
            return order_id, inflight_until, quote_gate_anchor

    should_quote = (
        (base_pos > 0 or base_pos - order_base >= -effective_max_base)
        and ask_qty >= BITMEX_LOT_SIZE
        and ask_price > 0
    )
    if not should_quote:
        if existing is not None and existing.cancellable:
            hbt.cancel(0, order_id, False)
            return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, quote_gate_anchor
        return order_id, inflight_until, quote_gate_anchor

    if existing is not None:
        if existing.cancellable and (existing.price != ask_price or existing.qty != ask_qty):
            hbt.modify(0, order_id, ask_price, ask_qty, False)
            return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, gate_depth.best_ask
        return order_id, inflight_until, quote_gate_anchor

    hbt.submit_sell_order(0, order_id, ask_price, ask_qty, GTX, LIMIT, False)
    return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS, gate_depth.best_ask


@njit
def submit_gate_hedge_order(
    hbt, order_id, metrics, side, hedge_qty, bitmex_delta_contracts, bitmex_exec_px, hedge_wait_start_ts
):
    gate_depth = hbt.depth(1)
    if gate_depth.best_bid <= 0 or gate_depth.best_ask <= 0:
        return order_id
    if hedge_qty < GATE_LOT_SIZE:
        return order_id

    state_before = hbt.state_values(1)
    pos_before = state_before.position
    value_before = state_before.trading_value
    trades_before = state_before.num_trades
    hedge_ts = hbt.current_timestamp
    order_id += 1
    slippage = GATE_TAKER_SLIPPAGE_BPS / 10_000.0

    if side < 0:
        expected_px = gate_depth.best_bid
        limit_px = round_to_tick(expected_px * (1.0 - slippage), GATE_TICK_SIZE)
        hbt.submit_sell_order(1, order_id, limit_px, hedge_qty, IOC, LIMIT, True)
    else:
        expected_px = gate_depth.best_ask
        limit_px = round_to_tick(expected_px * (1.0 + slippage), GATE_TICK_SIZE)
        hbt.submit_buy_order(1, order_id, limit_px, hedge_qty, IOC, LIMIT, True)

    state_after = hbt.state_values(1)
    if state_after.num_trades > trades_before:
        exec_contracts = abs(state_after.position - pos_before)
        exec_value = state_after.trading_value - value_before
        if exec_contracts > 0:
            exec_px = exec_value / (exec_contracts * GATE_CONTRACT_SIZE)
            if side < 0:
                metrics[8] += expected_px - exec_px
            else:
                metrics[8] += exec_px - expected_px
            hedge_base = exec_contracts * GATE_CONTRACT_SIZE
            if bitmex_exec_px > 0:
                if bitmex_delta_contracts > 0:
                    paired_edge = exec_px - bitmex_exec_px
                    metrics[16] += paired_edge
                    metrics[17] += 1
                else:
                    paired_edge = bitmex_exec_px - exec_px
                    metrics[18] += paired_edge
                    metrics[19] += 1
                metrics[13] += paired_edge * hedge_base
                metrics[14] += paired_edge
                metrics[15] += 1
        metrics[5] += state_after.num_trades - trades_before
        if hedge_wait_start_ts > 0:
            metrics[6] += hbt.current_timestamp - hedge_wait_start_ts
        else:
            metrics[6] += hbt.current_timestamp - hedge_ts
        metrics[7] += 1
    return order_id


@njit
def hedge_gate_net_exposure(hbt, order_id, metrics, bitmex_delta_contracts, bitmex_exec_px, hedge_wait_start_ts):
    gate_depth = hbt.depth(1)
    if gate_depth.best_bid <= 0 or gate_depth.best_ask <= 0:
        return order_id

    hedge_needed_base = current_bitmex_base(hbt) + current_gate_base(hbt)
    target_qty = abs(hedge_needed_base) / GATE_CONTRACT_SIZE
    if target_qty <= 0:
        return order_id

    if hedge_needed_base > 0:
        side = -1.0
    else:
        side = 1.0

    gate_pos = hbt.position(1)
    closing_qty = 0.0
    if side < 0 and gate_pos > 0:
        closing_qty = floor_to_lot(min(gate_pos, target_qty), GATE_LOT_SIZE)
    elif side > 0 and gate_pos < 0:
        closing_qty = floor_to_lot(min(abs(gate_pos), target_qty), GATE_LOT_SIZE)

    opening_qty = 0.0
    remaining_qty = target_qty - closing_qty
    if remaining_qty > 0:
        opening_qty = ceil_to_lot(remaining_qty, GATE_LOT_SIZE)

    if closing_qty >= GATE_LOT_SIZE:
        order_id = submit_gate_hedge_order(
            hbt, order_id, metrics, side, closing_qty, bitmex_delta_contracts, bitmex_exec_px, hedge_wait_start_ts
        )
    if opening_qty >= GATE_LOT_SIZE:
        order_id = submit_gate_hedge_order(
            hbt, order_id, metrics, side, opening_qty, bitmex_delta_contracts, bitmex_exec_px, hedge_wait_start_ts
        )
    return order_id


@njit
def update_risk_metrics(hbt, metrics):
    bitmex_base = current_bitmex_base(hbt)
    gate_base = current_gate_base(hbt)
    metrics[1] = max(metrics[1], abs(bitmex_base))
    metrics[2] = max(metrics[2], abs(gate_base))
    metrics[3] = max(metrics[3], abs(bitmex_base + gate_base))
    metrics[10] = max(metrics[10], abs(bitmex_base) + abs(gate_base))


@njit
def force_flatten(hbt, metrics):
    cancel_all_orders(hbt, 0)
    cancel_all_orders(hbt, 1)
    hbt.elapse(1_000_000_000)

    before = combined_equity_usdt(hbt)
    bitmex_depth = hbt.depth(0)
    gate_depth = hbt.depth(1)

    bitmex_pos = hbt.position(0)
    if bitmex_pos > 0:
        hbt.submit_sell_order(0, 90_001, bitmex_depth.best_bid, abs(bitmex_pos), IOC, MARKET, True)
    elif bitmex_pos < 0:
        hbt.submit_buy_order(0, 90_002, bitmex_depth.best_ask, abs(bitmex_pos), IOC, MARKET, True)

    gate_pos = hbt.position(1)
    if gate_pos > 0:
        hbt.submit_sell_order(1, 90_003, gate_depth.best_bid, abs(gate_pos), IOC, MARKET, True)
    elif gate_pos < 0:
        hbt.submit_buy_order(1, 90_004, gate_depth.best_ask, abs(gate_pos), IOC, MARKET, True)

    after = combined_equity_usdt(hbt)
    metrics[9] = after - before
    update_risk_metrics(hbt, metrics)


@njit
def run_strategy(hbt, recorder, metrics, toxic_enabled, end_close_ts_ns):
    bitmex_bid_order_id = 10_000
    bitmex_ask_order_id = 20_000
    gate_hedge_order_id = 30_000
    bitmex_bid_inflight_until = 0
    bitmex_ask_inflight_until = 0
    bitmex_bid_gate_anchor = 0.0
    bitmex_ask_gate_anchor = 0.0
    gate_hedge_inflight_until = 0
    pending_gate_hedge_due_ts = 0
    pending_gate_hedge_start_ts = 0
    pending_bitmex_delta_contracts = 0.0
    pending_bitmex_delta_value = 0.0
    last_record_ts = 0
    last_bitmex_pos = hbt.position(0)
    last_bitmex_trades = hbt.state_values(0).num_trades
    last_bitmex_trading_value = hbt.state_values(0).trading_value
    toxic_bid_cooldown_until = 0
    toxic_ask_cooldown_until = 0
    gate_signal_ts = np.zeros(GATE_SIGNAL_HISTORY_LEN, dtype=np.int64)
    gate_signal_bid = np.zeros(GATE_SIGNAL_HISTORY_LEN, dtype=np.float64)
    gate_signal_ask = np.zeros(GATE_SIGNAL_HISTORY_LEN, dtype=np.float64)
    gate_signal_bid_qty = np.zeros(GATE_SIGNAL_HISTORY_LEN, dtype=np.float64)
    gate_signal_ask_qty = np.zeros(GATE_SIGNAL_HISTORY_LEN, dtype=np.float64)
    gate_signal_write_idx = 0
    gate_signal_count = 0

    while hbt.elapse(ORDER_UPDATE_INTERVAL_NS) == 0:
        if hbt.current_timestamp >= end_close_ts_ns:
            break

        hbt.clear_inactive_orders(ALL_ASSETS)
        gate_signal_write_idx, gate_signal_count = record_gate_signal(
            hbt,
            gate_signal_ts,
            gate_signal_bid,
            gate_signal_ask,
            gate_signal_bid_qty,
            gate_signal_ask_qty,
            gate_signal_write_idx,
            gate_signal_count,
        )
        bitmex_bid_order_id, bitmex_bid_inflight_until, bitmex_bid_gate_anchor = manage_bitmex_bid(
            hbt,
            bitmex_bid_order_id,
            bitmex_bid_inflight_until,
            bitmex_bid_gate_anchor,
            toxic_enabled,
            toxic_bid_cooldown_until,
            gate_signal_ts,
            gate_signal_bid,
            gate_signal_ask,
            gate_signal_bid_qty,
            gate_signal_ask_qty,
            gate_signal_write_idx,
            gate_signal_count,
            metrics,
        )
        bitmex_ask_order_id, bitmex_ask_inflight_until, bitmex_ask_gate_anchor = manage_bitmex_ask(
            hbt,
            bitmex_ask_order_id,
            bitmex_ask_inflight_until,
            bitmex_ask_gate_anchor,
            toxic_enabled,
            toxic_ask_cooldown_until,
            gate_signal_ts,
            gate_signal_bid,
            gate_signal_ask,
            gate_signal_bid_qty,
            gate_signal_ask_qty,
            gate_signal_write_idx,
            gate_signal_count,
            metrics,
        )

        bitmex_state = hbt.state_values(0)
        if bitmex_state.num_trades > last_bitmex_trades:
            bitmex_depth = hbt.depth(0)
            bitmex_mid = (bitmex_depth.best_bid + bitmex_depth.best_ask) / 2.0
            delta_contracts = bitmex_state.position - last_bitmex_pos
            delta_trading_value = bitmex_state.trading_value - last_bitmex_trading_value
            bitmex_exec_px = 0.0
            if abs(delta_contracts) > 0 and delta_trading_value > 0:
                bitmex_exec_px = abs(delta_contracts) * BITMEX_CONTRACT_SIZE / delta_trading_value
            metrics[0] += abs(bitmex_base_from_contracts(delta_contracts, bitmex_mid))
            metrics[4] += bitmex_state.num_trades - last_bitmex_trades
            if delta_contracts > 0:
                metrics[11] += bitmex_state.num_trades - last_bitmex_trades
            elif delta_contracts < 0:
                metrics[12] += bitmex_state.num_trades - last_bitmex_trades
            if toxic_enabled:
                gate_depth = hbt.depth(1)
                adverse_anchor_move_bps = 0.0
                if delta_contracts > 0 and bitmex_bid_gate_anchor > 0 and gate_depth.best_bid > 0:
                    adverse_anchor_move_bps = -ratio_minus_one_bps(gate_depth.best_bid, bitmex_bid_gate_anchor)
                    if adverse_anchor_move_bps >= TOXIC_FILL_GATE_ANCHOR_MOVE_BPS:
                        toxic_bid_cooldown_until = max(
                            toxic_bid_cooldown_until,
                            hbt.current_timestamp + TOXIC_FILL_COOLDOWN_NS,
                        )
                        metrics[27] += 1
                        metrics[28] += adverse_anchor_move_bps
                elif delta_contracts < 0 and bitmex_ask_gate_anchor > 0 and gate_depth.best_ask > 0:
                    adverse_anchor_move_bps = ratio_minus_one_bps(gate_depth.best_ask, bitmex_ask_gate_anchor)
                    if adverse_anchor_move_bps >= TOXIC_FILL_GATE_ANCHOR_MOVE_BPS:
                        toxic_ask_cooldown_until = max(
                            toxic_ask_cooldown_until,
                            hbt.current_timestamp + TOXIC_FILL_COOLDOWN_NS,
                        )
                        metrics[27] += 1
                        metrics[28] += adverse_anchor_move_bps
            last_bitmex_pos = bitmex_state.position
            last_bitmex_trades = bitmex_state.num_trades
            last_bitmex_trading_value = bitmex_state.trading_value
            pending_bitmex_delta_contracts += delta_contracts
            pending_bitmex_delta_value += delta_trading_value
            if pending_gate_hedge_due_ts == 0:
                pending_gate_hedge_due_ts = hbt.current_timestamp + GATE_HEDGE_SEND_DELAY_NS
                pending_gate_hedge_start_ts = hbt.current_timestamp
            metrics[20] += 1

        if (
            pending_gate_hedge_due_ts > 0
            and hbt.current_timestamp >= pending_gate_hedge_due_ts
            and hbt.current_timestamp >= gate_hedge_inflight_until
        ):
            gate_depth = hbt.depth(1)
            if gate_depth.best_bid > 0 and gate_depth.best_ask > 0:
                pending_bitmex_exec_px = 0.0
                if abs(pending_bitmex_delta_contracts) > 0 and pending_bitmex_delta_value > 0:
                    pending_bitmex_exec_px = (
                        abs(pending_bitmex_delta_contracts) * BITMEX_CONTRACT_SIZE / pending_bitmex_delta_value
                    )
                previous_gate_hedge_order_id = gate_hedge_order_id
                gate_hedge_order_id = hedge_gate_net_exposure(
                    hbt,
                    gate_hedge_order_id,
                    metrics,
                    pending_bitmex_delta_contracts,
                    pending_bitmex_exec_px,
                    pending_gate_hedge_start_ts,
                )
                if gate_hedge_order_id != previous_gate_hedge_order_id:
                    gate_hedge_inflight_until = hbt.current_timestamp + GATE_HEDGE_INFLIGHT_NS
                    metrics[21] += 1
                pending_gate_hedge_due_ts = 0
                pending_gate_hedge_start_ts = 0
                pending_bitmex_delta_contracts = 0.0
                pending_bitmex_delta_value = 0.0

        update_risk_metrics(hbt, metrics)

        if hbt.current_timestamp - last_record_ts >= 1_000_000_000:
            recorder.record(hbt)
            last_record_ts = hbt.current_timestamp

    force_flatten(hbt, metrics)

    recorder.record(hbt)
    return True


def run_backtest(bitmex_npz: Path, gate_npz: Path, yyyymmdd: str, mode: str) -> Path:
    bitmex_asset = (
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
    gate_asset = (
        BacktestAsset()
        .data([str(gate_npz)])
        .linear_asset(GATE_CONTRACT_SIZE)
        .constant_order_latency(GATE_ORDER_ENTRY_LATENCY_NS, GATE_ORDER_RESPONSE_LATENCY_NS)
        .power_prob_queue_model3(3.0)
        .no_partial_fill_exchange()
        .trading_value_fee_model(0.0, 0.0)
        .tick_size(GATE_TICK_SIZE)
        .lot_size(GATE_LOT_SIZE)
        .last_trades_capacity(10_000)
    )

    hbt = HashMapMarketDepthBacktest([bitmex_asset, gate_asset])
    recorder = Recorder(2, 100_000)
    metrics = np.zeros(40, dtype=np.float64)
    toxic_enabled = mode == "toxic"
    metrics[31] = 1.0 if toxic_enabled else 0.0
    end_ts_ns = end_close_ts_ns(yyyymmdd)
    ok = run_strategy(hbt, recorder.recorder, metrics, toxic_enabled, end_ts_ns)
    if not ok:
        raise RuntimeError("strategy returned false")

    out = RESULT_DIR / f"bitmex_xbtusd_gate_btc_usdt_mm_arb_{mode}_{yyyymmdd}.npz"
    np.savez_compressed(out, **{"0": recorder.get(0), "1": recorder.get(1), "metrics": metrics})
    write_summary(out, metrics, yyyymmdd, mode)
    return out


def write_summary(
    result_npz: Path,
    metrics: np.ndarray | None = None,
    yyyymmdd: str = DATE,
    mode: str = "baseline",
) -> None:
    data = np.load(result_npz)
    bitmex = data["0"]
    gate = data["1"]
    if metrics is None:
        metrics = data["metrics"] if "metrics" in data.files else np.zeros(32, dtype=np.float64)

    bitmex_final = bitmex[-1]
    gate_final = gate[-1]
    bitmex_price = float(bitmex_final["price"])
    gate_price = float(gate_final["price"])

    bitmex_base = float(bitmex_final["position"]) * BITMEX_CONTRACT_SIZE / bitmex_price
    gate_base = float(gate_final["position"]) * GATE_CONTRACT_SIZE
    bitmex_equity_btc = (
        -float(bitmex_final["balance"])
        - float(bitmex_final["position"]) * BITMEX_CONTRACT_SIZE / bitmex_price
        - float(bitmex_final["fee"])
    )
    gate_equity_usdt = (
        float(gate_final["balance"])
        + float(gate_final["position"]) * gate_price * GATE_CONTRACT_SIZE
        - float(gate_final["fee"])
    )
    combined_equity_usdt = bitmex_equity_btc * bitmex_price + gate_equity_usdt
    total_fee_usdt = float(bitmex_final["fee"]) * bitmex_price + float(gate_final["fee"])
    avg_hedge_delay_ns = metrics[6] / metrics[7] if metrics[7] > 0 else 0.0
    avg_hedge_slippage = metrics[8] / metrics[7] if metrics[7] > 0 else 0.0
    avg_paired_edge = metrics[14] / metrics[15] if metrics[15] > 0 else 0.0
    avg_buy_paired_edge = metrics[16] / metrics[17] if metrics[17] > 0 else 0.0
    avg_sell_paired_edge = metrics[18] / metrics[19] if metrics[19] > 0 else 0.0
    avg_toxic_anchor_move_bps = metrics[28] / metrics[27] if metrics[27] > 0 else 0.0
    total_pnl_usdt = combined_equity_usdt
    pnl_status = "profit" if total_pnl_usdt > 0 else "loss" if total_pnl_usdt < 0 else "flat"
    pnl_status_zh = "赚钱" if total_pnl_usdt > 0 else "亏钱" if total_pnl_usdt < 0 else "不赚不亏"
    final_flat = abs(bitmex_base) < 1e-12 and abs(gate_base) < 1e-12

    summary = {
        "date": yyyymmdd,
        "mode": mode,
        "toxic_filter_enabled": bool(metrics[31] > 0),
        "bitmex_symbol": BITMEX_SYMBOL,
        "gate_symbol": GATE_SYMBOL,
        "open_long_spread_ratio": OPEN_LONG_SPREAD_RATIO,
        "close_spread_ratio": CLOSE_SPREAD_RATIO,
        "bitmex_bid_formula": "gate_bid * (1 - OPEN_LONG_SPREAD_RATIO)",
        "bitmex_ask_formula": "gate_ask * (1 - CLOSE_SPREAD_RATIO)",
        "bitmex_quote_bbo_clamp": True,
        "max_position_base": MAX_POSITION_BASE,
        "order_update_interval_ms": ORDER_UPDATE_INTERVAL_NS / 1_000_000.0,
        "gate_hedge_order_type": "IOC LIMIT",
        "gate_taker_slippage_bps": GATE_TAKER_SLIPPAGE_BPS,
        "gate_hedge_open_qty_rounding": "ceil_to_lot",
        "gate_hedge_close_qty_rounding": "floor_to_lot",
        "bitmex_command_inflight_ms": BITMEX_COMMAND_INFLIGHT_NS / 1_000_000.0,
        "bitmex_fill_report_delay_ms": BITMEX_FILL_REPORT_DELAY_NS / 1_000_000.0,
        "local_gate_send_delay_ms": LOCAL_GATE_SEND_DELAY_NS / 1_000_000.0,
        "gate_hedge_send_delay_ms": GATE_HEDGE_SEND_DELAY_NS / 1_000_000.0,
        "gate_hedge_inflight_ms": GATE_HEDGE_INFLIGHT_NS / 1_000_000.0,
        "gate_short_momentum_window_ms": GATE_SHORT_MOMENTUM_WINDOW_NS / 1_000_000.0,
        "gate_momentum_cancel_bps": GATE_MOMENTUM_CANCEL_BPS,
        "gate_microprice_cancel_bps": GATE_MICROPRICE_CANCEL_BPS,
        "toxic_fill_gate_anchor_move_bps": TOXIC_FILL_GATE_ANCHOR_MOVE_BPS,
        "toxic_fill_cooldown_ms": TOXIC_FILL_COOLDOWN_NS / 1_000_000.0,
        "bitmex_tick_size": BITMEX_TICK_SIZE,
        "gate_tick_size": GATE_TICK_SIZE,
        "bitmex_order_entry_latency_ms": BITMEX_ORDER_ENTRY_LATENCY_NS / 1_000_000.0,
        "bitmex_order_response_latency_ms": BITMEX_ORDER_RESPONSE_LATENCY_NS / 1_000_000.0,
        "gate_order_entry_latency_ms": GATE_ORDER_ENTRY_LATENCY_NS / 1_000_000.0,
        "gate_order_response_latency_ms": GATE_ORDER_RESPONSE_LATENCY_NS / 1_000_000.0,
        "bitmex_queue_model": "risk_adverse_queue_model",
        "gate_queue_model": "power_prob_queue_model3(3.0)",
        "bitmex_fill_hedge_enqueue_events": int(metrics[20]),
        "gate_hedge_send_events": int(metrics[21]),
        "toxic_quote_signal_events": int(metrics[22]),
        "toxic_bid_signal_cancel_events": int(metrics[23]),
        "toxic_ask_signal_cancel_events": int(metrics[24]),
        "toxic_bid_cooldown_cancel_events": int(metrics[25]),
        "toxic_ask_cooldown_cancel_events": int(metrics[26]),
        "toxic_fill_events": int(metrics[27]),
        "avg_toxic_anchor_move_bps": float(avg_toxic_anchor_move_bps),
        "toxic_bid_cooldown_suppress_events": int(metrics[29]),
        "toxic_ask_cooldown_suppress_events": int(metrics[30]),
        "toxic_bid_momentum_signal_events": int(metrics[32]),
        "toxic_ask_momentum_signal_events": int(metrics[33]),
        "toxic_bid_microprice_signal_events": int(metrics[34]),
        "toxic_ask_microprice_signal_events": int(metrics[35]),
        "pnl_status": pnl_status,
        "pnl_status_zh": pnl_status_zh,
        "total_pnl_usdt": total_pnl_usdt,
        "total_filled_base": float(metrics[0]),
        "realized_pnl_usdt": total_pnl_usdt,
        "bitmex_max_position_base": float(metrics[1]),
        "gate_max_position_base": float(metrics[2]),
        "max_gross_position_base": float(metrics[10]),
        "max_net_exposure_base": float(metrics[3]),
        "bitmex_maker_fills": int(metrics[4]),
        "bitmex_buy_fills": int(metrics[11]),
        "bitmex_sell_fills": int(metrics[12]),
        "gate_hedge_fills": int(metrics[5]),
        "paired_edge_pnl_usdt": float(metrics[13]),
        "avg_paired_edge_usdt_per_btc": float(avg_paired_edge),
        "avg_bitmex_buy_then_gate_sell_edge_usdt_per_btc": float(avg_buy_paired_edge),
        "avg_bitmex_sell_then_gate_buy_edge_usdt_per_btc": float(avg_sell_paired_edge),
        "paired_edge_events": int(metrics[15]),
        "avg_hedge_delay_ns": float(avg_hedge_delay_ns),
        "avg_hedge_delay_ms": float(avg_hedge_delay_ns / 1_000_000.0),
        "avg_hedge_slippage": float(avg_hedge_slippage),
        "force_close_pnl_usdt": float(metrics[9]),
        "total_fee_usdt": total_fee_usdt,
        "bitmex_fee_btc": float(bitmex_final["fee"]),
        "gate_fee_usdt": float(gate_final["fee"]),
        "bitmex_final_position_contracts": float(bitmex_final["position"]),
        "gate_final_position_contracts": float(gate_final["position"]),
        "bitmex_final_position_base": bitmex_base,
        "gate_final_position_base": gate_base,
        "net_base_position": bitmex_base + gate_base,
        "final_flat": final_flat,
        "bitmex_num_trades": int(bitmex_final["num_trades"]),
        "gate_num_trades": int(gate_final["num_trades"]),
        "bitmex_trading_value_btc": float(bitmex_final["trading_value"]),
        "gate_trading_value_usdt": float(gate_final["trading_value"]),
        "bitmex_equity_btc": bitmex_equity_btc,
        "gate_equity_usdt": gate_equity_usdt,
        "combined_equity_usdt_marked_at_bitmex_mid": combined_equity_usdt,
        "start_timestamp_ns": int(bitmex[0]["timestamp"]),
        "end_timestamp_ns": int(bitmex_final["timestamp"]),
        "records": int(len(bitmex)),
    }
    summary_path = result_npz.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    report_path = result_npz.with_suffix(".report.md")
    report_path.write_text(render_report(summary))
    print(render_console_summary(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


def signed_money(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.4f} USDT"


def signed_base(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.8f} BTC"


def render_console_summary(summary: dict) -> str:
    lines = [
        "",
        "========== 回测结果 ==========",
        f"结论: {summary['pnl_status_zh']} ({signed_money(summary['total_pnl_usdt'])})",
        f"日期: {summary['date']}",
        f"模式: {summary['mode']} (toxic_filter={summary['toxic_filter_enabled']})",
        f"交易对: BitMEX {summary['bitmex_symbol']} maker vs Gate {summary['gate_symbol']} taker",
        f"最终仓位归零: {'是' if summary['final_flat'] else '否'}",
        f"总成交 base: {summary['total_filled_base']:,.8f} BTC",
        f"BitMEX maker 成交次数: {summary['bitmex_maker_fills']}",
        f"  买成交: {summary['bitmex_buy_fills']}, 卖成交: {summary['bitmex_sell_fills']}",
        f"Gate hedge 成交次数: {summary['gate_hedge_fills']}",
        f"Gate hedge 发送事件: {summary['gate_hedge_send_events']}",
        f"Toxic fill: {summary['toxic_fill_events']}, avg anchor move: {summary['avg_toxic_anchor_move_bps']:,.4f} bps",
        f"Toxic signal cancel: bid={summary['toxic_bid_signal_cancel_events']}, ask={summary['toxic_ask_signal_cancel_events']}",
        f"Toxic cooldown suppress: bid={summary['toxic_bid_cooldown_suppress_events']}, ask={summary['toxic_ask_cooldown_suppress_events']}",
        f"平均 hedge 延迟: {summary['avg_hedge_delay_ms']:,.4f} ms",
        f"实际配对边际: {signed_money(summary['paired_edge_pnl_usdt'])}",
        f"平均配对边际: {summary['avg_paired_edge_usdt_per_btc']:,.4f} USDT/BTC",
        f"  BitMEX买->Gate卖: {summary['avg_bitmex_buy_then_gate_sell_edge_usdt_per_btc']:,.4f} USDT/BTC",
        f"  BitMEX卖->Gate买: {summary['avg_bitmex_sell_then_gate_buy_edge_usdt_per_btc']:,.4f} USDT/BTC",
        f"最大净敞口: {summary['max_net_exposure_base']:,.8f} BTC",
        f"期间最大持仓量(gross): {summary['max_gross_position_base']:,.8f} BTC",
        f"日终强平 PnL: {signed_money(summary['force_close_pnl_usdt'])}",
        f"总手续费: {signed_money(summary['total_fee_usdt'])}",
        "==============================",
        "",
    ]
    return "\n".join(lines)


def render_report(summary: dict) -> str:
    verdict = (
        f"本次回测结果为 **{summary['pnl_status_zh']}**，"
        f"总 PnL 为 **{signed_money(summary['total_pnl_usdt'])}**。"
    )
    return f"""# BitMEX XBTUSD vs Gate BTCUSDT 回测报告

{verdict}

## 参数

- 日期: `{summary['date']}`
- 模式: `{summary['mode']}`
- Toxic filter enabled: `{summary['toxic_filter_enabled']}`
- BitMEX maker: `{summary['bitmex_symbol']}`
- Gate hedge: `{summary['gate_symbol']}`
- `OPEN_LONG_SPREAD_RATIO`: `{summary['open_long_spread_ratio']}`
- `CLOSE_SPREAD_RATIO`: `{summary['close_spread_ratio']}`
- BitMEX bid 公式: `{summary['bitmex_bid_formula']}`
- BitMEX ask 公式: `{summary['bitmex_ask_formula']}`
- BitMEX BBO clamp: `{summary['bitmex_quote_bbo_clamp']}`
- `MAX_POSITION_BASE`: `{summary['max_position_base']}`
- 改单间隔: `{summary['order_update_interval_ms']} ms`
- Gate hedge 订单类型: `{summary['gate_hedge_order_type']}`
- Gate hedge 滑点保护: `{summary['gate_taker_slippage_bps']} bps`
- Gate hedge 开仓数量取整: `{summary['gate_hedge_open_qty_rounding']}`
- Gate hedge 平仓数量取整: `{summary['gate_hedge_close_qty_rounding']}`
- BitMEX command inflight: `{summary['bitmex_command_inflight_ms']} ms`
- Gate hedge send delay: `{summary['gate_hedge_send_delay_ms']} ms`
- BitMEX fill report delay: `{summary['bitmex_fill_report_delay_ms']} ms`
- Local Gate send delay: `{summary['local_gate_send_delay_ms']} ms`
- Gate hedge inflight: `{summary['gate_hedge_inflight_ms']} ms`
- Gate short momentum window: `{summary['gate_short_momentum_window_ms']} ms`
- Gate momentum cancel: `{summary['gate_momentum_cancel_bps']} bps`
- Gate microprice cancel: `{summary['gate_microprice_cancel_bps']} bps`
- Toxic fill anchor move: `{summary['toxic_fill_gate_anchor_move_bps']} bps`
- Toxic fill cooldown: `{summary['toxic_fill_cooldown_ms']} ms`
- BitMEX tick size: `{summary['bitmex_tick_size']}`
- Gate tick size: `{summary['gate_tick_size']}`
- BitMEX order latency: `{summary['bitmex_order_entry_latency_ms']} ms entry`, `{summary['bitmex_order_response_latency_ms']} ms response`
- Gate order latency: `{summary['gate_order_entry_latency_ms']} ms entry`, `{summary['gate_order_response_latency_ms']} ms response`
- BitMEX queue model: `{summary['bitmex_queue_model']}`
- Gate queue model: `{summary['gate_queue_model']}`

## 盈亏

- 结论: **{summary['pnl_status_zh']}**
- 总 PnL: **{signed_money(summary['total_pnl_usdt'])}**
- BitMEX equity: `{summary['bitmex_equity_btc']:,.8f} BTC`
- Gate equity: `{signed_money(summary['gate_equity_usdt'])}`
- 日终强制平仓 PnL: `{signed_money(summary['force_close_pnl_usdt'])}`
- 总手续费: `{signed_money(summary['total_fee_usdt'])}`
- BitMEX 手续费: `{summary['bitmex_fee_btc']:,.8f} BTC`
- Gate 手续费: `{signed_money(summary['gate_fee_usdt'])}`

## 成交和风险

- 总成交 base: `{summary['total_filled_base']:,.8f} BTC`
- BitMEX maker 成交次数: `{summary['bitmex_maker_fills']}`
- BitMEX 买成交次数: `{summary['bitmex_buy_fills']}`
- BitMEX 卖成交次数: `{summary['bitmex_sell_fills']}`
- Gate hedge 成交次数: `{summary['gate_hedge_fills']}`
- BitMEX fill hedge enqueue events: `{summary['bitmex_fill_hedge_enqueue_events']}`
- Gate hedge send events: `{summary['gate_hedge_send_events']}`
- Toxic quote signal events: `{summary['toxic_quote_signal_events']}`
- Toxic bid signal cancel events: `{summary['toxic_bid_signal_cancel_events']}`
- Toxic ask signal cancel events: `{summary['toxic_ask_signal_cancel_events']}`
- Toxic bid cooldown cancel events: `{summary['toxic_bid_cooldown_cancel_events']}`
- Toxic ask cooldown cancel events: `{summary['toxic_ask_cooldown_cancel_events']}`
- Toxic fill events: `{summary['toxic_fill_events']}`
- Avg toxic anchor move: `{summary['avg_toxic_anchor_move_bps']:,.4f} bps`
- Toxic bid cooldown suppress events: `{summary['toxic_bid_cooldown_suppress_events']}`
- Toxic ask cooldown suppress events: `{summary['toxic_ask_cooldown_suppress_events']}`
- 实际配对边际 PnL: `{signed_money(summary['paired_edge_pnl_usdt'])}`
- 平均实际配对边际: `{summary['avg_paired_edge_usdt_per_btc']:,.4f} USDT/BTC`
- BitMEX 买 -> Gate 卖平均边际: `{summary['avg_bitmex_buy_then_gate_sell_edge_usdt_per_btc']:,.4f} USDT/BTC`
- BitMEX 卖 -> Gate 买平均边际: `{summary['avg_bitmex_sell_then_gate_buy_edge_usdt_per_btc']:,.4f} USDT/BTC`
- BitMEX 最大仓位: `{summary['bitmex_max_position_base']:,.8f} BTC`
- Gate 最大仓位: `{summary['gate_max_position_base']:,.8f} BTC`
- 期间最大持仓量 gross: `{summary['max_gross_position_base']:,.8f} BTC`
- 最大净敞口: `{summary['max_net_exposure_base']:,.8f} BTC`
- 平均 hedge 延迟: `{summary['avg_hedge_delay_ms']:,.4f} ms`
- 平均 hedge 滑点: `{summary['avg_hedge_slippage']:,.6f}`

## 日终仓位

- 最终 BitMEX 仓位: `{summary['bitmex_final_position_contracts']:,.4f} contracts`, `{signed_base(summary['bitmex_final_position_base'])}`
- 最终 Gate 仓位: `{summary['gate_final_position_contracts']:,.4f} contracts`, `{signed_base(summary['gate_final_position_base'])}`
- 最终净仓位: `{signed_base(summary['net_base_position'])}`
- 仓位是否归零: `{'yes' if summary['final_flat'] else 'no'}`
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest BitMEX maker / Gate taker arbitrage with optional toxic filters."
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        default=list(DEFAULT_DATES),
        help="Dates as YYYYMMDD. Default: 20260512 20260513 20260514.",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "toxic", "both"],
        default="both",
        help="Run baseline, toxic-filter, or both modes.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use existing csv/npz files and do not download missing Tardis CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    modes = ("baseline", "toxic") if args.mode == "both" else (args.mode,)
    key = None if args.skip_download else tardis_key()
    results = []
    for yyyymmdd in args.dates:
        for exchange, symbol in ((BITMEX_EXCHANGE, BITMEX_SYMBOL), (GATE_EXCHANGE, GATE_SYMBOL)):
            data_types = ("trades", "incremental_book_L2")
            if exchange == GATE_EXCHANGE:
                data_types = ("trades", "book_ticker")
            for data_type in data_types:
                path = csv_path(exchange, data_type, symbol, yyyymmdd)
                if args.skip_download:
                    if not path.exists():
                        raise FileNotFoundError(f"missing {path}; rerun without --skip-download")
                    print(f"exists {path}")
                else:
                    download_file(exchange, data_type, symbol, yyyymmdd, key)

        bitmex_npz = convert_pair(BITMEX_EXCHANGE, BITMEX_SYMBOL, yyyymmdd)
        gate_npz = convert_pair(GATE_EXCHANGE, GATE_SYMBOL, yyyymmdd)
        for mode in modes:
            result = run_backtest(bitmex_npz, gate_npz, yyyymmdd, mode)
            results.append(result)
            print(f"result={result}")

    print("all_results=" + ",".join(str(path) for path in results))


if __name__ == "__main__":
    main()
