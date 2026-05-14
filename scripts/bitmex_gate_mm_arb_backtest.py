import gzip
import json
import math
import os
import urllib.request
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
from hftbacktest.data.validation import correct_event_order, correct_local_timestamp, validate_event_order
from hftbacktest.data.utils import tardis


DATE = "20260512"

BITMEX_EXCHANGE = "bitmex"
BITMEX_SYMBOL = "XBTUSD"
GATE_EXCHANGE = "gate-io-futures"
GATE_SYMBOL = "BTC_USDT"

OPEN_LONG_SPREAD_RATIO = 0.00040
CLOSE_SPREAD_RATIO = -0.001
MAX_POSITION_BASE = 0.079

BITMEX_TICK_SIZE = 0.1
BITMEX_LOT_SIZE = 100.0
BITMEX_CONTRACT_SIZE = 1.0

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
def ceil_to_tick(px, tick_size):
    return math.ceil(px / tick_size) * tick_size


@njit
def floor_to_tick(px, tick_size):
    return math.floor(px / tick_size) * tick_size


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
def cancel_all_side(hbt, asset_no, side):
    orders = hbt.orders(asset_no)
    values = orders.values()
    while values.has_next():
        order = values.get()
        if order.side == side and order.cancellable:
            hbt.cancel(asset_no, order.order_id, False)


@njit
def manage_bitmex_bid(hbt, order_id):
    bitmex_depth = hbt.depth(0)
    gate_depth = hbt.depth(1)

    if bitmex_depth.best_bid <= 0 or bitmex_depth.best_ask <= 0:
        cancel_all_side(hbt, 0, 1)
        return order_id
    if gate_depth.best_bid <= 0 or gate_depth.best_ask <= 0:
        cancel_all_side(hbt, 0, 1)
        return order_id

    base_pos = current_bitmex_base(hbt)
    remaining_base = MAX_POSITION_BASE - base_pos
    min_base = BITMEX_LOT_SIZE * BITMEX_CONTRACT_SIZE / bitmex_depth.best_bid

    raw_price = gate_depth.best_bid * (1.0 - OPEN_LONG_SPREAD_RATIO)
    bid_price = floor_to_tick(raw_price, BITMEX_TICK_SIZE)
    bid_qty = floor_to_lot(remaining_base * bid_price / BITMEX_CONTRACT_SIZE, BITMEX_LOT_SIZE)

    existing = hbt.orders(0).get(order_id)
    should_quote = remaining_base >= min_base and bid_qty >= BITMEX_LOT_SIZE and bid_price > 0
    if not should_quote:
        if existing is not None and existing.cancellable:
            hbt.cancel(0, order_id, False)
        return order_id

    if existing is not None:
        if existing.cancellable and (existing.price != bid_price or existing.qty != bid_qty):
            hbt.cancel(0, order_id, False)
            order_id += 1
            hbt.submit_buy_order(0, order_id, bid_price, bid_qty, GTX, LIMIT, False)
        return order_id

    hbt.submit_buy_order(0, order_id, bid_price, bid_qty, GTX, LIMIT, False)
    return order_id


@njit
def manage_gate_hedge_ask(hbt, order_id):
    gate_depth = hbt.depth(1)
    if gate_depth.best_bid <= 0 or gate_depth.best_ask <= 0:
        cancel_all_side(hbt, 1, -1)
        return order_id

    hedge_needed_base = current_bitmex_base(hbt) + current_gate_base(hbt)
    ask_price = ceil_to_tick(gate_depth.best_ask * (1.0 - CLOSE_SPREAD_RATIO), GATE_TICK_SIZE)
    ask_qty = floor_to_lot(hedge_needed_base / GATE_CONTRACT_SIZE, GATE_LOT_SIZE)

    existing = hbt.orders(1).get(order_id)
    should_quote = hedge_needed_base >= GATE_CONTRACT_SIZE and ask_qty >= GATE_LOT_SIZE and ask_price > 0
    if not should_quote:
        if existing is not None and existing.cancellable:
            hbt.cancel(1, order_id, False)
        return order_id

    if existing is not None:
        if existing.cancellable and (existing.price != ask_price or existing.qty != ask_qty):
            hbt.cancel(1, order_id, False)
            order_id += 1
            hbt.submit_sell_order(1, order_id, ask_price, ask_qty, GTX, LIMIT, False)
        return order_id

    hbt.submit_sell_order(1, order_id, ask_price, ask_qty, GTX, LIMIT, False)
    return order_id


@njit
def run_strategy(hbt, recorder):
    bitmex_bid_order_id = 10_000
    gate_ask_order_id = 20_000
    last_record_ts = 0

    while True:
        ret = hbt.wait_next_feed(True, 100_000_000)
        if ret == 1:
            break
        if ret < 0:
            return False

        hbt.clear_inactive_orders(ALL_ASSETS)
        bitmex_bid_order_id = manage_bitmex_bid(hbt, bitmex_bid_order_id)
        gate_ask_order_id = manage_gate_hedge_ask(hbt, gate_ask_order_id)

        if hbt.current_timestamp - last_record_ts >= 1_000_000_000:
            recorder.record(hbt)
            last_record_ts = hbt.current_timestamp

    recorder.record(hbt)
    return True


def run_backtest(bitmex_npz: Path, gate_npz: Path) -> Path:
    bitmex_asset = (
        BacktestAsset()
        .data([str(bitmex_npz)])
        .inverse_asset(BITMEX_CONTRACT_SIZE)
        .constant_order_latency(0, 0)
        .power_prob_queue_model3(3.0)
        .no_partial_fill_exchange()
        .trading_value_fee_model(0.000075, 0.00075)
        .tick_size(BITMEX_TICK_SIZE)
        .lot_size(BITMEX_LOT_SIZE)
        .last_trades_capacity(10_000)
    )
    gate_asset = (
        BacktestAsset()
        .data([str(gate_npz)])
        .linear_asset(GATE_CONTRACT_SIZE)
        .constant_order_latency(0, 0)
        .power_prob_queue_model3(3.0)
        .no_partial_fill_exchange()
        .trading_value_fee_model(0.00015, 0.0005)
        .tick_size(GATE_TICK_SIZE)
        .lot_size(GATE_LOT_SIZE)
        .last_trades_capacity(10_000)
    )

    hbt = HashMapMarketDepthBacktest([bitmex_asset, gate_asset])
    recorder = Recorder(2, 100_000)
    ok = run_strategy(hbt, recorder.recorder)
    if not ok:
        raise RuntimeError("strategy returned false")

    out = RESULT_DIR / f"bitmex_xbtusd_gate_btc_usdt_mm_arb_{DATE}.npz"
    recorder.to_npz(str(out))
    write_summary(out)
    return out


def write_summary(result_npz: Path) -> None:
    data = np.load(result_npz)
    bitmex = data["0"]
    gate = data["1"]

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

    summary = {
        "date": DATE,
        "bitmex_symbol": BITMEX_SYMBOL,
        "gate_symbol": GATE_SYMBOL,
        "open_long_spread_ratio": OPEN_LONG_SPREAD_RATIO,
        "close_spread_ratio": CLOSE_SPREAD_RATIO,
        "max_position_base": MAX_POSITION_BASE,
        "bitmex_final_position_contracts": float(bitmex_final["position"]),
        "gate_final_position_contracts": float(gate_final["position"]),
        "bitmex_final_position_base": bitmex_base,
        "gate_final_position_base": gate_base,
        "net_base_position": bitmex_base + gate_base,
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
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    key = tardis_key()
    for exchange, symbol in ((BITMEX_EXCHANGE, BITMEX_SYMBOL), (GATE_EXCHANGE, GATE_SYMBOL)):
        data_types = ("trades", "incremental_book_L2")
        if exchange == GATE_EXCHANGE:
            data_types = ("trades", "book_ticker")
        for data_type in data_types:
            download_file(exchange, data_type, symbol, DATE, key)

    bitmex_npz = convert_pair(BITMEX_EXCHANGE, BITMEX_SYMBOL, DATE)
    gate_npz = convert_pair(GATE_EXCHANGE, GATE_SYMBOL, DATE)
    result = run_backtest(bitmex_npz, gate_npz)
    print(f"result={result}")


if __name__ == "__main__":
    main()
