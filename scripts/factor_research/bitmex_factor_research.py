from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numba import njit

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hftbacktest import BUY_EVENT, SELL_EVENT, BacktestAsset, HashMapMarketDepthBacktest

from bitmex_single_market_mm_backtest import (
    BITMEX_CONTRACT_SIZE,
    BITMEX_EXCHANGE,
    BITMEX_LOT_SIZE,
    BITMEX_ORDER_ENTRY_LATENCY_NS,
    BITMEX_ORDER_RESPONSE_LATENCY_NS,
    BITMEX_TICK_SIZE,
    CSV_DIR,
    NPZ_DIR,
    ORDER_TTL_NS,
    convert_bitmex,
    download_file,
    end_close_ts_ns,
    tardis_key,
)

from factor_research.buckets import bucket_rows, factor_ic_rows, maker_fill_edge_bucket_rows, write_csv
from factor_research.factors import FACTOR_NAMES, build_factor_frame
from factor_research.labels import build_labels
from factor_research.maker_fill_labels import build_maker_fill_labels
from factor_research.plots import write_charts, write_fill_charts, write_lifecycle_fill_charts
from factor_research.report import render_report


RESULT_DIR = REPO_ROOT / "results" / "factor_research"
DEFAULT_DATES = ("20260516", "20260517", "20260518")
DEFAULT_HORIZONS_MS = (100, 250, 500, 1000)
BASE_COLUMNS = 9


@njit
def collect_samples(hbt, sample_interval_ns, end_ts_ns, rows):
    count = 0
    while hbt.elapse(sample_interval_ns) == 0:
        if hbt.current_timestamp >= end_ts_ns:
            break

        depth = hbt.depth(0)
        buy_qty = 0.0
        sell_qty = 0.0
        buy_count = 0.0
        sell_count = 0.0
        trades = hbt.last_trades(0)
        for trade in trades:
            if (trade.ev & BUY_EVENT) == BUY_EVENT:
                buy_qty += trade.qty
                buy_count += 1.0
            elif (trade.ev & SELL_EVENT) == SELL_EVENT:
                sell_qty += trade.qty
                sell_count += 1.0
        hbt.clear_last_trades(0)

        if depth.best_bid <= 0 or depth.best_ask <= 0:
            continue
        if count >= rows.shape[0]:
            break

        rows[count, 0] = hbt.current_timestamp
        rows[count, 1] = depth.best_bid
        rows[count, 2] = depth.best_ask
        rows[count, 3] = depth.best_bid_qty
        rows[count, 4] = depth.best_ask_qty
        rows[count, 5] = buy_qty
        rows[count, 6] = sell_qty
        rows[count, 7] = buy_count
        rows[count, 8] = sell_count
        count += 1
    return count


def build_asset(npz_path: Path, symbol: str) -> BacktestAsset:
    asset = BacktestAsset().data([str(npz_path)])
    # The factor layer does not submit orders, but keeping the same basic asset
    # settings makes the replay environment consistent with strategy scripts.
    if symbol == "XBTUSD":
        asset = asset.inverse_asset(BITMEX_CONTRACT_SIZE)
    else:
        asset = asset.linear_asset(BITMEX_CONTRACT_SIZE)
    return (
        asset.constant_order_latency(BITMEX_ORDER_ENTRY_LATENCY_NS, BITMEX_ORDER_RESPONSE_LATENCY_NS)
        .risk_adverse_queue_model()
        .no_partial_fill_exchange()
        .tick_size(BITMEX_TICK_SIZE)
        .lot_size(BITMEX_LOT_SIZE)
        .last_trades_capacity(100_000)
    )


def prepare_npz(symbol: str, yyyymmdd: str, skip_download: bool, buffer_rows: int | None) -> Path:
    out = NPZ_DIR / f"{BITMEX_EXCHANGE}_{symbol}_{yyyymmdd}.npz"
    if out.exists() and out.stat().st_size > 0:
        print(f"exists {out}")
        return out
    if skip_download:
        raise FileNotFoundError(f"{out} not found and --skip-download was set")
    key = tardis_key()
    download_file(BITMEX_EXCHANGE, "trades", symbol, yyyymmdd, key)
    download_file(BITMEX_EXCHANGE, "incremental_book_L2", symbol, yyyymmdd, key)
    return convert_bitmex(symbol, yyyymmdd, buffer_rows)


def run_one_day(symbol: str, yyyymmdd: str, sample_interval_ms: int, max_samples: int, npz_path: Path) -> np.ndarray:
    sample_interval_ns = sample_interval_ms * 1_000_000
    estimated = int((24 * 60 * 60 * 1_000) / sample_interval_ms) + 10_000
    capacity = max_samples if max_samples > 0 else estimated
    rows = np.zeros((capacity, BASE_COLUMNS), dtype=np.float64)
    hbt = HashMapMarketDepthBacktest([build_asset(npz_path, symbol)])
    count = collect_samples(hbt, sample_interval_ns, end_close_ts_ns(yyyymmdd), rows)
    return rows[:count].copy()


def concatenate_dicts(dicts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = dicts[0].keys()
    return {key: np.concatenate([item[key] for item in dicts]) for key in keys}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BitMEX high-frequency factor research.")
    parser.add_argument("--symbol", default="XBTUSDT", help="BitMEX symbol, e.g. XBTUSDT or XBTUSD.")
    parser.add_argument("--dates", nargs="+", default=list(DEFAULT_DATES), help="YYYYMMDD dates.")
    parser.add_argument("--skip-download", action="store_true", help="Use existing npz files only.")
    parser.add_argument("--buffer-rows", type=int, default=None, help="Optional tardis conversion buffer rows.")
    parser.add_argument("--sample-interval-ms", type=int, default=100, help="Market-state sampling interval.")
    parser.add_argument("--horizons-ms", nargs="+", type=int, default=list(DEFAULT_HORIZONS_MS))
    parser.add_argument("--bucket-horizon-ms", type=int, default=250)
    parser.add_argument("--buckets", type=int, default=10)
    parser.add_argument("--flow-decay", type=float, default=0.92)
    parser.add_argument("--maker-fee-rate", type=float, default=-0.0002)
    parser.add_argument("--hypothetical-order-qty", type=float, default=100.0)
    parser.add_argument("--queue-ahead-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--maker-fill-entry-latency-ms",
        type=float,
        default=BITMEX_ORDER_ENTRY_LATENCY_NS / 1_000_000.0,
        help="Latency before a hypothetical maker quote reaches the book.",
    )
    parser.add_argument(
        "--maker-fill-ttl-ms",
        type=float,
        default=ORDER_TTL_NS / 1_000_000.0,
        help="Maximum lifetime of a hypothetical maker quote.",
    )
    parser.add_argument("--max-samples-per-day", type=int, default=0)
    parser.add_argument("--result-tag", default="")
    parser.add_argument("--write-samples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_interval_ms <= 0:
        raise ValueError("--sample-interval-ms must be positive")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    dates = [str(date) for date in args.dates]
    horizons_ms = tuple(int(x) for x in args.horizons_ms)
    if args.bucket_horizon_ms not in horizons_ms:
        horizons_ms = tuple(sorted(set(horizons_ms + (args.bucket_horizon_ms,))))

    frames: list[dict[str, np.ndarray]] = []
    labels_list: list[dict[str, np.ndarray]] = []
    for yyyymmdd in dates:
        bitmex_npz = prepare_npz(args.symbol, yyyymmdd, args.skip_download, args.buffer_rows)
        raw = run_one_day(
            args.symbol,
            yyyymmdd,
            args.sample_interval_ms,
            args.max_samples_per_day,
            bitmex_npz,
        )
        print(f"{yyyymmdd}: sampled {len(raw):,} rows")
        frame = build_factor_frame(raw, args.sample_interval_ms, args.flow_decay)
        labels = build_labels(
            frame,
            horizons_ms,
            args.maker_fee_rate,
            args.hypothetical_order_qty,
            args.queue_ahead_multiplier,
        )
        labels.update(
            build_maker_fill_labels(
                frame,
                horizons_ms,
                args.maker_fee_rate,
                args.hypothetical_order_qty,
                args.queue_ahead_multiplier,
                args.maker_fill_entry_latency_ms,
                args.maker_fill_ttl_ms,
            )
        )
        frames.append(frame)
        labels_list.append(labels)

    data = concatenate_dicts(frames)
    labels = concatenate_dicts(labels_list)

    future_label_names = tuple(f"future_ret_{h}ms_bps" for h in horizons_ms)
    edge_labels = (
        f"bid_maker_edge_{args.bucket_horizon_ms}ms_bps",
        f"ask_maker_edge_{args.bucket_horizon_ms}ms_bps",
    )
    fill_labels = (
        f"bid_fill_prob_{args.bucket_horizon_ms}ms",
        f"ask_fill_prob_{args.bucket_horizon_ms}ms",
        f"bid_edge_when_filled_{args.bucket_horizon_ms}ms_bps",
        f"ask_edge_when_filled_{args.bucket_horizon_ms}ms_bps",
        f"bid_fill_queue_ratio_{args.bucket_horizon_ms}ms",
        f"ask_fill_queue_ratio_{args.bucket_horizon_ms}ms",
    )
    bucket_label = f"future_ret_{args.bucket_horizon_ms}ms_bps"
    tag = args.result_tag or f"{dates[0]}_{dates[-1]}_{args.sample_interval_ms}ms"
    prefix = RESULT_DIR / f"bitmex_{args.symbol.lower()}_{tag}"

    ic = factor_ic_rows(data, labels, FACTOR_NAMES, future_label_names)
    buckets = bucket_rows(data, labels, FACTOR_NAMES, bucket_label, edge_labels, args.buckets)
    fill_buckets = bucket_rows(
        data,
        labels,
        FACTOR_NAMES,
        fill_labels[0],
        (fill_labels[1], fill_labels[2], fill_labels[3], fill_labels[4], fill_labels[5]),
        args.buckets,
    )
    maker_fill_buckets = maker_fill_edge_bucket_rows(
        data,
        labels,
        FACTOR_NAMES,
        args.bucket_horizon_ms,
        args.buckets,
    )

    ic_path = prefix.with_suffix(".ic.csv")
    bucket_path = prefix.with_suffix(".buckets.csv")
    fill_bucket_path = prefix.with_suffix(".fill_buckets.csv")
    maker_fill_bucket_path = prefix.with_suffix(".maker_fill_edge_buckets.csv")
    report_path = prefix.with_suffix(".report.md")
    write_csv(ic_path, ic)
    write_csv(bucket_path, buckets)
    write_csv(fill_bucket_path, fill_buckets)
    write_csv(maker_fill_bucket_path, maker_fill_buckets)
    chart_files = write_charts(
        RESULT_DIR,
        prefix.name,
        ic,
        buckets,
        FACTOR_NAMES,
        bucket_label,
        edge_labels[0],
        edge_labels[1],
    )
    chart_files.update(
        write_fill_charts(
            RESULT_DIR,
            prefix.name,
            fill_buckets,
            FACTOR_NAMES,
            fill_labels[0],
            fill_labels[1],
            fill_labels[2],
            fill_labels[3],
        )
    )
    chart_files.update(
        write_lifecycle_fill_charts(
            RESULT_DIR,
            prefix.name,
            maker_fill_buckets,
            FACTOR_NAMES,
        )
    )

    output_files = {
        "ic_csv": ic_path,
        "bucket_csv": bucket_path,
        "fill_bucket_csv": fill_bucket_path,
        "maker_fill_edge_bucket_csv": maker_fill_bucket_path,
    }
    if args.write_samples:
        sample_path = prefix.with_suffix(".samples.npz")
        np.savez_compressed(sample_path, **data, **labels)
        output_files["samples_npz"] = sample_path

    report = render_report(
        args.symbol,
        dates,
        args.sample_interval_ms,
        horizons_ms,
        bucket_label,
        ic,
        buckets,
        maker_fill_buckets,
        output_files,
        chart_files,
        report_path,
    )
    report_path.write_text(report)
    print(f"wrote {ic_path}")
    print(f"wrote {bucket_path}")
    print(f"wrote {fill_bucket_path}")
    print(f"wrote {maker_fill_bucket_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
