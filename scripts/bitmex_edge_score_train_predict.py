from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bitmex_single_market_mm_backtest import BITMEX_ORDER_QTY, ORDER_TTL_NS
from factor_research.bitmex_factor_research import concatenate_dicts, prepare_npz, run_one_day
from factor_research.buckets import write_csv
from factor_research.edge_scoring import (
    fit_edge_models,
    predict_edge_scores,
    score_bucket_rows,
    target_arrays,
    write_model,
)
from factor_research.factors import build_factor_frame
from factor_research.maker_fill_labels import build_maker_fill_labels


RESULT_DIR = REPO_ROOT / "results" / "factor_research"


def date_range(start: str, end: str) -> list[str]:
    cur = datetime.strptime(start, "%Y%m%d")
    stop = datetime.strptime(end, "%Y%m%d")
    out = []
    while cur <= stop:
        out.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return out


def parse_dates(values: list[str] | None, start: str | None, end: str | None) -> list[str]:
    if values:
        return [str(x) for x in values]
    if start and end:
        return date_range(start, end)
    raise ValueError("provide explicit dates or start/end")


def load_factor_label_days(
    symbol: str,
    dates: list[str],
    skip_download: bool,
    buffer_rows: int | None,
    sample_interval_ms: int,
    max_samples_per_day: int,
    flow_decay: float,
    horizons_ms: tuple[int, ...],
    maker_fee_rate: float,
    order_qty: float,
    queue_ahead_multiplier: float,
    entry_latency_ms: float,
    ttl_ms: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    frames = []
    labels_list = []
    for yyyymmdd in dates:
        npz_path = prepare_npz(symbol, yyyymmdd, skip_download, buffer_rows)
        raw = run_one_day(symbol, yyyymmdd, sample_interval_ms, max_samples_per_day, npz_path)
        print(f"{yyyymmdd}: sampled {len(raw):,} rows", flush=True)
        frame = build_factor_frame(raw, sample_interval_ms, flow_decay)
        labels = build_maker_fill_labels(
            frame,
            horizons_ms,
            maker_fee_rate,
            order_qty,
            queue_ahead_multiplier,
            entry_latency_ms,
            ttl_ms,
        )
        frames.append(frame)
        labels_list.append(labels)
    return concatenate_dicts(frames), concatenate_dicts(labels_list)


def score_summary_rows(scores: dict[str, np.ndarray], labels: dict[str, np.ndarray], horizon_ms: int) -> list[dict[str, object]]:
    targets = target_arrays(labels, horizon_ms)
    rows = []
    for side in ("bid", "ask"):
        edge_pred = scores[f"pred_{side}_expected_edge_bps"]
        edge_if_filled_pred = scores[f"pred_{side}_edge_if_filled_bps"]
        fill_pred = scores[f"pred_{side}_fill_prob"]
        edge_actual = targets[f"{side}_expected_edge_bps"]
        edge_if_filled_actual = targets[f"{side}_edge_if_filled_bps"]
        fill_actual = targets[f"{side}_fill_prob"]
        valid_edge = np.isfinite(edge_pred) & np.isfinite(edge_actual)
        valid_edge_if_filled = np.isfinite(edge_if_filled_pred) & np.isfinite(edge_if_filled_actual)
        valid_fill = np.isfinite(fill_pred) & np.isfinite(fill_actual)
        edge_corr = (
            float(np.corrcoef(edge_pred[valid_edge], edge_actual[valid_edge])[0, 1])
            if int(valid_edge.sum()) > 2
            else float("nan")
        )
        edge_if_filled_corr = (
            float(
                np.corrcoef(
                    edge_if_filled_pred[valid_edge_if_filled],
                    edge_if_filled_actual[valid_edge_if_filled],
                )[0, 1]
            )
            if int(valid_edge_if_filled.sum()) > 2
            else float("nan")
        )
        fill_corr = (
            float(np.corrcoef(fill_pred[valid_fill], fill_actual[valid_fill])[0, 1])
            if int(valid_fill.sum()) > 2
            else float("nan")
        )
        rows.append(
            {
                "side": side,
                "samples": int(valid_edge.sum()),
                "pred_edge_mean_bps": float(np.nanmean(edge_pred[valid_edge])),
                "actual_edge_mean_bps": float(np.nanmean(edge_actual[valid_edge])),
                "edge_corr": edge_corr,
                "pred_edge_if_filled_mean_bps": float(np.nanmean(edge_if_filled_pred[valid_edge_if_filled])),
                "actual_edge_if_filled_mean_bps": float(np.nanmean(edge_if_filled_actual[valid_edge_if_filled])),
                "edge_if_filled_corr": edge_if_filled_corr,
                "pred_fill_prob_mean": float(np.nanmean(fill_pred[valid_fill])),
                "actual_fill_prob_mean": float(np.nanmean(fill_actual[valid_fill])),
                "fill_corr": fill_corr,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/predict maker edge scoring models.")
    parser.add_argument("--symbol", default="XBTUSDT")
    parser.add_argument("--train-dates", nargs="+", default=None)
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--test-dates", nargs="+", default=None)
    parser.add_argument("--test-start", default=None)
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--buffer-rows", type=int, default=None)
    parser.add_argument("--sample-interval-ms", type=int, default=100)
    parser.add_argument("--horizon-ms", type=int, default=250)
    parser.add_argument("--flow-decay", type=float, default=0.92)
    parser.add_argument("--maker-fee-rate", type=float, default=0.0)
    parser.add_argument("--hypothetical-order-qty", type=float, default=BITMEX_ORDER_QTY)
    parser.add_argument("--queue-ahead-multiplier", type=float, default=1.0)
    parser.add_argument("--maker-fill-entry-latency-ms", type=float, default=80.0)
    parser.add_argument("--maker-fill-ttl-ms", type=float, default=ORDER_TTL_NS / 1_000_000.0)
    parser.add_argument("--max-samples-per-day", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=2_000_000)
    parser.add_argument("--ridge-l2", type=float, default=100.0)
    parser.add_argument("--no-interactions", action="store_true")
    parser.add_argument("--clip-z", type=float, default=6.0)
    parser.add_argument("--score-buckets", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-scores", action="store_true")
    parser.add_argument("--result-tag", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train_dates = parse_dates(args.train_dates, args.train_start, args.train_end)
    test_dates = parse_dates(args.test_dates, args.test_start, args.test_end)
    horizons_ms = (int(args.horizon_ms),)
    tag = args.result_tag or f"edge_score_train_{train_dates[0]}_{train_dates[-1]}_test_{test_dates[0]}_{test_dates[-1]}"
    prefix = RESULT_DIR / f"bitmex_{args.symbol.lower()}_{tag}"

    print(f"train_dates={train_dates[0]}..{train_dates[-1]} n={len(train_dates)}", flush=True)
    train_data, train_labels = load_factor_label_days(
        args.symbol,
        train_dates,
        args.skip_download,
        args.buffer_rows,
        args.sample_interval_ms,
        args.max_samples_per_day,
        args.flow_decay,
        horizons_ms,
        args.maker_fee_rate,
        args.hypothetical_order_qty,
        args.queue_ahead_multiplier,
        args.maker_fill_entry_latency_ms,
        args.maker_fill_ttl_ms,
    )
    model = fit_edge_models(
        train_data,
        train_labels,
        args.horizon_ms,
        args.max_train_samples,
        args.ridge_l2,
        not args.no_interactions,
        args.clip_z,
        args.seed,
    )
    model_path = prefix.with_suffix(".edge_model.json")
    write_model(model_path, model)

    print(f"test_dates={test_dates[0]}..{test_dates[-1]} n={len(test_dates)}", flush=True)
    test_data, test_labels = load_factor_label_days(
        args.symbol,
        test_dates,
        args.skip_download,
        args.buffer_rows,
        args.sample_interval_ms,
        args.max_samples_per_day,
        args.flow_decay,
        horizons_ms,
        args.maker_fee_rate,
        args.hypothetical_order_qty,
        args.queue_ahead_multiplier,
        args.maker_fill_entry_latency_ms,
        args.maker_fill_ttl_ms,
    )
    scores = predict_edge_scores(model, test_data)

    summary_rows = score_summary_rows(scores, test_labels, args.horizon_ms)
    bucket_rows = score_bucket_rows(scores, test_labels, args.horizon_ms, args.score_buckets)
    summary_path = prefix.with_suffix(".edge_score_summary.csv")
    bucket_path = prefix.with_suffix(".edge_score_buckets.csv")
    write_csv(summary_path, summary_rows)
    write_csv(bucket_path, bucket_rows)

    if args.write_scores:
        score_path = prefix.with_suffix(".edge_scores.npz")
        np.savez_compressed(score_path, ts=test_data["ts"], **scores)
        print(f"wrote {score_path}", flush=True)

    print(f"wrote {model_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {bucket_path}", flush=True)
    for row in summary_rows:
        print(
            "side={side} samples={samples} edge_corr={edge_corr:.6f} "
            "edge_if_filled_corr={edge_if_filled_corr:.6f} "
            "pred_edge={pred_edge_mean_bps:.6f} actual_edge={actual_edge_mean_bps:.6f} "
            "fill_corr={fill_corr:.6f}".format(**row),
            flush=True,
        )


if __name__ == "__main__":
    main()
