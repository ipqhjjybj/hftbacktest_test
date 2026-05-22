from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = REPO_ROOT / "results" / "factor_research"


def date_range(start: str, end: str) -> list[str]:
    cur = datetime.strptime(start, "%Y%m%d")
    stop = datetime.strptime(end, "%Y%m%d")
    out = []
    while cur <= stop:
        out.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return out


def previous_dates(test_date: str, train_days: int) -> list[str]:
    cur = datetime.strptime(test_date, "%Y%m%d")
    return [(cur - timedelta(days=offset)).strftime("%Y%m%d") for offset in range(train_days, 0, -1)]


def run_command(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_buckets(test_date: str, bucket_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out = []
    by_side: dict[str, list[dict[str, str]]] = {}
    for row in bucket_rows:
        by_side.setdefault(row["side"], []).append(row)
    for side, rows in by_side.items():
        rows.sort(key=lambda row: int(row["bucket"]))
        if not rows:
            continue
        bottom = rows[0]
        top = rows[-1]
        out.append(
            {
                "test_date": test_date,
                "side": side,
                "buckets": len(rows),
                "bottom_pred_edge_bps": float(bottom["pred_edge_mean_bps"]),
                "top_pred_edge_bps": float(top["pred_edge_mean_bps"]),
                "bottom_actual_edge_bps": float(bottom["actual_expected_edge_mean_bps"]),
                "top_actual_edge_bps": float(top["actual_expected_edge_mean_bps"]),
                "actual_edge_spread_bps": float(top["actual_expected_edge_mean_bps"])
                - float(bottom["actual_expected_edge_mean_bps"]),
                "bottom_actual_edge_if_filled_bps": float(bottom["actual_edge_if_filled_mean_bps"]),
                "top_actual_edge_if_filled_bps": float(top["actual_edge_if_filled_mean_bps"]),
                "actual_edge_if_filled_spread_bps": float(top["actual_edge_if_filled_mean_bps"])
                - float(bottom["actual_edge_if_filled_mean_bps"]),
                "bottom_actual_fill_prob": float(bottom["actual_fill_prob_mean"]),
                "top_actual_fill_prob": float(top["actual_fill_prob_mean"]),
                "actual_fill_spread": float(top["actual_fill_prob_mean"]) - float(bottom["actual_fill_prob_mean"]),
                "top_samples": int(float(top["samples"])),
                "bottom_samples": int(float(bottom["samples"])),
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rolling walk-forward edge score validation.")
    parser.add_argument("--symbol", default="XBTUSDT")
    parser.add_argument("--test-start", default="20260512")
    parser.add_argument("--test-end", default="20260518")
    parser.add_argument("--train-days", type=int, default=14)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--sample-interval-ms", type=int, default=100)
    parser.add_argument("--horizon-ms", type=int, default=250)
    parser.add_argument("--maker-fee-rate", type=float, default=0.0)
    parser.add_argument("--max-samples-per-day", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=2_000_000)
    parser.add_argument("--ridge-l2", type=float, default=100.0)
    parser.add_argument("--no-interactions", action="store_true")
    parser.add_argument("--score-buckets", type=int, default=10)
    parser.add_argument("--result-tag", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable
    test_dates = date_range(args.test_start, args.test_end)
    run_tag = args.result_tag or (
        f"edge_score_wf_train{args.train_days}_{args.test_start}_{args.test_end}_"
        f"h{args.horizon_ms}_maker0"
    )

    daily_rows: list[dict[str, object]] = []
    bucket_summary_rows: list[dict[str, object]] = []
    daily_out = RESULT_DIR / f"bitmex_{args.symbol.lower()}_{run_tag}.edge_score_walk_forward_daily.csv"
    bucket_summary_out = RESULT_DIR / f"bitmex_{args.symbol.lower()}_{run_tag}.edge_score_walk_forward_bucket_summary.csv"

    for test_date in test_dates:
        train_dates = previous_dates(test_date, args.train_days)
        child_tag = f"{run_tag}_train_{train_dates[0]}_{train_dates[-1]}_test_{test_date}"
        cmd = [
            python,
            "scripts/bitmex_edge_score_train_predict.py",
            "--symbol",
            args.symbol,
            "--train-dates",
            *train_dates,
            "--test-dates",
            test_date,
            "--sample-interval-ms",
            str(args.sample_interval_ms),
            "--horizon-ms",
            str(args.horizon_ms),
            "--maker-fee-rate",
            str(args.maker_fee_rate),
            "--max-samples-per-day",
            str(args.max_samples_per_day),
            "--max-train-samples",
            str(args.max_train_samples),
            "--ridge-l2",
            str(args.ridge_l2),
            "--score-buckets",
            str(args.score_buckets),
            "--result-tag",
            child_tag,
        ]
        if args.skip_download:
            cmd.append("--skip-download")
        if args.no_interactions:
            cmd.append("--no-interactions")
        run_command(cmd)

        prefix = RESULT_DIR / f"bitmex_{args.symbol.lower()}_{child_tag}"
        summary_path = prefix.with_suffix(".edge_score_summary.csv")
        bucket_path = prefix.with_suffix(".edge_score_buckets.csv")
        for row in read_csv(summary_path):
            daily_rows.append(
                {
                    "test_date": test_date,
                    "train_start": train_dates[0],
                    "train_end": train_dates[-1],
                    **row,
                    "summary_csv": str(summary_path),
                    "bucket_csv": str(bucket_path),
                }
            )
        bucket_summary_rows.extend(summarize_buckets(test_date, read_csv(bucket_path)))
        write_csv(daily_out, daily_rows)
        write_csv(bucket_summary_out, bucket_summary_rows)

    print(f"daily={daily_out}", flush=True)
    print(f"bucket_summary={bucket_summary_out}", flush=True)
    for side in ("bid", "ask"):
        rows = [row for row in bucket_summary_rows if row["side"] == side]
        if not rows:
            continue
        edge_spread = sum(float(row["actual_edge_spread_bps"]) for row in rows) / len(rows)
        fill_spread = sum(float(row["actual_fill_spread"]) for row in rows) / len(rows)
        print(
            f"side={side} days={len(rows)} avg_top_minus_bottom_actual_edge_bps={edge_spread:.6f} "
            f"avg_top_minus_bottom_fill_prob={fill_spread:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
