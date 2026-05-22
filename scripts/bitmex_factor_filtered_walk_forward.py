from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = REPO_ROOT / "results"
FACTOR_RESULT_DIR = RESULT_DIR / "factor_research"


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
    dates = []
    for offset in range(train_days, 0, -1):
        dates.append((cur - timedelta(days=offset)).strftime("%Y%m%d"))
    return dates


def run_command(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def read_summary(path: Path) -> dict:
    with path.open() as file:
        return json.load(file)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rolling walk-forward factor-filtered maker test.")
    parser.add_argument("--symbol", default="XBTUSDT")
    parser.add_argument("--test-start", default="20260512")
    parser.add_argument("--test-end", default="20260518")
    parser.add_argument("--train-days", type=int, default=7)
    parser.add_argument("--sample-interval-ms", type=int, default=100)
    parser.add_argument("--horizons-ms", nargs="+", type=int, default=[100, 250, 500, 1000])
    parser.add_argument("--bucket-horizon-ms", type=int, default=250)
    parser.add_argument("--maker-fee-rate", type=float, default=0.0)
    parser.add_argument("--taker-fee-rate", type=float, default=0.0001)
    parser.add_argument("--exchange-model", choices=("no_partial", "strict_no_partial", "partial"), default="no_partial")
    parser.add_argument("--min-expected-edge-bps", type=float, default=0.02)
    parser.add_argument("--min-edge-if-filled-bps", type=float, default=0.25)
    parser.add_argument("--min-fill-prob", type=float, default=0.02)
    parser.add_argument("--min-fill-samples", type=int, default=1_000)
    parser.add_argument("--max-rules-per-side", type=int, default=8)
    parser.add_argument("--min-factor-matches", type=int, default=2)
    parser.add_argument("--result-tag", default="")
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable
    test_dates = date_range(args.test_start, args.test_end)
    run_tag = args.result_tag or (
        f"walk_forward_train{args.train_days}_{args.test_start}_{args.test_end}_"
        f"minmatch{args.min_factor_matches}_maker0_{args.exchange_model}"
    )
    rows = []
    out = RESULT_DIR / f"bitmex_xbtusdt_factor_filtered_maker_{run_tag}.walk_forward.csv"
    for test_date in test_dates:
        train_dates = previous_dates(test_date, args.train_days)
        train_tag = f"{run_tag}_train_{train_dates[0]}_{train_dates[-1]}_test_{test_date}"
        factor_prefix = FACTOR_RESULT_DIR / f"bitmex_{args.symbol.lower()}_{train_tag}"
        rules_csv = factor_prefix.with_suffix(".maker_fill_combo_rules.csv")

        factor_cmd = [
            python,
            "scripts/factor_research/bitmex_factor_research.py",
            "--symbol",
            args.symbol,
            "--dates",
            *train_dates,
            "--sample-interval-ms",
            str(args.sample_interval_ms),
            "--horizons-ms",
            *[str(x) for x in args.horizons_ms],
            "--bucket-horizon-ms",
            str(args.bucket_horizon_ms),
            "--maker-fee-rate",
            str(args.maker_fee_rate),
            "--result-tag",
            train_tag,
        ]
        if args.skip_download:
            factor_cmd.append("--skip-download")
        run_command(factor_cmd)

        backtest_tag = f"{run_tag}_test_{test_date}"
        backtest_cmd = [
            python,
            "scripts/bitmex_factor_filtered_maker.py",
            "--dates",
            test_date,
            "--exchange-model",
            args.exchange_model,
            "--maker-fee-rate",
            str(args.maker_fee_rate),
            "--taker-fee-rate",
            str(args.taker_fee_rate),
            "--rules-csv",
            str(rules_csv),
            "--min-expected-edge-bps",
            str(args.min_expected_edge_bps),
            "--min-edge-if-filled-bps",
            str(args.min_edge_if_filled_bps),
            "--min-fill-prob",
            str(args.min_fill_prob),
            "--min-fill-samples",
            str(args.min_fill_samples),
            "--max-rules-per-side",
            str(args.max_rules_per_side),
            "--min-factor-matches",
            str(args.min_factor_matches),
            "--result-tag",
            backtest_tag,
        ]
        if args.skip_download:
            backtest_cmd.append("--skip-download")
        run_command(backtest_cmd)

        summary_path = RESULT_DIR / f"bitmex_xbtusdt_factor_filtered_maker_{backtest_tag}_{test_date}.summary.json"
        summary = read_summary(summary_path)
        rows.append(
            {
                "test_date": test_date,
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "total_pnl_usdt": summary["total_pnl_usdt"],
                "gross_pnl_before_fee_usdt": summary["gross_pnl_before_fee_usdt"],
                "maker_rebate_usdt": summary["maker_rebate_usdt"],
                "fills": summary["fills"],
                "buy_fills": summary["buy_fills"],
                "sell_fills": summary["sell_fills"],
                "max_position_contracts_seen": summary["max_position_contracts_seen"],
                "selected_rules": summary["selected_rules"],
                "min_factor_matches": summary["min_factor_matches"],
                "selected_bid_factor_count": summary.get("selected_bid_factor_count", ""),
                "selected_ask_factor_count": summary.get("selected_ask_factor_count", ""),
                "selected_bid_factors": summary.get("selected_bid_factors", ""),
                "selected_ask_factors": summary.get("selected_ask_factors", ""),
                "bid_factor_match_possible": summary.get("bid_factor_match_possible", ""),
                "ask_factor_match_possible": summary.get("ask_factor_match_possible", ""),
                "place_bid": summary.get("place_bid", ""),
                "place_ask": summary.get("place_ask", ""),
                "suppress_bid": summary.get("suppress_bid", ""),
                "suppress_ask": summary.get("suppress_ask", ""),
                "factor_gate_bid": summary["factor_gate_bid"],
                "factor_gate_ask": summary["factor_gate_ask"],
                "rules_csv": str(rules_csv),
                "summary_json": str(summary_path),
            }
        )
        write_csv(out, rows)

    total_pnl = sum(float(row["total_pnl_usdt"]) for row in rows)
    total_gross = sum(float(row["gross_pnl_before_fee_usdt"]) for row in rows)
    total_fills = sum(int(row["fills"]) for row in rows)
    pos_days = sum(float(row["gross_pnl_before_fee_usdt"]) > 0 for row in rows)
    print(
        f"walk_forward={out} total_pnl={total_pnl:.6f} gross={total_gross:.6f} "
        f"fills={total_fills} positive_days={pos_days}/{len(rows)}"
    )


if __name__ == "__main__":
    main()
