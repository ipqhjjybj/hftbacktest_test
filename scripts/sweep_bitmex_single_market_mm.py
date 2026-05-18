import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_DATES = ("20260512", "20260513", "20260514")
ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
STRATEGY_SCRIPT = ROOT / "scripts" / "bitmex_single_market_mm_backtest.py"


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(float(item.strip())) for item in value.split(",") if item.strip()]


def tag_float(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text


def param_tag(params: dict) -> str:
    return (
        f"sw_hs{tag_float(params['base_half_spread_bps'])}"
        f"_ttl{params['order_ttl_ms']}"
        f"_mom{tag_float(params['momentum_cancel_bps'])}"
        f"_mic{tag_float(params['microprice_cancel_bps'])}"
        f"_vol{tag_float(params['vol_spread_multiplier'])}"
        f"_skew{tag_float(params['inventory_skew_bps'])}"
    )


def summary_path(tag: str, date: str) -> Path:
    return RESULT_DIR / f"bitmex_xbtusd_single_market_mm_{tag}_{date}.summary.json"


def load_summary(tag: str, date: str) -> dict:
    return json.loads(summary_path(tag, date).read_text())


def run_combo(params: dict, dates: list[str], skip_existing: bool, verbose_child: bool) -> list[dict]:
    tag = param_tag(params)
    paths = [summary_path(tag, date) for date in dates]
    if not skip_existing or not all(path.exists() for path in paths):
        cmd = [
            sys.executable,
            str(STRATEGY_SCRIPT),
            "--dates",
            *dates,
            "--skip-download",
            "--result-tag",
            tag,
            "--base-half-spread-bps",
            str(params["base_half_spread_bps"]),
            "--order-ttl-ms",
            str(params["order_ttl_ms"]),
            "--momentum-cancel-bps",
            str(params["momentum_cancel_bps"]),
            "--microprice-cancel-bps",
            str(params["microprice_cancel_bps"]),
            "--vol-spread-multiplier",
            str(params["vol_spread_multiplier"]),
            "--inventory-skew-bps",
            str(params["inventory_skew_bps"]),
            "--soft-position-contracts",
            str(params["soft_position_contracts"]),
            "--max-position-contracts",
            str(params["max_position_contracts"]),
        ]
        if verbose_child:
            subprocess.run(cmd, cwd=ROOT, check=True)
        else:
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            if proc.returncode != 0:
                print(proc.stdout)
                print(proc.stderr, file=sys.stderr)
                raise subprocess.CalledProcessError(proc.returncode, cmd)
    return [load_summary(tag, date) for date in dates]


def evaluate(params: dict, summaries: list[dict]) -> dict:
    pnls = [float(item["total_pnl_usdt"]) for item in summaries]
    fills = [int(item["maker_fills"]) for item in summaries]
    toxic = [int(item["toxic_fill_events"]) for item in summaries]
    min_pnl = min(pnls)
    total_pnl = sum(pnls)
    total_fills = sum(fills)
    total_toxic = sum(toxic)
    return {
        **params,
        "tag": param_tag(params),
        "all_positive": all(pnl > 0 for pnl in pnls),
        "min_daily_pnl_usdt": min_pnl,
        "total_pnl_usdt": total_pnl,
        "total_fills": total_fills,
        "total_toxic_fills": total_toxic,
        "toxic_fill_rate": total_toxic / total_fills if total_fills > 0 else 0.0,
        **{f"pnl_{summary['date']}": pnl for summary, pnl in zip(summaries, pnls)},
        **{f"fills_{summary['date']}": fill for summary, fill in zip(summaries, fills)},
        **{f"toxic_{summary['date']}": tox for summary, tox in zip(summaries, toxic)},
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep BitMEX XBTUSD single-market MM parameters and find combinations positive on every date."
    )
    parser.add_argument("--dates", nargs="+", default=list(DEFAULT_DATES))
    parser.add_argument("--half-spreads", default="2.5,3.0,3.5,4.0,4.5,5.0")
    parser.add_argument("--ttls-ms", default="200,300,500")
    parser.add_argument("--momentum-bps", default="0.8,1.0")
    parser.add_argument("--microprice-bps", default="0.5,0.8")
    parser.add_argument("--vol-multipliers", default="0.3,0.5")
    parser.add_argument("--inventory-skews", default="4.0")
    parser.add_argument("--soft-position-contracts", type=float, default=500.0)
    parser.add_argument("--max-position-contracts", type=float, default=1000.0)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N parameter combinations.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing summary files when present.")
    parser.add_argument("--verbose-child", action="store_true", help="Print each child backtest's full output.")
    parser.add_argument(
        "--out",
        default="results/bitmex_xbtusd_single_market_mm_sweep.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dates = list(args.dates)
    grid = list(
        itertools.product(
            parse_float_list(args.half_spreads),
            parse_int_list(args.ttls_ms),
            parse_float_list(args.momentum_bps),
            parse_float_list(args.microprice_bps),
            parse_float_list(args.vol_multipliers),
            parse_float_list(args.inventory_skews),
        )
    )
    if args.limit > 0:
        grid = grid[: args.limit]

    rows = []
    positives = []
    for idx, combo in enumerate(grid, start=1):
        params = {
            "base_half_spread_bps": combo[0],
            "order_ttl_ms": combo[1],
            "momentum_cancel_bps": combo[2],
            "microprice_cancel_bps": combo[3],
            "vol_spread_multiplier": combo[4],
            "inventory_skew_bps": combo[5],
            "soft_position_contracts": args.soft_position_contracts,
            "max_position_contracts": args.max_position_contracts,
        }
        print(f"[{idx}/{len(grid)}] {param_tag(params)}")
        summaries = run_combo(params, dates, args.skip_existing, args.verbose_child)
        row = evaluate(params, summaries)
        rows.append(row)
        if row["all_positive"]:
            positives.append(row)
            print(
                "  POSITIVE "
                f"total={row['total_pnl_usdt']:.6f} "
                f"min_daily={row['min_daily_pnl_usdt']:.6f} "
                f"fills={row['total_fills']}"
            )
        else:
            print(
                "  no "
                f"total={row['total_pnl_usdt']:.6f} "
                f"min_daily={row['min_daily_pnl_usdt']:.6f} "
                f"fills={row['total_fills']}"
            )

    rows.sort(key=lambda item: (item["all_positive"], item["min_daily_pnl_usdt"], item["total_pnl_usdt"]), reverse=True)
    out = ROOT / args.out
    write_csv(out, rows)

    print(f"sweep_csv={out}")
    print("top_candidates:")
    for row in rows[:10]:
        print(
            f"{row['tag']} all_positive={row['all_positive']} "
            f"total={row['total_pnl_usdt']:.6f} "
            f"min_daily={row['min_daily_pnl_usdt']:.6f} "
            f"fills={row['total_fills']} "
            f"toxic_rate={row['toxic_fill_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
