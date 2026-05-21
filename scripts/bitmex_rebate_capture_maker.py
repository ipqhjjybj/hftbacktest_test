import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bitmex_single_market_quality_mm_strategies as quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest BitMEX XBTUSDT rebate-capture maker.")
    parser.add_argument("--dates", nargs="+", default=["20260516", "20260517", "20260518"])
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--buffer-rows", type=int, default=None)
    parser.add_argument("--result-tag", default="")
    parser.add_argument("--exchange-model", choices=("no_partial", "strict_no_partial", "partial", "live_l2"), default="no_partial")
    parser.add_argument("--maker-fee-rate", type=float, default=-0.0002)
    parser.add_argument("--taker-fee-rate", type=float, default=0.0001)
    parser.add_argument("--base-half-spread-bps", type=float, default=2.5)
    parser.add_argument("--edge-threshold-bps", type=float, default=1.2)
    parser.add_argument("--vol-penalty-mult", type=float, default=0.6)
    parser.add_argument("--momentum-penalty-mult", type=float, default=1.2)
    parser.add_argument("--microprice-penalty-mult", type=float, default=1.0)
    parser.add_argument("--max-adverse-bps", type=float, default=4.0)
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    quality.STRATEGY_MODE = quality.STRATEGY_REBATE_GATED_MAKER
    quality.MAKER_FEE_RATE = args.maker_fee_rate
    quality.TAKER_FEE_RATE = args.taker_fee_rate
    quality.EXCHANGE_MODEL = args.exchange_model
    quality.RESULT_TAG = args.result_tag or f"rebate_capture_{args.exchange_model}"
    quality.REBATE_BASE_HALF_SPREAD_BPS = args.base_half_spread_bps
    quality.REBATE_EDGE_THRESHOLD_BPS = args.edge_threshold_bps
    quality.REBATE_VOL_PENALTY_MULT = args.vol_penalty_mult
    quality.REBATE_MOMENTUM_PENALTY_MULT = args.momentum_penalty_mult
    quality.REBATE_MICROPRICE_PENALTY_MULT = args.microprice_penalty_mult
    quality.REBATE_MAX_ADVERSE_BPS = args.max_adverse_bps


def write_aggregate(result_paths: list[Path], result_tag: str) -> Path:
    rows = []
    for result_path in result_paths:
        rows.append(json.loads(result_path.with_suffix(".summary.json").read_text()))

    out = quality.RESULT_DIR / f"bitmex_xbtusdt_rebate_capture_maker_{result_tag}.aggregate.csv"
    fields = [
        "date",
        "total_pnl_usdt",
        "gross_pnl_before_fee_usdt",
        "maker_rebate_usdt",
        "fills",
        "buy_fills",
        "sell_fills",
        "filled_base_btc",
        "max_position_contracts_seen",
        "gate_edge",
        "gate_vol",
        "gate_momentum",
        "gate_microprice",
        "gate_position",
        "force_close_pnl_usdt",
        "final_position_contracts",
    ]
    with out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    return out


def main() -> None:
    args = parse_args()
    apply_args(args)

    quality.CSV_DIR.mkdir(parents=True, exist_ok=True)
    quality.NPZ_DIR.mkdir(parents=True, exist_ok=True)
    quality.RESULT_DIR.mkdir(parents=True, exist_ok=True)

    key = None if args.skip_download else quality.tardis_key()
    outputs = []
    for yyyymmdd in args.dates:
        if not args.skip_download:
            quality.download_file(quality.BITMEX_EXCHANGE, "trades", quality.BITMEX_SYMBOL, yyyymmdd, key)
            quality.download_file(quality.BITMEX_EXCHANGE, "incremental_book_L2", quality.BITMEX_SYMBOL, yyyymmdd, key)
        bitmex_npz = quality.convert_bitmex(quality.BITMEX_SYMBOL, yyyymmdd, args.buffer_rows)
        outputs.append(quality.run_backtest(bitmex_npz, yyyymmdd))

    aggregate = write_aggregate(outputs, quality.RESULT_TAG)
    total_pnl = 0.0
    total_gross = 0.0
    total_rebate = 0.0
    total_fills = 0
    for result_path in outputs:
        summary = json.loads(result_path.with_suffix(".summary.json").read_text())
        total_pnl += summary["total_pnl_usdt"]
        total_gross += summary["gross_pnl_before_fee_usdt"]
        total_rebate += summary["maker_rebate_usdt"]
        total_fills += summary["fills"]
    print(
        f"aggregate={aggregate} total_pnl={total_pnl:.6f} gross={total_gross:.6f} "
        f"rebate={total_rebate:.6f} fills={total_fills}"
    )
    print("all_results=" + ",".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
