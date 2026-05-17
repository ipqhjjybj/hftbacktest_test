import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
from numba import njit

from hftbacktest import (
    BacktestAsset,
    GTX,
    LIMIT,
    HashMapMarketDepthBacktest,
    Recorder,
)

from bitmex_gate_mm_arb_backtest import (
    BITMEX_CONTRACT_SIZE,
    BITMEX_LOT_SIZE,
    BITMEX_ORDER_QTY,
    BITMEX_SYMBOL,
    BITMEX_TICK_SIZE,
    GATE_CONTRACT_SIZE,
    GATE_LOT_SIZE,
    GATE_SYMBOL,
    GATE_TICK_SIZE,
    NPZ_DIR,
    RESULT_DIR,
    ceil_to_tick,
    combined_equity_usdt,
    convert_pair,
    current_bitmex_base,
    current_gate_base,
    floor_to_tick,
    force_flatten,
    hedge_gate_net_exposure,
    npz_path,
    update_risk_metrics,
)


DATE = "20260512"
END_CLOSE_TS_NS = 1_778_630_400_000_000_000

ORDER_UPDATE_INTERVAL_NS = 10_000_000
BITMEX_COMMAND_INFLIGHT_NS = 80_000_000
BITMEX_ORDER_ENTRY_LATENCY_NS = 80_000_000
BITMEX_ORDER_RESPONSE_LATENCY_NS = 80_000_000
GATE_ORDER_ENTRY_LATENCY_NS = 20_000_000
GATE_ORDER_RESPONSE_LATENCY_NS = 20_000_000
GATE_HEDGE_INFLIGHT_NS = 20_000_000

BASE_BID_SPREAD_RATIO = 0.0008
BASE_ASK_SPREAD_RATIO = 0.0008
INVENTORY_SKEW_BPS = 12.0
MAX_REDUCE_QUOTE_CROSS_BPS = 2.0
SOFT_INVENTORY_LIMIT_BASE = 0.0010
HARD_INVENTORY_LIMIT_BASE = 0.0025
MAX_INVENTORY_HOLD_NS = 3_000_000_000
MAX_POSITION_BASE = 0.0030
MAX_GROSS_POSITION_BASE = 0.02
TARGET_BITMEX_BUY_FILLS = 10
TARGET_BITMEX_SELL_FILLS = 10
MAX_FILL_COUNT_IMBALANCE = 1


@njit
def clamp(value, low, high):
    return min(max(value, low), high)


@njit
def bitmex_base_from_contracts(contracts, price):
    if price <= 0:
        return 0.0
    return contracts * BITMEX_CONTRACT_SIZE / price


@njit
def inventory_ratio(net_base):
    if SOFT_INVENTORY_LIMIT_BASE <= 0:
        return 0.0
    return clamp(net_base / SOFT_INVENTORY_LIMIT_BASE, -1.0, 1.0)


@njit
def inventory_bid_spread(net_base):
    skew = INVENTORY_SKEW_BPS / 10_000.0
    max_cross = MAX_REDUCE_QUOTE_CROSS_BPS / 10_000.0
    ratio = inventory_ratio(net_base)
    return max(-max_cross, BASE_BID_SPREAD_RATIO + ratio * skew)


@njit
def inventory_ask_spread(net_base):
    skew = INVENTORY_SKEW_BPS / 10_000.0
    max_cross = MAX_REDUCE_QUOTE_CROSS_BPS / 10_000.0
    ratio = inventory_ratio(net_base)
    return max(-max_cross, BASE_ASK_SPREAD_RATIO - ratio * skew)


@njit
def gross_position_base(hbt):
    return abs(current_bitmex_base(hbt)) + abs(current_gate_base(hbt))


@njit
def allow_bid_by_fill_balance(buy_fills, sell_fills, net_base):
    if net_base < -SOFT_INVENTORY_LIMIT_BASE:
        return True
    if buy_fills >= TARGET_BITMEX_BUY_FILLS:
        return False
    return buy_fills - sell_fills < MAX_FILL_COUNT_IMBALANCE


@njit
def allow_ask_by_fill_balance(buy_fills, sell_fills, net_base):
    if net_base > SOFT_INVENTORY_LIMIT_BASE:
        return True
    if sell_fills >= TARGET_BITMEX_SELL_FILLS:
        return False
    return sell_fills - buy_fills < MAX_FILL_COUNT_IMBALANCE


@njit
def cancel_all_side(hbt, asset_no, side):
    orders = hbt.orders(asset_no)
    values = orders.values()
    while values.has_next():
        order = values.get()
        if order.side == side and order.cancellable:
            hbt.cancel(asset_no, order.order_id, False)


@njit
def manage_inventory_bid(hbt, order_id, inflight_until, buy_fills, sell_fills):
    if hbt.current_timestamp < inflight_until:
        return order_id, inflight_until

    bitmex_depth = hbt.depth(0)
    gate_depth = hbt.depth(1)
    if bitmex_depth.best_bid <= 0 or bitmex_depth.best_ask <= 0:
        cancel_all_side(hbt, 0, 1)
        return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS
    if gate_depth.best_bid <= 0 or gate_depth.best_ask <= 0:
        cancel_all_side(hbt, 0, 1)
        return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS

    bitmex_mid = (bitmex_depth.best_bid + bitmex_depth.best_ask) / 2.0
    order_base = bitmex_base_from_contracts(BITMEX_ORDER_QTY, bitmex_mid)
    net_base = current_bitmex_base(hbt) + current_gate_base(hbt)
    gross_base = gross_position_base(hbt)
    spread = inventory_bid_spread(net_base)
    bid_price = floor_to_tick(min(gate_depth.best_bid * (1.0 - spread), bitmex_depth.best_bid), BITMEX_TICK_SIZE)
    existing = hbt.orders(0).get(order_id)
    reduces_net = net_base < -SOFT_INVENTORY_LIMIT_BASE

    should_quote = (
        allow_bid_by_fill_balance(buy_fills, sell_fills, net_base)
        and net_base < SOFT_INVENTORY_LIMIT_BASE
        and abs(net_base + order_base) <= MAX_POSITION_BASE
        and (gross_base + order_base <= MAX_GROSS_POSITION_BASE or reduces_net)
        and bid_price > 0
        and BITMEX_ORDER_QTY >= BITMEX_LOT_SIZE
    )
    if not should_quote:
        if existing is not None and existing.cancellable:
            hbt.cancel(0, order_id, False)
            return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS
        return order_id, inflight_until

    if existing is not None:
        if existing.cancellable and (existing.price != bid_price or existing.qty != BITMEX_ORDER_QTY):
            hbt.modify(0, order_id, bid_price, BITMEX_ORDER_QTY, False)
            return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS
        return order_id, inflight_until

    hbt.submit_buy_order(0, order_id, bid_price, BITMEX_ORDER_QTY, GTX, LIMIT, False)
    return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS


@njit
def manage_inventory_ask(hbt, order_id, inflight_until, buy_fills, sell_fills):
    if hbt.current_timestamp < inflight_until:
        return order_id, inflight_until

    bitmex_depth = hbt.depth(0)
    gate_depth = hbt.depth(1)
    if bitmex_depth.best_bid <= 0 or bitmex_depth.best_ask <= 0:
        cancel_all_side(hbt, 0, -1)
        return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS
    if gate_depth.best_bid <= 0 or gate_depth.best_ask <= 0:
        cancel_all_side(hbt, 0, -1)
        return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS

    bitmex_mid = (bitmex_depth.best_bid + bitmex_depth.best_ask) / 2.0
    order_base = bitmex_base_from_contracts(BITMEX_ORDER_QTY, bitmex_mid)
    net_base = current_bitmex_base(hbt) + current_gate_base(hbt)
    gross_base = gross_position_base(hbt)
    spread = inventory_ask_spread(net_base)
    ask_price = ceil_to_tick(max(gate_depth.best_ask * (1.0 + spread), bitmex_depth.best_ask), BITMEX_TICK_SIZE)
    existing = hbt.orders(0).get(order_id)
    reduces_net = net_base > SOFT_INVENTORY_LIMIT_BASE

    should_quote = (
        allow_ask_by_fill_balance(buy_fills, sell_fills, net_base)
        and net_base > -SOFT_INVENTORY_LIMIT_BASE
        and abs(net_base - order_base) <= MAX_POSITION_BASE
        and (gross_base + order_base <= MAX_GROSS_POSITION_BASE or reduces_net)
        and ask_price > 0
        and BITMEX_ORDER_QTY >= BITMEX_LOT_SIZE
    )
    if not should_quote:
        if existing is not None and existing.cancellable:
            hbt.cancel(0, order_id, False)
            return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS
        return order_id, inflight_until

    if existing is not None:
        if existing.cancellable and (existing.price != ask_price or existing.qty != BITMEX_ORDER_QTY):
            hbt.modify(0, order_id, ask_price, BITMEX_ORDER_QTY, False)
            return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS
        return order_id, inflight_until

    hbt.submit_sell_order(0, order_id, ask_price, BITMEX_ORDER_QTY, GTX, LIMIT, False)
    return order_id, hbt.current_timestamp + BITMEX_COMMAND_INFLIGHT_NS


@njit
def maybe_inventory_hedge(hbt, gate_order_id, gate_inflight_until, inventory_enter_ts, metrics):
    if hbt.current_timestamp < gate_inflight_until:
        return gate_order_id, gate_inflight_until, inventory_enter_ts

    net_base = current_bitmex_base(hbt) + current_gate_base(hbt)
    abs_net = abs(net_base)
    if abs_net <= SOFT_INVENTORY_LIMIT_BASE:
        return gate_order_id, gate_inflight_until, inventory_enter_ts

    hold_ns = 0
    if inventory_enter_ts > 0:
        hold_ns = hbt.current_timestamp - inventory_enter_ts
    hard_trigger = abs_net >= HARD_INVENTORY_LIMIT_BASE
    hold_trigger = inventory_enter_ts > 0 and hold_ns >= MAX_INVENTORY_HOLD_NS
    if not hard_trigger and not hold_trigger:
        return gate_order_id, gate_inflight_until, inventory_enter_ts

    prev_order_id = gate_order_id
    gate_order_id = hedge_gate_net_exposure(hbt, gate_order_id, metrics, 0.0, 0.0, inventory_enter_ts)
    if gate_order_id != prev_order_id:
        metrics[20] += 1
        metrics[21] += 1
        if hard_trigger:
            metrics[26] += 1
        if hold_trigger:
            metrics[25] += 1
        gate_inflight_until = hbt.current_timestamp + GATE_HEDGE_INFLIGHT_NS
        inventory_enter_ts = 0
    return gate_order_id, gate_inflight_until, inventory_enter_ts


@njit
def run_inventory_strategy(hbt, recorder, metrics):
    bitmex_bid_order_id = 10_000
    bitmex_ask_order_id = 20_000
    gate_order_id = 30_000
    bitmex_bid_inflight_until = 0
    bitmex_ask_inflight_until = 0
    gate_inflight_until = 0
    inventory_enter_ts = 0
    last_record_ts = 0
    last_bitmex_pos = hbt.position(0)
    last_bitmex_trades = hbt.state_values(0).num_trades
    last_bitmex_trading_value = hbt.state_values(0).trading_value

    while hbt.elapse(ORDER_UPDATE_INTERVAL_NS) == 0:
        if hbt.current_timestamp >= END_CLOSE_TS_NS:
            break

        hbt.clear_inactive_orders(0)
        hbt.clear_inactive_orders(1)

        bitmex_bid_order_id, bitmex_bid_inflight_until = manage_inventory_bid(
            hbt, bitmex_bid_order_id, bitmex_bid_inflight_until, metrics[11], metrics[12]
        )
        bitmex_ask_order_id, bitmex_ask_inflight_until = manage_inventory_ask(
            hbt, bitmex_ask_order_id, bitmex_ask_inflight_until, metrics[11], metrics[12]
        )

        bitmex_state = hbt.state_values(0)
        if bitmex_state.num_trades > last_bitmex_trades:
            bitmex_depth = hbt.depth(0)
            bitmex_mid = (bitmex_depth.best_bid + bitmex_depth.best_ask) / 2.0
            delta_contracts = bitmex_state.position - last_bitmex_pos
            delta_trading_value = bitmex_state.trading_value - last_bitmex_trading_value
            metrics[0] += abs(bitmex_base_from_contracts(delta_contracts, bitmex_mid))
            metrics[4] += bitmex_state.num_trades - last_bitmex_trades
            if delta_contracts > 0:
                metrics[11] += bitmex_state.num_trades - last_bitmex_trades
            elif delta_contracts < 0:
                metrics[12] += bitmex_state.num_trades - last_bitmex_trades
            last_bitmex_pos = bitmex_state.position
            last_bitmex_trades = bitmex_state.num_trades
            last_bitmex_trading_value = bitmex_state.trading_value

        net_base = current_bitmex_base(hbt) + current_gate_base(hbt)
        if abs(net_base) > SOFT_INVENTORY_LIMIT_BASE:
            if inventory_enter_ts == 0:
                inventory_enter_ts = hbt.current_timestamp
            hold_ns = hbt.current_timestamp - inventory_enter_ts
            metrics[22] = max(metrics[22], hold_ns)
        else:
            inventory_enter_ts = 0

        if abs(net_base) > SOFT_INVENTORY_LIMIT_BASE:
            metrics[23] += 1
        if abs(net_base) > HARD_INVENTORY_LIMIT_BASE:
            metrics[24] += 1

        gate_order_id, gate_inflight_until, inventory_enter_ts = maybe_inventory_hedge(
            hbt, gate_order_id, gate_inflight_until, inventory_enter_ts, metrics
        )

        update_risk_metrics(hbt, metrics)

        if hbt.current_timestamp - last_record_ts >= 1_000_000_000:
            recorder.record(hbt)
            last_record_ts = hbt.current_timestamp

    force_flatten(hbt, metrics)
    recorder.record(hbt)
    return True


def run_backtest(bitmex_npz: Path, gate_npz: Path) -> Path:
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
    metrics = np.zeros(32, dtype=np.float64)
    ok = run_inventory_strategy(hbt, recorder.recorder, metrics)
    if not ok:
        raise RuntimeError("strategy returned false")

    out = RESULT_DIR / f"bitmex_xbtusd_gate_btc_usdt_inventory_{DATE}.npz"
    np.savez_compressed(out, **{"0": recorder.get(0), "1": recorder.get(1), "metrics": metrics})
    write_summary(out, metrics)
    return out


def signed_money(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.4f} USDT"


def signed_base(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.8f} BTC"


def write_summary(result_npz: Path, metrics: np.ndarray | None = None) -> None:
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
    total_pnl_usdt = bitmex_equity_btc * bitmex_price + gate_equity_usdt
    total_fee_usdt = float(bitmex_final["fee"]) * bitmex_price + float(gate_final["fee"])
    avg_hedge_delay_ns = metrics[6] / metrics[7] if metrics[7] > 0 else 0.0

    summary = {
        "date": DATE,
        "strategy": "inventory_based_bitmex_maker_gate_panic_hedge",
        "bitmex_symbol": BITMEX_SYMBOL,
        "gate_symbol": GATE_SYMBOL,
        "total_pnl_usdt": total_pnl_usdt,
        "pnl_status_zh": "赚钱" if total_pnl_usdt > 0 else "亏钱" if total_pnl_usdt < 0 else "不赚不亏",
        "base_bid_spread_ratio": BASE_BID_SPREAD_RATIO,
        "base_ask_spread_ratio": BASE_ASK_SPREAD_RATIO,
        "inventory_skew_bps": INVENTORY_SKEW_BPS,
        "max_reduce_quote_cross_bps": MAX_REDUCE_QUOTE_CROSS_BPS,
        "soft_inventory_limit_base": SOFT_INVENTORY_LIMIT_BASE,
        "hard_inventory_limit_base": HARD_INVENTORY_LIMIT_BASE,
        "max_inventory_hold_ms": MAX_INVENTORY_HOLD_NS / 1_000_000.0,
        "max_position_base": MAX_POSITION_BASE,
        "max_gross_position_limit_base": MAX_GROSS_POSITION_BASE,
        "target_bitmex_buy_fills": TARGET_BITMEX_BUY_FILLS,
        "target_bitmex_sell_fills": TARGET_BITMEX_SELL_FILLS,
        "max_fill_count_imbalance": MAX_FILL_COUNT_IMBALANCE,
        "bitmex_maker_fills": int(metrics[4]),
        "bitmex_buy_fills": int(metrics[11]),
        "bitmex_sell_fills": int(metrics[12]),
        "gate_hedge_fills": int(metrics[5]),
        "inventory_hedge_triggers": int(metrics[20]),
        "gate_hedge_send_events": int(metrics[21]),
        "hold_hedge_triggers": int(metrics[25]),
        "hard_hedge_triggers": int(metrics[26]),
        "total_filled_base": float(metrics[0]),
        "max_bitmex_position_base": float(metrics[1]),
        "max_gate_position_base": float(metrics[2]),
        "max_net_exposure_base": float(metrics[3]),
        "max_gross_position_base": float(metrics[10]),
        "max_inventory_hold_ms_observed": float(metrics[22] / 1_000_000.0),
        "soft_breach_samples": int(metrics[23]),
        "hard_breach_samples": int(metrics[24]),
        "avg_hedge_delay_ms": float(avg_hedge_delay_ns / 1_000_000.0),
        "force_close_pnl_usdt": float(metrics[9]),
        "total_fee_usdt": total_fee_usdt,
        "bitmex_final_position_contracts": float(bitmex_final["position"]),
        "gate_final_position_contracts": float(gate_final["position"]),
        "bitmex_final_position_base": bitmex_base,
        "gate_final_position_base": gate_base,
        "net_base_position": bitmex_base + gate_base,
        "records": int(len(bitmex)),
        "start_timestamp_ns": int(bitmex[0]["timestamp"]),
        "end_timestamp_ns": int(bitmex_final["timestamp"]),
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
            "========== 库存策略回测结果 ==========",
            f"结论: {summary['pnl_status_zh']} ({signed_money(summary['total_pnl_usdt'])})",
            f"日期: {summary['date']}",
            f"BitMEX maker 成交: {summary['bitmex_maker_fills']} "
            f"(buy={summary['bitmex_buy_fills']}, sell={summary['bitmex_sell_fills']})",
            f"Gate panic hedge 成交: {summary['gate_hedge_fills']}",
            f"库存 hedge 触发: {summary['inventory_hedge_triggers']} "
            f"(hard={summary['hard_hedge_triggers']}, hold={summary['hold_hedge_triggers']})",
            f"最大净敞口: {signed_base(summary['max_net_exposure_base'])}",
            f"最大持仓时间: {summary['max_inventory_hold_ms_observed']:,.2f} ms",
            f"日终强平 PnL: {signed_money(summary['force_close_pnl_usdt'])}",
            "======================================",
            "",
        ]
    )


def render_report(summary: dict) -> str:
    return f"""# BitMEX XBTUSD vs Gate BTCUSDT 库存策略回测

本次回测结果为 **{summary['pnl_status_zh']}**，总 PnL 为 **{signed_money(summary['total_pnl_usdt'])}**。

## 策略

- BitMEX 挂 maker bid/ask。
- 成交后不逐笔 Gate hedge。
- 净库存超过 soft limit 后停止加仓方向，只保留减仓方向 quote。
- 净库存超过 hard limit，或库存持有超过 max hold，才用 Gate IOC 做 panic hedge。

## 参数

- 日期: `{summary['date']}`
- base bid spread: `{summary['base_bid_spread_ratio']}`
- base ask spread: `{summary['base_ask_spread_ratio']}`
- inventory skew: `{summary['inventory_skew_bps']} bps`
- max reduce quote cross: `{summary['max_reduce_quote_cross_bps']} bps`
- soft inventory: `{signed_base(summary['soft_inventory_limit_base'])}`
- hard inventory: `{signed_base(summary['hard_inventory_limit_base'])}`
- max hold: `{summary['max_inventory_hold_ms']} ms`
- max position: `{signed_base(summary['max_position_base'])}`
- max gross position: `{signed_base(summary['max_gross_position_limit_base'])}`
- target BitMEX buy fills: `{summary['target_bitmex_buy_fills']}`
- target BitMEX sell fills: `{summary['target_bitmex_sell_fills']}`
- max fill count imbalance: `{summary['max_fill_count_imbalance']}`

## 结果

- 总 PnL: **{signed_money(summary['total_pnl_usdt'])}**
- 总成交 base: `{summary['total_filled_base']:,.8f} BTC`
- BitMEX maker fills: `{summary['bitmex_maker_fills']}`
- Gate panic hedge fills: `{summary['gate_hedge_fills']}`
- inventory hedge triggers: `{summary['inventory_hedge_triggers']}`
- hard hedge triggers: `{summary['hard_hedge_triggers']}`
- hold hedge triggers: `{summary['hold_hedge_triggers']}`
- 最大净敞口: `{signed_base(summary['max_net_exposure_base'])}`
- 最大 gross position: `{signed_base(summary['max_gross_position_base'])}`
- 最大库存持有时间: `{summary['max_inventory_hold_ms_observed']:,.2f} ms`
- 日终强平 PnL: `{signed_money(summary['force_close_pnl_usdt'])}`
- 总手续费: `{signed_money(summary['total_fee_usdt'])}`
"""


def day_end_ns(yyyymmdd: str) -> int:
    day = dt.datetime.strptime(yyyymmdd, "%Y%m%d").replace(tzinfo=dt.timezone.utc)
    return int((day + dt.timedelta(days=1)).timestamp() * 1_000_000_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BitMEX/Gate inventory backtest.")
    parser.add_argument("--date", default=DATE, help="UTC trading date in YYYYMMDD format")
    return parser.parse_args()


def main() -> None:
    global DATE, END_CLOSE_TS_NS
    args = parse_args()
    DATE = args.date
    END_CLOSE_TS_NS = day_end_ns(DATE)

    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    bitmex_npz = npz_path("bitmex", BITMEX_SYMBOL, DATE)
    gate_npz = npz_path("gate-io-futures", GATE_SYMBOL, DATE)
    if not bitmex_npz.exists():
        bitmex_npz = convert_pair("bitmex", BITMEX_SYMBOL, DATE)
    if not gate_npz.exists():
        gate_npz = convert_pair("gate-io-futures", GATE_SYMBOL, DATE)

    result = run_backtest(bitmex_npz, gate_npz)
    print(f"result={result}")


if __name__ == "__main__":
    main()
