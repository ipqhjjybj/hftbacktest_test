from __future__ import annotations

import numpy as np


BASE_COLUMNS = (
    "ts",
    "bid",
    "ask",
    "bid_qty",
    "ask_qty",
    "buy_qty",
    "sell_qty",
    "buy_count",
    "sell_count",
    "bid_qty_l2",
    "ask_qty_l2",
    "bid_qty_l3",
    "ask_qty_l3",
    "bid_qty_l4",
    "ask_qty_l4",
    "bid_qty_l5",
    "ask_qty_l5",
)

FACTOR_NAMES = (
    "spread_bps",
    "queue_imbalance",
    "microprice_bps",
    "trade_flow_imbalance",
    "trade_flow_ewm_imbalance",
    "momentum_100ms_bps",
    "momentum_250ms_bps",
    "momentum_1000ms_bps",
    "vol_1000ms_bps",
    "depth_imbalance_3",
    "depth_imbalance_5",
    "weighted_depth_imbalance_5",
    "bid_depth_slope_5",
    "ask_depth_slope_5",
    "top_bid_qty_change",
    "top_ask_qty_change",
    "ofi",
    "ofi_1000ms",
    "trade_qty_1000ms",
    "trade_count_1000ms",
    "momentum_3000ms_bps",
    "vol_250ms_bps",
)


def ratio_minus_one_bps(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full_like(numerator, np.nan, dtype=np.float64)
    mask = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
    out[mask] = (numerator[mask] / denominator[mask] - 1.0) * 10_000.0
    return out


def values_at_or_before(ts: np.ndarray, values: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(ts, target_ts, side="right") - 1
    out = np.full(ts.shape, np.nan, dtype=np.float64)
    valid = idx >= 0
    out[valid] = values[idx[valid]]
    return out


def values_at_or_after(ts: np.ndarray, values: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(ts, target_ts, side="left")
    out = np.full(ts.shape, np.nan, dtype=np.float64)
    valid = idx < len(ts)
    out[valid] = values[idx[valid]]
    return out


def ewm_imbalance(buy_qty: np.ndarray, sell_qty: np.ndarray, decay: float) -> np.ndarray:
    buy = 0.0
    sell = 0.0
    out = np.zeros(len(buy_qty), dtype=np.float64)
    for i in range(len(buy_qty)):
        buy = buy * decay + buy_qty[i]
        sell = sell * decay + sell_qty[i]
        total = buy + sell
        if total > 0:
            out[i] = (buy - sell) / total
    return out


def rolling_sum(values: np.ndarray, window_steps: int) -> np.ndarray:
    clean = np.nan_to_num(values.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    out = np.empty(len(clean), dtype=np.float64)
    cumsum = np.concatenate(([0.0], np.cumsum(clean)))
    window_steps = max(1, int(window_steps))
    for i in range(len(clean)):
        start = max(0, i + 1 - window_steps)
        out[i] = cumsum[i + 1] - cumsum[start]
    return out


def rolling_mean_abs(values: np.ndarray, window_steps: int) -> np.ndarray:
    abs_values = np.abs(np.nan_to_num(values.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0))
    out = np.full(len(abs_values), np.nan, dtype=np.float64)
    cumsum = np.concatenate(([0.0], np.cumsum(abs_values)))
    window_steps = max(2, int(window_steps))
    for i in range(len(abs_values)):
        start = max(0, i + 1 - window_steps)
        count = i + 1 - start
        if count > 1:
            out[i] = (cumsum[i + 1] - cumsum[start]) / count
    return out


def top_of_book_ofi(bid: np.ndarray, ask: np.ndarray, bid_qty: np.ndarray, ask_qty: np.ndarray) -> np.ndarray:
    out = np.zeros(len(bid), dtype=np.float64)
    for i in range(1, len(bid)):
        bid_part = 0.0
        ask_part = 0.0
        if bid[i] > bid[i - 1]:
            bid_part = bid_qty[i]
        elif bid[i] == bid[i - 1]:
            bid_part = bid_qty[i] - bid_qty[i - 1]
        else:
            bid_part = -bid_qty[i - 1]

        if ask[i] < ask[i - 1]:
            ask_part = -ask_qty[i]
        elif ask[i] == ask[i - 1]:
            ask_part = -(ask_qty[i] - ask_qty[i - 1])
        else:
            ask_part = ask_qty[i - 1]
        out[i] = bid_part + ask_part
    return out


def build_factor_frame(raw: np.ndarray, sample_interval_ms: int, flow_decay: float) -> dict[str, np.ndarray]:
    data = {name: raw[:, i].astype(np.float64, copy=False) for i, name in enumerate(BASE_COLUMNS)}
    bid = data["bid"]
    ask = data["ask"]
    bid_qty = data["bid_qty"]
    ask_qty = data["ask_qty"]
    mid = (bid + ask) / 2.0
    total_qty = bid_qty + ask_qty
    bid_levels = np.column_stack(
        (
            bid_qty,
            data["bid_qty_l2"],
            data["bid_qty_l3"],
            data["bid_qty_l4"],
            data["bid_qty_l5"],
        )
    )
    ask_levels = np.column_stack(
        (
            ask_qty,
            data["ask_qty_l2"],
            data["ask_qty_l3"],
            data["ask_qty_l4"],
            data["ask_qty_l5"],
        )
    )

    data["mid"] = mid
    data["spread_bps"] = ratio_minus_one_bps(ask, bid)
    data["queue_imbalance"] = np.divide(
        bid_qty - ask_qty,
        total_qty,
        out=np.zeros_like(total_qty, dtype=np.float64),
        where=total_qty > 0,
    )
    microprice = np.divide(
        ask * bid_qty + bid * ask_qty,
        total_qty,
        out=np.full_like(mid, np.nan, dtype=np.float64),
        where=total_qty > 0,
    )
    data["microprice_bps"] = ratio_minus_one_bps(microprice, mid)

    trade_total = data["buy_qty"] + data["sell_qty"]
    data["trade_flow_imbalance"] = np.divide(
        data["buy_qty"] - data["sell_qty"],
        trade_total,
        out=np.zeros_like(trade_total, dtype=np.float64),
        where=trade_total > 0,
    )
    data["trade_flow_ewm_imbalance"] = ewm_imbalance(data["buy_qty"], data["sell_qty"], flow_decay)

    bid_depth_3 = np.sum(bid_levels[:, :3], axis=1)
    ask_depth_3 = np.sum(ask_levels[:, :3], axis=1)
    depth_3 = bid_depth_3 + ask_depth_3
    data["depth_imbalance_3"] = np.divide(
        bid_depth_3 - ask_depth_3,
        depth_3,
        out=np.zeros_like(depth_3, dtype=np.float64),
        where=depth_3 > 0,
    )
    bid_depth_5 = np.sum(bid_levels, axis=1)
    ask_depth_5 = np.sum(ask_levels, axis=1)
    depth_5 = bid_depth_5 + ask_depth_5
    data["depth_imbalance_5"] = np.divide(
        bid_depth_5 - ask_depth_5,
        depth_5,
        out=np.zeros_like(depth_5, dtype=np.float64),
        where=depth_5 > 0,
    )
    weights = np.array([1.0, 0.5, 1.0 / 3.0, 0.25, 0.2], dtype=np.float64)
    bid_weighted = np.dot(bid_levels, weights)
    ask_weighted = np.dot(ask_levels, weights)
    weighted_total = bid_weighted + ask_weighted
    data["weighted_depth_imbalance_5"] = np.divide(
        bid_weighted - ask_weighted,
        weighted_total,
        out=np.zeros_like(weighted_total, dtype=np.float64),
        where=weighted_total > 0,
    )
    data["bid_depth_slope_5"] = np.divide(
        bid_levels[:, 0] - bid_levels[:, 4],
        bid_depth_5,
        out=np.zeros_like(bid_depth_5, dtype=np.float64),
        where=bid_depth_5 > 0,
    )
    data["ask_depth_slope_5"] = np.divide(
        ask_levels[:, 0] - ask_levels[:, 4],
        ask_depth_5,
        out=np.zeros_like(ask_depth_5, dtype=np.float64),
        where=ask_depth_5 > 0,
    )
    data["top_bid_qty_change"] = np.zeros_like(bid_qty, dtype=np.float64)
    data["top_ask_qty_change"] = np.zeros_like(ask_qty, dtype=np.float64)
    data["top_bid_qty_change"][1:] = bid_qty[1:] - bid_qty[:-1]
    data["top_ask_qty_change"][1:] = ask_qty[1:] - ask_qty[:-1]

    data["ofi"] = top_of_book_ofi(bid, ask, bid_qty, ask_qty)
    steps_1000 = max(1, int(1000 / sample_interval_ms))
    data["ofi_1000ms"] = rolling_sum(data["ofi"], steps_1000)
    data["trade_qty_1000ms"] = rolling_sum(data["buy_qty"] + data["sell_qty"], steps_1000)
    data["trade_count_1000ms"] = rolling_sum(data["buy_count"] + data["sell_count"], steps_1000)

    ts = data["ts"].astype(np.int64)
    for window_ms in (100, 250, 1000, 3000):
        past_mid = values_at_or_before(ts, mid, ts - window_ms * 1_000_000)
        data[f"momentum_{window_ms}ms_bps"] = ratio_minus_one_bps(mid, past_mid)

    one_step_ret = np.empty_like(mid, dtype=np.float64)
    one_step_ret[:] = np.nan
    one_step_ret[1:] = ratio_minus_one_bps(mid[1:], mid[:-1])
    for vol_window_ms in (250, 1000):
        lookback_steps = max(2, int(vol_window_ms / sample_interval_ms))
        data[f"vol_{vol_window_ms}ms_bps"] = rolling_mean_abs(one_step_ret, lookback_steps)
    return data
