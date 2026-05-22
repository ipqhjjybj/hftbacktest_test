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


def build_factor_frame(raw: np.ndarray, sample_interval_ms: int, flow_decay: float) -> dict[str, np.ndarray]:
    data = {name: raw[:, i].astype(np.float64, copy=False) for i, name in enumerate(BASE_COLUMNS)}
    bid = data["bid"]
    ask = data["ask"]
    bid_qty = data["bid_qty"]
    ask_qty = data["ask_qty"]
    mid = (bid + ask) / 2.0
    total_qty = bid_qty + ask_qty

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

    ts = data["ts"].astype(np.int64)
    for window_ms in (100, 250, 1000):
        past_mid = values_at_or_before(ts, mid, ts - window_ms * 1_000_000)
        data[f"momentum_{window_ms}ms_bps"] = ratio_minus_one_bps(mid, past_mid)

    vol_window_ms = 1000
    lookback_steps = max(2, int(vol_window_ms / sample_interval_ms))
    one_step_ret = np.empty_like(mid, dtype=np.float64)
    one_step_ret[:] = np.nan
    one_step_ret[1:] = ratio_minus_one_bps(mid[1:], mid[:-1])
    vol = np.full_like(mid, np.nan, dtype=np.float64)
    abs_ret = np.abs(np.nan_to_num(one_step_ret, nan=0.0))
    cumsum = np.concatenate(([0.0], np.cumsum(abs_ret)))
    for i in range(len(mid)):
        start = max(0, i + 1 - lookback_steps)
        count = i + 1 - start
        if count > 1:
            vol[i] = (cumsum[i + 1] - cumsum[start]) / count
    data["vol_1000ms_bps"] = vol
    return data
