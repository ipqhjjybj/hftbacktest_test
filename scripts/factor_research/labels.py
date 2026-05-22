from __future__ import annotations

import numpy as np

from .factors import ratio_minus_one_bps, values_at_or_after


def future_window_sum(ts: np.ndarray, values: np.ndarray, horizon_ms: int) -> np.ndarray:
    target_ts = ts + horizon_ms * 1_000_000
    end_idx = np.searchsorted(ts, target_ts, side="right")
    start_idx = np.arange(len(ts)) + 1
    cumsum = np.concatenate(([0.0], np.cumsum(values)))
    return cumsum[end_idx] - cumsum[start_idx]


def build_labels(
    data: dict[str, np.ndarray],
    horizons_ms: tuple[int, ...],
    maker_fee_rate: float,
    hypothetical_order_qty: float,
    queue_ahead_multiplier: float,
) -> dict[str, np.ndarray]:
    ts = data["ts"].astype(np.int64)
    bid = data["bid"]
    ask = data["ask"]
    mid = data["mid"]
    bid_qty = data["bid_qty"]
    ask_qty = data["ask_qty"]
    rebate_bps = -maker_fee_rate * 10_000.0
    labels: dict[str, np.ndarray] = {}
    for horizon_ms in horizons_ms:
        future_mid = values_at_or_after(ts, mid, ts + horizon_ms * 1_000_000)
        ret = ratio_minus_one_bps(future_mid, mid)
        labels[f"future_ret_{horizon_ms}ms_bps"] = ret
        labels[f"bid_adverse_{horizon_ms}ms_bps"] = -ret
        labels[f"ask_adverse_{horizon_ms}ms_bps"] = ret
        labels[f"bid_maker_edge_{horizon_ms}ms_bps"] = ((future_mid - bid) / mid) * 10_000.0 + rebate_bps
        labels[f"ask_maker_edge_{horizon_ms}ms_bps"] = ((ask - future_mid) / mid) * 10_000.0 + rebate_bps

        future_sell_qty = future_window_sum(ts, data["sell_qty"], horizon_ms)
        future_buy_qty = future_window_sum(ts, data["buy_qty"], horizon_ms)
        bid_fill_threshold = bid_qty * queue_ahead_multiplier + hypothetical_order_qty
        ask_fill_threshold = ask_qty * queue_ahead_multiplier + hypothetical_order_qty
        bid_fill = future_sell_qty >= bid_fill_threshold
        ask_fill = future_buy_qty >= ask_fill_threshold
        valid_horizon = np.isfinite(future_mid)

        bid_fill_prob = np.where(valid_horizon, bid_fill.astype(np.float64), np.nan)
        ask_fill_prob = np.where(valid_horizon, ask_fill.astype(np.float64), np.nan)
        labels[f"bid_fill_prob_{horizon_ms}ms"] = bid_fill_prob
        labels[f"ask_fill_prob_{horizon_ms}ms"] = ask_fill_prob
        labels[f"bid_fill_queue_ratio_{horizon_ms}ms"] = np.divide(
            future_sell_qty,
            bid_fill_threshold,
            out=np.full_like(future_sell_qty, np.nan, dtype=np.float64),
            where=bid_fill_threshold > 0,
        )
        labels[f"ask_fill_queue_ratio_{horizon_ms}ms"] = np.divide(
            future_buy_qty,
            ask_fill_threshold,
            out=np.full_like(future_buy_qty, np.nan, dtype=np.float64),
            where=ask_fill_threshold > 0,
        )
        bid_edge = labels[f"bid_maker_edge_{horizon_ms}ms_bps"]
        ask_edge = labels[f"ask_maker_edge_{horizon_ms}ms_bps"]
        labels[f"bid_edge_when_filled_{horizon_ms}ms_bps"] = np.where(bid_fill & valid_horizon, bid_edge, np.nan)
        labels[f"ask_edge_when_filled_{horizon_ms}ms_bps"] = np.where(ask_fill & valid_horizon, ask_edge, np.nan)
    return labels
