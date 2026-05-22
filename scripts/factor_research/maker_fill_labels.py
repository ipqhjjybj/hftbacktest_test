from __future__ import annotations

import numpy as np
from numba import njit


@njit
def _search_left(values: np.ndarray, target: float) -> int:
    lo = 0
    hi = len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit
def _search_right(values: np.ndarray, target: float) -> int:
    lo = 0
    hi = len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit
def _fill_maker_side(
    ts: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    bid_qty: np.ndarray,
    ask_qty: np.ndarray,
    mid: np.ndarray,
    active_qty: np.ndarray,
    quote_price: np.ndarray,
    side: int,
    horizons_ms: np.ndarray,
    maker_rebate_bps: float,
    order_qty: float,
    queue_ahead_multiplier: float,
    entry_latency_ns: int,
    ttl_ns: int,
    fill_prob: np.ndarray,
    post_only_reject: np.ndarray,
    at_bbo: np.ndarray,
    time_to_fill_ms: np.ndarray,
    edge_when_filled: np.ndarray,
    expected_edge: np.ndarray,
) -> None:
    n = len(ts)
    cumsum = np.zeros(n + 1, dtype=np.float64)
    for i in range(n):
        cumsum[i + 1] = cumsum[i] + active_qty[i]

    max_horizon_ns = 0
    for h in horizons_ms:
        h_ns = int(h) * 1_000_000
        if h_ns > max_horizon_ns:
            max_horizon_ns = h_ns

    for i in range(n):
        entry_ts = ts[i] + entry_latency_ns
        entry_idx = _search_left(ts, entry_ts)
        if entry_idx >= n:
            continue

        tail_idx = _search_left(ts, entry_ts + ttl_ns + max_horizon_ns)
        if tail_idx >= n:
            continue

        for h_pos in range(len(horizons_ms)):
            expected_edge[h_pos, i] = 0.0

        q = quote_price[i]
        entry_bid = bid[entry_idx]
        entry_ask = ask[entry_idx]
        if not np.isfinite(q) or not np.isfinite(entry_bid) or not np.isfinite(entry_ask):
            continue

        post_ok = False
        eligible = False
        queue_ahead = 0.0
        if side > 0:
            post_ok = q < entry_ask
            post_only_reject[i] = 0.0 if post_ok else 1.0
            if post_ok:
                if q == entry_bid:
                    at_bbo[i] = 1.0
                    queue_ahead = max(0.0, bid_qty[entry_idx] * queue_ahead_multiplier)
                    eligible = True
                elif q > entry_bid:
                    at_bbo[i] = 1.0
                    queue_ahead = 0.0
                    eligible = True
                else:
                    at_bbo[i] = 0.0
        else:
            post_ok = q > entry_bid
            post_only_reject[i] = 0.0 if post_ok else 1.0
            if post_ok:
                if q == entry_ask:
                    at_bbo[i] = 1.0
                    queue_ahead = max(0.0, ask_qty[entry_idx] * queue_ahead_multiplier)
                    eligible = True
                elif q < entry_ask:
                    at_bbo[i] = 1.0
                    queue_ahead = 0.0
                    eligible = True
                else:
                    at_bbo[i] = 0.0

        if not post_ok:
            fill_prob[i] = 0.0
            continue
        if not eligible:
            fill_prob[i] = 0.0
            continue

        threshold = queue_ahead + order_qty
        if threshold <= 0.0:
            fill_prob[i] = 0.0
            continue

        end_idx = _search_right(ts, entry_ts + ttl_ns)
        if end_idx <= entry_idx:
            fill_prob[i] = 0.0
            continue

        target_qty = cumsum[entry_idx] + threshold
        fill_cumsum_idx = _search_left(cumsum, target_qty)
        if fill_cumsum_idx <= entry_idx or fill_cumsum_idx > end_idx or fill_cumsum_idx > n:
            fill_prob[i] = 0.0
            continue

        fill_idx = fill_cumsum_idx - 1
        if fill_idx < entry_idx:
            fill_prob[i] = 0.0
            continue

        fill_prob[i] = 1.0
        time_to_fill_ms[i] = (ts[fill_idx] - entry_ts) / 1_000_000.0
        denom = mid[entry_idx]
        if denom <= 0.0 or not np.isfinite(denom):
            continue

        for h_pos in range(len(horizons_ms)):
            future_idx = _search_left(ts, ts[fill_idx] + int(horizons_ms[h_pos]) * 1_000_000)
            if future_idx >= n:
                continue
            future_mid = mid[future_idx]
            if not np.isfinite(future_mid):
                continue
            if side > 0:
                edge = (future_mid - q) / denom * 10_000.0 + maker_rebate_bps
            else:
                edge = (q - future_mid) / denom * 10_000.0 + maker_rebate_bps
            edge_when_filled[h_pos, i] = edge
            expected_edge[h_pos, i] = edge


def build_maker_fill_labels(
    data: dict[str, np.ndarray],
    horizons_ms: tuple[int, ...],
    maker_fee_rate: float,
    hypothetical_order_qty: float,
    queue_ahead_multiplier: float,
    entry_latency_ms: float,
    ttl_ms: float,
) -> dict[str, np.ndarray]:
    ts = data["ts"].astype(np.int64)
    n = len(ts)
    horizons = np.array(horizons_ms, dtype=np.int64)
    maker_rebate_bps = -maker_fee_rate * 10_000.0
    entry_latency_ns = int(entry_latency_ms * 1_000_000)
    ttl_ns = int(ttl_ms * 1_000_000)

    labels: dict[str, np.ndarray] = {}
    for prefix in ("bid", "ask"):
        labels[f"{prefix}_lifecycle_fill_prob"] = np.full(n, np.nan, dtype=np.float64)
        labels[f"{prefix}_post_only_reject_prob"] = np.full(n, np.nan, dtype=np.float64)
        labels[f"{prefix}_entry_at_bbo_prob"] = np.full(n, np.nan, dtype=np.float64)
        labels[f"{prefix}_time_to_fill_ms"] = np.full(n, np.nan, dtype=np.float64)
        for horizon_ms in horizons_ms:
            labels[f"{prefix}_lifecycle_edge_when_filled_{horizon_ms}ms_bps"] = np.full(
                n, np.nan, dtype=np.float64
            )
            labels[f"{prefix}_lifecycle_expected_edge_{horizon_ms}ms_bps"] = np.full(
                n, np.nan, dtype=np.float64
            )

    bid_edges = np.full((len(horizons_ms), n), np.nan, dtype=np.float64)
    bid_expected = np.full((len(horizons_ms), n), np.nan, dtype=np.float64)
    ask_edges = np.full((len(horizons_ms), n), np.nan, dtype=np.float64)
    ask_expected = np.full((len(horizons_ms), n), np.nan, dtype=np.float64)

    _fill_maker_side(
        ts,
        data["bid"],
        data["ask"],
        data["bid_qty"],
        data["ask_qty"],
        data["mid"],
        data["sell_qty"],
        data["bid"],
        1,
        horizons,
        maker_rebate_bps,
        hypothetical_order_qty,
        queue_ahead_multiplier,
        entry_latency_ns,
        ttl_ns,
        labels["bid_lifecycle_fill_prob"],
        labels["bid_post_only_reject_prob"],
        labels["bid_entry_at_bbo_prob"],
        labels["bid_time_to_fill_ms"],
        bid_edges,
        bid_expected,
    )
    _fill_maker_side(
        ts,
        data["bid"],
        data["ask"],
        data["bid_qty"],
        data["ask_qty"],
        data["mid"],
        data["buy_qty"],
        data["ask"],
        -1,
        horizons,
        maker_rebate_bps,
        hypothetical_order_qty,
        queue_ahead_multiplier,
        entry_latency_ns,
        ttl_ns,
        labels["ask_lifecycle_fill_prob"],
        labels["ask_post_only_reject_prob"],
        labels["ask_entry_at_bbo_prob"],
        labels["ask_time_to_fill_ms"],
        ask_edges,
        ask_expected,
    )
    for h_pos, horizon_ms in enumerate(horizons_ms):
        labels[f"bid_lifecycle_edge_when_filled_{horizon_ms}ms_bps"] = bid_edges[h_pos]
        labels[f"bid_lifecycle_expected_edge_{horizon_ms}ms_bps"] = bid_expected[h_pos]
        labels[f"ask_lifecycle_edge_when_filled_{horizon_ms}ms_bps"] = ask_edges[h_pos]
        labels[f"ask_lifecycle_expected_edge_{horizon_ms}ms_bps"] = ask_expected[h_pos]
    return labels
