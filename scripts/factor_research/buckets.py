from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x, y = finite_pair(x, y)
    if len(x) < 3:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 0 or y_std <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def ordinal_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    x, y = finite_pair(x, y)
    if len(x) < 3:
        return float("nan")
    return pearson_corr(ordinal_rank(x), ordinal_rank(y))


def factor_ic_rows(
    factors: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    factor_names: tuple[str, ...],
    label_names: tuple[str, ...],
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for factor_name in factor_names:
        x = factors[factor_name]
        for label_name in label_names:
            y = labels[label_name]
            valid = np.isfinite(x) & np.isfinite(y)
            rows.append(
                {
                    "factor": factor_name,
                    "label": label_name,
                    "samples": int(valid.sum()),
                    "pearson_ic": pearson_corr(x, y),
                    "rank_ic": rank_corr(x, y),
                }
            )
    return rows


def bucket_rows(
    factors: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    factor_names: tuple[str, ...],
    target_label: str,
    extra_labels: tuple[str, ...],
    buckets: int,
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    y = labels[target_label]
    extra = {name: labels[name] for name in extra_labels}
    for factor_name in factor_names:
        x = factors[factor_name]
        valid = np.isfinite(x) & np.isfinite(y)
        xv = x[valid]
        if len(xv) < buckets * 10 or np.nanstd(xv) <= 0:
            continue
        edges = np.nanquantile(xv, np.linspace(0.0, 1.0, buckets + 1))
        edges = np.unique(edges)
        if len(edges) <= 2:
            continue
        bucket_id = np.searchsorted(edges[1:-1], x, side="right")
        for b in range(len(edges) - 1):
            mask = valid & (bucket_id == b)
            count = int(mask.sum())
            if count == 0:
                continue
            row: dict[str, float | str | int] = {
                "factor": factor_name,
                "bucket": b + 1,
                "samples": count,
                "factor_min": float(np.nanmin(x[mask])),
                "factor_max": float(np.nanmax(x[mask])),
                "factor_mean": float(np.nanmean(x[mask])),
                target_label: float(np.nanmean(y[mask])),
                "positive_label_frac": float(np.nanmean(y[mask] > 0)),
            }
            for label_name, values in extra.items():
                extra_values = values[mask]
                extra_values = extra_values[np.isfinite(extra_values)]
                row[f"{label_name}_samples"] = int(len(extra_values))
                row[label_name] = float(np.nanmean(extra_values)) if len(extra_values) > 0 else float("nan")
            rows.append(row)
    return rows


def maker_fill_edge_bucket_rows(
    factors: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    factor_names: tuple[str, ...],
    horizon_ms: int,
    buckets: int,
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    side_specs = (
        (
            "bid",
            "bid_lifecycle_fill_prob",
            f"bid_lifecycle_edge_when_filled_{horizon_ms}ms_bps",
            f"bid_lifecycle_expected_edge_{horizon_ms}ms_bps",
            "bid_post_only_reject_prob",
            "bid_entry_at_bbo_prob",
            "bid_time_to_fill_ms",
        ),
        (
            "ask",
            "ask_lifecycle_fill_prob",
            f"ask_lifecycle_edge_when_filled_{horizon_ms}ms_bps",
            f"ask_lifecycle_expected_edge_{horizon_ms}ms_bps",
            "ask_post_only_reject_prob",
            "ask_entry_at_bbo_prob",
            "ask_time_to_fill_ms",
        ),
    )
    for factor_name in factor_names:
        x = factors[factor_name]
        factor_valid = np.isfinite(x)
        xv = x[factor_valid]
        if len(xv) < buckets * 10 or np.nanstd(xv) <= 0:
            continue
        edges = np.nanquantile(xv, np.linspace(0.0, 1.0, buckets + 1))
        edges = np.unique(edges)
        if len(edges) <= 2:
            continue
        bucket_id = np.searchsorted(edges[1:-1], x, side="right")
        for b in range(len(edges) - 1):
            base_mask = factor_valid & (bucket_id == b)
            if int(base_mask.sum()) == 0:
                continue
            for (
                side,
                fill_label,
                edge_label,
                expected_label,
                reject_label,
                at_bbo_label,
                time_label,
            ) in side_specs:
                fill = labels[fill_label]
                edge = labels[edge_label]
                expected = labels[expected_label]
                reject = labels[reject_label]
                at_bbo = labels[at_bbo_label]
                time_to_fill = labels[time_label]
                valid = base_mask & np.isfinite(fill) & np.isfinite(expected)
                samples = int(valid.sum())
                if samples == 0:
                    continue
                filled = valid & (fill > 0)
                fill_samples = int(filled.sum())
                edge_values = edge[filled]
                edge_values = edge_values[np.isfinite(edge_values)]
                time_values = time_to_fill[filled]
                time_values = time_values[np.isfinite(time_values)]
                row: dict[str, float | str | int] = {
                    "factor": factor_name,
                    "bucket": b + 1,
                    "side": side,
                    "samples": samples,
                    "factor_min": float(np.nanmin(x[base_mask])),
                    "factor_max": float(np.nanmax(x[base_mask])),
                    "factor_mean": float(np.nanmean(x[base_mask])),
                    "fill_samples": fill_samples,
                    "fill_prob": float(np.nanmean(fill[valid])),
                    "edge_if_filled_bps": float(np.nanmean(edge_values)) if len(edge_values) > 0 else float("nan"),
                    "expected_edge_bps": float(np.nanmean(expected[valid])),
                    "positive_fill_frac": float(np.nanmean(edge_values > 0)) if len(edge_values) > 0 else float("nan"),
                    "mean_time_to_fill_ms": float(np.nanmean(time_values)) if len(time_values) > 0 else float("nan"),
                    "post_only_reject_frac": float(np.nanmean(reject[valid])),
                    "entry_at_bbo_frac": float(np.nanmean(at_bbo[valid])),
                }
                rows.append(row)
    return rows


def _bucket_ids(values: np.ndarray, buckets: int) -> tuple[np.ndarray, int]:
    valid = np.isfinite(values)
    out = np.full(len(values), -1, dtype=np.int64)
    finite_values = values[valid]
    if len(finite_values) < buckets * 10 or np.nanstd(finite_values) <= 0:
        return out, 0
    edges = np.nanquantile(finite_values, np.linspace(0.0, 1.0, buckets + 1))
    edges = np.unique(edges)
    if len(edges) <= 2:
        return out, 0
    out[valid] = np.searchsorted(edges[1:-1], values[valid], side="right")
    return out, len(edges) - 1


def maker_fill_combo_rule_rows(
    factors: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    factor_names: tuple[str, ...],
    horizon_ms: int,
    buckets: int,
    combo_size: int = 2,
) -> list[dict[str, float | str | int]]:
    if combo_size != 2:
        raise ValueError("Only 2-factor combo rules are supported")

    rows: list[dict[str, float | str | int]] = []
    bucket_cache: dict[str, tuple[np.ndarray, int]] = {
        factor_name: _bucket_ids(factors[factor_name], buckets) for factor_name in factor_names
    }
    side_specs = (
        (
            "bid",
            "bid_lifecycle_fill_prob",
            f"bid_lifecycle_edge_when_filled_{horizon_ms}ms_bps",
            f"bid_lifecycle_expected_edge_{horizon_ms}ms_bps",
            "bid_post_only_reject_prob",
            "bid_entry_at_bbo_prob",
            "bid_time_to_fill_ms",
        ),
        (
            "ask",
            "ask_lifecycle_fill_prob",
            f"ask_lifecycle_edge_when_filled_{horizon_ms}ms_bps",
            f"ask_lifecycle_expected_edge_{horizon_ms}ms_bps",
            "ask_post_only_reject_prob",
            "ask_entry_at_bbo_prob",
            "ask_time_to_fill_ms",
        ),
    )

    for i, factor1 in enumerate(factor_names):
        x1 = factors[factor1]
        b1, b1_count = bucket_cache[factor1]
        if b1_count <= 0:
            continue
        for factor2 in factor_names[i + 1 :]:
            x2 = factors[factor2]
            b2, b2_count = bucket_cache[factor2]
            if b2_count <= 0:
                continue
            pair_valid = (b1 >= 0) & (b2 >= 0)
            if int(pair_valid.sum()) == 0:
                continue
            combo_id_all = b1 * b2_count + b2

            for (
                side,
                fill_label,
                edge_label,
                expected_label,
                reject_label,
                at_bbo_label,
                time_label,
            ) in side_specs:
                fill = labels[fill_label]
                edge = labels[edge_label]
                expected = labels[expected_label]
                reject = labels[reject_label]
                at_bbo = labels[at_bbo_label]
                time_to_fill = labels[time_label]

                valid = pair_valid & np.isfinite(fill) & np.isfinite(expected)
                idx = np.flatnonzero(valid)
                if len(idx) == 0:
                    continue
                combo_values = combo_id_all[idx]
                order = np.argsort(combo_values, kind="mergesort")
                sorted_idx = idx[order]
                sorted_combo = combo_values[order]
                starts = np.r_[0, np.flatnonzero(sorted_combo[1:] != sorted_combo[:-1]) + 1]
                ends = np.r_[starts[1:], len(sorted_combo)]

                for start, end in zip(starts, ends):
                    group_idx = sorted_idx[start:end]
                    samples = int(len(group_idx))
                    combo_id = int(sorted_combo[start])
                    bucket1 = combo_id // b2_count
                    bucket2 = combo_id % b2_count
                    filled_idx = group_idx[fill[group_idx] > 0]
                    edge_values = edge[filled_idx]
                    edge_values = edge_values[np.isfinite(edge_values)]
                    time_values = time_to_fill[filled_idx]
                    time_values = time_values[np.isfinite(time_values)]
                    rows.append(
                        {
                            "rule_type": "factor_pair",
                            "factor_count": 2,
                            "factor": f"{factor1}&{factor2}",
                            "factor1": factor1,
                            "bucket1": bucket1 + 1,
                            "factor1_min": float(np.nanmin(x1[group_idx])),
                            "factor1_max": float(np.nanmax(x1[group_idx])),
                            "factor1_mean": float(np.nanmean(x1[group_idx])),
                            "factor2": factor2,
                            "bucket2": bucket2 + 1,
                            "factor2_min": float(np.nanmin(x2[group_idx])),
                            "factor2_max": float(np.nanmax(x2[group_idx])),
                            "factor2_mean": float(np.nanmean(x2[group_idx])),
                            "side": side,
                            "samples": samples,
                            "fill_samples": int(len(filled_idx)),
                            "fill_prob": float(np.nanmean(fill[group_idx])),
                            "edge_if_filled_bps": float(np.nanmean(edge_values))
                            if len(edge_values) > 0
                            else float("nan"),
                            "expected_edge_bps": float(np.nanmean(expected[group_idx])),
                            "positive_fill_frac": float(np.nanmean(edge_values > 0))
                            if len(edge_values) > 0
                            else float("nan"),
                            "mean_time_to_fill_ms": float(np.nanmean(time_values))
                            if len(time_values) > 0
                            else float("nan"),
                            "post_only_reject_frac": float(np.nanmean(reject[group_idx])),
                            "entry_at_bbo_frac": float(np.nanmean(at_bbo[group_idx])),
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
