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
