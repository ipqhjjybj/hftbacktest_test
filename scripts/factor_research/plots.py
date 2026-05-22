from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def setup_matplotlib_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def label_horizon_ms(label: str) -> int:
    middle = label.removeprefix("future_ret_").removesuffix("ms_bps")
    return int(middle)


def rows_to_ic_matrix(
    ic_rows: list[dict[str, object]],
    factor_names: tuple[str, ...],
    value_key: str,
) -> tuple[list[int], np.ndarray]:
    horizons = sorted({label_horizon_ms(str(row["label"])) for row in ic_rows})
    matrix = np.full((len(factor_names), len(horizons)), np.nan, dtype=np.float64)
    factor_idx = {name: i for i, name in enumerate(factor_names)}
    horizon_idx = {horizon: i for i, horizon in enumerate(horizons)}
    for row in ic_rows:
        factor = str(row["factor"])
        if factor not in factor_idx:
            continue
        horizon = label_horizon_ms(str(row["label"]))
        matrix[factor_idx[factor], horizon_idx[horizon]] = float(row[value_key])
    return horizons, matrix


def plot_ic_heatmap(
    ic_rows: list[dict[str, object]],
    factor_names: tuple[str, ...],
    output_path: Path,
    value_key: str = "rank_ic",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons, matrix = rows_to_ic_matrix(ic_rows, factor_names, value_key)
    finite = matrix[np.isfinite(matrix)]
    limit = max(0.01, float(np.nanmax(np.abs(finite))) if finite.size else 0.01)

    fig_height = max(4.8, 0.42 * len(factor_names) + 1.8)
    fig, ax = plt.subplots(figsize=(8.8, fig_height), constrained_layout=True)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_title(f"{value_key} by factor and horizon")
    ax.set_xticks(np.arange(len(horizons)), [f"{h}ms" for h in horizons])
    ax.set_yticks(np.arange(len(factor_names)), factor_names)
    ax.set_xlabel("future mid return horizon")
    ax.set_ylabel("factor")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.9)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def grouped_bucket_rows(bucket_rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in bucket_rows:
        groups.setdefault(str(row["factor"]), []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda item: int(item["bucket"]))
    return groups


def subplot_grid_size(n: int) -> tuple[int, int]:
    cols = 3 if n > 4 else 2
    rows = int(np.ceil(n / cols))
    return rows, cols


def plot_bucket_future_returns(
    bucket_rows: list[dict[str, object]],
    factor_names: tuple[str, ...],
    bucket_label: str,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = grouped_bucket_rows(bucket_rows)
    rows, cols = subplot_grid_size(len(factor_names))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.0), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, factor in zip(axes_arr, factor_names):
        items = groups.get(factor, [])
        x = np.array([int(row["bucket"]) for row in items], dtype=np.float64)
        y = np.array([float(row[bucket_label]) for row in items], dtype=np.float64)
        samples = np.array([int(row["samples"]) for row in items], dtype=np.float64)
        ax.axhline(0, color="#666666", linewidth=0.8)
        if len(x) > 0:
            ax.plot(x, y, marker="o", linewidth=1.4)
            if np.nanmax(samples) > 0:
                size = 12.0 + 36.0 * samples / np.nanmax(samples)
                ax.scatter(x, y, s=size, alpha=0.25)
        ax.set_title(factor)
        ax.set_xlabel("factor bucket, low to high")
        ax.set_ylabel(f"{bucket_label} mean")
        ax.grid(True, alpha=0.25)
    for ax in axes_arr[len(factor_names) :]:
        ax.axis("off")
    fig.suptitle("Future mid return by factor bucket", fontsize=14)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_bucket_maker_edges(
    bucket_rows: list[dict[str, object]],
    factor_names: tuple[str, ...],
    bid_edge_label: str,
    ask_edge_label: str,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = grouped_bucket_rows(bucket_rows)
    rows, cols = subplot_grid_size(len(factor_names))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.0), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, factor in zip(axes_arr, factor_names):
        items = groups.get(factor, [])
        x = np.array([int(row["bucket"]) for row in items], dtype=np.float64)
        bid = np.array([float(row[bid_edge_label]) for row in items], dtype=np.float64)
        ask = np.array([float(row[ask_edge_label]) for row in items], dtype=np.float64)
        ax.axhline(0, color="#666666", linewidth=0.8)
        if len(x) > 0:
            ax.plot(x, bid, marker="o", linewidth=1.4, label="bid maker edge")
            ax.plot(x, ask, marker="s", linewidth=1.4, label="ask maker edge")
        ax.set_title(factor)
        ax.set_xlabel("factor bucket, low to high")
        ax.set_ylabel("mean edge bps")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes_arr[len(factor_names) :]:
        ax.axis("off")
    fig.suptitle("Approx maker edge by factor bucket", fontsize=14)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_bucket_fill_probability(
    bucket_rows: list[dict[str, object]],
    factor_names: tuple[str, ...],
    bid_fill_label: str,
    ask_fill_label: str,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = grouped_bucket_rows(bucket_rows)
    rows, cols = subplot_grid_size(len(factor_names))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.0), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, factor in zip(axes_arr, factor_names):
        items = groups.get(factor, [])
        x = np.array([int(row["bucket"]) for row in items], dtype=np.float64)
        bid = np.array([float(row[bid_fill_label]) for row in items], dtype=np.float64)
        ask = np.array([float(row[ask_fill_label]) for row in items], dtype=np.float64)
        if len(x) > 0:
            ax.plot(x, bid * 100.0, marker="o", linewidth=1.4, label="hypothetical bid fill")
            ax.plot(x, ask * 100.0, marker="s", linewidth=1.4, label="hypothetical ask fill")
        ax.set_title(factor)
        ax.set_xlabel("factor bucket, low to high")
        ax.set_ylabel("fill probability %")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes_arr[len(factor_names) :]:
        ax.axis("off")
    fig.suptitle("Hypothetical BBO fill probability by factor bucket", fontsize=14)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_bucket_fill_edges(
    bucket_rows: list[dict[str, object]],
    factor_names: tuple[str, ...],
    bid_edge_label: str,
    ask_edge_label: str,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = grouped_bucket_rows(bucket_rows)
    rows, cols = subplot_grid_size(len(factor_names))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.0), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, factor in zip(axes_arr, factor_names):
        items = groups.get(factor, [])
        x = np.array([int(row["bucket"]) for row in items], dtype=np.float64)
        bid = np.array([float(row[bid_edge_label]) for row in items], dtype=np.float64)
        ask = np.array([float(row[ask_edge_label]) for row in items], dtype=np.float64)
        ax.axhline(0, color="#666666", linewidth=0.8)
        if len(x) > 0:
            ax.plot(x, bid, marker="o", linewidth=1.4, label="bid edge if filled")
            ax.plot(x, ask, marker="s", linewidth=1.4, label="ask edge if filled")
        ax.set_title(factor)
        ax.set_xlabel("factor bucket, low to high")
        ax.set_ylabel("mean edge bps")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes_arr[len(factor_names) :]:
        ax.axis("off")
    fig.suptitle("Fill-conditioned maker edge by factor bucket", fontsize=14)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_charts(
    output_dir: Path,
    prefix_name: str,
    ic_rows: list[dict[str, object]],
    bucket_rows: list[dict[str, object]],
    factor_names: tuple[str, ...],
    bucket_label: str,
    bid_edge_label: str,
    ask_edge_label: str,
) -> dict[str, Path]:
    setup_matplotlib_cache(output_dir / "mplconfig")
    chart_paths = {
        "rank_ic_heatmap": output_dir / f"{prefix_name}.rank_ic_heatmap.png",
        "bucket_future_ret": output_dir / f"{prefix_name}.bucket_future_ret.png",
        "bucket_maker_edge": output_dir / f"{prefix_name}.bucket_maker_edge.png",
    }
    plot_ic_heatmap(ic_rows, factor_names, chart_paths["rank_ic_heatmap"], "rank_ic")
    plot_bucket_future_returns(bucket_rows, factor_names, bucket_label, chart_paths["bucket_future_ret"])
    plot_bucket_maker_edges(bucket_rows, factor_names, bid_edge_label, ask_edge_label, chart_paths["bucket_maker_edge"])
    return chart_paths


def write_fill_charts(
    output_dir: Path,
    prefix_name: str,
    fill_bucket_rows: list[dict[str, object]],
    factor_names: tuple[str, ...],
    bid_fill_label: str,
    ask_fill_label: str,
    bid_edge_label: str,
    ask_edge_label: str,
) -> dict[str, Path]:
    setup_matplotlib_cache(output_dir / "mplconfig")
    chart_paths = {
        "bucket_fill_probability": output_dir / f"{prefix_name}.bucket_fill_probability.png",
        "bucket_fill_edge": output_dir / f"{prefix_name}.bucket_fill_edge.png",
    }
    plot_bucket_fill_probability(
        fill_bucket_rows,
        factor_names,
        bid_fill_label,
        ask_fill_label,
        chart_paths["bucket_fill_probability"],
    )
    plot_bucket_fill_edges(
        fill_bucket_rows,
        factor_names,
        bid_edge_label,
        ask_edge_label,
        chart_paths["bucket_fill_edge"],
    )
    return chart_paths
