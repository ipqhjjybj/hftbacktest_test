from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .factors import FACTOR_NAMES


TARGET_SPECS = (
    ("bid_expected_edge_bps", "bid_lifecycle_expected_edge_{horizon_ms}ms_bps"),
    ("ask_expected_edge_bps", "ask_lifecycle_expected_edge_{horizon_ms}ms_bps"),
    ("bid_edge_if_filled_bps", "bid_lifecycle_edge_when_filled_{horizon_ms}ms_bps"),
    ("ask_edge_if_filled_bps", "ask_lifecycle_edge_when_filled_{horizon_ms}ms_bps"),
    ("bid_fill_prob", "bid_lifecycle_fill_prob"),
    ("ask_fill_prob", "ask_lifecycle_fill_prob"),
)


def raw_feature_matrix(data: dict[str, np.ndarray]) -> np.ndarray:
    cols = [data[name].astype(np.float64, copy=False) for name in FACTOR_NAMES]
    return np.column_stack(cols)


def sample_indices(n: int, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or n <= max_samples:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_samples, replace=False)
    idx.sort()
    return idx.astype(np.int64, copy=False)


def fit_standardizer(raw_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros(raw_x.shape[1], dtype=np.float64)
    std = np.ones(raw_x.shape[1], dtype=np.float64)
    for col in range(raw_x.shape[1]):
        values = raw_x[:, col]
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            continue
        mean[col] = float(np.mean(finite))
        col_std = float(np.std(finite))
        if np.isfinite(col_std) and col_std > 1e-12:
            std[col] = col_std
    return mean.astype(np.float64), std.astype(np.float64)


def transform_features(
    raw_x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    include_interactions: bool,
    clip_z: float,
) -> np.ndarray:
    z = (raw_x - mean) / std
    z = np.nan_to_num(z, nan=0.0, posinf=clip_z, neginf=-clip_z)
    z = np.clip(z, -clip_z, clip_z)
    parts = [z]
    parts.append(z * z)
    interactions = []
    if include_interactions:
        for i in range(z.shape[1]):
            for j in range(i + 1, z.shape[1]):
                interactions.append((z[:, i] * z[:, j])[:, None])
    if interactions:
        parts.append(np.hstack(interactions))
    out = np.hstack(parts).astype(np.float64, copy=False)
    return np.nan_to_num(out, nan=0.0, posinf=clip_z * clip_z, neginf=-(clip_z * clip_z))


def feature_output_names(include_interactions: bool) -> list[str]:
    names = list(FACTOR_NAMES)
    names.extend([f"{name}^2" for name in FACTOR_NAMES])
    if include_interactions:
        for i, left in enumerate(FACTOR_NAMES):
            for right in FACTOR_NAMES[i + 1 :]:
                names.append(f"{left}*{right}")
    return names


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge_l2: float) -> np.ndarray:
    valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    if int(valid.sum()) < x.shape[1] + 10:
        raise ValueError(f"not enough valid samples to fit ridge: {int(valid.sum())}")
    xv = x[valid]
    yv = y[valid].astype(np.float64, copy=False)
    xv = np.nan_to_num(xv, nan=0.0, posinf=0.0, neginf=0.0)
    yv = np.nan_to_num(yv, nan=0.0, posinf=0.0, neginf=0.0)
    design = np.column_stack((np.ones(len(xv), dtype=np.float64), xv))
    scale = max(float(len(design)), 1.0)
    xtx = np.einsum("ni,nj->ij", design, design) / scale
    penalty = np.eye(xtx.shape[0], dtype=np.float64) * (ridge_l2 / scale)
    penalty[0, 0] = 0.0
    xty = np.einsum("ni,n->i", design, yv) / scale
    return np.linalg.solve(xtx + penalty, xty)


def predict_linear(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return coef[0] + np.einsum("ij,j->i", x, coef[1:])


def target_arrays(labels: dict[str, np.ndarray], horizon_ms: int) -> dict[str, np.ndarray]:
    out = {}
    for target_name, template in TARGET_SPECS:
        out[target_name] = labels[template.format(horizon_ms=horizon_ms)]
    return out


def fit_edge_models(
    train_data: dict[str, np.ndarray],
    train_labels: dict[str, np.ndarray],
    horizon_ms: int,
    max_train_samples: int,
    ridge_l2: float,
    include_interactions: bool,
    clip_z: float,
    seed: int,
) -> dict[str, object]:
    raw = raw_feature_matrix(train_data)
    idx = sample_indices(len(raw), max_train_samples, seed)
    raw_sample = raw[idx]
    mean, std = fit_standardizer(raw_sample)
    x = transform_features(raw_sample, mean, std, include_interactions, clip_z)
    targets = target_arrays(train_labels, horizon_ms)

    models = {}
    for target_name, values in targets.items():
        y = values[idx]
        coef = fit_ridge(x, y, ridge_l2)
        pred = predict_linear(x, coef)
        valid = np.isfinite(y)
        corr = float(np.corrcoef(pred[valid], y[valid])[0, 1]) if int(valid.sum()) > 2 else float("nan")
        models[target_name] = {
            "coef": coef,
            "train_samples": int(valid.sum()),
            "train_target_mean": float(np.nanmean(y)),
            "train_pred_mean": float(np.nanmean(pred)),
            "train_corr": corr,
        }

    return {
        "feature_names": list(FACTOR_NAMES),
        "expanded_feature_names": feature_output_names(include_interactions),
        "mean": mean,
        "std": std,
        "horizon_ms": int(horizon_ms),
        "ridge_l2": float(ridge_l2),
        "include_interactions": bool(include_interactions),
        "clip_z": float(clip_z),
        "models": models,
    }


def predict_edge_scores(model: dict[str, object], data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    raw = raw_feature_matrix(data)
    x = transform_features(
        raw,
        model["mean"],
        model["std"],
        bool(model["include_interactions"]),
        float(model["clip_z"]),
    )
    out = {}
    for target_name, item in model["models"].items():
        pred = predict_linear(x, item["coef"])
        if target_name.endswith("_fill_prob"):
            pred = np.clip(pred, 0.0, 1.0)
        out[f"pred_{target_name}"] = pred
    return out


def score_bucket_rows(
    scores: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    horizon_ms: int,
    buckets: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    label_map = target_arrays(labels, horizon_ms)
    for side in ("bid", "ask"):
        score_name = f"pred_{side}_expected_edge_bps"
        edge_if_filled_score_name = f"pred_{side}_edge_if_filled_bps"
        fill_score_name = f"pred_{side}_fill_prob"
        actual_edge_name = f"{side}_expected_edge_bps"
        actual_edge_if_filled_name = f"{side}_edge_if_filled_bps"
        actual_fill_name = f"{side}_fill_prob"
        score = scores[score_name]
        edge_if_filled_score = scores[edge_if_filled_score_name]
        fill_score = scores[fill_score_name]
        actual_edge = label_map[actual_edge_name]
        actual_edge_if_filled = label_map[actual_edge_if_filled_name]
        actual_fill = label_map[actual_fill_name]
        valid = np.isfinite(score) & np.isfinite(actual_edge) & np.isfinite(actual_fill)
        sv = score[valid]
        if len(sv) < buckets * 10 or np.nanstd(sv) <= 0:
            continue
        edges = np.nanquantile(sv, np.linspace(0.0, 1.0, buckets + 1))
        edges = np.unique(edges)
        if len(edges) <= 2:
            continue
        bucket_id = np.searchsorted(edges[1:-1], score, side="right")
        for b in range(len(edges) - 1):
            mask = valid & (bucket_id == b)
            count = int(mask.sum())
            if count == 0:
                continue
            rows.append(
                {
                    "side": side,
                    "bucket": b + 1,
                    "samples": count,
                    "score_min": float(np.nanmin(score[mask])),
                    "score_max": float(np.nanmax(score[mask])),
                    "pred_edge_mean_bps": float(np.nanmean(score[mask])),
                    "actual_expected_edge_mean_bps": float(np.nanmean(actual_edge[mask])),
                    "pred_edge_if_filled_mean_bps": float(np.nanmean(edge_if_filled_score[mask])),
                    "actual_edge_if_filled_mean_bps": float(np.nanmean(actual_edge_if_filled[mask])),
                    "pred_fill_prob_mean": float(np.nanmean(fill_score[mask])),
                    "actual_fill_prob_mean": float(np.nanmean(actual_fill[mask])),
                    "positive_actual_edge_frac": float(np.nanmean(actual_edge[mask] > 0)),
                }
            )
    return rows


def model_to_jsonable(model: dict[str, object]) -> dict[str, object]:
    return {
        "feature_names": model["feature_names"],
        "expanded_feature_names": model["expanded_feature_names"],
        "mean": np.asarray(model["mean"]).tolist(),
        "std": np.asarray(model["std"]).tolist(),
        "horizon_ms": model["horizon_ms"],
        "ridge_l2": model["ridge_l2"],
        "include_interactions": model["include_interactions"],
        "clip_z": model["clip_z"],
        "models": {
            name: {
                key: (np.asarray(value).tolist() if key == "coef" else value)
                for key, value in item.items()
            }
            for name, item in model["models"].items()
        },
    }


def write_model(path: Path, model: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model_to_jsonable(model), indent=2, sort_keys=True))
