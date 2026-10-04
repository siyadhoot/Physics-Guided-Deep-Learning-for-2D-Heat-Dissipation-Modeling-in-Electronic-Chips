"""
Normalization for the leakage-free ML pipeline.

Rules
-----
* Statistics are computed from the TRAINING split only.
* Categorical channels (material, bc_type_*) are NOT z-scored.
* Existing train-only stats for Q / ax / ay / theta are reused where valid.
* New continuous channels (kx_base, ky_base, BC params, globals) get fresh
  train-only mean/std written to ``dataset/ml_normalization_stats.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .channels import CATEGORICAL_CHANNELS, CHANNEL_NAMES, TARGET_NAME
from .paths import ML_NORM_STATS, ORIGINAL_NORM_STATS


def load_ml_norm_stats(path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or ML_NORM_STATS
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m ml_data.build_aux_and_stats"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_original_norm_stats(path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or ORIGINAL_NORM_STATS
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_normalization(X: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    """
    Z-score continuous channels of X in-place-safe copy.

    X : (C, H, W)
    """
    out = X.astype(np.float32, copy=True)
    for i, name in enumerate(CHANNEL_NAMES):
        if name in CATEGORICAL_CHANNELS:
            continue
        ch_stats = stats["channels"][name]
        mean = float(ch_stats["mean"])
        std = float(ch_stats["std"])
        if std < 1e-12:
            out[i] = out[i] - mean
        else:
            out[i] = (out[i] - mean) / std
    return out


def apply_target_normalization(y: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    """Normalize theta target y : (1, H, W) or (H, W)."""
    tstats = stats["target"][TARGET_NAME]
    mean = float(tstats["mean"])
    std = float(tstats["std"])
    y = y.astype(np.float32, copy=True)
    if std < 1e-12:
        return y - mean
    return (y - mean) / std


def denormalize_target(y_norm: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    tstats = stats["target"][TARGET_NAME]
    mean = float(tstats["mean"])
    std = float(tstats["std"])
    return y_norm.astype(np.float32) * std + mean


def _running_update(acc: Dict[str, float], arr: np.ndarray) -> None:
    """Accumulate count / sum / sumsq / min / max for a flattened array."""
    flat = arr.astype(np.float64).ravel()
    acc["count"] += flat.size
    acc["sum"] += float(flat.sum())
    acc["sumsq"] += float(np.square(flat).sum())
    acc["min"] = min(acc["min"], float(flat.min()))
    acc["max"] = max(acc["max"], float(flat.max()))


def _finalize(acc: Dict[str, float]) -> Dict[str, float]:
    n = max(acc["count"], 1.0)
    mean = acc["sum"] / n
    var = max(acc["sumsq"] / n - mean * mean, 0.0)
    return {
        "mean": mean,
        "std": float(np.sqrt(var)),
        "min": acc["min"],
        "max": acc["max"],
        "count": int(acc["count"]),
        "split": "train",
    }


def empty_acc() -> Dict[str, float]:
    return {"count": 0.0, "sum": 0.0, "sumsq": 0.0, "min": float("inf"), "max": float("-inf")}


def save_ml_norm_stats(stats: Dict[str, Any], path: Optional[Path] = None) -> Path:
    path = path or ML_NORM_STATS
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return path
