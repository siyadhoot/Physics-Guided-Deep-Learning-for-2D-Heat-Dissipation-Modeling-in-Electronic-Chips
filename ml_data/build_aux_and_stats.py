"""
Build per-split BC / n-parameter aux caches and TRAIN-ONLY ML normalization
statistics.

Usage
-----
    python -m ml_data.build_aux_and_stats

This does NOT modify train.h5 / validation.h5 / test.h5 and does NOT run
the PDE solver.  Boundary parameters are reconstructed from stored seeds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np

from .channels import (
    CATEGORICAL_CHANNELS,
    CHANNEL_NAMES,
    EDGE_ORDER,
    TARGET_NAME,
    build_input_channels,
    build_kx_ky_base,
)
from .normalize import (
    _finalize,
    _running_update,
    empty_acc,
    load_original_norm_stats,
    save_ml_norm_stats,
)
from .paths import ML_AUX_DIR, SPLIT_CANONICAL, SPLIT_H5, aux_cache_path
from .reconstruct import EDGE_KEYS, reconstruct_bc_and_n


def _decode_str(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if isinstance(v, np.ndarray) and v.shape == ():
        v = v.item()
        if isinstance(v, bytes):
            return v.decode("utf-8")
    return str(v)


def build_aux_cache(split: str, verify_against_h5: bool = True) -> Path:
    """
    Reconstruct per-sample edge BC params + actual n scalars from seeds.

    Writes ``dataset/ml_aux/{split}_bc_params.npz``.
    """
    split = SPLIT_CANONICAL[split]
    h5_path = SPLIT_H5[split]
    out_path = aux_cache_path(split)
    ML_AUX_DIR.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        n = int(f["inputs/Q"].shape[0])
        seeds = np.asarray(f["metadata/seed"][:], dtype=np.int64)
        h_sink_h5 = np.asarray(f["metadata/h_sink"][:], dtype=np.float64)
        T_amb_h5 = np.asarray(f["metadata/T_ambient"][:], dtype=np.float64)
        n_d_h5 = np.asarray(f["metadata/n_dielectric"][:], dtype=np.float64)
        n_s_h5 = np.asarray(f["metadata/n_silicon"][:], dtype=np.float64)
        n_c_h5 = np.asarray(f["metadata/n_copper"][:], dtype=np.float64)
        bt_h5 = [_decode_str(f["metadata/boundary_type"][i]) for i in range(n)]

    data: Dict[str, Any] = {
        "seed": seeds,
        "h_sink": np.zeros(n, dtype=np.float64),
        "T_ambient": np.zeros(n, dtype=np.float64),
        "n_dielectric": np.zeros(n, dtype=np.float64),
        "n_silicon": np.zeros(n, dtype=np.float64),  # ACTUAL (corrected)
        "n_copper": np.zeros(n, dtype=np.float64),
        "n_silicon_metadata": n_s_h5.copy(),
        "n_silicon_was_overridden": np.zeros(n, dtype=np.int8),
        "boundary_type": np.array(bt_h5, dtype=object),
    }
    for edge in EDGE_KEYS:
        data[f"bc_type_{edge}"] = np.zeros(n, dtype=np.float32)
        data[f"bc_h_{edge}"] = np.zeros(n, dtype=np.float32)
        data[f"bc_T_{edge}"] = np.zeros(n, dtype=np.float32)

    mismatches = []
    for i in range(n):
        edge_params, n_actual, bc_meta = reconstruct_bc_and_n(int(seeds[i]))
        data["h_sink"][i] = bc_meta["h_sink"]
        data["T_ambient"][i] = bc_meta["T_ambient"]
        data["n_dielectric"][i] = n_actual[0]
        data["n_silicon"][i] = n_actual[1]
        data["n_copper"][i] = n_actual[2]
        if abs(n_actual[1] - n_s_h5[i]) > 1e-9:
            data["n_silicon_was_overridden"][i] = 1
        for edge in EDGE_KEYS:
            data[f"bc_type_{edge}"][i] = edge_params[edge]["type"]
            data[f"bc_h_{edge}"][i] = edge_params[edge]["h"]
            data[f"bc_T_{edge}"][i] = edge_params[edge]["T"]

        if verify_against_h5:
            if abs(bc_meta["h_sink"] - h_sink_h5[i]) > 1e-6:
                mismatches.append((i, "h_sink"))
            if abs(bc_meta["T_ambient"] - T_amb_h5[i]) > 1e-6:
                mismatches.append((i, "T_ambient"))
            if abs(n_actual[0] - n_d_h5[i]) > 1e-6:
                mismatches.append((i, "n_dielectric"))
            if abs(n_actual[2] - n_c_h5[i]) > 1e-6:
                mismatches.append((i, "n_copper"))
            if bc_meta["scenario"] != bt_h5[i]:
                mismatches.append((i, "boundary_type"))

        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"  [{split}] reconstructed {i + 1}/{n}")

    if mismatches:
        raise RuntimeError(
            f"Seed replay failed verification for {split}: "
            f"{len(mismatches)} mismatches, first={mismatches[:5]}"
        )

    n_over = int(data["n_silicon_was_overridden"].sum())
    print(f"  [{split}] n_silicon overrides corrected: {n_over}/{n}")

    # numpy object arrays need allow_pickle
    np.savez_compressed(out_path, **data)
    print(f"  wrote {out_path}")
    return out_path


def load_aux_cache(split: str) -> Dict[str, np.ndarray]:
    path = aux_cache_path(split)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing aux cache {path}. Run: python -m ml_data.build_aux_and_stats"
        )
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def edge_params_from_aux(aux: Dict[str, np.ndarray], idx: int) -> Dict[str, Dict[str, float]]:
    out = {}
    for edge in EDGE_KEYS:
        out[edge] = {
            "type": float(aux[f"bc_type_{edge}"][idx]),
            "h": float(aux[f"bc_h_{edge}"][idx]),
            "T": float(aux[f"bc_T_{edge}"][idx]),
        }
    return out


def compute_ml_normalization_stats() -> Path:
    """
    Compute mean/std/min/max for all continuous input channels + theta
    using the TRAINING split only.
    """
    split = "train"
    h5_path = SPLIT_H5[split]
    aux = load_aux_cache(split)
    original = load_original_norm_stats()

    accs = {name: empty_acc() for name in CHANNEL_NAMES if name not in CATEGORICAL_CHANNELS}
    acc_theta = empty_acc()

    with h5py.File(h5_path, "r") as f:
        n = int(f["inputs/Q"].shape[0])
        assert n == 6400, f"Expected 6400 train samples, got {n}"

        for i in range(n):
            Q = f["inputs/Q"][i]
            material = f["inputs/material"][i]
            ax = f["inputs/ax"][i]
            ay = f["inputs/ay"][i]
            theta = f["targets/theta"][i]

            kx_base, ky_base = build_kx_ky_base(
                material, ax, ay,
                float(f["metadata/k_ref_dielectric"][i]),
                float(f["metadata/k_ref_silicon"][i]),
                float(f["metadata/k_ref_copper"][i]),
            )
            edge_params = edge_params_from_aux(aux, i)
            X = build_input_channels(
                Q=Q,
                material=material,
                ax=ax,
                ay=ay,
                kx_base=kx_base,
                ky_base=ky_base,
                edge_params=edge_params,
                h_sink=float(aux["h_sink"][i]),
                T_ambient=float(aux["T_ambient"][i]),
                n_dielectric=float(aux["n_dielectric"][i]),
                n_silicon=float(aux["n_silicon"][i]),
                n_copper=float(aux["n_copper"][i]),
            )
            for c, name in enumerate(CHANNEL_NAMES):
                if name in CATEGORICAL_CHANNELS:
                    continue
                _running_update(accs[name], X[c])
            _running_update(acc_theta, theta)

            if (i + 1) % 500 == 0 or i + 1 == n:
                print(f"  [norm] processed {i + 1}/{n} train samples")

    channel_stats = {}
    for name in CHANNEL_NAMES:
        if name in CATEGORICAL_CHANNELS:
            channel_stats[name] = {
                "normalization": "none",
                "reason": "categorical / discrete code — not z-scored",
                "split": "train",
            }
        else:
            channel_stats[name] = _finalize(accs[name])
            channel_stats[name]["normalization"] = "zscore"

    # Cross-check reused channels against original train-only stats
    reuse_notes = {}
    for key in ("Q", "ax", "ay"):
        old = original[key]
        new = channel_stats[key]
        reuse_notes[key] = {
            "original_mean": old["mean"],
            "new_mean": new["mean"],
            "original_std": old["std"],
            "new_std": new["std"],
            "mean_abs_diff": abs(old["mean"] - new["mean"]),
            "std_abs_diff": abs(old["std"] - new["std"]),
        }

    stats = {
        "_note": (
            "Computed on the TRAINING split only (6400 samples). "
            "Categorical channels (material, bc_type_*) are not normalized. "
            "Continuous channels use z-score (x - mean) / std at load time. "
            "Validation/test must never contribute to these statistics. "
            "kx_base/ky_base are derived from material + k_ref + ax/ay "
            "(never from converged kx/ky). "
            "n_silicon uses the seed-reconstructed actual exponent."
        ),
        "split": "train",
        "n_samples": 6400,
        "channels": channel_stats,
        "target": {
            TARGET_NAME: {
                **_finalize(acc_theta),
                "normalization": "zscore",
            }
        },
        "reused_original_channel_crosscheck": reuse_notes,
        "categorical_channels": sorted(CATEGORICAL_CHANNELS),
        "channel_order": CHANNEL_NAMES,
    }
    path = save_ml_norm_stats(stats)
    print(f"  wrote {path}")
    return path


def main() -> None:
    print("Building aux BC caches (seed replay, no solver)...")
    for split in ("train", "validation", "test"):
        build_aux_cache(split, verify_against_h5=True)
    print("Computing TRAIN-ONLY ML normalization stats...")
    compute_ml_normalization_stats()
    print("Done.")


if __name__ == "__main__":
    main()
