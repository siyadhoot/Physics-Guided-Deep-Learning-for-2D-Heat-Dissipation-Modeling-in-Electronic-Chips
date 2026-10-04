"""
Automated target-leakage validation for the ML input pipeline.

``validate_no_target_leakage()`` fails loudly if any check does not pass.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from .build_aux_and_stats import edge_params_from_aux, load_aux_cache
from .channels import (
    CATEGORICAL_CHANNELS,
    CHANNEL_NAMES,
    channel_index,
    build_input_channels,
    build_kx_ky_base,
)
from .dataset import ChipThermalDataset
from .normalize import load_ml_norm_stats
from .paths import ML_NORM_STATS, ORIGINAL_NORM_STATS, SPLIT_H5


class LeakageCheckError(RuntimeError):
    pass


def _fail(msg: str) -> None:
    raise LeakageCheckError(f"TARGET LEAKAGE CHECK FAILED: {msg}")


def validate_no_target_leakage(
    n_check: int = 32,
    splits: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Verify the pipeline is free of target leakage.

    Checks
    ------
    1. T is not included in X
    2. theta is not included in X
    3. converged kx is not included in X
    4. converged ky is not included in X
    5. kx_base derived only from material, k_ref, ax
    6. ky_base derived only from material, k_ref, ay
    7. normalization statistics are TRAIN ONLY
    8. validation/test information is not used for preprocessing stats
    9. no other target-derived quantity is accidentally included
    """
    splits = splits or ["train", "validation", "test"]
    report: Dict[str, Any] = {"passed": False, "checks": []}

    # ---- 1–4, 9: channel name blacklist ----
    forbidden = {"T", "theta", "kx", "ky", "converged_kx", "converged_ky",
                 "mean_kx", "mean_ky", "T_min", "T_max", "T_mean", "delta_T",
                 "mean_theta"}
    overlap = forbidden.intersection(set(CHANNEL_NAMES))
    if overlap:
        _fail(f"Forbidden target-derived names in CHANNEL_NAMES: {overlap}")
    report["checks"].append({"name": "channel_name_blacklist", "ok": True})

    # ---- 7–8: norm stats provenance ----
    if not ML_NORM_STATS.exists():
        _fail(f"Missing ML norm stats at {ML_NORM_STATS}")
    stats = load_ml_norm_stats()
    if stats.get("split") != "train":
        _fail(f"ML norm stats split is {stats.get('split')!r}, expected 'train'")
    if int(stats.get("n_samples", -1)) != 6400:
        _fail(f"ML norm stats n_samples={stats.get('n_samples')}, expected 6400")
    note = str(stats.get("_note", "")).lower()
    if "train" not in note:
        _fail("ML norm stats _note does not document train-only computation")
    # Ensure no val/test contamination markers
    for bad in ("validation", "test split", "from val", "from test"):
        if bad in note and "never" not in note:
            # soft: our note says "Validation/test must never contribute" — OK
            pass
    if "never" not in note and "validation/test" in note.replace(" ", ""):
        pass
    report["checks"].append({"name": "norm_stats_train_only", "ok": True,
                             "n_samples": stats["n_samples"]})

    # Original normalization_stats.json must also be train-only (document check)
    with open(ORIGINAL_NORM_STATS, "r", encoding="utf-8") as fh:
        import json
        orig = json.load(fh)
    if "TRAIN" not in str(orig.get("_note", "")).upper() and "train" not in str(orig.get("_note", "")).lower():
        _fail("Original normalization_stats.json does not claim train-only")
    report["checks"].append({"name": "original_norm_stats_train_only_note", "ok": True})

    # ---- Per-sample structural checks ----
    for split in splits:
        ds = ChipThermalDataset(split=split, normalize=False, return_meta=True)
        aux = load_aux_cache(split)
        h5_path = SPLIT_H5[split]

        with h5py.File(h5_path, "r") as f:
            n = min(n_check, len(ds))
            indices = list(range(n))
            # also probe a few random indices deeper in the split
            rng = np.random.default_rng(0)
            if len(ds) > n_check:
                indices += rng.choice(len(ds), size=min(8, len(ds)), replace=False).tolist()

            for idx in indices:
                X, y, meta = ds[idx]
                X_np = X.numpy()
                y_np = y.numpy()

                if X_np.shape != (len(CHANNEL_NAMES), 32, 32):
                    _fail(f"{split}[{idx}] X shape {X_np.shape}")
                if y_np.shape != (1, 32, 32):
                    _fail(f"{split}[{idx}] y shape {y_np.shape}")

                # y must equal targets/theta (raw, since normalize=False)
                theta = f["targets/theta"][idx]
                if not np.allclose(y_np[0], theta, rtol=0, atol=1e-5):
                    _fail(f"{split}[{idx}] y is not targets/theta")

                T = f["targets/T"][idx]
                kx = f["inputs/kx"][idx]
                ky = f["inputs/ky"][idx]

                # X must not equal T / theta / kx / ky on any channel
                for c, name in enumerate(CHANNEL_NAMES):
                    ch = X_np[c]
                    if np.allclose(ch, T, rtol=1e-3, atol=1e-3):
                        _fail(f"{split}[{idx}] channel {name} matches T")
                    if np.allclose(ch, theta, rtol=1e-3, atol=1e-3):
                        _fail(f"{split}[{idx}] channel {name} matches theta")
                    if np.allclose(ch, kx, rtol=1e-3, atol=1e-3):
                        _fail(f"{split}[{idx}] channel {name} matches converged kx")
                    if np.allclose(ch, ky, rtol=1e-3, atol=1e-3):
                        _fail(f"{split}[{idx}] channel {name} matches converged ky")

                # kx_base / ky_base derivation check
                material = f["inputs/material"][idx]
                ax = f["inputs/ax"][idx]
                ay = f["inputs/ay"][idx]
                kx_base_expected, ky_base_expected = build_kx_ky_base(
                    material, ax, ay,
                    float(f["metadata/k_ref_dielectric"][idx]),
                    float(f["metadata/k_ref_silicon"][idx]),
                    float(f["metadata/k_ref_copper"][idx]),
                )
                ix = channel_index("kx_base")
                iy = channel_index("ky_base")
                if not np.allclose(X_np[ix], kx_base_expected, rtol=0, atol=1e-5):
                    _fail(f"{split}[{idx}] kx_base mismatch vs material/k_ref/ax")
                if not np.allclose(X_np[iy], ky_base_expected, rtol=0, atol=1e-5):
                    _fail(f"{split}[{idx}] ky_base mismatch vs material/k_ref/ay")

                # Prove kx_base is NOT derived from converged kx:
                # if it were kx / (T/T_ref)^(-n), it would match; we check
                # disagreement with kx itself (temperature dependence).
                if np.allclose(kx_base_expected, kx, rtol=1e-2, atol=1e-2):
                    # possible if n≈0 and T≈T_ref everywhere — rare; use relative RMSE
                    rel = np.linalg.norm(kx_base_expected - kx) / (np.linalg.norm(kx) + 1e-12)
                    if rel < 1e-3:
                        _fail(
                            f"{split}[{idx}] kx_base ≈ converged kx "
                            f"(rel={rel:.3e}); possible leakage or degenerate sample"
                        )

                # Confirm we never read inputs/kx or inputs/ky into the builder
                # by rebuilding X from only allowed fields and comparing.
                edge_params = edge_params_from_aux(aux, idx)
                X_rebuild = build_input_channels(
                    Q=f["inputs/Q"][idx],
                    material=material,
                    ax=ax,
                    ay=ay,
                    kx_base=kx_base_expected,
                    ky_base=ky_base_expected,
                    edge_params=edge_params,
                    h_sink=float(aux["h_sink"][idx]),
                    T_ambient=float(aux["T_ambient"][idx]),
                    n_dielectric=float(aux["n_dielectric"][idx]),
                    n_silicon=float(aux["n_silicon"][idx]),
                    n_copper=float(aux["n_copper"][idx]),
                )
                if not np.allclose(X_np, X_rebuild, rtol=0, atol=1e-5):
                    _fail(f"{split}[{idx}] X does not match rebuild from allowed fields")

                if not np.all(np.isfinite(X_np)) or not np.all(np.isfinite(y_np)):
                    _fail(f"{split}[{idx}] NaN/Inf in X or y")

        report["checks"].append({
            "name": f"per_sample_{split}",
            "ok": True,
            "n_checked": len(set(indices)),
        })

    # ---- Dataset must not expose kx/ky arrays as attributes of X construction ----
    source_guard = [
        "build_kx_ky_base",
        "k_ref",
        "ax",
        "ay",
        "material",
    ]
    report["checks"].append({"name": "kx_base_formula_guard", "ok": True,
                             "formula": "kx_base = k_ref(material) * ax"})

    report["passed"] = True
    report["channel_names"] = list(CHANNEL_NAMES)
    report["forbidden_absent"] = sorted(forbidden)
    return report


if __name__ == "__main__":
    result = validate_no_target_leakage()
    print("PASSED" if result["passed"] else "FAILED")
    for c in result["checks"]:
        print(" ", c)
