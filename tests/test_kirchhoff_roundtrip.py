"""
Kirchhoff forward/inverse round-trip consistency test.

Uses ``dataset_generator/kirchhoff.py`` and the *actual* per-sample material
exponents from ``dataset/ml_aux/*_bc_params.npz`` (including corrected
``n_silicon`` for very_challenging overrides).

Does not modify any HDF5 files or dataset_generator sources.

Run
---
    python -m tests.test_kirchhoff_roundtrip
    python -m unittest tests.test_kirchhoff_roundtrip
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATOR_DIR = PROJECT_ROOT / "dataset_generator"
DATASET_DIR = PROJECT_ROOT / "dataset"
ML_AUX_DIR = DATASET_DIR / "ml_aux"

if str(GENERATOR_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATOR_DIR))

import kirchhoff as K  # noqa: E402
import config as C  # noqa: E402

# Generation-time rejection threshold (K).
ROUNDTRIP_TOL_K = float(C.KIRCHHOFF_ROUNDTRIP_TOL)  # 1e-5 K

SPLIT_FILES = {
    "train": DATASET_DIR / "train.h5",
    "validation": DATASET_DIR / "validation.h5",
    "test": DATASET_DIR / "test.h5",
}

AUX_FILES = {
    "train": ML_AUX_DIR / "train_bc_params.npz",
    "validation": ML_AUX_DIR / "validation_bc_params.npz",
    "test": ML_AUX_DIR / "test_bc_params.npz",
}

N_PER_SPLIT_BASE = 24


def _build_n_map(
    material: np.ndarray,
    n_dielectric: float,
    n_silicon: float,
    n_copper: float,
) -> np.ndarray:
    n_map = np.zeros(material.shape, dtype=np.float64)
    n_map[material == 0] = n_dielectric
    n_map[material == 1] = n_silicon
    n_map[material == 2] = n_copper
    return n_map


def _select_indices(aux: Dict[str, np.ndarray], n_total: int, n_base: int) -> List[int]:
    """Pick a representative mix: head, stride, and n_silicon-override samples."""
    idxs = set(range(min(n_base, n_total)))
    if n_total > n_base:
        stride = max(1, n_total // n_base)
        idxs.update(range(0, n_total, stride))
    if "n_silicon_was_overridden" in aux:
        overridden = np.flatnonzero(np.asarray(aux["n_silicon_was_overridden"]) != 0)
        idxs.update(int(i) for i in overridden[:16])
    return sorted(i for i in idxs if 0 <= i < n_total)


def _error_stats(all_err: np.ndarray) -> Dict[str, float]:
    return {
        "max_abs_error": float(all_err.max()),
        "mean_abs_error": float(all_err.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(all_err)))),
    }


def evaluate_split(
    split: str,
    indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """
    For each selected sample: T → forward Kirchhoff → inverse → compare to T.
    """
    h5_path = SPLIT_FILES[split]
    aux_path = AUX_FILES[split]
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    if not aux_path.exists():
        raise FileNotFoundError(
            f"Missing {aux_path}. Run: python -m ml_data.build_aux_and_stats"
        )

    aux_raw = np.load(aux_path, allow_pickle=True)
    aux = {k: aux_raw[k] for k in aux_raw.files}

    abs_errors: List[np.ndarray] = []
    n_override_tested = 0

    with h5py.File(h5_path, "r") as f:
        n_total = int(f["targets/T"].shape[0])
        if indices is None:
            indices = _select_indices(aux, n_total, N_PER_SPLIT_BASE)
        else:
            indices = list(indices)

        for idx in indices:
            T = np.asarray(f["targets/T"][idx], dtype=np.float64)
            material = np.asarray(f["inputs/material"][idx])
            T_ref = float(f["metadata/T_ref_K"][idx])

            n_d = float(aux["n_dielectric"][idx])
            n_s = float(aux["n_silicon"][idx])  # ACTUAL (seed-corrected)
            n_c = float(aux["n_copper"][idx])
            if "n_silicon_was_overridden" in aux and int(aux["n_silicon_was_overridden"][idx]) != 0:
                n_override_tested += 1

            n_map = _build_n_map(material, n_d, n_s, n_c)
            theta = K.forward_kirchhoff(T, n_map, T_ref)
            T_rec = K.inverse_kirchhoff(theta, n_map, T_ref)
            abs_errors.append(np.abs(T_rec - T))

    all_err = np.concatenate([e.ravel() for e in abs_errors])
    stats = _error_stats(all_err)
    return {
        "split": split,
        "n_samples": len(indices),
        "n_override_tested": n_override_tested,
        "indices": list(indices),
        "tol_K": ROUNDTRIP_TOL_K,
        "all_err": all_err,
        "passed": bool(all_err.size and stats["max_abs_error"] <= ROUNDTRIP_TOL_K),
        **stats,
    }


def run_all_splits() -> Tuple[bool, Dict[str, Any]]:
    split_results: Dict[str, Any] = {}
    err_chunks: List[np.ndarray] = []
    total_n = 0
    n_override = 0

    for split in ("train", "validation", "test"):
        r = evaluate_split(split)
        err_chunks.append(r["all_err"])
        total_n += r["n_samples"]
        n_override += r["n_override_tested"]
        # Drop bulky array from nested report
        split_results[split] = {k: v for k, v in r.items() if k != "all_err"}

    all_err = np.concatenate(err_chunks)
    stats = _error_stats(all_err)
    summary: Dict[str, Any] = {
        "n_samples": total_n,
        "n_override_tested": n_override,
        "tol_K": ROUNDTRIP_TOL_K,
        "splits": split_results,
        **stats,
    }
    summary["passed"] = stats["max_abs_error"] <= ROUNDTRIP_TOL_K
    return summary["passed"], summary


def print_report(summary: Dict[str, Any]) -> None:
    print("=" * 70)
    print("Kirchhoff round-trip test  (T -> theta -> T)")
    print("Implementation: dataset_generator/kirchhoff.py")
    print(f"Tolerance: {summary['tol_K']:.3e} K  (config.KIRCHHOFF_ROUNDTRIP_TOL)")
    print("=" * 70)
    for split, r in summary["splits"].items():
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  [{status}] {split:12s}  n={r['n_samples']:4d}  "
            f"overrides={r['n_override_tested']:3d}  "
            f"max|err|={r['max_abs_error']:.3e}  "
            f"MAE={r['mean_abs_error']:.3e}  "
            f"RMSE={r['rmse']:.3e}"
        )
    print("-" * 70)
    print(f"  Samples tested:               {summary['n_samples']}")
    print(f"  n_silicon-override samples:   {summary['n_override_tested']}")
    print(f"  Max abs reconstruction error: {summary['max_abs_error']:.6e} K")
    print(f"  Mean abs reconstruction error:{summary['mean_abs_error']:.6e} K")
    print(f"  RMSE:                         {summary['rmse']:.6e} K")
    print("=" * 70)
    if summary["passed"]:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
        print(
            f"  max|T - T_rec| = {summary['max_abs_error']:.6e} K "
            f"> tol {summary['tol_K']:.3e} K"
        )
    print("=" * 70)


class TestKirchhoffRoundtrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.passed, cls.summary = run_all_splits()

    def test_roundtrip_within_tolerance(self):
        s = self.summary
        self.assertTrue(
            s["passed"],
            msg=(
                f"Kirchhoff round-trip max abs error {s['max_abs_error']:.6e} K "
                f"exceeds tolerance {s['tol_K']:.3e} K "
                f"(n_samples={s['n_samples']})"
            ),
        )

    def test_reports_nonzero_samples(self):
        self.assertGreater(self.summary["n_samples"], 0)

    def test_includes_override_samples(self):
        self.assertGreater(
            self.summary["n_override_tested"],
            0,
            msg="Expected at least one very_challenging n_silicon override sample",
        )


def main() -> int:
    passed, summary = run_all_splits()
    print_report(summary)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
