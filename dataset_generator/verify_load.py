"""
verify_load.py
===============
Step: automated data-loading verification test. Loads 5 samples each from
train / validation / test HDF5 files and checks:
  - array shapes == (32,32)
  - dtypes (float32 for continuous fields, int8 for material)
  - no NaN / Inf anywhere
  - correct HDF5 chunked reading (random-access + cross-chunk slice reads)
  - Kirchhoff round trip T -> theta -> T' consistency directly from file
"""

import os
import sys

import numpy as np
import h5py

import config as C
import kirchhoff as K

FILES = {
    "train": os.path.join(C.DATASET_DIR, "train.h5"),
    "val": os.path.join(C.DATASET_DIR, "validation.h5"),
    "test": os.path.join(C.DATASET_DIR, "test.h5"),
}

INPUT_FIELDS = ["Q", "material", "kx", "ky", "ax", "ay"]
TARGET_FIELDS = ["theta", "T"]

# The authoritative round-trip check (T -> theta -> psi^-1(theta) ~= T) is done
# analytically in float64 at GENERATION time (see sample_gen.py / kirchhoff.py)
# BEFORE casting to float32 for HDF5 storage, and must satisfy the project's
# 1e-5 K tolerance (it does, by ~6 orders of magnitude margin -- see
# dataset_statistics.json: worst-case ~2.8e-11 K). Once T and theta are stored
# as float32 (per the required HDF5 schema) and reloaded here, simply
# quantizing 300-500 K values to float32 already introduces ~1e-5-1e-4 K of
# rounding noise on its own (float32 relative eps ~1.2e-7 * ~5e2 K), which
# then propagates through the (mildly amplifying) inverse-power-law transform.
# So the independent float32 reconstruction below uses a separate, physically
# appropriate tolerance that reflects float32 storage precision, not the
# stricter float64 generation-time tolerance.
FLOAT32_ROUNDTRIP_TOL_K = 1e-3


def check_split(split, path, n_check=5):
    print(f"\n--- Verifying split={split} ({path}) ---")
    ok = True
    with h5py.File(path, "r") as h5:
        n_total = h5["inputs/Q"].shape[0]
        assert n_total > 0
        idxs = sorted(np.random.default_rng(0).choice(n_total, size=min(n_check, n_total), replace=False).tolist())
        print(f"  n_total={n_total}, sampling indices={idxs}")

        for name in INPUT_FIELDS:
            ds = h5[f"inputs/{name}"]
            expected_dtype = np.int8 if name == "material" else np.float32
            if ds.dtype != expected_dtype:
                print(f"  [FAIL] inputs/{name} dtype={ds.dtype}, expected {expected_dtype}")
                ok = False
            for i in idxs:
                arr = ds[i]
                if arr.shape != (C.GRID_SIZE, C.GRID_SIZE):
                    print(f"  [FAIL] inputs/{name}[{i}] shape={arr.shape}")
                    ok = False
                if not np.all(np.isfinite(arr)):
                    print(f"  [FAIL] inputs/{name}[{i}] contains NaN/Inf")
                    ok = False

        for name in TARGET_FIELDS:
            ds = h5[f"targets/{name}"]
            if ds.dtype != np.float32:
                print(f"  [FAIL] targets/{name} dtype={ds.dtype}, expected float32")
                ok = False
            for i in idxs:
                arr = ds[i]
                if arr.shape != (C.GRID_SIZE, C.GRID_SIZE):
                    print(f"  [FAIL] targets/{name}[{i}] shape={arr.shape}")
                    ok = False
                if not np.all(np.isfinite(arr)):
                    print(f"  [FAIL] targets/{name}[{i}] contains NaN/Inf")
                    ok = False

        # cross-chunk slice read (contiguous slice spanning multiple chunks)
        chunk = h5["inputs/Q"].chunks
        span = min(n_total, (chunk[0] if chunk else 1) * 2 + 3)
        big_slice = h5["inputs/Q"][0:span]
        if big_slice.shape[0] != span or not np.all(np.isfinite(big_slice)):
            print(f"  [FAIL] cross-chunk slice read of inputs/Q[0:{span}] failed")
            ok = False
        else:
            print(f"  [OK] cross-chunk slice read inputs/Q[0:{span}] shape={big_slice.shape}, chunks={chunk}")

        # Physical-range sanity check + Kirchhoff round-trip check.
        # Note: theta uses a PER-CELL material exponent n(x,y) (n_map), which is
        # not itself stored as a separate channel (only kx/ky, which already bake
        # in k(T) at the converged temperature, are stored per spec section 19).
        # Because n varies strongly by material (Si n~1-1.5 vs Cu n~0.1-0.4), a
        # naive whole-field T-vs-theta correlation is NOT a valid sanity check
        # under large delta_T (material identity dominates the theta offset).
        # The authoritative round-trip check (T -> theta -> psi^-1(theta) ~= T,
        # done per-cell with the SAME local n used to build theta) is performed
        # analytically for every sample at generation time and its result is
        # stored in metadata/kirchhoff_roundtrip_error_K -- verify it here.
        for i in idxs:
            T = h5["targets/T"][i].astype(np.float64)
            theta = h5["targets/theta"][i].astype(np.float64)
            mat = h5["inputs/material"][i]
            if T.min() < 250 or T.max() > 550:
                print(f"  [FAIL] {split}[{i}] T out of plausible physical range: "
                      f"[{T.min():.1f},{T.max():.1f}]")
                ok = False

            # independently reconstruct n_map from material map + stored per-material
            # exponents, then apply the analytical inverse Kirchhoff transform, and
            # compare against the stored T -- this exercises the exact pipeline a
            # downstream model would use on its OWN predicted theta.
            n_map = np.zeros_like(T)
            n_map[mat == 0] = float(h5["metadata/n_dielectric"][i])
            n_map[mat == 1] = float(h5["metadata/n_silicon"][i])
            n_map[mat == 2] = float(h5["metadata/n_copper"][i])
            T_ref = float(h5["metadata/T_ref_K"][i])
            T_reconstructed = K.inverse_kirchhoff(theta, n_map, T_ref)
            rt_err = float(np.max(np.abs(T - T_reconstructed)))
            if rt_err > FLOAT32_ROUNDTRIP_TOL_K:
                print(f"  [FAIL] {split}[{i}] independently-reconstructed Kirchhoff "
                      f"round-trip error {rt_err:.2e} K exceeds float32 tolerance "
                      f"{FLOAT32_ROUNDTRIP_TOL_K:.1e} K")
                ok = False
            # also report the true (float64, generation-time) round-trip error,
            # which is what the project's 1e-5 K spec tolerance applies to.
            true_rt_err = float(h5["metadata/kirchhoff_roundtrip_error_K"][i])
            if true_rt_err > C.KIRCHHOFF_ROUNDTRIP_TOL:
                print(f"  [FAIL] {split}[{i}] generation-time (float64) round-trip "
                      f"error {true_rt_err:.2e} K exceeds spec tolerance "
                      f"{C.KIRCHHOFF_ROUNDTRIP_TOL:.1e} K")
                ok = False

        print(f"  metadata keys: {list(h5['metadata'].keys())[:6]} ... "
              f"({len(h5['metadata'].keys())} total)")
        for i in idxs:
            sid = int(h5["metadata/sample_id"][i])
            tmax = float(h5["metadata/T_max"][i])
            print(f"    idx={i} sample_id={sid} T_max={tmax:.2f}K")

    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    all_ok = True
    for split, path in FILES.items():
        if not os.path.exists(path):
            print(f"[MISSING] {path}")
            all_ok = False
            continue
        all_ok &= check_split(split, path)

    print("\n" + "=" * 50)
    print(f"DATA LOADING VERIFICATION: {'ALL PASSED' if all_ok else 'FAILURES DETECTED'}")
    print("=" * 50)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
