"""
finalize_dataset.py
====================
Assembles the checkpoint shards produced by generate_dataset.py into the
final on-disk dataset artifacts:

    dataset/train.h5
    dataset/validation.h5
    dataset/test.h5
    dataset/metadata.csv
    dataset/normalization_stats.json
    dataset/dataset_statistics.json
"""

import os
import glob
import json

import numpy as np
import h5py
import pandas as pd

import config as C

FIELDS = ["Q", "material", "kx", "ky", "ax", "ay", "theta", "T"]
DTYPES = {
    "Q": "float32", "material": "int8", "kx": "float32", "ky": "float32",
    "ax": "float32", "ay": "float32", "theta": "float32", "T": "float32",
}

SPLIT_FILE = {"train": "train.h5", "val": "validation.h5", "test": "test.h5"}


def load_split_arrays_and_meta(split):
    shard_files = sorted(glob.glob(os.path.join(C.CHECKPOINT_DIR, f"{split}_shard_*.npz")))
    arrays = {f: [] for f in FIELDS}
    metas = []
    for sf in shard_files:
        data = np.load(sf)
        for f in FIELDS:
            arrays[f].append(data[f])
        meta_file = sf.replace(".npz", "_meta.json")
        with open(meta_file) as fh:
            metas.extend(json.load(fh))
    for f in FIELDS:
        arrays[f] = np.concatenate(arrays[f], axis=0).astype(DTYPES[f])
    return arrays, metas


def write_h5(split, arrays, metas, target_n):
    n = arrays["Q"].shape[0]
    assert n == target_n, f"{split}: expected {target_n} samples, got {n}"
    path = os.path.join(C.DATASET_DIR, SPLIT_FILE[split])
    with h5py.File(path, "w") as h5:
        grp_in = h5.create_group("inputs")
        for name in ("Q", "material", "kx", "ky", "ax", "ay"):
            grp_in.create_dataset(name, data=arrays[name],
                                   chunks=(min(64, n), C.GRID_SIZE, C.GRID_SIZE),
                                   compression="gzip", compression_opts=4)
        grp_t = h5.create_group("targets")
        for name in ("theta", "T"):
            grp_t.create_dataset(name, data=arrays[name],
                                  chunks=(min(64, n), C.GRID_SIZE, C.GRID_SIZE),
                                  compression="gzip", compression_opts=4)

        grp_m = h5.create_group("metadata")
        # boundary condition globals + per-sample scalar diagnostics
        numeric_keys = [
            "sample_id", "seed", "grid_size", "total_power_W", "max_power_W_m2",
            "num_hotspots", "silicon_fraction", "copper_fraction",
            "dielectric_fraction", "mean_kx", "mean_ky", "mean_anisotropy",
            "h_sink", "T_ambient", "T_min", "T_max", "T_mean", "delta_T",
            "mean_theta", "n_dielectric", "n_silicon", "n_copper",
            "k_ref_dielectric", "k_ref_silicon", "k_ref_copper", "T_ref_K",
            "solver_iterations", "pde_residual_mean",
            "pde_residual_max", "heat_balance_error_pct",
            "kirchhoff_roundtrip_error_K",
        ]
        for key in numeric_keys:
            grp_m.create_dataset(key, data=np.array([m[key] for m in metas], dtype=np.float64))

        str_keys = ["hotspot_type", "boundary_type", "difficulty_level", "split"]
        dt = h5py.string_dtype(encoding="utf-8")
        for key in str_keys:
            grp_m.create_dataset(key, data=np.array([str(m[key]) for m in metas], dtype=object), dtype=dt)

        h5.attrs["grid_size"] = C.GRID_SIZE
        h5.attrs["domain_length_m"] = C.DOMAIN_LENGTH_M
        h5.attrs["dx_m"] = C.DX
        h5.attrs["n_samples"] = n
        h5.attrs["split"] = split
        h5.attrs["T_ref_K"] = C.T_REF
        h5.attrs["d_chip_m"] = C.D_CHIP
    print(f"Wrote {path}  ({n} samples, {os.path.getsize(path)/1e6:.1f} MB)")


def build_metadata_csv(all_metas):
    df = pd.DataFrame(all_metas)
    cols = [
        "sample_id", "split", "grid_size", "total_power_W", "max_power_W_m2",
        "num_hotspots", "hotspot_type", "silicon_fraction", "copper_fraction",
        "dielectric_fraction", "mean_kx", "mean_ky", "mean_anisotropy",
        "boundary_type", "h_sink", "T_ambient", "T_min", "T_max", "T_mean",
        "delta_T", "mean_theta", "n_dielectric", "n_silicon", "n_copper",
        "k_ref_dielectric", "k_ref_silicon", "k_ref_copper", "T_ref_K",
        "solver_iterations", "pde_residual_mean",
        "pde_residual_max", "heat_balance_error_pct",
        "kirchhoff_roundtrip_error_K", "valid", "difficulty_level", "seed",
    ]
    df = df[cols]
    path = os.path.join(C.DATASET_DIR, "metadata.csv")
    df.to_csv(path, index=False)
    print(f"Wrote {path}  ({len(df)} rows)")
    return df


def build_normalization_stats(train_arrays):
    stats = {}
    for name in ("Q", "kx", "ky", "ax", "ay", "theta", "T"):
        arr = train_arrays[name].astype(np.float64)
        stats[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    stats["_note"] = (
        "Computed on the TRAINING split only. Raw physical-unit arrays in "
        "the HDF5 files are NOT normalized; apply (x - mean) / std (or "
        "min-max using min/max) at data-loading time using these stats."
    )
    path = os.path.join(C.DATASET_DIR, "normalization_stats.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {path}")
    return stats


def build_dataset_statistics(split_metas, split_counts, rejections_by_split):
    all_metas = sum(split_metas.values(), [])
    T_mins = [m["T_min"] for m in all_metas]
    T_maxs = [m["T_max"] for m in all_metas]
    T_means = [m["T_mean"] for m in all_metas]
    resid_mean = [m["pde_residual_mean"] for m in all_metas]
    resid_max = [m["pde_residual_max"] for m in all_metas]
    bal_err = [m["heat_balance_error_pct"] for m in all_metas]
    rt_err = [m["kirchhoff_roundtrip_error_K"] for m in all_metas]

    from collections import Counter
    difficulty_counts = Counter(m["difficulty_level"] for m in all_metas)
    pattern_counts = Counter(m["hotspot_type"] for m in all_metas)
    boundary_counts = Counter(m["boundary_type"] for m in all_metas)

    total_rejections = sum(sum(v.values()) for v in rejections_by_split.values())
    rejection_breakdown = {}
    for split, rej in rejections_by_split.items():
        for reason, count in rej.items():
            rejection_breakdown[reason] = rejection_breakdown.get(reason, 0) + count

    stats = {
        "total_samples": sum(split_counts.values()),
        "sample_counts_by_split": split_counts,
        "grid_size": C.GRID_SIZE,
        "domain_length_m": C.DOMAIN_LENGTH_M,
        "temperature_K": {
            "min": float(np.min(T_mins)), "max": float(np.max(T_maxs)),
            "mean": float(np.mean(T_means)), "std": float(np.std(T_means)),
        },
        "pde_residual": {
            "mean_of_means": float(np.mean(resid_mean)),
            "worst_max": float(np.max(resid_max)),
        },
        "heat_balance_error_pct": {
            "mean": float(np.mean(bal_err)), "worst": float(np.max(bal_err)),
        },
        "kirchhoff_roundtrip_error_K": {
            "mean": float(np.mean(rt_err)), "worst": float(np.max(rt_err)),
        },
        "total_rejections": total_rejections,
        "rejection_breakdown": rejection_breakdown,
        "difficulty_level_counts": dict(difficulty_counts),
        "power_pattern_type_counts": dict(pattern_counts),
        "boundary_scenario_counts": dict(boundary_counts),
    }
    path = os.path.join(C.DATASET_DIR, "dataset_statistics.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {path}")
    return stats


def main():
    targets = {"train": C.N_TRAIN, "val": C.N_VAL, "test": C.N_TEST}
    split_metas = {}
    split_counts = {}
    rejections_by_split = {}
    train_arrays = None
    all_metas_flat = []

    for split, target in targets.items():
        print(f"\nLoading shards for split={split} ...")
        arrays, metas = load_split_arrays_and_meta(split)
        for m in metas:
            m["split"] = split
        write_h5(split, arrays, metas, target)
        split_metas[split] = metas
        split_counts[split] = len(metas)
        all_metas_flat.extend(metas)
        if split == "train":
            train_arrays = arrays

        prog_path = os.path.join(C.CHECKPOINT_DIR, f"{split}_progress.json")
        with open(prog_path) as f:
            prog = json.load(f)
        rejections_by_split[split] = prog.get("rejections", {})

    build_metadata_csv(all_metas_flat)
    build_normalization_stats(train_arrays)
    build_dataset_statistics(split_metas, split_counts, rejections_by_split)

    print("\n=== FINALIZATION COMPLETE ===")
    print(json.dumps(split_counts, indent=2))


if __name__ == "__main__":
    main()
