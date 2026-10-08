"""Canonical paths for the ML data pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
GENERATOR_DIR = PROJECT_ROOT / "dataset_generator"
ML_AUX_DIR = DATASET_DIR / "ml_aux"

SPLIT_H5 = {
    "train": DATASET_DIR / "train.h5",
    "validation": DATASET_DIR / "validation.h5",
    "val": DATASET_DIR / "validation.h5",
    "test": DATASET_DIR / "test.h5",
}

# HDF5 attrs use "val"; filenames use "validation"
SPLIT_CANONICAL = {
    "train": "train",
    "validation": "validation",
    "val": "validation",
    "test": "test",
}

ORIGINAL_NORM_STATS = DATASET_DIR / "normalization_stats.json"
ML_NORM_STATS = DATASET_DIR / "ml_normalization_stats.json"


def aux_cache_path(split: str) -> Path:
    split = SPLIT_CANONICAL[split]
    return ML_AUX_DIR / f"{split}_bc_params.npz"
