"""
PyTorch Dataset / DataLoader for leakage-free chip thermal inputs.

Reads existing HDF5 ground-truth files and constructs FNO inputs dynamically:
  Q, material, ax, ay, k_ref_*  →  kx_base, ky_base
  seed-reconstructed aux cache  →  BC channels + actual n_*
  train-only stats              →  normalization

Never loads converged kx/ky, T, or theta into X.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .build_aux_and_stats import edge_params_from_aux, load_aux_cache
from .channels import (
    CHANNEL_NAMES,
    N_INPUT_CHANNELS,
    TARGET_NAME,
    build_input_channels,
    build_kx_ky_base,
)
from .normalize import apply_normalization, apply_target_normalization, load_ml_norm_stats
from .paths import SPLIT_CANONICAL, SPLIT_H5


class ChipThermalDataset(Dataset):
    """
    Leakage-free thermal dataset.

    Parameters
    ----------
    split : {'train', 'validation'/'val', 'test'}
    normalize : if True, apply train-only z-score to continuous channels
                and to the theta target.
    return_meta : if True, also return a dict of scalars useful for
                  inverse Kirchhoff / debugging.
    """

    def __init__(
        self,
        split: str = "train",
        normalize: bool = True,
        return_meta: bool = False,
        h5_path: Optional[Union[str, Path]] = None,
        norm_stats: Optional[dict] = None,
    ):
        self.split = SPLIT_CANONICAL[split]
        self.h5_path = Path(h5_path) if h5_path is not None else SPLIT_H5[self.split]
        self.normalize = normalize
        self.return_meta = return_meta
        self.channel_names = list(CHANNEL_NAMES)
        self.target_name = TARGET_NAME

        self.aux = load_aux_cache(self.split)
        self.norm_stats = norm_stats if norm_stats is not None else (
            load_ml_norm_stats() if normalize else None
        )

        self._h5: Optional[h5py.File] = None
        with h5py.File(self.h5_path, "r") as f:
            self.n = int(f["inputs/Q"].shape[0])
            self.grid_size = int(f["inputs/Q"].shape[1])
            # Cheap integrity checks (no full load)
            assert f["inputs/material"].shape[0] == self.n
            assert f["targets/theta"].shape[0] == self.n
            assert len(self.aux["seed"]) == self.n

    def _ensure_open(self) -> h5py.File:
        # Lazy open: required for DataLoader num_workers > 0 on Windows
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        f = self._ensure_open()

        Q = f["inputs/Q"][idx]
        material = f["inputs/material"][idx]
        ax = f["inputs/ax"][idx]
        ay = f["inputs/ay"][idx]
        theta = f["targets/theta"][idx]

        k_ref_d = float(f["metadata/k_ref_dielectric"][idx])
        k_ref_s = float(f["metadata/k_ref_silicon"][idx])
        k_ref_c = float(f["metadata/k_ref_copper"][idx])
        T_ref = float(f["metadata/T_ref_K"][idx])

        # Base conductivities from material + k_ref + anisotropy — NOT from kx/ky
        kx_base, ky_base = build_kx_ky_base(
            material, ax, ay, k_ref_d, k_ref_s, k_ref_c
        )

        edge_params = edge_params_from_aux(self.aux, idx)
        h_sink = float(self.aux["h_sink"][idx])
        T_ambient = float(self.aux["T_ambient"][idx])
        n_d = float(self.aux["n_dielectric"][idx])
        n_s = float(self.aux["n_silicon"][idx])  # actual (corrected)
        n_c = float(self.aux["n_copper"][idx])

        X = build_input_channels(
            Q=Q,
            material=material,
            ax=ax,
            ay=ay,
            kx_base=kx_base,
            ky_base=ky_base,
            edge_params=edge_params,
            h_sink=h_sink,
            T_ambient=T_ambient,
            n_dielectric=n_d,
            n_silicon=n_s,
            n_copper=n_c,
        )
        y = np.asarray(theta, dtype=np.float32)[None, ...]  # (1, H, W)

        if self.normalize:
            assert self.norm_stats is not None
            X = apply_normalization(X, self.norm_stats)
            y = apply_target_normalization(y, self.norm_stats)

        X_t = torch.from_numpy(np.ascontiguousarray(X))
        y_t = torch.from_numpy(np.ascontiguousarray(y))

        if not self.return_meta:
            return X_t, y_t

        meta = {
            "sample_id": int(f["metadata/sample_id"][idx]),
            "seed": int(self.aux["seed"][idx]),
            "h_sink": h_sink,
            "T_ambient": T_ambient,
            "T_ref_K": T_ref,
            "n_dielectric": n_d,
            "n_silicon": n_s,
            "n_copper": n_c,
            "k_ref_dielectric": k_ref_d,
            "k_ref_silicon": k_ref_s,
            "k_ref_copper": k_ref_c,
            "boundary_type": str(self.aux["boundary_type"][idx]),
            "channel_names": self.channel_names,
            "kx_base": kx_base,
            "ky_base": ky_base,
            "material": np.asarray(material),
        }
        return X_t, y_t, meta

    def __getstate__(self):
        # Don't pickle open HDF5 handles
        state = self.__dict__.copy()
        state["_h5"] = None
        return state


def make_dataloader(
    split: str,
    batch_size: int = 32,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    normalize: bool = True,
    return_meta: bool = False,
    **kwargs,
) -> DataLoader:
    if shuffle is None:
        shuffle = SPLIT_CANONICAL[split] == "train"
    ds = ChipThermalDataset(
        split=split, normalize=normalize, return_meta=return_meta
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs,
    )
