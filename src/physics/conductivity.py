"""
Modular Thermal Conductivity Model for 2D Electronic Chip Thermal Simulation.

Supports:
  1. Constant isotropic / anisotropic conductivity
  2. Spatially varying base conductivity kx_base(x,y), ky_base(x,y)
  3. Temperature-dependent conductivity k(T, x, y) = k_base(x,y) * (T / T_ref)^(-n(x,y))

Ensures physical constraint k > 0 for all valid inputs.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch


def compute_temperature_dependent_conductivity_torch(
    T: torch.Tensor,
    kx_base: torch.Tensor,
    ky_base: torch.Tensor,
    n_map: torch.Tensor,
    T_ref: float = 300.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute temperature-dependent anisotropic conductivity in PyTorch.

    kx(T,x,y) = kx_base(x,y) * (T / T_ref)^(-n(x,y))
    ky(T,x,y) = ky_base(x,y) * (T / T_ref)^(-n(x,y))

    Returns:
      (kx, ky) tensors, with strict check k > 0 (clamped at eps).
    """
    T_ratio = torch.clamp(T / T_ref, min=1e-4)
    factor = torch.pow(T_ratio, -n_map)

    kx = torch.clamp(kx_base * factor, min=eps)
    ky = torch.clamp(ky_base * factor, min=eps)

    if (kx <= 0).any() or (ky <= 0).any():
        raise ValueError("CRITICAL: Non-positive thermal conductivity detected (k <= 0)!")

    return kx, ky


def compute_temperature_dependent_conductivity_numpy(
    T: np.ndarray,
    kx_base: np.ndarray,
    ky_base: np.ndarray,
    n_map: np.ndarray,
    T_ref: float = 300.0,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """NumPy variant for thermal conductivity evaluation."""
    T_ratio = np.clip(T / T_ref, 1e-4, None)
    factor = np.power(T_ratio, -n_map)

    kx = np.clip(kx_base * factor, eps, None)
    ky = np.clip(ky_base * factor, eps, None)

    if np.any(kx <= 0) or np.any(ky <= 0):
        raise ValueError("CRITICAL: Non-positive thermal conductivity detected in NumPy calculation!")

    return kx, ky
