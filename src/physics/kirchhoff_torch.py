"""
Differentiable PyTorch and NumPy Kirchhoff transformations for heat dissipation modeling.

Forward Kirchhoff:
    theta(T) = integral_{T_ref}^{T} k(tau)/k_ref d(tau)
             = T_ref / (1 - n) * [ (T / T_ref)^(1 - n) - 1 ]     (n != 1)
             = T_ref * ln(T / T_ref)                             (n == 1)

Inverse Kirchhoff:
    T(theta) = T_ref * [ 1 + (1 - n) * theta / T_ref ]^(1 / (1 - n))  (n != 1)
    T(theta) = T_ref * exp(theta / T_ref)                             (n == 1)
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

N_EQ_1_TOL = 1e-3


def forward_kirchhoff_torch(
    T: torch.Tensor,
    n_map: torch.Tensor,
    T_ref: Union[float, torch.Tensor] = 300.0,
) -> torch.Tensor:
    """
    Forward Kirchhoff transformation in PyTorch (differentiable).

    T      : (..., H, W) Temperature tensor in Kelvin
    n_map  : (..., H, W) Spatially varying material exponent n
    T_ref  : Reference temperature in Kelvin
    """
    near_one = (torch.abs(n_map - 1.0) < N_EQ_1_TOL)
    far = ~near_one

    theta = torch.zeros_like(T)

    if torch.any(far):
        n_f = n_map[far]
        T_f = T[far]
        theta[far] = (T_ref / (1.0 - n_f)) * (torch.pow(torch.clamp(T_f / T_ref, min=1e-4), 1.0 - n_f) - 1.0)

    if torch.any(near_one):
        theta[near_one] = T_ref * torch.log(torch.clamp(T[near_one] / T_ref, min=1e-4))

    return theta


def inverse_kirchhoff_torch(
    theta: torch.Tensor,
    n_map: torch.Tensor,
    T_ref: Union[float, torch.Tensor] = 300.0,
) -> torch.Tensor:
    """
    Inverse Kirchhoff transformation in PyTorch (differentiable).

    theta  : (..., H, W) Kirchhoff variable tensor
    n_map  : (..., H, W) Spatially varying material exponent n
    T_ref  : Reference temperature in Kelvin
    """
    near_one = (torch.abs(n_map - 1.0) < N_EQ_1_TOL)
    far = ~near_one

    T = torch.zeros_like(theta)

    if torch.any(far):
        n_f = n_map[far]
        th_f = theta[far]
        base = 1.0 + (1.0 - n_f) * th_f / T_ref
        base_clamped = torch.clamp(base, min=1e-6)
        T[far] = T_ref * torch.pow(base_clamped, 1.0 / (1.0 - n_f))

    if torch.any(near_one):
        T[near_one] = T_ref * torch.exp(theta[near_one] / T_ref)

    return T


def kirchhoff_roundtrip_torch(
    T: torch.Tensor,
    n_map: torch.Tensor,
    T_ref: float = 300.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Verify round-trip T -> theta -> T_rec in PyTorch."""
    theta = forward_kirchhoff_torch(T, n_map, T_ref)
    T_rec = inverse_kirchhoff_torch(theta, n_map, T_ref)
    max_err = torch.max(torch.abs(T - T_rec))
    return max_err, theta, T_rec
