"""
Physics PDE Residual Computation for 2D Steady-State Anisotropic Heat Conduction.

PDE in Physical Space:
    d/dx ( kx(T,x,y) dT/dx ) + d/dy ( ky(T,x,y) dT/dy ) - (h_sink / d_chip) * (T - T_ambient) + Q_vol = 0

PDE in Kirchhoff Space:
    d/dx ( kx_base(x,y) d_theta/dx ) + d/dy ( ky_base(x,y) d_theta/dy ) - (h_sink / d_chip) * (T(theta) - T_ambient) + Q_vol = 0

Units: W/m^3
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

# Constants from dataset_generator/config.py
D_CHIP = 5.0e-4               # [m] 0.5 mm chip thickness
DOMAIN_LENGTH_M = 0.01        # [m] 10 mm chip width
GRID_SIZE = 32                # 32x32 grid
DX = DOMAIN_LENGTH_M / GRID_SIZE  # ~3.125e-4 m


def harmonic_mean_torch(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """Harmonic mean of face conductivities in PyTorch."""
    return 2.0 * a * b / (a + b + eps)


def compute_pde_residual_kirchhoff(
    theta: torch.Tensor,
    T: torch.Tensor,
    Q_vol: torch.Tensor,
    kx_base: torch.Tensor,
    ky_base: torch.Tensor,
    h_sink: torch.Tensor,
    T_ambient: torch.Tensor,
    dx: float = DX,
    dy: float = DX,
    d_chip: float = D_CHIP,
) -> torch.Tensor:
    """
    Compute 2D steady-state anisotropic heat equation residual in Kirchhoff space.

    Input shapes: [B, 1, H, W] or [B, H, W]
    Returns:
      Residual tensor R_pde [B, 1, H, W] in W/m^3.
    """
    while theta.ndim < 4:
        theta = theta.unsqueeze(0)
    while T.ndim < 4:
        T = T.unsqueeze(0)
    while Q_vol.ndim < 4:
        Q_vol = Q_vol.unsqueeze(0)
    while kx_base.ndim < 4:
        kx_base = kx_base.unsqueeze(0)
    while ky_base.ndim < 4:
        ky_base = ky_base.unsqueeze(0)

    if h_sink.ndim == 0:
        h_sink = h_sink.view(1, 1, 1, 1)
    elif h_sink.ndim == 1:
        h_sink = h_sink.view(-1, 1, 1, 1)
    elif h_sink.ndim == 3:
        h_sink = h_sink.unsqueeze(1)

    if T_ambient.ndim == 0:
        T_ambient = T_ambient.view(1, 1, 1, 1)
    elif T_ambient.ndim == 1:
        T_ambient = T_ambient.view(-1, 1, 1, 1)
    elif T_ambient.ndim == 3:
        T_ambient = T_ambient.unsqueeze(1)

    b, c, h, w = theta.shape

    # Interior x-face conductivities (harmonic mean between adjacent x cells)
    kxf = harmonic_mean_torch(kx_base[:, :, :-1, :], kx_base[:, :, 1:, :])  # [B, 1, H-1, W]
    # Fluxes across x-faces: kxf * (theta[i+1] - theta[i]) / dx
    flux_x = kxf * (theta[:, :, 1:, :] - theta[:, :, :-1, :]) / dx  # [B, 1, H-1, W]

    # Flux divergence in x at interior cells [B, 1, H-2, W]
    div_x = (flux_x[:, :, 1:, :] - flux_x[:, :, :-1, :]) / dx

    # Interior y-face conductivities (harmonic mean between adjacent y cells)
    kyf = harmonic_mean_torch(ky_base[:, :, :, :-1], ky_base[:, :, :, 1:])  # [B, 1, H, W-1]
    # Fluxes across y-faces: kyf * (theta[j+1] - theta[j]) / dy
    flux_y = kyf * (theta[:, :, :, 1:] - theta[:, :, :, :-1]) / dy  # [B, 1, H, W-1]

    # Flux divergence in y at interior cells [B, 1, H, W-2]
    div_y = (flux_y[:, :, :, 1:] - flux_y[:, :, :, :-1]) / dy

    # Match dimensions to [B, 1, H-2, W-2] interior
    div_x_int = div_x[:, :, :, 1:-1]
    div_y_int = div_y[:, :, 1:-1, :]

    # Out-of-plane lumped heat sink loss term
    sink_term = (h_sink / d_chip) * (T[:, :, 1:-1, 1:-1] - T_ambient)

    # Volumetric heat source term Q_vol [W/m^3]
    Q_int = Q_vol[:, :, 1:-1, 1:-1]

    # Total PDE residual at interior cells: div_x + div_y - sink + Q_vol
    R_int = div_x_int + div_y_int - sink_term + Q_int

    # Pad boundary cells with zeros to match full domain size [B, 1, H, W]
    R_pde = nn.functional.pad(R_int, (1, 1, 1, 1), mode="constant", value=0.0)

    return R_pde
