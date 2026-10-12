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


def compute_pde_residual_physical(
    T: torch.Tensor,
    Q_vol: torch.Tensor,
    kx_base: torch.Tensor,
    ky_base: torch.Tensor,
    n_map: torch.Tensor,
    h_sink: torch.Tensor,
    T_ambient: torch.Tensor,
    T_ref: float = 300.0,
    dx: float = DX,
    dy: float = DX,
    d_chip: float = D_CHIP,
) -> torch.Tensor:
    """
    Compute 2D steady-state anisotropic heat equation residual in Physical Temperature space (T-space).

    Evaluates:
        d/dx( kx(T) dT/dx ) + d/dy( ky(T) dT/dy ) - (h_sink / d_chip) * (T - T_ambient) + Q_vol = 0
    where:
        kx(T, x, y) = kx_base(x, y) * (clamp(T / T_ref, 0.5, 3.0))^(-n_map(x, y))
        ky(T, x, y) = ky_base(x, y) * (clamp(T / T_ref, 0.5, 3.0))^(-n_map(x, y))

    This matches the finite-volume discretization of solver.py directly, avoiding artificial
    Kirchhoff interface jumps across materials with different temperature exponents n.

    Returns:
        Residual tensor R_pde [B, 1, H, W] in W/m^3.
    """
    device = T.device
    Q_vol = torch.as_tensor(Q_vol, device=device, dtype=torch.float32)
    kx_base = torch.as_tensor(kx_base, device=device, dtype=torch.float32)
    ky_base = torch.as_tensor(ky_base, device=device, dtype=torch.float32)
    n_map = torch.as_tensor(n_map, device=device, dtype=torch.float32)
    h_sink = torch.as_tensor(h_sink, device=device, dtype=torch.float32)
    T_ambient = torch.as_tensor(T_ambient, device=device, dtype=torch.float32)

    while T.ndim < 4:
        T = T.unsqueeze(0)
    while Q_vol.ndim < 4:
        Q_vol = Q_vol.unsqueeze(0)
    while kx_base.ndim < 4:
        kx_base = kx_base.unsqueeze(0 if kx_base.ndim == 2 else 1)
    while ky_base.ndim < 4:
        ky_base = ky_base.unsqueeze(0 if ky_base.ndim == 2 else 1)
    while n_map.ndim < 4:
        n_map = n_map.unsqueeze(0 if n_map.ndim == 2 else 1)

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

    # Temperature-dependent conductivities matching solver.py Picard evaluation
    ratio = torch.clamp(T / T_ref, min=0.5, max=3.0)
    kx = kx_base * torch.pow(ratio, -n_map)
    ky = ky_base * torch.pow(ratio, -n_map)

    # Interior x-face conductivities (harmonic mean between adjacent x cells)
    kxf = harmonic_mean_torch(kx[:, :, :-1, :], kx[:, :, 1:, :])  # [B, 1, H-1, W]
    flux_x = kxf * (T[:, :, 1:, :] - T[:, :, :-1, :]) / dx        # [B, 1, H-1, W]
    div_x = (flux_x[:, :, 1:, :] - flux_x[:, :, :-1, :]) / dx     # [B, 1, H-2, W]

    # Interior y-face conductivities (harmonic mean between adjacent y cells)
    kyf = harmonic_mean_torch(ky[:, :, :, :-1], ky[:, :, :, 1:])  # [B, 1, H, W-1]
    flux_y = kyf * (T[:, :, :, 1:] - T[:, :, :, :-1]) / dy        # [B, 1, H, W-1]
    div_y = (flux_y[:, :, :, 1:] - flux_y[:, :, :, :-1]) / dy     # [B, 1, H, W-2]

    # Match dimensions to [B, 1, H-2, W-2] interior
    div_x_int = div_x[:, :, :, 1:-1]
    div_y_int = div_y[:, :, 1:-1, :]

    # Out-of-plane lumped heat sink loss term
    sink_term = (h_sink / d_chip) * (T[:, :, 1:-1, 1:-1] - T_ambient)

    # Volumetric heat source term Q_vol [W/m^3]
    Q_int = Q_vol[:, :, 1:-1, 1:-1]

    # Total PDE residual at interior cells
    R_int = div_x_int + div_y_int - sink_term + Q_int

    # Pad boundary cells with zeros to match full domain size [B, 1, H, W]
    R_pde = nn.functional.pad(R_int, (1, 1, 1, 1), mode="constant", value=0.0)

    return R_pde
