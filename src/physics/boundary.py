"""
Boundary Condition Handling and Physical Boundary-Cell Residuals for Chip Thermal PINN.

Boundary Condition Types (matching dataset_generator/solver.py):
  - Dirichlet (code 2): Prescribed edge temperature T = T_boundary at domain face
    Equivalent to h_eff = 2*k/dx, q_out = (2*k/dx) * (T_cell - T_boundary) [W/m^2]
  - Convective/Robin (code 1): Newton cooling -k*dT/dn = h_edge*(T_face - T_ambient)
    Resistance-in-series: h_eff = 1 / ((dx/2)/k + 1/h_edge) [W/(m^2.K)]
    Outward heat flux: q_out = h_eff * (T_cell - T_ambient) [W/m^2]
  - Insulated/Neumann (code 0): Zero normal heat flux across domain face (dT/dn = 0)
    h_eff = 0, q_out = 0 [W/m^2]

Energy-Balance Boundary Residual Formulation:
  For all 124 perimeter boundary cells on the 32x32 grid, energy conservation in W/m^3 is:
    R_boundary = (-div_cond + q_out*dx + (h_sink/d_chip)*(T - T_amb)*dx^2 - Q_vol*dx^2) / dx^2
  This matches the finite-volume Picard discretization in dataset_generator/solver.py:assemble_system
  identically, without double-counting the interior cells (evaluated separately by R_interior).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


def harmonic_mean_torch(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """Harmonic mean of adjacent cell conductivities."""
    return 2.0 * a * b / (a + b + eps)


def create_boundary_distance_mask(
    grid_size: int = 32,
    bc_types: Tuple[str, str, str, str] = ("insulated", "insulated", "insulated", "insulated"),
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Create boundary mask B(x,y) in [0, 1] for hard boundary constraint enforcement.
    bc_types: (x0/left, x1/right, y0/bottom, y1/top)
    B = 0 on edges with Dirichlet boundary condition; B = 1 everywhere else.
    """
    mask = torch.ones((1, 1, grid_size, grid_size), device=device, dtype=torch.float32)
    left, right, bottom, top = bc_types

    # x0 is row 0, x1 is row -1, y0 is col 0, y1 is col -1 (matching solver.py)
    if left == "dirichlet":
        mask[:, :, 0, :] = 0.0
    if right == "dirichlet":
        mask[:, :, -1, :] = 0.0
    if bottom == "dirichlet":
        mask[:, :, :, 0] = 0.0
    if top == "dirichlet":
        mask[:, :, :, -1] = 0.0

    return mask


def compute_boundary_residual_field_torch(
    theta_pred: torch.Tensor,
    T_pred: torch.Tensor,
    bc_type_left: Optional[torch.Tensor] = None,
    bc_type_right: Optional[torch.Tensor] = None,
    bc_type_bottom: Optional[torch.Tensor] = None,
    bc_type_top: Optional[torch.Tensor] = None,
    bc_h_left: Optional[torch.Tensor] = None,
    bc_h_right: Optional[torch.Tensor] = None,
    bc_h_bottom: Optional[torch.Tensor] = None,
    bc_h_top: Optional[torch.Tensor] = None,
    bc_T_left: Optional[torch.Tensor] = None,
    bc_T_right: Optional[torch.Tensor] = None,
    bc_T_bottom: Optional[torch.Tensor] = None,
    bc_T_top: Optional[torch.Tensor] = None,
    bc_type_x0: Optional[torch.Tensor] = None,
    bc_type_x1: Optional[torch.Tensor] = None,
    bc_type_y0: Optional[torch.Tensor] = None,
    bc_type_y1: Optional[torch.Tensor] = None,
    bc_h_x0: Optional[torch.Tensor] = None,
    bc_h_x1: Optional[torch.Tensor] = None,
    bc_h_y0: Optional[torch.Tensor] = None,
    bc_h_y1: Optional[torch.Tensor] = None,
    bc_T_x0: Optional[torch.Tensor] = None,
    bc_T_x1: Optional[torch.Tensor] = None,
    bc_T_y0: Optional[torch.Tensor] = None,
    bc_T_y1: Optional[torch.Tensor] = None,
    kx_base: Optional[torch.Tensor] = None,
    ky_base: Optional[torch.Tensor] = None,
    n_map: Optional[torch.Tensor] = None,
    Q_vol: Optional[torch.Tensor] = None,
    h_sink: Optional[torch.Tensor] = None,
    T_ambient: Optional[torch.Tensor] = None,
    dx: float = 3.125e-4,
    dy: float = 3.125e-4,
    d_chip: float = 5.0e-4,
    T_ref: float = 300.0,
    residual_scale: float = 1.0e6,
) -> torch.Tensor:
    """
    Evaluate physically correct boundary-cell energy balance residual across perimeter cells.

    Directly reproduces the finite-volume conservation equations from dataset_generator/solver.py:
      - x0 (West): row 0   (T[:, :, 0, :])
      - x1 (East): row -1  (T[:, :, -1, :])
      - y0 (South): col 0  (T[:, :, :, 0])
      - y1 (North): col -1 (T[:, :, :, -1])

    On each perimeter cell, conduction from interior neighbors, lateral conduction along the edge,
    outward heat escape through domain boundary faces (Robin, Dirichlet, or Neumann), out-of-plane
    package heat-sink loss, and volumetric heat sources are fully accounted for.

    Returns the physical boundary residual for the 124 perimeter cells, shape (B, 124),
    in W/m^3. residual_scale is retained for source compatibility but is applied by
    compute_boundary_residual_torch when forming the training loss.
    """
    while T_pred.ndim < 4:
        T_pred = T_pred.unsqueeze(0)

    device = T_pred.device
    dtype = T_pred.dtype
    B, _, H, W = T_pred.shape

    # Resolve conductivity base fields
    if kx_base is None:
        kx_base = torch.full((B, 1, H, W), 148.0, device=device, dtype=dtype)
    else:
        kx_base = torch.as_tensor(kx_base, device=device, dtype=dtype)
        while kx_base.ndim < 4:
            kx_base = kx_base.unsqueeze(0 if kx_base.ndim == 2 else 1)

    if ky_base is None:
        ky_base = torch.full((B, 1, H, W), 148.0, device=device, dtype=dtype)
    else:
        ky_base = torch.as_tensor(ky_base, device=device, dtype=dtype)
        while ky_base.ndim < 4:
            ky_base = ky_base.unsqueeze(0 if ky_base.ndim == 2 else 1)

    if n_map is None:
        n_map = torch.zeros((B, 1, H, W), device=device, dtype=dtype)
    else:
        n_map = torch.as_tensor(n_map, device=device, dtype=dtype)
        while n_map.ndim < 4:
            n_map = n_map.unsqueeze(0 if n_map.ndim == 2 else 1)

    if Q_vol is None:
        Q_vol = torch.zeros((B, 1, H, W), device=device, dtype=dtype)
    else:
        Q_vol = torch.as_tensor(Q_vol, device=device, dtype=dtype)
        while Q_vol.ndim < 4:
            Q_vol = Q_vol.unsqueeze(0 if Q_vol.ndim == 2 else 1)

    if h_sink is None:
        h_sink = torch.full((B, 1, 1, 1), 1000.0, device=device, dtype=dtype)
    else:
        h_sink = torch.as_tensor(h_sink, device=device, dtype=dtype).view(-1, 1, 1, 1)
        if h_sink.shape[0] == 1 and B > 1:
            h_sink = h_sink.expand(B, 1, 1, 1)

    if T_ambient is None:
        T_ambient = torch.full((B, 1, 1, 1), 300.0, device=device, dtype=dtype)
    else:
        T_ambient = torch.as_tensor(T_ambient, device=device, dtype=dtype).view(-1, 1, 1, 1)
        if T_ambient.shape[0] == 1 and B > 1:
            T_ambient = T_ambient.expand(B, 1, 1, 1)

    # 1. Temperature-dependent conductivities matching solver.py Picard evaluation
    ratio = torch.clamp(T_pred / T_ref, min=0.4, max=3.0)
    kx = kx_base * torch.pow(ratio, -n_map)
    ky = ky_base * torch.pow(ratio, -n_map)

    # 2. Interior face conductivities (harmonic mean)
    kxf = harmonic_mean_torch(kx[:, :, :-1, :], kx[:, :, 1:, :])  # [B, 1, H-1, W]
    kyf = harmonic_mean_torch(ky[:, :, :, :-1], ky[:, :, :, 1:])  # [B, 1, H, W-1]

    # 3. Net interior conduction entering each cell
    div_cond = torch.zeros((B, 1, H, W), device=device, dtype=dtype)

    flux_x = kxf * (T_pred[:, :, 1:, :] - T_pred[:, :, :-1, :])  # [B, 1, H-1, W]
    div_cond[:, :, 1:, :] -= flux_x
    div_cond[:, :, :-1, :] += flux_x

    flux_y = kyf * (T_pred[:, :, :, 1:] - T_pred[:, :, :, :-1])  # [B, 1, H, W-1]
    div_cond[:, :, :, 1:] -= flux_y
    div_cond[:, :, :, :-1] += flux_y

    # Helper to resolve edge parameters
    def resolve_edge(edge_alias: str, edge_key: str, opt_type_a, opt_type_k, opt_h_a, opt_h_k, opt_T_a, opt_T_k):
        raw_type = opt_type_k if opt_type_k is not None else opt_type_a
        raw_h = opt_h_k if opt_h_k is not None else opt_h_a
        raw_T = opt_T_k if opt_T_k is not None else opt_T_a

        if raw_type is None:
            t = torch.zeros((B, 1, 1, 1), device=device, dtype=dtype)
        else:
            t = torch.as_tensor(raw_type, device=device, dtype=dtype).view(-1, 1, 1, 1)
            if t.shape[0] == 1 and B > 1:
                t = t.expand(B, 1, 1, 1)

        if raw_h is None:
            h = torch.zeros((B, 1, 1, 1), device=device, dtype=dtype)
        else:
            h = torch.as_tensor(raw_h, device=device, dtype=dtype).view(-1, 1, 1, 1)
            if h.shape[0] == 1 and B > 1:
                h = h.expand(B, 1, 1, 1)

        if raw_T is None:
            t_env = torch.zeros((B, 1, 1, 1), device=device, dtype=dtype)
        else:
            t_env = torch.as_tensor(raw_T, device=device, dtype=dtype).view(-1, 1, 1, 1)
            if t_env.shape[0] == 1 and B > 1:
                t_env = t_env.expand(B, 1, 1, 1)

        return t, h, t_env

    # 4. Outward boundary heat escape fluxes
    bc_flux = torch.zeros((B, 1, H, W), device=device, dtype=dtype)

    # West (x0): row 0
    t_x0, h_x0, env_x0 = resolve_edge("left", "x0", bc_type_left, bc_type_x0, bc_h_left, bc_h_x0, bc_T_left, bc_T_x0)
    k_x0 = kx[:, :, 0:1, :]
    h_eff_x0 = 1.0 / ((dx / 2.0) / k_x0 + 1.0 / torch.clamp(h_x0, min=1e-6))
    coef_x0 = torch.where(torch.round(t_x0) == 2, 2.0 * k_x0,
              torch.where(torch.round(t_x0) == 1, h_eff_x0 * dx, torch.zeros_like(k_x0)))
    bc_flux[:, :, 0:1, :] += coef_x0 * (T_pred[:, :, 0:1, :] - env_x0)

    # East (x1): row -1
    t_x1, h_x1, env_x1 = resolve_edge("right", "x1", bc_type_right, bc_type_x1, bc_h_right, bc_h_x1, bc_T_right, bc_T_x1)
    k_x1 = kx[:, :, -1:, :]
    h_eff_x1 = 1.0 / ((dx / 2.0) / k_x1 + 1.0 / torch.clamp(h_x1, min=1e-6))
    coef_x1 = torch.where(torch.round(t_x1) == 2, 2.0 * k_x1,
              torch.where(torch.round(t_x1) == 1, h_eff_x1 * dx, torch.zeros_like(k_x1)))
    bc_flux[:, :, -1:, :] += coef_x1 * (T_pred[:, :, -1:, :] - env_x1)

    # South (y0): col 0
    t_y0, h_y0, env_y0 = resolve_edge("bottom", "y0", bc_type_bottom, bc_type_y0, bc_h_bottom, bc_h_y0, bc_T_bottom, bc_T_y0)
    k_y0 = ky[:, :, :, 0:1]
    h_eff_y0 = 1.0 / ((dy / 2.0) / k_y0 + 1.0 / torch.clamp(h_y0, min=1e-6))
    coef_y0 = torch.where(torch.round(t_y0) == 2, 2.0 * k_y0,
              torch.where(torch.round(t_y0) == 1, h_eff_y0 * dy, torch.zeros_like(k_y0)))
    bc_flux[:, :, :, 0:1] += coef_y0 * (T_pred[:, :, :, 0:1] - env_y0)

    # North (y1): col -1
    t_y1, h_y1, env_y1 = resolve_edge("top", "y1", bc_type_top, bc_type_y1, bc_h_top, bc_h_y1, bc_T_top, bc_T_y1)
    k_y1 = ky[:, :, :, -1:]
    h_eff_y1 = 1.0 / ((dy / 2.0) / k_y1 + 1.0 / torch.clamp(h_y1, min=1e-6))
    coef_y1 = torch.where(torch.round(t_y1) == 2, 2.0 * k_y1,
              torch.where(torch.round(t_y1) == 1, h_eff_y1 * dy, torch.zeros_like(k_y1)))
    bc_flux[:, :, :, -1:] += coef_y1 * (T_pred[:, :, :, -1:] - env_y1)

    # 5. Out-of-plane lumped heat-sink loss and volumetric power source
    sink_term = (h_sink / d_chip) * (dx * dy) * (T_pred - T_ambient)
    source_term = Q_vol * (dx * dy)

    # 6. Full cell residual in W/m^3 matching solver assemble_system
    R_cell = (-div_cond + bc_flux + sink_term - source_term) / (dx * dy)

    # Perimeter mask covering all 124 boundary cells
    boundary_mask = torch.zeros((1, 1, H, W), device=device, dtype=torch.bool)
    boundary_mask[:, :, 0, :] = True
    boundary_mask[:, :, -1, :] = True
    boundary_mask[:, :, :, 0] = True
    boundary_mask[:, :, :, -1] = True

    R_boundary = R_cell[boundary_mask.expand_as(R_cell)].view(B, -1)

    return R_boundary


def compute_boundary_residual_torch(*args, residual_scale: float = 1.0e6, **kwargs) -> torch.Tensor:
    """Existing training loss: mean squared physical residual scaled by residual_scale."""
    residual = compute_boundary_residual_field_torch(*args, residual_scale=residual_scale, **kwargs)
    return torch.mean((residual / residual_scale) ** 2)

