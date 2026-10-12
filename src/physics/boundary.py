"""
Boundary Condition Handling and Hard Constraint Envelopes for Chip Thermal PINN.

Boundary Condition Types:
  - Dirichlet: T = T_boundary (or theta = theta_boundary)
  - Convective (Robin): -k * dT/dn = h_edge * (T - T_ambient)
  - Insulated (Neumann): dT/dn = 0

Hard Boundary Envelope:
  T(x,y) = T_b(x,y) + B(x,y) * N(x,y)
  where B(x,y) = 0 on Dirichlet boundary edges and 1 in the domain interior.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


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


def compute_boundary_residual_torch(
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
    dx: float = 3.125e-4,
) -> torch.Tensor:
    """
    Evaluate boundary condition residual across domain edges with per-sample masking.
    
    Supports:
      - Dirichlet (code 2): (T_edge - T_boundary)^2  [K^2]
      - Insulated/Neumann (code 0): (T_edge - T_interior)^2  [K^2]
      - Convective/Robin (code 1): ((flux_cond - flux_conv) / h_total)^2  [K^2]

    Coordinate orientation matches solver.py:
      - x0 (left): row 0  (T[:, :, 0, :])
      - x1 (right): row -1 (T[:, :, -1, :])
      - y0 (bottom): col 0  (T[:, :, :, 0])
      - y1 (top): col -1 (T[:, :, :, -1])

    Returns scalar boundary MSE in K^2, or 0.0 * T_pred.sum() if no active edges.
    """
    while T_pred.ndim < 4:
        T_pred = T_pred.unsqueeze(0)

    device = T_pred.device
    B = T_pred.shape[0]

    # Resolve edge arguments (prefer x0/x1/y0/y1, fallback to legacy left/right/bottom/top)
    edge_specs = [
        ("x0", bc_type_x0 if bc_type_x0 is not None else bc_type_left,
               bc_h_x0 if bc_h_x0 is not None else bc_h_left,
               bc_T_x0 if bc_T_x0 is not None else bc_T_left),
        ("x1", bc_type_x1 if bc_type_x1 is not None else bc_type_right,
               bc_h_x1 if bc_h_x1 is not None else bc_h_right,
               bc_T_x1 if bc_T_x1 is not None else bc_T_right),
        ("y0", bc_type_y0 if bc_type_y0 is not None else bc_type_bottom,
               bc_h_y0 if bc_h_y0 is not None else bc_h_bottom,
               bc_T_y0 if bc_T_y0 is not None else bc_T_bottom),
        ("y1", bc_type_y1 if bc_type_y1 is not None else bc_type_top,
               bc_h_y1 if bc_h_y1 is not None else bc_h_top,
               bc_T_y1 if bc_T_y1 is not None else bc_T_top),
    ]

    bc_residuals = []

    for edge_name, b_type_raw, b_h_raw, b_T_raw in edge_specs:
        if b_type_raw is None:
            continue

        b_type = torch.as_tensor(b_type_raw, device=device, dtype=torch.float32).view(-1)
        if b_type.numel() == 1 and B > 1:
            b_type = b_type.expand(B)

        b_h = torch.as_tensor(b_h_raw if b_h_raw is not None else 0.0, device=device, dtype=torch.float32).view(-1)
        if b_h.numel() == 1 and B > 1:
            b_h = b_h.expand(B)

        b_T = torch.as_tensor(b_T_raw if b_T_raw is not None else 0.0, device=device, dtype=torch.float32).view(-1)
        if b_T.numel() == 1 and B > 1:
            b_T = b_T.expand(B)

        # Edge lines and adjacent interior lines
        if edge_name == "x0":
            line = T_pred[:, 0, 0, :]
            adj = T_pred[:, 0, 1, :]
            k_line = kx_base[:, 0, 0, :] if (kx_base is not None and kx_base.ndim == 4) else torch.tensor(148.0, device=device)
        elif edge_name == "x1":
            line = T_pred[:, 0, -1, :]
            adj = T_pred[:, 0, -2, :]
            k_line = kx_base[:, 0, -1, :] if (kx_base is not None and kx_base.ndim == 4) else torch.tensor(148.0, device=device)
        elif edge_name == "y0":
            line = T_pred[:, 0, :, 0]
            adj = T_pred[:, 0, :, 1]
            k_line = ky_base[:, 0, :, 0] if (ky_base is not None and ky_base.ndim == 4) else torch.tensor(148.0, device=device)
        else:  # y1
            line = T_pred[:, 0, :, -1]
            adj = T_pred[:, 0, :, -2]
            k_line = ky_base[:, 0, :, -1] if (ky_base is not None and ky_base.ndim == 4) else torch.tensor(148.0, device=device)

        # 1. Dirichlet boundary: T = T_boundary (code 2)
        mask_dir = (torch.round(b_type) == 2)
        if torch.any(mask_dir):
            err_dir = torch.mean((line[mask_dir] - b_T[mask_dir].unsqueeze(1)) ** 2)
            bc_residuals.append(err_dir)

        # 2. Insulated / Neumann boundary: dT/dn = 0 (code 0)
        mask_neu = (torch.round(b_type) == 0)
        if torch.any(mask_neu):
            err_neu = torch.mean((line[mask_neu] - adj[mask_neu]) ** 2)
            bc_residuals.append(err_neu)

        # 3. Convective / Robin boundary: -k dT/dn = h (T - T_ambient) (code 1)
        mask_rob = (torch.round(b_type) == 1)
        if torch.any(mask_rob):
            k_sub = k_line[mask_rob] if (isinstance(k_line, torch.Tensor) and k_line.ndim == 2) else k_line
            h_cond = k_sub / dx
            flux_cond = h_cond * (adj[mask_rob] - line[mask_rob])
            flux_conv = b_h[mask_rob].unsqueeze(1) * (line[mask_rob] - b_T[mask_rob].unsqueeze(1))
            denom = h_cond + b_h[mask_rob].unsqueeze(1) + 1e-6
            err_rob = torch.mean(((flux_cond - flux_conv) / denom) ** 2)
            bc_residuals.append(err_rob)

    if len(bc_residuals) == 0:
        return 0.0 * T_pred.sum()

    return torch.stack(bc_residuals).mean()
