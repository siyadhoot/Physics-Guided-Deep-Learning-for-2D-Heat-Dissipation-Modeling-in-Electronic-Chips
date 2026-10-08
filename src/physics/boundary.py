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
    bc_types: (left, right, bottom, top)
    B = 0 on edges with Dirichlet boundary condition; B = 1 everywhere else.
    """
    mask = torch.ones((1, 1, grid_size, grid_size), device=device, dtype=torch.float32)
    left, right, bottom, top = bc_types

    if left == "dirichlet":
        mask[:, :, :, 0] = 0.0
    if right == "dirichlet":
        mask[:, :, :, -1] = 0.0
    if bottom == "dirichlet":
        mask[:, :, 0, :] = 0.0
    if top == "dirichlet":
        mask[:, :, -1, :] = 0.0

    return mask


def compute_boundary_residual_torch(
    theta_pred: torch.Tensor,
    T_pred: torch.Tensor,
    bc_type_left: torch.Tensor,
    bc_type_right: torch.Tensor,
    bc_type_bottom: torch.Tensor,
    bc_type_top: torch.Tensor,
    bc_h_left: torch.Tensor,
    bc_h_right: torch.Tensor,
    bc_h_bottom: torch.Tensor,
    bc_h_top: torch.Tensor,
    bc_T_left: torch.Tensor,
    bc_T_right: torch.Tensor,
    bc_T_bottom: torch.Tensor,
    bc_T_top: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate boundary condition residual across domain edges.
    Returns scalar boundary error MSE across Dirichlet/convective edges.
    """
    b, c, h, w = theta_pred.shape
    bc_residuals = []

    # Left edge (col 0)
    # If bc_T_left > 0 and bc_type_left indicates Dirichlet, evaluate |T_pred[:, 0] - bc_T_left|
    if (bc_T_left > 0).any():
        err_left = torch.mean((T_pred[:, :, :, 0] - bc_T_left.view(-1, 1, 1)) ** 2)
        bc_residuals.append(err_left)

    # Right edge (col -1)
    if (bc_T_right > 0).any():
        err_right = torch.mean((T_pred[:, :, :, -1] - bc_T_right.view(-1, 1, 1)) ** 2)
        bc_residuals.append(err_right)

    # Bottom edge (row 0)
    if (bc_T_bottom > 0).any():
        err_bottom = torch.mean((T_pred[:, :, 0, :] - bc_T_bottom.view(-1, 1, 1)) ** 2)
        bc_residuals.append(err_bottom)

    # Top edge (row -1)
    if (bc_T_top > 0).any():
        err_top = torch.mean((T_pred[:, :, -1, :] - bc_T_top.view(-1, 1, 1)) ** 2)
        bc_residuals.append(err_top)

    if len(bc_residuals) == 0:
        return torch.tensor(0.0, device=theta_pred.device)

    return torch.stack(bc_residuals).mean()
