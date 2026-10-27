"""
Physics-Informed Fourier Neural Operator (PhysicsInformedFNO2d) for 2D Heat Dissipation.

Pipeline:
  Input X [B, 23, 32, 32] + Coordinates [B, 2, 32, 32]
      ↓
  FNO2d Backbone (width=32, modes=8x8, 4 layers)
      ↓
  Predicted Normalized Kirchhoff Potential theta_norm [B, 1, 32, 32]
      ↓
  Denormalize theta_norm -> theta_phys [B, 1, 32, 32]
      ↓
  Analytical Inverse Kirchhoff Transform -> Temperature T_pred [B, 1, 32, 32]
      ↓
  Physics Evaluation:
    - PDE Residual R_pde [B, 1, 32, 32] (W/m^3)
    - Boundary Condition Error
    - Separate Loss Components (data_loss, pde_loss, bc_loss, total_loss)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.models.fno import FNO2d
from src.physics.boundary import compute_boundary_residual_torch
from src.physics.kirchhoff_torch import inverse_kirchhoff_torch
from src.physics.pde_residual import compute_pde_residual_kirchhoff, compute_pde_residual_physical


class PhysicsInformedFNO2d(nn.Module):
    """
    Physics-Informed 2D Fourier Neural Operator.

    Integrates:
      - Supervised data loss on theta
      - Anisotropic Kirchhoff-space PDE residual loss
      - Boundary condition loss
      - Inverse Kirchhoff transformation for physical temperature T evaluation
    """

    def __init__(
        self,
        in_channels: int = 23,
        out_channels: int = 1,
        width: int = 32,
        modes1: int = 8,
        modes2: int = 8,
        n_layers: int = 4,
        use_grid: bool = True,
        w_data: float = 1.0,
        w_pde: float = 6.0e-11,
        w_bc: float = 6.0e-11,
    ):
        super().__init__()
        self.fno = FNO2d(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            modes1=modes1,
            modes2=modes2,
            n_layers=n_layers,
            use_grid=use_grid,
        )
        self.w_data = w_data
        self.w_pde = w_pde
        self.w_bc = w_bc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting normalized theta_norm [B, 1, H, W]."""
        return self.fno(x)

    def count_parameters(self, trainable_only: bool = True) -> int:
        return self.fno.count_parameters(trainable_only=trainable_only)

    def compute_physics_outputs(
        self,
        X: torch.Tensor,
        pred_theta_norm: torch.Tensor,
        meta: Dict[str, Any],
        target_mean: float,
        target_std: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Denormalize theta, perform inverse Kirchhoff to get T, and compute PDE residual.
        """
        device = X.device

        # 1. Denormalize theta: theta_phys = pred_theta_norm * std + mean
        theta_phys = pred_theta_norm * target_std + target_mean

        # 2. Extract physical parameter channels from input X [B, 23, 32, 32]
        # Channel 0: Q (normalized) -> extract physical Q_vol [W/m^3]
        # Channel 2: kx_base, Channel 3: ky_base
        # Meta contains scalar params: n_dielectric, n_silicon, n_copper, h_sink, T_ambient, T_ref_K
        material = meta["material"]  # numpy or tensor
        if isinstance(material, np.ndarray):
            material_t = torch.from_numpy(material).to(device)
        else:
            material_t = material.to(device)

        if material_t.ndim == 2:
            material_t = material_t.unsqueeze(0)

        # Build vectorized n_map tensor per-sample on device (Defect 4 fix)
        nd = torch.as_tensor(meta["n_dielectric"], device=device, dtype=torch.float32).view(-1, 1, 1)
        ns = torch.as_tensor(meta["n_silicon"], device=device, dtype=torch.float32).view(-1, 1, 1)
        nc = torch.as_tensor(meta["n_copper"], device=device, dtype=torch.float32).view(-1, 1, 1)
        if "T_ref_K" in meta:
            T_ref_val = meta["T_ref_K"]
            T_ref = float(T_ref_val) if isinstance(T_ref_val, (int, float)) else float(np.mean(T_ref_val))
        else:
            T_ref = 300.0

        n_map = torch.where(material_t == 0, nd, torch.where(material_t == 1, ns, nc))

        # Inverse Kirchhoff transform to physical temperature T_pred [K]
        T_pred = inverse_kirchhoff_torch(theta_phys, n_map.unsqueeze(1), T_ref=T_ref)

        # Physical conductivity channels kx_base, ky_base from meta / X
        if "kx_base" in meta:
            kx_base = torch.as_tensor(meta["kx_base"], device=device, dtype=torch.float32)
        else:
            kx_base = X[:, 2:3]

        if "ky_base" in meta:
            ky_base = torch.as_tensor(meta["ky_base"], device=device, dtype=torch.float32)
        else:
            ky_base = X[:, 3:4]

        while kx_base.ndim < 4:
            kx_base = kx_base.unsqueeze(0 if kx_base.ndim == 2 else 1)
        while ky_base.ndim < 4:
            ky_base = ky_base.unsqueeze(0 if ky_base.ndim == 2 else 1)

        # Volumetric power Q_vol [W/m^3] = Q_areal [W/m^2] / d_chip (0.5mm) (Defect 1 fix)
        d_chip = 5.0e-4
        if "Q_areal" in meta:
            Q_areal = torch.as_tensor(meta["Q_areal"], device=device, dtype=torch.float32)
            while Q_areal.ndim < 4:
                Q_areal = Q_areal.unsqueeze(1)
        else:
            # Denormalize input channel 0 using dataset stats (mean=45038.5, std=97418.0)
            # Physical Q_areal is non-negative
            Q_mean = 45038.5
            Q_std = 97418.0
            Q_areal = torch.clamp(X[:, 0:1] * Q_std + Q_mean, min=0.0)

        Q_vol = Q_areal / d_chip

        h_sink = torch.as_tensor(meta.get("h_sink", 1000.0), device=device, dtype=torch.float32)
        T_ambient = torch.as_tensor(meta.get("T_ambient", 300.0), device=device, dtype=torch.float32)

        # Compute PDE residual in Physical space (Conservative Finite-Volume matching solver.py)
        R_pde = compute_pde_residual_physical(
            T=T_pred,
            Q_vol=Q_vol,
            kx_base=kx_base,
            ky_base=ky_base,
            n_map=n_map,
            h_sink=h_sink,
            T_ambient=T_ambient,
            T_ref=T_ref,
            d_chip=d_chip,
        )

        return {
            "theta_phys": theta_phys,
            "T_pred": T_pred,
            "R_pde": R_pde,
            "n_map": n_map,
            "Q_vol": Q_vol,
            "kx_base": kx_base,
            "ky_base": ky_base,
        }

    def compute_all_losses(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        meta: Dict[str, Any],
        target_mean: float,
        target_std: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute separate data, PDE, boundary, and total loss components.
        """
        # 1. Forward pass
        pred_norm = self.forward(X)

        # 2. Supervised data loss on theta
        data_loss = nn.functional.mse_loss(pred_norm, y)

        # 3. Physics outputs & PDE residual
        phys_out = self.compute_physics_outputs(X, pred_norm, meta, target_mean, target_std)
        R_pde = phys_out["R_pde"]
        T_pred = phys_out["T_pred"]

        # Scaled PDE loss over physically defined interior cells (30x30 = 900 cells)
        # Excludes artificially zero-padded perimeter boundary cells from the MSE denominator
        R_interior = R_pde[:, :, 1:-1, 1:-1]
        pde_loss = torch.mean((R_interior / 1.0e6) ** 2)

        # 4. Boundary loss with per-sample masking and edge alignment
        has_bc = any(k in meta for k in (
            "bc_type_left", "bc_type_x0", "bc_T_left", "bc_T_x0"
        ))
        if has_bc:
            d_chip = 5.0e-4
            if "T_ref_K" in meta:
                T_ref_val = meta["T_ref_K"]
                T_ref = float(T_ref_val) if isinstance(T_ref_val, (int, float)) else float(np.mean(T_ref_val))
            else:
                T_ref = 300.0

            bc_loss = compute_boundary_residual_torch(
                theta_pred=phys_out["theta_phys"],
                T_pred=T_pred,
                bc_type_left=meta.get("bc_type_left", meta.get("bc_type_x0")),
                bc_type_right=meta.get("bc_type_right", meta.get("bc_type_x1")),
                bc_type_bottom=meta.get("bc_type_bottom", meta.get("bc_type_y0")),
                bc_type_top=meta.get("bc_type_top", meta.get("bc_type_y1")),
                bc_h_left=meta.get("bc_h_left", meta.get("bc_h_x0")),
                bc_h_right=meta.get("bc_h_right", meta.get("bc_h_x1")),
                bc_h_bottom=meta.get("bc_h_bottom", meta.get("bc_h_y0")),
                bc_h_top=meta.get("bc_h_top", meta.get("bc_h_y1")),
                bc_T_left=meta.get("bc_T_left", meta.get("bc_T_x0")),
                bc_T_right=meta.get("bc_T_right", meta.get("bc_T_x1")),
                bc_T_bottom=meta.get("bc_T_bottom", meta.get("bc_T_y0")),
                bc_T_top=meta.get("bc_T_top", meta.get("bc_T_y1")),
                kx_base=phys_out["kx_base"],
                ky_base=phys_out["ky_base"],
                n_map=phys_out["n_map"],
                Q_vol=phys_out["Q_vol"],
                h_sink=meta.get("h_sink", 1000.0),
                T_ambient=meta.get("T_ambient", 300.0),
                dx=3.125e-4,
                dy=3.125e-4,
                d_chip=d_chip,
                T_ref=T_ref,
                residual_scale=1.0e6,
            )
        else:
            bc_loss = 0.0 * T_pred.sum()

        total_loss = self.w_data * data_loss + self.w_pde * pde_loss + self.w_bc * bc_loss

        return {
            "data_loss": data_loss,
            "pde_loss": pde_loss,
            "bc_loss": bc_loss,
            "total_loss": total_loss,
            "pred_norm": pred_norm,
            "T_pred": T_pred,
            "R_pde": R_pde,
            "n_map": phys_out["n_map"],
            "Q_vol": phys_out["Q_vol"],
        }
