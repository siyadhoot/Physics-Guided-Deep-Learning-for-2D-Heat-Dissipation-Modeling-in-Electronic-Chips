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
from src.physics.pde_residual import compute_pde_residual_kirchhoff


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
        w_pde: float = 1e-4,
        w_bc: float = 1e-2,
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

        # Build n_map tensor
        n_d = float(np.mean(meta["n_dielectric"])) if isinstance(meta["n_dielectric"], (np.ndarray, list)) else float(meta["n_dielectric"])
        n_s = float(np.mean(meta["n_silicon"])) if isinstance(meta["n_silicon"], (np.ndarray, list)) else float(meta["n_silicon"])
        n_c = float(np.mean(meta["n_copper"])) if isinstance(meta["n_copper"], (np.ndarray, list)) else float(meta["n_copper"])
        T_ref = float(np.mean(meta["T_ref_K"])) if isinstance(meta["T_ref_K"], (np.ndarray, list)) else float(meta["T_ref_K"])

        n_map = torch.zeros_like(material_t, dtype=torch.float32)
        n_map[material_t == 0] = n_d
        n_map[material_t == 1] = n_s
        n_map[material_t == 2] = n_c

        # Inverse Kirchhoff transform to physical temperature T_pred [K]
        T_pred = inverse_kirchhoff_torch(theta_phys, n_map.unsqueeze(1), T_ref=T_ref)

        # Physical conductivity channels kx_base, ky_base from meta / X
        if "kx_base" in meta:
            kx_base = torch.from_numpy(np.asarray(meta["kx_base"])).to(device, dtype=torch.float32)
        else:
            kx_base = X[:, 2:3]

        if "ky_base" in meta:
            ky_base = torch.from_numpy(np.asarray(meta["ky_base"])).to(device, dtype=torch.float32)
        else:
            ky_base = X[:, 3:4]

        while kx_base.ndim < 4:
            kx_base = kx_base.unsqueeze(0)
        while ky_base.ndim < 4:
            ky_base = ky_base.unsqueeze(0)

        # Volumetric power Q_vol [W/m^3] = Q_areal [W/m^2] / d_chip (0.5mm)
        # Q channel is index 0
        Q_areal = X[:, 0:1]  # or from meta
        d_chip = 5.0e-4
        Q_vol = Q_areal * 1.0e4 / d_chip  # W/m^3 scale

        h_sink = torch.tensor(meta["h_sink"], device=device, dtype=torch.float32) if "h_sink" in meta else torch.tensor(1000.0, device=device)
        T_ambient = torch.tensor(meta["T_ambient"], device=device, dtype=torch.float32) if "T_ambient" in meta else torch.tensor(300.0, device=device)

        # Compute PDE residual in Kirchhoff space
        R_pde = compute_pde_residual_kirchhoff(
            theta=theta_phys,
            T=T_pred,
            Q_vol=Q_vol,
            kx_base=kx_base,
            ky_base=ky_base,
            h_sink=h_sink,
            T_ambient=T_ambient,
        )

        return {
            "theta_phys": theta_phys,
            "T_pred": T_pred,
            "R_pde": R_pde,
            "n_map": n_map,
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

        # Scaled PDE loss (normalized residual scale ~ 1e6 W/m^3)
        pde_loss = torch.mean((R_pde / 1.0e6) ** 2)

        # Boundary loss
        bc_loss = torch.tensor(0.0, device=X.device)
        if "bc_T_left" in meta:
            bc_loss = compute_boundary_residual_torch(
                theta_pred=phys_out["theta_phys"],
                T_pred=T_pred,
                bc_type_left=torch.tensor(meta["bc_type_left"], device=X.device) if "bc_type_left" in meta else torch.tensor(0),
                bc_type_right=torch.tensor(meta["bc_type_right"], device=X.device) if "bc_type_right" in meta else torch.tensor(0),
                bc_type_bottom=torch.tensor(meta["bc_type_bottom"], device=X.device) if "bc_type_bottom" in meta else torch.tensor(0),
                bc_type_top=torch.tensor(meta["bc_type_top"], device=X.device) if "bc_type_top" in meta else torch.tensor(0),
                bc_h_left=torch.tensor(meta.get("bc_h_left", 0.0), device=X.device),
                bc_h_right=torch.tensor(meta.get("bc_h_right", 0.0), device=X.device),
                bc_h_bottom=torch.tensor(meta.get("bc_h_bottom", 0.0), device=X.device),
                bc_h_top=torch.tensor(meta.get("bc_h_top", 0.0), device=X.device),
                bc_T_left=torch.tensor(meta.get("bc_T_left", 0.0), device=X.device),
                bc_T_right=torch.tensor(meta.get("bc_T_right", 0.0), device=X.device),
                bc_T_bottom=torch.tensor(meta.get("bc_T_bottom", 0.0), device=X.device),
                bc_T_top=torch.tensor(meta.get("bc_T_top", 0.0), device=X.device),
            )

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
        }
