"""
Unit tests for physics modules:
  - Differentiable Kirchhoff forward/inverse transformations
  - Conductivity model & physical positivity checks (k > 0)
  - Anisotropic PDE residual computation
  - Boundary condition masks & envelopes
  - PhysicsInformedFNO2d architecture & multi-loss logging
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pinn import PhysicsInformedFNO2d  # noqa: E402
from src.physics.boundary import create_boundary_distance_mask  # noqa: E402
from src.physics.conductivity import compute_temperature_dependent_conductivity_torch  # noqa: E402
from src.physics.kirchhoff_torch import (  # noqa: E402
    forward_kirchhoff_torch,
    inverse_kirchhoff_torch,
    kirchhoff_roundtrip_torch,
)
from src.physics.pde_residual import compute_pde_residual_kirchhoff  # noqa: E402


class TestPhysicsModules(unittest.TestCase):
    def setUp(self):
        self.b = 2
        self.h = 32
        self.w = 32
        self.T_ref = 300.0

    def test_kirchhoff_roundtrip_torch(self):
        """Verify T -> theta -> T_rec in PyTorch returns max abs error < 1e-4 K."""
        T_gt = torch.full((self.b, self.h, self.w), 320.0, dtype=torch.float32)
        n_map = torch.full((self.b, self.h, self.w), 1.2, dtype=torch.float32)

        max_err, theta, T_rec = kirchhoff_roundtrip_torch(T_gt, n_map, self.T_ref)
        self.assertLess(max_err.item(), 1e-4, f"Roundtrip error {max_err.item()} >= 1e-4 K")
        self.assertTrue(torch.isfinite(theta).all())
        self.assertTrue(torch.isfinite(T_rec).all())

    def test_conductivity_positivity(self):
        """Verify thermal conductivity computation enforces k > 0."""
        T = torch.full((self.b, self.h, self.w), 350.0, dtype=torch.float32)
        kx_base = torch.full((self.b, self.h, self.w), 148.0, dtype=torch.float32)
        ky_base = torch.full((self.b, self.h, self.w), 148.0, dtype=torch.float32)
        n_map = torch.full((self.b, self.h, self.w), 1.3, dtype=torch.float32)

        kx, ky = compute_temperature_dependent_conductivity_torch(T, kx_base, ky_base, n_map, self.T_ref)
        self.assertTrue((kx > 0).all(), "kx contains non-positive values")
        self.assertTrue((ky > 0).all(), "ky contains non-positive values")
        self.assertTrue(torch.isfinite(kx).all())
        self.assertTrue(torch.isfinite(ky).all())

    def test_pde_residual_shape_and_finiteness(self):
        """Verify anisotropic Kirchhoff PDE residual shape and finiteness."""
        theta = torch.randn(self.b, 1, self.h, self.w)
        T = torch.full((self.b, 1, self.h, self.w), 310.0)
        Q_vol = torch.full((self.b, 1, self.h, self.w), 1e6)
        kx_base = torch.full((self.b, 1, self.h, self.w), 148.0)
        ky_base = torch.full((self.b, 1, self.h, self.w), 148.0)
        h_sink = torch.tensor([1000.0, 1000.0])
        T_ambient = torch.tensor([300.0, 300.0])

        R_pde = compute_pde_residual_kirchhoff(
            theta=theta,
            T=T,
            Q_vol=Q_vol,
            kx_base=kx_base,
            ky_base=ky_base,
            h_sink=h_sink,
            T_ambient=T_ambient,
        )

        self.assertEqual(R_pde.shape, (self.b, 1, self.h, self.w))
        self.assertTrue(torch.isfinite(R_pde).all())

    def test_boundary_distance_mask(self):
        """Verify Dirichlet boundary distance mask B(x,y) = 0 on Dirichlet edges."""
        mask = create_boundary_distance_mask(
            grid_size=32,
            bc_types=("dirichlet", "insulated", "insulated", "dirichlet"),
        )
        self.assertEqual(mask.shape, (1, 1, 32, 32))
        self.assertEqual(mask[0, 0, :, 0].sum().item(), 0.0)  # Left Dirichlet edge = 0
        self.assertEqual(mask[0, 0, -1, :].sum().item(), 0.0) # Top Dirichlet edge = 0
        self.assertGreater(mask[0, 0, 15, 15].item(), 0.0)    # Interior = 1

    def test_pinn_architecture_and_losses(self):
        """Verify PhysicsInformedFNO2d construction, forward pass, and separate loss output."""
        model = PhysicsInformedFNO2d(
            in_channels=23,
            out_channels=1,
            width=32,
            modes1=8,
            modes2=8,
            n_layers=4,
        )
        X = torch.randn(self.b, 23, self.h, self.w)
        y = torch.randn(self.b, 1, self.h, self.w)

        meta = {
            "material": np.ones((self.b, self.h, self.w), dtype=np.int64),
            "n_dielectric": 0.2,
            "n_silicon": 1.3,
            "n_copper": 0.3,
            "T_ref_K": 300.0,
            "h_sink": np.array([1000.0, 1000.0]),
            "T_ambient": np.array([300.0, 300.0]),
            "kx_base": np.full((self.b, self.h, self.w), 148.0),
            "ky_base": np.full((self.b, self.h, self.w), 148.0),
        }

        losses = model.compute_all_losses(X, y, meta, target_mean=0.0, target_std=1.0)

        self.assertIn("data_loss", losses)
        self.assertIn("pde_loss", losses)
        self.assertIn("bc_loss", losses)
        self.assertIn("total_loss", losses)

        self.assertTrue(torch.isfinite(losses["data_loss"]).all())
        self.assertTrue(torch.isfinite(losses["pde_loss"]).all())
        self.assertTrue(torch.isfinite(losses["total_loss"]).all())


if __name__ == "__main__":
    unittest.main()
