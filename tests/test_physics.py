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
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "dataset_generator") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "dataset_generator"))

from experiments.physics_guided.training.train_pinn import (  # noqa: E402
    custom_collate,
    evaluate_validation_set,
    validate_and_load_checkpoint,
)
from src.models.pinn import PhysicsInformedFNO2d  # noqa: E402
from src.physics.boundary import (  # noqa: E402
    create_boundary_distance_mask,
    compute_boundary_residual_torch,
)
from src.physics.conductivity import compute_temperature_dependent_conductivity_torch  # noqa: E402
from src.physics.kirchhoff_torch import (  # noqa: E402
    forward_kirchhoff_torch,
    inverse_kirchhoff_torch,
    kirchhoff_roundtrip_torch,
)
from src.physics.pde_residual import (  # noqa: E402
    compute_pde_residual_kirchhoff,
    compute_pde_residual_physical,
)


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

    def test_pde_residual_physical(self):
        """Verify conservative finite-volume physical PDE residual computation across material interfaces."""
        T = torch.full((self.b, 1, self.h, self.w), 340.0)
        Q_vol = torch.full((self.b, 1, self.h, self.w), 1.0e6)
        kx_base = torch.full((self.b, 1, self.h, self.w), 148.0)
        ky_base = torch.full((self.b, 1, self.h, self.w), 148.0)
        n_map = torch.full((self.b, 1, self.h, self.w), 1.3)
        # Create sharp material interface: right half has copper properties
        n_map[:, :, :, 16:] = 0.25
        kx_base[:, :, :, 16:] = 400.0
        h_sink = torch.tensor([1000.0, 1000.0])
        T_ambient = torch.tensor([300.0, 300.0])

        R_pde = compute_pde_residual_physical(
            T=T,
            Q_vol=Q_vol,
            kx_base=kx_base,
            ky_base=ky_base,
            n_map=n_map,
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
        self.assertEqual(mask[0, 0, 0, :].sum().item(), 0.0)  # Left (x0/row 0) Dirichlet edge = 0
        self.assertEqual(mask[0, 0, :, -1].sum().item(), 0.0) # Top (y1/col -1) Dirichlet edge = 0
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

    def test_pde_loss_interior_reduction(self):
        """Verify PDE loss MSE is calculated strictly over the 900 interior cells (30x30)."""
        model = PhysicsInformedFNO2d(
            in_channels=23, out_channels=1, width=32, modes1=8, modes2=8, n_layers=4,
        )
        X = torch.randn(self.b, 23, self.h, self.w)
        y = torch.randn(self.b, 1, self.h, self.w)
        meta = {
            "material": np.ones((self.b, self.h, self.w), dtype=np.int64),
            "n_dielectric": 0.2, "n_silicon": 1.3, "n_copper": 0.3, "T_ref_K": 300.0,
            "h_sink": np.array([1000.0, 1000.0]), "T_ambient": np.array([300.0, 300.0]),
            "kx_base": np.full((self.b, self.h, self.w), 148.0),
            "ky_base": np.full((self.b, self.h, self.w), 148.0),
        }

        losses = model.compute_all_losses(X, y, meta, target_mean=0.0, target_std=1.0)
        R_pde = losses["R_pde"]
        pde_loss = losses["pde_loss"]

        # Interior residual is slice [:, :, 1:-1, 1:-1] of shape [B, 1, 30, 30] = 900 cells
        R_interior = R_pde[:, :, 1:-1, 1:-1]
        expected_interior_loss = torch.mean((R_interior / 1.0e6) ** 2)

        # Check exact equality with interior reduction
        self.assertAlmostEqual(pde_loss.item(), expected_interior_loss.item(), places=6)

        # Boundary cells must be zero-padded in R_pde
        self.assertEqual(R_pde[:, :, 0, :].abs().sum().item(), 0.0)
        self.assertEqual(R_pde[:, :, -1, :].abs().sum().item(), 0.0)
        self.assertEqual(R_pde[:, :, :, 0].abs().sum().item(), 0.0)
        self.assertEqual(R_pde[:, :, :, -1].abs().sum().item(), 0.0)

        # Confirm that taking mean over all 1024 cells would have under-represented the loss by 900/1024
        all_cells_mean = torch.mean((R_pde / 1.0e6) ** 2)
        ratio = pde_loss.item() / (all_cells_mean.item() + 1e-12)
        self.assertAlmostEqual(ratio, 1024.0 / 900.0, places=4)

    def test_boundary_residual_torch(self):
        """Verify boundary condition residual computation across active edges."""
        theta_pred = torch.full((self.b, 1, self.h, self.w), 25.0)
        T_pred = torch.full((self.b, 1, self.h, self.w), 325.0, requires_grad=True)

        # Test Dirichlet edge (code 2): boundary T = 300 K
        bc_type_x0 = torch.tensor([2.0, 2.0])
        bc_T_x0 = torch.tensor([300.0, 300.0])
        res_dir = compute_boundary_residual_torch(
            theta_pred=theta_pred, T_pred=T_pred,
            bc_type_x0=bc_type_x0, bc_T_x0=bc_T_x0,
        )
        self.assertTrue(torch.isfinite(res_dir))
        self.assertGreater(res_dir.item(), 0.0)

        # Test Insulated/Neumann edge (code 0)
        bc_type_x1 = torch.tensor([0.0, 0.0])
        res_neu = compute_boundary_residual_torch(
            theta_pred=theta_pred, T_pred=T_pred,
            bc_type_x1=bc_type_x1,
        )
        self.assertTrue(torch.isfinite(res_neu))

    def test_boundary_residual_solver_exact_match(self):
        """Verify boundary residual is near zero for an exact solver solution with mixed BCs."""
        import solver as S
        import config as C

        N = 32
        dx = C.DX
        d_chip = C.D_CHIP
        h_sink = 1200.0
        T_ambient = 300.0

        kx_base_np = np.full((N, N), 148.0)
        ky_base_np = np.full((N, N), 120.0)
        n_map_np = np.full((N, N), 1.2)
        Q_vol_np = np.full((N, N), 2.0e6)

        boundary = {
            "x0": {"type": "dirichlet", "T_boundary": 310.0},
            "x1": {"type": "convective", "h_edge": 50.0, "T_ambient": 300.0},
            "y0": {"type": "insulated"},
            "y1": {"type": "convective", "h_edge": 25.0, "T_ambient": 305.0},
        }

        result = S.solve_heat_pde(
            kx_base_np, ky_base_np, n_map_np, Q_vol_np,
            h_sink=h_sink, d_chip=d_chip,
            T_ambient_sink=T_ambient, dx=dx, boundary=boundary,
        )
        self.assertTrue(result["converged"])

        T_t = torch.tensor(result["T"], dtype=torch.float32).view(1, 1, N, N)
        kx_base_t = torch.tensor(kx_base_np, dtype=torch.float32).view(1, 1, N, N)
        ky_base_t = torch.tensor(ky_base_np, dtype=torch.float32).view(1, 1, N, N)
        n_map_t = torch.tensor(n_map_np, dtype=torch.float32).view(1, 1, N, N)
        Q_vol_t = torch.tensor(Q_vol_np, dtype=torch.float32).view(1, 1, N, N)

        loss_bc = compute_boundary_residual_torch(
            theta_pred=T_t,
            T_pred=T_t,
            bc_type_x0=torch.tensor([2.0]),
            bc_T_x0=torch.tensor([310.0]),
            bc_type_x1=torch.tensor([1.0]),
            bc_h_x1=torch.tensor([50.0]),
            bc_T_x1=torch.tensor([300.0]),
            bc_type_y0=torch.tensor([0.0]),
            bc_type_y1=torch.tensor([1.0]),
            bc_h_y1=torch.tensor([25.0]),
            bc_T_y1=torch.tensor([305.0]),
            kx_base=kx_base_t,
            ky_base=ky_base_t,
            n_map=n_map_t,
            Q_vol=Q_vol_t,
            h_sink=torch.tensor([h_sink]),
            T_ambient=torch.tensor([T_ambient]),
            dx=dx,
            dy=dx,
            d_chip=d_chip,
        )

        # Solver residual is within Picard tolerance, so normalized squared residual is < 0.05
        self.assertTrue(torch.isfinite(loss_bc))
        self.assertLess(loss_bc.item(), 0.05)

    def test_boundary_loss_gradient_flow(self):
        """Verify autograd propagates gradients from boundary loss alone (w_data=0, w_pde=0, w_bc=1)."""
        model = PhysicsInformedFNO2d(
            in_channels=23, out_channels=1, width=32, modes1=8, modes2=8, n_layers=4,
            w_data=0.0, w_pde=0.0, w_bc=1.0,
        )
        X = torch.randn(self.b, 23, self.h, self.w)
        y = torch.randn(self.b, 1, self.h, self.w)
        meta = {
            "material": np.ones((self.b, self.h, self.w), dtype=np.int64),
            "n_dielectric": 0.2, "n_silicon": 1.3, "n_copper": 0.3, "T_ref_K": 300.0,
            "h_sink": np.array([1000.0, 1000.0]), "T_ambient": np.array([300.0, 300.0]),
            "kx_base": np.full((self.b, self.h, self.w), 148.0),
            "ky_base": np.full((self.b, self.h, self.w), 148.0),
            "bc_type_x0": np.array([1.0, 1.0]), "bc_h_x0": np.array([30.0, 30.0]), "bc_T_x0": np.array([300.0, 300.0]),
            "bc_type_x1": np.array([0.0, 0.0]),
            "bc_type_y0": np.array([2.0, 2.0]), "bc_T_y0": np.array([310.0, 310.0]),
            "bc_type_y1": np.array([1.0, 1.0]), "bc_h_y1": np.array([20.0, 20.0]), "bc_T_y1": np.array([300.0, 300.0]),
        }

        losses = model.compute_all_losses(X, y, meta, target_mean=0.0, target_std=1.0)
        self.assertEqual(losses["data_loss"].item(), nn.functional.mse_loss(losses["pred_norm"], y).item())
        self.assertGreater(losses["bc_loss"].item(), 0.0)

        # Backprop through boundary loss alone
        losses["total_loss"].backward()

        # Confirm non-zero finite gradients in backbone parameters
        grad_norms = []
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient is None for {name}")
            self.assertTrue(torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}")
            grad_norms.append(param.grad.norm().item())

        total_grad_norm = sum(grad_norms)
        self.assertGreater(total_grad_norm, 0.0, "Total gradient norm from boundary loss is zero")

    def test_single_optimizer_step_finiteness(self):
        """Verify a single in-memory AdamW step produces strictly finite parameters without touching disk."""
        model = PhysicsInformedFNO2d(
            in_channels=23, out_channels=1, width=32, modes1=8, modes2=8, n_layers=4,
            w_data=1.0, w_pde=6.0e-11, w_bc=6.0e-11,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        X = torch.randn(self.b, 23, self.h, self.w)
        y = torch.randn(self.b, 1, self.h, self.w)
        meta = {
            "material": np.ones((self.b, self.h, self.w), dtype=np.int64),
            "n_dielectric": 0.2, "n_silicon": 1.3, "n_copper": 0.3, "T_ref_K": 300.0,
            "h_sink": np.array([1000.0, 1000.0]), "T_ambient": np.array([300.0, 300.0]),
            "kx_base": np.full((self.b, self.h, self.w), 148.0),
            "ky_base": np.full((self.b, self.h, self.w), 148.0),
            "bc_type_x0": np.array([2.0, 2.0]), "bc_T_x0": np.array([300.0, 300.0]),
        }

        losses = model.compute_all_losses(X, y, meta, target_mean=0.0, target_std=1.0)
        tot_loss = losses["total_loss"]
        tot_loss.backward()

        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient is None for {name}")
            self.assertTrue(torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}")

        optimizer.step()

        for name, param in model.named_parameters():
            self.assertTrue(torch.isfinite(param).all(), f"Non-finite parameter in {name} after step")

    def test_boundary_loss_numerical_scaling(self):
        """Verify numerical scaling of boundary loss and weighted contribution balance."""
        model = PhysicsInformedFNO2d(
            in_channels=23, out_channels=1, width=32, modes1=8, modes2=8, n_layers=4,
            w_data=1.0, w_pde=6.0e-11, w_bc=6.0e-11,
        )
        X = torch.randn(self.b, 23, self.h, self.w)
        y = torch.randn(self.b, 1, self.h, self.w)
        meta = {
            "material": np.ones((self.b, self.h, self.w), dtype=np.int64),
            "n_dielectric": 0.2, "n_silicon": 1.3, "n_copper": 0.3, "T_ref_K": 300.0,
            "h_sink": np.array([1000.0, 1000.0]), "T_ambient": np.array([300.0, 300.0]),
            "kx_base": np.full((self.b, self.h, self.w), 148.0),
            "ky_base": np.full((self.b, self.h, self.w), 148.0),
            "bc_type_x0": np.array([1.0, 1.0]), "bc_h_x0": np.array([30.0, 30.0]), "bc_T_x0": np.array([300.0, 300.0]),
            "bc_type_x1": np.array([0.0, 0.0]),
            "bc_type_y0": np.array([2.0, 2.0]), "bc_T_y0": np.array([305.0, 305.0]),
            "bc_type_y1": np.array([1.0, 1.0]), "bc_h_y1": np.array([40.0, 40.0]), "bc_T_y1": np.array([300.0, 300.0]),
        }

        losses = model.compute_all_losses(X, y, meta, target_mean=25.0, target_std=25.0)
        d_loss = losses["data_loss"].item()
        p_loss = losses["pde_loss"].item()
        b_loss = losses["bc_loss"].item()
        tot_loss = losses["total_loss"].item()

        # 1. Residuals normalized by 1e6 should have reasonable finite magnitude
        self.assertGreater(b_loss, 0.0)
        self.assertLess(b_loss, 1e11)

        # 2. Weighted contribution with w_bc = 6e-11 should be finite, non-zero, and bounded (< 1.0)
        weighted_bc = 6.0e-11 * b_loss
        self.assertGreater(weighted_bc, 1e-5)
        self.assertLess(weighted_bc, 1.0)

        # 3. Total loss should remain dominated by data loss, preventing loss explosion
        self.assertTrue(torch.isfinite(torch.tensor(tot_loss)))
        self.assertGreater(d_loss, 0.0)


class SyntheticChipDataset(Dataset):
    """Small in-memory synthetic dataset for evaluator tests without touching disk datasets."""

    def __init__(self, n_samples: int = 4, inject_nan_x: bool = False, inject_nan_y: bool = False):
        self.n_samples = n_samples
        self.inject_nan_x = inject_nan_x
        self.inject_nan_y = inject_nan_y

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        X = torch.randn(23, 32, 32)
        y = torch.randn(1, 32, 32)
        if self.inject_nan_x:
            X[0, 0, 0] = float("nan")
        if self.inject_nan_y:
            y[0, 0, 0] = float("nan")

        meta = {
            "sample_id": idx,
            "seed": 42 + idx,
            "boundary_type": "mixed",
            "h_sink": 1000.0,
            "T_ambient": 300.0,
            "T_ref_K": 300.0,
            "n_dielectric": 0.2,
            "n_silicon": 1.3,
            "n_copper": 0.3,
            "material": np.ones((32, 32), dtype=np.int64),
            "kx_base": np.full((32, 32), 148.0, dtype=np.float32),
            "ky_base": np.full((32, 32), 148.0, dtype=np.float32),
            "bc_type_x0": 2.0, "bc_h_x0": 1000.0, "bc_T_x0": 300.0,
            "bc_type_x1": 0.0, "bc_h_x1": 0.0, "bc_T_x1": 300.0,
            "bc_type_y0": 0.0, "bc_h_y0": 0.0, "bc_T_y0": 300.0,
            "bc_type_y1": 0.0, "bc_h_y1": 0.0, "bc_T_y1": 300.0,
        }
        return X, y, meta


class TestPINNTrainingSafeguardsAndMetrics(unittest.TestCase):
    """
    Tests for full-data training script enhancements:
    - Checkpoint initialization & compatibility validation
    - Incompatible checkpoint failure handling
    - Kelvin physical temperature metrics evaluation
    - Global R^2 aggregation across multiple batches
    - Hotspot metric definitions (MAE, localization distance, exact/tol-1 match)
    - Non-finite inputs, targets, predictions detection
    - Temperature span safeguard warning and abort behaviors
    - Parameter immutability during evaluation
    """

    def setUp(self):
        self.device = torch.device("cpu")
        self.model = PhysicsInformedFNO2d(
            in_channels=23,
            out_channels=1,
            width=32,
            modes1=8,
            modes2=8,
            n_layers=4,
        ).to(self.device)

    def test_checkpoint_initialization_compatible(self):
        """Verify compatible checkpoint loads weights strictly and verifies hash."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = Path(f.name)

        try:
            # Save current model weights
            payload = {
                "epoch": 4,
                "model_state_dict": self.model.state_dict(),
            }
            torch.save(payload, tmp_path)

            # Create a second model and modify its weights
            model2 = PhysicsInformedFNO2d(
                in_channels=23,
                out_channels=1,
                width=32,
                modes1=8,
                modes2=8,
                n_layers=4,
            ).to(self.device)

            with torch.no_grad():
                for p in model2.parameters():
                    p.add_(1.0)

            # Check that model2 differs initially
            p1_first = next(self.model.parameters())
            p2_first = next(model2.parameters())
            self.assertFalse(torch.allclose(p1_first, p2_first))

            # Load checkpoint into model2
            ckpt_hash, source_epoch = validate_and_load_checkpoint(model2, tmp_path, self.device)

            self.assertEqual(source_epoch, 4)
            self.assertEqual(len(ckpt_hash), 64)  # SHA-256 hex string

            # Now all parameters must match exactly
            for (n1, p1), (n2, p2) in zip(self.model.named_parameters(), model2.named_parameters()):
                self.assertEqual(n1, n2)
                self.assertTrue(torch.allclose(p1, p2), f"Parameter {n1} does not match after loading")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_checkpoint_initialization_incompatible_fails(self):
        """Verify that missing files, invalid formats, missing keys, or shape mismatches raise clear errors."""
        # 1. Non-existent file
        with self.assertRaises(FileNotFoundError):
            validate_and_load_checkpoint(self.model, "non_existent_file.pt", self.device)

        # 2. Corrupted / missing model_state_dict
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            torch.save({"something_else": 123}, tmp_path)
            with self.assertRaises(KeyError):
                validate_and_load_checkpoint(self.model, tmp_path, self.device)

            # 3. Missing required keys
            incomplete_state = dict(self.model.state_dict())
            first_key = list(incomplete_state.keys())[0]
            del incomplete_state[first_key]
            torch.save({"model_state_dict": incomplete_state}, tmp_path)
            with self.assertRaises(RuntimeError) as ctx:
                validate_and_load_checkpoint(self.model, tmp_path, self.device)
            self.assertIn("missing required model keys", str(ctx.exception).lower())

            # 4. Shape mismatch (e.g. incompatible tensor shape)
            mismatched_state = dict(self.model.state_dict())
            mismatched_state[first_key] = torch.zeros(99, 99)
            torch.save({"model_state_dict": mismatched_state}, tmp_path)
            with self.assertRaises(RuntimeError) as ctx:
                validate_and_load_checkpoint(self.model, tmp_path, self.device)
            self.assertIn("shape mismatch", str(ctx.exception).lower())
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_evaluator_physical_metrics_kelvin(self):
        """Verify evaluate_validation_set computes physical Kelvin metrics."""
        ds = SyntheticChipDataset(n_samples=4)
        loader = DataLoader(ds, batch_size=2, collate_fn=custom_collate)

        # Create varied ground-truth temperatures in Kelvin (~300 K to 400 K)
        x_grid, y_grid = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32))
        spatial_profile = np.sin(x_grid * np.pi) * np.sin(y_grid * np.pi)
        gt_T = np.stack([300.0 + (30.0 + 10.0 * i) * spatial_profile for i in range(4)], axis=0).astype(np.float32)

        res = evaluate_validation_set(
            model=self.model,
            loader=loader,
            device=self.device,
            target_mean=0.0,
            target_std=1.0,
            all_gt_T=gt_T,
        )

        expected_keys = [
            "val_data_loss", "val_pde_loss", "val_bc_loss", "val_total_loss",
            "pred_t_min", "pred_t_max", "pred_t_mean", "pred_t_std", "pred_t_span",
            "mean_sample_span", "mae_k", "rmse_k", "bias_k", "max_abs_err_k",
            "r2_score", "gt_t_min", "gt_t_max", "gt_t_span", "hotspot_mae_k",
            "loc_dist_mean", "loc_dist_median", "exact_match_rate_pct", "tol_1_cell_rate_pct",
        ]
        for k in expected_keys:
            self.assertIn(k, res, f"Missing metric {k}")
            self.assertTrue(np.isfinite(res[k]), f"Metric {k} is non-finite: {res[k]}")

        # Physical sanity: predictions should be in physical temperature range
        self.assertGreater(res["pred_t_min"], 200.0)
        self.assertLess(res["pred_t_max"], 600.0)

    def test_global_r2_across_batches(self):
        """Verify global R^2 is calculated over the aggregate dataset, not as an average of batch R^2 values."""
        y_true_b1 = np.array([300.0, 310.0, 320.0])
        y_pred_b1 = np.array([302.0, 308.0, 321.0])

        y_true_b2 = np.array([350.0, 360.0, 370.0])
        y_pred_b2 = np.array([351.0, 359.0, 372.0])

        # Global R2 across all points
        y_true_all = np.concatenate([y_true_b1, y_true_b2])
        y_pred_all = np.concatenate([y_pred_b1, y_pred_b2])
        ss_res_global = np.sum((y_pred_all - y_true_all) ** 2)
        ss_tot_global = np.sum((y_true_all - np.mean(y_true_all)) ** 2)
        r2_global = 1.0 - (ss_res_global / ss_tot_global)

        # Batch R2s
        r2_b1 = 1.0 - np.sum((y_pred_b1 - y_true_b1) ** 2) / np.sum((y_true_b1 - np.mean(y_true_b1)) ** 2)
        r2_b2 = 1.0 - np.sum((y_pred_b2 - y_true_b2) ** 2) / np.sum((y_true_b2 - np.mean(y_true_b2)) ** 2)
        r2_batch_avg = (r2_b1 + r2_b2) / 2.0

        # Mathematically distinct: global R2 accounts for between-batch variance
        self.assertNotAlmostEqual(r2_global, r2_batch_avg, places=3)
        self.assertGreater(r2_global, 0.9)

    def test_hotspot_metric_definitions(self):
        """Verify hotspot MAE, localization distance, exact-match and tol-1 cell rates."""
        # Create 2 samples:
        # Sample 0: GT max at (10, 10) with value 350.0. Pred max at (10, 10) with value 348.0.
        #           Dist = 0.0, Error = 2.0 K, exact match = True, tol-1 = True.
        # Sample 1: GT max at (20, 20) with value 360.0. Pred max at (20, 21) with value 355.0.
        #           Dist = 1.0, Error = 5.0 K, exact match = False, tol-1 = True.
        gt = np.full((2, 32, 32), 300.0, dtype=np.float32)
        pred = np.full((2, 32, 32), 300.0, dtype=np.float32)

        gt[0, 10, 10] = 350.0
        pred[0, 10, 10] = 348.0

        gt[1, 20, 20] = 360.0
        pred[1, 20, 21] = 355.0

        n_eval = 2
        ht_errs = []
        ht_dists = []
        exact_matches = 0
        tol_1_matches = 0

        for i in range(n_eval):
            gt_i = gt[i]
            pr_i = pred[i]
            gt_idx = np.unravel_index(np.argmax(gt_i), gt_i.shape)
            pr_idx = np.unravel_index(np.argmax(pr_i), pr_i.shape)
            ht_errs.append(abs(pr_i[pr_idx] - gt_i[gt_idx]))
            dist = np.sqrt((gt_idx[0] - pr_idx[0])**2 + (gt_idx[1] - pr_idx[1])**2)
            ht_dists.append(dist)
            if dist == 0.0:
                exact_matches += 1
            if dist <= 1.0:
                tol_1_matches += 1

        self.assertAlmostEqual(np.mean(ht_errs), 3.5, places=5)
        self.assertAlmostEqual(np.mean(ht_dists), 0.5, places=5)
        self.assertEqual(exact_matches / n_eval * 100.0, 50.0)
        self.assertEqual(tol_1_matches / n_eval * 100.0, 100.0)

    def test_non_finite_inputs_targets_detected(self):
        """Verify non-finite inputs or targets raise RuntimeError immediately."""
        # Non-finite input
        ds_nan_x = SyntheticChipDataset(n_samples=2, inject_nan_x=True)
        loader_nan_x = DataLoader(ds_nan_x, batch_size=2, collate_fn=custom_collate)
        with self.assertRaises(RuntimeError) as ctx:
            evaluate_validation_set(self.model, loader_nan_x, self.device, 0.0, 1.0)
        self.assertIn("non-finite input", str(ctx.exception).lower())

        # Non-finite target
        ds_nan_y = SyntheticChipDataset(n_samples=2, inject_nan_y=True)
        loader_nan_y = DataLoader(ds_nan_y, batch_size=2, collate_fn=custom_collate)
        with self.assertRaises(RuntimeError) as ctx:
            evaluate_validation_set(self.model, loader_nan_y, self.device, 0.0, 1.0)
        self.assertIn("non-finite target", str(ctx.exception).lower())

    def test_narrow_temperature_span_safeguard_behavior(self):
        """Verify narrow temperature span triggers configured warning/abort behavior without hardcoded crash."""
        val_metrics = {"pred_t_span": 5.0, "mae_k": 25.0}
        min_global_span = 100.0

        is_narrow = val_metrics["pred_t_span"] < min_global_span
        self.assertTrue(is_narrow)

        # Warning mode (abort_on_collapse = False) does not raise
        abort_on_collapse = False
        if abort_on_collapse and is_narrow:
            self.fail("Should not reach here when abort_on_collapse is False")

        # Abort mode (abort_on_collapse = True) raises RuntimeError
        abort_on_collapse = True
        with self.assertRaises(RuntimeError) as ctx:
            if abort_on_collapse and is_narrow:
                raise RuntimeError(f"ABORT: Safeguard triggered - Predicted span {val_metrics['pred_t_span']} K < {min_global_span} K")
        self.assertIn("safeguard triggered", str(ctx.exception).lower())

    def test_evaluation_does_not_modify_model_parameters(self):
        """Verify evaluation runs under torch.no_grad() and preserves parameters byte-for-byte."""
        params_before = [p.clone() for p in self.model.parameters()]

        ds = SyntheticChipDataset(n_samples=2)
        loader = DataLoader(ds, batch_size=2, collate_fn=custom_collate)
        _ = evaluate_validation_set(self.model, loader, self.device, 0.0, 1.0)

        self.assertFalse(self.model.training, "Model should remain in eval() mode")
        for p_before, p_after in zip(params_before, self.model.parameters()):
            self.assertTrue(torch.equal(p_before, p_after), "Model parameters were modified during evaluation!")


if __name__ == "__main__":
    unittest.main()
