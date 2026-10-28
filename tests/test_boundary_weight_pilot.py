import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.physics_guided.evaluate_boundary_pilot import ARM_WEIGHTS, temperature_metrics
from experiments.physics_guided.training.train_pinn import component_gradient_diagnostics, safe_gradient_ratio
from src.physics.boundary import compute_boundary_residual_field_torch, compute_boundary_residual_torch


class TestBoundaryPilot(unittest.TestCase):
    def test_arm_weights_controlled(self):
        self.assertEqual(ARM_WEIGHTS["A"], {"w_data": 1.0, "w_pde": 0.0, "w_bc": 0.0})
        self.assertEqual(ARM_WEIGHTS["B"]["w_data"], ARM_WEIGHTS["C"]["w_data"])
        self.assertEqual(ARM_WEIGHTS["B"]["w_pde"], ARM_WEIGHTS["C"]["w_pde"])
        self.assertNotEqual(ARM_WEIGHTS["B"]["w_bc"], ARM_WEIGHTS["C"]["w_bc"])
        losses = {"pde": 12.0, "bc": 3.0}
        self.assertEqual(ARM_WEIGHTS["A"]["w_pde"] * losses["pde"], 0.0)
        self.assertEqual(ARM_WEIGHTS["A"]["w_bc"] * losses["bc"], 0.0)

    def test_boundary_residual_shape_mask_and_loss_semantics(self):
        b, h, w = 2, 32, 32
        T = torch.full((b, 1, h, w), 310.0, requires_grad=True)
        args = dict(T_pred=T, theta_pred=T, kx_base=torch.ones_like(T) * 100,
                    ky_base=torch.ones_like(T) * 100, n_map=torch.zeros_like(T),
                    Q_vol=torch.ones_like(T) * 1e6, h_sink=torch.tensor([1000., 1000.]),
                    T_ambient=torch.tensor([300., 300.]))
        residual = compute_boundary_residual_field_torch(**args)
        loss = compute_boundary_residual_torch(**args)
        self.assertEqual(tuple(residual.shape), (b, 124))
        mask = torch.zeros(h, w, dtype=torch.bool)
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        self.assertEqual(int(mask.sum()), 124)
        torch.testing.assert_close(loss, torch.mean((residual / 1e6) ** 2))

    def test_temperature_metrics_masks_and_aggregation(self):
        true = np.zeros((2, 32, 32), dtype=np.float64)
        pred = true.copy()
        pred[:, 0, :] = 2.0
        result = temperature_metrics(pred, true)
        self.assertAlmostEqual(result["boundary_mae_k"], 64.0 / 124.0)
        self.assertEqual(result["interior_mae_k"], 0.0)
        self.assertEqual(result["boundary_rmse_k"], float(np.sqrt(128.0 / 124.0)))

    def test_gradient_diagnostics_are_non_mutating_and_safe(self):
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.grad = torch.tensor([[7.0, 9.0]])
        before = model.weight.grad.clone()
        x = torch.tensor([[1.0, 0.0]])
        diag = component_gradient_diagnostics(model(x).square().mean(), model)
        self.assertEqual(diag["active_parameters"], 1)
        self.assertEqual(diag["trainable_parameters"], 1)
        torch.testing.assert_close(model.weight.grad, before)
        self.assertEqual(safe_gradient_ratio(1.0, 0.0), 0.0)
        self.assertAlmostEqual(safe_gradient_ratio(2.0, 4.0), 0.5)

    def test_boundary_loss_gradient_reaches_trainable_parameter(self):
        temperature = torch.nn.Parameter(torch.tensor(315.0))
        T = temperature.expand(1, 1, 32, 32)
        residual = compute_boundary_residual_field_torch(
            theta_pred=T, T_pred=T, kx_base=torch.full_like(T, 100.0),
            ky_base=torch.full_like(T, 100.0), n_map=torch.zeros_like(T),
            Q_vol=torch.full_like(T, 1e6), h_sink=torch.tensor([1000.0]),
            T_ambient=torch.tensor([300.0]),
        )
        diagnostic = component_gradient_diagnostics(torch.mean((residual / 1e6) ** 2), torch.nn.ParameterList([temperature]))
        self.assertEqual(diagnostic["active_parameters"], 1)
        self.assertEqual(diagnostic["trainable_parameters"], 1)
        self.assertGreater(diagnostic["norm"], 0.0)

    def test_validation_evaluator_source_has_no_test_set_reference(self):
        source = (ROOT / "experiments/physics_guided/evaluate_boundary_pilot.py").read_text(encoding="utf-8")
        self.assertNotIn("test.h5", source)
        self.assertNotIn("compute_file_sha256", source)


if __name__ == "__main__":
    unittest.main()
