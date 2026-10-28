"""
Unit tests for experiments/physics_guided/evaluate_test_set.py.

Verifies test-evaluation logic, boundary masks, metric formulas, and checkpoint loading
using synthetic arrays and mocks WITHOUT accessing the real test.h5 dataset.
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

from experiments.physics_guided.evaluate_test_set import (
    calculate_r2_score,
    get_boundary_mask,
    load_baseline_fno,
    load_pinn_model,
    compute_perimeter_energy_residual,
)


class TestEvaluateTestSet(unittest.TestCase):
    """Test suite for evaluate_test_set.py pipeline."""

    def test_boundary_mask_dimensions_and_counts(self):
        """Verify boundary mask covers exactly 124 perimeter cells on 32x32 domain."""
        mask = get_boundary_mask(32, 32)
        self.assertEqual(mask.shape, (32, 32))
        self.assertEqual(mask.dtype, bool)

        n_boundary = int(np.sum(mask))
        n_interior = int(np.sum(~mask))

        # Perimeter of 32x32: 32 + 32 + 30 + 30 = 124 cells
        self.assertEqual(n_boundary, 124)
        # Interior: 30 x 30 = 900 cells
        self.assertEqual(n_interior, 900)
        self.assertEqual(n_boundary + n_interior, 1024)

        # Check edge rows and columns are completely True
        self.assertTrue(np.all(mask[0, :]))
        self.assertTrue(np.all(mask[-1, :]))
        self.assertTrue(np.all(mask[:, 0]))
        self.assertTrue(np.all(mask[:, -1]))
        # Check interior center is False
        self.assertFalse(np.any(mask[1:-1, 1:-1]))

    def test_r2_score_synthetic(self):
        """Verify R^2 calculation behavior on perfect, imperfect, and degenerate synthetic data."""
        # Perfect prediction
        y_true = np.array([[300.0, 310.0], [320.0, 330.0]], dtype=np.float64)
        y_pred = np.array([[300.0, 310.0], [320.0, 330.0]], dtype=np.float64)
        self.assertAlmostEqual(calculate_r2_score(y_true, y_pred), 1.0, places=6)

        # Mean predictor (R^2 = 0)
        y_pred_mean = np.full_like(y_true, np.mean(y_true))
        self.assertAlmostEqual(calculate_r2_score(y_true, y_pred_mean), 0.0, places=6)

        # Constant field (degenerate case)
        y_const = np.full((4, 4), 300.0)
        self.assertAlmostEqual(calculate_r2_score(y_const, y_const), 1.0, places=6)

    def test_missing_checkpoints_raise_filenotfound(self):
        """Verify loader functions fail cleanly with FileNotFoundError on non-existent paths."""
        bogus_path = PROJECT_ROOT / "experiments" / "non_existent_ckpt.pt"
        device = torch.device("cpu")
        with self.assertRaises(FileNotFoundError):
            load_baseline_fno(bogus_path, device)
        with self.assertRaises(FileNotFoundError):
            load_pinn_model(bogus_path, device)

    def test_real_checkpoints_loading(self):
        """Verify real checkpoints in repository load with correct parameter counts and eval mode."""
        device = torch.device("cpu")

        # Baseline FNO
        base_path = PROJECT_ROOT / "experiments" / "baseline" / "checkpoints" / "baseline" / "fno_baseline_best.pt"
        if base_path.exists():
            base_model, base_meta = load_baseline_fno(base_path, device)
            self.assertEqual(base_meta["n_parameters"], 530433)
            self.assertFalse(base_model.training)

        # PINN Checkpoint (full_data_v1 or pinn_best)
        pinn_path = PROJECT_ROOT / "experiments" / "physics_guided" / "checkpoints" / "pinn_full_data_v1" / "pinn_full_data_best.pt"
        if not pinn_path.exists():
            pinn_path = PROJECT_ROOT / "experiments" / "physics_guided" / "checkpoints" / "pinn_best.pt"

        if pinn_path.exists():
            pinn_model, pinn_meta = load_pinn_model(pinn_path, device)
            self.assertEqual(pinn_meta["n_parameters"], 530433)
            self.assertFalse(pinn_model.training)

    def test_synthetic_perimeter_energy_residual(self):
        """Verify physical boundary residual evaluation on synthetic batch."""
        B = 2
        H = 32
        W = 32
        device = torch.device("cpu")

        T_synthetic = torch.full((B, 1, H, W), 320.0, dtype=torch.float32)
        meta_synthetic = {
            "material": np.ones((B, H, W), dtype=np.int64),  # Silicon
            "n_dielectric": np.array([0.2, 0.2], dtype=np.float32),
            "n_silicon": np.array([1.2, 1.2], dtype=np.float32),
            "n_copper": np.array([0.25, 0.25], dtype=np.float32),
            "T_ref_K": np.array([300.0, 300.0], dtype=np.float32),
            "h_sink": np.array([1000.0, 1000.0], dtype=np.float32),
            "T_ambient": np.array([300.0, 300.0], dtype=np.float32),
            "kx_base": np.full((B, 1, H, W), 148.0, dtype=np.float32),
            "ky_base": np.full((B, 1, H, W), 148.0, dtype=np.float32),
            "Q_areal": np.zeros((B, 1, H, W), dtype=np.float32),
            "bc_type_x0": np.array([0.0, 0.0], dtype=np.float32),  # Insulated
            "bc_h_x0": np.array([0.0, 0.0], dtype=np.float32),
            "bc_T_x0": np.array([300.0, 300.0], dtype=np.float32),
            "bc_type_x1": np.array([0.0, 0.0], dtype=np.float32),
            "bc_h_x1": np.array([0.0, 0.0], dtype=np.float32),
            "bc_T_x1": np.array([300.0, 300.0], dtype=np.float32),
            "bc_type_y0": np.array([0.0, 0.0], dtype=np.float32),
            "bc_h_y0": np.array([0.0, 0.0], dtype=np.float32),
            "bc_T_y0": np.array([300.0, 300.0], dtype=np.float32),
            "bc_type_y1": np.array([0.0, 0.0], dtype=np.float32),
            "bc_h_y1": np.array([0.0, 0.0], dtype=np.float32),
            "bc_T_y1": np.array([300.0, 300.0], dtype=np.float32),
        }

        R_b, R_cell = compute_perimeter_energy_residual(T_synthetic, meta_synthetic)
        self.assertEqual(R_b.shape, (B, 124))
        self.assertEqual(R_cell.shape, (B, 1, H, W))
        self.assertTrue(torch.isfinite(R_b).all())
        self.assertTrue(torch.isfinite(R_cell).all())

    def test_no_optimizer_imported_or_called(self):
        """Ensure script is strictly an evaluation pipeline with no optimization routines."""
        script_path = PROJECT_ROOT / "experiments" / "physics_guided" / "evaluate_test_set.py"
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertNotIn("optimizer.step()", content)
        self.assertNotIn("optimizer.zero_grad()", content)
        self.assertNotIn("loss.backward()", content)
        self.assertIn("torch.inference_mode()", content)

    def test_plotting_functions_synthetic(self):
        """Verify all 5 visualization functions generate valid non-empty images."""
        import tempfile
        from experiments.physics_guided.plot_test_evaluation import (
            plot_figure_1_temperature_fields,
            plot_figure_2_absolute_error_heatmaps,
            plot_figure_3_boundary_interior_mae,
            plot_figure_4_boundary_physics_residual,
            plot_figure_5_hotspot_comparison,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            T_gt = np.random.uniform(300.0, 350.0, (32, 32))
            T_fno = T_gt + np.random.normal(0, 3.0, (32, 32))
            T_pinn = T_gt + np.random.normal(0, 3.0, (32, 32))
            meta = {"sample_idx": 0, "sample_id": 7200, "T_ambient": 300.0}

            dummy_metrics = {
                "baseline_model": {
                    "metrics": {
                        "mae_k": 3.1462, "boundary_mae_k": 3.3287, "interior_mae_k": 3.1210,
                        "boundary_residual_abs_W_m3": 2.1268e9,
                        "hotspot_loc_dist_mean_cells": 3.1769,
                        "hotspot_exact_match_pct": 33.125, "hotspot_tol_1_cell_pct": 61.750,
                    }
                },
                "pinn_model": {
                    "metrics": {
                        "mae_k": 3.2881, "boundary_mae_k": 3.5822, "interior_mae_k": 3.2476,
                        "boundary_residual_abs_W_m3": 1.9021e9,
                        "hotspot_loc_dist_mean_cells": 3.2755,
                        "hotspot_exact_match_pct": 32.375, "hotspot_tol_1_cell_pct": 61.125,
                    }
                }
            }

            p1 = tmp / "fig1.png"
            plot_figure_1_temperature_fields(T_gt, T_fno, T_pinn, meta, p1, dpi=72)
            self.assertTrue(p1.exists() and p1.stat().st_size > 0)

            p2 = tmp / "fig2.png"
            plot_figure_2_absolute_error_heatmaps(T_gt, T_fno, T_pinn, meta, p2, dpi=72)
            self.assertTrue(p2.exists() and p2.stat().st_size > 0)

            p3 = tmp / "fig3.png"
            plot_figure_3_boundary_interior_mae(dummy_metrics, p3, dpi=72)
            self.assertTrue(p3.exists() and p3.stat().st_size > 0)

            p4 = tmp / "fig4.png"
            plot_figure_4_boundary_physics_residual(dummy_metrics, p4, dpi=72)
            self.assertTrue(p4.exists() and p4.stat().st_size > 0)

            p5 = tmp / "fig5.png"
            plot_figure_5_hotspot_comparison(dummy_metrics, p5, dpi=72)
            self.assertTrue(p5.exists() and p5.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
