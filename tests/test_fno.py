"""
Unit tests for the 2D Fourier Neural Operator (FNO) baseline.

Tests model import, layer shapes, parameter counts, coordinate grid additions,
and execution of a real batch forward pass using the ml_data DataLoader.

Run
---
    python -m unittest tests.test_fno
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_data import N_INPUT_CHANNELS, make_dataloader  # noqa: E402
from models.fno import FNO2d, SpectralConv2d, build_fno_from_config  # noqa: E402


class TestFNO2d(unittest.TestCase):
    def setUp(self):
        self.in_channels = 23
        self.out_channels = 1
        self.width = 32
        self.modes1 = 8
        self.modes2 = 8
        self.n_layers = 4
        self.batch_size = 2
        self.grid_size = 32

    def test_import_and_instantiation(self):
        """Verify FNO2d and SpectralConv2d import and instantiate cleanly."""
        model = FNO2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            width=self.width,
            modes1=self.modes1,
            modes2=self.modes2,
            n_layers=self.n_layers,
        )
        self.assertIsInstance(model, FNO2d)
        n_params = model.count_parameters()
        self.assertGreater(n_params, 0)

    def test_spectral_conv2d_shape(self):
        """Verify SpectralConv2d forward pass tensor dimensions."""
        spec_conv = SpectralConv2d(
            in_channels=self.width,
            out_channels=self.width,
            modes1=self.modes1,
            modes2=self.modes2,
        )
        x_dummy = torch.randn(self.batch_size, self.width, self.grid_size, self.grid_size)
        out = spec_conv(x_dummy)
        self.assertEqual(out.shape, (self.batch_size, self.width, self.grid_size, self.grid_size))
        self.assertTrue(torch.isfinite(out).all())

    def test_dummy_forward_pass(self):
        """Verify FNO2d forward pass on synthetic input tensor."""
        model = FNO2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            width=self.width,
            modes1=self.modes1,
            modes2=self.modes2,
            n_layers=self.n_layers,
            use_grid=True,
        )
        model.eval()
        x_dummy = torch.randn(self.batch_size, self.in_channels, self.grid_size, self.grid_size)
        with torch.no_grad():
            out = model(x_dummy)
        self.assertEqual(out.shape, (self.batch_size, self.out_channels, self.grid_size, self.grid_size))
        self.assertTrue(torch.isfinite(out).all())

    def test_use_grid_flag(self):
        """Verify behavior with use_grid=False."""
        model_no_grid = FNO2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            width=self.width,
            modes1=self.modes1,
            modes2=self.modes2,
            n_layers=self.n_layers,
            use_grid=False,
        )
        model_no_grid.eval()
        x_dummy = torch.randn(self.batch_size, self.in_channels, self.grid_size, self.grid_size)
        with torch.no_grad():
            out = model_no_grid(x_dummy)
        self.assertEqual(out.shape, (self.batch_size, self.out_channels, self.grid_size, self.grid_size))

    def test_real_dataloader_batch_forward_pass(self):
        """Verify FNO2d execution on an actual sample batch from ml_data DataLoader."""
        loader = make_dataloader("train", batch_size=2, shuffle=False, num_workers=0)
        X, y = next(iter(loader))

        self.assertEqual(X.shape, (2, 23, 32, 32))
        self.assertEqual(y.shape, (2, 1, 32, 32))
        self.assertTrue(torch.isfinite(X).all(), "NaN/Inf found in input tensor X")
        self.assertTrue(torch.isfinite(y).all(), "NaN/Inf found in target tensor y")

        model = FNO2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            width=self.width,
            modes1=self.modes1,
            modes2=self.modes2,
            n_layers=self.n_layers,
        )
        model.eval()

        with torch.no_grad():
            pred = model(X)

        self.assertEqual(pred.shape, (2, 1, 32, 32))
        self.assertTrue(torch.isfinite(pred).all(), "NaN/Inf found in prediction tensor pred")

    def test_build_from_config(self):
        """Verify building model from a config dictionary."""
        cfg = {
            "model": {
                "in_channels": 23,
                "out_channels": 1,
                "width": 32,
                "modes1": 8,
                "modes2": 8,
                "n_layers": 4,
                "use_grid": True,
            }
        }
        model = build_fno_from_config(cfg)
        self.assertEqual(model.in_channels, 23)
        self.assertEqual(model.out_channels, 1)
        self.assertEqual(model.width, 32)


if __name__ == "__main__":
    unittest.main()
