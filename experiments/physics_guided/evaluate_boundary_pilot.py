"""Validation-only metrics for the three-arm boundary-weight pilot.

This entry point reads only the supplied checkpoint and dataset/validation.h5.
It deliberately does not import or invoke the production evaluator's dataset
integrity hashing workflow.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.physics_guided.training.train_pinn import custom_collate  # noqa: E402
from src.data import ChipThermalDataset  # noqa: E402
from src.data.normalize import load_ml_norm_stats  # noqa: E402
from src.models.pinn import PhysicsInformedFNO2d  # noqa: E402

ARM_WEIGHTS = {
    "A": {"w_data": 1.0, "w_pde": 0.0, "w_bc": 0.0},
    "B": {"w_data": 1.0, "w_pde": 6.0e-11, "w_bc": 6.0e-11},
    "C": {"w_data": 1.0, "w_pde": 6.0e-11, "w_bc": 1.5e-4},
}


def temperature_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Pixel-weighted K metrics; boundary is the perimeter, interior its complement."""
    if pred.shape != true.shape or pred.ndim != 3 or pred.shape[1:] != (32, 32):
        raise ValueError(f"Expected matching (N,32,32) arrays, got {pred.shape}, {true.shape}")
    err = pred - true
    boundary = np.zeros((32, 32), dtype=bool)
    boundary[0, :] = boundary[-1, :] = True
    boundary[:, 0] = boundary[:, -1] = True
    interior = ~boundary
    gt_flat = true.ravel()
    r2 = 1.0 - float(np.sum(err**2)) / max(float(np.sum((gt_flat - gt_flat.mean()) ** 2)), 1e-12)
    hotspot_errors, hotspot_distances = [], []
    for p, t in zip(pred, true):
        ti, pi = np.unravel_index(np.argmax(t), t.shape), np.unravel_index(np.argmax(p), p.shape)
        hotspot_errors.append(abs(float(p[pi] - t[ti])))
        hotspot_distances.append(float(np.hypot(ti[0] - pi[0], ti[1] - pi[1])))
    return {
        "mae_k": float(np.mean(np.abs(err))),
        "rmse_k": float(np.sqrt(np.mean(err**2))),
        "r2": r2,
        "boundary_mae_k": float(np.mean(np.abs(err[:, boundary]))),
        "boundary_rmse_k": float(np.sqrt(np.mean(err[:, boundary] ** 2))),
        "interior_mae_k": float(np.mean(np.abs(err[:, interior]))),
        "hotspot_temperature_error_k": float(np.mean(hotspot_errors)),
        "hotspot_localization_distance_cells": float(np.mean(hotspot_distances)),
        "mask_definition": "Boundary: union of first/last rows and first/last columns (124 cells); interior: remaining 30x30=900 cells.",
    }


def run_validation(checkpoint: Path, output_dir: Path, device: str | None = None, batch_size: int = 16) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite pilot output directory: {output_dir}")
    use_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(checkpoint, map_location=use_device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = PhysicsInformedFNO2d(
        in_channels=cfg.get("in_channels", 23), out_channels=cfg.get("out_channels", 1),
        width=cfg.get("width", 32), modes1=cfg.get("modes1", 8),
        modes2=cfg.get("modes2", 8), n_layers=cfg.get("n_layers", 4),
        w_data=cfg.get("w_data", 1.0), w_pde=cfg.get("w_pde", 6e-11), w_bc=cfg.get("w_bc", 6e-11),
    ).to(use_device)
    model.load_state_dict(ckpt["model_state_dict"])
    dataset = ChipThermalDataset("validation", normalize=True, return_meta=True)
    if len(dataset) != 800:
        raise ValueError(f"Expected 800 validation samples, found {len(dataset)}")
    stats = load_ml_norm_stats()["target"]["theta"]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate)
    predictions, residuals = [], []
    model.eval()
    with torch.inference_mode():
        for X, y, meta in loader:
            out = model.compute_all_losses(X.to(use_device), y.to(use_device), meta, float(stats["mean"]), float(stats["std"]))
            predictions.append(out["T_pred"][:, 0].cpu().numpy())
            residuals.append(out["R_boundary"].cpu().numpy())
    pred = np.concatenate(predictions)
    boundary_res = np.concatenate(residuals).astype(np.float64)
    with h5py.File(dataset.h5_path, "r") as val_file:
        true = np.asarray(val_file["targets/T"][:], dtype=np.float64)
    metrics = temperature_metrics(pred.astype(np.float64), true)
    metrics.update({
        "mean_boundary_residual_W_m3": float(np.mean(boundary_res)),
        "mean_absolute_boundary_residual_W_m3": float(np.mean(np.abs(boundary_res))),
        "boundary_residual_aggregation": "Arithmetic mean over 800 samples x 124 perimeter cells; physical finite-volume residual in W/m^3.",
        "n_validation_samples": 800,
        "checkpoint": str(checkpoint.resolve()),
    })
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(run_validation(args.checkpoint, args.output_dir, args.device, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
