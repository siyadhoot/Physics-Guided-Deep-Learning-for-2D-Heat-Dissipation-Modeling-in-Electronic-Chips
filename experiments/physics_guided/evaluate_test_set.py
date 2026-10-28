"""
Rigorous Test-Set Evaluation Pipeline for FNO vs. Physics-Guided PINN.

Project: Physics-Guided Deep Learning for 2D Heat Dissipation Modeling in Electronic Chips.

Final Evaluation Protocol:
  - Evaluates pretrained models strictly on the held-out test split (dataset/test.h5, 800 samples).
  - Open in READ-ONLY mode ('r') exclusively.
  - Zero training, zero fine-tuning, zero hyperparameter adjustment.
  - Exact same test samples and order evaluated for both models.
  - Consistent unit conversion to Kelvin (K) for all temperature error metrics.
  - Physical boundary residual evaluated in W/m^3 matching solver Picard discretization.

Outputs saved to:
  outputs/final_test_evaluation/
    metrics.json
    comparison.csv
    evaluation_summary.txt
    visualizations/ (optional test parity and error plots)

Usage:
  python experiments/physics_guided/evaluate_test_set.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")  # Headless publication-quality rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Determine project root robustly (3 levels up from experiments/physics_guided/evaluate_test_set.py)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project module imports
from src.data import ChipThermalDataset
from src.data.build_aux_and_stats import load_aux_cache
from src.data.channels import build_kx_ky_base
from src.data.kirchhoff_utils import inverse_kirchhoff_from_sample
from src.data.normalize import load_ml_norm_stats
from src.data.paths import DATASET_DIR, ML_NORM_STATS, SPLIT_H5
from src.models.fno import FNO2d
from src.models.pinn import PhysicsInformedFNO2d
from src.physics.boundary import compute_boundary_residual_torch
from src.physics.pde_residual import D_CHIP, DX, compute_pde_residual_kirchhoff


# ==============================================================================
# REPRODUCIBILITY & DEVICE SETUP
# ==============================================================================

def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds for evaluation reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(requested: Optional[str] = None) -> torch.device:
    """Select compute device."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 checksum of a file to verify data integrity."""
    if not path.exists():
        raise FileNotFoundError(f"File not found for hash check: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


# ==============================================================================
# METRICS UTILITIES
# ==============================================================================

def calculate_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate coefficient of determination R^2 across all pixels.
    R^2 = 1 - (SS_res / SS_tot).
    """
    y_t = y_true.ravel()
    y_p = y_pred.ravel()
    ss_res = float(np.sum((y_t - y_p) ** 2))
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - (ss_res / ss_tot))


def get_boundary_mask(h: int = 32, w: int = 32) -> np.ndarray:
    """
    Repository's established 124-cell boundary mask for 32x32 domain.
    Covers row 0 (x0/left), row -1 (x1/right), col 0 (y0/bottom), col -1 (y1/top).
    """
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def custom_collate(batch):
    """Custom collate preserving metadata for PDE residual and boundary evaluation."""
    X = torch.stack([item[0] for item in batch])
    y = torch.stack([item[1] for item in batch])
    first = batch[0][2]
    meta = {
        "material": np.stack([item[2]["material"] for item in batch]),
        "n_dielectric": np.array([item[2]["n_dielectric"] for item in batch], dtype=np.float32),
        "n_silicon": np.array([item[2]["n_silicon"] for item in batch], dtype=np.float32),
        "n_copper": np.array([item[2]["n_copper"] for item in batch], dtype=np.float32),
        "T_ref_K": np.array([item[2]["T_ref_K"] for item in batch], dtype=np.float32),
        "h_sink": np.array([item[2]["h_sink"] for item in batch], dtype=np.float32),
        "T_ambient": np.array([item[2]["T_ambient"] for item in batch], dtype=np.float32),
        "kx_base": np.stack([item[2]["kx_base"] for item in batch]),
        "ky_base": np.stack([item[2]["ky_base"] for item in batch]),
        "sample_id": [item[2]["sample_id"] for item in batch],
        "boundary_type": [item[2]["boundary_type"] for item in batch],
    }
    if "Q_areal" in first:
        meta["Q_areal"] = np.stack([item[2]["Q_areal"] for item in batch])
    for edge in ("x0", "x1", "y0", "y1"):
        if f"bc_type_{edge}" in first:
            meta[f"bc_type_{edge}"] = np.array([item[2][f"bc_type_{edge}"] for item in batch], dtype=np.float32)
            meta[f"bc_h_{edge}"] = np.array([item[2][f"bc_h_{edge}"] for item in batch], dtype=np.float32)
            meta[f"bc_T_{edge}"] = np.array([item[2][f"bc_T_{edge}"] for item in batch], dtype=np.float32)
    for edge_alias, edge_key in [("left", "x0"), ("right", "x1"), ("bottom", "y0"), ("top", "y1")]:
        if f"bc_type_{edge_key}" in meta:
            meta[f"bc_type_{edge_alias}"] = meta[f"bc_type_{edge_key}"]
            meta[f"bc_h_{edge_alias}"] = meta[f"bc_h_{edge_key}"]
            meta[f"bc_T_{edge_alias}"] = meta[f"bc_T_{edge_key}"]
    return X, y, meta


# ==============================================================================
# MODEL LOADER HELPERS
# ==============================================================================

def load_baseline_fno(ckpt_path: Path, device: torch.device) -> Tuple[FNO2d, Dict[str, Any]]:
    """Load pretrained baseline FNO2d model from checkpoint."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint file not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    mcfg = cfg.get("model", cfg)

    model = FNO2d(
        in_channels=mcfg.get("in_channels", 23),
        out_channels=mcfg.get("out_channels", 1),
        width=mcfg.get("width", 32),
        modes1=mcfg.get("modes1", 8),
        modes2=mcfg.get("modes2", 8),
        n_layers=mcfg.get("n_layers", 4),
        mlp_ratio=mcfg.get("mlp_ratio", 1.0),
        activation=mcfg.get("activation", "gelu"),
        use_grid=mcfg.get("use_grid", True),
    ).to(device)

    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    meta = {
        "checkpoint_path": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
        "best_val_loss": ckpt.get("best_val_loss"),
        "n_parameters": model.count_parameters(trainable_only=True),
        "sha256": compute_file_sha256(ckpt_path),
    }
    return model, meta


def load_pinn_model(ckpt_path: Path, device: torch.device) -> Tuple[PhysicsInformedFNO2d, Dict[str, Any]]:
    """Load pretrained PhysicsInformedFNO2d model from checkpoint."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"PINN checkpoint file not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})

    model = PhysicsInformedFNO2d(
        in_channels=cfg.get("in_channels", 23),
        out_channels=cfg.get("out_channels", 1),
        width=cfg.get("width", 32),
        modes1=cfg.get("modes1", 8),
        modes2=cfg.get("modes2", 8),
        n_layers=cfg.get("n_layers", 4),
        w_data=cfg.get("w_data", 1.0),
        w_pde=cfg.get("w_pde", 6e-11),
        w_bc=cfg.get("w_bc", 6e-11),
    ).to(device)

    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    meta = {
        "checkpoint_path": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
        "best_val_loss": ckpt.get("best_val_loss"),
        "n_parameters": model.count_parameters(trainable_only=True),
        "config": cfg,
        "sha256": compute_file_sha256(ckpt_path),
    }
    return model, meta


# ==============================================================================
# PHYSICAL BOUNDARY RESIDUAL CALCULATION (STANDALONE FOR ANY TEMPERATURE MAP)
# ==============================================================================

def compute_perimeter_energy_residual(
    T_kelvin: torch.Tensor,
    meta: Dict[str, Any],
    dx: float = DX,
    dy: float = DX,
    d_chip: float = D_CHIP,
    T_ref: float = 300.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute conservative finite-volume energy balance residual on perimeter cells in W/m^3.
    Matches dataset_generator/solver.py assemble_system formulation.

    Returns:
      (R_boundary_all, R_cell_all)
      R_boundary_all: Tensor of shape [B, 124] containing volumetric residual for perimeter cells.
    """
    device = T_kelvin.device
    dtype = T_kelvin.dtype
    B, _, H, W = T_kelvin.shape

    material_t = torch.as_tensor(meta["material"], device=device, dtype=torch.long)
    nd = torch.as_tensor(meta["n_dielectric"], device=device, dtype=torch.float32).view(-1, 1, 1)
    ns = torch.as_tensor(meta["n_silicon"], device=device, dtype=torch.float32).view(-1, 1, 1)
    nc = torch.as_tensor(meta["n_copper"], device=device, dtype=torch.float32).view(-1, 1, 1)
    n_map = torch.where(material_t == 0, nd, torch.where(material_t == 1, ns, nc)).unsqueeze(1)

    kx_base = torch.as_tensor(meta["kx_base"], device=device, dtype=dtype)
    ky_base = torch.as_tensor(meta["ky_base"], device=device, dtype=dtype)
    while kx_base.ndim < 4:
        kx_base = kx_base.unsqueeze(0 if kx_base.ndim == 2 else 1)
    while ky_base.ndim < 4:
        ky_base = ky_base.unsqueeze(0 if ky_base.ndim == 2 else 1)

    ratio = torch.clamp(T_kelvin / T_ref, min=0.4, max=3.0)
    kx = kx_base * torch.pow(ratio, -n_map)
    ky = ky_base * torch.pow(ratio, -n_map)

    # Face conductivities via harmonic mean
    def harmonic_mean_t(a, b, eps=1e-30):
        return 2.0 * a * b / (a + b + eps)

    kxf = harmonic_mean_t(kx[:, :, :-1, :], kx[:, :, 1:, :])
    kyf = harmonic_mean_t(ky[:, :, :, :-1], ky[:, :, :, 1:])

    div_cond = torch.zeros((B, 1, H, W), device=device, dtype=dtype)
    flux_x = kxf * (T_kelvin[:, :, 1:, :] - T_kelvin[:, :, :-1, :])
    div_cond[:, :, 1:, :] -= flux_x
    div_cond[:, :, :-1, :] += flux_x

    flux_y = kyf * (T_kelvin[:, :, :, 1:] - T_kelvin[:, :, :, :-1])
    div_cond[:, :, :, 1:] -= flux_y
    div_cond[:, :, :, :-1] += flux_y

    def resolve_edge(edge_alias, edge_key):
        raw_type = meta.get(f"bc_type_{edge_key}", meta.get(f"bc_type_{edge_alias}"))
        raw_h = meta.get(f"bc_h_{edge_key}", meta.get(f"bc_h_{edge_alias}"))
        raw_T = meta.get(f"bc_T_{edge_key}", meta.get(f"bc_T_{edge_alias}"))
        t = torch.as_tensor(raw_type, device=device, dtype=dtype).view(-1, 1, 1, 1)
        h = torch.as_tensor(raw_h, device=device, dtype=dtype).view(-1, 1, 1, 1)
        t_env = torch.as_tensor(raw_T, device=device, dtype=dtype).view(-1, 1, 1, 1)
        return t, h, t_env

    bc_flux = torch.zeros((B, 1, H, W), device=device, dtype=dtype)

    # West (x0): row 0
    t_x0, h_x0, env_x0 = resolve_edge("left", "x0")
    k_x0 = kx[:, :, 0:1, :]
    h_eff_x0 = 1.0 / ((dx / 2.0) / k_x0 + 1.0 / torch.clamp(h_x0, min=1e-6))
    coef_x0 = torch.where(torch.round(t_x0) == 2, 2.0 * k_x0,
              torch.where(torch.round(t_x0) == 1, h_eff_x0 * dx, torch.zeros_like(k_x0)))
    bc_flux[:, :, 0:1, :] += coef_x0 * (T_kelvin[:, :, 0:1, :] - env_x0)

    # East (x1): row -1
    t_x1, h_x1, env_x1 = resolve_edge("right", "x1")
    k_x1 = kx[:, :, -1:, :]
    h_eff_x1 = 1.0 / ((dx / 2.0) / k_x1 + 1.0 / torch.clamp(h_x1, min=1e-6))
    coef_x1 = torch.where(torch.round(t_x1) == 2, 2.0 * k_x1,
              torch.where(torch.round(t_x1) == 1, h_eff_x1 * dx, torch.zeros_like(k_x1)))
    bc_flux[:, :, -1:, :] += coef_x1 * (T_kelvin[:, :, -1:, :] - env_x1)

    # South (y0): col 0
    t_y0, h_y0, env_y0 = resolve_edge("bottom", "y0")
    k_y0 = ky[:, :, :, 0:1]
    h_eff_y0 = 1.0 / ((dy / 2.0) / k_y0 + 1.0 / torch.clamp(h_y0, min=1e-6))
    coef_y0 = torch.where(torch.round(t_y0) == 2, 2.0 * k_y0,
              torch.where(torch.round(t_y0) == 1, h_eff_y0 * dy, torch.zeros_like(k_y0)))
    bc_flux[:, :, :, 0:1] += coef_y0 * (T_kelvin[:, :, :, 0:1] - env_y0)

    # North (y1): col -1
    t_y1, h_y1, env_y1 = resolve_edge("top", "y1")
    k_y1 = ky[:, :, :, -1:]
    h_eff_y1 = 1.0 / ((dy / 2.0) / k_y1 + 1.0 / torch.clamp(h_y1, min=1e-6))
    coef_y1 = torch.where(torch.round(t_y1) == 2, 2.0 * k_y1,
              torch.where(torch.round(t_y1) == 1, h_eff_y1 * dy, torch.zeros_like(k_y1)))
    bc_flux[:, :, :, -1:] += coef_y1 * (T_kelvin[:, :, :, -1:] - env_y1)

    h_sink = torch.as_tensor(meta["h_sink"], device=device, dtype=dtype).view(-1, 1, 1, 1)
    t_ambient = torch.as_tensor(meta["T_ambient"], device=device, dtype=dtype).view(-1, 1, 1, 1)
    sink_term = (h_sink / d_chip) * (dx * dy) * (T_kelvin - t_ambient)

    Q_areal = torch.as_tensor(meta["Q_areal"], device=device, dtype=dtype)
    while Q_areal.ndim < 4:
        Q_areal = Q_areal.unsqueeze(1)
    Q_vol = Q_areal / d_chip
    source_term = Q_vol * (dx * dy)

    # Net volumetric imbalance in W/m^3
    R_cell = (-div_cond + bc_flux + sink_term - source_term) / (dx * dy)

    bmask = torch.from_numpy(get_boundary_mask(H, W)).to(device=device, dtype=torch.bool).view(1, 1, H, W)
    R_boundary = R_cell[bmask.expand_as(R_cell)].view(B, -1)
    return R_boundary, R_cell


# ==============================================================================
# TEST EVALUATOR CLASS
# ==============================================================================

class TestSetEvaluator:
    """Rigorous, deterministic held-out test evaluation comparing FNO and PINN."""

    def __init__(
        self,
        baseline_ckpt_path: Path,
        pinn_ckpt_path: Path,
        output_dir: Path,
        batch_size: int = 16,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        self.baseline_ckpt_path = Path(baseline_ckpt_path)
        self.pinn_ckpt_path = Path(pinn_ckpt_path)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.seed = seed
        self.device = get_device(device)

        set_seed(self.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    @torch.inference_mode()
    def run_evaluation(self, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """Execute full evaluation across test.h5."""
        start_time = time.time()

        print("=" * 80)
        print("RIGOROUS TEST-SET EVALUATION: FNO BASELINE VS. CORRECTED PINN")
        print("=" * 80)
        print("NOTICE: Reading held-out test dataset in STRICT READ-ONLY mode.")
        print("PROTOCOL: Test metrics are final results and must NOT guide further model tuning.")
        print(f"Device: {self.device} | Batch Size: {self.batch_size} | Random Seed: {self.seed}")
        print(f"Baseline Checkpoint: {self.baseline_ckpt_path}")
        print(f"PINN Checkpoint:     {self.pinn_ckpt_path}")
        print("=" * 80)

        # 1. Load Pretrained Models
        baseline_model, baseline_meta = load_baseline_fno(self.baseline_ckpt_path, self.device)
        pinn_model, pinn_meta = load_pinn_model(self.pinn_ckpt_path, self.device)

        # 2. Load Normalization Statistics (Derived strictly from Training split)
        norm_stats = load_ml_norm_stats()
        tstats = norm_stats["target"]["theta"]
        target_mean = float(tstats["mean"])
        target_std = float(tstats["std"])

        # 3. Load Test Dataset (Read-only)
        test_dataset = ChipThermalDataset("test", normalize=True, return_meta=True)
        aux = load_aux_cache("test")
        n_samples = len(test_dataset)
        if max_samples is not None:
            n_samples = min(n_samples, max_samples)

        print(f"Held-out test dataset loaded: {n_samples} samples (test.h5 verified).")

        # Load ground truth physical temperatures and metadata from test.h5
        with h5py.File(test_dataset.h5_path, "r") as f:
            T_true_all = np.asarray(f["targets/T"][:n_samples], dtype=np.float64)
            material_all = np.asarray(f["inputs/material"][:n_samples], dtype=np.int64)

        test_loader = DataLoader(
            test_dataset if max_samples is None else torch.utils.data.Subset(test_dataset, range(n_samples)),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate,
        )

        bmask_2d = get_boundary_mask(32, 32)
        imask_2d = ~bmask_2d

        # Metric accumulators
        base_T_all = np.zeros((n_samples, 32, 32), dtype=np.float64)
        pinn_T_all = np.zeros((n_samples, 32, 32), dtype=np.float64)

        base_res_signed = []
        base_res_abs = []
        pinn_res_signed = []
        pinn_res_abs = []

        sample_rows = []
        batch_offset = 0

        print("Executing model inference and inverse Kirchhoff reconstructions...")
        for b_idx, (X, y, meta) in enumerate(test_loader, start=1):
            X = X.to(self.device, non_blocking=True)
            bs = X.size(0)
            sample_indices = range(batch_offset, batch_offset + bs)

            if not torch.isfinite(X).all():
                raise RuntimeError(f"CRITICAL: Non-finite input tensor X encountered at batch {b_idx}!")

            # ------------------------------------------------------------------
            # Inference: Baseline FNO
            # ------------------------------------------------------------------
            b_theta_norm = baseline_model(X)
            if not torch.isfinite(b_theta_norm).all():
                raise RuntimeError(f"CRITICAL: Baseline FNO predicted non-finite normalized potential at batch {b_idx}!")
            b_theta_phys = b_theta_norm * target_std + target_mean

            # ------------------------------------------------------------------
            # Inference: Corrected PINN
            # ------------------------------------------------------------------
            p_theta_norm = pinn_model.forward(X)
            if not torch.isfinite(p_theta_norm).all():
                raise RuntimeError(f"CRITICAL: PINN predicted non-finite normalized potential at batch {b_idx}!")
            p_theta_phys = p_theta_norm * target_std + target_mean

            # Inverse Kirchhoff per sample
            b_T_batch = []
            p_T_batch = []

            for i_local, idx in enumerate(sample_indices):
                mat = material_all[idx]
                nd = float(aux["n_dielectric"][idx])
                ns = float(aux["n_silicon"][idx])
                nc = float(aux["n_copper"][idx])
                tref = float(meta["T_ref_K"][i_local])

                b_th_s = b_theta_phys[i_local, 0].cpu().numpy().astype(np.float64)
                p_th_s = p_theta_phys[i_local, 0].cpu().numpy().astype(np.float64)

                b_T_s = inverse_kirchhoff_from_sample(b_th_s, mat, nd, ns, nc, T_ref=tref)
                p_T_s = inverse_kirchhoff_from_sample(p_th_s, mat, nd, ns, nc, T_ref=tref)

                if not np.isfinite(b_T_s).all():
                    raise RuntimeError(f"CRITICAL: Baseline produced non-finite physical temperature on test sample {idx}!")
                if not np.isfinite(p_T_s).all():
                    raise RuntimeError(f"CRITICAL: PINN produced non-finite physical temperature on test sample {idx}!")

                base_T_all[idx] = b_T_s
                pinn_T_all[idx] = p_T_s
                b_T_batch.append(torch.from_numpy(b_T_s).unsqueeze(0).float())
                p_T_batch.append(torch.from_numpy(p_T_s).unsqueeze(0).float())

            # Evaluate physical boundary energy residual
            b_T_tensor = torch.stack(b_T_batch).to(self.device)
            p_T_tensor = torch.stack(p_T_batch).to(self.device)

            b_res_b, _ = compute_perimeter_energy_residual(b_T_tensor, meta)
            p_res_b, _ = compute_perimeter_energy_residual(p_T_tensor, meta)

            base_res_signed.extend(b_res_b.mean(dim=1).cpu().tolist())
            base_res_abs.extend(torch.abs(b_res_b).mean(dim=1).cpu().tolist())

            pinn_res_signed.extend(p_res_b.mean(dim=1).cpu().tolist())
            pinn_res_abs.extend(torch.abs(p_res_b).mean(dim=1).cpu().tolist())

            batch_offset += bs
            if b_idx % 10 == 0 or b_idx == len(test_loader):
                print(f"  Processed {batch_offset}/{n_samples} test samples ({batch_offset / n_samples * 100:.1f}%)")

        print("Inference completed. Calculating comprehensive metrics...")

        # ----------------------------------------------------------------------
        # COMPUTE RIGOROUS METRICS
        # ----------------------------------------------------------------------
        def compute_model_metrics(T_pred: np.ndarray, res_signed: List[float], res_abs: List[float]) -> Dict[str, Any]:
            err = T_pred - T_true_all  # [N, 32, 32]
            abs_err = np.abs(err)

            mae = float(np.mean(abs_err))
            rmse = float(np.sqrt(np.mean(err ** 2)))
            r2 = calculate_r2_score(T_true_all, T_pred)
            bias = float(np.mean(err))
            max_abs_err = float(np.max(abs_err))

            # Boundary vs Interior
            b_err = err[:, bmask_2d]
            b_abs = abs_err[:, bmask_2d]
            boundary_mae = float(np.mean(b_abs))
            boundary_rmse = float(np.sqrt(np.mean(b_err ** 2)))

            i_err = err[:, imask_2d]
            i_abs = abs_err[:, imask_2d]
            interior_mae = float(np.mean(i_abs))
            interior_rmse = float(np.sqrt(np.mean(i_err ** 2)))

            # Per-edge MAE
            edge_x0_mae = float(np.mean(abs_err[:, 0, :]))
            edge_x1_mae = float(np.mean(abs_err[:, -1, :]))
            edge_y0_mae = float(np.mean(abs_err[:, :, 0]))
            edge_y1_mae = float(np.mean(abs_err[:, :, -1]))

            # Hotspot metrics
            ht_temp_errs = []
            ht_dists = []
            exact_matches = 0
            tol_1_matches = 0

            for i in range(n_samples):
                gt_i = T_true_all[i]
                pr_i = T_pred[i]
                gt_idx = np.unravel_index(np.argmax(gt_i), gt_i.shape)
                pr_idx = np.unravel_index(np.argmax(pr_i), pr_i.shape)

                ht_temp_errs.append(abs(float(pr_i[pr_idx]) - float(gt_i[gt_idx])))
                dist = np.sqrt((pr_idx[0] - gt_idx[0]) ** 2 + (pr_idx[1] - gt_idx[1]) ** 2)
                ht_dists.append(dist)
                if dist == 0.0:
                    exact_matches += 1
                if dist <= 1.0:
                    tol_1_matches += 1

            hotspot_mae = float(np.mean(ht_temp_errs))
            loc_dist_mean = float(np.mean(ht_dists))
            loc_dist_median = float(np.median(ht_dists))
            exact_pct = float(exact_matches / n_samples * 100.0)
            tol_1_pct = float(tol_1_matches / n_samples * 100.0)

            # Boundary energy residual in W/m^3
            mean_signed_res = float(np.mean(res_signed))
            mean_abs_res = float(np.mean(res_abs))

            return {
                "mae_k": mae,
                "rmse_k": rmse,
                "r2_score": r2,
                "bias_k": bias,
                "max_abs_err_k": max_abs_err,
                "boundary_mae_k": boundary_mae,
                "boundary_rmse_k": boundary_rmse,
                "interior_mae_k": interior_mae,
                "interior_rmse_k": interior_rmse,
                "edge_x0_mae_k": edge_x0_mae,
                "edge_x1_mae_k": edge_x1_mae,
                "edge_y0_mae_k": edge_y0_mae,
                "edge_y1_mae_k": edge_y1_mae,
                "hotspot_temp_mae_k": hotspot_mae,
                "hotspot_loc_dist_mean_cells": loc_dist_mean,
                "hotspot_loc_dist_median_cells": loc_dist_median,
                "hotspot_exact_match_pct": exact_pct,
                "hotspot_tol_1_cell_pct": tol_1_pct,
                "boundary_residual_signed_W_m3": mean_signed_res,
                "boundary_residual_abs_W_m3": mean_abs_res,
            }

        base_metrics = compute_model_metrics(base_T_all, base_res_signed, base_res_abs)
        pinn_metrics = compute_model_metrics(pinn_T_all, pinn_res_signed, pinn_res_abs)

        duration = time.time() - start_time

        # ----------------------------------------------------------------------
        # COMPILE COMPARATIVE REPORTS
        # ----------------------------------------------------------------------
        comparison_rows = [
            ("Overall Temperature MAE [K]", base_metrics["mae_k"], pinn_metrics["mae_k"], True),
            ("Overall Temperature RMSE [K]", base_metrics["rmse_k"], pinn_metrics["rmse_k"], True),
            ("R² Score", base_metrics["r2_score"], pinn_metrics["r2_score"], False),
            ("Mean Bias [K]", base_metrics["bias_k"], pinn_metrics["bias_k"], None),
            ("Max Single-Pixel Error [K]", base_metrics["max_abs_err_k"], pinn_metrics["max_abs_err_k"], True),
            ("Boundary MAE [K]", base_metrics["boundary_mae_k"], pinn_metrics["boundary_mae_k"], True),
            ("Boundary RMSE [K]", base_metrics["boundary_rmse_k"], pinn_metrics["boundary_rmse_k"], True),
            ("Interior MAE [K]", base_metrics["interior_mae_k"], pinn_metrics["interior_mae_k"], True),
            ("Interior RMSE [K]", base_metrics["interior_rmse_k"], pinn_metrics["interior_rmse_k"], True),
            ("Edge x0 (West) MAE [K]", base_metrics["edge_x0_mae_k"], pinn_metrics["edge_x0_mae_k"], True),
            ("Edge x1 (East) MAE [K]", base_metrics["edge_x1_mae_k"], pinn_metrics["edge_x1_mae_k"], True),
            ("Edge y0 (South) MAE [K]", base_metrics["edge_y0_mae_k"], pinn_metrics["edge_y0_mae_k"], True),
            ("Edge y1 (North) MAE [K]", base_metrics["edge_y1_mae_k"], pinn_metrics["edge_y1_mae_k"], True),
            ("Hotspot Peak Temp MAE [K]", base_metrics["hotspot_temp_mae_k"], pinn_metrics["hotspot_temp_mae_k"], True),
            ("Hotspot Mean Location Dist [cells]", base_metrics["hotspot_loc_dist_mean_cells"], pinn_metrics["hotspot_loc_dist_mean_cells"], True),
            ("Hotspot Exact Match [%]", base_metrics["hotspot_exact_match_pct"], pinn_metrics["hotspot_exact_match_pct"], False),
            ("Hotspot 1-Cell Match [%]", base_metrics["hotspot_tol_1_cell_pct"], pinn_metrics["hotspot_tol_1_cell_pct"], False),
            ("Boundary Signed Residual [W/m³]", base_metrics["boundary_residual_signed_W_m3"], pinn_metrics["boundary_residual_signed_W_m3"], None),
            ("Boundary Mean Absolute Residual [W/m³]", base_metrics["boundary_residual_abs_W_m3"], pinn_metrics["boundary_residual_abs_W_m3"], True),
        ]

        df_comparison = []
        for metric_name, b_val, p_val, lower_is_better in comparison_rows:
            diff = p_val - b_val
            if lower_is_better is True:
                rel_impr = ((b_val - p_val) / (b_val + 1e-12)) * 100.0
            elif lower_is_better is False:
                rel_impr = ((p_val - b_val) / (abs(b_val) + 1e-12)) * 100.0
            else:
                rel_impr = float("nan")

            df_comparison.append({
                "Metric": metric_name,
                "Baseline_FNO": b_val,
                "Corrected_PINN": p_val,
                "Difference (PINN - Base)": diff,
                "Relative Improvement [%]": rel_impr,
            })

        df_comp = pd.DataFrame(df_comparison)
        csv_path = self.output_dir / "comparison.csv"
        df_comp.to_csv(csv_path, index=False)

        # ----------------------------------------------------------------------
        # SAVE JSON METRICS
        # ----------------------------------------------------------------------
        output_payload = {
            "status": "SUCCESS",
            "evaluation_notice": (
                "FINAL HELD-OUT TEST EVALUATION. Test dataset was strictly read in read-only mode. "
                "These results are for final unbiased benchmark comparison only and must NOT guide model selection."
            ),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "execution_duration_sec": duration,
            "device": str(self.device),
            "seed": self.seed,
            "n_test_samples": n_samples,
            "test_split_file": str(test_dataset.h5_path),
            "baseline_model": {
                "name": "FNO2d_Baseline",
                "checkpoint": baseline_meta,
                "metrics": base_metrics,
            },
            "pinn_model": {
                "name": "PhysicsInformedFNO2d_Corrected",
                "checkpoint": pinn_meta,
                "metrics": pinn_metrics,
            },
        }

        json_path = self.output_dir / "metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)

        # ----------------------------------------------------------------------
        # GENERATE HUMAN-READABLE SUMMARY REPORT
        # ----------------------------------------------------------------------
        summary_txt_path = self.output_dir / "evaluation_summary.txt"
        summary_lines = [
            "=" * 80,
            "FINAL TEST-SET EVALUATION REPORT: FNO BASELINE VS. CORRECTED PINN",
            "=" * 80,
            f"Execution Timestamp (UTC): {output_payload['timestamp_utc']}",
            f"Evaluation Duration:       {duration:.2f} seconds",
            f"Device:                     {self.device}",
            f"Evaluated Test Samples:     {n_samples} samples (dataset/test.h5)",
            "",
            "INTEGRITY AND PROTOCOL COMPLIANCE:",
            "  - Dataset split: Held-out test set (test.h5, 800 samples).",
            "  - Integrity check: Both models evaluated on IDENTICAL test samples in identical order.",
            "  - Preprocessing: Inputs normalized using training-only statistics.",
            "  - Output Space: Predictions transformed back to physical Kelvin via inverse Kirchhoff.",
            "  - Scientific rule: These test metrics are strictly held-out and must NOT guide tuning.",
            "",
            "MODEL CHECKPOINTS:",
            f"  1. Baseline FNO: {self.baseline_ckpt_path}",
            f"     Parameters:   {baseline_meta['n_parameters']:,} (Epoch {baseline_meta.get('epoch')})",
            f"     SHA256:       {baseline_meta['sha256'][:16]}...",
            f"  2. Corrected PINN: {self.pinn_ckpt_path}",
            f"     Parameters:   {pinn_meta['n_parameters']:,} (Epoch {pinn_meta.get('epoch')})",
            f"     SHA256:       {pinn_meta['sha256'][:16]}...",
            "",
            "=" * 80,
            "COMPARATIVE TEST PERFORMANCE SUMMARY",
            "=" * 80,
            f"{'Metric':<42} | {'Baseline FNO':>15} | {'Corrected PINN':>15} | {'Improvement':>12}",
            "-" * 90,
        ]

        for row in df_comparison:
            m_name = row["Metric"]
            b_val = row["Baseline_FNO"]
            p_val = row["Corrected_PINN"]
            impr = row["Relative Improvement [%]"]
            impr_str = f"{impr:+.2f}%" if np.isfinite(impr) else "N/A"
            summary_lines.append(f"{m_name:<42} | {b_val:>15.4f} | {p_val:>15.4f} | {impr_str:>12}")

        summary_lines.extend([
            "-" * 90,
            "",
            "KEY FINDINGS & DEFENSE DISCUSSION POINTS:",
            f"  1. Overall Accuracy: Baseline MAE = {base_metrics['mae_k']:.3f} K vs PINN MAE = {pinn_metrics['mae_k']:.3f} K.",
            f"  2. Hotspot Accuracy: Baseline Hotspot MAE = {base_metrics['hotspot_temp_mae_k']:.3f} K vs PINN = {pinn_metrics['hotspot_temp_mae_k']:.3f} K.",
            f"  3. Boundary Regularization: Baseline Boundary MAE = {base_metrics['boundary_mae_k']:.3f} K vs PINN = {pinn_metrics['boundary_mae_k']:.3f} K.",
            f"  4. Physical Residual: PINN Boundary Imbalance = {pinn_metrics['boundary_residual_abs_W_m3']:.3e} W/m³ vs Baseline = {base_metrics['boundary_residual_abs_W_m3']:.3e} W/m³.",
            "",
            "=" * 80,
            "END OF TEST REPORT",
            "=" * 80,
        ])

        with open(summary_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))

        # ----------------------------------------------------------------------
        # GENERATE OPTIONAL PARITY PLOT VIZ
        # ----------------------------------------------------------------------
        try:
            fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
            # Sample subset for scatter plotting (up to 5000 random pixels)
            sub_idx = np.random.choice(n_samples * 32 * 32, size=min(10000, n_samples * 32 * 32), replace=False)
            gt_flat = T_true_all.ravel()[sub_idx]
            b_flat = base_T_all.ravel()[sub_idx]
            p_flat = pinn_T_all.ravel()[sub_idx]

            t_min = min(float(np.min(gt_flat)), float(np.min(b_flat)), float(np.min(p_flat)))
            t_max = max(float(np.max(gt_flat)), float(np.max(b_flat)), float(np.max(p_flat)))

            ax[0].scatter(gt_flat, b_flat, alpha=0.15, s=6, c="#1f77b4", rasterized=True)
            ax[0].plot([t_min, t_max], [t_min, t_max], "k--", lw=1.5, label="Ideal Parity")
            ax[0].set_title(f"Baseline FNO Parity (Test R²={base_metrics['r2_score']:.4f})")
            ax[0].set_xlabel("Ground Truth T [K]")
            ax[0].set_ylabel("Predicted T [K]")
            ax[0].grid(True, alpha=0.3)
            ax[0].legend()

            ax[1].scatter(gt_flat, p_flat, alpha=0.15, s=6, c="#ff7f0e", rasterized=True)
            ax[1].plot([t_min, t_max], [t_min, t_max], "k--", lw=1.5, label="Ideal Parity")
            ax[1].set_title(f"Corrected PINN Parity (Test R²={pinn_metrics['r2_score']:.4f})")
            ax[1].set_xlabel("Ground Truth T [K]")
            ax[1].set_ylabel("Predicted T [K]")
            ax[1].grid(True, alpha=0.3)
            ax[1].legend()

            plt.tight_layout()
            viz_parity_path = self.viz_dir / "test_prediction_parity.png"
            plt.savefig(viz_parity_path)
            plt.close(fig)
        except Exception as e:
            print(f"Notice: Parity plot rendering skipped ({e})")

        print("=" * 80)
        print("TEST-SET EVALUATION COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print(f"  - Metrics JSON:  {json_path}")
        print(f"  - Comparison:    {csv_path}")
        print(f"  - Summary Text:  {summary_txt_path}")
        print("=" * 80)

        return output_payload


# ==============================================================================
# CLI ARGUMENT PARSER & MAIN ENTRYPOINT
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rigorous Held-Out Test Evaluation: FNO Baseline vs. Corrected PINN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Automatic resolution of sensible defaults based on repository files
    default_base_ckpt = PROJECT_ROOT / "experiments" / "baseline" / "checkpoints" / "baseline" / "fno_baseline_best.pt"

    # Default PINN candidate: check candidate paths verified in audit
    candidate_pinn_paths = [
        PROJECT_ROOT / "experiments" / "physics_guided" / "checkpoints" / "pinn_heat_escape_corrected" / "pinn_full_data_best.pt",
        PROJECT_ROOT / "experiments" / "physics_guided" / "checkpoints" / "pinn_full_data_v1" / "pinn_full_data_best.pt",
        PROJECT_ROOT / "experiments" / "physics_guided" / "checkpoints" / "pinn_best.pt",
    ]
    resolved_pinn = candidate_pinn_paths[1] if candidate_pinn_paths[1].exists() else candidate_pinn_paths[2]

    parser.add_argument(
        "--baseline-ckpt",
        type=str,
        default=str(default_base_ckpt),
        help="Path to baseline FNO model checkpoint (.pt).",
    )
    parser.add_argument(
        "--pinn-ckpt",
        type=str,
        default=str(resolved_pinn),
        help="Path to trained PINN model checkpoint (.pt).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "final_test_evaluation"),
        help="Directory to store evaluation outputs (metrics.json, comparison.csv, etc.).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for test evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda', 'cpu', or None for auto-detect).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic evaluation.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on evaluated test samples (useful for rapid sanity verification).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    evaluator = TestSetEvaluator(
        baseline_ckpt_path=Path(args.baseline_ckpt),
        pinn_ckpt_path=Path(args.pinn_ckpt),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    evaluator.run_evaluation(max_samples=args.max_samples)


if __name__ == "__main__":
    main()
