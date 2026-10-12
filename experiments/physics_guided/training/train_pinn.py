"""
Production-ready Full Training Script for PhysicsInformedFNO2d.

Physics-Guided Deep Learning for 2D Heat Dissipation Modeling in Electronic Chips.

Features:
  - Model: PhysicsInformedFNO2d (width=32, modes=8x8, 4 layers, ~530,433 parameters)
  - Data: Complete 6400 train samples & 800 validation samples (TEST SET IS STRICTLY LOCKED)
  - Physics Loss: Supervised Data Loss + Anisotropic Kirchhoff PDE Residual + Boundary Condition Error
  - Optimization: AdamW (lr=0.001, weight_decay=1e-4) + CosineAnnealingLR
  - Finiteness & Safety: Strict checks for inputs, targets, predictions, losses, and gradients
  - Overfitting Protection: Validation-based checkpointing (pinn_best.pt) & Early Stopping (patience=15)

Usage (on GPU / Google Colab / local):
    python experiments/physics_guided/training/train_pinn.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Determine project root robustly (4 levels up from experiments/physics_guided/training/train_pinn.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import ChipThermalDataset  # noqa: E402
from src.data.normalize import load_ml_norm_stats  # noqa: E402
from src.models.pinn import PhysicsInformedFNO2d  # noqa: E402
from src.physics.conductivity import compute_temperature_dependent_conductivity_torch  # noqa: E402
from src.physics.kirchhoff_torch import forward_kirchhoff_torch, inverse_kirchhoff_torch  # noqa: E402
from src.physics.pde_residual import compute_pde_residual_kirchhoff  # noqa: E402


def set_seed(seed: int = 42) -> None:
    """Set deterministic random seeds across Python, NumPy, PyTorch CPU, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Detect and report available device (CUDA GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"CUDA Available! Using GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        print("CUDA NOT Available. Using CPU.")
    return device


def custom_collate(batch):
    """Custom collate function preserving physics metadata required for PDE calculations."""
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


class EarlyStopping:
    """Validation-based early stopping."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        improved = metric < (self.best - self.min_delta)
        if improved:
            self.best = metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.should_stop = True
        return improved


@torch.no_grad()
def evaluate_validation_set(
    model: PhysicsInformedFNO2d,
    loader: DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float,
    all_gt_T: Optional[np.ndarray] = None,
    w_data: float = 1.0,
    w_pde: float = 6.0e-11,
    w_bc: float = 1.35e-5,
) -> Dict[str, Any]:
    """
    Evaluate complete validation dataset across loss components and physical temperature metrics.
    Computes Kelvin MAE, RMSE, R^2, bias, max error, hotspot metrics, and temperature spans.
    """
    model.eval()
    val_data_loss = 0.0
    val_pde_loss = 0.0
    val_bc_loss = 0.0
    n_samples = 0
    all_pred_T = []

    for X, y, meta in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if not torch.isfinite(X).all():
            raise RuntimeError("CRITICAL: Non-finite input X encountered in validation evaluation!")
        if not torch.isfinite(y).all():
            raise RuntimeError("CRITICAL: Non-finite target y encountered in validation evaluation!")

        loss_dict = model.compute_all_losses(X, y, meta, target_mean, target_std)

        d_loss = loss_dict["data_loss"]
        p_loss = loss_dict["pde_loss"]
        b_loss = loss_dict["bc_loss"]
        tot_loss = loss_dict["total_loss"]

        if not torch.isfinite(tot_loss).all():
            raise RuntimeError("CRITICAL: Non-finite loss encountered in validation evaluation!")

        bs = X.shape[0]
        val_data_loss += d_loss.item() * bs
        val_pde_loss += p_loss.item() * bs
        val_bc_loss += b_loss.item() * bs
        n_samples += bs

        # Physical temperature prediction in Kelvin via inverse Kirchhoff transform
        pred_norm = loss_dict["pred_norm"]
        theta_phys_pred = pred_norm * target_std + target_mean

        material_t = torch.as_tensor(meta["material"], device=device, dtype=torch.long)
        nd = torch.as_tensor(meta["n_dielectric"], device=device, dtype=torch.float32).view(-1, 1, 1)
        ns = torch.as_tensor(meta["n_silicon"], device=device, dtype=torch.float32).view(-1, 1, 1)
        nc = torch.as_tensor(meta["n_copper"], device=device, dtype=torch.float32).view(-1, 1, 1)
        n_map = torch.where(material_t == 0, nd, torch.where(material_t == 1, ns, nc))

        T_pred = inverse_kirchhoff_torch(theta_phys_pred, n_map.unsqueeze(1), T_ref=300.0)
        if not torch.isfinite(T_pred).all():
            raise RuntimeError("CRITICAL: Non-finite physical temperature predicted in validation evaluation!")

        all_pred_T.append(T_pred.squeeze(1).cpu().numpy())

    n_samples = max(n_samples, 1)
    v_data_mean = val_data_loss / n_samples
    v_pde_mean = val_pde_loss / n_samples
    v_bc_mean = val_bc_loss / n_samples
    v_total_mean = w_data * v_data_mean + w_pde * v_pde_mean + w_bc * v_bc_mean

    all_pred_T = np.concatenate(all_pred_T, axis=0)  # (N, 32, 32)
    pred_t_min = float(np.min(all_pred_T))
    pred_t_max = float(np.max(all_pred_T))
    pred_t_mean = float(np.mean(all_pred_T))
    pred_t_std = float(np.std(all_pred_T))
    pred_t_span = pred_t_max - pred_t_min
    sample_spans = np.max(all_pred_T, axis=(1, 2)) - np.min(all_pred_T, axis=(1, 2))
    mean_sample_span = float(np.mean(sample_spans))

    results = {
        "val_data_loss": v_data_mean,
        "val_pde_loss": v_pde_mean,
        "val_bc_loss": v_bc_mean,
        "val_total_loss": v_total_mean,
        "pred_t_min": pred_t_min,
        "pred_t_max": pred_t_max,
        "pred_t_mean": pred_t_mean,
        "pred_t_std": pred_t_std,
        "pred_t_span": pred_t_span,
        "mean_sample_span": mean_sample_span,
    }

    if all_gt_T is not None:
        n_eval = min(len(all_pred_T), len(all_gt_T))
        preds = all_pred_T[:n_eval]
        gts = all_gt_T[:n_eval]

        err = preds - gts
        abs_err = np.abs(err)
        mae_k = float(np.mean(abs_err))
        rmse_k = float(np.sqrt(np.mean(err**2)))
        bias_k = float(np.mean(err))
        max_abs_err_k = float(np.max(abs_err))

        ss_res = np.sum(err**2)
        ss_tot = np.sum((gts - np.mean(gts))**2)
        r2_score = float(1.0 - (ss_res / (ss_tot + 1e-12)))

        gt_t_min = float(np.min(gts))
        gt_t_max = float(np.max(gts))
        gt_t_span = gt_t_max - gt_t_min

        ht_errs = []
        ht_dists = []
        exact_matches = 0
        tol_1_matches = 0

        for i in range(n_eval):
            gt_i = gts[i]
            pr_i = preds[i]
            gt_idx = np.unravel_index(np.argmax(gt_i), gt_i.shape)
            pr_idx = np.unravel_index(np.argmax(pr_i), pr_i.shape)

            ht_errs.append(abs(pr_i[pr_idx] - gt_i[gt_idx]))
            dist = np.sqrt((gt_idx[0] - pr_idx[0])**2 + (gt_idx[1] - pr_idx[1])**2)
            ht_dists.append(dist)
            if dist == 0.0:
                exact_matches += 1
            if dist <= 1.0:
                tol_1_matches += 1

        ht_mae = float(np.mean(ht_errs))
        loc_dist_mean = float(np.mean(ht_dists))
        loc_dist_median = float(np.median(ht_dists))
        exact_rate = float(exact_matches / n_eval * 100.0)
        tol_1_rate = float(tol_1_matches / n_eval * 100.0)

        results.update({
            "mae_k": mae_k,
            "rmse_k": rmse_k,
            "bias_k": bias_k,
            "max_abs_err_k": max_abs_err_k,
            "r2_score": r2_score,
            "gt_t_min": gt_t_min,
            "gt_t_max": gt_t_max,
            "gt_t_span": gt_t_span,
            "hotspot_mae_k": ht_mae,
            "loc_dist_mean": loc_dist_mean,
            "loc_dist_median": loc_dist_median,
            "exact_match_rate_pct": exact_rate,
            "tol_1_cell_rate_pct": tol_1_rate,
        })

    return results


def train_one_epoch(
    model: PhysicsInformedFNO2d,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_mean: float,
    target_std: float,
    epoch: int,
    log_every: int = 100,
) -> Dict[str, float]:
    """Train single epoch with strict finiteness checks on inputs, targets, losses, and gradients."""
    model.train()
    running_data_loss = 0.0
    running_pde_loss = 0.0
    running_bc_loss = 0.0
    running_total_loss = 0.0
    n_samples = 0

    for step, (X, y, meta) in enumerate(loader, start=1):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if not torch.isfinite(X).all():
            raise RuntimeError(f"CRITICAL: Non-finite input X at epoch {epoch}, step {step}")
        if not torch.isfinite(y).all():
            raise RuntimeError(f"CRITICAL: Non-finite target y at epoch {epoch}, step {step}")

        optimizer.zero_grad(set_to_none=True)

        loss_dict = model.compute_all_losses(X, y, meta, target_mean, target_std)

        d_loss = loss_dict["data_loss"]
        p_loss = loss_dict["pde_loss"]
        b_loss = loss_dict["bc_loss"]
        tot_loss = loss_dict["total_loss"]

        if not torch.isfinite(tot_loss).all():
            raise RuntimeError(f"CRITICAL: Non-finite total loss at epoch {epoch}, step {step}")
        if not torch.isfinite(d_loss).all():
            raise RuntimeError(f"CRITICAL: Non-finite data loss at epoch {epoch}, step {step}")
        if not torch.isfinite(p_loss).all():
            raise RuntimeError(f"CRITICAL: Non-finite PDE loss at epoch {epoch}, step {step}")

        tot_loss.backward()

        # Check gradient finiteness across all parameters
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise RuntimeError(f"CRITICAL: Non-finite gradient in {name} at epoch {epoch}, step {step}")

        optimizer.step()

        bs = X.shape[0]
        running_data_loss += d_loss.item() * bs
        running_pde_loss += p_loss.item() * bs
        running_bc_loss += b_loss.item() * bs
        running_total_loss += tot_loss.item() * bs
        n_samples += bs

        if log_every and step % log_every == 0:
            print(
                f"  epoch {epoch:03d} step {step:04d} | "
                f"total_loss={tot_loss.item():.6e} | "
                f"data_loss={d_loss.item():.6e} | "
                f"pde_loss={p_loss.item():.6e}"
            )

    n_samples = max(n_samples, 1)
    return {
        "train_data_loss": running_data_loss / n_samples,
        "train_pde_loss": running_pde_loss / n_samples,
        "train_bc_loss": running_bc_loss / n_samples,
        "train_total_loss": running_total_loss / n_samples,
    }


def validate_and_load_checkpoint(
    model: PhysicsInformedFNO2d,
    init_ckpt_path: Union[str, Path],
    device: torch.device,
) -> Tuple[str, Optional[int]]:
    """
    Validate checkpoint path, architecture compatibility, tensor shapes, and load model weights.
    Loads model weights only (does NOT restore optimizer/scheduler/epoch state).
    Returns (sha256_hash, source_epoch).
    """
    path = Path(init_ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Initialization checkpoint file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Initialization checkpoint path is not a file: {path}")

    # Compute SHA-256 hash
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    ckpt_hash = h.hexdigest()

    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint at {path} does not contain 'model_state_dict'")

    source_state = ckpt["model_state_dict"]
    target_state = model.state_dict()

    # Verify keys
    missing_keys = set(target_state.keys()) - set(source_state.keys())
    if missing_keys:
        raise RuntimeError(f"Checkpoint is missing required model keys: {sorted(missing_keys)}")
    unexpected_keys = set(source_state.keys()) - set(target_state.keys())
    if unexpected_keys:
        raise RuntimeError(f"Checkpoint contains unexpected keys for current architecture: {sorted(unexpected_keys)}")

    # Verify tensor shapes
    shape_mismatches = []
    for k in target_state.keys():
        if target_state[k].shape != source_state[k].shape:
            shape_mismatches.append(f"{k}: expected {tuple(target_state[k].shape)}, got {tuple(source_state[k].shape)}")
    if shape_mismatches:
        raise RuntimeError(f"Model architecture tensor shape mismatch in checkpoint: {shape_mismatches}")

    # Load weights strictly
    model.load_state_dict(source_state, strict=True)
    source_epoch = ckpt.get("epoch")
    print(f"Successfully loaded model weights from checkpoint: {path}")
    print(f"  Source Epoch: {source_epoch} | SHA-256: {ckpt_hash[:16]}... (Trainable Parameters: {model.count_parameters(trainable_only=True):,})")
    return ckpt_hash, source_epoch


def save_checkpoint(
    path: Path,
    model: PhysicsInformedFNO2d,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    best_val_loss: float,
    cfg: dict,
    history: list,
    norm_stats: Optional[dict] = None,
    val_metrics: Optional[dict] = None,
) -> None:
    """Save model checkpoint safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_loss": best_val_loss,
        "config": cfg,
        "history": history,
        "val_metrics": val_metrics,
        "n_parameters": model.count_parameters(trainable_only=True),
        "norm_stats": norm_stats,
    }
    torch.save(payload, path)


def train_pinn(
    seed: int = 42,
    batch_size: int = 16,
    epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    patience: int = 15,
    min_delta: float = 1e-6,
    w_data: float = 1.0,
    w_pde: float = 6.0e-11,
    w_bc: float = 1.35e-5,
    init_ckpt: Optional[Union[str, Path]] = None,
    min_global_span: Optional[float] = 100.0,
    max_val_mae: Optional[float] = 50.0,
    abort_on_collapse: bool = False,
    best_criterion: str = "val_data_loss",
    ckpt_dir: Optional[Union[str, Path]] = None,
    results_dir: Optional[Union[str, Path]] = None,
    n_train_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute full production PINN training pipeline."""
    start_total_time = time.time()
    set_seed(seed)
    device = get_device()

    print("=" * 65)
    print("STARTING FULL PRODUCTION PHYSICS-INFORMED FNO (PINN) TRAINING")
    print("=" * 65)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {weight_decay}")
    print(f"Loss weights -> w_data: {w_data}, w_pde: {w_pde}, w_bc: {w_bc}")

    # 1. Load Data (TRAIN & VALIDATION ONLY - TEST SET IS LOCKED)
    train_ds = ChipThermalDataset("train", normalize=True, return_meta=True)
    val_ds = ChipThermalDataset("validation", normalize=True, return_meta=True)
    if n_train_samples is not None:
        train_ds = torch.utils.data.Subset(train_ds, range(n_train_samples))
    stats = load_ml_norm_stats()

    tstats = stats["target"]["theta"]
    target_mean = float(tstats["mean"])
    target_std = float(tstats["std"])

    # Load ground truth validation temperatures directly from validation.h5 for exact Kelvin metrics
    with h5py.File(val_ds.h5_path, "r") as f:
        all_gt_T = np.asarray(f["targets/T"][:], dtype=np.float32)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=custom_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=custom_collate,
    )

    n_train = len(train_ds)
    n_val = len(val_ds)
    print(f"Dataset sizes: Train={n_train:,}, Validation={n_val:,} (TEST SET IS LOCKED)")

    # 2. Construct Model & Assert Parameter Count
    model = PhysicsInformedFNO2d(
        in_channels=23,
        out_channels=1,
        width=32,
        modes1=8,
        modes2=8,
        n_layers=4,
        w_data=w_data,
        w_pde=w_pde,
        w_bc=w_bc,
    ).to(device)

    n_params = model.count_parameters(trainable_only=True)
    expected_params = 530433
    assert n_params == expected_params, (
        f"CRITICAL: Parameter count mismatch! Expected {expected_params:,}, got {n_params:,}"
    )
    print(f"Model constructed: PhysicsInformedFNO2d | Trainable parameters: {n_params:,} (PASS)")

    # Optional checkpoint initialization (loads weights only, fresh optimizer/scheduler)
    init_ckpt_hash = None
    init_ckpt_epoch = None
    if init_ckpt is not None:
        init_ckpt_hash, init_ckpt_epoch = validate_and_load_checkpoint(model, init_ckpt, device)

    # 3. Setup Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    # 4. Checkpoint & Results Paths (Dedicated Production Directories)
    if ckpt_dir is None:
        ckpt_dir = PROJECT_ROOT / "experiments" / "physics_guided" / "checkpoints" / "pinn_full_data_v1"
    else:
        ckpt_dir = Path(ckpt_dir)

    if results_dir is None:
        results_dir = PROJECT_ROOT / "outputs" / "pinn_full_data_v1"
    else:
        results_dir = Path(results_dir)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = ckpt_dir / "pinn_full_data_best.pt"
    last_ckpt_path = ckpt_dir / "pinn_full_data_last.pt"
    history_json_path = results_dir / "pinn_full_data_history.json"

    cfg = {
        "seed": seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "min_delta": min_delta,
        "w_data": w_data,
        "w_pde": w_pde,
        "w_bc": w_bc,
        "init_ckpt": str(init_ckpt) if init_ckpt else None,
        "init_ckpt_hash": init_ckpt_hash,
        "init_ckpt_epoch": init_ckpt_epoch,
        "min_global_span": min_global_span,
        "max_val_mae": max_val_mae,
        "abort_on_collapse": abort_on_collapse,
        "best_criterion": best_criterion,
        "in_channels": 23,
        "out_channels": 1,
        "width": 32,
        "modes1": 8,
        "modes2": 8,
        "n_layers": 4,
        "n_parameters": n_params,
        "n_train_samples": n_train,
        "n_val_samples": n_val,
    }

    # Initial Validation Evaluation
    init_val = evaluate_validation_set(
        model=model,
        loader=val_loader,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
        all_gt_T=all_gt_T,
        w_data=w_data,
        w_pde=w_pde,
        w_bc=w_bc,
    )
    print(
        f"Initial Val -> Total Loss: {init_val['val_total_loss']:.6e} "
        f"(Data: {init_val['val_data_loss']:.6e}, PDE: {init_val['val_pde_loss']:.6e}, BC: {init_val['val_bc_loss']:.6e}) | "
        f"MAE: {init_val.get('mae_k', float('nan')):.2f} K | R2: {init_val.get('r2_score', float('nan')):.4f} | "
        f"Span: {init_val.get('pred_t_span', float('nan')):.2f} K"
    )
    print("-" * 65)

    history = []
    best_val_metric = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
            epoch=epoch,
            log_every=100,
        )

        val_metrics = evaluate_validation_set(
            model=model,
            loader=val_loader,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
            all_gt_T=all_gt_T,
            w_data=w_data,
            w_pde=w_pde,
            w_bc=w_bc,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t0

        record = {
            "epoch": epoch,
            "train_total_loss": train_metrics["train_total_loss"],
            "train_data_loss": train_metrics["train_data_loss"],
            "train_pde_loss": train_metrics["train_pde_loss"],
            "train_bc_loss": train_metrics["train_bc_loss"],
            "learning_rate": current_lr,
            "epoch_time_sec": epoch_time,
        }
        record.update(val_metrics)
        history.append(record)

        # Numerical stability & collapse safeguards
        pred_span = val_metrics.get("pred_t_span", 0.0)
        val_mae = val_metrics.get("mae_k", 0.0)

        if min_global_span is not None and pred_span < min_global_span:
            warn_msg = (
                f"[SAFEGUARD WARNING] Epoch {epoch}: Predicted global temperature span "
                f"({pred_span:.2f} K) is below threshold ({min_global_span:.2f} K). "
                f"Possible temperature field collapse!"
            )
            print(warn_msg)
            if abort_on_collapse:
                raise RuntimeError(f"ABORT: Safeguard triggered - {warn_msg}")

        if max_val_mae is not None and val_mae > max_val_mae:
            warn_msg = (
                f"[SAFEGUARD WARNING] Epoch {epoch}: Validation MAE "
                f"({val_mae:.2f} K) exceeds threshold ({max_val_mae:.2f} K)."
            )
            print(warn_msg)
            if abort_on_collapse:
                raise RuntimeError(f"ABORT: Safeguard triggered - {warn_msg}")

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"Train Tot: {train_metrics['train_total_loss']:.6e} (PDE: {train_metrics['train_pde_loss']:.4e}) | "
            f"Val Tot: {val_metrics['val_total_loss']:.6e} (PDE: {val_metrics['val_pde_loss']:.4e}) | "
            f"MAE: {val_mae:.2f}K | R2: {val_metrics.get('r2_score', float('nan')):.4f} | "
            f"Span: {pred_span:.1f}K | LR: {current_lr:.3e} ({epoch_time:.1f}s)"
        )

        save_checkpoint(
            last_ckpt_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_metric,
            cfg,
            history,
            norm_stats=stats,
            val_metrics=val_metrics,
        )

        # Early stopping and checkpoint selection
        if best_criterion not in val_metrics:
            raise KeyError(f"Selected best_criterion '{best_criterion}' not found in validation metrics: {list(val_metrics.keys())}")
        current_metric = val_metrics[best_criterion]
        improved = early_stopping.step(current_metric)
        if improved:
            best_val_metric = current_metric
            best_epoch = epoch
            save_checkpoint(
                best_ckpt_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_metric,
                cfg,
                history,
                norm_stats=stats,
                val_metrics=val_metrics,
            )
            print(f"  -> New best checkpoint: {best_ckpt_path} ({best_criterion}={best_val_metric:.6e})")

        with open(history_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "config": cfg,
                "best_epoch": best_epoch,
                "best_criterion": best_criterion,
                "best_val_metric": best_val_metric,
                "history": history,
            }, f, indent=2)

        if early_stopping.should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch} (patience={early_stopping.patience})")
            break

    total_time = time.time() - start_total_time

    print("=" * 65)
    print("PINN TRAINING COMPLETE")
    print("=" * 65)
    print(f"Total training time:  {total_time:.2f} seconds ({total_time / 60.0:.2f} minutes)")
    print(f"Best epoch:            {best_epoch}")
    print(f"Best {best_criterion}: {best_val_metric:.6e}")
    print(f"Best checkpoint:       {best_ckpt_path}")
    print(f"History file:          {history_json_path}")
    print("=" * 65)

    return {
        "best_epoch": best_epoch,
        "best_val_metric": best_val_metric,
        "best_criterion": best_criterion,
        "total_time_sec": total_time,
        "best_ckpt_path": str(best_ckpt_path),
        "history_json_path": str(history_json_path),
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for production PINN training."""
    parser = argparse.ArgumentParser(description="Full Production Training for PhysicsInformedFNO2d.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15)")
    parser.add_argument("--min-delta", type=float, default=1e-6, help="Early stopping min delta (default: 1e-6)")
    parser.add_argument("--w-data", type=float, default=1.0, help="Data loss weight (default: 1.0)")
    parser.add_argument("--w-pde", type=float, default=6.0e-11, help="PDE loss weight (default: 6.0e-11)")
    parser.add_argument("--w-bc", type=float, default=1.35e-5, help="BC loss weight (default: 1.35e-5)")
    parser.add_argument("--init-ckpt", type=str, default=None, help="Path to checkpoint to initialize model weights (default: None)")
    parser.add_argument("--min-global-span", type=float, default=100.0, help="Minimum validation temperature span threshold for collapse safeguard (default: 100.0 K)")
    parser.add_argument("--max-val-mae", type=float, default=50.0, help="Maximum validation MAE threshold for safeguard (default: 50.0 K)")
    parser.add_argument("--abort-on-collapse", action="store_true", help="Abort training if span or MAE safeguards are violated (default: False, warn only)")
    parser.add_argument("--best-criterion", type=str, default="val_data_loss", choices=["val_data_loss", "val_total_loss", "mae_k"], help="Metric to select best checkpoint (default: val_data_loss)")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--results-dir", type=str, default=None, help="Directory to save results and history")
    parser.add_argument("--n-train-samples", type=int, default=None, help="Subset of training samples (default: None = all 6400)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()


def main():
    """Main entrypoint for running PINN training."""
    args = parse_args()
    train_pinn(
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        w_data=args.w_data,
        w_pde=args.w_pde,
        w_bc=args.w_bc,
        init_ckpt=args.init_ckpt,
        min_global_span=args.min_global_span,
        max_val_mae=args.max_val_mae,
        abort_on_collapse=args.abort_on_collapse,
        best_criterion=args.best_criterion,
        ckpt_dir=args.ckpt_dir,
        results_dir=args.results_dir,
        n_train_samples=args.n_train_samples,
    )


if __name__ == "__main__":
    # DO NOT EXECUTE TRAINING DIRECTLY UNLESS INTENDED
    main()
