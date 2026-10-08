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
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
    meta = {
        "material": np.stack([item[2]["material"] for item in batch]),
        "n_dielectric": [item[2]["n_dielectric"] for item in batch],
        "n_silicon": [item[2]["n_silicon"] for item in batch],
        "n_copper": [item[2]["n_copper"] for item in batch],
        "T_ref_K": [item[2]["T_ref_K"] for item in batch],
        "h_sink": np.array([item[2]["h_sink"] for item in batch]),
        "T_ambient": np.array([item[2]["T_ambient"] for item in batch]),
        "kx_base": np.stack([item[2]["kx_base"] for item in batch]),
        "ky_base": np.stack([item[2]["ky_base"] for item in batch]),
        "sample_id": [item[2]["sample_id"] for item in batch],
    }
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
) -> Dict[str, float]:
    """Evaluate complete validation dataset across all loss components."""
    model.eval()
    val_data_loss = 0.0
    val_pde_loss = 0.0
    val_bc_loss = 0.0
    val_total_loss = 0.0
    n_samples = 0

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
        val_total_loss += tot_loss.item() * bs
        n_samples += bs

    n_samples = max(n_samples, 1)
    return {
        "val_data_loss": val_data_loss / n_samples,
        "val_pde_loss": val_pde_loss / n_samples,
        "val_bc_loss": val_bc_loss / n_samples,
        "val_total_loss": val_total_loss / n_samples,
    }


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


def save_checkpoint(
    path: Path,
    model: PhysicsInformedFNO2d,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    best_val_loss: float,
    cfg: dict,
    history: list,
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
        "n_parameters": model.count_parameters(trainable_only=True),
    }
    torch.save(payload, path)


def train_pinn(
    seed: int = 42,
    batch_size: int = 8,
    epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    patience: int = 15,
    min_delta: float = 1e-6,
    w_data: float = 1.0,
    w_pde: float = 1e-4,
    w_bc: float = 1e-2,
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
    stats = load_ml_norm_stats()

    tstats = stats["target"]["theta"]
    target_mean = float(tstats["mean"])
    target_std = float(tstats["std"])

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

    # 3. Setup Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    # 4. Checkpoint & Results Paths
    ckpt_dir = PROJECT_ROOT / "experiments" / "physics_guided" / "checkpoints"
    results_dir = PROJECT_ROOT / "experiments" / "physics_guided" / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = ckpt_dir / "pinn_best.pt"
    last_ckpt_path = ckpt_dir / "pinn_last.pt"
    history_json_path = results_dir / "pinn_training_history.json"

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
        "in_channels": 23,
        "out_channels": 1,
        "width": 32,
        "modes1": 8,
        "modes2": 8,
        "n_layers": 4,
        "n_parameters": n_params,
    }

    # Initial Validation Evaluation
    init_val = evaluate_validation_set(model, val_loader, device, target_mean, target_std)
    print(f"Initial Untrained Val Total Loss: {init_val['val_total_loss']:.6e} "
          f"(Data: {init_val['val_data_loss']:.6e}, PDE: {init_val['val_pde_loss']:.6e})")
    print("-" * 65)

    history = []
    best_val_loss = float("inf")
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
            "val_total_loss": val_metrics["val_total_loss"],
            "val_data_loss": val_metrics["val_data_loss"],
            "val_pde_loss": val_metrics["val_pde_loss"],
            "val_bc_loss": val_metrics["val_bc_loss"],
            "learning_rate": current_lr,
            "epoch_time_sec": epoch_time,
        }
        history.append(record)

        val_tot = val_metrics["val_total_loss"]
        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"Train Total: {train_metrics['train_total_loss']:.6e} (PDE: {train_metrics['train_pde_loss']:.4e}) | "
            f"Val Total: {val_tot:.6e} (PDE: {val_metrics['val_pde_loss']:.4e}) | "
            f"LR: {current_lr:.3e} ({epoch_time:.1f}s)"
        )

        save_checkpoint(last_ckpt_path, model, optimizer, scheduler, epoch, best_val_loss, cfg, history)

        improved = early_stopping.step(val_tot)
        if improved:
            best_val_loss = val_tot
            best_epoch = epoch
            save_checkpoint(best_ckpt_path, model, optimizer, scheduler, epoch, best_val_loss, cfg, history)
            print(f"  -> New best checkpoint: {best_ckpt_path} (val_total_loss={best_val_loss:.6e})")

        # Save history JSON per epoch
        with open(history_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "config": cfg,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "history": history,
            }, f, indent=2)

        if early_stopping.should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch} (patience={early_stopping.patience})")
            break

    total_time = time.time() - start_total_time

    print("=" * 65)
    print("PINN TRAINING COMPLETE")
    print("=" * 65)
    print(f"Total training time: {total_time:.2f} seconds ({total_time / 60.0:.2f} minutes)")
    print(f"Best epoch:           {best_epoch}")
    print(f"Best val total loss:  {best_val_loss:.6e}")
    print(f"Best checkpoint:      {best_ckpt_path}")
    print(f"History file:         {history_json_path}")
    print("=" * 65)

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_time_sec": total_time,
        "best_ckpt_path": str(best_ckpt_path),
        "history_json_path": str(history_json_path),
    }


def main():
    """Main entrypoint for running PINN training."""
    train_pinn()


if __name__ == "__main__":
    # DO NOT EXECUTE TRAINING DIRECTLY UNLESS INTENDED
    main()
