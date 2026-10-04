"""
Small debug training script for baseline 2D Fourier Neural Operator (FNO).

Dataset subset: 100 training samples, 20 validation samples.
Epochs: 5, Batch size: 8, Optimizer: AdamW (lr=0.001, weight_decay=0.0001).

Strictly verifies:
  - Input/target finiteness before training
  - Prediction/loss/gradient finiteness during training
  - Validation prediction finiteness after every epoch

Saves best checkpoint to checkpoints/debug/fno_debug_best.pt
Saves history and plots to outputs/fno_debug/
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_data import ChipThermalDataset  # noqa: E402
from models.fno import FNO2d  # noqa: E402


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_debug_training(
    n_train_samples: int = 100,
    n_val_samples: int = 20,
    batch_size: int = 8,
    epochs: int = 5,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    seed: int = 42,
) -> Dict[str, Any]:
    start_total_time = time.time()
    set_seed(seed)

    device = torch.device("cpu")
    print("=" * 65)
    print("STARTING FNO BASELINE DEBUG TRAINING RUN (5 EPOCHS, 100 SAMPLES)")
    print("=" * 65)
    print(f"Device: {device}")
    print(f"Train samples: {n_train_samples}")
    print(f"Validation samples: {n_val_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")

    # 1. Load Subsets
    full_train_ds = ChipThermalDataset("train", normalize=True)
    full_val_ds = ChipThermalDataset("validation", normalize=True)

    train_indices = list(range(min(n_train_samples, len(full_train_ds))))
    val_indices = list(range(min(n_val_samples, len(full_val_ds))))

    train_ds = Subset(full_train_ds, train_indices)
    val_ds = Subset(full_val_ds, val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. Check initial batch finiteness
    first_X, first_y = next(iter(train_loader))
    if not torch.isfinite(first_X).all():
        raise ValueError("CRITICAL: Non-finite values (NaN/Inf) found in input X before training!")
    if not torch.isfinite(first_y).all():
        raise ValueError("CRITICAL: Non-finite values (NaN/Inf) found in target y before training!")
    print("[Check] Pre-training Data Finiteness: PASS (no NaN/Inf in X or y)")

    # 3. Model construction
    model = FNO2d(
        in_channels=23,
        out_channels=1,
        width=32,
        modes1=8,
        modes2=8,
        n_layers=4,
        use_grid=True,
    ).to(device)
    n_params = model.count_parameters()
    print(f"Model: FNO2d  Trainable Parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    ckpt_dir = PROJECT_ROOT / "checkpoints" / "debug"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "fno_debug_best.pt"

    output_dir = PROJECT_ROOT / "outputs" / "fno_debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.json"

    # Evaluate initial (untrained) train and val loss
    model.eval()
    initial_train_loss = 0.0
    with torch.no_grad():
        for X, y in train_loader:
            pred = model(X)
            if not torch.isfinite(pred).all():
                raise ValueError("CRITICAL: Initial train prediction contains NaN/Inf!")
            initial_train_loss += criterion(pred, y).item() * X.shape[0]
    initial_train_loss /= len(train_ds)

    initial_val_loss = 0.0
    with torch.no_grad():
        for X, y in val_loader:
            pred = model(X)
            if not torch.isfinite(pred).all():
                raise ValueError("CRITICAL: Initial validation prediction contains NaN/Inf!")
            initial_val_loss += criterion(pred, y).item() * X.shape[0]
    initial_val_loss /= len(val_ds)

    print(f"Initial Untrained Train Loss: {initial_train_loss:.6e}")
    print(f"Initial Untrained Val Loss:   {initial_val_loss:.6e}")
    print("-" * 65)

    history = []
    best_val_loss = float("inf")
    all_losses_finite = True
    all_grads_finite = True

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_train_loss = 0.0

        for batch_idx, (X, y) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            pred = model(X)
            if not torch.isfinite(pred).all():
                all_losses_finite = False
                raise ValueError(f"CRITICAL: Non-finite prediction at epoch {epoch}, batch {batch_idx}")

            loss = criterion(pred, y)
            if not torch.isfinite(loss).all():
                all_losses_finite = False
                raise ValueError(f"CRITICAL: Non-finite loss at epoch {epoch}, batch {batch_idx}")

            loss.backward()

            # Verify gradients
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    all_grads_finite = False
                    raise ValueError(f"CRITICAL: Non-finite gradient in {name} at epoch {epoch}")

            optimizer.step()
            running_train_loss += loss.item() * X.shape[0]

        epoch_train_loss = running_train_loss / len(train_ds)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                pred = model(X)
                if not torch.isfinite(pred).all():
                    all_losses_finite = False
                    raise ValueError(f"CRITICAL: Non-finite validation prediction at epoch {epoch}")
                val_loss_batch = criterion(pred, y)
                if not torch.isfinite(val_loss_batch).all():
                    all_losses_finite = False
                    raise ValueError(f"CRITICAL: Non-finite validation loss at epoch {epoch}")
                running_val_loss += val_loss_batch.item() * X.shape[0]

        epoch_val_loss = running_val_loss / len(val_ds)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        epoch_time = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "lr": current_lr,
            "epoch_time": epoch_time,
        }
        history.append(record)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {epoch_train_loss:.6e} | "
            f"Val Loss: {epoch_val_loss:.6e} | "
            f"LR: {current_lr:.3e} | "
            f"Time: {epoch_time:.2f} sec"
        )

        # Save best checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "history": history,
                    "config": {
                        "in_channels": 23,
                        "out_channels": 1,
                        "width": 32,
                        "modes1": 8,
                        "modes2": 8,
                        "n_layers": 4,
                        "use_grid": True,
                    },
                    "n_parameters": n_params,
                },
                best_ckpt_path,
            )

    total_runtime = time.time() - start_total_time
    final_train_loss = history[-1]["train_loss"]
    final_val_loss = history[-1]["val_loss"]

    train_loss_reduction = ((initial_train_loss - final_train_loss) / initial_train_loss) * 100.0
    val_loss_reduction = ((initial_val_loss - final_val_loss) / initial_val_loss) * 100.0

    print("=" * 65)
    print("DEBUG TRAINING SUMMARY")
    print("=" * 65)
    print(f"Initial Train Loss:        {initial_train_loss:.6e}")
    print(f"Final Train Loss:          {final_train_loss:.6e}")
    print(f"Training Loss Reduction:   {train_loss_reduction:.2f}%")
    print(f"Initial Validation Loss:   {initial_val_loss:.6e}")
    print(f"Final Validation Loss:     {final_val_loss:.6e}")
    print(f"Validation Loss Reduction: {val_loss_reduction:.2f}%")
    print(f"Best Val Loss:             {best_val_loss:.6e}")
    print(f"Total Runtime:             {total_runtime:.2f} seconds")
    print("=" * 65)

    # Save history
    history_data = {
        "initial_train_loss": initial_train_loss,
        "final_train_loss": final_train_loss,
        "initial_val_loss": initial_val_loss,
        "final_val_loss": final_val_loss,
        "train_loss_reduction_pct": train_loss_reduction,
        "val_loss_reduction_pct": val_loss_reduction,
        "best_val_loss": best_val_loss,
        "n_parameters": n_params,
        "epochs": history,
        "total_runtime_sec": total_runtime,
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)
    print(f"Saved training history to: {history_path}")
    print(f"Saved best checkpoint to:  {best_ckpt_path}")

    # Visual check: plot ground truth theta, predicted theta, and error for 2 validation samples
    model.eval()
    val_X_sample, val_y_sample = next(iter(val_loader))
    with torch.no_grad():
        val_pred_sample = model(val_X_sample)

    gt_samples = val_y_sample[:2, 0].numpy()
    pred_samples = val_pred_sample[:2, 0].numpy()
    error_samples = np.abs(gt_samples - pred_samples)

    np.savez_compressed(
        output_dir / "theta_comparison_samples.npz",
        ground_truth=gt_samples,
        predicted=pred_samples,
        abs_error=error_samples,
    )

    for i in range(min(2, gt_samples.shape[0])):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

        im0 = axes[0].imshow(gt_samples[i], cmap="inferno")
        axes[0].set_title(f"Sample {i+1}: Ground Truth θ")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(pred_samples[i], cmap="inferno")
        axes[1].set_title(f"Sample {i+1}: Predicted θ")
        fig.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(error_samples[i], cmap="viridis")
        axes[2].set_title(f"Sample {i+1}: Abs Error |θ - θ_pred|")
        fig.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plot_path = output_dir / f"theta_comparison_sample_{i+1}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved visualization plot: {plot_path}")

    return {
        "device": str(device),
        "n_train_samples": n_train_samples,
        "n_val_samples": n_val_samples,
        "batch_size": batch_size,
        "epochs": epochs,
        "initial_train_loss": initial_train_loss,
        "final_train_loss": final_train_loss,
        "initial_val_loss": initial_val_loss,
        "final_val_loss": final_val_loss,
        "train_loss_reduction_pct": train_loss_reduction,
        "val_loss_reduction_pct": val_loss_reduction,
        "all_losses_finite": all_losses_finite,
        "all_grads_finite": all_grads_finite,
        "checkpoint_saved": best_ckpt_path.exists(),
        "history_saved": history_path.exists(),
        "visualizations_created": (output_dir / "theta_comparison_sample_1.png").exists(),
        "total_runtime_sec": total_runtime,
    }


if __name__ == "__main__":
    run_debug_training()
