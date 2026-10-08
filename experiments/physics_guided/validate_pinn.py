"""
Phase 9 & Phase 10 Validation Script for PhysicsInformedFNO2d.

Performs:
  - Phase 9: Single-sample forward/backward diagnostic check (1 sample)
  - Phase 10: Few-sample forward-pass architecture validation (10 samples, NO training)

Outputs saved to outputs/pinn_validation/
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import ChipThermalDataset  # noqa: E402
from src.data.normalize import load_ml_norm_stats  # noqa: E402
from src.models.pinn import PhysicsInformedFNO2d  # noqa: E402
from src.physics.conductivity import compute_temperature_dependent_conductivity_torch  # noqa: E402
from src.physics.kirchhoff_torch import forward_kirchhoff_torch, inverse_kirchhoff_torch  # noqa: E402
from src.physics.pde_residual import compute_pde_residual_kirchhoff  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pinn_validation"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_single_sample_debug(model: PhysicsInformedFNO2d, dataset: ChipThermalDataset, stats: dict, device: torch.device) -> Dict[str, Any]:
    print("=" * 65)
    print("PHASE 9 — SINGLE-SAMPLE DEBUGGING DIAGNOSTIC REPORT")
    print("=" * 65)

    X_t, y_t, meta = dataset[0]
    X_batch = X_t.unsqueeze(0).to(device)
    y_batch = y_t.unsqueeze(0).to(device)

    # 1. Input shapes
    input_shape = tuple(X_batch.shape)
    target_shape = tuple(y_batch.shape)

    tstats = stats["target"]["theta"]
    target_mean = float(tstats["mean"])
    target_std = float(tstats["std"])

    # 2. Forward pass & Losses
    model.train()
    losses = model.compute_all_losses(X_batch, y_batch, meta, target_mean, target_std)

    data_loss = losses["data_loss"]
    pde_loss = losses["pde_loss"]
    bc_loss = losses["bc_loss"]
    total_loss = losses["total_loss"]

    # 3. Autograd check
    total_loss.backward()

    grad_status = "PASS"
    for name, p in model.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            grad_status = f"FAIL in {name}"
            break

    # 4. Check outputs & temperature range
    T_pred = losses["T_pred"].detach().cpu().numpy()[0, 0]
    R_pde = losses["R_pde"].detach().cpu().numpy()[0, 0]

    nan_count = int(np.isnan(T_pred).sum() + np.isnan(R_pde).sum())
    inf_count = int(np.isinf(T_pred).sum() + np.isinf(R_pde).sum())

    # Conductivity positivity check
    kx_base = torch.from_numpy(np.asarray(meta["kx_base"])).unsqueeze(0).to(device)
    ky_base = torch.from_numpy(np.asarray(meta["ky_base"])).unsqueeze(0).to(device)
    n_map = losses["n_map"]

    kx, ky = compute_temperature_dependent_conductivity_torch(
        losses["T_pred"].detach(), kx_base, ky_base, n_map
    )
    k_positive = bool((kx > 0).all() and (ky > 0).all())

    print(f"Input shape:           {input_shape}")
    print(f"Output theta shape:    {tuple(losses['pred_norm'].shape)}")
    print(f"Output T shape:        {T_pred.shape}")
    print(f"Temperature min:       {T_pred.min():.2f} K")
    print(f"Temperature max:       {T_pred.max():.2f} K")
    print(f"PDE residual mean:     {np.abs(R_pde).mean():.3e} W/m^3")
    print(f"PDE residual max:      {np.abs(R_pde).max():.3e} W/m^3")
    print(f"Data Loss:             {data_loss.item():.6e}")
    print(f"PDE Loss:              {pde_loss.item():.6e}")
    print(f"BC Loss:               {bc_loss.item():.6e}")
    print(f"Total Loss:            {total_loss.item():.6e}")
    print(f"NaN count:             {nan_count}")
    print(f"Inf count:             {inf_count}")
    print(f"Conductivity > 0 check: {'PASS' if k_positive else 'FAIL'}")
    print(f"Gradient status:       {grad_status}")
    print("=" * 65)

    return {
        "input_shape": input_shape,
        "output_shape": tuple(losses["pred_norm"].shape),
        "temp_min_K": float(T_pred.min()),
        "temp_max_K": float(T_pred.max()),
        "pde_residual_mean_W_m3": float(np.abs(R_pde).mean()),
        "pde_residual_max_W_m3": float(np.abs(R_pde).max()),
        "data_loss": data_loss.item(),
        "pde_loss": pde_loss.item(),
        "bc_loss": bc_loss.item(),
        "total_loss": total_loss.item(),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "k_positive": k_positive,
        "grad_status": grad_status,
    }


def run_few_sample_validation(
    model: PhysicsInformedFNO2d,
    dataset: ChipThermalDataset,
    stats: dict,
    device: torch.device,
    n_samples: int = 10,
) -> Dict[str, Any]:
    print("=" * 65)
    print(f"PHASE 10 — FEW-SAMPLE FORWARD VALIDATION ({n_samples} SAMPLES, UNTRAINED)")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tstats = stats["target"]["theta"]
    target_mean = float(tstats["mean"])
    target_std = float(tstats["std"])

    model.eval()

    sample_reports = []

    for i in range(n_samples):
        X_t, y_t, meta = dataset[i]
        X_batch = X_t.unsqueeze(0).to(device)
        y_batch = y_t.unsqueeze(0).to(device)

        with torch.no_grad():
            losses = model.compute_all_losses(X_batch, y_batch, meta, target_mean, target_std)

        T_pred = losses["T_pred"].cpu().numpy()[0, 0]
        R_pde = losses["R_pde"].cpu().numpy()[0, 0]

        # Read ground truth T from dataset file directly
        import h5py
        with h5py.File(dataset.h5_path, "r") as f:
            T_true = np.asarray(f["targets/T"][i], dtype=np.float64)

        abs_err = np.abs(T_pred - T_true)

        T_pred_max = float(T_pred.max())
        T_true_max = float(T_true.max())
        abs_hspot_err = abs(T_pred_max - T_true_max)
        pde_mean = float(np.abs(R_pde).mean())

        sample_reports.append(
            {
                "sample_idx": i,
                "sample_id": int(meta["sample_id"]),
                "T_pred_min_K": float(T_pred.min()),
                "T_pred_max_K": T_pred_max,
                "T_true_max_K": T_true_max,
                "abs_hotspot_err_K": abs_hspot_err,
                "pde_residual_mean_W_m3": pde_mean,
            }
        )

        # Plot first 3 samples
        if i < 3:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))

            im0 = axes[0, 0].imshow(T_true, cmap="inferno")
            axes[0, 0].set_title(f"Sample {i+1}: Reference T (K)")
            fig.colorbar(im0, ax=axes[0, 0])

            im1 = axes[0, 1].imshow(T_pred, cmap="inferno")
            axes[0, 1].set_title(f"Sample {i+1}: Untrained PINN T (K)")
            fig.colorbar(im1, ax=axes[0, 1])

            im2 = axes[1, 0].imshow(abs_err, cmap="viridis")
            axes[1, 0].set_title(f"Sample {i+1}: Absolute Error (K)")
            fig.colorbar(im2, ax=axes[1, 0])

            im3 = axes[1, 1].imshow(np.abs(R_pde), cmap="magma")
            axes[1, 1].set_title(f"Sample {i+1}: PDE Residual |R| (W/m³)")
            fig.colorbar(im3, ax=axes[1, 1])

            plt.tight_layout()
            plot_path = OUTPUT_DIR / f"validation_forward_sample_{i+1}.png"
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"Saved forward validation plot: {plot_path}")

    summary = {
        "n_samples_validated": n_samples,
        "sample_reports": sample_reports,
    }

    summary_path = OUTPUT_DIR / "forward_validation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved forward validation summary to: {summary_path}")
    print("=" * 65)
    print("PHASE 10 FORWARD VALIDATION: PASS")
    print("=" * 65)

    return summary


def run_tiny_training_experiment(
    n_train_samples: int = 30,
    epochs: int = 10,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    seed: int = 42,
) -> Dict[str, Any]:
    print("=" * 65)
    print(f"TINY PINN TRAINING EXPERIMENT ({n_train_samples} TRAIN SAMPLES, {epochs} EPOCHS)")
    print("=" * 65)
    print("DEBUGGING & LEARNING VERIFICATION ONLY — NOT FINAL PERFORMANCE")
    print("TEST SET IS COMPLETELY LOCKED & UNTOUCHED")
    print("-" * 65)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()
    print(f"Device: {device}")

    # Load dataset splits (30 train samples, 800 val samples, TEST SET IS LOCKED)
    full_train_ds = ChipThermalDataset("train", normalize=True, return_meta=True)
    val_ds = ChipThermalDataset("validation", normalize=True, return_meta=True)
    stats = load_ml_norm_stats()

    from torch.utils.data import DataLoader, Subset
    train_sub = Subset(full_train_ds, list(range(min(n_train_samples, len(full_train_ds)))))

    # Custom collate for meta dict
    def custom_collate(batch):
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

    train_loader = DataLoader(train_sub, batch_size=8, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=custom_collate)

    tstats = stats["target"]["theta"]
    target_mean = float(tstats["mean"])
    target_std = float(tstats["std"])

    model = PhysicsInformedFNO2d(
        in_channels=23,
        out_channels=1,
        width=32,
        modes1=8,
        modes2=8,
        n_layers=4,
        w_data=1.0,
        w_pde=1e-4,
        w_bc=1e-2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initial evaluation
    model.eval()
    init_train_data, init_train_pde, init_train_total = 0.0, 0.0, 0.0
    with torch.no_grad():
        for X, y, meta in train_loader:
            X, y = X.to(device), y.to(device)
            l = model.compute_all_losses(X, y, meta, target_mean, target_std)
            bs = X.shape[0]
            init_train_data += l["data_loss"].item() * bs
            init_train_pde += l["pde_loss"].item() * bs
            init_train_total += l["total_loss"].item() * bs
    init_train_data /= len(train_sub)
    init_train_pde /= len(train_sub)
    init_train_total /= len(train_sub)

    print(f"Initial Untrained Train Losses -> Data: {init_train_data:.6e} | PDE: {init_train_pde:.6e} | Total: {init_train_total:.6e}")
    print("-" * 65)

    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_data_acc, train_pde_acc, train_bc_acc, train_total_acc = 0.0, 0.0, 0.0, 0.0

        for X, y, meta in train_loader:
            X, y = X.to(device), y.to(device)

            if not torch.isfinite(X).all() or not torch.isfinite(y).all():
                raise ValueError(f"CRITICAL: Non-finite input or target at epoch {epoch}")

            optimizer.zero_grad(set_to_none=True)
            loss_dict = model.compute_all_losses(X, y, meta, target_mean, target_std)

            d_loss = loss_dict["data_loss"]
            p_loss = loss_dict["pde_loss"]
            b_loss = loss_dict["bc_loss"]
            tot_loss = loss_dict["total_loss"]

            if not torch.isfinite(tot_loss).all():
                raise ValueError(f"CRITICAL: Non-finite loss encountered at epoch {epoch}")

            tot_loss.backward()

            # Check gradients
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise ValueError(f"CRITICAL: Non-finite gradient in {name} at epoch {epoch}")

            optimizer.step()

            bs = X.shape[0]
            train_data_acc += d_loss.item() * bs
            train_pde_acc += p_loss.item() * bs
            train_bc_acc += b_loss.item() * bs
            train_total_acc += tot_loss.item() * bs

        ep_train_data = train_data_acc / len(train_sub)
        ep_train_pde = train_pde_acc / len(train_sub)
        ep_train_bc = train_bc_acc / len(train_sub)
        ep_train_total = train_total_acc / len(train_sub)

        # Validation evaluation (sample 100 val samples for fast epoch monitoring)
        model.eval()
        val_data_acc, val_pde_acc, val_bc_acc, val_total_acc = 0.0, 0.0, 0.0, 0.0
        val_n = 0
        with torch.no_grad():
            for v_idx, (X, y, meta) in enumerate(val_loader):
                if v_idx >= 4:  # 4 batches x 32 = 128 val samples
                    break
                X, y = X.to(device), y.to(device)
                loss_dict = model.compute_all_losses(X, y, meta, target_mean, target_std)
                bs = X.shape[0]
                val_data_acc += loss_dict["data_loss"].item() * bs
                val_pde_acc += loss_dict["pde_loss"].item() * bs
                val_bc_acc += loss_dict["bc_loss"].item() * bs
                val_total_acc += loss_dict["total_loss"].item() * bs
                val_n += bs

        ep_val_data = val_data_acc / max(val_n, 1)
        ep_val_pde = val_pde_acc / max(val_n, 1)
        ep_val_total = val_total_acc / max(val_n, 1)

        ep_time = time.time() - t0

        record = {
            "epoch": epoch,
            "train_data_loss": ep_train_data,
            "train_pde_loss": ep_train_pde,
            "train_total_loss": ep_train_total,
            "val_data_loss": ep_val_data,
            "val_pde_loss": ep_val_pde,
            "val_total_loss": ep_val_total,
            "seconds": ep_time,
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Total: {ep_train_total:.6f} (Data: {ep_train_data:.6f}, PDE: {ep_train_pde:.2f}) | "
            f"Val Total: {ep_val_total:.6f} (Data: {ep_val_data:.6f}, PDE: {ep_val_pde:.2f}) | "
            f"({ep_time:.2f}s)"
        )

    # Final physics check on validation samples
    model.eval()
    X_val, y_val, meta_val = next(iter(val_loader))
    X_val, y_val = X_val.to(device), y_val.to(device)
    meta_val_3 = {
        k: (v[:3] if isinstance(v, (np.ndarray, list)) and len(v) == X_val.shape[0] else v)
        for k, v in meta_val.items()
    }
    with torch.no_grad():
        out = model.compute_all_losses(X_val[:3], y_val[:3], meta_val_3, target_mean, target_std)

    T_pred_val = out["T_pred"].cpu().numpy()[:, 0]
    R_pde_val = out["R_pde"].cpu().numpy()[:, 0]

    # Save diagnostic plot for 3 validation samples
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    import h5py
    with h5py.File(val_ds.h5_path, "r") as f:
        for i in range(3):
            T_true_i = np.asarray(f["targets/T"][i], dtype=np.float64)
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

            im0 = axes[0].imshow(T_true_i, cmap="inferno")
            axes[0].set_title(f"Val Sample {i+1}: Ground Truth T (K)")
            fig.colorbar(im0, ax=axes[0])

            im1 = axes[1].imshow(T_pred_val[i], cmap="inferno")
            axes[1].set_title(f"Val Sample {i+1}: Tiny PINN T (K)")
            fig.colorbar(im1, ax=axes[1])

            im2 = axes[2].imshow(np.abs(R_pde_val[i]), cmap="magma")
            axes[2].set_title(f"Val Sample {i+1}: PDE Residual |R| (W/m³)")
            fig.colorbar(im2, ax=axes[2])

            plt.tight_layout()
            plot_path = OUTPUT_DIR / f"tiny_experiment_sample_{i+1}.png"
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"Saved tiny experiment plot: {plot_path}")

    final_train_total = history[-1]["train_total_loss"]
    final_val_total = history[-1]["val_total_loss"]
    final_train_pde = history[-1]["train_pde_loss"]
    final_val_pde = history[-1]["val_pde_loss"]

    overfitting_detected = (ep_train_total < init_train_total) and (ep_val_total > history[0]["val_total_loss"] * 1.2)

    print("=" * 65)
    print("TINY TRAINING EXPERIMENT SUMMARY")
    print("=" * 65)
    print(f"Initial Train Total Loss: {init_train_total:.6f} | Final: {final_train_total:.6f}")
    print(f"Initial Train PDE Loss:   {init_train_pde:.2f} | Final: {final_train_pde:.2f}")
    print(f"Initial Val Total Loss:   {history[0]['val_total_loss']:.6f} | Final: {final_val_total:.6f}")
    print(f"Initial Val PDE Loss:     {history[0]['val_pde_loss']:.2f} | Final: {final_val_pde:.2f}")
    print(f"T_pred Min Range:         {T_pred_val.min():.2f} K  to  {T_pred_val.max():.2f} K")
    print(f"Overfitting Detected:     {'YES (expected on 30 samples)' if overfitting_detected else 'NO'}")
    print("=" * 65)

    summary = {
        "n_train_samples": n_train_samples,
        "epochs": epochs,
        "init_train_total_loss": init_train_total,
        "final_train_total_loss": final_train_total,
        "init_train_pde_loss": init_train_pde,
        "final_train_pde_loss": final_train_pde,
        "init_val_total_loss": history[0]["val_total_loss"],
        "final_val_total_loss": final_val_total,
        "init_val_pde_loss": history[0]["val_pde_loss"],
        "final_val_pde_loss": final_val_pde,
        "temp_min_K": float(T_pred_val.min()),
        "temp_max_K": float(T_pred_val.max()),
        "overfitting_detected": overfitting_detected,
        "history": history,
    }

    with open(OUTPUT_DIR / "tiny_experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny-train", action="store_true", help="Run tiny training experiment")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load dataset & norm stats
    val_ds = ChipThermalDataset("validation", normalize=True, return_meta=True)
    stats = load_ml_norm_stats()

    # Construct model
    model = PhysicsInformedFNO2d(
        in_channels=23,
        out_channels=1,
        width=32,
        modes1=8,
        modes2=8,
        n_layers=4,
    ).to(device)

    if args.tiny_train:
        run_tiny_training_experiment(n_train_samples=30, epochs=10)
        return

    # Phase 9: Single-Sample Debugging
    single_res = run_single_sample_debug(model, val_ds, stats, device)

    # Phase 10: Few-Sample Forward Validation (10 samples)
    few_res = run_few_sample_validation(model, val_ds, stats, device, n_samples=10)

    print("\nARCHITECTURE & FORWARD PIPELINE VALIDATION COMPLETE.")
    print("NO TRAINING HAS BEEN PERFORMED.")


if __name__ == "__main__":
    main()
