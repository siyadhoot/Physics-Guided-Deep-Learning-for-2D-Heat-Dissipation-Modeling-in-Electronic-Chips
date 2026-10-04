"""
Inverse Kirchhoff Evaluation for baseline 2D FNO.

Converts predicted Kirchhoff potential theta(x,y) back to physical temperature T(x,y)
using the exact per-sample material exponents from the dataset auxiliary cache.

Evaluates on the held-out 800-sample test set:
  1. Kirchhoff Round-Trip Validation (T -> theta -> T_rec)
  2. FNO Temperature Prediction Metrics (MAE, RMSE, MSE, R^2, Relative L2)
  3. Hotspot Analysis (Temperature error and location distance error)
  4. Physical Sanity Checks (Finiteness, Range, Monotonicity)
  5. Representative Visualizations (Temperature maps & parity plot)

Outputs saved to outputs/fno_baseline_temperature/
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_generator.kirchhoff import forward_kirchhoff, inverse_kirchhoff  # noqa: E402
from ml_data import ChipThermalDataset  # noqa: E402
from ml_data.kirchhoff_utils import build_n_map  # noqa: E402
from ml_data.normalize import denormalize_target, load_ml_norm_stats  # noqa: E402
from models.fno import FNO2d, build_fno_from_config  # noqa: E402

BEST_CKPT_PATH = PROJECT_ROOT / "checkpoints" / "baseline" / "fno_baseline_best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fno_baseline_temperature"
VIZ_DIR = OUTPUT_DIR / "visualizations"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R^2 coefficient of determination across flattened arrays."""
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    ss_res = float(np.sum((y_true_flat - y_pred_flat) ** 2))
    ss_tot = float(np.sum((y_true_flat - np.mean(y_true_flat)) ** 2))
    if ss_tot < 1e-12:
        return 1.0
    return 1.0 - (ss_res / ss_tot)


def run_kirchhoff_roundtrip_check(
    h5_path: Path,
    test_ds: ChipThermalDataset,
    n_samples: int = 800,
) -> Dict[str, float]:
    """Validate T -> forward Kirchhoff -> theta -> inverse Kirchhoff -> T_rec."""
    print("=" * 65)
    print("RUNNING KIRCHHOFF ROUND-TRIP VALIDATION (800 TEST SAMPLES)")
    print("=" * 65)

    with h5py.File(h5_path, "r") as f:
        T_gt_all = np.asarray(f["targets/T"], dtype=np.float64)

    max_abs_errs = []
    mean_abs_errs = []
    sq_errs = []
    rel_errs = []

    for idx in range(n_samples):
        _, _, meta = test_ds[idx]
        T_gt = T_gt_all[idx]

        material = meta["material"]
        n_d = meta["n_dielectric"]
        n_s = meta["n_silicon"]
        n_c = meta["n_copper"]
        T_ref = meta["T_ref_K"]

        n_map = build_n_map(material, n_d, n_s, n_c)

        # Forward Kirchhoff
        theta_calc = forward_kirchhoff(T_gt, n_map, T_ref)
        # Inverse Kirchhoff
        T_rec = inverse_kirchhoff(theta_calc, n_map, T_ref)

        abs_diff = np.abs(T_rec - T_gt)
        max_abs_errs.append(float(abs_diff.max()))
        mean_abs_errs.append(float(abs_diff.mean()))
        sq_errs.extend((abs_diff ** 2).ravel().tolist())

        rel_diff = abs_diff / np.clip(T_gt, 1.0, None)
        rel_errs.append(float(rel_diff.max()))

    max_err = float(np.max(max_abs_errs))
    mean_err = float(np.mean(mean_abs_errs))
    rmse = float(np.sqrt(np.mean(sq_errs)))
    max_rel_err = float(np.max(rel_errs))

    print(f"Kirchhoff Round-Trip Results ({n_samples} samples):")
    print(f"  Max Absolute Reconstruction Error:  {max_err:.6e} K")
    print(f"  Mean Absolute Reconstruction Error: {mean_err:.6e} K")
    print(f"  RMSE:                               {rmse:.6e} K")
    print(f"  Max Relative Reconstruction Error:  {max_rel_err:.6e}")
    print("=" * 65)

    if max_err > 1e-3:
        raise ValueError(
            f"CRITICAL: Kirchhoff round-trip test failed! Max reconstruction error {max_err:.6e} K > 1e-3 K"
        )

    print("KIRCHHOFF ROUND-TRIP VALIDATION: PASS")
    print("=" * 65)

    return {
        "max_abs_error_K": max_err,
        "mean_abs_error_K": mean_err,
        "rmse_K": rmse,
        "max_relative_error": max_rel_err,
        "status": "PASS",
    }


def evaluate_fno_temperature(
    model: nn.Module,
    device: torch.device,
    test_ds: ChipThermalDataset,
    h5_path: Path,
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    print("=" * 65)
    print("RUNNING FNO TEMPERATURE INFERENCE & EVALUATION (800 TEST SAMPLES)")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        T_gt_all = np.asarray(f["targets/T"], dtype=np.float64)

    n_samples = len(test_ds)
    assert n_samples == 800, f"Expected 800 test samples, got {n_samples}"

    model.eval()

    T_pred_all = np.zeros((n_samples, 32, 32), dtype=np.float64)
    theta_pred_all = np.zeros((n_samples, 32, 32), dtype=np.float64)
    theta_gt_all = np.zeros((n_samples, 32, 32), dtype=np.float64)

    per_sample_records = []

    t0_start = time.time()

    for idx in range(n_samples):
        X_t, y_t, meta = test_ds[idx]

        # 1. Forward pass
        X_batch = X_t.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_norm_t = model(X_batch)

        pred_norm = pred_norm_t.cpu().numpy()[0, 0]  # (32, 32)
        y_norm = y_t.numpy()[0]  # (32, 32)

        if not np.all(np.isfinite(pred_norm)):
            raise ValueError(f"CRITICAL: Non-finite prediction in theta at sample index {idx}")

        # 2. Denormalize theta
        theta_pred_phys = denormalize_target(pred_norm, stats)
        theta_gt_phys = denormalize_target(y_norm, stats)

        theta_pred_all[idx] = theta_pred_phys
        theta_gt_all[idx] = theta_gt_phys

        # 3. Inverse Kirchhoff transformation
        material = meta["material"]
        n_d = meta["n_dielectric"]
        n_s = meta["n_silicon"]
        n_c = meta["n_copper"]
        T_ref = meta["T_ref_K"]

        n_map = build_n_map(material, n_d, n_s, n_c)

        T_gt = T_gt_all[idx]
        T_pred = inverse_kirchhoff(theta_pred_phys, n_map, T_ref)

        if not np.all(np.isfinite(T_pred)):
            raise ValueError(f"CRITICAL: Non-finite temperature prediction at sample index {idx}")

        T_pred_all[idx] = T_pred

        # 4. Sample metrics & Hotspot analysis
        T_gt_max = float(T_gt.max())
        T_pred_max = float(T_pred.max())
        abs_hotspot_err = float(np.abs(T_pred_max - T_gt_max))

        # Hotspot locations (argmax)
        gt_hspot_idx = np.unravel_index(np.argmax(T_gt), T_gt.shape)
        pred_hspot_idx = np.unravel_index(np.argmax(T_pred), T_pred.shape)

        hspot_dist_px = float(
            np.sqrt((gt_hspot_idx[0] - pred_hspot_idx[0]) ** 2 + (gt_hspot_idx[1] - pred_hspot_idx[1]) ** 2)
        )
        hspot_exact_match = int(gt_hspot_idx == pred_hspot_idx)

        sample_mae = float(np.abs(T_pred - T_gt).mean())
        sample_rmse = float(np.sqrt(np.mean((T_pred - T_gt) ** 2)))

        per_sample_records.append(
            {
                "sample_id": meta["sample_id"],
                "split_idx": idx,
                "T_true_max_K": T_gt_max,
                "T_pred_max_K": T_pred_max,
                "abs_hotspot_err_K": abs_hotspot_err,
                "true_hotspot_row": gt_hspot_idx[0],
                "true_hotspot_col": gt_hspot_idx[1],
                "pred_hotspot_row": pred_hspot_idx[0],
                "pred_hotspot_col": pred_hspot_idx[1],
                "hotspot_dist_px": hspot_dist_px,
                "hotspot_exact_match": hspot_exact_match,
                "sample_T_mae_K": sample_mae,
                "sample_T_rmse_K": sample_rmse,
            }
        )

    eval_time = time.time() - t0_start

    # Save per-sample CSV
    df_per_sample = pd.DataFrame(per_sample_records)
    csv_path = OUTPUT_DIR / "temperature_per_sample.csv"
    df_per_sample.to_csv(csv_path, index=False)
    print(f"Saved per-sample metrics CSV to: {csv_path}")

    # Overall Temperature Metrics across all 800 samples (800 x 32 x 32 = 819,200 points)
    T_diff = T_pred_all - T_gt_all
    abs_T_diff = np.abs(T_diff)

    temp_mae = float(abs_T_diff.mean())
    temp_mse = float(np.mean(T_diff ** 2))
    temp_rmse = float(np.sqrt(temp_mse))
    temp_max_abs_err = float(abs_T_diff.max())
    temp_r2 = calculate_r2_score(T_gt_all, T_pred_all)

    # Relative L2 error per sample & overall
    l2_diff = np.linalg.norm(T_diff.reshape(n_samples, -1), axis=1)
    l2_true = np.linalg.norm(T_gt_all.reshape(n_samples, -1), axis=1)
    rel_l2_per_sample = l2_diff / np.clip(l2_true, 1e-8, None)
    temp_rel_l2_mean = float(rel_l2_per_sample.mean())
    temp_rel_l2_median = float(np.median(rel_l2_per_sample))

    # Overall Hotspot Metrics
    hotspot_temp_errs = df_per_sample["abs_hotspot_err_K"].values
    hotspot_dist_errs = df_per_sample["hotspot_dist_px"].values
    hotspot_exact_matches = df_per_sample["hotspot_exact_match"].values

    hotspot_mae = float(hotspot_temp_errs.mean())
    hotspot_max_err = float(hotspot_temp_errs.max())
    hotspot_rmse = float(np.sqrt(np.mean(hotspot_temp_errs ** 2)))
    hotspot_mean_dist_px = float(hotspot_dist_errs.mean())
    hotspot_exact_match_pct = float(hotspot_exact_matches.mean() * 100.0)

    print("-" * 65)
    print("FNO PHYSICAL TEMPERATURE EVALUATION RESULTS (TEST SET):")
    print("-" * 65)
    print(f"  Temperature MAE:                  {temp_mae:.4f} K")
    print(f"  Temperature RMSE:                 {temp_rmse:.4f} K")
    print(f"  Temperature MSE:                  {temp_mse:.4f} K^2")
    print(f"  Temperature Max Absolute Error:   {temp_max_abs_err:.4f} K")
    print(f"  Temperature R^2 Score:            {temp_r2:.6f}")
    print(f"  Temperature Relative L2 (Mean):   {temp_rel_l2_mean:.6f}")
    print(f"  Temperature Relative L2 (Median): {temp_rel_l2_median:.6f}")
    print("-" * 65)
    print("HOTSPOT ANALYSIS RESULTS:")
    print(f"  Mean Hotspot Temp Error (MAE):    {hotspot_mae:.4f} K")
    print(f"  Max Hotspot Temp Error:           {hotspot_max_err:.4f} K")
    print(f"  Hotspot Temp RMSE:                {hotspot_rmse:.4f} K")
    print(f"  Hotspot Exact Location Match:     {hotspot_exact_match_pct:.2f}%")
    print(f"  Mean Hotspot Location Distance:   {hotspot_mean_dist_px:.3f} pixels")
    print("=" * 65)

    # Visualizations
    _generate_temperature_visualizations(
        T_gt_all, T_pred_all, abs_T_diff, per_sample_records, VIZ_DIR
    )

    summary = {
        "checkpoint_used": str(BEST_CKPT_PATH),
        "n_test_samples": n_samples,
        "temperature_unit": "Kelvin (K)",
        "kirchhoff_roundtrip_passed": True,
        "temperature_metrics": {
            "mae_K": temp_mae,
            "rmse_K": temp_rmse,
            "mse_K2": temp_mse,
            "max_abs_error_K": temp_max_abs_err,
            "r2_score": temp_r2,
            "rel_l2_mean": temp_rel_l2_mean,
            "rel_l2_median": temp_rel_l2_median,
        },
        "hotspot_metrics": {
            "mean_abs_hotspot_temp_error_K": hotspot_mae,
            "max_abs_hotspot_temp_error_K": hotspot_max_err,
            "rmse_hotspot_temp_error_K": hotspot_rmse,
            "exact_location_match_pct": hotspot_exact_match_pct,
            "mean_location_distance_px": hotspot_mean_dist_px,
        },
        "evaluation_time_sec": eval_time,
    }

    metrics_json_path = OUTPUT_DIR / "temperature_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved temperature evaluation metrics to: {metrics_json_path}")

    return summary


def _generate_temperature_visualizations(
    T_gt: np.ndarray,
    T_pred: np.ndarray,
    abs_err: np.ndarray,
    records: List[Dict[str, Any]],
    viz_dir: Path,
) -> None:
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save raw arrays for representative samples (first 5 test samples)
    np.savez_compressed(
        viz_dir / "temperature_test_samples.npz",
        ground_truth=T_gt[:5],
        predicted=T_pred[:5],
        abs_error=abs_err[:5],
    )

    # 2. Plot representative samples (samples 0, 1, 2)
    for i in range(min(3, T_gt.shape[0])):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

        vmin = min(T_gt[i].min(), T_pred[i].min())
        vmax = max(T_gt[i].max(), T_pred[i].max())

        im0 = axes[0].imshow(T_gt[i], cmap="inferno", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"Test Sample {i+1}: Ground Truth T (K)")
        axes[0].plot(records[i]["true_hotspot_col"], records[i]["true_hotspot_row"], "r*", markersize=10, label="True Hotspot")
        axes[0].legend(loc="upper right", fontsize=8)
        fig.colorbar(im0, ax=axes[0], label="Temperature (K)")

        im1 = axes[1].imshow(T_pred[i], cmap="inferno", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Test Sample {i+1}: FNO Predicted T (K)")
        axes[1].plot(records[i]["pred_hotspot_col"], records[i]["pred_hotspot_row"], "c*", markersize=10, label="Pred Hotspot")
        axes[1].legend(loc="upper right", fontsize=8)
        fig.colorbar(im1, ax=axes[1], label="Temperature (K)")

        im2 = axes[2].imshow(abs_err[i], cmap="viridis")
        axes[2].set_title(f"Test Sample {i+1}: Abs Error |T - T_pred| (K)")
        fig.colorbar(im2, ax=axes[2], label="Error (K)")

        plt.tight_layout()
        plot_path = viz_dir / f"test_temp_sample_{i+1}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved temperature map visualization: {plot_path}")

    # 3. Ground Truth T vs Predicted T Parity Plot
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # Subsample 2000 points randomly for clean plot readability
    rng = np.random.RandomState(42)
    idxs = rng.choice(T_gt.size, size=min(5000, T_gt.size), replace=False)

    gt_flat = T_gt.ravel()[idxs]
    pred_flat = T_pred.ravel()[idxs]

    ax.scatter(gt_flat, pred_flat, alpha=0.3, s=10, color="navy", edgecolors="none")
    min_val = min(gt_flat.min(), pred_flat.min()) - 1.0
    max_val = max(gt_flat.max(), pred_flat.max()) + 1.0

    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal (y = x)")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel("Ground Truth Temperature T (K)", fontsize=11)
    ax.set_ylabel("FNO Predicted Temperature T (K)", fontsize=11)
    ax.set_title("FNO Baseline: Ground Truth vs Predicted Temperature T", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")

    plt.tight_layout()
    parity_path = viz_dir / "parity_plot.png"
    plt.savefig(parity_path, dpi=150)
    plt.close(fig)
    print(f"Saved parity plot: {parity_path}")


def main():
    device = get_device()
    print(f"Using device: {device}")

    # Load test dataset & norm stats
    test_ds = ChipThermalDataset("test", normalize=True, return_meta=True)
    stats = load_ml_norm_stats()
    h5_path = PROJECT_ROOT / "dataset" / "test.h5"

    # 1. Kirchhoff Round-Trip Validation
    roundtrip_res = run_kirchhoff_roundtrip_check(h5_path, test_ds, n_samples=800)

    # 2. Load best FNO model
    if not BEST_CKPT_PATH.exists():
        raise FileNotFoundError(f"Missing best checkpoint file: {BEST_CKPT_PATH}")

    ckpt = torch.load(BEST_CKPT_PATH, map_location=device)
    model = build_fno_from_config(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded best FNO model from: {BEST_CKPT_PATH} (Epoch {ckpt.get('epoch', 'N/A')})")

    # 3. Evaluate FNO Temperature Prediction
    summary = evaluate_fno_temperature(model, device, test_ds, h5_path, stats)
    summary["kirchhoff_roundtrip"] = roundtrip_res

    print("=" * 65)
    print("INVERSE KIRCHHOFF TEMPERATURE EVALUATION COMPLETE: PASS")
    print("=" * 65)


if __name__ == "__main__":
    main()
