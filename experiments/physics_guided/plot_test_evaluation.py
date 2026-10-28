"""
Publication-Quality Visualization Suite for Final FNO vs. Corrected PINN Test Evaluation.

Project: Physics-Guided Deep Learning for 2D Heat Dissipation Modeling in Electronic Chips.

Generates Five Focused Figures for Research Report, Thesis, and Defense:
  1. temperature_field_comparison.png — Ground Truth vs. FNO vs. Corrected PINN
  2. absolute_error_comparison.png    — Absolute Temperature Error Heatmaps
  3. boundary_interior_mae.png        — Grouped Regional MAE (Boundary vs. Interior)
  4. boundary_physics_residual.png    — Boundary Energy Conservation Imbalance (W/m^3)
  5. hotspot_comparison.png           — Hotspot Localization Distance and Detection Rates

Usage:
  python experiments/physics_guided/plot_test_evaluation.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")  # Headless publication rendering
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Setup project root robustly
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import ChipThermalDataset
from src.data.build_aux_and_stats import load_aux_cache
from src.data.kirchhoff_utils import inverse_kirchhoff_from_sample
from src.data.normalize import load_ml_norm_stats
from experiments.physics_guided.evaluate_test_set import (
    get_boundary_mask,
    load_baseline_fno,
    load_pinn_model,
)

# Colors matching professional IEEE/Nature journal palettes
COLOR_BASELINE = "#1f77b4"  # Muted Blue
COLOR_PINN = "#d95f02"      # Physical Vermilion / Orange
COLOR_GT = "#2ca02c"        # Forest Green
COLOR_ACCENT = "#7570b3"    # Purple accent


def load_aggregate_metrics(metrics_path: Path) -> Dict[str, Any]:
    """Load verified aggregate test evaluation metrics from JSON."""
    if not metrics_path.exists():
        raise FileNotFoundError(f"Evaluation metrics JSON not found at: {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def reconstruct_sample_temperature(
    sample_idx: int,
    base_model: torch.nn.Module,
    pinn_model: torch.nn.Module,
    test_ds: ChipThermalDataset,
    aux: Dict[str, np.ndarray],
    target_mean: float,
    target_std: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Reconstruct physical Kelvin temperature maps for a specific test sample
    using the verified analytical inverse Kirchhoff pipeline.
    """
    X, _, meta = test_ds[sample_idx]
    X_batch = X.unsqueeze(0).to(device)

    with h5py.File(test_ds.h5_path, "r") as f:
        T_gt = np.asarray(f["targets/T"][sample_idx], dtype=np.float64)

    with torch.inference_mode():
        b_norm = base_model(X_batch)
        p_norm = pinn_model.forward(X_batch)

    b_th = (b_norm[0, 0] * target_std + target_mean).cpu().numpy().astype(np.float64)
    p_th = (p_norm[0, 0] * target_std + target_mean).cpu().numpy().astype(np.float64)

    mat = meta["material"]
    nd = float(aux["n_dielectric"][sample_idx])
    ns = float(aux["n_silicon"][sample_idx])
    nc = float(aux["n_copper"][sample_idx])
    tref = float(meta["T_ref_K"])

    T_fno = inverse_kirchhoff_from_sample(b_th, mat, nd, ns, nc, T_ref=tref)
    T_pinn = inverse_kirchhoff_from_sample(p_th, mat, nd, ns, nc, T_ref=tref)

    sample_meta = {
        "sample_idx": sample_idx,
        "sample_id": int(meta["sample_id"]),
        "boundary_type": str(meta.get("boundary_type", "mixed")),
        "T_ambient": float(meta["T_ambient"]),
        "h_sink": float(meta["h_sink"]),
    }
    return T_gt, T_fno, T_pinn, sample_meta


# ==============================================================================
# FIGURE 1: TEMPERATURE FIELD COMPARISON
# ==============================================================================

def plot_figure_1_temperature_fields(
    T_gt: np.ndarray,
    T_fno: np.ndarray,
    T_pinn: np.ndarray,
    meta: Dict[str, Any],
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Figure 1: Three-panel physical temperature field comparison on held-out test sample."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=dpi)

    vmin = min(float(T_gt.min()), float(T_fno.min()), float(T_pinn.min()))
    vmax = max(float(T_gt.max()), float(T_fno.max()), float(T_pinn.max()))

    # Find peak hotspots
    gt_r, gt_c = np.unravel_index(np.argmax(T_gt), T_gt.shape)
    fno_r, fno_c = np.unravel_index(np.argmax(T_fno), T_fno.shape)
    pinn_r, pinn_c = np.unravel_index(np.argmax(T_pinn), T_pinn.shape)

    # Panel A: Ground Truth
    im0 = axes[0].imshow(T_gt, cmap="inferno", vmin=vmin, vmax=vmax, origin="upper")
    axes[0].plot(gt_c, gt_r, "r*", markersize=13, markeredgecolor="white", markeredgewidth=1.2, label=f"Max: {T_gt[gt_r, gt_c]:.1f} K")
    axes[0].set_title("(a) Ground-Truth Temperature ($T_{\\mathrm{true}}$)", fontsize=12, fontweight="bold", pad=8)
    axes[0].set_xlabel("Grid X (Cells)", fontsize=10)
    axes[0].set_ylabel("Grid Y (Cells)", fontsize=10)
    axes[0].legend(loc="upper right", framealpha=0.85, fontsize=9)

    # Panel B: Baseline FNO
    fno_mae = np.mean(np.abs(T_fno - T_gt))
    im1 = axes[1].imshow(T_fno, cmap="inferno", vmin=vmin, vmax=vmax, origin="upper")
    axes[1].plot(fno_c, fno_r, "c*", markersize=13, markeredgecolor="black", markeredgewidth=1.0, label=f"Max: {T_fno[fno_r, fno_c]:.1f} K")
    axes[1].set_title(f"(b) Baseline FNO Prediction (MAE: {fno_mae:.2f} K)", fontsize=12, fontweight="bold", pad=8)
    axes[1].set_xlabel("Grid X (Cells)", fontsize=10)
    axes[1].set_ylabel("Grid Y (Cells)", fontsize=10)
    axes[1].legend(loc="upper right", framealpha=0.85, fontsize=9)

    # Panel C: Corrected PINN
    pinn_mae = np.mean(np.abs(T_pinn - T_gt))
    im2 = axes[2].imshow(T_pinn, cmap="inferno", vmin=vmin, vmax=vmax, origin="upper")
    axes[2].plot(pinn_c, pinn_r, "m*", markersize=13, markeredgecolor="white", markeredgewidth=1.0, label=f"Max: {T_pinn[pinn_r, pinn_c]:.1f} K")
    axes[2].set_title(f"(c) Corrected PINN Prediction (MAE: {pinn_mae:.2f} K)", fontsize=12, fontweight="bold", pad=8)
    axes[2].set_xlabel("Grid X (Cells)", fontsize=10)
    axes[2].set_ylabel("Grid Y (Cells)", fontsize=10)
    axes[2].legend(loc="upper right", framealpha=0.85, fontsize=9)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.18, 0.016, 0.68])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label("Temperature (K)", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(
        f"Figure 1: Full-Domain 2D Physical Temperature Field Comparison (Test Sample Index: {meta['sample_idx']}, Sample ID: {meta['sample_id']})\n"
        f"Domain: 10 mm × 10 mm (32 × 32 cells, dx = 0.3125 mm) | Ambient: {meta['T_ambient']:.1f} K",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.subplots_adjust(left=0.05, right=0.90, bottom=0.12, top=0.88, wspace=0.25)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Generated Figure 1: {out_path.name}")


# ==============================================================================
# FIGURE 2: ABSOLUTE TEMPERATURE ERROR HEATMAPS
# ==============================================================================

def plot_figure_2_absolute_error_heatmaps(
    T_gt: np.ndarray,
    T_fno: np.ndarray,
    T_pinn: np.ndarray,
    meta: Dict[str, Any],
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Figure 2: Side-by-side absolute temperature error heatmaps."""
    err_fno = np.abs(T_fno - T_gt)
    err_pinn = np.abs(T_pinn - T_gt)

    vmin = 0.0
    vmax = max(float(err_fno.max()), float(err_pinn.max()))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=dpi)

    # Baseline Error
    im0 = axes[0].imshow(err_fno, cmap="magma", vmin=vmin, vmax=vmax, origin="upper")
    axes[0].set_title(
        f"(a) Baseline FNO Absolute Error\nMean: {np.mean(err_fno):.2f} K | Max: {np.max(err_fno):.2f} K",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    axes[0].set_xlabel("Grid X (Cells)", fontsize=10)
    axes[0].set_ylabel("Grid Y (Cells)", fontsize=10)

    # Highlight boundary perimeter (124 cells)
    rect0 = patches.Rectangle((-0.5, -0.5), 32, 32, linewidth=1.5, edgecolor="cyan", facecolor="none", linestyle="--")
    axes[0].add_patch(rect0)

    # PINN Error
    im1 = axes[1].imshow(err_pinn, cmap="magma", vmin=vmin, vmax=vmax, origin="upper")
    axes[1].set_title(
        f"(b) Corrected PINN Absolute Error\nMean: {np.mean(err_pinn):.2f} K | Max: {np.max(err_pinn):.2f} K",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    axes[1].set_xlabel("Grid X (Cells)", fontsize=10)
    axes[1].set_ylabel("Grid Y (Cells)", fontsize=10)

    rect1 = patches.Rectangle((-0.5, -0.5), 32, 32, linewidth=1.5, edgecolor="cyan", facecolor="none", linestyle="--")
    axes[1].add_patch(rect1)

    cbar_ax = fig.add_axes([0.92, 0.16, 0.02, 0.70])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label("Absolute Temperature Error |$T_{\\mathrm{pred}} - T_{\\mathrm{true}}$| (K)", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(
        f"Figure 2: Spatial Error Distribution |$T - T_{{\\mathrm{{true}}}}$| on Held-Out Test Sample {meta['sample_idx']} (Sample ID: {meta['sample_id']})\n"
        f"Perimeter cyan dashed box highlights the 124 boundary cells",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.subplots_adjust(left=0.06, right=0.90, bottom=0.12, top=0.86, wspace=0.25)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Generated Figure 2: {out_path.name}")


# ==============================================================================
# FIGURE 3: BOUNDARY VS INTERIOR MAE
# ==============================================================================

def plot_figure_3_boundary_interior_mae(
    metrics: Dict[str, Any],
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Figure 3: Grouped bar chart comparing Boundary vs. Interior MAE across all 800 test samples."""
    base_m = metrics["baseline_model"]["metrics"]
    pinn_m = metrics["pinn_model"]["metrics"]

    categories = ["Boundary Region\n(124 Perimeter Cells)", "Interior Region\n(900 Core Cells)", "Full Domain\n(1024 Total Cells)"]
    fno_vals = [base_m["boundary_mae_k"], base_m["interior_mae_k"], base_m["mae_k"]]
    pinn_vals = [pinn_m["boundary_mae_k"], pinn_m["interior_mae_k"], pinn_m["mae_k"]]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=dpi)

    rects1 = ax.bar(x - width / 2, fno_vals, width, label="Baseline FNO", color=COLOR_BASELINE, edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width / 2, pinn_vals, width, label="Corrected PINN", color=COLOR_PINN, edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Mean Absolute Error (K)", fontsize=12, fontweight="bold")
    ax.set_title("Figure 3: Regional Error Decomposition: Boundary vs. Interior MAE (800 Held-Out Test Samples)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(framealpha=0.9, fontsize=11, loc="upper right")
    ax.set_ylim(0, max(max(fno_vals), max(pinn_vals)) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Attach numeric labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.4f} K",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    autolabel(rects1)
    autolabel(rects2)

    # Explanatory note
    ax.text(
        0.5, -0.16,
        "Boundary cells = outer edge rows & columns (124 cells) where Dirichlet/Robin/Neumann conditions are enforced.\n"
        "Interior cells = internal 30 × 30 core (900 cells). Errors evaluated in Kelvin against converged FV ground truth.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        style="italic",
        color="#333333",
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Generated Figure 3: {out_path.name}")


# ==============================================================================
# FIGURE 4: BOUNDARY PHYSICS RESIDUAL COMPARISON
# ==============================================================================

def plot_figure_4_boundary_physics_residual(
    metrics: Dict[str, Any],
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Figure 4: Bar comparison of mean absolute boundary residual in W/m^3."""
    base_res = metrics["baseline_model"]["metrics"]["boundary_residual_abs_W_m3"]
    pinn_res = metrics["pinn_model"]["metrics"]["boundary_residual_abs_W_m3"]

    # Express in Gigawatts per cubic meter (10^9 W/m^3) for clean readability
    base_gw = base_res / 1.0e9
    pinn_gw = pinn_res / 1.0e9

    pct_reduction = ((base_res - pinn_res) / base_res) * 100.0

    fig, ax = plt.subplots(figsize=(7.8, 5.2), dpi=dpi)

    models = ["Baseline FNO\n(Unconstrained)", "Corrected PINN\n(Boundary FV Regularized)"]
    vals = [base_gw, pinn_gw]
    colors = [COLOR_BASELINE, COLOR_PINN]

    bars = ax.bar(models, vals, width=0.45, color=colors, edgecolor="black", linewidth=1.0)

    ax.set_ylabel("Mean Absolute Boundary Residual (10⁹ W/m³)", fontsize=12, fontweight="bold")
    ax.set_title("Figure 4: Boundary Energy Conservation Fidelity (800 Held-Out Test Samples)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, max(vals) * 1.35)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Value labels
    for bar, val_w in zip(bars, [base_res, pinn_res]):
        height = bar.get_height()
        ax.annotate(
            f"{val_w:.4e} W/m³\n({height:.4f} GW/m³)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Highlight improvement badge with arrow
    mid_x = (bars[0].get_x() + bars[1].get_x() + bars[1].get_width()) / 2
    top_y = max(vals) * 1.15
    ax.annotate(
        f"▼ {pct_reduction:.2f}% Physical Violation Reduction",
        xy=(bars[1].get_x() + bars[1].get_width() / 2, pinn_gw + 0.08),
        xytext=(mid_x + 0.15, top_y),
        arrowprops=dict(facecolor="#2ca02c", edgecolor="#1e7120", width=2.0, headwidth=8.0, shrink=0.08),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f5e9", edgecolor="#2ca02c", linewidth=1.5),
        fontsize=10.5,
        fontweight="bold",
        color="#1b5e20",
        ha="center",
    )

    ax.text(
        0.5, -0.16,
        "Evaluates the discrete net volumetric imbalance on all 124 perimeter boundary cells matching\n"
        "dataset_generator/solver.py assemble_system. Lower residual indicates superior adherence to physical conservation.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        style="italic",
        color="#333333",
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Generated Figure 4: {out_path.name}")


# ==============================================================================
# FIGURE 5: HOTSPOT LOCALIZATION COMPARISON
# ==============================================================================

def plot_figure_5_hotspot_comparison(
    metrics: Dict[str, Any],
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Figure 5: Dual-panel comparison of hotspot spatial localization distance and detection rates."""
    base_m = metrics["baseline_model"]["metrics"]
    pinn_m = metrics["pinn_model"]["metrics"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), dpi=dpi)

    # Panel A: Mean Spatial Localization Distance (Cells) - Lower is better
    models = ["Baseline FNO", "Corrected PINN"]
    dists = [base_m["hotspot_loc_dist_mean_cells"], pinn_m["hotspot_loc_dist_mean_cells"]]
    colors = [COLOR_BASELINE, COLOR_PINN]

    bars0 = axes[0].bar(models, dists, width=0.45, color=colors, edgecolor="black", linewidth=0.9)
    axes[0].set_ylabel("Mean Localization Distance (Grid Cells)", fontsize=11, fontweight="bold")
    axes[0].set_title("(a) Mean Hotspot Offset Distance (Lower is Better)", fontsize=11, fontweight="bold", pad=10)
    axes[0].set_ylim(0, max(dists) * 1.30)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    for bar, d_cell in zip(bars0, dists):
        d_mm = d_cell * 0.3125
        axes[0].annotate(
            f"{d_cell:.4f} cells\n({d_mm:.3f} mm)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Panel B: Exact & Within-1-Cell Detection Rates (%) - Higher is better
    categories = ["Exact Pixel Match\n(Distance = 0 Cells)", "Within 1-Cell Radius\n(Distance ≤ 1.0 Cells)"]
    fno_rates = [base_m["hotspot_exact_match_pct"], base_m["hotspot_tol_1_cell_pct"]]
    pinn_rates = [pinn_m["hotspot_exact_match_pct"], pinn_m["hotspot_tol_1_cell_pct"]]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = axes[1].bar(x - width / 2, fno_rates, width, label="Baseline FNO", color=COLOR_BASELINE, edgecolor="black", linewidth=0.9)
    bars2 = axes[1].bar(x + width / 2, pinn_rates, width, label="Corrected PINN", color=COLOR_PINN, edgecolor="black", linewidth=0.9)

    axes[1].set_ylabel("Hotspot Detection Rate (%)", fontsize=11, fontweight="bold")
    axes[1].set_title("(b) Spatial Hotspot Capture Rate (Higher is Better)", fontsize=11, fontweight="bold", pad=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories, fontsize=10.5)
    axes[1].set_ylim(0, 80)
    axes[1].legend(loc="upper left", framealpha=0.9, fontsize=10)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    for rect in bars1:
        axes[1].annotate(
            f"{rect.get_height():.2f}%",
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    for rect in bars2:
        axes[1].annotate(
            f"{rect.get_height():.2f}%",
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    fig.suptitle(
        "Figure 5: Semiconductor Hotspot Peak Localization Precision (800 Held-Out Test Samples)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Generated Figure 5: {out_path.name}")


# ==============================================================================
# MAIN DRIVER
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Publication-Quality Test Evaluation Visualizations for FNO vs. Corrected PINN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "final_test_evaluation" / "metrics.json"),
        help="Path to evaluation metrics.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "final_test_evaluation" / "visualizations"),
        help="Directory to save generated figures.",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Index of test sample to use for temperature and error heatmaps (default: 0).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI resolution for publication figures.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device for sample inference ('cpu' or 'cuda').",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 75)
    print("GENERATING PUBLICATION-QUALITY TEST EVALUATION VISUALIZATIONS")
    print("=" * 75)
    print(f"Reading metrics from: {metrics_path}")
    print(f"Saving figures to:   {out_dir}")
    print(f"Sample Index:        {args.sample_idx} (First held-out test sample)")
    print(f"Figure Resolution:   {args.dpi} DPI")
    print("=" * 75)

    # 1. Load aggregate evaluation metrics
    metrics = load_aggregate_metrics(metrics_path)

    # 2. Reconstruct temperature fields for representative sample (Sample 0)
    base_ckpt = PROJECT_ROOT / "experiments" / "baseline" / "checkpoints" / "baseline" / "fno_baseline_best.pt"
    pinn_ckpt = PROJECT_ROOT / "experiments" / "physics_guided" / "checkpoints" / "pinn_full_data_v1" / "pinn_full_data_best.pt"
    if not pinn_ckpt.exists():
        pinn_ckpt = PROJECT_ROOT / "experiments" / "physics_guided" / "checkpoints" / "pinn_best.pt"

    base_model, _ = load_baseline_fno(base_ckpt, device)
    pinn_model, _ = load_pinn_model(pinn_ckpt, device)

    test_ds = ChipThermalDataset("test", normalize=True, return_meta=True)
    aux = load_aux_cache("test")
    norm_stats = load_ml_norm_stats()
    target_mean = float(norm_stats["target"]["theta"]["mean"])
    target_std = float(norm_stats["target"]["theta"]["std"])

    T_gt, T_fno, T_pinn, meta = reconstruct_sample_temperature(
        sample_idx=args.sample_idx,
        base_model=base_model,
        pinn_model=pinn_model,
        test_ds=test_ds,
        aux=aux,
        target_mean=target_mean,
        target_std=target_std,
        device=device,
    )

    # 3. Generate all 5 figures
    plot_figure_1_temperature_fields(
        T_gt=T_gt,
        T_fno=T_fno,
        T_pinn=T_pinn,
        meta=meta,
        out_path=out_dir / "temperature_field_comparison.png",
        dpi=args.dpi,
    )

    plot_figure_2_absolute_error_heatmaps(
        T_gt=T_gt,
        T_fno=T_fno,
        T_pinn=T_pinn,
        meta=meta,
        out_path=out_dir / "absolute_error_comparison.png",
        dpi=args.dpi,
    )

    plot_figure_3_boundary_interior_mae(
        metrics=metrics,
        out_path=out_dir / "boundary_interior_mae.png",
        dpi=args.dpi,
    )

    plot_figure_4_boundary_physics_residual(
        metrics=metrics,
        out_path=out_dir / "boundary_physics_residual.png",
        dpi=args.dpi,
    )

    plot_figure_5_hotspot_comparison(
        metrics=metrics,
        out_path=out_dir / "hotspot_comparison.png",
        dpi=args.dpi,
    )

    print("=" * 75)
    print("ALL 5 VISUALIZATIONS GENERATED SUCCESSFULLY (300 DPI)")
    print(f"Directory: {out_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
