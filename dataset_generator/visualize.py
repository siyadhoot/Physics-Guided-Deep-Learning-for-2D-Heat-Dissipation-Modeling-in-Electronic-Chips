"""
visualize.py
=============
Step: generate multi-panel inspection PNGs for representative samples,
drawn from train/val/test, spanning normal/challenging/very_challenging
difficulty and multiple power-pattern types, saved to
dataset/visualizations/.
"""

import os
import json

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

MATERIAL_CMAP = matplotlib.colors.ListedColormap(["#2b6cb0", "#7fb069", "#d9822b"])  # dielec, si, cu


def load_h5(path):
    return h5py.File(path, "r")


def material_boundary_mask(mat):
    b = np.zeros_like(mat, dtype=bool)
    b[:-1, :] |= mat[:-1, :] != mat[1:, :]
    b[1:, :] |= mat[:-1, :] != mat[1:, :]
    b[:, :-1] |= mat[:, :-1] != mat[:, 1:]
    b[:, 1:] |= mat[:, :-1] != mat[:, 1:]
    return b


def plot_sample(h5, idx, split, out_path):
    Q = h5["inputs/Q"][idx]
    mat = h5["inputs/material"][idx]
    kx = h5["inputs/kx"][idx]
    ky = h5["inputs/ky"][idx]
    T = h5["targets/T"][idx]
    theta = h5["targets/theta"][idx]

    sample_id = int(h5["metadata/sample_id"][idx])
    diff_level = h5["metadata/difficulty_level"][idx]
    pattern = h5["metadata/hotspot_type"][idx]
    Tmax = float(h5["metadata/T_max"][idx])
    dT = float(h5["metadata/delta_T"][idx])
    boundary_type = h5["metadata/boundary_type"][idx]

    fig, axes = plt.subplots(2, 4, figsize=(21, 10))

    im = axes[0, 0].imshow(Q.T, origin="lower", cmap="hot")
    axes[0, 0].set_title("Power Map Q(x,y) [W/m^2]")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)

    im = axes[0, 1].imshow(mat.T, origin="lower", cmap=MATERIAL_CMAP, vmin=-0.5, vmax=2.5)
    axes[0, 1].set_title("Material Map (0=Dielectric,1=Si,2=Cu)")
    cbar = plt.colorbar(im, ax=axes[0, 1], fraction=0.046, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Dielectric", "Silicon", "Copper"])

    im = axes[0, 2].imshow(kx.T, origin="lower", cmap="plasma")
    axes[0, 2].set_title("kx(x,y) [W/m.K]")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    im = axes[0, 3].imshow(ky.T, origin="lower", cmap="plasma")
    axes[0, 3].set_title("ky(x,y) [W/m.K]")
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046)

    im = axes[1, 0].imshow(T.T, origin="lower", cmap="inferno")
    axes[1, 0].set_title(f"Ground Truth T(x,y) [K]  (Tmax={Tmax:.1f}K, dT={dT:.1f}K)")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    im = axes[1, 1].imshow(theta.T, origin="lower", cmap="cividis")
    axes[1, 1].set_title("Kirchhoff theta(x,y)")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    # overlay: material boundaries on temperature contours
    ax = axes[1, 2]
    cf = ax.contourf(T.T, levels=20, cmap="inferno", origin="lower")
    bmask = material_boundary_mask(mat)
    ys, xs = np.where(bmask.T)
    ax.scatter(xs, ys, s=1.5, c="cyan", alpha=0.6, label="material boundary")
    ax.set_title("T contours + material boundary overlay")
    ax.legend(loc="upper right", fontsize=7)
    plt.colorbar(cf, ax=ax, fraction=0.046)

    ax = axes[1, 3]
    ax.axis("off")
    info_text = (
        f"sample_id: {sample_id}\nsplit: {split}\ndifficulty: {diff_level}\n"
        f"power pattern: {pattern}\nboundary scenario: {boundary_type}\n"
        f"T_max: {Tmax:.2f} K\ndelta_T: {dT:.2f} K\n"
        f"pde_residual_max: {float(h5['metadata/pde_residual_max'][idx]):.2e}\n"
        f"heat_balance_err: {float(h5['metadata/heat_balance_error_pct'][idx]):.4f} %\n"
        f"kirchhoff_rt_err: {float(h5['metadata/kirchhoff_roundtrip_error_K'][idx]):.2e} K\n"
        f"solver_iters: {int(h5['metadata/solver_iterations'][idx])}\n"
    )
    ax.text(0.02, 0.98, info_text, va="top", fontsize=11, family="monospace")

    fig.suptitle(f"Sample {sample_id} [{split}]  —  {diff_level} / pattern {pattern}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    files = {
        "train": os.path.join(C.DATASET_DIR, "train.h5"),
        "val": os.path.join(C.DATASET_DIR, "validation.h5"),
        "test": os.path.join(C.DATASET_DIR, "test.h5"),
    }
    handles = {s: load_h5(p) for s, p in files.items()}

    n_total = C.N_VISUALIZATION_SAMPLES
    # distribute across splits, weighted toward train, but include val/test for inspection
    n_train_v = int(n_total * 0.6)
    n_val_v = int(n_total * 0.2)
    n_test_v = n_total - n_train_v - n_val_v

    plan = []
    rng = np.random.default_rng(12345)
    for split, n_v, h5 in [("train", n_train_v, handles["train"]),
                            ("val", n_val_v, handles["val"]),
                            ("test", n_test_v, handles["test"])]:
        n = h5["inputs/Q"].shape[0]
        diffs = h5["metadata/difficulty_level"][:]
        # try to get a spread across difficulty levels
        idxs = []
        for level in [b"normal", b"challenging", b"very_challenging"]:
            level_str = level.decode() if isinstance(level, bytes) else level
            mask = np.array([d == level_str for d in diffs])
            cand = np.where(mask)[0]
            if len(cand) == 0:
                continue
            take = max(1, n_v // 3)
            idxs.extend(rng.choice(cand, size=min(take, len(cand)), replace=False).tolist())
        idxs = idxs[:n_v] if len(idxs) >= n_v else idxs + rng.choice(n, size=n_v - len(idxs), replace=False).tolist()
        plan.append((split, idxs))

    count = 0
    for split, idxs in plan:
        h5 = handles[split]
        for i, idx in enumerate(idxs):
            out_path = os.path.join(C.VIS_DIR, f"sample_{count:04d}_{split}.png")
            plot_sample(h5, int(idx), split, out_path)
            count += 1

    for h5 in handles.values():
        h5.close()

    print(f"Saved {count} visualization PNGs to {C.VIS_DIR}")


if __name__ == "__main__":
    main()
