"""
Smoke test for the leakage-free ML data pipeline.

Run after ``python -m ml_data.build_aux_and_stats``:

    python -m ml_data.smoke_test
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from .channels import CHANNEL_NAMES, channel_index
from .dataset import ChipThermalDataset, make_dataloader
from .kirchhoff_utils import inverse_kirchhoff_from_sample
from .leakage import validate_no_target_leakage
from .normalize import denormalize_target, load_ml_norm_stats


def main() -> int:
    print("=" * 70)
    print("SMOKE TEST: leakage-free chip thermal DataLoader")
    print("=" * 70)

    # 1–4: load one training sample
    ds = ChipThermalDataset(split="train", normalize=True, return_meta=True)
    X, y, meta = ds[0]
    print("\n[1] Loaded training sample 0")
    print("[2] Input channel names:")
    for i, name in enumerate(CHANNEL_NAMES):
        print(f"    {i:2d}: {name}")
    print(f"[3] X shape: {tuple(X.shape)}")
    print(f"[4] y shape: {tuple(y.shape)}")

    assert X.shape == (len(CHANNEL_NAMES), 32, 32), X.shape
    assert y.shape == (1, 32, 32), y.shape

    # 5–6: small batch, no NaN/Inf
    loader = make_dataloader("train", batch_size=8, shuffle=False, num_workers=0)
    Xb, yb = next(iter(loader))
    print(f"\n[5] Batch shapes: X={tuple(Xb.shape)} y={tuple(yb.shape)}")
    assert Xb.shape == (8, len(CHANNEL_NAMES), 32, 32)
    assert yb.shape == (8, 1, 32, 32)
    assert torch.isfinite(Xb).all() and torch.isfinite(yb).all()
    print("[6] No NaN/Inf in batch: OK")

    # 7: kx_base / ky_base
    ds_raw = ChipThermalDataset(split="train", normalize=False, return_meta=True)
    X0, y0, m0 = ds_raw[0]
    kx_b = X0[channel_index("kx_base")].numpy()
    ky_b = X0[channel_index("ky_base")].numpy()
    assert np.allclose(kx_b, m0["kx_base"], atol=1e-5)
    assert np.allclose(ky_b, m0["ky_base"], atol=1e-5)
    print(f"[7] kx_base/ky_base OK  "
          f"(kx_base range [{kx_b.min():.3g}, {kx_b.max():.3g}])")

    # 8: boundary channels
    for edge in ("left", "right", "bottom", "top"):
        t = X0[channel_index(f"bc_type_{edge}")].numpy()
        h = X0[channel_index(f"bc_h_{edge}")].numpy()
        Tv = X0[channel_index(f"bc_T_{edge}")].numpy()
        # interior should be zero for type/h/T edge maps
        interior = t[1:-1, 1:-1]
        assert np.allclose(interior, 0.0), edge
        print(f"[8] bc_{edge}: type unique={np.unique(t)}  "
              f"h max={h.max():.3g}  T max={Tv.max():.3g}")
    print(f"    boundary_type={m0['boundary_type']}")

    # 9: normalization
    stats = load_ml_norm_stats()
    assert stats["split"] == "train" and stats["n_samples"] == 6400
    Xn, yn, _ = ds[0]
    # material categorical: unchanged vs raw
    assert torch.allclose(
        Xn[channel_index("material")],
        X0[channel_index("material")],
    )
    print("[9] Normalization: train-only stats loaded; material not z-scored: OK")

    # inverse Kirchhoff sanity on raw theta
    theta = y0[0].numpy()
    T_hat = inverse_kirchhoff_from_sample(
        theta, m0["material"], m0["n_dielectric"], m0["n_silicon"], m0["n_copper"],
        T_ref=m0["T_ref_K"],
    )
    print(f"    inverse Kirchhoff T_hat range "
          f"[{T_hat.min():.3f}, {T_hat.max():.3f}] K")

    # 10: leakage check
    print("\n[10] Running validate_no_target_leakage()...")
    result = validate_no_target_leakage(n_check=16)
    assert result["passed"]
    print("     PASSED")

    # 11–12: val / test loaders
    for split in ("validation", "test"):
        ld = make_dataloader(split, batch_size=4, shuffle=False)
        Xv, yv = next(iter(ld))
        assert Xv.shape[0] == 4 and yv.shape[0] == 4
        assert torch.isfinite(Xv).all() and torch.isfinite(yv).all()
        print(f"[{11 if split == 'validation' else 12}] {split} loader OK  "
              f"X={tuple(Xv.shape)} y={tuple(yv.shape)}")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
