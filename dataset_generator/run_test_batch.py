"""
run_test_batch.py
==================
Step 1 of the execution mandate: generate 20 test samples, run the full
solver + Kirchhoff + validation pipeline, print diagnostics, and produce
inspection plots BEFORE committing to the full 8000-sample run.
"""

import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from sample_gen import generate_sample

N_TEST = 20
OUT_DIR = C.TEST_DIR


def main():
    print(f"=== Generating {N_TEST} test samples (grid {C.GRID_SIZE}x{C.GRID_SIZE}) ===")
    results = []
    logs = []
    t0 = time.time()
    attempt = 0
    while len(results) < N_TEST and attempt < N_TEST * 5:
        seed = 999_000_000 + attempt
        r, log = generate_sample(seed, sample_id=attempt, split="test_calibration")
        logs.append(log)
        attempt += 1
        if r is not None:
            results.append(r)
            m = r["meta"]
            print(f"[OK ] id={m['sample_id']:3d} diff={m['difficulty_level']:17s} "
                  f"pat={m['hotspot_type']} iters={m['solver_iterations']:3d} "
                  f"Tmax={m['T_max']:.2f}K dT={m['delta_T']:.2f}K "
                  f"resid_max={m['pde_residual_max']:.2e} bal_err={m['heat_balance_error_pct']:.3f}% "
                  f"rt_err={m['kirchhoff_roundtrip_error_K']:.2e}K")
        else:
            print(f"[REJ] seed={seed} reason={log['reason']}")
    dt = time.time() - t0
    print(f"\nCompleted: {len(results)}/{attempt} attempts valid in {dt:.1f}s "
          f"({dt/max(1,attempt):.3f}s/attempt)")

    # ---- aggregate stats ----
    Tmaxs = [r["meta"]["T_max"] for r in results]
    dTs = [r["meta"]["delta_T"] for r in results]
    resid = [r["meta"]["pde_residual_max"] for r in results]
    bal = [r["meta"]["heat_balance_error_pct"] for r in results]
    iters = [r["meta"]["solver_iterations"] for r in results]
    rt = [r["meta"]["kirchhoff_roundtrip_error_K"] for r in results]

    summary = {
        "n_valid": len(results),
        "n_attempts": attempt,
        "T_max_range": [float(np.min(Tmaxs)), float(np.max(Tmaxs))],
        "delta_T_range": [float(np.min(dTs)), float(np.max(dTs))],
        "residual_max_worst": float(np.max(resid)),
        "balance_error_pct_worst": float(np.max(bal)),
        "iterations_mean": float(np.mean(iters)),
        "iterations_max": int(np.max(iters)),
        "kirchhoff_roundtrip_worst_K": float(np.max(rt)),
        "seconds_per_sample": dt / max(1, attempt),
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    with open(f"{OUT_DIR}/test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(f"{OUT_DIR}/test_logs.json", "w") as f:
        json.dump(logs, f, indent=2)

    # ---- visual inspection plots ----
    n_show = min(6, len(results))
    for k in range(n_show):
        plot_sample(results[k], f"{OUT_DIR}/test_sample_{k:02d}.png")
    print(f"\nSaved {n_show} inspection plots to {OUT_DIR}")

    ok = (
        summary["n_valid"] == N_TEST and
        summary["T_max_range"][1] < C.REJECT_MAX_T and
        summary["residual_max_worst"] <= C.REJECT_MAX_RESIDUAL and
        summary["balance_error_pct_worst"] <= C.REJECT_MAX_HEAT_BALANCE_PCT and
        summary["kirchhoff_roundtrip_worst_K"] <= C.KIRCHHOFF_ROUNDTRIP_TOL
    )
    print(f"\n20-SAMPLE VALIDATION GATE: {'PASSED' if ok else 'FAILED'}")
    return ok


def plot_sample(result, path):
    s, m = result["sample"], result["meta"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fields = [
        ("Q (W/m^2)", s["Q"], "hot"),
        ("Material (0=Dielec,1=Si,2=Cu)", s["material"], "viridis"),
        ("kx (W/m.K)", s["kx"], "plasma"),
        ("ky (W/m.K)", s["ky"], "plasma"),
        ("ax", s["ax"], "coolwarm"),
        ("ay", s["ay"], "coolwarm"),
        ("T (K)", s["T"], "inferno"),
        ("theta (Kirchhoff)", s["theta"], "cividis"),
    ]
    for ax, (title, field, cmap) in zip(axes.ravel(), fields):
        im = ax.imshow(field.T, origin="lower", cmap=cmap)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        f"id={m['sample_id']} diff={m['difficulty_level']} pattern={m['hotspot_type']} "
        f"Tmax={m['T_max']:.1f}K dT={m['delta_T']:.1f}K resid={m['pde_residual_max']:.1e} "
        f"bal_err={m['heat_balance_error_pct']:.2f}%"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
