"""
sample_gen.py
=============
Builds ONE complete dataset sample: floorplan -> conductivity fields ->
power map -> boundary conditions -> non-linear anisotropic PDE solve ->
Kirchhoff transform -> validation / QC -> metadata record.
"""

import numpy as np

import config as C
import materials as M
import power as P
import boundary as B
import solver as S
import kirchhoff as K


def pick_difficulty(rng):
    r = rng.random()
    w = C.DIFFICULTY_WEIGHTS
    if r < w["normal"]:
        return "normal"
    elif r < w["normal"] + w["challenging"]:
        return "challenging"
    else:
        return "very_challenging"


def pick_pattern_type(rng, difficulty):
    if difficulty == "normal":
        pool = ["A", "B", "C", "D", "E", "F", "G", "J"]
    elif difficulty == "challenging":
        pool = ["G", "H", "I", "J", "B", "E"]
    else:
        pool = ["H", "I", "G", "E"]
    return str(rng.choice(pool))


def generate_sample(seed, sample_id, split):
    """
    Generate and validate a single sample.

    Returns (result_dict_or_None, log_entry_dict)
    log_entry_dict always contains status info for the rejection log / metadata.
    """
    rng = np.random.default_rng(seed)
    n = C.GRID_SIZE
    dx = C.DX

    difficulty = pick_difficulty(rng)
    pattern_type = pick_pattern_type(rng, difficulty)

    # ---------------- material / floorplan ----------------
    mat, fractions = M.generate_floorplan(rng, difficulty)
    k_ref_map, n_map, k_ref_scalar, n_scalar = M.assign_material_properties(mat, rng)

    if difficulty == "very_challenging" and rng.random() < 0.5:
        # push exponent for silicon toward the strong non-linear end
        n_map = np.where(mat == 1, rng.uniform(1.3, 1.5), n_map)

    strong_aniso = (rng.random() < C.ANISOTROPY_STRONG_PROB) or (difficulty != "normal" and rng.random() < 0.3)
    ax, ay = M.generate_anisotropy_fields(mat, rng, strong=strong_aniso)

    kx_base = k_ref_map * ax   # T-independent part of kx [W/(m.K)]
    ky_base = k_ref_map * ay

    # ---------------- power map ----------------
    Q_areal, power_info = P.generate_power_map(mat, rng, pattern_type, difficulty)  # W/m^2
    Q_vol = Q_areal / C.D_CHIP  # W/m^3

    # ---------------- boundary conditions ----------------
    boundary, bc_meta = B.generate_boundary_condition(rng, difficulty)

    # ---------------- solve non-linear PDE ----------------
    result = S.solve_heat_pde(
        kx_base, ky_base, n_map, Q_vol,
        h_sink=bc_meta["h_sink"], d_chip=bc_meta["d_chip"],
        T_ambient_sink=bc_meta["T_ambient"], dx=dx, boundary=boundary,
    )
    T = result["T"]
    kx_final, ky_final = result["kx_final"], result["ky_final"]
    converged = result["converged"]
    iterations = result["iterations"]

    log = {
        "sample_id": sample_id, "split": split, "seed": seed,
        "difficulty": difficulty, "pattern_type": pattern_type,
        "valid": False, "reason": "",
    }

    if not converged or not np.all(np.isfinite(T)):
        log["reason"] = "solver_not_converged_or_nan"
        return None, log

    if np.any(~np.isfinite(T)):
        log["reason"] = "nan_or_inf"
        return None, log

    T_min, T_max, T_mean = float(T.min()), float(T.max()), float(T.mean())
    delta_T = T_max - bc_meta["T_ambient"]

    if delta_T < C.REJECT_MIN_DELTA_T:
        log["reason"] = "delta_T_too_small"
        return None, log
    if T_max > C.REJECT_MAX_T:
        log["reason"] = "T_max_unphysical"
        return None, log

    # ---------------- PDE residual ----------------
    residual = S.compute_pde_residual(T, result["A"], result["b"], dx)
    resid_scale = max(np.max(np.abs(Q_vol)), 1.0)
    residual_norm = residual / resid_scale
    pde_residual_mean = float(np.mean(np.abs(residual_norm)))
    pde_residual_max = float(np.max(np.abs(residual_norm)))

    if pde_residual_max > C.REJECT_MAX_RESIDUAL:
        log["reason"] = "pde_residual_too_large"
        return None, log

    # ---------------- global heat balance ----------------
    balance = S.compute_heat_balance(
        T, kx_final, ky_final, Q_areal,
        h_sink=bc_meta["h_sink"], d_chip=bc_meta["d_chip"],
        T_ambient_sink=bc_meta["T_ambient"], dx=dx, boundary=boundary,
    )
    if balance["balance_error_pct"] > C.REJECT_MAX_HEAT_BALANCE_PCT:
        log["reason"] = "heat_balance_error_too_large"
        return None, log

    if len(np.unique(mat)) < 2:
        log["reason"] = "degenerate_material_map"
        return None, log

    # ---------------- Kirchhoff transform + round trip validation ----------------
    rt_error, theta, T_reconstructed = K.roundtrip_error(T, n_map, C.T_REF)
    if rt_error > C.KIRCHHOFF_ROUNDTRIP_TOL:
        # attempt one automatic fix: re-derive theta/T with float64 highest precision
        # (already float64) - if still failing, reject.
        log["reason"] = "kirchhoff_roundtrip_failed"
        return None, log

    # ---------------- assemble sample ----------------
    sample = {
        "Q": Q_areal.astype(np.float32),
        "material": mat.astype(np.int8),
        "kx": kx_final.astype(np.float32),
        "ky": ky_final.astype(np.float32),
        "ax": ax.astype(np.float32),
        "ay": ay.astype(np.float32),
        "theta": theta.astype(np.float32),
        "T": T.astype(np.float32),
    }

    total_power_W = float(np.sum(Q_areal) * dx * dx)
    mean_kx = float(np.mean(kx_final))
    mean_ky = float(np.mean(ky_final))
    mean_aniso = float(np.mean(kx_final / (ky_final + 1e-30)))

    meta = {
        "sample_id": sample_id,
        "split": split,
        "seed": int(seed),
        "grid_size": n,
        "total_power_W": total_power_W,
        "max_power_W_m2": float(np.max(Q_areal)),
        "num_hotspots": power_info["num_hotspots"],
        "hotspot_type": power_info["pattern_type"],
        "silicon_fraction": fractions["silicon"],
        "copper_fraction": fractions["copper"],
        "dielectric_fraction": fractions["dielectric"],
        "mean_kx": mean_kx,
        "mean_ky": mean_ky,
        "mean_anisotropy": mean_aniso,
        "boundary_type": bc_meta["scenario"],
        "h_sink": bc_meta["h_sink"],
        "T_ambient": bc_meta["T_ambient"],
        "T_min": T_min,
        "T_max": T_max,
        "T_mean": T_mean,
        "delta_T": delta_T,
        "mean_theta": float(np.mean(theta)),
        # per-material Kirchhoff/temperature-dependence exponents & reference
        # conductivities for THIS sample (randomized per sample within the
        # configured ranges) -- required downstream to reconstruct n_map =
        # n_dielectric*(material==0) + n_silicon*(material==1) + n_copper*(material==2)
        # and thus to apply the analytical inverse Kirchhoff transform to a
        # NETWORK-PREDICTED theta(x,y) field (T = psi^-1(theta; n_map, T_ref)).
        "n_dielectric": float(n_scalar[0]),
        "n_silicon": float(n_scalar[1]),
        "n_copper": float(n_scalar[2]),
        "k_ref_dielectric": float(k_ref_scalar[0]),
        "k_ref_silicon": float(k_ref_scalar[1]),
        "k_ref_copper": float(k_ref_scalar[2]),
        "T_ref_K": float(C.T_REF),
        "solver_iterations": iterations,
        "pde_residual_mean": pde_residual_mean,
        "pde_residual_max": pde_residual_max,
        "heat_balance_error_pct": balance["balance_error_pct"],
        "kirchhoff_roundtrip_error_K": rt_error,
        "valid": True,
        "difficulty_level": difficulty,
    }

    log["valid"] = True
    log["reason"] = "ok"
    return {"sample": sample, "meta": meta}, log
