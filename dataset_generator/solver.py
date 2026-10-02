"""
solver.py
=========
Finite-volume, non-linear, anisotropic 2D steady-state heat conduction
solver with:
  * harmonic-mean interface conductivity (sharp material interfaces)
  * temperature-dependent conductivity k(T) = k_ref * (T/T_ref)^-n
  * Robin/Dirichlet/Neumann edge boundary conditions (resistance-in-series
    treatment for Robin, half-cell-distance ghost treatment for Dirichlet)
  * an out-of-plane lumped "vertical heat-sink" loss term
    q_sink = (h_sink/d_chip) * (T - T_ambient)

Governing PDE (steady state):
    d/dx( kx(T,x,y) dT/dx ) + d/dy( ky(T,x,y) dT/dy )
        - (h_sink/d_chip)(T - T_ambient) + Q_vol(x,y) = 0

Q_vol [W/m^3] = Q_areal [W/m^2] / d_chip, i.e. the areal chip-surface power
density is converted to an equivalent volumetric source using the lumped
package thickness d_chip, so every term in the PDE is expressed in W/m^3.

The whole per-cell balance equation is multiplied through by dx^2 before
assembly purely for numerical conditioning (dx=dy); this does not change
the physical content of the model, all boundary/interface terms are scaled
consistently (see README for the full derivation).

Non-linearity is handled with Picard fixed-point iteration + adaptive
under-relaxation: at each iteration k(T) is refreshed from the current
temperature estimate, the resulting *linear* system is solved exactly with
a sparse direct solver, and iteration continues until
max_i |T^(m+1) - T^(m)| < tol.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

import config as C


def harmonic_mean(a, b):
    return 2.0 * a * b / (a + b + 1e-30)


def assemble_system(kx, ky, Q_vol, h_sink, d_chip, T_ambient_sink, dx, boundary):
    """
    Build the sparse linear system A @ T.ravel() = b for the CURRENT
    (already temperature-evaluated) conductivity fields kx, ky.

    boundary: dict with keys 'x0','x1','y0','y1', each a dict:
        {'type': 'dirichlet'|'convective'|'insulated',
         'T_boundary': float (dirichlet),
         'h_edge': float (convective), 'T_ambient': float (convective)}
    """
    N = kx.shape[0]
    idx = np.arange(N * N).reshape(N, N)
    diag = np.zeros((N, N))
    rhs = np.zeros((N, N))
    rows_list = []
    cols_list = []
    vals_list = []

    # ---- interior x-direction faces (between i, i+1) ----
    kxf = harmonic_mean(kx[:-1, :], kx[1:, :])           # (N-1, N)
    diag[:-1, :] += kxf
    diag[1:, :] += kxf
    r = idx[:-1, :].ravel()
    c = idx[1:, :].ravel()
    v = -kxf.ravel()
    rows_list += [r, c]
    cols_list += [c, r]
    vals_list += [v, v]

    # ---- interior y-direction faces (between j, j+1) ----
    kyf = harmonic_mean(ky[:, :-1], ky[:, 1:])           # (N, N-1)
    diag[:, :-1] += kyf
    diag[:, 1:] += kyf
    r = idx[:, :-1].ravel()
    c = idx[:, 1:].ravel()
    v = -kyf.ravel()
    rows_list += [r, c]
    cols_list += [c, r]
    vals_list += [v, v]

    # ---- boundary faces ----
    def edge_coef_and_rhs(k_line, spec):
        btype = spec["type"]
        if btype == "insulated":
            return np.zeros_like(k_line), np.zeros_like(k_line)
        elif btype == "dirichlet":
            coef = 2.0 * k_line
            return coef, coef * spec["T_boundary"]
        elif btype == "convective":
            h_edge = spec["h_edge"]
            h_eff = 1.0 / ((dx / 2.0) / k_line + 1.0 / h_edge)
            coef = h_eff * dx
            t_amb = spec.get("T_ambient", T_ambient_sink)
            return coef, coef * t_amb
        else:
            raise ValueError(f"Unknown boundary type: {btype}")

    coef, contrib = edge_coef_and_rhs(kx[0, :], boundary["x0"])
    diag[0, :] += coef
    rhs[0, :] += contrib

    coef, contrib = edge_coef_and_rhs(kx[-1, :], boundary["x1"])
    diag[-1, :] += coef
    rhs[-1, :] += contrib

    coef, contrib = edge_coef_and_rhs(ky[:, 0], boundary["y0"])
    diag[:, 0] += coef
    rhs[:, 0] += contrib

    coef, contrib = edge_coef_and_rhs(ky[:, -1], boundary["y1"])
    diag[:, -1] += coef
    rhs[:, -1] += contrib

    # ---- vertical (out-of-plane) sink term ----
    sink_coef = (h_sink / d_chip) * dx * dx
    diag += sink_coef
    rhs += sink_coef * T_ambient_sink

    # ---- volumetric source term ----
    rhs += Q_vol * dx * dx

    rows_all = np.concatenate(rows_list + [idx.ravel()])
    cols_all = np.concatenate(cols_list + [idx.ravel()])
    vals_all = np.concatenate(vals_list + [diag.ravel()])

    A = sparse.coo_matrix((vals_all, (rows_all, cols_all)), shape=(N * N, N * N)).tocsr()
    b = rhs.ravel()
    return A, b


def solve_heat_pde(kx_base, ky_base, n_map, Q_vol, h_sink, d_chip,
                    T_ambient_sink, dx, boundary, T_ref=C.T_REF,
                    max_iter=C.SOLVER_MAX_PICARD_ITER, tol=C.SOLVER_TOL_K,
                    relax0=C.SOLVER_RELAXATION):
    """
    Solve the non-linear steady state PDE via Picard iteration.

    kx_base, ky_base : (N,N) k_ref * anisotropy_ratio  [W/(m.K)] (T-independent part)
    n_map             : (N,N) local temperature exponent for k(T)
    Q_vol             : (N,N) volumetric source [W/m^3]

    Returns dict with T, kx_final, ky_final, iterations, converged, A, b
    (A,b are the final assembled linear system, reused for residual checks).
    """
    N = kx_base.shape[0]
    T = np.full((N, N), T_ambient_sink, dtype=np.float64)
    relax = relax0
    converged = False
    prev_diff = None
    lo_r, hi_r = C.T_CLIP_RATIO
    iterations = max_iter
    A, b = None, None

    for it in range(1, max_iter + 1):
        ratio = np.clip(T / T_ref, lo_r, hi_r)
        kx = kx_base * ratio ** (-n_map)
        ky = ky_base * ratio ** (-n_map)

        A, b = assemble_system(kx, ky, Q_vol, h_sink, d_chip, T_ambient_sink, dx, boundary)
        try:
            T_solved_flat = spsolve(A, b)
        except Exception:
            iterations = it
            converged = False
            break
        T_solved = T_solved_flat.reshape(N, N)

        if not np.all(np.isfinite(T_solved)):
            iterations = it
            converged = False
            break

        diff = float(np.max(np.abs(T_solved - T)))

        if prev_diff is not None and diff > prev_diff * 1.5:
            relax = max(C.SOLVER_RELAXATION_MIN, relax * 0.6)

        T_new = T + relax * (T_solved - T)

        if diff < tol:
            T = T_new
            converged = True
            iterations = it
            break

        T = T_new
        prev_diff = diff
    else:
        iterations = max_iter

    # final conductivities evaluated at the converged (or last) temperature
    ratio = np.clip(T / T_ref, lo_r, hi_r)
    kx_final = kx_base * ratio ** (-n_map)
    ky_final = ky_base * ratio ** (-n_map)

    if A is None or not np.all(np.isfinite(T)):
        converged = False

    return {
        "T": T,
        "kx_final": kx_final,
        "ky_final": ky_final,
        "iterations": iterations,
        "converged": converged,
        "A": A,
        "b": b,
    }


def compute_pde_residual(T, A, b, dx):
    """Residual field in W/m^3 (same scale as Q_vol) from the final linear system."""
    N = T.shape[0]
    resid_flat = A.dot(T.ravel()) - b
    residual = resid_flat.reshape(N, N) / (dx * dx)
    return residual


def compute_heat_balance(T, kx, ky, Q_areal, h_sink, d_chip, T_ambient_sink, dx, boundary):
    """
    Global energy balance check [W]:
        total_Q_W ~= total_sink_W + total_edge_W
    """
    total_Q_W = float(np.sum(Q_areal) * dx * dx)
    total_sink_W = float(np.sum(h_sink * (T - T_ambient_sink)) * dx * dx)

    def edge_flux_W(k_line, T_line, spec):
        btype = spec["type"]
        if btype == "insulated":
            return 0.0
        elif btype == "dirichlet":
            flux_out = 2.0 * k_line * (T_line - spec["T_boundary"]) / dx
        elif btype == "convective":
            h_edge = spec["h_edge"]
            h_eff = 1.0 / ((dx / 2.0) / k_line + 1.0 / h_edge)
            t_amb = spec.get("T_ambient", T_ambient_sink)
            flux_out = h_eff * (T_line - t_amb)
        else:
            raise ValueError(f"Unknown boundary type: {btype}")
        return float(np.sum(flux_out) * dx * d_chip)

    total_edge_W = (
        edge_flux_W(kx[0, :], T[0, :], boundary["x0"]) +
        edge_flux_W(kx[-1, :], T[-1, :], boundary["x1"]) +
        edge_flux_W(ky[:, 0], T[:, 0], boundary["y0"]) +
        edge_flux_W(ky[:, -1], T[:, -1], boundary["y1"])
    )

    denom = max(abs(total_Q_W), 1e-12)
    balance_error_pct = abs(total_Q_W - (total_sink_W + total_edge_W)) / denom * 100.0

    return {
        "total_Q_W": total_Q_W,
        "total_sink_W": total_sink_W,
        "total_edge_W": total_edge_W,
        "balance_error_pct": balance_error_pct,
    }
