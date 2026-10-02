"""
kirchhoff.py
============
Forward and inverse Kirchhoff temperature transformation for a material
with temperature-dependent conductivity k(T) = k_ref * (T/T_ref)^(-n).

Forward:
    theta(T) = integral_{T_ref}^{T} k(tau)/k_ref d(tau)
             = T_ref/(1-n) * [ (T/T_ref)^(1-n) - 1 ]      (n != 1)
             = T_ref * ln(T/T_ref)                          (n == 1)

Inverse:
    T(theta) = T_ref * [ 1 + (1-n)*theta/T_ref ]^(1/(1-n))  (n != 1)
    T(theta) = T_ref * exp(theta/T_ref)                      (n == 1)

theta does not depend on the magnitude of k_ref, only on T, n and T_ref,
since we integrate the *ratio* k(tau)/k_ref. n is applied per-cell using
each cell's local material exponent (spatially varying n_map), which is
what the project spec calls evaluating theta "relative to the material's
isotropic base component k_ref(T)".
"""

import numpy as np

N_EQ_1_TOL = 1e-3


def forward_kirchhoff(T, n_map, T_ref):
    T = np.asarray(T, dtype=np.float64)
    n_map = np.asarray(n_map, dtype=np.float64)
    theta = np.empty_like(T)

    near_one = np.abs(n_map - 1.0) < N_EQ_1_TOL
    far = ~near_one

    if np.any(far):
        n_f = n_map[far]
        theta[far] = (T_ref / (1.0 - n_f)) * ((T[far] / T_ref) ** (1.0 - n_f) - 1.0)
    if np.any(near_one):
        theta[near_one] = T_ref * np.log(T[near_one] / T_ref)
    return theta


def inverse_kirchhoff(theta, n_map, T_ref):
    theta = np.asarray(theta, dtype=np.float64)
    n_map = np.asarray(n_map, dtype=np.float64)
    T = np.empty_like(theta)

    near_one = np.abs(n_map - 1.0) < N_EQ_1_TOL
    far = ~near_one

    if np.any(far):
        n_f = n_map[far]
        base = 1.0 + (1.0 - n_f) * theta[far] / T_ref
        base = np.clip(base, 1e-6, None)  # guard against invalid fractional powers
        T[far] = T_ref * base ** (1.0 / (1.0 - n_f))
    if np.any(near_one):
        T[near_one] = T_ref * np.exp(theta[near_one] / T_ref)
    return T


def roundtrip_error(T, n_map, T_ref):
    """max abs error of T -> theta -> T_reconstructed, for validation."""
    theta = forward_kirchhoff(T, n_map, T_ref)
    T_back = inverse_kirchhoff(theta, n_map, T_ref)
    return float(np.max(np.abs(T - T_back))), theta, T_back
