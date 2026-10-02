"""
materials.py
============
Generates realistic, simplified electronic-chip floorplans as 2D material
maps with 3 classes: 0=dielectric, 1=silicon, 2=copper.

Floorplans are built from geometric primitives (rectangles, strips,
interconnect traces, memory-array blocks, irregular blown-up regions) so
that sharp, physically meaningful material interfaces are preserved
(no per-pixel random noise).
"""

import numpy as np
from scipy import ndimage

import config as C


def _rand_rect(rng, n, min_frac=0.06, max_frac=0.45):
    """Return (i0,i1,j0,j1) integer bounding box for a rectangle."""
    min_size = max(1, int(min_frac * n))
    max_size = max(min_size + 1, int(max_frac * n))
    h = rng.integers(min_size, max_size + 1)
    w = rng.integers(min_size, max_size + 1)
    i0 = rng.integers(0, max(1, n - h + 1))
    j0 = rng.integers(0, max(1, n - w + 1))
    return i0, min(n, i0 + h), j0, min(n, j0 + w)


def _add_strip(mat, rng, n, material_id, horizontal=True):
    """Thin conductive/interconnect trace strip."""
    thickness = max(1, int(rng.uniform(0.02, 0.06) * n))
    if horizontal:
        j0 = rng.integers(0, n)
        i0, i1, _, _ = _rand_rect(rng, n, 0.3, 0.95)
        j1 = min(n, j0 + thickness)
        mat[i0:i1, j0:j1] = material_id
    else:
        i0 = rng.integers(0, n)
        _, _, j0, j1 = _rand_rect(rng, n, 0.3, 0.95)
        i1 = min(n, i0 + thickness)
        mat[i0:i1, j0:j1] = material_id


def _add_memory_array(mat, rng, n, material_a, material_b):
    """Repetitive grid of small alternating blocks -> memory-array look."""
    i0, i1, j0, j1 = _rand_rect(rng, n, 0.25, 0.55)
    rows = rng.integers(3, 8)
    cols = rng.integers(3, 8)
    h = max(1, (i1 - i0) // rows)
    w = max(1, (j1 - j0) // cols)
    for r in range(rows):
        for c in range(cols):
            ii0 = i0 + r * h
            jj0 = j0 + c * w
            ii1 = min(n, ii0 + max(1, int(h * 0.8)))
            jj1 = min(n, jj0 + max(1, int(w * 0.8)))
            mid = material_a if (r + c) % 2 == 0 else material_b
            mat[ii0:ii1, jj0:jj1] = mid


def _add_irregular_blob(mat, rng, n, material_id):
    """Irregular connected region via thresholded smoothed noise."""
    noise = rng.standard_normal((n, n))
    sigma = rng.uniform(1.5, 4.0)
    noise = ndimage.gaussian_filter(noise, sigma=sigma)
    thresh = np.quantile(noise, rng.uniform(0.6, 0.85))
    mask = noise > thresh
    # keep only the largest connected component for a "blob" look
    labeled, num = ndimage.label(mask)
    if num > 0:
        sizes = ndimage.sum(mask, labeled, range(1, num + 1))
        biggest = np.argmax(sizes) + 1
        mask = labeled == biggest
    mat[mask] = material_id


def generate_floorplan(rng, difficulty="normal"):
    """
    Build an (N,N) int8 material map with realistic chip-like structure.

    Copper and dielectric areas are grown (via randomly placed rectangles /
    strips) until per-sample TARGET area fractions are reached, so that the
    dataset-level composition distribution matches the project spec:
    generally 60-80% Si+Dielectric with copper in smaller targeted regions,
    but with challenging/very_challenging samples deliberately allowed much
    larger copper structures.

    Returns
    -------
    mat : (N,N) int8  material ids {0,1,2}
    fractions : dict  {silicon, copper, dielectric} area fractions
    """
    n = C.GRID_SIZE
    mat = np.ones((n, n), dtype=np.int8)  # base: silicon die

    # optional thin dielectric isolation border (not every sample)
    if rng.random() < 0.5:
        border = 1
        mat[:border, :] = 0
        mat[-border:, :] = 0
        mat[:, :border] = 0
        mat[:, -border:] = 0

    # ---- per-sample target area fractions (drives 60-80% Si/Dielectric rule) ----
    if difficulty == "normal":
        target_copper = rng.uniform(0.10, 0.32)
        target_dielectric = rng.uniform(0.12, 0.30)
        rect_hi = 0.35
    elif difficulty == "challenging":
        target_copper = rng.uniform(0.20, 0.45)
        target_dielectric = rng.uniform(0.10, 0.25)
        rect_hi = 0.42
    else:  # very_challenging: allow large copper structures
        target_copper = rng.uniform(*C.CHALLENGING_COPPER_FRACTION_RANGE)
        target_dielectric = rng.uniform(0.08, 0.22)
        rect_hi = 0.50

    # ---- grow copper regions (blocks + interconnect strips) toward target ----
    attempts = 0
    while (mat == 2).mean() < target_copper and attempts < 60:
        attempts += 1
        if rng.random() < 0.65:
            i0, i1, j0, j1 = _rand_rect(rng, n, 0.10, rect_hi)
            mat[i0:i1, j0:j1] = 2
        else:
            _add_strip(mat, rng, n, 2, horizontal=bool(rng.integers(0, 2)))

    # ---- grow dielectric isolation regions toward target ----
    attempts = 0
    while (mat == 0).mean() < target_dielectric and attempts < 60:
        attempts += 1
        i0, i1, j0, j1 = _rand_rect(rng, n, 0.06, 0.26)
        mat[i0:i1, j0:j1] = 0

    # occasional memory-array-like block (alternating Si/dielectric or Si/Cu)
    if rng.random() < 0.35:
        pair = rng.choice([(1, 0), (1, 2), (0, 2)])
        _add_memory_array(mat, rng, n, int(pair[0]), int(pair[1]))

    # occasional irregular connected blob (processing block / odd-shaped region)
    if rng.random() < 0.35:
        blob_material = int(rng.choice([0, 1, 2]))
        _add_irregular_blob(mat, rng, n, blob_material)

    fractions = _compute_fractions(mat)

    # rescue: guarantee non-degenerate multi-material map
    if len(np.unique(mat)) < 2:
        i0, i1, j0, j1 = _rand_rect(rng, n, 0.2, 0.4)
        mat[i0:i1, j0:j1] = 2
        fractions = _compute_fractions(mat)

    return mat, fractions


def _compute_fractions(mat):
    n = mat.size
    return {
        "dielectric": float(np.sum(mat == 0)) / n,
        "silicon": float(np.sum(mat == 1)) / n,
        "copper": float(np.sum(mat == 2)) / n,
    }


def material_interfaces_mask(mat):
    """Boolean mask (N,N) of cells adjacent to a different material (boundary pixels)."""
    diff = np.zeros_like(mat, dtype=bool)
    diff[:-1, :] |= mat[:-1, :] != mat[1:, :]
    diff[1:, :] |= mat[:-1, :] != mat[1:, :]
    diff[:, :-1] |= mat[:, :-1] != mat[:, 1:]
    diff[:, 1:] |= mat[:, :-1] != mat[:, 1:]
    return diff


def assign_material_properties(mat, rng):
    """
    Sample per-material scalar properties (k_ref, n exponent) for this
    sample and expand into full 2D fields: k_ref_map, n_map.
    """
    k_ref_map = np.zeros(mat.shape, dtype=np.float64)
    n_map = np.zeros(mat.shape, dtype=np.float64)
    k_ref_scalar = {}
    n_scalar = {}
    for mid in (0, 1, 2):
        lo, hi = C.MATERIAL_K_RANGE[mid]
        k_val = rng.uniform(lo, hi)
        lo_n, hi_n = C.MATERIAL_N_RANGE[mid]
        n_val = rng.uniform(lo_n, hi_n)
        k_ref_scalar[mid] = k_val
        n_scalar[mid] = n_val
        mask = mat == mid
        k_ref_map[mask] = k_val
        n_map[mask] = n_val
    return k_ref_map, n_map, k_ref_scalar, n_scalar


def generate_anisotropy_fields(mat, rng, strong=False):
    """
    Generate spatially-correlated anisotropy ratio fields ax(x,y), ay(x,y)
    (kx = k_ref*ax, ky = k_ref*ay). Each connected material region gets its
    own (ax,ay) pair, followed by mild smoothing to avoid pixel noise while
    preserving material-boundary sharpness (smoothing is masked per-region).
    """
    n = mat.shape[0]
    ax = np.ones((n, n), dtype=np.float64)
    ay = np.ones((n, n), dtype=np.float64)

    lo, hi = C.ANISOTROPY_STRONG_RANGE if strong else C.ANISOTROPY_STD_RANGE

    for mid in (0, 1, 2):
        region_mask = mat == mid
        if not np.any(region_mask):
            continue
        labeled, num = ndimage.label(region_mask)
        for lab in range(1, num + 1):
            sub = labeled == lab
            a_x = rng.uniform(lo, hi)
            # keep the ratio a_x/a_y within the configured range too
            a_y = rng.uniform(lo, hi)
            ax[sub] = a_x
            ay[sub] = a_y
    return ax, ay
