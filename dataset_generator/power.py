"""
power.py
========
Generates realistic 2D power-density maps Q(x,y) [W/m^2] for the chip
surface: distributed background power + Gaussian hotspots + rectangular
active blocks, following pattern types A-J from the project spec.
"""

import numpy as np
from scipy import ndimage

import config as C
from materials import material_interfaces_mask


def _gaussian_hotspot(n, x0, y0, amp, sx, sy, theta):
    """Rotated anisotropic 2D Gaussian evaluated on an (n,n) normalized grid."""
    xs = np.linspace(0, 1, n)
    ys = np.linspace(0, 1, n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Xc = X - x0
    Yc = Y - y0
    ct, st = np.cos(theta), np.sin(theta)
    Xr = Xc * ct + Yc * st
    Yr = -Xc * st + Yc * ct
    return amp * np.exp(-(Xr ** 2 / (2 * sx ** 2) + Yr ** 2 / (2 * sy ** 2)))


def _pick_interface_location(mat, rng, kind="any"):
    """Pick a normalized (x0,y0) location near a material interface."""
    n = mat.shape[0]
    iface = material_interfaces_mask(mat)
    idxs = np.argwhere(iface)
    if len(idxs) == 0:
        return rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
    pick = idxs[rng.integers(0, len(idxs))]
    x0 = (pick[0] + 0.5) / n
    y0 = (pick[1] + 0.5) / n
    return float(x0), float(y0)


def generate_power_map(mat, rng, pattern_type, difficulty="normal"):
    """
    Build Q(x,y) [W/m^2] according to the requested pattern type.

    Returns
    -------
    Q : (N,N) float64 [W/m^2]
    info : dict  (num_hotspots, pattern_type, total_power diagnostics filled later)
    """
    n = C.GRID_SIZE
    bg_lo, bg_hi = C.BACKGROUND_POWER_RANGE_W_CM2
    background = rng.uniform(bg_lo, bg_hi) * C.W_CM2_TO_W_M2
    Q = np.full((n, n), background, dtype=np.float64)

    peak_lo, peak_hi = C.HOTSPOT_PEAK_RANGE_W_CM2
    num_hotspots = 1

    def add_hotspot(x0, y0, amp_wcm2=None, sx=None, sy=None, theta=None):
        amp = (amp_wcm2 if amp_wcm2 is not None else rng.uniform(peak_lo, peak_hi)) * C.W_CM2_TO_W_M2
        sx = sx if sx is not None else rng.uniform(0.03, 0.12)
        sy = sy if sy is not None else rng.uniform(0.03, 0.12)
        theta = theta if theta is not None else rng.uniform(0, np.pi)
        Q_add = _gaussian_hotspot(n, x0, y0, amp, sx, sy, theta)
        return Q_add

    if pattern_type == "A":  # single central hotspot
        num_hotspots = 1
        Q += add_hotspot(0.5, 0.5)

    elif pattern_type == "B":  # multiple distributed hotspots
        num_hotspots = int(rng.integers(3, 8))
        for _ in range(num_hotspots):
            Q += add_hotspot(rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85))

    elif pattern_type == "C":  # corner hotspot
        num_hotspots = 1
        corner = rng.choice([(0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9)])
        Q += add_hotspot(float(corner[0]), float(corner[1]), sx=rng.uniform(0.05, 0.1), sy=rng.uniform(0.05, 0.1))

    elif pattern_type == "D":  # edge hotspot
        num_hotspots = 1
        edge_choice = rng.integers(0, 4)
        pos = rng.uniform(0.2, 0.8)
        edges = [(0.05, pos), (0.95, pos), (pos, 0.05), (pos, 0.95)]
        x0, y0 = edges[edge_choice]
        Q += add_hotspot(x0, y0)

    elif pattern_type == "E":  # several narrow hotspots
        num_hotspots = int(rng.integers(4, 9))
        for _ in range(num_hotspots):
            Q += add_hotspot(rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9),
                              sx=rng.uniform(0.015, 0.04), sy=rng.uniform(0.015, 0.04))

    elif pattern_type == "F":  # large diffuse heating
        num_hotspots = 1
        Q += add_hotspot(rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7),
                          sx=rng.uniform(0.2, 0.4), sy=rng.uniform(0.2, 0.4),
                          amp_wcm2=rng.uniform(peak_lo * 0.3, peak_hi * 0.5))

    elif pattern_type == "G":  # two closely adjacent hotspots
        num_hotspots = 2
        x0, y0 = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
        dx_, dy_ = rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08)
        Q += add_hotspot(x0, y0)
        Q += add_hotspot(np.clip(x0 + dx_, 0.05, 0.95), np.clip(y0 + dy_, 0.05, 0.95))

    elif pattern_type == "H":  # hotspot adjacent to high-contrast interface
        num_hotspots = 1
        x0, y0 = _pick_interface_location(mat, rng)
        Q += add_hotspot(x0, y0, sx=rng.uniform(0.03, 0.07), sy=rng.uniform(0.03, 0.07))

    elif pattern_type == "I":  # hotspot crossing a material boundary
        num_hotspots = 1
        x0, y0 = _pick_interface_location(mat, rng)
        Q += add_hotspot(x0, y0, sx=rng.uniform(0.06, 0.12), sy=rng.uniform(0.06, 0.12))

    elif pattern_type == "J":  # asymmetric workload distribution
        num_hotspots = int(rng.integers(2, 5))
        bias_x = rng.uniform(0.1, 0.4)
        for _ in range(num_hotspots):
            Q += add_hotspot(np.clip(rng.normal(bias_x, 0.1), 0.05, 0.95),
                              rng.uniform(0.1, 0.9))
    else:
        num_hotspots = 1
        Q += add_hotspot(0.5, 0.5)

    # occasionally add a rectangular "active block" (CPU core / GPU EU / mem block)
    if rng.random() < 0.5:
        blk_lo, blk_hi = C.ACTIVE_BLOCK_POWER_RANGE_W_CM2
        amp = rng.uniform(blk_lo, blk_hi) * C.W_CM2_TO_W_M2
        i0 = rng.integers(0, n - n // 4)
        j0 = rng.integers(0, n - n // 4)
        h = rng.integers(n // 8, n // 3)
        w = rng.integers(n // 8, n // 3)
        i1, j1 = min(n, i0 + h), min(n, j0 + w)
        block = np.zeros((n, n))
        block[i0:i1, j0:j1] = amp
        block = ndimage.gaussian_filter(block, sigma=0.4)  # slightly soften block edges
        Q += block

    # extra critical hotspots for challenging / very_challenging difficulty
    if difficulty in ("challenging", "very_challenging"):
        extra = int(rng.integers(1, 3))
        for _ in range(extra):
            x0, y0 = _pick_interface_location(mat, rng)
            sx = rng.uniform(0.015, 0.05) if difficulty == "very_challenging" else rng.uniform(0.03, 0.08)
            amp_scale = rng.uniform(0.8, 1.3)
            Q += add_hotspot(x0, y0, sx=sx, sy=sx, amp_wcm2=rng.uniform(peak_lo, peak_hi) * amp_scale)
            num_hotspots += 1

    Q = np.clip(Q, 0.0, None)
    return Q, {"num_hotspots": int(num_hotspots), "pattern_type": pattern_type}
