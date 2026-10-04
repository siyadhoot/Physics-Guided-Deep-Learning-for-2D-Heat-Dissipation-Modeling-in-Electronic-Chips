"""
Input-channel construction for the leakage-free FNO pipeline.

X shape: (C, H, W) with C = N_INPUT_CHANNELS
y shape: (1, H, W)  — Kirchhoff potential theta
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np

# Edge order in channels: left(x0), right(x1), bottom(y0), top(y1)
EDGE_ORDER = ("x0", "x1", "y0", "y1")
EDGE_NAMES = ("left", "right", "bottom", "top")

CHANNEL_NAMES: List[str] = (
    [
        "Q",
        "material",
        "kx_base",
        "ky_base",
        "ax",
        "ay",
    ]
    + [f"bc_type_{name}" for name in EDGE_NAMES]
    + [f"bc_h_{name}" for name in EDGE_NAMES]
    + [f"bc_T_{name}" for name in EDGE_NAMES]
    + [
        "h_sink",
        "T_ambient",
        "n_dielectric",
        "n_silicon",
        "n_copper",
    ]
)

TARGET_NAME = "theta"
N_INPUT_CHANNELS = len(CHANNEL_NAMES)

# Channels that are categorical / discrete — never z-score normalized
CATEGORICAL_CHANNELS = frozenset(
    ["material"] + [f"bc_type_{name}" for name in EDGE_NAMES]
)


def _paint_edge(field: np.ndarray, edge: str, value: float) -> None:
    """Write ``value`` onto the boundary edge of a (H,W) array (in-place)."""
    if edge == "x0":  # left / west  (i = 0)
        field[0, :] = value
    elif edge == "x1":  # right / east (i = -1)
        field[-1, :] = value
    elif edge == "y0":  # bottom / south (j = 0)
        field[:, 0] = value
    elif edge == "y1":  # top / north (j = -1)
        field[:, -1] = value
    else:
        raise ValueError(f"Unknown edge key: {edge}")


def build_k_ref_map(
    material: np.ndarray,
    k_ref_dielectric: float,
    k_ref_silicon: float,
    k_ref_copper: float,
) -> np.ndarray:
    """k_ref(x,y) from material IDs and per-sample scalar k_ref values."""
    k_ref = np.zeros(material.shape, dtype=np.float64)
    k_ref[material == 0] = k_ref_dielectric
    k_ref[material == 1] = k_ref_silicon
    k_ref[material == 2] = k_ref_copper
    return k_ref


def build_kx_ky_base(
    material: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    k_ref_dielectric: float,
    k_ref_silicon: float,
    k_ref_copper: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    kx_base = k_ref(material) * ax
    ky_base = k_ref(material) * ay

    Uses metadata k_ref scalars — NEVER derived from converged kx/ky.
    """
    k_ref = build_k_ref_map(material, k_ref_dielectric, k_ref_silicon, k_ref_copper)
    kx_base = (k_ref * np.asarray(ax, dtype=np.float64)).astype(np.float32)
    ky_base = (k_ref * np.asarray(ay, dtype=np.float64)).astype(np.float32)
    return kx_base, ky_base


def build_input_channels(
    Q: np.ndarray,
    material: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    kx_base: np.ndarray,
    ky_base: np.ndarray,
    edge_params: Mapping[str, Mapping[str, float]],
    h_sink: float,
    T_ambient: float,
    n_dielectric: float,
    n_silicon: float,
    n_copper: float,
) -> np.ndarray:
    """
    Assemble raw (unnormalized) input tensor X of shape (C, H, W).

    Boundary channels are zero in the interior; edge pixels carry the
    physical type / h / T parameter for that edge.
    Global scalars are broadcast as constant fields.
    """
    H, W = Q.shape
    channels: List[np.ndarray] = [
        np.asarray(Q, dtype=np.float32),
        np.asarray(material, dtype=np.float32),
        np.asarray(kx_base, dtype=np.float32),
        np.asarray(ky_base, dtype=np.float32),
        np.asarray(ax, dtype=np.float32),
        np.asarray(ay, dtype=np.float32),
    ]

    # bc_type_*
    for edge in EDGE_ORDER:
        field = np.zeros((H, W), dtype=np.float32)
        _paint_edge(field, edge, float(edge_params[edge]["type"]))
        channels.append(field)

    # bc_h_*
    for edge in EDGE_ORDER:
        field = np.zeros((H, W), dtype=np.float32)
        _paint_edge(field, edge, float(edge_params[edge]["h"]))
        channels.append(field)

    # bc_T_*
    for edge in EDGE_ORDER:
        field = np.zeros((H, W), dtype=np.float32)
        _paint_edge(field, edge, float(edge_params[edge]["T"]))
        channels.append(field)

    # global condition channels
    channels.append(np.full((H, W), float(h_sink), dtype=np.float32))
    channels.append(np.full((H, W), float(T_ambient), dtype=np.float32))
    channels.append(np.full((H, W), float(n_dielectric), dtype=np.float32))
    channels.append(np.full((H, W), float(n_silicon), dtype=np.float32))
    channels.append(np.full((H, W), float(n_copper), dtype=np.float32))

    X = np.stack(channels, axis=0)
    assert X.shape[0] == N_INPUT_CHANNELS, (X.shape, N_INPUT_CHANNELS)
    return X


def channel_index(name: str) -> int:
    return CHANNEL_NAMES.index(name)
