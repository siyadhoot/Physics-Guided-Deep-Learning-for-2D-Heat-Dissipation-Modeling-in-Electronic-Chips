"""Kirchhoff helpers using the *actual* per-sample n parameters."""

from __future__ import annotations

import sys
from typing import Mapping, Optional, Union

import numpy as np

from .paths import GENERATOR_DIR

if str(GENERATOR_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATOR_DIR))

import kirchhoff as K  # noqa: E402


def build_n_map(
    material: np.ndarray,
    n_dielectric: float,
    n_silicon: float,
    n_copper: float,
) -> np.ndarray:
    """Construct spatially varying n(x,y) from material IDs and scalars."""
    n_map = np.zeros(material.shape, dtype=np.float64)
    n_map[material == 0] = n_dielectric
    n_map[material == 1] = n_silicon
    n_map[material == 2] = n_copper
    return n_map


def inverse_kirchhoff_from_sample(
    theta: np.ndarray,
    material: np.ndarray,
    n_dielectric: float,
    n_silicon: float,
    n_copper: float,
    T_ref: float = 300.0,
) -> np.ndarray:
    """
    T = psi^{-1}(theta; n_map, T_ref)

    ``n_silicon`` must be the *actual* exponent used at generation time
    (from the aux BC cache), not necessarily ``metadata/n_silicon``.
    """
    n_map = build_n_map(material, n_dielectric, n_silicon, n_copper)
    return K.inverse_kirchhoff(theta, n_map, T_ref)
