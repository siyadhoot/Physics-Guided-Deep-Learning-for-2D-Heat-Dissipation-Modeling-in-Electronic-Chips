"""
Reconstruct per-edge boundary conditions and *actual* material exponents
from the stored sample seed — without re-running the PDE solver.

Why this is needed
------------------
HDF5 metadata stores only ``boundary_type`` (scenario name), ``h_sink``,
and ``T_ambient``.  Per-edge ``h_edge``, which edge is Dirichlet vs
convective vs insulated (for shuffled scenarios), and ``T_boundary`` are
*not* stored.

They are fully determined by the sample seed and the generator RNG
sequence in ``dataset_generator/sample_gen.py``.  Replaying that sequence
up to (and including) ``generate_boundary_condition`` recovers the exact
BC dict that was used by the solver.

Additionally, for some ``very_challenging`` samples the generator
overwrites silicon ``n`` on ``n_map`` but does *not* update
``metadata/n_silicon``.  Seed replay recovers the true exponent that was
used for the Kirchhoff transform.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Tuple

import numpy as np

from .paths import GENERATOR_DIR

# Import generator modules (same RNG sequence as sample_gen.generate_sample)
if str(GENERATOR_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATOR_DIR))

import config as C  # noqa: E402
import materials as M  # noqa: E402
import power as P  # noqa: E402
import boundary as B  # noqa: E402


BC_TYPE_CODE = {
    "insulated": 0,
    "convective": 1,
    "dirichlet": 2,
}

EDGE_KEYS = ("x0", "x1", "y0", "y1")  # left, right, bottom, top


def _pick_difficulty(rng: np.random.Generator) -> str:
    r = rng.random()
    w = C.DIFFICULTY_WEIGHTS
    if r < w["normal"]:
        return "normal"
    if r < w["normal"] + w["challenging"]:
        return "challenging"
    return "very_challenging"


def _pick_pattern_type(rng: np.random.Generator, difficulty: str) -> str:
    if difficulty == "normal":
        pool = ["A", "B", "C", "D", "E", "F", "G", "J"]
    elif difficulty == "challenging":
        pool = ["G", "H", "I", "J", "B", "E"]
    else:
        pool = ["H", "I", "G", "E"]
    return str(rng.choice(pool))


def replay_pre_solve(seed: int) -> Dict[str, Any]:
    """
    Replay ``generate_sample`` RNG up through boundary generation.

    Does **not** call the PDE solver.  Returns material fields, anisotropy,
    power map, boundary dict, and the *actual* per-material n scalars used
    on ``n_map`` (including the very_challenging silicon override).
    """
    rng = np.random.default_rng(int(seed))
    difficulty = _pick_difficulty(rng)
    pattern_type = _pick_pattern_type(rng, difficulty)

    mat, fractions = M.generate_floorplan(rng, difficulty)
    k_ref_map, n_map, k_ref_scalar, n_scalar = M.assign_material_properties(mat, rng)

    # Mirror sample_gen.py: optional silicon-n override (n_scalar NOT updated there)
    n_actual = {
        0: float(n_scalar[0]),
        1: float(n_scalar[1]),
        2: float(n_scalar[2]),
    }
    if difficulty == "very_challenging" and rng.random() < 0.5:
        n_si = float(rng.uniform(1.3, 1.5))
        n_map = np.where(mat == 1, n_si, n_map)
        n_actual[1] = n_si

    strong_aniso = (rng.random() < C.ANISOTROPY_STRONG_PROB) or (
        difficulty != "normal" and rng.random() < 0.3
    )
    ax, ay = M.generate_anisotropy_fields(mat, rng, strong=strong_aniso)
    kx_base = k_ref_map * ax
    ky_base = k_ref_map * ay

    Q_areal, power_info = P.generate_power_map(mat, rng, pattern_type, difficulty)
    boundary, bc_meta = B.generate_boundary_condition(rng, difficulty)

    return {
        "material": mat,
        "k_ref_map": k_ref_map,
        "n_map": n_map,
        "k_ref_scalar": k_ref_scalar,
        "n_scalar_metadata": {0: float(n_scalar[0]), 1: float(n_scalar[1]), 2: float(n_scalar[2])},
        "n_actual": n_actual,
        "ax": ax,
        "ay": ay,
        "kx_base": kx_base,
        "ky_base": ky_base,
        "Q": Q_areal,
        "power_info": power_info,
        "boundary": boundary,
        "bc_meta": bc_meta,
        "difficulty": difficulty,
        "pattern_type": pattern_type,
        "fractions": fractions,
    }


def edge_params_from_boundary(boundary: Dict[str, dict]) -> Dict[str, Dict[str, float]]:
    """
    Flatten a solver boundary dict into per-edge (type_code, h, T_param).

    T_param:
      - Dirichlet  -> T_boundary
      - convective -> T_ambient (of that edge)
      - insulated  -> 0
    """
    out = {}
    for edge in EDGE_KEYS:
        spec = boundary[edge]
        btype = spec["type"]
        code = BC_TYPE_CODE[btype]
        if btype == "convective":
            h = float(spec["h_edge"])
            T_param = float(spec["T_ambient"])
        elif btype == "dirichlet":
            h = 0.0
            T_param = float(spec["T_boundary"])
        else:
            h = 0.0
            T_param = 0.0
        out[edge] = {"type": float(code), "h": h, "T": T_param}
    return out


def reconstruct_bc_and_n(seed: int) -> Tuple[Dict[str, Dict[str, float]], Dict[int, float], Dict[str, Any]]:
    """Convenience wrapper: seed -> (edge_params, n_actual, bc_meta)."""
    rep = replay_pre_solve(seed)
    return edge_params_from_boundary(rep["boundary"]), rep["n_actual"], rep["bc_meta"]
