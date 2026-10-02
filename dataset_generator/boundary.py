"""
boundary.py
===========
Builds a per-sample boundary-condition specification: one entry per domain
edge (x0=west/x=0, x1=east/x=1, y0=south/y=0, y1=north/y=1), plus the global
out-of-plane package parameters (h_sink, d_chip, T_ambient).
"""

import numpy as np

import config as C


def generate_boundary_condition(rng, difficulty="normal"):
    scenario = rng.choice(C.BOUNDARY_SCENARIOS)
    T_ambient = C.T_AMBIENT_NOMINAL + rng.uniform(-C.T_AMBIENT_JITTER, C.T_AMBIENT_JITTER)
    h_sink = rng.uniform(*C.H_SINK_RANGE)

    def conv(h_lo=None, h_hi=None, t_amb=None):
        h_lo = C.H_EDGE_RANGE[0] if h_lo is None else h_lo
        h_hi = C.H_EDGE_RANGE[1] if h_hi is None else h_hi
        return {"type": "convective", "h_edge": rng.uniform(h_lo, h_hi),
                "T_ambient": T_ambient if t_amb is None else t_amb}

    def insulated():
        return {"type": "insulated"}

    def dirichlet(t=None):
        return {"type": "dirichlet", "T_boundary": T_ambient if t is None else t}

    if scenario == "all_edges_convective":
        boundary = {k: conv() for k in ("x0", "x1", "y0", "y1")}

    elif scenario == "bottom_cooled_sides_insulated":
        # y0 = "bottom" strongly cooled (near heat-sink attach edge)
        boundary = {
            "y0": conv(h_lo=200.0, h_hi=C.H_EDGE_RANGE[1] * 5),
            "y1": insulated(), "x0": insulated(), "x1": insulated(),
        }

    elif scenario == "top_cooled_sides_insulated":
        boundary = {
            "y1": conv(h_lo=200.0, h_hi=C.H_EDGE_RANGE[1] * 5),
            "y0": insulated(), "x0": insulated(), "x1": insulated(),
        }

    elif scenario == "asymmetric_cooling":
        edges = ["x0", "x1", "y0", "y1"]
        rng.shuffle(edges)
        boundary = {}
        boundary[edges[0]] = conv(h_lo=150.0, h_hi=C.H_EDGE_RANGE[1] * 4)
        boundary[edges[1]] = conv()
        boundary[edges[2]] = insulated()
        boundary[edges[3]] = insulated()

    elif scenario == "mixed_dirichlet_convective":
        edges = ["x0", "x1", "y0", "y1"]
        rng.shuffle(edges)
        boundary = {
            edges[0]: dirichlet(),
            edges[1]: conv(),
            edges[2]: conv(),
            edges[3]: insulated(),
        }

    elif scenario == "all_insulated_sink_only":
        boundary = {k: insulated() for k in ("x0", "x1", "y0", "y1")}

    else:
        boundary = {k: conv() for k in ("x0", "x1", "y0", "y1")}

    if difficulty == "very_challenging":
        # push h_sink to the weaker end occasionally -> larger, harder gradients
        if rng.random() < 0.4:
            h_sink = rng.uniform(C.H_SINK_RANGE[0], C.H_SINK_RANGE[0] * 3)

    meta = {
        "scenario": scenario,
        "h_sink": float(h_sink),
        "d_chip": float(C.D_CHIP),
        "T_ambient": float(T_ambient),
    }
    return boundary, meta
