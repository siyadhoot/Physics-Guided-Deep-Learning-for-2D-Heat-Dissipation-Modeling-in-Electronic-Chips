# ML Input Design — Leakage-Free FNO Pipeline

This document describes the corrected machine-learning input formulation for
the Physics-Guided 2D Heat Dissipation dataset. The original numerical
simulation files (`train.h5`, `validation.h5`, `test.h5`) are **preserved
unchanged** as ground truth. The ML pipeline constructs leakage-free inputs
dynamically at load time.

---

## 1. Original dataset structure

Each HDF5 split stores:

| Group | Arrays | Meaning |
|---|---|---|
| `inputs/Q` | `(N,32,32)` | Areal power density \[W/m²\] |
| `inputs/material` | `(N,32,32)` int8 | `{0,1,2}` = dielectric / silicon / copper |
| `inputs/kx`, `inputs/ky` | `(N,32,32)` | Conductivity at **converged** temperature |
| `inputs/ax`, `inputs/ay` | `(N,32,32)` | Anisotropy ratios (T-independent) |
| `targets/theta` | `(N,32,32)` | Kirchhoff potential (primary ML target) |
| `targets/T` | `(N,32,32)` | Ground-truth temperature \[K\] |
| `metadata/*` | `(N,)` | Per-sample scalars (see below) |

Relevant metadata scalars (per sample):

- `k_ref_dielectric`, `k_ref_silicon`, `k_ref_copper`
- `n_dielectric`, `n_silicon`, `n_copper` *(see § limitation on `n_silicon`)*
- `h_sink`, `T_ambient`, `T_ref_K`
- `boundary_type` (scenario name only)
- `seed` (used to reconstruct exact per-edge BCs)

File-level constants: `d_chip_m = 5e-4`, `T_ref_K = 300`, `grid_size = 32`.

---

## 2. Original FNO input design

The dataset README example stacked:

```text
X = [Q, material, kx, ky, ax, ay]     # shape (6, 32, 32)
y = [theta, T]                        # shape (2, 32, 32)
```

This feeds **converged** `kx`/`ky` into the network.

---

## 3. Why converged kx/ky cause target leakage

Physical conductivity:

```text
kx(T,x,y) = kx_base(x,y) · (T / T_ref)^(-n(x,y))
ky(T,x,y) = ky_base(x,y) · (T / T_ref)^(-n(x,y))
```

with

```text
kx_base = k_ref(material) · ax
ky_base = k_ref(material) · ay
```

The stored `inputs/kx` and `inputs/ky` are evaluated at the **converged**
temperature field `T`. Therefore they are functionally dependent on the
solution the model is asked to predict. Giving them to the FNO is target
leakage: the network can exploit information derived from `T`.

---

## 4. Corrected FNO input design

The FNO receives only quantities known **before** solving the heat equation:

- power map `Q`
- material layout
- base thermal properties (`k_ref` → `kx_base`, `ky_base`)
- anisotropy (`ax`, `ay`)
- boundary conditions (per-edge type + parameters)
- ambient / sink conditions
- temperature-dependence exponents `n_*`

It predicts `theta`; physical temperature is recovered by the inverse
Kirchhoff transform.

```text
KNOWN (pre-solve)
    Q, material, kx_base, ky_base, ax, ay, BCs, h_sink, T_ambient, n_*
        │
        ▼
       FNO
        │
        ▼
    theta_hat(x,y)
        │
        ▼
  inverse Kirchhoff (n_map from material + n_*)
        │
        ▼
    T_hat(x,y)
```

---

## 5. Exact input channels

`X` has shape `(C, 32, 32)` with `C = 23` and channel order:

| # | Name | Description |
|---|---|---|
| 0 | `Q` | Power density \[W/m²\] |
| 1 | `material` | Material ID `{0,1,2}` (categorical; not z-scored) |
| 2 | `kx_base` | `k_ref(material) * ax` |
| 3 | `ky_base` | `k_ref(material) * ay` |
| 4 | `ax` | Anisotropy ratio x |
| 5 | `ay` | Anisotropy ratio y |
| 6–9 | `bc_type_{left,right,bottom,top}` | Edge type codes on edge pixels: `0` insulated, `1` convective, `2` Dirichlet |
| 10–13 | `bc_h_{left,right,bottom,top}` | `h_edge` on convective edges; else `0` |
| 14–17 | `bc_T_{left,right,bottom,top}` | `T_boundary` (Dirichlet) or `T_ambient` (convective); else `0` |
| 18 | `h_sink` | Out-of-plane sink coefficient (broadcast) |
| 19 | `T_ambient` | Ambient temperature (broadcast) |
| 20 | `n_dielectric` | Exponent for material 0 (broadcast) |
| 21 | `n_silicon` | **Actual** exponent for material 1 (broadcast) |
| 22 | `n_copper` | Exponent for material 2 (broadcast) |

Edge orientation matches the solver: `left=x0`, `right=x1`, `bottom=y0`, `top=y1`.

---

## 6. Exact target

```text
y = theta          # shape (1, 32, 32)
```

- Primary training target: Kirchhoff potential `targets/theta`
- `targets/T` is retained for physical evaluation only and is **never** an input
- After prediction: `T_hat = inverse_kirchhoff(theta_hat, n_map, T_ref)`

---

## 7. Boundary-condition representation

### What the solver uses

Each sample has a per-edge dict (`x0,x1,y0,y1`) with type ∈
`{insulated, convective, dirichlet}` plus `h_edge` / `T_boundary` /
`T_ambient` as applicable (`dataset_generator/boundary.py`).

Six scenarios appear in the dataset:

- `all_edges_convective`
- `bottom_cooled_sides_insulated`
- `top_cooled_sides_insulated`
- `asymmetric_cooling` *(edge assignment shuffled)*
- `mixed_dirichlet_convective` *(edge assignment shuffled)*
- `all_insulated_sink_only`

### What was stored in HDF5

Only `boundary_type` (scenario name), `h_sink`, and `T_ambient`.
**Per-edge `h_edge`, edge-role assignment, and Dirichlet temperatures were
not stored.**

### How they are recovered (no solver rerun)

Each sample’s `seed` fully determines the generator RNG sequence. Replaying
`sample_gen` up through `generate_boundary_condition` (without calling the
PDE solver) reconstructs the exact boundary dict. Results are cached in:

```text
dataset/ml_aux/{train,validation,test}_bc_params.npz
```

Physical encoding (not an opaque scenario integer):

- Spatial type channels on edge pixels (`0/1/2`)
- Spatial `h` and `T` parameter channels on edge pixels
- Global `h_sink` and `T_ambient` as constant fields

---

## 8. Global conditioning variables

| Variable | Source | Encoding |
|---|---|---|
| `h_sink` | metadata / aux cache | broadcast channel |
| `T_ambient` | metadata / aux cache | broadcast channel |
| `n_dielectric` | aux (matches metadata) | broadcast channel |
| `n_silicon` | aux (**actual**, seed-corrected) | broadcast channel |
| `n_copper` | aux (matches metadata) | broadcast channel |

Constants **not** added as channels (identical for all samples):

- `T_ref = 300 K`
- `d_chip = 5e-4 m`

---

## 9. Kirchhoff transformation

Unchanged from generation (`dataset_generator/kirchhoff.py`):

```text
θ(T) = T_ref/(1−n) · [(T/T_ref)^(1−n) − 1]     (n ≠ 1)
θ(T) = T_ref · ln(T/T_ref)                        (n = 1)

T(θ) = T_ref · [1 + (1−n)θ/T_ref]^(1/(1−n))       (n ≠ 1)
T(θ) = T_ref · exp(θ/T_ref)                        (n = 1)
```

`n_map` is built from `material` and the **actual** per-sample `n_*`
(from the aux cache). Helper: `ml_data.inverse_kirchhoff_from_sample`.

### Known generator quirk (handled)

For some `very_challenging` samples, `sample_gen.py` overwrites silicon `n`
on `n_map` but does **not** update `metadata/n_silicon`. The aux cache stores
the true exponent used at generation time. Use aux `n_silicon` for inverse
Kirchhoff, not raw metadata, when they differ.

---

## 10. Normalization

File: `dataset/ml_normalization_stats.json` (**TRAIN only**, 6400 samples).

| Channel class | Strategy |
|---|---|
| Continuous spatial / global (`Q`, `kx_base`, `ky_base`, `ax`, `ay`, `bc_h_*`, `bc_T_*`, `h_sink`, `T_ambient`, `n_*`) | z-score: `(x − mean) / std` |
| Categorical (`material`, `bc_type_*`) | **not** normalized (discrete codes) |
| Target `theta` | z-score using train-only stats |

- Validation / test must never contribute to these statistics.
- Existing `normalization_stats.json` values for `Q`/`ax`/`ay`/`theta` are
  cross-checked; ML stats are recomputed for the new channel set.
- Raw HDF5 arrays remain in physical units.

---

## 11. Data split

Unchanged:

| Split | Samples | File |
|---|---|---|
| train | 6400 | `dataset/train.h5` |
| validation | 800 | `dataset/validation.h5` |
| test | 800 | `dataset/test.h5` |

No reshuffling, no regeneration of simulations.

---

## 12. Leakage checks

```bash
python -m ml_data.leakage
# or via smoke test
python -m ml_data.smoke_test
```

`validate_no_target_leakage()` verifies:

1. `T` not in `X`
2. `theta` not in `X`
3. converged `kx` not in `X`
4. converged `ky` not in `X`
5. `kx_base` from `material`, `k_ref`, `ax` only
6. `ky_base` from `material`, `k_ref`, `ay` only
7. normalization stats are TRAIN only
8. val/test not used for preprocessing stats
9. no other target-derived quantity in channel names / content

The check **raises** on failure.

---

## 13. Why this formulation is physically valid

The steady heat equation depends on:

- sources `Q`
- material-dependent base conductivity and anisotropy
- nonlinear exponent `n` (known a priori per material sample)
- edge BCs and package sink / ambient conditions

All of these are known before the solve. The Kirchhoff map converts the
nonlinear conductivity problem into a form well suited for learning
`θ(x,y)`; the inverse map recovers physical `T` without feeding `T` into
the network. Using `kx_base`/`ky_base` instead of converged `kx`/`ky`
preserves the correct pre-solve information while removing solution leakage.

---

## Variable table

| Variable | Source | FNO Input? | Reason |
|---|---|---|---|
| `Q` | HDF5 `inputs/Q` | **Yes** | Known power map |
| `material` | HDF5 `inputs/material` | **Yes** | Known floorplan |
| `ax`, `ay` | HDF5 `inputs/ax`,`ay` | **Yes** | T-independent anisotropy |
| `k_ref_*` | HDF5 metadata | Indirect | Used to build `kx_base`/`ky_base` |
| `kx_base` | `k_ref(material)*ax` | **Yes** | Pre-solve conductivity base |
| `ky_base` | `k_ref(material)*ay` | **Yes** | Pre-solve conductivity base |
| `kx`, `ky` (converged) | HDF5 `inputs/kx`,`ky` | **No** | Derived from converged `T` (leakage) |
| `theta` | HDF5 `targets/theta` | **Target only** | Primary prediction |
| `T` | HDF5 `targets/T` | **No** | Solution; eval only |
| `boundary_type` | metadata | Indirect | Scenario name; edges reconstructed |
| `bc_type_*` | seed replay → aux | **Yes** | Physical edge BC type |
| `bc_h_*` | seed replay → aux | **Yes** | Edge convection `h_edge` |
| `bc_T_*` | seed replay → aux | **Yes** | Edge Dirichlet / ambient temperature |
| `h_sink` | metadata / aux | **Yes** | Package sink BC |
| `T_ambient` | metadata / aux | **Yes** | Ambient / Robin reference |
| `n_dielectric` | metadata / aux | **Yes** | Known material nonlinearity |
| `n_silicon` | aux (actual) | **Yes** | Known nonlinearity (seed-corrected) |
| `n_copper` | metadata / aux | **Yes** | Known material nonlinearity |
| `T_ref` | constant 300 K | No (fixed) | Same for all samples |
| `d_chip` | constant 5e-4 m | No (fixed) | Same for all samples |
| `T_min/T_max/...` | metadata diagnostics | **No** | Derived from solution |
| `mean_kx/mean_ky` | metadata diagnostics | **No** | Derived from converged fields |
| `pde_residual_*` | metadata QC | **No** | Solver diagnostics |

---

## How to use

```bash
# One-time: reconstruct BCs from seeds + compute train-only ML stats
python -m ml_data.build_aux_and_stats

# Smoke test (no training)
python -m ml_data.smoke_test
```

```python
from ml_data import ChipThermalDataset, make_dataloader, validate_no_target_leakage

train_loader = make_dataloader("train", batch_size=32, shuffle=True)
X, y = next(iter(train_loader))
# X: [B, 23, 32, 32]   y: [B, 1, 32, 32]

validate_no_target_leakage()  # raises if leakage detected
```

Implementation lives in `ml_data/`. Original HDF5 files are never modified.
