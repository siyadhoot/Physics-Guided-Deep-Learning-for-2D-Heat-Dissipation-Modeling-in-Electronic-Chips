# Physics-Guided 2D Heat Dissipation Dataset for Electronic Chips

This dataset was generated end-to-end (floorplans → power maps → non-linear
anisotropic FV heat solver → Kirchhoff transform → validation → HDF5) by the
scripts in `../dataset_generator/`. It supports the research workflow:

```
Power Map Q(x,y) + Material/Floorplan k(x,y)
        │
        ▼
Kirchhoff Temperature Transformation θ(x,y)
        │
        ▼
Anisotropic Wavelet-Fourier Neural Operator   (Input → θ, trained separately)
        │
        ▼
Hard-Constrained Boundary Envelope
        │
        ▼
Inverse Kirchhoff Transformation   θ → T
        │
        ▼
Predicted Temperature Map T(x,y)
```

The ground truth `T(x,y)` is **not** heuristic Gaussian smoothing — it is the
converged numerical solution of the non-linear, anisotropic, steady-state
heat-conduction PDE described below, for every one of the 8,000 samples.

---

## 1. Governing physics

Steady-state non-linear anisotropic heat equation with an out-of-plane
(vertical) lumped package loss term:

```
∂/∂x( kx(T,x,y) ∂T/∂x ) + ∂/∂y( ky(T,x,y) ∂T/∂y )
      − (h_sink / d_chip) · (T − T_ambient) + Q_vol(x,y) = 0
```

* `kx(T,x,y) = kx_base(x,y) · (T/T_ref)^(−n(x,y))`, similarly for `ky`
  (temperature-dependent, anisotropic, per-material conductivity).
* `kx_base = k_ref(material) · a_x(x,y)`, `ky_base = k_ref(material) · a_y(x,y)`
  — anisotropy ratios `a_x, a_y` are spatially correlated with material
  regions (not pixel noise), standard range `kx/ky ∈ [0.5, 2.0]`, with
  occasional stronger anisotropy `[0.3, 3.0]`.
* `Q_vol = Q_areal / d_chip` — the areal chip-surface power density
  `Q_areal(x,y)` [W/m²] is converted to an equivalent volumetric source using
  the lumped package thickness `d_chip`, so every term in the PDE is
  expressed consistently in **W/m³**.
* `h_sink / d_chip · (T − T_ambient)` — effective out-of-plane heat-sink loss
  (2D-equivalent package model), `h_sink ∈ [100, 5000] W/(m².K)`,
  `d_chip = 0.5 mm`.

### Boundary conditions (per edge: x=0, x=1, y=0, y=1)

| Type | Equation |
|---|---|
| Dirichlet | `T = T_boundary` |
| Robin / Convective | `−k ∂T/∂n = h_edge (T − T_ambient)` |
| Neumann (insulated) | `∂T/∂n = 0` |

Six boundary scenarios are sampled per sample: all-edges-convective,
bottom-cooled/sides-insulated, top-cooled/sides-insulated, asymmetric
cooling, mixed Dirichlet+convective, and all-insulated (sink-only).

---

## 2. Numerical solver

Implemented in `dataset_generator/solver.py`:

* **Finite-volume discretization**, cell-centered, `dx = dy` (32×32 grid over
  a 10 mm × 10 mm domain ⇒ `dx ≈ 3.125e-4 m`).
* **Harmonic-mean interface conductivity** (critical for sharp material
  interfaces, e.g. 400 vs 1.5 W/m·K):
  `k_{i+1/2,j} = 2 k_{i,j} k_{i+1,j} / (k_{i,j} + k_{i+1,j})`.
* **Robin boundary treatment** via a resistance-in-series lumped coefficient
  `h_eff = 1 / ( (dx/2)/k_cell + 1/h_edge )`, robust for any `h_edge`.
* **Dirichlet boundary treatment** via a half-cell-distance ghost node
  (`coefficient = 2·k_cell`).
* **Non-linear Picard fixed-point iteration**: `T⁽⁰⁾ = T_ambient`; at each
  iteration `k(T)` is refreshed and the resulting *linear* system is solved
  exactly with `scipy.sparse.linalg.spsolve`, with adaptive under-relaxation
  for stability. Converged when `max|T⁽ᵐ⁺¹⁾ − T⁽ᵐ⁾| < 1e-5 K`
  (typically 9-15 iterations).

### Validation & rejection (every sample)

Recorded in `metadata.csv` / HDF5 `metadata/` group and enforced at
generation time:

* PDE residual `R = ∇·(k∇T) − (h_sink/d_chip)(T−T_amb) + Q` (mean & max,
  reject if max > 1e-2 after normalization by source scale).
* Global energy balance: `∫Q dA` vs. heat leaving via the vertical sink +
  edges (reject if error > 1%). **Achieved: worst-case ≈ 5×10⁻⁵ %.**
* Reject on solver non-convergence, NaN/Inf, `ΔT < 0.1 K`, `T_max > 500 K`,
  or a degenerate (single-material) floorplan.
* Kirchhoff round-trip `T → θ → ψ⁻¹(θ)` verified analytically in float64 at
  generation time; must be `< 1e-5 K` (**achieved: worst-case ≈ 3×10⁻¹¹ K**).

8,443 attempts were needed to produce 8,000 valid training-block samples
(≈5-9% rejection rate per split, all rejections were `T_max_unphysical`,
i.e. extreme power/weak-sink combinations exceeding 500 K) — see
`dataset_statistics.json` for the full breakdown.

---

## 3. Kirchhoff transformation

For a material with `k(T) = k_ref·(T/T_ref)^(−n)`:

```
θ(T) = ∫[T_ref→T] k(τ)/k_ref dτ
     = T_ref/(1−n) · [ (T/T_ref)^(1−n) − 1 ]     (n ≠ 1)
     = T_ref · ln(T/T_ref)                        (n = 1, |n-1|<1e-3)
```

`θ` depends only on local `T`, the material exponent `n`, and `T_ref` — **not**
on the magnitude of `k_ref` — so it is evaluated per-cell using each cell's
local material exponent `n(x,y)` (as required by the workflow: "θ evaluated
relative to the material's isotropic base component", with `a_x, a_y`
preserved as separate spatial channels).

### Inverse transform

```
T(θ) = T_ref · [ 1 + (1−n)θ/T_ref ]^(1/(1−n))     (n ≠ 1)
T(θ) = T_ref · exp(θ/T_ref)                        (n = 1)
```

**Important for downstream model use:** `n(x,y)` is randomized *per sample*
within each material's configured range (see §5) and is **not** stored as a
raster channel. To apply the inverse transform to a network's *predicted*
`θ(x,y)`, reconstruct `n_map` from the `material` channel and the per-sample
scalars stored in metadata:

```python
n_map = np.zeros_like(material, dtype=float)
n_map[material == 0] = meta["n_dielectric"]
n_map[material == 1] = meta["n_silicon"]
n_map[material == 2] = meta["n_copper"]
T = inverse_kirchhoff(theta_pred, n_map, T_ref=meta["T_ref_K"])
```

(`inverse_kirchhoff` is provided in `dataset_generator/kirchhoff.py`.)

---

## 4. Physical units reference

| Quantity | Symbol | Units |
|---|---|---|
| Domain size | `Lx, Ly` | 0.01 m (10 mm) |
| Grid | `N × N` | 32 × 32 (configurable) |
| Cell size | `dx = dy` | ≈ 3.125×10⁻⁴ m |
| Power density (input) | `Q(x,y)` | W/m² (areal, at chip surface) |
| Volumetric source (internal) | `Q_vol = Q/d_chip` | W/m³ |
| Thermal conductivity | `kx, ky` | W/(m·K) |
| Anisotropy ratio | `ax, ay` | dimensionless, `= kx/k_ref`, `ky/k_ref` |
| Temperature | `T` | K |
| Kirchhoff potential | `θ` | W/m (a temperature-like scalar; see §3) |
| Chip effective thickness | `d_chip` | 0.5 mm |
| Heat-sink coefficient | `h_sink` | 100–5000 W/(m²·K) |
| Edge convection coefficient | `h_edge` | 5–300 W/(m²·K) |
| Ambient temperature | `T_ambient` | ≈300 K ± 5 K jitter |
| Reference temperature | `T_ref` | 300 K |

---

## 5. Materials

| ID | Material | `k_ref` range [W/m·K] | `n` exponent range |
|---|---|---|---|
| 0 | Dielectric | 1.5 – 2.5 | 0.0 – 0.5 |
| 1 | Silicon | 120 – 170 (nominal 148) | 1.0 – 1.5 |
| 2 | Copper | 350 – 450 (nominal 400) | 0.1 – 0.4 |

Floorplans (`dataset_generator/materials.py`) are built from geometric
primitives — rectangles, interconnect strips, memory-array-like alternating
blocks, and irregular connected blobs — with **per-sample target area
fractions** for copper/dielectric (varying by difficulty level) so that the
dataset-level composition matches the "generally 60-80% Si+Dielectric, with
some large-copper challenging samples" rule (achieved training-set mean
combined Si+Dielectric fraction ≈ 75%, with challenging/very_challenging
samples reaching up to ~74% copper). Sharp material interfaces are always
preserved (no smoothing of the material map itself).

---

## 6. Power maps

Background distributed power (0.1–2.0 W/cm²) + 1-10 Gaussian hotspots
(rotated, anisotropic widths) + optional rectangular "active block" regions,
following pattern types **A–J** (single central, multiple distributed,
corner, edge, narrow multi-hotspot, large diffuse, adjacent pair, hotspot at
a material interface, hotspot crossing a boundary, asymmetric workload).
`challenging` / `very_challenging` samples additionally inject hotspots
deliberately positioned at detected material-interface pixels.

---

## 7. Difficulty distribution (achieved)

| Level | Target | Achieved |
|---|---|---|
| Normal | 60% | 61.5% |
| Challenging | 25% | 25.3% |
| Very challenging | 15% | 13.2% |

("Very challenging" additionally pushes silicon `n` up to 1.3-1.5, uses
larger copper structures, weaker `h_sink`, and interface-crossing
micro-hotspots.)

---

## 8. Dataset size & splits

| Split | Samples | Seed block |
|---|---|---|
| Train | 6,400 | `[0, 2,000,000)` |
| Validation | 800 | `[2,000,000, 5,000,000)` |
| Test | 800 | `[5,000,000, ∞)` |

Splits use **disjoint random-seed blocks** — every floorplan, power map,
anisotropy pattern, and boundary condition combination is independently
sampled per split, verified to have **zero duplicate samples within or
across splits** (MD5 hash of `Q`+`material` arrays). Total samples and the
train/val/test fractions are configurable in `dataset_generator/config.py`
(`TOTAL_SAMPLES = 8000` by default; set to 5000/10000/etc. and re-run).

---

## 9. File hierarchy

```
dataset/
├── train.h5                 # 6400 samples
├── validation.h5             # 800 samples
├── test.h5                   # 800 samples
├── metadata.csv               # 8000 rows, all splits combined
├── normalization_stats.json  # per-channel mean/std/min/max (TRAIN split only)
├── dataset_statistics.json   # aggregate QC summary
├── README.md                  # this file
└── visualizations/
    ├── sample_0000_train.png
    ├── sample_0001_train.png
    └── ... (36 total, spanning train/val/test and all difficulty levels)
```

### HDF5 internal structure (`train.h5` / `validation.h5` / `test.h5`)

```
inputs/Q          (N, 32, 32)  float32   W/m²   areal power density
inputs/material   (N, 32, 32)  int8      {0,1,2} = {dielectric, silicon, copper}
inputs/kx         (N, 32, 32)  float32   W/(m.K) at converged T
inputs/ky         (N, 32, 32)  float32   W/(m.K) at converged T
inputs/ax         (N, 32, 32)  float32   kx / k_ref  (anisotropy ratio)
inputs/ay         (N, 32, 32)  float32   ky / k_ref  (anisotropy ratio)
targets/theta     (N, 32, 32)  float32   Kirchhoff-transformed field
targets/T         (N, 32, 32)  float32   K, ground-truth temperature
metadata/*        (N,)         numeric or UTF-8 string datasets, e.g.:
    sample_id, seed, grid_size, total_power_W, max_power_W_m2, num_hotspots,
    hotspot_type, boundary_type, difficulty_level, h_sink, T_ambient,
    T_min, T_max, T_mean, delta_T, mean_theta,
    n_dielectric, n_silicon, n_copper,           # per-sample material exponents
    k_ref_dielectric, k_ref_silicon, k_ref_copper, T_ref_K,
    solver_iterations, pde_residual_mean, pde_residual_max,
    heat_balance_error_pct, kirchhoff_roundtrip_error_K,
    silicon_fraction, copper_fraction, dielectric_fraction,
    mean_kx, mean_ky, mean_anisotropy
```

File-level attrs: `grid_size`, `domain_length_m`, `dx_m`, `n_samples`,
`split`, `T_ref_K`, `d_chip_m`.

**Data is stored in raw physical units — not normalized.** Use
`normalization_stats.json` (computed on the training split only) to
normalize at load time.

---

## 10. Loading the dataset (PyTorch example)

```python
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ChipThermalDataset(Dataset):
    def __init__(self, h5_path, norm_stats=None):
        self.h5_path = h5_path
        self.norm_stats = norm_stats
        with h5py.File(h5_path, "r") as f:
            self.n = f["inputs/Q"].shape[0]
        self._h5 = None  # opened lazily per-worker for multiprocessing safety

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        self._ensure_open()
        f = self._h5
        Q = f["inputs/Q"][idx]
        material = f["inputs/material"][idx].astype(np.float32)
        kx = f["inputs/kx"][idx]
        ky = f["inputs/ky"][idx]
        ax = f["inputs/ax"][idx]
        ay = f["inputs/ay"][idx]
        theta = f["targets/theta"][idx]
        T = f["targets/T"][idx]

        x = np.stack([Q, material, kx, ky, ax, ay], axis=0)  # (6, 32, 32)
        y = np.stack([theta, T], axis=0)                      # (2, 32, 32)
        return torch.from_numpy(x), torch.from_numpy(y)

train_ds = ChipThermalDataset("dataset/train.h5")
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)

x, y = next(iter(train_loader))
print(x.shape, y.shape)  # torch.Size([32, 6, 32, 32]) torch.Size([32, 2, 32, 32])
```

Arrays are plain NumPy-backed HDF5 datasets (channel-first `(N,H,W)`), so
they load equally well into a TensorFlow `tf.data.Dataset` via
`h5py` + `tf.data.Dataset.from_generator`, and are architecture-agnostic
(usable for CNNs, U-Nets, standard FNOs, or Wavelet-Fourier Neural
Operators).

---

## 11. Regenerating / resizing the dataset

Everything is driven by `dataset_generator/config.py`:

* `TOTAL_SAMPLES` (default 8000) — change to 5000 / 10000 / etc.
* `GRID_SIZE` (default 32) — change to 64 for higher resolution (re-run the
  20-sample gate in `run_test_batch.py` first to confirm solver stability at
  the new resolution before committing to a full run).
* All material, power, boundary, solver, and validation constants.

Pipeline (resumable — checkpoints are written to `dataset_generator/checkpoints/`
after every batch of `BATCH_SIZE` valid samples, so an interrupted run simply
picks up where it left off):

```bash
cd dataset_generator
python run_test_batch.py       # Step 1: 20-sample physics/validation gate
python generate_dataset.py     # Step 2: full N-sample generation (train/val/test)
python finalize_dataset.py     # Assemble shards -> HDF5 + metadata + stats
python visualize.py            # Multi-panel inspection PNGs
python verify_load.py          # Automated load / shape / dtype / NaN test
```

---

## 12. Achieved quality summary (this run)

From `dataset_statistics.json`:

* **8,000 / 8,000** valid samples (6,400 / 800 / 800), zero duplicates.
* `T` range across the whole dataset: **295.0 – 499.8 K** (all `< 500 K`
  reject threshold, all `ΔT ≥ 0.1 K`).
* PDE residual (normalized by local source scale): mean-of-means
  `5.5×10⁻⁹`, worst-case max `1.3×10⁻⁶` (threshold: `1×10⁻²`).
* Global heat-balance error: mean `2.7×10⁻⁶ %`, worst-case `5.3×10⁻⁵ %`
  (threshold: `1 %`).
* Kirchhoff round-trip error (float64, generation time): mean `2.8×10⁻¹³ K`,
  worst-case `2.8×10⁻¹¹ K` (threshold: `1×10⁻⁵ K`).
* Total rejected candidate samples: 519 (all `T_max_unphysical`, i.e.
  extreme power/weak-sink combinations exceeding 500 K) out of 8,519
  attempts across all three splits.
