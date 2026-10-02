"""
config.py
=========
Central configuration for the Physics-Guided Heat Dissipation dataset generator.

Every tunable constant used anywhere in the pipeline (grid size, sample counts,
material properties, boundary condition ranges, solver tolerances, ...) lives
here so the whole dataset can be regenerated at a different resolution / size
by editing a single file.
"""

import os

# --------------------------------------------------------------------------
# 0. PATHS
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
VIS_DIR = os.path.join(DATASET_DIR, "visualizations")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "dataset_generator", "checkpoints")
TEST_DIR = os.path.join(PROJECT_ROOT, "dataset_generator", "test_run")

for d in (DATASET_DIR, VIS_DIR, CHECKPOINT_DIR, TEST_DIR):
    os.makedirs(d, exist_ok=True)

# --------------------------------------------------------------------------
# 1. DATASET SIZE / SPLITS  (configurable: 5000 / 8000 / 10000 ...)
# --------------------------------------------------------------------------
TOTAL_SAMPLES = 8000
TRAIN_FRACTION = 0.80
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10

N_TRAIN = int(round(TOTAL_SAMPLES * TRAIN_FRACTION))   # 6400
N_VAL = int(round(TOTAL_SAMPLES * VAL_FRACTION))        # 800
N_TEST = TOTAL_SAMPLES - N_TRAIN - N_VAL                # 800

# Disjoint random-seed blocks per split -> no duplicate scenarios, and the
# test split draws from an entirely separate seed / geometry-parameter block
# so it exercises out-of-distribution floorplans & power configurations.
SEED_BLOCK = {
    "train": 0,
    "val": 2_000_000,
    "test": 5_000_000,
}

# --------------------------------------------------------------------------
# 2. GRID / DOMAIN  (configurable: change GRID_SIZE to 64 for higher res)
# --------------------------------------------------------------------------
GRID_SIZE = 32                       # NxN grid
DOMAIN_LENGTH_M = 0.01               # 10 mm x 10 mm physical chip
DX = DOMAIN_LENGTH_M / GRID_SIZE     # ~3.125e-4 m per cell
DY = DX

# --------------------------------------------------------------------------
# 3. MATERIAL PROPERTIES
# --------------------------------------------------------------------------
# material id -> name
MATERIAL_NAMES = {0: "dielectric", 1: "silicon", 2: "copper"}

# Nominal / reference thermal conductivity range at T_ref=300K [W/(m.K)]
MATERIAL_K_RANGE = {
    0: (1.5, 2.5),      # dielectric
    1: (120.0, 170.0),  # silicon
    2: (350.0, 450.0),  # copper
}
MATERIAL_K_NOMINAL = {0: 2.0, 1: 148.0, 2: 400.0}

# Temperature-dependence exponent n :  k(T) = k_ref * (T/T_ref)^(-n)
MATERIAL_N_RANGE = {
    0: (0.0, 0.5),
    1: (1.0, 1.5),
    2: (0.1, 0.4),
}

T_REF = 300.0  # [K] reference temperature for k(T) and Kirchhoff transform

# --------------------------------------------------------------------------
# 4. ANISOTROPY
# --------------------------------------------------------------------------
ANISOTROPY_STD_RANGE = (0.5, 2.0)      # kx/ky ratio, standard cases
ANISOTROPY_STRONG_RANGE = (0.3, 3.0)   # occasional stronger anisotropy
ANISOTROPY_STRONG_PROB = 0.15          # fraction of samples using the strong range

# --------------------------------------------------------------------------
# 5. POWER MAP
# --------------------------------------------------------------------------
BACKGROUND_POWER_RANGE_W_CM2 = (0.1, 2.0)   # W/cm^2 -> converted to W/m^2 internally
HOTSPOT_COUNT_RANGE = (1, 8)
HOTSPOT_PEAK_RANGE_W_CM2 = (5.0, 80.0)      # W/cm^2 peak added power at hotspot center
ACTIVE_BLOCK_POWER_RANGE_W_CM2 = (2.0, 25.0)

POWER_PATTERN_TYPES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

W_CM2_TO_W_M2 = 1.0e4  # 1 W/cm^2 = 1e4 W/m^2

# --------------------------------------------------------------------------
# 6. OUT-OF-PLANE / PACKAGE MODEL
# --------------------------------------------------------------------------
D_CHIP = 5.0e-4                # [m] effective chip thickness (0.5 mm)
H_SINK_RANGE = (100.0, 5000.0)  # [W/(m^2.K)] effective heat-sink convective coeff.
H_EDGE_RANGE = (5.0, 60.0)      # [W/(m^2.K)] weak lateral/edge convection
T_AMBIENT_NOMINAL = 300.0       # [K]
T_AMBIENT_JITTER = 5.0          # +/- K random jitter around nominal

BOUNDARY_SCENARIOS = [
    "all_edges_convective",
    "bottom_cooled_sides_insulated",
    "top_cooled_sides_insulated",
    "asymmetric_cooling",
    "mixed_dirichlet_convective",
    "all_insulated_sink_only",
]

# --------------------------------------------------------------------------
# 7. SOLVER
# --------------------------------------------------------------------------
SOLVER_MAX_PICARD_ITER = 200
SOLVER_TOL_K = 1e-5           # [K] convergence tolerance on max |dT|
SOLVER_RELAXATION = 0.8       # Picard under-relaxation factor
SOLVER_RELAXATION_MIN = 0.15
T_CLIP_RATIO = (0.4, 3.0)     # clip (T/T_ref) before applying exponent -n for stability

# --------------------------------------------------------------------------
# 8. VALIDATION / REJECTION RULES
# --------------------------------------------------------------------------
REJECT_MIN_DELTA_T = 0.1     # [K]
REJECT_MAX_T = 500.0         # [K]
REJECT_MAX_RESIDUAL = 1e-2   # dimensionless-scaled PDE residual tolerance
REJECT_MAX_HEAT_BALANCE_PCT = 1.0  # [%]
KIRCHHOFF_ROUNDTRIP_TOL = 1e-5     # [K]

# --------------------------------------------------------------------------
# 9. DIFFICULTY DISTRIBUTION
# --------------------------------------------------------------------------
DIFFICULTY_WEIGHTS = {
    "normal": 0.60,
    "challenging": 0.25,
    "very_challenging": 0.15,
}

# --------------------------------------------------------------------------
# 10. MATERIAL COMPOSITION TARGETS
# --------------------------------------------------------------------------
SI_DIELECTRIC_FRACTION_RANGE = (0.60, 0.80)  # combined Si+Dielectric fraction (typical)
CHALLENGING_COPPER_FRACTION_RANGE = (0.30, 0.55)  # large-copper challenging samples

# --------------------------------------------------------------------------
# 11. BATCHING / RESUMABILITY
# --------------------------------------------------------------------------
BATCH_SIZE = 500          # samples per checkpoint batch
MAX_ATTEMPT_MULTIPLIER = 3  # safety cap: max attempts = target * this, per split

# --------------------------------------------------------------------------
# 12. VISUALIZATION
# --------------------------------------------------------------------------
N_VISUALIZATION_SAMPLES = 36
