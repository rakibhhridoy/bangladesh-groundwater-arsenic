"""
Configuration for Hypothesis 4: ArseniCFM Foundation Model
Multi-modal (satellite + geochemical) arsenic prediction with Alpha Earth embeddings.
"""

import os
from pathlib import Path

# ─── PATHS ────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent
DATA_DIR       = PROJECT_ROOT / 'data'
HYPO1_OUTPUT   = PROJECT_ROOT / 'hypo1' / 'output'
HYPO2_OUTPUT   = PROJECT_ROOT / 'hypo2' / 'output'
HYPO3_OUTPUT   = PROJECT_ROOT / 'hypo3' / 'output'
OUTPUT_DIR     = PROJECT_ROOT / 'hypo4' / 'output'
FIGURES_DIR    = OUTPUT_DIR / 'figures'
TABLES_DIR     = OUTPUT_DIR / 'tables'
MAPS_DIR       = OUTPUT_DIR / 'maps'
EMBEDDINGS_DIR = OUTPUT_DIR / 'embeddings'
MODELS_DIR     = OUTPUT_DIR / 'models'
LOGS_DIR       = OUTPUT_DIR / 'logs'
DRAFT_DIR      = PROJECT_ROOT / 'hypo4' / 'draft'

for d in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, MAPS_DIR,
          EMBEDDINGS_DIR, MODELS_DIR, LOGS_DIR, DRAFT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Sentinel-2 imagery
SENTINEL_DIR   = PROJECT_ROOT / 'EarthEngine'
SENTINEL_BANDS = ['WVP', 'B8A', 'B8', 'B5', 'B2', 'B3', 'B4']
SENTINEL_SCALE = 10000.0   # divide raw DN → reflectance [0,1]
N_SENTINEL     = 1808      # total TIF files

# Source datasets
INTEGRATED_FILE = HYPO1_OUTPUT / 'data_phase1_integrated.csv'
SAT_TABLE       = HYPO1_OUTPUT / 'tables' / 'Table_12_satellite_features.csv'

# ─── ALPHA EARTH ──────────────────────────────────────────────────────────────
# Set environment variable ALPHA_EARTH_API_KEY before running if available
ALPHA_EARTH_API_KEY  = os.environ.get('ALPHA_EARTH_API_KEY', '')
ALPHA_EARTH_BASE_URL = 'https://api.alpha-earth.com/v1'
ALPHA_EARTH_EMBED_DIM = 64   # embedding dimension produced by Alpha Earth

# ─── MODEL ARCHITECTURE ───────────────────────────────────────────────────────
SAT_EMBED_DIM  = 64    # Alpha Earth or statistical fallback
GEO_EMBED_DIM  = 32    # geochemical encoder output
CTX_EMBED_DIM  = 16    # context encoder output (lat/lon/depth/season)
FUSED_DIM      = SAT_EMBED_DIM + GEO_EMBED_DIM + CTX_EMBED_DIM  # 112
HIDDEN_DIM     = 256
ATTN_HEADS     = 4
DROPOUT        = 0.2

# ─── TRAINING ─────────────────────────────────────────────────────────────────
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-3
N_EPOCHS       = 100
PATIENCE       = 15
WEIGHT_DECAY   = 1e-4
CV_FOLDS       = 5
SPATIAL_BLOCK_DEG = 1.0   # degrees for GroupKFold block CV

# Multi-task loss weights (sum = 1.0 scale)
TASK_WEIGHTS = {
    'concentration': 1.0,   # As µg/L regression (main task)
    'exceedance':    0.5,    # P(As > 10 µg/L)
    'mechanism':     0.3,    # reductive / desorption / mixed
    'depth_zone':    0.2,    # Shallow / Intermediate / Medium_Deep / Deep
}

# ─── CLASSIFICATION LABELS ────────────────────────────────────────────────────
DEPTH_ZONES = {
    'Shallow':      (0,   30),
    'Intermediate': (30,  80),
    'Medium_Deep':  (80,  150),
    'Deep':         (150, 500),
}
DEPTH_ZONE_LABELS = list(DEPTH_ZONES.keys())
DEPTH_ZONE_TO_IDX = {z: i for i, z in enumerate(DEPTH_ZONE_LABELS)}

MECHANISM_LABELS = ['Reductive', 'Desorption', 'Mixed']
MECHANISM_TO_IDX = {m: i for i, m in enumerate(MECHANISM_LABELS)}

# ─── GEOCHEMICAL FEATURE COLUMNS ──────────────────────────────────────────────
GEO_FEATURE_COLS = [
    'pH', 'EC', 'TDS', 'ORP', 'DO',
    'HCO3-', 'Cl-', 'SO42-', 'Na+', 'K+', 'Ca2+', 'Mg2+',
    'Fe2+', 'Mn2+', 'PO43-', 'NO3-', 'Al3+', 'B+',
]
CONTEXT_COLS = ['Latitude', 'Longitude', 'Depth', 'Season_bin']

# Satellite index columns (pre-computed in Table_12)
SAT_INDEX_COLS = ['NDVI_mean', 'EVI_mean', 'NDWI_mean', 'MNDWI_mean', 'NDMI_mean']

# Season
SEASON_DRY = 'Dry'
SEASON_WET = 'Wet'

# ─── PHYSIOGRAPHIC ZONES ─────────────────────────────────────────────────────
PHYSIOGRAPHIC_ZONES = {
    'Barind_Tract':           {'lat': (23.8, 25.6), 'lon': (88.0, 89.2)},
    'Northern_Terrace':       {'lat': (25.2, 26.7), 'lon': (88.0, 89.8)},
    'Brahmaputra_Floodplain': {'lat': (23.5, 25.5), 'lon': (89.5, 91.2)},
    'Ganges_Floodplain':      {'lat': (23.0, 24.5), 'lon': (88.5, 90.2)},
    'GBM_Delta':              {'lat': (21.5, 23.5), 'lon': (89.0, 90.5)},
    'Meghna_Floodplain':      {'lat': (22.5, 24.5), 'lon': (90.5, 91.5)},
    'Eastern_Hills':          {'lat': (21.5, 24.8), 'lon': (91.5, 92.7)},
}
PHYS_ZONE_LABELS = list(PHYSIOGRAPHIC_ZONES.keys())

# ─── EMBEDDING ANALYSIS ───────────────────────────────────────────────────────
UMAP_N_COMPONENTS = 2
UMAP_N_NEIGHBORS  = 15
UMAP_MIN_DIST     = 0.1
N_CLUSTERS        = 6   # KMeans on embeddings

# ─── FEW-SHOT EXPERIMENTS ─────────────────────────────────────────────────────
FEW_SHOT_K = [5, 10, 20, 50, 100]

# ─── RISK MAPPING ─────────────────────────────────────────────────────────────
BANGLADESH_BBOX = {
    'lat_min': 20.5, 'lat_max': 26.8,
    'lon_min': 87.8, 'lon_max': 92.8,
}
RISK_GRID_DEG = 0.1   # ~10km grid for national risk map
KNN_NEIGHBORS = 15

# ─── COMPUTE ──────────────────────────────────────────────────────────────────
N_JOBS       = 2   # NEVER -1: Mac multiprocessing OOM
RANDOM_STATE = 42
DPI          = 300
WHO_AS_LIMIT = 10.0

print("=" * 65)
print("HYPOTHESIS 4 — ArseniCFM Foundation Model")
print("=" * 65)
print(f"  Alpha Earth key  : {'SET' if ALPHA_EARTH_API_KEY else 'NOT SET (fallback mode)'}")
print(f"  Sentinel TIFs    : {SENTINEL_DIR}")
print(f"  Embed dims       : sat={SAT_EMBED_DIM} geo={GEO_EMBED_DIM} ctx={CTX_EMBED_DIM} → fused={FUSED_DIM}")
print(f"  Output dir       : {OUTPUT_DIR}")
print("=" * 65)
