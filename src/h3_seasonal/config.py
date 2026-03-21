"""
Configuration for Hypothesis 3 analysis
Seasonal Hydro-Climatic Forcing and Depth-Modulated Arsenic Mobilization
"""

import os
from pathlib import Path

# ─── PATHS ────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).parent.parent
DATA_DIR        = PROJECT_ROOT / 'data'
HYPO1_OUTPUT    = PROJECT_ROOT / 'hypo1' / 'output'
HYPO2_OUTPUT    = PROJECT_ROOT / 'hypo2' / 'output'
OUTPUT_DIR      = PROJECT_ROOT / 'hypo3' / 'output'
FIGURES_DIR     = OUTPUT_DIR / 'figures'
TABLES_DIR      = OUTPUT_DIR / 'tables'
MAPS_DIR        = OUTPUT_DIR / 'maps'
RESULTS_DIR     = OUTPUT_DIR / 'results'

for d in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, MAPS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

INTEGRATED_FILE = HYPO1_OUTPUT / 'data_phase1_integrated.csv'
WEATHER_FILE    = DATA_DIR / 'weatherData.csv'

# ─── TARGET & CONSTANTS ───────────────────────────────────────────────────────
TARGET       = 'As'
WHO_AS_LIMIT = 10.0    # µg/L
ALPHA        = 0.05
RANDOM_STATE = 42
DPI          = 300

# ─── SEASON LABELS ────────────────────────────────────────────────────────────
SEASON_DRY = 'Dry'
SEASON_WET = 'Wet'
SEASON_LABELS = [SEASON_DRY, SEASON_WET]
SEASON_COLORS = {SEASON_DRY: '#d73027', SEASON_WET: '#4575b4'}

# ─── DEPTH ZONES (inherited from H2) ─────────────────────────────────────────
DEPTH_ZONES = {
    'Shallow':      (0,   30),
    'Intermediate': (30,  80),
    'Medium_Deep':  (80,  150),
    'Deep':         (150, 500),
}
DEPTH_ZONE_LABELS = list(DEPTH_ZONES.keys())
ZONE_COLORS = {
    'Shallow':      '#d73027',
    'Intermediate': '#fc8d59',
    'Medium_Deep':  '#4575b4',
    'Deep':         '#313695',
}

# ─── PHYSIOGRAPHIC ZONES (inherited from H2) ──────────────────────────────────
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

# ─── VARIABLE GROUPS ──────────────────────────────────────────────────────────
# Ions and chemistry
IONS = ['Na+', 'K+', 'Mg2+', 'Ca2+', 'NO3-', 'SO42-', 'PO43-',
        'HCO3-', 'Cl-', 'Fe2+', 'Mn2+', 'Al3+', 'B+', 'Cr3+']

REDOX_VARS  = ['ORP', 'DO', 'Fe2+', 'Mn2+', 'SO42-', 'HCO3-']
SATELLITE_VARS = ['NDVI_mean', 'EVI_mean', 'NDWI_mean', 'MNDWI_mean', 'NDMI_mean']
WEATHER_VARS   = ['swvl1', 't2m', 'tcc', 'lai_hv']  # from ERA5; added by data_loader

# Variables requiring log-transform
LOG_TRANSFORM_VARS = ['As', 'Fe2+', 'Mn2+', 'Al3+', 'PO43-', 'NO3-']

# DOC proxies (no direct DOC measurement)
DOC_PROXY_VARS = ['TDS', 'EC']

# Molar masses for stoichiometry (g/mol)
MW_As  = 74.922
MW_PO4 = 94.971
MW_Cl  = 35.453

# ─── SEM PATHWAY VARIABLES ────────────────────────────────────────────────────
SEM_DOC_PROXY  = 'TDS'   # mediator 1 in reductive pathway
SEM_REDOX_MED  = 'ORP'   # mediator 2
SEM_FE_MED     = 'Fe2+'  # mediator 3
SEM_PO4_MED    = 'PO43-' # mediator in desorption pathway
SEM_CONC_MED   = 'Cl-'   # mediator in concentration pathway

# ─── SPATIAL ─────────────────────────────────────────────────────────────────
SPATIAL_BLOCK_SIZE = 1.0   # degrees, for spatial block CV
KNN_NEIGHBORS      = 8     # for Moran's I weight matrix
KRIGE_MAX_N        = 400   # max training points per kriging fit
KRIGE_GRID_SIZE    = 50    # grid resolution for risk maps
BANGLADESH_BBOX    = {'lat_min': 20.5, 'lat_max': 26.8,
                      'lon_min': 87.8, 'lon_max': 92.8}

# ─── STATISTICAL ──────────────────────────────────────────────────────────────
N_BOOTSTRAP    = 500   # bootstrap iterations (safe for Mac)
N_PERMUTATIONS = 499
CV_FOLDS       = 5

# ─── ML ───────────────────────────────────────────────────────────────────────
RF_N_ESTIMATORS  = 300   # safe (n_jobs=2)
XGB_N_ESTIMATORS = 200
N_JOBS           = 2     # NEVER use -1: Mac multiprocessing OOM

# ─── ENTROPY ──────────────────────────────────────────────────────────────────
ENTROPY_BINS   = 20         # histogram bins for Shannon entropy
RENYI_ALPHAS   = [0.5, 1.0, 2.0]   # Rényi α values
ENTROPY_GRID   = 0.5        # degrees, for spatial entropy ratio map

# ─── STOICHIOMETRY ────────────────────────────────────────────────────────────
# Literature competitive adsorption constant range (Dixit & Hering, 2003)
LIT_RATIO_LOW  = 0.01
LIT_RATIO_HIGH = 0.15

# ─── PLOTTING ─────────────────────────────────────────────────────────────────
FIGURE_SIZE = (14, 9)
COLORMAP    = 'RdYlBu_r'

print("=" * 65)
print("HYPOTHESIS 3 CONFIGURATION")
print("=" * 65)
print("H3: Seasonal Hydro-Climatic Forcing & Arsenic Mobilization")
print(f"  Depth zones   : {DEPTH_ZONE_LABELS}")
print(f"  Phys zones    : {PHYS_ZONE_LABELS}")
print(f"  Satellite vars: {SATELLITE_VARS}")
print(f"  Weather vars  : {WEATHER_VARS}")
print(f"  N bootstrap   : {N_BOOTSTRAP}")
print(f"  Output dir    : {OUTPUT_DIR}")
print("=" * 65)
