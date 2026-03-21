"""
Configuration for Hypothesis 2 analysis
Depth-Stratified Geochemical Transitions and Process Partitioning of Arsenic Mobilization
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / 'data'
HYPO1_OUTPUT = PROJECT_ROOT / 'hypo1' / 'output'
OUTPUT_DIR   = PROJECT_ROOT / 'hypo2' / 'output'
FIGURES_DIR  = OUTPUT_DIR / 'figures'
TABLES_DIR   = OUTPUT_DIR / 'tables'
MAPS_DIR     = OUTPUT_DIR / 'maps'
RESULTS_DIR  = OUTPUT_DIR / 'results'

for d in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, MAPS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Data sources
DATA_FILE     = DATA_DIR / 'Data.csv'
INTEGRATED_FILE = HYPO1_OUTPUT / 'data_phase1_integrated.csv'  # 45-feature dataset from hypo1

# ─────────────────────────────────────────────
# TARGET & IONS
# ─────────────────────────────────────────────
TARGET = 'As'
WHO_AS_LIMIT = 10.0  # µg/L

IONS = [
    'Na+', 'K+', 'Mg2+', 'Ca2+', 'NO3-', 'SO42-', 'PO43-', 'CO3-',
    'HCO3-', 'Cl-', 'Br-', 'I-', 'Al3+', 'Cu2+', 'Fe2+', 'Mn2+', 'Cr3+', 'B+'
]

WATER_QUALITY_PARAMS = ['pH', 'DO', 'EC', 'TDS', 'Salinity', 'Temperature', 'Turbidity', 'ORP']

# ─────────────────────────────────────────────
# DEPTH ZONES (Holocene / Pleistocene framework)
# Boundary ~150 m from Winkel et al. 2011 applied to Bangladesh
# ─────────────────────────────────────────────
DEPTH_ZONES = {
    'Shallow':     (0,   30),   # Zone 1 — unconfined Holocene, surface recharge
    'Intermediate':(30,  80),   # Zone 2 — Holocene confined, classic Bangladesh hotspot
    'Medium_Deep': (80,  150),  # Zone 3 — Holocene/Pleistocene boundary
    'Deep':        (150, 500),  # Zone 4 — Pleistocene, geogenic mineral weathering
}
DEPTH_ZONE_LABELS = list(DEPTH_ZONES.keys())   # ordered list
HOLOCENE_PLEISTOCENE_BOUNDARY = 150  # m

# ─────────────────────────────────────────────
# PHYSIOGRAPHIC ZONE ASSIGNMENT
# Source: Habib et al. (2024), Saha & Rahman (2020),
#         standard Bangladesh physiographic zone literature
# Primary method: coordinate bounding boxes
# Fallback: district name lookup
# ─────────────────────────────────────────────
PHYSIOGRAPHIC_ZONES = {
    'Barind_Tract': {
        'lat': (23.8, 25.6), 'lon': (88.0, 89.2),
        'districts': ['Rajshahi', 'Chapainawabganj', 'Naogaon', 'Natore', 'Joypurhat'],
        'geology': 'Pleistocene terrace, orange-gray clay, low As',
        'reference': 'Habib et al. (2024)',
    },
    'Northern_Terrace': {
        'lat': (25.2, 26.7), 'lon': (88.0, 89.8),
        'districts': ['Rangpur', 'Dinajpur', 'Thakurgaon', 'Panchagarh',
                      'Nilphamari', 'Lalmonirhat', 'Kurigram', 'Gaibandha'],
        'geology': 'Himalayan piedmont, mixed Pleistocene-Holocene alluvial',
        'reference': 'General Bangladesh geology',
    },
    'Brahmaputra_Floodplain': {
        'lat': (23.5, 25.5), 'lon': (89.5, 91.2),
        'districts': ['Mymensingh', 'Jamalpur', 'Sherpur', 'Sirajganj',
                      'Tangail', 'Kishoreganj', 'Netrokona', 'Bogura', 'Pabna'],
        'geology': 'Young Holocene alluvial, Brahmaputra-Jamuna floodplain',
        'reference': 'Goswami et al. (2019)',
    },
    'Ganges_Floodplain': {
        'lat': (23.0, 24.5), 'lon': (88.5, 90.2),
        'districts': ['Kushtia', 'Meherpur', 'Chuadanga', 'Jhenaidah',
                      'Magura', 'Narail', 'Faridpur', 'Rajbari',
                      'Gopalganj', 'Madaripur', 'Shariatpur'],
        'geology': 'Ganges floodplain, mixed alluvial-deltaic, moderate-high As',
        'reference': 'Jha et al. (2023)',
    },
    'GBM_Delta': {
        'lat': (21.5, 23.5), 'lon': (89.0, 90.5),
        'districts': ['Khulna', 'Bagerhat', 'Satkhira', 'Barisal',
                      'Jhalokathi', 'Pirojpur', 'Bhola', 'Patuakhali', 'Barguna'],
        'geology': 'GBM delta, waterlogged, organic-rich Holocene, As hotspot',
        'reference': 'Shahid et al. (2018)',
    },
    'Meghna_Floodplain': {
        'lat': (22.5, 24.5), 'lon': (90.5, 91.5),
        'districts': ['Comilla', 'Chandpur', 'Lakshmipur', 'Noakhali',
                      'Feni', 'Brahmanbaria', 'Dhaka', 'Narayanganj',
                      'Munshiganj', 'Narsingdi', 'Manikganj', 'Gazipur'],
        'geology': 'Holocene Meghna floodplain, As-enriched sediment',
        'reference': 'Saha & Rahman (2020)',
    },
    'Eastern_Hills': {
        'lat': (21.5, 24.8), 'lon': (91.5, 92.7),
        'districts': ['Chittagong', "Cox's Bazar", 'Rangamati',
                      'Khagrachhari', 'Bandarban', 'Sylhet',
                      'Moulvibazar', 'Habiganj', 'Sunamganj'],
        'geology': 'Tertiary hill rocks, geogenic mineral weathering, low-As baseline',
        'reference': 'General Bangladesh geology',
    },
}

# ─────────────────────────────────────────────
# VARIABLE GROUPS FOR PROCESS PARTITIONING
# ─────────────────────────────────────────────

# Geogenic proxies — infer mineral weathering / sediment provenance
GEOGENIC_VARS = ['B+', 'Al3+', 'Mn2+', 'Fe2+', 'Ca2+', 'Mg2+', 'Cr3+', 'ORP', 'Depth']

# Diagnostic ion ratios for geogenic source discrimination
GEOGENIC_RATIOS = {
    'B_Cl_ratio':   ('B+',   'Cl-'),   # Marine sediment / evaporite provenance
    'Ca_Mg_ratio':  ('Ca2+', 'Mg2+'),  # Carbonate vs. silicate weathering
    'Mn_Fe_ratio':  ('Mn2+', 'Fe2+'),  # Thermodynamic redox sequence
    'HCO3_Cl_ratio':('HCO3-','Cl-'),   # Carbonate dissolution vs. marine water
    'Na_Cl_ratio':  ('Na+',  'Cl-'),   # Silicate weathering (feldspars) vs. marine
    'Al_Fe_ratio':  ('Al3+', 'Fe2+'),  # Aluminosilicate vs. Fe-oxide weathering
}

# Anthropogenic proxies — fertilizer, land use, irrigation
ANTHROPOGENIC_VARS = ['PO43-', 'NO3-']

# Satellite-derived land-use proxies (present in integrated dataset)
SATELLITE_VARS = ['NDVI_mean', 'EVI_mean', 'NDWI_mean', 'MNDWI_mean', 'NDMI_mean']

# Redox indicators
REDOX_VARS = ['ORP', 'DO', 'Fe2+', 'Mn2+', 'SO42-', 'HCO3-']

# Spatial variables
SPATIAL_VARS = ['Latitude', 'Longitude', 'District', 'Upazila']

# Variables requiring log transformation (right-skewed)
LOG_TRANSFORM_VARS = ['As', 'Fe2+', 'Mn2+', 'Al3+', 'Cu2+', 'B+', 'PO43-', 'NO3-']

# ─────────────────────────────────────────────
# STATISTICAL PARAMETERS
# ─────────────────────────────────────────────
ALPHA          = 0.05
RANDOM_STATE   = 42
TEST_SIZE      = 0.2
N_BOOTSTRAP    = 500    # Bootstrap iterations for variance partitioning CIs (was 1000 — OOM risk)
N_PERMUTATIONS = 499    # Permutation test for partial R²

# ML parameters
RF_N_ESTIMATORS  = 300   # was 500 — reduced to avoid OOM on macOS spawn workers
XGB_N_ESTIMATORS = 200   # was 300
CV_FOLDS         = 5

# Bayesian parameters
BAYES_DRAWS   = 2000
BAYES_TUNE    = 1000
BAYES_CHAINS  = 2

# GWR / spatial parameters
SPATIAL_BANDWIDTH = 'AICc'  # adaptive bandwidth selection
GWR_KERNEL        = 'gaussian'

# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────
DPI         = 300
FIGURE_SIZE = (14, 9)
COLORMAP    = 'RdYlBu_r'

ZONE_COLORS = {
    'Shallow':      '#d73027',
    'Intermediate': '#fc8d59',
    'Medium_Deep':  '#4575b4',
    'Deep':         '#313695',
}

GEO_ZONE_COLORS = {
    'Barind_Tract':           '#8B4513',
    'Northern_Terrace':       '#DAA520',
    'Brahmaputra_Floodplain': '#228B22',
    'Ganges_Floodplain':      '#20B2AA',
    'GBM_Delta':              '#4169E1',
    'Meghna_Floodplain':      '#DC143C',
    'Eastern_Hills':          '#9400D3',
}

# ─────────────────────────────────────────────────────────
print("=" * 65)
print("HYPOTHESIS 2 CONFIGURATION")
print("=" * 65)
print("H2: Depth-Stratified Geochemical Transitions")
print(f"  Depth zones   : {list(DEPTH_ZONES.keys())}")
print(f"  Geo zones     : {list(PHYSIOGRAPHIC_ZONES.keys())}")
print(f"  Geogenic vars : {len(GEOGENIC_VARS)}")
print(f"  Anthropogenic : {len(ANTHROPOGENIC_VARS)} ions + {len(SATELLITE_VARS)} satellite")
print(f"  WHO As limit  : {WHO_AS_LIMIT} µg/L")
print(f"  Output dir    : {OUTPUT_DIR}")
print("=" * 65)
