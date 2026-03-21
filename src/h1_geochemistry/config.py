"""
Configuration file for Hypothesis 1 analysis
Paths, constants, and settings
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'hypo1' / 'output'
FIGURES_DIR = OUTPUT_DIR / 'figures'
TABLES_DIR = OUTPUT_DIR / 'tables'
RESULTS_DIR = OUTPUT_DIR / 'results'

# Create output directories if they don't exist
for dir_path in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data file paths
DATA_FILE = DATA_DIR / 'Data.csv'
DRY_SEASON_FILE = DATA_DIR / 'DrySeason.csv'
WET_SEASON_FILE = DATA_DIR / 'WetSeason.csv'
CHEMICAL_DRY_FILE = DATA_DIR / 'ChemicalDry.csv'
CHEMICAL_WET_FILE = DATA_DIR / 'ChemicalWet.csv'

# Ion list (18 ions as per methodology)
IONS = ['Na+', 'K+', 'Mg2+', 'Ca2+', 'NO3-', 'SO42-', 'PO43-', 'CO3-',
        'HCO3-', 'Cl-', 'Br-', 'I-', 'Al3+', 'Cu2+', 'Fe2+', 'Mn2+', 'Cr3+', 'B+']

# Target variable
TARGET = 'As'

# Redox-sensitive ions (Hypothesis 1 focus)
REDOX_IONS = ['Fe2+', 'HCO3-', 'SO42-']

# Secondary variables for interpretation
SECONDARY_IONS = ['Na+', 'Cl-', 'Ca2+', 'Mg2+']

# Water quality parameters
WATER_QUALITY_PARAMS = ['pH', 'DO', 'EC', 'TDS', 'Salinity', 'Temperature', 'Turbidity', 'ORP']

# Spatial variables
SPATIAL_VARS = ['Latitude', 'Longitude', 'District', 'Upazila']

# Temporal variable
TEMPORAL_VAR = 'Season'

# Statistical thresholds
ALPHA = 0.05  # Significance level
ALPHA_BONFERRONI = 0.05 / len(IONS)  # Multiple comparison correction

# WHO Arsenic guideline
WHO_AS_LIMIT = 10.0  # µg/L

# Spatial weight matrix parameters
SPATIAL_WEIGHT_K = 8  # k-nearest neighbors for spatial weights
SPATIAL_WEIGHT_DISTANCE = None  # Or specify distance in km

# Clustering parameters
N_CLUSTERS_RANGE = range(2, 6)  # Test 2-5 clusters
LINKAGE_METHOD = 'ward'  # Hierarchical clustering linkage
DISTANCE_METRIC = 'euclidean'

# PCA parameters
PCA_VARIANCE_THRESHOLD = 0.80  # Retain components explaining 80% variance

# Regression parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Plotting parameters
DPI = 300
FIGURE_SIZE = (12, 8)
COLORMAP = 'viridis'

# Log transformations (variables to log-transform due to right-skewness)
LOG_TRANSFORM_VARS = ['As', 'Fe2+', 'HCO3-', 'Mn2+', 'Cu2+', 'Al3+']

# Print configuration summary
print("=" * 60)
print("HYPOTHESIS 1 CONFIGURATION")
print("=" * 60)
print(f"Project Root: {PROJECT_ROOT}")
print(f"Data Directory: {DATA_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"Number of ions analyzed: {len(IONS)}")
print(f"WHO As limit: {WHO_AS_LIMIT} µg/L")
print(f"Significance level (α): {ALPHA}")
print("=" * 60)
