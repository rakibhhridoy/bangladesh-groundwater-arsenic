"""
Data loading and preparation for Hypothesis 2
Assigns depth zones and physiographic zones to all 1,807 samples
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_FILE, INTEGRATED_FILE, IONS, TARGET, WATER_QUALITY_PARAMS,
    SPATIAL_VARS, WHO_AS_LIMIT, LOG_TRANSFORM_VARS, OUTPUT_DIR,
    DEPTH_ZONES, DEPTH_ZONE_LABELS, PHYSIOGRAPHIC_ZONES,
    GEOGENIC_VARS, GEOGENIC_RATIOS, ANTHROPOGENIC_VARS, SATELLITE_VARS,
    HOLOCENE_PLEISTOCENE_BOUNDARY, TABLES_DIR
)


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

def load_data():
    """Load integrated dataset (hypo1 output) or fall back to raw Data.csv."""
    if INTEGRATED_FILE.exists():
        df = pd.read_csv(INTEGRATED_FILE)
        print(f"Loaded integrated dataset: {df.shape}  [{INTEGRATED_FILE.name}]")
    else:
        df = pd.read_csv(DATA_FILE)
        print(f"Loaded raw dataset: {df.shape}  [{DATA_FILE.name}]")

    # Normalise column names: strip whitespace
    df.columns = df.columns.str.strip()

    # Ensure Season column exists
    if 'Season' not in df.columns and 'Date' in df.columns:
        df['Season'] = 'Unknown'

    print(f"  Columns : {list(df.columns)}")
    return df


# ─────────────────────────────────────────────
# DEPTH ZONE ASSIGNMENT
# ─────────────────────────────────────────────

def assign_depth_zones(df):
    """
    Classify each sample into one of 4 depth zones.

    Zones (from config):
        Shallow      : 0–30 m   (unconfined Holocene)
        Intermediate : 30–80 m  (Holocene confined)
        Medium_Deep  : 80–150 m (Holocene/Pleistocene boundary)
        Deep         : >150 m   (Pleistocene)
    """
    if 'Depth' not in df.columns:
        raise ValueError("'Depth' column not found in dataset.")

    conditions = []
    choices    = []
    for zone, (lo, hi) in DEPTH_ZONES.items():
        conditions.append((df['Depth'] >= lo) & (df['Depth'] < hi))
        choices.append(zone)

    df['Depth_Zone'] = np.select(conditions, choices, default='Deep')

    # Ordered categorical for consistent sorting
    df['Depth_Zone'] = pd.Categorical(
        df['Depth_Zone'], categories=DEPTH_ZONE_LABELS, ordered=True
    )

    # Holocene / Pleistocene binary flag (Winkel et al. 2011 boundary)
    df['Aquifer_Type'] = np.where(
        df['Depth'] < HOLOCENE_PLEISTOCENE_BOUNDARY, 'Holocene', 'Pleistocene'
    )

    print("\nDepth Zone Distribution:")
    for zone in DEPTH_ZONE_LABELS:
        lo, hi = DEPTH_ZONES[zone]
        n = (df['Depth_Zone'] == zone).sum()
        mean_as = df.loc[df['Depth_Zone'] == zone, TARGET].mean()
        pct_who = (df.loc[df['Depth_Zone'] == zone, TARGET] > WHO_AS_LIMIT).mean() * 100
        print(f"  {zone:15s} ({lo:3d}–{hi:3d} m): n={n:4d} | "
              f"Mean As={mean_as:6.2f} µg/L | {pct_who:.1f}% >WHO")

    holo_n  = (df['Aquifer_Type'] == 'Holocene').sum()
    pleis_n = (df['Aquifer_Type'] == 'Pleistocene').sum()
    print(f"\n  Holocene  (<{HOLOCENE_PLEISTOCENE_BOUNDARY} m): n={holo_n}")
    print(f"  Pleistocene (≥{HOLOCENE_PLEISTOCENE_BOUNDARY} m): n={pleis_n}")

    return df


# ─────────────────────────────────────────────
# PHYSIOGRAPHIC ZONE ASSIGNMENT
# ─────────────────────────────────────────────

def _zone_from_coordinates(lat, lon):
    """Return physiographic zone name from lat/lon bounding boxes."""
    for zone, info in PHYSIOGRAPHIC_ZONES.items():
        lat_lo, lat_hi = info['lat']
        lon_lo, lon_hi = info['lon']
        if lat_lo <= lat <= lat_hi and lon_lo <= lon <= lon_hi:
            return zone
    return 'Unclassified'


def _zone_from_district(district):
    """Fallback: return zone from district name lookup."""
    if not isinstance(district, str):
        return 'Unclassified'
    d = district.strip().title()
    for zone, info in PHYSIOGRAPHIC_ZONES.items():
        if d in [x.title() for x in info['districts']]:
            return zone
    return 'Unclassified'


def assign_physiographic_zones(df):
    """
    Assign each sample to a Bangladesh physiographic zone.
    Primary method: coordinate bounding boxes.
    Fallback: District name lookup.
    """
    has_coords = ('Latitude' in df.columns and 'Longitude' in df.columns)
    has_district = 'District' in df.columns

    if has_coords:
        df['Phys_Zone'] = df.apply(
            lambda r: _zone_from_coordinates(r['Latitude'], r['Longitude']), axis=1
        )
        # Resolve unclassified via district if available
        if has_district:
            mask = df['Phys_Zone'] == 'Unclassified'
            df.loc[mask, 'Phys_Zone'] = df.loc[mask, 'District'].apply(_zone_from_district)
    elif has_district:
        df['Phys_Zone'] = df['District'].apply(_zone_from_district)
    else:
        df['Phys_Zone'] = 'Unclassified'

    print("\nPhysiographic Zone Distribution:")
    for zone in list(PHYSIOGRAPHIC_ZONES.keys()) + ['Unclassified']:
        n = (df['Phys_Zone'] == zone).sum()
        if n == 0:
            continue
        mean_as = df.loc[df['Phys_Zone'] == zone, TARGET].mean()
        print(f"  {zone:28s}: n={n:4d} | Mean As={mean_as:6.2f} µg/L")

    return df


# ─────────────────────────────────────────────
# GEOGENIC RATIO COMPUTATION
# ─────────────────────────────────────────────

def compute_geogenic_ratios(df):
    """
    Compute diagnostic ion ratios for geogenic source discrimination.
    All ratios computed from raw (non-log) concentrations with a small
    offset to avoid division by zero.
    """
    eps = 1e-6
    print("\nComputing geogenic ratios:")
    for ratio_name, (num, den) in GEOGENIC_RATIOS.items():
        if num in df.columns and den in df.columns:
            df[ratio_name] = df[num] / (df[den] + eps)
            print(f"  {ratio_name:20s} = {num} / {den}  "
                  f"(median={df[ratio_name].median():.4f})")
        else:
            missing = [v for v in [num, den] if v not in df.columns]
            print(f"  SKIP {ratio_name}: missing {missing}")

    return df


# ─────────────────────────────────────────────
# LOG TRANSFORMATION
# ─────────────────────────────────────────────

def log_transform(df):
    """Log10-transform right-skewed variables (handles zeros with offset)."""
    print("\nLog-transforming skewed variables:")
    for var in LOG_TRANSFORM_VARS:
        if var in df.columns:
            col_log = f'{var}_log'
            df[col_log] = np.log10(df[var].clip(lower=0) + 1e-6)
            skew_raw = df[var].skew()
            skew_log = df[col_log].skew()
            print(f"  {var:12s}: skew {skew_raw:+.2f} → {skew_log:+.2f}")
    return df


# ─────────────────────────────────────────────
# QUALITY FLAGS
# ─────────────────────────────────────────────

def flag_data_quality(df):
    """Flag physically implausible values (same thresholds as hypo1)."""
    flags = pd.Series(False, index=df.index)

    if 'pH' in df.columns:
        flags |= (df['pH'] > 14) | (df['pH'] < 0)
    if 'Salinity' in df.columns:
        flags |= (df['Salinity'] < 0)
    if 'EC' in df.columns:
        flags |= (df['EC'] < 0)
    if 'ORP' in df.columns:
        flags |= (df['ORP'] < -1000)

    df['QC_Flag'] = flags
    print(f"\nQuality flags: {flags.sum()} samples flagged as implausible")
    return df


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def load_and_prepare_data():
    """
    Full data preparation pipeline for Hypothesis 2.

    Returns
    -------
    df : pd.DataFrame
        Prepared dataset with depth zones, physiographic zones,
        geogenic ratios, log transforms, and QC flags.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 2: DATA PREPARATION")
    print("=" * 70)

    df = load_data()

    # Drop rows where As is missing
    before = len(df)
    df = df[df[TARGET].notna()].copy()
    print(f"\nDropped {before - len(df)} rows with missing As. Remaining: {len(df)}")

    # Assign depth zones
    df = assign_depth_zones(df)

    # Assign physiographic zones
    df = assign_physiographic_zones(df)

    # Compute geogenic ratios
    df = compute_geogenic_ratios(df)

    # Log transforms
    df = log_transform(df)

    # Quality flags
    df = flag_data_quality(df)

    # WHO exceedance summary
    n_above = (df[TARGET] > WHO_AS_LIMIT).sum()
    print(f"\nWHO Exceedance (>{WHO_AS_LIMIT} µg/L): {n_above} / {len(df)} "
          f"({100 * n_above / len(df):.1f}%)")
    print(f"\nFinal dataset shape: {df.shape}")

    # Save prepared dataset
    out_path = OUTPUT_DIR / 'data_h2_prepared.csv'
    df.to_csv(out_path, index=False)
    print(f"Prepared data saved → {out_path}")

    # Zone summary table
    zone_summary = df.groupby('Depth_Zone', observed=True).agg(
        N=(TARGET, 'count'),
        As_mean=(TARGET, 'mean'),
        As_median=(TARGET, 'median'),
        As_std=(TARGET, 'std'),
        Pct_above_WHO=(TARGET, lambda x: (x > WHO_AS_LIMIT).mean() * 100),
        Depth_mean=('Depth', 'mean'),
    ).round(2)
    zone_summary.to_csv(TABLES_DIR / 'H2_T01_depth_zone_summary.csv')
    print(f"Zone summary saved → {TABLES_DIR}/H2_T01_depth_zone_summary.csv")

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)

    return df


if __name__ == "__main__":
    df = load_and_prepare_data()
