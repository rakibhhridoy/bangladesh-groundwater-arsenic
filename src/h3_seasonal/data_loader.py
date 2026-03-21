"""
H3 Data Loader
Loads integrated dataset, classifies zones, computes ΔAs (matched pairs + kriging),
and integrates ERA5 weather data by well location and sampling date.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from config import (
    INTEGRATED_FILE, WEATHER_FILE, HYPO1_OUTPUT, TARGET, DEPTH_ZONES, DEPTH_ZONE_LABELS,
    PHYSIOGRAPHIC_ZONES, PHYS_ZONE_LABELS, SATELLITE_VARS, WEATHER_VARS,
    LOG_TRANSFORM_VARS, SEASON_DRY, SEASON_WET, MW_As, MW_PO4, MW_Cl,
    KRIGE_MAX_N, RANDOM_STATE
)


# ─── ZONE CLASSIFIERS ─────────────────────────────────────────────────────────

def assign_depth_zone(depth):
    for zone, (lo, hi) in DEPTH_ZONES.items():
        if lo <= depth < hi:
            return zone
    return 'Deep'


def assign_phys_zone(lat, lon):
    for zone, bounds in PHYSIOGRAPHIC_ZONES.items():
        if (bounds['lat'][0] <= lat <= bounds['lat'][1] and
                bounds['lon'][0] <= lon <= bounds['lon'][1]):
            return zone
    return 'Unclassified'


# ─── WEATHER INTEGRATION ──────────────────────────────────────────────────────

def load_weather(df_main):
    """
    Match each well sample to nearest ERA5 grid point at sampling date.
    Extracts: swvl1 (soil moisture), t2m (temperature), tcc (cloud cover), lai_hv (LAI).
    Returns df_main with weather columns appended.
    """
    try:
        print("  Loading ERA5 weather data (this may take ~30s)...")
        weather = pd.read_csv(WEATHER_FILE, parse_dates=['time'])
        # Filter to 2021 (groundwater sampling year)
        weather = weather[weather['time'].dt.year == 2021].copy()
        if len(weather) == 0:
            # Try without year filter — use all available
            weather = pd.read_csv(WEATHER_FILE, parse_dates=['time'])
            print(f"  Warning: No 2021 data; using all {len(weather)} weather records")

        # Parse sampling dates from main data
        df_main = df_main.copy()
        df_main['Date_parsed'] = pd.to_datetime(df_main['Date'], dayfirst=True, errors='coerce')

        # For each well, find nearest weather grid point and nearest time
        avail_weather_cols = [c for c in WEATHER_VARS if c in weather.columns]
        print(f"  Available weather vars: {avail_weather_cols}")

        rows = []
        # Build KD-tree on weather lat/lon for fast lookup
        from scipy.spatial import cKDTree
        weather_coords = weather[['latitude', 'longitude']].drop_duplicates().values
        kd = cKDTree(weather_coords)

        for _, row in df_main.iterrows():
            # Nearest spatial grid point
            _, idx = kd.query([row['Latitude'], row['Longitude']])
            nlat, nlon = weather_coords[idx]

            # Nearest time
            sub = weather[(weather['latitude'] == nlat) & (weather['longitude'] == nlon)]
            if len(sub) == 0 or pd.isna(row.get('Date_parsed')):
                rows.append({c: np.nan for c in avail_weather_cols})
                continue

            time_deltas = (sub['time'] - row['Date_parsed']).abs()
            nearest = sub.loc[time_deltas.idxmin(), avail_weather_cols]
            rows.append(nearest.to_dict())

        weather_df = pd.DataFrame(rows, index=df_main.index)
        df_main = pd.concat([df_main, weather_df], axis=1)
        print(f"  Weather integration complete: {avail_weather_cols} added")
        return df_main

    except FileNotFoundError:
        print(f"  Warning: {WEATHER_FILE} not found. Skipping weather integration.")
        for col in WEATHER_VARS:
            df_main[col] = np.nan
        return df_main
    except Exception as e:
        print(f"  Warning: Weather integration failed: {e}. Skipping.")
        for col in WEATHER_VARS:
            df_main[col] = np.nan
        return df_main


# ─── DELTA COMPUTATION (matched pairs + kriging for unmatched) ────────────────

def compute_delta(df):
    """
    Compute seasonal differences (ΔAs, ΔPO₄, ΔORP, ΔFe, ΔCl, ΔEC) for each well.
    Strategy:
      - 686 wells with both seasons → direct subtraction
      - 261+ wells with only one season → kriging to estimate missing season
    Returns df_delta with one row per unique well location.
    """
    print("\n  Computing seasonal deltas...")

    dry = df[df['Season'] == SEASON_DRY].copy()
    wet = df[df['Season'] == SEASON_WET].copy()

    # Round coords for matching (3 decimals ~ 100m precision)
    for d in [dry, wet, df]:
        d['lat_r'] = d['Latitude'].round(3)
        d['lon_r'] = d['Longitude'].round(3)

    DELTA_VARS = [TARGET, 'PO43-', 'ORP', 'Fe2+', 'Cl-', 'EC', 'TDS', 'HCO3-', 'DO', 'pH']
    DELTA_VARS = [v for v in DELTA_VARS if v in df.columns]

    # ── True pairs ──────────────────────────────────────────────────────────
    num_cols = (DELTA_VARS + ['Depth']
                + [c for c in SATELLITE_VARS if c in dry.columns]
                + [c for c in WEATHER_VARS if c in dry.columns])
    num_cols = [c for c in num_cols if c in dry.columns]

    dry_grp_num = dry.groupby(['lat_r', 'lon_r'])[num_cols].mean(numeric_only=True).reset_index()

    # Attach categorical zone labels (first value per group)
    cat_cols = [c for c in ['Depth_Zone', 'Phys_Zone'] if c in dry.columns]
    if cat_cols:
        dry_grp_cat = dry.groupby(['lat_r', 'lon_r'])[cat_cols].first().reset_index()
        dry_grp = pd.merge(dry_grp_num, dry_grp_cat, on=['lat_r', 'lon_r'], how='left')
    else:
        dry_grp = dry_grp_num

    wet_grp = wet.groupby(['lat_r', 'lon_r'])[DELTA_VARS].mean(numeric_only=True).reset_index()

    paired = pd.merge(dry_grp, wet_grp, on=['lat_r', 'lon_r'],
                      suffixes=('_dry', '_wet'))
    paired['is_kriged'] = False

    print(f"  True matched pairs: {len(paired)}")

    # ── Kriging for unmatched wells ─────────────────────────────────────────
    try:
        from pykrige.ok import OrdinaryKriging

        matched_coords = set(zip(paired['lat_r'], paired['lon_r']))

        dry_only = dry_grp[~dry_grp.apply(
            lambda r: (r['lat_r'], r['lon_r']) in matched_coords, axis=1)]
        wet_only = wet_grp[~wet_grp.apply(
            lambda r: (r['lat_r'], r['lon_r']) in matched_coords, axis=1)]

        krig_rows = []
        # Krig wet As for dry-only wells
        if len(dry_only) > 0 and len(wet) >= 15:
            wet_sub = wet.sample(min(KRIGE_MAX_N, len(wet)), random_state=RANDOM_STATE)
            as_log_wet = np.log10(wet_sub[TARGET].clip(lower=0.001))
            OK = OrdinaryKriging(wet_sub['Longitude'], wet_sub['Latitude'],
                                 as_log_wet, variogram_model='spherical',
                                 enable_plotting=False, verbose=False)
            z, _ = OK.execute('points',
                              dry_only['lon_r'].values, dry_only['lat_r'].values)
            for i, (_, row) in enumerate(dry_only.iterrows()):
                r = {f'{v}_dry': row.get(v, np.nan) for v in DELTA_VARS}
                r[f'{TARGET}_wet'] = float(10 ** z[i])
                for v in DELTA_VARS:
                    if v != TARGET:
                        r[f'{v}_wet'] = np.nan  # only As kriged
                r['lat_r'] = row['lat_r']
                r['lon_r'] = row['lon_r']
                r['Depth'] = row.get('Depth', np.nan)
                r['Depth_Zone'] = row.get('Depth_Zone', 'Unknown')
                r['Phys_Zone'] = row.get('Phys_Zone', 'Unclassified')
                for sv in SATELLITE_VARS:
                    r[sv] = row.get(sv, np.nan)
                r['is_kriged'] = True
                krig_rows.append(r)
            print(f"  Kriged wet-season As for {len(dry_only)} dry-only wells")

        # Krig dry As for wet-only wells
        if len(wet_only) > 0 and len(dry) >= 15:
            dry_sub = dry.sample(min(KRIGE_MAX_N, len(dry)), random_state=RANDOM_STATE)
            as_log_dry = np.log10(dry_sub[TARGET].clip(lower=0.001))
            OK = OrdinaryKriging(dry_sub['Longitude'], dry_sub['Latitude'],
                                 as_log_dry, variogram_model='spherical',
                                 enable_plotting=False, verbose=False)
            z, _ = OK.execute('points',
                              wet_only['lon_r'].values, wet_only['lat_r'].values)
            for i, (_, row) in enumerate(wet_only.iterrows()):
                r = {f'{v}_wet': row.get(v, np.nan) for v in DELTA_VARS
                     if v in wet_only.columns}
                r[f'{TARGET}_dry'] = float(10 ** z[i])
                for v in DELTA_VARS:
                    if v != TARGET:
                        r[f'{v}_dry'] = np.nan
                r['lat_r'] = row['lat_r']
                r['lon_r'] = row['lon_r']
                r['Depth'] = np.nan
                r['Depth_Zone'] = 'Unknown'
                r['Phys_Zone'] = 'Unclassified'
                for sv in SATELLITE_VARS:
                    r[sv] = row.get(sv, np.nan)
                r['is_kriged'] = True
                krig_rows.append(r)
            print(f"  Kriged dry-season As for {len(wet_only)} wet-only wells")

        if krig_rows:
            krig_df = pd.DataFrame(krig_rows)
            # Add missing suffix columns with NaN
            for v in DELTA_VARS:
                for suf in ('_dry', '_wet'):
                    col = v + suf
                    if col not in krig_df.columns:
                        krig_df[col] = np.nan
            # Align columns
            for col in paired.columns:
                if col not in krig_df.columns:
                    krig_df[col] = np.nan
            paired = pd.concat([paired, krig_df[paired.columns]], ignore_index=True)

    except ImportError:
        print("  pykrige not available; using only matched pairs for ΔAs")
    except Exception as e:
        print(f"  Kriging warning: {e}; using only matched pairs")

    # ── Compute Δ values ────────────────────────────────────────────────────
    for v in DELTA_VARS:
        dry_col = f'{v}_dry'
        wet_col = f'{v}_wet'
        if dry_col in paired.columns and wet_col in paired.columns:
            paired[f'Delta_{v}'] = paired[wet_col] - paired[dry_col]

    # Molar conversions for stoichiometry
    paired['As_umol_dry']  = paired[f'{TARGET}_dry']  / MW_As
    paired['As_umol_wet']  = paired[f'{TARGET}_wet']  / MW_As
    paired['Delta_As_umol'] = paired['As_umol_wet'] - paired['As_umol_dry']

    if 'PO43-_dry' in paired.columns:
        # Assume PO43- in mg/L → µmol/L (*1000/MW)
        paired['PO4_umol_dry'] = paired['PO43-_dry'] * 1000 / MW_PO4
        paired['PO4_umol_wet'] = paired['PO43-_wet'] * 1000 / MW_PO4
        paired['Delta_PO4_umol'] = paired['PO4_umol_wet'] - paired['PO4_umol_dry']

    # Season binary for SEM/regression
    paired['Season_bin'] = 0.5  # already averaged dry+wet; use full df for SEM

    # Rename coordinates
    paired.rename(columns={'lat_r': 'Latitude', 'lon_r': 'Longitude'}, inplace=True)

    print(f"  Total delta rows (matched + kriged): {len(paired)}")
    return paired


# ─── MAIN LOADER ─────────────────────────────────────────────────────────────

def load_and_prepare_data():
    """
    Load integrated dataset, assign zones, add weather, compute ΔAs.
    Returns: df (full), df_dry, df_wet, df_delta
    """
    print("=" * 70)
    print("H3 DATA LOADER")
    print("=" * 70)

    # ── Load integrated dataset ───────────────────────────────────────────
    print(f"\n  Loading: {INTEGRATED_FILE}")
    df = pd.read_csv(INTEGRATED_FILE)
    print(f"  Shape: {df.shape}")

    # ── Merge satellite features by row position (Location_ID = 1-based index) ─
    SAT_FILE = HYPO1_OUTPUT / 'tables' / 'Table_12_satellite_features.csv'
    sat_cols_present = [c for c in SATELLITE_VARS if c in df.columns]
    if not sat_cols_present and SAT_FILE.exists():
        sat = pd.read_csv(SAT_FILE)
        sat_mean_cols = ['NDVI_mean', 'EVI_mean', 'NDWI_mean', 'MNDWI_mean', 'NDMI_mean']
        sat_keep = [c for c in sat_mean_cols if c in sat.columns]
        if len(sat) == len(df) and sat_keep:
            for col in sat_keep:
                df[col] = sat[col].values
            print(f"  Satellite features merged: {sat_keep}")
        else:
            print(f"  [WARN] Satellite file shape mismatch or no mean cols; skipping")
    elif sat_cols_present:
        print(f"  Satellite features already in dataset: {sat_cols_present}")

    # ── Depth zone classification ─────────────────────────────────────────
    if 'Depth_Zone' not in df.columns:
        df['Depth_Zone'] = df['Depth'].apply(assign_depth_zone)
    print(f"  Depth zone counts:\n{df['Depth_Zone'].value_counts().to_string()}")

    # ── Physiographic zone ────────────────────────────────────────────────
    if 'Phys_Zone' not in df.columns:
        df['Phys_Zone'] = df.apply(
            lambda r: assign_phys_zone(r['Latitude'], r['Longitude']), axis=1)

    # ── Log transforms ────────────────────────────────────────────────────
    for col in LOG_TRANSFORM_VARS:
        if col in df.columns:
            df[f'{col}_log'] = np.log10(df[col].clip(lower=1e-6))

    # Season binary
    df['Season_bin'] = (df['Season'] == SEASON_WET).astype(int)

    # ── Weather integration ───────────────────────────────────────────────
    df = load_weather(df)

    # ── Split by season ───────────────────────────────────────────────────
    df_dry = df[df['Season'] == SEASON_DRY].copy().reset_index(drop=True)
    df_wet = df[df['Season'] == SEASON_WET].copy().reset_index(drop=True)
    print(f"\n  Dry season: n={len(df_dry)}  |  Wet season: n={len(df_wet)}")

    # ── Seasonal delta dataset ────────────────────────────────────────────
    df_delta = compute_delta(df)

    # Ensure Depth_Zone in delta for stratified analyses
    if 'Depth_Zone' not in df_delta.columns or df_delta['Depth_Zone'].isna().all():
        df_delta['Depth_Zone'] = df_delta.get('Depth', pd.Series(dtype=float)).apply(
            lambda d: assign_depth_zone(d) if pd.notna(d) else 'Unknown')

    print(f"\n  Data preparation complete.")
    print(f"  Full df: {df.shape} | Dry: {df_dry.shape} | Wet: {df_wet.shape} | Delta: {df_delta.shape}")
    print("=" * 70)

    return df, df_dry, df_wet, df_delta
