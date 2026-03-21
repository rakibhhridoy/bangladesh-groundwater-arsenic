"""
H4 Data Loader
Loads integrated dataset + satellite embeddings and prepares multi-modal feature matrix.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from config import (
    INTEGRATED_FILE, SAT_TABLE, HYPO1_OUTPUT,
    GEO_FEATURE_COLS, CONTEXT_COLS, SAT_INDEX_COLS,
    DEPTH_ZONES, DEPTH_ZONE_LABELS, DEPTH_ZONE_TO_IDX,
    MECHANISM_LABELS, MECHANISM_TO_IDX,
    SEASON_DRY, SEASON_WET, WHO_AS_LIMIT, RANDOM_STATE,
    PHYSIOGRAPHIC_ZONES, PHYS_ZONE_LABELS
)


# ─── ZONE HELPERS ─────────────────────────────────────────────────────────────

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


def assign_mechanism(row):
    """
    Rule-based mechanism label from H2/H3 findings.
    Reductive: low ORP, high Fe2+
    Desorption: high PO43-, moderate ORP
    Mixed: otherwise
    """
    try:
        orp = float(row.get('ORP', 0))
        fe  = float(row.get('Fe2+', 0))
        po4 = float(row.get('PO43-', 0))
        if orp < -100 and fe > 0.3:
            return 'Reductive'
        if po4 > 0.1:
            return 'Desorption'
        return 'Mixed'
    except Exception:
        return 'Mixed'


# ─── MAIN LOADER ──────────────────────────────────────────────────────────────

def load_and_prepare_data():
    """
    Load integrated dataset with all features.
    Returns: df (full DataFrame with all columns including labels)
    """
    print("=" * 65)
    print("H4 DATA LOADER")
    print("=" * 65)

    # ── 1. Integrated geochemistry dataset ───────────────────────────────
    print(f"\n  Loading: {INTEGRATED_FILE}")
    df = pd.read_csv(INTEGRATED_FILE)
    print(f"  Shape: {df.shape}")

    # ── 2. Merge satellite indices (Table_12) ─────────────────────────────
    sat_cols_present = [c for c in SAT_INDEX_COLS if c in df.columns]
    if not sat_cols_present and SAT_TABLE.exists():
        sat = pd.read_csv(SAT_TABLE)
        sat_keep = [c for c in SAT_INDEX_COLS if c in sat.columns]
        if len(sat) == len(df) and sat_keep:
            for col in sat_keep:
                df[col] = sat[col].values
            print(f"  Satellite indices merged: {sat_keep}")
        else:
            print(f"  [WARN] SAT_TABLE shape mismatch; skipping")
    elif sat_cols_present:
        print(f"  Satellite indices already present: {sat_cols_present}")

    # ── 3. Depth zone & physiographic zone ───────────────────────────────
    if 'Depth_Zone' not in df.columns:
        df['Depth_Zone'] = df['Depth'].apply(assign_depth_zone)
    if 'Phys_Zone' not in df.columns:
        df['Phys_Zone'] = df.apply(
            lambda r: assign_phys_zone(r['Latitude'], r['Longitude']), axis=1)

    # ── 4. Season binary ─────────────────────────────────────────────────
    if 'Season_bin' not in df.columns:
        df['Season_bin'] = (df['Season'] == SEASON_WET).astype(int)

    # ── 5. Round As to 2 decimal places (instrument precision) ───────────
    df['As'] = df['As'].round(2)

    # ── 6. Classification labels ──────────────────────────────────────────
    df['Exceedance'] = (df['As'] > WHO_AS_LIMIT).astype(int)
    df['Mechanism']  = df.apply(assign_mechanism, axis=1)
    df['Mech_idx']   = df['Mechanism'].map(MECHANISM_TO_IDX).fillna(2).astype(int)
    df['Zone_idx']   = df['Depth_Zone'].map(DEPTH_ZONE_TO_IDX).fillna(0).astype(int)

    # ── 6. Location ID for embedding lookup ──────────────────────────────
    # The integrated dataset rows 1..N correspond to Location_IDs 1..N
    if 'Location_ID' not in df.columns:
        df['Location_ID'] = np.arange(1, len(df) + 1)

    # ── 7. Fill missing geo features with median ──────────────────────────
    for col in GEO_FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0.0

    # ── 8. Log-transform As target ────────────────────────────────────────
    df['As_log10'] = np.log10(df['As'].clip(lower=0.001))

    print(f"\n  Final df shape: {df.shape}")
    print(f"  As range: {df['As'].min():.2f} – {df['As'].max():.2f} µg/L")
    print(f"  WHO exceedance: {df['Exceedance'].mean()*100:.1f}%")
    print(f"  Mechanism counts: {df['Mechanism'].value_counts().to_dict()}")
    print(f"  Depth zones: {df['Depth_Zone'].value_counts().to_dict()}")
    print("=" * 65)
    return df


def build_feature_matrices(df, embeddings, location_ids):
    """
    Align satellite embeddings with the geochemistry DataFrame.

    Parameters
    ----------
    df           : DataFrame from load_and_prepare_data()
    embeddings   : (N_embed, 64) normalized satellite embeddings
    location_ids : (N_embed,) Location_ID for each embedding row

    Returns
    -------
    X_sat   : (N, 64) satellite embedding matrix (aligned to df rows)
    X_geo   : (N, n_geo) geochemical feature matrix
    X_ctx   : (N, 4) context features [lat, lon, depth, season]
    y_conc  : (N,) log10(As) for regression
    y_exc   : (N,) binary WHO exceedance
    y_mech  : (N,) mechanism class index
    y_zone  : (N,) depth zone class index
    """
    # Build lookup: location_id → embedding row
    id_to_emb = {int(lid): emb for lid, emb in zip(location_ids, embeddings)}
    embed_dim = embeddings.shape[1]

    X_sat = np.stack([
        id_to_emb.get(int(lid), np.zeros(embed_dim))
        for lid in df['Location_ID']
    ]).astype(np.float32)

    X_geo = df[[c for c in GEO_FEATURE_COLS if c in df.columns]].values.astype(np.float32)

    ctx_cols = [c for c in CONTEXT_COLS if c in df.columns]
    X_ctx = df[ctx_cols].values.astype(np.float32)

    y_conc = df['As_log10'].values.astype(np.float32)
    y_exc  = df['Exceedance'].values.astype(np.int64)
    y_mech = df['Mech_idx'].values.astype(np.int64)
    y_zone = df['Zone_idx'].values.astype(np.int64)

    print(f"\n  Feature matrices:")
    print(f"    X_sat  : {X_sat.shape}")
    print(f"    X_geo  : {X_geo.shape}")
    print(f"    X_ctx  : {X_ctx.shape}")
    print(f"    y_conc : {y_conc.shape}  mean={y_conc.mean():.3f}")

    return X_sat, X_geo, X_ctx, y_conc, y_exc, y_mech, y_zone


def make_spatial_blocks(df, block_deg=1.0):
    """Assign each sample a spatial block ID for GroupKFold CV."""
    lat_block = (df['Latitude'] // block_deg).astype(int)
    lon_block = (df['Longitude'] // block_deg).astype(int)
    block_id  = lat_block.astype(str) + '_' + lon_block.astype(str)
    return block_id.values
