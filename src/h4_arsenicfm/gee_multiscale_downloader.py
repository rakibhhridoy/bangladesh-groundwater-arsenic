"""
AlphaEarth Multi-Scale Contextual Embedding Downloader — ArseniCFM (H4)

Strategy:
  For each well, extract AlphaEarth embedding statistics at 3 spatial scales:
    250m  → micro-scale landscape (immediate well catchment)
    1km   → local scale (village / floodplain unit)
    2km   → regional scale (sedimentary structure)

  Per scale, compute 5 statistics × 64 bands = 320 features:
    mean, stdDev, p10, p50, p90

  Total: 3 × 320 = 960 features → PCA → 64-dim (matches model input)

  Water masking: JRC Global Surface Water (occurrence > 50%) applied before
  statistics — removes river/pond pixels that would dilute the land signal.

Usage:
    python gee_multiscale_downloader.py --project ee-arsenicbd
    python gee_multiscale_downloader.py --project ee-arsenicbd --install
"""

import numpy as np
import pandas as pd
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    INTEGRATED_FILE, EMBEDDINGS_DIR, TABLES_DIR, RANDOM_STATE
)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DATASET_ID   = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
WATER_DS     = 'JRC/GSW1_4/GlobalSurfaceWater'
BAND_NAMES   = [f'A{i:02d}' for i in range(64)]
SCALES_M     = {'s250': 250, 's1k': 1000, 's2k': 2000}
STATS        = ['mean', 'stdDev', 'p10', 'p50', 'p90']
N_RAW_FEATS  = len(SCALES_M) * len(BAND_NAMES) * len(STATS)   # 960
PCA_DIM      = 64
GEE_SCALE_M  = 10          # native AlphaEarth resolution
CHUNK_SIZE   = 100         # features per server-side map request


# ─── GEE SETUP ────────────────────────────────────────────────────────────────

def init_gee(project=None):
    try:
        import ee
    except ImportError:
        print("Run: pip install earthengine-api"); sys.exit(1)
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project) if project else ee.Initialize()
    print("  GEE initialized")
    return ee


def build_masked_image(ee, year):
    """AlphaEarth mosaic over Bangladesh with water pixels masked."""
    bangladesh = ee.Geometry.BBox(87.8, 20.5, 92.8, 26.8)

    image = (ee.ImageCollection(DATASET_ID)
             .filterDate(f'{year}-01-01', f'{year+1}-01-01')
             .filterBounds(bangladesh)
             .mosaic()
             .select(BAND_NAMES))

    # JRC water mask — mask out pixels with >50% occurrence as water
    water_occ = ee.Image(WATER_DS).select('occurrence')
    land_mask  = water_occ.lt(50).unmask(1)   # 1=land, 0=water
    image_masked = image.updateMask(land_mask)

    n_tiles = (ee.ImageCollection(DATASET_ID)
               .filterDate(f'{year}-01-01', f'{year+1}-01-01')
               .filterBounds(bangladesh).size().getInfo())
    print(f"  AlphaEarth tiles covering Bangladesh: {n_tiles}")
    print(f"  Water mask applied (JRC GSW, occurrence > 50%)")
    return image_masked


# ─── FEATURE EXTRACTION ───────────────────────────────────────────────────────

def _make_reducer(ee):
    """Combined reducer: mean + stdDev + p10 + p50 + p90."""
    return (ee.Reducer.mean()
            .combine(ee.Reducer.stdDev(),                sharedInputs=True)
            .combine(ee.Reducer.percentile([10, 50, 90]), sharedInputs=True))


def _extract_scale_chunk(ee, image, chunk_df, radius_m, scale_prefix, reducer):
    """
    Server-side fc.map() for one scale over a chunk of points.
    Returns list of dicts {loc_id, <scale_prefix>_<band>_<stat>: value, ...}
    """
    def make_feature(row):
        return ee.Feature(
            ee.Geometry.Point([row['Longitude'], row['Latitude']]),
            {'loc_id': int(row['Location_ID'])}
        )
    features = [make_feature(r) for _, r in chunk_df.iterrows()]
    fc = ee.FeatureCollection(features)

    def add_stats(feat):
        buf   = feat.geometry().buffer(radius_m)
        stats = image.reduceRegion(
            reducer   = reducer,
            geometry  = buf,
            scale     = GEE_SCALE_M,
            maxPixels = 1e8,
            bestEffort= True,
            tileScale = 4,
        )
        # Prefix all keys with scale name
        prefixed = ee.Dictionary(
            stats.keys().map(lambda k: ee.String(scale_prefix).cat('_').cat(k))
        ).combine(ee.Dictionary(stats.values().map(lambda v: v)))

        # Build correctly keyed dict: prefix_key → value
        keys_new = stats.keys().map(lambda k: ee.String(scale_prefix).cat('_').cat(k))
        vals     = stats.values()
        prefixed = ee.Dictionary.fromLists(keys_new, vals)
        return feat.set(prefixed)

    fc_stats = fc.map(add_stats)
    info     = fc_stats.getInfo()

    rows = []
    for feat in info.get('features', []):
        props  = feat['properties']
        loc_id = props.get('loc_id')
        row    = {'loc_id': loc_id}
        for band in BAND_NAMES:
            for stat in ['mean', 'stdDev', 'p10', 'p50', 'p90']:
                key     = f'{scale_prefix}_{band}_{stat}'
                row[key] = props.get(key, np.nan)
        rows.append(row)
    return rows


def extract_multiscale_embeddings(ee, image, coords, year):
    """
    Extract 960-dim multi-scale statistics for all unique well locations.
    Returns DataFrame (N_unique, 960+1) with loc_id and all feature columns.
    """
    reducer = _make_reducer(ee)
    all_scale_dfs = []

    for scale_name, radius_m in SCALES_M.items():
        print(f"\n  Scale {scale_name} ({radius_m}m radius)...")
        n_chunks = (len(coords) + CHUNK_SIZE - 1) // CHUNK_SIZE
        scale_rows = []

        for i in range(n_chunks):
            chunk = coords.iloc[i * CHUNK_SIZE : (i+1) * CHUNK_SIZE]
            print(f"    Chunk {i+1}/{n_chunks} ({len(chunk)} pts)...", end=' ')
            try:
                rows = _extract_scale_chunk(
                    ee, image, chunk, radius_m, scale_name, reducer)
                scale_rows.extend(rows)
                n_ok = sum(1 for r in rows
                           if any(v is not None and not (isinstance(v, float) and np.isnan(v))
                                  for k, v in r.items() if k != 'loc_id'))
                print(f"OK ({n_ok}/{len(rows)} non-null)")
            except Exception as e:
                print(f"FAILED: {e}")
                for _, row in chunk.iterrows():
                    empty = {'loc_id': int(row['Location_ID'])}
                    for band in BAND_NAMES:
                        for stat in ['mean', 'stdDev', 'p10', 'p50', 'p90']:
                            empty[f'{scale_name}_{band}_{stat}'] = np.nan
                    scale_rows.append(empty)

        all_scale_dfs.append(pd.DataFrame(scale_rows).set_index('loc_id'))

    # Merge all scales on loc_id
    combined = pd.concat(all_scale_dfs, axis=1).reset_index()
    combined.rename(columns={'index': 'loc_id'}, inplace=True)
    print(f"\n  Combined feature matrix: {combined.shape}")
    return combined


# ─── PCA COMPRESSION ──────────────────────────────────────────────────────────

def compress_to_64(feature_df, loc_ids_1807):
    """
    PCA: 960-dim → 64-dim.
    Expands from 838 unique locations to 1807 rows via nearest-neighbour.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial import cKDTree

    feat_cols = [c for c in feature_df.columns if c != 'loc_id']
    X_raw = feature_df[feat_cols].values.astype(np.float32)

    # Fill NaN with column median
    col_medians = np.nanmedian(X_raw, axis=0)
    nan_mask    = np.isnan(X_raw)
    X_filled    = np.where(nan_mask, col_medians[None, :], X_raw)

    print(f"\n  PCA: {X_filled.shape[1]}-dim → {PCA_DIM}-dim")
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_filled)

    pca = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_sc)
    var_explained = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"  PCA cumulative variance explained: {var_explained*100:.1f}%")

    # Save PCA model params for future use
    np.save(EMBEDDINGS_DIR / 'multiscale_pca_components.npy',
            pca.components_)
    np.save(EMBEDDINGS_DIR / 'multiscale_pca_mean.npy',
            scaler.mean_)
    np.save(EMBEDDINGS_DIR / 'multiscale_pca_scale.npy',
            scaler.scale_)

    # Build lookup: unique loc_id → 64-dim embedding
    unique_loc_ids = feature_df['loc_id'].values
    id_to_emb = {int(lid): X_pca[i] for i, lid in enumerate(unique_loc_ids)}

    # Load full dataset to get lat/lon for NN expansion
    df_full = pd.read_csv(INTEGRATED_FILE)
    if 'Location_ID' not in df_full.columns:
        df_full['Location_ID'] = np.arange(1, len(df_full) + 1)

    # Build KD-tree from unique downloaded locations
    df_unique = df_full[df_full['Location_ID'].isin(unique_loc_ids)] \
                       [['Location_ID', 'Latitude', 'Longitude']] \
                       .drop_duplicates('Location_ID') \
                       .set_index('Location_ID')
    tree = cKDTree(df_unique[['Latitude', 'Longitude']].values)

    # Expand to all 1807 rows
    X_1807 = np.zeros((len(loc_ids_1807), PCA_DIM), dtype=np.float32)
    for i, lid in enumerate(loc_ids_1807):
        if int(lid) in id_to_emb:
            X_1807[i] = id_to_emb[int(lid)]
        else:
            _, idx = tree.query([df_full.iloc[i]['Latitude'],
                                 df_full.iloc[i]['Longitude']])
            nearest_lid = df_unique.index[idx]
            X_1807[i] = id_to_emb.get(int(nearest_lid),
                                       np.zeros(PCA_DIM))

    nonzero = (np.abs(X_1807).sum(axis=1) > 0).sum()
    print(f"  Expanded: {X_1807.shape}  non-zero: {nonzero}/{len(loc_ids_1807)}")
    return X_1807, pca, var_explained


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def download_multiscale(year=2021, project=None, force=False, install=False):
    cache_raw = EMBEDDINGS_DIR / 'multiscale_raw_features.csv'
    cache_emb = EMBEDDINGS_DIR / 'multiscale_pca64_embeddings.npy'
    cache_ids = EMBEDDINGS_DIR / 'multiscale_location_ids.csv'

    # ── Load well coordinates ─────────────────────────────────────────────
    df_full = pd.read_csv(INTEGRATED_FILE)
    if 'Location_ID' not in df_full.columns:
        df_full['Location_ID'] = np.arange(1, len(df_full) + 1)
    df_full['lat_r'] = df_full['Latitude'].round(4)
    df_full['lon_r'] = df_full['Longitude'].round(4)
    coords = (df_full.drop_duplicates(subset=['lat_r', 'lon_r'])
                     [['Location_ID', 'Latitude', 'Longitude']]
                     .reset_index(drop=True))
    loc_ids_1807 = df_full['Location_ID'].values
    print(f"  Unique well locations: {len(coords)} → expand to {len(loc_ids_1807)} rows")

    # ── GEE download ──────────────────────────────────────────────────────
    if not cache_raw.exists() or force:
        ee = init_gee(project)
        image = build_masked_image(ee, year)
        feature_df = extract_multiscale_embeddings(ee, image, coords, year)
        feature_df.to_csv(cache_raw, index=False)
        print(f"  Saved raw features: {cache_raw}")
    else:
        print(f"  Loading cached raw features: {cache_raw}")
        feature_df = pd.read_csv(cache_raw)

    print(f"  Raw feature shape: {feature_df.shape}")
    print(f"  NaN rate: {feature_df.iloc[:, 1:].isna().mean().mean()*100:.1f}%")

    # ── PCA compression ───────────────────────────────────────────────────
    if not cache_emb.exists() or force:
        X_64, pca, var_exp = compress_to_64(feature_df, loc_ids_1807)
        np.save(cache_emb, X_64)
        pd.DataFrame({'location_id': loc_ids_1807}).to_csv(cache_ids, index=False)
        print(f"\n  Saved PCA embeddings: {cache_emb}")
        print(f"  Variance explained by {PCA_DIM} PCs: {var_exp*100:.1f}%")
    else:
        print(f"  Loading cached PCA embeddings: {cache_emb}")
        X_64 = np.load(cache_emb)

    # ── Diagnostics ───────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Multi-Scale Embedding Diagnostics")
    print("=" * 55)
    print(f"  Shape          : {X_64.shape}")
    nonzero = (np.abs(X_64).sum(axis=1) > 0).sum()
    print(f"  Non-zero rows  : {nonzero}/{len(X_64)}")
    norms = np.linalg.norm(X_64[X_64.sum(1) != 0], axis=1)
    print(f"  Mean L2 norm   : {norms.mean():.4f}")
    print(f"  Cache          : {cache_emb}")
    print("=" * 55)

    # ── Install as primary cache ──────────────────────────────────────────
    if install:
        import shutil
        # Backup previous
        prev = EMBEDDINGS_DIR / 'sentinel_statistical_embeddings.npy'
        bak  = EMBEDDINGS_DIR / 'previous_embeddings_backup.npy'
        if prev.exists():
            shutil.copy(prev, bak)
            print(f"  Backed up previous → {bak.name}")
        shutil.copy(cache_emb, prev)
        shutil.copy(cache_ids, EMBEDDINGS_DIR / 'sentinel_location_ids.csv')
        print("  Installed multi-scale PCA embeddings as primary cache")
        print("  Run: python main.py --phases B C D E F G")

    return X_64, loc_ids_1807


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AlphaEarth multi-scale contextual embedding downloader')
    parser.add_argument('--year',    type=int, default=2021)
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--force',   action='store_true',
                        help='Re-download even if cache exists')
    parser.add_argument('--install', action='store_true',
                        help='Install as primary model cache after download')
    args = parser.parse_args()

    print("=" * 55)
    print("AlphaEarth Multi-Scale Downloader — ArseniCFM H4")
    print("=" * 55)
    print(f"  Scales   : {list(SCALES_M.keys())} m")
    print(f"  Stats    : {STATS}")
    print(f"  Raw dim  : {N_RAW_FEATS} → PCA → {PCA_DIM}")
    print(f"  Year     : {args.year}")
    print(f"  Water    : JRC GSW mask (occurrence > 50%)")
    print("=" * 55)

    download_multiscale(
        year    = args.year,
        project = args.project,
        force   = args.force,
        install = args.install,
    )
