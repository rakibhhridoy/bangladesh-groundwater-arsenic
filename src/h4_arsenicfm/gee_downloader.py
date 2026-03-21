"""
AlphaEarth GEE Downloader — ArseniCFM (H4)

Downloads 64-dim AlphaEarth embeddings directly from Google Earth Engine
at all 1,807 well locations. No Google Drive needed.

Dataset: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
Bands:   A00 – A63  (64 dimensions, unit-length vectors)
Scale:   10 m
Year:    2021 (matches groundwater sampling year)

Usage:
    python gee_downloader.py            # authenticate + download
    python gee_downloader.py --year 2020  # use a different year

Requirements:
    pip install earthengine-api

Authentication (one-time):
    earthengine authenticate
    # OR call ee.Authenticate() inside the script (opens browser)
"""

import numpy as np
import pandas as pd
import argparse
import sys
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    INTEGRATED_FILE, SAT_TABLE, EMBEDDINGS_DIR, TABLES_DIR, RANDOM_STATE
)

DATASET_ID  = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
N_BANDS     = 64
BAND_NAMES  = [f'A{i:02d}' for i in range(N_BANDS)]   # A00 … A63
GEE_SCALE   = 10        # 10 m native resolution
CHUNK_SIZE  = 200       # points per GEE request (safe limit)
CACHE_PATH  = EMBEDDINGS_DIR / 'alphaearth_gee_embeddings.npy'
CACHE_IDS   = EMBEDDINGS_DIR / 'alphaearth_gee_location_ids.csv'


# ─── GEE AUTHENTICATION ───────────────────────────────────────────────────────

def init_gee(project: str = None):
    """
    Initialize Google Earth Engine.
    Tries service-account credentials first, then interactive auth.
    """
    try:
        import ee
    except ImportError:
        print("  earthengine-api not installed.")
        print("  Run: pip install earthengine-api")
        sys.exit(1)

    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        print("  GEE initialized (existing credentials)")
    except Exception:
        print("  No existing credentials — opening browser for authentication...")
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        print("  GEE authenticated and initialized")
    return ee


# ─── LOAD WELL COORDINATES ────────────────────────────────────────────────────

def load_well_coords():
    """
    Load (Location_ID, Latitude, Longitude) for all 1,807 wells.
    Returns DataFrame sorted by Location_ID.
    """
    df = pd.read_csv(INTEGRATED_FILE)
    if 'Location_ID' not in df.columns:
        df['Location_ID'] = np.arange(1, len(df) + 1)

    # De-duplicate by location (take first row per unique lat/lon)
    df['lat_r'] = df['Latitude'].round(4)
    df['lon_r'] = df['Longitude'].round(4)
    coords = df.drop_duplicates(subset=['lat_r', 'lon_r'])[
        ['Location_ID', 'Latitude', 'Longitude']
    ].reset_index(drop=True)

    print(f"  Unique well locations: {len(coords)}")
    return coords


# ─── SINGLE-CHUNK DOWNLOAD ────────────────────────────────────────────────────

def _sample_chunk(ee, image, chunk_df):
    """
    Sample AlphaEarth embedding at a chunk of well locations.
    Returns list of dicts: [{Location_ID, A00, A01, …, A63}, …]
    """
    features = [
        ee.Feature(
            ee.Geometry.Point([row['Longitude'], row['Latitude']]),
            {'loc_id': int(row['Location_ID'])}
        )
        for _, row in chunk_df.iterrows()
    ]
    fc = ee.FeatureCollection(features)

    sampled = image.sampleRegions(
        collection=fc,
        scale=GEE_SCALE,
        geometries=False,
        tileScale=4,
    )

    info = sampled.getInfo()   # triggers server-side computation

    rows = []
    for feat in info.get('features', []):
        props = feat['properties']
        loc_id = props.get('loc_id')
        emb = [props.get(b, np.nan) for b in BAND_NAMES]
        rows.append({'Location_ID': loc_id, 'embedding': emb})
    return rows


# ─── MAIN DOWNLOAD ────────────────────────────────────────────────────────────

def download_embeddings(year: int = 2021, project: str = None,
                        force: bool = False):
    """
    Download AlphaEarth 64-dim embeddings for all well locations.

    Parameters
    ----------
    year    : Sampling year (use year matching groundwater collection, default 2021)
    project : GEE Cloud project ID (optional, needed for some accounts)
    force   : Re-download even if cache exists

    Returns
    -------
    embeddings   : (N, 64) float32 normalized embeddings
    location_ids : (N,) int Location_ID
    """
    if CACHE_PATH.exists() and CACHE_IDS.exists() and not force:
        print(f"  Cache found: {CACHE_PATH}")
        embeddings   = np.load(CACHE_PATH)
        location_ids = pd.read_csv(CACHE_IDS)['location_id'].values
        print(f"  Loaded: {embeddings.shape}")
        return embeddings, location_ids

    ee = init_gee(project=project)

    # ── Load AlphaEarth image for target year ─────────────────────────────
    print(f"\n  Loading AlphaEarth embeddings ({year})...")
    bangladesh = ee.Geometry.BBox(87.8, 20.5, 92.8, 26.8)
    collection = ee.ImageCollection(DATASET_ID)
    image = (collection
             .filterDate(f'{year}-01-01', f'{year+1}-01-01')
             .filterBounds(bangladesh)
             .mosaic()
             .select(BAND_NAMES))

    n_tiles = (collection
               .filterDate(f'{year}-01-01', f'{year+1}-01-01')
               .filterBounds(bangladesh).size().getInfo())
    print(f"  Tiles covering Bangladesh: {n_tiles}")
    info = image.getInfo()
    actual_bands = [b['id'] for b in info.get('bands', [])]
    print(f"  Available bands: {len(actual_bands)}  ({actual_bands[:5]}…)")

    # ── Load well coordinates ─────────────────────────────────────────────
    coords = load_well_coords()

    # ── Chunked download ──────────────────────────────────────────────────
    all_rows = []
    n_chunks = (len(coords) + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"\n  Downloading in {n_chunks} chunks of {CHUNK_SIZE} points...")

    for i in range(n_chunks):
        chunk = coords.iloc[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        print(f"  Chunk {i+1}/{n_chunks} ({len(chunk)} pts)...", end=' ')
        try:
            rows = _sample_chunk(ee, image, chunk)
            all_rows.extend(rows)
            print(f"OK ({len(rows)} returned)")
        except Exception as e:
            print(f"FAILED: {e}")
            # Fill with NaN for failed chunk
            for _, row in chunk.iterrows():
                all_rows.append({
                    'Location_ID': int(row['Location_ID']),
                    'embedding': [np.nan] * N_BANDS
                })

    # ── Assemble matrix ───────────────────────────────────────────────────
    print(f"\n  Assembling embedding matrix...")
    loc_id_to_emb = {
        r['Location_ID']: np.array(r['embedding'], dtype=np.float32)
        for r in all_rows if r['Location_ID'] is not None
    }

    # Align to coords order (sorted by Location_ID)
    location_ids = coords['Location_ID'].values
    embeddings   = np.stack([
        loc_id_to_emb.get(int(lid), np.zeros(N_BANDS, dtype=np.float32))
        for lid in location_ids
    ])

    # Replace NaN with 0
    embeddings = np.where(np.isfinite(embeddings), embeddings, 0.0).astype(np.float32)

    n_valid = np.sum(np.abs(embeddings).sum(axis=1) > 0)
    print(f"  Valid embeddings: {n_valid}/{len(location_ids)}")
    print(f"  Embedding shape: {embeddings.shape}")

    # AlphaEarth embeddings are unit-length — verify
    norms = np.linalg.norm(embeddings[embeddings.sum(axis=1) != 0], axis=1)
    print(f"  Mean L2 norm: {norms.mean():.4f}  (expected ≈ 1.0 for AlphaEarth)")

    # ── Save ──────────────────────────────────────────────────────────────
    np.save(CACHE_PATH, embeddings)
    pd.DataFrame({'location_id': location_ids}).to_csv(CACHE_IDS, index=False)

    # Save raw (not z-scored — AlphaEarth embeddings are already unit-length)
    raw_path = EMBEDDINGS_DIR / 'alphaearth_gee_embeddings_raw.npy'
    np.save(raw_path, embeddings)

    print(f"\n  Saved: {CACHE_PATH}")
    print(f"  Saved: {CACHE_IDS}")

    # ── Summary table ─────────────────────────────────────────────────────
    summary = pd.DataFrame({
        'Location_ID': location_ids,
        **{f'A{i:02d}': embeddings[:, i] for i in range(N_BANDS)},
    })
    summary.to_csv(TABLES_DIR / 'H4_AlphaEarth_GEE_embeddings.csv', index=False)
    print(f"  Saved CSV: H4_AlphaEarth_GEE_embeddings.csv ({summary.shape})")

    return embeddings, location_ids


# ─── VERIFY / INSPECT ─────────────────────────────────────────────────────────

def inspect_embeddings(embeddings, location_ids):
    """Print quick diagnostic of downloaded embeddings."""
    print("\n" + "=" * 55)
    print("AlphaEarth Embedding Diagnostics")
    print("=" * 55)
    print(f"  Shape      : {embeddings.shape}")
    print(f"  Dtype      : {embeddings.dtype}")
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  L2 norms   : mean={norms.mean():.4f}  std={norms.std():.4f}")
    print(f"  Value range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
    nonzero = (np.abs(embeddings).sum(axis=1) > 0).sum()
    print(f"  Non-zero   : {nonzero}/{len(embeddings)}")
    print(f"  Cache path : {CACHE_PATH}")
    print("=" * 55)
    print("\nTo use in ArseniCFM pipeline, run:")
    print("  python main.py --phases B C D E F G")
    print("(Phase A will auto-load the GEE cache)")


# ─── PATCH main.py TO USE GEE EMBEDDINGS ──────────────────────────────────────

def patch_main_to_use_gee():
    """
    The main.py Phase A auto-detects the cache file name.
    GEE embeddings are saved to 'alphaearth_gee_embeddings.npy' —
    rename to 'sentinel_statistical_embeddings.npy' so main.py picks them up.
    """
    gee_path  = EMBEDDINGS_DIR / 'alphaearth_gee_embeddings.npy'
    stat_path = EMBEDDINGS_DIR / 'sentinel_statistical_embeddings.npy'
    gee_ids   = EMBEDDINGS_DIR / 'alphaearth_gee_location_ids.csv'
    stat_ids  = EMBEDDINGS_DIR / 'sentinel_location_ids.csv'

    if gee_path.exists():
        import shutil
        # Backup statistical fallback
        bak = EMBEDDINGS_DIR / 'sentinel_statistical_embeddings_backup.npy'
        if stat_path.exists() and not bak.exists():
            shutil.copy(stat_path, bak)
            print(f"  Backed up statistical embeddings → {bak.name}")

        shutil.copy(gee_path, stat_path)
        shutil.copy(gee_ids, stat_ids)
        print(f"  GEE embeddings installed as primary cache")
        print(f"  Run: python main.py --phases B C D E F G")
    else:
        print("  GEE embeddings not yet downloaded — run download first")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download AlphaEarth GEE embeddings for ArseniCFM')
    parser.add_argument('--year',    type=int, default=2021,
                        help='Year for AlphaEarth embeddings (default: 2021)')
    parser.add_argument('--project', type=str, default=None,
                        help='Google Cloud project ID (optional)')
    parser.add_argument('--force',   action='store_true',
                        help='Force re-download even if cache exists')
    parser.add_argument('--install', action='store_true',
                        help='After download, install GEE embeddings as primary cache')
    args = parser.parse_args()

    print("=" * 55)
    print("AlphaEarth GEE Downloader — ArseniCFM H4")
    print("=" * 55)
    print(f"  Dataset : {DATASET_ID}")
    print(f"  Year    : {args.year}")
    print(f"  Bands   : {N_BANDS} (A00–A63)")
    print(f"  Scale   : {GEE_SCALE}m")
    print(f"  Output  : {CACHE_PATH}")
    print("=" * 55)

    embeddings, location_ids = download_embeddings(
        year=args.year, project=args.project, force=args.force)
    inspect_embeddings(embeddings, location_ids)

    if args.install:
        patch_main_to_use_gee()
