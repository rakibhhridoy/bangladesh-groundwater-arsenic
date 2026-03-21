"""
Sentinel-2 Feature Extractor for ArseniCFM
Loads 1,808 GeoTIFFs and computes 64-dim statistical embeddings per location.
Used as the fallback when Alpha Earth API is unavailable.

Feature vector (64 dimensions total):
  - Per-band stats (7 bands × 5 stats: mean, std, p5, p95, median) = 35
  - Spectral indices (NDVI, EVI, NDWI, MNDWI, NDMI × 3 stats each) = 15
  - Spatial texture (7 bands: gradient variance)                     =  7
  - Band correlations (upper triangle: 7×6/2 = 21, take first 7)   =  7
Total: 64
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("  [WARN] rasterio not installed; Sentinel extraction unavailable")

from config import (
    SENTINEL_DIR, SENTINEL_BANDS, SENTINEL_SCALE,
    EMBEDDINGS_DIR, TABLES_DIR, RANDOM_STATE, N_SENTINEL
)

EMBED_DIM = 64
BAND_IDX = {b: i for i, b in enumerate(SENTINEL_BANDS)}


# ─── SINGLE-FILE FEATURE EXTRACTION ──────────────────────────────────────────

def extract_features_from_tif(tif_path):
    """
    Load a 7-band Sentinel-2 TIF and compute a 64-dim statistical embedding.
    Returns np.ndarray shape (64,) or None on failure.
    """
    if not HAS_RASTERIO:
        return None
    try:
        with rasterio.open(tif_path) as src:
            data = src.read().astype(np.float32) / SENTINEL_SCALE  # (7, H, W)
    except Exception:
        return None

    n_bands, H, W = data.shape
    flat = data.reshape(n_bands, -1)  # (7, H*W)
    valid_mask = np.all(np.isfinite(flat), axis=0)
    flat = flat[:, valid_mask]
    if flat.shape[1] < 10:
        return np.zeros(EMBED_DIM, dtype=np.float32)

    feats = []

    # ── 1. Per-band statistics (35 dims) ─────────────────────────────────
    for b in range(n_bands):
        v = flat[b]
        feats += [
            float(v.mean()),
            float(v.std()),
            float(np.percentile(v, 5)),
            float(np.percentile(v, 95)),
            float(np.median(v)),
        ]  # 5 per band × 7 bands = 35

    # ── 2. Spectral indices (15 dims) ─────────────────────────────────────
    B2   = flat[BAND_IDX['B2']]   # Blue
    B3   = flat[BAND_IDX['B3']]   # Green
    B4   = flat[BAND_IDX['B4']]   # Red
    B8   = flat[BAND_IDX['B8']]   # NIR
    B8A  = flat[BAND_IDX['B8A']]  # Red-edge
    B5   = flat[BAND_IDX['B5']]   # Red-edge 1
    WVP  = flat[BAND_IDX['WVP']]  # Water vapour

    def safe_index(a, b):
        denom = a + b
        return np.where(denom > 1e-6, (a - b) / denom, 0.0)

    NDVI  = safe_index(B8, B4)
    EVI   = 2.5 * safe_index(B8 - B4, B8 + 6*B4 - 7.5*B2 + 1)
    NDWI  = safe_index(B3, B8)
    MNDWI = safe_index(B3, B8A)
    NDMI  = safe_index(B8A, B5)

    for idx in [NDVI, EVI, NDWI, MNDWI, NDMI]:
        feats += [float(idx.mean()), float(idx.std()), float(np.percentile(idx, 75))]
    # 5 × 3 = 15

    # ── 3. Spatial texture: gradient variance (7 dims) ────────────────────
    for b in range(n_bands):
        band_2d = data[b]  # original (H, W)
        gy = np.diff(band_2d, axis=0)
        gx = np.diff(band_2d, axis=1)
        feats.append(float(np.nanstd(gy) + np.nanstd(gx)))
    # 7

    # ── 4. Inter-band correlations (7 dims: first 7 unique pairs) ─────────
    corr_pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3),(3,4)]
    for i, j in corr_pairs:
        c = np.corrcoef(flat[i], flat[j])[0, 1]
        feats.append(float(c) if np.isfinite(c) else 0.0)
    # 7

    # Total should be 35 + 15 + 7 + 7 = 64
    arr = np.array(feats, dtype=np.float32)
    if len(arr) != EMBED_DIM:
        # Pad or truncate to exactly 64
        if len(arr) < EMBED_DIM:
            arr = np.pad(arr, (0, EMBED_DIM - len(arr)))
        else:
            arr = arr[:EMBED_DIM]

    # Replace NaN/Inf
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


# ─── BATCH EXTRACTION ─────────────────────────────────────────────────────────

def extract_all_embeddings(force_recompute=False):
    """
    Extract 64-dim features for all 1,808 Sentinel-2 TIFs.
    Saves to EMBEDDINGS_DIR/sentinel_statistical_embeddings.npy (N, 64)
    and     EMBEDDINGS_DIR/sentinel_location_ids.csv

    Returns: embeddings (N, 64), location_ids (N,)
    """
    cache_emb  = EMBEDDINGS_DIR / 'sentinel_statistical_embeddings.npy'
    cache_ids  = EMBEDDINGS_DIR / 'sentinel_location_ids.csv'

    if cache_emb.exists() and cache_ids.exists() and not force_recompute:
        print(f"  Loading cached embeddings: {cache_emb}")
        embeddings = np.load(cache_emb)
        location_ids = pd.read_csv(cache_ids)['location_id'].values
        print(f"  Loaded: {embeddings.shape}")
        return embeddings, location_ids

    print(f"\n  Extracting Sentinel-2 features from {SENTINEL_DIR}")
    print(f"  Embedding dim: {EMBED_DIM}")

    tif_files = sorted(SENTINEL_DIR.glob('Sentinel2_Arsenic_Location_*.tif'),
                       key=lambda p: int(p.stem.split('_')[-1]))

    embeddings = []
    location_ids = []
    failures = 0

    for tif in tif_files:
        loc_id = int(tif.stem.split('_')[-1])
        feats = extract_features_from_tif(tif)
        if feats is None:
            feats = np.zeros(EMBED_DIM, dtype=np.float32)
            failures += 1
        embeddings.append(feats)
        location_ids.append(loc_id)

        if len(location_ids) % 200 == 0:
            print(f"    Processed {len(location_ids)}/{len(tif_files)}...")

    embeddings = np.stack(embeddings, axis=0)   # (N, 64)
    location_ids = np.array(location_ids)

    # Normalize: z-score per feature
    mu  = embeddings.mean(axis=0, keepdims=True)
    sig = embeddings.std(axis=0, keepdims=True) + 1e-8
    embeddings_norm = (embeddings - mu) / sig

    np.save(cache_emb, embeddings_norm)
    pd.DataFrame({'location_id': location_ids}).to_csv(cache_ids, index=False)

    # Save raw stats too (for inspection)
    np.save(EMBEDDINGS_DIR / 'sentinel_stat_embeddings_raw.npy', embeddings)
    np.save(EMBEDDINGS_DIR / 'sentinel_stat_normalization_mu.npy', mu)
    np.save(EMBEDDINGS_DIR / 'sentinel_stat_normalization_sig.npy', sig)

    print(f"  Extracted {len(embeddings)} locations | failures: {failures}")
    print(f"  Saved: {cache_emb}")
    return embeddings_norm, location_ids


# ─── FEATURE NAMES ────────────────────────────────────────────────────────────

def get_feature_names():
    """Return list of 64 feature names for interpretability."""
    names = []
    stats = ['mean', 'std', 'p05', 'p95', 'med']
    for b in SENTINEL_BANDS:
        for s in stats:
            names.append(f'{b}_{s}')
    for idx_name in ['NDVI', 'EVI', 'NDWI', 'MNDWI', 'NDMI']:
        for s in ['mean', 'std', 'p75']:
            names.append(f'{idx_name}_{s}')
    for b in SENTINEL_BANDS:
        names.append(f'{b}_gradient_var')
    pairs = ['B1B2', 'B1B3', 'B1B4', 'B2B3', 'B2B4', 'B3B4', 'B4B5']
    for p in pairs:
        names.append(f'corr_{p}')
    return names[:EMBED_DIM]


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    embeddings, loc_ids = extract_all_embeddings()
    feat_names = get_feature_names()
    print(f"\n  Feature names ({len(feat_names)}): {feat_names[:5]}...")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Mean L2 norm: {np.linalg.norm(embeddings, axis=1).mean():.3f}")
