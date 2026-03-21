"""
Prithvi-100M Encoder for ArseniCFM (Tier 2)
--------------------------------------------
Loads IBM-NASA Prithvi-EO-V1-100M (MAE ViT) from /tmp/prithvi/,
processes 1,807 Sentinel-2 TIFs, extracts 768-dim CLS tokens,
reduces to 64-dim via PCA, and saves as the primary embedding cache.

Usage:
    cd hypo4/
    python prithvi_encoder.py            # extract + cache (dry run)
    python prithvi_encoder.py --install  # also replace sentinel_statistical_embeddings.npy
    python prithvi_encoder.py --test     # forward-pass test only (1 TIF)
"""

import sys
import argparse
import numpy as np
import torch
import rasterio
import yaml
from pathlib import Path
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
import joblib

# ── paths ──────────────────────────────────────────────────────────────────────
PRITHVI_DIR  = Path('/tmp/prithvi')
SENTINEL_DIR = Path('/Users/rakibhhridoy/AsGW/GroundWater/EarthEngine')
EMBED_DIR    = Path('/Users/rakibhhridoy/AsGW/GroundWater/hypo4/output/embeddings')

CKPT_PATH    = PRITHVI_DIR / 'Prithvi_EO_V1_100M.pt'
CFG_PATH     = PRITHVI_DIR / 'config.yaml'
MODEL_SCRIPT = PRITHVI_DIR / 'prithvi_mae.py'

OUT_CLS      = EMBED_DIR / 'prithvi_cls768_tokens.npy'   # raw 768-dim
OUT_EMBED    = EMBED_DIR / 'prithvi_pca64_embeddings.npy' # PCA 64-dim
OUT_IDS      = EMBED_DIR / 'prithvi_location_ids.npy'
OUT_PCA      = EMBED_DIR / 'prithvi_pca_model.pkl'

# Primary cache (read by main.py / fm_claim_extended.py via alpha_earth.py)
PRIMARY_CACHE = EMBED_DIR / 'sentinel_statistical_embeddings.npy'

# ── band mapping ───────────────────────────────────────────────────────────────
# Our 7-band TIFs:  [WVP(0), B8A(1), B8(2), B5(3), B2(4), B3(5), B4(6)]
# Prithvi expects: [B02,    B03,    B04,    B05,   B06*,  B07*      ]
#   B02=Blue  → idx 4 (B2)
#   B03=Green → idx 5 (B3)
#   B04=Red   → idx 6 (B4)
#   B05=NIR   → idx 3 (B5)
#   B06=SWIR1 → idx 1 (B8A) — best available proxy
#   B07=SWIR2 → idx 2 (B8)  — best available proxy
BAND_MAP = [4, 5, 6, 3, 1, 2]

# Prithvi training normalisation (raw DN; applied after *10000 de-scaling)
PRITHVI_MEAN = np.array([775.229, 1080.993, 1228.586, 2497.202, 2204.214, 1610.832],
                         dtype=np.float32)
PRITHVI_STD  = np.array([1281.526, 1270.030, 1399.480, 1368.345, 1291.676, 1154.506],
                          dtype=np.float32)

IMG_SIZE  = 224
NUM_FRAMES = 3   # Prithvi trained with T=3; we repeat single-frame 3×
IN_CHANS   = 6
EMBED_DIM  = 768
OUT_DIM    = 64


# ── model loading ──────────────────────────────────────────────────────────────

def _load_prithvi_mae():
    """Instantiate PrithviMAE from local script + checkpoint."""
    if str(PRITHVI_DIR) not in sys.path:
        sys.path.insert(0, str(PRITHVI_DIR))
    from prithvi_mae import PrithviMAE

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    ma = cfg['model_args']

    patch_size = (ma.get('tubelet_size', 1), ma['patch_size'], ma['patch_size'])

    model = PrithviMAE(
        img_size      = ma['img_size'],
        patch_size    = patch_size,
        num_frames    = ma['num_frames'],
        in_chans      = ma['in_chans'],
        embed_dim     = ma['embed_dim'],
        depth         = ma['depth'],
        num_heads     = ma['num_heads'],
        decoder_embed_dim = ma['decoder_embed_dim'],
        decoder_depth     = ma['decoder_depth'],
        decoder_num_heads = ma['decoder_num_heads'],
    )

    state_dict = torch.load(CKPT_PATH, map_location='cpu')
    # Checkpoint is a plain OrderedDict with encoder.*/decoder.* keys
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys")
    model.eval()
    return model


# ── preprocessing ──────────────────────────────────────────────────────────────

def preprocess_tif(tif_path: Path) -> np.ndarray:
    """
    Load TIF (7-band, stored as reflectance ÷ 10000), select 6 Prithvi bands,
    scale back to DN range, normalize with Prithvi stats, resize to 224×224.
    Returns: (6, 224, 224) float32
    """
    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)   # (7, H, W) reflectance*10000 stored as-is

    img = data[BAND_MAP, :, :]                  # (6, H, W)

    # Resize to IMG_SIZE × IMG_SIZE
    H, W = img.shape[1], img.shape[2]
    if H != IMG_SIZE or W != IMG_SIZE:
        scale = (1.0, IMG_SIZE / H, IMG_SIZE / W)
        img = zoom(img, scale, order=1)

    # Normalize with Prithvi training stats (data already in ~DN range stored in TIF)
    img = (img - PRITHVI_MEAN[:, None, None]) / PRITHVI_STD[:, None, None]
    return img.astype(np.float32)


# ── feature extraction ─────────────────────────────────────────────────────────

@torch.no_grad()
def extract_cls_token(model, img_np: np.ndarray) -> np.ndarray:
    """
    img_np: (6, 224, 224) — already preprocessed
    Returns: (768,) CLS token from last encoder block
    """
    # Shape expected by PrithviViT: (B, C, T, H, W)
    tensor = torch.tensor(img_np[None, :, None, :, :], dtype=torch.float32)  # (1,6,1,224,224)
    tensor = tensor.expand(-1, -1, NUM_FRAMES, -1, -1).contiguous()           # (1,6,3,224,224)

    features = model.encoder.forward_features(tensor)   # list[12] of (1, N+1, 768)
    last_feat = features[-1]                             # (1, N+1, 768)
    cls_token = last_feat[0, 0, :].numpy()               # (768,)
    return cls_token


def extract_all(model, verbose=True):
    """
    Extract CLS tokens for all TIFs in SENTINEL_DIR.
    Returns cls_tokens (N, 768), loc_ids (N,).
    """
    tif_files = sorted(SENTINEL_DIR.glob('Sentinel2_Arsenic_Location_*.tif'),
                       key=lambda p: int(p.stem.split('_')[-1]))
    if verbose:
        print(f"  Found {len(tif_files)} TIF files in {SENTINEL_DIR}")

    cls_tokens = []
    loc_ids    = []
    errors     = 0

    for i, tif in enumerate(tif_files):
        loc_id = int(tif.stem.split('_')[-1])
        try:
            img = preprocess_tif(tif)
            cls = extract_cls_token(model, img)
            cls_tokens.append(cls)
            loc_ids.append(loc_id)
        except Exception as e:
            print(f"  [warn] TIF {tif.name}: {e}")
            cls_tokens.append(np.zeros(EMBED_DIM, dtype=np.float32))
            loc_ids.append(loc_id)
            errors += 1

        if verbose and (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(tif_files)} extracted...")

    if errors:
        print(f"  [warn] {errors} TIFs failed (zero-filled)")

    return np.stack(cls_tokens).astype(np.float32), np.array(loc_ids)


# ── PCA reduction ──────────────────────────────────────────────────────────────

def pca_reduce(cls_tokens: np.ndarray, n_components: int = OUT_DIM):
    """Fit PCA on cls_tokens (N, 768) → (N, 64)."""
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(cls_tokens)
    var_exp = pca.explained_variance_ratio_.sum()
    print(f"  PCA {cls_tokens.shape[1]}→{n_components}: variance explained = {var_exp:.3f}")
    return reduced.astype(np.float32), pca


# ── expand to 1807 rows ────────────────────────────────────────────────────────

def expand_to_dataset(embeddings: np.ndarray, loc_ids: np.ndarray,
                      integrated_csv: Path) -> np.ndarray:
    """
    TIFs are per unique location (838). Dataset has 1,807 rows (location repeats).
    Match embeddings by location_id to produce (1807, 64).
    """
    import pandas as pd
    df = pd.read_csv(integrated_csv)

    # Build loc→embedding map
    loc_map = {lid: embeddings[i] for i, lid in enumerate(loc_ids)}

    # Map to full dataset
    n = len(df)
    out = np.zeros((n, embeddings.shape[1]), dtype=np.float32)
    missing = 0
    for i, row in enumerate(df.itertuples()):
        lid = getattr(row, 'location_id', None) or getattr(row, 'Location_ID', None)
        if lid is not None and lid in loc_map:
            out[i] = loc_map[lid]
        else:
            missing += 1
    if missing:
        print(f"  [warn] {missing} rows had no matching location_id; zero-filled")
    return out


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--install', action='store_true',
                        help='Replace sentinel_statistical_embeddings.npy with Prithvi embeddings')
    parser.add_argument('--test', action='store_true',
                        help='Run forward-pass test on first TIF and exit')
    parser.add_argument('--force', action='store_true',
                        help='Recompute even if cache exists')
    args = parser.parse_args()

    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Prithvi-100M Encoder — ArseniCFM Tier 2")
    print("=" * 65)
    print(f"  Checkpoint : {CKPT_PATH}")
    print(f"  Sentinel   : {SENTINEL_DIR}")
    print(f"  Output     : {EMBED_DIR}")
    print("=" * 65)

    # ── load model ──
    print("\n[1/4] Loading Prithvi-100M...")
    model = _load_prithvi_mae()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model: {type(model.encoder).__name__}, {n_params:.1f}M params, embed_dim={EMBED_DIM}")

    # ── test mode ──
    if args.test:
        tifs = sorted(SENTINEL_DIR.glob('Sentinel2_Arsenic_Location_*.tif'))
        if not tifs:
            print("  No TIFs found!"); return
        tif = tifs[0]
        print(f"\n[TEST] Processing {tif.name}...")
        img = preprocess_tif(tif)
        print(f"  Preprocessed: {img.shape}, mean={img.mean():.3f}, std={img.std():.3f}")
        cls = extract_cls_token(model, img)
        print(f"  CLS token: {cls.shape}, norm={np.linalg.norm(cls):.3f}")
        print("  [TEST PASSED]")
        return

    # ── extract CLS tokens ──
    if OUT_CLS.exists() and not args.force:
        print(f"\n[2/4] Loading cached CLS tokens: {OUT_CLS}")
        cls_tokens = np.load(OUT_CLS)
        loc_ids = np.load(OUT_IDS)
    else:
        print("\n[2/4] Extracting Prithvi CLS tokens...")
        cls_tokens, loc_ids = extract_all(model, verbose=True)
        np.save(OUT_CLS, cls_tokens)
        np.save(OUT_IDS, loc_ids)
        print(f"  Saved CLS tokens: {cls_tokens.shape} → {OUT_CLS}")

    print(f"  CLS tokens shape : {cls_tokens.shape}")
    print(f"  NaN rate         : {np.isnan(cls_tokens).mean():.3f}")
    print(f"  Mean L2 norm     : {np.linalg.norm(cls_tokens, axis=1).mean():.4f}")

    # ── PCA reduction ──
    if OUT_EMBED.exists() and not args.force:
        print(f"\n[3/4] Loading cached PCA embeddings: {OUT_EMBED}")
        embeddings_64 = np.load(OUT_EMBED)
    else:
        print("\n[3/4] PCA reduction 768 → 64...")
        embeddings_64, pca = pca_reduce(cls_tokens, n_components=OUT_DIM)
        np.save(OUT_EMBED, embeddings_64)
        joblib.dump(pca, OUT_PCA)
        print(f"  Saved PCA embeddings: {embeddings_64.shape} → {OUT_EMBED}")
        print(f"  Saved PCA model      → {OUT_PCA}")

    print(f"\n[4/4] Embedding diagnostics:")
    print(f"  Shape        : {embeddings_64.shape}")
    print(f"  Non-zero rows: {(np.linalg.norm(embeddings_64, axis=1) > 0).sum()}/{len(embeddings_64)}")
    print(f"  Mean L2 norm : {np.linalg.norm(embeddings_64, axis=1).mean():.4f}")

    # ── install ──
    if args.install:
        import shutil

        # Expand to 1807 rows if needed
        n_rows = embeddings_64.shape[0]
        integrated = Path('/Users/rakibhhridoy/AsGW/GroundWater/hypo1/output/data_phase1_integrated.csv')
        if n_rows != 1807 and integrated.exists():
            print(f"\n  Expanding {n_rows} unique-location embeddings → 1807 dataset rows...")
            embeddings_1807 = expand_to_dataset(embeddings_64, loc_ids, integrated)
        else:
            embeddings_1807 = embeddings_64

        # Back up previous cache
        if PRIMARY_CACHE.exists():
            backup = PRIMARY_CACHE.with_name('previous_prithvi_backup.npy')
            shutil.copy(PRIMARY_CACHE, backup)
            print(f"  Backed up previous → {backup.name}")

        np.save(PRIMARY_CACHE, embeddings_1807)
        print(f"  Installed Prithvi embeddings as primary cache: {PRIMARY_CACHE}")
        print(f"  Shape: {embeddings_1807.shape}")

    print("\n  Done.")
    print("  Next: python main.py --phases B C D E F G")


if __name__ == '__main__':
    main()
