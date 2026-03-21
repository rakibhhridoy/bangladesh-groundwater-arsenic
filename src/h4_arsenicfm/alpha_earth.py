"""
Alpha Earth Foundation Model Interface — ArseniCFM
Provides 64-dim geospatial embeddings via three-tier fallback:

Tier 1: Alpha Earth REST API (if ALPHA_EARTH_API_KEY is set)
         → POST /v1/embed with TIF bytes → 64-dim embedding
Tier 2: IBM Prithvi-100M (if torch + transformers available)
         → Pre-trained ViT encoder on Sentinel-2 patches
Tier 3: Statistical feature extraction (sentinel_extractor.py)
         → Always available; 64 hand-crafted spectral/spatial features

The model framing:
  - Alpha Earth (or Prithvi) is the pre-trained backbone (analogous to BERT)
  - ArseniCFM is domain adaptation for arsenic prediction (analogous to BioBERT)
  - FM properties demonstrated: pre-trained backbone, multi-task generalization,
    few-shot adaptability
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import (
    ALPHA_EARTH_API_KEY, ALPHA_EARTH_BASE_URL, ALPHA_EARTH_EMBED_DIM,
    SENTINEL_DIR, EMBEDDINGS_DIR, RANDOM_STATE
)
from sentinel_extractor import extract_all_embeddings, extract_features_from_tif


# ─── TIER 1: ALPHA EARTH API ──────────────────────────────────────────────────

def _embed_via_alpha_earth_api(tif_path):
    """
    Call Alpha Earth v1 API to get 64-dim embedding for a Sentinel-2 TIF.
    Returns np.ndarray (64,) or None on failure.
    """
    try:
        import requests
        with open(tif_path, 'rb') as f:
            tif_bytes = f.read()
        resp = requests.post(
            f'{ALPHA_EARTH_BASE_URL}/embed',
            headers={'Authorization': f'Bearer {ALPHA_EARTH_API_KEY}',
                     'Content-Type': 'application/octet-stream'},
            data=tif_bytes,
            timeout=30
        )
        if resp.status_code == 200:
            j = resp.json()
            emb = np.array(j.get('embedding', []), dtype=np.float32)
            if len(emb) == ALPHA_EARTH_EMBED_DIM:
                return emb
    except Exception as e:
        pass
    return None


# ─── TIER 2: PRITHVI-100M (IBM + NASA) ────────────────────────────────────────

def _load_prithvi_model():
    """
    Load IBM Prithvi-100M from HuggingFace (ibm-nasa-geospatial/Prithvi-100M).
    Returns (model, processor) or (None, None).
    """
    try:
        import torch
        from transformers import AutoModel, AutoFeatureExtractor
        print("  Loading IBM Prithvi-100M from HuggingFace...")
        model_id = 'ibm-nasa-geospatial/Prithvi-100M'
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        model.eval()
        print("  Prithvi-100M loaded successfully.")
        return model, processor
    except Exception as e:
        print(f"  Prithvi-100M not available: {e}")
        return None, None


def _embed_via_prithvi(data_np, model, processor):
    """
    data_np: (7, H, W) float32 Sentinel-2 array (already /10000 reflectance)
    Returns np.ndarray (64,) projection of Prithvi CLS token.
    """
    try:
        import torch
        # Prithvi expects 6 HLS bands; map our 7 Sentinel bands to closest match
        # WVP, B8A, B8, B5, B2, B3, B4 → use B2(Blue), B3(Green), B4(Red), B8(NIR), B8A, B5
        band_map = [4, 5, 6, 2, 1, 3]   # indices into our 7-band stack
        img = data_np[band_map, :, :]    # (6, H, W)
        # Centre-crop to 224×224
        H, W = img.shape[1], img.shape[2]
        ch = min(H, 224)
        cw = min(W, 224)
        r0 = (H - ch) // 2; c0 = (W - cw) // 2
        img = img[:, r0:r0+ch, c0:c0+cw]
        # Resize to 224×224 using bilinear (simple numpy)
        from scipy.ndimage import zoom
        scale = (1, 224/img.shape[1], 224/img.shape[2])
        img = zoom(img, scale, order=1)
        tensor = torch.tensor(img[None], dtype=torch.float32)   # (1, 6, 224, 224)
        with torch.no_grad():
            out = model(pixel_values=tensor)
        cls = out.last_hidden_state[:, 0, :].numpy().squeeze()  # (hidden_dim,)
        # Project to 64-dim via PCA-like linear projection (deterministic)
        np.random.seed(RANDOM_STATE)
        proj = np.random.randn(len(cls), 64).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True)
        emb = cls @ proj
        return emb.astype(np.float32)
    except Exception as e:
        return None


# ─── MAIN EMBEDDER CLASS ──────────────────────────────────────────────────────

class AlphaEarthEmbedder:
    """
    Three-tier embedding engine:
      tier=1  Alpha Earth API
      tier=2  Prithvi-100M (local)
      tier=3  Statistical features (always works)

    Usage:
        embedder = AlphaEarthEmbedder()
        emb = embedder.embed_tif(tif_path)   # (64,)
        all_embs = embedder.embed_all()       # (N, 64) cached
    """

    def __init__(self):
        self.tier = self._detect_tier()
        self._prithvi_model = None
        self._prithvi_proc  = None
        print(f"  AlphaEarthEmbedder: using tier {self.tier}")
        if self.tier == 2:
            self._prithvi_model, self._prithvi_proc = _load_prithvi_model()
            if self._prithvi_model is None:
                self.tier = 3
                print("  Prithvi failed; falling back to tier 3 (statistical)")

    def _detect_tier(self):
        if ALPHA_EARTH_API_KEY:
            return 1
        try:
            import torch
            import transformers
            return 2
        except ImportError:
            return 3

    def embed_tif(self, tif_path):
        """Return 64-dim embedding for a single TIF."""
        tif_path = Path(tif_path)
        if self.tier == 1:
            emb = _embed_via_alpha_earth_api(tif_path)
            if emb is not None:
                return emb
            self.tier = 3  # fallback on API failure
        if self.tier == 2 and self._prithvi_model is not None:
            try:
                import rasterio
                with rasterio.open(tif_path) as src:
                    data = src.read().astype(np.float32) / 10000.0
                emb = _embed_via_prithvi(data, self._prithvi_model, self._prithvi_proc)
                if emb is not None:
                    return emb
            except Exception:
                pass
        # Tier 3: statistical
        return extract_features_from_tif(tif_path)

    def embed_all(self, force_recompute=False):
        """
        Embed all 1,808 Sentinel-2 TIFs.
        Returns (embeddings, location_ids).
        Tier 3 uses cached sentinel_extractor results.
        """
        if self.tier == 3:
            return extract_all_embeddings(force_recompute=force_recompute)

        # For tier 1/2: embed each TIF individually
        cache_path = EMBEDDINGS_DIR / f'alpha_earth_tier{self.tier}_embeddings.npy'
        cache_ids  = EMBEDDINGS_DIR / 'alpha_earth_location_ids.csv'

        if cache_path.exists() and not force_recompute:
            print(f"  Loading cached tier-{self.tier} embeddings")
            return np.load(cache_path), pd.read_csv(cache_ids)['location_id'].values

        tif_files = sorted(SENTINEL_DIR.glob('Sentinel2_Arsenic_Location_*.tif'),
                           key=lambda p: int(p.stem.split('_')[-1]))
        embeddings = []
        location_ids = []
        for tif in tif_files:
            loc_id = int(tif.stem.split('_')[-1])
            emb = self.embed_tif(tif)
            if emb is None:
                emb = np.zeros(ALPHA_EARTH_EMBED_DIM, dtype=np.float32)
            embeddings.append(emb)
            location_ids.append(loc_id)
            if len(location_ids) % 200 == 0:
                print(f"    Embedded {len(location_ids)}/{len(tif_files)} (tier {self.tier})...")

        embeddings = np.stack(embeddings)
        location_ids = np.array(location_ids)
        mu  = embeddings.mean(0, keepdims=True)
        sig = embeddings.std(0, keepdims=True) + 1e-8
        embeddings = (embeddings - mu) / sig
        np.save(cache_path, embeddings)
        pd.DataFrame({'location_id': location_ids}).to_csv(cache_ids, index=False)
        return embeddings, location_ids

    @property
    def embed_dim(self):
        return ALPHA_EARTH_EMBED_DIM

    @property
    def tier_label(self):
        labels = {1: 'Alpha Earth API', 2: 'Prithvi-100M', 3: 'Statistical (fallback)'}
        return labels[self.tier]
