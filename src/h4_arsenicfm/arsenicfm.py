"""
ArseniCFM — Foundation Model for Arsenic Contamination Prediction

Architecture:
  Satellite Encoder (Alpha Earth, frozen) ──┐
  Geochemical Encoder (MLP, trained)        ├─→ Cross-Modal Attention → Multi-task heads
  Context Encoder (MLP, trained)            ┘

Multi-task outputs:
  1. As concentration (log10 µg/L) — regression
  2. P(As > 10 µg/L)               — binary classification
  3. Mechanism type                 — 3-class (Reductive/Desorption/Mixed)
  4. Depth zone                     — 4-class (Shallow/Intermediate/Medium_Deep/Deep)

The satellite encoder weights are kept frozen during fine-tuning to demonstrate
the foundation model property: pre-trained geospatial representations transfer
to a novel downstream task (groundwater arsenic) with minimal domain-specific data.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from config import (
    SAT_EMBED_DIM, GEO_EMBED_DIM, CTX_EMBED_DIM, FUSED_DIM,
    HIDDEN_DIM, ATTN_HEADS, DROPOUT,
    MECHANISM_LABELS, DEPTH_ZONE_LABELS, TASK_WEIGHTS,
    RANDOM_STATE
)


# ─── SKLEARN-COMPATIBLE FALLBACK ─────────────────────────────────────────────
# When PyTorch is unavailable, ArseniCFM falls back to an sklearn-based
# gradient boosting model that still demonstrates the multi-modal fusion concept.

class ArseniCFM_SklearnFallback:
    """
    Sklearn-based ArseniCFM for environments without PyTorch.
    Concatenates sat + geo + ctx features, trains XGBoost per task.
    """
    def __init__(self):
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        self.models = {}

    def fit(self, X_sat, X_geo, X_ctx, y_conc, y_exc, y_mech, y_zone):
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        X = np.concatenate([X_sat, X_geo, X_ctx], axis=1)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        print("  Fitting sklearn fallback models...")
        self.models['concentration'] = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, random_state=RANDOM_STATE)
        self.models['concentration'].fit(X, y_conc)
        self.models['exceedance'] = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, random_state=RANDOM_STATE)
        self.models['exceedance'].fit(X, y_exc)
        self.models['mechanism'] = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=RANDOM_STATE)
        self.models['mechanism'].fit(X, y_mech)
        self.models['depth_zone'] = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=RANDOM_STATE)
        self.models['depth_zone'].fit(X, y_zone)
        return self

    def predict(self, X_sat, X_geo, X_ctx):
        X = np.concatenate([X_sat, X_geo, X_ctx], axis=1)
        X = self.scaler.transform(X)
        return {
            'concentration': self.models['concentration'].predict(X),
            'exceedance':    self.models['exceedance'].predict_proba(X)[:, 1],
            'mechanism':     self.models['mechanism'].predict(X),
            'depth_zone':    self.models['depth_zone'].predict(X),
        }


# ─── PYTORCH MODEL ────────────────────────────────────────────────────────────

if HAS_TORCH:

    class GeoChemEncoder(nn.Module):
        """MLP encoder for geochemical ion measurements → 32-dim embedding."""
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, GEO_EMBED_DIM),
            )

        def forward(self, x):
            return self.net(x)


    class ContextEncoder(nn.Module):
        """MLP encoder for lat/lon/depth/season → 16-dim embedding."""
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 32),
                nn.ReLU(),
                nn.Linear(32, CTX_EMBED_DIM),
            )

        def forward(self, x):
            return self.net(x)


    class CrossModalAttention(nn.Module):
        """
        Multi-head attention between satellite and geochemical embeddings.
        Satellite embedding attends to geochemical context, producing
        a context-aware satellite representation.
        """
        def __init__(self, sat_dim=SAT_EMBED_DIM, geo_dim=GEO_EMBED_DIM, n_heads=ATTN_HEADS):
            super().__init__()
            # Project satellite and geo to same dimension for attention
            self.q_proj = nn.Linear(sat_dim, 64)
            self.k_proj = nn.Linear(geo_dim, 64)
            self.v_proj = nn.Linear(geo_dim, 64)
            self.out_proj = nn.Linear(64, sat_dim)
            self.n_heads = n_heads
            self.head_dim = 64 // n_heads

        def forward(self, sat, geo):
            """
            sat: (B, sat_dim)
            geo: (B, geo_dim)
            Returns: (B, sat_dim) cross-attended satellite embedding
            """
            B = sat.size(0)
            # Unsqueeze for sequence dim (seq_len=1)
            q = self.q_proj(sat).unsqueeze(1)   # (B, 1, 64)
            k = self.k_proj(geo).unsqueeze(1)    # (B, 1, 64)
            v = self.v_proj(geo).unsqueeze(1)    # (B, 1, 64)
            # Scaled dot-product attention
            scale = self.head_dim ** -0.5
            attn  = (q @ k.transpose(-2, -1)) * scale   # (B, 1, 1)
            attn  = F.softmax(attn, dim=-1)
            out   = (attn @ v).squeeze(1)                # (B, 64)
            return self.out_proj(out)                    # (B, sat_dim)


    class ArseniCFM(nn.Module):
        """
        ArseniCFM: Foundation Model for Arsenic Contamination Prediction

        Satellite embeddings (Alpha Earth, frozen) are domain-adapted through
        cross-modal attention with geochemical context, then fused with context
        features for multi-task prediction.

        Input:
            x_sat:  (B, 64)  satellite embedding (from Alpha Earth / fallback)
            x_geo:  (B, n_geo) geochemical features
            x_ctx:  (B, 4)  context [lat, lon, depth, season]

        Output: dict with 4 task predictions
        """
        def __init__(self, geo_in_dim, ctx_in_dim, freeze_satellite=True):
            super().__init__()
            self.freeze_satellite = freeze_satellite

            # Satellite projection (can be frozen after Alpha Earth feature extraction)
            self.sat_proj = nn.Sequential(
                nn.Linear(SAT_EMBED_DIM, SAT_EMBED_DIM),
                nn.LayerNorm(SAT_EMBED_DIM),
                nn.GELU(),
            )

            self.geo_encoder = GeoChemEncoder(geo_in_dim)
            self.ctx_encoder = ContextEncoder(ctx_in_dim)
            self.cross_attn  = CrossModalAttention()

            # Fusion MLP
            self.fusion = nn.Sequential(
                nn.Linear(FUSED_DIM, HIDDEN_DIM),
                nn.LayerNorm(HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
                nn.GELU(),
            )

            fused_out = HIDDEN_DIM // 2

            # Task heads
            self.head_concentration = nn.Linear(fused_out, 1)
            self.head_exceedance    = nn.Linear(fused_out, 1)
            self.head_mechanism     = nn.Linear(fused_out, len(MECHANISM_LABELS))
            self.head_depth_zone    = nn.Linear(fused_out, len(DEPTH_ZONE_LABELS))

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x_sat, x_geo, x_ctx):
            # Satellite: optionally freeze
            if self.freeze_satellite:
                with torch.no_grad():
                    sat_emb = self.sat_proj(x_sat)
            else:
                sat_emb = self.sat_proj(x_sat)

            geo_emb = self.geo_encoder(x_geo)     # (B, 32)
            ctx_emb = self.ctx_encoder(x_ctx)     # (B, 16)

            # Cross-modal attention: satellite attends to geochemistry
            sat_ctx = self.cross_attn(sat_emb, geo_emb)   # (B, 64)

            # Fusion
            fused = torch.cat([sat_ctx, geo_emb, ctx_emb], dim=1)  # (B, 112)
            h = self.fusion(fused)

            return {
                'concentration': self.head_concentration(h).squeeze(-1),
                'exceedance':    torch.sigmoid(self.head_exceedance(h).squeeze(-1)),
                'mechanism':     self.head_mechanism(h),
                'depth_zone':    self.head_depth_zone(h),
            }

        def get_embeddings(self, x_sat, x_geo, x_ctx):
            """Return fused representation (before task heads) for analysis."""
            sat_emb = self.sat_proj(x_sat)
            geo_emb = self.geo_encoder(x_geo)
            ctx_emb = self.ctx_encoder(x_ctx)
            sat_ctx = self.cross_attn(sat_emb, geo_emb)
            fused   = torch.cat([sat_ctx, geo_emb, ctx_emb], dim=1)
            return self.fusion(fused)


    class ArseniCFMLoss(nn.Module):
        """Weighted multi-task loss for ArseniCFM."""
        def __init__(self):
            super().__init__()
            self.w = TASK_WEIGHTS

        def forward(self, preds, targets):
            loss_conc = F.mse_loss(preds['concentration'], targets['concentration'])
            loss_exc  = F.binary_cross_entropy(preds['exceedance'], targets['exceedance'].float())
            loss_mech = F.cross_entropy(preds['mechanism'], targets['mechanism'])
            loss_zone = F.cross_entropy(preds['depth_zone'], targets['depth_zone'])
            total = (self.w['concentration'] * loss_conc +
                     self.w['exceedance']    * loss_exc +
                     self.w['mechanism']     * loss_mech +
                     self.w['depth_zone']    * loss_zone)
            return total, {
                'concentration': loss_conc.item(),
                'exceedance':    loss_exc.item(),
                'mechanism':     loss_mech.item(),
                'depth_zone':    loss_zone.item(),
                'total':         total.item(),
            }


# ─── MODEL FACTORY ────────────────────────────────────────────────────────────

def build_model(geo_in_dim, ctx_in_dim, freeze_satellite=True):
    """Build ArseniCFM (PyTorch) or fallback (sklearn)."""
    if HAS_TORCH:
        model = ArseniCFM(geo_in_dim, ctx_in_dim, freeze_satellite)
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  ArseniCFM (PyTorch): {n_params:,} params | {n_trainable:,} trainable")
        return model
    else:
        print("  ArseniCFM (sklearn fallback): PyTorch not available")
        return ArseniCFM_SklearnFallback()
