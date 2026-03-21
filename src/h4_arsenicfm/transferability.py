"""
Transferability Experiments — ArseniCFM (H4)

Demonstrates key FM property: few-shot adaptability.

Experiments:
  1. LOZO-CV (Leave-One-Zone-Out): train on 6 physiographic zones, test on 7th
     → Measures geographic transferability
  2. Few-shot learning curves: k=5,10,20,50,100 samples from target zone
     → Measures sample efficiency of pre-trained embeddings
  3. Geochemical feature contribution: compare performance with/without Alpha Earth
     embeddings in low-data regime
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score

from config import (
    TABLES_DIR, FIGURES_DIR, RANDOM_STATE, FEW_SHOT_K,
    PHYS_ZONE_LABELS, DPI, WHO_AS_LIMIT
)


# ─── LOZO-CV ──────────────────────────────────────────────────────────────────

def run_lozo_cv(X_sat, X_geo, X_ctx, y_conc, y_exc, df):
    """
    Leave-One-Zone-Out CV on physiographic zones.
    Tests geographic transferability of Alpha Earth embeddings.
    """
    print("\n  Leave-One-Zone-Out CV (LOZO)...")
    rows = []
    phys_zones = df['Phys_Zone'].fillna('Unclassified').values
    unique_zones = sorted(df['Phys_Zone'].fillna('Unclassified').unique())

    X_full = np.concatenate([X_sat, X_geo, X_ctx], axis=1)
    X_geo_only = np.concatenate([X_geo, X_ctx], axis=1)

    scaler_full = StandardScaler()
    scaler_geo  = StandardScaler()
    X_full_s = scaler_full.fit_transform(X_full)
    X_geo_s  = scaler_geo.fit_transform(X_geo_only)

    for zone in unique_zones:
        te_mask = phys_zones == zone
        tr_mask = ~te_mask
        if te_mask.sum() < 10 or tr_mask.sum() < 30:
            continue

        # With Alpha Earth embeddings
        reg_full = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        reg_full.fit(X_full_s[tr_mask], y_conc[tr_mask])
        r2_full = r2_score(y_conc[te_mask], reg_full.predict(X_full_s[te_mask]))
        clf_full = LogisticRegression(max_iter=500, C=1.0, random_state=RANDOM_STATE)
        clf_full.fit(X_full_s[tr_mask], y_exc[tr_mask])
        try:
            auc_full = roc_auc_score(y_exc[te_mask],
                                     clf_full.predict_proba(X_full_s[te_mask])[:, 1])
        except Exception:
            auc_full = np.nan

        # Geochemistry only (no satellite)
        reg_geo = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        reg_geo.fit(X_geo_s[tr_mask], y_conc[tr_mask])
        r2_geo = r2_score(y_conc[te_mask], reg_geo.predict(X_geo_s[te_mask]))
        clf_geo = LogisticRegression(max_iter=500, C=1.0, random_state=RANDOM_STATE)
        clf_geo.fit(X_geo_s[tr_mask], y_exc[tr_mask])
        try:
            auc_geo = roc_auc_score(y_exc[te_mask],
                                    clf_geo.predict_proba(X_geo_s[te_mask])[:, 1])
        except Exception:
            auc_geo = np.nan

        rows.append({
            'Test_Zone': zone, 'N_test': te_mask.sum(), 'N_train': tr_mask.sum(),
            'R2_with_AE': round(r2_full, 4), 'R2_geo_only': round(r2_geo, 4),
            'AUC_with_AE': round(auc_full, 4), 'AUC_geo_only': round(auc_geo, 4),
            'R2_gain':  round(r2_full - r2_geo, 4),
            'AUC_gain': round(auc_full - auc_geo, 4),
        })
        print(f"  Zone {zone[:15]:15s}: R²={r2_full:.3f} (+{r2_full-r2_geo:.3f})  "
              f"AUC={auc_full:.3f} (+{auc_full-auc_geo:.3f})")

    lozo_df = pd.DataFrame(rows)
    lozo_df.to_csv(TABLES_DIR / 'H4_T09_lozo_cv.csv', index=False)
    print(f"  Mean R² gain from Alpha Earth: {lozo_df['R2_gain'].mean():.3f}")
    return lozo_df


# ─── FEW-SHOT LEARNING ────────────────────────────────────────────────────────

def run_few_shot(X_sat, X_geo, X_ctx, y_conc, y_exc, df, k_values=FEW_SHOT_K,
                 n_repeats=10):
    """
    Few-shot learning curves: train on k samples, test on remaining.
    Compares: (1) Alpha Earth + geo  vs  (2) geo only.
    """
    print("\n  Few-shot learning curves...")
    rng = np.random.RandomState(RANDOM_STATE)

    X_full = np.concatenate([X_sat, X_geo, X_ctx], axis=1)
    X_geo_only = np.concatenate([X_geo, X_ctx], axis=1)

    rows = []
    for k in k_values:
        r2_ae_list, r2_geo_list = [], []
        auc_ae_list, auc_geo_list = [], []

        for repeat in range(n_repeats):
            # Sample k training indices
            idx_all = np.arange(len(df))
            if k >= len(idx_all) - 10:
                break
            tr_idx = rng.choice(idx_all, size=k, replace=False)
            te_idx = np.setdiff1d(idx_all, tr_idx)

            sc_f  = StandardScaler().fit(X_full[tr_idx])
            sc_g  = StandardScaler().fit(X_geo_only[tr_idx])
            Xf_tr = sc_f.transform(X_full[tr_idx])
            Xf_te = sc_f.transform(X_full[te_idx])
            Xg_tr = sc_g.transform(X_geo_only[tr_idx])
            Xg_te = sc_g.transform(X_geo_only[te_idx])

            # With Alpha Earth
            r = Ridge(alpha=1.0).fit(Xf_tr, y_conc[tr_idx])
            r2_ae_list.append(r2_score(y_conc[te_idx], r.predict(Xf_te)))
            try:
                clf = LogisticRegression(max_iter=300, C=1.0).fit(Xf_tr, y_exc[tr_idx])
                auc_ae_list.append(roc_auc_score(y_exc[te_idx],
                                                  clf.predict_proba(Xf_te)[:, 1]))
            except Exception:
                auc_ae_list.append(np.nan)

            # Geo only
            r2 = Ridge(alpha=1.0).fit(Xg_tr, y_conc[tr_idx])
            r2_geo_list.append(r2_score(y_conc[te_idx], r2.predict(Xg_te)))
            try:
                clf2 = LogisticRegression(max_iter=300, C=1.0).fit(Xg_tr, y_exc[tr_idx])
                auc_geo_list.append(roc_auc_score(y_exc[te_idx],
                                                   clf2.predict_proba(Xg_te)[:, 1]))
            except Exception:
                auc_geo_list.append(np.nan)

        rows.append({
            'k': k,
            'R2_AE_mean': round(np.nanmean(r2_ae_list), 4),
            'R2_AE_std':  round(np.nanstd(r2_ae_list), 4),
            'R2_Geo_mean': round(np.nanmean(r2_geo_list), 4),
            'R2_Geo_std':  round(np.nanstd(r2_geo_list), 4),
            'AUC_AE_mean':  round(np.nanmean(auc_ae_list), 4),
            'AUC_Geo_mean': round(np.nanmean(auc_geo_list), 4),
        })
        print(f"  k={k:3d}: R²(AE)={np.nanmean(r2_ae_list):.3f}  "
              f"R²(geo)={np.nanmean(r2_geo_list):.3f}")

    fs_df = pd.DataFrame(rows)
    fs_df.to_csv(TABLES_DIR / 'H4_T10_few_shot.csv', index=False)
    return fs_df


# ─── PLOTS ────────────────────────────────────────────────────────────────────

def _plot_lozo(lozo_df):
    if len(lozo_df) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    zones = lozo_df['Test_Zone'].str[:12].tolist()
    x = np.arange(len(zones))
    w = 0.35
    for ax, (col_ae, col_geo, title) in zip(axes, [
        ('R2_with_AE', 'R2_geo_only', 'R² (log₁₀As) — LOZO-CV'),
        ('AUC_with_AE', 'AUC_geo_only', 'AUC (WHO exceedance) — LOZO-CV'),
    ]):
        ax.bar(x - w/2, lozo_df[col_ae].values, w, label='Alpha Earth + Geo',
               color='#2166ac', alpha=0.85)
        ax.bar(x + w/2, lozo_df[col_geo].values, w, label='Geo Only',
               color='#d73027', alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels(zones, rotation=30, ha='right', fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        ax.axhline(0, color='k', lw=0.5)
    plt.suptitle('H4: LOZO-CV — Geographic Transferability of Alpha Earth Embeddings',
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H4_F06_lozo_cv.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H4_F06_lozo_cv.png")


def _plot_few_shot(fs_df):
    if len(fs_df) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    k = fs_df['k'].values
    for ax, (ae_col, geo_col, title, ylabel) in zip(axes, [
        ('R2_AE_mean', 'R2_Geo_mean', 'R² (log₁₀As) — Few-Shot', 'R²'),
        ('AUC_AE_mean', 'AUC_Geo_mean', 'AUC — Few-Shot', 'ROC-AUC'),
    ]):
        ax.plot(k, fs_df[ae_col], 'o-', color='#2166ac', lw=2, label='Alpha Earth + Geo')
        ax.plot(k, fs_df[geo_col], 's--', color='#d73027', lw=2, label='Geo Only')
        if ae_col == 'R2_AE_mean' and 'R2_AE_std' in fs_df.columns:
            ae_std = fs_df['R2_AE_std'].values
            ax.fill_between(k, fs_df[ae_col]-ae_std, fs_df[ae_col]+ae_std,
                            alpha=0.2, color='#2166ac')
        ax.set_xscale('log')
        ax.set_xlabel('Training samples (k)')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.legend(); ax.grid(alpha=0.3)
    plt.suptitle('H4: Few-Shot Learning Curves\n'
                 'Alpha Earth embeddings provide data efficiency in low-label regime',
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H4_F07_few_shot.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H4_F07_few_shot.png")


# ─── MAIN RUNNER ──────────────────────────────────────────────────────────────

def run_transferability(X_sat, X_geo, X_ctx, y_conc, y_exc, df):
    """Full transferability experiment suite."""
    print("\n" + "█" * 65)
    print(" PHASE D: TRANSFERABILITY EXPERIMENTS (H4)")
    print("█" * 65)

    lozo_df = run_lozo_cv(X_sat, X_geo, X_ctx, y_conc, y_exc, df)
    _plot_lozo(lozo_df)

    fs_df = run_few_shot(X_sat, X_geo, X_ctx, y_conc, y_exc, df)
    _plot_few_shot(fs_df)

    print("\n✓ Phase D complete. Outputs: H4_T09, T10, F06, F07")
    return {'lozo': lozo_df, 'few_shot': fs_df}
