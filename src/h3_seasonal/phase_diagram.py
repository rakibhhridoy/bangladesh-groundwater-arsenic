"""
Phase C — Mechanism Phase Diagram + GMM Clustering (H3.2)
Novel: NDVI-NDWI satellite space as hydrogeochemical phase diagram.
Shannon entropy test for mechanism heterogeneity in Shallow zone.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import levene, mannwhitneyu
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

from config import (
    TARGET, DEPTH_ZONE_LABELS, ZONE_COLORS, SEASON_DRY, SEASON_WET,
    SATELLITE_VARS, FIGURES_DIR, TABLES_DIR, DPI, ALPHA, RANDOM_STATE
)
from entropy_analysis import shannon_entropy, bootstrap_delta_H


# ─── SHALLOW ENTROPY TEST ─────────────────────────────────────────────────────

def shallow_entropy_test(df_delta):
    """
    Test H3.2: Shannon entropy significantly higher in shallow wet season.
    Also test variance increase (Levene) and mutual opposition evidence.
    """
    print("\n" + "=" * 70)
    print("PHASE C-1: SHALLOW ZONE ENTROPY + VARIANCE TEST (H3.2)")
    print("=" * 70)

    rows = []
    for zone in DEPTH_ZONE_LABELS:
        sub = df_delta[df_delta['Depth_Zone'] == zone].dropna(subset=[f'{TARGET}_dry', f'{TARGET}_wet'])
        if len(sub) < 20:
            continue

        dry_as = sub[f'{TARGET}_dry'].values
        wet_as = sub[f'{TARGET}_wet'].values
        delta_as = sub[f'Delta_{TARGET}'].values if f'Delta_{TARGET}' in sub.columns else wet_as - dry_as

        # Shannon entropy change
        dH_mean, dH_lo, dH_hi = bootstrap_delta_H(dry_as, wet_as)

        # Levene test for variance change
        stat, lev_p = levene(dry_as, wet_as)

        # Fraction with negative ΔAs (dilution-dominated)
        frac_negative = (delta_as < 0).mean()

        # Bimodality check: are there two populations?
        try:
            from astropy.stats import bimodality_coefficient as bc
            bimoc = bc(delta_as)
        except Exception:
            # Manual Sarle's bimodality coefficient: (skew²+1)/kurtosis
            from scipy.stats import skew, kurtosis
            sk = skew(delta_as)
            ku = kurtosis(delta_as)
            bimoc = (sk ** 2 + 1) / (ku + 3 + 1e-9)

        rows.append({
            'Depth_Zone': zone, 'N': len(sub),
            'DeltaH_mean': round(dH_mean, 4),
            'DeltaH_CI_lo': round(dH_lo, 4),
            'DeltaH_CI_hi': round(dH_hi, 4),
            'DeltaH_Significant': 'YES' if (dH_lo > 0 or dH_hi < 0) else 'NO',
            'Levene_p': round(lev_p, 6),
            'Variance_Increases_Wet': 'YES' if lev_p < ALPHA and wet_as.var() > dry_as.var() else 'NO',
            'Frac_Negative_DeltaAs': round(frac_negative, 3),
            'Bimodality_Coeff': round(bimoc, 4),
            'H3.2_Prediction': 'Shallow should have highest entropy increase',
        })
        print(f"  {zone:15s}: ΔH={dH_mean:+.4f}[{dH_lo:+.4f},{dH_hi:+.4f}]  "
              f"Levene_p={lev_p:.4f}  f(ΔAs<0)={frac_negative:.2f}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(TABLES_DIR / 'H3_T07_shallow_entropy_bimodality.csv', index=False)
    return df_out


# ─── GMM CLUSTERING ───────────────────────────────────────────────────────────

def gmm_cluster_shallow(df_delta):
    """
    Gaussian mixture model on (NDVI, NDWI, PO43-, ΔAs) for Shallow wells.
    Identifies dilution-dominated vs. desorption-dominated populations.
    """
    print("\n" + "=" * 70)
    print("PHASE C-2: GMM CLUSTERING IN SATELLITE SPACE (H3.2)")
    print("=" * 70)

    shallow = df_delta[df_delta['Depth_Zone'] == 'Shallow'].copy()
    feature_cols = [c for c in ['NDVI_mean', 'NDWI_mean', 'PO43-_dry', f'Delta_{TARGET}']
                    if c in shallow.columns]
    shallow_feat = shallow[feature_cols].dropna()

    if len(shallow_feat) < 20:
        print(f"  SKIP: only {len(shallow_feat)} complete Shallow rows for GMM")
        return None, None

    # Also need NDVI, NDWI for phase diagram
    ndvi_col = 'NDVI_mean' if 'NDVI_mean' in shallow_feat.columns else None
    ndwi_col = 'NDWI_mean' if 'NDWI_mean' in shallow_feat.columns else None

    X = shallow_feat.values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # GMM model selection: n=2 vs n=3 by BIC
    bic_scores = {}
    gmm_models = {}
    for n in [2, 3]:
        gmm = GaussianMixture(n_components=n, covariance_type='full',
                              random_state=RANDOM_STATE, n_init=5)
        gmm.fit(X_sc)
        bic_scores[n] = gmm.bic(X_sc)
        gmm_models[n] = gmm

    best_n = min(bic_scores, key=bic_scores.get)
    best_gmm = gmm_models[best_n]
    clusters = best_gmm.predict(X_sc)

    # Silhouette
    sil = silhouette_score(X_sc, clusters) if len(set(clusters)) > 1 else 0.0

    shallow_feat = shallow_feat.copy()
    shallow_feat['GMM_Cluster'] = clusters
    shallow_feat['Lat'] = shallow['Latitude'].values[:len(shallow_feat)] if 'Latitude' in shallow.columns else np.nan
    shallow_feat['Lon'] = shallow['Longitude'].values[:len(shallow_feat)] if 'Longitude' in shallow.columns else np.nan

    # Label clusters: cluster with highest NDVI & lowest NDWI = desorption-dominated
    cluster_stats = []
    for c in sorted(shallow_feat['GMM_Cluster'].unique()):
        sub_c = shallow_feat[shallow_feat['GMM_Cluster'] == c]
        delta_col = f'Delta_{TARGET}'
        cluster_stats.append({
            'Cluster': c, 'N': len(sub_c),
            'Mean_DeltaAs': round(sub_c[delta_col].mean(), 3) if delta_col in sub_c.columns else np.nan,
            'Mean_NDVI': round(sub_c['NDVI_mean'].mean(), 4) if 'NDVI_mean' in sub_c.columns else np.nan,
            'Mean_NDWI': round(sub_c['NDWI_mean'].mean(), 4) if 'NDWI_mean' in sub_c.columns else np.nan,
            'Mean_PO4': round(sub_c['PO43-_dry'].mean(), 3) if 'PO43-_dry' in sub_c.columns else np.nan,
            'BIC_best_n': best_n, 'Silhouette': round(sil, 4),
        })

    cluster_df = pd.DataFrame(cluster_stats)

    # Assign mechanism labels
    if len(cluster_df) >= 2:
        desorption_cl = cluster_df.loc[cluster_df['Mean_DeltaAs'].idxmax(), 'Cluster']
        cluster_df['Mechanism'] = cluster_df['Cluster'].apply(
            lambda c: 'Desorption-dominated' if c == desorption_cl else 'Dilution-dominated')
    else:
        cluster_df['Mechanism'] = 'Single-cluster'

    cluster_df.to_csv(TABLES_DIR / 'H3_T07b_gmm_cluster_stats.csv', index=False)
    print(f"\n  Best GMM: n={best_n} (BIC={bic_scores[best_n]:.1f}), Silhouette={sil:.4f}")
    print(cluster_df.to_string(index=False))

    # SVM decision boundary in NDVI-NDWI space
    svm_result = None
    if ndvi_col and ndwi_col:
        svm_result = _fit_svm_boundary(shallow_feat, ndvi_col, ndwi_col, clusters)

    _plot_phase_diagram(shallow_feat, cluster_df, ndvi_col, ndwi_col, svm_result)
    _plot_gmm_map(shallow_feat, cluster_df)

    return cluster_df, shallow_feat


def _fit_svm_boundary(df_feat, ndvi_col, ndwi_col, clusters):
    """SVM classifier to find the decision boundary in NDVI-NDWI space."""
    X_svm = df_feat[[ndvi_col, ndwi_col]].values
    # Binary: ΔAs > 0 = desorption; ΔAs <= 0 = dilution
    delta_col = [c for c in df_feat.columns if 'Delta_As' in c or 'Delta_' + 'As' in c]
    if not delta_col:
        return None
    y_bin = (df_feat[delta_col[0]] > 0).astype(int).values

    if len(set(y_bin)) < 2:
        return None

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_svm)
    svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=RANDOM_STATE)
    svm.fit(X_sc, y_bin)
    return {'model': svm, 'scaler': scaler, 'ndvi_col': ndvi_col, 'ndwi_col': ndwi_col}


def _plot_phase_diagram(df_feat, cluster_df, ndvi_col, ndwi_col, svm_result):
    """H3_F06: NDVI × NDWI phase diagram coloured by ΔAs."""
    if ndvi_col is None or ndwi_col is None:
        return

    delta_col = [c for c in df_feat.columns if 'Delta_' in c and 'As' in c and 'umol' not in c]
    if not delta_col:
        return
    delta_col = delta_col[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: ΔAs colour map
    ax = axes[0]
    vabs = np.nanpercentile(np.abs(df_feat[delta_col]), 95)
    sc = ax.scatter(df_feat[ndwi_col], df_feat[ndvi_col],
                    c=df_feat[delta_col], cmap='RdBu_r',
                    vmin=-vabs, vmax=vabs, s=50, alpha=0.8, edgecolors='k', linewidths=0.3)
    plt.colorbar(sc, ax=ax, label='ΔAs (µg/L) = As_wet − As_dry')

    # SVM boundary
    if svm_result:
        ndwi_range = np.linspace(df_feat[ndwi_col].min(), df_feat[ndwi_col].max(), 100)
        ndvi_range = np.linspace(df_feat[ndvi_col].min(), df_feat[ndvi_col].max(), 100)
        XX, YY = np.meshgrid(ndwi_range, ndvi_range)
        grid = np.column_stack([XX.ravel(), YY.ravel()])
        grid_sc = svm_result['scaler'].transform(grid)
        Z = svm_result['model'].predict(grid_sc).reshape(XX.shape)
        ax.contour(XX, YY, Z, levels=[0.5], colors=['black'], linewidths=2, linestyles='--')
        ax.text(ndwi_range[10], ndvi_range[-5], 'Phase Boundary', fontsize=9,
                color='black', fontstyle='italic')

    ax.set_xlabel(f'{ndwi_col} (Recharge proxy ↑)', fontsize=11)
    ax.set_ylabel(f'{ndvi_col} (Agricultural intensity ↑)', fontsize=11)
    ax.set_title('H3.2: Seasonal Mechanism Phase Diagram\n'
                 'Red: desorption dominates; Blue: dilution dominates', fontsize=11)
    ax.axhline(0, color='grey', lw=0.5, ls=':')
    ax.axvline(0, color='grey', lw=0.5, ls=':')
    ax.grid(alpha=0.2)

    # Right: GMM cluster assignments
    ax2 = axes[1]
    cluster_colors = {0: '#d73027', 1: '#4575b4', 2: '#74c476'}
    mech_map = dict(zip(cluster_df['Cluster'], cluster_df['Mechanism']))
    for c in sorted(df_feat['GMM_Cluster'].unique()):
        mask = df_feat['GMM_Cluster'] == c
        label = mech_map.get(c, f'Cluster {c}')
        ax2.scatter(df_feat.loc[mask, ndwi_col], df_feat.loc[mask, ndvi_col],
                    color=cluster_colors.get(c, 'grey'), s=50, alpha=0.8,
                    edgecolors='k', linewidths=0.3, label=label)
    ax2.set_xlabel(f'{ndwi_col} (Recharge proxy ↑)', fontsize=11)
    ax2.set_ylabel(f'{ndvi_col} (Agricultural intensity ↑)', fontsize=11)
    ax2.set_title(f'H3.2: GMM Cluster Assignments (n_components={cluster_df["BIC_best_n"].iloc[0]})\n'
                  f'Silhouette={cluster_df["Silhouette"].iloc[0]:.3f}', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2)

    plt.suptitle('H3 Phase C: Hydrogeochemical Phase Diagram (Shallow Zone)\n'
                 '(First diagram of mechanism opposition in satellite index space)',
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H3_F06_phase_diagram.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H3_F06_phase_diagram.png")


def _plot_gmm_map(df_feat, cluster_df):
    """H3_F07: GMM cluster spatial map."""
    if 'Lat' not in df_feat.columns or df_feat['Lat'].isna().all():
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    cluster_colors = {0: '#d73027', 1: '#4575b4', 2: '#74c476'}
    mech_map = dict(zip(cluster_df['Cluster'], cluster_df['Mechanism']))

    for c in sorted(df_feat['GMM_Cluster'].unique()):
        mask = df_feat['GMM_Cluster'] == c
        label = mech_map.get(c, f'Cluster {c}')
        ax.scatter(df_feat.loc[mask, 'Lon'], df_feat.loc[mask, 'Lat'],
                   color=cluster_colors.get(c, 'grey'), s=60, alpha=0.8,
                   edgecolors='k', linewidths=0.4, label=label)

    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title('H3.2: Spatial Distribution of GMM Mechanism Clusters\n'
                 '(Shallow zone wells only)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H3_F07_gmm_cluster_map.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H3_F07_gmm_cluster_map.png")


def run_phase_diagram(df, df_delta):
    """Run complete Phase C: entropy test + GMM clustering + phase diagram."""
    print("\n" + "█" * 70)
    print(" PHASE C: MECHANISM PHASE DIAGRAM + GMM CLUSTERING (H3.2)")
    print("█" * 70)

    entropy_test_df = shallow_entropy_test(df_delta)
    cluster_df, shallow_feat = gmm_cluster_shallow(df_delta)

    print("\n✓ Phase C complete. Outputs: H3_T07, H3_F06, H3_F07")
    return {'entropy_test': entropy_test_df, 'gmm_clusters': cluster_df,
            'shallow_features': shallow_feat}
