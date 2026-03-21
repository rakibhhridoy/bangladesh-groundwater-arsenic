"""
Phase C — Geogenic Proxy Construction and Analysis
Constructs a geogenic index from ion ratios + PCA.
Tests H2.3: geogenic processes dominate As release in deep aquifers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from config import (
    TARGET, DEPTH_ZONE_LABELS, ZONE_COLORS, GEO_ZONE_COLORS,
    GEOGENIC_VARS, GEOGENIC_RATIOS, WHO_AS_LIMIT,
    FIGURES_DIR, TABLES_DIR, DPI, ALPHA, PHYSIOGRAPHIC_ZONES
)

# All geogenic proxy columns (base vars + ratios)
RATIO_NAMES = list(GEOGENIC_RATIOS.keys())


def _available(df, cols):
    return [c for c in cols if c in df.columns]


# ─────────────────────────────────────────────
# RATIO CORRELATIONS WITH As
# ─────────────────────────────────────────────

def geogenic_ratio_correlations(df):
    """
    Spearman correlations of each geogenic ratio and base variable with As.
    By depth zone — key test for H2.3.
    Table H2_T09.
    """
    print("\n" + "="*70)
    print("PHASE C-1: GEOGENIC RATIO CORRELATIONS WITH As BY DEPTH ZONE")
    print("="*70)

    proxy_vars = _available(df, RATIO_NAMES + GEOGENIC_VARS)
    rows = []

    for zone in ['ALL'] + DEPTH_ZONE_LABELS:
        sub = df if zone == 'ALL' else df[df['Depth_Zone'] == zone]
        for var in proxy_vars:
            pair = sub[[var, TARGET]].dropna()
            if len(pair) < 15:
                continue
            rho, p = stats.spearmanr(pair[var], pair[TARGET])
            rows.append({
                'Depth_Zone': zone, 'Variable': var,
                'Spearman_rho': round(rho, 4),
                'p_value': round(p, 6),
                'Significant': 'YES' if p < ALPHA else 'NO',
                'N': len(pair),
            })

    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(TABLES_DIR / 'H2_T09_geogenic_corr_by_zone.csv', index=False)

    # Print As correlations for all zones
    print("\n  Spearman ρ (geogenic proxies vs. As) by depth zone:")
    pivot = corr_df.pivot_table(
        index='Variable', columns='Depth_Zone', values='Spearman_rho'
    )
    print(pivot.round(3).to_string())

    return corr_df


# ─────────────────────────────────────────────
# PCA OF GEOGENIC PROXIES
# ─────────────────────────────────────────────

def geogenic_pca(df):
    """
    PCA on geogenic proxy variables to identify end-member compositions.
    PC1 loaded on mineral-weathering indicators → 'Geogenic PC1 score'.
    Table H2_T10, Figure H2_F05.
    """
    print("\n" + "="*70)
    print("PHASE C-2: PCA OF GEOGENIC PROXIES")
    print("="*70)

    pca_vars = _available(df, RATIO_NAMES + ['Al3+', 'Mn2+', 'Fe2+', 'B+', 'Cr3+', 'ORP'])
    sub = df[pca_vars + ['Depth_Zone', TARGET, 'Depth']].dropna()

    if len(sub) < 30 or len(pca_vars) < 3:
        print("  SKIP: insufficient data for PCA.")
        return df, None, None

    X = sub[pca_vars].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    pca = PCA(n_components=min(len(pca_vars), 6))
    scores = pca.fit_transform(X_sc)

    # Loadings table
    loadings = pd.DataFrame(
        pca.components_.T,
        index=pca_vars,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)]
    ).round(4)

    var_explained = pd.Series(
        pca.explained_variance_ratio_ * 100,
        index=[f'PC{i+1}' for i in range(pca.n_components_)]
    ).round(2)

    loadings.loc['Variance_Explained_%'] = var_explained
    loadings.to_csv(TABLES_DIR / 'H2_T10_geogenic_pca_loadings.csv')
    print(f"\n  Variance explained: {var_explained.values}")
    print(f"  PC1 top loadings:\n{loadings['PC1'].drop('Variance_Explained_%').abs().sort_values(ascending=False).head(5)}")

    # Assign PC1 score as 'Geogenic_Index'
    df_pca = sub.copy()
    df_pca['Geogenic_PC1'] = scores[:, 0]
    df_pca['Geogenic_PC2'] = scores[:, 1]
    df = df.copy()
    df.loc[sub.index, 'Geogenic_PC1'] = scores[:, 0]
    df.loc[sub.index, 'Geogenic_PC2'] = scores[:, 1]

    # Geogenic_PC1 vs As
    rho, p = stats.spearmanr(df_pca['Geogenic_PC1'], df_pca[TARGET])
    print(f"\n  Geogenic PC1 vs As: ρ={rho:.4f}  p={p:.4e}")

    # By depth zone
    print("  Geogenic PC1 by depth zone:")
    for zone in DEPTH_ZONE_LABELS:
        s = df_pca[df_pca['Depth_Zone'] == zone]['Geogenic_PC1']
        if len(s) > 0:
            print(f"    {zone:15s}: mean={s.mean():+.3f}  std={s.std():.3f}  n={len(s)}")

    # PCA biplot
    _plot_geogenic_pca_biplot(df_pca, loadings, pca_vars, pca)

    return df, loadings, df_pca


def _plot_geogenic_pca_biplot(df_pca, loadings, pca_vars, pca):
    """Figure H2_F05 — Geogenic PCA biplot coloured by depth zone."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # PC1 vs PC2 coloured by depth zone
    ax = axes[0]
    for zone in DEPTH_ZONE_LABELS:
        sub = df_pca[df_pca['Depth_Zone'] == zone]
        ax.scatter(sub['Geogenic_PC1'], sub['Geogenic_PC2'],
                   c=ZONE_COLORS[zone], label=zone, alpha=0.45, s=25)

    # Loadings arrows
    scale = 3
    for i, var in enumerate(pca_vars):
        ax.annotate('', xy=(scale * pca.components_[0, i],
                             scale * pca.components_[1, i]),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
        ax.text(scale * pca.components_[0, i] * 1.1,
                scale * pca.components_[1, i] * 1.1,
                var, fontsize=8, color='black')

    ax.axhline(0, color='grey', lw=0.5, ls='--')
    ax.axvline(0, color='grey', lw=0.5, ls='--')
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    ax.set_title('Geogenic PCA Biplot — Depth Zone', fontsize=12)
    ax.legend(fontsize=9)

    # PC1 boxplot by zone
    ax2 = axes[1]
    data = [df_pca.loc[df_pca['Depth_Zone'] == z, 'Geogenic_PC1'].dropna()
            for z in DEPTH_ZONE_LABELS]
    bp = ax2.boxplot(data, patch_artist=True, notch=True, labels=DEPTH_ZONE_LABELS)
    for patch, zone in zip(bp['boxes'], DEPTH_ZONE_LABELS):
        patch.set_facecolor(ZONE_COLORS[zone])
        patch.set_alpha(0.75)
    ax2.axhline(0, color='grey', ls='--', lw=1)
    ax2.set_ylabel('Geogenic PC1 Score', fontsize=11)
    ax2.set_title('Geogenic PC1 Score by Depth Zone\n(positive = more geogenic)', fontsize=12)
    ax2.tick_params(axis='x', rotation=20)

    plt.suptitle('H2 Phase C: Geogenic Proxy PCA Analysis', fontsize=13)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H2_F05_geogenic_pca.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H2_F05_geogenic_pca.png")


# ─────────────────────────────────────────────
# PHYSIOGRAPHIC ZONE ANALYSIS
# ─────────────────────────────────────────────

def physiographic_zone_analysis(df):
    """
    As statistics and geogenic proxy means by physiographic zone.
    Table H2_T11.
    """
    print("\n" + "="*70)
    print("PHASE C-3: PHYSIOGRAPHIC ZONE ANALYSIS")
    print("="*70)

    if 'Phys_Zone' not in df.columns:
        print("  SKIP: Phys_Zone not assigned.")
        return None

    agg_dict = {TARGET: ['mean', 'median', 'std', 'count',
                          lambda x: (x > WHO_AS_LIMIT).mean() * 100]}
    for ratio in RATIO_NAMES:
        if ratio in df.columns:
            agg_dict[ratio] = 'mean'

    zone_stats = df.groupby('Phys_Zone').agg(agg_dict).round(3)
    zone_stats.columns = ['_'.join(c).strip('_') for c in zone_stats.columns]
    zone_stats.to_csv(TABLES_DIR / 'H2_T11_physiographic_zone_stats.csv')

    print("\n  As by physiographic zone:")
    for zone in df['Phys_Zone'].unique():
        sub = df[df['Phys_Zone'] == zone][TARGET].dropna()
        if len(sub) == 0:
            continue
        info = PHYSIOGRAPHIC_ZONES.get(zone, {})
        print(f"  {zone:28s}: n={len(sub):4d}  mean={sub.mean():6.2f}  "
              f">{WHO_AS_LIMIT}={100*(sub>WHO_AS_LIMIT).mean():.1f}%  "
              f"[{info.get('geology','')[:40]}]")

    return zone_stats


# ─────────────────────────────────────────────
# GEOGENIC INDEX SCORE
# ─────────────────────────────────────────────

def compute_geogenic_index(df):
    """
    Construct a simple normalised geogenic index (0–1) as weighted sum of
    standardised geogenic proxies. Weights based on PC1 loadings if available,
    else equal weights.
    Table H2_T12.
    """
    print("\n" + "="*70)
    print("PHASE C-4: COMPOSITE GEOGENIC INDEX")
    print("="*70)

    index_vars = _available(df, RATIO_NAMES + ['Al3+', 'Mn2+'])
    if len(index_vars) < 2:
        print("  SKIP: insufficient variables for composite index.")
        return df

    sub = df[index_vars].copy()
    # Standardise each variable 0–1 (min-max)
    for v in index_vars:
        lo, hi = sub[v].quantile(0.01), sub[v].quantile(0.99)
        sub[v] = (sub[v] - lo) / (hi - lo + 1e-9)
        sub[v] = sub[v].clip(0, 1)

    df['Geogenic_Index'] = sub.mean(axis=1)

    rho, p = stats.spearmanr(df['Geogenic_Index'].dropna(),
                              df.loc[df['Geogenic_Index'].notna(), TARGET])
    print(f"  Geogenic Index vs As: ρ={rho:.4f}  p={p:.4e}")

    idx_zone = df.groupby('Depth_Zone', observed=True)['Geogenic_Index'].agg(
        ['mean', 'std', 'count']
    ).round(4)
    idx_zone.to_csv(TABLES_DIR / 'H2_T12_geogenic_index_by_zone.csv')
    print("\n  Geogenic Index by depth zone:")
    print(idx_zone.to_string())

    return df


# ─────────────────────────────────────────────
# FIGURE: GEOGENIC RATIOS HEATMAP BY ZONE
# ─────────────────────────────────────────────

def plot_ratio_heatmap(df):
    """Figure H2_F06 — Mean geogenic ratios by depth zone (heatmap)."""
    ratio_cols = _available(df, RATIO_NAMES)
    if not ratio_cols:
        return

    ratio_means = df.groupby('Depth_Zone', observed=True)[ratio_cols].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    # Normalise each column for visual comparison
    ratio_norm = (ratio_means - ratio_means.min()) / (ratio_means.max() - ratio_means.min() + 1e-9)
    sns.heatmap(ratio_norm.T, annot=ratio_means.T.round(3), fmt='.3f',
                cmap='RdYlBu_r', ax=ax, linewidths=0.5,
                cbar_kws={'label': 'Normalised value (0–1)'})
    ax.set_title('H2 Phase C: Geogenic Ion Ratios by Depth Zone\n'
                 '(Values annotated = raw means; colour = normalised)', fontsize=12)
    ax.set_xlabel('Depth Zone', fontsize=11)
    ax.set_ylabel('Geogenic Ratio', fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H2_F06_geogenic_ratio_heatmap.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H2_F06_geogenic_ratio_heatmap.png")


# ─────────────────────────────────────────────
# PHASE C RUNNER
# ─────────────────────────────────────────────

def run_phase_c(df):
    """Run complete Phase C: Geogenic proxy analysis."""
    print("\n" + "█"*70)
    print(" PHASE C: GEOGENIC PROXY CONSTRUCTION AND ANALYSIS (H2.3)")
    print("█"*70)

    corr_df               = geogenic_ratio_correlations(df)
    df, loadings, df_pca  = geogenic_pca(df)
    phys_stats            = physiographic_zone_analysis(df)
    df                    = compute_geogenic_index(df)
    plot_ratio_heatmap(df)

    print("\n✓ Phase C complete. Outputs:")
    print("  Tables : H2_T09, H2_T10, H2_T11, H2_T12")
    print("  Figures: H2_F05, H2_F06")

    return df, {
        'geogenic_correlations': corr_df,
        'pca_loadings': loadings,
        'physiographic_stats': phys_stats,
    }


if __name__ == "__main__":
    from data_loader import load_and_prepare_data
    df = load_and_prepare_data()
    df, results = run_phase_c(df)
