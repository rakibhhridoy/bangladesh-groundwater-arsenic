"""
Embedding Space Analysis — ArseniCFM (H4)

Demonstrates that Alpha Earth geospatial embeddings encode geochemically
meaningful structure WITHOUT any arsenic-specific training.

Analyses:
  1. UMAP visualization coloured by As, mechanism, depth zone, phys zone, season
  2. KMeans clustering on embeddings → compare to known geochemical zones
  3. Embedding similarity vs spatial distance (Mantel test analog)
  4. Embedding similarity vs geochemical similarity
  5. PCA variance explained — information content of embedding space
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from config import (
    FIGURES_DIR, TABLES_DIR, RANDOM_STATE, DPI,
    UMAP_N_COMPONENTS, UMAP_N_NEIGHBORS, UMAP_MIN_DIST,
    N_CLUSTERS, DEPTH_ZONE_LABELS, MECHANISM_LABELS,
    PHYS_ZONE_LABELS, WHO_AS_LIMIT, GEO_FEATURE_COLS
)

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("  [WARN] umap-learn not installed; using PCA for visualization")


# ─── DIMENSIONALITY REDUCTION ─────────────────────────────────────────────────

def reduce_to_2d(embeddings, method='umap'):
    """Reduce (N, 64) embeddings to (N, 2) for visualization."""
    if method == 'umap' and HAS_UMAP:
        reducer = umap.UMAP(
            n_components=UMAP_N_COMPONENTS,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            random_state=RANDOM_STATE,
            low_memory=True,
        )
        coords = reducer.fit_transform(embeddings)
        method_used = 'UMAP'
    else:
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        coords = pca.fit_transform(embeddings)
        method_used = 'PCA'
    print(f"  Reduced to 2D via {method_used}: {coords.shape}")
    return coords, method_used


# ─── UMAP VISUALIZATION ───────────────────────────────────────────────────────

def _scatter_colored(ax, coords, color_vals, cmap, title, label, vmin=None, vmax=None):
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=color_vals,
                    cmap=cmap, s=8, alpha=0.7, linewidths=0,
                    vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, label=label, shrink=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2')
    ax.grid(alpha=0.2)


def _scatter_categorical(ax, coords, labels, unique_labels, colors, title):
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.7,
                   label=lab, color=colors[i], linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2')
    ax.legend(fontsize=7, markerscale=3); ax.grid(alpha=0.2)


def plot_embedding_space(coords, df, method_used='UMAP'):
    """Panel figure: 6 coloured UMAP plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    cmap_pool = ['tab10', 'Set2', 'Set3']

    # 1. As concentration
    as_log = np.log10(df['As'].clip(lower=0.001).values)
    _scatter_colored(axes[0], coords, as_log, 'RdYlBu_r',
                     f'{method_used}: Coloured by log₁₀(As µg/L)',
                     'log₁₀(As)')

    # 2. WHO exceedance
    exc = df['Exceedance'].values
    _scatter_colored(axes[1], coords, exc, 'RdBu_r',
                     f'{method_used}: WHO Exceedance (1=unsafe)',
                     'Exceedance')

    # 3. Mechanism
    mech_vals = df['Mechanism'].values
    mechs_uniq = sorted(df['Mechanism'].unique())
    mech_colors = plt.get_cmap('Set2').colors[:len(mechs_uniq)]
    mech_int = np.array([mechs_uniq.index(m) for m in mech_vals])
    _scatter_colored(axes[2], coords, mech_int, 'Set2',
                     f'{method_used}: Mechanism Type',
                     'Mechanism')
    # Add legend manually
    for i, m in enumerate(mechs_uniq):
        axes[2].scatter([], [], color=mech_colors[i], label=m, s=30)
    axes[2].legend(fontsize=8)

    # 4. Depth zone
    zones = df['Depth_Zone'].values
    zone_uniq = DEPTH_ZONE_LABELS
    zone_colors = ['#d73027', '#fc8d59', '#4575b4', '#313695']
    zone_int = np.array([zone_uniq.index(z) if z in zone_uniq else 0 for z in zones])
    _scatter_colored(axes[3], coords, zone_int, 'RdBu',
                     f'{method_used}: Depth Zone',
                     'Zone')
    for i, z in enumerate(zone_uniq):
        axes[3].scatter([], [], color=zone_colors[i], label=z, s=30)
    axes[3].legend(fontsize=7)

    # 5. Physiographic zone
    phys = df['Phys_Zone'].fillna('Unclassified').values
    phys_uniq = sorted(df['Phys_Zone'].fillna('Unclassified').unique())
    phys_cmap = plt.get_cmap('tab10')
    phys_int = np.array([phys_uniq.index(p) for p in phys])
    _scatter_colored(axes[4], coords, phys_int, 'tab10',
                     f'{method_used}: Physiographic Zone',
                     'Zone')
    for i, z in enumerate(phys_uniq[:7]):
        axes[4].scatter([], [], color=phys_cmap(i/10), label=z[:12], s=30)
    axes[4].legend(fontsize=6)

    # 6. Season
    season_bin = df['Season_bin'].values
    _scatter_colored(axes[5], coords, season_bin, 'coolwarm',
                     f'{method_used}: Season (0=Dry, 1=Wet)',
                     'Season')

    plt.suptitle(f'H4: Alpha Earth Embedding Space ({method_used})\n'
                 'Frozen embeddings reveal geochemically meaningful structure',
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H4_F04_embedding_umap.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H4_F04_embedding_umap.png")


# ─── CLUSTERING ANALYSIS ──────────────────────────────────────────────────────

def cluster_analysis(embeddings, df):
    """KMeans on embeddings; compare clusters to known zones via ARI."""
    print("\n  Clustering embeddings...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    labels_pred = km.fit_predict(embeddings)

    # Adjusted Rand Index vs geochemical labels
    zone_int = df['Depth_Zone'].map(
        {z: i for i, z in enumerate(DEPTH_ZONE_LABELS)}).fillna(0).values
    phys_int = df['Phys_Zone'].fillna('Unclassified').astype('category').cat.codes.values
    exc_int  = df['Exceedance'].values

    ari_zone  = adjusted_rand_score(zone_int, labels_pred)
    ari_phys  = adjusted_rand_score(phys_int, labels_pred)
    ari_exc   = adjusted_rand_score(exc_int,  labels_pred)
    sil_score = silhouette_score(embeddings, labels_pred, sample_size=500)

    rows = [
        {'Comparison': 'Depth Zone', 'ARI': round(ari_zone, 4), 'N_clusters': N_CLUSTERS},
        {'Comparison': 'Phys Zone',  'ARI': round(ari_phys, 4), 'N_clusters': N_CLUSTERS},
        {'Comparison': 'WHO Exceed', 'ARI': round(ari_exc, 4),  'N_clusters': N_CLUSTERS},
        {'Comparison': 'Silhouette', 'ARI': round(sil_score, 4), 'N_clusters': N_CLUSTERS},
    ]
    print(f"  ARI (depth zone): {ari_zone:.4f}")
    print(f"  ARI (phys zone):  {ari_phys:.4f}")
    print(f"  ARI (WHO exceed): {ari_exc:.4f}")
    print(f"  Silhouette:       {sil_score:.4f}")
    return pd.DataFrame(rows), labels_pred


# ─── MANTEL TEST (embedding vs spatial distance) ───────────────────────────────

def mantel_embedding_vs_spatial(embeddings, df, n_sample=500):
    """
    Spearman correlation between pairwise embedding distances and
    pairwise spatial (haversine-approx) distances.
    """
    np.random.seed(RANDOM_STATE)
    idx = np.random.choice(len(df), min(n_sample, len(df)), replace=False)
    emb_sub = embeddings[idx]
    lat = df['Latitude'].values[idx]
    lon = df['Longitude'].values[idx]

    emb_dist = cdist(emb_sub, emb_sub, metric='euclidean')
    spa_dist  = cdist(np.column_stack([lat, lon]),
                      np.column_stack([lat, lon]), metric='euclidean')

    # Upper triangle only
    mask = np.triu(np.ones(len(idx), dtype=bool), k=1)
    rho, p = spearmanr(emb_dist[mask], spa_dist[mask])
    print(f"  Mantel (emb vs spatial): ρ={rho:.4f}  p={p:.4f}")
    return float(rho), float(p)


def mantel_embedding_vs_geochem(embeddings, df, n_sample=500):
    """
    Spearman correlation between pairwise embedding distances and
    pairwise geochemical Euclidean distances.
    """
    from sklearn.preprocessing import StandardScaler
    geo_cols = [c for c in GEO_FEATURE_COLS if c in df.columns]
    X_geo = StandardScaler().fit_transform(df[geo_cols].fillna(0).values)

    np.random.seed(RANDOM_STATE)
    idx = np.random.choice(len(df), min(n_sample, len(df)), replace=False)
    emb_sub = embeddings[idx]
    geo_sub = X_geo[idx]

    emb_dist = cdist(emb_sub, emb_sub)
    geo_dist = cdist(geo_sub, geo_sub)
    mask = np.triu(np.ones(len(idx), dtype=bool), k=1)
    rho, p = spearmanr(emb_dist[mask], geo_dist[mask])
    print(f"  Mantel (emb vs geochem): ρ={rho:.4f}  p={p:.4f}")
    return float(rho), float(p)


# ─── PCA VARIANCE ─────────────────────────────────────────────────────────────

def pca_variance_analysis(embeddings):
    """Explain cumulative variance — information richness of embedding space."""
    pca = PCA(n_components=min(64, embeddings.shape[0]), random_state=RANDOM_STATE)
    pca.fit(embeddings)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n90 = int(np.argmax(cum_var >= 0.90)) + 1
    n95 = int(np.argmax(cum_var >= 0.95)) + 1
    print(f"  PCA: {n90} dims explain 90% variance; {n95} dims explain 95%")
    return pca.explained_variance_ratio_, cum_var, n90, n95


def _plot_pca_variance(var_ratio, cum_var):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(range(1, 21), var_ratio[:20], color='#4575b4', alpha=0.8)
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('H4: PCA of Alpha Earth Embeddings\n(Top 20 components)')
    axes[0].grid(axis='y', alpha=0.3)
    axes[1].plot(range(1, len(cum_var)+1), cum_var * 100, color='#2166ac', lw=2)
    axes[1].axhline(90, color='r', ls='--', lw=1, label='90%')
    axes[1].axhline(95, color='g', ls='--', lw=1, label='95%')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Variance Explained (%)')
    axes[1].set_title('H4: Cumulative Variance — Embedding Information Content')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H4_F05_pca_variance.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H4_F05_pca_variance.png")


# ─── MAIN RUNNER ──────────────────────────────────────────────────────────────

def run_embedding_analysis(embeddings, df):
    """Full embedding analysis suite."""
    print("\n" + "█" * 65)
    print(" PHASE C: EMBEDDING SPACE ANALYSIS (H4)")
    print("█" * 65)

    # 1. Reduce to 2D
    coords, method = reduce_to_2d(embeddings, method='umap')
    np.save(EMBEDDINGS_DIR / 'embedding_2d_coords.npy', coords)
    pd.DataFrame(coords, columns=['dim1', 'dim2']).to_csv(
        TABLES_DIR / 'H4_T04_embedding_2d.csv', index=False)

    # 2. UMAP panel
    plot_embedding_space(coords, df, method)

    # 3. Clustering
    cluster_df, labels_pred = cluster_analysis(embeddings, df)
    cluster_df.to_csv(TABLES_DIR / 'H4_T06_clustering_ari.csv', index=False)

    # 4. Mantel tests
    rho_spa, p_spa = mantel_embedding_vs_spatial(embeddings, df)
    rho_geo, p_geo = mantel_embedding_vs_geochem(embeddings, df)
    mantel_df = pd.DataFrame([
        {'Test': 'Emb vs Spatial distance', 'Spearmans_rho': round(rho_spa, 4), 'p': round(p_spa, 6)},
        {'Test': 'Emb vs Geochem distance', 'Spearmans_rho': round(rho_geo, 4), 'p': round(p_geo, 6)},
    ])
    mantel_df.to_csv(TABLES_DIR / 'H4_T07_mantel_tests.csv', index=False)

    # 5. PCA variance
    var_ratio, cum_var, n90, n95 = pca_variance_analysis(embeddings)
    _plot_pca_variance(var_ratio, cum_var)
    pca_df = pd.DataFrame({'PC': range(1, len(var_ratio)+1),
                           'Var_ratio': var_ratio,
                           'Cum_var': cum_var})
    pca_df.to_csv(TABLES_DIR / 'H4_T08_pca_variance.csv', index=False)

    print("\n✓ Phase C complete. Outputs: H4_T04, T06, T07, T08, F04, F05")
    return {'coords_2d': coords, 'cluster_labels': labels_pred,
            'mantel': mantel_df, 'pca_n90': n90, 'pca_n95': n95}


from config import EMBEDDINGS_DIR
