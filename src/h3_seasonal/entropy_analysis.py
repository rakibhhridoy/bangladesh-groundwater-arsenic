"""
Phase A — Shannon / Rényi Entropy and Mutual Information (H3.8)
Novel diagnostic: seasonal mechanism multiplicity via information theory.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import entropy as scipy_entropy, levene
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')

from config import (
    TARGET, DEPTH_ZONE_LABELS, ZONE_COLORS, SEASON_LABELS, SEASON_COLORS,
    WHO_AS_LIMIT, FIGURES_DIR, TABLES_DIR, DPI, ALPHA, N_BOOTSTRAP,
    RANDOM_STATE, ENTROPY_BINS, RENYI_ALPHAS, ENTROPY_GRID,
    BANGLADESH_BBOX, SEASON_DRY, SEASON_WET
)


# ─── ENTROPY FUNCTIONS ────────────────────────────────────────────────────────

def _hist_probs(vals, bins=ENTROPY_BINS):
    """Normalised histogram → probability vector (zeros removed)."""
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(vals) < 5:
        return np.array([1.0])
    counts, _ = np.histogram(vals, bins=bins)
    probs = counts / counts.sum()
    return probs[probs > 0]


def shannon_entropy(vals, bins=ENTROPY_BINS):
    """Shannon entropy H (bits) for a distribution of values."""
    return float(scipy_entropy(_hist_probs(vals, bins), base=2))


def renyi_entropy(vals, alpha, bins=ENTROPY_BINS):
    """Rényi entropy H_α. α=1 → Shannon."""
    probs = _hist_probs(vals, bins)
    if alpha == 1:
        return float(scipy_entropy(probs, base=2))
    val = np.log2(np.sum(probs ** alpha)) / (1 - alpha)
    return float(val)


def bootstrap_delta_H(dry_vals, wet_vals, n_boot=N_BOOTSTRAP, bins=ENTROPY_BINS, seed=RANDOM_STATE):
    """Bootstrap CI for ΔH = H_wet − H_dry."""
    rng = np.random.default_rng(seed)
    n_dry, n_wet = len(dry_vals), len(wet_vals)
    delta_boot = []
    for _ in range(n_boot):
        H_d = shannon_entropy(rng.choice(dry_vals, n_dry, replace=True), bins)
        H_w = shannon_entropy(rng.choice(wet_vals, n_wet, replace=True), bins)
        delta_boot.append(H_w - H_d)
    arr = np.array(delta_boot)
    return arr.mean(), np.percentile(arr, 2.5), np.percentile(arr, 97.5)


# ─── MUTUAL INFORMATION ───────────────────────────────────────────────────────

def mutual_info_season_as(df_zone):
    """I(As ; Season) for a zone subset."""
    sub = df_zone[['As', 'Season']].dropna()
    if len(sub) < 20:
        return np.nan
    as_binned = pd.cut(sub['As'], bins=ENTROPY_BINS, labels=False)
    season_bin = (sub['Season'] == SEASON_WET).astype(int)
    return mutual_info_score(season_bin, as_binned.fillna(-1))


# ─── PHASE A RUNNER ──────────────────────────────────────────────────────────

def run_entropy_analysis(df):
    """
    H3.8: Shannon entropy, Rényi entropy, mutual information.
    Returns dict of results tables.
    """
    print("\n" + "█" * 70)
    print(" PHASE A: INFORMATION-THEORETIC SEASONAL ANALYSIS (H3.8)")
    print("█" * 70)

    # ── 1. Shannon entropy by zone × season ─────────────────────────────
    print("\n  Computing Shannon entropy Δ by depth zone...")
    entropy_rows = []
    for zone in ['ALL'] + DEPTH_ZONE_LABELS:
        sub = df if zone == 'ALL' else df[df['Depth_Zone'] == zone]
        dry_as = sub[sub['Season'] == SEASON_DRY][TARGET].dropna().values
        wet_as = sub[sub['Season'] == SEASON_WET][TARGET].dropna().values

        if len(dry_as) < 10 or len(wet_as) < 10:
            continue

        H_dry = shannon_entropy(dry_as)
        H_wet = shannon_entropy(wet_as)
        dH_mean, dH_lo, dH_hi = bootstrap_delta_H(dry_as, wet_as)
        significant = 'YES' if (dH_lo > 0 or dH_hi < 0) else 'NO'

        # Levene test for variance change
        _, lev_p = levene(dry_as, wet_as)

        entropy_rows.append({
            'Depth_Zone': zone,
            'N_Dry': len(dry_as), 'N_Wet': len(wet_as),
            'H_Dry_bits': round(H_dry, 4),
            'H_Wet_bits': round(H_wet, 4),
            'DeltaH_mean': round(dH_mean, 4),
            'DeltaH_CI_lo': round(dH_lo, 4),
            'DeltaH_CI_hi': round(dH_hi, 4),
            'DeltaH_Significant': significant,
            'Levene_p': round(lev_p, 6),
            'Interpretation': (
                'Mechanism heterogeneity increases in wet season'
                if dH_mean > 0 else
                'Mechanism homogenizes in wet season'
            ),
        })
        print(f"  {zone:15s}: H_dry={H_dry:.3f}  H_wet={H_wet:.3f}  "
              f"ΔH={dH_mean:+.3f} [{dH_lo:+.3f},{dH_hi:+.3f}]  sig={significant}")

    entropy_df = pd.DataFrame(entropy_rows)
    entropy_df.to_csv(TABLES_DIR / 'H3_T03_shannon_entropy_by_zone.csv', index=False)

    # ── 2. Rényi entropy ──────────────────────────────────────────────────
    print("\n  Computing Rényi entropy H_α...")
    renyi_rows = []
    for zone in ['ALL'] + DEPTH_ZONE_LABELS:
        sub = df if zone == 'ALL' else df[df['Depth_Zone'] == zone]
        for season in [SEASON_DRY, SEASON_WET]:
            vals = sub[sub['Season'] == season][TARGET].dropna().values
            if len(vals) < 10:
                continue
            for alpha in RENYI_ALPHAS:
                renyi_rows.append({
                    'Depth_Zone': zone, 'Season': season, 'Alpha': alpha,
                    'Renyi_H': round(renyi_entropy(vals, alpha), 4),
                    'N': len(vals),
                })
    renyi_df = pd.DataFrame(renyi_rows)
    renyi_df.to_csv(TABLES_DIR / 'H3_T28_renyi_entropy.csv', index=False)

    # ── 3. Mutual information I(As ; Season) ──────────────────────────────
    print("\n  Computing mutual information I(As; Season) by zone...")
    mi_rows = []
    for zone in ['ALL'] + DEPTH_ZONE_LABELS:
        sub = df if zone == 'ALL' else df[df['Depth_Zone'] == zone]
        mi = mutual_info_season_as(sub)
        mi_rows.append({'Depth_Zone': zone, 'Mutual_Info_bits': round(mi, 6) if pd.notna(mi) else np.nan,
                        'N': len(sub), 'Interpretation': 'High: season strongly informs As'
                        if pd.notna(mi) and mi > 0.1 else 'Low: season weakly informs As'})
        print(f"  {zone:15s}: I(As;Season)={mi:.4f} bits")
    mi_df = pd.DataFrame(mi_rows)
    mi_df.to_csv(TABLES_DIR / 'H3_T04_mutual_information.csv', index=False)

    # ── 4. Spatial entropy ratio map ──────────────────────────────────────
    print("\n  Computing spatial entropy ratio map...")
    spatial_rows = _spatial_entropy_map(df)
    spatial_df = pd.DataFrame(spatial_rows)
    if len(spatial_df) > 0:
        spatial_df.to_csv(TABLES_DIR / 'H3_T27_spatial_entropy_ratio.csv', index=False)

    # ── 5. Figures ────────────────────────────────────────────────────────
    _plot_entropy_bars(entropy_df)
    _plot_renyi_profiles(renyi_df)
    _plot_spatial_entropy(spatial_df)

    print("\n✓ Phase A complete. Outputs: H3_T03, H3_T04, H3_T27, H3_T28")
    return {
        'shannon': entropy_df,
        'renyi': renyi_df,
        'mutual_info': mi_df,
        'spatial_entropy': spatial_df,
    }


def _spatial_entropy_map(df):
    """Compute Shannon entropy ratio (H_wet/H_dry) on 0.5° grid cells."""
    rows = []
    lat_bins = np.arange(BANGLADESH_BBOX['lat_min'], BANGLADESH_BBOX['lat_max'], ENTROPY_GRID)
    lon_bins = np.arange(BANGLADESH_BBOX['lon_min'], BANGLADESH_BBOX['lon_max'], ENTROPY_GRID)

    for lat in lat_bins:
        for lon in lon_bins:
            mask = ((df['Latitude'] >= lat) & (df['Latitude'] < lat + ENTROPY_GRID) &
                    (df['Longitude'] >= lon) & (df['Longitude'] < lon + ENTROPY_GRID))
            sub = df[mask]
            if len(sub) < 10:
                continue
            dry_as = sub[sub['Season'] == SEASON_DRY][TARGET].dropna().values
            wet_as = sub[sub['Season'] == SEASON_WET][TARGET].dropna().values
            if len(dry_as) < 5 or len(wet_as) < 5:
                continue
            H_dry = shannon_entropy(dry_as)
            H_wet = shannon_entropy(wet_as)
            ratio = H_wet / (H_dry + 1e-9)
            rows.append({
                'lat_centre': lat + ENTROPY_GRID / 2,
                'lon_centre': lon + ENTROPY_GRID / 2,
                'H_Dry': round(H_dry, 4),
                'H_Wet': round(H_wet, 4),
                'Entropy_Ratio': round(ratio, 4),
                'N_dry': len(dry_as), 'N_wet': len(wet_as),
            })
    return rows


# ─── FIGURES ─────────────────────────────────────────────────────────────────

def _plot_entropy_bars(entropy_df):
    """H3_F04: Shannon entropy ΔH by depth zone with bootstrap CI."""
    if entropy_df is None or len(entropy_df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: H_dry vs H_wet side by side
    ax = axes[0]
    zones = entropy_df['Depth_Zone'].tolist()
    x = np.arange(len(zones))
    w = 0.35
    ax.bar(x - w/2, entropy_df['H_Dry_bits'], w, label='Dry season', color='#d73027', alpha=0.85)
    ax.bar(x + w/2, entropy_df['H_Wet_bits'], w, label='Wet season', color='#4575b4', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(zones, rotation=20, fontsize=9)
    ax.set_ylabel('Shannon Entropy H (bits)', fontsize=11)
    ax.set_title('H3.8: Shannon Entropy by Season and Depth Zone', fontsize=12)
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    # Right: ΔH with CI
    ax2 = axes[1]
    dH = entropy_df['DeltaH_mean'].values
    lo = entropy_df['DeltaH_mean'].values - entropy_df['DeltaH_CI_lo'].values
    hi = entropy_df['DeltaH_CI_hi'].values - entropy_df['DeltaH_mean'].values
    colors = ['#d73027' if v > 0 else '#4575b4' for v in dH]
    ax2.bar(x, dH, yerr=[lo, hi], color=colors, alpha=0.85, capsize=5)
    ax2.axhline(0, color='black', lw=1.5, ls='--')
    ax2.set_xticks(x); ax2.set_xticklabels(zones, rotation=20, fontsize=9)
    ax2.set_ylabel('ΔH = H_wet − H_dry (bits)', fontsize=11)
    ax2.set_title('H3.8: Seasonal Entropy Change (ΔH)\n(Positive = more mechanism heterogeneity in wet season)', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)

    for i, (row, color) in enumerate(zip(entropy_df.itertuples(), colors)):
        ax2.text(i, dH[i] + (0.02 if dH[i] >= 0 else -0.05),
                 '★' if row.DeltaH_Significant == 'YES' else '',
                 ha='center', fontsize=14, color='black')

    plt.suptitle('H3 Phase A: Information Entropy of Seasonal As Distributions\n'
                 '(★ = bootstrap CI excludes zero)', fontsize=13)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H3_F04_shannon_entropy.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H3_F04_shannon_entropy.png")


def _plot_renyi_profiles(renyi_df):
    """H3_F05: Rényi entropy H_α profiles by depth zone."""
    if renyi_df is None or len(renyi_df) == 0:
        return

    zones = DEPTH_ZONE_LABELS
    n_zones = len(zones)
    fig, axes = plt.subplots(1, n_zones, figsize=(4 * n_zones, 5), sharey=False)
    if n_zones == 1:
        axes = [axes]

    for ax, zone in zip(axes, zones):
        sub = renyi_df[renyi_df['Depth_Zone'] == zone]
        for season, color, ls in [(SEASON_DRY, '#d73027', '-'), (SEASON_WET, '#4575b4', '--')]:
            s = sub[sub['Season'] == season]
            if len(s) == 0:
                continue
            ax.plot(s['Alpha'], s['Renyi_H'], color=color, ls=ls,
                    marker='o', label=season, linewidth=2)
        ax.set_xlabel('α (Rényi order)', fontsize=10)
        ax.set_ylabel('H_α (bits)', fontsize=10)
        ax.set_title(f'{zone}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle('H3 Phase A: Rényi Entropy Profiles H_α\n'
                 '(α<1: emphasises rare high-As events; α>1: emphasises bulk)',
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H3_F05_renyi_entropy.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H3_F05_renyi_entropy.png")


def _plot_spatial_entropy(spatial_df):
    """H3_F03: Spatial entropy ratio map."""
    if spatial_df is None or len(spatial_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(spatial_df['lon_centre'], spatial_df['lat_centre'],
                    c=spatial_df['Entropy_Ratio'], cmap='RdBu_r',
                    vmin=0.7, vmax=1.3, s=200, alpha=0.85, edgecolors='k', linewidths=0.5)
    plt.colorbar(sc, ax=ax, label='Entropy Ratio H_wet/H_dry')
    ax.axhline(0, color='k', lw=0)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title('H3.8: Spatial Entropy Ratio Map (H_wet / H_dry)\n'
                 'Red (>1): Mechanism heterogeneity increases in monsoon\n'
                 'Blue (<1): Mechanism homogenizes in monsoon', fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H3_F03_spatial_entropy_ratio.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H3_F03_spatial_entropy_ratio.png")
