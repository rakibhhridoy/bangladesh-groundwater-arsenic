"""
Phase D — Anthropogenic Proxy Analysis
Tests H2.2: anthropogenic variables (PO4, NO3, NDVI, Season) explain more
As variance in shallow aquifers (<30 m) than in deep aquifers (>150 m).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

from config import (
    TARGET, DEPTH_ZONE_LABELS, ZONE_COLORS, WHO_AS_LIMIT,
    ANTHROPOGENIC_VARS, SATELLITE_VARS,
    FIGURES_DIR, TABLES_DIR, DPI, ALPHA
)

ALL_ANTHRO = ANTHROPOGENIC_VARS + SATELLITE_VARS


def _available(df, cols):
    return [c for c in cols if c in df.columns]


# ─────────────────────────────────────────────
# ANTHROPOGENIC CORRELATIONS BY ZONE
# ─────────────────────────────────────────────

def anthropogenic_correlations_by_zone(df):
    """
    Spearman correlations of anthropogenic proxies with As, stratified by depth zone.
    Key test for H2.2: PO4/NDVI correlations with As should be strongest in Zone 1.
    Table H2_T13.
    """
    print("\n" + "="*70)
    print("PHASE D-1: ANTHROPOGENIC CORRELATIONS WITH As BY DEPTH ZONE")
    print("="*70)

    anthro_vars = _available(df, ALL_ANTHRO)
    rows = []

    for zone in ['ALL'] + DEPTH_ZONE_LABELS:
        sub = df if zone == 'ALL' else df[df['Depth_Zone'] == zone]
        for var in anthro_vars:
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
    corr_df.to_csv(TABLES_DIR / 'H2_T13_anthropogenic_corr_by_zone.csv', index=False)

    print("\n  Spearman ρ (anthropogenic proxies vs. As):")
    pivot = corr_df.pivot_table(
        index='Variable', columns='Depth_Zone', values='Spearman_rho'
    )
    print(pivot.round(3).to_string())

    return corr_df


# ─────────────────────────────────────────────
# PO4 DEPTH GRADIENT
# ─────────────────────────────────────────────

def phosphate_depth_gradient(df):
    """
    Does PO4 concentration decrease with depth?
    This would confirm agricultural surface input attenuating with depth.
    Table H2_T14.
    """
    print("\n" + "="*70)
    print("PHASE D-2: PHOSPHATE DEPTH GRADIENT")
    print("="*70)

    if 'PO43-' not in df.columns:
        print("  SKIP: PO43- not available.")
        return None

    sub = df[['Depth', 'PO43-', TARGET, 'Depth_Zone']].dropna()
    rho, p = stats.spearmanr(sub['Depth'], sub['PO43-'])
    print(f"  PO43- vs Depth: ρ={rho:.4f}  p={p:.4e}")
    print(f"  PO43- {'decreases' if rho < 0 else 'increases'} with depth "
          f"[{'✓ consistent with surface input' if rho < 0 else '✗ unexpected'}]")

    # Mean PO4 by depth zone
    po4_zone = df.groupby('Depth_Zone', observed=True)['PO43-'].agg(
        ['mean', 'median', 'std', 'count']
    ).round(4)
    po4_zone.to_csv(TABLES_DIR / 'H2_T14_po4_by_depth_zone.csv')
    print("\n  Mean PO43- by depth zone:")
    print(po4_zone.to_string())

    # NO3 gradient
    if 'NO3-' in df.columns:
        sub2 = df[['Depth', 'NO3-']].dropna()
        rho2, p2 = stats.spearmanr(sub2['Depth'], sub2['NO3-'])
        print(f"\n  NO3- vs Depth: ρ={rho2:.4f}  p={p2:.4e} "
              f"[{'decreases' if rho2 < 0 else 'increases'} with depth]")

    return po4_zone


# ─────────────────────────────────────────────
# NDVI GRADIENT WITH DEPTH
# ─────────────────────────────────────────────

def ndvi_as_depth_interaction(df):
    """
    Tests whether NDVI predicts As differently by depth zone.
    Low NDVI (high agricultural intensity) → higher As in shallow zones?
    Table H2_T15.
    """
    print("\n" + "="*70)
    print("PHASE D-3: NDVI-As RELATIONSHIP BY DEPTH ZONE")
    print("="*70)

    ndvi_col = 'NDVI_mean' if 'NDVI_mean' in df.columns else None
    if ndvi_col is None:
        print("  SKIP: NDVI_mean not available.")
        return None

    rows = []
    for zone in DEPTH_ZONE_LABELS:
        sub = df[df['Depth_Zone'] == zone][[ndvi_col, TARGET]].dropna()
        if len(sub) < 10:
            continue
        rho, p = stats.spearmanr(sub[ndvi_col], sub[TARGET])
        # Partial correlation controlling for PO4
        if 'PO43-' in df.columns:
            sub3 = df[df['Depth_Zone'] == zone][[ndvi_col, TARGET, 'PO43-']].dropna()
            if len(sub3) >= 15:
                try:
                    from pingouin import partial_corr
                    pc = partial_corr(data=sub3, x=ndvi_col, y=TARGET, covar='PO43-',
                                      method='spearman')
                    partial_rho = pc['r'].values[0]
                    partial_p   = pc['p-val'].values[0]
                except Exception:
                    partial_rho, partial_p = np.nan, np.nan
            else:
                partial_rho, partial_p = np.nan, np.nan
        else:
            partial_rho, partial_p = np.nan, np.nan

        rows.append({
            'Depth_Zone': zone, 'N': len(sub),
            'Spearman_rho_NDVI_As': round(rho, 4),
            'p_value': round(p, 6),
            'Significant': 'YES' if p < ALPHA else 'NO',
            'Partial_rho_NDVI_As_ctrl_PO4': round(partial_rho, 4) if not np.isnan(partial_rho) else np.nan,
            'Partial_p': round(partial_p, 6) if not np.isnan(partial_p) else np.nan,
        })
        print(f"  {zone:15s}: ρ(NDVI,As)={rho:+.4f}  p={p:.4e}")

    ndvi_df = pd.DataFrame(rows)
    ndvi_df.to_csv(TABLES_DIR / 'H2_T15_ndvi_as_by_zone.csv', index=False)
    return ndvi_df


# ─────────────────────────────────────────────
# SEASONAL EFFECT BY ZONE (Irrigation proxy)
# ─────────────────────────────────────────────

def seasonal_effect_by_zone(df):
    """
    Does the seasonal (Dry/Wet) effect on As differ by depth zone?
    Dry season = max irrigation pumping → stronger anthropogenic influence in shallow zones.
    Table H2_T16.
    """
    print("\n" + "="*70)
    print("PHASE D-4: SEASONAL EFFECT ON As BY DEPTH ZONE")
    print("="*70)

    if 'Season' not in df.columns:
        print("  SKIP: Season column not available.")
        return None

    rows = []
    for zone in DEPTH_ZONE_LABELS:
        sub = df[df['Depth_Zone'] == zone]
        dry = sub[sub['Season'] == 'Dry'][TARGET].dropna()
        wet = sub[sub['Season'] == 'Wet'][TARGET].dropna()
        if len(dry) < 5 or len(wet) < 5:
            continue

        stat, p = stats.mannwhitneyu(dry, wet, alternative='two-sided')
        diff_pct = (dry.median() - wet.median()) / (wet.median() + 1e-9) * 100

        rows.append({
            'Depth_Zone': zone,
            'N_Dry': len(dry), 'N_Wet': len(wet),
            'Median_As_Dry': round(dry.median(), 2),
            'Median_As_Wet': round(wet.median(), 2),
            'Diff_pct': round(diff_pct, 1),
            'MannWhitney_U': round(stat, 2),
            'p_value': round(p, 6),
            'Significant': 'YES' if p < ALPHA else 'NO',
        })
        print(f"  {zone:15s}: Dry={dry.median():.2f}  Wet={wet.median():.2f}  "
              f"Δ={diff_pct:+.1f}%  p={p:.4e}")

    seas_df = pd.DataFrame(rows)
    seas_df.to_csv(TABLES_DIR / 'H2_T16_seasonal_effect_by_zone.csv', index=False)
    return seas_df


# ─────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────

def plot_anthropogenic_gradient(df):
    """
    Figure H2_F07 — PO4, NO3, NDVI vs. depth zone (combined panel).
    """
    plot_vars = _available(df, ['PO43-', 'NO3-', 'NDVI_mean', 'EVI_mean'])
    if not plot_vars:
        return

    n_cols = 2
    n_rows = int(np.ceil(len(plot_vars) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = np.array(axes).flatten()

    palette = {z: ZONE_COLORS[z] for z in DEPTH_ZONE_LABELS}

    for i, var in enumerate(plot_vars):
        ax = axes[i]
        plot_df = df[['Depth_Zone', var]].dropna()
        sns.boxplot(data=plot_df, x='Depth_Zone', y=var,
                    palette=palette, order=DEPTH_ZONE_LABELS, ax=ax,
                    flierprops=dict(marker='o', alpha=0.3, ms=3))
        ax.set_title(f'{var} by Depth Zone', fontsize=12, fontweight='bold')
        ax.set_xlabel('Depth Zone', fontsize=10)
        ax.set_ylabel(var, fontsize=10)
        ax.tick_params(axis='x', rotation=20)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('H2 Phase D: Anthropogenic Proxy Gradient Across Depth Zones\n'
                 '(Decreasing gradient expected for surface-applied inputs)', fontsize=13)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H2_F07_anthropogenic_gradient.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H2_F07_anthropogenic_gradient.png")


def plot_po4_as_by_zone(df):
    """
    Figure H2_F08 — PO4 vs. As scatter by depth zone (4-panel).
    """
    if 'PO43-' not in df.columns:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for i, zone in enumerate(DEPTH_ZONE_LABELS):
        ax = axes[i]
        sub = df[df['Depth_Zone'] == zone][['PO43-', TARGET]].dropna()
        ax.scatter(sub['PO43-'], sub[TARGET],
                   c=ZONE_COLORS[zone], alpha=0.45, s=25, edgecolors='none')
        # OLS line
        if len(sub) > 10:
            z = np.polyfit(np.log10(sub['PO43-'] + 1), np.log10(sub[TARGET] + 1e-6), 1)
            xfit = np.linspace(sub['PO43-'].min(), sub['PO43-'].max(), 100)
            yfit = 10 ** (z[0] * np.log10(xfit + 1) + z[1])
            ax.plot(xfit, yfit, 'k-', lw=1.5, alpha=0.8)

        rho, p = stats.spearmanr(sub['PO43-'], sub[TARGET])
        ax.set_xlabel('PO43- (µg/L)', fontsize=10)
        ax.set_ylabel('As (µg/L)', fontsize=10)
        ax.set_title(f'{zone}  (n={len(sub)}, ρ={rho:.3f}, p={p:.3f})',
                     fontsize=11, fontweight='bold')
        ax.axhline(WHO_AS_LIMIT, color='red', ls='--', lw=1, alpha=0.6)
        ax.set_yscale('log')
        ax.grid(alpha=0.2)

    plt.suptitle('H2 Phase D: PO4³⁻ vs. Arsenic by Depth Zone\n'
                 '(Testing agricultural phosphate-As coupling by depth)', fontsize=13)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H2_F08_po4_as_by_zone.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H2_F08_po4_as_by_zone.png")


# ─────────────────────────────────────────────
# PHASE D RUNNER
# ─────────────────────────────────────────────

def run_phase_d(df):
    """Run complete Phase D: Anthropogenic proxy analysis."""
    print("\n" + "█"*70)
    print(" PHASE D: ANTHROPOGENIC PROXY ANALYSIS (H2.2)")
    print("█"*70)

    corr_df    = anthropogenic_correlations_by_zone(df)
    po4_zone   = phosphate_depth_gradient(df)
    ndvi_df    = ndvi_as_depth_interaction(df)
    seas_df    = seasonal_effect_by_zone(df)
    plot_anthropogenic_gradient(df)
    plot_po4_as_by_zone(df)

    print("\n✓ Phase D complete. Outputs:")
    print("  Tables : H2_T13, H2_T14, H2_T15, H2_T16")
    print("  Figures: H2_F07, H2_F08")

    return {
        'anthropogenic_correlations': corr_df,
        'phosphate_gradient': po4_zone,
        'ndvi_analysis': ndvi_df,
        'seasonal_effect': seas_df,
    }


if __name__ == "__main__":
    from data_loader import load_and_prepare_data
    df = load_and_prepare_data()
    run_phase_d(df)
