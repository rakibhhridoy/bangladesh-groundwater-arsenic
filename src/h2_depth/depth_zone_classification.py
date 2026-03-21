"""
Phase A — Depth Zone Classification and Statistical Characterization
Tests whether As and key geochemical variables differ significantly across depth zones.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scikit_posthocs import posthoc_dunn
import warnings
warnings.filterwarnings('ignore')

from config import (
    TARGET, WHO_AS_LIMIT, DEPTH_ZONES, DEPTH_ZONE_LABELS, ZONE_COLORS,
    IONS, WATER_QUALITY_PARAMS, GEOGENIC_VARS, ANTHROPOGENIC_VARS,
    FIGURES_DIR, TABLES_DIR, DPI, ALPHA
)


def zone_descriptive_statistics(df):
    """
    Compute descriptive statistics for As and all key variables by depth zone.
    Table H2_T02 — Zone characteristics.
    """
    print("\n" + "="*70)
    print("PHASE A-1: DESCRIPTIVE STATISTICS BY DEPTH ZONE")
    print("="*70)

    analyse_vars = [TARGET] + IONS + WATER_QUALITY_PARAMS
    analyse_vars = [v for v in analyse_vars if v in df.columns]

    rows = []
    for zone in DEPTH_ZONE_LABELS:
        sub = df[df['Depth_Zone'] == zone]
        if len(sub) == 0:
            continue
        for var in analyse_vars:
            if var not in sub.columns:
                continue
            col = sub[var].dropna()
            rows.append({
                'Depth_Zone': zone,
                'Variable': var,
                'N': len(col),
                'Mean': col.mean(),
                'Median': col.median(),
                'Std': col.std(),
                'Q1': col.quantile(0.25),
                'Q3': col.quantile(0.75),
                'Min': col.min(),
                'Max': col.max(),
                'CV_pct': (col.std() / col.mean() * 100) if col.mean() != 0 else np.nan,
            })

    stats_df = pd.DataFrame(rows).round(4)
    stats_df.to_csv(TABLES_DIR / 'H2_T02_zone_descriptive_stats.csv', index=False)
    print(f"  Saved: H2_T02_zone_descriptive_stats.csv  ({len(stats_df)} rows)")

    # Focused print: As by zone
    as_stats = stats_df[stats_df['Variable'] == TARGET].set_index('Depth_Zone')
    print("\n  Arsenic by Depth Zone:")
    for zone in DEPTH_ZONE_LABELS:
        if zone in as_stats.index:
            r = as_stats.loc[zone]
            print(f"    {zone:15s}: n={int(r.N):4d}  mean={r.Mean:6.2f}  "
                  f"median={r.Median:5.2f}  std={r.Std:6.2f} µg/L")

    return stats_df


def kruskal_wallis_by_zone(df):
    """
    Kruskal-Wallis test: are As and key variables significantly different across zones?
    Followed by Dunn post-hoc with Bonferroni correction.
    Table H2_T03.
    """
    print("\n" + "="*70)
    print("PHASE A-2: KRUSKAL-WALLIS TEST ACROSS DEPTH ZONES")
    print("="*70)

    test_vars = [TARGET] + [v for v in GEOGENIC_VARS + ANTHROPOGENIC_VARS if v in df.columns]

    kw_rows  = []
    dunn_dfs = {}

    for var in test_vars:
        if var not in df.columns:
            continue
        groups = [
            df.loc[df['Depth_Zone'] == z, var].dropna().values
            for z in DEPTH_ZONE_LABELS
            if len(df.loc[df['Depth_Zone'] == z, var].dropna()) > 0
        ]
        if len(groups) < 2:
            continue

        stat, p = stats.kruskal(*groups)
        sig = 'YES' if p < ALPHA else 'NO'

        # Effect size: eta-squared approximation
        n_total = sum(len(g) for g in groups)
        eta2 = (stat - len(groups) + 1) / (n_total - len(groups))

        kw_rows.append({
            'Variable': var, 'H_statistic': round(stat, 4),
            'p_value': round(p, 6), 'Significant': sig,
            'Eta_squared': round(max(eta2, 0), 4),
            'N_groups': len(groups), 'N_total': n_total,
        })
        print(f"  {var:12s}: H={stat:7.3f}  p={p:.4e}  η²={max(eta2,0):.4f}  [{sig}]")

        # Dunn post-hoc for significant variables
        if p < ALPHA:
            dunn_data = pd.DataFrame({
                z: pd.Series(df.loc[df['Depth_Zone'] == z, var].dropna().values)
                for z in DEPTH_ZONE_LABELS
            })
            try:
                dunn_p = posthoc_dunn(
                    df[df['Depth_Zone'].isin(DEPTH_ZONE_LABELS)][[var, 'Depth_Zone']].dropna(),
                    val_col=var, group_col='Depth_Zone', p_adjust='bonferroni'
                )
                dunn_dfs[var] = dunn_p
            except Exception:
                pass

    kw_df = pd.DataFrame(kw_rows)
    kw_df.to_csv(TABLES_DIR / 'H2_T03_kruskal_wallis_by_zone.csv', index=False)
    print(f"\n  Significant variables (p<{ALPHA}): "
          f"{kw_df[kw_df['Significant']=='YES']['Variable'].tolist()}")

    # Save Dunn tables
    for var, ddf in dunn_dfs.items():
        safe_name = var.replace('+', 'pos').replace('-', 'neg').replace('/', '_')
        ddf.to_csv(TABLES_DIR / f'H2_T03_dunn_{safe_name}.csv')

    return kw_df, dunn_dfs


def plot_as_by_depth_zone(df):
    """
    Figure H2_F01 — Boxplot of As by depth zone + WHO limit line.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Raw As
    ax = axes[0]
    data = [df.loc[df['Depth_Zone'] == z, TARGET].dropna() for z in DEPTH_ZONE_LABELS]
    bp = ax.boxplot(data, patch_artist=True, notch=True, labels=DEPTH_ZONE_LABELS)
    for patch, zone in zip(bp['boxes'], DEPTH_ZONE_LABELS):
        patch.set_facecolor(ZONE_COLORS[zone])
        patch.set_alpha(0.75)
    ax.axhline(WHO_AS_LIMIT, color='red', ls='--', lw=1.5, label='WHO 10 µg/L')
    ax.set_ylabel('Arsenic (µg/L)', fontsize=12)
    ax.set_xlabel('Depth Zone', fontsize=12)
    ax.set_title('Arsenic Concentration by Depth Zone (Raw)', fontsize=13)
    ax.legend()
    ax.tick_params(axis='x', rotation=20)

    # Log As
    ax2 = axes[1]
    log_col = f'{TARGET}_log' if f'{TARGET}_log' in df.columns else TARGET
    data_log = [df.loc[df['Depth_Zone'] == z, log_col].dropna() for z in DEPTH_ZONE_LABELS]
    bp2 = ax2.boxplot(data_log, patch_artist=True, notch=True, labels=DEPTH_ZONE_LABELS)
    for patch, zone in zip(bp2['boxes'], DEPTH_ZONE_LABELS):
        patch.set_facecolor(ZONE_COLORS[zone])
        patch.set_alpha(0.75)
    ax2.axhline(np.log10(WHO_AS_LIMIT), color='red', ls='--', lw=1.5, label='WHO (log scale)')
    ax2.set_ylabel('log₁₀(Arsenic) (µg/L)', fontsize=12)
    ax2.set_xlabel('Depth Zone', fontsize=12)
    ax2.set_title('Arsenic Concentration by Depth Zone (Log₁₀)', fontsize=13)
    ax2.legend()
    ax2.tick_params(axis='x', rotation=20)

    plt.suptitle(
        'H2 Phase A: Arsenic Distribution Across Depth Zones\n'
        '(Shallow=0–30m, Intermediate=30–80m, Medium_Deep=80–150m, Deep>150m)',
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H2_F01_as_by_depth_zone.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H2_F01_as_by_depth_zone.png")


def plot_multivar_by_depth(df):
    """
    Figure H2_F02 — Key geochemical variables vs. depth zone (violin + box).
    """
    plot_vars = [TARGET, 'Fe2+', 'Mn2+', 'PO43-', 'ORP', 'DO']
    plot_vars = [v for v in plot_vars if v in df.columns]

    n_cols = 3
    n_rows = int(np.ceil(len(plot_vars) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = axes.flatten()

    palette = {z: ZONE_COLORS[z] for z in DEPTH_ZONE_LABELS}

    for i, var in enumerate(plot_vars):
        ax = axes[i]
        plot_df = df[['Depth_Zone', var]].dropna()
        sns.violinplot(data=plot_df, x='Depth_Zone', y=var,
                       palette=palette, order=DEPTH_ZONE_LABELS,
                       inner='box', ax=ax, cut=0)
        ax.set_title(var, fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(f'{var} (µg/L or mV or mg/L)', fontsize=9)
        ax.tick_params(axis='x', rotation=25, labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('H2 Phase A: Key Variables by Depth Zone', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H2_F02_variables_by_depth_zone.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H2_F02_variables_by_depth_zone.png")


def who_exceedance_by_zone(df):
    """
    Table H2_T04 — WHO exceedance statistics by depth zone.
    """
    rows = []
    for zone in DEPTH_ZONE_LABELS:
        sub = df[df['Depth_Zone'] == zone][TARGET].dropna()
        lo, hi = DEPTH_ZONES[zone]
        rows.append({
            'Depth_Zone': zone,
            'Depth_Range_m': f'{lo}–{hi}',
            'N': len(sub),
            'N_above_WHO': (sub > WHO_AS_LIMIT).sum(),
            'Pct_above_WHO': round((sub > WHO_AS_LIMIT).mean() * 100, 1),
            'N_above_50': (sub > 50).sum(),
            'Pct_above_50': round((sub > 50).mean() * 100, 1),
            'Mean_As': round(sub.mean(), 2),
            'Median_As': round(sub.median(), 2),
        })

    exceedance_df = pd.DataFrame(rows)
    exceedance_df.to_csv(TABLES_DIR / 'H2_T04_who_exceedance_by_zone.csv', index=False)

    print("\n  WHO Exceedance by Depth Zone:")
    for _, r in exceedance_df.iterrows():
        print(f"    {r['Depth_Zone']:15s} ({r['Depth_Range_m']:7s} m): "
              f"{r['Pct_above_WHO']:5.1f}% >WHO  |  {r['Pct_above_50']:5.1f}% >50 µg/L")

    return exceedance_df


def run_phase_a(df):
    """Run complete Phase A analysis."""
    print("\n" + "█"*70)
    print(" PHASE A: DEPTH ZONE CLASSIFICATION & CHARACTERIZATION")
    print("█"*70)

    stats_df     = zone_descriptive_statistics(df)
    kw_df, dunn  = kruskal_wallis_by_zone(df)
    exceedance   = who_exceedance_by_zone(df)
    plot_as_by_depth_zone(df)
    plot_multivar_by_depth(df)

    print("\n✓ Phase A complete. Outputs:")
    print("  Tables : H2_T02, H2_T03, H2_T04")
    print("  Figures: H2_F01, H2_F02")

    return {
        'zone_stats': stats_df,
        'kruskal_wallis': kw_df,
        'dunn_posthoc': dunn,
        'exceedance': exceedance,
    }


if __name__ == "__main__":
    from data_loader import load_and_prepare_data
    df = load_and_prepare_data()
    run_phase_a(df)
