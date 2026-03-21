"""
Phase B — Redox Transition Analysis
Tests H2.1: Dissolved oxygen and ORP systematically decrease with depth,
creating distinct redox zones that control As speciation and mobility.
Also tracks the Mn/Fe thermodynamic reduction sequence with depth.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

from config import (
    TARGET, DEPTH_ZONE_LABELS, ZONE_COLORS, WHO_AS_LIMIT,
    FIGURES_DIR, TABLES_DIR, DPI, ALPHA, DEPTH_ZONES
)

REDOX_VARS = ['ORP', 'DO', 'Fe2+', 'Mn2+', 'SO42-', 'HCO3-']


# ─────────────────────────────────────────────
# REDOX CLASSIFICATION
# ─────────────────────────────────────────────

def classify_redox_conditions(df):
    """
    Classify each sample into oxidizing / suboxic / anoxic using ORP and DO.
    Thresholds follow Eh-pH literature (Smedley & Kinniburgh 2002).
    """
    conditions = []
    if 'ORP' in df.columns:
        cond = pd.cut(
            df['ORP'],
            bins=[-2000, 0, 200, 5000],
            labels=['Anoxic', 'Suboxic', 'Oxidizing']
        )
        conditions.append(('ORP', cond))
    if 'DO' in df.columns:
        cond_do = pd.cut(
            df['DO'],
            bins=[-1, 2, 4, 100],
            labels=['Anaerobic', 'Suboxic', 'Oxic']
        )
        conditions.append(('DO', cond_do))

    if conditions:
        # Primary: ORP-based; secondary: DO-based
        df['Redox_Class'] = conditions[0][1]
        mask_null = df['Redox_Class'].isnull()
        if len(conditions) > 1 and mask_null.any():
            df.loc[mask_null, 'Redox_Class'] = conditions[1][1][mask_null]

    return df


def redox_class_by_zone(df):
    """
    Table H2_T05 — Redox class distribution by depth zone.
    Tests H2.1: anoxic fraction increases with depth.
    """
    print("\n" + "="*70)
    print("PHASE B-1: REDOX CLASSIFICATION BY DEPTH ZONE")
    print("="*70)

    df = classify_redox_conditions(df)

    if 'Redox_Class' not in df.columns:
        print("  SKIP: ORP and DO not available.")
        return df, None

    ct = pd.crosstab(df['Depth_Zone'], df['Redox_Class'], margins=True)
    ct_pct = pd.crosstab(df['Depth_Zone'], df['Redox_Class'], normalize='index') * 100

    ct.to_csv(TABLES_DIR / 'H2_T05_redox_class_by_zone_count.csv')
    ct_pct.round(1).to_csv(TABLES_DIR / 'H2_T05_redox_class_by_zone_pct.csv')

    print("\n  Redox class distribution (%):")
    print(ct_pct.round(1).to_string())

    # As mean by redox class and zone
    if TARGET in df.columns and 'Redox_Class' in df.columns:
        as_redox = df.groupby(['Depth_Zone', 'Redox_Class'], observed=True)[TARGET].agg(
            ['mean', 'median', 'count']
        ).round(2)
        as_redox.to_csv(TABLES_DIR / 'H2_T05_as_by_redox_zone.csv')

    return df, ct_pct


# ─────────────────────────────────────────────
# ORP / DO vs DEPTH REGRESSION
# ─────────────────────────────────────────────

def redox_depth_regression(df):
    """
    Test whether ORP and DO decrease significantly with depth.
    Uses Spearman correlation + OLS regression (depth → ORP/DO).
    Table H2_T06.
    """
    print("\n" + "="*70)
    print("PHASE B-2: ORP/DO vs DEPTH REGRESSION")
    print("="*70)

    rows = []
    for var in ['ORP', 'DO']:
        if var not in df.columns or 'Depth' not in df.columns:
            continue
        sub = df[['Depth', var]].dropna()
        if len(sub) < 30:
            continue

        rho, p_sp = stats.spearmanr(sub['Depth'], sub[var])
        r, p_pe   = stats.pearsonr(sub['Depth'], sub[var])

        # OLS
        X = sm.add_constant(sub['Depth'])
        ols = sm.OLS(sub[var], X).fit()

        rows.append({
            'Variable': var,
            'N': len(sub),
            'Spearman_rho': round(rho, 4),
            'Spearman_p': round(p_sp, 6),
            'Pearson_r': round(r, 4),
            'Pearson_p': round(p_pe, 6),
            'OLS_slope': round(ols.params['Depth'], 6),
            'OLS_intercept': round(ols.params['const'], 4),
            'OLS_R2': round(ols.rsquared, 4),
            'OLS_p': round(ols.f_pvalue, 6),
            'Direction': 'Decreases' if rho < 0 else 'Increases',
            'H2.1_Supported': 'YES' if rho < 0 and p_sp < ALPHA else 'NO',
        })
        print(f"  {var}: ρ={rho:.4f}  p={p_sp:.4e}  slope={ols.params['Depth']:.4f}  "
              f"[{'✓ decreases' if rho < 0 else '✗ does not decrease'} with depth]")

    reg_df = pd.DataFrame(rows)
    reg_df.to_csv(TABLES_DIR / 'H2_T06_redox_depth_regression.csv', index=False)
    return reg_df


# ─────────────────────────────────────────────
# Mn / Fe REDUCTION SEQUENCE
# ─────────────────────────────────────────────

def manganese_iron_sequence(df):
    """
    Tests the thermodynamic reduction sequence: Mn²⁺ accumulates before Fe²⁺ with depth.
    Mn reduces at higher Eh than Fe → Mn/Fe ratio should change with depth.
    Table H2_T07.
    """
    print("\n" + "="*70)
    print("PHASE B-3: Mn/Fe THERMODYNAMIC REDUCTION SEQUENCE")
    print("="*70)

    if 'Mn_Fe_ratio' not in df.columns:
        if 'Mn2+' in df.columns and 'Fe2+' in df.columns:
            df['Mn_Fe_ratio'] = df['Mn2+'] / (df['Fe2+'] + 1e-6)
        else:
            print("  SKIP: Mn2+ or Fe2+ not available.")
            return df, None

    seq_rows = []
    for zone in DEPTH_ZONE_LABELS:
        sub = df[df['Depth_Zone'] == zone]
        for col in ['Mn2+', 'Fe2+', 'Mn_Fe_ratio']:
            if col not in sub.columns:
                continue
            vals = sub[col].dropna()
            seq_rows.append({
                'Depth_Zone': zone,
                'Variable': col,
                'Mean': round(vals.mean(), 4),
                'Median': round(vals.median(), 4),
                'Std': round(vals.std(), 4),
            })

    seq_df = pd.DataFrame(seq_rows)
    seq_df.to_csv(TABLES_DIR / 'H2_T07_mn_fe_sequence_by_zone.csv', index=False)

    # Spearman: Mn/Fe vs. Depth
    sub = df[['Depth', 'Mn_Fe_ratio']].dropna()
    if len(sub) > 30:
        rho, p = stats.spearmanr(sub['Depth'], sub['Mn_Fe_ratio'])
        print(f"  Mn/Fe ratio vs Depth: ρ={rho:.4f}  p={p:.4e}")
        print(f"  Interpretation: Mn/Fe {'increases' if rho > 0 else 'decreases'} with depth "
              f"({'✓ consistent with thermodynamic sequence' if rho > 0 else '✗ unexpected'})")

    # Spearman: As vs Mn/Fe by zone
    print("\n  As vs Mn/Fe correlation by depth zone:")
    for zone in DEPTH_ZONE_LABELS:
        sub = df[df['Depth_Zone'] == zone][[TARGET, 'Mn_Fe_ratio']].dropna()
        if len(sub) < 10:
            continue
        rho, p = stats.spearmanr(sub[TARGET], sub['Mn_Fe_ratio'])
        print(f"    {zone:15s}: ρ={rho:+.4f}  p={p:.4e}")

    return df, seq_df


# ─────────────────────────────────────────────
# As SPECIATION SHIFT WITH DEPTH
# ─────────────────────────────────────────────

def arsenic_speciation_by_zone(df):
    """
    Predict As(III)/As(V) dominance from ORP-pH (thermodynamic fields).
    As(III) dominant when Eh < 0 and pH > 7 approximately.
    Table H2_T08.
    """
    print("\n" + "="*70)
    print("PHASE B-4: PREDICTED As SPECIATION BY DEPTH ZONE")
    print("="*70)

    if 'ORP' not in df.columns or 'pH' not in df.columns:
        print("  SKIP: ORP or pH not available.")
        return None

    # Simple thermodynamic classification (boundary at ORP~0 mV, pH~7)
    df['As_Species'] = np.where(
        (df['ORP'] < 100) & (df['pH'] > 6.5), 'As(III)', 'As(V)'
    )

    spec_table = df.groupby(['Depth_Zone', 'As_Species'], observed=True).agg(
        N=(TARGET, 'count'),
        As_mean=(TARGET, 'mean'),
        As_median=(TARGET, 'median'),
    ).round(2)

    spec_table.to_csv(TABLES_DIR / 'H2_T08_as_speciation_by_zone.csv')

    print("\n  Predicted As(III) fraction by depth zone:")
    for zone in DEPTH_ZONE_LABELS:
        sub = df[df['Depth_Zone'] == zone]
        n_iii = (sub['As_Species'] == 'As(III)').sum()
        n_tot = len(sub)
        if n_tot == 0:
            continue
        mean_as_iii = sub.loc[sub['As_Species'] == 'As(III)', TARGET].mean()
        print(f"    {zone:15s}: As(III)={n_iii}/{n_tot} "
              f"({100*n_iii/n_tot:.1f}%)  mean As(III)={mean_as_iii:.2f} µg/L")

    return spec_table


# ─────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────

def plot_redox_depth_profile(df):
    """
    Figure H2_F03 — ORP, DO, Fe²⁺, Mn²⁺ vs. depth (scatter + LOWESS).
    """
    plot_vars = [v for v in ['ORP', 'DO', 'Fe2+', 'Mn2+'] if v in df.columns]
    if not plot_vars:
        return

    fig, axes = plt.subplots(1, len(plot_vars), figsize=(5 * len(plot_vars), 7),
                              sharey=True)
    if len(plot_vars) == 1:
        axes = [axes]

    cmap = plt.cm.RdYlBu_r
    depth_norm = mcolors.Normalize(vmin=df['Depth'].min(), vmax=df['Depth'].max())

    for ax, var in zip(axes, plot_vars):
        sub = df[['Depth', var, 'Depth_Zone']].dropna()
        sc = ax.scatter(sub[var], sub['Depth'],
                        c=sub['Depth'], cmap=cmap, norm=depth_norm,
                        alpha=0.35, s=20, rasterized=True)
        # LOWESS smoother
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(sub['Depth'], sub[var], frac=0.3)
            ax.plot(smoothed[:, 0], smoothed[:, 1], 'k-', lw=2, label='LOWESS')
        except Exception:
            pass

        ax.set_xlabel(var, fontsize=11)
        ax.set_title(var, fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.axhline(30,  color='#d73027', ls=':', lw=1, alpha=0.7, label='30 m')
        ax.axhline(80,  color='#fc8d59', ls=':', lw=1, alpha=0.7, label='80 m')
        ax.axhline(150, color='#4575b4', ls=':', lw=1, alpha=0.7, label='150 m')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel('Depth (m)', fontsize=12)
    plt.colorbar(sc, ax=axes[-1], label='Depth (m)', shrink=0.7)

    plt.suptitle('H2 Phase B: Redox Indicator Profiles with Depth\n'
                 '(Dashed lines = zone boundaries)', fontsize=13)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H2_F03_redox_depth_profile.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H2_F03_redox_depth_profile.png")


def plot_as_vs_orp_by_zone(df):
    """
    Figure H2_F04 — As vs. ORP coloured by depth zone.
    """
    if 'ORP' not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    for zone in DEPTH_ZONE_LABELS:
        sub = df[df['Depth_Zone'] == zone][['ORP', TARGET]].dropna()
        ax.scatter(sub['ORP'], sub[TARGET],
                   c=ZONE_COLORS[zone], label=zone,
                   alpha=0.5, s=25, edgecolors='none')

    ax.axhline(WHO_AS_LIMIT, color='red', ls='--', lw=1.5, label='WHO 10 µg/L')
    ax.axvline(0, color='grey', ls=':', lw=1, alpha=0.7, label='ORP=0 mV')
    ax.set_xlabel('ORP (mV)', fontsize=12)
    ax.set_ylabel('Arsenic (µg/L)', fontsize=12)
    ax.set_title('H2 Phase B: Arsenic vs. ORP by Depth Zone\n'
                 '(Redox control on As mobility)', fontsize=13)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)

    fig.savefig(FIGURES_DIR / 'H2_F04_as_vs_orp_by_zone.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H2_F04_as_vs_orp_by_zone.png")


# ─────────────────────────────────────────────
# PHASE B RUNNER
# ─────────────────────────────────────────────

def run_phase_b(df):
    """Run complete Phase B: Redox transition analysis."""
    print("\n" + "█"*70)
    print(" PHASE B: REDOX TRANSITION ANALYSIS (H2.1)")
    print("█"*70)

    df, redox_dist   = redox_class_by_zone(df)
    reg_df           = redox_depth_regression(df)
    df, seq_df       = manganese_iron_sequence(df)
    spec_table       = arsenic_speciation_by_zone(df)
    plot_redox_depth_profile(df)
    plot_as_vs_orp_by_zone(df)

    print("\n✓ Phase B complete. Outputs:")
    print("  Tables : H2_T05, H2_T06, H2_T07, H2_T08")
    print("  Figures: H2_F03, H2_F04")

    return df, {
        'redox_distribution': redox_dist,
        'regression': reg_df,
        'mn_fe_sequence': seq_df,
        'as_speciation': spec_table,
    }


if __name__ == "__main__":
    from data_loader import load_and_prepare_data
    df = load_and_prepare_data()
    df, results = run_phase_b(df)
