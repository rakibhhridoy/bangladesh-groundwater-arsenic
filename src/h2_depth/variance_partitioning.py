"""
Phase E — Variance Partitioning
Quantifies how much As variance is explained by geogenic, anthropogenic,
and spatial variable groups — separately for each depth zone.
Tests H2.2 (anthropogenic dominant in shallow) and H2.3 (geogenic dominant in deep).
Method: Hierarchical partial R² (Legendre & Legendre 2012).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

from config import (
    TARGET, DEPTH_ZONE_LABELS, ZONE_COLORS, WHO_AS_LIMIT,
    GEOGENIC_VARS, GEOGENIC_RATIOS, ANTHROPOGENIC_VARS, SATELLITE_VARS,
    SPATIAL_VARS, FIGURES_DIR, TABLES_DIR, DPI, ALPHA,
    N_BOOTSTRAP, RANDOM_STATE
)

# Predefined predictor groups for partitioning
GEOGENIC_PREDICTORS  = ['B+', 'Al3+', 'Mn2+', 'Fe2+', 'Ca2+', 'Cr3+', 'ORP',
                         'B_Cl_ratio', 'Ca_Mg_ratio', 'Mn_Fe_ratio', 'HCO3_Cl_ratio']
ANTHRO_PREDICTORS    = ['PO43-', 'NO3-', 'NDVI_mean', 'EVI_mean']
SPATIAL_PREDICTORS   = ['Latitude', 'Longitude']


def _available(df, cols):
    return [c for c in cols if c in df.columns]


def _adjusted_r2(r2, n, k):
    if n <= k + 1 or k == 0:
        return np.nan
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def _fit_r2(X, y):
    """OLS R² for a feature matrix X against target y."""
    if X.shape[1] == 0 or len(X) < X.shape[1] + 2:
        return 0.0
    Xc = sm.add_constant(X, has_constant='add')
    try:
        ols = sm.OLS(y, Xc).fit()
        return max(ols.rsquared, 0.0)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────
# CORE VARIANCE PARTITIONING
# ─────────────────────────────────────────────

def partition_variance(df, zone='ALL'):
    """
    Compute partial R² for three predictor groups:
      G = Geogenic
      A = Anthropogenic
      S = Spatial

    Following Legendre & Legendre (2012) set algebra:
      [a]  = R²(G)  - R²(G|A,S)   pure geogenic
      [b]  = R²(A)  - R²(A|G,S)   pure anthropogenic
      [c]  = R²(S)  - R²(S|G,A)   pure spatial
      [d]  = shared (joint) fractions
      [e]  = unexplained

    Returns dict with partial fractions.
    """
    sub = df if zone == 'ALL' else df[df['Depth_Zone'] == zone]
    if len(sub) < 30:
        return None

    g_cols = _available(sub, GEOGENIC_PREDICTORS)
    a_cols = _available(sub, ANTHRO_PREDICTORS)
    s_cols = _available(sub, SPATIAL_PREDICTORS)
    all_cols = list(set(g_cols + a_cols + s_cols))

    # Target: log-As if available, else raw
    y_col = f'{TARGET}_log' if f'{TARGET}_log' in sub.columns else TARGET
    complete = sub[all_cols + [y_col]].dropna()
    if len(complete) < 30:
        return None

    y = complete[y_col].values
    scaler = StandardScaler()

    def Xsc(cols):
        if not cols:
            return np.zeros((len(complete), 0))
        return scaler.fit_transform(complete[cols].values)

    G = Xsc(g_cols)
    A = Xsc(a_cols)
    S = Xsc(s_cols)
    GA = np.hstack([G, A])
    GS = np.hstack([G, S])
    AS = np.hstack([A, S])
    GAS = np.hstack([G, A, S])

    r2_G   = _fit_r2(G,   y)
    r2_A   = _fit_r2(A,   y)
    r2_S   = _fit_r2(S,   y)
    r2_GA  = _fit_r2(GA,  y)
    r2_GS  = _fit_r2(GS,  y)
    r2_AS  = _fit_r2(AS,  y)
    r2_GAS = _fit_r2(GAS, y)

    # Pure fractions via set algebra
    pure_G = r2_GAS - r2_AS          # [a]
    pure_A = r2_GAS - r2_GS          # [b]
    pure_S = r2_GAS - r2_GA          # [c]
    joint  = r2_GAS - pure_G - pure_A - pure_S  # [d] shared
    unexpl = max(0, 1 - r2_GAS)      # [e]

    return {
        'zone': zone, 'N': len(complete),
        'R2_total': round(r2_GAS, 4),
        'pure_geogenic': round(max(pure_G, 0), 4),
        'pure_anthropogenic': round(max(pure_A, 0), 4),
        'pure_spatial': round(max(pure_S, 0), 4),
        'joint_shared': round(max(joint, 0), 4),
        'unexplained': round(unexpl, 4),
        'R2_G': round(r2_G, 4),
        'R2_A': round(r2_A, 4),
        'R2_S': round(r2_S, 4),
        'n_geogenic_vars': len(g_cols),
        'n_anthro_vars': len(a_cols),
        'n_spatial_vars': len(s_cols),
        'H2.2_Supported': 'YES' if pure_A > pure_G else 'NO',  # anthropogenic > geogenic
        'H2.3_Supported': 'YES' if pure_G > pure_A else 'NO',  # geogenic > anthropogenic
    }


def run_variance_partitioning(df):
    """
    Variance partitioning for ALL samples and each depth zone separately.
    Table H2_T17.
    """
    print("\n" + "="*70)
    print("PHASE E-1: VARIANCE PARTITIONING — GEOGENIC vs ANTHROPOGENIC vs SPATIAL")
    print("="*70)

    rows = []
    for zone in ['ALL'] + DEPTH_ZONE_LABELS:
        result = partition_variance(df, zone)
        if result:
            rows.append(result)
            print(f"\n  Zone: {zone:15s}  (n={result['N']})")
            print(f"    Pure Geogenic      : {result['pure_geogenic']*100:5.1f}%  "
                  f"[H2.3: {result['H2.3_Supported']}]")
            print(f"    Pure Anthropogenic : {result['pure_anthropogenic']*100:5.1f}%  "
                  f"[H2.2: {result['H2.2_Supported']}]")
            print(f"    Pure Spatial       : {result['pure_spatial']*100:5.1f}%")
            print(f"    Shared             : {result['joint_shared']*100:5.1f}%")
            print(f"    Unexplained        : {result['unexplained']*100:5.1f}%")
            print(f"    Total R²           : {result['R2_total']*100:5.1f}%")

    vp_df = pd.DataFrame(rows)
    vp_df.to_csv(TABLES_DIR / 'H2_T17_variance_partitioning.csv', index=False)
    return vp_df


# ─────────────────────────────────────────────
# BOOTSTRAP CONFIDENCE INTERVALS
# ─────────────────────────────────────────────

def bootstrap_variance_partitioning(df, zone='ALL', n_boot=200):
    """
    Bootstrap 95% CIs for variance partitioning fractions.
    Table H2_T18.
    """
    print(f"\n  Bootstrapping variance partitioning (zone={zone}, B={n_boot})...")
    rng = np.random.RandomState(RANDOM_STATE)

    sub = df if zone == 'ALL' else df[df['Depth_Zone'] == zone]
    g_cols = _available(sub, GEOGENIC_PREDICTORS)
    a_cols = _available(sub, ANTHRO_PREDICTORS)
    s_cols = _available(sub, SPATIAL_PREDICTORS)
    y_col  = f'{TARGET}_log' if f'{TARGET}_log' in sub.columns else TARGET
    all_cols = list(set(g_cols + a_cols + s_cols))
    complete = sub[all_cols + [y_col]].dropna()

    if len(complete) < 30:
        return None

    boot_results = {'pure_G': [], 'pure_A': [], 'pure_S': [], 'joint': []}

    for _ in range(n_boot):
        idx = rng.choice(len(complete), size=len(complete), replace=True)
        sample = complete.iloc[idx]
        y = sample[y_col].values
        scaler = StandardScaler()

        def Xsc(cols):
            return scaler.fit_transform(sample[cols].values) if cols else np.zeros((len(sample), 0))

        G, A, S = Xsc(g_cols), Xsc(a_cols), Xsc(s_cols)
        r2_GAS = _fit_r2(np.hstack([G, A, S]), y)
        r2_AS  = _fit_r2(np.hstack([A, S]), y)
        r2_GS  = _fit_r2(np.hstack([G, S]), y)
        r2_GA  = _fit_r2(np.hstack([G, A]), y)

        boot_results['pure_G'].append(max(r2_GAS - r2_AS, 0))
        boot_results['pure_A'].append(max(r2_GAS - r2_GS, 0))
        boot_results['pure_S'].append(max(r2_GAS - r2_GA, 0))
        boot_results['joint'].append(max(r2_GAS - sum([
            max(r2_GAS - r2_AS, 0),
            max(r2_GAS - r2_GS, 0),
            max(r2_GAS - r2_GA, 0)
        ]), 0))

    ci_rows = []
    for key, label in [('pure_G', 'Geogenic'), ('pure_A', 'Anthropogenic'),
                       ('pure_S', 'Spatial'), ('joint', 'Shared')]:
        vals = np.array(boot_results[key])
        ci_rows.append({
            'Component': label, 'Zone': zone,
            'Mean': round(vals.mean(), 4),
            'CI_2.5': round(np.percentile(vals, 2.5), 4),
            'CI_97.5': round(np.percentile(vals, 97.5), 4),
            'Std': round(vals.std(), 4),
        })
        print(f"    {label:16s}: {vals.mean()*100:5.1f}% "
              f"[{np.percentile(vals,2.5)*100:.1f}–{np.percentile(vals,97.5)*100:.1f}%]")

    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(TABLES_DIR / f'H2_T18_bootstrap_CI_{zone}.csv', index=False)
    return ci_df


# ─────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────

def plot_variance_partitioning(vp_df):
    """
    Figure H2_F09 — Stacked bar chart of variance components by depth zone.
    """
    zones = vp_df['zone'].tolist()
    pure_g = vp_df['pure_geogenic'].values
    pure_a = vp_df['pure_anthropogenic'].values
    pure_s = vp_df['pure_spatial'].values
    joint  = vp_df['joint_shared'].values
    unexpl = vp_df['unexplained'].values

    x = np.arange(len(zones))
    width = 0.6

    fig, ax = plt.subplots(figsize=(12, 7))
    b1 = ax.bar(x, pure_g,           width, label='Pure Geogenic',      color='#d73027', alpha=0.85)
    b2 = ax.bar(x, pure_a,           width, bottom=pure_g,              label='Pure Anthropogenic', color='#4575b4', alpha=0.85)
    b3 = ax.bar(x, pure_s,           width, bottom=pure_g + pure_a,     label='Pure Spatial',      color='#74add1', alpha=0.85)
    b4 = ax.bar(x, joint,            width, bottom=pure_g+pure_a+pure_s,label='Shared (Joint)',    color='#fee090', alpha=0.85)
    b5 = ax.bar(x, unexpl,           width, bottom=pure_g+pure_a+pure_s+joint, label='Unexplained', color='#d9d9d9', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(zones, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Fraction of As Variance (R²)', fontsize=12)
    ax.set_title(
        'H2 Phase E: Variance Partitioning — Geogenic vs. Anthropogenic vs. Spatial\n'
        '(Tests H2.2: Anthropogenic > Geogenic in shallow zones;\n'
        ' H2.3: Geogenic > Anthropogenic in deep zones)',
        fontsize=12
    )
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v*100:.0f}%'))
    ax.grid(axis='y', alpha=0.3)

    # Annotate R² total
    for i, row in vp_df.iterrows():
        ax.text(i, 1.01, f"R²={row['R2_total']*100:.1f}%",
                ha='center', va='bottom', fontsize=8, color='black')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H2_F09_variance_partitioning.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H2_F09_variance_partitioning.png")


def plot_geogenic_vs_anthro(vp_df):
    """
    Figure H2_F10 — Pure geogenic vs. pure anthropogenic R² along depth gradient.
    """
    zones = [z for z in DEPTH_ZONE_LABELS if z in vp_df['zone'].values]
    zone_df = vp_df[vp_df['zone'].isin(zones)].set_index('zone').loc[zones]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(zones))

    ax.plot(x, zone_df['pure_geogenic'] * 100,   'o-', color='#d73027',
            lw=2.5, ms=10, label='Pure Geogenic R²', markerfacecolor='white', markeredgewidth=2)
    ax.plot(x, zone_df['pure_anthropogenic'] * 100, 's-', color='#4575b4',
            lw=2.5, ms=10, label='Pure Anthropogenic R²', markerfacecolor='white', markeredgewidth=2)

    ax.fill_between(x, zone_df['pure_geogenic'] * 100,
                    zone_df['pure_anthropogenic'] * 100,
                    where=zone_df['pure_geogenic'] > zone_df['pure_anthropogenic'],
                    alpha=0.15, color='#d73027', label='Geogenic dominant')
    ax.fill_between(x, zone_df['pure_geogenic'] * 100,
                    zone_df['pure_anthropogenic'] * 100,
                    where=zone_df['pure_anthropogenic'] >= zone_df['pure_geogenic'],
                    alpha=0.15, color='#4575b4', label='Anthropogenic dominant')

    ax.set_xticks(x)
    ax.set_xticklabels(zones, fontsize=11)
    ax.set_xlabel('Depth Zone (shallow → deep)', fontsize=12)
    ax.set_ylabel('Pure Partial R² (%)', fontsize=12)
    ax.set_title(
        'H2 Phase E: Process Dominance Transition Across Depth\n'
        '(Expected: anthropogenic → geogenic with increasing depth)',
        fontsize=13
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0)

    # Crossover annotation
    g_arr = zone_df['pure_geogenic'].values * 100
    a_arr = zone_df['pure_anthropogenic'].values * 100
    crossover = np.where(np.diff(np.sign(g_arr - a_arr)))[0]
    for cx in crossover:
        ax.axvline(cx + 0.5, color='grey', ls='--', lw=1.5, alpha=0.7)
        ax.text(cx + 0.55, ax.get_ylim()[1] * 0.95,
                'Mechanism\ntransition', fontsize=9, color='grey', va='top')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H2_F10_geogenic_vs_anthro_by_depth.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H2_F10_geogenic_vs_anthro_by_depth.png")


# ─────────────────────────────────────────────
# PHASE E RUNNER
# ─────────────────────────────────────────────

def run_phase_e(df):
    """Run complete Phase E: Variance partitioning."""
    print("\n" + "█"*70)
    print(" PHASE E: VARIANCE PARTITIONING (H2.2 & H2.3)")
    print("█"*70)

    vp_df = run_variance_partitioning(df)

    # Bootstrap CIs for overall and shallow/deep zones
    ci_all     = bootstrap_variance_partitioning(df, zone='ALL',     n_boot=200)
    ci_shallow = bootstrap_variance_partitioning(df, zone='Shallow', n_boot=200)
    ci_deep    = bootstrap_variance_partitioning(df, zone='Deep',    n_boot=200)

    plot_variance_partitioning(vp_df)
    plot_geogenic_vs_anthro(vp_df)

    print("\n✓ Phase E complete. Outputs:")
    print("  Tables : H2_T17, H2_T18_*")
    print("  Figures: H2_F09, H2_F10")

    return vp_df, {
        'ci_all': ci_all,
        'ci_shallow': ci_shallow,
        'ci_deep': ci_deep,
    }


if __name__ == "__main__":
    from data_loader import load_and_prepare_data
    df = load_and_prepare_data()
    run_phase_e(df)
