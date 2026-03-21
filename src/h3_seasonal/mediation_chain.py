"""
Phase B — Sequential Mediation Analysis (H3.1)
Causal chain: Season → DOC (TDS) → ORP → Fe²⁺ → As
Computes standardised path coefficients and bootstrap indirect effects per depth zone.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

from config import (
    TARGET, DEPTH_ZONE_LABELS, ZONE_COLORS, FIGURES_DIR, TABLES_DIR,
    DPI, ALPHA, N_BOOTSTRAP, RANDOM_STATE, SEASON_WET, SEM_DOC_PROXY,
    SEM_REDOX_MED, SEM_FE_MED
)


def _std(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def _ols_beta(formula, data, predictor):
    """Fit OLS, return coefficient for predictor (0 if fails)."""
    try:
        m = smf.ols(formula, data=data).fit()
        key = f'Q("{predictor}")'
        return m.params.get(key, m.params.get(predictor, 0.0))
    except Exception:
        return 0.0


def mediate_chain(df, zone_label, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    """
    Fit 3-link sequential mediation:
      a1: Season → TDS_std
      a2: TDS_std → ORP_std
      a3: ORP_std → Fe_std
      b:  Fe_std  → As_log
    Bootstrap indirect effect = a1 × a2 × a3 × b.
    """
    needed = ['Season_bin', SEM_DOC_PROXY, SEM_REDOX_MED, SEM_FE_MED, f'{TARGET}_log']
    sub = df[[c for c in needed if c in df.columns]].dropna()
    if len(sub) < 30:
        return None

    # Standardise
    for col in [SEM_DOC_PROXY, SEM_REDOX_MED, SEM_FE_MED]:
        if col in sub.columns:
            sub[col + '_std'] = _std(sub[col])

    doc = SEM_DOC_PROXY + '_std'
    orp = SEM_REDOX_MED + '_std'
    fe  = SEM_FE_MED    + '_std'
    y   = f'{TARGET}_log'

    if not all(c in sub.columns for c in [doc, orp, fe, y]):
        return None

    def _get_coeffs(data):
        a1 = _ols_beta(f'Q("{doc}") ~ Season_bin',        data, 'Season_bin')
        a2 = _ols_beta(f'Q("{orp}") ~ Q("{doc}")',         data, doc)
        a3 = _ols_beta(f'Q("{fe}") ~ Q("{orp}")',          data, orp)
        b  = _ols_beta(f'Q("{y}") ~ Q("{fe}")',             data, fe)
        c_prime = _ols_beta(
            f'Q("{y}") ~ Season_bin + Q("{doc}") + Q("{orp}") + Q("{fe}")',
            data, 'Season_bin'
        )
        return a1, a2, a3, b, c_prime

    a1, a2, a3, b, c_prime = _get_coeffs(sub)
    total_indirect = a1 * a2 * a3 * b

    # Total effect c
    try:
        c_total = smf.ols(f'Q("{y}") ~ Season_bin', data=sub).fit().params['Season_bin']
    except Exception:
        c_total = np.nan

    mediation_prop = total_indirect / (c_total + 1e-9) if pd.notna(c_total) else np.nan

    # Bootstrap
    rng = np.random.default_rng(seed)
    boot_indirect = []
    for _ in range(n_boot):
        idx = rng.choice(len(sub), len(sub), replace=True)
        samp = sub.iloc[idx]
        try:
            ba1, ba2, ba3, bb, _ = _get_coeffs(samp)
            boot_indirect.append(ba1 * ba2 * ba3 * bb)
        except Exception:
            pass

    boot_arr = np.array(boot_indirect)
    ci_lo = np.percentile(boot_arr, 2.5)  if len(boot_arr) else np.nan
    ci_hi = np.percentile(boot_arr, 97.5) if len(boot_arr) else np.nan
    sig = 'YES' if (ci_lo > 0 or ci_hi < 0) else 'NO'

    return {
        'Zone': zone_label, 'N': len(sub),
        'a1_Season_TDS': round(a1, 5),
        'a2_TDS_ORP':    round(a2, 5),
        'a3_ORP_Fe':     round(a3, 5),
        'b_Fe_As':       round(b,  5),
        'c_prime_direct':round(c_prime, 5),
        'c_total':       round(c_total, 5) if pd.notna(c_total) else np.nan,
        'Indirect_effect': round(total_indirect, 6),
        'Boot_CI_lo':    round(ci_lo, 6),
        'Boot_CI_hi':    round(ci_hi, 6),
        'Significant':   sig,
        'Mediation_proportion': round(mediation_prop, 4) if pd.notna(mediation_prop) else np.nan,
    }


def run_mediation_chain(df):
    """Phase B: run 3-link mediation for each depth zone."""
    print("\n" + "█" * 70)
    print(" PHASE B: SEQUENTIAL MEDIATION ANALYSIS — DOC→ORP→Fe→As (H3.1)")
    print("█" * 70)

    rows = []
    for zone in ['ALL'] + DEPTH_ZONE_LABELS:
        sub = df if zone == 'ALL' else df[df['Depth_Zone'] == zone]
        print(f"\n  Zone: {zone} (n={len(sub)})")
        result = mediate_chain(sub, zone)
        if result:
            rows.append(result)
            print(f"    a1={result['a1_Season_TDS']:+.4f}  "
                  f"a2={result['a2_TDS_ORP']:+.4f}  "
                  f"a3={result['a3_ORP_Fe']:+.4f}  "
                  f"b={result['b_Fe_As']:+.4f}")
            print(f"    Indirect={result['Indirect_effect']:+.5f} "
                  f"[{result['Boot_CI_lo']:+.5f},{result['Boot_CI_hi']:+.5f}] "
                  f"sig={result['Significant']}")
            print(f"    Mediation proportion: {result['Mediation_proportion']:.3f}")
        else:
            print(f"    SKIP: insufficient data")

    med_df = pd.DataFrame(rows)
    med_df.to_csv(TABLES_DIR / 'H3_T05_mediation_path_coefficients.csv', index=False)

    # Bootstrap CI table
    ci_df = med_df[['Zone', 'N', 'Indirect_effect', 'Boot_CI_lo', 'Boot_CI_hi',
                     'Significant', 'Mediation_proportion']].copy()
    ci_df.to_csv(TABLES_DIR / 'H3_T06_mediation_bootstrap_CI.csv', index=False)

    _plot_mediation(med_df)
    _plot_path_diagram(med_df)

    print("\n✓ Phase B complete. Outputs: H3_T05, H3_T06, H3_F02, H3_F12")
    return med_df


def _plot_mediation(med_df):
    """H3_F02: Indirect effects with CI by depth zone."""
    if med_df is None or len(med_df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: path coefficients heatmap
    ax = axes[0]
    path_cols = ['a1_Season_TDS', 'a2_TDS_ORP', 'a3_ORP_Fe', 'b_Fe_As', 'c_prime_direct']
    path_labels = ['a₁: Season→TDS', 'a₂: TDS→ORP', 'a₃: ORP→Fe²⁺', 'b: Fe²⁺→As', "c': Direct"]
    matrix = med_df.set_index('Zone')[path_cols].values
    im = ax.imshow(matrix.T, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(med_df))); ax.set_xticklabels(med_df['Zone'], rotation=20, fontsize=9)
    ax.set_yticks(range(len(path_cols))); ax.set_yticklabels(path_labels, fontsize=10)
    plt.colorbar(im, ax=ax, label='Standardised β')
    ax.set_title('Path Coefficients (Standardised β)', fontsize=11)
    for i, row in enumerate(med_df.itertuples()):
        for j, col in enumerate(path_cols):
            val = getattr(row, col)
            ax.text(i, j, f'{val:.3f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(val) > 0.3 else 'black')

    # Right: indirect effects with CI
    ax2 = axes[1]
    x = np.arange(len(med_df))
    ind = med_df['Indirect_effect'].values
    lo  = ind - med_df['Boot_CI_lo'].values
    hi  = med_df['Boot_CI_hi'].values - ind
    colors = ['#d73027' if s == 'YES' else '#aaaaaa' for s in med_df['Significant']]
    ax2.bar(x, ind, color=colors, alpha=0.85)
    ax2.errorbar(x, ind, yerr=[lo, hi], fmt='none', color='black', capsize=5, linewidth=1.5)
    ax2.axhline(0, color='black', lw=1.5, ls='--')
    ax2.set_xticks(x); ax2.set_xticklabels(med_df['Zone'], rotation=20, fontsize=9)
    ax2.set_ylabel('Indirect Effect (a₁×a₂×a₃×b)', fontsize=11)
    ax2.set_title('Bootstrap 95% CI for Indirect Effect\n(Red = significant; Grey = ns)', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('H3 Phase B: Sequential Mediation — DOC→ORP→Fe²⁺→As\nby Depth Zone',
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H3_F02_mediation_results.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H3_F02_mediation_results.png")


def _plot_path_diagram(med_df):
    """H3_F12: Schematic path diagram with standardised coefficients for ALL zone."""
    row = med_df[med_df['Zone'] == 'ALL']
    if len(row) == 0:
        row = med_df.iloc[[0]]
    row = row.iloc[0]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    nodes = [('Season\n(Wet=1)', 0.05, 0.5),
             ('DOC\n(TDS)', 0.25, 0.5),
             ('ORP', 0.45, 0.5),
             ('Fe²⁺', 0.65, 0.5),
             ('As\n(log)', 0.85, 0.5)]
    labels = [row.a1_Season_TDS, row.a2_TDS_ORP, row.a3_ORP_Fe, row.b_Fe_As]
    coeff_pos = [0.15, 0.35, 0.55, 0.75]

    for (name, x, y) in nodes:
        ax.add_patch(plt.Circle((x, y), 0.07, color='steelblue', zorder=3))
        ax.text(x, y, name, ha='center', va='center', fontsize=9,
                color='white', fontweight='bold', zorder=4)

    for i in range(len(nodes) - 1):
        x1, x2 = nodes[i][1] + 0.07, nodes[i+1][1] - 0.07
        ax.annotate('', xy=(x2, 0.5), xytext=(x1, 0.5),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        ax.text(coeff_pos[i], 0.62, f'β={labels[i]:+.3f}', ha='center', fontsize=10,
                color='#d73027' if abs(labels[i]) > 0.1 else 'grey')

    # Direct path
    ax.annotate('', xy=(0.85, 0.35), xytext=(0.05, 0.35),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='grey', linestyle='dashed'))
    ax.text(0.45, 0.25, f"Direct c'={row.c_prime_direct:+.3f}", ha='center', fontsize=9, color='grey')

    # Indirect effect
    ind_txt = (f"Indirect effect = {row.Indirect_effect:+.4f}\n"
               f"95% CI [{row.Boot_CI_lo:+.4f}, {row.Boot_CI_hi:+.4f}]\n"
               f"Significant: {row.Significant} | Mediation prop: {row.Mediation_proportion:.2f}")
    ax.text(0.45, 0.1, ind_txt, ha='center', fontsize=10, color='black',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_title(f'H3 Phase B: Mediation Path Diagram (Zone: {row.Zone}, n={row.N})\n'
                 f'Pathway: Season → DOC → ORP → Fe²⁺ → As',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H3_F12_mediation_path_diagram.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H3_F12_mediation_path_diagram.png")
