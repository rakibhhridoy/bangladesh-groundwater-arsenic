"""
H2 Supplementary: Continuous Depth Spline Analysis
Tests whether the 80m mechanism crossover is a sharp threshold or gradual transition.
Uses GAM (via pygam or statsmodels lowess) and rolling-window partial correlations.

Outputs:
  tables/H2_T_spline_crossover.csv
  figures/H2_F_depth_spline.png
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import TARGET, FIGURES_DIR, TABLES_DIR, DPI

DATA = Path(__file__).parent.parent / 'hypo1' / 'output' / 'data_phase1_integrated.csv'


def rolling_partial_corr(df, target, predictor, covariate, depth_col='Depth',
                          window=80, step=5, min_n=20):
    """
    Rolling-window partial Spearman correlation of target ~ predictor
    controlling for covariate, along continuous depth.
    Returns DataFrame with [depth_mid, rho, p, n].
    """
    df = df[[depth_col, target, predictor, covariate]].dropna()
    depths = np.arange(df[depth_col].min(), df[depth_col].max() - window, step)
    rows = []
    for d in depths:
        sub = df[(df[depth_col] >= d) & (df[depth_col] < d + window)]
        if len(sub) < min_n:
            continue
        # Partial correlation: residualise predictor and target on covariate
        from numpy.polynomial import polynomial as P
        def residualise(y, x):
            mask = np.isfinite(y) & np.isfinite(x)
            if mask.sum() < 5:
                return y
            coef = np.polyfit(x[mask], y[mask], 1)
            return y - np.polyval(coef, x)

        y_res = residualise(sub[target].values, sub[covariate].values)
        x_res = residualise(sub[predictor].values, sub[covariate].values)
        rho, p = spearmanr(x_res, y_res)
        rows.append({'depth_mid': d + window / 2, 'rho': rho, 'p': p, 'n': len(sub)})
    return pd.DataFrame(rows)


def run_depth_spline():
    print("\n" + "="*65)
    print("H2 CONTINUOUS DEPTH SPLINE ANALYSIS")
    print("="*65)

    df = pd.read_csv(DATA)
    df = df[(df['Latitude'] >= 20.5) & (df['Longitude'] >= 87.8)].copy()

    # Log-transform As
    df['As_log'] = np.log10(df[TARGET].clip(lower=0.01))

    # Filter to valid depth & mechanism proxies
    needed = ['Depth', TARGET, 'As_log', 'PO43-', 'Fe2+', 'ORP', 'pH', 'HCO3-']
    df = df[needed].dropna()
    print(f"  Valid samples: {len(df)}")
    print(f"  Depth range: {df['Depth'].min():.0f}–{df['Depth'].max():.0f} m")

    # ── 1. Rolling partial correlations: PO4 and Fe vs depth ─────────────────
    print("\n  Computing rolling partial correlations (window=80m, step=5m)...")
    po4_roll = rolling_partial_corr(df, 'As_log', 'PO43-', 'Fe2+', window=80, step=5)
    fe_roll  = rolling_partial_corr(df, 'As_log', 'Fe2+',  'PO43-', window=80, step=5)
    print(f"    PO4 rolling: {len(po4_roll)} depth windows")
    print(f"    Fe  rolling: {len(fe_roll)} depth windows")

    # ── 2. LOWESS smoothed As vs depth ────────────────────────────────────────
    df_sorted = df.sort_values('Depth')
    as_smooth = lowess(df_sorted['As_log'], df_sorted['Depth'], frac=0.3)

    # ── 3. Find crossover depth ───────────────────────────────────────────────
    merged = po4_roll.merge(fe_roll, on='depth_mid', suffixes=('_po4', '_fe'))
    merged['dominance'] = merged['rho_po4'] - merged['rho_fe']
    # Crossover = depth where dominance changes sign
    crossovers = []
    for i in range(len(merged) - 1):
        if merged['dominance'].iloc[i] * merged['dominance'].iloc[i+1] < 0:
            d1 = merged['depth_mid'].iloc[i]
            d2 = merged['depth_mid'].iloc[i+1]
            crossovers.append((d1 + d2) / 2)

    crossover_depth = crossovers[0] if crossovers else np.nan
    print(f"\n  Crossover depth (PO4 ρ = Fe ρ): {crossover_depth:.1f} m")
    print(f"  (Existing hard zone boundary: 80 m)")

    # ── 4. Zone-averaged partial correlations (for comparison) ────────────────
    zone_bins = [0, 30, 80, 150, 999]
    zone_labels = ['Shallow\n(<30m)', 'Intermediate\n(30–80m)',
                   'Medium-Deep\n(80–150m)', 'Deep\n(>150m)']
    df['zone_bin'] = pd.cut(df['Depth'], bins=zone_bins, labels=zone_labels)

    zone_rows = []
    for z in zone_labels:
        sub = df[df['zone_bin'] == z]
        if len(sub) < 10:
            continue
        def partial_rho(y, x, cov):
            def resid(a, b):
                mask = np.isfinite(a) & np.isfinite(b)
                if mask.sum() < 5: return a
                c = np.polyfit(b[mask], a[mask], 1)
                return a - np.polyval(c, b)
            return spearmanr(resid(y.values, cov.values),
                             resid(x.values, cov.values))

        rho_po4, p_po4 = partial_rho(sub['As_log'], sub['PO43-'], sub['Fe2+'])
        rho_fe,  p_fe  = partial_rho(sub['As_log'], sub['Fe2+'],  sub['PO43-'])
        zone_rows.append({
            'Zone': z.replace('\n', ' '), 'n': len(sub),
            'PO4_partial_rho': round(rho_po4, 3), 'PO4_p': round(p_po4, 4),
            'Fe_partial_rho':  round(rho_fe, 3),  'Fe_p':  round(p_fe, 4),
            'Dominant': 'PO4' if rho_po4 > rho_fe else 'Fe',
        })
    zone_df = pd.DataFrame(zone_rows)
    print("\n  Zone-averaged partial correlations:")
    print(zone_df.to_string(index=False))

    # ── 5. Save table ─────────────────────────────────────────────────────────
    out_tbl = TABLES_DIR / 'H2_T_spline_crossover.csv'
    spline_tbl = merged[['depth_mid', 'rho_po4', 'p_po4', 'n_po4',
                          'rho_fe', 'p_fe', 'n_fe', 'dominance']].copy()
    spline_tbl['crossover_flag'] = False
    if crossovers:
        spline_tbl.loc[
            (spline_tbl['depth_mid'] >= crossovers[0] - 5) &
            (spline_tbl['depth_mid'] <= crossovers[0] + 5), 'crossover_flag'
        ] = True
    spline_tbl.to_csv(out_tbl, index=False)
    print(f"\n  Table saved: {out_tbl}")

    # ── 6. Figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle('H2: Mechanism Shift — Continuous Depth Analysis\n'
                 '(Rolling 80m windows, step 5m)',
                 fontsize=13, fontweight='bold')

    # Panel A: Rolling partial correlations vs depth
    ax = axes[0]
    ax.plot(po4_roll['rho'], po4_roll['depth_mid'], 'o-',
            color='#e74c3c', ms=4, lw=1.5, label='PO₄ partial ρ (vs Fe)')
    ax.plot(fe_roll['rho'],  fe_roll['depth_mid'],  's-',
            color='#2980b9', ms=4, lw=1.5, label='Fe partial ρ (vs PO₄)')
    ax.axvline(0, color='gray', lw=0.8, linestyle=':')
    if not np.isnan(crossover_depth):
        ax.axhline(crossover_depth, color='#27ae60', lw=2.0, linestyle='--',
                   label=f'Crossover: {crossover_depth:.0f} m')
    ax.axhline(80, color='black', lw=1.5, linestyle='--', alpha=0.5,
               label='Hard zone boundary: 80 m')
    ax.set_xlabel('Partial Spearman ρ with log(As)', fontsize=11)
    ax.set_ylabel('Well Depth (m)', fontsize=11)
    ax.invert_yaxis()
    ax.set_title('Rolling Partial Correlations\n(window=80m)', fontsize=11)
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)

    # Panel B: PO4 dominance score vs depth
    ax2 = axes[1]
    merged_sorted = merged.sort_values('depth_mid')
    colors = ['#e74c3c' if d > 0 else '#2980b9' for d in merged_sorted['dominance']]
    ax2.barh(merged_sorted['depth_mid'], merged_sorted['dominance'],
             height=4, color=colors, alpha=0.7)
    ax2.axvline(0, color='black', lw=1.5)
    if not np.isnan(crossover_depth):
        ax2.axhline(crossover_depth, color='#27ae60', lw=2, linestyle='--',
                    label=f'Crossover: {crossover_depth:.0f} m')
    ax2.axhline(80, color='black', lw=1.5, linestyle='--', alpha=0.5)
    ax2.set_xlabel('PO₄ dominance score\n(ρ_PO4 − ρ_Fe)', fontsize=11)
    ax2.set_ylabel('Well Depth (m)', fontsize=11)
    ax2.invert_yaxis()
    ax2.set_title('Mechanism Dominance by Depth\n(+ve = PO₄, −ve = Fe)', fontsize=11)
    # Shade PO4 vs Fe zones
    ax2.axvspan(-1, 0, alpha=0.07, color='#2980b9')
    ax2.axvspan(0, 1, alpha=0.07, color='#e74c3c')
    ax2.text(0.25, 180, 'Fe dominant', color='#2980b9', fontsize=9, ha='center')
    ax2.text(-0.25, 20, 'PO₄ dominant', color='#e74c3c', fontsize=9, ha='center')
    ax2.legend(fontsize=8.5)
    ax2.grid(True, alpha=0.3)

    # Panel C: LOWESS As vs depth
    ax3 = axes[2]
    ax3.scatter(df['As_log'], df['Depth'], alpha=0.25, s=6, color='gray')
    ax3.plot(as_smooth[:, 1], as_smooth[:, 0], 'r-', lw=2.5, label='LOWESS smooth')
    ax3.axhline(80, color='black', lw=1.5, linestyle='--', alpha=0.7,
                label='80 m boundary')
    if not np.isnan(crossover_depth):
        ax3.axhline(crossover_depth, color='#27ae60', lw=2, linestyle='--',
                    label=f'Spline crossover: {crossover_depth:.0f} m')
    ax3.set_xlabel('log₁₀(As) [µg/L]', fontsize=11)
    ax3.set_ylabel('Well Depth (m)', fontsize=11)
    ax3.invert_yaxis()
    ax3.set_title('Arsenic Concentration\nvs Continuous Depth', fontsize=11)
    ax3.legend(fontsize=8.5)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_fig = FIGURES_DIR / 'H2_F_depth_spline.png'
    fig.savefig(out_fig, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {out_fig}")

    return {
        'crossover_depth': crossover_depth,
        'spline_table': spline_tbl,
        'zone_table': zone_df,
    }


if __name__ == '__main__':
    result = run_depth_spline()
    print(f"\nCrossover depth: {result['crossover_depth']:.1f} m")
