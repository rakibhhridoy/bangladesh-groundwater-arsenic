"""
Phase — Mechanism Shift Analysis
Tests H2.4: The dominant As mobilization mechanism shifts with depth.
Phosphate competition peaks in shallow zones; reductive dissolution peaks in deep zones.
Method: Depth-stratified partial correlations + interaction regression + β-coefficient comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols as smf_ols
import warnings
warnings.filterwarnings('ignore')

from config import (
    TARGET, DEPTH_ZONE_LABELS, ZONE_COLORS, WHO_AS_LIMIT,
    FIGURES_DIR, TABLES_DIR, DPI, ALPHA
)

MECHANISMS = {
    'Phosphate_Competition': 'PO43-',
    'Reductive_Dissolution': 'Fe2+',
    'Bicarbonate_Buffering': 'HCO3-',
    'pH_Desorption':         'pH',
}


def _available(df, cols):
    return [c for c in cols if c in df.columns]


# ─────────────────────────────────────────────
# PARTIAL CORRELATIONS BY DEPTH ZONE
# ─────────────────────────────────────────────

def partial_correlations_by_zone(df):
    """
    Compute partial Spearman correlations for each mechanism proxy with As,
    controlling for all other mechanism proxies — stratified by depth zone.
    Tests whether mechanism ranking changes with depth (H2.4).
    Table H2_T19.
    """
    print("\n" + "="*70)
    print("MECHANISM SHIFT — PARTIAL CORRELATIONS BY DEPTH ZONE")
    print("="*70)

    mech_vars = _available(df, list(MECHANISMS.values()))
    if len(mech_vars) < 2:
        print("  SKIP: insufficient mechanism variables.")
        return None

    rows = []
    for zone in ['ALL'] + DEPTH_ZONE_LABELS:
        sub = df if zone == 'ALL' else df[df['Depth_Zone'] == zone]
        y_col = f'{TARGET}_log' if f'{TARGET}_log' in sub.columns else TARGET
        cols_needed = mech_vars + [y_col]
        complete = sub[cols_needed].dropna()

        if len(complete) < 20:
            continue

        for mech_name, mech_var in MECHANISMS.items():
            if mech_var not in complete.columns:
                continue

            # Partial correlation: mech_var vs As, controlling for other mech_vars
            covars = [v for v in mech_vars if v != mech_var and v in complete.columns]

            try:
                import pingouin as pg
                result = pg.partial_corr(
                    data=complete, x=mech_var, y=y_col,
                    covar=covars, method='spearman'
                )
                rho  = result['r'].values[0]
                p    = result['p-val'].values[0]
                ci95 = result['CI95%'].values[0]
            except Exception:
                # Fallback: residual correlation after OLS
                try:
                    X_cov = sm.add_constant(complete[covars])
                    resid_x = sm.OLS(complete[mech_var], X_cov).fit().resid
                    resid_y = sm.OLS(complete[y_col],    X_cov).fit().resid
                    rho, p  = stats.spearmanr(resid_x, resid_y)
                    ci95    = [np.nan, np.nan]
                except Exception:
                    continue

            rows.append({
                'Depth_Zone': zone,
                'Mechanism': mech_name,
                'Variable': mech_var,
                'Partial_rho': round(rho, 4),
                'p_value': round(p, 6),
                'CI_lower': round(ci95[0], 4) if not np.isnan(ci95[0]) else np.nan,
                'CI_upper': round(ci95[1], 4) if not np.isnan(ci95[0]) else np.nan,
                'Significant': 'YES' if p < ALPHA else 'NO',
                'N': len(complete),
            })

    partial_df = pd.DataFrame(rows)
    partial_df.to_csv(TABLES_DIR / 'H2_T19_partial_corr_by_zone.csv', index=False)

    # Summary: mechanism ranking per zone
    print("\n  Mechanism ranking by partial ρ (controlling other mechanisms):")
    for zone in ['ALL'] + DEPTH_ZONE_LABELS:
        zone_df = partial_df[partial_df['Depth_Zone'] == zone].sort_values(
            'Partial_rho', ascending=False, key=abs
        )
        if len(zone_df) == 0:
            continue
        print(f"\n    {zone}:")
        for _, r in zone_df.iterrows():
            sig = '✓' if r['Significant'] == 'YES' else '✗'
            print(f"      {sig} {r['Mechanism']:30s}: ρ={r['Partial_rho']:+.4f}  p={r['p_value']:.4e}")

    return partial_df


# ─────────────────────────────────────────────
# INTERACTION REGRESSION: MECHANISM × DEPTH ZONE
# ─────────────────────────────────────────────

def interaction_regression(df):
    """
    Fits: log(As) ~ PO4 × DepthZone + Fe × DepthZone + pH + ORP
    Tests whether mechanism coefficients differ significantly by depth zone.
    Table H2_T20.
    """
    print("\n" + "="*70)
    print("MECHANISM SHIFT — INTERACTION REGRESSION (As ~ Mechanism × DepthZone)")
    print("="*70)

    y_col = f'{TARGET}_log' if f'{TARGET}_log' in df.columns else TARGET
    pred_map = {
        'PO43-': 'PO4', 'Fe2+': 'Fe', 'HCO3-': 'HCO3', 'pH': 'pH', 'ORP': 'ORP'
    }

    # Rename for formula compatibility
    rdf = df.copy()
    for orig, safe in pred_map.items():
        if orig in rdf.columns:
            rdf[safe] = rdf[orig]

    rdf['logAs'] = rdf[y_col]
    rdf['Zone']  = rdf['Depth_Zone'].astype(str)

    avail_preds = [safe for orig, safe in pred_map.items() if orig in df.columns]
    formula_vars = ' + '.join(avail_preds[:2])  # PO4 and Fe (primary mechanisms)
    formula_int  = ' + '.join([f'{v}*Zone' for v in avail_preds[:2]])

    # Additive model
    add_cols = ['logAs', 'Zone'] + avail_preds
    rdf_clean = rdf[add_cols].dropna()

    results_rows = []
    for mech_var in avail_preds[:2]:
        try:
            form = f'logAs ~ {mech_var} * C(Zone)'
            model = smf_ols(form, data=rdf_clean).fit()

            # Extract coefficients for each zone interaction
            for zone in DEPTH_ZONE_LABELS:
                # Find matching interaction term
                matching = [c for c in model.params.index
                            if mech_var in c and zone in c and ':' in c]
                if matching:
                    coef = model.params[matching[0]]
                    p_val = model.pvalues[matching[0]]
                else:
                    # Base zone: main effect
                    coef = model.params.get(mech_var, np.nan)
                    p_val = model.pvalues.get(mech_var, np.nan)

                results_rows.append({
                    'Mechanism_Var': mech_var,
                    'Depth_Zone': zone,
                    'Beta_coef': round(coef, 5),
                    'p_value': round(p_val, 6),
                    'Significant': 'YES' if p_val < ALPHA else 'NO',
                    'Model_R2': round(model.rsquared, 4),
                    'N': len(rdf_clean),
                })

            print(f"  {mech_var} × Zone interaction model: R²={model.rsquared:.4f}")

        except Exception as e:
            print(f"  WARNING {mech_var}: {e}")

    inter_df = pd.DataFrame(results_rows)
    inter_df.to_csv(TABLES_DIR / 'H2_T20_interaction_regression.csv', index=False)
    return inter_df


# ─────────────────────────────────────────────
# β COEFFICIENT COMPARISON ACROSS ZONES
# ─────────────────────────────────────────────

def beta_comparison_across_zones(df):
    """
    Fit simple OLS for each mechanism × depth zone combination.
    Compare standardised β coefficients to quantify mechanism dominance per zone.
    Table H2_T21.
    """
    print("\n" + "="*70)
    print("MECHANISM SHIFT — STANDARDISED β COMPARISON")
    print("="*70)

    y_col = f'{TARGET}_log' if f'{TARGET}_log' in df.columns else TARGET
    mech_vars = _available(df, list(MECHANISMS.values()))

    rows = []
    for zone in DEPTH_ZONE_LABELS:
        sub = df[df['Depth_Zone'] == zone]
        cols_needed = [y_col] + mech_vars
        complete = sub[cols_needed].dropna()
        if len(complete) < 20:
            continue

        # Standardise all variables
        y_std = (complete[y_col] - complete[y_col].mean()) / complete[y_col].std()

        for mech_name, mech_var in MECHANISMS.items():
            if mech_var not in complete.columns:
                continue
            x = complete[mech_var]
            x_std = (x - x.mean()) / (x.std() + 1e-9)

            try:
                X = sm.add_constant(x_std)
                model = sm.OLS(y_std, X).fit()
                beta  = model.params.iloc[1]
                p_val = model.pvalues.iloc[1]
                r2    = model.rsquared
            except Exception:
                continue

            rows.append({
                'Depth_Zone': zone, 'Mechanism': mech_name, 'Variable': mech_var,
                'Std_Beta': round(beta, 4),
                'R2': round(r2, 4),
                'p_value': round(p_val, 6),
                'Significant': 'YES' if p_val < ALPHA else 'NO',
                'N': len(complete),
            })

    beta_df = pd.DataFrame(rows)
    beta_df.to_csv(TABLES_DIR / 'H2_T21_beta_by_zone.csv', index=False)

    # Print ranking
    print("\n  Mechanism β-rank per depth zone:")
    for zone in DEPTH_ZONE_LABELS:
        sub = beta_df[beta_df['Depth_Zone'] == zone].sort_values('Std_Beta', ascending=False, key=abs)
        if len(sub) == 0:
            continue
        print(f"\n    {zone}:")
        for _, r in sub.iterrows():
            sig = '✓' if r['Significant'] == 'YES' else '✗'
            print(f"      {sig} {r['Mechanism']:30s}: β={r['Std_Beta']:+.4f}  R²={r['R2']:.4f}")

    return beta_df


# ─────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────

def plot_mechanism_shift_heatmap(partial_df):
    """Figure H2_F11 — Partial ρ heatmap: mechanism × depth zone."""
    if partial_df is None:
        return

    pivot = partial_df.pivot_table(
        index='Mechanism', columns='Depth_Zone', values='Partial_rho'
    )
    # Keep zone order
    pivot = pivot.reindex(columns=[z for z in DEPTH_ZONE_LABELS if z in pivot.columns])

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0,
                linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Partial Spearman ρ'})
    ax.set_title(
        'H2 Mechanism Shift: Partial ρ (Mechanism vs. As) by Depth Zone\n'
        '(Controlling all other mechanisms; expected: PO4 dominant shallow, Fe dominant deep)',
        fontsize=12
    )
    ax.set_xlabel('Depth Zone', fontsize=11)
    ax.set_ylabel('Mechanism', fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H2_F11_mechanism_shift_heatmap.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H2_F11_mechanism_shift_heatmap.png")


def plot_beta_by_zone(beta_df):
    """Figure H2_F12 — Standardised β across depth zones (line chart per mechanism)."""
    if beta_df is None or len(beta_df) == 0:
        return

    mechanisms = beta_df['Mechanism'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(mechanisms)))

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(DEPTH_ZONE_LABELS))

    for mech, col in zip(mechanisms, colors):
        sub = beta_df[beta_df['Mechanism'] == mech].set_index('Depth_Zone')
        vals = [sub.loc[z, 'Std_Beta'] if z in sub.index else np.nan
                for z in DEPTH_ZONE_LABELS]
        ax.plot(x, vals, 'o-', color=col, lw=2, ms=9, label=mech)

    ax.axhline(0, color='grey', ls='--', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(DEPTH_ZONE_LABELS, fontsize=11)
    ax.set_xlabel('Depth Zone (shallow → deep)', fontsize=12)
    ax.set_ylabel('Standardised β coefficient', fontsize=12)
    ax.set_title(
        'H2 Mechanism Shift: Standardised β by Depth Zone\n'
        '(Diverging lines = mechanism transition with depth)',
        fontsize=13
    )
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H2_F12_beta_by_zone.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H2_F12_beta_by_zone.png")


# ─────────────────────────────────────────────
# PHASE RUNNER
# ─────────────────────────────────────────────

def run_mechanism_shift(df):
    """Run complete mechanism shift analysis (H2.4)."""
    print("\n" + "█"*70)
    print(" MECHANISM SHIFT ANALYSIS (H2.4)")
    print("█"*70)

    partial_df  = partial_correlations_by_zone(df)
    inter_df    = interaction_regression(df)
    beta_df     = beta_comparison_across_zones(df)
    plot_mechanism_shift_heatmap(partial_df)
    plot_beta_by_zone(beta_df)

    print("\n✓ Mechanism shift analysis complete. Outputs:")
    print("  Tables : H2_T19, H2_T20, H2_T21")
    print("  Figures: H2_F11, H2_F12")

    return {
        'partial_correlations': partial_df,
        'interaction_regression': inter_df,
        'beta_comparison': beta_df,
    }


if __name__ == "__main__":
    from data_loader import load_and_prepare_data
    df = load_and_prepare_data()
    run_mechanism_shift(df)
