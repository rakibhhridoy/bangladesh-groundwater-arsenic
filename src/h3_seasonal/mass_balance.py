"""
Phase E — Three-Component Seasonal Mass Balance (H3.4)
Decomposes observed ΔAs into: concentration, reductive dissolution, competitive desorption.
Propensity score matching: Barind Tract vs. GBM Delta natural experiment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from config import (
    TARGET, DEPTH_ZONE_LABELS, ZONE_COLORS, SEASON_DRY, SEASON_WET,
    FIGURES_DIR, TABLES_DIR, DPI, ALPHA, RANDOM_STATE, MW_Cl
)


def _std(x):
    return (x - x.mean()) / (x.std() + 1e-9)


# ─── THREE-COMPONENT MASS BALANCE ─────────────────────────────────────────────

def compute_mass_balance(df_delta, df_dry):
    """
    Three-component decomposition of ΔAs:
      Component 1 — Concentration: ΔAs_conc = As_dry × (Cl_dry/Cl_wet − 1)
      Component 2 — Reductive:     ΔAs_redox = β_ORP × ΔORP + β_Fe × ΔFe  (from dry-season OLS)
      Component 3 — Desorption:    ΔAs_desorb = β_PO4 × ΔPO₄
      Residual:                    ΔAs_obs − sum(3 components)
    """
    print("\n" + "=" * 70)
    print("PHASE E-1: THREE-COMPONENT SEASONAL MASS BALANCE (H3.4)")
    print("=" * 70)

    df = df_delta.copy()
    delta_as_col = f'Delta_{TARGET}'
    if delta_as_col not in df.columns:
        print("  WARNING: Delta_As column not found, skipping mass balance")
        return df, None

    # ── Get dry-season OLS β coefficients ───────────────────────────────
    needed_dry = [TARGET, 'ORP', 'Fe2+', 'PO43-', 'Cl-']
    avail = [c for c in needed_dry if c in df_dry.columns]
    beta = {}

    if len(avail) >= 3:
        df_dry_clean = df_dry[avail + [f'{TARGET}_log']].dropna()
        pred_cols = [c for c in ['ORP', 'Fe2+', 'PO43-'] if c in df_dry_clean.columns]
        for col in pred_cols:
            df_dry_clean[col + '_std'] = _std(df_dry_clean[col])

        formula = f'{TARGET}_log ~ ' + ' + '.join(f'Q("{c}_std")' for c in pred_cols)
        try:
            model = smf.ols(formula, data=df_dry_clean).fit()
            for col in pred_cols:
                key = f'Q("{col}_std")'
                beta[col] = model.params.get(key, model.params.get(col + '_std', 0.0))
            print(f"  Dry-season OLS β: {beta}")
        except Exception as e:
            print(f"  WARNING: OLS failed: {e}")

    # ── Compute three components ─────────────────────────────────────────
    # Component 1: Concentration (Cl-based)
    if 'Cl-_dry' in df.columns and 'Cl-_wet' in df.columns:
        cl_ratio = df['Cl-_dry'] / (df['Cl-_wet'] + 1e-9)
        df['ΔAs_concentration'] = df[f'{TARGET}_dry'] * (cl_ratio - 1)
    else:
        df['ΔAs_concentration'] = np.nan

    # Component 2: Reductive dissolution (ORP + Fe)
    comp2 = pd.Series(0.0, index=df.index)
    if 'Delta_ORP' in df.columns and 'ORP' in beta:
        orp_std_scale = df_dry['ORP'].std() if 'ORP' in df_dry.columns else 1.0
        comp2 += beta.get('ORP', 0) * (df['Delta_ORP'] / (orp_std_scale + 1e-9))
    if 'Delta_Fe2+' in df.columns and 'Fe2+' in beta:
        fe_std_scale = df_dry['Fe2+'].std() if 'Fe2+' in df_dry.columns else 1.0
        comp2 += beta.get('Fe2+', 0) * (df['Delta_Fe2+'] / (fe_std_scale + 1e-9))
    df['ΔAs_reductive'] = comp2

    # Component 3: Competitive desorption (PO₄)
    if 'Delta_PO43-' in df.columns and 'PO43-' in beta:
        po4_std_scale = df_dry['PO43-'].std() if 'PO43-' in df_dry.columns else 1.0
        df['ΔAs_desorption'] = beta.get('PO43-', 0) * (df['Delta_PO43-'] / (po4_std_scale + 1e-9))
    else:
        df['ΔAs_desorption'] = np.nan

    # Residual
    comps = ['ΔAs_concentration', 'ΔAs_reductive', 'ΔAs_desorption']
    comp_sum = df[comps].sum(axis=1, skipna=True)
    df['ΔAs_residual'] = df[delta_as_col] - comp_sum

    # ── Summary table ────────────────────────────────────────────────────
    mb_rows = []
    for zone in ['ALL'] + DEPTH_ZONE_LABELS:
        sub = df if zone == 'ALL' else df[df['Depth_Zone'] == zone]
        sub = sub.dropna(subset=[delta_as_col])
        if len(sub) < 5:
            continue
        obs_mean = sub[delta_as_col].mean()
        row = {'Depth_Zone': zone, 'N': len(sub),
               'DeltaAs_observed': round(obs_mean, 3)}
        for c, label in zip(comps, ['Concentration', 'Reductive', 'Desorption']):
            mean_c = sub[c].mean() if c in sub.columns else np.nan
            frac   = mean_c / (obs_mean + 1e-9) if pd.notna(mean_c) else np.nan
            row[f'ΔAs_{label}'] = round(mean_c, 3) if pd.notna(mean_c) else np.nan
            row[f'Frac_{label}'] = round(frac, 4) if pd.notna(frac) else np.nan
        row['ΔAs_Residual'] = round(sub['ΔAs_residual'].mean(), 3)
        mb_rows.append(row)
        print(f"  {zone:15s}: obs={obs_mean:.2f}  "
              f"conc={row.get('ΔAs_Concentration', np.nan):.2f}  "
              f"redox={row.get('ΔAs_Reductive', np.nan):.2f}  "
              f"desorb={row.get('ΔAs_Desorption', np.nan):.2f}")

    mb_df = pd.DataFrame(mb_rows)
    mb_df.to_csv(TABLES_DIR / 'H3_T13_mass_balance_components.csv', index=False)

    # NDWI amplitude vs. concentration component
    if 'NDWI_mean' in df.columns and 'ΔAs_concentration' in df.columns:
        valid = df[['NDWI_mean', 'ΔAs_concentration', 'Depth_Zone']].dropna()
        ndwi_rows = []
        for zone in ['ALL'] + DEPTH_ZONE_LABELS:
            s = valid if zone == 'ALL' else valid[valid['Depth_Zone'] == zone]
            if len(s) < 10:
                continue
            rho, p = stats.spearmanr(s['NDWI_mean'], s['ΔAs_concentration'])
            ndwi_rows.append({'Depth_Zone': zone, 'N': len(s),
                               'Spearman_rho': round(rho, 5), 'p_value': round(p, 6),
                               'Significant': 'YES' if p < ALPHA else 'NO',
                               'Interpretation': 'NDWI predicts concentration dilution'
                               if (p < ALPHA and rho < 0) else 'No significant NDWI-concentration link'})
            print(f"  NDWI vs ΔAs_conc {zone}: ρ={rho:+.4f}  p={p:.4f}")
        ndwi_df = pd.DataFrame(ndwi_rows)
        ndwi_df.to_csv(TABLES_DIR / 'H3_T14_ndwi_concentration_component.csv', index=False)

    return df, mb_df


# ─── PROPENSITY SCORE MATCHING ────────────────────────────────────────────────

def propensity_score_matching(df_delta):
    """
    Natural experiment: Barind Tract (heavy drawdown) vs. GBM Delta (light drawdown).
    Match on (depth, As_dry, Fe_dry, pH_dry) via propensity score.
    Compare ΔAs_concentration component between matched groups.
    """
    print("\n" + "=" * 70)
    print("PHASE E-2: PROPENSITY SCORE MATCHING — Barind vs. GBM Delta (H3.4)")
    print("=" * 70)

    if 'Phys_Zone' not in df_delta.columns:
        print("  SKIP: Phys_Zone column not found")
        return None

    barind = df_delta[df_delta['Phys_Zone'] == 'Barind_Tract'].copy()
    gbm    = df_delta[df_delta['Phys_Zone'] == 'GBM_Delta'].copy()

    print(f"  Barind Tract: n={len(barind)} | GBM Delta: n={len(gbm)}")

    if len(barind) < 10 or len(gbm) < 10:
        print("  SKIP: insufficient samples in one/both groups")
        return None

    # Confounder variables (available in delta dataset)
    conf_vars = [c for c in ['Depth', f'{TARGET}_dry', 'Fe2+_dry', 'pH_dry', 'ORP_dry']
                 if c in df_delta.columns]
    if len(conf_vars) < 2:
        conf_vars = [c for c in df_delta.columns
                     if '_dry' in c and df_delta[c].dtype in [np.float64, np.int64]][:4]

    treatment_label = 'Barind_Tract'
    combined = pd.concat([barind, gbm], ignore_index=True)
    combined['Treatment'] = (combined['Phys_Zone'] == treatment_label).astype(int)
    combined_clean = combined[conf_vars + ['Treatment', 'ΔAs_concentration']
                              if 'ΔAs_concentration' in combined.columns
                              else conf_vars + ['Treatment']].dropna()

    if len(combined_clean) < 20:
        print("  SKIP: insufficient complete rows for matching")
        return None

    X = combined_clean[conf_vars].values
    y = combined_clean['Treatment'].values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    ps_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=500)
    ps_model.fit(X_sc, y)
    ps_scores = ps_model.predict_proba(X_sc)[:, 1]
    combined_clean = combined_clean.copy()
    combined_clean['PS_score'] = ps_scores

    # Match each treated unit to nearest control
    treated  = combined_clean[combined_clean['Treatment'] == 1]
    controls = combined_clean[combined_clean['Treatment'] == 0]

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(controls['PS_score'].values.reshape(-1, 1))
    dists, idxs = nn.kneighbors(treated['PS_score'].values.reshape(-1, 1))

    matched_controls = controls.iloc[idxs.flatten()]

    result_rows = []
    outcome_col = 'ΔAs_concentration' if 'ΔAs_concentration' in combined_clean.columns else None

    if outcome_col:
        treated_outcome  = treated[outcome_col].values
        control_outcome  = matched_controls[outcome_col].values
        diff = treated_outcome.mean() - control_outcome.mean()
        t_stat, t_p = stats.ttest_ind(treated_outcome, control_outcome, nan_policy='omit')
        u_stat, u_p = stats.mannwhitneyu(treated_outcome, control_outcome, alternative='two-sided')

        result_rows.append({
            'Comparison': f'{treatment_label} vs GBM_Delta',
            'N_treated': len(treated), 'N_control': len(controls),
            'N_matched': len(treated),
            'Treated_mean_outcome': round(treated_outcome.mean(), 3),
            'Control_mean_outcome': round(control_outcome.mean(), 3),
            'ATT_estimate': round(diff, 3),
            'ttest_p': round(t_p, 5),
            'MannWhitney_p': round(u_p, 5),
            'Significant': 'YES' if u_p < ALPHA else 'NO',
            'Interpretation': (
                'Barind Tract shows stronger concentration effect (irrigation drawdown dominant)'
                if diff > 0 else 'GBM Delta shows stronger concentration effect'
            ),
        })
        print(f"  ATT (Average Treatment Effect on Treated): {diff:+.3f} µg/L")
        print(f"  MannWhitney p={u_p:.5f}  Significant: {'YES' if u_p < ALPHA else 'NO'}")

    ps_df = pd.DataFrame(result_rows)
    ps_df.to_csv(TABLES_DIR / 'H3_T15_propensity_matching_result.csv', index=False)
    return ps_df


def run_mass_balance(df_delta, df_dry):
    """Run complete Phase E."""
    print("\n" + "█" * 70)
    print(" PHASE E: THREE-COMPONENT MASS BALANCE + PROPENSITY MATCHING (H3.4)")
    print("█" * 70)

    df_delta_mb, mb_df = compute_mass_balance(df_delta, df_dry)
    ps_df = propensity_score_matching(df_delta_mb)

    if mb_df is not None:
        _plot_mass_balance_bars(mb_df)
    _plot_ndwi_conc_scatter(df_delta_mb)

    print("\n✓ Phase E complete. Outputs: H3_T13, H3_T14, H3_T15, H3_F10, H3_F11")
    return {'mass_balance': mb_df, 'propensity': ps_df, 'df_delta': df_delta_mb}


def _plot_mass_balance_bars(mb_df):
    """H3_F10: Stacked bar chart of three components by depth zone."""
    zones = [z for z in DEPTH_ZONE_LABELS if z in mb_df['Depth_Zone'].values]
    if not zones:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(zones))
    comps = [('Frac_Concentration', '#2166ac', 'Concentration/Dilution'),
             ('Frac_Reductive',     '#d73027', 'Reductive Dissolution'),
             ('Frac_Desorption',    '#1a9641', 'Competitive Desorption')]

    bottoms = np.zeros(len(zones))
    for col, color, label in comps:
        if col not in mb_df.columns:
            continue
        vals = mb_df[mb_df['Depth_Zone'].isin(zones)].set_index('Depth_Zone').reindex(zones)[col].fillna(0).values
        ax.bar(x, vals * 100, bottom=bottoms * 100, color=color, alpha=0.85, label=label, edgecolor='k', lw=0.5)
        bottoms += vals

    ax.set_xticks(x); ax.set_xticklabels(zones, fontsize=11)
    ax.set_ylabel('% Contribution to Total Seasonal ΔAs', fontsize=11)
    ax.set_title('H3.4: Three-Component Seasonal Mass Balance by Depth Zone\n'
                 '(Concentration + Reductive Dissolution + Competitive Desorption)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H3_F10_mass_balance_bars.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H3_F10_mass_balance_bars.png")


def _plot_ndwi_conc_scatter(df_delta):
    """H3_F11: NDWI vs. concentration component (Barind vs. GBM highlighted)."""
    if 'NDWI_mean' not in df_delta.columns or 'ΔAs_concentration' not in df_delta.columns:
        return
    valid = df_delta[['NDWI_mean', 'ΔAs_concentration', 'Phys_Zone', 'Depth_Zone']].dropna()
    if len(valid) < 10:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    highlight = {'Barind_Tract': ('#d73027', 's', 80), 'GBM_Delta': ('#4575b4', '^', 80)}
    for zone, (color, marker, s) in highlight.items():
        sub = valid[valid['Phys_Zone'] == zone]
        ax.scatter(sub['NDWI_mean'], sub['ΔAs_concentration'],
                   color=color, marker=marker, s=s, alpha=0.8, label=zone, edgecolors='k', lw=0.4)
    others = valid[~valid['Phys_Zone'].isin(highlight.keys())]
    ax.scatter(others['NDWI_mean'], others['ΔAs_concentration'],
               color='grey', s=20, alpha=0.4, label='Other zones')

    from statsmodels.nonparametric.smoothers_lowess import lowess
    lw = lowess(valid['ΔAs_concentration'], valid['NDWI_mean'], frac=0.5)
    ax.plot(lw[:, 0], lw[:, 1], 'k-', lw=2, label='LOESS fit')
    ax.axhline(0, color='grey', ls='--', lw=1)

    ax.set_xlabel('NDWI (Surface water extent / recharge proxy)', fontsize=11)
    ax.set_ylabel('ΔAs_concentration (physical concentration component, µg/L)', fontsize=11)
    ax.set_title('H3.4: NDWI vs. Concentration Component of Seasonal ΔAs\n'
                 '(Barind Tract ■ vs. GBM Delta ▲ — natural experiment)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H3_F11_ndwi_concentration_scatter.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H3_F11_ndwi_concentration_scatter.png")
