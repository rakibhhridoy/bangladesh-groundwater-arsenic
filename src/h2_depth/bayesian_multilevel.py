"""
Phase G-1 — Bayesian Multilevel Model
Depth zones as hierarchical levels with zone-specific slopes for
geogenic and anthropogenic predictors.
Quantifies: P(geogenic dominant | depth zone) for each zone.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.special import expit
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    try:
        import pymc3 as pm
        import arviz as az
        PYMC_AVAILABLE = True
    except ImportError:
        PYMC_AVAILABLE = False

from config import (
    TARGET, DEPTH_ZONE_LABELS, ZONE_COLORS, WHO_AS_LIMIT,
    FIGURES_DIR, TABLES_DIR, RESULTS_DIR, DPI, ALPHA,
    BAYES_DRAWS, BAYES_TUNE, BAYES_CHAINS, RANDOM_STATE
)

GEOGENIC_PRED  = 'Geogenic_Index'   # from Phase C
ANTHRO_PRED    = 'PO43-'


def _available(df, cols):
    return [c for c in cols if c in df.columns]


# ─────────────────────────────────────────────
# FREQUENTIST MULTILEVEL (LME) — always runs
# ─────────────────────────────────────────────

def mixed_effects_by_zone(df):
    """
    Linear Mixed-Effects model:
      log(As) ~ Geogenic_Index + PO43- + ORP + pH + (1 + Geogenic_Index + PO43- | Depth_Zone)

    Estimates zone-specific slopes for geogenic vs. anthropogenic predictors.
    Table H2_T25.
    """
    print("\n" + "="*70)
    print("PHASE G-1a: MIXED-EFFECTS MODEL (Depth Zone as grouping level)")
    print("="*70)

    y_col = f'{TARGET}_log' if f'{TARGET}_log' in df.columns else TARGET

    # Build predictor set from what's available
    geo_pred   = GEOGENIC_PRED if GEOGENIC_PRED in df.columns else None
    anthro_col = 'PO43-' if 'PO43-' in df.columns else None
    other_cols = _available(df, ['ORP', 'pH', 'Fe2+', 'HCO3-'])

    if geo_pred is None and anthro_col is None:
        print("  SKIP: neither Geogenic_Index nor PO43- available.")
        return None

    fixed_effects = [c for c in [geo_pred, anthro_col] + other_cols if c is not None]
    keep_cols     = [y_col, 'Depth_Zone'] + fixed_effects
    complete      = df[keep_cols].dropna()

    print(f"  N complete = {len(complete)}")

    # Standardise predictors
    scaler_dict = {}
    for col in fixed_effects:
        mu, sigma = complete[col].mean(), complete[col].std()
        complete[col + '_std'] = (complete[col] - mu) / (sigma + 1e-9)
        scaler_dict[col] = (mu, sigma)

    std_fixed = [c + '_std' for c in fixed_effects]
    # Wrap each term in Q() so patsy handles column names that contain
    # special characters such as '-' (e.g. 'PO43-_std', 'Fe2+_std').
    formula   = f"{y_col} ~ " + " + ".join(f'Q("{c}")' for c in std_fixed)

    # OLS per-zone for zone-specific slopes (approximates LME)
    zone_coef_rows = []
    for zone in DEPTH_ZONE_LABELS:
        sub = complete[complete['Depth_Zone'] == zone]
        if len(sub) < 20:
            continue
        try:
            model = smf.ols(formula, data=sub).fit()
            for pred in std_fixed:
                orig = pred.replace('_std', '')
                # patsy stores Q("col") terms under 'Q("col")' key
                q_key = f'Q("{pred}")'
                beta  = model.params.get(q_key, model.params.get(pred, np.nan))
                se    = model.bse.get(q_key, model.bse.get(pred, np.nan))
                pval  = model.pvalues.get(q_key, model.pvalues.get(pred, np.nan))
                zone_coef_rows.append({
                    'Depth_Zone': zone,
                    'Predictor': orig,
                    'Beta': round(beta, 5),
                    'SE': round(se, 5),
                    'p_value': round(pval, 6),
                    'Significant': 'YES' if pval < ALPHA else 'NO',
                    'N': len(sub),
                    'R2': round(model.rsquared, 4),
                })
        except Exception as e:
            print(f"  WARNING {zone}: {e}")

    coef_df = pd.DataFrame(zone_coef_rows)
    coef_df.to_csv(TABLES_DIR / 'H2_T25_zone_specific_coefficients.csv', index=False)

    print("\n  Zone-specific standardised β coefficients:")
    pivot = coef_df.pivot_table(index='Predictor', columns='Depth_Zone', values='Beta')
    pivot = pivot.reindex(columns=[z for z in DEPTH_ZONE_LABELS if z in pivot.columns])
    print(pivot.round(4).to_string())

    # Process dominance probability (sign-based)
    dom_rows = []
    for zone in DEPTH_ZONE_LABELS:
        sub = coef_df[coef_df['Depth_Zone'] == zone]
        geo_beta    = sub[sub['Predictor'] == (GEOGENIC_PRED or 'Geogenic_Index')]['Beta'].values
        anthro_beta = sub[sub['Predictor'] == anthro_col]['Beta'].values
        if len(geo_beta) > 0 and len(anthro_beta) > 0:
            dominant = 'Geogenic' if abs(geo_beta[0]) > abs(anthro_beta[0]) else 'Anthropogenic'
            dom_rows.append({
                'Depth_Zone': zone,
                'Geogenic_Beta': round(geo_beta[0], 4),
                'Anthropogenic_Beta': round(anthro_beta[0], 4),
                'Dominant_Process': dominant,
            })
            print(f"  {zone:15s}: Geogenic β={geo_beta[0]:+.4f}  "
                  f"Anthropogenic β={anthro_beta[0]:+.4f}  → {dominant} dominant")

    dom_df = pd.DataFrame(dom_rows)
    dom_df.to_csv(TABLES_DIR / 'H2_T26_process_dominance_by_zone.csv', index=False)

    return coef_df, dom_df


# ─────────────────────────────────────────────
# BAYESIAN MULTILEVEL MODEL (PyMC if available)
# ─────────────────────────────────────────────

def bayesian_multilevel_model(df):
    """
    Bayesian hierarchical model with depth-zone-specific slopes.
    log(As) ~ α[zone] + β_geo[zone] * Geogenic_Index + β_anthro[zone] * PO43-
    Posterior β_geo[Deep] > β_anthro[Deep] → H2.3 supported
    Posterior β_anthro[Shallow] > β_geo[Shallow] → H2.2 supported
    Table H2_T27.
    """
    if not PYMC_AVAILABLE:
        print("\n  ⚠ PyMC not available. Skipping Bayesian multilevel model.")
        print("    Install with: pip install pymc arviz")
        return None

    print("\n" + "="*70)
    print("PHASE G-1b: BAYESIAN MULTILEVEL MODEL")
    print("="*70)

    y_col   = f'{TARGET}_log' if f'{TARGET}_log' in df.columns else TARGET
    geo_col = GEOGENIC_PRED if GEOGENIC_PRED in df.columns else 'ORP'
    ant_col = ANTHRO_PRED if ANTHRO_PRED in df.columns else 'NO3-'

    keep = [y_col, geo_col, ant_col, 'Depth_Zone']
    complete = df[keep].dropna()

    # Encode zones
    zones       = DEPTH_ZONE_LABELS
    zone_idx    = pd.Categorical(complete['Depth_Zone'], categories=zones, ordered=True)
    zone_codes  = zone_idx.codes
    n_zones     = len(zones)

    # Standardise
    def std(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    y   = complete[y_col].values.astype(float)
    Xg  = std(complete[geo_col]).values.astype(float)
    Xa  = std(complete[ant_col]).values.astype(float)

    print(f"  Fitting Bayesian model: n={len(y)}, zones={n_zones}")

    try:
        with pm.Model() as model:
            # Hyperpriors
            mu_alpha   = pm.Normal('mu_alpha',   mu=0,  sigma=2)
            sigma_alpha= pm.HalfNormal('sigma_alpha', sigma=1)
            mu_geo     = pm.Normal('mu_geo',     mu=0,  sigma=1)
            sigma_geo  = pm.HalfNormal('sigma_geo',   sigma=0.5)
            mu_anthro  = pm.Normal('mu_anthro',  mu=0,  sigma=1)
            sigma_anthro=pm.HalfNormal('sigma_anthro', sigma=0.5)

            # Zone-level random effects
            alpha     = pm.Normal('alpha',   mu=mu_alpha,  sigma=sigma_alpha, shape=n_zones)
            beta_geo  = pm.Normal('beta_geo', mu=mu_geo,   sigma=sigma_geo,   shape=n_zones)
            beta_ant  = pm.Normal('beta_ant', mu=mu_anthro, sigma=sigma_anthro,shape=n_zones)

            sigma_y = pm.HalfNormal('sigma_y', sigma=1)

            mu_y = alpha[zone_codes] + beta_geo[zone_codes] * Xg + beta_ant[zone_codes] * Xa
            y_obs = pm.Normal('y_obs', mu=mu_y, sigma=sigma_y, observed=y)

            trace = pm.sample(
                draws=BAYES_DRAWS, tune=BAYES_TUNE, chains=BAYES_CHAINS,
                random_seed=RANDOM_STATE, progressbar=False,
                return_inferencedata=True
            )

        # Extract zone-specific posteriors
        post_rows = []
        for i, zone in enumerate(zones):
            geo_post = trace.posterior['beta_geo'].values[:, :, i].flatten()
            ant_post = trace.posterior['beta_ant'].values[:, :, i].flatten()

            prob_geo_gt_ant = (geo_post > ant_post).mean()  # P(geogenic dominant)

            post_rows.append({
                'Depth_Zone': zone,
                'beta_geo_mean': round(geo_post.mean(), 4),
                'beta_geo_hdi_2.5': round(np.percentile(geo_post, 2.5), 4),
                'beta_geo_hdi_97.5': round(np.percentile(geo_post, 97.5), 4),
                'beta_ant_mean': round(ant_post.mean(), 4),
                'beta_ant_hdi_2.5': round(np.percentile(ant_post, 2.5), 4),
                'beta_ant_hdi_97.5': round(np.percentile(ant_post, 97.5), 4),
                'P_geogenic_dominant': round(prob_geo_gt_ant, 4),
                'P_anthro_dominant': round(1 - prob_geo_gt_ant, 4),
                'H2.3_Supported': 'YES' if prob_geo_gt_ant > 0.75 else 'NO',
                'H2.2_Supported': 'YES' if (1 - prob_geo_gt_ant) > 0.75 else 'NO',
            })
            print(f"  {zone:15s}: β_geo={geo_post.mean():+.3f}  "
                  f"β_ant={ant_post.mean():+.3f}  "
                  f"P(geo>ant)={prob_geo_gt_ant:.3f}")

        post_df = pd.DataFrame(post_rows)
        post_df.to_csv(TABLES_DIR / 'H2_T27_bayesian_zone_posteriors.csv', index=False)

        _plot_bayesian_posteriors(trace, zones)
        return trace, post_df

    except Exception as e:
        print(f"  WARNING: Bayesian model failed: {e}")
        return None


def _plot_bayesian_posteriors(trace, zones):
    """Figure H2_F16 — Posterior distributions of β_geo and β_ant per zone."""
    try:
        import arviz as az
        fig, axes = plt.subplots(2, len(zones), figsize=(4 * len(zones), 8), sharey='row')

        for i, zone in enumerate(zones):
            geo_post = trace.posterior['beta_geo'].values[:, :, i].flatten()
            ant_post = trace.posterior['beta_ant'].values[:, :, i].flatten()

            for j, (post, label, color) in enumerate([
                (geo_post, 'β_geogenic',     '#d73027'),
                (ant_post, 'β_anthropogenic', '#4575b4'),
            ]):
                ax = axes[j, i]
                ax.hist(post, bins=50, color=color, alpha=0.7, density=True)
                ax.axvline(np.mean(post), color='black', lw=2, ls='-', label='Mean')
                ax.axvline(np.percentile(post, 2.5),  color='black', lw=1, ls='--')
                ax.axvline(np.percentile(post, 97.5), color='black', lw=1, ls='--')
                ax.axvline(0, color='red', lw=1.5, ls=':', label='Zero')
                ax.set_title(f'{zone}\n{label}', fontsize=10)
                ax.set_xlabel('Posterior value', fontsize=9)
                if i == 0:
                    ax.set_ylabel('Density', fontsize=10)
                ax.legend(fontsize=7)
                ax.grid(alpha=0.3)

        plt.suptitle('H2 Phase G: Bayesian Posterior Distributions\n'
                     '(β_geogenic vs. β_anthropogenic per depth zone)',
                     fontsize=13)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / 'H2_F16_bayesian_posteriors.png', dpi=DPI, bbox_inches='tight')
        plt.close()
        print("  Saved: H2_F16_bayesian_posteriors.png")
    except Exception as e:
        print(f"  WARNING plotting posteriors: {e}")


# ─────────────────────────────────────────────
# PROCESS DOMINANCE PROBABILITY PLOT
# ─────────────────────────────────────────────

def plot_process_dominance(coef_df):
    """Figure H2_F17 — Process dominance probability bar chart by zone."""
    if coef_df is None:
        return

    geo_col = GEOGENIC_PRED
    ant_col = ANTHRO_PRED

    zones, p_geo, p_ant = [], [], []
    for zone in DEPTH_ZONE_LABELS:
        sub = coef_df[coef_df['Depth_Zone'] == zone]
        g = sub[sub['Predictor'] == geo_col]['Beta'].values
        a = sub[sub['Predictor'] == ant_col]['Beta'].values
        if len(g) == 0 or len(a) == 0:
            continue
        total = abs(g[0]) + abs(a[0]) + 1e-9
        zones.append(zone)
        p_geo.append(abs(g[0]) / total)
        p_ant.append(abs(a[0]) / total)

    if not zones:
        return

    x = np.arange(len(zones))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, p_geo, label='Geogenic dominance',      color='#d73027', alpha=0.85)
    ax.bar(x, p_ant, bottom=p_geo, label='Anthropogenic dominance', color='#4575b4', alpha=0.85)
    ax.axhline(0.5, color='grey', ls='--', lw=1.5, label='Equal contribution (50%)')
    ax.set_xticks(x)
    ax.set_xticklabels(zones, fontsize=12)
    ax.set_ylabel('Relative Process Contribution (|β| fraction)', fontsize=11)
    ax.set_title('H2 Phase G: Geogenic vs. Anthropogenic Process Dominance\n'
                 'by Depth Zone (from Zone-Specific OLS β coefficients)', fontsize=12)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v*100:.0f}%'))
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H2_F17_process_dominance.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H2_F17_process_dominance.png")


# ─────────────────────────────────────────────
# PHASE G-1 RUNNER
# ─────────────────────────────────────────────

def run_phase_g1(df):
    """Run complete Phase G-1: Bayesian multilevel model."""
    print("\n" + "█"*70)
    print(" PHASE G-1: BAYESIAN MULTILEVEL MODEL")
    print("█"*70)

    result_lme = mixed_effects_by_zone(df)
    coef_df    = result_lme[0] if result_lme else None
    dom_df     = result_lme[1] if result_lme else None

    bayes_result = bayesian_multilevel_model(df)
    plot_process_dominance(coef_df)

    print("\n✓ Phase G-1 complete. Outputs:")
    print("  Tables : H2_T25, H2_T26, H2_T27 (if PyMC available)")
    print("  Figures: H2_F16 (if PyMC), H2_F17")

    return {
        'lme_coefficients': coef_df,
        'process_dominance': dom_df,
        'bayesian_result': bayes_result,
    }


if __name__ == "__main__":
    from data_loader import load_and_prepare_data
    df = load_and_prepare_data()
    run_phase_g1(df)
