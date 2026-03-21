"""
Phase F — SHAP Variable Importance by Depth Zone
Trains Random Forest and XGBoost models per depth zone.
SHAP analysis reveals which variables drive As prediction in each zone.
Directly tests H2.2 (anthropogenic features top in shallow) and H2.3 (geogenic features top in deep).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from config import (
    TARGET, DEPTH_ZONE_LABELS, ZONE_COLORS, WHO_AS_LIMIT,
    GEOGENIC_VARS, GEOGENIC_RATIOS, ANTHROPOGENIC_VARS, SATELLITE_VARS,
    FIGURES_DIR, TABLES_DIR, DPI, ALPHA,
    RF_N_ESTIMATORS, XGB_N_ESTIMATORS, CV_FOLDS, RANDOM_STATE
)

GEOGENIC_FEATURES  = ['B+', 'Al3+', 'Mn2+', 'Fe2+', 'Ca2+', 'Cr3+', 'ORP',
                       'B_Cl_ratio', 'Ca_Mg_ratio', 'Mn_Fe_ratio', 'HCO3_Cl_ratio']
ANTHRO_FEATURES    = ['PO43-', 'NO3-', 'NDVI_mean', 'EVI_mean']
HYDRO_FEATURES     = ['pH', 'DO', 'EC', 'TDS', 'HCO3-', 'SO42-', 'Depth']
ALL_FEATURES       = GEOGENIC_FEATURES + ANTHRO_FEATURES + HYDRO_FEATURES


def _available(df, cols):
    return [c for c in cols if c in df.columns]


def _label_feature(feat):
    """Classify feature as Geogenic, Anthropogenic, or Hydrochemical."""
    if feat in GEOGENIC_FEATURES:
        return 'Geogenic'
    if feat in ANTHRO_FEATURES:
        return 'Anthropogenic'
    return 'Hydrochemical'


# ─────────────────────────────────────────────
# MODEL TRAINING PER DEPTH ZONE
# ─────────────────────────────────────────────

def train_models_by_zone(df):
    """
    Train Random Forest for each depth zone + ALL samples.
    Returns dict: zone → {model, X, y, feature_names, cv_r2}.
    Table H2_T22 — Model performance by zone.
    """
    print("\n" + "="*70)
    print("PHASE F-1: RANDOM FOREST MODEL TRAINING BY DEPTH ZONE")
    print("="*70)

    y_col     = f'{TARGET}_log' if f'{TARGET}_log' in df.columns else TARGET
    feat_cols = _available(df, ALL_FEATURES)
    print(f"  Features available: {len(feat_cols)}: {feat_cols}")

    zone_models = {}
    perf_rows   = []

    for zone in ['ALL'] + DEPTH_ZONE_LABELS:
        sub = df if zone == 'ALL' else df[df['Depth_Zone'] == zone]
        complete = sub[feat_cols + [y_col]].dropna()

        if len(complete) < 30:
            print(f"  SKIP {zone}: n={len(complete)} (< 30)")
            continue

        X = complete[feat_cols].values
        y = complete[y_col].values

        rf = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_features='sqrt',
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=2      # capped: n_jobs=-1 floods macOS spawn workers and OOMs
        )

        # Cross-validated R²
        kf     = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_r2  = cross_val_score(rf, X, y, cv=kf, scoring='r2')
        cv_rmse= cross_val_score(rf, X, y, cv=kf,
                                 scoring='neg_root_mean_squared_error')

        rf.fit(X, y)
        train_r2 = rf.score(X, y)

        zone_models[zone] = {
            'model': rf,
            'X': X, 'y': y,
            'feature_names': feat_cols,
            'n': len(complete),
        }

        perf_rows.append({
            'Depth_Zone': zone, 'N': len(complete),
            'CV_R2_mean': round(cv_r2.mean(), 4),
            'CV_R2_std':  round(cv_r2.std(),  4),
            'CV_RMSE_mean': round(-cv_rmse.mean(), 4),
            'Train_R2': round(train_r2, 4),
            'n_features': len(feat_cols),
        })
        print(f"  {zone:15s}: n={len(complete):4d}  "
              f"CV R²={cv_r2.mean():.4f}±{cv_r2.std():.4f}  "
              f"Train R²={train_r2:.4f}")

    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(TABLES_DIR / 'H2_T22_model_performance_by_zone.csv', index=False)
    return zone_models, perf_df


# ─────────────────────────────────────────────
# SHAP ANALYSIS PER ZONE
# ─────────────────────────────────────────────

def compute_shap_by_zone(zone_models):
    """
    Compute SHAP values for each zone model.
    Returns dict: zone → shap_values array.
    """
    if not SHAP_AVAILABLE:
        print("  SKIP: shap not installed (pip install shap)")
        return {}

    print("\n" + "="*70)
    print("PHASE F-2: SHAP VALUE COMPUTATION BY DEPTH ZONE")
    print("="*70)

    shap_results = {}
    for zone, info in zone_models.items():
        model  = info['model']
        X      = info['X']
        feats  = info['feature_names']
        n_bg   = min(100, len(X))

        try:
            explainer   = shap.TreeExplainer(model, data=X[:n_bg])
            shap_vals   = explainer.shap_values(X)
            shap_results[zone] = {
                'shap_values': shap_vals,
                'X': X,
                'feature_names': feats,
                'expected_value': explainer.expected_value,
            }
            print(f"  {zone:15s}: SHAP computed (n={len(X)})")
        except Exception as e:
            print(f"  WARNING {zone}: {e}")

    return shap_results


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE TABLE
# ─────────────────────────────────────────────

def feature_importance_by_zone(zone_models, shap_results):
    """
    Compile feature importance (SHAP mean |value| or RF impurity) by zone.
    Classify each feature as Geogenic / Anthropogenic / Hydrochemical.
    Table H2_T23.
    """
    print("\n" + "="*70)
    print("PHASE F-3: FEATURE IMPORTANCE BY DEPTH ZONE")
    print("="*70)

    rows = []
    for zone, info in zone_models.items():
        feats = info['feature_names']
        model = info['model']

        if zone in shap_results and SHAP_AVAILABLE:
            shap_vals = shap_results[zone]['shap_values']
            importance = np.abs(shap_vals).mean(axis=0)
            imp_type = 'SHAP_mean_abs'
        else:
            importance = model.feature_importances_
            imp_type = 'RF_impurity'

        # Normalise to sum=1
        imp_norm = importance / (importance.sum() + 1e-9)

        for feat, imp, imp_raw in zip(feats, imp_norm, importance):
            rows.append({
                'Depth_Zone': zone, 'Feature': feat,
                'Importance_normalised': round(imp, 6),
                'Importance_raw': round(imp_raw, 6),
                'Importance_type': imp_type,
                'Process_type': _label_feature(feat),
                'Rank': 0,
            })

    imp_df = pd.DataFrame(rows)
    # Assign rank within each zone
    imp_df['Rank'] = imp_df.groupby('Depth_Zone')['Importance_normalised'].rank(
        ascending=False
    ).astype(int)

    imp_df.to_csv(TABLES_DIR / 'H2_T23_feature_importance_by_zone.csv', index=False)

    # Print top-5 per zone
    print("\n  Top-5 features by depth zone:")
    for zone in ['ALL'] + DEPTH_ZONE_LABELS:
        sub = imp_df[(imp_df['Depth_Zone'] == zone) & (imp_df['Rank'] <= 5)].sort_values('Rank')
        if len(sub) == 0:
            continue
        print(f"\n    {zone}:")
        for _, r in sub.iterrows():
            print(f"      #{int(r.Rank)} {r.Feature:20s} {r.Process_type:15s}  "
                  f"{r.Importance_normalised*100:5.2f}%")

    # Process-type fraction by zone
    frac_rows = []
    for zone in ['ALL'] + DEPTH_ZONE_LABELS:
        sub = imp_df[imp_df['Depth_Zone'] == zone]
        if len(sub) == 0:
            continue
        for ptype in ['Geogenic', 'Anthropogenic', 'Hydrochemical']:
            frac = sub[sub['Process_type'] == ptype]['Importance_normalised'].sum()
            frac_rows.append({'Depth_Zone': zone, 'Process_type': ptype,
                              'Importance_fraction': round(frac, 4)})

    frac_df = pd.DataFrame(frac_rows)
    frac_df.to_csv(TABLES_DIR / 'H2_T24_process_importance_fraction_by_zone.csv', index=False)

    print("\n  Process-type importance fraction by zone:")
    pivot = frac_df.pivot_table(
        index='Process_type', columns='Depth_Zone', values='Importance_fraction'
    )
    print(pivot.round(3).to_string())

    return imp_df, frac_df


# ─────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────

def plot_shap_summary_by_zone(shap_results):
    """Figure H2_F13 — SHAP beeswarm plots per depth zone (2×2 panel)."""
    if not SHAP_AVAILABLE or not shap_results:
        return

    zones_to_plot = [z for z in DEPTH_ZONE_LABELS if z in shap_results]
    n_zones = len(zones_to_plot)
    if n_zones == 0:
        return

    n_cols = 2
    n_rows = int(np.ceil(n_zones / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = np.array(axes).flatten()

    for i, zone in enumerate(zones_to_plot):
        ax = axes[i]
        plt.sca(ax)
        info = shap_results[zone]
        shap.summary_plot(
            info['shap_values'], info['X'],
            feature_names=info['feature_names'],
            plot_type='dot', show=False, max_display=12,
            color_bar=False
        )
        ax.set_title(f'{zone}  (n={zone_models_ref.get(zone, {}).get("n", "?")})',
                     fontsize=12, fontweight='bold')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('H2 Phase F: SHAP Summary Plots by Depth Zone\n'
                 '(Features ordered by mean |SHAP| importance)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H2_F13_shap_summary_by_zone.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H2_F13_shap_summary_by_zone.png")


zone_models_ref = {}  # global ref for plot_shap_summary_by_zone


def plot_importance_comparison(imp_df, frac_df):
    """Figure H2_F14 — Feature importance comparison across zones."""
    if imp_df is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: stacked bar of process-type fractions
    ax = axes[0]
    zones  = [z for z in DEPTH_ZONE_LABELS if z in frac_df['Depth_Zone'].values]
    ptypes = ['Geogenic', 'Anthropogenic', 'Hydrochemical']
    colors = {'Geogenic': '#d73027', 'Anthropogenic': '#4575b4', 'Hydrochemical': '#74c476'}

    bottoms = np.zeros(len(zones))
    for ptype in ptypes:
        vals = []
        for zone in zones:
            sub = frac_df[(frac_df['Depth_Zone'] == zone) &
                          (frac_df['Process_type'] == ptype)]
            vals.append(sub['Importance_fraction'].values[0] if len(sub) > 0 else 0)
        ax.bar(zones, vals, bottom=bottoms, label=ptype,
               color=colors[ptype], alpha=0.85)
        bottoms += np.array(vals)

    ax.set_ylabel('Normalised Feature Importance Fraction', fontsize=11)
    ax.set_title('Process-Type Importance by Depth Zone', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v*100:.0f}%'))
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', alpha=0.3)

    # Right: top features for Shallow vs Deep
    ax2 = axes[1]
    shallow_imp = imp_df[imp_df['Depth_Zone'] == 'Shallow'].sort_values(
        'Importance_normalised', ascending=False).head(10)
    deep_imp    = imp_df[imp_df['Depth_Zone'] == 'Deep'].sort_values(
        'Importance_normalised', ascending=False).head(10)

    y_sh = np.arange(len(shallow_imp))
    y_dp = np.arange(len(deep_imp))

    ax2.barh(y_sh, shallow_imp['Importance_normalised'] * 100,
             color=[colors.get(_label_feature(f), '#aaa') for f in shallow_imp['Feature']],
             alpha=0.8, label='Shallow zone')
    ax2.set_yticks(y_sh)
    ax2.set_yticklabels(shallow_imp['Feature'], fontsize=9)
    ax2.set_xlabel('Importance (%)', fontsize=11)
    ax2.set_title('Top Features: Shallow Zone', fontsize=12)
    ax2.grid(axis='x', alpha=0.3)

    plt.suptitle('H2 Phase F: Feature Importance by Process Type and Depth Zone',
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H2_F14_importance_comparison.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H2_F14_importance_comparison.png")


def plot_shap_dependence_po4_fe(shap_results, zone_models):
    """Figure H2_F15 — SHAP dependence: PO4 (shallow) and Fe (deep)."""
    if not SHAP_AVAILABLE or not shap_results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    pairs = [('Shallow', 'PO43-', '#d73027', axes[0]),
             ('Deep',    'Fe2+',  '#313695', axes[1])]

    for zone, feat, color, ax in pairs:
        if zone not in shap_results:
            continue
        info  = shap_results[zone]
        feats = info['feature_names']
        if feat not in feats:
            ax.text(0.5, 0.5, f'{feat} not available', ha='center', transform=ax.transAxes)
            continue

        idx        = feats.index(feat)
        shap_vals  = info['shap_values'][:, idx]
        feat_vals  = info['X'][:, idx]

        sc = ax.scatter(feat_vals, shap_vals, c=feat_vals, cmap='RdYlBu_r',
                        alpha=0.5, s=20)
        plt.colorbar(sc, ax=ax, label=feat)
        ax.axhline(0, color='grey', ls='--', lw=1)
        ax.set_xlabel(f'{feat} concentration', fontsize=11)
        ax.set_ylabel(f'SHAP value (impact on log As)', fontsize=11)
        ax.set_title(f'{zone} zone: SHAP dependence\n{feat}', fontsize=12)
        ax.grid(alpha=0.2)

    plt.suptitle('H2 Phase F: SHAP Dependence Plots\n'
                 '(Left: phosphate in shallow zone; Right: iron in deep zone)',
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H2_F15_shap_dependence.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H2_F15_shap_dependence.png")


# ─────────────────────────────────────────────
# PHASE F RUNNER
# ─────────────────────────────────────────────

def run_phase_f(df):
    """Run complete Phase F: SHAP variable importance by depth zone."""
    print("\n" + "█"*70)
    print(" PHASE F: SHAP VARIABLE IMPORTANCE BY DEPTH ZONE (H2.2 & H2.3)")
    print("█"*70)

    if not SHAP_AVAILABLE:
        print("  ⚠ shap library not installed. Run: pip install shap")
        print("  Falling back to RF feature importance only.")

    zone_models, perf_df = train_models_by_zone(df)
    global zone_models_ref
    zone_models_ref = zone_models

    shap_results = compute_shap_by_zone(zone_models)
    imp_df, frac_df = feature_importance_by_zone(zone_models, shap_results)

    if SHAP_AVAILABLE and shap_results:
        plot_shap_summary_by_zone(shap_results)
        plot_shap_dependence_po4_fe(shap_results, zone_models)

    plot_importance_comparison(imp_df, frac_df)

    print("\n✓ Phase F complete. Outputs:")
    print("  Tables : H2_T22, H2_T23, H2_T24")
    print("  Figures: H2_F13, H2_F14, H2_F15")

    return {
        'zone_models': zone_models,
        'shap_results': shap_results,
        'importance': imp_df,
        'process_fractions': frac_df,
        'performance': perf_df,
    }


if __name__ == "__main__":
    from data_loader import load_and_prepare_data
    df = load_and_prepare_data()
    run_phase_f(df)
