"""
Phase F — Ensemble Satellite Forecast Model (H3.5)
RF + XGBoost + Ridge + ElasticNet stacked ensemble.
SHAP interaction effects. MAPIE conformal prediction intervals.
Leave-one-physiographic-zone-out transferability test.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from mapie.regression import MapieRegressor
    MAPIE_AVAILABLE = True
except ImportError:
    MAPIE_AVAILABLE = False

from config import (
    TARGET, DEPTH_ZONE_LABELS, PHYS_ZONE_LABELS, SATELLITE_VARS, ZONE_COLORS,
    FIGURES_DIR, TABLES_DIR, DPI, ALPHA, RANDOM_STATE,
    RF_N_ESTIMATORS, XGB_N_ESTIMATORS, CV_FOLDS, N_JOBS,
    SPATIAL_BLOCK_SIZE, BANGLADESH_BBOX
)


def _build_features(df_delta):
    """
    Build feature matrix for satellite forecast model.
    Features: satellite indices + depth zone (encoded) + phys zone (encoded) + depth.
    Target: ΔAs (Delta_As).
    """
    target_col = f'Delta_{TARGET}'
    feature_cols = [c for c in SATELLITE_VARS if c in df_delta.columns]
    feature_cols += [c for c in ['Depth'] if c in df_delta.columns]

    # Encode categorical zones
    df = df_delta.copy()
    for cat_col in ['Depth_Zone', 'Phys_Zone']:
        if cat_col in df.columns:
            le = LabelEncoder()
            df[cat_col + '_enc'] = le.fit_transform(df[cat_col].fillna('Unknown'))
            feature_cols.append(cat_col + '_enc')

    avail = [c for c in feature_cols if c in df.columns]
    complete = df[avail + [target_col, 'Latitude', 'Longitude',
                           'Phys_Zone' if 'Phys_Zone' in df.columns else avail[0]]
                  ].dropna(subset=[target_col])
    complete = complete.dropna(subset=avail[:3])  # require at least satellite cols

    X = complete[avail].values
    y = complete[target_col].values
    return X, y, avail, complete


def _spatial_blocks(lat, lon, block_size=SPATIAL_BLOCK_SIZE):
    """Assign spatial block IDs for block CV."""
    lat_block = (lat / block_size).astype(int)
    lon_block = (lon / block_size).astype(int)
    return lat_block * 1000 + lon_block


def build_ensemble():
    """Build stacked generalization ensemble."""
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_features='sqrt',
                                      min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=N_JOBS)),
        ('ridge', Ridge(alpha=1.0)),
        ('en',    ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000)),
    ]
    if XGB_AVAILABLE:
        estimators.insert(1, ('xgb', xgb.XGBRegressor(
            n_estimators=XGB_N_ESTIMATORS, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
            verbosity=0, n_jobs=N_JOBS)))

    meta = Ridge(alpha=1.0)
    stack = StackingRegressor(estimators=estimators, final_estimator=meta,
                               cv=5, n_jobs=1)  # n_jobs=1 for Mac safety
    return stack


def run_satellite_forecast(df_delta):
    """Phase F: ensemble satellite model with SHAP + conformal + transferability."""
    print("\n" + "█" * 70)
    print(" PHASE F: ENSEMBLE SATELLITE FORECAST MODEL (H3.5)")
    print("█" * 70)

    X, y, feature_names, complete = _build_features(df_delta)
    if len(complete) < 30:
        print(f"  SKIP: only {len(complete)} complete rows for satellite model")
        return None

    print(f"  Feature matrix: {X.shape}  |  Features: {feature_names}")

    # ── Spatial block CV ──────────────────────────────────────────────────
    blocks = _spatial_blocks(complete['Latitude'].values, complete['Longitude'].values)
    groups = blocks

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    ensemble = build_ensemble()

    print(f"\n  Running spatial block CV (n_splits={CV_FOLDS})...")
    gkf = GroupKFold(n_splits=min(CV_FOLDS, len(np.unique(groups))))
    cv_preds = cross_val_predict(ensemble, X_sc, y, cv=gkf, groups=groups, n_jobs=1)

    cv_r2   = r2_score(y, cv_preds)
    cv_rmse = np.sqrt(mean_squared_error(y, cv_preds))
    print(f"  Ensemble CV-R²={cv_r2:.4f}  RMSE={cv_rmse:.4f}")

    # Fit on full data
    ensemble.fit(X_sc, y)

    # Per-base-learner performance
    perf_rows = [{'Model': 'Ensemble (stacked)', 'CV_R2': round(cv_r2, 4),
                  'CV_RMSE': round(cv_rmse, 4), 'N': len(y)}]
    # named_estimators_ is a Bunch: {name: fitted_estimator}
    for name, est in ensemble.named_estimators_.items():
        try:
            sub_preds = cross_val_predict(est, X_sc, y,
                                           cv=gkf, groups=groups, n_jobs=1)
            perf_rows.append({
                'Model': name, 'CV_R2': round(r2_score(y, sub_preds), 4),
                'CV_RMSE': round(np.sqrt(mean_squared_error(y, sub_preds)), 4),
                'N': len(y),
            })
        except Exception:
            pass

    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(TABLES_DIR / 'H3_T16_ensemble_performance.csv', index=False)
    print(perf_df.to_string(index=False))

    # ── SHAP analysis ─────────────────────────────────────────────────────
    shap_df, interact_df = None, None
    if SHAP_AVAILABLE:
        shap_df, interact_df = _shap_analysis(ensemble, X_sc, feature_names, complete)

    # ── MAPIE conformal prediction ────────────────────────────────────────
    mapie_result = None
    if MAPIE_AVAILABLE:
        mapie_result = _conformal_prediction(X_sc, y, feature_names, complete, groups, gkf)

    # ── Transferability test ──────────────────────────────────────────────
    transfer_df = _transferability_test(X_sc, y, complete, feature_names, gkf)

    # ── Figures ───────────────────────────────────────────────────────────
    _plot_predicted_vs_observed(y, cv_preds)
    if mapie_result is not None:
        _plot_conformal_map(complete, mapie_result)

    print("\n✓ Phase F complete. Outputs: H3_T16-T20, H3_F12-F14")
    return {'performance': perf_df, 'shap': shap_df, 'shap_interactions': interact_df,
            'conformal': mapie_result, 'transferability': transfer_df,
            'scaler': scaler, 'model': ensemble, 'feature_names': feature_names}


def _shap_analysis(ensemble, X_sc, feature_names, complete):
    """SHAP values + interaction matrix for satellite features."""
    print("\n  Computing SHAP values...")
    try:
        # Use RF base learner for SHAP (TreeExplainer)
        rf_est = ensemble.named_estimators_.get('rf')
        if rf_est is None:
            print("  No RF base learner found for SHAP")
            return None, None

        n_bg = min(100, len(X_sc))
        explainer = shap.TreeExplainer(rf_est, data=X_sc[:n_bg])
        shap_vals = explainer.shap_values(X_sc)

        mean_abs = np.abs(shap_vals).mean(axis=0)
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_mean_abs': mean_abs,
            'SHAP_norm': mean_abs / (mean_abs.sum() + 1e-9),
        }).sort_values('SHAP_mean_abs', ascending=False)
        shap_df.to_csv(TABLES_DIR / 'H3_T17_shap_satellite_importance.csv', index=False)
        print("  SHAP feature importance saved")

        # SHAP interaction values (subsample — expensive)
        interact_df = None
        n_interact = min(200, len(X_sc))
        try:
            print(f"  Computing SHAP interaction values (n={n_interact})...")
            interact_vals = explainer.shap_interaction_values(X_sc[:n_interact])
            interact_mean = np.abs(interact_vals).mean(axis=0)
            interact_df = pd.DataFrame(interact_mean, index=feature_names, columns=feature_names)
            interact_df.to_csv(TABLES_DIR / 'H3_T18_shap_interaction_matrix.csv')
            print("  SHAP interaction matrix saved")
        except Exception as e:
            print(f"  SHAP interaction skipped: {e}")

        _plot_shap(shap_vals, X_sc, feature_names, interact_df)
        return shap_df, interact_df

    except Exception as e:
        print(f"  SHAP failed: {e}")
        return None, None


def _conformal_prediction(X_sc, y, feature_names, complete, groups, gkf):
    """MAPIE conformal prediction intervals + calibration assessment."""
    print("\n  Fitting MAPIE conformal predictor (α=0.10)...")
    try:
        base_rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE,
                                         n_jobs=N_JOBS)
        mapie = MapieRegressor(estimator=base_rf, method='plus', cv=5)
        mapie.fit(X_sc, y)
        y_pred, y_pis = mapie.predict(X_sc, alpha=0.10)

        coverage = float(np.mean((y >= y_pis[:, 0, 0]) & (y <= y_pis[:, 1, 0])))
        mean_width = float(np.mean(y_pis[:, 1, 0] - y_pis[:, 0, 0]))

        coverage_df = pd.DataFrame([{
            'Method': 'MAPIE (plus, α=0.10)',
            'Nominal_coverage': 0.90, 'Empirical_coverage': round(coverage, 4),
            'Mean_PI_width': round(mean_width, 3),
            'Calibrated': 'YES' if abs(coverage - 0.90) < 0.05 else 'NO',
        }])
        coverage_df.to_csv(TABLES_DIR / 'H3_T19_conformal_coverage.csv', index=False)
        print(f"  Conformal coverage: nominal=0.90  empirical={coverage:.3f}  "
              f"PI width={mean_width:.2f}  Calibrated: {'YES' if abs(coverage - 0.90) < 0.05 else 'NO'}")

        result = {'y_pred': y_pred, 'y_pis': y_pis, 'coverage': coverage,
                  'mean_width': mean_width, 'lats': complete['Latitude'].values,
                  'lons': complete['Longitude'].values}
        return result

    except Exception as e:
        print(f"  MAPIE failed: {e}. Try: pip install MAPIE")
        return None


def _transferability_test(X_sc, y, complete, feature_names, gkf):
    """Leave-one-physiographic-zone-out CV."""
    print("\n  Leave-one-zone-out transferability test...")
    if 'Phys_Zone' not in complete.columns:
        return None

    phys_zones = complete['Phys_Zone'].unique()
    transfer_rows = []
    rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=N_JOBS)

    for held_out_zone in phys_zones:
        train_mask = complete['Phys_Zone'] != held_out_zone
        test_mask  = complete['Phys_Zone'] == held_out_zone
        if train_mask.sum() < 20 or test_mask.sum() < 5:
            continue
        rf.fit(X_sc[train_mask], y[train_mask])
        preds = rf.predict(X_sc[test_mask])
        r2  = r2_score(y[test_mask], preds) if len(y[test_mask]) > 1 else np.nan
        rmse = np.sqrt(mean_squared_error(y[test_mask], preds))
        transfer_rows.append({
            'Held_out_zone': held_out_zone, 'N_test': int(test_mask.sum()),
            'N_train': int(train_mask.sum()),
            'Transfer_R2': round(r2, 4), 'Transfer_RMSE': round(rmse, 4),
        })
        print(f"  {held_out_zone:25s}: R²={r2:.4f}  RMSE={rmse:.4f}")

    transfer_df = pd.DataFrame(transfer_rows)
    transfer_df.to_csv(TABLES_DIR / 'H3_T20_transferability_lozo.csv', index=False)
    return transfer_df


def _plot_shap(shap_vals, X_sc, feature_names, interact_df):
    """H3_F13: SHAP beeswarm + interaction heatmap."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: beeswarm (bar fallback)
        ax = axes[0]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        order = np.argsort(mean_abs)
        colors = ['#d73027' if v > 0 else '#4575b4'
                  for v in shap_vals.mean(axis=0)[order]]
        ax.barh(range(len(order)), mean_abs[order],
                color=[ZONE_COLORS.get(feature_names[i], '#74c476') for i in order], alpha=0.8)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([feature_names[i] for i in order], fontsize=10)
        ax.set_xlabel('Mean |SHAP value| — impact on predicted ΔAs', fontsize=11)
        ax.set_title('H3.5: SHAP Feature Importance\n(Satellite + Zone predictors of ΔAs)', fontsize=11)
        ax.grid(axis='x', alpha=0.3)

        # Right: interaction heatmap
        ax2 = axes[1]
        if interact_df is not None:
            im = ax2.imshow(interact_df.values, cmap='RdBu_r', aspect='auto')
            ax2.set_xticks(range(len(interact_df.columns)))
            ax2.set_xticklabels(interact_df.columns, rotation=45, ha='right', fontsize=8)
            ax2.set_yticks(range(len(interact_df.index)))
            ax2.set_yticklabels(interact_df.index, fontsize=8)
            plt.colorbar(im, ax=ax2, label='Mean |SHAP interaction|')
            ax2.set_title('H3.5: SHAP Interaction Matrix\n'
                          '(Off-diagonal: feature pair interaction effects)', fontsize=11)
        else:
            ax2.text(0.5, 0.5, 'SHAP interactions\nnot computed', ha='center',
                     transform=ax2.transAxes, fontsize=12)

        plt.suptitle('H3 Phase F: SHAP Analysis — Satellite Predictors of Seasonal ΔAs',
                     fontsize=13)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / 'H3_F13_shap_satellite.png', dpi=DPI, bbox_inches='tight')
        plt.close()
        print("  Saved: H3_F13_shap_satellite.png")
    except Exception as e:
        print(f"  SHAP plot failed: {e}")


def _plot_predicted_vs_observed(y, cv_preds):
    """H3_F13: Predicted vs. observed ΔAs."""
    fig, ax = plt.subplots(figsize=(8, 7))
    vabs = np.percentile(np.abs(y), 95)
    ax.scatter(y, cv_preds, alpha=0.4, s=20, color='steelblue', edgecolors='k', lw=0.2)
    ax.plot([-vabs, vabs], [-vabs, vabs], 'k--', lw=2, label='1:1 line')
    r2 = r2_score(y, cv_preds)
    ax.set_xlabel('Observed ΔAs (µg/L)', fontsize=12)
    ax.set_ylabel('Predicted ΔAs — Satellite Ensemble (µg/L)', fontsize=12)
    ax.set_title(f'H3.5: Ensemble Satellite Model — Predicted vs. Observed ΔAs\n'
                 f'Spatial block CV-R²={r2:.4f}', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H3_F13_predicted_vs_observed.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H3_F13_predicted_vs_observed.png")


def _plot_conformal_map(complete, mapie_result):
    """H3_F14: Conformal prediction interval width map."""
    try:
        pi_width = mapie_result['y_pis'][:, 1, 0] - mapie_result['y_pis'][:, 0, 0]
        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(complete['Longitude'], complete['Latitude'],
                        c=pi_width, cmap='YlOrRd', s=30, alpha=0.8,
                        edgecolors='k', linewidths=0.3)
        plt.colorbar(sc, ax=ax, label='PI Width = uncertainty in predicted ΔAs (µg/L)')
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        ax.set_title(f'H3.5: Conformal Prediction Interval Width Map\n'
                     f'(Coverage: {mapie_result["coverage"]:.2%} nominal=90%)\n'
                     f'Wide intervals = high uncertainty → field sampling priority',
                     fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / 'H3_F14_conformal_uncertainty_map.png',
                    dpi=DPI, bbox_inches='tight')
        plt.close()
        print("  Saved: H3_F14_conformal_uncertainty_map.png")
    except Exception as e:
        print(f"  Conformal map plot failed: {e}")
