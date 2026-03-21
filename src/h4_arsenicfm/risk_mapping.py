"""
National Risk Mapping — ArseniCFM (H4)

Generates national-scale arsenic risk maps at ~10km resolution using
kNN interpolation from ArseniCFM predictions on the training data.

Outputs:
  - Continuous As concentration map (log10 µg/L)
  - Binary WHO exceedance probability map
  - Uncertainty map (interquartile range from kNN ensemble)
  - Zone-stratified risk summary table
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from scipy.interpolate import griddata
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from config import (
    FIGURES_DIR, TABLES_DIR, MAPS_DIR,
    BANGLADESH_BBOX, RISK_GRID_DEG, KNN_NEIGHBORS,
    WHO_AS_LIMIT, RANDOM_STATE, DPI,
    PHYS_ZONE_LABELS, PHYSIOGRAPHIC_ZONES
)


# ─── GRID GENERATION ──────────────────────────────────────────────────────────

def make_prediction_grid():
    """Generate lat/lon grid over Bangladesh bounding box."""
    lats = np.arange(BANGLADESH_BBOX['lat_min'],
                     BANGLADESH_BBOX['lat_max'], RISK_GRID_DEG)
    lons = np.arange(BANGLADESH_BBOX['lon_min'],
                     BANGLADESH_BBOX['lon_max'], RISK_GRID_DEG)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    grid_pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    return lat_grid, lon_grid, grid_pts


# ─── KNN RISK MAP ─────────────────────────────────────────────────────────────

def build_risk_maps_knn(df, X_sat, X_geo, X_ctx, y_conc_pred, y_exc_pred):
    """
    Spatial kNN interpolation of ArseniCFM predictions onto national grid.
    Uses distance-weighted kNN for smooth maps.

    Parameters
    ----------
    df           : DataFrame with Latitude, Longitude
    y_conc_pred  : (N,) predicted log10(As)
    y_exc_pred   : (N,) predicted P(As > 10 µg/L)
    """
    print("\n  Building national kNN risk maps...")
    lat_grid, lon_grid, grid_pts = make_prediction_grid()

    coords_train = df[['Latitude', 'Longitude']].values
    scaler = StandardScaler().fit(coords_train)
    coords_s = scaler.transform(coords_train)
    grid_s   = scaler.transform(grid_pts)

    # kNN for concentration
    knn_conc = KNeighborsRegressor(n_neighbors=KNN_NEIGHBORS, weights='distance')
    knn_conc.fit(coords_s, y_conc_pred)
    conc_pred = knn_conc.predict(grid_s)

    # kNN for exceedance probability
    knn_exc = KNeighborsRegressor(n_neighbors=KNN_NEIGHBORS, weights='distance')
    knn_exc.fit(coords_s, y_exc_pred)
    exc_pred = knn_exc.predict(grid_s).clip(0, 1)

    # Uncertainty: IQR from kNN neighbors
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=KNN_NEIGHBORS).fit(coords_s)
    _, indices = nn.kneighbors(grid_s)
    neighbor_conc = y_conc_pred[indices]
    uncertainty   = np.percentile(neighbor_conc, 75, axis=1) - \
                    np.percentile(neighbor_conc, 25, axis=1)

    # Reshape to grid
    conc_grid = conc_pred.reshape(lat_grid.shape)
    exc_grid  = exc_pred.reshape(lat_grid.shape)
    unc_grid  = uncertainty.reshape(lat_grid.shape)

    return lat_grid, lon_grid, conc_grid, exc_grid, unc_grid


# ─── GRIDDATA INTERPOLATION ───────────────────────────────────────────────────

def build_risk_maps_griddata(df, y_conc_pred, y_exc_pred):
    """Griddata interpolation (linear) — faster alternative to kNN."""
    print("\n  Building griddata interpolation maps...")
    lat_grid, lon_grid, grid_pts = make_prediction_grid()
    coords = df[['Latitude', 'Longitude']].values

    conc_grid = griddata(coords, y_conc_pred, (lat_grid, lon_grid), method='linear')
    exc_grid  = griddata(coords, y_exc_pred,  (lat_grid, lon_grid), method='linear')

    # Fill NaN from linear with nearest
    conc_nn  = griddata(coords, y_conc_pred,  (lat_grid, lon_grid), method='nearest')
    exc_nn   = griddata(coords, y_exc_pred,   (lat_grid, lon_grid), method='nearest')
    conc_grid = np.where(np.isfinite(conc_grid), conc_grid, conc_nn)
    exc_grid  = np.where(np.isfinite(exc_grid),  exc_grid,  exc_nn)
    exc_grid  = np.clip(exc_grid, 0, 1)

    return lat_grid, lon_grid, conc_grid, exc_grid, np.zeros_like(conc_grid)


# ─── PLOTTING ─────────────────────────────────────────────────────────────────

def plot_risk_maps(lat_grid, lon_grid, conc_grid, exc_grid, unc_grid,
                   df, y_conc_true, y_exc_true):
    """3-panel national risk map."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # Panel 1: As concentration
    ax = axes[0]
    cf = ax.contourf(lon_grid, lat_grid, conc_grid,
                     levels=20, cmap='RdYlBu_r', vmin=0, vmax=2.5)
    plt.colorbar(cf, ax=ax, label='Predicted log₁₀(As µg/L)')
    ax.scatter(df['Longitude'], df['Latitude'], c=y_conc_true,
               cmap='RdYlBu_r', s=8, alpha=0.5, edgecolors='k', linewidths=0.3,
               vmin=0, vmax=2.5, label='Observed')
    ax.set_title('ArseniCFM: Predicted As Concentration\nlog₁₀(µg/L)', fontsize=11)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.grid(alpha=0.3)

    # Panel 2: WHO exceedance probability
    ax = axes[1]
    cf2 = ax.contourf(lon_grid, lat_grid, exc_grid,
                      levels=20, cmap='RdBu_r', vmin=0, vmax=1)
    plt.colorbar(cf2, ax=ax, label='P(As > 10 µg/L)')
    ax.contour(lon_grid, lat_grid, exc_grid, levels=[0.5],
               colors='k', linewidths=1.5, linestyles='--')
    ax.set_title('ArseniCFM: WHO Exceedance Risk\nP(As > 10 µg/L)', fontsize=11)
    ax.set_xlabel('Longitude')
    ax.grid(alpha=0.3)

    # Panel 3: Uncertainty
    ax = axes[2]
    cf3 = ax.contourf(lon_grid, lat_grid, unc_grid,
                      levels=15, cmap='Purples', vmin=0)
    plt.colorbar(cf3, ax=ax, label='Prediction IQR (log₁₀ units)')
    ax.set_title('ArseniCFM: Prediction Uncertainty\n(kNN IQR)', fontsize=11)
    ax.set_xlabel('Longitude')
    ax.grid(alpha=0.3)

    # Add Bangladesh bounding box annotation
    for ax in axes:
        ax.set_xlim(BANGLADESH_BBOX['lon_min'], BANGLADESH_BBOX['lon_max'])
        ax.set_ylim(BANGLADESH_BBOX['lat_min'], BANGLADESH_BBOX['lat_max'])

    plt.suptitle('H4: ArseniCFM National Arsenic Risk Maps — Bangladesh\n'
                 'Multi-modal foundation model prediction (Alpha Earth + Geochemistry)',
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H4_F08_national_risk_map.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H4_F08_national_risk_map.png")


def plot_risk_summary_by_zone(df, y_conc_pred, y_exc_pred):
    """Bar chart of mean predicted As and exceedance % by physiographic zone."""
    df2 = df.copy()
    df2['pred_As_log'] = y_conc_pred
    df2['pred_As_ug']  = 10 ** np.clip(y_conc_pred, -3, 5)
    df2['pred_exc']    = (y_exc_pred > 0.5).astype(int)

    zone_summary = df2.groupby('Phys_Zone').agg(
        N=('pred_As_log', 'count'),
        pred_As_median=('pred_As_ug', 'median'),
        pred_exc_pct=('pred_exc', 'mean'),
    ).reset_index()
    zone_summary['pred_exc_pct'] *= 100
    zone_summary.to_csv(TABLES_DIR / 'H4_T11_zone_risk_summary.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(zone_summary))
    axes[0].bar(x, zone_summary['pred_As_median'], color='#d73027', alpha=0.8)
    axes[0].axhline(WHO_AS_LIMIT, color='black', ls='--', lw=2, label='WHO limit')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(zone_summary['Phys_Zone'].str[:12], rotation=30, ha='right')
    axes[0].set_ylabel('Median Predicted As (µg/L)')
    axes[0].set_title('ArseniCFM: Predicted As by Physiographic Zone')
    axes[0].legend()

    axes[1].bar(x, zone_summary['pred_exc_pct'], color='#4575b4', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(zone_summary['Phys_Zone'].str[:12], rotation=30, ha='right')
    axes[1].set_ylabel('WHO Exceedance Rate (%)')
    axes[1].set_title('ArseniCFM: Predicted Exceedance Rate by Zone')

    plt.suptitle('H4: Zonal Risk Summary', fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H4_F09_zone_risk_summary.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H4_F09_zone_risk_summary.png")
    return zone_summary


# ─── MAIN RUNNER ──────────────────────────────────────────────────────────────

def run_risk_mapping(df, X_sat, X_geo, X_ctx, y_conc_pred, y_exc_pred,
                     y_conc_true, y_exc_true):
    """Full risk mapping pipeline."""
    print("\n" + "█" * 65)
    print(" PHASE E: NATIONAL RISK MAPPING (H4)")
    print("█" * 65)

    # Try kNN first, fall back to griddata
    try:
        lat_grid, lon_grid, conc_grid, exc_grid, unc_grid = build_risk_maps_knn(
            df, X_sat, X_geo, X_ctx, y_conc_pred, y_exc_pred)
    except Exception as e:
        print(f"  kNN map failed ({e}); using griddata")
        lat_grid, lon_grid, conc_grid, exc_grid, unc_grid = build_risk_maps_griddata(
            df, y_conc_pred, y_exc_pred)

    # Save grid data
    np.save(MAPS_DIR / 'H4_risk_conc_grid.npy', conc_grid)
    np.save(MAPS_DIR / 'H4_risk_exc_grid.npy', exc_grid)
    np.save(MAPS_DIR / 'H4_risk_unc_grid.npy', unc_grid)

    # Point-level predictions CSV
    risk_pts = df[['Latitude', 'Longitude', 'Phys_Zone', 'Depth_Zone', 'As']].copy()
    risk_pts['pred_As_log10'] = y_conc_pred
    risk_pts['pred_As_ug']    = 10 ** np.clip(y_conc_pred, -3, 5)
    risk_pts['pred_exc_prob'] = y_exc_pred
    risk_pts['true_exc']      = y_exc_true
    risk_pts.to_csv(TABLES_DIR / 'H4_T12_risk_predictions.csv', index=False)

    plot_risk_maps(lat_grid, lon_grid, conc_grid, exc_grid, unc_grid,
                   df, y_conc_true, y_exc_true)
    zone_summary = plot_risk_summary_by_zone(df, y_conc_pred, y_exc_pred)

    print("\n✓ Phase E complete. Outputs: H4_T11, T12, F08, F09")
    return {'zone_summary': zone_summary, 'grid_shape': conc_grid.shape}
