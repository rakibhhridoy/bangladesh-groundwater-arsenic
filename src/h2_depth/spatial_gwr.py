"""
Phase G-2 — Spatial Analysis: GWR + Kriging + Risk Maps
Geographically Weighted Regression (GWR) for spatially varying coefficients.
Ordinary Kriging stratified by depth zone for As risk maps.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
import matplotlib.patches as mpatches
from scipy import stats
from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    from mgwr.gwr import GWR, MGWR
    from mgwr.sel_bw import Sel_BW
    GWR_AVAILABLE = True
except ImportError:
    GWR_AVAILABLE = False

try:
    from pykrige.ok import OrdinaryKriging
    PYKRIGE_AVAILABLE = True
except ImportError:
    PYKRIGE_AVAILABLE = False

from config import (
    TARGET, DEPTH_ZONE_LABELS, ZONE_COLORS, WHO_AS_LIMIT,
    FIGURES_DIR, TABLES_DIR, MAPS_DIR, DPI, ALPHA,
    RANDOM_STATE, GWR_KERNEL
)

GEOGENIC_PRED  = 'Geogenic_Index'
ANTHRO_PRED    = 'PO43-'

# Memory-safety caps  ─────────────────────────────────────────────────────
# Sel_BW is O(n²) per iteration. Subsample for bandwidth search only.
GWR_BW_SUBSAMPLE  = 400   # max points used for Sel_BW bandwidth selection
# OrdinaryKriging inverts an n×n covariance matrix. Cap per zone.
KRIGE_MAX_N       = 400   # max training points per zone for kriging
# Kriging prediction grid (rows × cols). Keep small to limit cross-cov matrix.
KRIGE_GRID_SIZE   = 50    # 50×50 = 2500 grid points  (was 80×80 = 6400)
# ─────────────────────────────────────────────────────────────────────────


def _available(df, cols):
    return [c for c in cols if c in df.columns]


# ─────────────────────────────────────────────
# SPATIAL AUTOCORRELATION (Moran's I)
# ─────────────────────────────────────────────

def morans_i_by_zone(df):
    """
    Compute Moran's I for As in each depth zone.
    Tests whether spatial clustering of As differs by zone.
    Table H2_T28.
    """
    print("\n" + "="*70)
    print("PHASE G-2a: SPATIAL AUTOCORRELATION (Moran's I) BY DEPTH ZONE")
    print("="*70)

    try:
        from libpysal.weights import KNN
        from esda.moran import Moran
        SPATIAL_AVAILABLE = True
    except ImportError:
        SPATIAL_AVAILABLE = False
        print("  ⚠ libpysal/esda not available. Install: pip install libpysal esda")

    rows = []
    for zone in ['ALL'] + DEPTH_ZONE_LABELS:
        sub = df if zone == 'ALL' else df[df['Depth_Zone'] == zone]
        sub = sub[['Latitude', 'Longitude', TARGET]].dropna()

        if len(sub) < 20:
            continue

        if SPATIAL_AVAILABLE:
            try:
                coords = sub[['Latitude', 'Longitude']].values
                w = KNN.from_array(coords, k=min(8, len(sub) - 1))
                w.transform = 'R'
                mi = Moran(sub[TARGET].values, w)
                rows.append({
                    'Depth_Zone': zone, 'N': len(sub),
                    'Morans_I': round(mi.I, 5),
                    'Expected_I': round(mi.EI, 5),
                    'p_value': round(mi.p_sim, 6),
                    'z_score': round(mi.z_sim, 4),
                    'Significant': 'YES' if mi.p_sim < ALPHA else 'NO',
                    'Interpretation': 'Clustered' if mi.I > mi.EI and mi.p_sim < ALPHA else
                                      'Random' if mi.p_sim >= ALPHA else 'Dispersed',
                })
                print(f"  {zone:15s}: I={mi.I:.5f}  p={mi.p_sim:.4e}  [{rows[-1]['Interpretation']}]")
            except Exception as e:
                print(f"  WARNING {zone}: {e}")
        else:
            # Fallback: simple distance-weighted correlation
            coords = sub[['Latitude', 'Longitude']].values
            as_vals = sub[TARGET].values
            rows.append({
                'Depth_Zone': zone, 'N': len(sub),
                'Morans_I': np.nan, 'p_value': np.nan,
                'Significant': 'N/A', 'Interpretation': 'libpysal required',
            })

    moran_df = pd.DataFrame(rows)
    if len(moran_df) > 0:
        moran_df.to_csv(TABLES_DIR / 'H2_T28_morans_i_by_zone.csv', index=False)
    return moran_df


# ─────────────────────────────────────────────
# GEOGRAPHICALLY WEIGHTED REGRESSION
# ─────────────────────────────────────────────

def run_gwr(df):
    """
    GWR: log(As) ~ Geogenic_Index + PO43- with spatially varying coefficients.
    Produces local coefficient maps showing where geogenic/anthropogenic processes dominate.
    Table H2_T29.
    """
    print("\n" + "="*70)
    print("PHASE G-2b: GEOGRAPHICALLY WEIGHTED REGRESSION (GWR)")
    print("="*70)

    if not GWR_AVAILABLE:
        print("  ⚠ mgwr not available. Install: pip install mgwr")
        print("  Falling back to OLS with spatial stratification.")
        return _spatial_ols_fallback(df)

    y_col   = f'{TARGET}_log' if f'{TARGET}_log' in df.columns else TARGET
    geo_col = GEOGENIC_PRED if GEOGENIC_PRED in df.columns else 'ORP'
    ant_col = ANTHRO_PRED if ANTHRO_PRED in df.columns else 'NO3-'

    keep = ['Latitude', 'Longitude', y_col, geo_col, ant_col]
    complete = df[keep].dropna()
    if len(complete) < 50:
        print(f"  SKIP: n={len(complete)} too small for GWR.")
        return None

    # Standardise
    scaler = StandardScaler()
    coords = complete[['Latitude', 'Longitude']].values
    y      = complete[y_col].values.reshape(-1, 1)
    X      = scaler.fit_transform(complete[[geo_col, ant_col]].values)
    X      = np.hstack([np.ones((len(X), 1)), X])  # intercept

    try:
        # ── Bandwidth search on a subsample to avoid O(n²) OOM ────────────
        n_full = len(coords)
        if n_full > GWR_BW_SUBSAMPLE:
            rng   = np.random.RandomState(42)
            idx_s = rng.choice(n_full, GWR_BW_SUBSAMPLE, replace=False)
            coords_s, y_s, X_s = coords[idx_s], y[idx_s], X[idx_s]
            print(f"  Bandwidth search on subsample n={GWR_BW_SUBSAMPLE} "
                  f"(full n={n_full}) to prevent OOM")
        else:
            coords_s, y_s, X_s = coords, y, X

        bw_selector = Sel_BW(coords_s, y_s, X_s, kernel=GWR_KERNEL, fixed=False)
        bw = bw_selector.search()
        print(f"  Optimal bandwidth (AICc): {bw:.2f}")

        # ── Fit GWR on full dataset with the found bandwidth ───────────────
        gwr_model  = GWR(coords, y, X, bw, kernel=GWR_KERNEL, fixed=False)
        gwr_results= gwr_model.fit()

        # Local coefficients
        local_df = complete.copy()
        local_df['GWR_intercept']  = gwr_results.params[:, 0]
        local_df['GWR_geo_coef']   = gwr_results.params[:, 1]
        local_df['GWR_anthro_coef']= gwr_results.params[:, 2]
        local_df['GWR_R2_local']   = gwr_results.localR2
        local_df['GWR_dominant']   = np.where(
            abs(gwr_results.params[:, 1]) > abs(gwr_results.params[:, 2]),
            'Geogenic', 'Anthropogenic'
        )

        local_df.to_csv(TABLES_DIR / 'H2_T29_gwr_local_coefficients.csv', index=False)

        # Summary
        geo_dom  = (local_df['GWR_dominant'] == 'Geogenic').mean() * 100
        ant_dom  = 100 - geo_dom
        print(f"  GWR R² (global): {gwr_results.R2:.4f}")
        print(f"  Geogenic dominant locations : {geo_dom:.1f}%")
        print(f"  Anthropogenic dominant locs : {ant_dom:.1f}%")

        _plot_gwr_coefficients(local_df)
        return local_df, gwr_results

    except Exception as e:
        print(f"  WARNING GWR failed: {e}")
        return _spatial_ols_fallback(df)


def _spatial_ols_fallback(df):
    """OLS fallback when mgwr is not installed: zone-specific OLS spatial summary."""
    print("  Using OLS spatial fallback (zone-stratified)...")
    y_col   = f'{TARGET}_log' if f'{TARGET}_log' in df.columns else TARGET
    geo_col = GEOGENIC_PRED if GEOGENIC_PRED in df.columns else 'ORP'
    ant_col = ANTHRO_PRED if ANTHRO_PRED in df.columns else 'NO3-'

    rows = []
    for zone in DEPTH_ZONE_LABELS:
        sub = df[df['Depth_Zone'] == zone][['Latitude', 'Longitude', y_col, geo_col, ant_col]].dropna()
        if len(sub) < 15:
            continue
        scaler = StandardScaler()
        X = scaler.fit_transform(sub[[geo_col, ant_col]])
        Xc = np.column_stack([np.ones(len(X)), X])
        try:
            import statsmodels.api as sm
            model = sm.OLS(sub[y_col].values, Xc).fit()
            rows.append({
                'Depth_Zone': zone, 'N': len(sub),
                'R2': round(model.rsquared, 4),
                'Geo_Beta': round(model.params[1], 4),
                'Anthro_Beta': round(model.params[2], 4),
                'Dominant': 'Geogenic' if abs(model.params[1]) > abs(model.params[2]) else 'Anthropogenic',
            })
        except Exception:
            pass

    fallback_df = pd.DataFrame(rows)
    fallback_df.to_csv(TABLES_DIR / 'H2_T29_ols_spatial_fallback.csv', index=False)
    print(fallback_df.to_string(index=False))
    return fallback_df


def _plot_gwr_coefficients(local_df):
    """Figure H2_F18 — GWR local coefficient maps (geo vs. anthro)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Bangladesh bounding box
    lat_min, lat_max = 20.5, 26.8
    lon_min, lon_max = 87.8, 92.8

    grid_lat = np.linspace(lat_min, lat_max, 100)
    grid_lon = np.linspace(lon_min, lon_max, 100)
    grid_lon_mg, grid_lat_mg = np.meshgrid(grid_lon, grid_lat)

    for ax, col, title, cmap in [
        (axes[0], 'GWR_geo_coef',   'Geogenic β (GWR)',      'RdYlBu_r'),
        (axes[1], 'GWR_anthro_coef','Anthropogenic β (GWR)', 'RdYlBu_r'),
        (axes[2], 'GWR_R2_local',   'Local R² (GWR)',        'YlOrRd'),
    ]:
        zi = griddata(
            (local_df['Longitude'], local_df['Latitude']),
            local_df[col],
            (grid_lon_mg, grid_lat_mg),
            method='linear'
        )
        im = ax.contourf(grid_lon_mg, grid_lat_mg, zi, levels=20, cmap=cmap)
        ax.scatter(local_df['Longitude'], local_df['Latitude'],
                   c=local_df[col], cmap=cmap, s=15, alpha=0.5,
                   edgecolors='k', linewidths=0.3)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

    plt.suptitle('H2 Phase G: Geographically Weighted Regression Coefficient Maps\n'
                 '(Spatial variation in geogenic vs. anthropogenic As drivers)',
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(MAPS_DIR / 'H2_F18_gwr_coefficient_maps.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H2_F18_gwr_coefficient_maps.png")


# ─────────────────────────────────────────────
# KRIGING RISK MAPS BY DEPTH ZONE
# ─────────────────────────────────────────────

def kriging_risk_maps(df):
    """
    Ordinary Kriging for As prediction by depth zone.
    Generates 4 risk maps (one per zone) + WHO exceedance probability maps.
    """
    print("\n" + "="*70)
    print("PHASE G-2c: KRIGING RISK MAPS BY DEPTH ZONE")
    print("="*70)

    if not PYKRIGE_AVAILABLE:
        print("  ⚠ pykrige not available. Install: pip install pykrige")
        print("  Falling back to IDW interpolation.")
        return _idw_risk_maps(df)

    lat_min, lat_max = 20.5, 26.8
    lon_min, lon_max = 87.8, 92.8
    grid_lon = np.linspace(lon_min, lon_max, KRIGE_GRID_SIZE)
    grid_lat = np.linspace(lat_min, lat_max, KRIGE_GRID_SIZE)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    krig_rows = []
    for i, zone in enumerate(DEPTH_ZONE_LABELS):
        sub = df[df['Depth_Zone'] == zone][['Latitude', 'Longitude', TARGET]].dropna()
        ax  = axes[i]

        if len(sub) < 15:
            ax.text(0.5, 0.5, f'{zone}\nn={len(sub)} (insufficient)',
                    ha='center', transform=ax.transAxes, fontsize=12)
            continue

        # Cap training points — kriging inverts an n×n covariance matrix
        n_zone = len(sub)
        if n_zone > KRIGE_MAX_N:
            sub = sub.sample(n=KRIGE_MAX_N, random_state=RANDOM_STATE)
            print(f"  {zone}: subsampled to {KRIGE_MAX_N} points for kriging (was {n_zone})")

        # Log-transform for kriging
        as_log = np.log10(sub[TARGET].values + 1e-6)

        try:
            OK = OrdinaryKriging(
                sub['Longitude'].values, sub['Latitude'].values, as_log,
                variogram_model='spherical',
                enable_plotting=False, verbose=False
            )
            z, ss = OK.execute('grid', grid_lon, grid_lat)
            z_back = 10 ** z  # back-transform

            # Exceedance probability (log-normal approximation)
            z_thresh = np.log10(WHO_AS_LIMIT)
            pexceed  = 1 - stats.norm.cdf((z_thresh - z) / (np.sqrt(ss) + 1e-9))

            # Plot As prediction
            levels = [0, 5, 10, 25, 50, 100, 200, 500]
            cmap   = plt.cm.RdYlBu_r
            norm   = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            im     = ax.contourf(grid_lon, grid_lat, z_back, levels=levels,
                                 cmap=cmap, norm=norm, extend='max')
            ax.scatter(sub['Longitude'], sub['Latitude'],
                       c=sub[TARGET], cmap=cmap, norm=norm,
                       s=20, alpha=0.7, edgecolors='k', linewidths=0.3)
            plt.colorbar(im, ax=ax, label='As (µg/L)', shrink=0.8)

            ax.set_title(f'{zone}  (n={len(sub)})\nKriging — As (µg/L)', fontsize=11)
            ax.set_xlabel('Longitude', fontsize=9)
            ax.set_ylabel('Latitude', fontsize=9)
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)

            # Save exceedance map
            fig2, ax2 = plt.subplots(figsize=(8, 7))
            im2 = ax2.contourf(grid_lon, grid_lat, pexceed, levels=20,
                               cmap='Reds', vmin=0, vmax=1)
            ax2.scatter(sub['Longitude'], sub['Latitude'], c='blue', s=10, alpha=0.3)
            plt.colorbar(im2, ax=ax2, label='P(As > 10 µg/L)')
            ax2.set_title(f'{zone}: WHO Exceedance Probability Map', fontsize=12)
            ax2.set_xlim(lon_min, lon_max)
            ax2.set_ylim(lat_min, lat_max)
            fig2.savefig(MAPS_DIR / f'H2_F20_exceedance_prob_{zone}.png',
                         dpi=DPI, bbox_inches='tight')
            plt.close(fig2)

            krig_rows.append({
                'Depth_Zone': zone, 'N_wells': len(sub),
                'Kriged_mean_As': round(float(z_back.mean()), 2),
                'Kriged_max_As': round(float(z_back.max()), 2),
                'Mean_variance': round(float(ss.mean()), 4),
                'Area_above_WHO_pct': round(float((pexceed > 0.5).mean() * 100), 1),
            })
            print(f"  {zone}: kriging complete, mean As={z_back.mean():.2f} µg/L, "
                  f"area >WHO probability>50%: {(pexceed>0.5).mean()*100:.1f}%")

        except Exception as e:
            print(f"  WARNING {zone}: {e}")
            ax.text(0.5, 0.5, f'{zone}\nKriging failed:\n{str(e)[:60]}',
                    ha='center', transform=ax.transAxes, fontsize=9)

    plt.suptitle('H2 Phase G: Arsenic Risk Maps by Depth Zone (Kriging)\n'
                 '(Shallow: 0–30m | Intermediate: 30–80m | Medium_Deep: 80–150m | Deep: >150m)',
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(MAPS_DIR / 'H2_F19_kriging_risk_maps.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H2_F19_kriging_risk_maps.png")
    print("  Saved: H2_F20_exceedance_prob_*.png")

    krig_df = pd.DataFrame(krig_rows)
    if len(krig_df) > 0:
        krig_df.to_csv(TABLES_DIR / 'H2_T30_kriging_summary_by_zone.csv', index=False)
    return krig_df


def _idw_risk_maps(df):
    """IDW fallback when pykrige is not available."""
    print("  Using IDW interpolation fallback...")

    lat_min, lat_max = 20.5, 26.8
    lon_min, lon_max = 87.8, 92.8
    grid_lon = np.linspace(lon_min, lon_max, KRIGE_GRID_SIZE)
    grid_lat = np.linspace(lat_min, lat_max, KRIGE_GRID_SIZE)
    grid_lon_mg, grid_lat_mg = np.meshgrid(grid_lon, grid_lat)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for i, zone in enumerate(DEPTH_ZONE_LABELS):
        sub = df[df['Depth_Zone'] == zone][['Latitude', 'Longitude', TARGET]].dropna()
        ax  = axes[i]

        if len(sub) < 10:
            continue

        # IDW
        zi = griddata(
            (sub['Longitude'].values, sub['Latitude'].values),
            sub[TARGET].values,
            (grid_lon_mg, grid_lat_mg),
            method='linear'
        )

        im = ax.contourf(grid_lon_mg, grid_lat_mg, zi,
                         levels=[0,5,10,25,50,100,200,500],
                         cmap='RdYlBu_r', extend='max')
        ax.scatter(sub['Longitude'], sub['Latitude'],
                   c=sub[TARGET], cmap='RdYlBu_r', s=20, alpha=0.7)
        plt.colorbar(im, ax=ax, label='As (µg/L)')
        ax.set_title(f'{zone} (n={len(sub)}) — IDW Interpolation')

    plt.suptitle('H2: Arsenic Risk Maps by Depth Zone (IDW Interpolation)', fontsize=13)
    plt.tight_layout()
    fig.savefig(MAPS_DIR / 'H2_F19_idw_risk_maps.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H2_F19_idw_risk_maps.png")
    return None


# ─────────────────────────────────────────────
# PHASE G-2 RUNNER
# ─────────────────────────────────────────────

def run_phase_g2(df):
    """Run complete Phase G-2: Spatial analysis."""
    print("\n" + "█"*70)
    print(" PHASE G-2: SPATIAL ANALYSIS — GWR + KRIGING RISK MAPS")
    print("█"*70)

    moran_df = morans_i_by_zone(df)
    gwr_result = run_gwr(df)
    krig_df  = kriging_risk_maps(df)

    print("\n✓ Phase G-2 complete. Outputs:")
    print("  Tables : H2_T28, H2_T29, H2_T30")
    print("  Maps   : H2_F18 (GWR), H2_F19 (kriging), H2_F20 (exceedance)")

    return {
        'moran': moran_df,
        'gwr': gwr_result,
        'kriging': krig_df,
    }


if __name__ == "__main__":
    from data_loader import load_and_prepare_data
    df = load_and_prepare_data()
    run_phase_g2(df)
