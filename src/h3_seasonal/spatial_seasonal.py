"""
Phase G — Seasonal Moran's I Shift + 56-Cell Risk Matrix (H3.6)
Spatial autocorrelation of As changes seasonally → mechanism heterogeneity.
56-cell WHO exceedance risk matrix: 7 phys × 4 depth × 2 seasons.
Seasonal GWR: separate coefficient surfaces for Dry vs Wet.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from config import (
    TARGET, DEPTH_ZONE_LABELS, ZONE_COLORS, PHYS_ZONE_LABELS,
    SEASON_DRY, SEASON_WET, SEASON_COLORS,
    FIGURES_DIR, TABLES_DIR, MAPS_DIR,
    DPI, ALPHA, RANDOM_STATE, WHO_AS_LIMIT,
    BANGLADESH_BBOX, KNN_NEIGHBORS, KRIGE_GRID_SIZE
)

# ─── OPTIONAL IMPORTS ─────────────────────────────────────────────────────────
try:
    import libpysal.weights as lps_w
    from esda.moran import Moran, Moran_Local
    MORAN_AVAILABLE = True
except ImportError:
    MORAN_AVAILABLE = False
    print("  [WARN] libpysal/esda not available — Moran's I will be skipped")

try:
    from mgwr.gwr import GWR, MGWR
    from mgwr.sel_bw import Sel_BW
    import mgwr.sel_bw as sel_bw_mod
    GWR_AVAILABLE = True
except ImportError:
    GWR_AVAILABLE = False
    print("  [WARN] mgwr not available — GWR analysis will be skipped")


# ─── MORAN'S I ────────────────────────────────────────────────────────────────

def compute_morans_i(vals, lat, lon, k=KNN_NEIGHBORS):
    """
    Compute global Moran's I for a variable using k-NN spatial weights.
    Returns (I, expected_I, p_value, z_score) or None on failure.
    """
    if not MORAN_AVAILABLE:
        return None
    try:
        coords = list(zip(lon, lat))
        w = lps_w.KNN.from_array(coords, k=k)
        w.transform = 'r'
        mi = Moran(vals, w, permutations=499)
        return {
            'I': round(mi.I, 6),
            'EI': round(mi.EI, 6),
            'p_sim': round(mi.p_sim, 6),
            'z_sim': round(mi.z_sim, 4),
            'Significant': 'YES' if mi.p_sim < ALPHA else 'NO',
        }
    except Exception as e:
        print(f"    Moran's I failed: {e}")
        return None


def run_seasonal_morans(df):
    """
    Compute Moran's I for As separately in Dry and Wet season × depth zone.
    Returns DataFrame of results.
    """
    print("\n" + "=" * 70)
    print("PHASE G-1: SEASONAL MORAN'S I (H3.6)")
    print("=" * 70)

    rows = []
    for season in [SEASON_DRY, SEASON_WET]:
        sub_s = df[df['Season'] == season].copy()
        for zone in ['ALL'] + DEPTH_ZONE_LABELS:
            sub = sub_s if zone == 'ALL' else sub_s[sub_s['Depth_Zone'] == zone]
            sub = sub.dropna(subset=[TARGET, 'Latitude', 'Longitude'])
            if len(sub) < 20:
                continue

            vals = np.log1p(sub[TARGET].values)
            lat  = sub['Latitude'].values
            lon  = sub['Longitude'].values
            result = compute_morans_i(vals, lat, lon)
            if result is None:
                # Fallback: report NaN row
                result = {'I': np.nan, 'EI': np.nan, 'p_sim': np.nan,
                          'z_sim': np.nan, 'Significant': 'SKIP'}

            rows.append({
                'Season': season, 'Depth_Zone': zone, 'N': len(sub),
                **result,
            })
            print(f"  {season:4s} | {zone:15s}: I={result['I']:.5f}  "
                  f"p={result['p_sim']:.4f}  sig={result['Significant']}")

    moran_df = pd.DataFrame(rows)
    moran_df.to_csv(TABLES_DIR / 'H3_T21_seasonal_morans_i.csv', index=False)

    # Delta-I: Wet − Dry (clustered As changes sign → mechanisms shift)
    delta_rows = []
    for zone in ['ALL'] + DEPTH_ZONE_LABELS:
        sub_df = moran_df[moran_df['Depth_Zone'] == zone]
        dry_row = sub_df[sub_df['Season'] == SEASON_DRY]
        wet_row = sub_df[sub_df['Season'] == SEASON_WET]
        if len(dry_row) and len(wet_row):
            delta_I = wet_row['I'].values[0] - dry_row['I'].values[0]
            delta_rows.append({
                'Depth_Zone': zone,
                'I_dry': dry_row['I'].values[0],
                'I_wet': wet_row['I'].values[0],
                'Delta_I': round(delta_I, 6),
                'Interpretation': (
                    'Clustering INCREASES wet season → spatially coherent mobilization'
                    if delta_I > 0.05 else
                    'Clustering DECREASES wet season → dilution disperses spatial pattern'
                    if delta_I < -0.05 else
                    'No substantial change in spatial clustering'
                ),
            })
            print(f"  ΔI(wet−dry) {zone:15s}: {delta_I:+.5f}")

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(TABLES_DIR / 'H3_T22_delta_morans_i.csv', index=False)
    return moran_df, delta_df


# ─── LISA CLUSTERS ────────────────────────────────────────────────────────────

def compute_lisa_clusters(df, season_label, depth_zone='ALL'):
    """
    Local Indicators of Spatial Association (LISA) for a given season/zone.
    Returns df with LISA cluster labels and significance.
    """
    if not MORAN_AVAILABLE:
        return None

    sub = df[df['Season'] == season_label]
    if depth_zone != 'ALL':
        sub = sub[sub['Depth_Zone'] == depth_zone]
    sub = sub.dropna(subset=[TARGET, 'Latitude', 'Longitude']).copy()
    if len(sub) < 20:
        return None

    try:
        coords = list(zip(sub['Longitude'].values, sub['Latitude'].values))
        w = lps_w.KNN.from_array(coords, k=KNN_NEIGHBORS)
        w.transform = 'r'
        vals = np.log1p(sub[TARGET].values)
        lm = Moran_Local(vals, w, permutations=499)

        # LISA quad labels: 1=HH, 2=LH, 3=LL, 4=HL, 0=ns
        quad_labels = {0: 'ns', 1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
        sub = sub.copy()
        sub['LISA_q'] = lm.q
        sub['LISA_p'] = lm.p_sim
        sub['LISA_label'] = sub.apply(
            lambda r: quad_labels[int(r['LISA_q'])] if r['LISA_p'] < ALPHA else 'ns',
            axis=1
        )
        sub['LISA_Is'] = lm.Is
        return sub
    except Exception as e:
        print(f"    LISA failed: {e}")
        return None


def run_lisa_maps(df):
    """Generate LISA cluster maps for Dry and Wet seasons (Shallow zone)."""
    print("\n" + "=" * 70)
    print("PHASE G-2: LISA CLUSTER MAPS")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    quad_colors = {'HH': '#d73027', 'HL': '#fc8d59', 'LH': '#91bfdb',
                   'LL': '#4575b4', 'ns': '#cccccc'}

    all_lisa = {}
    for ax, season in zip(axes, [SEASON_DRY, SEASON_WET]):
        lisa_df = compute_lisa_clusters(df, season_label=season, depth_zone='ALL')
        if lisa_df is None:
            ax.set_title(f'{season}: LISA not available')
            continue
        all_lisa[season] = lisa_df

        for quad, color in quad_colors.items():
            mask = lisa_df['LISA_label'] == quad
            if mask.sum() == 0:
                continue
            ax.scatter(lisa_df.loc[mask, 'Longitude'],
                       lisa_df.loc[mask, 'Latitude'],
                       c=color, s=25, alpha=0.85, label=quad,
                       edgecolors='k', linewidths=0.2)

        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        ax.set_title(f'H3.6: LISA Clusters — {season} Season\n'
                     f'(HH=High-high hotspot, LL=Low-low coldspot)', fontsize=10)
        ax.legend(fontsize=8, title='Cluster type')
        ax.grid(alpha=0.2)

    plt.suptitle('H3 Phase G: Local Spatial Autocorrelation of Arsenic\n'
                 '(Change in cluster type reveals mechanism shift)', fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H3_F15_lisa_cluster_maps.png', dpi=DPI,
                bbox_inches='tight')
    plt.close()
    print("  Saved: H3_F15_lisa_cluster_maps.png")

    # LISA transition table: fraction HH that switch to LL
    if SEASON_DRY in all_lisa and SEASON_WET in all_lisa:
        _save_lisa_transition(all_lisa[SEASON_DRY], all_lisa[SEASON_WET])

    return all_lisa


def _save_lisa_transition(dry_df, wet_df):
    """Count wells that transition between LISA classes across seasons."""
    merged = dry_df[['Latitude', 'Longitude', 'LISA_label']].rename(
        columns={'LISA_label': 'LISA_dry'}).merge(
        wet_df[['Latitude', 'Longitude', 'LISA_label']].rename(
            columns={'LISA_label': 'LISA_wet'}),
        on=['Latitude', 'Longitude'], how='inner')

    ct = pd.crosstab(merged['LISA_dry'], merged['LISA_wet'],
                     margins=True, margins_name='Total')
    ct.to_csv(TABLES_DIR / 'H3_T23_lisa_season_transition.csv')
    print(f"  LISA transition table: {len(merged)} matched wells")


# ─── 56-CELL RISK MATRIX ──────────────────────────────────────────────────────

def build_56cell_risk_matrix(df):
    """
    7 physiographic zones × 4 depth zones × 2 seasons = 56 cells.
    Each cell: N, % above WHO 10 µg/L, median As.
    """
    print("\n" + "=" * 70)
    print("PHASE G-3: 56-CELL WHO RISK MATRIX (H3.6)")
    print("=" * 70)

    rows = []
    for season in [SEASON_DRY, SEASON_WET]:
        sub_s = df[df['Season'] == season]
        for phys in PHYS_ZONE_LABELS:
            sub_p = sub_s[sub_s.get('Phys_Zone', pd.Series(dtype=str)) == phys] if 'Phys_Zone' in df.columns else sub_s
            for depth in DEPTH_ZONE_LABELS:
                sub = sub_p[sub_p['Depth_Zone'] == depth].dropna(subset=[TARGET])
                n = len(sub)
                if n == 0:
                    rows.append({
                        'Season': season, 'Phys_Zone': phys,
                        'Depth_Zone': depth, 'N': 0,
                        'Exceedance_pct': np.nan,
                        'Median_As': np.nan,
                        'Risk_Level': 'Insufficient',
                    })
                    continue
                exc = (sub[TARGET] > WHO_AS_LIMIT).mean() * 100
                med_as = sub[TARGET].median()
                risk = ('Critical' if exc > 60 else
                        'High'     if exc > 40 else
                        'Moderate' if exc > 20 else
                        'Low')
                rows.append({
                    'Season': season, 'Phys_Zone': phys,
                    'Depth_Zone': depth, 'N': n,
                    'Exceedance_pct': round(exc, 2),
                    'Median_As': round(med_as, 3),
                    'Risk_Level': risk,
                })

    risk_df = pd.DataFrame(rows)
    risk_df.to_csv(TABLES_DIR / 'H3_T24_56cell_risk_matrix.csv', index=False)
    print(f"  Risk matrix: {len(risk_df)} cells saved")

    # Print summary for Phys_Zone if available
    if 'Phys_Zone' in df.columns:
        _print_risk_summary(risk_df)
        _plot_risk_matrix_heatmap(risk_df)
        _compute_seasonal_risk_amplification(risk_df)

    return risk_df


def _print_risk_summary(risk_df):
    print("\n  56-cell risk matrix summary (mean exceedance by zone × season):")
    pivot = risk_df.groupby(['Phys_Zone', 'Season'])['Exceedance_pct'].mean().unstack()
    print(pivot.to_string())


def _compute_seasonal_risk_amplification(risk_df):
    """
    Seasonal Risk Amplification (SRA) = exceedance_wet / exceedance_dry.
    SRA > 1 means monsoon worsens arsenic risk.
    """
    sra_rows = []
    dry_df = risk_df[risk_df['Season'] == SEASON_DRY].set_index(['Phys_Zone', 'Depth_Zone'])
    wet_df = risk_df[risk_df['Season'] == SEASON_WET].set_index(['Phys_Zone', 'Depth_Zone'])
    common = dry_df.index.intersection(wet_df.index)

    for idx in common:
        exc_dry = dry_df.loc[idx, 'Exceedance_pct']
        exc_wet = wet_df.loc[idx, 'Exceedance_pct']
        if pd.isna(exc_dry) or pd.isna(exc_wet) or exc_dry < 1:
            continue
        sra = exc_wet / exc_dry
        sra_rows.append({
            'Phys_Zone': idx[0], 'Depth_Zone': idx[1],
            'Exc_dry': exc_dry, 'Exc_wet': exc_wet,
            'SRA': round(sra, 4),
            'Interpretation': (
                'Monsoon AMPLIFIES risk (SRA>1.2)' if sra > 1.2 else
                'Monsoon REDUCES risk (SRA<0.8)'   if sra < 0.8 else
                'Risk stable across seasons'
            ),
        })

    sra_df = pd.DataFrame(sra_rows)
    sra_df.to_csv(TABLES_DIR / 'H3_T25_seasonal_risk_amplification.csv', index=False)
    if len(sra_df):
        print(f"\n  SRA > 1.2 (monsoon amplifies): "
              f"{(sra_df['SRA'] > 1.2).sum()} / {len(sra_df)} cells")
        print(f"  Mean SRA: {sra_df['SRA'].mean():.3f}")
    return sra_df


def _plot_risk_matrix_heatmap(risk_df):
    """H3_F16: Heatmap of 56-cell risk matrix for Dry and Wet seasons."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    for ax, season in zip(axes, [SEASON_DRY, SEASON_WET]):
        sub = risk_df[risk_df['Season'] == season]
        if sub.empty:
            continue
        pivot = sub.pivot_table(
            index='Phys_Zone', columns='Depth_Zone',
            values='Exceedance_pct', aggfunc='mean'
        ).reindex(columns=DEPTH_ZONE_LABELS)
        # order rows by mean exceedance
        pivot = pivot.reindex(
            pivot.mean(axis=1).sort_values(ascending=False).index
        )

        im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn_r',
                       vmin=0, vmax=100)
        ax.set_xticks(range(len(DEPTH_ZONE_LABELS)))
        ax.set_xticklabels(DEPTH_ZONE_LABELS, rotation=20, fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(f'{season} Season\n% Wells > WHO 10 µg/L', fontsize=11)
        plt.colorbar(im, ax=ax, label='Exceedance %')

        for i in range(len(pivot.index)):
            for j in range(len(DEPTH_ZONE_LABELS)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                            fontsize=8,
                            color='white' if val > 60 else 'black')

    plt.suptitle('H3 Phase G: 56-Cell WHO Arsenic Exceedance Risk Matrix\n'
                 '(7 Physiographic × 4 Depth × 2 Season)', fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H3_F16_risk_matrix_heatmap.png', dpi=DPI,
                bbox_inches='tight')
    plt.close()
    print("  Saved: H3_F16_risk_matrix_heatmap.png")


# ─── SEASONAL GWR ─────────────────────────────────────────────────────────────

def run_seasonal_gwr(df):
    """
    Separate GWR fits for Dry and Wet seasons.
    Δ(GWR coefficients) = β_wet − β_dry reveals spatial shift in mechanism.
    """
    print("\n" + "=" * 70)
    print("PHASE G-4: SEASONAL GWR — COEFFICIENT SHIFT (H3.6)")
    print("=" * 70)

    if not GWR_AVAILABLE:
        print("  SKIP: mgwr not installed — install with: pip install mgwr")
        return None, None

    results = {}
    for season in [SEASON_DRY, SEASON_WET]:
        sub = df[df['Season'] == season].dropna(
            subset=[TARGET, 'Latitude', 'Longitude', 'Fe2+', 'PO43-', 'ORP']
        ).copy()
        if len(sub) < 30:
            print(f"  SKIP {season}: n={len(sub)} < 30")
            continue

        coords = np.column_stack([sub['Longitude'].values, sub['Latitude'].values])
        y = np.log1p(sub[TARGET].values).reshape(-1, 1)
        X = np.column_stack([
            sub['Fe2+'].values,
            sub['PO43-'].values,
            sub['ORP'].values,
        ])
        # Standardise X
        X_mean = X.mean(axis=0)
        X_std  = X.std(axis=0) + 1e-9
        X_sc   = (X - X_mean) / X_std

        try:
            selector = Sel_BW(coords, y, X_sc)
            bw = selector.search(bw_min=20, bw_max=min(len(sub) - 1, 200))
            gwr_model = GWR(coords, y, X_sc, bw=bw)
            gwr_res   = gwr_model.fit()

            results[season] = {
                'bw': bw, 'n': len(sub),
                'coords': coords,
                'lat': sub['Latitude'].values,
                'lon': sub['Longitude'].values,
                'coef_fe':  gwr_res.params[:, 1],
                'coef_po4': gwr_res.params[:, 2],
                'coef_orp': gwr_res.params[:, 3],
                'R2_local': gwr_res.localR2,
            }
            print(f"  {season}: bw={bw:.1f}, n={len(sub)}")
            print(f"    β_Fe  : mean={gwr_res.params[:, 1].mean():+.4f}")
            print(f"    β_PO₄ : mean={gwr_res.params[:, 2].mean():+.4f}")
            print(f"    β_ORP : mean={gwr_res.params[:, 3].mean():+.4f}")
        except Exception as e:
            print(f"  GWR {season} failed: {e}")

    # Δ coefficients
    if SEASON_DRY in results and SEASON_WET in results:
        _save_gwr_delta(results)
        _plot_gwr_coefficients(results)

    return results


def _save_gwr_delta(results):
    """Save summary of GWR coefficient shift wet−dry."""
    dry = results[SEASON_DRY]
    wet = results[SEASON_WET]

    # Build summary table (mean coefficients)
    rows = []
    for var, d_key, w_key in [
        ('Fe2+',  'coef_fe',  'coef_fe'),
        ('PO43-', 'coef_po4', 'coef_po4'),
        ('ORP',   'coef_orp', 'coef_orp'),
    ]:
        b_dry = dry[d_key].mean()
        b_wet = wet[w_key].mean()
        rows.append({
            'Variable': var,
            'Beta_dry_mean': round(b_dry, 5),
            'Beta_wet_mean': round(b_wet, 5),
            'Delta_beta': round(b_wet - b_dry, 5),
            'Interpretation': (
                f'{var} effect STRENGTHENS in wet season (Δβ>0)' if b_wet > b_dry
                else f'{var} effect WEAKENS in wet season (Δβ<0)'
            ),
        })
    gwr_df = pd.DataFrame(rows)
    gwr_df.to_csv(TABLES_DIR / 'H3_T26_gwr_seasonal_delta.csv', index=False)
    print("\n  GWR Δ-coefficient table:")
    print(gwr_df.to_string(index=False))


def _plot_gwr_coefficients(results):
    """H3_F17: Side-by-side GWR coefficient maps for Fe²⁺ and PO₄³⁻."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    var_keys = [('Fe2+', 'coef_fe'), ('PO43-', 'coef_po4')]
    seasons  = [SEASON_DRY, SEASON_WET]

    for row_idx, (var_label, key) in enumerate(var_keys):
        vals_all = np.concatenate([results[s][key] for s in seasons
                                   if s in results])
        vabs = np.nanpercentile(np.abs(vals_all), 95)

        for col_idx, season in enumerate(seasons):
            if season not in results:
                continue
            ax   = axes[row_idx][col_idx]
            res  = results[season]
            sc   = ax.scatter(res['lon'], res['lat'],
                              c=res[key], cmap='RdBu_r',
                              vmin=-vabs, vmax=vabs, s=30,
                              alpha=0.85, edgecolors='k', linewidths=0.2)
            plt.colorbar(sc, ax=ax, label=f'GWR β ({var_label})')
            ax.set_xlabel('Longitude', fontsize=9)
            ax.set_ylabel('Latitude', fontsize=9)
            ax.set_title(f'{var_label} — {season} Season GWR β', fontsize=10)
            ax.grid(alpha=0.2)

    plt.suptitle('H3 Phase G: Seasonal GWR Coefficient Maps\n'
                 '(Colour shift reveals spatial mechanism change between seasons)',
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H3_F17_gwr_seasonal_coefficients.png',
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H3_F17_gwr_seasonal_coefficients.png")


# ─── MAIN ENTRY ───────────────────────────────────────────────────────────────

def run_spatial_seasonal(df, df_delta):
    """Run complete Phase G: Moran → LISA → 56-cell risk → seasonal GWR."""
    print("\n" + "█" * 70)
    print(" PHASE G: SEASONAL MORAN'S I + 56-CELL RISK MATRIX + GWR (H3.6)")
    print("█" * 70)

    moran_df, delta_df = run_seasonal_morans(df)
    lisa_results       = run_lisa_maps(df)
    risk_df            = build_56cell_risk_matrix(df)
    gwr_results        = run_seasonal_gwr(df)

    print("\n✓ Phase G complete. Outputs: H3_T21-T26, H3_F15-F17")
    return {
        'moran': moran_df, 'delta_moran': delta_df,
        'lisa':  lisa_results, 'risk_matrix': risk_df,
        'gwr':   gwr_results,
    }
