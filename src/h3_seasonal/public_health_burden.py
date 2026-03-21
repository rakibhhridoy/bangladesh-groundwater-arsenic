"""
Public Health Burden Analysis — Bangladesh Groundwater Arsenic
=================================================================
Estimates population-at-risk by overlaying zone/depth exceedance rates
with WorldPop Bangladesh 2020 population raster.

Outputs:
  H3_T33_population_at_risk_by_zone.csv
  H3_T34_seasonal_exposure_burden.csv
  H3_T35_depth_substitution_benefit.csv
  H3_T36_district_attributable_burden.csv
  H3_F22_population_risk_map.png

Usage:
    python public_health_burden.py              # downloads WorldPop if needed
    python public_health_burden.py --no-download  # use offline if file exists
"""

import numpy as np
import pandas as pd
import warnings
import argparse
import sys
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent / 'hypo1'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'hypo3'))

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
HYPO1_OUT    = PROJECT_ROOT / 'hypo1' / 'output'
HYPO3_OUT    = PROJECT_ROOT / 'hypo3' / 'output'
TABLES_DIR   = HYPO3_OUT / 'tables'
FIGURES_DIR  = HYPO3_OUT / 'figures'
DATA_DIR     = PROJECT_ROOT / 'data'

INTEGRATED_FILE = HYPO1_OUT / 'bangladesh_groundwater_arsenic_1807samples.csv'
WORLDPOP_FILE   = DATA_DIR / 'bgd_ppp_2020_1km_Aggregated.tif'
WORLDPOP_URL    = (
    'https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/'
    '2020/BGD/bgd_ppp_2020_1km_Aggregated.tif'
)

WHO_LIMIT      = 10.0
BANGLADESH_POP = 169_356_251   # Census 2022 estimate

# Physiographic zone bounding boxes (lat_min, lat_max, lon_min, lon_max)
ZONE_BBOX = {
    'Barind_Tract':           (23.8, 25.6, 88.0, 89.2),
    'Northern_Terrace':       (25.2, 26.7, 88.0, 89.8),
    'Brahmaputra_Floodplain': (23.5, 25.5, 89.5, 91.2),
    'Ganges_Floodplain':      (23.0, 24.5, 88.5, 90.2),
    'GBM_Delta':              (21.5, 23.5, 89.0, 90.5),
    'Meghna_Floodplain':      (22.5, 24.5, 90.5, 91.5),
    'Eastern_Hills':          (21.5, 24.8, 91.5, 92.7),
}

DEPTH_ZONES = {
    'Shallow':      (0,   30),
    'Intermediate': (30,  80),
    'Medium_Deep':  (80,  150),
    'Deep':         (150, 9999),
}


# ── WorldPop download ─────────────────────────────────────────────────────────

def download_worldpop():
    """Download WorldPop Bangladesh 2020 1km raster if not cached."""
    if WORLDPOP_FILE.exists():
        print(f"  WorldPop file found: {WORLDPOP_FILE}")
        return True
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import urllib.request
        print(f"  Downloading WorldPop Bangladesh 2020 (~15 MB)...")
        print(f"  URL: {WORLDPOP_URL}")
        urllib.request.urlretrieve(WORLDPOP_URL, WORLDPOP_FILE)
        print(f"  Saved: {WORLDPOP_FILE}")
        return True
    except Exception as e:
        print(f"  [WARN] Download failed: {e}")
        print(f"  Manually download from: {WORLDPOP_URL}")
        print(f"  Save to: {WORLDPOP_FILE}")
        return False


def load_worldpop():
    """Load WorldPop raster, return (lats, lons, pop_grid)."""
    try:
        import rasterio
        with rasterio.open(WORLDPOP_FILE) as src:
            pop = src.read(1).astype(np.float32)
            pop = np.where(pop < 0, 0, pop)   # no-data → 0
            h, w = pop.shape
            bounds = src.bounds
            lons = np.linspace(bounds.left,  bounds.right, w)
            lats = np.linspace(bounds.top,   bounds.bottom, h)
            return lats, lons, pop
    except ImportError:
        print("  [WARN] rasterio not installed — using proxy population model")
        return None, None, None
    except Exception as e:
        print(f"  [WARN] WorldPop load error: {e} — using proxy model")
        return None, None, None


def zone_population(lats, lons, pop, lat_min, lat_max, lon_min, lon_max):
    """Sum population within a bounding box."""
    if pop is None:
        return None
    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    lon_mask = (lons >= lon_min) & (lons <= lon_max)
    sub = pop[np.ix_(np.where(lat_mask)[0], np.where(lon_mask)[0])]
    return float(sub.sum())


# ── Proxy population model (fallback if rasterio/WorldPop unavailable) ────────

# Fraction of Bangladesh population per physiographic zone
# from literature (BGS/DPHE 2001, BBS 2022 census)
ZONE_POP_FRACTION = {
    'Barind_Tract':           0.073,
    'Northern_Terrace':       0.081,
    'Brahmaputra_Floodplain': 0.168,
    'Ganges_Floodplain':      0.141,
    'GBM_Delta':              0.225,
    'Meghna_Floodplain':      0.189,
    'Eastern_Hills':          0.123,
}


def get_zone_populations(lats, lons, pop_grid):
    """Return dict of zone → estimated population using WorldPop or proxy."""
    pops = {}
    if pop_grid is not None:
        total_extracted = 0
        for zone, (lat_min, lat_max, lon_min, lon_max) in ZONE_BBOX.items():
            p = zone_population(lats, lons, pop_grid,
                                lat_min, lat_max, lon_min, lon_max)
            pops[zone] = p if p is not None else 0
            total_extracted += pops[zone]
        # Normalise to match official census total × groundwater-dependent fraction
        gw_fraction = 0.80   # ~80% of Bangladesh population uses groundwater
        scale = BANGLADESH_POP * gw_fraction / total_extracted if total_extracted > 0 else 1
        pops = {z: v * scale for z, v in pops.items()}
        print(f"  WorldPop total extracted: {total_extracted:,.0f} → "
              f"scaled to {sum(pops.values()):,.0f} GW-dependent")
    else:
        gw_dep = BANGLADESH_POP * 0.80
        for zone, frac in ZONE_POP_FRACTION.items():
            pops[zone] = gw_dep * frac
        print(f"  Using proxy population fractions (WorldPop unavailable)")
    return pops


# ── Load exceedance data ───────────────────────────────────────────────────────

def load_exceedance_data(df):
    """Compute WHO exceedance rates by zone × depth × season from raw data."""
    results = []
    for zone, (lat_min, lat_max, lon_min, lon_max) in ZONE_BBOX.items():
        mask_zone = (
            (df['Latitude']  >= lat_min) & (df['Latitude']  <= lat_max) &
            (df['Longitude'] >= lon_min) & (df['Longitude'] <= lon_max)
        )
        for depth_name, (d_min, d_max) in DEPTH_ZONES.items():
            mask_depth = (df['Depth'] >= d_min) & (df['Depth'] < d_max)
            mask_both  = mask_zone & mask_depth
            sub = df[mask_both]
            if len(sub) < 5:
                continue
            for season in ['Dry', 'Wet', 'All']:
                if season == 'All':
                    s = sub
                elif 'Season' in df.columns:
                    s = sub[sub['Season'] == season]
                else:
                    continue
                if len(s) < 5:
                    continue
                exc = (s['As'] > WHO_LIMIT).mean() if 'As' in s.columns \
                    else (s['Arsenic_ugL'] > WHO_LIMIT).mean()
                results.append({
                    'Zone': zone,
                    'Depth_Zone': depth_name,
                    'Season': season,
                    'N': len(s),
                    'Exceedance_Rate': exc,
                    'Mean_As': s['As'].mean() if 'As' in s.columns
                               else s['Arsenic_ugL'].mean(),
                })
    return pd.DataFrame(results)


def load_sra_data():
    """Load pre-computed SRA from H3 output."""
    sra_path = TABLES_DIR / 'H3_T25_seasonal_risk_amplification.csv'
    if sra_path.exists():
        return pd.read_csv(sra_path)
    return None


# ── Table 1: Population at risk by zone ───────────────────────────────────────

def table_population_at_risk(df, zone_pops, exc_df):
    """H3_T33: Population at risk per zone (dry and wet season)."""
    rows = []
    for zone, pop in zone_pops.items():
        zone_exc = exc_df[exc_df['Zone'] == zone]

        # Overall zone exceedance (all depths, both seasons)
        exc_all = zone_exc[zone_exc['Season'] == 'All']
        exc_dry = zone_exc[zone_exc['Season'] == 'Dry']
        exc_wet = zone_exc[zone_exc['Season'] == 'Wet']

        exc_rate_all = exc_all['Exceedance_Rate'].mean() if len(exc_all) else np.nan
        exc_rate_dry = exc_dry['Exceedance_Rate'].mean() if len(exc_dry) else np.nan
        exc_rate_wet = exc_wet['Exceedance_Rate'].mean() if len(exc_wet) else np.nan

        par_dry = pop * exc_rate_dry if not np.isnan(exc_rate_dry) else np.nan
        par_wet = pop * exc_rate_wet if not np.isnan(exc_rate_wet) else np.nan
        par_add = par_wet - par_dry if not np.isnan(par_dry) and not np.isnan(par_wet) else np.nan

        rows.append({
            'Zone':                        zone,
            'GW_Pop_Estimate':             round(pop),
            'Exceedance_Dry_pct':          round(exc_rate_dry * 100, 1),
            'Exceedance_Wet_pct':          round(exc_rate_wet * 100, 1),
            'Exceedance_All_pct':          round(exc_rate_all * 100, 1),
            'Pop_At_Risk_Dry':             round(par_dry) if not np.isnan(par_dry) else None,
            'Pop_At_Risk_Wet':             round(par_wet) if not np.isnan(par_wet) else None,
            'Additional_Exposed_Monsoon':  round(par_add) if not np.isnan(par_add) else None,
        })

    t = pd.DataFrame(rows).sort_values('Pop_At_Risk_Wet', ascending=False)
    t.to_csv(TABLES_DIR / 'H3_T33_population_at_risk_by_zone.csv', index=False)
    print(f"\n  H3_T33 Population at Risk:")
    print(f"  Total GW-dependent pop: {sum(zone_pops.values()):,.0f}")
    total_dry = t['Pop_At_Risk_Dry'].sum()
    total_wet = t['Pop_At_Risk_Wet'].sum()
    total_add = t['Additional_Exposed_Monsoon'].sum()
    print(f"  At risk (dry season):   {total_dry:,.0f} ({total_dry/sum(zone_pops.values())*100:.1f}%)")
    print(f"  At risk (wet season):   {total_wet:,.0f} ({total_wet/sum(zone_pops.values())*100:.1f}%)")
    print(f"  Additional monsoon exp: {total_add:,.0f}")
    return t


# ── Table 2: Seasonal exposure burden ─────────────────────────────────────────

def table_seasonal_burden(zone_pops, sra_df):
    """H3_T34: Seasonal exposure burden using SRA from H3."""
    if sra_df is None:
        print("  [SKIP] SRA data not found — skipping seasonal burden table")
        return None

    rows = []
    for _, row in sra_df.iterrows():
        zone  = row.get('Zone', row.get('Physiographic_Zone', ''))
        depth = row.get('Depth_Zone', row.get('Depth', ''))
        sra   = row.get('SRA', np.nan)
        exc_dry = row.get('Exceedance_Dry', row.get('WHO_Exc_Dry', np.nan))
        exc_wet = row.get('Exceedance_Wet', row.get('WHO_Exc_Wet', np.nan))

        pop = zone_pops.get(zone, 0)
        # Rough depth fraction: Shallow 22%, Intermediate 30%, M-D 27%, Deep 21%
        depth_frac = {'Shallow': 0.22, 'Intermediate': 0.30,
                      'Medium_Deep': 0.27, 'Deep': 0.21}.get(depth, 0.25)
        depth_pop = pop * depth_frac

        par_dry = depth_pop * exc_dry if not np.isnan(exc_dry) else np.nan
        par_wet = depth_pop * exc_wet if not np.isnan(exc_wet) else np.nan
        monsoon_extra = par_wet - par_dry if not np.isnan(par_dry) and not np.isnan(par_wet) else np.nan

        rows.append({
            'Zone':              zone,
            'Depth_Zone':        depth,
            'SRA':               round(sra, 3) if not np.isnan(sra) else np.nan,
            'Exc_Dry_pct':       round(exc_dry * 100, 1) if not np.isnan(exc_dry) else np.nan,
            'Exc_Wet_pct':       round(exc_wet * 100, 1) if not np.isnan(exc_wet) else np.nan,
            'Depth_Pop':         round(depth_pop),
            'PAR_Dry':           round(par_dry) if not np.isnan(par_dry) else np.nan,
            'PAR_Wet':           round(par_wet) if not np.isnan(par_wet) else np.nan,
            'Monsoon_Extra_PAR': round(monsoon_extra) if not np.isnan(monsoon_extra) else np.nan,
        })

    t = pd.DataFrame(rows)
    t.to_csv(TABLES_DIR / 'H3_T34_seasonal_exposure_burden.csv', index=False)
    total_extra = t['Monsoon_Extra_PAR'].dropna().sum()
    print(f"\n  H3_T34 Seasonal Burden: monsoon adds ~{total_extra:,.0f} to exposure")
    return t


# ── Table 3: Depth substitution benefit ───────────────────────────────────────

def table_depth_substitution(zone_pops, exc_df):
    """H3_T35: If Intermediate users switched to Deep wells — lives protected."""
    rows = []
    for zone, pop in zone_pops.items():
        ze = exc_df[(exc_df['Zone'] == zone) & (exc_df['Season'] == 'All')]

        exc_int  = ze[ze['Depth_Zone'] == 'Intermediate']['Exceedance_Rate']
        exc_deep = ze[ze['Depth_Zone'] == 'Deep']['Exceedance_Rate']

        if len(exc_int) == 0 or len(exc_deep) == 0:
            continue

        exc_int  = exc_int.values[0]
        exc_deep = exc_deep.values[0]
        delta    = exc_int - exc_deep   # reduction in exceedance probability

        # ~30% of GW population uses Intermediate-depth wells
        intermediate_pop = pop * 0.30
        protected = intermediate_pop * delta

        rows.append({
            'Zone':                  zone,
            'Exc_Intermediate_pct':  round(exc_int  * 100, 1),
            'Exc_Deep_pct':          round(exc_deep * 100, 1),
            'Delta_Exceedance_ppt':  round(delta    * 100, 1),
            'Intermediate_Pop':      round(intermediate_pop),
            'Pop_Protected_Switch':  round(protected),
            'Protection_Efficiency': round(protected / (pop * 0.30) * 100, 1),
        })

    t = pd.DataFrame(rows).sort_values('Pop_Protected_Switch', ascending=False)
    t.to_csv(TABLES_DIR / 'H3_T35_depth_substitution_benefit.csv', index=False)
    total_prot = t['Pop_Protected_Switch'].sum()
    print(f"\n  H3_T35 Depth Substitution: switching Intermediate→Deep "
          f"would protect ~{total_prot:,.0f} people")
    return t


# ── Table 4: Attributable Burden Index ────────────────────────────────────────

def table_attributable_burden(zone_pops, exc_df, sra_df):
    """H3_T36: Zone-level Attributable Burden Index = exceedance × SRA × pop_density."""
    rows = []
    # Mean SRA per zone
    sra_by_zone = {}
    if sra_df is not None:
        sra_col = [c for c in sra_df.columns if 'SRA' in c.upper()][0] \
                  if any('SRA' in c for c in sra_df.columns) else None
        zone_col = [c for c in sra_df.columns if 'Zone' in c][0] \
                   if any('Zone' in c for c in sra_df.columns) else None
        if sra_col and zone_col:
            for zone in ZONE_BBOX:
                sub = sra_df[sra_df[zone_col] == zone][sra_col].dropna()
                sra_by_zone[zone] = sub.mean() if len(sub) else 1.0

    for zone, pop in zone_pops.items():
        ze = exc_df[(exc_df['Zone'] == zone) & (exc_df['Season'] == 'All')]
        exc_mean = ze['Exceedance_Rate'].mean() if len(ze) else np.nan
        sra_mean = sra_by_zone.get(zone, 1.0)

        # Approximate area from bbox
        lat_min, lat_max, lon_min, lon_max = ZONE_BBOX[zone]
        area_km2 = (lat_max - lat_min) * 111 * (lon_max - lon_min) * 111 * \
                   np.cos(np.radians((lat_min + lat_max) / 2))
        pop_density = pop / area_km2

        # Attributable Burden Index = exceedance_rate × SRA × normalized_pop_density
        abi = exc_mean * sra_mean * (pop_density / 1000)  # per 1000/km²

        rows.append({
            'Zone':              zone,
            'GW_Pop':            round(pop),
            'Area_km2':          round(area_km2),
            'Pop_Density_km2':   round(pop_density, 1),
            'Exceedance_All_pct': round(exc_mean * 100, 1) if not np.isnan(exc_mean) else np.nan,
            'Mean_SRA':          round(sra_mean, 3),
            'ABI_Score':         round(abi, 4) if not np.isnan(exc_mean) else np.nan,
            'Priority_Tier':     'High' if (not np.isnan(exc_mean) and exc_mean > 0.40) else
                                 'Medium' if (not np.isnan(exc_mean) and exc_mean > 0.25) else 'Low',
        })

    t = pd.DataFrame(rows).sort_values('ABI_Score', ascending=False)
    t.to_csv(TABLES_DIR / 'H3_T36_attributable_burden_index.csv', index=False)
    print(f"\n  H3_T36 Attributable Burden Index:")
    print(t[['Zone', 'Exceedance_All_pct', 'Mean_SRA', 'ABI_Score', 'Priority_Tier']]
          .to_string(index=False))
    return t


# ── Figure: Population risk map ───────────────────────────────────────────────

def plot_risk_map(par_table, sra_table, burden_table):
    """H3_F22: 3-panel population risk figure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(1, 3, figsize=(16, 6))

        # Panel A: Population at risk bar chart
        ax = axes[0]
        if par_table is not None:
            zones   = [z.replace('_', '\n') for z in par_table['Zone']]
            par_dry = par_table['Pop_At_Risk_Dry'].values / 1e6
            par_wet = par_table['Pop_At_Risk_Wet'].values / 1e6
            x = np.arange(len(zones))
            ax.bar(x - 0.2, par_dry, 0.4, label='Dry season', color='#d4a853', alpha=0.85)
            ax.bar(x + 0.2, par_wet, 0.4, label='Wet season', color='#3a7abf', alpha=0.85)
            ax.set_xticks(x); ax.set_xticklabels(zones, fontsize=7)
            ax.set_ylabel('Population at risk (millions)')
            ax.set_title('(a) Population at Risk by Zone', fontweight='bold', fontsize=10)
            ax.legend(fontsize=8)

        # Panel B: Monsoon additional exposure
        ax = axes[1]
        if par_table is not None and 'Additional_Exposed_Monsoon' in par_table.columns:
            add = par_table['Additional_Exposed_Monsoon'].values / 1e6
            colors = ['#c0392b' if a > 0 else '#27ae60' for a in add]
            zones_short = [z.replace('_Floodplain', '\nFP').replace('_', '\n')
                           for z in par_table['Zone']]
            ax.bar(range(len(zones_short)), add, color=colors, alpha=0.85)
            ax.set_xticks(range(len(zones_short)))
            ax.set_xticklabels(zones_short, fontsize=7)
            ax.set_ylabel('Additional people exposed (millions)')
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_title('(b) Monsoon Amplification of Exposure', fontweight='bold', fontsize=10)

        # Panel C: Attributable Burden Index
        ax = axes[2]
        if burden_table is not None and 'ABI_Score' in burden_table.columns:
            bt = burden_table.sort_values('ABI_Score', ascending=True)
            colors_abi = ['#c0392b' if t == 'High' else
                          '#e67e22' if t == 'Medium' else '#27ae60'
                          for t in bt['Priority_Tier']]
            zones_bt = [z.replace('_', '\n') for z in bt['Zone']]
            ax.barh(range(len(zones_bt)), bt['ABI_Score'], color=colors_abi, alpha=0.85)
            ax.set_yticks(range(len(zones_bt))); ax.set_yticklabels(zones_bt, fontsize=7)
            ax.set_xlabel('Attributable Burden Index')
            ax.set_title('(c) Attributable Burden Index', fontweight='bold', fontsize=10)
            patches = [mpatches.Patch(color='#c0392b', label='High priority'),
                       mpatches.Patch(color='#e67e22', label='Medium priority'),
                       mpatches.Patch(color='#27ae60', label='Low priority')]
            ax.legend(handles=patches, fontsize=7)

        plt.tight_layout()
        out = FIGURES_DIR / 'H3_F22_population_risk_map.png'
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: {out.name}")
    except Exception as e:
        print(f"\n  [WARN] Plot failed: {e}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main(download=True):
    print("=" * 65)
    print("PUBLIC HEALTH BURDEN ANALYSIS — Bangladesh Arsenic")
    print("=" * 65)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load integrated geochemical dataset
    df = pd.read_csv(INTEGRATED_FILE)
    # Normalise As column name
    if 'As' not in df.columns and 'Arsenic_ugL' not in df.columns:
        as_cols = [c for c in df.columns if c.lower().startswith('as') or 'arsenic' in c.lower()]
        if as_cols:
            df = df.rename(columns={as_cols[0]: 'As'})
    if 'As' not in df.columns:
        df['As'] = df.iloc[:, df.columns.str.lower().str.contains('as').argmax()]
    # Normalise Season column
    if 'Season' not in df.columns:
        seas_cols = [c for c in df.columns if 'season' in c.lower()]
        if seas_cols:
            df = df.rename(columns={seas_cols[0]: 'Season'})
    print(f"\n  Dataset: {len(df)} samples | As col present: {'As' in df.columns}")
    print(f"  WHO exceedance (overall): {(df['As'] > WHO_LIMIT).mean()*100:.1f}%")

    # 2. Compute zone×depth×season exceedance rates from raw data
    print("\n  Computing exceedance rates by zone × depth × season...")
    exc_df = load_exceedance_data(df)
    print(f"  Exceedance table: {len(exc_df)} zone-depth-season combinations")

    # 3. Load WorldPop population raster
    if download:
        download_worldpop()
    lats, lons, pop_grid = load_worldpop()
    zone_pops = get_zone_populations(lats, lons, pop_grid)
    total_gw_pop = sum(zone_pops.values())
    print(f"\n  Total GW-dependent population: {total_gw_pop:,.0f}")

    # 4. Load SRA from H3
    sra_df = load_sra_data()
    if sra_df is not None:
        print(f"  SRA table loaded: {len(sra_df)} rows")

    # 5. Generate tables
    print("\n" + "-" * 55)
    par_table     = table_population_at_risk(df, zone_pops, exc_df)
    burden_table  = table_seasonal_burden(zone_pops, sra_df)
    subst_table   = table_depth_substitution(zone_pops, exc_df)
    abi_table     = table_attributable_burden(zone_pops, exc_df, sra_df)

    # 6. Figure
    plot_risk_map(par_table, burden_table, abi_table)

    # 7. Summary for manuscript
    print("\n" + "=" * 65)
    print("MANUSCRIPT NUMBERS SUMMARY")
    print("=" * 65)
    total_dry = par_table['Pop_At_Risk_Dry'].sum()
    total_wet = par_table['Pop_At_Risk_Wet'].sum()
    total_add = par_table['Additional_Exposed_Monsoon'].sum()
    print(f"  GW-dependent population:              {total_gw_pop:>12,.0f}")
    print(f"  At risk (dry season, As>10 µg/L):     {total_dry:>12,.0f}  "
          f"({total_dry/total_gw_pop*100:.1f}%)")
    print(f"  At risk (wet season):                 {total_wet:>12,.0f}  "
          f"({total_wet/total_gw_pop*100:.1f}%)")
    print(f"  Additional monsoon exposure:          {total_add:>12,.0f}")
    if subst_table is not None:
        total_prot = subst_table['Pop_Protected_Switch'].sum()
        print(f"  Protected by Intermediate→Deep shift: {total_prot:>12,.0f}")
    print("=" * 65)
    print("Outputs:")
    for f in ['H3_T33_population_at_risk_by_zone.csv',
              'H3_T34_seasonal_exposure_burden.csv',
              'H3_T35_depth_substitution_benefit.csv',
              'H3_T36_attributable_burden_index.csv',
              'H3_F22_population_risk_map.png']:
        path = (TABLES_DIR if f.endswith('.csv') else FIGURES_DIR) / f
        print(f"  {'✓' if path.exists() else '?'} {f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-download', action='store_true',
                        help='Skip WorldPop download (use cached file)')
    args = parser.parse_args()
    main(download=not args.no_download)
