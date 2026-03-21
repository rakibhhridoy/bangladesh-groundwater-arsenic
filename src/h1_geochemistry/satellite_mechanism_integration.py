"""
PHASE 2D: SATELLITE-MECHANISM INTEGRATION
Link satellite-derived land use to geochemical mechanisms

Hypothesis: Agricultural land use (high NDVI) correlates with:
- Higher phosphate (PO₄³⁻) from fertilizers
- Elevated As/Fe ratios (phosphate competition)
- Higher As contamination in agricultural regions

Also test water body influence (NDWI):
- Surface water (high NDWI) may contribute aquifer recharge
- Surface water intrusion can mobilize As

Output tables:
- Table_21_satellite_land_use_clustering.csv
- Table_22_mechanism_by_land_use.csv

Output figures:
- Figure_15_satellite_mechanism_integration.png (4-panel)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

from config import TABLES_DIR, FIGURES_DIR, OUTPUT_DIR, DPI, FIGURE_SIZE

def cluster_land_use_types(df):
    """
    Use satellite indices to identify land use clusters

    K-means clustering on:
    - NDVI: vegetation greenness (agriculture vs barren)
    - NDWI: water presence (wetland vs dry)
    - MNDWI: surface water (water vs land)

    Expected clusters:
    1. Agricultural (high NDVI)
    2. Urban/Developed (low NDVI, variable water)
    3. Wetland/Rice paddies (moderate NDVI, high water)
    4. Barren/Low vegetation (low NDVI, low water)
    """

    print("\n1. Land Use Clustering (K-means on Satellite Indices):")
    print("-"*70)

    # Select satellite features
    sat_cols = ['NDVI_mean', 'NDWI_mean', 'MNDWI_mean']
    available_cols = [col for col in sat_cols if col in df.columns]

    if len(available_cols) < 2:
        print(f"   ⚠ Insufficient satellite data for clustering")
        return None

    df_sat = df[available_cols + ['As']].dropna()

    if len(df_sat) < 50:
        print(f"   ⚠ Insufficient samples ({len(df_sat)}) for clustering")
        return None

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_sat[available_cols])

    # K-means with 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df_sat['Land_Use_Cluster'] = clusters

    # Characterize clusters
    print(f"\n   Cluster Characteristics:")
    print(f"   {'Cluster':10} | {'Count':8} | {'NDVI':8} | {'NDWI':8} | {'MNDWI':8} | {'As':10}")
    print(f"   {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")

    cluster_names = {}
    for clust in range(4):
        subset = df_sat[df_sat['Land_Use_Cluster'] == clust]
        n_samples = len(subset)

        ndvi_mean = subset['NDVI_mean'].mean() if 'NDVI_mean' in available_cols else np.nan
        ndwi_mean = subset['NDWI_mean'].mean() if 'NDWI_mean' in available_cols else np.nan
        mndwi_mean = subset['MNDWI_mean'].mean() if 'MNDWI_mean' in available_cols else np.nan
        as_mean = subset['As'].mean()

        print(f"   {clust:10} | {n_samples:8} | {ndvi_mean:8.3f} | {ndwi_mean:8.3f} | {mndwi_mean:8.3f} | {as_mean:10.2f}")

        # Auto-name clusters
        if ndvi_mean > 0.1:
            cluster_names[clust] = 'Agricultural'
        elif mndwi_mean > 0.2:
            cluster_names[clust] = 'Wetland/Water'
        elif ndvi_mean < -0.2:
            cluster_names[clust] = 'Barren'
        else:
            cluster_names[clust] = 'Mixed/Urban'

    df_sat['Land_Use_Type'] = df_sat['Land_Use_Cluster'].map(cluster_names)

    print(f"\n   Land use type distribution:")
    for lu_type, count in df_sat['Land_Use_Type'].value_counts().items():
        print(f"   {lu_type:20}: {count:5} samples ({100*count/len(df_sat):5.1f}%)")

    df_sat.to_csv(TABLES_DIR / 'Table_21_satellite_land_use_clustering.csv', index=False)
    print(f"\n   Saved: Table_21_satellite_land_use_clustering.csv")

    return df_sat, cluster_names

def mechanism_by_land_use(df, df_sat):
    """
    Analyze geochemical mechanisms stratified by land use

    Expected:
    - Agricultural areas: PO₄³⁻ high, As/Fe high (phosphate mechanism dominant)
    - Wetland/water: Fe²⁺ high, As high (reductive mechanism)
    - Urban/developed: Mixed signals
    - Barren: Low As, low mobilization
    """

    print("\n2. Geochemical Mechanisms by Land Use Type:")
    print("-"*70)

    # Merge with land use classification
    df_merged = df.copy()
    df_merged['Land_Use_Type'] = df_sat['Land_Use_Type'].values

    mechanism_summary = []

    for lu_type in df_merged['Land_Use_Type'].unique():
        if pd.isna(lu_type):
            continue

        subset = df_merged[df_merged['Land_Use_Type'] == lu_type]

        # Calculate key variables
        as_mean = subset['As'].mean()
        po4_mean = subset['PO43-'].mean() if 'PO43-' in subset.columns else np.nan
        fe_mean = subset['Fe2+'].mean() if 'Fe2+' in subset.columns else np.nan
        hco3_mean = subset['HCO3-'].mean() if 'HCO3-' in subset.columns else np.nan
        ph_mean = subset['pH'].mean() if 'pH' in subset.columns else np.nan

        # As/Fe ratio (phosphate competition indicator)
        if 'Fe2+' in subset.columns:
            as_fe_ratio = (subset['As'] / subset['Fe2+']).replace([np.inf, -np.inf], np.nan).mean()
        else:
            as_fe_ratio = np.nan

        # Correlations within land use type
        if len(subset) > 5:
            r_as_po4, _ = stats.spearmanr(subset['As'], subset['PO43-']) if 'PO43-' in subset.columns else (np.nan, np.nan)
            r_as_fe, _ = stats.spearmanr(subset['As'], subset['Fe2+']) if 'Fe2+' in subset.columns else (np.nan, np.nan)
        else:
            r_as_po4, r_as_fe = np.nan, np.nan

        mechanism_summary.append({
            'Land_Use_Type': lu_type,
            'N_Samples': len(subset),
            'As_Mean': as_mean,
            'PO4_Mean': po4_mean,
            'Fe_Mean': fe_mean,
            'HCO3_Mean': hco3_mean,
            'pH_Mean': ph_mean,
            'As_Fe_Ratio': as_fe_ratio,
            'r_As_PO4': r_as_po4,
            'r_As_Fe': r_as_fe,
        })

    mech_df = pd.DataFrame(mechanism_summary)

    print(f"\n   Mean Chemistry by Land Use Type:")
    print(mech_df[['Land_Use_Type', 'N_Samples', 'As_Mean', 'PO4_Mean', 'Fe_Mean', 'As_Fe_Ratio']].to_string(index=False))

    print(f"\n   Local Correlations by Land Use Type:")
    print(mech_df[['Land_Use_Type', 'N_Samples', 'r_As_PO4', 'r_As_Fe']].to_string(index=False))

    # ANOVA: do As levels differ by land use?
    groups = [subset['As'].values for subset in [df_merged[df_merged['Land_Use_Type'] == lu]['As'].values
                                                  for lu in df_merged['Land_Use_Type'].unique()]]
    if len(groups) > 1:
        f_stat, p_val = stats.f_oneway(*[g for g in groups if len(g) > 0])
        print(f"\n   ANOVA: As levels differ by land use? F={f_stat:.4f}, p={p_val:.6f}")
        if p_val < 0.05:
            print(f"   ✓ SIGNIFICANT spatial variation in As by land use type")

    mech_df.to_csv(TABLES_DIR / 'Table_22_mechanism_by_land_use.csv', index=False)
    print(f"\n   Saved: Table_22_mechanism_by_land_use.csv")

    return mech_df

def create_satellite_mechanism_figure(df_sat, mech_df):
    """Create 4-panel figure showing satellite-mechanism integration"""

    print("\n3. Creating satellite-mechanism integration figure...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel 1: NDVI vs PO₄³⁻ colored by As
    ax = axes[0, 0]
    if 'NDVI_mean' in df_sat.columns and 'PO43-' in df_sat.columns:
        scatter = ax.scatter(df_sat['NDVI_mean'], df_sat['PO43-'],
                           c=df_sat['As'], cmap='YlOrRd', s=30, alpha=0.6)
        r, p = stats.spearmanr(df_sat['NDVI_mean'], df_sat['PO43-'])
        ax.set_xlabel('NDVI (Agricultural Intensity)', fontweight='bold')
        ax.set_ylabel('Phosphate (mg/L)', fontweight='bold')
        ax.set_title(f'Agriculture & Phosphate\nρ = {r:.3f}, p = {p:.2e}', fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('As (µg/L)', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'NDVI/PO₄³⁻ data unavailable', ha='center', va='center', transform=ax.transAxes)
    ax.grid(alpha=0.3)

    # Panel 2: Land use clusters spatial
    ax = axes[0, 1]
    if 'NDVI_mean' in df_sat.columns and 'MNDWI_mean' in df_sat.columns:
        for lu_type in df_sat['Land_Use_Type'].unique():
            subset = df_sat[df_sat['Land_Use_Type'] == lu_type]
            ax.scatter(subset['NDVI_mean'], subset['MNDWI_mean'], label=lu_type, s=30, alpha=0.6)
        ax.set_xlabel('NDVI (Vegetation)', fontweight='bold')
        ax.set_ylabel('MNDWI (Surface Water)', fontweight='bold')
        ax.set_title('Land Use Type Classification\n(K-means on Satellite Indices)', fontweight='bold')
        ax.legend(loc='best')
    ax.grid(alpha=0.3)

    # Panel 3: As by land use
    ax = axes[1, 0]
    if 'Land_Use_Type' in df_sat.columns:
        land_use_types = []
        as_values = []
        for lu_type in mech_df['Land_Use_Type'].unique():
            subset = df_sat[df_sat['Land_Use_Type'] == lu_type]['As']
            land_use_types.append(lu_type)
            as_values.append(subset)

        bp = ax.boxplot(as_values, labels=land_use_types)
        ax.set_ylabel('Arsenic (µg/L)', fontweight='bold')
        ax.set_title('As Distribution by Land Use Type', fontweight='bold')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(alpha=0.3, axis='y')

    # Panel 4: As/Fe ratio by land use (mechanism proxy)
    ax = axes[1, 1]
    if 'As_Fe_Ratio' in mech_df.columns:
        mechanisms = mech_df['Land_Use_Type'].values
        ratios = mech_df['As_Fe_Ratio'].values
        colors = ['green' if 'Agricultural' in m else 'blue' if 'Wetland' in m else 'gray' for m in mechanisms]
        ax.bar(range(len(ratios)), ratios, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(ratios)))
        ax.set_xticklabels(mechanisms, rotation=45, ha='right')
        ax.set_ylabel('Mean As/Fe Ratio', fontweight='bold')
        ax.set_title('Mechanism Proxy: Phosphate Mobilization Efficiency\n(Higher As/Fe = More Mobilization per Fe)',
                    fontweight='bold')
        ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Satellite Evidence for Geochemical Mechanisms\nLinked Land Use Patterns to As Mobilization Pathways',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Figure_15_satellite_mechanism_integration.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"   Saved: Figure_15_satellite_mechanism_integration.png")

def run_satellite_mechanism_integration(df):
    """Main satellite-mechanism integration analysis"""

    print("\n" + "="*80)
    print("PHASE 2D: SATELLITE-MECHANISM INTEGRATION")
    print("="*80)
    print("\nHypothesis: Agricultural land use (high NDVI) drives phosphate-related")
    print("arsenic mobilization, while wetlands show Fe-reduction signatures.")

    # Cluster by land use
    result = cluster_land_use_types(df)

    if result is None:
        print("\n⚠ Could not perform land use clustering")
        return None

    df_sat, cluster_names = result

    # Analyze mechanisms by land use
    mech_df = mechanism_by_land_use(df, df_sat)

    # Create figure
    create_satellite_mechanism_figure(df_sat, mech_df)

    print("\n" + "="*80)
    print("✓ SATELLITE-MECHANISM INTEGRATION COMPLETE")
    print("="*80)

    return {
        'land_use': df_sat,
        'mechanisms_by_lu': mech_df,
        'cluster_names': cluster_names
    }

if __name__ == "__main__":
    print("Loading integrated dataset...")
    df = pd.read_csv(OUTPUT_DIR / 'data_phase1_integrated.csv')

    results = run_satellite_mechanism_integration(df)
    if results:
        print("\nSatellite-mechanism integration complete.")
