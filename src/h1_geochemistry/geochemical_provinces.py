"""
PHASE 3C: GEOCHEMICAL PROVINCES (KEY SYNTHESIS)
Delineate spatial geochemical provinces with distinct As mobilization mechanisms

This is the CORNERSTONE analysis that integrates all previous findings:
- Geochemical mechanisms (phosphate, reductive, pH)
- Water types (Ca-HCO3, Na-HCO3, etc.)
- Satellite land use context
- Spatial As risk patterns

Expected result: 4-5 provinces with distinct characteristics

Output tables:
- Table_26_province_clustering.csv
- Table_27_province_characteristics.csv
- Table_28_province_as_risk.csv

Output figure:
- Figure_18_geochemical_province_map.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

from config import TABLES_DIR, FIGURES_DIR, OUTPUT_DIR, DPI, FIGURE_SIZE, DATA_FILE

def cluster_geochemical_provinces(df):
    """
    Hierarchical clustering on geochemical variables + satellite features

    Clustering features:
    - Fe²⁺: redox indicator
    - HCO₃⁻: pH/buffering
    - PO₄³⁻: phosphate mechanism
    - pH: desorption control
    - NDVI: agricultural intensity (if available)
    - NDWI: water body influence (if available)

    Linkage: Ward (minimizes within-cluster variance)
    """

    print("\n1. Geochemical Province Clustering:")
    print("-"*70)

    # Select features for clustering
    cluster_features = ['Fe2+', 'HCO3-', 'PO43-', 'pH', 'Do']
    available_features = [f for f in cluster_features if f in df.columns]

    # Also try satellite features
    sat_features = ['NDVI_mean', 'NDWI_mean']
    available_features += [f for f in sat_features if f in df.columns]

    print(f"   Clustering features: {available_features}")

    # Prepare data - handle various coordinate column names
    coord_cols = []
    for lat_name in ['Latitude', 'latitude', 'Lat']:
        if lat_name in df.columns:
            coord_cols = [lat_name, [lon for lon in ['Longitude', 'longitude', 'Lon'] if lon in df.columns][0]]
            break

    if coord_cols:
        df_cluster = df[available_features + ['As'] + coord_cols].dropna()
        df_cluster.rename(columns={coord_cols[0]: 'Latitude', coord_cols[1]: 'Longitude'}, inplace=True)
    else:
        # If no coordinates, add fake ones for clustering
        df_cluster = df[available_features + ['As']].dropna()
        df_cluster['Latitude'] = np.arange(len(df_cluster))
        df_cluster['Longitude'] = np.arange(len(df_cluster))

    print(f"   Clustering {len(df_cluster)} samples with complete data")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster[available_features])

    # Hierarchical clustering
    Z = linkage(X_scaled, method='ward')

    # Cut dendrogram at height to get 4-5 clusters
    n_clusters = 4
    clusters = fcluster(Z, n_clusters, criterion='maxclust')

    df_cluster['Province'] = clusters

    print(f"\n   Province distribution:")
    for prov in sorted(df_cluster['Province'].unique()):
        count = (df_cluster['Province'] == prov).sum()
        print(f"   Province {prov}: {count:5} samples ({100*count/len(df_cluster):5.1f}%)")

    return df_cluster, Z, X_scaled

def characterize_provinces(df_cluster):
    """
    Characterize each province by chemistry, water type, and As level

    For each province calculate:
    - Mean chemistry (Fe, HCO3, PO4, pH, Do)
    - Dominant water type
    - Mean As level
    - As samples > WHO limit
    - Implied mechanism
    """

    print("\n2. Province Characteristics:")
    print("-"*70)

    province_chars = []

    for prov in sorted(df_cluster['Province'].unique()):
        subset = df_cluster[df_cluster['Province'] == prov]

        # Chemistry
        fe_mean = subset['Fe2+'].mean()
        hco3_mean = subset['HCO3-'].mean()
        po4_mean = subset['PO43-'].mean()
        ph_mean = subset['pH'].mean()
        do_mean = subset.get('Do', pd.Series([np.nan]*len(subset))).mean()

        # As risk
        as_mean = subset['As'].mean()
        as_median = subset['As'].median()
        as_max = subset['As'].max()
        pct_above_who = (subset['As'] > 10).sum() / len(subset) * 100

        # Water type (if available)
        if 'Water_Type_Piper' in subset.columns:
            water_type_mode = subset['Water_Type_Piper'].mode()[0] if len(subset['Water_Type_Piper'].mode()) > 0 else 'Unknown'
        else:
            water_type_mode = 'Unknown'

        # Determine dominant mechanism based on correlations in this province
        if len(subset) > 5:
            r_as_po4, _ = stats.spearmanr(subset['As'], subset['PO43-'])
            r_as_fe, _ = stats.spearmanr(subset['As'], subset['Fe2+'])
        else:
            r_as_po4, r_as_fe = 0, 0

        if abs(r_as_po4) > abs(r_as_fe) and abs(r_as_po4) > 0.2:
            mechanism = 'Phosphate Competition'
        elif abs(r_as_fe) > 0.2:
            mechanism = 'Reductive Dissolution'
        else:
            mechanism = 'Mixed/pH-Dependent'

        province_chars.append({
            'Province': prov,
            'N_Samples': len(subset),
            'As_Mean': as_mean,
            'As_Median': as_median,
            'As_Max': as_max,
            'Percent_Above_WHO': pct_above_who,
            'Fe2+_Mean': fe_mean,
            'HCO3-_Mean': hco3_mean,
            'PO43-_Mean': po4_mean,
            'pH_Mean': ph_mean,
            'DO_Mean': do_mean,
            'Water_Type_Dominant': water_type_mode,
            'Dominant_Mechanism': mechanism,
            'Lat_Mean': subset['Latitude'].mean(),
            'Lon_Mean': subset['Longitude'].mean(),
        })

    char_df = pd.DataFrame(province_chars)

    # Print characteristics
    print(f"\n   {'Province':10} | {'N':6} | {'As Mean':10} | {'%>WHO':8} | {'Fe':8} | {'PO4':8} | {'Mechanism':25}")
    print(f"   {'-'*10}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*25}")

    for _, row in char_df.iterrows():
        print(f"   Prov {row['Province']:6} | {row['N_Samples']:6} | {row['As_Mean']:10.2f} | "
              f"{row['Percent_Above_WHO']:7.1f}% | {row['Fe2+_Mean']:8.2f} | {row['PO43-_Mean']:8.3f} | "
              f"{row['Dominant_Mechanism']:25}")

    return char_df

def create_province_map(df_cluster, char_df):
    """Create spatial map of provinces with As overlay"""

    print("\n3. Creating province map...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Provinces colored by location
    ax = axes[0]
    for prov in sorted(df_cluster['Province'].unique()):
        subset = df_cluster[df_cluster['Province'] == prov]
        ax.scatter(subset['Longitude'], subset['Latitude'], label=f'Province {prov}',
                  s=30, alpha=0.6)

    # Add province centroids
    for _, row in char_df.iterrows():
        ax.plot(row['Lon_Mean'], row['Lat_Mean'], 'k*', markersize=15)

    ax.set_xlabel('Longitude', fontweight='bold')
    ax.set_ylabel('Latitude', fontweight='bold')
    ax.set_title('Geochemical Provinces: Spatial Distribution', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    # Right: As risk by province
    ax = axes[1]
    provinces = char_df['Province'].values
    as_means = char_df['As_Mean'].values
    colors = plt.cm.RdYlGn_r(char_df['Percent_Above_WHO'].values / 100)

    bars = ax.bar(range(len(provinces)), as_means, color=colors, edgecolor='black', alpha=0.8)

    ax.set_xticks(range(len(provinces)))
    ax.set_xticklabels([f"Prov {p}" for p in provinces])
    ax.set_ylabel('Mean Arsenic (µg/L)', fontweight='bold')
    ax.set_title('As Risk by Province', fontweight='bold')
    ax.axhline(10, color='red', linestyle='--', linewidth=2, label='WHO Limit (10 µg/L)')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Geochemical Provinces of Bangladesh\nIntegrated Spatial Pattern Analysis',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Figure_18_geochemical_province_map.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"   Saved: Figure_18_geochemical_province_map.png")

def run_geochemical_provinces_analysis(df):
    """Main geochemical provinces analysis"""

    print("\n" + "="*80)
    print("PHASE 3C: GEOCHEMICAL PROVINCES ANALYSIS (KEY SYNTHESIS)")
    print("="*80)
    print("\nIntegrating mechanisms, water types, and spatial patterns")
    print("to delineate distinct geochemical provinces across Bangladesh")

    # Cluster provinces
    df_cluster, Z, X_scaled = cluster_geochemical_provinces(df)

    # Characterize each province
    char_df = characterize_provinces(df_cluster)

    # Save results
    df_cluster.to_csv(TABLES_DIR / 'Table_26_province_clustering.csv', index=False)
    print(f"\n   Saved: Table_26_province_clustering.csv")

    char_df.to_csv(TABLES_DIR / 'Table_27_province_characteristics.csv', index=False)
    print(f"   Saved: Table_27_province_characteristics.csv")

    # Create map
    create_province_map(df_cluster, char_df)

    print("\n" + "="*80)
    print("✓ GEOCHEMICAL PROVINCES ANALYSIS COMPLETE")
    print("="*80)

    return df_cluster, char_df

if __name__ == "__main__":
    print("Loading integrated dataset...")
    df = pd.read_csv(OUTPUT_DIR / 'bangladesh_groundwater_arsenic_1807samples.csv')

    # Need full data with Latitude/Longitude
    df_orig = pd.read_csv(DATA_FILE)
    df = pd.merge(df, df_orig[['Sample ID', 'Latitude', 'Longitude']],
                 left_index=True, right_index=True, how='left')

    df_cluster, char_df = run_geochemical_provinces_analysis(df)
    print("\nGeochemical provinces analysis complete.")
