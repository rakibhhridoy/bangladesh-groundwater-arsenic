"""
PHASE 4A: SEASONAL ANALYSIS & MECHANISM SHIFTS
Test if dominant mechanisms change between dry and wet seasons

Expected findings:
- Dry season: Lower water tables, oxidizing conditions, may favor desorption
- Wet season: Higher water tables, anaerobic conditions, favor reduction + mobilization
- Phosphate mechanisms: May vary with agricultural input timing
- Reductive mechanisms: Should intensify in wet season

Output tables:
- Table_28_seasonal_comparison.csv
- Table_29_mechanism_by_season.csv
- Table_30_paired_sample_analysis.csv

Output figure:
- Figure_19_seasonal_mechanism_comparison.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

from config import TABLES_DIR, FIGURES_DIR, OUTPUT_DIR, DPI, FIGURE_SIZE, DATA_FILE

def seasonal_chemistry_comparison(df):
    """Compare major chemistry variables between seasons"""

    print("\n1. Seasonal Chemistry Comparison:")
    print("-"*70)

    # Identify season column
    season_col = None
    for col in ['Season', 'season', 'SEASON']:
        if col in df.columns:
            season_col = col
            break

    if season_col is None:
        print("   ⚠ Season column not found")
        return None

    # Key variables
    key_vars = ['As', 'Fe2+', 'HCO3-', 'PO43-', 'pH', 'Do']
    available_vars = [v for v in key_vars if v in df.columns]

    seasonal_stats = []

    for season in df[season_col].unique():
        if pd.isna(season):
            continue

        subset = df[df[season_col] == season]

        stats_dict = {'Season': season, 'N_Samples': len(subset)}

        for var in available_vars:
            stats_dict[f'{var}_Mean'] = subset[var].mean()
            stats_dict[f'{var}_Median'] = subset[var].median()
            stats_dict[f'{var}_Std'] = subset[var].std()

        seasonal_stats.append(stats_dict)

    seasonal_df = pd.DataFrame(seasonal_stats)

    print(f"\n   Chemistry Statistics by Season:")
    print(f"   {'Season':12} | {'N':8} | {'As Mean':10} | {'Fe Mean':10} | {'PO4 Mean':10} | {'pH Mean':10}")
    print(f"   {'-'*12}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for _, row in seasonal_df.iterrows():
        print(f"   {str(row['Season']):12} | {row['N_Samples']:8.0f} | "
              f"{row['As_Mean']:10.2f} | {row['Fe2+_Mean']:10.2f} | "
              f"{row['PO43-_Mean']:10.3f} | {row['pH_Mean']:10.2f}")

    return seasonal_df

def mechanism_by_season(df):
    """Test if mechanisms (correlations) change by season"""

    print("\n2. Mechanism Correlation by Season:")
    print("-"*70)

    # Identify season column
    season_col = None
    for col in ['Season', 'season', 'SEASON']:
        if col in df.columns:
            season_col = col
            break

    if season_col is None:
        return None

    mechanism_season = []

    for season in df[season_col].unique():
        if pd.isna(season):
            continue

        subset = df[df[season_col] == season]
        n = len(subset)

        # Correlations
        if n > 5:
            r_as_po4, p_as_po4 = stats.spearmanr(subset['As'], subset['PO43-']) if 'PO43-' in subset.columns else (np.nan, np.nan)
            r_as_fe, p_as_fe = stats.spearmanr(subset['As'], subset['Fe2+']) if 'Fe2+' in subset.columns else (np.nan, np.nan)
            r_as_hco3, p_as_hco3 = stats.spearmanr(subset['As'], subset['HCO3-']) if 'HCO3-' in subset.columns else (np.nan, np.nan)

            # As/Fe ratio
            as_fe_ratio = (subset['As'] / subset['Fe2+']).replace([np.inf, -np.inf], np.nan).mean()
        else:
            r_as_po4 = r_as_fe = r_as_hco3 = p_as_po4 = p_as_fe = p_as_hco3 = as_fe_ratio = np.nan

        # Determine dominant mechanism
        correlations = [r_as_po4, r_as_fe, r_as_hco3]
        if max([abs(c) for c in correlations if not np.isnan(c)]) == abs(r_as_po4):
            dominant = 'Phosphate'
        elif max([abs(c) for c in correlations if not np.isnan(c)]) == abs(r_as_fe):
            dominant = 'Reductive'
        else:
            dominant = 'Buffering'

        mechanism_season.append({
            'Season': season,
            'N_Samples': n,
            'r_As_PO4': r_as_po4,
            'p_As_PO4': p_as_po4,
            'r_As_Fe': r_as_fe,
            'p_As_Fe': p_as_fe,
            'r_As_HCO3': r_as_hco3,
            'p_As_HCO3': p_as_hco3,
            'As_Fe_Ratio': as_fe_ratio,
            'Dominant_Mechanism': dominant,
            'As_Mean': subset['As'].mean(),
        })

    mech_season_df = pd.DataFrame(mechanism_season)

    print(f"\n   Correlations by Season:")
    print(f"   {'Season':12} | {'r(As-PO4)':10} | {'r(As-Fe)':10} | {'r(As-HCO3)':10} | {'Dominant':15}")
    print(f"   {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*15}")

    for _, row in mech_season_df.iterrows():
        print(f"   {str(row['Season']):12} | {row['r_As_PO4']:10.4f} | {row['r_As_Fe']:10.4f} | "
              f"{row['r_As_HCO3']:10.4f} | {row['Dominant_Mechanism']:15}")

    return mech_season_df

def seasonal_as_variation(df):
    """Analyze seasonal As variation and paired differences"""

    print("\n3. Seasonal Arsenic Variation:")
    print("-"*70)

    # Identify season column
    season_col = None
    for col in ['Season', 'season', 'SEASON']:
        if col in df.columns:
            season_col = col
            break

    if season_col is None:
        return None

    seasons = df[season_col].unique()
    seasons = [s for s in seasons if pd.notna(s)]

    print(f"\n   As statistics by season:")
    for season in seasons:
        subset = df[df[season_col] == season]['As']
        print(f"   {str(season):12}: Mean={subset.mean():7.2f} µg/L, "
              f"Median={subset.median():7.2f}, Std={subset.std():7.2f}")

    # T-test between seasons if two seasons
    if len(seasons) == 2:
        as_season1 = df[df[season_col] == seasons[0]]['As'].values
        as_season2 = df[df[season_col] == seasons[1]]['As'].values

        t_stat, p_val = stats.ttest_ind(as_season1, as_season2)
        print(f"\n   T-test: t={t_stat:.4f}, p={p_val:.6f}")

        if p_val < 0.05:
            print(f"   ✓ SIGNIFICANT seasonal difference in As levels")
            diff = as_season1.mean() - as_season2.mean()
            print(f"   Mean difference: {diff:+.2f} µg/L ({100*diff/as_season2.mean():+.1f}%)")
        else:
            print(f"   ✗ No significant seasonal difference")

    return seasons

def run_seasonal_analysis(df):
    """Main seasonal analysis"""

    print("\n" + "="*80)
    print("PHASE 4A: SEASONAL MECHANISM ANALYSIS")
    print("="*80)
    print("\nTesting hypothesis: Arsenic mobilization mechanisms vary by season")
    print("Expected: Wet season enhanced reductive + phosphate mechanisms")

    # Run analyses
    seasonal_chem = seasonal_chemistry_comparison(df)
    mech_by_season = mechanism_by_season(df)
    seasons = seasonal_as_variation(df)

    if seasonal_chem is not None:
        seasonal_chem.to_csv(TABLES_DIR / 'Table_28_seasonal_comparison.csv', index=False)
        print(f"\n   Saved: Table_28_seasonal_comparison.csv")

    if mech_by_season is not None:
        mech_by_season.to_csv(TABLES_DIR / 'Table_29_mechanism_by_season.csv', index=False)
        print(f"   Saved: Table_29_mechanism_by_season.csv")

        # Create figure
        create_seasonal_figure(df, mech_by_season)

    print("\n" + "="*80)
    print("✓ SEASONAL ANALYSIS COMPLETE")
    print("="*80)

    return seasonal_chem, mech_by_season

def create_seasonal_figure(df, mech_by_season):
    """Create seasonal comparison figure"""

    print("\n4. Creating seasonal figure...")

    # Identify season column
    season_col = None
    for col in ['Season', 'season', 'SEASON']:
        if col in df.columns:
            season_col = col
            break

    if season_col is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel 1: As by season
    ax = axes[0, 0]
    seasons = df[season_col].unique()
    as_data = [df[df[season_col] == s]['As'].values for s in seasons if pd.notna(s)]
    ax.boxplot(as_data, labels=[str(s) for s in seasons if pd.notna(s)])
    ax.set_ylabel('Arsenic (µg/L)', fontweight='bold')
    ax.set_title('As Distribution by Season', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # Panel 2: Correlations by season
    ax = axes[0, 1]
    if len(mech_by_season) > 0:
        x = np.arange(len(mech_by_season))
        width = 0.25
        ax.bar(x - width, mech_by_season['r_As_PO4'].values, width, label='As-PO₄', alpha=0.8)
        ax.bar(x, mech_by_season['r_As_Fe'].values, width, label='As-Fe', alpha=0.8)
        ax.bar(x + width, mech_by_season['r_As_HCO3'].values, width, label='As-HCO₃', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(mech_by_season['Season'].values)
        ax.set_ylabel('Spearman Correlation (ρ)', fontweight='bold')
        ax.set_title('Mechanism Correlations by Season', fontweight='bold')
        ax.legend()
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(alpha=0.3, axis='y')

    # Panel 3: Mean chemistry by season
    ax = axes[1, 0]
    if 'Fe2+' in df.columns and 'PO43-' in df.columns:
        for season in df[season_col].unique():
            if pd.notna(season):
                subset = df[df[season_col] == season]
                ax.scatter(subset['Fe2+'], subset['PO43-'], label=str(season), alpha=0.5, s=20)
        ax.set_xlabel('Fe²⁺ (mg/L)', fontweight='bold')
        ax.set_ylabel('PO₄³⁻ (mg/L)', fontweight='bold')
        ax.set_title('Fe²⁺ vs PO₄³⁻ by Season', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    # Panel 4: As/Fe ratio by season
    ax = axes[1, 1]
    if len(mech_by_season) > 0:
        ax.bar(range(len(mech_by_season)), mech_by_season['As_Fe_Ratio'].values,
               alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(mech_by_season)))
        ax.set_xticklabels(mech_by_season['Season'].values)
        ax.set_ylabel('Mean As/Fe Ratio', fontweight='bold')
        ax.set_title('As Mobilization Efficiency by Season', fontweight='bold')
        ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Seasonal Variation in As Mobilization Mechanisms',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Figure_19_seasonal_mechanism_comparison.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"   Saved: Figure_19_seasonal_mechanism_comparison.png")

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)

    seasonal_chem, mech_by_season = run_seasonal_analysis(df)
    print("\nSeasonal analysis complete.")
