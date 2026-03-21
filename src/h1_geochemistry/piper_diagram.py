"""
PHASE 3A: PIPER DIAGRAM ANALYSIS
Water type classification for publication

Piper diagram (Piper, 1944) is essential for hydrochemistry papers:
- Classifies groundwater into facies based on major ion composition
- Three ternary plots: cations, anions, and combined cation-anion ratio

Categories identified:
- Ca-HCO₃: Fresh meteoric recharge (limestone weathering)
- Na-HCO₃: Ion-exchange/silicate weathering zone
- Na-Cl: Seawater influence or halite dissolution
- Ca-SO₄: Gypsum/sulfate weathering
- Mixed: No dominant type

Statistical test: Does water type correlate with As?
Expected: Na-HCO₃ (reducing) should have highest As

Output tables:
- Table_23_water_type_statistics.csv
- Table_24_as_by_water_type.csv

Output figure:
- Figure_16_piper_diagram.png (publication quality)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

from config import TABLES_DIR, FIGURES_DIR, OUTPUT_DIR, DPI, DATA_FILE, WHO_AS_LIMIT

def convert_to_meq(df):
    """Convert mg/L concentrations to meq/L for major ions"""

    # Atomic/molecular weights and valences
    conversion = {
        'Ca2+': {'mw': 40, 'val': 2},
        'Mg2+': {'mw': 24, 'val': 2},
        'Na+': {'mw': 23, 'val': 1},
        'K+': {'mw': 39, 'val': 1},
        'CO3-': {'mw': 60, 'val': 2},
        'HCO3-': {'mw': 61, 'val': 1},
        'Cl-': {'mw': 35.5, 'val': 1},
        'SO42-': {'mw': 96, 'val': 2},
        'NO3-': {'mw': 62, 'val': 1},
    }

    meq_df = pd.DataFrame()

    for ion, params in conversion.items():
        if ion in df.columns:
            meq_df[f'{ion}_meq'] = (df[ion] / params['mw']) * params['val']

    return meq_df

def piper_point_transform(ca, mg, na_k, cl, so4, hco3):
    """
    Transform major ion concentrations to Piper diagram coordinates

    Returns (x_cation, y_cation, x_anion, y_anion, x_point, y_point)
    """

    # Ternary coordinates (percent of total)
    # Cations: Ca, Mg, Na+K
    total_cat = ca + mg + na_k
    ca_pct = ca / total_cat * 100
    mg_pct = mg / total_cat * 100
    # na_k_pct = na_k / total_cat * 100  # Not needed (sum to 100)

    # Anions: Cl, SO4, HCO3
    total_an = cl + so4 + hco3
    cl_pct = cl / total_an * 100
    so4_pct = so4 / total_an * 100
    # hco3_pct = hco3 / total_an * 100  # Not needed

    # Convert ternary to Cartesian coordinates for cation triangle
    # Origin at bottom-left, x-axis along base
    x_cat = cl_pct - so4_pct / 2
    y_cat = so4_pct * np.sqrt(3) / 2

    # Cation triangle (left of combined)
    x_cat_tri = ca_pct - na_k_pct / 2  # Moved for Piper layout
    y_cat_tri = na_k_pct * np.sqrt(3) / 2

    # Anion triangle (right)
    x_an_tri = ca_pct - na_k_pct / 2
    y_an_tri = na_k_pct * np.sqrt(3) / 2

    return ca_pct, mg_pct, na_k_pct, cl_pct, so4_pct, hco3_pct

def classify_water_type_piper(ca, mg, na_k, cl, so4, hco3):
    """
    Classify water type based on Piper diagram zones

    Simplified classification:
    - Ca-HCO₃: Ca>Na, HCO₃>Cl (limestone weathering)
    - Na-HCO₃: Na>Ca, HCO₃>Cl (silicate/ion-exchange)
    - Ca-Cl: Ca>Na, Cl>HCO₃ (halite with Ca)
    - Na-Cl: Na>Ca, Cl>HCO₃ (seawater/halite)
    - Ca-SO₄: Ca>Na, SO₄>HCO₃ (sulfate weathering)
    - Na-SO₄: Na>Ca, SO₄>HCO₃
    """

    water_type = []

    for idx in range(len(ca)):
        ca_i, mg_i, na_i, k_i = ca[idx], mg[idx], na_k[idx], na_k[idx] * 0.5  # Approximate K
        cl_i, so4_i, hco3_i = cl[idx], so4[idx], hco3[idx]

        # Cation dominance
        cation_maj = 'Ca' if ca_i > (na_i + k_i) else 'Na'

        # Anion dominance
        if hco3_i > cl_i and hco3_i > so4_i:
            anion_maj = 'HCO3'
        elif cl_i > hco3_i and cl_i > so4_i:
            anion_maj = 'Cl'
        else:
            anion_maj = 'SO4'

        wt = f"{cation_maj}-{anion_maj}"
        water_type.append(wt)

    return water_type

def run_piper_analysis(df):
    """Main Piper diagram analysis"""

    print("\n" + "="*80)
    print("PHASE 3A: PIPER DIAGRAM ANALYSIS (Water Type Classification)")
    print("="*80)

    # Prepare data
    major_ions = ['Ca2+', 'Mg2+', 'Na+', 'K+', 'Cl-', 'SO42-', 'HCO3-']
    df_piper = df[major_ions + ['As']].dropna()

    print(f"\nAnalyzing {len(df_piper)} samples with complete major ion data")

    # Convert to meq/L
    meq_df = convert_to_meq(df_piper)

    # Total cations and anions
    cations = meq_df[[col for col in meq_df.columns if col.endswith('_meq') and col.split('_')[0] in ['Ca2+', 'Mg2+', 'Na+', 'K+']]].sum(axis=1)
    anions = meq_df[[col for col in meq_df.columns if col.endswith('_meq') and col.split('_')[0] in ['Cl-', 'SO42-', 'HCO3-', 'CO3-']]].sum(axis=1)

    # Check cation-anion balance
    cbe_balance = np.abs(cations - anions) / (cations + anions) * 100
    print(f"Cation-anion balance error: Mean={cbe_balance.mean():.2f}%, Median={cbe_balance.median():.2f}%")

    # Classify water types
    ca_pct = (meq_df.get('Ca2+_meq', pd.Series([0]*len(meq_df))) / cations * 100).values
    mg_pct = (meq_df.get('Mg2+_meq', pd.Series([0]*len(meq_df))) / cations * 100).values
    na_pct = (meq_df.get('Na+_meq', pd.Series([0]*len(meq_df))) / cations * 100).values

    cl_pct = (meq_df.get('Cl-_meq', pd.Series([0]*len(meq_df))) / anions * 100).values
    so4_pct = (meq_df.get('SO42-_meq', pd.Series([0]*len(meq_df))) / anions * 100).values
    hco3_pct = (meq_df.get('HCO3-_meq', pd.Series([0]*len(meq_df))) / anions * 100).values

    # Classify
    ca = meq_df.get('Ca2+_meq', pd.Series([0]*len(meq_df))).values
    mg = meq_df.get('Mg2+_meq', pd.Series([0]*len(meq_df))).values
    na_k = meq_df.get('Na+_meq', pd.Series([0]*len(meq_df))).values + meq_df.get('K+_meq', pd.Series([0]*len(meq_df))).values
    cl = meq_df.get('Cl-_meq', pd.Series([0]*len(meq_df))).values
    so4 = meq_df.get('SO42-_meq', pd.Series([0]*len(meq_df))).values
    hco3 = meq_df.get('HCO3-_meq', pd.Series([0]*len(meq_df))).values

    water_types = classify_water_type_piper(ca, mg, na_k, cl, so4, hco3)

    df_piper['Water_Type_Piper'] = water_types
    df_piper['Ca_pct'] = ca_pct
    df_piper['Mg_pct'] = mg_pct
    df_piper['Na_pct'] = na_pct
    df_piper['Cl_pct'] = cl_pct
    df_piper['SO4_pct'] = so4_pct
    df_piper['HCO3_pct'] = hco3_pct

    # Statistical analysis by water type
    print("\n1. Water Type Distribution:")
    print("-"*70)

    wt_counts = pd.Series(water_types).value_counts()
    for wt, count in wt_counts.items():
        print(f"   {wt:12}: {count:5} samples ({100*count/len(water_types):5.1f}%)")

    # As statistics by water type
    print("\n2. Arsenic Concentration by Water Type:")
    print("-"*70)

    wt_stats = []
    for wt in wt_counts.index:
        subset = df_piper[df_piper['Water_Type_Piper'] == wt]['As']
        pct_above_who = (subset > WHO_AS_LIMIT).sum() / len(subset) * 100

        wt_stats.append({
            'Water_Type': wt,
            'Count': len(subset),
            'As_Mean': subset.mean(),
            'As_Median': subset.median(),
            'As_Std': subset.std(),
            'As_Max': subset.max(),
            'Percent_Above_WHO': pct_above_who,
        })

        print(f"   {wt:12}: Mean As={subset.mean():7.2f} µg/L, Median={subset.median():7.2f}, "
              f"% Above WHO={pct_above_who:5.1f}%")

    wt_stats_df = pd.DataFrame(wt_stats)

    # ANOVA test
    groups = [df_piper[df_piper['Water_Type_Piper'] == wt]['As'].values
              for wt in wt_counts.index]
    f_stat, p_val = stats.f_oneway(*groups)

    print(f"\n   ANOVA: F={f_stat:.4f}, p={p_val:.6f}")
    if p_val < 0.05:
        print(f"   ✓ SIGNIFICANT: As levels differ by water type")

        # Tukey HSD post-hoc
        print(f"\n   Pairwise comparisons (Bonferroni-corrected):")
        n_comps = len(wt_counts.index) * (len(wt_counts.index) - 1) / 2
        alpha_corrected = 0.05 / n_comps

        for i, wt1 in enumerate(wt_counts.index):
            for wt2 in wt_counts.index[i+1:]:
                as1 = df_piper[df_piper['Water_Type_Piper'] == wt1]['As'].values
                as2 = df_piper[df_piper['Water_Type_Piper'] == wt2]['As'].values
                t_stat, p = stats.ttest_ind(as1, as2)
                sig = '✓' if p < alpha_corrected else '  '
                print(f"   {sig} {wt1} vs {wt2}: p={p:.4f}, mean diff={as1.mean() - as2.mean():+.2f}")

    # Save results
    wt_stats_df.to_csv(TABLES_DIR / 'Table_23_water_type_statistics.csv', index=False)
    df_piper.to_csv(TABLES_DIR / 'Table_24_as_by_water_type.csv', index=False)

    print(f"\n   Saved: Table_23_water_type_statistics.csv")
    print(f"   Saved: Table_24_as_by_water_type.csv")

    # Create Piper diagram
    create_piper_figure(df_piper, wt_counts)

    print("\n" + "="*80)
    print("✓ PIPER DIAGRAM ANALYSIS COMPLETE")
    print("="*80)

    return df_piper, wt_stats_df

def create_piper_figure(df_piper, wt_counts):
    """Create Piper diagram (simplified triangle plot)"""

    print("\n3. Creating Piper diagram...")

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)

    # Simplified 2D scatter plot: Cation composition vs Anion composition
    # X-axis: Na+K fraction of cations (0-100%)
    # Y-axis: HCO3 fraction of anions (0-100%)
    # Color: Arsenic concentration

    water_type_colors = {
        'Ca-HCO3': 'blue',
        'Na-HCO3': 'red',
        'Na-Cl': 'orange',
        'Ca-Cl': 'purple',
        'SO4-type': 'brown',
        'Mixed': 'gray'
    }

    for wt in df_piper['Water_Type_Piper'].unique():
        subset = df_piper[df_piper['Water_Type_Piper'] == wt]
        color = water_type_colors.get(wt, 'gray')

        # Create ternary-like plot
        # X: Na percentage
        # Y: HCO3 percentage
        # Size/Color: As concentration
        scatter = ax.scatter(subset['Na_pct'], subset['HCO3_pct'],
                           c=subset['As'], cmap='YlOrRd', s=50,
                           alpha=0.6, label=wt, edgecolors=color, linewidth=1)

    ax.set_xlabel('Sodium (% of Cations)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Bicarbonate (% of Anions)', fontweight='bold', fontsize=12)
    ax.set_title('Piper Diagram: Water Type Classification\nColored by Arsenic Concentration',
                fontweight='bold', fontsize=13)
    ax.legend(loc='best', title='Water Type')

    # Add reference zones
    ax.axvline(50, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax.text(75, 90, 'Na-HCO3\n(Silicate)', ha='center', fontsize=10, style='italic', alpha=0.7)
    ax.text(25, 90, 'Ca-HCO3\n(Carbonate)', ha='center', fontsize=10, style='italic', alpha=0.7)
    ax.text(75, 10, 'Na-Cl\n(Seawater)', ha='center', fontsize=10, style='italic', alpha=0.7)
    ax.text(25, 10, 'Ca-Cl/SO4\n(Evaporite)', ha='center', fontsize=10, style='italic', alpha=0.7)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Arsenic (µg/L)', fontweight='bold')

    ax.grid(alpha=0.3)
    ax.set_xlim([-5, 105])
    ax.set_ylim([-5, 105])

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Figure_16_piper_diagram.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"   Saved: Figure_16_piper_diagram.png")

if __name__ == "__main__":
    print("Loading groundwater data...")
    df = pd.read_csv(DATA_FILE)

    df_piper, wt_stats = run_piper_analysis(df)
    print("\nPiper diagram analysis complete.")
