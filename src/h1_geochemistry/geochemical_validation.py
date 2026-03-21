"""
PHASE 1: GEOCHEMICAL VALIDATION & ION RATIO ANALYSIS
Extends data_loader.py with geochemical QA/QC

Calculates:
1. Charge balance error (CBE) - analytical quality
2. Ion ratios for water type interpretation
3. Water chemistry categorization

Output tables:
- Table_9_charge_balance_error.csv
- Table_10_ion_ratios.csv
- Table_11_water_chemistry_categories.csv

Output figures:
- Figure_8_cbe_histogram.png
- Figure_9_ratios_vs_arsenic.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config import (
    DATA_FILE, IONS, TARGET, WATER_QUALITY_PARAMS,
    OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, DPI, FIGURE_SIZE, WHO_AS_LIMIT
)

def meq_conversion():
    """Molecular weights for meq/L conversion"""
    # Atomic/molecular weights (g/mol)
    mw = {
        'Na+': 23, 'K+': 39, 'Mg2+': 12, 'Ca2+': 20,
        'NO3-': 62, 'SO42-': 96, 'PO43-': 95, 'CO3-': 60,
        'HCO3-': 61, 'Cl-': 35.5, 'Br-': 80, 'I-': 127,
        'Al3+': 27, 'Cu2+': 64, 'Fe2+': 56, 'Mn2+': 55, 'Cr3+': 52, 'B+': 11
    }

    # Valences for meq/L calculation
    valence = {
        'Na+': 1, 'K+': 1, 'Mg2+': 2, 'Ca2+': 2,
        'NO3-': 1, 'SO42-': 2, 'PO43-': 3, 'CO3-': 2,
        'HCO3-': 1, 'Cl-': 1, 'Br-': 1, 'I-': 1,
        'Al3+': 3, 'Cu2+': 2, 'Fe2+': 2, 'Mn2+': 2, 'Cr3+': 3, 'B+': 1
    }
    return mw, valence

def calculate_charge_balance_error(df):
    """
    Calculate charge balance error (CBE)

    CBE (%) = |Sum(Cations) - Sum(Anions)| / |Sum(Cations) + Sum(Anions)| * 100

    Cations: Na+, K+, Mg2+, Ca2+, Al3+, Cu2+, Fe2+, Mn2+, Cr3+
    Anions: NO3-, SO42-, PO43-, CO3-, HCO3-, Cl-, Br-, I-

    Target: CBE < 5% (excellent), < 10% (acceptable), > 10% (quality issue)
    """
    mw, valence = meq_conversion()

    # Cations
    cations = ['Na+', 'K+', 'Mg2+', 'Ca2+', 'Al3+', 'Cu2+', 'Fe2+', 'Mn2+', 'Cr3+']
    anions = ['NO3-', 'SO42-', 'PO43-', 'CO3-', 'HCO3-', 'Cl-', 'Br-', 'I-']

    cation_meq = pd.DataFrame()
    for cat in cations:
        if cat in df.columns:
            # meq/L = (mg/L / MW) * valence
            cation_meq[cat] = (df[cat] / mw[cat]) * valence[cat]

    anion_meq = pd.DataFrame()
    for an in anions:
        if an in df.columns:
            anion_meq[an] = (df[an] / mw[an]) * valence[an]

    sum_cations = cation_meq.sum(axis=1)
    sum_anions = anion_meq.sum(axis=1)

    # Calculate CBE
    cbe = np.abs(sum_cations - sum_anions) / (sum_cations + sum_anions) * 100
    cbe = cbe.fillna(0)

    return pd.DataFrame({
        'Sample_ID': df.get('Sample_ID', range(len(df))),
        'Sum_Cations_meq': sum_cations,
        'Sum_Anions_meq': sum_anions,
        'CBE_percent': cbe,
        'CBE_Quality': cbe.apply(lambda x: 'Excellent' if x < 5 else ('Acceptable' if x < 10 else 'Poor'))
    })

def calculate_ion_ratios(df):
    """
    Calculate diagnostic ion ratios for water type interpretation

    Ratios:
    - HCO3/Cl: >1 bicarbonate dominant, <0.5 chloride dominant
    - Na/Cl: >1 excess Na (silicate weathering), <1 excess Cl (halite)
    - Ca/Mg: >2 limestone, 1-2 mixed
    - Fe2+/Mn2+: redox conditions
    - As/Fe: mobilization efficiency
    - SO4/Cl: agricultural/industrial influence
    """

    ratios_df = pd.DataFrame({
        'Sample_ID': df.get('Sample_ID', range(len(df))),
        'As': df.get('As', np.nan),
        'Fe2+': df.get('Fe2+', np.nan),
        'HCO3-': df.get('HCO3-', np.nan),
        'Cl-': df.get('Cl-', np.nan),
        'Na+': df.get('Na+', np.nan),
        'Ca2+': df.get('Ca2+', np.nan),
        'Mg2+': df.get('Mg2+', np.nan),
        'Mn2+': df.get('Mn2+', np.nan),
        'SO42-': df.get('SO42-', np.nan),
        'PO43-': df.get('PO43-', np.nan),
    })

    # Avoid division by zero
    ratios_df['HCO3_Cl_ratio'] = np.where(
        ratios_df['Cl-'] > 0.1,
        ratios_df['HCO3-'] / ratios_df['Cl-'],
        np.nan
    )

    ratios_df['Na_Cl_ratio'] = np.where(
        ratios_df['Cl-'] > 0.1,
        ratios_df['Na+'] / ratios_df['Cl-'],
        np.nan
    )

    ratios_df['Ca_Mg_ratio'] = np.where(
        ratios_df['Mg2+'] > 0.1,
        ratios_df['Ca2+'] / ratios_df['Mg2+'],
        np.nan
    )

    ratios_df['Fe_Mn_ratio'] = np.where(
        ratios_df['Mn2+'] > 0.01,
        ratios_df['Fe2+'] / ratios_df['Mn2+'],
        np.nan
    )

    ratios_df['As_Fe_ratio'] = np.where(
        ratios_df['Fe2+'] > 0.1,
        ratios_df['As'] / ratios_df['Fe2+'],
        np.nan
    )

    ratios_df['SO4_Cl_ratio'] = np.where(
        ratios_df['Cl-'] > 0.1,
        ratios_df['SO42-'] / ratios_df['Cl-'],
        np.nan
    )

    ratios_df['PO4_Cl_ratio'] = np.where(
        ratios_df['Cl-'] > 0.1,
        ratios_df['PO43-'] / ratios_df['Cl-'],
        np.nan
    )

    return ratios_df

def categorize_water_chemistry(df):
    """
    Pre-classify samples based on dominant ions for Piper diagram

    Categories:
    - Ca-HCO3: Carbonate weathering (Ca > Na+K, HCO3 > Cl)
    - Na-HCO3: Silicate weathering + ion exchange (Na > Ca, HCO3 > Cl)
    - Na-Cl: Seawater intrusion or halite (Na, Cl dominant)
    - Ca-SO4: Gypsum/sulfate (Ca, SO4 dominant)
    - Mixed: No dominant type
    """

    chemistry = pd.DataFrame({
        'Sample_ID': df.get('Sample_ID', range(len(df))),
        'Na+': df.get('Na+', np.nan),
        'K+': df.get('K+', np.nan),
        'Ca2+': df.get('Ca2+', np.nan),
        'Mg2+': df.get('Mg2+', np.nan),
        'Cl-': df.get('Cl-', np.nan),
        'HCO3-': df.get('HCO3-', np.nan),
        'SO42-': df.get('SO42-', np.nan),
        'As': df.get('As', np.nan),
    })

    # Cation dominance (Na+K vs Ca vs Mg)
    na_k = chemistry['Na+'] + chemistry['K+']
    ca = chemistry['Ca2+']

    # Anion dominance (HCO3 vs Cl vs SO4)
    hco3 = chemistry['HCO3-']
    cl = chemistry['Cl-']
    so4 = chemistry['SO42-']

    # Classify
    water_type = []
    for idx in chemistry.index:
        cat_maj = None
        if ca.iloc[idx] > na_k.iloc[idx]:
            cat_maj = 'Ca'
        else:
            cat_maj = 'Na'

        an_maj = None
        if hco3.iloc[idx] > cl.iloc[idx] and hco3.iloc[idx] > so4.iloc[idx]:
            an_maj = 'HCO3'
        elif cl.iloc[idx] > hco3.iloc[idx] and cl.iloc[idx] > so4.iloc[idx]:
            an_maj = 'Cl'
        else:
            an_maj = 'SO4'

        chemistry.loc[idx, 'Water_Type'] = f"{cat_maj}-{an_maj}"

    return chemistry[['Sample_ID', 'Water_Type', 'As']]

def run_geochemical_validation(df):
    """Main validation function"""

    print("\n" + "="*80)
    print("PHASE 1A: GEOCHEMICAL VALIDATION")
    print("="*80)

    # Calculate charge balance error
    print("\n1. Calculating charge balance error...")
    cbe_df = calculate_charge_balance_error(df)

    print(f"   CBE Statistics:")
    print(f"   Mean CBE: {cbe_df['CBE_percent'].mean():.2f}%")
    print(f"   Median CBE: {cbe_df['CBE_percent'].median():.2f}%")
    print(f"   Samples with CBE < 5%: {(cbe_df['CBE_percent'] < 5).sum()} ({100*(cbe_df['CBE_percent'] < 5).sum()/len(cbe_df):.1f}%)")
    print(f"   Samples with CBE < 10%: {(cbe_df['CBE_percent'] < 10).sum()} ({100*(cbe_df['CBE_percent'] < 10).sum()/len(cbe_df):.1f}%)")
    print(f"   Samples with CBE > 10%: {(cbe_df['CBE_percent'] > 10).sum()} ({100*(cbe_df['CBE_percent'] > 10).sum()/len(cbe_df):.1f}%)")

    # Save CBE table
    cbe_df.to_csv(TABLES_DIR / 'Table_9_charge_balance_error.csv', index=False)
    print(f"   Saved: {TABLES_DIR / 'Table_9_charge_balance_error.csv'}")

    # Calculate ion ratios
    print("\n2. Calculating ion ratios...")
    ratios_df = calculate_ion_ratios(df)

    print(f"   Key ratios calculated:")
    print(f"   HCO3/Cl ratio: mean = {ratios_df['HCO3_Cl_ratio'].mean():.3f}")
    print(f"   Na/Cl ratio: mean = {ratios_df['Na_Cl_ratio'].mean():.3f}")
    print(f"   Ca/Mg ratio: mean = {ratios_df['Ca_Mg_ratio'].mean():.3f}")
    print(f"   As/Fe ratio: mean = {ratios_df['As_Fe_ratio'].mean():.6f}")

    ratios_df.to_csv(TABLES_DIR / 'Table_10_ion_ratios.csv', index=False)
    print(f"   Saved: {TABLES_DIR / 'Table_10_ion_ratios.csv'}")

    # Categorize water chemistry
    print("\n3. Categorizing water chemistry...")
    chem_cat_df = categorize_water_chemistry(df)
    print(f"   Water types distribution:")
    print(chem_cat_df['Water_Type'].value_counts())

    chem_cat_df.to_csv(TABLES_DIR / 'Table_11_water_chemistry_categories.csv', index=False)
    print(f"   Saved: {TABLES_DIR / 'Table_11_water_chemistry_categories.csv'}")

    # Create figures
    print("\n4. Creating publication figures...")

    # Figure 8: CBE histogram
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.hist(cbe_df['CBE_percent'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(5, color='green', linestyle='--', linewidth=2, label='Excellent (<5%)')
    ax.axvline(10, color='orange', linestyle='--', linewidth=2, label='Acceptable (<10%)')
    ax.set_xlabel('Charge Balance Error (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Charge Balance Error\nGeochemical Data Quality Assessment',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Figure_8_cbe_histogram.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"   Saved: Figure_8_cbe_histogram.png")

    # Figure 9: Key ratios vs arsenic (4-panel)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # HCO3/Cl vs As
    ax = axes[0, 0]
    scatter = ax.scatter(ratios_df['HCO3_Cl_ratio'], ratios_df['As'],
                        c=ratios_df['As'], cmap='YlOrRd', alpha=0.6, s=30)
    ax.set_xlabel('HCO3⁻/Cl⁻ Ratio', fontweight='bold')
    ax.set_ylabel('Arsenic (µg/L)', fontweight='bold')
    ax.set_title('Bicarbonate-Chloride Ratio vs Arsenic')
    ax.axvline(1, color='gray', linestyle=':', alpha=0.5)

    # Na/Cl vs As
    ax = axes[0, 1]
    scatter = ax.scatter(ratios_df['Na_Cl_ratio'], ratios_df['As'],
                        c=ratios_df['As'], cmap='YlOrRd', alpha=0.6, s=30)
    ax.set_xlabel('Na⁺/Cl⁻ Ratio', fontweight='bold')
    ax.set_ylabel('Arsenic (µg/L)', fontweight='bold')
    ax.set_title('Sodium-Chloride Ratio vs Arsenic')
    ax.axvline(1, color='gray', linestyle=':', alpha=0.5)

    # Ca/Mg vs As
    ax = axes[1, 0]
    scatter = ax.scatter(ratios_df['Ca_Mg_ratio'], ratios_df['As'],
                        c=ratios_df['As'], cmap='YlOrRd', alpha=0.6, s=30)
    ax.set_xlabel('Ca²⁺/Mg²⁺ Ratio', fontweight='bold')
    ax.set_ylabel('Arsenic (µg/L)', fontweight='bold')
    ax.set_title('Calcium-Magnesium Ratio vs Arsenic')
    ax.axvline(2, color='gray', linestyle=':', alpha=0.5)

    # As/Fe vs As
    ax = axes[1, 1]
    scatter = ax.scatter(ratios_df['As_Fe_ratio'], ratios_df['As'],
                        c=ratios_df['As'], cmap='YlOrRd', alpha=0.6, s=30)
    ax.set_xlabel('As/Fe Ratio', fontweight='bold')
    ax.set_ylabel('Arsenic (µg/L)', fontweight='bold')
    ax.set_title('Arsenic-Iron Ratio vs Total Arsenic')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('As (µg/L)', fontweight='bold')

    plt.suptitle('Diagnostic Ion Ratios vs Arsenic Concentration',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Figure_9_ratios_vs_arsenic.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"   Saved: Figure_9_ratios_vs_arsenic.png")

    print("\n" + "="*80)
    print("✓ GEOCHEMICAL VALIDATION COMPLETE")
    print("="*80)

    return {
        'cbe_df': cbe_df,
        'ratios_df': ratios_df,
        'chem_cat_df': chem_cat_df
    }

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)

    results = run_geochemical_validation(df)
    print("\nGeochemical validation results:")
    print(f"  Samples processed: {len(df)}")
    print(f"  CBE statistics: Mean = {results['cbe_df']['CBE_percent'].mean():.2f}%")
    print(f"  Water types identified: {results['chem_cat_df']['Water_Type'].nunique()}")
