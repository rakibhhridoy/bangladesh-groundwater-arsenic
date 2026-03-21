"""
PHASE 2B: REDUCTIVE DISSOLUTION MECHANISM (EXTENDED)
Extends existing reductive dissolution analysis with redox stratification

Mechanism: In anaerobic conditions (low dissolved oxygen):
1. Organic matter oxidation consumes O₂
2. Fe(III) oxides reduced to Fe(II) in solution
3. Adsorbed As mobilized by: a) Fe surface reduction, b) pH increase from HCO₃⁻

Key Variables:
- DO (dissolved oxygen): <2 anaerobic, 2-4 suboxic, >4 oxic
- Fe²⁺: Product of Fe(III) reduction
- HCO₃⁻: Associated with anaerobic respiration
- Eh-pH: Thermodynamic indicator of redox state

Hypothesis: Fe²⁺-As correlation is STRONGER in anaerobic samples than oxic

Output tables:
- Table_16_redox_stratified_correlations.csv
- Table_17_redox_classification.csv
- Table_18_fe_hco3_product_analysis.csv

Output figures:
- Figure_13_reductive_mechanism_evidence.png (4-panel)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

from config import TABLES_DIR, FIGURES_DIR, OUTPUT_DIR, DPI, FIGURE_SIZE

def classify_redox_state(df):
    """
    Classify samples into redox categories based on dissolved oxygen

    Categories:
    - Anaerobic: DO < 2 mg/L (strongly reducing)
    - Suboxic: 2 ≤ DO < 4 mg/L (moderately reducing)
    - Oxic: DO ≥ 4 mg/L (oxidizing)

    Also estimate Eh (redox potential) if ORP available
    """

    print("\n1. Redox State Classification:")
    print("-"*70)

    df_clean = df[['As', 'Fe2+', 'HCO3-', 'DO', 'ORP']].dropna() if 'DO' in df.columns else df[['As', 'Fe2+', 'HCO3-']].dropna()

    if 'DO' in df_clean.columns:
        # Classify by dissolved oxygen
        redox_class = []
        for do in df_clean['DO']:
            if do < 2:
                redox_class.append('Anaerobic')
            elif do < 4:
                redox_class.append('Suboxic')
            else:
                redox_class.append('Oxic')

        df_clean['Redox_Class'] = redox_class

        redox_counts = pd.Series(redox_class).value_counts()
        print(f"\n   Redox class distribution:")
        for rc, count in redox_counts.items():
            print(f"   {rc:12}: {count:5} samples ({100*count/len(df_clean):5.1f}%)")

        # Calculate mean parameters by redox class
        print(f"\n   Mean chemistry by redox class:")
        print(f"   {'Class':12} | {'As (µg/L)':12} | {'Fe²⁺ (mg/L)':12} | {'HCO₃⁻':12}")
        print(f"   {'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

        for rc in ['Anaerobic', 'Suboxic', 'Oxic']:
            subset = df_clean[df_clean['Redox_Class'] == rc]
            if len(subset) > 0:
                as_mean = subset['As'].mean()
                fe_mean = subset['Fe2+'].mean()
                hco3_mean = subset['HCO3-'].mean()
                print(f"   {rc:12} | {as_mean:12.2f} | {fe_mean:12.2f} | {hco3_mean:12.2f}")

    else:
        print("   ⚠ DO not available, cannot classify redox state")
        df_clean['Redox_Class'] = 'Unknown'

    return df_clean

def redox_stratified_correlations(df_redox):
    """
    Compare As-Fe correlation across redox classes

    Hypothesis: Fe-As correlation is strongest under anaerobic conditions
    """

    print("\n2. Redox-Stratified Correlations:")
    print("-"*70)

    results = []

    if 'Redox_Class' not in df_redox.columns:
        return None

    for redox_class in ['Anaerobic', 'Suboxic', 'Oxic', 'Unknown']:
        subset = df_redox[df_redox['Redox_Class'] == redox_class]

        if len(subset) < 3:
            continue

        # Correlation: As vs Fe²⁺
        r_as_fe, p_as_fe = stats.spearmanr(subset['As'], subset['Fe2+'])

        # Correlation: As vs HCO₃⁻
        r_as_hco3, p_as_hco3 = stats.spearmanr(subset['As'], subset['HCO3-'])

        # Correlation: Fe²⁺ vs HCO₃⁻ (test redox couple)
        r_fe_hco3, p_fe_hco3 = stats.spearmanr(subset['Fe2+'], subset['HCO3-'])

        # As/Fe ratio
        as_fe_ratio = np.where(subset['Fe2+'] > 0.1, subset['As'] / subset['Fe2+'], np.nan)
        as_fe_mean = np.nanmean(as_fe_ratio)

        results.append({
            'Redox_Class': redox_class,
            'N_Samples': len(subset),
            'r_As_Fe': r_as_fe,
            'p_As_Fe': p_as_fe,
            'r_As_HCO3': r_as_hco3,
            'p_As_HCO3': p_as_hco3,
            'r_Fe_HCO3': r_fe_hco3,
            'p_Fe_HCO3': p_fe_hco3,
            'As_Fe_Ratio_Mean': as_fe_mean,
            'As_Mean': subset['As'].mean(),
            'Fe_Mean': subset['Fe2+'].mean(),
        })

        print(f"\n   {redox_class} (n={len(subset)}):")
        print(f"   r(As-Fe²⁺) = {r_as_fe:.4f}, p = {p_as_fe:.6f}")
        print(f"   r(As-HCO₃⁻) = {r_as_hco3:.4f}, p = {p_as_hco3:.6f}")
        print(f"   r(Fe²⁺-HCO₃⁻) = {r_fe_hco3:.4f}, p = {p_fe_hco3:.6f}")
        print(f"   Mean As/Fe = {as_fe_mean:.6f}")

    results_df = pd.DataFrame(results)

    # Test hypothesis: does As-Fe correlation increase with reduction?
    # Extract anaerobic vs oxic correlation difference
    anaerobic_as_fe = results_df[results_df['Redox_Class'] == 'Anaerobic']['r_As_Fe'].values
    oxic_as_fe = results_df[results_df['Redox_Class'] == 'Oxic']['r_As_Fe'].values

    if len(anaerobic_as_fe) > 0 and len(oxic_as_fe) > 0:
        diff = anaerobic_as_fe[0] - oxic_as_fe[0]
        print(f"\n   Difference (Anaerobic - Oxic): {diff:+.4f}")
        if diff > 0:
            print(f"   ✓ As-Fe correlation STRONGER in anaerobic conditions (supports hypothesis)")
        else:
            print(f"   ✗ As-Fe correlation NOT stronger in anaerobic (hypothesis not supported)")

    return results_df

def fe_hco3_product_analysis(df):
    """
    Analyze Fe²⁺ × HCO₃⁻ product as indicator of anaerobic respiration

    Rationale: Fe²⁺ and HCO₃⁻ both increase during anaerobic decomposition
    Their product reflects intensity of redox transformation

    Expected: High Fe×HCO₃ should correlate with high As (if mechanism is redox)
    """

    print("\n3. Fe²⁺ × HCO₃⁻ Product Analysis:")
    print("-"*70)

    df_clean = df[['As', 'Fe2+', 'HCO3-']].dropna()

    # Calculate product (note: need to avoid units issue - use raw values)
    df_clean['Fe_HCO3_Product'] = df_clean['Fe2+'] * df_clean['HCO3-']

    # Correlation with As
    r_as_product, p_as_product = stats.spearmanr(df_clean['As'], df_clean['Fe_HCO3_Product'])

    print(f"\n   Spearman r(As - Fe²⁺×HCO₃⁻): {r_as_product:.4f}, p = {p_as_product:.6f}")

    if p_as_product < 0.05:
        print(f"   ✓ SIGNIFICANT correlation")
    else:
        print(f"   ✗ Not significant")

    # Categorize by product quartiles
    df_clean['FH_Quartile'] = pd.qcut(df_clean['Fe_HCO3_Product'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    print(f"\n   As concentration by Fe²⁺×HCO₃⁻ quartile:")
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        subset = df_clean[df_clean['FH_Quartile'] == q]['As']
        print(f"   {q}: Mean = {subset.mean():.2f} µg/L, Median = {subset.median():.2f} µg/L")

    return df_clean

def run_reductive_mechanism_analysis(df):
    """Main reductive mechanism analysis"""

    print("\n" + "="*80)
    print("PHASE 2B: REDUCTIVE DISSOLUTION MECHANISM ANALYSIS (EXTENDED)")
    print("="*80)
    print("\nHypothesis: Anaerobic respiration drives Fe reduction and As mobilization")
    print("Prediction: Fe²⁺-As correlation should be strongest in anaerobic samples")

    # Classify by redox state
    df_redox = classify_redox_state(df)

    # Stratified correlations
    if 'DO' in df.columns or 'Redox_Class' in df_redox.columns:
        stratified_corr = redox_stratified_correlations(df_redox)
        if stratified_corr is not None:
            stratified_corr.to_csv(TABLES_DIR / 'Table_16_redox_stratified_correlations.csv', index=False)
            print(f"\n   Saved: Table_16_redox_stratified_correlations.csv")
    else:
        stratified_corr = None
        print(f"\n   ⚠ Cannot calculate stratified correlations without DO data")

    # Fe×HCO₃ product analysis
    df_product = fe_hco3_product_analysis(df)
    df_product[['As', 'Fe2+', 'HCO3-', 'Fe_HCO3_Product']].to_csv(
        TABLES_DIR / 'Table_18_fe_hco3_product_analysis.csv', index=False
    )
    print(f"   Saved: Table_18_fe_hco3_product_analysis.csv")

    # Create Figure 13
    if stratified_corr is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))

        # Panel 1: As concentration by redox class
        ax = axes[0, 0]
        redox_as = []
        labels = []
        for rc in ['Anaerobic', 'Suboxic', 'Oxic']:
            subset = df_redox[df_redox['Redox_Class'] == rc]['As']
            if len(subset) > 0:
                redox_as.append(subset)
                labels.append(f"{rc}\n(n={len(subset)})")

        ax.boxplot(redox_as, labels=labels)
        ax.set_ylabel('Arsenic (µg/L)', fontweight='bold')
        ax.set_title('As Distribution by Redox Class', fontweight='bold')
        ax.grid(alpha=0.3, axis='y')

        # Panel 2: As-Fe correlation by redox class
        ax = axes[0, 1]
        redox_classes = stratified_corr['Redox_Class'].values
        correlations = stratified_corr['r_As_Fe'].values
        colors = ['red' if rc == 'Anaerobic' else 'orange' if rc == 'Suboxic' else 'blue' if rc == 'Oxic' else 'gray' for rc in redox_classes]
        ax.bar(range(len(correlations)), correlations, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(correlations)))
        ax.set_xticklabels(redox_classes, rotation=45)
        ax.set_ylabel('Spearman r(As-Fe²⁺)', fontweight='bold')
        ax.set_title('As-Fe²⁺ Correlation by Redox Conditions', fontweight='bold')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(alpha=0.3, axis='y')

        # Panel 3: Fe×HCO₃ product vs As
        ax = axes[1, 0]
        ax.scatter(df_product['Fe_HCO3_Product'], df_product['As'], alpha=0.5, s=20, color='darkred')
        r, p = stats.spearmanr(df_product['Fe_HCO3_Product'], df_product['As'])
        ax.set_xlabel('Fe²⁺ × HCO₃⁻ Product', fontweight='bold')
        ax.set_ylabel('Arsenic (µg/L)', fontweight='bold')
        ax.set_title(f'Redox Product vs As\nρ = {r:.3f}, p = {p:.2e}', fontweight='bold')
        ax.grid(alpha=0.3)

        # Panel 4: Fe²⁺ vs As by redox class
        ax = axes[1, 1]
        for rc, color in [('Anaerobic', 'red'), ('Suboxic', 'orange'), ('Oxic', 'blue')]:
            subset = df_redox[df_redox['Redox_Class'] == rc]
            if len(subset) > 0:
                ax.scatter(subset['Fe2+'], subset['As'], label=rc, alpha=0.6, s=20, color=color)
        ax.set_xlabel('Fe²⁺ (mg/L)', fontweight='bold')
        ax.set_ylabel('Arsenic (µg/L)', fontweight='bold')
        ax.set_title('As vs Fe²⁺ Colored by Redox Conditions', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.suptitle('Reductive Dissolution Mechanism Evidence\nIs As mobilization driven by anaerobic Fe reduction?',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'Figure_13_reductive_mechanism_evidence.png', dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"   Saved: Figure_13_reductive_mechanism_evidence.png")

    # Save redox classification
    df_redox[['As', 'Fe2+', 'HCO3-', 'Redox_Class']].to_csv(
        TABLES_DIR / 'Table_17_redox_classification.csv', index=False
    )
    print(f"   Saved: Table_17_redox_classification.csv")

    print("\n" + "="*80)
    print("✓ REDUCTIVE MECHANISM ANALYSIS COMPLETE")
    print("="*80)

    return {
        'redox_classification': df_redox,
        'stratified_correlations': stratified_corr,
        'fe_hco3_product': df_product
    }

if __name__ == "__main__":
    print("Loading integrated dataset...")
    df = pd.read_csv(OUTPUT_DIR / 'bangladesh_groundwater_arsenic_1807samples.csv')

    results = run_reductive_mechanism_analysis(df)
    print("\nReductive mechanism analysis results saved.")
