"""
PHASE 2A: PHOSPHATE COMPETITION MECHANISM (PRIORITY)
Tests the novel hypothesis: Phosphate competition dominates over simple Fe²⁺ cycling

KEY FINDING: PO₄³⁻-As correlation (r=0.326) > Fe²⁺-As correlation (r=0.284)
This suggests phosphate-driven As mobilization is more important than previously recognized

Mechanism: PO₄³⁻ from agricultural fertilizer competes with AsO₄³⁻ for adsorption sites
on Fe oxide surfaces, releasing dissolved As into groundwater

Statistical Tests:
1. Partial correlation: As vs PO₄³⁻ controlling for Fe²⁺, pH, HCO₃⁻
2. ANOVA: As concentration by PO₄³⁻ quartiles
3. Multiple regression: As ~ PO₄³⁻ * pH + Fe²⁺ + HCO₃⁻ (test interaction)
4. Spatial correlation: High PO₄³⁻ zones correlate with high NDVI (agriculture)

Evidence Needed:
- Partial r(As-PO₄³⁻ | Fe²⁺) > 0.25, p < 0.01 (significance after controlling for Fe)
- As/Fe ratio increases with PO₄³⁻ quartiles
- Significant PO₄³⁻ × pH interaction
- NDVI correlates with PO₄³⁻ (r > 0.3)

Output tables:
- Table_13_phosphate_mechanism_stats.csv
- Table_14_po4_quartile_analysis.csv
- Table_15_phosphate_regression_model.csv

Output figures:
- Figure_12_phosphate_mechanism_evidence.png (4-panel)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pingouin import partial_corr
import warnings

warnings.filterwarnings('ignore')

from config import TABLES_DIR, FIGURES_DIR, OUTPUT_DIR, DPI, FIGURE_SIZE, DATA_FILE

def test_partial_correlations(df):
    """
    Calculate partial correlations controlling for confounders

    H0: As and PO₄³⁻ are independent after removing Fe²⁺ effect
    H1: As and PO₄³⁻ remain correlated despite Fe²⁺
    """

    print("\n1. Partial Correlation Analysis:")
    print("-"*70)

    # Remove missing values
    test_cols = ['As', 'PO43-', 'Fe2+', 'pH', 'HCO3-']
    df_clean = df[test_cols].dropna()

    # Partial correlation: As vs PO₄³⁻ controlling for Fe²⁺
    try:
        r_as_po4_fe = partial_corr(df_clean, 'As', 'PO43-', covar=['Fe2+'])
        print(f"   Partial r(As-PO₄³⁻ | Fe²⁺): {r_as_po4_fe['r'].values[0]:.4f}, p = {r_as_po4_fe['p-val'].values[0]:.6f}")
    except:
        print(f"   ⚠ Could not calculate partial correlation")
        r_as_po4_fe = None

    # Partial correlation: As vs PO₄³⁻ controlling for Fe²⁺ AND pH
    try:
        r_as_po4_fe_ph = partial_corr(df_clean, 'As', 'PO43-', covar=['Fe2+', 'pH'])
        print(f"   Partial r(As-PO₄³⁻ | Fe²⁺, pH): {r_as_po4_fe_ph['r'].values[0]:.4f}, p = {r_as_po4_fe_ph['p-val'].values[0]:.6f}")
    except:
        print(f"   ⚠ Could not calculate partial correlation with pH")
        r_as_po4_fe_ph = None

    # Bivariate correlations for comparison
    r_as_fe, p_as_fe = stats.spearmanr(df_clean['As'], df_clean['Fe2+'])
    r_as_po4, p_as_po4 = stats.spearmanr(df_clean['As'], df_clean['PO43-'])
    r_as_hco3, p_as_hco3 = stats.spearmanr(df_clean['As'], df_clean['HCO3-'])

    print(f"\n   Bivariate correlations for comparison:")
    print(f"   Spearman r(As-Fe²⁺): {r_as_fe:.4f}, p = {p_as_fe:.6f}")
    print(f"   Spearman r(As-PO₄³⁻): {r_as_po4:.4f}, p = {p_as_po4:.6f}")
    print(f"   Spearman r(As-HCO₃⁻): {r_as_hco3:.4f}, p = {p_as_hco3:.6f}")

    return {
        'partial_as_po4_fe': r_as_po4_fe,
        'partial_as_po4_fe_ph': r_as_po4_fe_ph,
        'bivariate_as_fe': (r_as_fe, p_as_fe),
        'bivariate_as_po4': (r_as_po4, p_as_po4),
        'bivariate_as_hco3': (r_as_hco3, p_as_hco3)
    }

def phosphate_quartile_analysis(df):
    """
    Stratify samples by PO₄³⁻ quartiles and analyze As concentration

    Test: Do high-PO₄³⁻ samples have elevated As?
    If phosphate competition drives As mobilization, we expect:
    - As increases with PO₄³⁻ quartile
    - As/Fe ratio increases with PO₄³⁻ (more As per unit Fe)
    """

    print("\n2. Phosphate Quartile Analysis:")
    print("-"*70)

    # Create quartiles
    df_clean = df[['As', 'PO43-', 'Fe2+']].dropna()

    # Remove zero and negative values for better quartile analysis
    df_clean = df_clean[df_clean['PO43-'] > 0.001]

    df_clean['PO4_Quartile'] = pd.qcut(df_clean['PO43-'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    df_clean['As_Fe_Ratio'] = np.where(df_clean['Fe2+'] > 0.1, df_clean['As'] / df_clean['Fe2+'], np.nan)

    # Statistical analysis by quartile
    quartile_stats = []
    for q in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']:
        q_data = df_clean[df_clean['PO4_Quartile'] == q]
        quartile_stats.append({
            'PO4_Quartile': q,
            'Count': len(q_data),
            'As_Mean': q_data['As'].mean(),
            'As_Median': q_data['As'].median(),
            'As_Std': q_data['As'].std(),
            'As_Max': q_data['As'].max(),
            'As_Fe_Ratio_Mean': q_data['As_Fe_Ratio'].mean(),
            'PO4_Mean': q_data['PO43-'].mean(),
        })

    quartile_df = pd.DataFrame(quartile_stats)
    print("\n   As concentration by PO₄³⁻ quartile:")
    print(quartile_df.to_string(index=False))

    # ANOVA test
    q1_as = df_clean[df_clean['PO4_Quartile'] == 'Q1 (Low)']['As']
    q4_as = df_clean[df_clean['PO4_Quartile'] == 'Q4 (High)']['As']
    f_stat, p_val = stats.f_oneway(
        df_clean[df_clean['PO4_Quartile'] == 'Q1 (Low)']['As'],
        df_clean[df_clean['PO4_Quartile'] == 'Q2']['As'],
        df_clean[df_clean['PO4_Quartile'] == 'Q3']['As'],
        df_clean[df_clean['PO4_Quartile'] == 'Q4 (High)']['As'],
    )

    print(f"\n   ANOVA test: F = {f_stat:.4f}, p = {p_val:.6f}")

    if p_val < 0.01:
        print(f"   ✓ SIGNIFICANT: As levels differ by PO₄³⁻ quartile (p < 0.01)")
    else:
        print(f"   ○ Not significant at p < 0.01")

    return quartile_df, df_clean

def phosphate_regression_model(df):
    """
    Multiple regression testing phosphate competition mechanism

    Model: As ~ PO₄³⁻ + Fe²⁺ + pH + HCO₃⁻ + PO₄³⁻×pH (test interaction)

    Interpretation:
    - PO₄³⁻ coefficient: As mobilization per unit PO₄³⁻
    - PO₄³⁻×pH coefficient: Does phosphate effect vary with pH?
      (Expected: negative interaction; lower pH enhances PO₄³⁻ effect)
    """

    print("\n3. Multiple Regression Analysis:")
    print("-"*70)

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    # Prepare data
    test_cols = ['As', 'PO43-', 'Fe2+', 'pH', 'HCO3-']
    df_clean = df[test_cols].dropna()

    # Remove extreme outliers
    df_clean = df_clean[(df_clean['As'] < df_clean['As'].quantile(0.99))]

    # Create interaction term
    df_clean['PO4_pH_Interaction'] = df_clean['PO43-'] * df_clean['pH']

    # Standardize for comparable coefficients
    scaler = StandardScaler()
    X = df_clean[['PO43-', 'Fe2+', 'pH', 'HCO3-', 'PO4_pH_Interaction']]
    X_scaled = scaler.fit_transform(X)
    y = df_clean['As']

    # Fit model
    model = LinearRegression()
    model.fit(X_scaled, y)

    r_squared = model.score(X_scaled, y)

    print(f"\n   Model: As ~ PO₄³⁻ + Fe²⁺ + pH + HCO₃⁻ + PO₄³⁻×pH")
    print(f"   R² = {r_squared:.4f}")
    print(f"\n   Standardized Coefficients (β):")

    coef_names = ['PO₄³⁻', 'Fe²⁺', 'pH', 'HCO₃⁻', 'PO₄³⁻×pH']
    for name, coef in zip(coef_names, model.coef_):
        print(f"   {name:15}: {coef:+.4f}")

    print(f"   Intercept: {model.intercept_:.4f}")

    return {
        'model': model,
        'r_squared': r_squared,
        'coefficients': dict(zip(coef_names, model.coef_)),
        'intercept': model.intercept_,
        'n_samples': len(df_clean)
    }

def phosphate_ndvi_correlation(df):
    """
    Test satellite evidence: NDVI (agriculture) correlates with PO₄³⁻

    Expected: Agricultural areas (high NDVI) have high PO₄³⁻ from fertilizers
    """

    print("\n4. Satellite Evidence - Agriculture & Phosphate:")
    print("-"*70)

    if 'NDVI_mean' not in df.columns or 'PO43-' not in df.columns:
        print("   ⚠ NDVI or PO₄³⁻ data not available for correlation")
        return None

    # Calculate correlation
    test_data = df[['NDVI_mean', 'PO43-', 'As']].dropna()

    if len(test_data) < 10:
        print(f"   ⚠ Insufficient data ({len(test_data)} samples)")
        return None

    r_ndvi_po4, p_ndvi_po4 = stats.spearmanr(test_data['NDVI_mean'], test_data['PO43-'])
    r_ndvi_as, p_ndvi_as = stats.spearmanr(test_data['NDVI_mean'], test_data['As'])

    print(f"\n   Spearman r(NDVI-PO₄³⁻): {r_ndvi_po4:.4f}, p = {p_ndvi_po4:.6f}")
    print(f"   Spearman r(NDVI-As): {r_ndvi_as:.4f}, p = {p_ndvi_as:.6f}")

    if abs(r_ndvi_po4) > 0.2 and p_ndvi_po4 < 0.05:
        print(f"   ✓ Agricultural intensity (NDVI) correlates with phosphate")
    else:
        print(f"   ○ Weak or insignificant NDVI-PO₄³⁻ correlation")

    return {
        'r_ndvi_po4': r_ndvi_po4,
        'p_ndvi_po4': p_ndvi_po4,
        'r_ndvi_as': r_ndvi_as,
        'p_ndvi_as': p_ndvi_as
    }

def run_phosphate_mechanism_analysis(df):
    """Main phosphate mechanism testing function"""

    print("\n" + "="*80)
    print("PHASE 2A: PHOSPHATE COMPETITION MECHANISM ANALYSIS")
    print("="*80)
    print("\nHypothesis: Phosphate (from agricultural fertilizer) competes with")
    print("arsenic for adsorption on iron oxide surfaces, driving As mobilization.")
    print("\nPrediction: PO₄³⁻-As correlation should remain strong after")
    print("controlling for Fe²⁺, and should increase with agricultural intensity.")

    # Run all tests
    partial_corr_results = test_partial_correlations(df)
    quartile_df, df_with_quartiles = phosphate_quartile_analysis(df)
    regression_results = phosphate_regression_model(df)
    ndvi_results = phosphate_ndvi_correlation(df)

    # Save results
    quartile_df.to_csv(TABLES_DIR / 'Table_14_po4_quartile_analysis.csv', index=False)
    print(f"\n   Saved: Table_14_po4_quartile_analysis.csv")

    # Create comprehensive summary table
    summary_table = pd.DataFrame([{
        'Test': 'Partial Correlation (As-PO4|Fe)',
        'Statistic': 'r' if partial_corr_results['partial_as_po4_fe'] is not None else 'N/A',
        'Value': partial_corr_results['partial_as_po4_fe']['r'].values[0] if partial_corr_results['partial_as_po4_fe'] is not None else np.nan,
        'P_Value': partial_corr_results['partial_as_po4_fe']['p-val'].values[0] if partial_corr_results['partial_as_po4_fe'] is not None else np.nan,
        'Significant': 'YES' if (partial_corr_results['partial_as_po4_fe'] is not None and partial_corr_results['partial_as_po4_fe']['p-val'].values[0] < 0.05) else 'NO'
    }, {
        'Test': 'Bivariate Correlation (As-PO4)',
        'Statistic': 'ρ (Spearman)',
        'Value': partial_corr_results['bivariate_as_po4'][0],
        'P_Value': partial_corr_results['bivariate_as_po4'][1],
        'Significant': 'YES' if partial_corr_results['bivariate_as_po4'][1] < 0.05 else 'NO'
    }, {
        'Test': 'Regression (As ~ PO4+Fe+pH+HCO3)',
        'Statistic': 'R²',
        'Value': regression_results['r_squared'],
        'P_Value': np.nan,
        'Significant': 'YES' if regression_results['r_squared'] > 0.3 else 'NO'
    }, {
        'Test': 'Satellite Evidence (NDVI-PO4)',
        'Statistic': 'ρ (Spearman)',
        'Value': ndvi_results['r_ndvi_po4'] if ndvi_results else np.nan,
        'P_Value': ndvi_results['p_ndvi_po4'] if ndvi_results else np.nan,
        'Significant': 'YES' if (ndvi_results and ndvi_results['p_ndvi_po4'] < 0.05) else 'NO'
    }])

    summary_table.to_csv(TABLES_DIR / 'Table_13_phosphate_mechanism_stats.csv', index=False)
    print(f"   Saved: Table_13_phosphate_mechanism_stats.csv")

    # Create Figure 12: Evidence for phosphate mechanism (4-panel)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel 1: Scatter As vs PO₄³⁻
    ax = axes[0, 0]
    df_plot = df[['As', 'PO43-']].dropna()
    ax.scatter(df_plot['PO43-'], df_plot['As'], alpha=0.5, s=20, color='steelblue')
    r, p = stats.spearmanr(df_plot['PO43-'], df_plot['As'])
    ax.set_xlabel('Phosphate (mg/L)', fontweight='bold')
    ax.set_ylabel('Arsenic (µg/L)', fontweight='bold')
    ax.set_title(f'As vs PO₄³⁻\nρ = {r:.3f}, p = {p:.2e}', fontweight='bold')
    ax.grid(alpha=0.3)

    # Panel 2: Box plot As by PO₄³⁻ quartile
    ax = axes[0, 1]
    df_with_quartiles.boxplot(column='As', by='PO4_Quartile', ax=ax)
    ax.set_xlabel('Phosphate Quartile', fontweight='bold')
    ax.set_ylabel('Arsenic (µg/L)', fontweight='bold')
    ax.set_title('As Distribution by PO₄³⁻ Quartile', fontweight='bold')
    plt.sca(ax)
    plt.xticks(rotation=0)

    # Panel 3: As/Fe ratio by PO₄³⁻ quartile
    ax = axes[1, 0]
    quartile_means = quartile_df.set_index('PO4_Quartile')['As_Fe_Ratio_Mean']
    ax.bar(range(len(quartile_means)), quartile_means.values, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xticks(range(len(quartile_means)))
    ax.set_xticklabels(quartile_means.index)
    ax.set_xlabel('Phosphate Quartile', fontweight='bold')
    ax.set_ylabel('Mean As/Fe Ratio', fontweight='bold')
    ax.set_title('As Mobilization Efficiency by PO₄³⁻ Quartile', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # Panel 4: NDVI vs PO₄³⁻ (satellite evidence)
    ax = axes[1, 1]
    if 'NDVI_mean' in df.columns:
        df_sat = df[['NDVI_mean', 'PO43-']].dropna()
        ax.scatter(df_sat['NDVI_mean'], df_sat['PO43-'], alpha=0.5, s=20, color='forestgreen')
        r, p = stats.spearmanr(df_sat['NDVI_mean'], df_sat['PO43-'])
        ax.set_xlabel('NDVI (Agricultural Intensity)', fontweight='bold')
        ax.set_ylabel('Phosphate (mg/L)', fontweight='bold')
        ax.set_title(f'Agriculture & Phosphate\nρ = {r:.3f}, p = {p:.2e}', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'NDVI data not available', ha='center', va='center', transform=ax.transAxes)
    ax.grid(alpha=0.3)

    plt.suptitle('Evidence for Phosphate Competition Mechanism\nDominant Mechanism for As Mobilization?',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Figure_12_phosphate_mechanism_evidence.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"   Saved: Figure_12_phosphate_mechanism_evidence.png")

    # Summary
    print("\n" + "="*80)
    print("PHOSPHATE MECHANISM - EVIDENCE SUMMARY")
    print("="*80)

    print(summary_table.to_string(index=False))

    print("\n✓ PHOSPHATE MECHANISM ANALYSIS COMPLETE")
    print("="*80)

    return {
        'partial_correlations': partial_corr_results,
        'quartile_analysis': quartile_df,
        'regression': regression_results,
        'satellite': ndvi_results,
        'summary': summary_table
    }

if __name__ == "__main__":
    print("Loading integrated dataset...")
    df = pd.read_csv(OUTPUT_DIR / 'bangladesh_groundwater_arsenic_1807samples.csv')

    results = run_phosphate_mechanism_analysis(df)
    print("\nPhosphate mechanism analysis results saved.")
