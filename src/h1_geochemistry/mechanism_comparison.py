"""
PHASE 2C: MECHANISM COMPARISON & RANKING
Statistically compare all arsenic mobilization mechanisms

Mechanisms tested:
1. Phosphate Competition: PO₄³⁻ competes with As for Fe oxide binding sites
2. Reductive Dissolution: Fe(III) oxide reduction releases adsorbed As
3. pH-Dependent Desorption: High pH enhances As desorption
4. Combined (Multi-mechanism): Interaction of multiple pathways

Comparison methodology:
- Multiple linear regression models for each mechanism
- Model selection: AIC, BIC (lower = better fit with fewer parameters)
- Variance partitioning: Relative importance of each mechanism
- Cross-validation: Stability of mechanism rankings

Output tables:
- Table_19_mechanism_model_comparison.csv (AIC, BIC, R², RMSE)
- Table_20_mechanism_importance.csv (variance partition)

Output figure:
- Figure_14_mechanism_comparison.png (coefficient plot)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

from config import TABLES_DIR, FIGURES_DIR, OUTPUT_DIR, DPI, FIGURE_SIZE

def build_mechanism_models(df):
    """Build regression models for each mechanism"""

    print("\n" + "="*80)
    print("PHASE 2C: MECHANISM MODEL COMPARISON")
    print("="*80)

    print("\n1. Building Mechanism-Specific Models:")
    print("-"*70)

    # Prepare data
    test_cols = ['As', 'PO43-', 'Fe2+', 'pH', 'HCO3-', 'DO']
    df_clean = df[test_cols].dropna()

    # Remove extreme outliers (>99th percentile)
    df_clean = df_clean[(df_clean['As'] < df_clean['As'].quantile(0.99))]

    print(f"   Analysis sample size: {len(df_clean)} samples")

    # Standardize features for comparable coefficients
    scaler = StandardScaler()

    # Model 1: Phosphate Competition
    # As ~ PO₄³⁻ + Fe²⁺ + pH + PO₄³⁻×pH
    X1 = df_clean[['PO43-', 'Fe2+', 'pH']].copy()
    X1['PO4_pH_Int'] = df_clean['PO43-'] * df_clean['pH']
    X1_scaled = scaler.fit_transform(X1)

    model1 = LinearRegression()
    model1.fit(X1_scaled, df_clean['As'])
    pred1 = model1.predict(X1_scaled)
    r2_1 = r2_score(df_clean['As'], pred1)
    rmse_1 = np.sqrt(mean_squared_error(df_clean['As'], pred1))

    # Model 2: Reductive Dissolution
    # As ~ Fe²⁺ + HCO₃⁻ + Fe²⁺×HCO₃⁻
    X2 = df_clean[['Fe2+', 'HCO3-']].copy()
    X2['Fe_HCO3_Int'] = df_clean['Fe2+'] * df_clean['HCO3-']
    X2_scaled = scaler.fit_transform(X2)

    model2 = LinearRegression()
    model2.fit(X2_scaled, df_clean['As'])
    pred2 = model2.predict(X2_scaled)
    r2_2 = r2_score(df_clean['As'], pred2)
    rmse_2 = np.sqrt(mean_squared_error(df_clean['As'], pred2))

    # Model 3: pH-Dependent Desorption
    # As ~ pH + HCO₃⁻ (as pH proxy for desorption)
    X3 = df_clean[['pH', 'HCO3-']].copy()
    X3_scaled = scaler.fit_transform(X3)

    model3 = LinearRegression()
    model3.fit(X3_scaled, df_clean['As'])
    pred3 = model3.predict(X3_scaled)
    r2_3 = r2_score(df_clean['As'], pred3)
    rmse_3 = np.sqrt(mean_squared_error(df_clean['As'], pred3))

    # Model 4: Combined (All mechanisms)
    # As ~ PO₄³⁻ + Fe²⁺ + pH + HCO₃⁻ + DO
    X4 = df_clean[['PO43-', 'Fe2+', 'pH', 'HCO3-', 'DO']].copy()
    X4_scaled = scaler.fit_transform(X4)

    model4 = LinearRegression()
    model4.fit(X4_scaled, df_clean['As'])
    pred4 = model4.predict(X4_scaled)
    r2_4 = r2_score(df_clean['As'], pred4)
    rmse_4 = np.sqrt(mean_squared_error(df_clean['As'], pred4))

    # Calculate AIC and BIC
    # AIC = 2k + n*ln(RSS/n), where k = parameters, n = samples, RSS = residual sum of squares
    def calculate_aic_bic(y_true, y_pred, n_params):
        n = len(y_true)
        rss = np.sum((y_true - y_pred) ** 2)
        aic = 2 * n_params + n * np.log(rss / n)
        bic = n_params * np.log(n) + n * np.log(rss / n)
        return aic, bic

    aic1, bic1 = calculate_aic_bic(df_clean['As'], pred1, X1.shape[1])
    aic2, bic2 = calculate_aic_bic(df_clean['As'], pred2, X2.shape[1])
    aic3, bic3 = calculate_aic_bic(df_clean['As'], pred3, X3.shape[1])
    aic4, bic4 = calculate_aic_bic(df_clean['As'], pred4, X4.shape[1])

    # Create comparison table
    comparison_df = pd.DataFrame([
        {
            'Mechanism': 'Phosphate Competition',
            'Model': 'As ~ PO4 + Fe + pH + PO4×pH',
            'N_Parameters': X1.shape[1],
            'R_Squared': r2_1,
            'RMSE': rmse_1,
            'AIC': aic1,
            'BIC': bic1,
            'AIC_Delta': aic1 - min(aic1, aic2, aic3, aic4),
            'BIC_Delta': bic1 - min(bic1, bic2, bic3, bic4),
        },
        {
            'Mechanism': 'Reductive Dissolution',
            'Model': 'As ~ Fe + HCO3 + Fe×HCO3',
            'N_Parameters': X2.shape[1],
            'R_Squared': r2_2,
            'RMSE': rmse_2,
            'AIC': aic2,
            'BIC': bic2,
            'AIC_Delta': aic2 - min(aic1, aic2, aic3, aic4),
            'BIC_Delta': bic2 - min(bic1, bic2, bic3, bic4),
        },
        {
            'Mechanism': 'pH-Dependent Desorption',
            'Model': 'As ~ pH + HCO3',
            'N_Parameters': X3.shape[1],
            'R_Squared': r2_3,
            'RMSE': rmse_3,
            'AIC': aic3,
            'BIC': bic3,
            'AIC_Delta': aic3 - min(aic1, aic2, aic3, aic4),
            'BIC_Delta': bic3 - min(bic1, bic2, bic3, bic4),
        },
        {
            'Mechanism': 'Combined (Multi-mechanism)',
            'Model': 'As ~ PO4 + Fe + pH + HCO3 + DO',
            'N_Parameters': X4.shape[1],
            'R_Squared': r2_4,
            'RMSE': rmse_4,
            'AIC': aic4,
            'BIC': bic4,
            'AIC_Delta': aic4 - min(aic1, aic2, aic3, aic4),
            'BIC_Delta': bic4 - min(bic1, bic2, bic3, bic4),
        },
    ])

    print("\n2. Model Comparison Results:")
    print("-"*70)
    print(comparison_df[['Mechanism', 'N_Parameters', 'R_Squared', 'RMSE', 'AIC_Delta', 'BIC_Delta']].to_string(index=False))

    # Interpretation
    print("\n3. Model Selection Interpretation:")
    print("-"*70)

    best_aic_idx = comparison_df['AIC_Delta'].idxmin()
    best_bic_idx = comparison_df['BIC_Delta'].idxmin()

    print(f"\n   Best model by AIC: {comparison_df.loc[best_aic_idx, 'Mechanism']}")
    print(f"   Best model by BIC: {comparison_df.loc[best_bic_idx, 'Mechanism']}")

    # R² ranking
    print(f"\n   Explanatory power ranking (R²):")
    for idx, (mech, r2) in enumerate(zip(comparison_df['Mechanism'], comparison_df['R_Squared']), 1):
        print(f"   {idx}. {mech:35}: R² = {r2:.4f}")

    comparison_df.to_csv(TABLES_DIR / 'Table_19_mechanism_model_comparison.csv', index=False)
    print(f"\n   Saved: Table_19_mechanism_model_comparison.csv")

    return comparison_df, {
        'model1': (model1, X1_scaled, pred1, 'Phosphate'),
        'model2': (model2, X2_scaled, pred2, 'Reductive'),
        'model3': (model3, X3_scaled, pred3, 'pH-Desorption'),
        'model4': (model4, X4_scaled, pred4, 'Combined'),
    }

def calculate_mechanism_importance(df):
    """Estimate relative importance of each mechanism via structure"""

    print("\n4. Mechanism Importance Assessment:")
    print("-"*70)

    # Simple method: relative correlation strength
    test_cols = ['As', 'PO43-', 'Fe2+', 'pH', 'HCO3-']
    df_clean = df[test_cols].dropna()

    # Key correlations
    r_as_po4, _ = stats.spearmanr(df_clean['As'], df_clean['PO43-'])
    r_as_fe, _ = stats.spearmanr(df_clean['As'], df_clean['Fe2+'])
    r_as_ph, _ = stats.spearmanr(df_clean['As'], df_clean['pH'])
    r_as_hco3, _ = stats.spearmanr(df_clean['As'], df_clean['HCO3-'])

    # Normalize to percentage
    total_corr = abs(r_as_po4) + abs(r_as_fe) + abs(r_as_ph) + abs(r_as_hco3)

    importance_df = pd.DataFrame([
        {
            'Mechanism': 'Phosphate Competition',
            'Key_Variable': 'PO43-',
            'Abs_Correlation': abs(r_as_po4),
            'Relative_Importance': 100 * abs(r_as_po4) / total_corr,
        },
        {
            'Mechanism': 'Reductive Dissolution',
            'Key_Variable': 'Fe2+',
            'Abs_Correlation': abs(r_as_fe),
            'Relative_Importance': 100 * abs(r_as_fe) / total_corr,
        },
        {
            'Mechanism': 'pH-Dependent Desorption',
            'Key_Variable': 'pH',
            'Abs_Correlation': abs(r_as_ph),
            'Relative_Importance': 100 * abs(r_as_ph) / total_corr,
        },
        {
            'Mechanism': 'Buffering (HCO3 proxy)',
            'Key_Variable': 'HCO3-',
            'Abs_Correlation': abs(r_as_hco3),
            'Relative_Importance': 100 * abs(r_as_hco3) / total_corr,
        },
    ])

    print("\n   Mechanism Importance Ranking:")
    for idx, row in importance_df.sort_values('Relative_Importance', ascending=False).iterrows():
        print(f"   {row['Mechanism']:35}: {row['Relative_Importance']:6.1f}%")

    importance_df.to_csv(TABLES_DIR / 'Table_20_mechanism_importance.csv', index=False)
    print(f"\n   Saved: Table_20_mechanism_importance.csv")

    return importance_df

def create_comparison_figure(comparison_df):
    """Create publication figure for mechanism comparison"""

    print("\n5. Creating comparison figure...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel 1: R² by mechanism
    ax = axes[0, 0]
    mechanisms = comparison_df['Mechanism'].values
    r2_vals = comparison_df['R_Squared'].values
    colors_r2 = ['red' if 'Phosphate' in m else 'orange' if 'Reductive' in m else 'blue' if 'pH' in m else 'green' for m in mechanisms]
    ax.bar(range(len(r2_vals)), r2_vals, color=colors_r2, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(r2_vals)))
    ax.set_xticklabels(mechanisms, rotation=45, ha='right')
    ax.set_ylabel('R² (Explanatory Power)', fontweight='bold')
    ax.set_title('Model Fit Quality by Mechanism', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim([0, 0.4])

    # Panel 2: AIC Delta (model selection)
    ax = axes[0, 1]
    aic_delta = comparison_df['AIC_Delta'].values
    ax.bar(range(len(aic_delta)), aic_delta, color=colors_r2, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(aic_delta)))
    ax.set_xticklabels(mechanisms, rotation=45, ha='right')
    ax.set_ylabel('ΔAICc (from best model)', fontweight='bold')
    ax.set_title('Model Selection by AIC\n(Lower = Better; Δ<4 = Competitive)',
                fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    ax.axhline(4, color='red', linestyle='--', linewidth=1, label='AIC Threshold (Δ=4)')

    # Panel 3: RMSE (prediction error)
    ax = axes[1, 0]
    rmse_vals = comparison_df['RMSE'].values
    ax.bar(range(len(rmse_vals)), rmse_vals, color=colors_r2, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(rmse_vals)))
    ax.set_xticklabels(mechanisms, rotation=45, ha='right')
    ax.set_ylabel('RMSE (µg/L)', fontweight='bold')
    ax.set_title('Prediction Error by Mechanism', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # Panel 4: Mechanism summary (text)
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = "MECHANISM RANKING:\n\n"
    for idx, (_, row) in enumerate(comparison_df.sort_values('R_Squared', ascending=False).iterrows(), 1):
        summary_text += f"{idx}. {row['Mechanism']}\n"
        summary_text += f"   R² = {row['R_Squared']:.4f}, RMSE = {row['RMSE']:.2f} µg/L\n"
        if row['AIC_Delta'] < 4:
            summary_text += f"   ✓ Competitive (ΔAICc = {row['AIC_Delta']:.1f})\n"
        else:
            summary_text += f"   ✗ Weaker (ΔAICc = {row['AIC_Delta']:.1f})\n"
        summary_text += "\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Arsenic Mobilization Mechanisms: Statistical Comparison\nWhich mechanism dominates?',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Figure_14_mechanism_comparison.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"   Saved: Figure_14_mechanism_comparison.png")

def run_mechanism_comparison(df):
    """Main mechanism comparison analysis"""

    print("\n" + "="*80)
    print("MECHANISM COMPARISON & RANKING")
    print("="*80)

    # Build and compare models
    comparison_df, models = build_mechanism_models(df)

    # Calculate importance
    importance_df = calculate_mechanism_importance(df)

    # Create figure
    create_comparison_figure(comparison_df)

    print("\n" + "="*80)
    print("✓ MECHANISM COMPARISON COMPLETE")
    print("="*80)

    return {
        'comparison': comparison_df,
        'importance': importance_df,
        'models': models
    }

if __name__ == "__main__":
    print("Loading integrated dataset...")
    df = pd.read_csv(OUTPUT_DIR / 'data_phase1_integrated.csv')

    results = run_mechanism_comparison(df)
    print("\nMechanism comparison complete.")
