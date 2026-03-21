"""
Phase 3: Bivariate Correlational Analysis
Testing core mechanism: As correlation with Fe, HCO3-, SO4-
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr, f_oneway
from pingouin import partial_corr
import matplotlib.pyplot as plt
import seaborn as sns
from config import (
    IONS, TARGET, REDOX_IONS, ALPHA, OUTPUT_DIR,
    FIGURES_DIR, TABLES_DIR, DPI
)

def calculate_correlation_matrix(df, variables=None, method='spearman'):
    """
    Calculate correlation matrix using Spearman rank correlation
    (preferred for geochemical data, non-parametric)
    """
    if variables is None:
        variables = IONS + [TARGET]

    available_vars = [v for v in variables if v in df.columns]

    # Use only complete cases
    df_complete = df[available_vars].dropna()

    print(f"\nCalculating {method.upper()} correlation matrix")
    print(f"Sample size: {len(df_complete)}")

    if method.lower() == 'spearman':
        corr_matrix = df_complete.corr(method='spearman')
    else:
        corr_matrix = df_complete.corr(method='pearson')

    return corr_matrix, df_complete

def calculate_pvalues(df, variables=None, method='spearman'):
    """
    Calculate p-values for correlations
    """
    if variables is None:
        variables = IONS + [TARGET]

    available_vars = [v for v in variables if v in df.columns]
    df_complete = df[available_vars].dropna()

    n_vars = len(available_vars)
    pval_matrix = np.zeros((n_vars, n_vars))

    for i, var1 in enumerate(available_vars):
        for j, var2 in enumerate(available_vars):
            if i == j:
                pval_matrix[i, j] = 0
            elif i > j:
                pval_matrix[i, j] = pval_matrix[j, i]
            else:
                if method.lower() == 'spearman':
                    _, p_value = spearmanr(df_complete[var1], df_complete[var2])
                else:
                    _, p_value = pearsonr(df_complete[var1], df_complete[var2])
                pval_matrix[i, j] = p_value

    pval_df = pd.DataFrame(pval_matrix, index=available_vars, columns=available_vars)
    return pval_df

def test_key_correlations(df, alpha=ALPHA):
    """
    Test primary hypothesis correlations:
    H1: As strongly correlated with Fe, HCO3-, negative with SO4-
    """
    print("\n" + "="*70)
    print("KEY CORRELATION TESTS (HYPOTHESIS 1 CRITICAL)")
    print("="*70)

    complete_pairs = []

    # Test As-Fe correlation
    if 'Fe2+' in df.columns:
        mask = df[[TARGET, 'Fe2+']].notna().all(axis=1)
        data_clean = df[mask]
        if len(data_clean) > 3:
            rho_fe, p_fe = spearmanr(data_clean[TARGET], data_clean['Fe2+'])
            complete_pairs.append({
                'Pair': f'{TARGET} - Fe2+',
                'N': len(data_clean),
                'Spearman_rho': rho_fe,
                'p_value': p_fe,
                'Significant': 'YES' if p_fe < alpha else 'NO',
                'Expected': '>0.45'
            })
            print(f"\n{TARGET} - Fe2+:")
            print(f"  N = {len(data_clean)}")
            print(f"  ρ = {rho_fe:.4f} (Expected: >0.45)")
            print(f"  p-value = {p_fe:.6f} (α = {alpha})")
            print(f"  Result: {'✓ SUPPORTS H1' if rho_fe > 0.45 and p_fe < alpha else '✗ Does not support H1'}")

    # Test As-HCO3- correlation
    if 'HCO3-' in df.columns:
        mask = df[[TARGET, 'HCO3-']].notna().all(axis=1)
        data_clean = df[mask]
        if len(data_clean) > 3:
            rho_hco3, p_hco3 = spearmanr(data_clean[TARGET], data_clean['HCO3-'])
            complete_pairs.append({
                'Pair': f'{TARGET} - HCO3-',
                'N': len(data_clean),
                'Spearman_rho': rho_hco3,
                'p_value': p_hco3,
                'Significant': 'YES' if p_hco3 < alpha else 'NO',
                'Expected': '>0.40'
            })
            print(f"\n{TARGET} - HCO3-:")
            print(f"  N = {len(data_clean)}")
            print(f"  ρ = {rho_hco3:.4f} (Expected: >0.40)")
            print(f"  p-value = {p_hco3:.6f}")
            print(f"  Result: {'✓ SUPPORTS H1' if rho_hco3 > 0.40 and p_hco3 < alpha else '✗ Does not support H1'}")

    # Test As-SO4- correlation (should be negative)
    if 'SO42-' in df.columns:
        mask = df[[TARGET, 'SO42-']].notna().all(axis=1)
        data_clean = df[mask]
        if len(data_clean) > 3:
            rho_so4, p_so4 = spearmanr(data_clean[TARGET], data_clean['SO42-'])
            complete_pairs.append({
                'Pair': f'{TARGET} - SO42-',
                'N': len(data_clean),
                'Spearman_rho': rho_so4,
                'p_value': p_so4,
                'Significant': 'YES' if p_so4 < alpha else 'NO',
                'Expected': '<-0.25'
            })
            print(f"\n{TARGET} - SO42-:")
            print(f"  N = {len(data_clean)}")
            print(f"  ρ = {rho_so4:.4f} (Expected: <-0.25)")
            print(f"  p-value = {p_so4:.6f}")
            print(f"  Result: {'✓ SUPPORTS H1' if rho_so4 < -0.25 and p_so4 < alpha else '✗ Does not support H1'}")

    key_corr_df = pd.DataFrame(complete_pairs)
    return key_corr_df

def calculate_partial_correlation(df, var1, var2, control_var):
    """
    Calculate partial correlation (control for confounding variable)
    """
    mask = df[[var1, var2, control_var]].notna().all(axis=1)
    data_clean = df[mask]

    if len(data_clean) > 5:
        pcorr = partial_corr(data=data_clean, x=var1, y=var2, covar=control_var, method='spearman')
        return pcorr
    else:
        return None

def plot_correlation_heatmap(corr_matrix, title='Correlation Matrix'):
    """
    Create publication-quality correlation heatmap
    """
    fig, ax = plt.subplots(figsize=(14, 12))

    # Create mask for upper triangle (optional)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )

    ax.set_title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_key_correlations(df):
    """
    Create scatter plots for key hypothesis correlations
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    key_pairs = [
        (TARGET, 'Fe2+', 'red', '>0.45'),
        (TARGET, 'HCO3-', 'blue', '>0.40'),
        (TARGET, 'SO42-', 'green', '<-0.25')
    ]

    for idx, (var1, var2, color, expected) in enumerate(key_pairs):
        if var1 in df.columns and var2 in df.columns:
            mask = df[[var1, var2]].notna().all(axis=1)
            data_clean = df[mask]

            rho, p_val = spearmanr(data_clean[var1], data_clean[var2])

            axes[idx].scatter(data_clean[var2], data_clean[var1], alpha=0.5, color=color, s=30)

            # Add regression line
            z = np.polyfit(data_clean[var2], data_clean[var1], 1)
            p = np.poly1d(z)
            x_line = np.linspace(data_clean[var2].min(), data_clean[var2].max(), 100)
            axes[idx].plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2)

            axes[idx].set_xlabel(var2, fontsize=12)
            axes[idx].set_ylabel(var1, fontsize=12)
            axes[idx].set_title(f'{var1} vs {var2}\nρ={rho:.3f}, p={p_val:.4f}\nExpected: {expected}',
                               fontsize=11, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def run_phase_3_analysis(df):
    """
    Run complete Phase 3: Bivariate Correlational Analysis
    """
    print("\n" + "="*70)
    print("PHASE 3: BIVARIATE CORRELATIONAL ANALYSIS")
    print("="*70)

    # Calculate correlation matrices
    corr_matrix, df_complete = calculate_correlation_matrix(df, method='spearman')
    pval_matrix = calculate_pvalues(df, method='spearman')

    # Save correlation matrices
    corr_matrix.to_csv(TABLES_DIR / 'Table_4_correlation_matrix.csv')
    pval_matrix.to_csv(TABLES_DIR / 'Table_5_pvalue_matrix.csv')

    # Test key correlations
    key_corr_results = test_key_correlations(df, alpha=ALPHA)
    key_corr_results.to_csv(TABLES_DIR / 'Table_6_key_correlations.csv', index=False)

    # Create visualizations
    fig1 = plot_correlation_heatmap(corr_matrix, 'Spearman Correlation Matrix - All Variables')
    fig1.savefig(FIGURES_DIR / 'Figure_3_correlation_heatmap.png', dpi=DPI, bbox_inches='tight')
    print(f"Figure saved: Figure_3_correlation_heatmap.png")
    plt.close(fig1)

    fig2 = plot_key_correlations(df)
    fig2.savefig(FIGURES_DIR / 'Figure_4_key_correlations_scatter.png', dpi=DPI, bbox_inches='tight')
    print(f"Figure saved: Figure_4_key_correlations_scatter.png")
    plt.close(fig2)

    print("\n" + "="*70)
    print("KEY CORRELATION RESULTS SUMMARY")
    print("="*70)
    print(key_corr_results.to_string(index=False))

    return {
        'corr_matrix': corr_matrix,
        'pval_matrix': pval_matrix,
        'key_correlations': key_corr_results,
        'n_complete_cases': len(df_complete)
    }

if __name__ == "__main__":
    from data_loader import load_and_prepare_data
    df, _, _ = load_and_prepare_data()
    results = run_phase_3_analysis(df)
