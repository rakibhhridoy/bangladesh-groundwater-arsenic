"""
Phase 4: Principal Component Analysis (PCA)
Testing if As clusters with redox indicators (Fe, HCO3-, SO4-)
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from config import (
    IONS, TARGET, REDOX_IONS, PCA_VARIANCE_THRESHOLD,
    OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, DPI
)

def prepare_pca_data(df, variables=None):
    """
    Prepare data for PCA: standardize and handle missing values
    """
    if variables is None:
        variables = IONS + [TARGET]

    available_vars = [v for v in variables if v in df.columns]

    # Get complete cases
    df_complete = df[available_vars].dropna()

    print(f"\nPCA Data Preparation:")
    print(f"  Variables: {len(available_vars)}")
    print(f"  Complete cases: {len(df_complete)}/{len(df)}")

    # Standardize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_complete)

    return data_scaled, available_vars, df_complete, scaler

def run_pca(data_scaled, variables, variance_threshold=PCA_VARIANCE_THRESHOLD):
    """
    Run PCA and determine number of components
    """
    print(f"\n{'='*70}")
    print(f"PRINCIPAL COMPONENT ANALYSIS")
    print(f"{'='*70}")

    # Fit PCA with all components
    pca_full = PCA()
    pca_full.fit(data_scaled)

    # Cumulative variance explained
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)

    # Determine number of components
    n_components = np.argmax(cumsum_var >= variance_threshold) + 1

    print(f"\nVariance explained by components:")
    for i in range(min(6, len(pca_full.explained_variance_ratio_))):
        print(f"  PC{i+1}: {pca_full.explained_variance_ratio_[i]:.4f} ({cumsum_var[i]:.4f} cumulative)")

    print(f"\nComponents explaining {variance_threshold*100:.0f}% variance: {n_components}")

    # Fit final PCA with selected components
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)

    return pca, pca_full, data_pca, cumsum_var, n_components

def analyze_loadings(pca, variables):
    """
    Analyze PCA loadings - which ions load on each PC?
    """
    print(f"\n{'='*70}")
    print(f"PCA LOADINGS ANALYSIS")
    print(f"{'='*70}")

    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=variables
    )

    # Sort by absolute loading
    for col in loadings_df.columns:
        print(f"\n{col} Loadings (top variables):")
        top_vars = loadings_df[col].abs().sort_values(ascending=False).head(8)
        for var, loading in top_vars.items():
            actual_loading = loadings_df.loc[var, col]
            print(f"  {var:12} : {actual_loading:7.4f} (|{actual_loading:.4f}|)")

    return loadings_df

def test_hypothesis_1(pca, loadings_df, target=TARGET):
    """
    Test if PC1 contains redox signature:
    - Positive loadings: As, Fe, HCO3-
    - Negative loadings: SO4-
    """
    print(f"\n{'='*70}")
    print(f"HYPOTHESIS 1 TEST: REDOX SIGNATURE IN PC1")
    print(f"{'='*70}")

    pc1_loadings = loadings_df['PC1'].copy()

    # Check if As, Fe, HCO3 are on positive side
    redox_positive = ['As', 'Fe2+', 'HCO3-']
    redox_negative = ['SO42-']

    print(f"\nExpected pattern for reductive dissolution mechanism:")
    print(f"  Should have positive loadings: {redox_positive}")
    print(f"  Should have negative loadings: {redox_negative}")

    print(f"\nActual PC1 loadings:")
    for var in redox_positive + redox_negative:
        if var in pc1_loadings.index:
            loading = pc1_loadings[var]
            print(f"  {var:12} : {loading:7.4f}")

    # Calculate hypothesis support score
    support_score = 0
    if pc1_loadings.get('As', 0) > 0.3:
        support_score += 1
    if pc1_loadings.get('Fe2+', 0) > 0.3:
        support_score += 1
    if pc1_loadings.get('HCO3-', 0) > 0.3:
        support_score += 1
    if pc1_loadings.get('SO42-', 0) < -0.2:
        support_score += 1

    print(f"\nHypothesis support: {support_score}/4 criteria met")
    if support_score >= 3:
        print("✓ STRONG SUPPORT for reductive dissolution mechanism")
    elif support_score >= 2:
        print("○ MODERATE SUPPORT for reductive dissolution mechanism")
    else:
        print("✗ WEAK SUPPORT for reductive dissolution mechanism")

    return support_score

def calculate_communalities(pca, n_components):
    """
    Calculate communalities: variance in each variable explained by selected PCs
    """
    communalities = np.sum(pca.components_**2, axis=0)
    communalities_df = pd.DataFrame({
        'Variable': pca.feature_names_in_ if hasattr(pca, 'feature_names_in_') else range(len(communalities)),
        'Communality': communalities
    })

    return communalities_df.sort_values('Communality', ascending=False)

def plot_scree_plot(pca_full, cumsum_var, variance_threshold):
    """
    Create scree plot showing variance explained
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    n_components = len(pca_full.explained_variance_ratio_)
    pc_labels = [f'PC{i+1}' for i in range(min(15, n_components))]

    ax.bar(range(1, min(16, n_components + 1)),
           pca_full.explained_variance_ratio_[:min(15, n_components)],
           alpha=0.7, color='steelblue', label='Individual')

    ax.plot(range(1, min(16, n_components + 1)),
            cumsum_var[:min(15, n_components)],
            marker='o', color='red', linewidth=2, label='Cumulative')

    ax.axhline(y=variance_threshold, color='green', linestyle='--', linewidth=2,
               label=f'{variance_threshold*100:.0f}% threshold')

    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Variance Explained', fontsize=12)
    ax.set_title('PCA Scree Plot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig

def plot_biplot(pca, loadings_df, data_pca, color_var=None):
    """
    Create PCA biplot: PC1 vs PC2 with variable vectors
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot scores (samples)
    if color_var is not None:
        scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.5, c=color_var, cmap='viridis', s=30)
        plt.colorbar(scatter, label='Arsenic (µg/L)')
    else:
        ax.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.5, s=30)

    # Plot loadings (vectors)
    scale_factor = 3.5
    for i, var in enumerate(loadings_df.index):
        ax.arrow(0, 0,
                loadings_df.iloc[i, 0] * scale_factor,
                loadings_df.iloc[i, 1] * scale_factor,
                head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.6)

        # Label
        ax.text(loadings_df.iloc[i, 0] * scale_factor * 1.15,
               loadings_df.iloc[i, 1] * scale_factor * 1.15,
               var, fontsize=10, fontweight='bold')

    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('PCA Biplot (PC1 vs PC2)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    return fig

def plot_loadings_heatmap(loadings_df):
    """
    Create heatmap of PCA loadings
    """
    fig, ax = plt.subplots(figsize=(8, 12))

    sns.heatmap(loadings_df.iloc[:, :min(5, loadings_df.shape[1])],
                annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Loading'}, ax=ax)

    ax.set_title('PCA Loadings Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Variable', fontsize=12)

    return fig

def run_phase_4_analysis(df, target_for_coloring=TARGET):
    """
    Run complete Phase 4: PCA Analysis
    """
    print("\n" + "="*70)
    print("PHASE 4: PRINCIPAL COMPONENT ANALYSIS")
    print("="*70)

    # Prepare data
    data_scaled, variables, df_complete, scaler = prepare_pca_data(
        df, variables=IONS + [TARGET]
    )

    # Run PCA
    pca, pca_full, data_pca, cumsum_var, n_components = run_pca(
        data_scaled, variables, variance_threshold=PCA_VARIANCE_THRESHOLD
    )

    # Analyze loadings
    loadings_df = analyze_loadings(pca, variables)
    loadings_df.to_csv(TABLES_DIR / 'Table_7_pca_loadings.csv')

    # Test hypothesis
    support_score = test_hypothesis_1(pca, loadings_df, target=TARGET)

    # Communalities
    comm_df = calculate_communalities(pca, pca.n_components_)
    comm_df.to_csv(TABLES_DIR / 'Table_8_communalities.csv', index=False)

    # Visualizations
    fig1 = plot_scree_plot(pca_full, cumsum_var, PCA_VARIANCE_THRESHOLD)
    fig1.savefig(FIGURES_DIR / 'Figure_5_scree_plot.png', dpi=DPI, bbox_inches='tight')
    print(f"Figure saved: Figure_5_scree_plot.png")
    plt.close(fig1)

    # Biplot with arsenic coloring
    if TARGET in df_complete.columns:
        color_var = df_complete[TARGET].values
    else:
        color_var = None

    fig2 = plot_biplot(pca, loadings_df, data_pca, color_var=color_var)
    fig2.savefig(FIGURES_DIR / 'Figure_6_pca_biplot.png', dpi=DPI, bbox_inches='tight')
    print(f"Figure saved: Figure_6_pca_biplot.png")
    plt.close(fig2)

    fig3 = plot_loadings_heatmap(loadings_df)
    fig3.savefig(FIGURES_DIR / 'Figure_7_loadings_heatmap.png', dpi=DPI, bbox_inches='tight')
    print(f"Figure saved: Figure_7_loadings_heatmap.png")
    plt.close(fig3)

    return {
        'pca': pca,
        'pca_full': pca_full,
        'loadings_df': loadings_df,
        'communalities': comm_df,
        'data_pca': data_pca,
        'support_score': support_score,
        'n_components': n_components,
        'cumsum_var': cumsum_var
    }

if __name__ == "__main__":
    from data_loader import load_and_prepare_data
    df, _, _ = load_and_prepare_data()
    results = run_phase_4_analysis(df)
