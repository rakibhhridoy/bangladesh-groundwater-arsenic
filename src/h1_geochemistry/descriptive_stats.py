"""
Phase 2: Univariate Analysis - Descriptive Statistics
Testing Hypothesis 1: Arsenic distribution and redox chemistry
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from config import (
    IONS, TARGET, WHO_AS_LIMIT, OUTPUT_DIR,
    FIGURES_DIR, TABLES_DIR, REDOX_IONS, DPI, COLORMAP
)

def calculate_descriptive_statistics(df, variables=None):
    """
    Calculate comprehensive descriptive statistics for variables
    """
    if variables is None:
        variables = IONS + [TARGET]

    available_vars = [v for v in variables if v in df.columns]

    stats_dict = {}

    for var in available_vars:
        data = df[var].dropna()

        stats_dict[var] = {
            'N': len(data),
            'Mean': data.mean(),
            'Median': data.median(),
            'Std': data.std(),
            'Min': data.min(),
            'Max': data.max(),
            'Q1': data.quantile(0.25),
            'Q3': data.quantile(0.75),
            'IQR': data.quantile(0.75) - data.quantile(0.25),
            'Skewness': data.skew(),
            'Kurtosis': data.kurtosis(),
            'CV%': (data.std() / data.mean() * 100) if data.mean() != 0 else np.nan
        }

    stats_df = pd.DataFrame(stats_dict).T
    return stats_df

def test_normality(df, variables=None):
    """
    Test normality using Shapiro-Wilk test
    (for n < 5000)
    """
    if variables is None:
        variables = IONS + [TARGET]

    print("\n" + "="*70)
    print("NORMALITY TEST (Shapiro-Wilk)")
    print("="*70)

    normality_results = []

    for var in variables:
        if var in df.columns:
            data = df[var].dropna()

            if len(data) > 5000:
                # Use a sample for large datasets
                data_sample = data.sample(n=5000, random_state=42)
            else:
                data_sample = data

            if len(data_sample) > 3:  # Minimum for Shapiro-Wilk
                stat, p_value = stats.shapiro(data_sample)
                is_normal = "Yes" if p_value > 0.05 else "No"

                normality_results.append({
                    'Variable': var,
                    'N': len(data),
                    'W_statistic': stat,
                    'p_value': p_value,
                    'Normal_p>0.05': is_normal
                })

                print(f"{var:12} | W={stat:.4f} | p={p_value:.4f} | Normal: {is_normal}")

    normality_df = pd.DataFrame(normality_results)
    return normality_df

def fit_distributions(df, variable=TARGET):
    """
    Fit different distributions to a variable
    Test: Normal, Log-normal, Gamma
    """
    print(f"\n{'='*70}")
    print(f"DISTRIBUTION FITTING: {variable}")
    print("="*70)

    data = df[variable].dropna()

    # Normal distribution
    mu_norm, sigma_norm = stats.norm.fit(data)
    ks_norm = stats.kstest(data, 'norm', args=(mu_norm, sigma_norm))

    # Log-normal distribution
    shape_lognorm, loc_lognorm, scale_lognorm = stats.lognorm.fit(data)
    ks_lognorm = stats.kstest(
        data,
        lambda x: stats.lognorm.cdf(x, shape_lognorm, loc_lognorm, scale_lognorm)
    )

    # Gamma distribution
    shape_gamma, loc_gamma, scale_gamma = stats.gamma.fit(data)
    ks_gamma = stats.kstest(
        data,
        lambda x: stats.gamma.cdf(x, shape_gamma, loc_gamma, scale_gamma)
    )

    fit_results = {
        'Normal': {'KS_stat': ks_norm.statistic, 'p_value': ks_norm.pvalue},
        'Log-normal': {'KS_stat': ks_lognorm.statistic, 'p_value': ks_lognorm.pvalue},
        'Gamma': {'KS_stat': ks_gamma.statistic, 'p_value': ks_gamma.pvalue}
    }

    fit_df = pd.DataFrame(fit_results).T
    print(fit_df)

    return fit_df

def calculate_who_exceedance(df, target=TARGET, limit=WHO_AS_LIMIT):
    """
    Calculate WHO Arsenic exceedance statistics
    """
    print(f"\n{'='*70}")
    print(f"ARSENIC EXCEEDANCE ANALYSIS (WHO Limit = {limit} µg/L)")
    print("="*70)

    data = df[target].dropna()

    above_limit = (data > limit).sum()
    above_50 = (data > 50).sum()
    above_100 = (data > 100).sum()

    exceedance_stats = {
        'Total_Samples': len(data),
        'Above_WHO_10': above_limit,
        'Percent_Above_WHO': 100 * above_limit / len(data),
        'Above_50': above_50,
        'Percent_Above_50': 100 * above_50 / len(data),
        'Above_100': above_100,
        'Percent_Above_100': 100 * above_100 / len(data),
        'Mean': data.mean(),
        'Median': data.median(),
        'Max': data.max(),
        'Percentile_95': data.quantile(0.95),
        'Percentile_75': data.quantile(0.75),
    }

    for key, val in exceedance_stats.items():
        print(f"{key:25} : {val:8.2f}")

    return exceedance_stats

def plot_distribution(df, variables=None):
    """
    Create comprehensive distribution plots
    """
    if variables is None:
        variables = [TARGET] + REDOX_IONS[:3]

    available_vars = [v for v in variables if v in df.columns]

    fig, axes = plt.subplots(len(available_vars), 2, figsize=(14, 4*len(available_vars)))
    if len(available_vars) == 1:
        axes = axes.reshape(1, -1)

    for idx, var in enumerate(available_vars):
        data = df[var].dropna()

        # Histogram
        axes[idx, 0].hist(data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[idx, 0].set_xlabel(var)
        axes[idx, 0].set_ylabel('Frequency')
        axes[idx, 0].set_title(f'{var} - Histogram')
        axes[idx, 0].axvline(data.mean(), color='red', linestyle='--', label=f'Mean={data.mean():.2f}')
        axes[idx, 0].axvline(data.median(), color='green', linestyle='--', label=f'Median={data.median():.2f}')
        axes[idx, 0].legend()

        # Q-Q plot
        stats.probplot(data, dist="norm", plot=axes[idx, 1])
        axes[idx, 1].set_title(f'{var} - Q-Q Plot')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Figure_1_distribution_analysis.png', dpi=DPI, bbox_inches='tight')
    print(f"\nFigure saved: Figure_1_distribution_analysis.png")
    plt.close()

def plot_arsenic_exceedance(df, target=TARGET):
    """
    Create WHO exceedance visualization
    """
    data = df[target].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram with WHO limit line
    axes[0].hist(data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(WHO_AS_LIMIT, color='red', linestyle='--', linewidth=2, label=f'WHO Limit ({WHO_AS_LIMIT} µg/L)')
    axes[0].set_xlabel(f'{target} (µg/L)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{target} Distribution with WHO Limit')
    axes[0].legend()
    axes[0].set_xlim(0, min(data.quantile(0.95) * 1.1, 500))

    # Pie chart of exceedance
    above = (data > WHO_AS_LIMIT).sum()
    below = (data <= WHO_AS_LIMIT).sum()
    sizes = [above, below]
    labels = [f'Above WHO\n({above}, {100*above/(above+below):.1f}%)',
              f'Below WHO\n({below}, {100*below/(above+below):.1f}%)']
    colors = ['#ff6b6b', '#51cf66']

    axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('WHO Arsenic Exceedance')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Figure_2_arsenic_exceedance.png', dpi=DPI, bbox_inches='tight')
    print(f"Figure saved: Figure_2_arsenic_exceedance.png")
    plt.close()

def generate_summary_table(df):
    """
    Generate and save comprehensive summary table
    """
    print("\n" + "="*70)
    print("GENERATING SUMMARY TABLE 1: DESCRIPTIVE STATISTICS")
    print("="*70)

    stats_df = calculate_descriptive_statistics(df, IONS + [TARGET])

    # Round for presentation
    stats_df_display = stats_df.copy()
    for col in stats_df_display.columns:
        stats_df_display[col] = stats_df_display[col].round(2)

    # Save table
    stats_df.to_csv(TABLES_DIR / 'Table_1_descriptive_statistics.csv')
    print(f"Table saved: Table_1_descriptive_statistics.csv")

    return stats_df_display

def run_phase_2_analysis(df):
    """
    Run complete Phase 2: Univariate Analysis
    """
    print("\n" + "="*70)
    print("PHASE 2: UNIVARIATE ANALYSIS")
    print("="*70)

    # Descriptive statistics
    stats_df = generate_summary_table(df)
    print("\nDescriptive Statistics Summary:")
    print(stats_df.to_string())

    # Normality testing
    normality_df = test_normality(df, [TARGET] + REDOX_IONS)
    normality_df.to_csv(TABLES_DIR / 'Table_2_normality_test.csv', index=False)

    # Distribution fitting
    fit_results = fit_distributions(df, TARGET)
    fit_results.to_csv(TABLES_DIR / 'Table_3_distribution_fit.csv')

    # WHO Exceedance
    exceedance_stats = calculate_who_exceedance(df, TARGET)

    # Plotting
    plot_distribution(df, [TARGET, 'Fe2+', 'HCO3-', 'SO42-'])
    plot_arsenic_exceedance(df, TARGET)

    return {
        'stats_df': stats_df,
        'normality_df': normality_df,
        'fit_results': fit_results,
        'exceedance_stats': exceedance_stats
    }

if __name__ == "__main__":
    from data_loader import load_and_prepare_data
    df, _, _ = load_and_prepare_data()
    results = run_phase_2_analysis(df)
