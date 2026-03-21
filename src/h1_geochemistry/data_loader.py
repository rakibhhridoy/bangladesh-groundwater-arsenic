"""
Data loading and preparation for Hypothesis 1 analysis
Phase 1: Data cleaning & quality control
"""

import pandas as pd
import numpy as np
import warnings
from config import (
    DATA_FILE, DRY_SEASON_FILE, WET_SEASON_FILE,
    IONS, TARGET, WATER_QUALITY_PARAMS, SPATIAL_VARS,
    TEMPORAL_VAR, REDOX_IONS, WHO_AS_LIMIT,
    LOG_TRANSFORM_VARS, ALPHA, OUTPUT_DIR
)

warnings.filterwarnings('ignore')

def load_data():
    """Load main dataset"""
    print("Loading data from:", DATA_FILE)
    df = pd.read_csv(DATA_FILE)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df

def assess_missing_values(df, report_to_file=True):
    """
    Assess missing values in dataset
    Reports % missing for each variable
    """
    print("\n" + "="*70)
    print("MISSING VALUE ASSESSMENT")
    print("="*70)

    missing_summary = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percent': (df.isnull().sum() / len(df) * 100).round(2)
    }).sort_values('Missing_Percent', ascending=False)

    print(missing_summary.to_string(index=False))

    # Save to file
    if report_to_file:
        missing_summary.to_csv(
            OUTPUT_DIR / 'missing_values_report.csv',
            index=False
        )
        print(f"\nMissing values report saved to: {OUTPUT_DIR}/missing_values_report.csv")

    return missing_summary

def remove_missing_target(df, target=TARGET):
    """Remove rows where target variable (As) is missing"""
    initial_n = len(df)
    df = df[df[target].notna()].copy()
    removed = initial_n - len(df)
    print(f"\nRemoved {removed} samples with missing {target}")
    print(f"Remaining samples: {len(df)}")
    return df

def remove_missing_redox_set(df):
    """
    Create indicator for complete redox set
    Keep only samples with all: As, Fe2+, HCO3-, SO42-, pH
    """
    redox_vars = REDOX_IONS + ['pH', TARGET]

    initial_n = len(df)
    df['Complete_Redox_Set'] = df[redox_vars].notna().all(axis=1)

    complete_n = df['Complete_Redox_Set'].sum()
    print(f"\nRedox set completeness:")
    print(f"  Total samples: {initial_n}")
    print(f"  Complete redox set: {complete_n} ({100*complete_n/initial_n:.1f}%)")
    print(f"  Missing redox set: {initial_n - complete_n}")

    return df

def detect_outliers_univariate(df, variables=None, threshold=3.0):
    """
    Detect univariate outliers using IQR method
    threshold = 3.0 means Q3 + 3*IQR
    """
    if variables is None:
        variables = IONS + [TARGET] + WATER_QUALITY_PARAMS

    print("\n" + "="*70)
    print("UNIVARIATE OUTLIER DETECTION (IQR Method)")
    print("="*70)

    outliers_dict = {}

    for var in variables:
        if var in df.columns:
            Q1 = df[var].quantile(0.25)
            Q3 = df[var].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outlier_mask = (df[var] < lower_bound) | (df[var] > upper_bound)
            n_outliers = outlier_mask.sum()

            if n_outliers > 0:
                outliers_dict[var] = {
                    'count': n_outliers,
                    'percent': 100 * n_outliers / len(df),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'indices': df[outlier_mask].index.tolist()
                }
                print(f"{var:12} | Outliers: {n_outliers:4} ({100*n_outliers/len(df):5.2f}%) | "
                      f"Range: [{lower_bound:.2f}, {upper_bound:.2f}]")

    return outliers_dict

def handle_outliers(df, outliers_dict, action='flag'):
    """
    Handle outliers
    action = 'flag' : add flag column
    action = 'remove': remove outliers (only if physically implausible)
    """
    if action == 'flag':
        df['Is_Outlier'] = False
        for var, info in outliers_dict.items():
            df.loc[info['indices'], 'Is_Outlier'] = True
        print(f"\nFlagged {df['Is_Outlier'].sum()} samples as outliers")

    return df

def log_transform_variables(df):
    """Log-transform right-skewed variables"""
    print("\n" + "="*70)
    print("LOG TRANSFORMATION OF SKEWED VARIABLES")
    print("="*70)

    transformed_vars = []
    for var in LOG_TRANSFORM_VARS:
        if var in df.columns:
            # Create log-transformed column (handle zeros by adding small value)
            df[f'{var}_log'] = np.log10(df[var] + 1e-6)
            transformed_vars.append(var)

            skew_original = df[var].skew()
            skew_log = df[f'{var}_log'].skew()
            print(f"{var:12} | Original skew: {skew_original:7.2f} | Log skew: {skew_log:7.2f}")

    print(f"\nCreated log-transformed columns for {len(transformed_vars)} variables")
    return df

def standardize_variables(df, variables=None):
    """Standardize variables to mean=0, SD=1"""
    from sklearn.preprocessing import StandardScaler

    if variables is None:
        variables = IONS + [TARGET]

    print(f"\nStandardizing {len(variables)} variables")

    scaler = StandardScaler()
    available_vars = [v for v in variables if v in df.columns]

    df[f'{available_vars[0]}_std'] = scaler.fit_transform(df[[available_vars[0]]])

    for var in available_vars[1:]:
        df[f'{var}_std'] = scaler.fit_transform(df[[var]])

    return df, scaler

def load_and_prepare_data(remove_outliers=False, standardize=True, log_transform=True):
    """
    Main function: Load and prepare data for analysis

    Returns:
        df: Prepared dataframe
        missing_summary: Missing value report
        outliers: Detected outliers
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 1: DATA PREPARATION")
    print("="*70)

    # Load data
    df = load_data()

    # Assess missing values
    missing_summary = assess_missing_values(df)

    # Remove rows with missing target
    df = remove_missing_target(df)

    # Create redox set indicator
    df = remove_missing_redox_set(df)

    # Detect outliers
    outliers = detect_outliers_univariate(df, threshold=3.0)

    # Flag outliers
    df = handle_outliers(df, outliers, action='flag')

    # Data quality summary
    print("\n" + "="*70)
    print("DATA QUALITY SUMMARY")
    print("="*70)
    print(f"Total samples: {len(df)}")
    print(f"Complete redox set: {df['Complete_Redox_Set'].sum()}")
    print(f"Flagged as outliers: {df['Is_Outlier'].sum()}")
    print(f"Data quality: {100 * (1 - df['Is_Outlier'].sum() / len(df)):.1f}%")

    # WHO Arsenic Exceedance
    print(f"\nARSENIC EXCEEDANCE (WHO limit = {WHO_AS_LIMIT} µg/L):")
    above_who = (df[TARGET] > WHO_AS_LIMIT).sum()
    above_50 = (df[TARGET] > 50).sum()
    print(f"  Above WHO limit (10 µg/L): {above_who} ({100*above_who/len(df):.1f}%)")
    print(f"  Severe (>50 µg/L): {above_50} ({100*above_50/len(df):.1f}%)")

    # Apply transformations
    if log_transform:
        df = log_transform_variables(df)

    if standardize:
        df, _ = standardize_variables(df)

    print("\n" + "="*70)
    print("DATA PREPARATION COMPLETE")
    print("="*70)
    print(f"Final dataset shape: {df.shape}")
    print(f"Columns created: {len(df.columns) - len(load_data().columns)}")

    return df, missing_summary, outliers

if __name__ == "__main__":
    # Run if executed directly
    df, missing_summary, outliers = load_and_prepare_data()

    # Save prepared dataset
    output_path = OUTPUT_DIR / 'data_prepared.csv'
    df.to_csv(output_path, index=False)
    print(f"\nPrepared data saved to: {output_path}")
