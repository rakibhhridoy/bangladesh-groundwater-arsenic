"""
PHASE 1C: DATA INTEGRATION
Merge groundwater chemistry + satellite features + ion ratios

Input tables:
- Table_9_charge_balance_error.csv (from geochemical_validation.py)
- Table_10_ion_ratios.csv (from geochemical_validation.py)
- Table_11_water_chemistry_categories.csv (from geochemical_validation.py)
- Table_12_satellite_features.csv (from satellite_features.py)
- Data.csv (original groundwater data)

Output:
- data_phase1_integrated.csv (1,807 samples × ~60 features)

This integrated dataset is used for all Phase 2-4 analyses
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import DATA_FILE, TABLES_DIR, OUTPUT_DIR

def create_integrated_dataset():
    """Merge all data sources by Sample_ID"""

    print("\n" + "="*80)
    print("PHASE 1C: DATA INTEGRATION")
    print("="*80)

    # Load original groundwater data
    print("\n1. Loading original groundwater data...")
    df_base = pd.read_csv(DATA_FILE)
    print(f"   Groundwater data: {df_base.shape} (samples × features)")
    print(f"   Columns: {list(df_base.columns)[:10]}...")

    # Load geochemical validation results
    print("\n2. Loading geochemical validation results...")

    # Charge balance error
    try:
        cbe_df = pd.read_csv(TABLES_DIR / 'Table_9_charge_balance_error.csv')
        print(f"   CBE data: {cbe_df.shape}")
    except FileNotFoundError:
        print(f"   ⚠ CBE data not found, skipping...")
        cbe_df = None

    # Ion ratios
    try:
        ratios_df = pd.read_csv(TABLES_DIR / 'Table_10_ion_ratios.csv')
        print(f"   Ion ratios: {ratios_df.shape}")
    except FileNotFoundError:
        print(f"   ⚠ Ion ratios not found, skipping...")
        ratios_df = None

    # Water chemistry categories
    try:
        chem_cat_df = pd.read_csv(TABLES_DIR / 'Table_11_water_chemistry_categories.csv')
        print(f"   Water chemistry categories: {chem_cat_df.shape}")
    except FileNotFoundError:
        print(f"   ⚠ Water chemistry categories not found, skipping...")
        chem_cat_df = None

    # Satellite features
    print("\n3. Loading satellite features...")
    try:
        satellite_df = pd.read_csv(TABLES_DIR / 'Table_12_satellite_features.csv')
        print(f"   Satellite features: {satellite_df.shape}")
    except FileNotFoundError:
        print(f"   ⚠ Satellite features not found, skipping...")
        satellite_df = None

    # Start with base data
    df_integrated = df_base.copy()

    # Merge CBE data
    if cbe_df is not None:
        print("\n4. Merging charge balance error data...")
        # Match by Sample_ID or row index
        if 'Sample_ID' in cbe_df.columns and 'Sample_ID' in df_integrated.columns:
            df_integrated = df_integrated.merge(
                cbe_df[['Sample_ID', 'CBE_percent', 'CBE_Quality']],
                on='Sample_ID',
                how='left'
            )
        else:
            # Use index-based merge
            df_integrated = df_integrated.assign(**{
                'CBE_percent': cbe_df['CBE_percent'].values,
                'CBE_Quality': cbe_df['CBE_Quality'].values
            })
        print(f"   Integrated data shape: {df_integrated.shape}")

    # Merge ion ratios
    if ratios_df is not None:
        print("\n5. Merging ion ratio data...")
        # Keep only ratio columns (not chemistry columns already in base)
        ratio_cols = [col for col in ratios_df.columns if 'ratio' in col.lower()]
        if ratio_cols:
            if 'Sample_ID' in ratios_df.columns and 'Sample_ID' in df_integrated.columns:
                df_integrated = df_integrated.merge(
                    ratios_df[['Sample_ID'] + ratio_cols],
                    on='Sample_ID',
                    how='left'
                )
            else:
                for col in ratio_cols:
                    df_integrated[col] = ratios_df[col].values
        print(f"   Integrated data shape: {df_integrated.shape}")

    # Merge water chemistry categories
    if chem_cat_df is not None:
        print("\n6. Merging water chemistry categories...")
        if 'Sample_ID' in chem_cat_df.columns and 'Sample_ID' in df_integrated.columns:
            # Only merge Water_Type column
            df_integrated = df_integrated.merge(
                chem_cat_df[['Sample_ID', 'Water_Type']],
                on='Sample_ID',
                how='left'
            )
        print(f"   Integrated data shape: {df_integrated.shape}")

    # Merge satellite features
    if satellite_df is not None:
        print("\n7. Merging satellite features...")
        # Try to match by Location_ID
        if 'Location_ID' in satellite_df.columns:
            if 'Sample_ID' in df_integrated.columns:
                # Assume Sample_ID corresponds to Location_ID
                df_integrated = df_integrated.merge(
                    satellite_df,
                    left_on='Sample_ID',
                    right_on='Location_ID',
                    how='left',
                    suffixes=('', '_sat')
                )
                # Remove duplicate Location_ID column if created
                if 'Location_ID' in df_integrated.columns and 'Sample_ID' in df_integrated.columns:
                    # Keep only one ID column
                    pass
        print(f"   Integrated data shape: {df_integrated.shape}")

    # Summary statistics
    print("\n8. Dataset Integration Summary:")
    print(f"   Total samples: {len(df_integrated)}")
    print(f"   Total features: {df_integrated.shape[1]}")
    print(f"   Missing values: {df_integrated.isnull().sum().sum()} ({100*df_integrated.isnull().sum().sum()/(len(df_integrated)*df_integrated.shape[1]):.2f}%)")
    print(f"\n   Feature categories:")
    print(f"   - Original groundwater chemistry: {len(df_base.columns)}")
    if cbe_df is not None:
        print(f"   - Geochemical validation: 2")
    if ratios_df is not None:
        ratio_cols = [col for col in ratios_df.columns if 'ratio' in col.lower()]
        print(f"   - Ion ratios: {len(ratio_cols)}")
    if chem_cat_df is not None:
        print(f"   - Water chemistry categories: 1")
    if satellite_df is not None:
        sat_cols = [col for col in satellite_df.columns if col != 'Location_ID' and col != 'Sample_ID']
        print(f"   - Satellite features: {len(sat_cols)}")

    # Save integrated dataset
    output_path = OUTPUT_DIR / 'data_phase1_integrated.csv'
    df_integrated.to_csv(output_path, index=False)
    print(f"\n9. Saving integrated dataset...")
    print(f"   Saved to: {output_path}")

    # Create data quality report
    print("\n10. Creating data quality report...")

    report_lines = [
        "PHASE 1 DATA INTEGRATION REPORT",
        "="*80,
        "",
        f"Integration Date: {pd.Timestamp.now()}",
        f"Total Samples: {len(df_integrated)}",
        f"Total Features: {df_integrated.shape[1]}",
        "",
        "MISSING DATA SUMMARY:",
        "-"*80,
    ]

    missing_pct = df_integrated.isnull().sum() / len(df_integrated) * 100
    missing_summary = missing_pct[missing_pct > 0].sort_values(ascending=False)

    if len(missing_summary) > 0:
        for col, pct in missing_summary.items():
            report_lines.append(f"{col}: {pct:.2f}% ({int(missing_pct[col] * len(df_integrated))} samples)")
    else:
        report_lines.append("No missing data detected")

    report_lines.extend([
        "",
        "FEATURE STATISTICS:",
        "-"*80,
        "",
    ])

    # Add basic statistics for key columns
    for col in ['As', 'Fe2+', 'HCO3-', 'PO43-', 'pH', 'Do', 'CBE_percent']:
        if col in df_integrated.columns:
            report_lines.append(f"{col}:")
            report_lines.append(f"  Mean: {df_integrated[col].mean():.4f}")
            report_lines.append(f"  Median: {df_integrated[col].median():.4f}")
            report_lines.append(f"  Std Dev: {df_integrated[col].std():.4f}")
            report_lines.append(f"  Min: {df_integrated[col].min():.4f}")
            report_lines.append(f"  Max: {df_integrated[col].max():.4f}")
            report_lines.append("")

    # Save report
    report_path = OUTPUT_DIR / 'integration_report.txt'
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))

    print(f"   Saved to: {report_path}")

    print("\n" + "="*80)
    print("✓ DATA INTEGRATION COMPLETE")
    print("="*80)

    return df_integrated

def verify_integration(df):
    """Verify integrated dataset quality"""

    print("\n11. Data Integration Verification:")
    print("-"*80)

    # Check for required columns
    required_cols = ['Sample_ID', 'As', 'Fe2+', 'HCO3-', 'pH']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"  ⚠ Missing required columns: {missing_cols}")
    else:
        print(f"  ✓ All required columns present")

    # Check data types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"  ✓ {len(numeric_cols)} numeric columns")

    # Check for outliers in key variables
    outlier_count = 0
    for col in ['As', 'Fe2+', 'PO43-']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
            outlier_count += outliers
            if outliers > 0:
                print(f"  - {col}: {outliers} potential outliers")

    print(f"\n  ✓ Integration verification complete")

    return True

if __name__ == "__main__":
    df_integrated = create_integrated_dataset()
    verify_integration(df_integrated)

    print(f"\nIntegrated dataset summary:")
    print(df_integrated.describe())
