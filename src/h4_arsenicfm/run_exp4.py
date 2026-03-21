"""
Run Exp 4 (Permutation Test) only, then generate the 6-panel figure.
Reads Exp 1-3 results from saved CSVs.

Usage:
    python run_exp4.py              # n_perms=30 (fast, ~10 min)
    python run_exp4.py --perms 100  # full (slow, ~60 min)
"""

import sys
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    INTEGRATED_FILE, EMBEDDINGS_DIR, TABLES_DIR, FIGURES_DIR,
    GEO_FEATURE_COLS, CONTEXT_COLS, PHYSIOGRAPHIC_ZONES,
    WHO_AS_LIMIT, RANDOM_STATE
)

# Import experiments from the main module
from fm_claim_extended import (
    load_all, exp4_permutation_test, plot_fm_evidence
)


def main(n_perms=30):
    print("=" * 65)
    print("EXP 4 ONLY — Permutation Test + Final Figure")
    print(f"n_perms = {n_perms}")
    print("=" * 65)

    # Load data
    df, X_ae, X_geo, X_ctx, X_sat_idx, y_exc, sat_idx_cols = load_all()

    # Run Exp 4
    perm_results = exp4_permutation_test(df, X_ae, X_ctx, y_exc,
                                         n_perms=n_perms)

    # Load saved Exp 1-3 results
    ablation_df  = pd.read_csv(TABLES_DIR / 'H4_T16_embedding_ablation.csv')
    spatial_df   = pd.read_csv(TABLES_DIR / 'H4_T17_spatial_baselines.csv')
    fewshot_df   = pd.read_csv(TABLES_DIR / 'H4_T18_fewshot_adaptation.csv')
    screening_df = pd.read_csv(TABLES_DIR / 'H4_T15_fm_screening_experiment.csv')

    # Aggregate few-shot by k
    fewshot_agg = fewshot_df.groupby('k').agg(
        AUC_sat_mean=('AUC_sat_mean', 'mean'),
        AUC_sat_std=('AUC_sat_mean', 'std'),
        AUC_full_mean=('AUC_full_mean', 'mean'),
        AUC_full_std=('AUC_full_mean', 'std'),
    ).reset_index()

    # Generate 6-panel figure
    plot_fm_evidence(ablation_df, spatial_df, fewshot_agg,
                     perm_results, screening_df)

    print("\nDone.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--perms', type=int, default=30)
    args = parser.parse_args()
    main(n_perms=args.perms)
