"""
ArseniCFM Main Orchestrator — H4
Runs all phases of the foundation model pipeline.

Usage:
    python main.py              # run all phases
    python main.py --phases A B # run specific phases
    python main.py --force      # recompute cached embeddings

Phases:
    A: Sentinel-2 feature extraction → 64-dim statistical embeddings
    B: Linear probing on frozen embeddings (FM property 1)
    C: Embedding space analysis (UMAP, clustering, Mantel)
    D: Full ArseniCFM training + ablation (FM property 2)
    E: Transferability experiments (FM property 3)
    F: National risk mapping
    G: Results synthesis
"""

import sys
import argparse
import logging
import traceback
import numpy as np
from datetime import datetime
from pathlib import Path

# ─── LOGGING ──────────────────────────────────────────────────────────────────
from config import LOGS_DIR
log_file = LOGS_DIR / f'H4_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ─── PHASE RUNNERS ────────────────────────────────────────────────────────────

def phase_a_embeddings(args):
    """Extract Sentinel-2 statistical embeddings (or load cache)."""
    from sentinel_extractor import extract_all_embeddings
    embeddings, location_ids = extract_all_embeddings(force_recompute=args.force)
    return embeddings, location_ids


def phase_b_probing(embeddings, y_exc, y_mech, y_zone, y_conc, groups):
    """Linear probing on frozen embeddings."""
    from probing import run_probing
    return run_probing(embeddings, y_exc, y_mech, y_zone, y_conc, groups)


def phase_c_embedding_analysis(embeddings, df):
    """UMAP, clustering, Mantel tests."""
    from embedding_analysis import run_embedding_analysis
    return run_embedding_analysis(embeddings, df)


def phase_d_training(X_sat, X_geo, X_ctx, y_conc, y_exc, y_mech, y_zone, groups):
    """Full ArseniCFM training + ablation."""
    from trainer import run_training
    return run_training(X_sat, X_geo, X_ctx, y_conc, y_exc, y_mech, y_zone,
                        groups, run_ablation_study=True)


def phase_e_transferability(X_sat, X_geo, X_ctx, y_conc, y_exc, df):
    """LOZO-CV + few-shot learning curves."""
    from transferability import run_transferability
    return run_transferability(X_sat, X_geo, X_ctx, y_conc, y_exc, df)


def phase_f_risk_mapping(df, X_sat, X_geo, X_ctx, y_conc, y_exc, fold_metrics):
    """National risk mapping using best-fold predictions."""
    from risk_mapping import run_risk_mapping
    # Use training predictions as proxy (would use held-out test in production)
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    X_full = np.concatenate([X_sat, X_geo, X_ctx], axis=1)
    sc = StandardScaler().fit(X_full)
    reg = Ridge(alpha=1.0).fit(sc.transform(X_full), y_conc)
    clf_prob = Ridge(alpha=1.0).fit(sc.transform(X_full), y_exc.astype(float))
    y_conc_pred = reg.predict(sc.transform(X_full))
    y_exc_pred  = np.clip(clf_prob.predict(sc.transform(X_full)), 0, 1)
    return run_risk_mapping(df, X_sat, X_geo, X_ctx, y_conc_pred, y_exc_pred,
                            y_conc, y_exc)


def phase_g_synthesis(probing_r, training_r, ablation_r, transfer_r, tier_label):
    """Results synthesis."""
    from results_synthesis import run_results_synthesis
    return run_results_synthesis(probing_r, training_r, ablation_r,
                                 transfer_r, tier_label)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ArseniCFM H4 Pipeline')
    parser.add_argument('--phases', nargs='+', default=['A','B','C','D','E','F','G'],
                        choices=['A','B','C','D','E','F','G'],
                        help='Phases to run')
    parser.add_argument('--force', action='store_true',
                        help='Force recompute cached embeddings')
    args = parser.parse_args()

    phases = set(args.phases)
    logger.info("=" * 65)
    logger.info("H4: ArseniCFM Foundation Model Pipeline")
    logger.info(f"Phases: {sorted(phases)}")
    logger.info("=" * 65)

    results = {}

    # ── Load data (always) ────────────────────────────────────────────────
    logger.info("\n[DATA] Loading integrated dataset...")
    from data_loader import load_and_prepare_data, build_feature_matrices, make_spatial_blocks
    df = load_and_prepare_data()

    # ── Phase A: Embeddings ───────────────────────────────────────────────
    embeddings, location_ids = None, None
    if 'A' in phases:
        logger.info("\n[A] Sentinel-2 Embedding Extraction")
        try:
            embeddings, location_ids = phase_a_embeddings(args)
            logger.info(f"  Embeddings: {embeddings.shape}")
        except Exception as e:
            logger.error(f"  Phase A failed: {e}\n{traceback.format_exc()}")

    if embeddings is None:
        # Try loading cache
        from config import EMBEDDINGS_DIR
        cache = EMBEDDINGS_DIR / 'sentinel_statistical_embeddings.npy'
        if cache.exists():
            import pandas as pd
            embeddings = np.load(cache)
            location_ids = pd.read_csv(
                EMBEDDINGS_DIR / 'sentinel_location_ids.csv')['location_id'].values
            logger.info(f"  Loaded cached embeddings: {embeddings.shape}")
        else:
            # No embeddings available: use zeros + warn
            embeddings = np.zeros((len(df), 64), dtype=np.float32)
            location_ids = np.arange(1, len(df)+1)
            logger.warning("  No embeddings found; using zeros (Phase A required)")

    # ── Build feature matrices ────────────────────────────────────────────
    logger.info("\n[DATA] Building feature matrices...")
    X_sat, X_geo, X_ctx, y_conc, y_exc, y_mech, y_zone = \
        build_feature_matrices(df, embeddings, location_ids)
    groups = make_spatial_blocks(df)

    # Detect embedding tier
    ae_nonzero = float(np.abs(X_sat).sum()) > 0
    tier_label = 'Statistical (fallback)' if ae_nonzero else 'Zeros (Phase A needed)'

    # ── Phase B: Linear Probing ───────────────────────────────────────────
    probing_r = None
    if 'B' in phases:
        logger.info("\n[B] Linear Probing")
        try:
            probing_r = phase_b_probing(X_sat, y_exc, y_mech, y_zone, y_conc, groups)
        except Exception as e:
            logger.error(f"  Phase B failed: {e}\n{traceback.format_exc()}")
    results['probing'] = probing_r

    # ── Phase C: Embedding Analysis ───────────────────────────────────────
    if 'C' in phases:
        logger.info("\n[C] Embedding Space Analysis")
        try:
            results['embedding_analysis'] = phase_c_embedding_analysis(X_sat, df)
        except Exception as e:
            logger.error(f"  Phase C failed: {e}\n{traceback.format_exc()}")

    # ── Phase D: Training ─────────────────────────────────────────────────
    training_r, ablation_r = None, None
    if 'D' in phases:
        logger.info("\n[D] ArseniCFM Training + Ablation")
        try:
            fold_metrics, artifacts, ablation_df = phase_d_training(
                X_sat, X_geo, X_ctx, y_conc, y_exc, y_mech, y_zone, groups)
            training_r = fold_metrics
            ablation_r = ablation_df
        except Exception as e:
            logger.error(f"  Phase D failed: {e}\n{traceback.format_exc()}")
    results['training'] = training_r
    results['ablation'] = ablation_r

    # ── Phase E: Transferability ──────────────────────────────────────────
    transfer_r = None
    if 'E' in phases:
        logger.info("\n[E] Transferability Experiments")
        try:
            transfer_r = phase_e_transferability(X_sat, X_geo, X_ctx, y_conc, y_exc, df)
        except Exception as e:
            logger.error(f"  Phase E failed: {e}\n{traceback.format_exc()}")
    results['transfer'] = transfer_r

    # ── Phase F: Risk Mapping ─────────────────────────────────────────────
    if 'F' in phases:
        logger.info("\n[F] National Risk Mapping")
        try:
            results['risk_mapping'] = phase_f_risk_mapping(
                df, X_sat, X_geo, X_ctx, y_conc, y_exc,
                training_r if training_r else [])
        except Exception as e:
            logger.error(f"  Phase F failed: {e}\n{traceback.format_exc()}")

    # ── Phase G: Synthesis ────────────────────────────────────────────────
    if 'G' in phases:
        logger.info("\n[G] Results Synthesis")
        try:
            results['synthesis'] = phase_g_synthesis(
                probing_r, training_r, ablation_r, transfer_r, tier_label)
        except Exception as e:
            logger.error(f"  Phase G failed: {e}\n{traceback.format_exc()}")

    logger.info("\n" + "=" * 65)
    logger.info("H4 COMPLETE")
    logger.info("=" * 65)
    for phase, key in [('B','probing'),('D','training'),('E','transfer')]:
        status = '✓' if results.get(key) is not None else '✗'
        logger.info(f"  Phase {phase}: {status}")
    logger.info(f"  Log: {log_file}")
    return results


if __name__ == '__main__':
    main()
