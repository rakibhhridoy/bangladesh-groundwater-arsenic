"""
Hypothesis 3 — Main Orchestrator
Seasonal Hydro-Climatic Forcing and Depth-Modulated Arsenic Mobilization

Run order:
  A — Entropy analysis        (entropy_analysis.py)
  B — Sequential mediation    (mediation_chain.py)
  C — Phase diagram / GMM     (phase_diagram.py)
  D — Stoichiometry           (stoichiometry.py)
  E — Mass balance            (mass_balance.py)
  F — Satellite forecast      (satellite_forecast.py)
  G — Spatial / seasonal      (spatial_seasonal.py)
  H — Structural equation     (sem_analysis.py)
  I — Results synthesis       (results_synthesis.py)

Usage:
    python main.py [--phases A B C ...]   # run specific phases
    python main.py                        # run all phases
"""

import sys
import argparse
import traceback
import logging
from datetime import datetime
from pathlib import Path

# ─── Setup logging BEFORE any module imports ──────────────────────────────────
LOG_DIR = Path(__file__).parent / 'output' / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f'H3_analysis_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ─── Module imports (lazy to avoid crash on missing optional deps) ─────────────
def _import_phase(name, module_name):
    try:
        mod = __import__(module_name)
        log.info(f"  ✓ Imported {module_name}")
        return mod
    except Exception as e:
        log.warning(f"  ✗ Could not import {module_name}: {e}")
        return None


def _run_phase(label, fn, *args, **kwargs):
    """Execute a phase function with error isolation."""
    log.info(f"\n{'='*70}")
    log.info(f"  STARTING PHASE {label}")
    log.info(f"{'='*70}")
    try:
        result = fn(*args, **kwargs)
        log.info(f"  ✓ Phase {label} COMPLETE")
        return result
    except Exception as e:
        log.error(f"  ✗ Phase {label} ERROR: {e}")
        log.error(traceback.format_exc())
        return None


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main(phases_to_run=None):
    log.info("=" * 70)
    log.info("  HYPOTHESIS 3 ANALYSIS — START")
    log.info(f"  Log: {LOG_FILE}")
    log.info("=" * 70)

    # ── 0. Data loading ────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("  PHASE 0: DATA LOADING")
    log.info("=" * 70)

    data_loader = _import_phase('data_loader', 'data_loader')
    if data_loader is None:
        log.error("Cannot load data — aborting.")
        sys.exit(1)

    try:
        df, df_dry, df_wet, df_delta = data_loader.load_and_prepare_data()
        log.info(f"  Dataset: {len(df):,} total | {len(df_dry):,} dry | "
                 f"{len(df_wet):,} wet | {len(df_delta):,} delta pairs")
    except Exception as e:
        log.error(f"Data loading failed: {e}")
        log.error(traceback.format_exc())
        sys.exit(1)

    # ── Phase registry ─────────────────────────────────────────────────────────
    all_phases = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    if phases_to_run is None:
        phases_to_run = all_phases

    phase_results = {}

    # ── A: Entropy ─────────────────────────────────────────────────────────────
    if 'A' in phases_to_run:
        mod = _import_phase('A', 'entropy_analysis')
        if mod:
            r = _run_phase('A (Entropy)', mod.run_entropy_analysis, df)
            if r:
                phase_results['entropy']     = r.get('shannon')
                phase_results['mutual_info'] = r.get('mutual_info')

    # ── B: Mediation chain ─────────────────────────────────────────────────────
    if 'B' in phases_to_run:
        mod = _import_phase('B', 'mediation_chain')
        if mod:
            r = _run_phase('B (Mediation)', mod.run_mediation_chain, df)
            phase_results['mediation'] = r

    # ── C: Phase diagram + GMM ─────────────────────────────────────────────────
    if 'C' in phases_to_run:
        mod = _import_phase('C', 'phase_diagram')
        if mod:
            r = _run_phase('C (Phase Diagram)', mod.run_phase_diagram, df, df_delta)
            if r:
                phase_results['gmm_clusters']    = r.get('gmm_clusters')
                phase_results['entropy_shallow'] = r.get('entropy_test')

    # ── D: Stoichiometry ───────────────────────────────────────────────────────
    if 'D' in phases_to_run:
        mod = _import_phase('D', 'stoichiometry')
        if mod:
            r = _run_phase('D (Stoichiometry)', mod.run_stoichiometry, df_delta)
            if r:
                phase_results['stoich_ratio'] = r.get('ratio')
                phase_results['evi_corr']     = r.get('evi_corr')
                # Update df_delta with R_displacement column
                if r.get('df_delta') is not None:
                    df_delta = r['df_delta']

    # ── E: Mass balance ────────────────────────────────────────────────────────
    if 'E' in phases_to_run:
        mod = _import_phase('E', 'mass_balance')
        if mod:
            r = _run_phase('E (Mass Balance)', mod.run_mass_balance,
                           df_delta, df_dry)
            if r:
                phase_results['mass_balance']   = r.get('mass_balance')
                phase_results['psm_comparison'] = r.get('propensity')

    # ── F: Satellite forecast ──────────────────────────────────────────────────
    if 'F' in phases_to_run:
        mod = _import_phase('F', 'satellite_forecast')
        if mod:
            r = _run_phase('F (Satellite Forecast)', mod.run_satellite_forecast,
                           df_delta)
            if r:
                phase_results['satellite_cv']       = r.get('cv_results')
                phase_results['satellite_shap']     = r.get('shap_df')
                phase_results['satellite_conformal']= r.get('conformal')
                phase_results['satellite_transfer'] = r.get('transfer')

    # ── G: Spatial / seasonal ──────────────────────────────────────────────────
    if 'G' in phases_to_run:
        mod = _import_phase('G', 'spatial_seasonal')
        if mod:
            r = _run_phase('G (Spatial Seasonal)', mod.run_spatial_seasonal,
                           df, df_delta)
            if r:
                phase_results['moran']       = r.get('moran')
                phase_results['risk_matrix'] = r.get('risk_matrix')
                phase_results['gwr']         = r.get('gwr')

    # ── H: SEM ─────────────────────────────────────────────────────────────────
    if 'H' in phases_to_run:
        mod = _import_phase('H', 'sem_analysis')
        if mod:
            r = _run_phase('H (SEM)', mod.run_sem_analysis, df)
            if r:
                phase_results['sem_paths']     = r.get('paths')
                phase_results['sem_fit']       = r.get('fit')
                phase_results['sem_dominance'] = r.get('dominance')

    # ── I: Synthesis ───────────────────────────────────────────────────────────
    if 'I' in phases_to_run:
        mod = _import_phase('I', 'results_synthesis')
        if mod:
            r = _run_phase('I (Synthesis)', mod.run_results_synthesis,
                           df, phase_results)
            if r:
                phase_results['verdict']    = r.get('verdict')
                phase_results['risk_maps']  = r.get('risk_maps')

    # ── Final summary ──────────────────────────────────────────────────────────
    log.info("\n" + "█" * 70)
    log.info("  HYPOTHESIS 3 ANALYSIS — COMPLETE")
    log.info("█" * 70)
    log.info(f"  Phases run: {', '.join(phases_to_run)}")
    log.info(f"  Log saved : {LOG_FILE}")

    completed = [k for k, v in phase_results.items() if v is not None]
    skipped   = [k for k, v in phase_results.items() if v is None]
    log.info(f"  Results populated : {completed}")
    if skipped:
        log.info(f"  Results missing   : {skipped}")

    return phase_results


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run Hypothesis 3 analysis for Bangladesh groundwater arsenic'
    )
    parser.add_argument(
        '--phases', nargs='+', default=None,
        metavar='PHASE',
        help='Phases to run (A B C D E F G H I). Default: all.'
    )
    args = parser.parse_args()

    phase_list = [p.upper() for p in args.phases] if args.phases else None
    main(phases_to_run=phase_list)
