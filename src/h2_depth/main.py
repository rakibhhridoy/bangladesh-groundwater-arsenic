"""
HYPOTHESIS 2: MAIN ANALYSIS ORCHESTRATOR
Depth-Stratified Geochemical Transitions and Process Partitioning of Arsenic Mobilization

Run this file to execute the complete H2 analysis pipeline (Phases A–G).

Usage:
    cd hypo2/
    python main.py
"""

# ── Mac-safe environment setup ─────────────────────────────────────────────
# Must happen BEFORE any library imports.

# 1. Force non-interactive backend: the default macOS 'macosx' backend
#    requires a GUI event loop and crashes / hangs when plt.subplots() or
#    sns.violinplot() is called from a terminal script.
import matplotlib
matplotlib.use('Agg')

# 2. Suppress tqdm progress bars emitted by pingouin's bootstrap CI
#    computation inside partial_corr() — otherwise Mechanism Shift takes
#    ~10 extra minutes of noisy spinner output.
import os
os.environ.setdefault('TQDM_DISABLE', '1')

import sys
import time
from pathlib import Path
from datetime import datetime

# ── Module imports ─────────────────────────────────────────────────────────
from data_loader              import load_and_prepare_data
from depth_zone_classification import run_phase_a
from redox_transition          import run_phase_b
from geogenic_proxy            import run_phase_c
from anthropogenic_proxy       import run_phase_d
from variance_partitioning     import run_phase_e
from mechanism_shift           import run_mechanism_shift
from shap_by_depth             import run_phase_f
from bayesian_multilevel       import run_phase_g1
from spatial_gwr               import run_phase_g2
from results_synthesis         import run_results_synthesis
from config                    import OUTPUT_DIR, RESULTS_DIR, TABLES_DIR, FIGURES_DIR, MAPS_DIR


# ── Helpers ────────────────────────────────────────────────────────────────

def banner(text):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def phase_done(label, t0):
    elapsed = time.time() - t0
    print(f"\n{'─'*80}")
    print(f"✓  {label} — {elapsed:.1f}s")
    print(f"{'─'*80}\n")


def phase_warn(label, err):
    print(f"\n{'─'*80}")
    print(f"⚠  {label} — WARNING: {err}")
    print(f"{'─'*80}\n")


# ── Pipeline ───────────────────────────────────────────────────────────────

def main():
    run_start = time.time()

    banner(
        "HYPOTHESIS 2: DEPTH-STRATIFIED GEOCHEMICAL TRANSITIONS\n"
        "  AND PROCESS PARTITIONING OF ARSENIC MOBILIZATION\n"
        "  Bangladesh Groundwater — n=1,807 samples\n"
        "\n"
        "  Sub-hypotheses:\n"
        "    H2.1  Redox conditions transition systematically with depth\n"
        "    H2.2  Anthropogenic variables dominate As variance in shallow zones\n"
        "    H2.3  Geogenic processes dominate As variance in deep zones\n"
        "    H2.4  Dominant mechanism shifts with depth (PO4 → Fe)\n"
        "\n"
        "  Phases A–G: Depth classification → Redox → Geogenic proxies\n"
        "              → Anthropogenic proxies → Variance partitioning\n"
        "              → Mechanism shift → SHAP → Bayesian → Spatial GWR"
    )

    all_results = {}

    # ── DATA PREPARATION ───────────────────────────────────────────────────
    banner("DATA PREPARATION")
    t0 = time.time()
    try:
        df = load_and_prepare_data()
        phase_done("Data Preparation", t0)
    except Exception as e:
        print(f"✗ FATAL: Data preparation failed: {e}")
        sys.exit(1)

    # ── PHASE A: DEPTH ZONE CLASSIFICATION ────────────────────────────────
    banner("PHASE A: DEPTH ZONE CLASSIFICATION & CHARACTERISATION")
    t0 = time.time()
    try:
        phase_a = run_phase_a(df)
        all_results['phase_a'] = phase_a
        phase_done("Phase A", t0)
    except Exception as e:
        phase_warn("Phase A", e)
        all_results['phase_a'] = {'error': str(e)}

    # ── PHASE B: REDOX TRANSITION ──────────────────────────────────────────
    banner("PHASE B: REDOX TRANSITION ANALYSIS (H2.1)")
    t0 = time.time()
    try:
        df, phase_b = run_phase_b(df)
        all_results['phase_b'] = phase_b
        phase_done("Phase B", t0)
    except Exception as e:
        phase_warn("Phase B", e)
        all_results['phase_b'] = {'error': str(e)}

    # ── PHASE C: GEOGENIC PROXIES ──────────────────────────────────────────
    banner("PHASE C: GEOGENIC PROXY CONSTRUCTION (H2.3)")
    t0 = time.time()
    try:
        df, phase_c = run_phase_c(df)
        all_results['phase_c'] = phase_c
        phase_done("Phase C", t0)
    except Exception as e:
        phase_warn("Phase C", e)
        all_results['phase_c'] = {'error': str(e)}

    # ── PHASE D: ANTHROPOGENIC PROXIES ────────────────────────────────────
    banner("PHASE D: ANTHROPOGENIC PROXY ANALYSIS (H2.2)")
    t0 = time.time()
    try:
        phase_d = run_phase_d(df)
        all_results['phase_d'] = phase_d
        phase_done("Phase D", t0)
    except Exception as e:
        phase_warn("Phase D", e)
        all_results['phase_d'] = {'error': str(e)}

    # ── PHASE E: VARIANCE PARTITIONING ────────────────────────────────────
    banner("PHASE E: VARIANCE PARTITIONING (H2.2 & H2.3)")
    t0 = time.time()
    try:
        vp_df, vp_ci = run_phase_e(df)
        all_results['phase_e'] = {'variance_partitioning': vp_df, 'ci': vp_ci}
        phase_done("Phase E", t0)
    except Exception as e:
        phase_warn("Phase E", e)
        all_results['phase_e'] = {'error': str(e)}

    # ── MECHANISM SHIFT ────────────────────────────────────────────────────
    banner("MECHANISM SHIFT ANALYSIS (H2.4)")
    t0 = time.time()
    try:
        mshift = run_mechanism_shift(df)
        all_results['mechanism_shift'] = mshift
        phase_done("Mechanism Shift", t0)
    except Exception as e:
        phase_warn("Mechanism Shift", e)
        all_results['mechanism_shift'] = {'error': str(e)}

    # ── PHASE F: SHAP ──────────────────────────────────────────────────────
    banner("PHASE F: SHAP VARIABLE IMPORTANCE BY DEPTH ZONE")
    t0 = time.time()
    try:
        phase_f = run_phase_f(df)
        all_results['phase_f'] = phase_f
        phase_done("Phase F", t0)
    except Exception as e:
        phase_warn("Phase F", e)
        all_results['phase_f'] = {'error': str(e)}

    # ── PHASE G-1: BAYESIAN MULTILEVEL ────────────────────────────────────
    banner("PHASE G-1: BAYESIAN MULTILEVEL MODEL")
    t0 = time.time()
    try:
        phase_g1 = run_phase_g1(df)
        all_results['phase_g1'] = phase_g1
        phase_done("Phase G-1", t0)
    except Exception as e:
        phase_warn("Phase G-1", e)
        all_results['phase_g1'] = {'error': str(e)}

    # ── PHASE G-2: SPATIAL GWR + KRIGING ──────────────────────────────────
    banner("PHASE G-2: SPATIAL ANALYSIS — GWR + KRIGING RISK MAPS")
    t0 = time.time()
    try:
        phase_g2 = run_phase_g2(df)
        all_results['phase_g2'] = phase_g2
        phase_done("Phase G-2", t0)
    except Exception as e:
        phase_warn("Phase G-2", e)
        all_results['phase_g2'] = {'error': str(e)}

    # ── RESULTS SYNTHESIS ─────────────────────────────────────────────────
    banner("RESULTS SYNTHESIS")
    t0 = time.time()
    try:
        verdict_df, summary_df = run_results_synthesis(df, all_results)
        all_results['synthesis'] = {'verdicts': verdict_df, 'summary': summary_df}
        phase_done("Results Synthesis", t0)
    except Exception as e:
        phase_warn("Results Synthesis", e)
        all_results['synthesis'] = {'error': str(e)}

    # ── FINAL REPORT ───────────────────────────────────────────────────────
    total_elapsed = time.time() - run_start
    banner("HYPOTHESIS 2 — ANALYSIS COMPLETE")

    print(f"  Total execution time : {total_elapsed:.1f}s  ({total_elapsed/60:.1f} min)")
    print(f"  Output directory     : {OUTPUT_DIR}")
    print(f"\n  Output summary:")
    print(f"    Tables  : {TABLES_DIR}")
    print(f"    Figures : {FIGURES_DIR}")
    print(f"    Maps    : {MAPS_DIR}")
    print(f"    Results : {RESULTS_DIR}")

    print(f"\n  Phase completion status:")
    phase_keys = ['phase_a','phase_b','phase_c','phase_d','phase_e',
                  'mechanism_shift','phase_f','phase_g1','phase_g2','synthesis']
    labels     = ['A — Depth Classification','B — Redox Transition',
                  'C — Geogenic Proxies','D — Anthropogenic Proxies',
                  'E — Variance Partitioning','Mechanism Shift',
                  'F — SHAP','G-1 — Bayesian','G-2 — Spatial GWR','Synthesis']
    for key, lbl in zip(phase_keys, labels):
        info = all_results.get(key, {})
        if isinstance(info, dict) and 'error' in info:
            status = f"⚠  WARNING ({info['error'][:50]})"
        else:
            status = "✓  COMPLETE"
        print(f"    {lbl:30s}: {status}")

    # Save log
    log_path = RESULTS_DIR / f'H2_analysis_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    with open(log_path, 'w') as lf:
        lf.write(f"Hypothesis 2 Analysis Log\nRun: {datetime.now()}\n")
        lf.write(f"Total time: {total_elapsed:.1f}s\n\n")
        for key, lbl in zip(phase_keys, labels):
            info = all_results.get(key, {})
            status = 'ERROR' if (isinstance(info, dict) and 'error' in info) else 'OK'
            lf.write(f"{lbl}: {status}\n")
    print(f"\n  Log saved → {log_path}")

    return all_results


if __name__ == "__main__":
    results = main()
