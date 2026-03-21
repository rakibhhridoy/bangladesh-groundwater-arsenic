"""
FM Screening Experiment — ArseniCFM H4

Demonstrates the key foundation model novelty:
  At a held-out physiographic zone with NO lab data,
  can satellite + context alone predict WHO exceedance?

Protocol:
  LOZO-CV (Leave-One-Zone-Out):
    Train on 6 zones → Test on 7th
    Scenario A: Full model  (sat + geo + ctx)  ← requires lab data
    Scenario B: Sat-only    (sat + ctx)         ← NO lab data needed
    Scenario C: Ctx-only    (lat/lon/depth/season) ← spatial baseline
    Scenario D: Random      (50% coin flip)         ← null baseline

Output:
  output/tables/H4_T15_fm_screening_experiment.csv
  output/figures/H4_F10_fm_screening.png
"""

import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    INTEGRATED_FILE, EMBEDDINGS_DIR, TABLES_DIR, FIGURES_DIR,
    GEO_FEATURE_COLS, CONTEXT_COLS, PHYSIOGRAPHIC_ZONES,
    WHO_AS_LIMIT, RANDOM_STATE, N_JOBS
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.dummy import DummyClassifier


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def assign_phys_zone(lat, lon):
    for zone, bounds in PHYSIOGRAPHIC_ZONES.items():
        if (bounds['lat'][0] <= lat <= bounds['lat'][1] and
                bounds['lon'][0] <= lon <= bounds['lon'][1]):
            return zone
    return 'Unclassified'


def load_data():
    df = pd.read_csv(INTEGRATED_FILE)
    if 'Location_ID' not in df.columns:
        df['Location_ID'] = np.arange(1, len(df) + 1)
    df['As'] = df['As'].round(2)
    df['Exceedance'] = (df['As'] > WHO_AS_LIMIT).astype(int)
    df['Season_bin'] = (df['Season'] == 'Wet').astype(int)
    df['Phys_Zone'] = df.apply(
        lambda r: assign_phys_zone(r['Latitude'], r['Longitude']), axis=1)

    for col in GEO_FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0.0

    return df


def load_embeddings(df):
    cache_emb = EMBEDDINGS_DIR / 'multiscale_pca64_embeddings.npy'
    cache_ids = EMBEDDINGS_DIR / 'multiscale_location_ids.csv'

    # Prefer multi-scale; fall back to single-point GEE; then statistical
    for emb_path, ids_path in [
        (cache_emb, cache_ids),
        (EMBEDDINGS_DIR / 'alphaearth_gee_embeddings.npy',
         EMBEDDINGS_DIR / 'alphaearth_gee_location_ids.csv'),
        (EMBEDDINGS_DIR / 'sentinel_statistical_embeddings.npy',
         EMBEDDINGS_DIR / 'sentinel_location_ids.csv'),
    ]:
        if emb_path.exists() and ids_path.exists():
            emb   = np.load(emb_path).astype(np.float32)
            lids  = pd.read_csv(ids_path).iloc[:, 0].values
            print(f"  Embeddings: {emb_path.name}  {emb.shape}")
            break
    else:
        print("  WARNING: no embeddings found — using zeros")
        emb  = np.zeros((len(df), 64), dtype=np.float32)
        lids = df['Location_ID'].values

    # Align to df rows
    id_to_emb = {int(lid): emb[i] for i, lid in enumerate(lids)}
    dim = emb.shape[1]
    X_sat = np.stack([
        id_to_emb.get(int(lid), np.zeros(dim))
        for lid in df['Location_ID']
    ]).astype(np.float32)
    return X_sat


# ─── MAIN EXPERIMENT ──────────────────────────────────────────────────────────

def run_screening():
    print("=" * 65)
    print("FM SCREENING EXPERIMENT — Leave-One-Zone-Out")
    print("=" * 65)

    df       = load_data()
    X_sat    = load_embeddings(df)
    geo_cols = [c for c in GEO_FEATURE_COLS if c in df.columns]
    ctx_cols = [c for c in CONTEXT_COLS     if c in df.columns]

    X_geo  = df[geo_cols].values.astype(np.float32)
    X_ctx  = df[ctx_cols].values.astype(np.float32)
    y_exc  = df['Exceedance'].values

    # Build feature matrices for each scenario
    X_full = np.concatenate([X_sat, X_geo, X_ctx], axis=1)   # requires lab data
    X_satc = np.concatenate([X_sat, X_ctx],         axis=1)   # sat + context only
    X_ctxo = X_ctx                                            # context only

    phys_zones   = df['Phys_Zone'].values
    unique_zones = [z for z in PHYSIOGRAPHIC_ZONES.keys()
                    if (phys_zones == z).sum() >= 20]         # need ≥20 test pts

    print(f"\n  Zones with ≥20 samples: {unique_zones}")
    print(f"  Exceedance prevalence: {y_exc.mean()*100:.1f}%\n")

    records = []

    for zone in unique_zones:
        te = phys_zones == zone
        tr = ~te
        n_tr = tr.sum(); n_te = te.sum()
        prev_te = y_exc[te].mean()

        print(f"  Zone: {zone:<28s}  tr={n_tr}  te={n_te}  "
              f"exc_te={prev_te*100:.0f}%")

        # Skip zones where test set is all-one-class (AUC undefined)
        if len(np.unique(y_exc[te])) < 2:
            print(f"    SKIP (only one class in test set)")
            continue

        sc_full = StandardScaler().fit(X_full[tr])
        sc_satc = StandardScaler().fit(X_satc[tr])
        sc_ctxo = StandardScaler().fit(X_ctxo[tr])

        row = {'Zone': zone, 'n_train': n_tr, 'n_test': n_te,
               'prevalence_test': round(float(prev_te), 3)}

        # ── Scenario A: Full (sat + geo + ctx) ────────────────────────────
        clf_A = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=RANDOM_STATE)
        clf_A.fit(sc_full.transform(X_full[tr]), y_exc[tr])
        prob_A = clf_A.predict_proba(sc_full.transform(X_full[te]))[:, 1]
        row['AUC_full']    = round(float(roc_auc_score(y_exc[te], prob_A)), 4)
        row['AP_full']     = round(float(average_precision_score(y_exc[te], prob_A)), 4)

        # ── Scenario B: Sat + ctx only (NO lab data) ──────────────────────
        clf_B = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=RANDOM_STATE)
        clf_B.fit(sc_satc.transform(X_satc[tr]), y_exc[tr])
        prob_B = clf_B.predict_proba(sc_satc.transform(X_satc[te]))[:, 1]
        row['AUC_sat_only']= round(float(roc_auc_score(y_exc[te], prob_B)), 4)
        row['AP_sat_only'] = round(float(average_precision_score(y_exc[te], prob_B)), 4)

        # ── Scenario C: Context only (lat/lon/depth/season) ───────────────
        clf_C = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=RANDOM_STATE)
        clf_C.fit(sc_ctxo.transform(X_ctxo[tr]), y_exc[tr])
        prob_C = clf_C.predict_proba(sc_ctxo.transform(X_ctxo[te]))[:, 1]
        row['AUC_ctx_only']= round(float(roc_auc_score(y_exc[te], prob_C)), 4)
        row['AP_ctx_only'] = round(float(average_precision_score(y_exc[te], prob_C)), 4)

        # ── Scenario D: Random (stratified dummy) ─────────────────────────
        rng = np.random.RandomState(RANDOM_STATE)
        prob_D = rng.uniform(0, 1, size=n_te)
        row['AUC_random']  = round(float(roc_auc_score(y_exc[te], prob_D)), 4)
        row['AP_random']   = round(float(average_precision_score(y_exc[te], prob_D)), 4)

        # Satellite lift over context-only
        row['AUC_sat_lift'] = round(row['AUC_sat_only'] - row['AUC_ctx_only'], 4)

        records.append(row)
        print(f"    AUC  full={row['AUC_full']:.3f}  "
              f"sat={row['AUC_sat_only']:.3f}  "
              f"ctx={row['AUC_ctx_only']:.3f}  "
              f"rand={row['AUC_random']:.3f}  "
              f"lift={row['AUC_sat_lift']:+.3f}")

    results = pd.DataFrame(records)

    # ── Summary row ───────────────────────────────────────────────────────
    auc_cols = ['AUC_full', 'AUC_sat_only', 'AUC_ctx_only', 'AUC_random', 'AUC_sat_lift']
    mean_row = {'Zone': 'MEAN'}
    for c in auc_cols:
        mean_row[c] = round(float(results[c].mean()), 4)
    mean_row['AP_full']    = round(float(results['AP_full'].mean()), 4)
    mean_row['AP_sat_only']= round(float(results['AP_sat_only'].mean()), 4)
    mean_row['n_train']    = int(results['n_train'].mean())
    mean_row['n_test']     = int(results['n_test'].mean())
    mean_row['prevalence_test'] = round(float(results['prevalence_test'].mean()), 3)

    results = pd.concat([results, pd.DataFrame([mean_row])], ignore_index=True)

    # ── Save table ────────────────────────────────────────────────────────
    out_csv = TABLES_DIR / 'H4_T15_fm_screening_experiment.csv'
    results.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FM SCREENING SUMMARY (LOZO-CV)")
    print("=" * 65)
    mr = results[results['Zone'] == 'MEAN'].iloc[0]
    print(f"  Full model (sat+geo+ctx)  : AUC = {mr['AUC_full']:.3f}")
    print(f"  Sat-only  (NO lab data)   : AUC = {mr['AUC_sat_only']:.3f}  ← FM SCREENING CLAIM")
    print(f"  Context-only (lat/lon)    : AUC = {mr['AUC_ctx_only']:.3f}")
    print(f"  Random baseline           : AUC = {mr['AUC_random']:.3f}")
    print(f"  Satellite lift over ctx   : ΔAUC = {mr['AUC_sat_lift']:+.3f}")
    print("=" * 65)

    # ── Figure ────────────────────────────────────────────────────────────
    _plot_screening(results)

    return results


def _plot_screening(results):
    data = results[results['Zone'] != 'MEAN'].copy()
    zones = data['Zone'].tolist()
    n = len(zones)
    x = np.arange(n)
    w = 0.20

    colors = {
        'Full (sat+geo+ctx)': '#2166ac',
        'Sat-only (no lab)':  '#f46d43',
        'Context-only':       '#a6d96a',
        'Random':             '#d9d9d9',
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - 1.5*w, data['AUC_full'],     w, label='Full (sat+geo+ctx)',
           color=colors['Full (sat+geo+ctx)'], alpha=0.9, edgecolor='k', lw=0.5)
    ax.bar(x - 0.5*w, data['AUC_sat_only'], w, label='Sat-only (no lab)',
           color=colors['Sat-only (no lab)'],  alpha=0.9, edgecolor='k', lw=0.5)
    ax.bar(x + 0.5*w, data['AUC_ctx_only'], w, label='Context-only',
           color=colors['Context-only'],       alpha=0.9, edgecolor='k', lw=0.5)
    ax.bar(x + 1.5*w, data['AUC_random'],   w, label='Random baseline',
           color=colors['Random'],            alpha=0.9, edgecolor='k', lw=0.5)

    ax.axhline(0.5, color='k', lw=1, ls='--', alpha=0.4)

    # Annotate sat-only bars
    for i, (_, row) in enumerate(data.iterrows()):
        ax.text(i - 0.5*w, row['AUC_sat_only'] + 0.005,
                f"{row['AUC_sat_only']:.2f}", ha='center', va='bottom',
                fontsize=7.5, color='#f46d43', fontweight='bold')

    short_labels = [z.replace('_', '\n') for z in zones]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_ylabel('ROC-AUC (WHO exceedance)', fontsize=11)
    ax.set_ylim(0.3, 1.02)
    ax.set_title(
        'Foundation Model Screening Experiment\n'
        'LOZO-CV: WHO exceedance prediction at held-out zones\n'
        '(Orange = satellite-only, no lab geochemistry required)',
        fontsize=11, pad=8)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    # Mean AUC annotations
    mr = results[results['Zone'] == 'MEAN'].iloc[0]
    txt = (f"Mean AUC\n"
           f"Full: {mr['AUC_full']:.3f}\n"
           f"Sat:  {mr['AUC_sat_only']:.3f}\n"
           f"Ctx:  {mr['AUC_ctx_only']:.3f}")
    ax.text(0.01, 0.98, txt, transform=ax.transAxes,
            va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    out_fig = FIGURES_DIR / 'H4_F10_fm_screening.png'
    plt.savefig(out_fig, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved figure: {out_fig}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    run_screening()
