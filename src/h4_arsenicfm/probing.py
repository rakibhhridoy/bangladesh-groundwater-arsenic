"""
Linear Probing Experiments — ArseniCFM (H4)

Demonstrates the FM property: frozen Alpha Earth embeddings contain
rich geospatial information transferable to arsenic contamination tasks
WITHOUT any fine-tuning.

Experiments:
  1. Linear probe: embedding → WHO exceedance (binary)
  2. Linear probe: embedding → mechanism type (3-class)
  3. Linear probe: embedding → depth zone (4-class)
  4. Linear probe: embedding → log10(As) (regression)
  5. Comparison: random embeddings vs Alpha Earth (positive control)
  6. Feature importance: which of the 64 embedding dims are most predictive?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
from sklearn.dummy import DummyClassifier, DummyRegressor

from config import (
    TABLES_DIR, FIGURES_DIR, RANDOM_STATE, CV_FOLDS,
    MECHANISM_LABELS, DEPTH_ZONE_LABELS, WHO_AS_LIMIT, DPI
)


# ─── PROBE RUNNER ─────────────────────────────────────────────────────────────

def _linear_probe_cls(X, y, groups, label, n_splits=CV_FOLDS):
    """Logistic regression probe, spatial block CV. Returns mean AUC/Acc."""
    gkf = GroupKFold(n_splits=n_splits)
    scaler = StandardScaler()
    aucs, accs = [], []
    for tr, te in gkf.split(X, y, groups):
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(max_iter=500, random_state=RANDOM_STATE,
                                 C=1.0, solver='lbfgs', multi_class='auto')
        clf.fit(Xtr, y[tr])
        prob = clf.predict_proba(Xte)
        pred = clf.predict(Xte)
        accs.append(accuracy_score(y[te], pred))
        try:
            if prob.shape[1] == 2:
                aucs.append(roc_auc_score(y[te], prob[:, 1]))
            else:
                aucs.append(roc_auc_score(y[te], prob, multi_class='ovr'))
        except Exception:
            aucs.append(np.nan)
    return np.nanmean(aucs), np.nanmean(accs)


def _linear_probe_reg(X, y, groups, label, n_splits=CV_FOLDS):
    """Ridge regression probe, spatial block CV. Returns mean R²."""
    gkf = GroupKFold(n_splits=n_splits)
    scaler = StandardScaler()
    r2s = []
    for tr, te in gkf.split(X, y, groups):
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        reg = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        reg.fit(Xtr, y[tr])
        r2s.append(r2_score(y[te], reg.predict(Xte)))
    return float(np.mean(r2s))


def _dummy_cls(X, y, groups, n_splits=CV_FOLDS):
    gkf = GroupKFold(n_splits=n_splits)
    accs = []
    for tr, te in gkf.split(X, y, groups):
        d = DummyClassifier(strategy='most_frequent')
        d.fit(X[tr], y[tr])
        accs.append(accuracy_score(y[te], d.predict(X[te])))
    return float(np.mean(accs))


def _dummy_reg(X, y, groups, n_splits=CV_FOLDS):
    gkf = GroupKFold(n_splits=n_splits)
    r2s = []
    for tr, te in gkf.split(X, y, groups):
        d = DummyRegressor(strategy='mean')
        d.fit(X[tr], y[tr])
        r2s.append(r2_score(y[te], d.predict(X[te])))
    return float(np.mean(r2s))


# ─── FEATURE IMPORTANCE ───────────────────────────────────────────────────────

def _embedding_dim_importance(X_emb, y, groups, task='classification',
                               n_top=20, n_splits=3):
    """
    Evaluate each embedding dimension individually as a linear probe.
    Returns top-N most predictive dims and their scores.
    """
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for dim in range(X_emb.shape[1]):
        x_d = X_emb[:, dim:dim+1]
        fold_scores = []
        for tr, te in gkf.split(x_d, y, groups):
            if task == 'classification':
                clf = LogisticRegression(max_iter=200, C=1.0, random_state=RANDOM_STATE)
                clf.fit(x_d[tr], y[tr])
                fold_scores.append(accuracy_score(y[te], clf.predict(x_d[te])))
            else:
                reg = Ridge()
                reg.fit(x_d[tr], y[tr])
                fold_scores.append(r2_score(y[te], reg.predict(x_d[te])))
        scores.append(float(np.mean(fold_scores)))
    return np.array(scores)


# ─── MAIN RUNNER ──────────────────────────────────────────────────────────────

def run_probing(embeddings, y_exc, y_mech, y_zone, y_conc, groups):
    """
    Run all linear probing experiments on frozen Alpha Earth embeddings.
    Returns dict of results.
    """
    print("\n" + "█" * 65)
    print(" PHASE B: LINEAR PROBING EXPERIMENTS (H4)")
    print("█" * 65)

    X = embeddings.astype(np.float32)
    X_rand = np.random.RandomState(RANDOM_STATE).randn(*X.shape).astype(np.float32)

    rows = []

    # ── 1. WHO Exceedance ─────────────────────────────────────────────────
    print("\n  [1] Probing: WHO exceedance (binary)")
    auc_ae, acc_ae = _linear_probe_cls(X, y_exc, groups, 'WHO_Exc')
    auc_rnd, acc_rnd = _linear_probe_cls(X_rand, y_exc, groups, 'WHO_Exc_rand')
    dummy_acc = _dummy_cls(X, y_exc, groups)
    rows.append({'Task': 'WHO_Exceedance', 'Type': 'Binary',
                 'Alpha_Earth_AUC': round(auc_ae, 4), 'Alpha_Earth_Acc': round(acc_ae, 4),
                 'Random_Acc': round(acc_rnd, 4), 'Dummy_Acc': round(dummy_acc, 4),
                 'Lift_over_Dummy': round(acc_ae - dummy_acc, 4)})
    print(f"    Alpha Earth: AUC={auc_ae:.3f}  Acc={acc_ae:.3f}  "
          f"(random: {acc_rnd:.3f}  dummy: {dummy_acc:.3f})")

    # ── 2. Mechanism type ─────────────────────────────────────────────────
    print("\n  [2] Probing: Mechanism type (3-class)")
    auc_m, acc_m = _linear_probe_cls(X, y_mech, groups, 'Mechanism')
    _, acc_m_rnd = _linear_probe_cls(X_rand, y_mech, groups, 'Mechanism_rand')
    dummy_m = _dummy_cls(X, y_mech, groups)
    rows.append({'Task': 'Mechanism_Type', 'Type': '3-class',
                 'Alpha_Earth_AUC': round(auc_m, 4), 'Alpha_Earth_Acc': round(acc_m, 4),
                 'Random_Acc': round(acc_m_rnd, 4), 'Dummy_Acc': round(dummy_m, 4),
                 'Lift_over_Dummy': round(acc_m - dummy_m, 4)})
    print(f"    Alpha Earth: AUC={auc_m:.3f}  Acc={acc_m:.3f}  "
          f"(random: {acc_m_rnd:.3f}  dummy: {dummy_m:.3f})")

    # ── 3. Depth zone ─────────────────────────────────────────────────────
    print("\n  [3] Probing: Depth zone (4-class)")
    auc_z, acc_z = _linear_probe_cls(X, y_zone, groups, 'Depth_Zone')
    _, acc_z_rnd = _linear_probe_cls(X_rand, y_zone, groups, 'Depth_Zone_rand')
    dummy_z = _dummy_cls(X, y_zone, groups)
    rows.append({'Task': 'Depth_Zone', 'Type': '4-class',
                 'Alpha_Earth_AUC': round(auc_z, 4), 'Alpha_Earth_Acc': round(acc_z, 4),
                 'Random_Acc': round(acc_z_rnd, 4), 'Dummy_Acc': round(dummy_z, 4),
                 'Lift_over_Dummy': round(acc_z - dummy_z, 4)})
    print(f"    Alpha Earth: AUC={auc_z:.3f}  Acc={acc_z:.3f}  "
          f"(random: {acc_z_rnd:.3f}  dummy: {dummy_z:.3f})")

    # ── 4. As concentration ───────────────────────────────────────────────
    print("\n  [4] Probing: log10(As) regression")
    r2_ae = _linear_probe_reg(X, y_conc, groups, 'As_conc')
    r2_rnd = _linear_probe_reg(X_rand, y_conc, groups, 'As_conc_rand')
    dummy_r2 = _dummy_reg(X, y_conc, groups)
    rows.append({'Task': 'As_Concentration', 'Type': 'Regression',
                 'Alpha_Earth_R2': round(r2_ae, 4),
                 'Random_R2': round(r2_rnd, 4), 'Dummy_R2': round(dummy_r2, 4),
                 'Lift_over_Dummy': round(r2_ae - dummy_r2, 4)})
    print(f"    Alpha Earth: R²={r2_ae:.3f}  (random: {r2_rnd:.3f}  dummy: {dummy_r2:.3f})")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(TABLES_DIR / 'H4_T02_linear_probing.csv', index=False)
    print(f"\n  Saved: H4_T02_linear_probing.csv")

    # ── 5. Per-dimension importance ───────────────────────────────────────
    print("\n  [5] Embedding dimension importance for WHO exceedance...")
    dim_scores = _embedding_dim_importance(X, y_exc, groups, task='classification')
    top_idx = np.argsort(dim_scores)[::-1][:20]
    dim_df = pd.DataFrame({'dim': top_idx, 'accuracy': dim_scores[top_idx]})
    dim_df.to_csv(TABLES_DIR / 'H4_T03_embedding_dim_importance.csv', index=False)

    # ── Plots ─────────────────────────────────────────────────────────────
    _plot_probing_comparison(results_df)
    _plot_dim_importance(dim_scores)

    return {'probing': results_df, 'dim_importance': dim_scores}


def _plot_probing_comparison(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    tasks = df['Task'].tolist()
    x = np.arange(len(tasks))
    w = 0.25
    # Use Acc for classification, R2 for regression
    ae_vals  = df['Alpha_Earth_Acc'].fillna(df.get('Alpha_Earth_R2', pd.Series())).values
    rnd_vals = df['Random_Acc'].fillna(df.get('Random_R2', pd.Series())).values
    dum_vals = df['Dummy_Acc'].fillna(df.get('Dummy_R2', pd.Series())).values

    ax.bar(x - w, ae_vals,  w, label='Alpha Earth (frozen)', color='#2166ac', alpha=0.85)
    ax.bar(x,     rnd_vals, w, label='Random embedding',     color='#d73027', alpha=0.6)
    ax.bar(x + w, dum_vals, w, label='Dummy baseline',       color='#969696', alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(tasks, rotation=20, ha='right')
    ax.set_ylabel('Score (Acc / R²)')
    ax.set_title('H4: Linear Probing — Frozen Alpha Earth Embeddings\n'
                 '(no fine-tuning; spatial block CV)', fontsize=11)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H4_F02_linear_probing.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H4_F02_linear_probing.png")


def _plot_dim_importance(dim_scores):
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(dim_scores))
    colors = ['#2166ac' if s > np.percentile(dim_scores, 75) else '#969696'
              for s in dim_scores]
    ax.bar(x, dim_scores, color=colors, alpha=0.8)
    ax.set_xlabel('Embedding Dimension Index (0–63)')
    ax.set_ylabel('Linear Probe Accuracy (WHO exceedance)')
    ax.set_title('H4: Per-Dimension Predictive Power of Alpha Earth Embeddings\n'
                 '(Blue = top-25% most informative dimensions)', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H4_F03_dim_importance.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H4_F03_dim_importance.png")
