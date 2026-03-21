"""
ArseniCFM Trainer
Spatial block CV, multi-task training, ablation studies, SHAP explainability.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    roc_auc_score, accuracy_score, f1_score
)
from sklearn.preprocessing import StandardScaler

from config import (
    BATCH_SIZE, LEARNING_RATE, N_EPOCHS, PATIENCE, WEIGHT_DECAY,
    CV_FOLDS, RANDOM_STATE, TABLES_DIR, FIGURES_DIR, MODELS_DIR,
    TASK_WEIGHTS, DPI, N_JOBS, WHO_AS_LIMIT
)
from arsenicfm import build_model, ArseniCFMLoss


# ─── SKLEARN-COMPATIBLE CV ────────────────────────────────────────────────────

def train_eval_sklearn(X_sat, X_geo, X_ctx, y_conc, y_exc, y_mech, y_zone,
                       groups, n_splits=CV_FOLDS):
    """Sklearn fallback: GBM multi-task, spatial block CV."""
    from arsenicfm import ArseniCFM_SklearnFallback
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    for fold, (tr, te) in enumerate(gkf.split(X_sat, y_conc, groups)):
        model = ArseniCFM_SklearnFallback()
        model.fit(X_sat[tr], X_geo[tr], X_ctx[tr],
                  y_conc[tr], y_exc[tr], y_mech[tr], y_zone[tr])
        preds = model.predict(X_sat[te], X_geo[te], X_ctx[te])
        metrics = _compute_metrics(preds, y_conc[te], y_exc[te], y_mech[te], y_zone[te])
        metrics['fold'] = fold + 1
        fold_metrics.append(metrics)
        print(f"  Fold {fold+1}: R²={metrics['r2_conc']:.3f}  AUC={metrics['auc_exc']:.3f}")
    return fold_metrics, model


# ─── PYTORCH TRAINING ─────────────────────────────────────────────────────────

def _make_tensors(*arrays):
    return tuple(torch.tensor(a) for a in arrays)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x_sat, x_geo, x_ctx, yc, ye, ym, yz in loader:
        x_sat, x_geo, x_ctx = x_sat.to(device), x_geo.to(device), x_ctx.to(device)
        yc, ye, ym, yz = yc.to(device), ye.to(device), ym.to(device), yz.to(device)
        optimizer.zero_grad()
        preds = model(x_sat, x_geo, x_ctx)
        targets = {'concentration': yc, 'exceedance': ye, 'mechanism': ym, 'depth_zone': yz}
        loss, _ = criterion(preds, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_fold(model, loader, device):
    model.eval()
    pred_conc, pred_exc, pred_mech, pred_zone = [], [], [], []
    for x_sat, x_geo, x_ctx, *_ in loader:
        p = model(x_sat.to(device), x_geo.to(device), x_ctx.to(device))
        pred_conc.append(p['concentration'].cpu().numpy())
        pred_exc.append(p['exceedance'].cpu().numpy())
        pred_mech.append(p['mechanism'].argmax(1).cpu().numpy())
        pred_zone.append(p['depth_zone'].argmax(1).cpu().numpy())
    return {
        'concentration': np.concatenate(pred_conc),
        'exceedance':    np.concatenate(pred_exc),
        'mechanism':     np.concatenate(pred_mech),
        'depth_zone':    np.concatenate(pred_zone),
    }


def train_eval_pytorch(X_sat, X_geo, X_ctx, y_conc, y_exc, y_mech, y_zone,
                       groups, n_splits=CV_FOLDS, freeze_satellite=True):
    """PyTorch ArseniCFM with spatial block CV."""
    device = torch.device('cpu')  # Mac-safe (MPS optional)
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    best_model_state = None
    best_r2 = -np.inf

    scaler_geo = StandardScaler()
    scaler_ctx = StandardScaler()
    scaler_sat = StandardScaler()

    X_sat_s = scaler_sat.fit_transform(X_sat)
    X_geo_s = scaler_geo.fit_transform(X_geo)
    X_ctx_s = scaler_ctx.fit_transform(X_ctx)

    for fold, (tr, te) in enumerate(gkf.split(X_sat_s, y_conc, groups)):
        print(f"\n  ── Fold {fold+1}/{n_splits} ──")
        model = build_model(X_geo.shape[1], X_ctx.shape[1], freeze_satellite).to(device)
        criterion = ArseniCFMLoss().to(device)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPOCHS)

        # Datasets
        Xst, Xgt, Xct = (torch.tensor(X_sat_s[tr], dtype=torch.float32),
                          torch.tensor(X_geo_s[tr], dtype=torch.float32),
                          torch.tensor(X_ctx_s[tr], dtype=torch.float32))
        yt = (torch.tensor(y_conc[tr], dtype=torch.float32),
              torch.tensor(y_exc[tr],  dtype=torch.float32),
              torch.tensor(y_mech[tr], dtype=torch.long),
              torch.tensor(y_zone[tr], dtype=torch.long))

        tr_loader = DataLoader(TensorDataset(Xst, Xgt, Xct, *yt),
                               batch_size=BATCH_SIZE, shuffle=True)
        # Test set
        Xste = torch.tensor(X_sat_s[te], dtype=torch.float32)
        Xgte = torch.tensor(X_geo_s[te], dtype=torch.float32)
        Xcte = torch.tensor(X_ctx_s[te], dtype=torch.float32)
        yte  = (torch.tensor(y_conc[te], dtype=torch.float32),
                torch.tensor(y_exc[te],  dtype=torch.float32),
                torch.tensor(y_mech[te], dtype=torch.long),
                torch.tensor(y_zone[te], dtype=torch.long))
        te_loader = DataLoader(TensorDataset(Xste, Xgte, Xcte, *yte),
                               batch_size=BATCH_SIZE)

        best_val = np.inf
        patience_count = 0
        train_losses = []

        for epoch in range(N_EPOCHS):
            loss = train_epoch(model, tr_loader, optimizer, criterion, device)
            train_losses.append(loss)
            scheduler.step()
            if loss < best_val:
                best_val = loss
                patience_count = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_count += 1
            if patience_count >= PATIENCE:
                print(f"    Early stop at epoch {epoch+1}")
                break
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1:3d}  loss={loss:.4f}")

        model.load_state_dict(best_state)
        preds = eval_fold(model, te_loader, device)
        metrics = _compute_metrics(preds, y_conc[te], y_exc[te], y_mech[te], y_zone[te])
        metrics['fold'] = fold + 1
        metrics['best_val_loss'] = best_val
        fold_metrics.append(metrics)
        print(f"  Fold {fold+1}: R²={metrics['r2_conc']:.3f}  AUC={metrics['auc_exc']:.3f}  "
              f"mech_acc={metrics['acc_mech']:.3f}")

        if metrics['r2_conc'] > best_r2:
            best_r2 = metrics['r2_conc']
            best_model_state = best_state

    # Save best model
    if best_model_state is not None:
        torch.save(best_model_state, MODELS_DIR / 'arsenicfm_best.pt')
        print(f"\n  Best model saved: R²={best_r2:.3f}")

    return fold_metrics, {'scaler_geo': scaler_geo, 'scaler_ctx': scaler_ctx,
                          'scaler_sat': scaler_sat, 'state': best_model_state}


# ─── METRICS ──────────────────────────────────────────────────────────────────

def _compute_metrics(preds, y_conc, y_exc, y_mech, y_zone):
    metrics = {}
    # Concentration
    p_c = preds['concentration']
    metrics['r2_conc']   = float(r2_score(y_conc, p_c))
    metrics['mae_conc']  = float(mean_absolute_error(y_conc, p_c))
    metrics['rmse_conc'] = float(np.sqrt(mean_squared_error(y_conc, p_c)))
    # Back-transform: 10^log10
    y_as_true = 10 ** y_conc
    y_as_pred = 10 ** np.clip(p_c, -3, 5)
    metrics['r2_as_raw'] = float(r2_score(y_as_true, y_as_pred))
    metrics['mae_as_raw'] = float(mean_absolute_error(y_as_true, y_as_pred))
    # Exceedance
    p_e = preds['exceedance']
    try:
        metrics['auc_exc'] = float(roc_auc_score(y_exc, p_e))
    except Exception:
        metrics['auc_exc'] = np.nan
    metrics['acc_exc'] = float(accuracy_score(y_exc, (p_e > 0.5).astype(int)))
    # Mechanism
    p_m = preds['mechanism']
    metrics['acc_mech'] = float(accuracy_score(y_mech, p_m))
    try:
        metrics['f1_mech'] = float(f1_score(y_mech, p_m, average='weighted'))
    except Exception:
        metrics['f1_mech'] = np.nan
    # Depth zone
    p_z = preds['depth_zone']
    metrics['acc_zone'] = float(accuracy_score(y_zone, p_z))
    return metrics


# ─── ABLATION STUDY ───────────────────────────────────────────────────────────

def run_ablation(X_sat, X_geo, X_ctx, y_conc, y_exc, y_mech, y_zone, groups):
    """
    Ablation: test contribution of each modality.
    Variants:
      full          = sat + geo + ctx
      no_sat        = zeros for sat
      no_geo        = zeros for geo
      no_ctx        = zeros for ctx
      geo_only      = no sat, no ctx
    """
    print("\n  Running ablation study...")
    ablations = {
        'full':      (X_sat,                        X_geo,                   X_ctx),
        'no_sat':    (np.zeros_like(X_sat),          X_geo,                   X_ctx),
        'no_geo':    (X_sat,                         np.zeros_like(X_geo),    X_ctx),
        'no_ctx':    (X_sat,                         X_geo,                   np.zeros_like(X_ctx)),
        'geo_only':  (np.zeros_like(X_sat),          X_geo,                   np.zeros_like(X_ctx)),
        'sat_only':  (X_sat,                         np.zeros_like(X_geo),    np.zeros_like(X_ctx)),
    }
    rows = []
    for name, (xs, xg, xc) in ablations.items():
        print(f"  Ablation: {name}")
        if HAS_TORCH:
            folds, _ = train_eval_pytorch(xs, xg, xc, y_conc, y_exc, y_mech, y_zone,
                                          groups, n_splits=3)
        else:
            folds, _ = train_eval_sklearn(xs, xg, xc, y_conc, y_exc, y_mech, y_zone,
                                          groups, n_splits=3)
        mean_r2  = np.mean([f['r2_conc'] for f in folds])
        mean_auc = np.mean([f['auc_exc'] for f in folds])
        rows.append({'Ablation': name, 'R2_log10As': round(mean_r2, 4),
                     'AUC_WHO': round(mean_auc, 4)})
        print(f"    {name}: R²={mean_r2:.3f}  AUC={mean_auc:.3f}")
    abl_df = pd.DataFrame(rows)
    abl_df.to_csv(TABLES_DIR / 'H4_T05_ablation_study.csv', index=False)
    print(f"  Saved: H4_T05_ablation_study.csv")
    return abl_df


# ─── MAIN RUNNER ──────────────────────────────────────────────────────────────

def run_training(X_sat, X_geo, X_ctx, y_conc, y_exc, y_mech, y_zone, groups,
                 run_ablation_study=True):
    """
    Full training pipeline: spatial block CV + ablation.
    Returns fold_metrics, model_artifacts, ablation_df.
    """
    print("\n" + "█" * 65)
    print(" ARSENICFM TRAINING (H4)")
    print("█" * 65)

    if HAS_TORCH:
        fold_metrics, artifacts = train_eval_pytorch(
            X_sat, X_geo, X_ctx, y_conc, y_exc, y_mech, y_zone, groups)
    else:
        fold_metrics, artifacts = train_eval_sklearn(
            X_sat, X_geo, X_ctx, y_conc, y_exc, y_mech, y_zone, groups)

    # Summary table
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(TABLES_DIR / 'H4_T01_cv_metrics.csv', index=False)
    print(f"\n  CV Summary:")
    for col in ['r2_conc', 'mae_conc', 'auc_exc', 'acc_mech', 'acc_zone']:
        if col in metrics_df:
            print(f"    {col}: {metrics_df[col].mean():.3f} ± {metrics_df[col].std():.3f}")

    # Ablation
    abl_df = None
    if run_ablation_study:
        abl_df = run_ablation(X_sat, X_geo, X_ctx, y_conc, y_exc, y_mech, y_zone, groups)

    # Plot CV metrics
    _plot_cv_metrics(metrics_df)

    return fold_metrics, artifacts, abl_df


def _plot_cv_metrics(metrics_df):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    metrics = [('r2_conc', 'R² (log₁₀As)', 'R²'),
               ('auc_exc', 'AUC (WHO >', 'AUC'),
               ('acc_mech', 'Accuracy (Mechanism)', 'Acc'),
               ('acc_zone', 'Accuracy (Depth Zone)', 'Acc')]
    for ax, (col, title, ylabel) in zip(axes, metrics):
        if col not in metrics_df.columns:
            continue
        vals = metrics_df[col].values
        ax.bar(range(1, len(vals)+1), vals, color='#4575b4', alpha=0.8)
        ax.axhline(vals.mean(), color='red', ls='--', lw=1.5,
                   label=f'Mean={vals.mean():.3f}')
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Fold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('ArseniCFM: Spatial Block CV Performance (5 folds)', fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H4_F01_cv_metrics.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H4_F01_cv_metrics.png")
