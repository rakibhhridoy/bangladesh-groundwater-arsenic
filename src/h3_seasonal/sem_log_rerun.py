"""
H3 SEM Re-run: Log-transformed mediators
Fixes the mixed-scale problem: previously TDS, ORP, Fe, PO4 were in
original units while As was log10-transformed, creating scale artifacts.

This version log-transforms all positive mediators and handles ORP
(can be negative) by shift-log: log(ORP + |min| + 1).

Outputs:
  tables/H3_T_sem_log_corrected.csv
  tables/H3_T_sem_log_indirect_effects.csv
  figures/H3_F_sem_log_comparison.png
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (TARGET, FIGURES_DIR, TABLES_DIR, DPI,
                    DEPTH_ZONE_LABELS)

DATA = Path(__file__).parent.parent / 'hypo1' / 'output' / 'bangladesh_groundwater_arsenic_1807samples.csv'

ZONES_TO_RUN = ['ALL'] + DEPTH_ZONE_LABELS


def log_transform_mediators(df):
    """
    Apply consistent log10 transform to all mediators.
    ORP can be negative: use shift log10(ORP - min(ORP) + 1).
    """
    df = df.copy()

    # As: already log10
    df['As_log'] = np.log10(df[TARGET].clip(lower=0.01))

    # TDS: strictly positive
    if 'TDS' in df.columns:
        df['TDS_log'] = np.log10(df['TDS'].clip(lower=0.1))

    # ORP: can be negative — shift to make all positive
    if 'ORP' in df.columns:
        shift = abs(df['ORP'].min()) + 1
        df['ORP_log'] = np.log10(df['ORP'] + shift)

    # Fe2+: clip to >0
    if 'Fe2+' in df.columns:
        df['Fe_log'] = np.log10(df['Fe2+'].clip(lower=0.001))

    # PO4: clip to >0
    if 'PO43-' in df.columns:
        df['PO4_log'] = np.log10(df['PO43-'].clip(lower=0.001))

    # Cl: proxy for dilution/concentration
    if 'Cl-' in df.columns:
        df['Cl_log'] = np.log10(df['Cl-'].clip(lower=0.1))

    return df


def _beta(formula, predictor, data):
    """OLS standardised path coefficient."""
    data_clean = data.dropna(subset=[s.strip() for s in formula.replace('~', '+').split('+')])
    if len(data_clean) < 10:
        return np.nan, np.nan, np.nan
    try:
        model = smf.ols(formula, data=data_clean).fit()
        b = model.params.get(predictor, np.nan)
        p = model.pvalues.get(predictor, np.nan)
        se = model.bse.get(predictor, np.nan)
        return b, p, se
    except Exception:
        return np.nan, np.nan, np.nan


def fit_sem_ols_log(data, zone_label):
    """
    OLS path decomposition on log-transformed variables.
    Pathways:
      Reductive:  Season -> TDS_log -> ORP_log -> Fe_log -> As_log
      Desorption: Season -> PO4_log -> As_log
      Dilution:   Season -> Cl_log  -> As_log
    """
    rows = []

    # ── Reductive pathway ─────────────────────────────────────────────────
    if all(c in data.columns for c in ['Season_bin', 'TDS_log']):
        b, p, se = _beta('TDS_log ~ Season_bin', 'Season_bin', data)
        rows.append({'pathway': 'Reductive', 'from': 'Season', 'to': 'TDS',
                     'beta': b, 'p': p, 'se': se})

    if all(c in data.columns for c in ['TDS_log', 'ORP_log']):
        b, p, se = _beta('ORP_log ~ TDS_log', 'TDS_log', data)
        rows.append({'pathway': 'Reductive', 'from': 'TDS', 'to': 'ORP',
                     'beta': b, 'p': p, 'se': se})

    if all(c in data.columns for c in ['ORP_log', 'Fe_log']):
        b, p, se = _beta('Fe_log ~ ORP_log', 'ORP_log', data)
        rows.append({'pathway': 'Reductive', 'from': 'ORP', 'to': 'Fe',
                     'beta': b, 'p': p, 'se': se})

    if all(c in data.columns for c in ['Fe_log', 'As_log']):
        b, p, se = _beta('As_log ~ Fe_log', 'Fe_log', data)
        rows.append({'pathway': 'Reductive', 'from': 'Fe', 'to': 'As',
                     'beta': b, 'p': p, 'se': se})

    # ── Desorption pathway ────────────────────────────────────────────────
    if all(c in data.columns for c in ['Season_bin', 'PO4_log']):
        b, p, se = _beta('PO4_log ~ Season_bin', 'Season_bin', data)
        rows.append({'pathway': 'Desorption', 'from': 'Season', 'to': 'PO4',
                     'beta': b, 'p': p, 'se': se})

    if all(c in data.columns for c in ['PO4_log', 'As_log']):
        b, p, se = _beta('As_log ~ PO4_log', 'PO4_log', data)
        rows.append({'pathway': 'Desorption', 'from': 'PO4', 'to': 'As',
                     'beta': b, 'p': p, 'se': se})

    # ── Dilution pathway ──────────────────────────────────────────────────
    if all(c in data.columns for c in ['Season_bin', 'Cl_log']):
        b, p, se = _beta('Cl_log ~ Season_bin', 'Season_bin', data)
        rows.append({'pathway': 'Dilution', 'from': 'Season', 'to': 'Cl',
                     'beta': b, 'p': p, 'se': se})

    if all(c in data.columns for c in ['Cl_log', 'As_log']):
        b, p, se = _beta('As_log ~ Cl_log', 'Cl_log', data)
        rows.append({'pathway': 'Dilution', 'from': 'Cl', 'to': 'As',
                     'beta': b, 'p': p, 'se': se})

    path_df = pd.DataFrame(rows)
    path_df['zone'] = zone_label
    path_df['n'] = len(data.dropna())
    return path_df


def compute_indirect_effects(path_df, zone_label):
    """Compute indirect effects as product of path coefficients."""
    rows = []

    def get_beta(frm, to):
        mask = (path_df['from'] == frm) & (path_df['to'] == to)
        if mask.any():
            return path_df.loc[mask, 'beta'].values[0]
        return np.nan

    # Reductive: Season -> TDS -> ORP -> Fe -> As
    b_st  = get_beta('Season', 'TDS')
    b_to  = get_beta('TDS',    'ORP')
    b_of  = get_beta('ORP',    'Fe')
    b_fa  = get_beta('Fe',     'As')
    ie_red = b_st * b_to * b_of * b_fa

    # Desorption: Season -> PO4 -> As
    b_sp  = get_beta('Season', 'PO4')
    b_pa  = get_beta('PO4',    'As')
    ie_des = b_sp * b_pa

    # Dilution: Season -> Cl -> As
    b_sc  = get_beta('Season', 'Cl')
    b_ca  = get_beta('Cl',     'As')
    ie_dil = b_sc * b_ca

    ratio = abs(ie_des / ie_red) if (ie_red != 0 and not np.isnan(ie_red)) else np.nan

    rows.append({
        'zone': zone_label,
        'ie_reductive':  round(ie_red,  6),
        'ie_desorption': round(ie_des,  6),
        'ie_dilution':   round(ie_dil,  6),
        'desorption_dominance_ratio': round(ratio, 1) if not np.isnan(ratio) else np.nan,
        'dominant_pathway': ('Desorption' if abs(ie_des) > abs(ie_red) else 'Reductive')
                             if not np.isnan(ie_des) and not np.isnan(ie_red) else 'NA',
    })
    return pd.DataFrame(rows)


def run_sem_log():
    print("\n" + "="*65)
    print("H3 SEM RE-RUN: LOG-TRANSFORMED MEDIATORS")
    print("="*65)

    df = pd.read_csv(DATA)
    df = df[(df['Latitude'] >= 20.5) & (df['Longitude'] >= 87.8)].copy()
    df['Season_bin'] = (df['Season'] == 'Wet').astype(int)

    # Depth zones
    bins   = [0, 30, 80, 150, 9999]
    labels = DEPTH_ZONE_LABELS
    df['Depth_Zone'] = pd.cut(df['Depth'], bins=bins, labels=labels)

    # Log-transform all mediators
    df = log_transform_mediators(df)

    all_paths = []
    all_ie    = []

    for zone in ZONES_TO_RUN:
        sub = df if zone == 'ALL' else df[df['Depth_Zone'] == zone]
        n = sub.dropna(subset=['As_log', 'Season_bin']).shape[0]
        print(f"\n  Zone: {zone} (n={n})")

        if n < 15:
            print("    SKIP: insufficient samples")
            continue

        paths = fit_sem_ols_log(sub, zone)
        ie    = compute_indirect_effects(paths, zone)
        all_paths.append(paths)
        all_ie.append(ie)

        # Print indirect effects
        for _, row in ie.iterrows():
            print(f"    IE reductive  : {row['ie_reductive']:.6f}")
            print(f"    IE desorption : {row['ie_desorption']:.6f}")
            print(f"    IE dilution   : {row['ie_dilution']:.6f}")
            print(f"    Desorption/Reductive ratio: {row['desorption_dominance_ratio']}")
            print(f"    Dominant pathway: {row['dominant_pathway']}")

    paths_df = pd.concat(all_paths, ignore_index=True)
    ie_df    = pd.concat(all_ie,    ignore_index=True)

    # Save tables
    paths_df.to_csv(TABLES_DIR / 'H3_T_sem_log_corrected.csv',      index=False)
    ie_df.to_csv(   TABLES_DIR / 'H3_T_sem_log_indirect_effects.csv', index=False)
    print(f"\n  Tables saved to {TABLES_DIR}")

    # ── Comparison figure: original vs log-transform indirect effects ─────────
    orig_file = TABLES_DIR / 'H3_T28_sem_indirect_effects.csv'
    if orig_file.exists():
        orig = pd.read_csv(orig_file)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('H3 SEM: Original vs Log-Transformed Mediators\nIndirect Effects Comparison',
                     fontsize=12, fontweight='bold')

        zones_plot = ie_df['zone'].tolist()
        x = np.arange(len(zones_plot))
        w = 0.35

        for ax_idx, (df_plot, title) in enumerate([(ie_df, 'Log-transformed (corrected)')]):
            ax = axes[0]
            ax.bar(x - w/2, df_plot['ie_desorption'], w, label='Desorption', color='#e74c3c', alpha=0.8)
            ax.bar(x + w/2, df_plot['ie_reductive'],  w, label='Reductive',  color='#2980b9', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(df_plot['zone'], rotation=30, ha='right', fontsize=9)
            ax.set_ylabel('Indirect effect (log-scale path coefficients)', fontsize=10)
            ax.set_title(title, fontsize=11)
            ax.legend(fontsize=9)
            ax.axhline(0, color='black', lw=0.8)
            ax.grid(True, alpha=0.3, axis='y')

        # Panel B: dominance ratio
        ax2 = axes[1]
        ratios = ie_df['desorption_dominance_ratio'].replace([np.inf, -np.inf], np.nan)
        colors = ['#e74c3c' if r > 1 else '#2980b9' for r in ratios.fillna(0)]
        bars = ax2.bar(x, ratios.fillna(0), color=colors, alpha=0.8)
        ax2.axhline(1, color='black', lw=1.5, linestyle='--',
                    label='Equal dominance (ratio=1)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(ie_df['zone'], rotation=30, ha='right', fontsize=9)
        ax2.set_ylabel('|IE_desorption| / |IE_reductive| ratio', fontsize=10)
        ax2.set_title('Mechanism Dominance Ratio\n(Log-corrected SEM)', fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, ratios.fillna(0)):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                         f'{val:.1f}×', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        out = FIGURES_DIR / 'H3_F_sem_log_comparison.png'
        fig.savefig(out, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Figure saved: {out}")
    else:
        # Simple figure without comparison
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(ie_df))
        w = 0.35
        ax.bar(x - w/2, ie_df['ie_desorption'], w, label='Desorption',
               color='#e74c3c', alpha=0.8)
        ax.bar(x + w/2, ie_df['ie_reductive'],  w, label='Reductive',
               color='#2980b9', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(ie_df['zone'], rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Indirect effect (log path coefficients)', fontsize=11)
        ax.set_title('H3 SEM Indirect Effects — Log-Transformed Mediators', fontsize=12)
        ax.legend(); ax.axhline(0, color='black', lw=0.8); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        out = FIGURES_DIR / 'H3_F_sem_log_comparison.png'
        fig.savefig(out, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Figure saved: {out}")

    return ie_df


if __name__ == '__main__':
    result = run_sem_log()
    print("\n=== FINAL INDIRECT EFFECTS (LOG-CORRECTED) ===")
    print(result.to_string(index=False))
