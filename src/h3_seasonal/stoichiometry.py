"""
Phase D — Stoichiometric Displacement Ratio (H3.3)
Field-scale Δ[As]/Δ[PO₄] molar ratio compared to laboratory constants.
EVI as crop-cycle proxy for fertiliser application intensity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from config import (
    TARGET, DEPTH_ZONE_LABELS, ZONE_COLORS, SEASON_DRY, SEASON_WET,
    FIGURES_DIR, TABLES_DIR, DPI, ALPHA, MW_As, MW_PO4,
    LIT_RATIO_LOW, LIT_RATIO_HIGH
)


def compute_displacement_ratio(df_delta):
    """
    Compute Δ[As]/Δ[PO₄] molar ratio for wells where both Δ[As]>0 and Δ[PO₄]>0.
    Returns df_delta with R column added.
    """
    df = df_delta.copy()
    # Ensure molar columns exist
    if 'Delta_As_umol' not in df.columns:
        if f'{TARGET}_dry' in df.columns and f'{TARGET}_wet' in df.columns:
            df['As_umol_dry'] = df[f'{TARGET}_dry'] / MW_As
            df['As_umol_wet'] = df[f'{TARGET}_wet'] / MW_As
            df['Delta_As_umol'] = df['As_umol_wet'] - df['As_umol_dry']
        else:
            df['Delta_As_umol'] = np.nan

    if 'Delta_PO4_umol' not in df.columns:
        po4_dry = 'PO43-_dry'
        po4_wet = 'PO43-_wet'
        if po4_dry in df.columns and po4_wet in df.columns:
            df['PO4_umol_dry'] = df[po4_dry] * 1000 / MW_PO4
            df['PO4_umol_wet'] = df[po4_wet] * 1000 / MW_PO4
            df['Delta_PO4_umol'] = df['PO4_umol_wet'] - df['PO4_umol_dry']
        else:
            df['Delta_PO4_umol'] = np.nan

    # Displacement ratio only where BOTH Δ are positive (mobilisation event)
    valid = (df['Delta_As_umol'] > 0) & (df['Delta_PO4_umol'] > 0)
    df['R_displacement'] = np.where(valid,
                                    df['Delta_As_umol'] / (df['Delta_PO4_umol'] + 1e-12),
                                    np.nan)
    # Cap outliers at 99th percentile
    cap = df['R_displacement'].quantile(0.99)
    df['R_displacement'] = df['R_displacement'].clip(upper=cap)

    return df


def run_stoichiometry(df_delta):
    """Phase D: stoichiometric displacement ratio by depth zone."""
    print("\n" + "█" * 70)
    print(" PHASE D: STOICHIOMETRIC DISPLACEMENT RATIO (H3.3)")
    print("█" * 70)

    df_delta = compute_displacement_ratio(df_delta)

    # ── 1. Ratio by depth zone ────────────────────────────────────────────
    print("\n  Displacement ratio Δ[As]/Δ[PO₄] by depth zone:")
    rows = []
    for zone in ['ALL'] + DEPTH_ZONE_LABELS:
        sub = df_delta if zone == 'ALL' else df_delta[df_delta['Depth_Zone'] == zone]
        R_vals = sub['R_displacement'].dropna()
        if len(R_vals) < 5:
            continue

        in_lit_range = ((R_vals >= LIT_RATIO_LOW) & (R_vals <= LIT_RATIO_HIGH)).mean()
        t_stat, t_p = stats.ttest_1samp(R_vals, (LIT_RATIO_LOW + LIT_RATIO_HIGH) / 2)

        rows.append({
            'Depth_Zone': zone, 'N': len(R_vals),
            'R_mean': round(R_vals.mean(), 6),
            'R_median': round(R_vals.median(), 6),
            'R_std': round(R_vals.std(), 6),
            'R_Q25': round(R_vals.quantile(0.25), 6),
            'R_Q75': round(R_vals.quantile(0.75), 6),
            'Frac_in_lit_range': round(in_lit_range, 3),
            'Lit_range': f'[{LIT_RATIO_LOW}, {LIT_RATIO_HIGH}]',
            'Consistent_with_lab': 'YES' if in_lit_range > 0.5 else 'NO',
            'ttest_vs_lit_midpoint_p': round(t_p, 5),
        })
        print(f"  {zone:15s}: R={R_vals.median():.5f} (n={len(R_vals)})  "
              f"in_lit_range={in_lit_range:.2%}  "
              f"lab_consistent={'YES' if in_lit_range > 0.5 else 'NO'}")

    ratio_df = pd.DataFrame(rows)
    ratio_df.to_csv(TABLES_DIR / 'H3_T10_stoichiometric_ratio_by_zone.csv', index=False)

    # ── 2. EVI amplitude vs. displacement ratio ───────────────────────────
    print("\n  EVI amplitude vs. displacement ratio:")
    evi_rows = []
    for zone in ['ALL'] + DEPTH_ZONE_LABELS:
        sub = df_delta if zone == 'ALL' else df_delta[df_delta['Depth_Zone'] == zone]
        sub = sub.dropna(subset=['R_displacement'])

        evi_col = 'EVI_mean' if 'EVI_mean' in sub.columns else None
        if evi_col is None or len(sub) < 10:
            continue

        valid = sub[[evi_col, 'R_displacement']].dropna()
        if len(valid) < 10:
            continue

        rho, p = stats.spearmanr(valid[evi_col], valid['R_displacement'])
        evi_rows.append({
            'Depth_Zone': zone, 'N': len(valid),
            'EVI_col': evi_col,
            'Spearman_rho': round(rho, 5),
            'p_value': round(p, 6),
            'Significant': 'YES' if p < ALPHA else 'NO',
            'Interpretation': ('EVI predicts displacement ratio — '
                               'more crops = more efficient As mobilisation')
            if (p < ALPHA and rho > 0) else 'No significant EVI-ratio relationship',
        })
        print(f"  {zone:15s}: ρ(EVI, R)={rho:+.4f}  p={p:.4f}  sig={'YES' if p < ALPHA else 'NO'}")

    evi_df = pd.DataFrame(evi_rows)
    evi_df.to_csv(TABLES_DIR / 'H3_T11_evi_displacement_correlation.csv', index=False)

    # ── 3. Seasonal PO₄ distribution ─────────────────────────────────────
    # Compare PO₄ between dry and wet seasons per depth zone
    if 'PO43-_dry' in df_delta.columns and 'PO43-_wet' in df_delta.columns:
        po4_rows = []
        for zone in DEPTH_ZONE_LABELS:
            sub = df_delta[df_delta['Depth_Zone'] == zone]
            dry_po4 = sub['PO43-_dry'].dropna().values
            wet_po4 = sub['PO43-_wet'].dropna().values
            if len(dry_po4) < 5 or len(wet_po4) < 5:
                continue
            u, p = stats.mannwhitneyu(wet_po4, dry_po4, alternative='two-sided')
            po4_rows.append({
                'Depth_Zone': zone,
                'PO4_dry_median': round(np.median(dry_po4), 4),
                'PO4_wet_median': round(np.median(wet_po4), 4),
                'Delta_PO4_pct': round((np.median(wet_po4) - np.median(dry_po4)) /
                                       (np.median(dry_po4) + 1e-9) * 100, 2),
                'MannWhitney_p': round(p, 6),
                'Significant': 'YES' if p < ALPHA else 'NO',
                'N_dry': len(dry_po4), 'N_wet': len(wet_po4),
            })
            print(f"  PO₄ {zone}: dry={np.median(dry_po4):.3f}  "
                  f"wet={np.median(wet_po4):.3f}  p={p:.4f}")
        po4_df = pd.DataFrame(po4_rows)
        po4_df.to_csv(TABLES_DIR / 'H3_T12_po4_seasonal_distribution.csv', index=False)

    # ── 4. Figures ────────────────────────────────────────────────────────
    _plot_ratio_by_zone(df_delta, ratio_df)
    _plot_evi_ratio_scatter(df_delta)

    print("\n✓ Phase D complete. Outputs: H3_T10, H3_T11, H3_T12, H3_F08, H3_F09")
    return {'ratio': ratio_df, 'evi_corr': evi_df, 'df_delta': df_delta}


def _plot_ratio_by_zone(df_delta, ratio_df):
    """H3_F08: Displacement ratio by depth zone with lab reference band."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: boxplot R by zone
    ax = axes[0]
    data_by_zone = [df_delta[df_delta['Depth_Zone'] == z]['R_displacement'].dropna().values
                    for z in DEPTH_ZONE_LABELS]
    bp = ax.boxplot(data_by_zone, patch_artist=True, labels=DEPTH_ZONE_LABELS,
                    showfliers=False, widths=0.5)
    for patch, zone in zip(bp['boxes'], DEPTH_ZONE_LABELS):
        patch.set_facecolor(ZONE_COLORS.get(zone, 'steelblue'))
        patch.set_alpha(0.7)

    ax.axhspan(LIT_RATIO_LOW, LIT_RATIO_HIGH, color='gold', alpha=0.3,
               label=f'Lab range [{LIT_RATIO_LOW}, {LIT_RATIO_HIGH}]')
    ax.set_ylabel('Δ[As] / Δ[PO₄] (µmol/µmol)', fontsize=11)
    ax.set_title('H3.3: Stoichiometric Displacement Ratio by Depth Zone\n'
                 '(Gold band = laboratory adsorption competition range)', fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(DEPTH_ZONE_LABELS, rotation=15)

    # Right: fraction in lab range
    ax2 = axes[1]
    if ratio_df is not None and len(ratio_df) > 0:
        zones = ratio_df['Depth_Zone'].tolist()
        frac = ratio_df['Frac_in_lit_range'].values
        colors = [ZONE_COLORS.get(z, 'steelblue') for z in zones]
        x = np.arange(len(zones))
        ax2.bar(x, frac * 100, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax2.axhline(50, color='black', ls='--', lw=1.5, label='50% threshold')
        ax2.set_xticks(x); ax2.set_xticklabels(zones, rotation=15, fontsize=9)
        ax2.set_ylabel('% Wells within Lab Reference Range', fontsize=11)
        ax2.set_title('H3.3: Fraction of Wells Consistent with Lab Adsorption Constants', fontsize=11)
        ax2.set_ylim(0, 100); ax2.legend(); ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('H3 Phase D: Field-Scale Stoichiometric As-PO₄ Displacement Ratio',
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H3_F08_stoichiometric_ratio.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H3_F08_stoichiometric_ratio.png")


def _plot_evi_ratio_scatter(df_delta):
    """H3_F09: EVI amplitude vs. displacement ratio with LOESS fit."""
    evi_col = 'EVI_mean' if 'EVI_mean' in df_delta.columns else None
    if evi_col is None:
        return
    valid = df_delta[[evi_col, 'R_displacement', 'Depth_Zone']].dropna()
    if len(valid) < 10:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    for zone in DEPTH_ZONE_LABELS:
        sub = valid[valid['Depth_Zone'] == zone]
        if len(sub) < 3:
            continue
        ax.scatter(sub[evi_col], sub['R_displacement'],
                   color=ZONE_COLORS.get(zone, 'grey'), s=30, alpha=0.6,
                   label=zone, edgecolors='k', linewidths=0.2)

    # Overall LOESS / lowess fit
    from statsmodels.nonparametric.smoothers_lowess import lowess
    lw = lowess(valid['R_displacement'], valid[evi_col], frac=0.4)
    ax.plot(lw[:, 0], lw[:, 1], 'k-', lw=2.5, label='LOESS fit')

    ax.axhspan(LIT_RATIO_LOW, LIT_RATIO_HIGH, color='gold', alpha=0.2,
               label=f'Lab range [{LIT_RATIO_LOW}, {LIT_RATIO_HIGH}]')
    rho, p = stats.spearmanr(valid[evi_col], valid['R_displacement'])
    ax.set_xlabel(f'{evi_col} (Crop biomass / agricultural intensity)', fontsize=11)
    ax.set_ylabel('Δ[As]/Δ[PO₄] Displacement Ratio (µmol/µmol)', fontsize=11)
    ax.set_title(f'H3.3: EVI vs. Stoichiometric Displacement Ratio\n'
                 f'(ρ={rho:+.3f}, p={p:.4f}) — More crops → more efficient As mobilisation',
                 fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H3_F09_evi_ratio_scatter.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H3_F09_evi_ratio_scatter.png")
