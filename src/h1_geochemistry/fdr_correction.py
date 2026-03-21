"""
FDR Correction: Benjamini-Hochberg across all Spearman tests in H1 and H2.
Collects all p-values from saved tables, applies BH correction,
reports which results flip from significant to non-significant.

Outputs:
  hypo1/output/tables/H1_T_fdr_corrected.csv
  hypo2/output/tables/H2_T_fdr_corrected.csv
  hypo1/output/tables/ALL_TESTS_FDR_SUMMARY.csv
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from pathlib import Path

ROOT    = Path(__file__).parent.parent
H1_OUT  = ROOT / 'hypo1' / 'output' / 'tables'
H2_OUT  = ROOT / 'hypo2' / 'output' / 'tables'
DATA    = ROOT / 'hypo1' / 'output' / 'data_phase1_integrated.csv'
ALPHA   = 0.05

def collect_h1_tests(df):
    """Re-compute all H1 Spearman correlations for As vs each ion, by zone."""
    TARGET = 'As'
    IONS = ['Na+', 'K+', 'Mg2+', 'Ca2+', 'Fe2+', 'Mn2+',
            'NO3-', 'SO42-', 'PO43-', 'HCO3-', 'Cl-',
            'Al3+', 'B+', 'pH', 'ORP', 'DO', 'EC', 'TDS']

    bins   = [0, 30, 80, 150, 9999]
    labels = ['Shallow', 'Intermediate', 'Medium-Deep', 'Deep']
    df['Depth_Zone'] = pd.cut(df['Depth'], bins=bins, labels=labels)

    rows = []
    for zone in ['ALL'] + labels:
        sub = df if zone == 'ALL' else df[df['Depth_Zone'] == zone]
        for ion in IONS:
            if ion not in sub.columns:
                continue
            clean = sub[[TARGET, ion]].dropna()
            if len(clean) < 10:
                continue
            rho, p = spearmanr(clean[TARGET], clean[ion])
            rows.append({
                'hypothesis': 'H1',
                'test': f'As ~ {ion}',
                'zone': zone,
                'n': len(clean),
                'rho': round(rho, 4),
                'p_raw': p,
                'significant_raw': p < ALPHA,
            })
    return pd.DataFrame(rows)


def collect_h2_tests(df):
    """Re-compute H2 partial correlations and mechanism shift tests."""
    TARGET = 'As'
    df['As_log'] = np.log10(df[TARGET].clip(lower=0.01))

    bins   = [0, 30, 80, 150, 9999]
    labels = ['Shallow', 'Intermediate', 'Medium-Deep', 'Deep']
    df['Depth_Zone'] = pd.cut(df['Depth'], bins=bins, labels=labels)

    MECHS = {'PO43-': 'Fe2+', 'Fe2+': 'PO43-',
             'HCO3-': 'Fe2+', 'ORP': 'PO43-'}

    rows = []
    for zone in ['ALL'] + labels:
        sub = df if zone == 'ALL' else df[df['Depth_Zone'] == zone]

        # Raw Spearman As ~ depth
        clean = sub[['Depth', 'As_log']].dropna()
        if len(clean) >= 10:
            rho, p = spearmanr(clean['As_log'], clean['Depth'])
            rows.append({'hypothesis': 'H2', 'test': 'As_log ~ Depth',
                         'zone': zone, 'n': len(clean),
                         'rho': round(rho, 4), 'p_raw': p,
                         'significant_raw': p < ALPHA})

        # ORP ~ Depth
        clean2 = sub[['Depth', 'ORP']].dropna()
        if len(clean2) >= 10:
            rho, p = spearmanr(clean2['ORP'], clean2['Depth'])
            rows.append({'hypothesis': 'H2', 'test': 'ORP ~ Depth',
                         'zone': zone, 'n': len(clean2),
                         'rho': round(rho, 4), 'p_raw': p,
                         'significant_raw': p < ALPHA})

        # Mechanism proxies ~ As (partial)
        for pred, covar in MECHS.items():
            if pred not in sub.columns or covar not in sub.columns:
                continue
            clean3 = sub[['As_log', pred, covar]].dropna()
            if len(clean3) < 10:
                continue

            def resid(y, x):
                mask = np.isfinite(y) & np.isfinite(x)
                if mask.sum() < 5: return y
                c = np.polyfit(x[mask], y[mask], 1)
                return y - np.polyval(c, x)

            y_res = resid(clean3['As_log'].values, clean3[covar].values)
            x_res = resid(clean3[pred].values, clean3[covar].values)
            rho, p = spearmanr(x_res, y_res)
            rows.append({'hypothesis': 'H2',
                         'test': f'As_log ~ {pred} | {covar}',
                         'zone': zone, 'n': len(clean3),
                         'rho': round(rho, 4), 'p_raw': p,
                         'significant_raw': p < ALPHA})

    return pd.DataFrame(rows)


def apply_fdr(df, alpha=ALPHA):
    """Apply Benjamini-Hochberg FDR correction."""
    p_vals = df['p_raw'].values
    reject, p_corrected, _, _ = multipletests(p_vals, alpha=alpha, method='fdr_bh')
    df = df.copy()
    df['p_fdr']              = np.round(p_corrected, 6)
    df['significant_fdr']    = reject
    df['flipped']            = df['significant_raw'] & ~df['significant_fdr']
    df['newly_significant']  = ~df['significant_raw'] & df['significant_fdr']
    return df


def run_fdr():
    print("\n" + "="*65)
    print("FDR CORRECTION: BENJAMINI-HOCHBERG")
    print("="*65)

    df = pd.read_csv(DATA)
    df = df[(df['Latitude'] >= 20.5) & (df['Longitude'] >= 87.8)].copy()

    print("\nCollecting H1 Spearman tests...")
    h1_tests = collect_h1_tests(df.copy())
    print(f"  H1: {len(h1_tests)} tests")

    print("Collecting H2 tests...")
    h2_tests = collect_h2_tests(df.copy())
    print(f"  H2: {len(h2_tests)} tests")

    # Combine and apply FDR within each hypothesis separately
    print("\nApplying BH-FDR correction...")

    h1_fdr = apply_fdr(h1_tests)
    h2_fdr = apply_fdr(h2_tests)

    # Also apply globally across H1+H2 combined
    all_tests = pd.concat([h1_fdr, h2_fdr], ignore_index=True)
    global_fdr = apply_fdr(all_tests.rename(columns={'p_raw': 'p_raw'}))

    # Summary
    for hyp, fdr_df in [('H1', h1_fdr), ('H2', h2_fdr)]:
        n_raw = fdr_df['significant_raw'].sum()
        n_fdr = fdr_df['significant_fdr'].sum()
        n_flip = fdr_df['flipped'].sum()
        print(f"\n  {hyp}:")
        print(f"    Significant (raw α=0.05): {n_raw}/{len(fdr_df)}")
        print(f"    Significant (BH-FDR):     {n_fdr}/{len(fdr_df)}")
        print(f"    Flipped (sig → ns):        {n_flip}")
        if n_flip > 0:
            print(f"    Flipped tests:")
            print(fdr_df[fdr_df['flipped']][
                ['test', 'zone', 'rho', 'p_raw', 'p_fdr']
            ].to_string(index=False))

    # Key results check
    print("\n  KEY FINDINGS UNDER FDR CORRECTION:")
    key_tests = [
        ('H1', 'As ~ PO43-', 'ALL'),
        ('H1', 'As ~ Fe2+', 'ALL'),
        ('H1', 'As ~ HCO3-', 'ALL'),
        ('H2', 'As_log ~ ORP | PO43-', 'Shallow'),
        ('H2', "As_log ~ PO43- | Fe2+", 'Shallow'),
        ('H2', "As_log ~ Fe2+ | PO43-", 'Medium-Deep'),
        ('H2', 'ORP ~ Depth', 'ALL'),
    ]
    for hyp, test, zone in key_tests:
        src = h1_fdr if hyp == 'H1' else h2_fdr
        mask = (src['test'] == test) & (src['zone'] == zone)
        if mask.any():
            row = src[mask].iloc[0]
            status = "✓ HOLDS" if row['significant_fdr'] else "✗ FLIPS"
            print(f"    {status}  {hyp} {zone}: {test}  "
                  f"ρ={row['rho']:.3f} p_raw={row['p_raw']:.4f} "
                  f"p_fdr={row['p_fdr']:.4f}")

    # Save
    h1_fdr.to_csv(H1_OUT / 'H1_T_fdr_corrected.csv', index=False)
    h2_fdr.to_csv(H2_OUT / 'H2_T_fdr_corrected.csv', index=False)
    all_tests_out = H1_OUT / 'ALL_TESTS_FDR_SUMMARY.csv'
    pd.concat([h1_fdr, h2_fdr]).to_csv(all_tests_out, index=False)
    print(f"\n  Tables saved:")
    print(f"    {H1_OUT / 'H1_T_fdr_corrected.csv'}")
    print(f"    {H2_OUT / 'H2_T_fdr_corrected.csv'}")
    print(f"    {all_tests_out}")

    return h1_fdr, h2_fdr


if __name__ == '__main__':
    h1, h2 = run_fdr()
