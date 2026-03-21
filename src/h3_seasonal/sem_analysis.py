"""
Phase H — Structural Equation Model (H3.7)
Simultaneous three-pathway partitioning:
  Pathway 1 (Reductive):   Season → TDS → ORP → Fe²⁺ → As
  Pathway 2 (Desorption):  Season → PO₄ → As
  Pathway 3 (Dilution):    Season → Cl⁻ → As
Multi-group SEM across 4 depth zones.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from config import (
    TARGET, DEPTH_ZONE_LABELS, ZONE_COLORS, FIGURES_DIR, TABLES_DIR,
    DPI, ALPHA, RANDOM_STATE, N_BOOTSTRAP,
    SEM_DOC_PROXY, SEM_REDOX_MED, SEM_FE_MED, SEM_PO4_MED, SEM_CONC_MED
)

# ─── OPTIONAL semopy IMPORT ───────────────────────────────────────────────────
try:
    import semopy
    SEMOPY_AVAILABLE = True
except ImportError:
    SEMOPY_AVAILABLE = False
    print("  [WARN] semopy not available — using OLS path decomposition fallback")


# ─── SEM MODEL SPECIFICATION ──────────────────────────────────────────────────

# Three concurrent pathways in one model
SEM_SPEC = """
# Pathway 1: Reductive dissolution
TDS  ~ Season_bin
ORP  ~ TDS
Fe   ~ ORP
As_log ~ Fe + ORP + TDS

# Pathway 2: Phosphate desorption
PO4  ~ Season_bin
As_log ~ PO4

# Pathway 3: Dilution / concentration
Cl   ~ Season_bin
As_log ~ Cl

# Direct seasonal effect (after mediation)
As_log ~ Season_bin
"""

RENAME_MAP = {
    SEM_DOC_PROXY: 'TDS',
    SEM_REDOX_MED: 'ORP',
    SEM_FE_MED:    'Fe',
    SEM_PO4_MED:   'PO4',
    SEM_CONC_MED:  'Cl',
    f'{TARGET}_log': 'As_log',
}


def _prep_sem_data(df):
    """Standardise and rename variables for SEM."""
    needed_raw = [SEM_DOC_PROXY, SEM_REDOX_MED, SEM_FE_MED,
                  SEM_PO4_MED, SEM_CONC_MED, f'{TARGET}_log', 'Season_bin']
    available  = [c for c in needed_raw if c in df.columns]
    sub = df[available].dropna().copy()

    # Standardise mediators (but NOT Season_bin, it stays 0/1)
    std_cols = [c for c in available if c != 'Season_bin']
    for col in std_cols:
        mu, sd = sub[col].mean(), sub[col].std()
        sub[col] = (sub[col] - mu) / (sd + 1e-9)

    sub = sub.rename(columns=RENAME_MAP)
    return sub


# ─── semopy FIT ───────────────────────────────────────────────────────────────

def fit_sem_semopy(data, zone_label):
    """
    Fit SEM with semopy. Returns dict of results.
    """
    if not SEMOPY_AVAILABLE:
        return None

    # Build spec matching available columns
    available = set(data.columns)
    spec_lines = []
    for line in SEM_SPEC.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        lhs, rhs_str = line.split('~')
        lhs = lhs.strip()
        rhs_terms = [t.strip() for t in rhs_str.split('+')]
        rhs_ok = [t for t in rhs_terms if t in available]
        if lhs in available and rhs_ok:
            spec_lines.append(f"{lhs} ~ {' + '.join(rhs_ok)}")
    spec = '\n'.join(spec_lines)
    if not spec:
        return None

    try:
        model = semopy.Model(spec)
        model.fit(data)
        params = model.inspect()

        # Fit statistics
        try:
            stats_res = model.calc_stats()
            chi2 = stats_res.get('chi2', {})
            cfi  = float(stats_res.get('CFI', np.nan))
            rmsea= float(stats_res.get('RMSEA', np.nan))
            srmr = float(stats_res.get('SRMR', np.nan))
        except Exception:
            chi2, cfi, rmsea, srmr = {}, np.nan, np.nan, np.nan

        return {
            'zone': zone_label,
            'n': len(data),
            'params': params,
            'cfi': cfi, 'rmsea': rmsea, 'srmr': srmr,
            'spec': spec,
        }
    except Exception as e:
        print(f"    semopy failed for {zone_label}: {e}")
        return None


# ─── OLS PATH FALLBACK ────────────────────────────────────────────────────────

def fit_sem_ols_fallback(data, zone_label):
    """
    OLS-based path decomposition when semopy unavailable.
    Computes standardised path coefficients for each link.
    """
    import statsmodels.formula.api as smf

    def _beta(formula, pred, d):
        try:
            m = smf.ols(formula, data=d).fit()
            # Try Q-quoted version first
            key = f'Q("{pred}")'
            return float(m.params.get(key, m.params.get(pred, np.nan)))
        except Exception:
            return np.nan

    rows = []
    # Reductive links
    if 'TDS' in data.columns:
        b = _beta('TDS ~ Season_bin', 'Season_bin', data)
        rows.append({'from': 'Season', 'to': 'TDS', 'beta': b, 'pathway': 'Reductive'})
    if 'ORP' in data.columns and 'TDS' in data.columns:
        b = _beta('ORP ~ TDS', 'TDS', data)
        rows.append({'from': 'TDS', 'to': 'ORP', 'beta': b, 'pathway': 'Reductive'})
    if 'Fe' in data.columns and 'ORP' in data.columns:
        b = _beta('Fe ~ ORP', 'ORP', data)
        rows.append({'from': 'ORP', 'to': 'Fe', 'beta': b, 'pathway': 'Reductive'})
    if 'Fe' in data.columns and 'As_log' in data.columns:
        b = _beta('As_log ~ Fe', 'Fe', data)
        rows.append({'from': 'Fe', 'to': 'As_log', 'beta': b, 'pathway': 'Reductive'})

    # Desorption links
    if 'PO4' in data.columns:
        b = _beta('PO4 ~ Season_bin', 'Season_bin', data)
        rows.append({'from': 'Season', 'to': 'PO4', 'beta': b, 'pathway': 'Desorption'})
    if 'PO4' in data.columns and 'As_log' in data.columns:
        b = _beta('As_log ~ PO4', 'PO4', data)
        rows.append({'from': 'PO4', 'to': 'As_log', 'beta': b, 'pathway': 'Desorption'})

    # Dilution links
    if 'Cl' in data.columns:
        b = _beta('Cl ~ Season_bin', 'Season_bin', data)
        rows.append({'from': 'Season', 'to': 'Cl', 'beta': b, 'pathway': 'Dilution'})
    if 'Cl' in data.columns and 'As_log' in data.columns:
        b = _beta('As_log ~ Cl', 'Cl', data)
        rows.append({'from': 'Cl', 'to': 'As_log', 'beta': b, 'pathway': 'Dilution'})

    path_df = pd.DataFrame(rows)
    path_df['zone'] = zone_label
    path_df['n']    = len(data)

    # Compute indirect effects per pathway
    indirect = {}
    for pathway in ['Reductive', 'Desorption', 'Dilution']:
        sub = path_df[path_df['pathway'] == pathway]['beta'].dropna()
        indirect[pathway] = sub.prod() if len(sub) else np.nan

    return {
        'zone': zone_label, 'n': len(data),
        'params': path_df,
        'indirect': indirect,
        'cfi': np.nan, 'rmsea': np.nan, 'srmr': np.nan,
    }


# ─── MULTI-GROUP SEM ──────────────────────────────────────────────────────────

def run_sem_analysis(df):
    """Phase H: SEM across 'ALL' and each depth zone."""
    print("\n" + "█" * 70)
    print(" PHASE H: STRUCTURAL EQUATION MODEL (H3.7)")
    print("█" * 70)

    all_results = {}
    path_rows   = []
    fit_rows    = []

    for zone in ['ALL'] + DEPTH_ZONE_LABELS:
        sub = df if zone == 'ALL' else df[df['Depth_Zone'] == zone]
        data = _prep_sem_data(sub)
        if len(data) < 30:
            print(f"  SKIP {zone}: n={len(data)} < 30")
            continue
        print(f"\n  Zone: {zone} (n={len(data)})")

        if SEMOPY_AVAILABLE:
            result = fit_sem_semopy(data, zone)
        else:
            result = None

        if result is None:
            result = fit_sem_ols_fallback(data, zone)

        all_results[zone] = result

        # Collect path coefficients
        params = result.get('params')
        if isinstance(params, pd.DataFrame):
            for _, row in params.iterrows():
                path_rows.append({
                    'Zone': zone, 'N': result['n'],
                    'From': row.get('lval', row.get('from', '')),
                    'To':   row.get('rval', row.get('to', '')),
                    'Beta': row.get('Estimate', row.get('beta', np.nan)),
                    'p_value': row.get('p-value', np.nan),
                    'Pathway': row.get('pathway', _infer_pathway(
                        row.get('lval', row.get('from', '')),
                        row.get('rval', row.get('to', ''))
                    )),
                })

        fit_rows.append({
            'Zone': zone, 'N': result['n'],
            'CFI': result.get('cfi', np.nan),
            'RMSEA': result.get('rmsea', np.nan),
            'SRMR': result.get('srmr', np.nan),
        })

        # Print indirect effects
        if 'indirect' in result:
            for pw, val in result['indirect'].items():
                print(f"    Indirect ({pw}): {val:+.5f}" if pd.notna(val) else
                      f"    Indirect ({pw}): NA")

    path_df = pd.DataFrame(path_rows)
    fit_df  = pd.DataFrame(fit_rows)

    path_df.to_csv(TABLES_DIR / 'H3_T29_sem_path_coefficients.csv', index=False)
    fit_df.to_csv(TABLES_DIR  / 'H3_T30_sem_fit_statistics.csv',    index=False)

    # Pathway dominance table
    dominance_df = _pathway_dominance_table(all_results)
    dominance_df.to_csv(TABLES_DIR / 'H3_T31_pathway_dominance.csv', index=False)

    # Figures
    _plot_sem_path_heatmap(path_df)
    _plot_pathway_dominance(dominance_df)

    print("\n✓ Phase H complete. Outputs: H3_T29, H3_T30, H3_T31, H3_F18, H3_F19")
    return {'results': all_results, 'paths': path_df, 'fit': fit_df,
            'dominance': dominance_df}


def _infer_pathway(from_var, to_var):
    reductive_nodes = {'Season', 'TDS', 'ORP', 'Fe'}
    desorption_nodes = {'Season', 'PO4', 'As_log'}
    dilution_nodes   = {'Season', 'Cl', 'As_log'}
    if from_var in reductive_nodes and to_var in reductive_nodes:
        return 'Reductive'
    if from_var in desorption_nodes and to_var == 'As_log':
        return 'Desorption'
    if from_var in dilution_nodes and to_var == 'As_log':
        return 'Dilution'
    return 'Other'


def _pathway_dominance_table(all_results):
    """
    For each zone: which pathway has the largest absolute indirect effect?
    """
    rows = []
    for zone, result in all_results.items():
        if result is None:
            continue
        indirect = result.get('indirect', {})
        if not indirect:
            # Estimate from path coefficients via product
            params = result.get('params')
            if params is None:
                continue
            indirect = {}
            for pw in ['Reductive', 'Desorption', 'Dilution']:
                sub = params[params.get('Pathway', params.get('pathway', '')) == pw
                             if 'Pathway' in params.columns or 'pathway' in params.columns
                             else pd.Series(False)]['Beta' if 'Beta' in params.columns else 'beta'].dropna()
                indirect[pw] = float(sub.prod()) if len(sub) else np.nan

        abs_ind = {k: abs(v) for k, v in indirect.items() if pd.notna(v)}
        if not abs_ind:
            dominant = 'Undetermined'
        else:
            dominant = max(abs_ind, key=abs_ind.get)

        rows.append({
            'Zone': zone, 'N': result['n'],
            'Indirect_Reductive':  indirect.get('Reductive', np.nan),
            'Indirect_Desorption': indirect.get('Desorption', np.nan),
            'Indirect_Dilution':   indirect.get('Dilution', np.nan),
            'Dominant_Pathway': dominant,
        })
        print(f"  {zone:15s}: dominant={dominant}  "
              f"red={indirect.get('Reductive', np.nan):.4f}  "
              f"des={indirect.get('Desorption', np.nan):.4f}  "
              f"dil={indirect.get('Dilution', np.nan):.4f}"
              if all(pd.notna(v) for v in indirect.values()) else
              f"  {zone:15s}: dominant={dominant}")

    return pd.DataFrame(rows)


def _plot_sem_path_heatmap(path_df):
    """H3_F18: Heatmap of standardised path coefficients × depth zone."""
    if path_df.empty:
        return

    beta_col = 'Beta'
    key_links = [
        ('Season', 'TDS'), ('TDS', 'ORP'), ('ORP', 'Fe'), ('Fe', 'As_log'),
        ('Season', 'PO4'), ('PO4', 'As_log'),
        ('Season', 'Cl'),  ('Cl', 'As_log'),
        ('Season', 'As_log'),
    ]
    link_labels = [f'{f}→{t}' for f, t in key_links]
    zones = path_df['Zone'].unique().tolist()

    matrix = np.full((len(key_links), len(zones)), np.nan)
    for j, zone in enumerate(zones):
        sub = path_df[path_df['Zone'] == zone]
        for i, (from_v, to_v) in enumerate(key_links):
            row = sub[(sub['From'] == from_v) & (sub['To'] == to_v)]
            if len(row):
                matrix[i, j] = row[beta_col].values[0]

    fig, ax = plt.subplots(figsize=(max(10, len(zones) * 1.5), 8))
    vabs = np.nanpercentile(np.abs(matrix[~np.isnan(matrix)]), 95) if not np.all(np.isnan(matrix)) else 0.5
    im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r',
                   vmin=-vabs, vmax=vabs)
    ax.set_xticks(range(len(zones))); ax.set_xticklabels(zones, rotation=25, fontsize=9)
    ax.set_yticks(range(len(link_labels))); ax.set_yticklabels(link_labels, fontsize=9)
    plt.colorbar(im, ax=ax, label='Standardised β')
    ax.set_title('H3.7: SEM Standardised Path Coefficients × Depth Zone\n'
                 '(Red=positive; Blue=negative)', fontsize=12)

    for i in range(len(key_links)):
        for j in range(len(zones)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=8,
                        color='white' if abs(v) > 0.3 else 'black')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H3_F18_sem_path_heatmap.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H3_F18_sem_path_heatmap.png")


def _plot_pathway_dominance(dominance_df):
    """H3_F19: Stacked bar chart of absolute indirect effects by zone."""
    if dominance_df.empty:
        return

    zones = dominance_df['Zone'].tolist()
    pathways = ['Indirect_Reductive', 'Indirect_Desorption', 'Indirect_Dilution']
    labels   = ['Reductive', 'Desorption', 'Dilution']
    colors   = ['#d73027', '#fc8d59', '#4575b4']

    # Absolute values for display
    data_abs = dominance_df[pathways].abs().fillna(0).values

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(zones))
    bottom = np.zeros(len(zones))
    for i, (pw, label, color) in enumerate(zip(pathways, labels, colors)):
        heights = data_abs[:, i]
        ax.bar(x, heights, bottom=bottom, color=color, alpha=0.85,
               label=label, edgecolor='black', linewidth=0.4)
        bottom += heights

    ax.set_xticks(x); ax.set_xticklabels(zones, rotation=20, fontsize=9)
    ax.set_ylabel('|Indirect Effect| (Standardised)', fontsize=11)
    ax.set_title('H3.7: SEM Pathway Dominance by Depth Zone\n'
                 '(Stacked absolute indirect effects)', fontsize=12)
    ax.legend(title='Pathway', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'H3_F19_pathway_dominance.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved: H3_F19_pathway_dominance.png")
