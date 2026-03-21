"""
Publication-ready 6-panel combined figure for Hypothesis 1 (H1)
Bangladesh Groundwater Arsenic Study — STOTEN / EST style
REWRITTEN: high-quality, spacious, 22×15 inches, DPI 300
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ── Paths ───────────────────────────────────────────────────────────────────
BASE        = '/Users/rakibhhridoy/AsGW/GroundWater/hypo1/output'
DATA_CSV    = f'{BASE}/bangladesh_groundwater_arsenic_1807samples.csv'
TABLE23_CSV = f'{BASE}/tables/Table_23_water_type_statistics.csv'
TABLE26_CSV = f'{BASE}/tables/Table_26_province_clustering.csv'
OUT_PNG     = f'{BASE}/figures/H1_combined_figure.png'

# ── Palette ─────────────────────────────────────────────────────────────────
C_BLUE   = '#2166ac'
C_RED    = '#d6604d'
C_GREEN  = '#4dac26'
C_PURPLE = '#762a83'
C_SALMON = '#f4a582'
C_LBLUE  = '#92c5de'
C_TEAL   = '#1b7837'
PALETTE  = [C_BLUE, C_RED, C_GREEN, C_PURPLE, C_SALMON, C_LBLUE, C_TEAL]

# ── Global rcParams ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':           'DejaVu Sans',
    'font.size':             11,
    'axes.labelsize':        13,
    'axes.titlesize':        12,
    'axes.titleweight':      'bold',
    'xtick.labelsize':       11,
    'ytick.labelsize':       11,
    'legend.fontsize':       10,
    'axes.linewidth':        1.0,
    'xtick.major.width':     0.8,
    'ytick.major.width':     0.8,
    'xtick.minor.width':     0.5,
    'ytick.minor.width':     0.5,
    'xtick.major.size':      4,
    'ytick.major.size':      4,
    'figure.facecolor':      'white',
    'axes.facecolor':        '#fafafa',
    'savefig.facecolor':     'white',
})

# ── Load data ────────────────────────────────────────────────────────────────
df  = pd.read_csv(DATA_CSV)
t23 = pd.read_csv(TABLE23_CSV)
t26 = pd.read_csv(TABLE26_CSV)

# ── Ion name map ─────────────────────────────────────────────────────────────
NAME_MAP = {
    'PO43-':  'PO\u2084\u00b3\u207b',
    'HCO3-':  'HCO\u2083\u207b',
    'Fe2+':   'Fe\u00b2\u207a',
    'Mn2+':   'Mn\u00b2\u207a',
    'SO42-':  'SO\u2084\u00b2\u207b',
    'Na+':    'Na\u207a',
    'Ca2+':   'Ca\u00b2\u207a',
    'Mg2+':   'Mg\u00b2\u207a',
    'K+':     'K\u207a',
    'Cl-':    'Cl\u207b',
    'NO3-':   'NO\u2083\u207b',
    'CO3-':   'CO\u2083\u00b2\u207b',
    'Br-':    'Br\u207b',
    'I-':     'I\u207b',
    'Al3+':   'Al\u00b3\u207a',
    'Cu2+':   'Cu\u00b2\u207a',
    'Cr3+':   'Cr\u00b3\u207a',
    'B+':     'B',
    'pH':     'pH',
    'ORP':    'ORP',
    'EC':     'EC',
    'TDS':    'TDS',
    'Depth':  'Depth',
    'DO':     'DO',
    'Temperature': 'Temp.',
}

# ── Figure layout ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    2, 3,
    figsize=(22, 15),
    constrained_layout=True,
    facecolor='white',
)
fig.set_constrained_layout_pads(
    h_pad=0.15, w_pad=0.15, hspace=0.12, wspace=0.12
)

axA, axB, axC = axes[0]
axD, axE, axF = axes[1]


def add_panel_label(ax, letter):
    ax.text(-0.08, 1.02, letter,
            transform=ax.transAxes,
            fontsize=14, fontweight='bold',
            va='bottom', ha='right',
            color='#111111')


def light_grid(ax, axis='both'):
    ax.grid(True, color='grey', alpha=0.18, linewidth=0.55,
            linestyle='-', which='major', axis=axis)
    ax.set_axisbelow(True)


# ═══════════════════════════════════════════════════════════════════════════
# Panel A — Arsenic Concentration Distribution (log histogram)
# ═══════════════════════════════════════════════════════════════════════════
As_vals = df['As'].dropna()
As_vals = As_vals[As_vals > 0]

bins = np.logspace(np.log10(As_vals.min()), np.log10(As_vals.max()), 51)
axA.hist(As_vals, bins=bins, color=C_LBLUE, edgecolor='white',
         linewidth=0.4, alpha=0.88)
axA.set_xscale('log')
light_grid(axA)

who_line = axA.axvline(
    10, color=C_RED, linewidth=1.5, linestyle='--',
    label='WHO 10 \u00b5g/L', zorder=5)
mean_val = As_vals.mean()
mean_line = axA.axvline(
    mean_val, color=C_BLUE, linewidth=1.5, linestyle='--',
    label=f'Mean {mean_val:.1f} \u00b5g/L', zorder=5)

pct_above = (As_vals > 10).mean() * 100
axA.text(0.97, 0.95,
         f'{pct_above:.0f}% exceed WHO 10 \u00b5g/L',
         transform=axA.transAxes, fontsize=11,
         va='top', ha='right',
         color=C_RED, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.7))

axA.set_xlabel('Arsenic (\u00b5g/L)')
axA.set_ylabel('Frequency')
axA.set_title('Arsenic Concentration Distribution')
axA.legend(frameon=True, framealpha=0.9, edgecolor='#cccccc',
           loc='upper left', handlelength=1.5, borderpad=0.5,
           labelspacing=0.35, fontsize=10)
add_panel_label(axA, 'A')


# ═══════════════════════════════════════════════════════════════════════════
# Panel B — Top-10 Spearman Correlations with As (horizontal bars)
# ═══════════════════════════════════════════════════════════════════════════
exclude_cols = {
    'As', 'Sample ID', 'District', 'Upazila', 'Date', 'Season', 'GW',
    'CBE_percent', 'CBE_Quality', 'Latitude', 'Longitude',
    'HCO3_Cl_ratio', 'Na_Cl_ratio', 'Ca_Mg_ratio', 'Fe_Mn_ratio',
    'As_Fe_ratio', 'SO4_Cl_ratio', 'PO4_Cl_ratio', 'Salinity', 'Turbidity',
}
geo_cols = [c for c in df.columns
            if c not in exclude_cols
            and df[c].dtype in (float, int, np.float64, np.int64)]

corrs = {}
for col in geo_cols:
    valid = df[['As', col]].dropna()
    if len(valid) > 10:
        rho, _ = stats.spearmanr(valid['As'], valid[col])
        corrs[col] = rho

corr_s     = pd.Series(corrs).dropna()
top10      = corr_s.abs().nlargest(10).index
top10_vals = corr_s[top10].sort_values()

colors_b  = [C_BLUE if v > 0 else C_RED for v in top10_vals]
labels_b  = [NAME_MAP.get(k, k) for k in top10_vals.index]

bars = axB.barh(labels_b, top10_vals.values, color=colors_b,
                height=0.7, edgecolor='none')
axB.axvline(0, color='#333333', linewidth=1.0, zorder=3)

# Value annotations: inside bar if |val|>0.08 (white bold), else outside
max_abs = top10_vals.abs().max()
for bar, val in zip(bars, top10_vals.values):
    cx = bar.get_x() + bar.get_width() / 2
    cy = bar.get_y() + bar.get_height() / 2
    if abs(val) > 0.08:
        x_text = val * 0.82
        ha = 'right' if val > 0 else 'left'
        axB.text(x_text, cy, f'{val:.3f}',
                 va='center', ha=ha, fontsize=9,
                 color='white', fontweight='bold')
    else:
        offset = 0.005 if val > 0 else -0.005
        ha = 'left' if val > 0 else 'right'
        axB.text(val + offset, cy, f'{val:.3f}',
                 va='center', ha=ha, fontsize=9,
                 color='#222222', fontweight='bold')

light_grid(axB, axis='x')
pad = 0.05
axB.set_xlim(-(max_abs + pad), max_abs + pad)
axB.set_xlabel('Spearman \u03c1 with As')
axB.set_title('Spearman Correlations with Arsenic')
add_panel_label(axB, 'B')


# ═══════════════════════════════════════════════════════════════════════════
# Panel C — PCA Biplot (PC1 vs PC2)
# ═══════════════════════════════════════════════════════════════════════════
pca_cols = ['pH', 'EC', 'TDS', 'ORP', 'HCO3-', 'Cl-', 'SO42-',
            'Na+', 'K+', 'Ca2+', 'Mg2+', 'Fe2+', 'Mn2+', 'PO43-', 'As']
pca_cols = [c for c in pca_cols if c in df.columns]

df_pca = df[pca_cols].dropna()
X      = StandardScaler().fit_transform(df_pca)
pca    = PCA(n_components=2, random_state=42)
scores = pca.fit_transform(X)
ev     = pca.explained_variance_ratio_ * 100

log_as = np.log10(df_pca['As'].values + 0.1)
sc = axC.scatter(scores[:, 0], scores[:, 1],
                 c=log_as, cmap='RdYlBu_r',
                 alpha=0.5, s=15,
                 linewidths=0, rasterized=True)
cb = fig.colorbar(sc, ax=axC, pad=0.03, shrink=0.82)
cb.set_label('log\u2081\u2080[As] (\u00b5g/L)', fontsize=10)
cb.ax.tick_params(labelsize=9)

# Top 5 loading arrows
loadings = pca.components_.T          # (n_features, 2)
load_mag = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
top5_idx = np.argsort(load_mag)[-5:]
scale    = 2.8
for i in top5_idx:
    lx, ly = loadings[i, 0] * scale, loadings[i, 1] * scale
    axC.annotate('', xy=(lx, ly), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='#111111',
                                 lw=1.1, shrinkA=0, shrinkB=0))
    vname = NAME_MAP.get(pca_cols[i], pca_cols[i])
    axC.text(lx * 1.14, ly * 1.14, vname, fontsize=9,
             color='#111111', ha='center', va='center',
             path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

axC.set_xlabel(f'PC1 ({ev[0]:.1f}%)')
axC.set_ylabel(f'PC2 ({ev[1]:.1f}%)')
axC.axhline(0, color='grey', lw=0.5, alpha=0.5)
axC.axvline(0, color='grey', lw=0.5, alpha=0.5)
axC.set_title('PCA Biplot (PC1 vs PC2)')
light_grid(axC)
add_panel_label(axC, 'C')


# ═══════════════════════════════════════════════════════════════════════════
# Panel D — Hydrochemical Facies (pie chart from Table_23)
# ═══════════════════════════════════════════════════════════════════════════
wt_counts = t23.set_index('Water_Type')['Count']
wt_labels = list(wt_counts.index)
wt_sizes  = list(wt_counts.values)
total_wt  = sum(wt_sizes)

pie_colors = PALETTE[:len(wt_labels)]


def autopct_fmt(pct):
    return f'{pct:.0f}%' if pct > 5 else ''


wedges, texts, autotexts = axD.pie(
    wt_sizes, labels=None, autopct=autopct_fmt,
    colors=pie_colors, startangle=90,
    pctdistance=0.75,
    wedgeprops=dict(linewidth=0.8, edgecolor='white'))

for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight('bold')

# Legend below pie
legend_items = []
for i, (label, cnt) in enumerate(zip(wt_labels, wt_sizes)):
    pct_val  = cnt / total_wt * 100
    row      = t23[t23['Water_Type'] == label]
    mean_as  = row['As_Mean'].values[0] if len(row) else 0.0
    legend_items.append(
        mpatches.Patch(
            color=pie_colors[i],
            label=f'{label}\n{pct_val:.0f}% | As={mean_as:.1f} \u00b5g/L'))

axD.legend(handles=legend_items,
           fontsize=8, loc='upper center',
           bbox_to_anchor=(0.5, -0.02),
           ncol=2, frameon=True,
           framealpha=0.9, edgecolor='#cccccc',
           handlelength=1.5, borderpad=0.5,
           labelspacing=0.5, columnspacing=0.8)
axD.set_title('Hydrochemical Facies', fontsize=12, fontweight='bold', pad=6)
add_panel_label(axD, 'D')


# ═══════════════════════════════════════════════════════════════════════════
# Panel E — Eh–pH Stability Domain
# ═══════════════════════════════════════════════════════════════════════════
df_ep     = df[['pH', 'ORP', 'As']].dropna()
log_as_ep = np.log10(df_ep['As'].values + 0.1)

sc2 = axE.scatter(df_ep['pH'], df_ep['ORP'],
                  c=log_as_ep, cmap='RdYlBu_r',
                  alpha=0.5, s=15,
                  linewidths=0, rasterized=True)
cb2 = fig.colorbar(sc2, ax=axE, pad=0.03, shrink=0.82)
cb2.set_label('log\u2081\u2080[As]', fontsize=10)
cb2.ax.tick_params(labelsize=9)

axE.axvline(7,  color='grey', linewidth=1.0, linestyle='--', alpha=0.7, zorder=3)
axE.axhline(0,  color='grey', linewidth=1.0, linestyle='--', alpha=0.7, zorder=3)

axE.set_xlabel('pH')
axE.set_ylabel('ORP (mV)')
axE.set_title('Eh\u2013pH Stability Domain')
light_grid(axE)
add_panel_label(axE, 'E')


# ═══════════════════════════════════════════════════════════════════════════
# Panel F — Geochemical Province Map
# ═══════════════════════════════════════════════════════════════════════════
df_map  = df[['Latitude', 'Longitude', 'As']].copy()
t26_sub = t26[['Latitude', 'Longitude', 'Province']].copy()
df_map  = df_map.merge(t26_sub, on=['Latitude', 'Longitude'], how='left')

# Physiographic province bounding boxes (fallback assignment)
province_bbox = {
    'GBM Delta':        (20.5, 23.5, 87.8, 90.0),
    'Brahmaputra Fl.':  (23.5, 26.8, 88.0, 90.5),
    'Meghna Fl.':       (22.5, 24.5, 90.0, 91.5),
    'Sylhet Basin':     (24.0, 26.0, 91.0, 92.8),
    'Chittagong':       (20.5, 23.5, 91.5, 92.8),
    'Barind Tract':     (24.0, 26.8, 87.8, 89.5),
    'Eastern Hills':    (20.5, 24.0, 91.5, 92.8),
}

prov_names = {
    1: 'Province 1 (Reductive)',
    2: 'Province 2 (Phosphate)',
    3: 'Province 3 (Mixed)',
    4: 'Province 4',
}
df_map['Prov_Label'] = df_map['Province'].map(prov_names)


def assign_bbox(row):
    lat, lon = row['Latitude'], row['Longitude']
    for name, (lat_min, lat_max, lon_min, lon_max) in province_bbox.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return name
    return 'Other'


mask_no_prov = df_map['Prov_Label'].isna()
if mask_no_prov.any():
    df_map.loc[mask_no_prov, 'Prov_Label'] = \
        df_map[mask_no_prov].apply(assign_bbox, axis=1)

unique_provs   = sorted(df_map['Prov_Label'].dropna().unique())
prov_color_map = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(unique_provs)}

df_map['As_clip'] = df_map['As'].clip(upper=60)
point_sizes       = 8 + df_map['As_clip'] / 6

for prov, grp in df_map.groupby('Prov_Label'):
    axF.scatter(grp['Longitude'], grp['Latitude'],
                s=point_sizes[grp.index],
                c=prov_color_map.get(prov, 'grey'),
                alpha=0.6, linewidths=0, rasterized=True,
                label=prov)

# Bangladesh rough outline
bd_lon = [88.0, 92.7, 92.6, 90.0, 88.2, 88.0, 88.0]
bd_lat = [20.7, 20.5, 24.6, 26.6, 26.5, 20.7, 20.7]
axF.plot(bd_lon, bd_lat, color='#222222', linewidth=1.2, alpha=0.9, zorder=5)

axF.set_xlim(87.5, 93.2)
axF.set_ylim(20.0, 27.2)
axF.set_xlabel('Longitude (\u00b0E)')
axF.set_ylabel('Latitude (\u00b0N)')
axF.set_title('Geochemical Province Map')

all_handles, all_labels = axF.get_legend_handles_labels()
max_items = 7
leg = axF.legend(
    all_handles[:max_items], all_labels[:max_items],
    fontsize=9, loc='lower right',
    frameon=True, framealpha=0.9, edgecolor='#cccccc',
    markerscale=1.5, handletextpad=0.5,
    borderpad=0.5, labelspacing=0.35, ncol=1)

light_grid(axF)
add_panel_label(axF, 'F')


# ═══════════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════════
plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

import os
size_kb = os.path.getsize(OUT_PNG) / 1024
print(f'H1 combined figure saved: {OUT_PNG}')
print(f'File size: {size_kb:.1f} KB')
