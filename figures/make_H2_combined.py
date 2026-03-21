"""
H2 Combined 6-panel publication figure — fully rewritten for STOTEN/EST quality.
figsize=(22,15), DPI=300, constrained_layout, all font/line specs per TARGET SPECS.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import linregress
import os

# ── Paths ───────────────────────────────────────────────────────────────────
BASE     = '/Users/rakibhhridoy/AsGW/GroundWater'
TABLES   = f'{BASE}/hypo2/output/tables'
MAIN_CSV = f'{BASE}/hypo1/output/data_phase1_integrated.csv'
OUT_PNG  = f'{BASE}/hypo2/output/figures/H2_combined_figure.png'

# ── Palette & zone order ─────────────────────────────────────────────────────
PAL         = ['#2166ac', '#d6604d', '#4dac26', '#762a83']
ZONE_ORDER  = ['Shallow', 'Intermediate', 'Medium_Deep', 'Deep']
ZONE_SHORT  = ['Shallow', 'Interm.', 'Med-Deep', 'Deep']
ZONE_COLORS = dict(zip(ZONE_ORDER, PAL))

# ── Global rcParams ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'axes.facecolor'   : 'white',
    'figure.facecolor' : 'white',
    'axes.grid'        : True,
    'grid.color'       : '#cccccc',
    'grid.alpha'       : 0.25,
    'grid.linewidth'   : 0.5,
    'axes.labelsize'   : 13,   # axes labels 13 pt
    'xtick.labelsize'  : 11,   # tick labels 11 pt
    'ytick.labelsize'  : 11,
    'legend.fontsize'  : 10,   # legend 10 pt
    'axes.linewidth'   : 1.0,  # axis spine linewidth 1.0
    'xtick.major.width': 0.9,
    'ytick.major.width': 0.9,
})

# ── Load main data ────────────────────────────────────────────────────────────
df = pd.read_csv(MAIN_CSV)

def assign_zone(depth):
    if depth <= 30:    return 'Shallow'
    elif depth <= 80:  return 'Intermediate'
    elif depth <= 150: return 'Medium_Deep'
    else:              return 'Deep'

df['Depth_Zone'] = df['Depth'].apply(assign_zone)
df['Depth_Zone'] = pd.Categorical(df['Depth_Zone'], categories=ZONE_ORDER, ordered=True)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(22, 15), constrained_layout=True)
fig.patch.set_facecolor('white')
axes = axes.flatten()   # A=0  B=1  C=2  D=3  E=4  F=5

def add_panel_label(ax, letter):
    """Bold panel letter, 14 pt, slightly outside top-left corner."""
    ax.text(-0.10, 1.05, letter, transform=ax.transAxes,
            fontsize=14, fontweight='bold', ha='left', va='top')

# ════════════════════════════════════════════════════════════════════════════
# Panel A — Arsenic by Depth Zone (boxplot + log scale + WHO line)
# ════════════════════════════════════════════════════════════════════════════
ax = axes[0]

who_df  = pd.read_csv(f'{TABLES}/H2_T04_who_exceedance_by_zone.csv')
who_pct = dict(zip(who_df['Depth_Zone'], who_df['Pct_above_WHO']))

data_by_zone = [df[df['Depth_Zone'] == z]['As'].dropna().values for z in ZONE_ORDER]

bp = ax.boxplot(
    data_by_zone,
    patch_artist=True,
    showfliers=True,
    flierprops=dict(marker='o', markersize=2.5, alpha=0.3, linestyle='none'),
    medianprops=dict(color='black', linewidth=2.0),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    boxprops=dict(linewidth=1.5),
    widths=0.55,
)

for patch, z in zip(bp['boxes'], ZONE_ORDER):
    patch.set_facecolor(ZONE_COLORS[z])
    patch.set_alpha(0.75)
for flier, z in zip(bp['fliers'], ZONE_ORDER):
    flier.set(color=ZONE_COLORS[z], alpha=0.3)

ax.set_yscale('log')
ax.set_ylim(0.05, 3000)

# WHO line — dashed red, 1.5 pt
ax.axhline(10, color='#cc0000', linestyle='--', linewidth=1.5, label='WHO 10 µg/L', zorder=3)

# x-tick labels with two-line depth ranges, 11 pt
ax.set_xticks(range(1, 5))
ax.set_xticklabels(
    ['Shallow\n(0–30 m)', 'Interm.\n(30–80 m)', 'Med-Deep\n(80–150 m)', 'Deep\n(>150 m)'],
    fontsize=11
)
ax.set_ylabel('Arsenic (µg/L)', fontsize=13)
ax.set_title('Arsenic by Depth Zone', fontsize=12, fontweight='bold', pad=8)

# WHO legend only — upper left
ax.legend(fontsize=10, loc='upper left', framealpha=0.88,
          edgecolor='#bbbbbb', handlelength=1.5)

# % exceedance annotations at y=400 (log scale), bold red 10 pt
for i, z in enumerate(ZONE_ORDER, 1):
    pct = who_pct.get(z, None)
    if pct is not None:
        ax.text(i, 400, f'{pct:.0f}%', ha='center', va='center',
                fontsize=10, color='#cc0000', fontweight='bold')

add_panel_label(ax, 'A')

# ════════════════════════════════════════════════════════════════════════════
# Panel B — ORP vs Depth (scatter per zone + running-mean trend, inverted y)
# ════════════════════════════════════════════════════════════════════════════
ax = axes[1]

for z, zs in zip(ZONE_ORDER, ZONE_SHORT):
    sub = df[df['Depth_Zone'] == z][['ORP', 'Depth']].dropna()
    ax.scatter(sub['ORP'], sub['Depth'], c=ZONE_COLORS[z],
               alpha=0.30, s=8, linewidths=0, label=zs)

# Running mean — 20 m bins, thick black 2 pt
valid = df[['ORP', 'Depth']].dropna().sort_values('Depth')
bins  = np.arange(0, valid['Depth'].max() + 20, 20)
bin_mid, bin_orp = [], []
for b0, b1 in zip(bins[:-1], bins[1:]):
    sub = valid[(valid['Depth'] >= b0) & (valid['Depth'] < b1)]['ORP']
    if len(sub) >= 5:
        bin_mid.append((b0 + b1) / 2)
        bin_orp.append(sub.mean())

ax.plot(bin_orp, bin_mid, color='black', linewidth=2.0, linestyle='-',
        label='Running mean', zorder=5)

ax.axvline(0, color='#555555', linestyle=':', linewidth=1.0)
ax.invert_yaxis()

ax.set_xlabel('ORP (mV)', fontsize=13)
ax.set_ylabel('Depth (m)', fontsize=13)
ax.set_title('ORP–Depth Profile', fontsize=12, fontweight='bold', pad=8)

# Legend: 4 zones + running mean, lower left
ax.legend(fontsize=10, loc='lower left', framealpha=0.88,
          edgecolor='#bbbbbb', handlelength=1.5, labelspacing=0.4)

# Spearman annotation — lower right, italic 10 pt
ax.text(0.97, 0.04, 'ρ=−0.077, p=0.001',
        transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
        style='italic', color='#333333')

add_panel_label(ax, 'B')

# ════════════════════════════════════════════════════════════════════════════
# Panel C — Mechanism Shift Heatmap (2 × 4: PO4 & Fe x 4 zones)
# ════════════════════════════════════════════════════════════════════════════
ax = axes[2]

pc_df     = pd.read_csv(f'{TABLES}/H2_T19_partial_corr_by_zone.csv')
mechs_map = {'Phosphate_Competition': 'PO₄³⁻', 'Reductive_Dissolution': 'Fe²⁺'}
plot_mechs = list(mechs_map.keys())

pc_filt = pc_df[pc_df['Depth_Zone'].isin(ZONE_ORDER) &
                pc_df['Mechanism'].isin(plot_mechs)].copy()

hm_data  = np.full((2, 4), np.nan)
sig_data = np.full((2, 4), '', dtype=object)
for mi, mech in enumerate(plot_mechs):
    for zi, zone in enumerate(ZONE_ORDER):
        row = pc_filt[(pc_filt['Mechanism'] == mech) & (pc_filt['Depth_Zone'] == zone)]
        if not row.empty:
            hm_data[mi, zi]  = row['Partial_rho'].values[0]
            sig_data[mi, zi] = '*' if row['Significant'].values[0] == 'YES' else ''

norm = TwoSlopeNorm(vmin=-0.4, vcenter=0, vmax=0.4)
im   = ax.imshow(hm_data, cmap='RdBu_r', norm=norm, aspect='auto')

cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.ax.tick_params(labelsize=10)
cb.set_label('Partial ρ with As', fontsize=11)

ax.set_xticks(range(4))
ax.set_xticklabels(ZONE_SHORT, fontsize=11)
ax.set_yticks(range(2))
ax.set_yticklabels([mechs_map[m] for m in plot_mechs], fontsize=12)
ax.set_title('Mechanism Shift Heatmap (~80 m crossover)', fontsize=12, fontweight='bold', pad=8)

# Cell annotations — 10 pt bold; white if |val| > 0.22 else black
for mi in range(2):
    for zi in range(4):
        val = hm_data[mi, zi]
        if not np.isnan(val):
            txt   = f'{val:.3f}{sig_data[mi, zi]}'
            color = 'white' if abs(val) > 0.22 else 'black'
            ax.text(zi, mi, txt, ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')

add_panel_label(ax, 'C')

# ════════════════════════════════════════════════════════════════════════════
# Panel D — Variance Partitioning (stacked bar)
# ════════════════════════════════════════════════════════════════════════════
ax = axes[3]

vp_df    = pd.read_csv(f'{TABLES}/H2_T17_variance_partitioning.csv')
vp_zones = vp_df[vp_df['zone'].isin(ZONE_ORDER)].copy()
vp_zones['zone'] = pd.Categorical(vp_zones['zone'], categories=ZONE_ORDER, ordered=True)
vp_zones = vp_zones.sort_values('zone')

geo_pct    = vp_zones['pure_geogenic'].values * 100
anthro_pct = vp_zones['pure_anthropogenic'].values * 100
shared_pct = vp_zones['joint_shared'].values * 100
x = np.arange(len(vp_zones))
bar_w = 0.55

ax.bar(x, geo_pct,    bar_w, label='Geogenic',      color='#2166ac', alpha=0.85)
ax.bar(x, anthro_pct, bar_w, bottom=geo_pct,
       label='Anthropogenic', color='#d6604d', alpha=0.85)
ax.bar(x, shared_pct, bar_w, bottom=geo_pct + anthro_pct,
       label='Shared', color='#92c5de', alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(ZONE_SHORT, fontsize=11)
ax.set_ylabel('Variance Explained (%)', fontsize=13)
ax.set_title('Variance Partitioning by Depth', fontsize=12, fontweight='bold', pad=8)
ax.legend(fontsize=10, loc='upper right', framealpha=0.88, edgecolor='#bbbbbb')
ax.set_ylim(0, ax.get_ylim()[1] * 1.20)

add_panel_label(ax, 'D')

# ════════════════════════════════════════════════════════════════════════════
# Panel E — PO₄ vs As log-log scatter + regression per zone
# ════════════════════════════════════════════════════════════════════════════
ax = axes[4]

rho_by_zone = {}
for z in ZONE_ORDER:
    row = pc_df[(pc_df['Depth_Zone'] == z) & (pc_df['Mechanism'] == 'Phosphate_Competition')]
    if not row.empty:
        rho_by_zone[z] = row['Partial_rho'].values[0]

rho_lines = []
for z, zs in zip(ZONE_ORDER, ZONE_SHORT):
    sub = df[df['Depth_Zone'] == z][['PO43-', 'As']].dropna()
    sub = sub[(sub['PO43-'] > 0) & (sub['As'] > 0)]

    # Scatter — s=18 per spec (larger markers)
    ax.scatter(sub['PO43-'], sub['As'], c=ZONE_COLORS[z],
               alpha=0.40, s=18, linewidths=0)

    # Regression line — 2 pt
    if len(sub) > 5:
        lx, ly = np.log10(sub['PO43-']), np.log10(sub['As'])
        slope, intercept, *_ = linregress(lx, ly)
        xi = np.linspace(lx.min(), lx.max(), 80)
        ax.plot(10**xi, 10**(slope * xi + intercept),
                color=ZONE_COLORS[z], linewidth=2.0, alpha=0.9,
                label=zs)

    if z in rho_by_zone:
        rho_lines.append(f'{zs}: ρ={rho_by_zone[z]:.3f}')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('PO₄³⁻ (mg/L)', fontsize=13)
ax.set_ylabel('As (µg/L)', fontsize=13)
ax.set_title('Phosphate–Arsenic Association', fontsize=12, fontweight='bold', pad=8)

# Legend: zone names only, 4 items, upper left
ax.legend(fontsize=10, loc='upper left', framealpha=0.88,
          edgecolor='#bbbbbb', handlelength=1.5, labelspacing=0.4)

# ρ text block — lower right, italic 10 pt
rho_text = 'Partial ρ:\n' + '\n'.join(rho_lines)
ax.text(0.98, 0.03, rho_text, transform=ax.transAxes,
        fontsize=10, ha='right', va='bottom', color='#333333',
        style='italic',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.80, ec='#cccccc'))

add_panel_label(ax, 'E')

# ════════════════════════════════════════════════════════════════════════════
# Panel F — Arsenic by Physiographic Zone (horizontal bar, sorted ascending)
# ════════════════════════════════════════════════════════════════════════════
ax = axes[5]

phys_df = pd.read_csv(f'{TABLES}/H2_T11_physiographic_zone_stats.csv')
phys_df = phys_df[phys_df['Phys_Zone'] != 'Unclassified'].copy()
phys_df = phys_df.sort_values('As_mean', ascending=True).reset_index(drop=True)

def trim_zone(name, maxlen=15):
    s = name.replace('_', ' ')
    return s if len(s) <= maxlen else s[:maxlen].rstrip()

colors_phys = ['#d6604d' if m > 10 else '#2166ac' for m in phys_df['As_mean']]
se = phys_df['As_std'] / np.sqrt(phys_df['As_count'])

ax.barh(range(len(phys_df)), phys_df['As_mean'],
        xerr=se, color=colors_phys, alpha=0.82, height=0.65,
        error_kw=dict(elinewidth=1.2, capsize=3, ecolor='#444444'))

# WHO line — dashed red 1.5 pt
ax.axvline(10, color='#cc0000', linestyle='--', linewidth=1.5)

ax.set_yticks(range(len(phys_df)))
ax.set_yticklabels([trim_zone(z) for z in phys_df['Phys_Zone']], fontsize=10)
ax.set_xlabel('Mean As (µg/L)', fontsize=13)
ax.set_title('Arsenic by Physiographic Zone', fontsize=12, fontweight='bold', pad=8)

# 2-patch legend
red_patch  = mpatches.Patch(color='#d6604d', alpha=0.82, label='Mean > WHO 10')
blue_patch = mpatches.Patch(color='#2166ac', alpha=0.82, label='Mean ≤ WHO 10')
ax.legend(handles=[red_patch, blue_patch], fontsize=10,
          loc='lower right', framealpha=0.88, edgecolor='#bbbbbb')

add_panel_label(ax, 'F')

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

sz_kb = os.path.getsize(OUT_PNG) / 1024
print(f"H2 combined figure saved.")
print(f"Output : {OUT_PNG}")
print(f"Size   : {sz_kb:.1f} KB")
