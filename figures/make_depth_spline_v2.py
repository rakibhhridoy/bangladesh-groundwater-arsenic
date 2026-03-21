"""
H2 Depth Spline Figure — Publication-quality, blue-theme, 2x2 layout.
Replaces the original wide 1x3 layout.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────
BASE     = '/Users/rakibhhridoy/AsGW/GroundWater'
DATA_CSV = f'{BASE}/hypo1/output/data_phase1_integrated.csv'
OUT_PNG  = f'{BASE}/hypo2/output/figures/H2_F_depth_spline.png'

# ── Blue palette (matching H1/H2) ────────────────────────────────────────
C_DARK   = '#053061'
C_PO4    = '#2166ac'   # PO4 line — primary blue
C_FE     = '#92c5de'   # Fe line — light blue
C_CROSS  = '#d6604d'   # crossover marker — accent red
C_FILL_P = '#d1e5f0'   # PO4 dominance fill
C_FILL_F = '#f4a582'   # Fe dominance fill (warm)
C_GRAY   = '#bdbdbd'
C_SCATTER = '#4393c3'

# ── rcParams ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_CSV)
df = df[(df['Latitude'] >= 20.5) & (df['Longitude'] >= 87.8)].copy()
df['As_log'] = np.log10(df['As'].clip(lower=0.01))
needed = ['Depth', 'As', 'As_log', 'PO43-', 'Fe2+']
df = df[needed].dropna()
print(f"Samples: {len(df)}, Depth: {df['Depth'].min():.0f}–{df['Depth'].max():.0f} m")

# ── Rolling partial correlations ──────────────────────────────────────────
def rolling_partial(df, target, pred, cov, window=80, step=5, min_n=20):
    df_sub = df[['Depth', target, pred, cov]].dropna()
    depths = np.arange(df_sub['Depth'].min(), df_sub['Depth'].max() - window, step)
    rows = []
    for d in depths:
        s = df_sub[(df_sub['Depth'] >= d) & (df_sub['Depth'] < d + window)]
        if len(s) < min_n:
            continue
        def resid(y, x):
            m = np.isfinite(y) & np.isfinite(x)
            if m.sum() < 5: return y
            c = np.polyfit(x[m], y[m], 1)
            return y - np.polyval(c, x)
        y_r = resid(s[target].values, s[cov].values)
        x_r = resid(s[pred].values, s[cov].values)
        rho, p = spearmanr(x_r, y_r)
        rows.append({'depth_mid': d + window/2, 'rho': rho, 'p': p, 'n': len(s)})
    return pd.DataFrame(rows)

po4_roll = rolling_partial(df, 'As_log', 'PO43-', 'Fe2+')
fe_roll  = rolling_partial(df, 'As_log', 'Fe2+', 'PO43-')

# Find crossover
merged = po4_roll.merge(fe_roll, on='depth_mid', suffixes=('_po4', '_fe'))
merged['dominance'] = merged['rho_po4'] - merged['rho_fe']
crossovers = []
for i in range(len(merged)-1):
    if merged['dominance'].iloc[i] * merged['dominance'].iloc[i+1] < 0:
        crossovers.append((merged['depth_mid'].iloc[i] + merged['depth_mid'].iloc[i+1]) / 2)
cross_d = crossovers[0] if crossovers else 130.0
print(f"Crossover depth: {cross_d:.0f} m")

# LOWESS
df_s = df.sort_values('Depth')
as_smooth = lowess(df_s['As_log'].values, df_s['Depth'].values, frac=0.3)

# ── Figure: 2x2 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.subplots_adjust(hspace=0.32, wspace=0.30)

# ═══════════════════════════════════════════════════════════════════════════
# Panel A: Rolling partial correlations vs depth
# ═══════════════════════════════════════════════════════════════════════════
ax = axes[0, 0]
ax.plot(po4_roll['rho'], po4_roll['depth_mid'], 'o-',
        color=C_PO4, ms=5, lw=2.0, label='PO4 partial rho', zorder=3)
ax.plot(fe_roll['rho'], fe_roll['depth_mid'], 's-',
        color=C_FE, ms=5, lw=2.0, label='Fe2+ partial rho', zorder=3)
ax.axvline(0, color=C_GRAY, lw=0.8, ls=':')

# Crossover annotation — place text to the right of the line, vertically centered
ax.axhline(cross_d, color=C_CROSS, lw=2.0, ls='--', alpha=0.8, zorder=2)
ax.annotate(f'Crossover ~{cross_d:.0f} m', xy=(-0.10, cross_d),
            fontsize=10, color=C_CROSS, fontweight='bold',
            ha='left', va='bottom',
            xytext=(0, -14), textcoords='offset points')

# Zone boundaries
for bd, lbl in [(30, '30 m'), (80, '80 m'), (150, '150 m')]:
    ax.axhline(bd, color=C_GRAY, lw=1.0, ls='--', alpha=0.4)

ax.set_xlabel('Partial Spearman rho with log(As)')
ax.set_ylabel('Well Depth (m)')
ax.invert_yaxis()
ax.set_title('Rolling Partial Correlations')
ax.legend(loc='center right', framealpha=0.9)
ax.set_xlim(-0.15, 0.55)
ax.text(-0.12, 0.98, '(A)', transform=ax.transAxes,
        fontsize=16, fontweight='bold', va='top')

# ═══════════════════════════════════════════════════════════════════════════
# Panel B: Butterfly / Mirror chart — PO4 right, Fe left
# ═══════════════════════════════════════════════════════════════════════════
ax2 = axes[0, 1]
ms = merged.sort_values('depth_mid')
depths = ms['depth_mid'].values
rho_po4 = ms['rho_po4'].values
rho_fe  = ms['rho_fe'].values

# Smooth with rolling mean for cleaner curves
from scipy.ndimage import uniform_filter1d
smooth_n = 3
rho_po4_s = uniform_filter1d(rho_po4, smooth_n)
rho_fe_s  = uniform_filter1d(rho_fe, smooth_n)

# PO4 fills to the RIGHT of center (positive x)
ax2.fill_betweenx(depths, 0, rho_po4_s, alpha=0.45, color=C_PO4,
                   label='PO4 partial rho', zorder=2)
ax2.plot(rho_po4_s, depths, color=C_PO4, lw=2.0, zorder=3)

# Fe fills to the LEFT of center (negative x = mirrored)
ax2.fill_betweenx(depths, 0, -rho_fe_s, alpha=0.45, color=C_FE,
                   label='Fe2+ partial rho', zorder=2)
ax2.plot(-rho_fe_s, depths, color=C_FE, lw=2.0, zorder=3)

# Central axis
ax2.axvline(0, color='#333333', lw=1.5, zorder=4)

# Crossover line
ax2.axhline(cross_d, color=C_CROSS, lw=2.0, ls='--', alpha=0.8, zorder=5)
ax2.annotate(f'Crossover ~{cross_d:.0f} m', xy=(0.0, cross_d),
             fontsize=10, color=C_CROSS, fontweight='bold',
             ha='center', va='bottom',
             xytext=(0, -12), textcoords='offset points')

# Zone boundary lines
for bd in [30, 80, 150]:
    ax2.axhline(bd, color=C_GRAY, lw=0.8, ls=':', alpha=0.4)

# Side labels
ax2.text(0.78, 0.08, 'PO4', transform=ax2.transAxes,
         fontsize=13, color=C_PO4, fontweight='bold', ha='center', va='bottom')
ax2.text(0.22, 0.92, 'Fe', transform=ax2.transAxes,
         fontsize=13, color='#5a8fad', fontweight='bold', ha='center', va='top')

# Arrow annotations for direction
ax2.annotate('', xy=(0.35, depths.min()+10), xytext=(0.05, depths.min()+10),
             arrowprops=dict(arrowstyle='->', color=C_PO4, lw=1.5))
ax2.annotate('', xy=(-0.25, depths.max()-10), xytext=(-0.05, depths.max()-10),
             arrowprops=dict(arrowstyle='->', color='#5a8fad', lw=1.5))

ax2.set_xlabel('Partial rho (mirrored)', fontsize=12)
ax2.set_ylabel('Well Depth (m)')
ax2.invert_yaxis()
ax2.set_title('Mechanism Butterfly: PO4 vs Fe')
xmax = max(rho_po4_s.max(), rho_fe_s.max()) * 1.3
ax2.set_xlim(-xmax, xmax)
ax2.legend(loc='center right', framealpha=0.9, fontsize=9.5)
ax2.text(-0.12, 0.98, '(B)', transform=ax2.transAxes,
         fontsize=16, fontweight='bold', va='top')

# ═══════════════════════════════════════════════════════════════════════════
# Panel C: LOWESS arsenic vs depth
# ═══════════════════════════════════════════════════════════════════════════
ax3 = axes[1, 0]
ax3.scatter(df_s['Depth'], df_s['As_log'], s=8, alpha=0.15,
            color=C_SCATTER, edgecolors='none', zorder=1, label='Individual wells')
ax3.plot(as_smooth[:, 0], as_smooth[:, 1], '-', color=C_CROSS, lw=2.5,
         zorder=3, label='LOWESS trend')

# Zone boundaries
for bd in [30, 80, 150]:
    ax3.axvline(bd, color=C_GRAY, lw=1.0, ls='--', alpha=0.5)

# WHO line
who_log = np.log10(10)
ax3.axhline(who_log, color='red', lw=1.5, ls=':', alpha=0.6, label='WHO 10 µg/L')

ax3.set_xlabel('Well Depth (m)')
ax3.set_ylabel('log10(As) [ug/L]')
ax3.set_title('Arsenic vs Continuous Depth')
ax3.legend(loc='lower left', framealpha=0.9, fontsize=10)
ax3.text(-0.12, 0.98, '(C)', transform=ax3.transAxes,
         fontsize=16, fontweight='bold', va='top')

# ═══════════════════════════════════════════════════════════════════════════
# Panel D: Zone-averaged partial correlations (paired dot plot)
# ═══════════════════════════════════════════════════════════════════════════
ax4 = axes[1, 1]

zones = ['Shallow\n(<30 m)', 'Intermediate\n(30–80 m)',
         'Medium-Deep\n(80–150 m)', 'Deep\n(>150 m)']
zone_bins = [0, 30, 80, 150, 999]
df['zone'] = pd.cut(df['Depth'], bins=zone_bins, labels=zones)

zone_data = []
for z in zones:
    sub = df[df['zone'] == z]
    if len(sub) < 10:
        zone_data.append((z, np.nan, np.nan, np.nan, np.nan))
        continue
    def resid(y, x):
        m = np.isfinite(y) & np.isfinite(x)
        if m.sum() < 5: return y
        c = np.polyfit(x[m], y[m], 1)
        return y - np.polyval(c, x)
    yr = resid(sub['As_log'].values, sub['Fe2+'].values)
    xr = resid(sub['PO43-'].values, sub['Fe2+'].values)
    rho_po4, p_po4 = spearmanr(xr, yr)
    yr2 = resid(sub['As_log'].values, sub['PO43-'].values)
    xr2 = resid(sub['Fe2+'].values, sub['PO43-'].values)
    rho_fe, p_fe = spearmanr(xr2, yr2)
    zone_data.append((z, rho_po4, p_po4, rho_fe, p_fe))

y_pos = np.arange(len(zones))
po4_vals = [d[1] for d in zone_data]
fe_vals  = [d[3] for d in zone_data]

# Connecting lines
for i in range(len(zones)):
    ax4.plot([po4_vals[i], fe_vals[i]], [y_pos[i], y_pos[i]],
             color=C_GRAY, lw=1.5, zorder=1)

# Dots
ax4.scatter(po4_vals, y_pos, s=120, color=C_PO4, zorder=3, edgecolors='white',
            linewidth=1.5, label='PO4 partial rho')
ax4.scatter(fe_vals, y_pos, s=120, color=C_FE, zorder=3, edgecolors='white',
            linewidth=1.5, label='Fe2+ partial rho')

# Significance annotations
for i, (z, rp, pp, rf, pf) in enumerate(zone_data):
    sig_p = '***' if pp < 0.001 else '**' if pp < 0.01 else '*' if pp < 0.05 else 'ns'
    sig_f = '***' if pf < 0.001 else '**' if pf < 0.01 else '*' if pf < 0.05 else 'ns'
    ax4.annotate(f'{rp:.3f}{sig_p}', (rp, y_pos[i]),
                 textcoords='offset points', xytext=(0, 12),
                 fontsize=9, color=C_PO4, ha='center', fontweight='bold')
    ax4.annotate(f'{rf:.3f}{sig_f}', (rf, y_pos[i]),
                 textcoords='offset points', xytext=(0, -15),
                 fontsize=9, color='#5a93b5', ha='center')

ax4.axvline(0, color=C_GRAY, lw=0.8, ls=':')
ax4.set_yticks(y_pos)
ax4.set_yticklabels(zones)
ax4.set_xlabel('Partial Spearman rho with log(As)')
ax4.set_title('Zone-Averaged Mechanism Comparison')
ax4.legend(loc='lower right', framealpha=0.9)
ax4.invert_yaxis()
ax4.set_xlim(-0.1, 0.5)
ax4.set_ylim(len(zones) - 1 + 0.6, -0.6)  # add padding top and bottom (y inverted)
ax4.text(-0.12, 0.98, '(D)', transform=ax4.transAxes,
         fontsize=16, fontweight='bold', va='top')

# ── Save ──────────────────────────────────────────────────────────────────
fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {OUT_PNG}")
plt.close()
