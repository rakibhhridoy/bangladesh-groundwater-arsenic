#!/usr/bin/env python3
"""
High-quality (vector PDF + SVG + 600 dpi PNG) regeneration of all four main
figures for the JHM revision. Summary panels are drawn from the canonical saved
result tables; continuous panels from the cleaned dataset.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import rankdata
import statsmodels.api as sm
from pathlib import Path
import figstyle as fs

fs.set_style()
import os
ROOT = Path(os.environ.get('PAPER1_ROOT', Path(__file__).resolve().parents[3]))
HERE = Path(__file__).resolve().parent
H2D, H3D = ROOT/'src/hypo2/output/tables', ROOT/'src/hypo3/output/tables'  # written by src/h2_depth and src/h3_seasonal
df = pd.read_csv(ROOT/'data/integrated/bangladesh_groundwater_arsenic_1807samples_CLEAN.csv')

ZORD = ['Shallow', 'Intermediate', 'Medium_Deep', 'Deep']
ZSHORT = ['Shallow', 'Inter.', 'Med-Deep', 'Deep']
ZSUB = ['<30 m', '30–80 m', '80–150 m', '>150 m']
def zlabels(): return [f'{a}\n{b}' for a, b in zip(ZSHORT, ZSUB)]

def style_bar_labels(ax, bars, vals, fmt='{:.0f}', dy=3, fs_=7.5, color=fs.INK):
    for b, v in zip(bars, vals):
        ax.annotate(fmt.format(v), (b.get_x()+b.get_width()/2, b.get_height()),
                    textcoords='offset points', xytext=(0, dy), ha='center',
                    fontsize=fs_, color=color)

# =====================================================================
# FIGURE 1 (H1)
# =====================================================================
def fig_h1():
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import AgglomerativeClustering
    from scipy.stats import spearmanr
    fig = plt.figure(figsize=(7.2, 6.1), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    ION = {'PO43-':r'PO$_4^{3-}$','HCO3-':r'HCO$_3^-$','Fe2+':r'Fe$^{2+}$','Mn2+':r'Mn$^{2+}$',
           'SO42-':r'SO$_4^{2-}$','Cl-':r'Cl$^-$','Ca2+':r'Ca$^{2+}$','Mg2+':r'Mg$^{2+}$',
           'Na+':r'Na$^+$','K+':r'K$^+$','NO3-':r'NO$_3^-$','B+':'B','Br-':r'Br$^-$',
           'I-':r'I$^-$','Al3+':r'Al$^{3+}$','Cu2+':r'Cu$^{2+}$','Cr3+':r'Cr$^{3+}$','CO3-':r'CO$_3^{2-}$'}
    rho = {c: spearmanr(df['As'], df[c], nan_policy='omit')[0] for c in ION}
    rho = dict(sorted(rho.items(), key=lambda kv: kv[1], reverse=True))

    # (A) arsenic distribution
    axA = fs.clean(fig.add_subplot(gs[0, 0]))
    As = df['As'].replace(0, np.nan).dropna()
    hh, _, _ = axA.hist(np.log10(As), bins=42, color=fs.BAR, edgecolor='white', linewidth=0.4)
    axA.set_ylim(0, hh.max()*1.20)
    for x, lab, c, ha in [(1, 'WHO 10', fs.INK, 'right'), (np.log10(50), 'BD 50', fs.WARM2, 'left')]:
        axA.axvline(x, color=c, ls='--', lw=1.4)
        axA.text(x + (-0.09 if ha == 'right' else 0.09), hh.max()*1.13, lab, color=c,
                 fontsize=7.2, ha=ha, va='center')
    axA.set_xlabel(r'log$_{10}$ arsenic ($\mu$g L$^{-1}$)'); axA.set_ylabel('Number of samples')
    fs.panel_tag(axA, 'A')

    # (B) Spearman correlations
    axB = fs.clean(fig.add_subplot(gs[0, 1])); axB.grid(axis='y', visible=False)
    vals = list(rho.values()); labs = [ION[c] for c in rho]
    cols = [fs.PO4 if c=='PO43-' else fs.HCO3 if c=='HCO3-' else fs.FE if c=='Fe2+' else fs.MUTE for c in rho]
    axB.barh(range(len(vals)), vals, color=cols, edgecolor=fs.INK, linewidth=0.3,
             hatch=fs.HATCH_1)
    axB.set_yticks(range(len(vals))); axB.set_yticklabels(labs, fontsize=7.5)
    axB.invert_yaxis(); axB.axvline(0, color=fs.INK, lw=0.7)
    axB.set_xlabel(r'Spearman $\rho$ with arsenic')
    for i, c in enumerate(rho):
        if c in ('PO43-', 'HCO3-', 'Fe2+'):
            axB.text(vals[i]+0.006, i, f'{vals[i]:.3f}', va='center', fontsize=7.5, fontweight='bold')
    fs.panel_tag(axB, 'B')

    # (C) PO4/Fe vs As scatter
    axC = fs.clean(fig.add_subplot(gs[1, 0]))
    for col, c, lab in [('PO43-', fs.PO4, r'PO$_4^{3-}$'), ('Fe2+', fs.FE, r'Fe$^{2+}$')]:
        x, y = np.log10(df[col].replace(0, np.nan)), np.log10(df['As'].replace(0, np.nan))
        m = x.notna() & y.notna()
        axC.scatter(x[m], y[m], s=6, alpha=0.18, color=c, edgecolors='none', rasterized=True)
        b, a = np.polyfit(x[m], y[m], 1); xs = np.linspace(x[m].min(), x[m].max(), 50)
        axC.plot(xs, a+b*xs, color=c, lw=2.3, label=f'{lab}  '+r'$\rho$='+f'{rho[col]:.3f}')
    axC.set_xlabel(r'log$_{10}$ concentration (mg L$^{-1}$)')
    axC.set_ylabel(r'log$_{10}$ arsenic ($\mu$g L$^{-1}$)')
    axC.legend(loc='lower right'); fs.panel_tag(axC, 'C')

    # (D) interpolated phosphate surface, the spatial companion to Fig. 1
    axD = fs.clean(fig.add_subplot(gs[1, 1])); axD.grid(False)
    import geopandas as gpd
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    from shapely.geometry import Point
    from matplotlib.colors import LinearSegmentedColormap
    ROOT = Path(__file__).resolve().parent.parent
    bgd = gpd.read_file(ROOT / 'data' / 'raw' / 'shapefiles' / 'bangladesh.geojson')
    wp = (df[(df['PO43-'] > 0)].groupby(['Longitude', 'Latitude'])['PO43-']
            .median().reset_index())
    wp.columns = ['lon', 'lat', 'po4']
    minx, miny, maxx, maxy = bgd.total_bounds
    gx, gy = np.mgrid[minx:maxx:260j, miny:maxy:260j]
    gz = griddata((wp.lon, wp.lat), np.log10(wp.po4), (gx, gy), method='linear')
    msk = np.isnan(gz)
    gz = gaussian_filter(np.where(msk, np.nanmean(gz), gz), sigma=4.0)
    gz = np.where(msk, np.nan, gz)
    poly = bgd.union_all()
    inside = gpd.GeoSeries([Point(a, b) for a, b in zip(gx.ravel(), gy.ravel())],
                           crs=bgd.crs).within(poly).values.reshape(gx.shape)
    gz = np.where(inside, gz, np.nan)
    # the shared magnitude ramp: distinct from the warm arsenic surface of Fig. 1
    # and from Fe's identity blue, which panels B and C use in this same figure
    pcmap = LinearSegmentedColormap.from_list('po4_warm', fs.WARMR)
    LEVP = np.log10([0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0])
    cfp = axD.contourf(gx, gy, gz, levels=LEVP, cmap=pcmap, extend='both', zorder=1)
    axD.contour(gx, gy, gz, levels=[np.log10(1.03)], colors=[fs.CONTOUR],
                linewidths=1.2, zorder=3)
    bgd.boundary.plot(ax=axD, color=fs.INK, lw=0.7, zorder=4)
    axD.scatter(wp.lon, wp.lat, s=1.6, c='#3a3a37', alpha=0.5, lw=0, zorder=5)
    axD.set_xlim(minx-0.08, maxx+0.08); axD.set_ylim(miny-0.08, maxy+0.08)
    axD.set_aspect(1/np.cos(np.radians(23.7)))
    axD.set_xticks([88, 90, 92]); axD.set_yticks([21, 23, 25])
    axD.tick_params(labelsize=7.5)
    axD.set_xlabel('longitude (' + u'\N{DEGREE SIGN}' + 'E)', fontsize=8)
    axD.set_ylabel('latitude (' + u'\N{DEGREE SIGN}' + 'N)', fontsize=8)
    cbp = fig.colorbar(cfp, ax=axD, fraction=0.040, pad=0.02, ticks=LEVP)
    cbp.ax.set_yticklabels(['0.1', '0.25', '0.5', '1', '2', '3', '4', '6'], fontsize=7)
    cbp.set_label(r'PO$_4^{3-}$ (mg L$^{-1}$, log scale)', fontsize=7.5)
    cbp.outline.set_visible(False)
    axD.plot([], [], color=fs.CONTOUR, lw=1.2, label='national median 1.03')
    axD.legend(loc='lower left', fontsize=6.6, frameon=True, framealpha=0.9,
               edgecolor='none', borderpad=0.35)
    fs.panel_tag(axD, 'D')
    return fig

# =====================================================================
# helpers for H2/H3 from saved tables
# =====================================================================
t01 = pd.read_csv(H2D/'H2_T01_depth_zone_summary.csv').set_index('Depth_Zone').reindex(ZORD)
t19 = pd.read_csv(H2D/'H2_T19_partial_corr_by_zone.csv')
pc = t19[t19.Variable.isin(['PO43-','Fe2+'])].pivot_table(index='Depth_Zone', columns='Variable', values='Partial_rho').reindex(ZORD)
t11 = pd.read_csv(H2D/'H2_T11_physiographic_zone_stats.csv')
t11 = t11[t11.Phys_Zone != 'Unclassified'].sort_values('As_mean', ascending=False)
t16 = pd.read_csv(H2D/'H2_T16_seasonal_effect_by_zone.csv').set_index('Depth_Zone').reindex(ZORD)

def fig_h2():
    fig = plt.figure(figsize=(7.2, 7.4), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.85, 0.85, 1.25])
    # (A) mean arsenic by depth class -- single axis
    axA = fs.clean(fig.add_subplot(gs[0, 0])); x = np.arange(4)
    axA.bar(x, t01['As_mean'], color=fs.BAR, edgecolor=fs.INK, linewidth=0.5, width=0.66,
            hatch=fs.HATCH_1)
    axA.axhline(10, color=fs.INK, ls='--', lw=1.0)
    axA.set_xlim(-0.62, 4.05)
    axA.text(3.55, 10, 'WHO 10', fontsize=6.6, color=fs.INK, ha='left', va='center',
             bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none'))
    axA.set_ylabel(r'mean As ($\mu$g L$^{-1}$)', fontsize=8)
    axA.set_xticks(x); axA.set_xticklabels([], fontsize=7.5); axA.grid(axis='x', visible=False)
    fs.panel_tag(axA, 'A')
    # (A2) WHO exceedance -- its own panel, shared x
    axA2 = fs.clean(fig.add_subplot(gs[1, 0]))
    axA2.bar(x, t01['Pct_above_WHO'], color=fs.WARM1, edgecolor='white', linewidth=0.5,
             width=0.66, hatch=fs.HATCH_1)
    for xi, pv in zip(x, t01['Pct_above_WHO']):
        axA2.annotate(f'{pv:.1f}', (xi, pv), textcoords='offset points', xytext=(0, 3),
                      fontsize=7.0, color=fs.INK, ha='center')
    axA2.set_ylabel('WHO exceedance (%)', fontsize=8)
    axA2.set_ylim(0, max(t01['Pct_above_WHO'])*1.22)
    axA2.set_xticks(x); axA2.set_xticklabels(zlabels(), fontsize=7.2); axA2.grid(axis='x', visible=False)
    fs.panel_tag(axA2, 'B')
    # (C) the partition holds inside every depth class
    # Replaces the partial-correlation heatmap, whose eight values are already
    # shown as bars in Fig. 5D. This is the paper's central claim and had no figure.
    axB = fs.clean(fig.add_subplot(gs[0:2, 1])); axB.grid(axis='x', visible=False)
    import pandas as _pd
    gdf = _pd.read_csv(ROOT / 'revision_r2' / 'inputs' / 'wells_with_usgs_geology.csv')
    HOL = {'dsd', 'ac', 'dsl', 'ppc', 'dt'}
    OLD = {'afo', 'rb', 'rm', 'QTdd'}
    UNITS = HOL | OLD | {'asl', 'asc', 'asd', 'afy', 'de', 'dm', 'ava', 'csd'}
    gdf = gdf[gdf['GLG'].isin(UNITS)].copy()
    gdf['age'] = np.where(gdf['GLG'].isin(HOL), 0, np.where(gdf['GLG'].isin(OLD), 2, 1))
    gdf['dc'] = _pd.cut(gdf['Depth'], [0, 30, 80, 150, 1e4], labels=[0, 1, 2, 3], right=False)
    gdf = gdf.dropna(subset=['dc', 'As'])
    SER = [(0, 'Holocene deltaic', '#b3261e', fs.HATCH_1),
           (1, 'alluvial / other', '#f2946a', fs.HATCH_1),
           (2, 'Pleistocene + older', '#fbd5bd', fs.HATCH_2)]
    y = np.arange(4); h = 0.26
    for k, (code, lab, col, hat) in enumerate(SER):
        pct, nn = [], []
        for z in range(4):
            w = gdf[(gdf['age'] == code) & (gdf['dc'] == z)]['As']
            pct.append(100*(w > 10).mean() if len(w) else np.nan); nn.append(len(w))
        axB.barh(y + (k-1)*h, pct, h, color=col, edgecolor=fs.INK, linewidth=0.4,
                 hatch=hat, label=lab)
        for yi, v, c in zip(y + (k-1)*h, pct, nn):
            axB.annotate(f'{v:.0f}', (v, yi), textcoords='offset points', xytext=(3, 0),
                         va='center', fontsize=6.4, color=fs.INK)
    axB.set_yticks(y)
    axB.set_yticklabels(['Shallow\n$<$30 m', 'Inter.\n30--80 m',
                         'Med-Deep\n80--150 m', 'Deep\n$>$150 m'], fontsize=7.4)
    axB.set_ylim(3.55, -0.95); axB.set_xlim(0, 68)
    axB.set_xlabel('WHO exceedance (%)', fontsize=8)
    axB.legend(loc='lower right', fontsize=6.8, frameon=True, framealpha=0.92,
               edgecolor='none', borderpad=0.35)
    axB.annotate('Holocene stays high and Pleistocene stays low in every\n'
                 'depth class ($p < 0.01$ throughout), so the partition\n'
                 'is not a depth effect', (0.5, -0.88),
                 xycoords=('axes fraction', 'data'),
                 ha='center', va='center', fontsize=6.5, color=fs.INK, linespacing=1.4)
    fs.panel_tag(axB, 'C')

    # (C) scatter coloured by depth
    axC = fs.clean(fig.add_subplot(gs[2, 0]))
    d = df[(df['PO43-'] > 0) & (df['As'] > 0)]
    lx, ly = np.log10(d['PO43-']), np.log10(d['As'])
    sc = axC.scatter(lx, ly, c=d['Depth'].clip(0, 300), cmap=fs.SEQ2, s=7, alpha=0.5, edgecolors='none', rasterized=True)
    lo = sm.nonparametric.lowess(ly, lx, frac=0.5); axC.plot(lo[:, 0], lo[:, 1], color=fs.PO4, lw=2.4, label='LOWESS')
    axC.set_xlabel(r'log$_{10}$ PO$_4^{3-}$ (mg L$^{-1}$)'); axC.set_ylabel(r'log$_{10}$ As ($\mu$g L$^{-1}$)')
    axC.legend(loc='lower right')
    fig.colorbar(sc, ax=axC, fraction=0.046, pad=0.03, label='depth (m)'); fs.panel_tag(axC, 'D')
    # (D) physiographic zones
    axD = fs.clean(fig.add_subplot(gs[2, 1])); axD.grid(axis='y', visible=False); y = np.arange(len(t11))
    norm = Normalize(t11['As_mean'].min(), t11['As_mean'].max())
    axD.barh(y, t11['As_mean'], color=plt.cm.YlOrRd(norm(t11['As_mean'])),
             edgecolor=fs.INK, linewidth=0.4, hatch=fs.HATCH_1)
    axD.set_yticks(y); axD.set_yticklabels([z.replace('_', ' ') for z in t11['Phys_Zone']], fontsize=8)
    axD.invert_yaxis(); axD.axvline(10, color=fs.INK, ls='--', lw=1.0)
    axD.set_xlabel(r'mean arsenic ($\mu$g L$^{-1}$)', fontsize=8)
    axD.text(10.6, len(t11)-0.4, 'WHO 10', fontsize=6.6, color=fs.INK, ha='left', va='center')
    fs.panel_tag(axD, 'E')
    return fig

def fig_h2_spline():
    others = ['HCO3-', 'SO42-', 'NO3-', 'Cl-', 'Na+', 'Ca2+', 'Mg2+', 'pH', 'ORP']
    fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=True); gs = GridSpec(2, 2, figure=fig)
    dd = df.dropna(subset=['Depth']).sort_values('Depth').assign(As_log=lambda x: np.log10(x['As'].clip(lower=0.1)))
    def roll(sub, pred, cov):
        if len(sub) < 20: return np.nan
        C = np.c_[np.ones(len(sub)), rankdata(sub[cov])]; r = lambda y: y - C@np.linalg.lstsq(C, y, rcond=None)[0]
        return np.corrcoef(r(rankdata(sub['As_log'])), r(rankdata(sub[pred])))[0, 1]
    cen, rp, rf = [], [], []
    for c in np.arange(20, 260, 5):
        w = dd[(dd['Depth'] >= c-40) & (dd['Depth'] < c+40)]
        if len(w) >= 20: cen.append(c); rp.append(roll(w, 'PO43-', 'Fe2+')); rf.append(roll(w, 'Fe2+', 'PO43-'))
    cen, rp, rf = map(np.array, (cen, rp, rf)); s = np.where(rf >= rp)[0]; cross = cen[s[0]] if len(s) else cen[np.argmin(abs(rp-rf))]
    # (A)
    axA = fs.clean(fig.add_subplot(gs[0, 0]))
    axA.plot(rp, cen, color=fs.PO4, lw=2.2, label=r'PO$_4^{3-}$'); axA.plot(rf, cen, color=fs.FE, lw=2.2, label=r'Fe$^{2+}$')
    axA.axhspan(32, 153, color=fs.GREY, alpha=0.16, lw=0, zorder=0,
                label='95% CI 32–153 m')
    axA.axhline(68, color=fs.INK, ls='--', lw=1.1, zorder=1, label='crossover 68 m')
    axA.invert_yaxis(); axA.set_ylim(cen.max()+8, -46)
    axA.set_xlabel(r'partial $\rho$ with log(As)', fontsize=8)
    axA.set_ylabel('depth (m)', fontsize=8)
    axA.legend(loc='upper center', ncol=2, frameon=False, fontsize=7.0,
               columnspacing=1.1, handlelength=1.6)
    fs.panel_tag(axA, 'A')
    # (B) butterfly
    axB = fs.clean(fig.add_subplot(gs[0, 1])); axB.grid(axis='y', visible=False); y = np.arange(4)
    axB.barh(y, -pc['PO43-'], color=fs.PO4, edgecolor='white', linewidth=0.3,
             hatch=fs.HATCH_1, label=r'PO$_4^{3-}$')
    axB.barh(y, pc['Fe2+'], color=fs.FE, edgecolor='white', linewidth=0.3,
             hatch=fs.HATCH_2, label=r'Fe$^{2+}$')
    axB.set_yticks(y); axB.set_yticklabels(ZSHORT, fontsize=8); axB.invert_yaxis(); axB.axvline(0, color=fs.INK, lw=0.8)
    axB.set_xticklabels([f'{abs(v):.2f}' for v in axB.get_xticks()], fontsize=8)
    axB.set_xlabel(r'partial $\rho$   ($\leftarrow$ PO$_4$ | Fe $\rightarrow$)', fontsize=8)
    axB.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=7.5, borderpad=0.4)
    fs.panel_tag(axB, 'B')
    # (C) LOWESS As vs depth
    axC = fs.clean(fig.add_subplot(gs[1, 0])); dp = df[(df['As'] > 0) & df['Depth'].notna()]
    edges = np.array([0,20,40,60,80,100,130,160,200,250,320])
    mid, med, q1, q3, nb = [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        w = dp[(dp['Depth'] >= a) & (dp['Depth'] < b)]['As']
        if len(w) < 15: continue
        mid.append((a+b)/2); med.append(w.median())
        q1.append(w.quantile(.25)); q3.append(w.quantile(.75)); nb.append(len(w))
    mid, med, q1, q3 = map(np.array, (mid, med, q1, q3))
    axC.fill_between(mid, q1, q3, color=fs.PO4, alpha=0.18, lw=0, label='interquartile range')
    axC.plot(mid, med, 'o-', color=fs.PO4, lw=2.0, ms=4.5, label='median')
    axC.axhline(10, color=fs.INK, ls='--', lw=1.0)
    axC.text(316, 11, 'WHO 10', fontsize=6.6, color=fs.INK, ha='right', va='bottom')
    axC.set_yscale('log'); axC.set_xlim(0, 320)
    axC.set_ylim(q1.min()*0.55, q3.max()*3.2)
    for m, k in zip(mid, nb):
        axC.annotate(f'{k}', (m, 0.965), xycoords=('data', 'axes fraction'),
                     fontsize=5.6, color=fs.GREY, ha='center', va='top')
    axC.set_xlabel('depth (m)', fontsize=8)
    axC.set_ylabel(r'As ($\mu$g L$^{-1}$)', fontsize=8)
    axC.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=7.2)
    fs.panel_tag(axC, 'C')
    # (D) zone-avg partials
    axD = fs.clean(fig.add_subplot(gs[1, 1])); axD.grid(axis='x', visible=False); x = np.arange(4); w = 0.38
    axD.bar(x-w/2, pc['PO43-'], w, color=fs.PO4, label=r'PO$_4^{3-}$', edgecolor='white',
            linewidth=0.3, hatch=fs.HATCH_1)
    axD.bar(x+w/2, pc['Fe2+'], w, color=fs.FE, label=r'Fe$^{2+}$', edgecolor='white',
            linewidth=0.3, hatch=fs.HATCH_2)
    axD.set_xticks(x); axD.set_xticklabels(ZSHORT, fontsize=8)
    axD.set_ylabel(r'partial $\rho$', fontsize=8)
    axD.set_ylim(0, max(pc['PO43-'].max(), pc['Fe2+'].max())*1.28)
    axD.legend(loc='upper center', ncol=2, frameon=False, fontsize=7.5)
    fs.panel_tag(axD, 'D')
    return fig

def fig_h3():
    sem = pd.read_csv(H3D/'H3_T_sem_log_indirect_effects.csv').set_index('zone')
    ent = pd.read_csv(H3D/'H3_T03_shannon_entropy_by_zone.csv').set_index('Depth_Zone')
    risk = pd.read_csv(H3D/'H3_T24_56cell_risk_matrix.csv')
    fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=True); gs = GridSpec(2, 2, figure=fig)
    # (A) SEM
    axA = fs.clean(fig.add_subplot(gs[0, 0])); axA.grid(axis='x', visible=False)
    zs = ['ALL', 'Shallow', 'Intermediate', 'Medium_Deep', 'Deep']; s = sem.reindex(zs); x = np.arange(len(zs)); w = 0.38
    axA.bar(x-w/2, s['ie_desorption'].abs(), w, color=fs.PO4, label='desorption',
            edgecolor='white', linewidth=0.3, hatch=fs.HATCH_1)
    axA.bar(x+w/2, s['ie_reductive'].abs(), w, color=fs.FE, label='reductive',
            edgecolor='white', linewidth=0.3, hatch=fs.HATCH_2)
    axA.set_yscale('log'); axA.set_xticks(x); axA.set_xticklabels([z.replace('_', '\n') for z in zs], fontsize=7.5)
    axA.set_ylabel('|standardised indirect effect|')
    axA.legend(loc='upper right'); fs.panel_tag(axA, 'A')
    # (B) risk matrix
    axB = fig.add_subplot(gs[0, 1]); axB.grid(False)
    risk['row'] = risk['Season'].str[0] + '·' + risk['Depth_Zone'].map({'Shallow':'S','Intermediate':'I','Medium_Deep':'M','Deep':'D'})
    piv = risk.pivot_table(index='row', columns='Phys_Zone', values='Exceedance_pct').reindex([f'{s}·{d}' for s in 'DW' for d in 'SIMD'])
    piv.columns = [c.replace('_Floodplain','').replace('_Tract','').replace('_Terrace','').replace('_',' ') for c in piv.columns]
    im = axB.imshow(piv.values, cmap=fs.SEQ, aspect='auto', vmin=0, vmax=80)
    axB.set_xticks(range(len(piv.columns))); axB.set_xticklabels(piv.columns, fontsize=7, rotation=40, ha='right')
    axB.set_yticks(range(len(piv.index))); axB.set_yticklabels(piv.index, fontsize=7)
    axB.axhline(3.5, color='white', lw=1.5)
    fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04, label='% > WHO'); fs.panel_tag(axB, 'B')
    # (C) seasonal As
    axC = fs.clean(fig.add_subplot(gs[1, 0])); axC.grid(axis='x', visible=False); x = np.arange(4); w = 0.38
    axC.bar(x-w/2, t16['Median_As_Dry'], w, color=fs.DRY, label='dry', edgecolor='white',
            linewidth=0.3, hatch=fs.HATCH_1)
    axC.bar(x+w/2, t16['Median_As_Wet'], w, color=fs.WET, label='wet', edgecolor='white',
            linewidth=0.3, hatch=fs.HATCH_2)
    axC.axhline(10, color=fs.PO4, ls='--', lw=1.1)
    for i, z in enumerate(ZORD):
        if str(t16.loc[z, 'Significant']) == 'YES':
            axC.text(i, max(t16.loc[z, ['Median_As_Dry', 'Median_As_Wet']])+0.4, '*', ha='center', fontsize=15, fontweight='bold')
    axC.set_xticks(x); axC.set_xticklabels(ZSHORT, fontsize=8); axC.set_ylabel(r'median As ($\mu$g L$^{-1}$)')
    axC.legend(loc='upper right'); fs.panel_tag(axC, 'C')
    # (D) entropy
    axD = fs.clean(fig.add_subplot(gs[1, 1])); axD.grid(axis='x', visible=False); e = ent.reindex(ZORD); x = np.arange(4); w = 0.38
    axD.bar(x-w/2, e['H_Dry_bits'], w, color=fs.DRY, label='dry', edgecolor='white',
            linewidth=0.3, hatch=fs.HATCH_1)
    axD.bar(x+w/2, e['H_Wet_bits'], w, color=fs.WET, label='wet', edgecolor='white',
            linewidth=0.3, hatch=fs.HATCH_2)
    axD.set_xticks(x); axD.set_xticklabels(ZSHORT, fontsize=8); axD.set_ylabel('Shannon entropy (bits)')
    axD.legend(loc='upper right'); fs.panel_tag(axD, 'D')
    return fig

if __name__ == '__main__':
    for stem, fn in [('H1_combined_figure', fig_h1), ('H2_combined_figure', fig_h2),
                     ('H2_F_depth_spline', fig_h2_spline), ('H3_combined_figure', fig_h3)]:
        out = fs.save_all(fn(), stem, HERE)
        print('wrote', ', '.join(out))
    print('done')
