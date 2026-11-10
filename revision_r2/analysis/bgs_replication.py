"""External replication of Paper 1's core findings in the DPHE/BGS (2001)
National Hydrochemical Survey and Special Study Areas.

Independent agency, independent sampling campaign (1998-2000 vs 2020-2021),
independent laboratory. Data downloaded 2026-08-31 from
https://www.bgs.ac.uk/groundwater/quality/groundwater-and-health/arsenic-contamination-of-groundwater/bangladesh-reports
"""
import numpy as np, pandas as pd, geopandas as gpd
from scipy import stats
from scipy.stats import rankdata
from shapely.geometry import Point

import os
from pathlib import Path
R = os.environ.get("PAPER1_ROOT", str(Path(__file__).resolve().parents[2]))
SHP = f"{R}/data/raw/usgs_ofr97470h"
D = f"{R}/data/external/BGS_DPHE_2001"

LAB = {'dsd':'Deltaic sand','ac':'Chandina alluvium','dsl':'Deltaic silt','ppc':'Marsh clay and peat',
       'dt':'Tidal deltaic','asl':'Alluvial silt','asc':'Alluvial silt + clay','asd':'Alluvial sand',
       'afy':'Young gravelly sand','de':'Estuarine','dm':'Tidal mud','ava':'Valley alluvium',
       'afo':'Old gravelly sand','csd':'Beach / dune sand','rb':'Barind residuum',
       'rm':'Madhupur residuum','QTdd':'Dihing / Dupi Tila'}
HOL = {'dsd','ac','dsl','ppc','dt'}          # Holocene deltaic, as in the manuscript
OLD = {'afo','rb','rm','QTdd'}               # Pleistocene terrace and older

def uncensor(s, mode='recens'):
    """BGS reports non-detects as '< DL', with two different limits per analyte.
    Substitution at DL/2 invents an ordering among values that were never
    distinguished, and because As and P are co-censored in 641 of the same wells
    it inflates rank correlation. The default here is recensoring, the standard
    treatment for multiply-censored data: everything below the HIGHEST detection
    limit is set to one common value so it all ties, which is exactly what the
    measurement supports. 'drop', 'half' and 'dl' are kept for sensitivity."""
    s = s.astype(str).str.strip()
    cen = s.str.startswith('<')
    val = pd.to_numeric(s.str.lstrip('< '), errors='coerce')
    cen = cen & val.notna()
    if mode == 'drop':   return val.where(~cen), cen
    if mode == 'half':   return val.where(~cen, val*0.5), cen
    if mode == 'dl':     return val, cen
    dlmax = val[cen].max() if cen.any() else np.nan
    if np.isnan(dlmax):  return val, cen
    return val.where(val >= dlmax, dlmax*0.5), cen

def load(path):
    d = pd.read_csv(path, skiprows=4, low_memory=False, encoding='latin-1')
    return d[d.SAMPLE_ID.notna() & (d.SAMPLE_ID.astype(str).str.strip() != '')].copy()

def prho(y, x, z):
    """Partial Spearman of y with x controlling z."""
    m = y.notna() & x.notna() & z.notna()
    if m.sum() < 20: return np.nan, m.sum()
    C = np.c_[np.ones(m.sum()), rankdata(z[m])]
    r = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    return np.corrcoef(r(rankdata(y[m])), r(rankdata(x[m])))[0, 1], int(m.sum())

def eps2(groups):
    H, p = stats.kruskal(*groups); k = len(groups); N = sum(len(g) for g in groups)
    return (H-k+1)/(N-k), H, p

def line(*a): print(*a, flush=True)

def parse_n(d, c):
    return int(d[c].astype(str).str.strip().str.startswith('<').sum())

# ======================================================================
line("="*74); line("BGS/DPHE NATIONAL HYDROCHEMICAL SURVEY  (n = 3,534 wells, 1998-2000)"); line("="*74)
nhs = load(f"{D}/BGS_NHS_national_3534.csv")
for c, out in [('As','As'), ('P_','P'), ('Fe','Fe'), ('Mn','Mn')]:
    nhs[out], nhs[out+'_cen'] = uncensor(nhs[c])   # recensored
nhs['Depth'] = pd.to_numeric(nhs['WELL_DEPTH'], errors='coerce')
nhs['PO4'] = nhs['P'] * 3.066            # P (mg/L) -> PO4 (mg/L) for magnitude comparison
line(f"  censored: As {nhs.As_cen.sum()} ({100*nhs.As_cen.mean():.0f}%), "
     f"P {nhs.P_cen.sum()} ({100*nhs.P_cen.mean():.0f}%), Fe {nhs.Fe_cen.sum()}")

line("\n-- TEST 1  national correlate ranking (Spearman rho with As)")
line("   manuscript (BWDB 2020-21): PO4 0.343 | HCO3 0.305 | Fe 0.284")
for v in ['P','Fe','Mn']:
    r, p = stats.spearmanr(nhs['As'], nhs[v], nan_policy='omit')
    line(f"     {v:4s} rho = {r:+.3f}   p = {p:.3g}")
line("\n   sensitivity of As-P to the censoring treatment "
     f"(As censored {nhs.As_cen.sum()}, P censored {nhs.P_cen.sum()}, both {(nhs.As_cen & nhs.P_cen).sum()})")
for m, why in [('drop','uncensored pairs only, drops the informative low-low block'),
               ('half','substitute DL/2, invents ordering among ties'),
               ('dl','substitute DL'),
               ('recens','recensor to the highest DL (used above)')]:
    a, _ = uncensor(nhs['As'], m); b, _ = uncensor(nhs['P_'], m)
    msk = a.notna() & b.notna()
    r, _p = stats.spearmanr(a[msk], b[msk])
    line(f"     {m:7s} rho = {r:+.3f}  n = {msk.sum():5d}   {why}")

line("\n-- TEST 2  deposit-type partition (USGS OFR 97-470H, the manuscript's own layer)")
geo = gpd.read_file(f"{SHP}/geo8bg.shp").set_crs("EPSG:4326", allow_override=True)
pts = gpd.GeoDataFrame(nhs, geometry=[Point(xy) for xy in zip(nhs.LONG_DEG, nhs.LAT_DEG)], crs="EPSG:4326")
j = gpd.sjoin(pts, geo[['GLG','geometry']], how='left', predicate='within')
j = j[~j.index.duplicated()]
nhs['GLG'] = j['GLG']
g = nhs[nhs.GLG.isin(LAB) & nhs.As.notna()].copy()
line(f"   deposit type resolved for {nhs.GLG.isin(LAB).sum()} of {len(nhs)} wells; "
     f"{len(g)} also have arsenic")

rows = []
for k, s in g.groupby('GLG'):
    if len(s) < 15: continue
    rows.append(dict(unit=LAB[k], code=k, n=len(s), pct=100*(s.As > 10).mean(),
                     medAs=s.As.median(), medPO4=s.PO4.median()))
t = pd.DataFrame(rows).sort_values('pct', ascending=False)
line(f"   {'unit':22s} {'n':>5s} {'>10 ug/L':>9s} {'med As':>8s} {'med PO4':>8s}")
for r_ in t.itertuples():
    line(f"     {r_.unit:22s} {r_.n:5d} {r_.pct:8.1f}% {r_.medAs:8.2f} {r_.medPO4:8.2f}")

kept = t.code.tolist()
gg = g[g.GLG.isin(kept)]
e_dep, H_dep, p_dep = eps2([s.As.values for _, s in gg.groupby('GLG')])
bins = [0, 30, 80, 150, 1e4]
gg = gg.assign(dclass=pd.cut(gg.Depth, bins, labels=['Shallow','Inter.','Med-Deep','Deep'], right=False))
dsub = [s.As.values for _, s in gg.dropna(subset=['dclass']).groupby('dclass', observed=True) if len(s) >= 15]
e_dep_d, H_d, p_d = eps2(dsub)
line(f"\n   deposit type : Kruskal-Wallis H = {H_dep:.1f}  eps^2 = {e_dep:.3f}  (p = {p_dep:.2g})")
line(f"   depth class  : Kruskal-Wallis H = {H_d:.1f}  eps^2 = {e_dep_d:.3f}  (p = {p_d:.2g})")
line(f"   ratio        : {e_dep/e_dep_d:.1f}x        manuscript: 238.2 / 16.1, eps^2 0.135 / 0.008, 17.1x")

line("\n-- TEST 3  Holocene deltaic vs Pleistocene terrace and older")
h = g[g.GLG.isin(HOL)]; o = g[g.GLG.isin(OLD)]
for nm, col in [('As (ug/L)','As'), ('PO4 (mg/L)','PO4')]:
    a, b = h[col].dropna(), o[col].dropna()
    U, p = stats.mannwhitneyu(a, b)
    line(f"   {nm:12s} Holocene {a.median():7.2f} (n={len(a)})   Pleistocene {b.median():6.2f} (n={len(b)})   p = {p:.2g}")
line(f"   WHO exceedance   Holocene {100*(h.As>10).mean():.1f}%   Pleistocene {100*(o.As>10).mean():.1f}%")
line("   manuscript: As 8.14 vs 0.60, exceedance 46.4% vs 3.8%, PO4 1.249 vs 0.553")

line("\n-- TEST 4  does the phosphate partition survive inside each depth class")
for lab, lo, hi in [('Shallow <30 m',0,30), ('Inter. 30-80 m',30,80), ('Med-Deep 80-150 m',80,150), ('Deep >150 m',150,1e4)]:
    s = g[(g.Depth >= lo) & (g.Depth < hi)]
    a = s[s.GLG.isin(HOL)].PO4.dropna(); b = s[s.GLG.isin(OLD)].PO4.dropna()
    if len(a) < 10 or len(b) < 10:
        line(f"   {lab:19s} n too small (Hol {len(a)}, Pleist {len(b)})"); continue
    U, p = stats.mannwhitneyu(a, b)
    line(f"   {lab:19s} PO4 {a.median():6.2f} vs {b.median():5.2f}   n {len(a):4d}/{len(b):3d}   p = {p:.2g}")

line("\n-- TEST 5  depth-stratified partial correlations with log(As)")
line("   manuscript As-PO4 partial rho by class: 0.359, 0.286, 0.160, 0.071")
nhs['lAs'] = np.log10(nhs['As'].clip(lower=0.1))
for lab, lo, hi in [('Shallow <30 m',0,30), ('Inter. 30-80 m',30,80), ('Med-Deep 80-150 m',80,150), ('Deep >150 m',150,1e4)]:
    s = nhs[(nhs.Depth >= lo) & (nhs.Depth < hi)]
    rp, np_ = prho(s['lAs'], s['P'], s['Fe'])
    rf, nf  = prho(s['lAs'], s['Fe'], s['P'])
    line(f"   {lab:19s} As-P|Fe {rp:+.3f}   As-Fe|P {rf:+.3f}   n = {np_}")

# ======================================================================
line("\n"+"="*74); line("BGS/DPHE SPECIAL STUDY AREAS  (n = 271, Nawabganj / Faridpur / Lakshmipur)"); line("="*74)
ssa = load(f"{D}/BGS_special_study_areas_271.csv")
for c, out in [('AsTot','As'), ('Ptot','P'), ('Fe','Fe'), ('HCO3','HCO3'), ('DOC','DOC'), ('Mn','Mn')]:
    ssa[out], _ = uncensor(ssa[c])
line(f"  censored: As {parse_n(ssa,'AsTot')}, P {parse_n(ssa,'Ptot')}, DOC {parse_n(ssa,'DOC')} of {len(ssa)}")
ssa['Depth'] = pd.to_numeric(ssa['WELL_DEPTH'], errors='coerce')

line("\n-- TEST 6  the three leading correlates, with HCO3 available")
for v in ['P','HCO3','Fe','Mn','DOC']:
    m = ssa['As'].notna() & ssa[v].notna()
    r, p = stats.spearmanr(ssa.loc[m,'As'], ssa.loc[m,v])
    line(f"     As-{v:5s} rho = {r:+.3f}   p = {p:.3g}   n = {m.sum()}")

line("\n-- TEST 7  what DOC is actually attached to")
for v in ['P','As','Fe','HCO3']:
    m = ssa['DOC'].notna() & ssa[v].notna()
    r, p = stats.spearmanr(ssa.loc[m,'DOC'], ssa.loc[m,v])
    line(f"     DOC-{v:5s} rho = {r:+.3f}   p = {p:.3g}   n = {m.sum()}")
r_ap, n_ap = prho(ssa['As'], ssa['P'], ssa['DOC'])
r_raw, _   = stats.spearmanr(ssa['As'], ssa['P'], nan_policy='omit')
line(f"\n   As-P Spearman        {r_raw:+.3f}")
line(f"   As-P controlling DOC {r_ap:+.3f}   (n = {n_ap})")
