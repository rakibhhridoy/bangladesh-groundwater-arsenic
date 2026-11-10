"""R2 Discussion: nitrate is unreliable, show phosphate (same spectrophotometric
procedure) is not similarly affected. Data-internal diagnostics only; no QC records."""
import os
from pathlib import Path
import pandas as pd, numpy as np
from scipy import stats
R=os.environ.get("PAPER1_ROOT", str(Path(__file__).resolve().parents[2])); O=f"{R}/revision_r2/output"
d=pd.read_csv(f"{R}/data/integrated/bangladesh_groundwater_arsenic_1807samples_CLEAN.csv")
d=d[(d.pH<14)&(d.DO<25)].copy()
out=[]; P=lambda *a:(out.append(" ".join(str(x) for x in a)),print(*a,flush=True))
sp=lambda a,b:stats.spearmanr(a,b,nan_policy='omit')
P("[1] arsenate interference in molybdenum-blue PO4, worst case")
base=sp(d.As,d['PO43-'])[0]; P(f"  As-PO4 rho as reported {base:.3f}")
for f,lab in [(1.0,'100% As(V), 1:1 molar response (worst case)'),(0.13,'13% As(V) (thermodynamic estimate upper)')]:
    eq=d.As*f/74.92*94.97/1000          # ug/L As -> mg/L PO4-equivalent
    c=(d['PO43-']-eq).clip(lower=0)
    P(f"  {lab:46s}: max PO4-eq {eq.max():.3f} mg/L, median {eq.median():.4f}; median share of PO4 {100*(eq/d['PO43-'].replace(0,np.nan)).median():.1f}%; rho {sp(d.As,c)[0]:.3f}")
P("[2] between-campaign reproducibility at paired wells (station ID)")
full=pd.read_csv(f"{R}/data/integrated/bangladesh_groundwater_arsenic_1807samples_CLEAN.csv")
full['sid']=full['Sample ID'].astype(str).str.strip(); full=full[~full.sid.str.endswith('-')]
pw=full.pivot_table(index='sid',columns='Season',values=['PO43-','NO3-','As','Fe2+','HCO3-','Cl-','SO42-','Mn2+'],aggfunc='first')
for v in ['PO43-','NO3-','As','Fe2+','HCO3-','Cl-','SO42-','Mn2+']:
    a=pw[v,'Dry'].values; b=pw[v,'Wet'].values; m=~(np.isnan(a)|np.isnan(b))
    P(f"  {v:6s} dry-vs-wet Spearman at same well {stats.spearmanr(a[m],b[m])[0]:+.3f} (n={m.sum()})  distinct values {full[v].nunique()}")
P("[3] redox consistency (nitrate failed these)")
for v in ['PO43-','NO3-']:
    P(f"  {v:6s} rho with ORP {sp(d[v],d.ORP)[0]:+.3f}, Fe {sp(d[v],d['Fe2+'])[0]:+.3f}, DO {sp(d[v],d.DO)[0]:+.3f}, HCO3 {sp(d[v],d['HCO3-'])[0]:+.3f}")
d['redox']=pd.cut(d.ORP,[-2000,-100,0,2000],labels=['strongly reducing','reducing','oxidising'])
P("  medians by redox class:",d.groupby('redox',observed=True)[['PO43-','NO3-']].median().round(2).to_dict())
P("  expected: PO4 rises in reducing water (release with Fe-oxide reduction); NO3 should vanish where Fe2+>1 mg/L")
P(f"  share of samples with Fe2+>1 mg/L carrying NO3>1 mg/L: {100*((d['Fe2+']>1)&(d['NO3-']>1)).sum()/(d['Fe2+']>1).sum():.1f}%")
P("[4] resolution and bounds")
for v in ['PO43-','NO3-']:
    x=d[v]; P(f"  {v:6s} distinct {x.nunique()}, at min {100*(x==x.min()).mean():.1f}% (={x.min()}), at max {100*(x==x.max()).mean():.1f}% (={x.max()})")
inner=(d['PO43-']>d['PO43-'].min())&(d['PO43-']<d['PO43-'].max())
P(f"  As-PO4 rho excluding bound values {sp(d.As[inner],d['PO43-'][inner])[0]:.3f} (n={inner.sum()})")
P("[5] independent laboratories")
h=pd.read_csv(f"{R}/revision_r2/inputs/hoque2010_A5_1_hydrochem.csv")
h=h[h.As.notna()&h.P.notna()]; h['As_']=h.As.where(h.As>0,0.5)   # negatives are <DL codes
P(f"  Hoque 2010 (UCL, n={len(h)}): As-P rho {sp(h.As_,h.P)[0]:.3f} p={sp(h.As_,h.P)[1]:.2g}; As-Fe {sp(h.As_,h.Fe)[0]:.3f}")
P("  BGS/DPHE 2001 (n=3,534, different laboratory and method): As-P 0.675 (r03/bgs_replication)")
open(f"{O}/r05_summary.txt","w").write("\n".join(out))
