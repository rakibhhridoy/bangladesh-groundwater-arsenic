"""Well unit = BWDB station ID, as defined in the source report (BWDB 2023, s.1.3:
901 wells, each with a separate ID and GPS value). Supersedes the location-based rule
in r01 for counting. Flags IDs whose depth or coordinates disagree between campaigns,
and tests whether the headline results depend on them."""
import os
from pathlib import Path
import pandas as pd, numpy as np
from scipy import stats
R=os.environ.get("PAPER1_ROOT", str(Path(__file__).resolve().parents[2])); O=f"{R}/revision_r2/output"
d=pd.read_csv(f"{R}/data/integrated/bangladesh_groundwater_arsenic_1807samples_CLEAN.csv")
d['sid']=d['Sample ID'].astype(str).str.strip()
d['District']=d.District.str.strip().replace({'Shathkhira':'Satkhira','Sariatpur':'Shariatpur'})
out=[]; P=lambda *a:(out.append(" ".join(str(x) for x in a)),print(*a,flush=True))
trunc=d.sid.str.endswith('-')            # truncated IDs (e.g. 'CDUM2LW-') cannot be resolved
P(f"records {len(d)}; truncated IDs {trunc.sum()} ({d[trunc].sid.unique().tolist()})")
ids=d[~trunc].groupby('sid').Season.agg(lambda x:tuple(sorted(set(x))))
both=ids[ids==('Dry','Wet')].index; dry_only=ids[ids==('Dry',)].index; wet_only=ids[ids==('Wet',)].index
P(f"station IDs {len(ids)} (+ {d[trunc].shape[0]} truncated-ID records): in both campaigns {len(both)}, dry only {len(dry_only)}, wet only {len(wet_only)}")
p=d[d.sid.isin(both)].pivot_table(index='sid',columns='Season',values=['Depth','Latitude','Longitude'],aggfunc='first')
dz=(p['Depth','Dry']-p['Depth','Wet']).abs()
km=np.hypot((p['Latitude','Dry']-p['Latitude','Wet'])*111,(p['Longitude','Dry']-p['Longitude','Wet'])*102)
bad_z=dz[dz>5].index; bad_xy=km[km>1].index
P(f"paired IDs with depth disagreement >5 m: {len(bad_z)}; with coordinates >1 km apart: {len(bad_xy)}; either: {len(set(bad_z)|set(bad_xy))}")
flag=pd.DataFrame({'depth_dry':p['Depth','Dry'],'depth_wet':p['Depth','Wet'],'dz':dz,'km':km.round(2)})
flag[(flag.dz>5)|(flag.km>1)].sort_values('dz',ascending=False).to_csv(f"{O}/r01b_inconsistent_ids.csv")
d['inconsistent']=d.sid.isin(set(bad_z)|set(bad_xy))|trunc
d[['Sample ID','sid','Season','District','Depth','inconsistent']].to_csv(f"{O}/r01b_station_key.csv",index=False)
P(f"records flagged inconsistent (incl. truncated): {d.inconsistent.sum()}")
# robustness of headline numbers to dropping flagged records
c=d[~d.inconsistent]
P(f"As-PO4 rho all {stats.spearmanr(d.As,d['PO43-'])[0]:.3f} | excluding flagged {stats.spearmanr(c.As,c['PO43-'])[0]:.3f}")
dc=lambda x:pd.cut(x.Depth,[0,30,80,150,1e4],right=False,labels=['<30','30-80','80-150','>150'])
for lab,x in [('all',d),('excl. flagged',c)]:
    P(f"  {lab:14s} As-PO4 by depth class:",{k:round(stats.spearmanr(v.As,v['PO43-'])[0],3) for k,v in x.groupby(dc(x),observed=True)})
g=pd.read_csv(f"{R}/revision_r2/inputs/wells_with_usgs_geology.csv").merge(d[['Sample ID','Season','Depth','inconsistent']],on=['Sample ID','Season','Depth'])
HOL={'dsd','ac','dsl','ppc','dt'}; OLD={'afo','rb','rm','QTdd'}
for lab,x in [('all',g),('excl. flagged',g[~g.inconsistent])]:
    h=x[x.GLG.isin(HOL)]; o=x[x.GLG.isin(OLD)]
    P(f"  {lab:14s} HOL {100*(h.As>10).mean():.1f}% (n={len(h)}) vs OLD {100*(o.As>10).mean():.1f}% (n={len(o)})")
P("\nby campaign (Fe: UV-VIS wet, AAS dry; PO4 VIS both; BWDB 2023 Table 3-1)")
for s in ['Dry','Wet']:
    x=d[d.Season==s]; P(f"  {s}: As-PO4 {stats.spearmanr(x.As,x['PO43-'])[0]:.3f}  As-Fe {stats.spearmanr(x.As,x['Fe2+'])[0]:.3f}  As-HCO3 {stats.spearmanr(x.As,x['HCO3-'])[0]:.3f}  n={len(x)}")
open(f"{O}/r01b_summary.txt","w").write("\n".join(out))
