"""R1 lines 486-489: practical value of mapped deposit type depends on map resolution.
Source map Alam et al. (1990), 1:1,000,000, digitised by Persits et al. (1997).
Distance from every well to the nearest mapped unit boundary; Holocene-vs-Pleistocene
contrast and deposit-type effect size recomputed after excluding wells near boundaries."""
import os
from pathlib import Path
import pandas as pd, numpy as np, geopandas as gpd
from scipy import stats
from shapely.geometry import Point
R=os.environ.get("PAPER1_ROOT", str(Path(__file__).resolve().parents[2])); O=f"{R}/revision_r2/output"
LAB=['dsd','ac','dsl','ppc','dt','asl','asc','asd','afy','de','dm','ava','afo','csd','rb','rm','QTdd']
HOL={'dsd','ac','dsl','ppc','dt'}; OLD={'afo','rb','rm','QTdd'}
geo=gpd.read_file(f"{R}/data/raw/usgs_ofr97470h/geo8bg.shp").set_crs("EPSG:4326",allow_override=True).to_crs("EPSG:32646")
g=pd.read_csv(f"{R}/revision_r2/inputs/wells_with_usgs_geology.csv")
pts=gpd.GeoSeries([Point(x,y) for x,y in zip(g.Longitude,g.Latitude)],crs="EPSG:4326").to_crs("EPSG:32646")
# boundaries between DIFFERENT units only
diss=geo.dissolve('GLG').reset_index()
bnd=diss.boundary.union_all()
coast=geo.union_all().boundary
inner=bnd.difference(coast.buffer(1))
g['dist_km']=pts.distance(inner).values/1000
g=g[g.GLG.isin(LAB)&g.As.notna()].copy()
g.to_csv(f"{O}/r08_boundary_distance.csv",index=False)
out=[]; P=lambda *a:(out.append(" ".join(str(x) for x in a)),print(*a,flush=True))
P("distance to nearest inter-unit boundary (km): quantiles",g.dist_km.quantile([.1,.25,.5,.75,.9]).round(2).to_dict())
def eps2(d):
    sub=[x.As.values for _,x in d.groupby('GLG') if len(x)>=15]; H,p=stats.kruskal(*sub); k=len(sub); N=sum(map(len,sub)); return (H-k+1)/(N-k),k
for cut in (0,0.5,1,2,3,5):
    s=g[g.dist_km>=cut]; h=s[s.GLG.isin(HOL)]; o=s[s.GLG.isin(OLD)]; e,k=eps2(s)
    P(f"  >= {cut:3.1f} km from boundary: n={len(s):4d} ({100*len(s)/len(g):.0f}%) eps2={e:.3f} (k={k}) | HOL {len(h)} {100*(h.As>10).mean():.1f}% | OLD {len(o)} {100*(o.As>10).mean():.1f}% | PO4 {h['PO43-'].median():.2f} vs {o['PO43-'].median():.2f}")
open(f"{O}/r08_summary.txt","w").write("\n".join(out))
