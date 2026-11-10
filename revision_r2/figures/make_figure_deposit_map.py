"""New figure (R1, line 189): arsenic and phosphate on the depositional map.
Fill = mapped deposit age group (warm steps, dark = high-arsenic Holocene deltaic),
points = well medians (station ID) in neutral ink, sized by As (A) and PO4 (B);
ring in A marks median above WHO 10 ug/L. Layout agreed with the author 2026-11-09."""
import os
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np, pandas as pd, geopandas as gpd
R=os.environ.get("PAPER1_ROOT", str(Path(__file__).resolve().parents[2])); O=f"{R}/revision_r2/output"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"svg.fonttype":"none","pdf.fonttype":42})
INK,MUT="#0b0b0b","#52514e"
HOL={'dsd','ac','dsl','ppc','dt'}; OLD={'afo','rb','rm','QTdd'}
C_HOL,C_MID,C_OLD="#b3261e","#f2946a","#fbd5bd"
geo=gpd.read_file(f"{R}/data/raw/usgs_ofr97470h/geo8bg.shp").set_crs("EPSG:4326",allow_override=True)
bgd=gpd.read_file(f"{R}/data/raw/shapefiles/bangladesh.geojson")
geo['grp']=np.where(geo.GLG.isin(HOL),'HOL',np.where(geo.GLG.isin(OLD),'OLD','MID'))
geo=geo.clip(bgd.union_all())
g=pd.read_csv(f"{R}/revision_r2/inputs/wells_with_usgs_geology.csv")
g['sid']=g['Sample ID'].astype(str).str.strip()
g['key']=np.where(g.sid.str.endswith('-'),g.sid+'@'+g.Latitude.round(4).astype(str),g.sid)
w=g.groupby('key').agg(lon=('Longitude','first'),lat=('Latitude','first'),As=('As','median'),PO4=('PO43-','median')).reset_index()
def size_As(v): return 3+np.sqrt(np.clip(v,0,300))*4.2
def size_P(v):  return 2+np.clip(v,0,6)*30
fig,axs=plt.subplots(1,2,figsize=(7.4,4.9),dpi=300)
for ax,var,sz,lab in [(axs[0],'As',size_As,'A'),(axs[1],'PO4',size_P,'B')]:
    for k,c in [('MID',C_MID),('OLD',C_OLD),('HOL',C_HOL)]:
        geo[geo.grp==k].plot(ax=ax,color=c,lw=0,alpha=0.85,zorder=1)
    bgd.boundary.plot(ax=ax,color=INK,lw=0.6,zorder=2)
    v=w[var].values; o=np.argsort(-v)
    ax.scatter(w.lon.values[o],w.lat.values[o],s=sz(v[o]),facecolor="#2f2f2c",edgecolor="white",lw=0.35,alpha=0.85,zorder=3)
    if var=='As':
        e=w[w.As>10]; ax.scatter(e.lon,e.lat,s=sz(e.As.values)+16,facecolor='none',edgecolor=INK,lw=0.6,zorder=4)
    ax.set_xlim(88.0,92.7); ax.set_ylim(20.6,26.7); ax.set_aspect(1/np.cos(np.radians(23.7)))
    ax.set_xticks([88,89,90,91,92]); ax.set_yticks([21,22,23,24,25,26])
    ax.tick_params(labelsize=6.8,color=MUT,length=2)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.text(0.02,0.98,lab,transform=ax.transAxes,fontsize=11,fontweight='bold',va='top',color=INK)
# size legends
hA=[Line2D([],[],ls='',marker='o',markersize=np.sqrt(size_As(x)),markerfacecolor="#2f2f2c",markeredgecolor='white',label=f"{x:g}") for x in (1,10,50,200)]
hA.append(Line2D([],[],ls='',marker='o',markersize=np.sqrt(size_As(50)+16),markerfacecolor='none',markeredgecolor=INK,label="> WHO 10"))
axs[0].legend(handles=hA,title="arsenic (µg/L)",loc='lower left',fontsize=6.4,title_fontsize=6.6,frameon=True,framealpha=0.92,edgecolor="#d9d8d3",labelspacing=0.9,borderpad=0.6)
hP=[Line2D([],[],ls='',marker='o',markersize=np.sqrt(size_P(x)),markerfacecolor="#2f2f2c",markeredgecolor='white',label=f"{x:g}") for x in (0.2,1,3,5)]
axs[1].legend(handles=hP,title="phosphate (mg/L)",loc='lower left',fontsize=6.4,title_fontsize=6.6,frameon=True,framealpha=0.92,edgecolor="#d9d8d3",labelspacing=1.0,borderpad=0.6)
fig.legend(handles=[Patch(facecolor=C_HOL,label="Holocene deltaic"),Patch(facecolor=C_MID,label="alluvial, estuarine and other"),Patch(facecolor=C_OLD,label="Pleistocene terrace and older")],
           loc='lower center',ncol=3,fontsize=7,frameon=False,bbox_to_anchor=(0.5,-0.01))
fig.tight_layout(rect=(0,0.05,1,1))
for e in ("pdf","png","svg"): fig.savefig(f"{O}/Figure_deposit_map.{e}",facecolor="white",bbox_inches="tight",dpi=600 if e=="png" else None)
print("wells",len(w))
