"""Figure 1: interpolated arsenic surface with contours and the sampling network.
Sequential warm ramp for magnitude (dataviz reference palette). No figure title or
subtitle -- the caption carries that."""
import os
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import numpy as np, pandas as pd, geopandas as gpd
from scipy.interpolate import griddata
from shapely.geometry import Point

R=os.environ.get("PAPER1_ROOT", str(Path(__file__).resolve().parents[2]))
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"svg.fonttype":"none","pdf.fonttype":42})
INK,MUT="#0b0b0b","#52514e"
RAMP=["#fdece0","#fbd5bd","#f8b894","#f2946a","#eb6834","#d9482c","#b3261e","#7d1a15"]
cmap=LinearSegmentedColormap.from_list("as_warm",RAMP)

d=pd.read_csv(f"{R}/data/integrated/bangladesh_groundwater_arsenic_1807samples_CLEAN.csv")
d=d[(d.pH<14)&(d.DO<25)].copy()
bgd=gpd.read_file(f"{R}/data/raw/shapefiles/bangladesh.geojson")
divs=gpd.read_file(f"{R}/data/raw/shapefiles/bgd_divisions.geojson")
riv=gpd.read_file(f"{R}/data/raw/shapefiles/bgd_hydrorivers.geojson")
sa=gpd.read_file(f"{R}/data/raw/shapefiles/south_asia.geojson")

# well-level median so repeat visits do not double-weight a location
# one point per well, keyed on the BWDB station identifier (BWDB 2023, s.1.3);
# truncated identifiers are kept apart by their coordinates
d['sid']=d['Sample ID'].astype(str).str.strip()
d['key']=np.where(d.sid.str.endswith('-'),d.sid+'@'+d.Latitude.round(4).astype(str)+','+d.Longitude.round(4).astype(str),d.sid)
w=d.groupby('key').agg(lon=('Longitude','first'),lat=('Latitude','first'),As=('As','median')).reset_index(drop=True)
print(f"  wells: {len(w)}  samples: {len(d)}")

minx,miny,maxx,maxy=bgd.total_bounds
gx,gy=np.mgrid[minx:maxx:340j, miny:maxy:340j]
gz=griddata((w.lon,w.lat), np.log10(w.As.clip(lower=0.05)), (gx,gy), method='linear')
from scipy.ndimage import gaussian_filter
m=np.isnan(gz); z0=np.where(m,np.nanmean(gz),gz)
gz=gaussian_filter(z0,sigma=4.5); gz=np.where(m,np.nan,gz)
# mask to the country outline
poly=bgd.union_all()
pts=gpd.GeoSeries([Point(a,b) for a,b in zip(gx.ravel(),gy.ravel())],crs=bgd.crs)
gz=np.where(pts.within(poly).values.reshape(gx.shape), gz, np.nan)

fig,ax=plt.subplots(figsize=(7.9,7.4),dpi=300)
fig.patch.set_facecolor("white")
LEV=np.log10([0.5,1,2,5,10,25,50,100,300])
cf=ax.contourf(gx,gy,gz,levels=LEV,cmap=cmap,extend='both',zorder=1,alpha=0.92)
ax.contour(gx,gy,gz,levels=[np.log10(10)],colors=['#104281'],linewidths=1.3,zorder=4)
riv.plot(ax=ax,color="#7ea9d8",lw=0.3,alpha=0.45,zorder=2)
divs.boundary.plot(ax=ax,color="#8b8a85",lw=0.4,linestyle="--",zorder=3)
bgd.boundary.plot(ax=ax,color=INK,lw=0.9,zorder=5)

exc=w[w.As>10]
ax.scatter(w.lon,w.lat,s=5,c="#2f2f2c",alpha=0.6,lw=0,zorder=6,label=f"monitoring well (n = {len(w)})")
ax.scatter(exc.lon,exc.lat,s=26,facecolors='none',edgecolors="#104281",lw=0.8,zorder=7,
           label=f"exceeds WHO 10 µg/L ({100*len(exc)/len(w):.0f}%)")

cb=fig.colorbar(cf,ax=ax,fraction=0.032,pad=0.02,ticks=LEV)
cb.ax.set_yticklabels(["0.5","1","2","5","10","25","50","100","300"],fontsize=7.5)
cb.set_label("dissolved arsenic (µg/L, log scale)",fontsize=8,color=MUT)
cb.outline.set_visible(False); cb.ax.tick_params(length=2,color=MUT)

axin=ax.inset_axes([0.003,0.355,0.273,0.295])
sa.plot(ax=axin,color="#eeeeec",edgecolor="#b9b8b3",lw=0.3)
bgd.plot(ax=axin,color="#b3261e",edgecolor=INK,lw=0.5)
axin.set_xlim(66.0,100.0); axin.set_ylim(5.0,33.0)
axin.set_aspect(1/np.cos(np.radians(19.0)))
axin.set_xticks([]); axin.set_yticks([]); axin.set_facecolor("white")
axin.patch.set_alpha(0.96)
# region label, bottom centre inside the frame. A white halo carries it over
# Sri Lanka and the southern Indian coast, which occupy this part of the extent.
axin.text(0.5,0.075,"South Asia",transform=axin.transAxes,ha="center",va="center",
          fontsize=6.6,color=MUT,zorder=6,
          bbox=dict(boxstyle="round,pad=0.22",fc="white",ec="none",alpha=0.86))
for sp in axin.spines.values(): sp.set_color("#b9b8b3"); sp.set_linewidth(0.5)

ax.set_xlim(86.00, 92.95)
from matplotlib.lines import Line2D
h,l=ax.get_legend_handles_labels()
h.append(Line2D([0],[0],color="#104281",lw=1.3)); l.append("interpolated WHO 10 µg/L contour")
ax.legend(h,l,loc="lower left",frameon=True,framealpha=0.92,edgecolor="#d9d8d3",
          fontsize=7.4,handletextpad=0.5,borderpad=0.5)
ax.set_xlabel("longitude (°E)",fontsize=8,color=MUT); ax.set_ylabel("latitude (°N)",fontsize=8,color=MUT)
ax.tick_params(labelsize=7.5,color=MUT); ax.set_aspect(1/np.cos(np.radians(23.7)))
for sp in ax.spines.values(): sp.set_visible(False)
# scale bar
x0,y0=minx+0.28,miny+0.20; km=100/111.32
ax.plot([x0,x0+km],[y0,y0],color=INK,lw=2,solid_capstyle='butt',zorder=8)
ax.text(x0+km/2,y0+0.055,"100 km",ha='center',fontsize=7,color=INK,zorder=8)
ax.annotate("N",xy=(maxx-0.30,maxy-0.30),ha='center',fontsize=10,fontweight='bold',color=INK,zorder=8)
ax.annotate("",xy=(maxx-0.30,maxy-0.34),xytext=(maxx-0.30,maxy-0.62),
            arrowprops=dict(arrowstyle='-|>',color=INK,lw=1.1),zorder=8)
fig.tight_layout()
for e in ("pdf","png"):
    fig.savefig(f"{R}/revision_r2/output/Figure1_study_area_map.{e}",
                facecolor="white",bbox_inches="tight",dpi=600 if e=="png" else None)
print("  wrote Figure1_study_area_map .pdf .png"); dx=(maxx-minx)/339*111.32*np.cos(np.radians(23.7)); dy=(maxy-miny)/339*111.32; print(f"  grid cell {dx:.2f} x {dy:.2f} km; smoothing sigma 4.5 cells = {4.5*dx:.1f} km; exceed {100*(w.As>10).mean():.1f}%")
