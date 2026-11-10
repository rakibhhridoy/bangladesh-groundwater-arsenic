"""R2 lines 84-85 & 422-429, R1 lines 333-342: depth-specific As-PO4 / As-Fe relationship
with cluster-bootstrap CIs; within depositional age; by physiographic zone (manuscript
line 369 claim); alternative Bengal Basin depth bins (R2 Methodology-3).
R2 Discussion: 68 m vs >150 m crossover, harmonised rolling window in both datasets,
depth-coverage comparison, and BWDB reweighted to the BGS depth distribution."""
import os
from pathlib import Path
import pandas as pd, numpy as np, sys
from scipy import stats
from scipy.stats import rankdata
R=os.environ.get("PAPER1_ROOT", str(Path(__file__).resolve().parents[2])); O=f"{R}/revision_r2/output"
sys.path.insert(0,str(Path(__file__).resolve().parent))
HOL={'dsd','ac','dsl','ppc','dt'}; OLD={'afo','rb','rm','QTdd'}
out=[]; P=lambda *a: (out.append(" ".join(str(x) for x in a)), print(*a, flush=True))
g=pd.read_csv(f"{R}/revision_r2/inputs/wells_with_usgs_geology.csv")
full=pd.read_csv(f"{R}/data/integrated/bangladesh_groundwater_arsenic_1807samples_CLEAN.csv")
full['well']=pd.read_csv(f"{O}/r01c_station_wellkey.csv").well.values
g=g.merge(full[['Sample ID','Season','Depth','Latitude','Longitude','well']],on=['Sample ID','Season','Depth','Latitude','Longitude'])
g=g[g.As.notna()&g.Depth.notna()&g['PO43-'].notna()&g['Fe2+'].notna()].copy()
g['lAs']=np.log10(g.As.clip(lower=0.1)); g['PO4']=g['PO43-']; g['Fe']=g['Fe2+']
g['age']=np.where(g.GLG.isin(HOL),'Holocene',np.where(g.GLG.isin(OLD),'Pleistocene','other'))

def prho(d,x,z,y='lAs'):
    """partial Spearman of y with x controlling list z (rank residualisation)"""
    C=np.c_[np.ones(len(d)),*[rankdata(d[c]) for c in z]] if z else np.ones((len(d),1))
    r=lambda v: v-C@np.linalg.lstsq(C,v,rcond=None)[0]
    return np.corrcoef(r(rankdata(d[y])),r(rankdata(d[x])))[0,1]
def boot(d,fn,B=1000,seed=42):
    rng=np.random.default_rng(seed); grp=[v for _,v in d.groupby('well')]; n=len(grp); vals=[]
    for _ in range(B):
        s=pd.concat([grp[i] for i in rng.integers(0,n,n)]); vals.append(fn(s))
    return np.nanpercentile(vals,[2.5,97.5])
def row(d,lab):
    rp=stats.spearmanr(d.As,d.PO4); rf=stats.spearmanr(d.As,d.Fe)
    pp=prho(d,'PO4',['Fe']); pf=prho(d,'Fe',['PO4'])
    cp=boot(d,lambda s:stats.spearmanr(s.As,s.PO4)[0]); cpp=boot(d,lambda s:prho(s,'PO4',['Fe']))
    cf=boot(d,lambda s:stats.spearmanr(s.As,s.Fe)[0])
    P(f"  {lab:24s} n={len(d):4d} wells={d.well.nunique():4d} | As-PO4 rho={rp[0]:+.3f} [{cp[0]:+.2f},{cp[1]:+.2f}] p={rp[1]:.1g} | As-PO4|Fe {pp:+.3f} [{cpp[0]:+.2f},{cpp[1]:+.2f}]"
      f" | As-Fe rho={rf[0]:+.3f} [{cf[0]:+.2f},{cf[1]:+.2f}] | As-Fe|PO4 {pf:+.3f}")
    return dict(group=lab,n=len(d),wells=d.well.nunique(),rho_PO4=rp[0],lo_PO4=cp[0],hi_PO4=cp[1],p_PO4=rp[1],prho_PO4_Fe=pp,plo=cpp[0],phi=cpp[1],rho_Fe=rf[0],lo_Fe=cf[0],hi_Fe=cf[1],prho_Fe_PO4=pf)
rows=[]
P("[A] national"); rows.append(row(g,'all'))
P("[B] operational depth classes")
bins=[0,30,80,150,1e4]; labs=['<30 m','30-80 m','80-150 m','>150 m']
g['dc']=pd.cut(g.Depth,bins,right=False,labels=labs)
for k in labs: rows.append(row(g[g.dc==k],k))
P("[C] depth class x depositional age")
for a in ['Holocene','Pleistocene']:
    for k in labs:
        s=g[(g.dc==k)&(g.age==a)]
        if len(s)>=25: rows.append(row(s,f"{a} {k}"))
        else: P(f"  {a} {k}: n={len(s)} too small")
P("[D] alternative Bengal Basin depth schemes")
for name,b,l in [('DPHE shallow/deep at 150 m',[0,150,1e4],['<150','>150']),
                 ('three-tier 50/150 m',[0,50,150,1e4],['<50','50-150','>150']),
                 ('quartiles',None,None)]:
    P(f"  -- {name}")
    if b is None:
        g['alt']=pd.qcut(g.Depth.rank(method='first'),4,labels=['Q1','Q2','Q3','Q4']); l=['Q1','Q2','Q3','Q4']
        P("     quartile edges m:",np.round(np.quantile(g.Depth,[.25,.5,.75]),0))
    else: g['alt']=pd.cut(g.Depth,b,right=False,labels=l)
    for k in l: rows.append(row(g[g.alt==k],f"{name}: {k}"))
P("[E] physiographic zone: As-PO4|Fe and As-Fe|PO4, shallow (<80 m) vs deep (>=80 m)")
zt=[]
for z,s in g.groupby('physio'):
    a=s[s.Depth<80]; b=s[s.Depth>=80]
    if len(a)>=20 and len(b)>=20:
        r=(prho(a,'PO4',['Fe']),prho(a,'Fe',['PO4']),prho(b,'PO4',['Fe']),prho(b,'Fe',['PO4']))
        lead_s='PO4' if r[0]>r[1] else 'Fe'; lead_d='PO4' if r[2]>r[3] else 'Fe'
        P(f"  {z:28s} <80m n={len(a):3d} PO4 {r[0]:+.2f} Fe {r[1]:+.2f} ({lead_s}) | >=80m n={len(b):3d} PO4 {r[2]:+.2f} Fe {r[3]:+.2f} ({lead_d})  shift={'yes' if (lead_s,lead_d)==('PO4','Fe') else 'no'}")
        zt.append(dict(zone=z,n_sh=len(a),n_dp=len(b),po4_sh=r[0],fe_sh=r[1],po4_dp=r[2],fe_dp=r[3]))
    else: P(f"  {z:28s} insufficient (n<80m={len(a)}, n>=80m={len(b)})")
pd.DataFrame(zt).to_csv(f"{O}/r03_zones.csv",index=False)
pd.DataFrame(rows).to_csv(f"{O}/r03_depthwise_correlations.csv",index=False)

# ---------- crossover, harmonised ----------
def rolling(d,window=80,step=5,min_n=20):
    r=[]
    for lo in np.arange(0,d.Depth.max()-window+step,step):
        s=d[(d.Depth>=lo)&(d.Depth<lo+window)]
        if len(s)<min_n: continue
        r.append((lo+window/2,prho(s,'PO4',['Fe']),prho(s,'Fe',['PO4']),len(s)))
    return pd.DataFrame(r,columns=['mid','po4','fe','n'])
def cross(rr):
    """first depth where Fe partial exceeds PO4 partial and stays above for >=3 windows"""
    diff=(rr.fe-rr.po4).values
    for i in range(len(diff)-2):
        if (diff[i:i+3]>0).all():
            if i==0: return rr.mid.iloc[0]
            x0,x1=rr.mid.iloc[i-1],rr.mid.iloc[i]; y0,y1=diff[i-1],diff[i]
            return x0+(0-y0)*(x1-x0)/(y1-y0)
    return np.nan
bw=rolling(g); P("\n[F] BWDB rolling crossover (80 m window, 5 m step):",round(cross(bw),1),"m")
# BGS
from bgs_replication import load, uncensor
D=f"{R}/data/external/BGS_DPHE_2001"
nhs=load(f"{D}/BGS_NHS_national_3534.csv")
for c,o in [('As','As'),('P_','P'),('Fe','Fe')]: nhs[o],_=uncensor(nhs[c])
nhs['Depth']=pd.to_numeric(nhs.WELL_DEPTH,errors='coerce'); nhs['PO4']=nhs.P*3.066
nhs=nhs[nhs.As.notna()&nhs.PO4.notna()&nhs.Fe.notna()&nhs.Depth.notna()].copy()
nhs['lAs']=np.log10(nhs.As.clip(lower=0.1)); nhs['well']=np.arange(len(nhs))
bg=rolling(nhs); P("    BGS rolling crossover (same method):",round(cross(bg),1),"m")
P("    BGS rolling windows (mid, PO4|Fe, Fe|PO4, n):"); P(bg.iloc[::4].round(3).to_string(index=False))
P("    BWDB rolling windows:"); P(bw.iloc[::4].round(3).to_string(index=False))
bw.to_csv(f"{O}/r03_rolling_bwdb.csv",index=False); bg.to_csv(f"{O}/r03_rolling_bgs.csv",index=False)
P("\n[G] depth coverage")
edges=[0,30,50,80,100,150,200,300,1000]
cb=pd.cut(g.Depth,edges,right=False).value_counts(sort=False); cg=pd.cut(nhs.Depth,edges,right=False).value_counts(sort=False)
P(pd.DataFrame({'BWDB n':cb,'BWDB %':(100*cb/cb.sum()).round(1),'BGS n':cg,'BGS %':(100*cg/cg.sum()).round(1)}).to_string())
P(f"    median depth BWDB {g.Depth.median():.0f} m, BGS {nhs.Depth.median():.0f} m; BWDB wells >150 m {(g.Depth>150).sum()}, BGS {(nhs.Depth>150).sum()}")
# bootstrap crossover for BGS
rng=np.random.default_rng(42); bc=[]
for _ in range(300):
    s=nhs.iloc[rng.integers(0,len(nhs),len(nhs))]; bc.append(cross(rolling(s)))
bc=np.array(bc); P(f"    BGS crossover bootstrap: found in {np.isfinite(bc).sum()}/300, median {np.nanmedian(bc):.0f} m, 95% CI {np.nanpercentile(bc,[2.5,97.5]).round(0)}")
# [H] BWDB resampled to BGS depth distribution (importance weights by depth bin)
w=(cg/cg.sum())/(cb/cb.sum()); g['wt']=pd.cut(g.Depth,edges,right=False).map(w).astype(float)
xs=[]
for b in range(300):
    s=g.sample(len(g),replace=True,weights='wt',random_state=b); xs.append(cross(rolling(s)))
xs=np.array(xs); P(f"[H] BWDB reweighted to BGS depth mix: crossover found {np.isfinite(xs).sum()}/300, median {np.nanmedian(xs):.0f} m, 95% CI {np.nanpercentile(xs,[2.5,97.5]).round(0)}")
# [I] same-depth-range comparison: correlations in both datasets within common classes
P("[I] harmonised class partials As-PO4|Fe / As-Fe|PO4 in both datasets")
for k,lo,hi in [('<30',0,30),('30-80',30,80),('80-150',80,150),('>150',150,1e4)]:
    a=g[(g.Depth>=lo)&(g.Depth<hi)]; b=nhs[(nhs.Depth>=lo)&(nhs.Depth<hi)]
    P(f"  {k:7s} BWDB n={len(a):4d} PO4 {prho(a,'PO4',['Fe']):+.3f} Fe {prho(a,'Fe',['PO4']):+.3f} | BGS n={len(b):4d} PO4 {prho(b,'PO4',['Fe']):+.3f} Fe {prho(b,'Fe',['PO4']):+.3f}")
open(f"{O}/r03_summary.txt","w").write("\n".join(out))
