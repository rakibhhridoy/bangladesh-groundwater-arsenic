"""How method-dependent is the 68 m crossover? Grid over covariate set, window width,
crossing rule; plus cluster-bootstrap of the Fe-minus-PO4 partial difference per depth
class (is Fe actually stronger at depth?). Manuscript value came from
the submitted crossover analysis (4 covariates, first sign change, 80 m window)."""
import os
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
R=os.environ.get("PAPER1_ROOT", str(Path(__file__).resolve().parents[2])); O=f"{R}/revision_r2/output"
df=pd.read_csv(f"{R}/data/integrated/bangladesh_groundwater_arsenic_1807samples_CLEAN.csv")
df['well']=pd.read_csv(f"{O}/r01c_station_wellkey.csv").well.values
d=df[(df['As']>0)&(df['PO43-']>0)&(df['Fe2+']>0)&(df['HCO3-']>0)&(df['Depth']>0)&(df.pH<14)&(df.DO<25)].copy()
for a,b in [('lAs','As'),('lP','PO43-'),('lFe','Fe2+'),('lH','HCO3-')]: d[a]=np.log(d[b])
out=[]; P=lambda *a:(out.append(" ".join(str(x) for x in a)),print(*a,flush=True))
def ps(x,y,Z):
    rx=stats.rankdata(x); ry=stats.rankdata(y)
    C=np.column_stack([np.ones(len(x))]+[stats.rankdata(Z[:,j]) for j in range(Z.shape[1])])
    e=lambda v:v-C@np.linalg.lstsq(C,v,rcond=None)[0]
    return np.corrcoef(e(rx),e(ry))[0,1]
COV={'Fe/PO4 only (as BGS comparison & Fig. 5)':([], ),'+HCO3':(['lH'],),'+HCO3,pH,DO (manuscript 68 m)':(['lH','pH','DO'],)}
def curve(dd,extra,win):
    cs=np.arange(10,251,5); r=[]
    for c in cs:
        s=dd[(dd.Depth>=c-win/2)&(dd.Depth<=c+win/2)]
        if len(s)<60: continue
        rp=ps(s.lAs.values,s.lP.values,s[['lFe']+extra].values); rf=ps(s.lAs.values,s.lFe.values,s[['lP']+extra].values)
        r.append((c,rp-rf))
    return np.array(r)
def first(r):
    i=np.where((r[:-1,1]>0)&(r[1:,1]<=0))[0]
    if not len(i): return np.nan
    i=i[0]; x0,x1,y0,y1=r[i,0],r[i+1,0],r[i,1],r[i+1,1]; return x0+(x1-x0)*y0/(y0-y1)
def sustained(r,k=3):
    for i in range(1,len(r)-k+1):
        if r[i-1,1]>0 and (r[i:i+k,1]<=0).all():
            x0,x1,y0,y1=r[i-1,0],r[i,0],r[i-1,1],r[i,1]; return x0+(x1-x0)*y0/(y0-y1)
    return np.nan
P(f"n={len(d)}")
P(f"{'covariates':42s} {'window':>6s} {'first':>7s} {'sustained(3)':>13s}")
grid=[]
for lab,(extra,) in COV.items():
    for win in (60,80,120):
        r=curve(d,extra,win); a,b=first(r),sustained(r)
        P(f"{lab:42s} {win:6d} {a:7.0f} {b:13.0f}"); grid.append(dict(cov=lab,win=win,first=a,sustained=b))
pd.DataFrame(grid).to_csv(f"{O}/r07_grid.csv",index=False)
P("\nFe-minus-PO4 partial difference per depth class (manuscript covariates), cluster bootstrap by well B=1000")
d['dc']=pd.cut(d.Depth,[0,30,80,150,1e4],right=False,labels=['<30','30-80','80-150','>150'])
rng=np.random.default_rng(42)
for k in ['<30','30-80','80-150','>150']:
    s=d[d.dc==k]; grp=[v for _,v in s.groupby('well')]; n=len(grp)
    f=lambda q: ps(q.lAs.values,q.lFe.values,q[['lP','lH','pH','DO']].values)-ps(q.lAs.values,q.lP.values,q[['lFe','lH','pH','DO']].values)
    b=[f(pd.concat([grp[i] for i in rng.integers(0,n,n)])) for _ in range(1000)]
    P(f"  {k:7s} n={len(s)} Fe-PO4 = {f(s):+.3f}  95% CI [{np.percentile(b,2.5):+.3f}, {np.percentile(b,97.5):+.3f}]  P(Fe>PO4)={np.mean(np.array(b)>0):.2f}")
open(f"{O}/r07_summary.txt","w").write("\n".join(out))
