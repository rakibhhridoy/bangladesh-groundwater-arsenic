"""R2 Results-1 (17 categories vs 4 bins; depth confound) and R2 Methodology-4
(surface geology vs deep wells). Fair comparisons of mapped deposit type and depth,
and the shallow-only test. Cluster bootstrap by well (r01c_station_wellkey.csv)."""
import os
from pathlib import Path
import pandas as pd, numpy as np
from scipy import stats
import statsmodels.formula.api as smf, statsmodels.api as sm
from statsmodels.stats.contingency_tables import StratifiedTable
from sklearn.model_selection import KFold
R=os.environ.get("PAPER1_ROOT", str(Path(__file__).resolve().parents[2])); O=f"{R}/revision_r2/output"
LAB=['dsd','ac','dsl','ppc','dt','asl','asc','asd','afy','de','dm','ava','afo','csd','rb','rm','QTdd']
HOL={'dsd','ac','dsl','ppc','dt'}; OLD={'afo','rb','rm','QTdd'}
g=pd.read_csv(f"{R}/revision_r2/inputs/wells_with_usgs_geology.csv")
key=pd.read_csv(f"{O}/r01c_station_wellkey.csv")
full=pd.read_csv(f"{R}/data/integrated/bangladesh_groundwater_arsenic_1807samples_CLEAN.csv")
full['well']=key.well.values
# attach well id: geology file is CLEAN minus 2 impossible records, same order
g=g.merge(full[['Sample ID','Season','Depth','Latitude','Longitude','well']],on=['Sample ID','Season','Depth','Latitude','Longitude'],how='left')
assert g.well.notna().all(), g.well.isna().sum()
g=g[g.GLG.isin(LAB)&g.As.notna()&g.Depth.notna()].copy()
g['age']=np.where(g.GLG.isin(HOL),'HOL',np.where(g.GLG.isin(OLD),'OLD','ALL'))
g['dclass']=pd.cut(g.Depth,[0,30,80,150,1e4],right=False,labels=['S','I','MD','D'])
g['rA']=stats.rankdata(g.As)/len(g); g['ld']=np.log(g.Depth.clip(lower=1)); LB=g.ld.min()-1e-6; UB=g.ld.max()+1e-6
out=[]; P=lambda *a: out.append(" ".join(str(x) for x in a))
def eps2(d,col,minn=15):
    sub=[x.As.values for _,x in d.groupby(col,observed=True) if len(x)>=minn]
    H,p=stats.kruskal(*sub); k=len(sub); N=sum(map(len,sub)); return (H-k+1)/(N-k),H,p,k,N
P("N samples",len(g),"wells",g.well.nunique())
e=eps2(g,'GLG'); P(f"deposit type: eps2={e[0]:.3f} H={e[1]:.1f} k={e[3]}")
e=eps2(g,'dclass'); P(f"4 depth classes: eps2={e[0]:.3f} H={e[1]:.1f}")
P("\n[1] equal granularity: depth in k quantile bins")
for k in (4,8,12,17,25):
    g['dq']=pd.qcut(g.Depth.rank(method='first'),k,labels=False); e=eps2(g,'dq'); P(f"  depth {k} bins eps2={e[0]:.3f}")
P(f"  Spearman As~depth rho={stats.spearmanr(g.As,g.Depth)[0]:.3f}  rho^2={stats.spearmanr(g.As,g.Depth)[0]**2:.4f}")
# [2] continuous depth, flexible, cross-validated on rank(As); same for deposit type
P("\n[2] 10-fold CV R2 on rank(As), grouped folds by well")
from sklearn.model_selection import GroupKFold
def cvr2(formula,d,folds=10):
    gkf=GroupKFold(folds); ss=0; st=0
    for tr,te in gkf.split(d,groups=d.well):
        m=smf.ols(formula,d.iloc[tr]).fit(); pr=m.predict(d.iloc[te]); y=d.iloc[te].rA
        ss+=((y-pr)**2).sum(); st+=((y-d.iloc[tr].rA.mean())**2).sum()
    return 1-ss/st
res={}
for lab,f in [('depth linear','rA~ld'),('depth spline df4','rA~bs(ld,df=4,lower_bound=LB,upper_bound=UB)'),('depth spline df8','rA~bs(ld,df=8,lower_bound=LB,upper_bound=UB)'),('depth spline df16','rA~bs(ld,df=16,lower_bound=LB,upper_bound=UB)'),
              ('deposit type','rA~C(GLG)'),('deposit + depth spline df4','rA~C(GLG)+bs(ld,df=4,lower_bound=LB,upper_bound=UB)')]:
    fit=smf.ols(f,g).fit(); res[lab]=fit; P(f"  {lab:28s} in-sample R2={fit.rsquared:.3f} adjR2={fit.rsquared_adj:.3f} CV R2={cvr2(f,g):.3f}")
an=sm.stats.anova_lm(res['deposit + depth spline df4'],typ=2); sst=((g.rA-g.rA.mean())**2).sum()
P("  type-II partial R2 (share of total SS) in joint model:"); 
for t in an.index[:-1]: P(f"    {t:16s} SS/SST={an.loc[t,'sum_sq']/sst:.3f}  F={an.loc[t,'F']:.1f} p={an.loc[t,'PR(>F)']:.2g}")
# [3] depth-residualised deposit effect
r=g.rA-smf.ols('rA~bs(ld,df=4,lower_bound=LB,upper_bound=UB)',g).fit().fittedvalues; g['res']=r
sub=[x.res.values for _,x in g.groupby('GLG') if len(x)>=15]; H,p=stats.kruskal(*sub);k=len(sub);N=sum(map(len,sub))
P(f"\n[3] deposit type on depth-residualised rank(As): eps2={(H-k+1)/(N-k):.3f} p={p:.2g}")
# [4] well-level (one value per well) to rule out repeat-visit weighting
w=g.groupby('well').agg(As=('As','median'),Depth=('Depth','median'),GLG=('GLG','first'),age=('age','first')).reset_index()
e=eps2(w,'GLG'); w['dc']=pd.cut(w.Depth,[0,30,80,150,1e4],right=False); e2=eps2(w,'dc')
P(f"[4] well-level: n wells={len(w)} deposit eps2={e[0]:.3f}  depth-class eps2={e2[0]:.3f}")
# [5] shallow-only
def cboot(d,fn,B=2000,seed=42):
    rng=np.random.default_rng(seed); ws=d.well.unique(); grp={k:v for k,v in d.groupby('well')}
    vals=[]
    for _ in range(B):
        s=pd.concat([grp[x] for x in rng.choice(ws,len(ws))]); vals.append(fn(s))
    return np.nanpercentile(vals,[2.5,97.5])
exc=lambda d:100*(d.As>10).mean()
P("\n[5] shallow-only test (cluster bootstrap 95% CI by well, B=2000)")
rows=[]
for cut in (30,50,70,1e4):
    s=g[g.Depth<cut]; h=s[s.age=='HOL']; o=s[s.age=='OLD']
    ch=cboot(h,exc); co=cboot(o,exc); e=eps2(s,'GLG')
    mw=stats.mannwhitneyu(h.As,o.As); mp=stats.mannwhitneyu(h['PO43-'],o['PO43-'])
    lab='all' if cut>1e3 else f'<{int(cut)} m'
    P(f"  {lab:8s} n={len(s)} eps2 deposit={e[0]:.3f} (k={e[3]}) | HOL n={len(h)} wells={h.well.nunique()} {exc(h):.1f}% [{ch[0]:.1f},{ch[1]:.1f}]"
      f" | OLD n={len(o)} wells={o.well.nunique()} {exc(o):.1f}% [{co[0]:.1f},{co[1]:.1f}] | As MWU p={mw.pvalue:.1g}"
      f" | PO4 med HOL {h['PO43-'].median():.2f} OLD {o['PO43-'].median():.2f} p={mp.pvalue:.1g}")
    rows.append(dict(cut=lab,n=len(s),eps2=e[0],k=e[3],hol_n=len(h),hol_exc=exc(h),hol_lo=ch[0],hol_hi=ch[1],old_n=len(o),old_exc=exc(o),old_lo=co[0],old_hi=co[1],p_As=mw.pvalue,po4_hol=h['PO43-'].median(),po4_old=o['PO43-'].median(),p_po4=mp.pvalue))
pd.DataFrame(rows).to_csv(f"{O}/r02_shallow_test.csv",index=False)
# [6] within depth class + CMH
P("\n[6] HOL vs OLD exceedance within depth class; Mantel-Haenszel")
tabs=[]
for k,s in g[g.age!='ALL'].groupby('dclass',observed=True):
    h=s[s.age=='HOL']; o=s[s.age=='OLD']
    a=(h.As>10).sum(); b=(h.As<=10).sum(); c=(o.As>10).sum(); d_=(o.As<=10).sum()
    tabs.append(np.array([[a,b],[c,d_]])+0.5)
    P(f"  {k}: HOL {len(h)} {exc(h):.1f}% (median depth {h.Depth.median():.0f}) | OLD {len(o)} {exc(o):.1f}% (median depth {o.Depth.median():.0f})")
st=StratifiedTable(tabs); P(f"  MH pooled OR={st.oddsratio_pooled:.1f} 95%CI {st.oddsratio_pooled_confint()} p={st.test_null_odds().pvalue:.2g}; homogeneity p={st.test_equal_odds().pvalue:.2f}")
# [7] deposit-type mean depth spread (R2 note Table 1 33 vs 146 m) and depth-matched contrast:
P("\n[7] depth-matched HOL vs OLD (nearest depth, caliper 5 m, 1:1 by sample)")
h=g[g.age=='HOL'].copy(); o=g[g.age=='OLD'].copy(); used=set(); pairs=[]
for i,r in o.sort_values('Depth').iterrows():
    c=h[~h.index.isin(used)]; j=(c.Depth-r.Depth).abs().idxmin()
    if abs(c.loc[j,'Depth']-r.Depth)<=5: used.add(j); pairs.append((i,j))
oi=[a for a,b in pairs]; hi=[b for a,b in pairs]
P(f"  matched pairs={len(pairs)}: OLD {exc(o.loc[oi]):.1f}% vs HOL {exc(h.loc[hi]):.1f}%; median depth {o.loc[oi].Depth.median():.0f} vs {h.loc[hi].Depth.median():.0f}; Wilcoxon p={stats.wilcoxon(o.loc[oi].As.values,h.loc[hi].As.values).pvalue:.2g}")
# [8] station IDs whose two campaigns map to different deposit types (coordinate disagreement)
multi=g.groupby('well').GLG.nunique(); bad=multi[multi>1].index
c=g[~g.well.isin(bad)]; h=c[c.age=='HOL']; o=c[c.age=='OLD']; e=eps2(c,'GLG')
P(f"\n[8] IDs spanning two deposit types: {len(bad)}; excluding them n={len(c)} wells={c.well.nunique()} eps2={e[0]:.3f} | HOL {exc(h):.1f}% (n={len(h)}) OLD {exc(o):.1f}% (n={len(o)})")
open(f"{O}/r02_summary.txt","w").write("\n".join(out)); print("\n".join(out))
