"""R2 lines 124-128: is the gradient-boosted model adequately validated for depth-specific
assessment, particularly for PO4 and Fe? Same features and hyperparameters as the
manuscript (scripts/h1_geochemistry/shap_analysis.py; target log1p(As))."""
import os
from pathlib import Path
import pandas as pd, numpy as np, shap, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score
R=os.environ.get("PAPER1_ROOT", str(Path(__file__).resolve().parents[2])); O=f"{R}/revision_r2/output"
d=pd.read_csv(f"{R}/data/integrated/bangladesh_groundwater_arsenic_1807samples_CLEAN.csv")
d['well']=pd.read_csv(f"{O}/r01c_station_wellkey.csv").well.values
d=d[(d.pH<14)&(d.DO<25)&(d.ORP>-1000)]
F=['PO43-','Fe2+','pH','HCO3-','DO','Mn2+','Ca2+','Na+','Cl-','SO42-','ORP','EC','TDS','Temperature']
d=d.dropna(subset=F+['As','Depth']).reset_index(drop=True)
X=d[F].values; y=np.log1p(d.As.values)
d['blk']=(np.floor(d.Latitude)).astype(int).astype(str)+'_'+(np.floor(d.Longitude)).astype(int).astype(str)
d['dc']=pd.cut(d.Depth,[0,30,80,150,1e4],right=False,labels=['<30','30-80','80-150','>150'])
mk=lambda: GradientBoostingRegressor(n_estimators=300,learning_rate=0.05,max_depth=5,random_state=42,validation_fraction=0.2,n_iter_no_change=50)
out=[]; P=lambda *a:(out.append(" ".join(str(x) for x in a)),print(*a,flush=True))
P(f"n={len(d)} wells={d.well.nunique()} spatial 1-degree blocks={d.blk.nunique()}")
def oof(splitter,groups=None,seeds=(42,)):
    preds=[]
    for s in seeds:
        p=np.zeros(len(y))
        sp=splitter(s)
        for tr,te in (sp.split(X,y,groups) if groups is not None else sp.split(X,y)):
            m=mk().fit(X[tr],y[tr]); p[te]=m.predict(X[te])
        preds.append(p)
    return np.mean(preds,0)
schemes={'random 5-fold (as in manuscript)':(lambda s:KFold(5,shuffle=True,random_state=s),None),
         'grouped by well, 5-fold':(lambda s:GroupKFold(5),d.well.values),
         'spatial 1-degree blocks, 5-fold':(lambda s:GroupKFold(5),d.blk.values)}
rows=[]
for name,(sp,gr) in schemes.items():
    p=oof(sp,gr); r=dict(scheme=name,all=r2_score(y,p))
    for k in ['<30','30-80','80-150','>150']:
        m=(d.dc==k).values; r[k]=r2_score(y[m],p[m])
    rows.append(r); P(f"  {name:34s} R2 all {r['all']:.3f} | "+" ".join(f"{k}:{r[k]:.3f}" for k in ['<30','30-80','80-150','>150']))
pd.DataFrame(rows).to_csv(f"{O}/r06_cv.csv",index=False)
P("\nwithin-class models, grouped-by-well 5-fold CV, and SHAP share (mean |SHAP| %) with bootstrap rank stability (B=50)")
srows=[]
for k in ['<30','30-80','80-150','>150']:
    m=(d.dc==k).values; Xk,yk,gk=X[m],y[m],d.well.values[m]
    p=np.zeros(len(yk))
    for tr,te in GroupKFold(5).split(Xk,yk,gk): p[te]=mk().fit(Xk[tr],yk[tr]).predict(Xk[te])
    mod=mk().fit(Xk,yk); sv=np.abs(shap.TreeExplainer(mod).shap_values(Xk)).mean(0); share=100*sv/sv.sum()
    rng=np.random.default_rng(42); ranks={'PO43-':[],'Fe2+':[],'HCO3-':[]}; ws=np.unique(gk)
    for b in range(50):
        pick=rng.choice(ws,len(ws)); idx=np.concatenate([np.where(gk==w)[0] for w in pick])
        mb=mk().fit(Xk[idx],yk[idx]); s=np.abs(shap.TreeExplainer(mb).shap_values(Xk[idx])).mean(0)
        order=(-s).argsort().argsort()+1
        for f in ranks: ranks[f].append(order[F.index(f)])
    top=[F[i] for i in (-share).argsort()[:4]]
    P(f"  {k:7s} n={m.sum():4d} CV R2={r2_score(yk,p):.3f} | SHAP PO4 {share[F.index('PO43-')]:.1f}% (rank median {np.median(ranks['PO43-']):.0f}, 90% {np.percentile(ranks['PO43-'],[5,95]).astype(int)}) "
      f"Fe {share[F.index('Fe2+')]:.1f}% (rank {np.median(ranks['Fe2+']):.0f}, {np.percentile(ranks['Fe2+'],[5,95]).astype(int)}) HCO3 {share[F.index('HCO3-')]:.1f}% (rank {np.median(ranks['HCO3-']):.0f}) | top4 {top}")
    srows.append(dict(depth=k,n=int(m.sum()),cv_r2=r2_score(yk,p),shap_PO4=share[F.index('PO43-')],shap_Fe=share[F.index('Fe2+')],shap_HCO3=share[F.index('HCO3-')],
        rank_PO4_med=np.median(ranks['PO43-']),rank_PO4_lo=np.percentile(ranks['PO43-'],5),rank_PO4_hi=np.percentile(ranks['PO43-'],95),
        rank_Fe_med=np.median(ranks['Fe2+']),rank_Fe_lo=np.percentile(ranks['Fe2+'],5),rank_Fe_hi=np.percentile(ranks['Fe2+'],95)))
pd.DataFrame(srows).to_csv(f"{O}/r06_shap_by_depth.csv",index=False)
open(f"{O}/r06_summary.txt","w").write("\n".join(out))
