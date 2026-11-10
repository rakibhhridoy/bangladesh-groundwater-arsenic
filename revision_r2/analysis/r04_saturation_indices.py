"""R2 Results-2: secondary Fe sinks (siderite) decoupling Fe from As and PO4.
Saturation indices by PHREEQC (phreeqpython, wateq4f.dat) from field pH and temperature
and laboratory major ions. Dissolved Fe is entered as Fe(II) (total dissolved Fe; the
source reports it as Fe2+), so SI(siderite) and SI(vivianite) are upper bounds in
oxic waters. Tests whether the As-PO4 association depends on siderite saturation."""
import os
from pathlib import Path
import pandas as pd, numpy as np
from scipy import stats
from scipy.stats import rankdata
import phreeqpython
R=os.environ.get("PAPER1_ROOT", str(Path(__file__).resolve().parents[2])); O=f"{R}/revision_r2/output"
d=pd.read_csv(f"{R}/data/integrated/bangladesh_groundwater_arsenic_1807samples_CLEAN.csv")
d['well']=pd.read_csv(f"{O}/r01c_station_wellkey.csv").well.values
d=d[(d.pH>3)&(d.pH<11)&d['HCO3-'].notna()&d['Fe2+'].notna()&d['PO43-'].notna()&d.As.notna()].copy()
d['T']=d.Temperature.where((d.Temperature>15)&(d.Temperature<40),26.0)
pp=phreeqpython.PhreeqPython(database='wateq4f.dat')
MIN=['Siderite','Vivianite','Calcite','Rhodochrosite','Hydroxyapatite']
z=lambda v: max(float(v),0.0) if pd.notna(v) else 0.0
res=[]
for r in d.itertuples(index=False):
    sol={'units':'mg/L','temp':r.T,'pH':r.pH,'Ca':z(r._6),'Mg':z(r._5),'Na':z(r._3),'K':z(r._4),'Cl':z(r._12),
         'S(6)':f"{z(r._8)} as SO4",'N(5)':f"{z(r._7)} as NO3",'Alkalinity':f"{z(r._11)+z(r._10)*61/30} as HCO3",
         'Fe(2)':z(r._17),'Mn(2)':z(r._18),'P':f"{max(z(r._9),1e-4)} as PO4"}
    s=pp.add_solution(sol); res.append([s.si(m) for m in MIN]+[s.I]); s.forget()
S=pd.DataFrame(res,columns=['SI_'+m for m in MIN]+['I'],index=d.index); d=d.join(S)
d.to_csv(f"{O}/r04_saturation_indices.csv",index=False)
out=[]; P=lambda *a:(out.append(" ".join(str(x) for x in a)),print(*a,flush=True))
P(f"n={len(d)}")
for m in MIN:
    c='SI_'+m; P(f"  {m:15s} median SI {d[c].median():+.2f}  IQR [{d[c].quantile(.25):+.2f},{d[c].quantile(.75):+.2f}]  supersaturated {100*(d[c]>0).mean():.1f}%  (>|0.5| {100*(d[c]>0.5).mean():.1f}%)")
d['dc']=pd.cut(d.Depth,[0,30,80,150,1e4],right=False,labels=['<30','30-80','80-150','>150'])
P("siderite supersaturated % by depth class:",d.groupby('dc',observed=True).SI_Siderite.apply(lambda x:round(100*(x>0).mean(),1)).to_dict())
def prho(s,x,zs,y='As'):
    C=np.c_[np.ones(len(s)),*[rankdata(s[c]) for c in zs]]
    r=lambda v:v-C@np.linalg.lstsq(C,v,rcond=None)[0]
    a,b=r(rankdata(s[y])),r(rankdata(s[x])); return stats.pearsonr(a,b)
def boot(s,fn,B=1000,seed=42):
    rng=np.random.default_rng(seed); grp=[v for _,v in s.groupby('well')]; n=len(grp)
    return np.nanpercentile([fn(pd.concat([grp[i] for i in rng.integers(0,n,n)])) for _ in range(B)],[2.5,97.5])
P("\nAs-PO4 association by siderite saturation state")
rows=[]
for lab,s in [('undersaturated SI<0',d[d.SI_Siderite<0]),('near equilibrium |SI|<=0.5',d[d.SI_Siderite.abs()<=0.5]),('supersaturated SI>0',d[d.SI_Siderite>0]),('strongly under SI<-0.5',d[d.SI_Siderite<-0.5]),('strongly super SI>0.5',d[d.SI_Siderite>0.5])]:
    r0=stats.spearmanr(s.As,s['PO43-']); r1=prho(s,'PO43-',['Fe2+']); rf=stats.spearmanr(s.As,s['Fe2+'])
    ci=boot(s,lambda q:stats.spearmanr(q.As,q['PO43-'])[0]); ci1=boot(s,lambda q:prho(q,'PO43-',['Fe2+'])[0])
    P(f"  {lab:28s} n={len(s):4d} As-PO4 {r0[0]:+.3f} [{ci[0]:+.2f},{ci[1]:+.2f}] p={r0[1]:.1g} | As-PO4|Fe {r1[0]:+.3f} [{ci1[0]:+.2f},{ci1[1]:+.2f}] | As-Fe {rf[0]:+.3f} | median Fe {s['Fe2+'].median():.2f} PO4 {s['PO43-'].median():.2f} As {s.As.median():.2f}")
    rows.append(dict(group=lab,n=len(s),rho=r0[0],lo=ci[0],hi=ci[1],prho=r1[0],plo=ci1[0],phi=ci1[1],rho_Fe=rf[0]))
pd.DataFrame(rows).to_csv(f"{O}/r04_by_siderite_state.csv",index=False)
# interaction test: does SI_siderite moderate the As-PO4 slope? rank regression
import statsmodels.formula.api as smf
q=d.assign(rA=rankdata(d.As)/len(d),rP=rankdata(d['PO43-'])/len(d),rF=rankdata(d['Fe2+'])/len(d),sup=(d.SI_Siderite>0).astype(int))
m=smf.ols('rA~rP*sup+rF',q).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(q.well)[0]})
P(f"\nrank regression rA ~ rPO4*supersat + rFe (cluster SE): rPO4 {m.params['rP']:.3f} (p={m.pvalues['rP']:.1g}); rPO4:supersat {m.params['rP:sup']:+.3f} (p={m.pvalues['rP:sup']:.2f})")
# Fe:As decoupling expected under siderite removal: Fe/As lower where supersaturated at given As?
P(f"Spearman SI_siderite vs As {stats.spearmanr(d.SI_Siderite,d.As)[0]:+.3f}; vs PO4 {stats.spearmanr(d.SI_Siderite,d['PO43-'])[0]:+.3f}; vs Fe {stats.spearmanr(d.SI_Siderite,d['Fe2+'])[0]:+.3f}")
P(f"Vivianite supersaturated {100*(d.SI_Vivianite>0).mean():.1f}%; As-PO4 where vivianite undersaturated: {stats.spearmanr(d[d.SI_Vivianite<0].As,d[d.SI_Vivianite<0]['PO43-'])[0]:+.3f} (n={(d.SI_Vivianite<0).sum()})")
open(f"{O}/r04_summary.txt","w").write("\n".join(out))
