"""Builds supp_tables_r2.tex (Supplementary Tables S6 revised, S7-S15) from the
analysis outputs, and recomputes the paired between-campaign comparison (S2) on
station-ID pairs. Prints the numbers quoted in the S2 text."""
import os
from pathlib import Path
import pandas as pd, numpy as np
from scipy import stats
R=os.environ.get("PAPER1_ROOT", str(Path(__file__).resolve().parents[2])); G=f"{R}/revision_r2/output"; A=G
L=[]; T=lambda x:L.append(x)
def tab(caption,label,colspec,header,rows,size=r"\small"):
    T(r"\begin{table}[H]"+"\n"+r"\centering"+"\n"+rf"\caption{{{caption}}}"+"\n"+rf"\label{{{label}}}"+"\n"+size+"\n"+rf"\begin{{tabular}}{{{colspec}}}"+"\n"+r"\toprule")
    T(header+r" \\"+"\n"+r"\midrule")
    for r in rows: T(r+r" \\")
    T(r"\bottomrule"+"\n"+r"\end{tabular}"+"\n"+r"\end{table}"+"\n")
LEQ=r'$\leq$'; GEQ=r'$\geq$'; ABS=r'$|$SI$|$'
f2=lambda v:f"{v:.2f}"; f3=lambda v:f"{v:.3f}"; f1=lambda v:f"{v:.1f}"
def pv(p):
    if p<1e-3:
        m,e=f"{p:.1e}".split('e'); return rf"${m}\times10^{{{int(e)}}}$"
    return f"{p:.3f}"
# ---------- S6 paired, station ID ----------
d=pd.read_csv(f"{R}/data/integrated/bangladesh_groundwater_arsenic_1807samples_CLEAN.csv")
d['sid']=d['Sample ID'].astype(str).str.strip(); d=d[~d.sid.str.endswith('-')]
V=['As','Fe2+','PO43-','ORP','EC','Cl-','pH','HCO3-','Depth']
p=d.pivot_table(index='sid',columns='Season',values=V,aggfunc='first').dropna(subset=[('As','Dry'),('As','Wet')])
p['dc']=pd.cut(p['Depth','Dry'],[0,30,80,150,1e4],right=False,labels=['Shallow','Intermediate','Medium-Deep','Deep'])
rows=[]
for k in ['Shallow','Intermediate','Medium-Deep','Deep',None]:
    q=p if k is None else p[p.dc==k]; a,b=q['As','Dry'],q['As','Wet']
    rows.append(f"{'All' if k is None else k} & {len(q)} & {f2(a.median())} & {f2(b.median())} & {pv(stats.wilcoxon(a,b).pvalue)} & {f1(100*(b>a).mean())}")
tab(r"Within-well paired between-campaign change in arsenic for the 878 wells sampled in both campaigns, matched by station identifier (medians in $\mu$g/L; depth class from the dry-season record). Wilcoxon signed-rank test.",
    "tab:s2_paired","lrrrrr",r"\textbf{Depth class} & \textbf{$n$} & \textbf{Median dry} & \textbf{Median wet} & \textbf{Wilcoxon $p$} & \textbf{\% wet $>$ dry}",rows)
a,b=p['As','Dry'],p['As','Wet']
print(f"S2: n={len(p)} median {a.median():.2f}->{b.median():.2f} p={stats.wilcoxon(a,b).pvalue:.2g} wet>dry {100*(b>a).mean():.1f}%")
print(f"   exceed {100*(a>10).mean():.1f}->{100*(b>10).mean():.1f}; safe->unsafe {100*((a<=10)&(b>10)).mean():.1f}% unsafe->safe {100*((a>10)&(b<=10)).mean():.1f}%")
for v in ['EC','Cl-','pH','HCO3-']: print(f"   {v} median {p[v,'Dry'].median():.2f}->{p[v,'Wet'].median():.2f}")
dA=p['As','Wet']-p['As','Dry']
for v in ['Fe2+','ORP','PO43-']:
    dv=p[v,'Wet']-p[v,'Dry']; m=dA.notna()&dv.notna(); print(f"   dAs~d{v} rho {stats.spearmanr(dA[m],dv[m])[0]:.3f}")
print(f"   PO4 median {p['PO43-','Dry'].median():.2f}->{p['PO43-','Wet'].median():.2f}")
# ---------- S7 geology vs depth ----------
sh=pd.read_csv(f"{A}/r02_shallow_test.csv")
rows=[]
for r in sh.itertuples():
    rows.append(f"{r.cut.replace('<','$<$')} & {r.n} & {f3(r.eps2)} & {r.hol_n} & {f1(r.hol_exc)} [{f1(r.hol_lo)}, {f1(r.hol_hi)}] & {r.old_n} & {f1(r.old_exc)} [{f1(r.old_lo)}, {f1(r.old_hi)}] & {f2(r.po4_hol)} / {f2(r.po4_old)}")
tab(r"Deposit-type contrast restricted to shallow wells. Exceedance is the percentage above 10~$\mu$g/L, with 95\% intervals from 2{,}000 bootstrap resamples of whole wells. $\varepsilon^2$ is the Kruskal--Wallis effect size across deposit types with at least fifteen samples. Phosphate is the median (mg/L) on Holocene / Pleistocene sediment.",
    "tab:s_shallow","lrrrrrrr",r"\textbf{Wells} & \textbf{$n$} & \textbf{$\varepsilon^2$} & \textbf{$n$ Hol.} & \textbf{Holocene \%} & \textbf{$n$ Pl.} & \textbf{Pleistocene \%} & \textbf{PO$_4^{3-}$}",rows,r"\footnotesize")
txt=open(f"{A}/r02_summary.txt").read()
rows=[r"Deposit type (17 units), $\varepsilon^2$ & 0.135",
      r"Depth, 4 operational classes, $\varepsilon^2$ & 0.008",
      r"Depth, 17 quantile bins, $\varepsilon^2$ & 0.048",
      r"Depth, linear (log depth), cross-validated $R^2$ & 0.007",
      r"Depth, spline df 4 / 8 / 16, cross-validated $R^2$ & 0.008 / 0.008 / 0.021",
      r"Deposit type, cross-validated $R^2$ & 0.125",
      r"Deposit type + depth spline, cross-validated $R^2$ & 0.138",
      r"Joint model, unique share: deposit type / depth & 0.149 / 0.018",
      r"Deposit type on depth-detrended ranks, $\varepsilon^2$ & 0.141",
      r"Well-level medians (859 wells): deposit / depth class $\varepsilon^2$ & 0.199 / 0.010",
      r"178 depth-matched pairs ($\pm$5 m), Holocene / Pleistocene \% $>$WHO & 43.8 / 3.9",
      r"Mantel--Haenszel odds ratio across depth classes [95\% CI] & 18.9 [9.4, 38.3]",
      r"Excluding 26 identifiers in two deposit types, $\varepsilon^2$ & 0.137",
      r"Same exclusion, Holocene / Pleistocene \% $>$WHO & 46.3 / 3.9"]
tab(r"Mapped deposit type compared with depth on equal terms (1{,}662 samples, 859 wells). Cross-validated $R^2$ is for arsenic ranks with folds that hold out whole wells (10 folds).",
    "tab:s_fair","lr",r"\textbf{Comparison} & \textbf{Value}",rows)
# ---------- S8 alternative depth schemes ----------
t=pd.read_csv(f"{A}/r03_depthwise_correlations.csv")
alt=t[t.group.str.contains(':')]
NM={'DPHE shallow/deep at 150 m':'Two-tier, 150 m','three-tier 50/150 m':'Three-tier, 50 and 150 m','quartiles':'Depth quartiles'}
rows=[f"{NM[g.split(': ')[0]]} & {g.split(': ')[1].replace('<','$<$').replace('>',GEQ)} & {n} & {f3(r)} [{f2(lo)}, {f2(hi)}] & {f3(pp)} [{f2(pl)}, {f2(ph)}] & {f3(rf)}"
      for g,n,r,lo,hi,pp,pl,ph,rf in alt[['group','n','rho_PO4','lo_PO4','hi_PO4','prho_PO4_Fe','plo','phi','rho_Fe']].itertuples(index=False)]
tab(r"Arsenic--phosphate association under alternative depth schemes. Spearman $\rho$ and the partial $\rho$ controlling for Fe$^{2+}$, with 95\% intervals from 1{,}000 bootstrap resamples of whole wells. Quartile boundaries fall at 30, 72 and 116~m.",
    "tab:s_altbins","llrrrr",r"\textbf{Scheme} & \textbf{Class} & \textbf{$n$} & \textbf{As--PO$_4^{3-}$} & \textbf{As--PO$_4^{3-}$ $|$ Fe} & \textbf{As--Fe$^{2+}$}",rows,r"\footnotesize")
# ---------- S9 crossover grid ----------
gr=pd.read_csv(f"{A}/r07_grid.csv"); rows=[]
for r in gr.itertuples():
    rows.append(f"{r.cov.split(' (')[0]} & {r.win} & {r.first:.0f} & {r.sustained:.0f}")
tab(r"Sensitivity of the apparent depth at which the partial correlation of arsenic with Fe$^{2+}$ overtakes that with PO$_4^{3-}$ (m), by the additional covariates controlled, the rolling-window width (m), and the crossing rule (first sign change, or first change sustained over three windows). The submitted manuscript reported 68~m from the third covariate set with an 80~m window. Below, the difference between the two partial correlations (Fe minus PO$_4^{3-}$, all covariates) by depth class with 95\% intervals from 1{,}000 bootstrap resamples of whole wells: $<$30~m $-0.137$ [$-0.304$, 0.047]; 30--80~m $-0.094$ [$-0.234$, 0.038]; 80--150~m 0.094 [$-0.057$, 0.232]; $\geq$150~m 0.101 [$-0.072$, 0.272].",
    "tab:s_grid","lrrr",r"\textbf{Covariates beyond Fe / PO$_4^{3-}$} & \textbf{Window} & \textbf{First} & \textbf{Sustained}",
    [x.replace('Fe/PO4 only','none').replace('+HCO3,pH,DO','HCO$_3^-$, pH, DO').replace('+HCO3','HCO$_3^-$') for x in rows])
# ---------- S10 physiographic units ----------
z=pd.read_csv(f"{A}/r03_zones.csv"); rows=[]
for r in z.itertuples():
    ls='PO$_4$' if r.po4_sh>r.fe_sh else 'Fe'; ld='PO$_4$' if r.po4_dp>r.fe_dp else 'Fe'
    rows.append(f"{r.zone} & {r.n_sh} & {f2(r.po4_sh)} / {f2(r.fe_sh)} ({ls}) & {r.n_dp} & {f2(r.po4_dp)} / {f2(r.fe_dp)} ({ld})")
tab(r"Partial correlations of arsenic with PO$_4^{3-}$ (controlling Fe$^{2+}$) and with Fe$^{2+}$ (controlling PO$_4^{3-}$) above and below 80~m, by physiographic unit \citep{Persits1997}, for units with at least 20 samples in each interval. The larger of the two is given in parentheses.",
    "tab:s_zones","lrlrl",r"\textbf{Unit} & \textbf{$n<$80 m} & \textbf{PO$_4$ / Fe} & \textbf{$n\geq$80 m} & \textbf{PO$_4$ / Fe}",rows,r"\footnotesize")
# ---------- S11 saturation ----------
si=pd.read_csv(f"{A}/r04_saturation_indices.csv"); rows=[]
for m in ['Siderite','Vivianite','Calcite','Rhodochrosite','Hydroxyapatite']:
    c=si['SI_'+m]; rows.append(f"{m} & {c.median():+.2f} & [{c.quantile(.25):+.2f}, {c.quantile(.75):+.2f}] & {f1(100*(c>0).mean())}")
tab(r"Saturation indices (PHREEQC, \texttt{wateq4f} database; $n = 1{,}806$). Dissolved iron is entered as Fe(II), so values for siderite and vivianite are upper bounds in oxic waters.",
    "tab:s_si","lrrr",r"\textbf{Mineral} & \textbf{Median SI} & \textbf{IQR} & \textbf{\% supersaturated}",rows)
bs=pd.read_csv(f"{A}/r04_by_siderite_state.csv"); rows=[]
for r in bs.itertuples():
    g=r.group.replace('<=',LEQ).replace('<','$<$').replace('>','$>$').replace('|SI|',ABS)
    rows.append(f"{g} & {r.n} & {f3(r.rho)} [{f2(r.lo)}, {f2(r.hi)}] & {f3(r.prho)} [{f2(r.plo)}, {f2(r.phi)}] & {f3(r.rho_Fe)}")
tab(r"Arsenic--phosphate association by siderite saturation state, with 95\% intervals from 1{,}000 bootstrap resamples of whole wells. The interaction between phosphate and siderite supersaturation in a rank regression of arsenic (with Fe$^{2+}$, well-clustered errors) is $-0.019$ ($p = 0.74$).",
    "tab:s_sid","lrrrr",r"\textbf{Siderite state} & \textbf{$n$} & \textbf{As--PO$_4^{3-}$} & \textbf{As--PO$_4^{3-}$ $|$ Fe} & \textbf{As--Fe$^{2+}$}",rows,r"\footnotesize")
# ---------- S12 GB validation ----------
cv=pd.read_csv(f"{A}/r06_cv.csv"); rows=[f"{r.scheme} & {f3(r.all)} & {f3(r._3)} & {f3(r._4)} & {f3(r._5)} & {f3(r._6)}" for r in cv.itertuples()]
tab(r"Gradient-boosted model, five-fold cross-validated $R^2$ on $\log(1+\mathrm{As})$ overall and out-of-fold within each depth class, under three partitions.",
    "tab:s_cv","lrrrrr",r"\textbf{Partition} & \textbf{All} & \textbf{$<$30} & \textbf{30--80} & \textbf{80--150} & \textbf{$\geq$150}",rows,r"\footnotesize")
sb=pd.read_csv(f"{A}/r06_shap_by_depth.csv"); rows=[]
for r in sb.itertuples():
    dl=r.depth.replace('<','$<$').replace('>',GEQ)
    rows.append(f"{dl} & {r.n} & {f3(r.cv_r2)} & {f1(r.shap_PO4)} ({r.rank_PO4_med:.0f}; {r.rank_PO4_lo:.0f}--{r.rank_PO4_hi:.0f}) & {f1(r.shap_Fe)} ({r.rank_Fe_med:.0f}; {r.rank_Fe_lo:.0f}--{r.rank_Fe_hi:.0f}) & {f1(r.shap_HCO3)}")
tab(r"Models fitted within each depth class. Cross-validated $R^2$ holds out whole wells. SHAP share is the percentage of mean absolute SHAP value, with the median rank among 14 features and its 90\% range over 50 bootstrap resamples of whole wells in parentheses.",
    "tab:s_shap","lrrrrr",r"\textbf{Depth (m)} & \textbf{$n$} & \textbf{CV $R^2$} & \textbf{PO$_4^{3-}$ \%} & \textbf{Fe$^{2+}$ \%} & \textbf{HCO$_3^-$ \%}",rows,r"\footnotesize")
# ---------- S13 boundary distance ----------
bd=pd.read_csv(f"{A}/r08_boundary_distance.csv")
HOL={'dsd','ac','dsl','ppc','dt'}; OLD={'afo','rb','rm','QTdd'}; rows=[]
def e2(s):
    sub=[x.As.values for _,x in s.groupby('GLG') if len(x)>=15]; H,_=stats.kruskal(*sub); k=len(sub); N=sum(map(len,sub)); return (H-k+1)/(N-k),k
for c in (0,0.5,1,2,3,5):
    s=bd[bd.dist_km>=c]; h=s[s.GLG.isin(HOL)]; o=s[s.GLG.isin(OLD)]; e,k=e2(s)
    rows.append(f"{c:g} & {len(s)} & {f3(e)} ({k}) & {len(h)} & {f1(100*(h.As>10).mean())} & {len(o)} & {f1(100*(o.As>10).mean())}")
tab(r"Deposit-type contrast after excluding samples near a mapped unit boundary of the 1:1{,}000{,}000 geological map. Median distance to the nearest boundary is 1.4~km.",
    "tab:s_bound","rrrrrrr",r"\textbf{Min. distance (km)} & \textbf{$n$} & \textbf{$\varepsilon^2$ ($k$)} & \textbf{$n$ Hol.} & \textbf{Hol. \%} & \textbf{$n$ Pl.} & \textbf{Pl. \%}",rows,r"\footnotesize")
# ---------- S14 record consistency ----------
inc=pd.read_csv(f"{A}/r01b_inconsistent_ids.csv")
rows=[r"Station identifiers in the data & 924",
      r"Sampled in both campaigns & 878",
      r"Dry-season campaign only / wet-season only & 28 / 18",
      r"Records with a truncated identifier & 5",
      rf"Paired identifiers with depth differing by $>$5~m & {int((inc.dz>5).sum())}",
      rf"Paired identifiers with coordinates differing by $>$1~km & {int((inc.km>1).sum())}",
      r"Identifiers whose two records map to different deposit types & 26",
      r"As--PO$_4^{3-}$ $\rho$, all records / excluding inconsistent & 0.343 / 0.326",
      r"Holocene / Pleistocene \% $>$WHO, excluding inconsistent & 46.5 / 4.1"]
tab(r"Well identity across campaigns. The source report describes 901 wells \citep{BWDB2023}; the excess of identifiers in the data is attributed to identifiers recorded differently between campaigns. Inconsistent records are retained in all analyses.",
    "tab:s_ids","lr",r"\textbf{Item} & \textbf{Value}",rows)
# ---------- S15 analytical methods ----------
M=[("Ca, Mg, Na, K","AAS","AAS"),("As, Cr","AAS","AAS"),("Fe, Mn, Cu, Al","UV-VIS spectrophotometry","AAS"),
   (r"PO$_4^{3-}$ (molybdenum blue)","VIS spectrophotometry","VIS spectrophotometry"),
   (r"NO$_3^-$, SO$_4^{2-}$, B, Br$^-$, I$^-$","VIS spectrophotometry","VIS spectrophotometry"),
   (r"CO$_3^{2-}$, HCO$_3^-$, Cl$^-$","Titration","Titration")]
tab(r"Analytical method by analyte and campaign, from \citet[Table 3-1]{BWDB2023}.","tab:s_methods","lll",
    r"\textbf{Analyte} & \textbf{Wet season 2020} & \textbf{Dry season 2021}",[f"{a} & {b} & {c}" for a,b,c in M])
ORDER=["tab:s2_paired","tab:s_shallow","tab:s_altbins","tab:s_grid","tab:s_zones","tab:s_fair","tab:s_si","tab:s_sid","tab:s_cv","tab:s_shap","tab:s_bound","tab:s_ids","tab:s_methods"]
blocks={}
for b in L:
    import re
    m=re.search(r"\\label\{([^}]*)\}",b); 
    if m: blocks[m.group(1)]=b
    else: blocks.setdefault('_rows',[]).append(b)
# reassemble: each table is begin..end across several L entries
txt="\n".join(L); parts=re.split(r"(?=\\begin\{table\}\[H\])",txt); bl={}
for part in parts:
    m=re.search(r"\\label\{([^}]*)\}",part)
    if m: bl[m.group(1)]=part
assert set(bl)==set(ORDER),set(bl)^set(ORDER)
open(f"{G}/supp_table_paired_r2.tex","w").write(bl[ORDER[0]])
open(f"{G}/supp_tables_r2.tex","w").write("\n".join(bl[k] for k in ORDER[1:]))
for i,k in enumerate(ORDER): print(f"S{6+i} {k}"); print("wrote supp_tables_r2.tex")
