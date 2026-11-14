"""Figure 5 for the GSD major revision. Same four-panel layout as the submitted
figure (JHMRevision/make_figures_hq.py: fig_h2_spline), with three changes:
panel A drops the 68 m crossover line and its CI band (crossover depth withdrawn,
see analysis/r07) and adds well-level bootstrap 95% bands; panels B and D use the
class partials As-PO4|Fe and As-Fe|PO4 of the revised table (analysis/r03), with
bootstrap 95% CIs in D. Panel C draws arsenic in the warm arsenic ramp
(it used the phosphate colour in the submitted figure)."""
import os
from pathlib import Path
import sys, numpy as np, pandas as pd
R=os.environ.get("PAPER1_ROOT", str(Path(__file__).resolve().parents[2])); sys.path.insert(0,f"{R}/revision_r2/figures/lib")
import make_figures_hq as M, figstyle as fs
from scipy.stats import rankdata
src=open(f"{R}/revision_r2/figures/lib/make_figures_hq.py").read()
a=src.index("def fig_h2_spline():"); b=src.index("def fig_h3():"); code=src[a:b]
# panel A: remove crossover band and line
code=code.replace("""    axA.axhspan(32, 153, color=fs.GREY, alpha=0.16, lw=0, zorder=0,
                label='95% CI 32–153 m')
    axA.axhline(68, color=fs.INK, ls='--', lw=1.1, zorder=1, label='crossover 68 m')
""","""    axA.fill_betweenx(cen, bandP[0], bandP[1], color=fs.PO4, alpha=0.16, lw=0, zorder=0)
    axA.fill_betweenx(cen, bandF[0], bandF[1], color=fs.FE, alpha=0.16, lw=0, zorder=0)
    axA.axvline(0, color=fs.INK, lw=0.6, zorder=0)
""")
assert "bandP" in code
code=code.replace("    # (A)\n","""    # well-level bootstrap bands for the rolling curves
    key=pd.read_csv(f"{R}/revision_r2/output/r01b_station_key.csv")
    dd['well']=key.loc[dd.index,'sid'].values
    grp=[g for _,g in dd.groupby('well')]; rng=np.random.default_rng(42); BP=[];BF=[]
    for _ in range(300):
        s=pd.concat([grp[i] for i in rng.integers(0,len(grp),len(grp))])
        bp=[];bf=[]
        for c in cen:
            w=s[(s['Depth']>=c-40)&(s['Depth']<c+40)]
            bp.append(roll(w,'PO43-','Fe2+')); bf.append(roll(w,'Fe2+','PO43-'))
        BP.append(bp); BF.append(bf)
    bandP=np.nanpercentile(BP,[2.5,97.5],axis=0); bandF=np.nanpercentile(BF,[2.5,97.5],axis=0)
    # (A)
""")
# panels B and D: revised class partials with CIs
code=code.replace("    # (D) zone-avg partials\n","""    # (D) class partials with bootstrap CIs
""")
code=code.replace("""    axD.bar(x-w/2, pc['PO43-'], w, color=fs.PO4, label=r'PO$_4^{3-}$', edgecolor='white',
            linewidth=0.3, hatch=fs.HATCH_1)
    axD.bar(x+w/2, pc['Fe2+'], w, color=fs.FE, label=r'Fe$^{2+}$', edgecolor='white',
            linewidth=0.3, hatch=fs.HATCH_2)""","""    axD.bar(x-w/2, pc['PO43-'], w, color=fs.PO4, label=r'PO$_4^{3-}$ | Fe', edgecolor='white',
            linewidth=0.3, hatch=fs.HATCH_1, yerr=[pc['PO43-']-ci['plo'], ci['phi']-pc['PO43-']],
            error_kw=dict(ecolor=fs.INK, lw=0.8, capsize=2))
    axD.bar(x+w/2, pc['Fe2+'], w, color=fs.FE, label=r'Fe$^{2+}$ | PO$_4$', edgecolor='white',
            linewidth=0.3, hatch=fs.HATCH_2)
    axD.axhline(0, color=fs.INK, lw=0.6)""")
code=code.replace("axD.set_ylim(0, max(pc['PO43-'].max(), pc['Fe2+'].max())*1.28)","axD.set_ylim(min(-0.06, ci['plo'].min()-0.02), max(ci['phi'].max(), pc['Fe2+'].max())*1.22)")
# panel C: arsenic in the warm arsenic ramp, not the phosphate colour
code=code.replace("axC.fill_between(mid, q1, q3, color=fs.PO4, alpha=0.18","axC.fill_between(mid, q1, q3, color=fs.WARMR[4], alpha=0.22")
code=code.replace("axC.plot(mid, med, 'o-', color=fs.PO4,","axC.plot(mid, med, 'o-', color=fs.WARM2,")
assert "fs.PO4, alpha=0.18, lw=0, label='interquartile" not in code
ns={**M.__dict__}; 
t=pd.read_csv(f"{R}/revision_r2/output/r03_depthwise_correlations.csv").set_index('group')
rows=['<30 m','30-80 m','80-150 m','>150 m']
ns['pc']=pd.DataFrame({'PO43-':t.loc[rows,'prho_PO4_Fe'].values,'Fe2+':t.loc[rows,'prho_Fe_PO4'].values})
ns['ci']=pd.DataFrame({'plo':t.loc[rows,'plo'].values,'phi':t.loc[rows,'phi'].values})
ns['R']=R
exec(code,ns)
fig=ns['fig_h2_spline']()
out=fs.save_all(fig,'H2_F_depth_spline',__import__('pathlib').Path(f"{R}/revision_r2/output"))
print('wrote',out)
