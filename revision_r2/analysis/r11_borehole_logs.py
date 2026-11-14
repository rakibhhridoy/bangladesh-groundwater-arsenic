"""R2 Methodology-4: match nested monitoring wells to the digitised BWDB borehole
logs (r10) and test whether the deposit-type contrast holds once the subsurface
position of each well is known.

Per well (station identifier, median over campaigns):
  lith_at_depth   lithology of the log at the well's recorded depth
  fine_above_m    total clay + silt/silty-clay thickness between surface and well
  aquitards_above number of fine-grained units >= AQ m thick above the well
  first_aquifer   True if no such unit separates the well from the surface
Tests: arsenic by first vs deeper aquifer; arsenic vs fine thickness (partial on
depth); Holocene vs Pleistocene exceedance among logged wells, among
first-aquifer wells and among wells beneath an aquitard; rank regression with
deposit age, depth, aquifer position and fine thickness, nest-cluster bootstrap.
"""
import os
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import rankdata
import statsmodels.formula.api as smf

R = os.environ.get("PAPER1_ROOT", str(Path(__file__).resolve().parents[2]))
O = f"{R}/revision_r2/output"
HOL = {'dsd', 'ac', 'dsl', 'ppc', 'dt'}; OLD = {'afo', 'rb', 'rm', 'QTdd'}
FINE = {"clay", "silt/silty clay"}
AQ = 5.0   # minimum aquitard thickness (m); 10 m as sensitivity

# log image -> BWDB nest prefix, from the site name on each log.
# 'depth' = one of two nests at the same place, chosen as the nest whose depth
# range fits the log; 'ambiguous' = two nests, no basis to choose, excluded.
MATCH = {
 "img-040-000": ("SYSYPZ", "name"), "img-040-001": ("MVMVPZ", "name"), "img-040-002": ("CHHAPZ", "name"),
 "img-041-003": ("DHGRPZ", "name"), "img-041-004": ("DHGOPZ", "name"), "img-041-005": ("CNGOPZ", "name"),
 "img-041-006": ("JYJYPZ", "name"), "img-041-007": ("DIDIPZ", "name"), "img-042-008": ("CDCDPZ", "name"),
 "img-042-009": ("CUCUPZ", "name"), "img-042-010": ("FNFNPZ", "name"), "img-043-011": ("GAGAPZ", "name"),
 "img-043-012": ("KUKUPZ", "name"), "img-043-013": ("PGPGPZ", "name"), "img-043-014": ("PGTTPZ", "name"),
 "img-043-015": ("RARAPZ", "name"), "img-043-016": ("THTHPZ", "depth"), "img-044-017": ("NLSYPZ", "name"),
 "img-044-018": ("NLDAPZ", "name"), "img-044-019": ("DHBAPZ", "name"), "img-044-020": ("DHMJPZ", "name"),
 "img-044-021": ("DHJAPZ", "name"), "img-045-022": ("MNMNPZ", "name"), "img-045-023": ("MUSRPZ", "name"),
 "img-045-024": ("NGNGPZ", "name"), "img-045-025": ("NGRUPZ", "name"), "img-045-026": ("NGSHPZ", "name"),
 "img-045-027": ("NSNSPZ", "name"), "img-045-028": ("TATAPZ", "name"), "img-045-029": ("JAJAPZ", "name"),
 "img-045-030": ("MYMYPZ", "name"), "img-046-031": ("SRSRPZ", "name"), "img-046-032": ("CNCNPZ", "name"),
 "img-046-033": ("NANAPZ", "name"), "img-046-034": ("NTNTPZ", "name"), "img-046-035": ("PAPAPZ", "name"),
 "img-046-036": ("RJRJPZ", "name"), "img-046-037": ("SJULPZ", "name"), "img-046-038": ("KSKSPZ", "name"),
 "img-046-039": ("BRBRPZ", "name"), "img-047-040": ("BOBOPZ", "name"), "img-047-041": ("BOSAPZ", "name"),
 "img-047-042": ("CHSAPZ|CHSDPZ", "ambiguous"), "img-047-043": ("CXBAPZ", "name"),
 "img-047-044": ("NKHTPZ|NKHT2PZ", "ambiguous"), "img-048-045": ("NKCMPZ", "name"),
 "img-048-046": ("BHBHPZ", "name"), "img-048-047": ("PKKLPZ", "name"), "img-048-048": ("PKPKPZ", "name"),
 "img-048-049": ("PRPRPZ", "name"), "img-048-050": ("BARAPZ", "name"), "img-048-051": ("CHDCHDPZ", "name"),
 "img-048-052": ("JEJEPZ", "name"), "img-048-053": ("KHDUPZ", "name"), "img-049-054": ("KHPKPZ", "name"),
 "img-049-055": ("KTAMPZ", "name"), "img-049-056": ("KTBAPZ", "name"), "img-049-057": ("MGMGPZ", "depth"),
 "img-049-058": ("NRLOPZ", "name"), "img-049-059": ("SKASPZ", "name"), "img-049-060": ("FABHPZ", "name"),
 "img-049-061": ("FAFAPZ", "name"), "img-049-062": ("MDMDPZ", "name"), "img-050-063": ("RBRBPZ", "name"),
 "img-050-064": ("GPGPPZ", "name"), "img-050-065": ("BSGOPZ", "name"), "img-050-066": ("SYZKPZ", "name"),
 "img-050-067": ("SUDEPZ", "name"), "img-050-068": ("HBHBPZ", "name")}

out = []; P = lambda *a: (out.append(" ".join(str(x) for x in a)), print(*a, flush=True))


def well_metrics(L, depth, aq=AQ):
    L = L.sort_values("top_m")
    if depth > L.bottom_m.max(): return None
    at = L[(L.top_m <= depth) & (L.bottom_m > depth)].lithology
    lith = at.iloc[0] if len(at) else L.iloc[-1].lithology
    above = L[L.top_m < depth].copy(); above["bot"] = above.bottom_m.clip(upper=depth)
    above["th"] = above.bot - above.top_m
    fine = above[above.lithology.isin(FINE)]
    # merge adjacent fine layers into units before counting aquitards
    units = []; cur = None
    for r in above.itertuples():
        f = r.lithology in FINE
        if f and cur is not None: cur += r.th
        elif f: cur = r.th
        elif cur is not None: units.append(cur); cur = None
    # a trailing fine unit is the one the well itself sits in: not between well and surface
    # a fine unit that the well sits in, or the topmost surface cap, is not an aquitard *between*
    surf = 0.0
    for r in above.itertuples():
        if r.lithology in FINE: surf += r.th
        else: break
    n_aq = sum(u >= aq for u in units) - (1 if surf >= aq else 0)
    return dict(lith_at_depth=lith, in_sand=lith not in FINE, fine_above_m=round(fine.th.sum(), 2),
                surface_cap_m=round(surf, 2), aquitards_above=max(n_aq, 0), first_aquifer=n_aq <= 0)


def cluster_boot(d, fn, B=2000, seed=42):
    rng = np.random.default_rng(seed); grp = [g for _, g in d.groupby("nest")]; v = []
    for _ in range(B):
        s = pd.concat([grp[i] for i in rng.integers(0, len(grp), len(grp))])
        try: v.append(fn(s))
        except Exception: pass
    return np.nanpercentile(v, [2.5, 97.5])


def main():
    L = pd.read_csv(f"{O}/r10_log_layers.csv")
    meta = pd.read_csv(f"{O}/r10_log_meta.csv").set_index("image")
    d = pd.read_csv(f"{R}/data/integrated/bangladesh_groundwater_arsenic_1807samples_CLEAN.csv")
    d["sid"] = d["Sample ID"].astype(str).str.strip()
    d["nest"] = d.sid.str.replace(r"-.*$", "", regex=True)
    g = pd.read_csv(f"{R}/revision_r2/inputs/wells_with_usgs_geology.csv")[["Sample ID", "Season", "Depth", "GLG"]]
    d = d.merge(g, on=["Sample ID", "Season", "Depth"], how="left")
    w = d.groupby("sid").agg(nest=("nest", "first"), depth=("Depth", "median"), As=("As", "median"),
                             PO4=("PO43-", "median"), Fe=("Fe2+", "median"), GLG=("GLG", "first")).reset_index()
    w["age"] = np.where(w.GLG.isin(HOL), "Holocene", np.where(w.GLG.isin(OLD), "Pleistocene", "other"))

    rows = []
    for im, (nest, how) in MATCH.items():
        if how == "ambiguous": continue
        LL = L[L.image == im]
        for r in w[w.nest == nest].itertuples():
            m = well_metrics(LL, r.depth)
            if m is None: continue
            m10 = well_metrics(LL, r.depth, aq=10.0)
            rows.append(dict(sid=r.sid, nest=nest, image=im, match=how, depth=r.depth, As=r.As, PO4=r.PO4, Fe=r.Fe,
                             age=r.age, GLG=r.GLG, log_depth=meta.loc[im, "total_depth_m"], **m,
                             first_aquifer_10=m10["first_aquifer"]))
    x = pd.DataFrame(rows)
    x.to_csv(f"{O}/r11_logged_wells.csv", index=False)
    nn = x.nest.nunique()
    P(f"logs 69; matched to nests {sum(v[1]!='ambiguous' for v in MATCH.values())} "
      f"(by name {sum(v[1]=='name' for v in MATCH.values())}, by depth {sum(v[1]=='depth' for v in MATCH.values())}, "
      f"ambiguous excluded {sum(v[1]=='ambiguous' for v in MATCH.values())})")
    P(f"logged wells {len(x)} in {nn} nests; median depth {x.depth.median():.0f} m; "
      f"wells deeper than their log dropped: {sum(len(w[w.nest==v[0]]) for v in MATCH.values() if v[1]!='ambiguous') - len(x)}")
    P(f"deposit age: {x.age.value_counts().to_dict()}")
    P(f"well depth falls in sand: {x.in_sand.mean()*100:.1f}% ({x.in_sand.sum()}/{len(x)}); "
      f"lithology at depth: {x.lith_at_depth.value_counts().to_dict()}")
    P(f"first aquifer (no >= {AQ:.0f} m aquitard above): {x.first_aquifer.sum()} wells; beneath aquitard: {(~x.first_aquifer).sum()}")
    ex = lambda s: (s.As > 10).mean() * 100

    P("\n[1] arsenic by aquifer position (all logged wells)")
    for k, s in x.groupby("first_aquifer"):
        P(f"  {'first aquifer' if k else 'beneath aquitard':17s} n={len(s):3d} median As {s.As.median():6.2f} "
          f"exceed {ex(s):5.1f}% median depth {s.depth.median():.0f} m")
    u = stats.mannwhitneyu(x[x.first_aquifer].As, x[~x.first_aquifer].As)
    P(f"  Mann-Whitney p = {u.pvalue:.2g}")

    P("\n[2] arsenic vs fine-grained thickness above the well")
    rho = stats.spearmanr(x.fine_above_m, x.As)
    def prho(s, a, b, z):
        C = np.c_[np.ones(len(s)), rankdata(s[z])]
        r_ = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
        return np.corrcoef(r_(rankdata(s[a])), r_(rankdata(s[b])))[0, 1]
    pr = prho(x, "fine_above_m", "As", "depth")
    ci = cluster_boot(x, lambda s: prho(s, "fine_above_m", "As", "depth"))
    P(f"  Spearman rho = {rho.statistic:+.3f} (p={rho.pvalue:.2g}); partial on depth = {pr:+.3f} [{ci[0]:+.2f}, {ci[1]:+.2f}]")
    sc = stats.spearmanr(x.surface_cap_m, x.As)
    P(f"  surface clay-cap thickness vs As: rho = {sc.statistic:+.3f} (p={sc.pvalue:.2g})")
    capmed = x.groupby("age").surface_cap_m.median()
    P(f"  median surface cap: Holocene {capmed['Holocene']:.1f} m, Pleistocene {capmed['Pleistocene']:.1f} m")
    capw = {g: stats.spearmanr(s.surface_cap_m, s.As) for g, s in x.groupby("age")}
    for g, r in capw.items(): P(f"    within {g:11s} rho = {r.statistic:+.3f} (p={r.pvalue:.2g})")
    ns = x.groupby("nest").agg(cap=("surface_cap_m", "first"), As=("As", "median"))
    capn = stats.spearmanr(ns.cap, ns.As)
    P(f"    by site (n={len(ns)}, median As) rho = {capn.statistic:+.3f} (p={capn.pvalue:.2g})")
    yc = x[x.age.isin(["Holocene", "Pleistocene"])].copy()
    yc["rAs"] = rankdata(yc.As) / len(yc); yc["hol"] = (yc.age == "Holocene").astype(int)
    yc["rdep"] = rankdata(yc.depth) / len(yc); yc["rcap"] = rankdata(yc.surface_cap_m) / len(yc)
    mc = smf.ols("rAs ~ hol + rdep + rcap", yc).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(yc.nest)[0]})
    P(f"    with deposit age and depth: cap {mc.params['rcap']:+.3f} (p={mc.pvalues['rcap']:.2g}), age {mc.params['hol']:+.3f} (p={mc.pvalues['hol']:.2g})")

    P("\n[3] Holocene vs Pleistocene among logged wells")
    tab = []
    for lab, s in [("all logged wells", x), ("first aquifer", x[x.first_aquifer]),
                   ("beneath aquitard", x[~x.first_aquifer]),
                   ("first aquifer (10 m rule)", x[x.first_aquifer_10]),
                   ("in sand at depth", x[x.in_sand]),
                   ("matched by name only", x[x.match == "name"])]:
        h = s[s.age == "Holocene"]; o = s[s.age == "Pleistocene"]
        if len(h) and len(o):
            cih = cluster_boot(h, ex, B=1000); cio = cluster_boot(o, ex, B=1000)
            P(f"  {lab:26s} HOL {ex(h):5.1f}% [{cih[0]:.1f}-{cih[1]:.1f}] n={len(h):3d} ({h.nest.nunique()} nests) | "
              f"OLD {ex(o):5.1f}% [{cio[0]:.1f}-{cio[1]:.1f}] n={len(o):3d} ({o.nest.nunique()} nests)")
            tab.append(dict(subset=lab, hol=ex(h), hol_lo=cih[0], hol_hi=cih[1], n_hol=len(h), nests_hol=h.nest.nunique(),
                            old=ex(o), old_lo=cio[0], old_hi=cio[1], n_old=len(o), nests_old=o.nest.nunique()))
    pd.DataFrame(tab).to_csv(f"{O}/r11_deposit_by_aquifer.csv", index=False)

    P("\n[4] rank regression, rank(As) ~ age + depth + aquifer position + fine thickness (Holocene/Pleistocene wells)")
    y = x[x.age.isin(["Holocene", "Pleistocene"])].copy()
    y["rAs"] = rankdata(y.As) / len(y); y["hol"] = (y.age == "Holocene").astype(int)
    y["rdep"] = rankdata(y.depth) / len(y); y["rfine"] = rankdata(y.fine_above_m) / len(y)
    y["fa"] = y.first_aquifer.astype(int)
    f = "rAs ~ hol + rdep + fa + rfine"
    m = smf.ols(f, y).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(y.nest)[0]})
    for k in ["hol", "rdep", "fa", "rfine"]:
        P(f"  {k:6s} {m.params[k]:+.3f}  SE {m.bse[k]:.3f}  p={m.pvalues[k]:.2g}")
    m0 = smf.ols("rAs ~ rdep + fa + rfine", y).fit()
    P(f"  R2 full {m.rsquared:.3f}; without deposit age {m0.rsquared:.3f}; unique share of age {m.rsquared-m0.rsquared:.3f}")
    m1 = smf.ols("rAs ~ hol + rdep", y).fit()
    P(f"  R2 age + depth only {m1.rsquared:.3f}; adding aquifer position and fine thickness +{m.rsquared-m1.rsquared:.3f}")

    P("\n[5] As-PO4 association among logged wells")
    for lab, s in [("first aquifer", x[x.first_aquifer]), ("beneath aquitard", x[~x.first_aquifer])]:
        r = stats.spearmanr(s.PO4, s.As, nan_policy="omit")
        P(f"  {lab:17s} n={len(s):3d} rho {r.statistic:+.3f} (p={r.pvalue:.2g})")

    open(f"{O}/r11_summary.txt", "w").write("\n".join(out))

    # Supplementary table (LaTeX)
    v = pd.read_csv(f"{O}/r10_validation.csv"); vb = v[v["kind"].isna()]; vt = v[v["kind"] == "total depth"]
    T = pd.read_csv(f"{O}/r11_deposit_by_aquifer.csv")
    lab = {"all logged wells": "All logged wells", "first aquifer": "First aquifer (no aquitard $\\geq$5~m above)",
           "beneath aquitard": "Beneath an aquitard $\\geq$5~m", "first aquifer (10 m rule)": "First aquifer, 10~m aquitard rule",
           "in sand at depth": "Well depth in sand", "matched by name only": "Logs matched by site name only"}
    rows = [f"{lab[r.subset]} & {r.hol:.1f} [{r.hol_lo:.1f}--{r.hol_hi:.1f}] & {r.n_hol} ({r.nests_hol}) & "
            f"{r.old:.1f} [{r.old_lo:.1f}--{r.old_hi:.1f}] & {r.n_old} ({r.nests_old}) \\\\" for r in T.itertuples()]
    fa = x[x.first_aquifer]; bq = x[~x.first_aquifer]
    cap = (f"Mapped deposit type and subsurface position for {len(x)} nested monitoring wells in {nn} clustered sites "
           f"matched to the digitised borehole logs of \\citet[Section 3.5]{{BWDB2023}}. Exceedance is the percentage of wells "
           f"above the WHO guideline, with 95\\% intervals from a bootstrap over sites; n is wells (sites). "
           f"Against the aquifer boundaries stated in the text of the source report for seven sites, the digitised logs agree "
           f"to a median of {vb.error_m.abs().median():.1f}~m ({(vb.error_m.abs()<=2.2).sum()} of {len(vb)} boundaries within 2.2~m) "
           f"and total depths to within {vt.error_m.abs().max():.1f}~m. "
           f"The thickness of the fine-grained cap at the surface correlates with arsenic across all logged wells "
           f"($\\rho = {sc.statistic:+.2f}$, $p = {sc.pvalue:.3f}$), but Holocene sites carry thicker caps "
           f"(median {capmed['Holocene']:.1f}~m against {capmed['Pleistocene']:.1f}~m on Pleistocene sites) and the association "
           f"does not hold within either deposit type (Holocene $\\rho = {capw['Holocene'].statistic:+.2f}$, "
           f"Pleistocene $\\rho = {capw['Pleistocene'].statistic:+.2f}$), by site ($\\rho = {capn.statistic:+.2f}$, "
           f"$p = {capn.pvalue:.2f}$, {len(ns)} sites) or once deposit type and depth enter the rank regression ($p = {mc.pvalues['rcap']:.2f}$).")
    tex = ("\\begin{table}[H]\n\\centering\n\\caption{" + cap + "}\n\\label{tab:s_logs}\n\\small\n"
           "\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{\\textbf{Holocene}} & \\multicolumn{2}{c}{\\textbf{Pleistocene}} \\\\\n"
           "\\textbf{Subset} & \\textbf{\\% $>$WHO} & \\textbf{n} & \\textbf{\\% $>$WHO} & \\textbf{n} \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(f"{O}/supp_table_logs_r2.tex", "w").write(tex)


if __name__ == "__main__":
    main()
