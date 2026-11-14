"""Check the digitised logs against the aquifer boundaries stated in the text of
BWDB (2023) Sections 3.5.1-3.5.7 for the seven example sites. Each stated
boundary is compared with the nearest sand/fine boundary in the digitised log,
ignoring fine units thinner than 3 m (the report text describes main units)."""
import os, numpy as np, pandas as pd
from pathlib import Path
R = os.environ.get("PAPER1_ROOT", str(Path(__file__).resolve().parents[2]))
O = f"{R}/revision_r2/output"
FINE = {"clay", "silt/silty clay"}
TEXT = {  # image: stated boundaries (m) between aquifer and aquitard, report text
 "img-040-000": [6, 54, 70, 95, 108, 122, 140, 160],          # Sylhet
 "img-040-002": [42.68, 176.83, 246.95],                      # Hathazari (304.88 = base)
 "img-041-003": [6.10, 134.15, 189.02],                       # Green Road (292.68 = base)
 "img-042-008": [60.98, 67.07],                               # Chandpur
 "img-043-011": [9],                                          # Gaibandha
 "img-044-019": [6.10, 167.68, 195.12],                       # Banani
 "img-047-042": [6.1, 42.68, 48.78, 152.44, 158.54, 195.12, 204.27, 298.78]}  # Sandwip
BASE = {"img-040-002": 304.88, "img-041-003": 292.68, "img-043-011": 75, "img-044-019": 292.68,
        "img-047-042": 301.83, "img-040-000": 304.88, "img-042-008": 304.88}
L = pd.read_csv(f"{O}/r10_log_layers.csv"); meta = pd.read_csv(f"{O}/r10_log_meta.csv").set_index("image")
rows = []
for im, stated in TEXT.items():
    d = L[L.image == im]; units = []
    for r in d.itertuples():
        k = r.lithology in FINE
        if units and units[-1][0] == k: units[-1][2] = r.bottom_m
        else: units.append([k, r.top_m, r.bottom_m])
    # absorb fine units < 3 m into the surrounding sand
    m = []
    for u in units:
        if u[0] and u[2] - u[1] < 3 and m and not m[-1][0]: m[-1][2] = u[2]; continue
        if m and m[-1][0] == u[0]: m[-1][2] = u[2]
        else: m.append(list(u))
    bnd = np.array([u[1] for u in m[1:]])
    for s in stated:
        e = bnd[np.abs(bnd - s).argmin()] - s
        rows.append(dict(image=im, stated_m=s, digitised_m=round(s + e, 2), error_m=round(e, 2)))
    rows.append(dict(image=im, stated_m=BASE[im], digitised_m=meta.loc[im, "total_depth_m"],
                     error_m=round(meta.loc[im, "total_depth_m"] - BASE[im], 2), kind="total depth"))
v = pd.DataFrame(rows); v.to_csv(f"{O}/r10_validation.csv", index=False)
b = v[v.get("kind").isna()] if "kind" in v else v
t = v[v.get("kind") == "total depth"]
print(f"boundaries {len(b)}: median |error| {b.error_m.abs().median():.2f} m, 90th pct {b.error_m.abs().quantile(.9):.2f} m, max {b.error_m.abs().max():.2f} m")
print(f"total depths {len(t)}: median |error| {t.error_m.abs().median():.2f} m, max {t.error_m.abs().max():.2f} m")
print(v.to_string())
