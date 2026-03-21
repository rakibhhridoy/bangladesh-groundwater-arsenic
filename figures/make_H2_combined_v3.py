#!/usr/bin/env python3
"""Generate H2 combined 4-panel figure (publication quality)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

# ── Load data ──
df = pd.read_csv("/Users/rakibhhridoy/AsGW/GroundWater/hypo1/output/data_phase1_integrated.csv")

# ── Depth zone assignment ──
bins = [0, 30, 80, 150, np.inf]
labels = ["Shallow", "Intermediate", "Medium_Deep", "Deep"]
df["Depth_Zone"] = pd.cut(df["Depth"], bins=bins, labels=labels, right=True, include_lowest=True)

# ── Physiographic zone assignment (first match wins) ──
zones_bb = [
    ("Barind_Tract",            23.8, 25.6, 88.0, 89.2),
    ("Northern_Terrace",        25.2, 26.7, 88.0, 89.8),
    ("Brahmaputra_Floodplain",  23.5, 25.5, 89.5, 91.2),
    ("Ganges_Floodplain",       23.0, 24.5, 88.5, 90.2),
    ("GBM_Delta",               21.5, 23.5, 89.0, 90.5),
    ("Meghna_Floodplain",       22.5, 24.5, 90.5, 91.5),
    ("Eastern_Hills",           21.5, 24.8, 91.5, 92.7),
]

def assign_phys(row):
    lat, lon = row["Latitude"], row["Longitude"]
    for name, lat_lo, lat_hi, lon_lo, lon_hi in zones_bb:
        if lat_lo <= lat <= lat_hi and lon_lo <= lon <= lon_hi:
            return name
    return "Unclassified"

df["Phys_Zone"] = df.apply(assign_phys, axis=1)

# ── Partial correlation helper ──
def partial_corr_spearman(df_sub, x, y, covar):
    """Spearman partial correlation of x-y controlling for covar."""
    mask = df_sub[[x, y, covar]].dropna()
    if len(mask) < 10:
        return np.nan, 1.0
    rx = stats.spearmanr(mask[x], mask[covar])[0]
    ry = stats.spearmanr(mask[y], mask[covar])[0]
    rxy = stats.spearmanr(mask[x], mask[y])[0]
    denom = np.sqrt((1 - rx**2) * (1 - ry**2))
    if denom == 0:
        return np.nan, 1.0
    pr = (rxy - rx * ry) / denom
    n = len(mask)
    t_stat = pr * np.sqrt((n - 3) / (1 - pr**2 + 1e-12))
    p_val = 2 * stats.t.sf(np.abs(t_stat), df=n - 3)
    return pr, p_val

# ── Styling constants ──
zone_order = ["Shallow", "Intermediate", "Medium_Deep", "Deep"]
zone_colors = ["#a6cee3", "#5b9bd5", "#2b6cb0", "#1a3a5c"]  # blue gradient
zone_color_map = dict(zip(zone_order, zone_colors))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.titlesize": 14,
})

fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
fig.patch.set_facecolor("white")

# ═══════════════════════════════════════════
# Panel A: Arsenic by Depth Zone (Boxplot + violin)
# ═══════════════════════════════════════════
ax = axes[0, 0]
data_by_zone = [df.loc[df["Depth_Zone"] == z, "As"].dropna().values for z in zone_order]

parts = ax.violinplot(data_by_zone, positions=range(4), showextrema=False, widths=0.7)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(zone_colors[i])
    pc.set_alpha(0.3)

bp = ax.boxplot(data_by_zone, positions=range(4), widths=0.35, patch_artist=True,
                showfliers=False, medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2))
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(zone_colors[i])
    patch.set_edgecolor("black")
    patch.set_alpha(0.8)

ax.set_yscale("log")
ax.axhline(10, color="red", ls="--", lw=1.5, alpha=0.8, label="WHO 10 \u00b5g/L")
ax.set_xticks(range(4))
ax.set_xticklabels(["Shallow\n(0\u201330 m)", "Intermediate\n(30\u201380 m)",
                     "Medium-Deep\n(80\u2013150 m)", "Deep\n(150+ m)"])
ax.set_ylabel("Arsenic (\u00b5g/L)")
ax.set_title("Arsenic Distribution by Depth Zone")
ax.legend(loc="upper right", fontsize=10)

# Annotate % > WHO
for i, vals in enumerate(data_by_zone):
    pct = 100 * np.sum(vals > 10) / len(vals) if len(vals) > 0 else 0
    ymax = np.percentile(vals, 95) if len(vals) > 0 else 100
    ax.text(i, ymax * 1.8, f"{pct:.1f}%\n>WHO", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=zone_colors[i])

ax.text(-0.08, 1.05, "(A)", transform=ax.transAxes, fontsize=16, fontweight="bold")

# ═══════════════════════════════════════════
# Panel B: Mechanism Shift Heatmap
# ═══════════════════════════════════════════
ax = axes[0, 1]

heatmap_data = np.zeros((2, 4))
pval_data = np.zeros((2, 4))
for j, z in enumerate(zone_order):
    sub = df[df["Depth_Zone"] == z]
    pr_po4, p_po4 = partial_corr_spearman(sub, "PO43-", "As", "Fe2+")
    pr_fe, p_fe = partial_corr_spearman(sub, "Fe2+", "As", "PO43-")
    heatmap_data[0, j] = pr_po4
    heatmap_data[1, j] = pr_fe
    pval_data[0, j] = p_po4
    pval_data[1, j] = p_fe

vmax = max(0.4, np.nanmax(np.abs(heatmap_data)))
im = ax.imshow(heatmap_data, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

ax.set_xticks(range(4))
ax.set_xticklabels(["Shallow", "Intermediate", "Med-Deep", "Deep"], fontsize=11)
ax.set_yticks([0, 1])
ax.set_yticklabels(["PO4 -> As\n(ctrl Fe2+)", "Fe2+ -> As\n(ctrl PO4)"], fontsize=11)

# Annotate cells
for i in range(2):
    for j in range(4):
        val = heatmap_data[i, j]
        p = pval_data[i, j]
        stars = ""
        if p < 0.001:
            stars = "***"
        elif p < 0.01:
            stars = "**"
        elif p < 0.05:
            stars = "*"
        txt_color = "white" if abs(val) > 0.25 else "black"
        ax.text(j, i, f"{val:.3f}{stars}", ha="center", va="center",
                fontsize=12, fontweight="bold", color=txt_color)

cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Partial Spearman \u03c1", fontsize=11)

ax.set_title("Mechanism Shift: Partial Correlations with As")

# Draw crossover annotation — between Intermediate and Med-Deep columns
ax.annotate("Crossover\n~80 m", xy=(1.5, 1.65), xytext=(1.5, 1.65),
            ha="center", va="top", fontsize=10,
            fontweight="bold", color="red", annotation_clip=False)
ax.annotate("", xy=(1.1, 1.45), xytext=(1.9, 1.45),
            arrowprops=dict(arrowstyle="<->", color="red", lw=2),
            annotation_clip=False)

ax.text(-0.08, 1.05, "(B)", transform=ax.transAxes, fontsize=16, fontweight="bold")

# ═══════════════════════════════════════════
# Panel C: PO4-As Scatter by Depth Zone
# ═══════════════════════════════════════════
ax = axes[1, 0]

for i, z in enumerate(zone_order):
    sub = df[df["Depth_Zone"] == z].dropna(subset=["PO43-", "As"])
    sub = sub[(sub["PO43-"] > 0) & (sub["As"] > 0)]
    ax.scatter(sub["PO43-"], sub["As"], c=zone_colors[i], alpha=0.4, s=18,
               edgecolors="none", label=None)
    # regression line in log space
    if len(sub) > 10:
        logx = np.log10(sub["PO43-"].values)
        logy = np.log10(sub["As"].values)
        slope, intercept, _, _, _ = stats.linregress(logx, logy)
        xfit = np.linspace(logx.min(), logx.max(), 100)
        yfit = slope * xfit + intercept
        rho, pval = spearmanr(sub["PO43-"], sub["As"])
        star = ""
        if pval < 0.001: star = "***"
        elif pval < 0.01: star = "**"
        elif pval < 0.05: star = "*"
        ax.plot(10**xfit, 10**yfit, color=zone_colors[i], lw=2.2,
                label=f"{z}: rho={rho:.3f}{star}")

ax.set_xscale("log")
ax.set_yscale("log")
ax.axhline(10, color="red", ls="--", lw=1, alpha=0.6)
ax.set_xlabel("PO4 (mg/L)")
ax.set_ylabel("Arsenic (ug/L)")
ax.set_title("PO4-As Association by Depth Zone")
ax.legend(fontsize=9.5, loc="lower left", framealpha=0.9, edgecolor="gray")

ax.text(-0.08, 1.05, "(C)", transform=ax.transAxes, fontsize=16, fontweight="bold")

# ═══════════════════════════════════════════
# Panel D: Arsenic by Physiographic Zone (Horizontal Bar)
# ═══════════════════════════════════════════
ax = axes[1, 1]

phys_stats = []
for z in df["Phys_Zone"].unique():
    sub = df[df["Phys_Zone"] == z]["As"].dropna()
    if len(sub) >= 5:
        phys_stats.append({
            "zone": z.replace("_", " "),
            "mean": sub.mean(),
            "se": sub.sem(),
            "pct_who": 100 * (sub > 10).sum() / len(sub),
            "n": len(sub),
        })

phys_df = pd.DataFrame(phys_stats).sort_values("mean")

# Color by intensity
norm = plt.Normalize(phys_df["mean"].min(), phys_df["mean"].max())
cmap = plt.cm.YlOrRd
bar_colors = [cmap(norm(v)) for v in phys_df["mean"]]

y_pos = range(len(phys_df))
bars = ax.barh(y_pos, phys_df["mean"], xerr=phys_df["se"],
               color=bar_colors, edgecolor="gray", linewidth=0.5,
               capsize=3, height=0.65)

ax.axvline(10, color="red", ls="--", lw=1.5, alpha=0.8, label="WHO 10 \u00b5g/L")
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{z}\n(n={n})" for z, n in zip(phys_df["zone"], phys_df["n"])],
                   fontsize=10)
ax.set_xlabel("Mean Arsenic (\u00b5g/L)")
ax.set_title("Arsenic by Physiographic Zone")

# Annotate % > WHO
for i, (_, row) in enumerate(phys_df.iterrows()):
    ax.text(row["mean"] + row["se"] + 1.5, i, f"{row['pct_who']:.1f}% >WHO",
            va="center", fontsize=9.5, fontweight="bold",
            color="darkred" if row["pct_who"] > 30 else "black")

ax.legend(loc="lower right", fontsize=10)
ax.text(-0.08, 1.05, "(D)", transform=ax.transAxes, fontsize=16, fontweight="bold")

# ── Final layout ──
plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.5)

out_path = "/Users/rakibhhridoy/AsGW/GroundWater/hypo2/output/figures/H2_combined_figure.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out_path}")
print(f"Size: {pd.io.common.file_exists(out_path)}")

# Print summary stats for verification
print("\n── Panel A: Depth zone counts & % >WHO ──")
for z in zone_order:
    sub = df[df["Depth_Zone"] == z]["As"].dropna()
    print(f"  {z}: n={len(sub)}, median={sub.median():.1f}, mean={sub.mean():.1f}, %>WHO={100*(sub>10).sum()/len(sub):.1f}%")

print("\n── Panel B: Partial correlations ──")
for j, z in enumerate(zone_order):
    print(f"  {z}: PO4 partial rho={heatmap_data[0,j]:.3f} (p={pval_data[0,j]:.4f}), Fe partial rho={heatmap_data[1,j]:.3f} (p={pval_data[1,j]:.4f})")

print("\n── Panel D: Physiographic zones ──")
for _, row in phys_df.iterrows():
    print(f"  {row['zone']}: mean={row['mean']:.1f}, %>WHO={row['pct_who']:.1f}%, n={row['n']}")
