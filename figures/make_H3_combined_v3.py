#!/usr/bin/env python3
"""Generate H3 combined 4-panel figure for the manuscript."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

# ---------- Global style ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
})

# Colors
C_DESORP = "#E06060"   # coral-red
C_REDUCT = "#4682B4"   # steel blue
C_DILUT  = "#888888"   # gray
C_DRY    = "#E8825C"   # coral/orange
C_WET    = "#4682B4"   # steel blue

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ================================================================
# PANEL A: SEM Indirect Effects (log-scale absolute values)
# ================================================================
ax = axes[0, 0]

zones = ["ALL", "Shallow", "Intermediate", "Medium_Deep", "Deep"]
zone_labels = ["Overall", "Shallow", "Intermediate", "Med-Deep", "Deep"]

desorption = [-0.0844, -0.0751, -0.0813, -0.0751, -0.0714]
reductive  = [-0.000592, -0.002427, -0.000072, -0.000376, -0.000515]
dilution   = [0.000029, -0.010162, -0.000668, -0.000575, 0.007460]

abs_desorp = [abs(v) for v in desorption]
abs_reduct = [abs(v) for v in reductive]
abs_dilut  = [abs(v) for v in dilution]

x = np.arange(len(zones))
w = 0.25

bars1 = ax.bar(x - w, abs_desorp, w, color=C_DESORP, edgecolor="white", linewidth=0.5, label="Desorption", zorder=3)
bars2 = ax.bar(x,     abs_reduct, w, color=C_REDUCT, edgecolor="white", linewidth=0.5, label="Reductive", zorder=3)
bars3 = ax.bar(x + w, abs_dilut,  w, color=C_DILUT,  edgecolor="white", linewidth=0.5, label="Dilution", zorder=3)

ax.set_yscale("log")
ax.set_ylim(1e-5, 0.2)
ax.set_xticks(x)
ax.set_xticklabels(zone_labels, fontsize=11)
ax.set_ylabel("|Indirect Effect|  (log scale)", fontsize=13)
ax.set_title("SEM Indirect Effects by Pathway", fontsize=14, fontweight="bold", pad=10)
ax.legend(fontsize=10, frameon=True, edgecolor="#cccccc", loc="upper right")
ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)

# Annotate ratio on Overall bar
ax.annotate("100×", xy=(0 - w, abs_desorp[0]), xytext=(0.3, 0.12),
            fontsize=11, fontweight="bold", color=C_DESORP,
            arrowprops=dict(arrowstyle="->", color=C_DESORP, lw=1.5))

ax.text(-0.08, 1.05, "(A)", transform=ax.transAxes, fontsize=16, fontweight="bold")

# ================================================================
# PANEL B: 56-Cell Risk Matrix Heatmap
# ================================================================
ax = axes[0, 1]

risk_df = pd.read_csv("/Users/rakibhhridoy/AsGW/GroundWater/hypo3/output/tables/H3_T24_56cell_risk_matrix.csv")

seasons = ["Dry", "Wet"]
depth_zones = ["Shallow", "Intermediate", "Medium_Deep", "Deep"]
phys_zones = ["Barind_Tract", "Northern_Terrace", "Brahmaputra_Floodplain",
              "Ganges_Floodplain", "GBM_Delta", "Meghna_Floodplain", "Eastern_Hills"]
phys_labels = ["Barind", "N.Terrace", "Brahmaputra", "Ganges", "GBM Delta", "Meghna", "E.Hills"]

row_labels = []
for s in seasons:
    for d in depth_zones:
        short_d = d.replace("Medium_Deep", "Med-Deep")
        row_labels.append(f"{s}-{short_d}")

matrix = np.full((len(seasons)*len(depth_zones), len(phys_zones)), np.nan)

for i_s, season in enumerate(seasons):
    for i_d, dz in enumerate(depth_zones):
        row_idx = i_s * len(depth_zones) + i_d
        for i_p, pz in enumerate(phys_zones):
            mask = (risk_df["Season"] == season) & (risk_df["Phys_Zone"] == pz) & (risk_df["Depth_Zone"] == dz)
            subset = risk_df.loc[mask, "Exceedance_pct"]
            if len(subset) > 0 and not pd.isna(subset.values[0]):
                matrix[row_idx, i_p] = subset.values[0]

# Custom colormap: green -> yellow -> red
cmap = LinearSegmentedColormap.from_list("risk", ["#2ecc71", "#f1c40f", "#e74c3c"], N=256)
cmap.set_bad(color="#eeeeee")

im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=70)

# Annotate cells
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = matrix[i, j]
        if np.isnan(val):
            ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#999999")
        else:
            color = "white" if val > 45 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=8, fontweight="bold", color=color)

ax.set_xticks(range(len(phys_labels)))
ax.set_xticklabels(phys_labels, fontsize=10, rotation=35, ha="right")
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=10)

# Horizontal line separating Dry and Wet
ax.axhline(y=3.5, color="black", linewidth=1.5)

cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("WHO Exceedance (%)", fontsize=11)

ax.set_title("Seasonal Risk Matrix (WHO >10 µg/L)", fontsize=14, fontweight="bold", pad=10)
ax.text(-0.08, 1.05, "(B)", transform=ax.transAxes, fontsize=16, fontweight="bold")

# ================================================================
# PANEL C: Seasonal Arsenic by Depth (from raw data)
# ================================================================
ax = axes[1, 0]

df = pd.read_csv("/Users/rakibhhridoy/AsGW/GroundWater/hypo1/output/data_phase1_integrated.csv")

# Assign depth zones
def assign_depth_zone(d):
    if d <= 30:
        return "Shallow"
    elif d <= 80:
        return "Intermediate"
    elif d <= 150:
        return "Medium_Deep"
    else:
        return "Deep"

df["Depth_Zone"] = df["Depth"].apply(assign_depth_zone)

dz_order = ["Shallow", "Intermediate", "Medium_Deep", "Deep"]
dz_labels = ["Shallow\n(0-30 m)", "Intermediate\n(30-80 m)", "Med-Deep\n(80-150 m)", "Deep\n(>150 m)"]

dry_medians = []
wet_medians = []
for dz in dz_order:
    dry_medians.append(df[(df["Depth_Zone"] == dz) & (df["Season"] == "Dry")]["As"].median())
    wet_medians.append(df[(df["Depth_Zone"] == dz) & (df["Season"] == "Wet")]["As"].median())

x = np.arange(len(dz_order))
w = 0.35

bars_dry = ax.bar(x - w/2, dry_medians, w, color=C_DRY, edgecolor="white", linewidth=0.5, label="Dry Season", zorder=3)
bars_wet = ax.bar(x + w/2, wet_medians, w, color=C_WET, edgecolor="white", linewidth=0.5, label="Wet Season", zorder=3)

# WHO line
ax.axhline(y=10, color="#cc0000", linewidth=1.2, linestyle="--", zorder=2, label="WHO 10 µg/L")

# Significance stars
sig_stars = ["***", "***", "", ""]
for i, star in enumerate(sig_stars):
    if star:
        max_val = max(dry_medians[i], wet_medians[i])
        ax.text(i, max_val + 0.8, star, ha="center", va="bottom", fontsize=13, fontweight="bold", color="#333333")

ax.set_xticks(x)
ax.set_xticklabels(dz_labels, fontsize=11)
ax.set_ylabel("Median Arsenic (µg/L)", fontsize=13)
ax.set_title("Seasonal Arsenic Variation by Depth Zone", fontsize=14, fontweight="bold", pad=10)
ax.legend(fontsize=10, frameon=True, edgecolor="#cccccc", loc="upper right")
ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
ax.text(-0.08, 1.05, "(C)", transform=ax.transAxes, fontsize=16, fontweight="bold")

# ================================================================
# PANEL D: Shannon Entropy by Depth Zone and Season
# ================================================================
ax = axes[1, 1]

dz_labels_d = ["Shallow", "Intermediate", "Med-Deep", "Deep"]
dry_H = [1.328, 1.721, 1.584, 1.777]
wet_H = [1.801, 2.218, 2.051, 1.123]

x = np.arange(len(dz_labels_d))
w = 0.35

bars_d = ax.bar(x - w/2, dry_H, w, color=C_DRY, edgecolor="white", linewidth=0.5, label="Dry Season", zorder=3)
bars_w = ax.bar(x + w/2, wet_H, w, color=C_WET, edgecolor="white", linewidth=0.5, label="Wet Season", zorder=3)

# Annotate delta-H
for i in range(len(dz_labels_d)):
    dh = wet_H[i] - dry_H[i]
    max_val = max(dry_H[i], wet_H[i])
    sign = "+" if dh > 0 else ""
    color = "#2e7d32" if dh > 0 else "#c62828"
    ax.text(i, max_val + 0.06, f"ΔH={sign}{dh:.2f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=color)

ax.set_xticks(x)
ax.set_xticklabels(dz_labels_d, fontsize=11)
ax.set_ylabel("Shannon Entropy (H)", fontsize=13)
ax.set_ylim(0, 2.7)
ax.set_title("Mechanism Diversity: Shannon Entropy by Depth", fontsize=14, fontweight="bold", pad=10)
ax.legend(fontsize=10, frameon=True, edgecolor="#cccccc", loc="upper left")
ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)

# Annotations for key message
ax.annotate("Diversity ↑", xy=(0 + w/2, wet_H[0]), xytext=(0.5, 2.35),
            fontsize=10, color="#2e7d32", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.2))
ax.annotate("Homogenisation ↓", xy=(3 + w/2, wet_H[3]), xytext=(2.5, 1.55),
            fontsize=10, color="#c62828", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.2))

ax.text(-0.08, 1.05, "(D)", transform=ax.transAxes, fontsize=16, fontweight="bold")

# ================================================================
# Save
# ================================================================
plt.tight_layout(w_pad=3.0, h_pad=3.0)
out_path = "/Users/rakibhhridoy/AsGW/GroundWater/hypo3/output/figures/H3_combined_figure.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out_path}")
print(f"Size: {__import__('os').path.getsize(out_path) / 1024:.0f} KB")
