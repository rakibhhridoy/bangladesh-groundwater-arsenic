"""
make_H3_combined.py
====================
Generates H3_combined_figure.png — a 6-panel publication figure for
Hypothesis 3 (Seasonal Hydro-Climatic Forcing).

Panels
------
A  SEM pathway schematic (conceptual, horizontal chain + desorption arc)
B  SEM indirect effects by depth zone (desorption vs reductive)
C  56-cell risk matrix heatmap (7 zones × 4 depths × 2 seasons, WHO exceedance %)
D  Seasonal arsenic shift bars (Dry vs Wet median, by depth zone)
E  Shannon entropy (Dry vs Wet) by depth zone
F  Satellite ensemble null result (R² bar chart)

Design rules
------------
- figsize=(22, 15), constrained_layout=True
- axes labels 13 pt, tick labels 11 pt, legend 10 pt, panel labels 14 pt bold, titles 12 pt bold
- DPI 300
- No tight_layout; use constrained_layout throughout
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE  = "/Users/rakibhhridoy/AsGW/GroundWater"
T_H3  = os.path.join(BASE, "hypo3/output/tables")
T_H2  = os.path.join(BASE, "hypo2/output/tables")
OUT   = os.path.join(BASE, "hypo3/output/figures/H3_combined_figure.png")

# ── Global font sizes ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.labelsize":     13,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    10,
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

PANEL_KW = dict(fontsize=14, fontweight="bold", ha="left", va="top",
                transform=None)   # will be set per axis

# ── Load tables ────────────────────────────────────────────────────────────────
t03  = pd.read_csv(os.path.join(T_H3, "H3_T03_shannon_entropy_by_zone.csv"))
t16  = pd.read_csv(os.path.join(T_H3, "H3_T16_ensemble_performance.csv"))
t24  = pd.read_csv(os.path.join(T_H3, "H3_T24_56cell_risk_matrix.csv"))
t29  = pd.read_csv(os.path.join(T_H3, "H3_T29_sem_path_coefficients.csv"))
t_h2 = pd.read_csv(os.path.join(T_H2, "H2_T16_seasonal_effect_by_zone.csv"))

# ── Colour palette ─────────────────────────────────────────────────────────────
DRY_C  = "#E07B54"    # warm orange
WET_C  = "#4A90D9"    # cool blue
RED_C  = "#D94F3D"    # reductive chain
DES_C  = "#7B2D8B"    # desorption (purple)
DIL_C  = "#5B9E6B"    # dilution (green)
BOX_FC = "#e8f4f8"    # box facecolor
BOX_EC = "#2166ac"    # box edgecolor

# ═════════════════════════════════════════════════════════════════════════════
# Figure skeleton — 2 rows × 3 cols with extra vertical space
# ═════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 15), constrained_layout=True)
gs  = GridSpec(2, 3, figure=fig)

ax_A = fig.add_subplot(gs[0, 0])
ax_B = fig.add_subplot(gs[0, 1])
ax_C = fig.add_subplot(gs[0, 2])
ax_D = fig.add_subplot(gs[1, 0])
ax_E = fig.add_subplot(gs[1, 1])
ax_F = fig.add_subplot(gs[1, 2])

# ═════════════════════════════════════════════════════════════════════════════
# Panel A — SEM pathway schematic (horizontal chain + desorption arc)
# ═════════════════════════════════════════════════════════════════════════════
ax_A.set_xlim(-0.3, 10.3)
ax_A.set_ylim(0, 5)
ax_A.axis("off")

# Box layout: x positions at 1, 3.5, 6, 8.5; y = 2.5 for main chain
BW, BH = 1.4, 0.6
CHAIN_Y = 2.5
CHAIN_X = [1.0, 3.5, 6.0, 8.5]
BOX_LABELS = ["DOC/OC", "ORP", "Fe²⁺", "As"]

def draw_fancy_box(ax, cx, cy, label, bw=BW, bh=BH,
                   fc=BOX_FC, ec=BOX_EC, fontsize=11):
    patch = FancyBboxPatch(
        (cx - bw/2, cy - bh/2), bw, bh,
        boxstyle="round,pad=0.3",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc,
        zorder=4,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", zorder=5, color="#1a1a2e")

for cx, lbl in zip(CHAIN_X, BOX_LABELS):
    # Highlight As box differently
    fc = "#fde8e8" if lbl == "As" else BOX_FC
    ec = "#b2182b" if lbl == "As" else BOX_EC
    draw_fancy_box(ax_A, cx, CHAIN_Y, lbl, fc=fc, ec=ec)

# Arrows between chain boxes — reductive pathway (red)
CHAIN_BETAS = ["β=−0.18", "β=−0.32", "β=+0.19"]
for i in range(len(CHAIN_X) - 1):
    x0 = CHAIN_X[i]   + BW/2
    x1 = CHAIN_X[i+1] - BW/2
    y0 = y1 = CHAIN_Y
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="->",
        color=RED_C,
        linewidth=1.5,
        mutation_scale=18,
        zorder=3,
    )
    ax_A.add_patch(arr)
    mx = (x0 + x1) / 2
    ax_A.text(mx, y0 + 0.22, CHAIN_BETAS[i],
              ha="center", va="bottom", fontsize=10, color=RED_C, fontweight="bold",
              path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# Desorption arc: DOC/OC → As (dashed purple, above the chain)
# Use a ConnectionPatch-style annotation with arc
ax_A.annotate(
    "",
    xy=(CHAIN_X[3] - BW/2, CHAIN_Y + BH/2 - 0.05),
    xytext=(CHAIN_X[0] + BW/2, CHAIN_Y + BH/2 - 0.05),
    arrowprops=dict(
        arrowstyle="->",
        color=DES_C,
        lw=2.0,
        linestyle="dashed",
        connectionstyle="arc3,rad=-0.45",
    ),
    zorder=6,
)
# Desorption label above the arc
ax_A.text(4.75, 4.15,
          "Desorption\n(indirect = −0.084)",
          ha="center", va="center",
          fontsize=9, color=DES_C, fontweight="bold",
          path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

# Legend
leg_handles = [
    mpatches.Patch(color=RED_C,       label="Reductive chain (DOC→ORP→Fe→As)"),
    mpatches.Patch(color=DES_C,       label="Desorption path (DOC→As, dashed)"),
]
ax_A.legend(handles=leg_handles, loc="lower center", fontsize=9,
            framealpha=0.85, edgecolor="#cccccc", borderpad=0.6,
            bbox_to_anchor=(0.5, 0.02))

ax_A.text(-0.04, 1.03, "A", transform=ax_A.transAxes,
          fontsize=14, fontweight="bold", ha="left", va="top")
ax_A.set_title("Seasonal Mobilisation Pathways (SEM)", pad=8)

# ═════════════════════════════════════════════════════════════════════════════
# Panel B — SEM indirect effects by depth zone
# ═════════════════════════════════════════════════════════════════════════════
depth_zones = ["ALL", "Shallow", "Intermediate", "Medium_Deep", "Deep"]
short_dz    = ["All", "Shallow", "Interm.", "Med-Deep", "Deep"]

def compute_indirect(df, zone, pathway):
    sub = df[(df["Zone"] == zone) & (df["Pathway"] == pathway)]
    betas = sub["Beta"].values
    if len(betas) == 0:
        return np.nan
    return float(np.prod(betas))

desorption_vals = []
reductive_vals  = []
for z in depth_zones:
    dv = compute_indirect(t29, z, "Desorption")
    rv = compute_indirect(t29, z, "Reductive")
    # Fallback to hard-coded global if missing
    if np.isnan(dv):
        dv = -0.084
    if np.isnan(rv):
        rv = -0.0006
    desorption_vals.append(dv)
    reductive_vals.append(rv)

x_b = np.arange(len(depth_zones))
w_b = 0.32

bars_d = ax_B.bar(x_b - w_b/2, desorption_vals, w_b,
                  label="Desorption", color=DES_C,
                  edgecolor="white", linewidth=0.8, alpha=0.90)
bars_r = ax_B.bar(x_b + w_b/2, reductive_vals, w_b,
                  label="Reductive", color=RED_C,
                  edgecolor="white", linewidth=0.8, alpha=0.90)

ax_B.axhline(0, color="black", linewidth=0.8, linestyle="--")

# Annotate bar values
for i, (dv, rv) in enumerate(zip(desorption_vals, reductive_vals)):
    offset = 0.001
    va_d = "bottom" if dv >= 0 else "top"
    va_r = "bottom" if rv >= 0 else "top"
    ax_B.text(i - w_b/2, dv + (offset if dv >= 0 else -offset),
              f"{dv:.3f}", ha="center", va=va_d, fontsize=8.5, color="#333333")
    ax_B.text(i + w_b/2, rv + (offset if rv >= 0 else -offset),
              f"{rv:.4f}", ha="center", va=va_r, fontsize=8.5, color="#333333")

ax_B.set_xticks(x_b)
ax_B.set_xticklabels(short_dz, rotation=25, ha="right")
ax_B.set_ylabel("Indirect Effect (β product)")
ax_B.set_title("SEM Indirect Effects by Pathway", pad=8)
ax_B.legend(loc="lower right", framealpha=0.85, edgecolor="#cccccc")
ax_B.spines[["top", "right"]].set_visible(False)

ax_B.text(-0.08, 1.03, "B", transform=ax_B.transAxes,
          fontsize=14, fontweight="bold", ha="left", va="top")

# ═════════════════════════════════════════════════════════════════════════════
# Panel C — 56-cell risk matrix heatmap (WHO exceedance %)
# ═════════════════════════════════════════════════════════════════════════════
ZONE_ORDER  = ["Barind_Tract", "Northern_Terrace", "Brahmaputra_Floodplain",
               "Ganges_Floodplain", "GBM_Delta", "Meghna_Floodplain", "Eastern_Hills"]
SHORT_Z     = ["Barind", "N.Terr.", "Brahma.", "Ganges", "GBM", "Meghna", "E.Hills"]
DEPTH_ORDER = ["Shallow", "Intermediate", "Medium_Deep", "Deep"]
SHORT_D     = ["Shallow", "Interm.", "Med-D.", "Deep"]

def build_exceedance_matrix(df, season):
    mat = np.full((len(DEPTH_ORDER), len(ZONE_ORDER)), np.nan)
    sub = df[df["Season"] == season]
    for _, row in sub.iterrows():
        d = row["Depth_Zone"]
        z = row["Phys_Zone"]
        if d in DEPTH_ORDER and z in ZONE_ORDER:
            r = DEPTH_ORDER.index(d)
            c = ZONE_ORDER.index(z)
            mat[r, c] = row["Exceedance_pct"]
    return mat

mat_dry = build_exceedance_matrix(t24, "Dry")
mat_wet = build_exceedance_matrix(t24, "Wet")
# Stack: Dry (rows 0-3) then Wet (rows 4-7)
mat_all = np.vstack([mat_dry, mat_wet])

cmap_heat = plt.cm.get_cmap("RdYlGn_r")
im = ax_C.imshow(mat_all, cmap=cmap_heat, vmin=0, vmax=60,
                 aspect="auto", interpolation="none")

# Row labels
row_labels = [f"Dry – {d}" for d in SHORT_D] + [f"Wet – {d}" for d in SHORT_D]
ax_C.set_yticks(np.arange(8))
ax_C.set_yticklabels(row_labels, fontsize=10)
ax_C.set_xticks(np.arange(7))
ax_C.set_xticklabels(SHORT_Z, rotation=20, ha="right", fontsize=10)

# Divider between Dry and Wet blocks
ax_C.axhline(3.5, color="white", linewidth=3.0, zorder=5)

# Cell annotations — WHO exceedance %
for ri, (season, depth) in enumerate([
        ("Dry", "Shallow"), ("Dry", "Intermediate"),
        ("Dry", "Medium_Deep"), ("Dry", "Deep"),
        ("Wet", "Shallow"), ("Wet", "Intermediate"),
        ("Wet", "Medium_Deep"), ("Wet", "Deep")]):
    for ci, zone in enumerate(ZONE_ORDER):
        sub = t24[(t24["Season"]==season) &
                  (t24["Depth_Zone"]==depth) &
                  (t24["Phys_Zone"]==zone)]
        if sub.empty or sub["N"].values[0] == 0:
            continue
        v = sub["Exceedance_pct"].values[0]
        if not np.isnan(v):
            # Dark text on light cells, white on dark
            text_col = "white" if v > 40 else "black"
            ax_C.text(ci, ri, f"{v:.0f}%",
                      ha="center", va="center",
                      fontsize=8, color=text_col, fontweight="bold")

# DRY / WET side labels
ax_C.text(-0.18, 1.5/8, "DRY", transform=ax_C.transAxes,
          ha="center", va="center", fontsize=11, rotation=90,
          fontweight="bold", color=DRY_C)
ax_C.text(-0.18, 5.5/8, "WET", transform=ax_C.transAxes,
          ha="center", va="center", fontsize=11, rotation=90,
          fontweight="bold", color=WET_C)

# Colorbar
div = make_axes_locatable(ax_C)
cax = div.append_axes("right", size="4%", pad=0.06)
cb  = plt.colorbar(im, cax=cax)
cb.set_label("WHO exceedance (%)", fontsize=11)
cb.ax.tick_params(labelsize=10)

ax_C.set_title("56-Cell Seasonal Risk Matrix", pad=8)
ax_C.text(-0.25, 1.03, "C", transform=ax_C.transAxes,
          fontsize=14, fontweight="bold", ha="left", va="top")

# ═════════════════════════════════════════════════════════════════════════════
# Panel D — Seasonal arsenic comparison (Dry vs Wet median by depth)
# ═════════════════════════════════════════════════════════════════════════════
depth_d   = t_h2["Depth_Zone"].tolist()
short_d_labels = [d.replace("Medium_Deep", "Med-Deep") for d in depth_d]
med_dry   = t_h2["Median_As_Dry"].values
med_wet   = t_h2["Median_As_Wet"].values
sig_flag  = t_h2["Significant"].values
p_vals    = t_h2["p_value"].values

x_d = np.arange(len(depth_d))
w_d = 0.32

b1 = ax_D.bar(x_d - w_d/2, med_dry, w_d, label="Dry season",
              color=DRY_C, edgecolor="white", linewidth=0.8, alpha=0.92)
b2 = ax_D.bar(x_d + w_d/2, med_wet, w_d, label="Wet season",
              color=WET_C, edgecolor="white", linewidth=0.8, alpha=0.92)

# WHO 10 µg/L reference line
who_line = ax_D.axhline(10, color="#b2182b", linewidth=1.4,
                         linestyle="--", zorder=3)
ax_D.text(len(depth_d) - 0.5, 10.4, "WHO 10 µg/L",
          ha="right", va="bottom", fontsize=9, color="#b2182b", fontstyle="italic")

# Significance stars
for i, (dy, wy, sig, pv) in enumerate(zip(med_dry, med_wet, sig_flag, p_vals)):
    top = max(dy, wy)
    y_star = top + 0.4
    if sig == "YES":
        if pv < 0.001:
            star = "***"
        elif pv < 0.01:
            star = "**"
        else:
            star = "*"
        ax_D.text(i, y_star, star, ha="center", va="bottom",
                  fontsize=12, color="black", fontweight="bold")
    else:
        ax_D.text(i, y_star, "ns", ha="center", va="bottom",
                  fontsize=9, color="#888888")

ax_D.set_xticks(x_d)
ax_D.set_xticklabels(short_d_labels, rotation=25, ha="right")
ax_D.set_ylabel("Median As (µg/L)")
ax_D.set_title("Seasonal Arsenic Variation by Depth", pad=8)
ax_D.legend(loc="upper right", framealpha=0.85, edgecolor="#cccccc")
ax_D.spines[["top", "right"]].set_visible(False)

ax_D.text(-0.10, 1.03, "D", transform=ax_D.transAxes,
          fontsize=14, fontweight="bold", ha="left", va="top")

# ═════════════════════════════════════════════════════════════════════════════
# Panel E — Shannon entropy (Dry vs Wet, by depth zone)
# ═════════════════════════════════════════════════════════════════════════════
# Use depth zones only (exclude ALL row if present)
t03_depth = t03[t03["Depth_Zone"] != "ALL"].reset_index(drop=True)
if t03_depth.empty:
    t03_depth = t03.copy()

zones_e  = t03_depth["Depth_Zone"].tolist()
short_e  = [z.replace("Medium_Deep", "Med-Deep")
              .replace("Intermediate", "Interm.") for z in zones_e]
h_dry_e  = t03_depth["H_Dry_bits"].values
h_wet_e  = t03_depth["H_Wet_bits"].values
sig_e    = t03_depth["DeltaH_Significant"].values

x_e = np.arange(len(zones_e))
w_e = 0.32

ax_E.bar(x_e - w_e/2, h_dry_e, w_e, label="Dry season",
         color=DRY_C, edgecolor="white", linewidth=0.8, alpha=0.92)
ax_E.bar(x_e + w_e/2, h_wet_e, w_e, label="Wet season",
         color=WET_C, edgecolor="white", linewidth=0.8, alpha=0.92)

for i, (hd, hw, sig) in enumerate(zip(h_dry_e, h_wet_e, sig_e)):
    top = max(hd, hw)
    if sig == "YES":
        ax_E.text(i, top + 0.02, "*", ha="center", va="bottom",
                  fontsize=13, color="black", fontweight="bold")

ax_E.set_xticks(x_e)
ax_E.set_xticklabels(short_e, rotation=25, ha="right")
ax_E.set_ylabel("Shannon Entropy (bits)")
ax_E.set_title("Mechanism Entropy by Zone", pad=8)
ax_E.legend(loc="upper left", framealpha=0.85, edgecolor="#cccccc")
ax_E.spines[["top", "right"]].set_visible(False)

ax_E.text(-0.10, 1.03, "E", transform=ax_E.transAxes,
          fontsize=14, fontweight="bold", ha="left", va="top")

# ═════════════════════════════════════════════════════════════════════════════
# Panel F — Satellite ensemble null result (R² bar chart)
# ═════════════════════════════════════════════════════════════════════════════
MODEL_NAMES = {
    "Ensemble (stacked)": "Ensemble",
    "rf":                 "RF",
    "xgb":                "XGBoost",
    "ridge":              "Ridge",
    "en":                 "Elastic Net",
}
t16_plot = t16.copy()
t16_plot["ShortName"] = (t16_plot["Model"]
                         .map(MODEL_NAMES)
                         .fillna(t16_plot["Model"]))
YLIM_F_LOW = -0.12
t16_plot["CV_R2_disp"] = t16_plot["CV_R2"].clip(lower=YLIM_F_LOW)

# Red for negative R², blue for positive
bar_colours = [WET_C if r >= 0 else "#D94F3D" for r in t16_plot["CV_R2"]]

x_f = np.arange(len(t16_plot))
ax_F.bar(x_f, t16_plot["CV_R2_disp"], color=bar_colours,
         edgecolor="white", linewidth=0.8, alpha=0.90)
ax_F.axhline(0, color="black", linewidth=1.0, linestyle="--", label="R²=0 baseline")

# Annotate actual R² values; for clipped bars show actual value at clip boundary
for i, r2 in enumerate(t16_plot["CV_R2"]):
    disp = t16_plot["CV_R2_disp"].iloc[i]
    is_clipped = r2 < YLIM_F_LOW
    if is_clipped:
        # Place label at clipping boundary with downward arrow to signal truncation
        ax_F.text(i, YLIM_F_LOW - 0.003, f"↓ {r2:.3f}",
                  ha="center", va="top", fontsize=9.5, color="#D94F3D",
                  fontweight="bold")
    else:
        offset = 0.004 if disp >= 0 else -0.010
        va = "bottom" if disp >= 0 else "top"
        ax_F.text(i, disp + offset, f"{r2:.3f}",
                  ha="center", va=va, fontsize=9.5, color="black", fontweight="bold")

ax_F.set_ylim(YLIM_F_LOW, 0.04)

# Annotate null result
ax_F.text(0.98, 0.97, "Null result: R²≈0.008",
          transform=ax_F.transAxes,
          ha="right", va="top", fontsize=10,
          color="#b2182b", fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.3", facecolor="#fde8e8",
                    edgecolor="#b2182b", alpha=0.85))

ax_F.set_xticks(x_f)
ax_F.set_xticklabels(t16_plot["ShortName"], rotation=30, ha="right")
ax_F.set_ylabel("Cross-validated R²")
ax_F.set_title("Satellite Forecast Performance (Null Result)", pad=8)
ax_F.legend(loc="lower right", framealpha=0.85, edgecolor="#cccccc")
ax_F.spines[["top", "right"]].set_visible(False)

ax_F.text(-0.10, 1.03, "F", transform=ax_F.transAxes,
          fontsize=14, fontweight="bold", ha="left", va="top")

# ── Save ───────────────────────────────────────────────────────────────────────
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT}")
print(f"Size:  {os.path.getsize(OUT) / 1024:.1f} KB")
