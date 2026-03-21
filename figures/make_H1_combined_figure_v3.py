"""
Generate publication-quality 4-panel combined figure for Hypothesis 1.
H1: Multi-Mechanism Arsenic Mobilisation at National Scale.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Load data ────────────────────────────────────────────────────────────────
DATA = "/Users/rakibhhridoy/AsGW/GroundWater/hypo1/output/bangladesh_groundwater_arsenic_1807samples.csv"
PROV = "/Users/rakibhhridoy/AsGW/GroundWater/hypo1/output/tables/Table_26_province_clustering.csv"
OUT  = "/Users/rakibhhridoy/AsGW/GroundWater/hypo1/output/figures/H1_combined_figure.png"

df = pd.read_csv(DATA)
prov = pd.read_csv(PROV)

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
})

fig = plt.figure(figsize=(16, 12))

# Use GridSpec: 2x2, but Panel C gets two sub-axes
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.30)

BLUE   = "#3C6E9C"
TEAL   = "#2A9D8F"
RED    = "#C44E52"
PANEL_LABEL_KW = dict(fontsize=16, fontweight="bold", va="top", ha="left",
                       transform=None)  # will use ax.transAxes

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL A — Arsenic concentration distribution (log-scale histogram)
# ═══════════════════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0, 0])

as_vals = df["As"].dropna()
as_pos = as_vals[as_vals > 0]

# Log-spaced bins
bins = np.logspace(np.log10(as_pos.min()), np.log10(as_pos.max()), 50)
ax_a.hist(as_pos, bins=bins, color=BLUE, edgecolor="white", linewidth=0.4, alpha=0.88)
ax_a.set_xscale("log")

# Threshold lines
for thresh, label, xoff in [(10, "WHO 10 µg/L", 1.08), (50, "Bangladesh\n50 µg/L", 1.08)]:
    ax_a.axvline(thresh, color=RED, ls="--", lw=1.5, zorder=5)
    ax_a.text(thresh * xoff, ax_a.get_ylim()[1] * 0.92 if thresh == 10 else ax_a.get_ylim()[1] * 0.75,
              label, color=RED, fontsize=9.5, va="top", fontweight="bold")

# Percentages
pct_who = (as_vals >= 10).mean() * 100
pct_bd  = (as_vals >= 50).mean() * 100
ax_a.text(0.03, 0.50,
          f"{pct_who:.1f}% > WHO\n{pct_bd:.1f}% > BD std",
          transform=ax_a.transAxes, ha="left", va="center",
          fontsize=11, fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999", alpha=0.9))

ax_a.set_xlabel("Arsenic concentration (µg/L)")
ax_a.set_ylabel("Number of samples")
ax_a.set_title("Arsenic Concentration Distribution", fontweight="bold")
ax_a.yaxis.grid(True, color="#DDDDDD", linewidth=0.5)
ax_a.set_axisbelow(True)

# Fix threshold annotation y-positions after limits are set
ylim = ax_a.get_ylim()

# Panel label
ax_a.text(0.02, 0.98, "(A)", transform=ax_a.transAxes, fontsize=16,
          fontweight="bold", va="top", ha="left")

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL B — Spearman correlations with Arsenic (horizontal bar)
# ═══════════════════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[0, 1])

corr_vars = ["PO43-", "HCO3-", "Fe2+", "EC", "TDS", "K+", "pH",
             "Mg2+", "Na+", "SO42-", "NO3-", "DO", "Mn2+", "Depth"]

rhos, pvals = [], []
for v in corr_vars:
    if v in df.columns:
        mask = df["As"].notna() & df[v].notna()
        r, p = stats.spearmanr(df.loc[mask, "As"], df.loc[mask, v])
        rhos.append(r)
        pvals.append(p)
    else:
        rhos.append(np.nan)
        pvals.append(np.nan)

corr_df = pd.DataFrame({"var": corr_vars, "rho": rhos, "p": pvals}).dropna()
corr_df["abs_rho"] = corr_df["rho"].abs()
corr_df = corr_df.sort_values("abs_rho", ascending=True)

colors = [BLUE if r >= 0 else RED for r in corr_df["rho"]]
bars = ax_b.barh(corr_df["var"], corr_df["rho"], color=colors, edgecolor="white",
                 linewidth=0.4, height=0.7)

# Annotate rho values
for bar, rho in zip(bars, corr_df["rho"]):
    xpos = bar.get_width()
    offset = 0.008 if rho >= 0 else -0.008
    ha = "left" if rho >= 0 else "right"
    ax_b.text(xpos + offset, bar.get_y() + bar.get_height() / 2,
              f"{rho:.3f}", va="center", ha=ha, fontsize=9, fontweight="bold")

ax_b.axvline(0, color="#333333", lw=0.8)
ax_b.set_xlabel("Spearman ρ with Arsenic")
ax_b.set_title("Spearman Correlations with Arsenic", fontweight="bold")
ax_b.xaxis.grid(True, color="#DDDDDD", linewidth=0.5)
ax_b.set_axisbelow(True)
ax_b.set_xlim(-0.25, 0.42)

ax_b.text(0.02, 0.98, "(B)", transform=ax_b.transAxes, fontsize=16,
          fontweight="bold", va="top", ha="left")

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL C — PO4 vs As and Fe vs As dual scatter (log-log hexbin)
# ═══════════════════════════════════════════════════════════════════════════════
gs_c = gs[1, 0].subgridspec(1, 2, wspace=0.35)
ax_c1 = fig.add_subplot(gs_c[0, 0])
ax_c2 = fig.add_subplot(gs_c[0, 1])

def hexbin_scatter(ax, x_col, y_col, xlabel, ylabel, df_src):
    """Draw a log-log hexbin scatter with regression and Spearman annotation."""
    mask = (df_src[x_col] > 0) & (df_src[y_col] > 0) & df_src[x_col].notna() & df_src[y_col].notna()
    x = df_src.loc[mask, x_col].values
    y = df_src.loc[mask, y_col].values

    hb = ax.hexbin(x, y, gridsize=35, cmap="YlGnBu", mincnt=1,
                   xscale="log", yscale="log", linewidths=0.2, norm=LogNorm())

    # Regression line in log-log space
    lx, ly = np.log10(x), np.log10(y)
    slope, intercept, _, _, _ = stats.linregress(lx, ly)
    xs = np.linspace(lx.min(), lx.max(), 100)
    ax.plot(10**xs, 10**(slope * xs + intercept), color=RED, lw=2, ls="-", zorder=5)

    rho, p = stats.spearmanr(x, y)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    ax.text(0.05, 0.05, f"ρ = {rho:.3f}{sig}\nn = {len(x)}",
            transform=ax.transAxes, va="bottom", fontsize=10.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999", alpha=0.9))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.4)
    ax.xaxis.grid(True, color="#EEEEEE", linewidth=0.4)
    ax.set_axisbelow(True)
    return hb

hexbin_scatter(ax_c1, "PO43-", "As", "PO₄³⁻ (mg/L)", "As (µg/L)", df)
ax_c1.set_title("Phosphate vs Arsenic", fontweight="bold", fontsize=12)

hb2 = hexbin_scatter(ax_c2, "Fe2+", "As", "Fe²⁺ (mg/L)", "As (µg/L)", df)
ax_c2.set_title("Iron vs Arsenic", fontweight="bold", fontsize=12)

# Shared colorbar
cb = fig.colorbar(hb2, ax=[ax_c1, ax_c2], shrink=0.75, pad=0.02, label="Count")

ax_c1.text(0.02, 0.98, "(C)", transform=ax_c1.transAxes, fontsize=16,
           fontweight="bold", va="top", ha="left",
           bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL D — Geochemical Province Map
# ═══════════════════════════════════════════════════════════════════════════════
ax_d = fig.add_subplot(gs[1, 1])

province_colors = ["#2A9D8F", "#E76F51", "#264653", "#E9C46A"]

# Filter to Bangladesh bounds (remove outliers)
prov = prov[(prov["Latitude"] >= 20.5) & (prov["Latitude"] <= 27) &
            (prov["Longitude"] >= 87.5) & (prov["Longitude"] <= 93)]

# Compute per-province stats
n_prov = prov["Province"].nunique()
prov_stats = prov.groupby("Province").agg(
    mean_as=("As", "mean"),
    count=("As", "size")
).reset_index()

# Assign descriptive names based on mean As
prov_stats = prov_stats.sort_values("mean_as")
desc_names = {
    0: "Low-As",
    1: "Moderate-As",
    2: "Elevated-As",
    3: "High-As",
}

# Map province IDs to ordered labels
sorted_ids = prov_stats["Province"].tolist()
id_to_rank = {pid: rank for rank, pid in enumerate(sorted_ids)}

for rank, pid in enumerate(sorted_ids):
    row = prov_stats[prov_stats["Province"] == pid].iloc[0]
    label = f"{desc_names[rank]} (mean {row['mean_as']:.1f} µg/L, n={int(row['count'])})"
    mask = prov["Province"] == pid
    ax_d.scatter(prov.loc[mask, "Longitude"], prov.loc[mask, "Latitude"],
                 c=province_colors[rank], s=8, alpha=0.6, label=label,
                 edgecolors="none", rasterized=True)

ax_d.set_xlabel("Longitude (°E)")
ax_d.set_ylabel("Latitude (°N)")
ax_d.set_title("Geochemical Provinces", fontweight="bold")
ax_d.legend(loc="lower left", fontsize=8.5, framealpha=0.9, edgecolor="#999999",
            markerscale=2.5, handletextpad=0.3)
ax_d.set_aspect("equal", adjustable="datalim")
ax_d.yaxis.grid(True, color="#EEEEEE", linewidth=0.4)
ax_d.xaxis.grid(True, color="#EEEEEE", linewidth=0.4)
ax_d.set_axisbelow(True)

ax_d.text(0.02, 0.98, "(D)", transform=ax_d.transAxes, fontsize=16,
          fontweight="bold", va="top", ha="left")

# ── Save ─────────────────────────────────────────────────────────────────────
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {OUT}")
print(f"File size: {__import__('os').path.getsize(OUT) / 1024:.0f} KB")
