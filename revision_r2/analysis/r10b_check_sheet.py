"""Side-by-side check sheet: each original BWDB log image next to its digitised
column drawn on the same depth scale, with matched wells marked. Writes
r10_check_all.pdf (all 69) and r10_spotcheck.pdf (10 logs for the authors)."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import r10_digitise_logs as M

O = M.O
COL = {"clay": "#4d4d4d", "silt/silty clay": "#a6a6a6", "very fine sand": "#aaeaea",
       "fine sand": "#8fbeec", "medium sand": "#6097d7", "coarse sand": "#3d79c7"}
L = pd.read_csv(f"{O}/r10_log_layers.csv"); meta = pd.read_csv(f"{O}/r10_log_meta.csv")
wells = pd.read_csv(f"{O}/r11_logged_wells.csv") if os.path.exists(f"{O}/r11_logged_wells.csv") else None
SPOT = ["img-040-001", "img-041-005", "img-042-009", "img-043-015", "img-045-028",
        "img-046-034", "img-047-044", "img-048-045", "img-049-057", "img-050-063"]


def panel(fig, gs, im):
    r = meta[meta.image == im].iloc[0]
    img = Image.open(f"{M.IMG}/{im}.png").convert("RGB")
    a1 = fig.add_subplot(gs[0]); a1.imshow(img); a1.axis("off")
    a2 = fig.add_subplot(gs[1])
    for t in L[L.image == im].itertuples():
        a2.add_patch(plt.Rectangle((0, t.top_m), 1, t.bottom_m - t.top_m, color=COL[t.lithology], lw=0))
        a2.plot([0, 1], [t.top_m, t.top_m], color="k", lw=0.3)
    if wells is not None:
        for wv in wells[wells.image == im].itertuples():
            a2.plot([1.05], [wv.depth], marker="<", color="k", ms=5, clip_on=False)
            a2.text(1.2, wv.depth, f"{wv.depth:.0f} m, As {wv.As:.0f}", va="center", fontsize=6)
    a2.set_xlim(0, 1); a2.set_ylim(r.total_depth_m, 0); a2.set_xticks([])
    a2.set_ylabel("Depth (m)", fontsize=7); a2.tick_params(labelsize=6)
    a2.set_title(f"{im}\n{r.source}, {r.total_depth_m:.0f} m", fontsize=6)


def sheet(images, path, per=4):
    with PdfPages(path) as pdf:
        for k in range(0, len(images), per):
            fig = plt.figure(figsize=(11, 8.2))
            outer = fig.add_gridspec(1, per, wspace=0.55)
            for j, im in enumerate(images[k:k + per]):
                panel(fig, outer[j].subgridspec(1, 2, width_ratios=[1.0, 0.45], wspace=0.15), im)
            h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COL.values()]
            fig.legend(h, list(COL), loc="lower center", ncol=6, fontsize=7, frameon=False)
            pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    sheet(list(meta.image), f"{O}/r10_check_all.pdf")
    sheet(SPOT, f"{O}/r10_spotcheck.pdf", per=2)
    print("wrote r10_check_all.pdf and r10_spotcheck.pdf")
