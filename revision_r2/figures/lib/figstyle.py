"""Shared publication figure style for the JHM revision.
Vector-first (PDF/SVG) with embedded, editable fonts. Colourblind-safe palette.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

# ---- consistent, colourblind-safe palette ----
PO4   = '#C0392B'   # phosphate (vermillion)
FE    = '#2166AC'   # iron (blue)
HCO3  = '#1B7837'   # bicarbonate (green)
WARM1 = '#d9482c'   # arsenic warm, mid   (replaces the retired purple accent)
WARM2 = '#7d1a15'   # arsenic warm, deep  (Bangladesh 50 threshold)
DRY   = '#D8973C'   # dry season (ochre)
WET   = '#2166AC'   # wet season (blue)
GREY  = '#9AA0A6'
INK   = '#222222'
BAR   = '#C9D3DD'   # neutral bars (un-highlighted)
MUTE  = '#C2C7CC'   # de-emphasised / "other" series
SEQ   = 'YlOrRd'    # sequential  (arsenic / exceedance / risk)
SEQ2  = 'cividis'   # sequential  (secondary continuous, e.g. depth)
DIV   = 'RdYlBu_r'  # diverging   (unused; kept for reference)
# the paper's single sequential ramp, shared by every magnitude surface and heatmap
WARMR = ['#fdece0', '#fbd5bd', '#f8b894', '#f2946a', '#eb6834', '#d9482c', '#b3261e', '#7d1a15']
CONTOUR = '#104281'   # dark blue, the one cool mark, so contours read over the warm ramp
# hatch encodes the series, so paired bars survive greyscale and colour blindness
HATCH_1 = '///'
HATCH_2 = '\\\\\\'

def set_style():
    mpl.rcParams.update({
        # vector-friendly, editable text in PDF/SVG
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'dejavusans',
        'font.size': 9.5,
        'axes.titlesize': 10.5, 'axes.titleweight': 'bold',
        'axes.labelsize': 9.5, 'axes.labelcolor': INK,
        'axes.edgecolor': '#444444', 'axes.linewidth': 0.8,
        'axes.titlepad': 7, 'axes.labelpad': 3,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': '#E6E6E6', 'grid.linewidth': 0.6,
        'xtick.color': INK, 'ytick.color': INK,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'legend.fontsize': 8.5, 'legend.frameon': False,
        'figure.dpi': 150, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.03,
        'lines.linewidth': 2.0, 'lines.solid_capstyle': 'round',
        'hatch.linewidth': 0.6,
    })

def panel_tag(ax, tag, x=0.018, y=0.97):
    """Bold panel letter inside the top-left corner, with a white halo so it
    never collides with titles or data."""
    ax.text(x, y, tag, transform=ax.transAxes, fontsize=12.5,
            fontweight='bold', va='top', ha='left', color=INK, zorder=20,
            bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.78))

def clean(ax):
    ax.tick_params(length=3)
    ax.set_axisbelow(True)
    return ax

def save_all(fig, stem, outdir):
    """Save a figure as vector PDF + SVG + 600-dpi PNG."""
    outdir = Path(outdir)
    paths = []
    for ext in ('pdf', 'svg', 'png'):
        p = outdir / f'{stem}.{ext}'
        fig.savefig(p)
        paths.append(p.name)
    plt.close(fig)
    return paths
