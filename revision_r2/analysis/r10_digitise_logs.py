"""Digitise the 69 BWDB exploratory borehole logs (BWDB 2023, Section 3.5,
Figures 3-2 to 3-8, pdf pages 40-50) into layer tables.

Each log is an embedded raster image: a title, a depth axis with labelled ticks,
one lithology column and a legend of seven classes. For every image the script
  1. finds the lithology column (the widest block of non-white columns),
  2. calibrates depth by OCR of the tick labels (tesseract) and a linear fit,
  3. classifies each pixel row of the column by its mean colour against the
     legend colours, smoothed with a running mode,
  4. writes contiguous layers (top_m, bottom_m, class).
Grey rows are split by brightness into clay (dark) and silt / silty clay
(lighter); the two lighter grey patterns share one mean colour and are pooled.
"""
import os, re, glob, subprocess, json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

R = os.environ.get("PAPER1_ROOT", str(Path(__file__).resolve().parents[2]))
PDF = os.environ.get("BWDB_PDF", f"{R}/data/external/BWDB_2023/16594.pdf")
O = f"{R}/revision_r2/output"; os.makedirs(O, exist_ok=True)
IMG = os.environ.get("LOG_IMG_DIR", f"{O}/r10_log_images")

# legend colours measured from the legend swatches of the logs (RGB means)
SAND = {"very fine sand": (170, 234, 234), "fine sand": (143, 190, 236),
        "medium sand": (96, 151, 215), "coarse sand": (61, 121, 199)}
CLASSES = ["clay", "silt/silty clay", "very fine sand", "fine sand", "medium sand", "coarse sand"]
FINE = {"clay", "silt/silty clay"}


def extract():
    os.makedirs(IMG, exist_ok=True)
    if not glob.glob(f"{IMG}/img-04*.png"):
        subprocess.run(["pdfimages", "-png", "-p", "-f", "40", "-l", "50", PDF, f"{IMG}/img"], check=True)
    return sorted(glob.glob(f"{IMG}/img-0[45][0-9]-*.png"))


def ocr_tsv(im, digits=True, psm="6"):
    tmp = f"{IMG}/_ocr.png"; im.save(tmp)
    cfg = ["-c", "tessedit_char_whitelist=0123456789"] if digits else []
    t = subprocess.run(["tesseract", tmp, "-", "--psm", psm, *cfg, "tsv"], capture_output=True, text=True).stdout
    rows = [r.split("\t") for r in t.strip().split("\n")[1:]]
    return [r for r in rows if len(r) == 12 and r[11].strip()]


def column_box(a):
    nw = a.sum(2) < 720
    cf = nw.mean(0); thr = 0.5 * cf.max()
    best = (0, 0, 0); x = 0
    while x < len(cf):
        if cf[x] > thr:
            x0 = x
            while x < len(cf) and cf[x] > thr: x += 1
            if x - x0 > best[0]: best = (x - x0, x0, x)
        x += 1
    _, x0, x1 = best
    rf = nw[:, x0 + 2:x1 - 2].mean(1)
    best = (0, 0, 0); y = 0
    while y < len(rf):
        if rf[y] > 0.5:
            y0 = y
            while y < len(rf) and rf[y] > 0.5: y += 1
            if y - y0 > best[0]: best = (y - y0, y0, y)
        y += 1
    _, y0, y1 = best
    return x0, x1, y0, y1


NICE = (5, 10, 20, 25, 50)
# Label step (m) read by eye for the logs whose OCR scale failed the tick check
# or found no ticks. OCR still supplies the label positions; the step resolves
# dropped or extra digits ("2" for 20, "500" for "50.0").
STEP_BY_EYE = {
    "img-041-003": 50, "img-041-004": 50, "img-041-005": 20, "img-042-010": 50,
    "img-043-011": 20, "img-043-012": 20, "img-043-015": 50, "img-043-016": 50,
    "img-044-019": 50, "img-044-021": 50, "img-045-023": 50, "img-045-024": 50,
    "img-045-025": 50, "img-045-026": 50, "img-045-028": 20, "img-045-029": 50,
    "img-045-030": 50, "img-046-032": 20, "img-046-034": 50, "img-046-035": 50,
    "img-046-037": 50, "img-047-041": 50, "img-047-042": 50, "img-047-043": 50,
    "img-047-044": 50, "img-048-045": 50, "img-048-046": 50, "img-048-048": 50,
    "img-048-050": 50, "img-049-054": 50, "img-049-057": 50, "img-049-060": 50,
    "img-049-062": 50, "img-050-063": 50, "img-050-067": 50}
# (label_m, tick_y_px) read by eye from the tick marks, where neither OCR nor the
# 10 m tick rule works (Bhanga has 20 m minor ticks).
MANUAL = {"img-047-041": (150, 568), "img-048-050": (300, 653),
          "img-049-057": (300, 644), "img-049-060": (300, 654)}


def tick_spacing(a, x0, y0, y1):
    """Median spacing of all tick marks (major and minor) left of the axis."""
    dark = a.sum(2) < 380
    best = (0, None)
    for x in range(max(0, x0 - 8), x0 + 2):
        col = dark[y0 - 2:y1 + 2, x]; run = cur = 0
        for v in col:
            cur = cur + 1 if v else 0; run = max(run, cur)
        if run > best[0]: best = (run, x)
    ax = best[1]; out = []
    for dx in (2, 3):
        ys = np.where(dark[y0 - 3:y1 + 3, ax - dx])[0] + y0 - 3
        if len(ys) < 3: continue
        grp = np.split(ys, np.where(np.diff(ys) > 2)[0] + 1)
        c = np.array([g.mean() for g in grp if len(g) <= 3])
        if len(c) >= 3:
            dd = np.diff(c); md = np.median(dd)
            reg = dd[np.abs(dd / md - 1) < 0.3] if md > 0 else dd[:0]
            if len(reg) < 2: continue
            md = reg.mean()   # mean of regular gaps, sub-pixel
            n = round((c[-1] - c[0]) / md)
            out.append((c[-1] - c[0]) / n)   # spacing over the whole run, not one gap
    return ax, (min(out) if out else None)


def depth_axis(img, x0, y0, y1, name=None):
    """depth = b * (y - y0): the column top is 0 m in every log. b is the median
    of label/offset ratios over OCR'd tick labels, checked against the tick
    spacing (spacing * b must be a round plotting interval)."""
    a = np.asarray(img).astype(int)
    ax, sp = tick_spacing(a, x0, y0, y1)
    if name in MANUAL:
        v, yl = MANUAL[name]; bb = v / (yl - y0)
        return (-bb * y0, bb), {"source": "manual", "tick_m": round(sp * bb, 2) if sp else None}
    pts = []
    for S in (4, 3, 5):
        strip = img.crop((0, max(0, y0 - 12), max(1, ax - 2), min(img.height, y1 + 12)))
        big = strip.resize((strip.width * S, strip.height * S), Image.LANCZOS).convert("L")
        for r in ocr_tsv(big, psm="6") + ocr_tsv(big, psm="11"):
            v = r[11].strip()
            if not v.isdigit() or float(r[10]) < 30: continue
            yc = max(0, y0 - 12) + (int(r[7]) + int(r[9]) / 2) / S
            if float(v) > 0 and yc - y0 > 10: pts.append((round(yc), float(v)))
    pts = sorted(set(pts))
    if not pts: return None, {"why": "no labels"}
    if name in STEP_BY_EYE:
        st = STEP_BY_EYE[name]; cand = []
        for y, d in pts:
            for c in (d, d * 10, d / 10):
                if c >= st and c <= 400 and abs(c / st - round(c / st)) < 1e-6: cand.append(c / (y - y0))
        agree = []
        if cand:
            cand = np.array(cand)
            sup = [(np.abs(cand / r - 1) < 0.03).sum() for r in cand]
            r0 = cand[int(np.argmax(sup))]; agree = cand[np.abs(cand / r0 - 1) < 0.03]
        if len(agree) >= 2:
            bb = np.median(agree)
            return (-bb * y0, bb), {"source": "ocr+step", "n_labels": int(len(agree)), "tick_m": round(sp * bb, 2) if sp else None}
        if sp:   # minor ticks are 10 m apart on every log where both could be read
            bb = 10.0 / sp
            return (-bb * y0, bb), {"source": "10 m ticks", "n_labels": int(len(agree)), "tick_m": 10.0}
        return None, {"why": "no labels match step, no ticks"}
    rat = np.array([d / (y - y0) for y, d in pts]); med = np.median(rat)
    ok = np.abs(rat / med - 1) < 0.04
    if ok.sum() < 2: return None, {"why": "labels disagree"}
    bb = np.median(rat[ok])
    tick = sp * bb if sp else None
    nice = tick is not None and any(abs(tick / n - 1) < 0.06 for n in NICE)
    if not nice: return None, {"why": f"tick {tick:.2f} m not round" if tick else "no ticks"}
    return (-bb * y0, bb), {"source": "ocr", "n_labels": int(ok.sum()), "tick_m": round(tick, 2)}


def classify_rows(a, x0, x1, y0, y1):
    core = a[y0:y1, x0 + 3:x1 - 3].astype(float)
    m = core.mean(1)
    lab = []
    for r, g, b in m:
        if abs(b - r) < 25 and abs(g - r) < 25:
            lab.append("clay" if (r + g + b) / 3 < 120 else "silt/silty clay")
        else:
            lab.append(min(SAND, key=lambda k: sum((np.array((r, g, b)) - SAND[k]) ** 2)))
    # running mode, window 7 rows, removes the thin black boundary lines
    lab = np.array(lab); out = lab.copy(); k = 3
    for i in range(len(lab)):
        w = lab[max(0, i - k):i + k + 1]
        v, c = np.unique(w, return_counts=True); out[i] = v[c.argmax()]
    return out


def layers(lab, y0, a, b, min_px=3):
    segs = []; s = 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            segs.append([s, i, lab[s]]); s = i
    # absorb slivers thinner than min_px into the thicker neighbour
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for j, (s0, s1, c) in enumerate(segs):
            if s1 - s0 < min_px:
                nb = j - 1 if j == len(segs) - 1 or (j > 0 and segs[j - 1][1] - segs[j - 1][0] >= segs[j + 1][1] - segs[j + 1][0]) else j + 1
                lo, hi = min(j, nb), max(j, nb)
                keepc = segs[nb][2]
                segs[lo:hi + 1] = [[segs[lo][0], segs[hi][1], keepc]]
                changed = True; break
        # merge equal neighbours
        k = 0
        while k < len(segs) - 1:
            if segs[k][2] == segs[k + 1][2]:
                segs[k:k + 2] = [[segs[k][0], segs[k + 1][1], segs[k][2]]]
            else: k += 1
    f = lambda py: a + b * (y0 + py)
    return [(round(f(s0), 2), round(f(s1), 2), c) for s0, s1, c in segs]


def title(img, y0):
    top = img.crop((0, 0, img.width, max(10, y0 - 14)))
    big = top.resize((top.width * 3, top.height * 3), Image.LANCZOS)
    tmp = f"{IMG}/_t.png"; big.save(tmp)
    return " ".join(subprocess.run(["tesseract", tmp, "-", "--psm", "6"], capture_output=True, text=True).stdout.split())


def main():
    rows, meta = [], []
    for f in extract():
        img = Image.open(f).convert("RGB"); a = np.asarray(img).astype(int)
        x0, x1, y0, y1 = column_box(a)
        cal, info = depth_axis(img, x0, y0, y1, os.path.basename(f)[:-4])
        name = os.path.basename(f)[:-4]
        if cal is None:
            meta.append(dict(image=name, ok=False, title=title(img, y0), **info)); continue
        aa, bb = cal
        lab = classify_rows(a, x0, x1, y0, y1)
        L = layers(lab, y0, aa, bb)
        meta.append(dict(image=name, ok=True, title=title(img, y0), top_m=round(aa + bb * y0, 2),
                         total_depth_m=round(aa + bb * y1, 2), m_per_px=round(bb, 4),
                         n_layers=len(L), **info))
        for t, btm, c in L: rows.append(dict(image=name, top_m=t, bottom_m=btm, lithology=c))
    pd.DataFrame(meta).to_csv(f"{O}/r10_log_meta.csv", index=False)
    pd.DataFrame(rows).to_csv(f"{O}/r10_log_layers.csv", index=False)
    m = pd.DataFrame(meta)
    print(f"images {len(m)}, calibrated {int(m.ok.sum())}")
    print(m.drop(columns=["title"]).to_string())


if __name__ == "__main__":
    main()
