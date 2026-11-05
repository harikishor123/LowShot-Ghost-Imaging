#!/usr/bin/env python3
"""Full, read-only audit of the onion root-tip microscopy dataset (STEP 2).

The script never writes to, renames or deletes anything in the dataset folders. It
hashes every original file before and after the run and fails loudly if anything
changed. All outputs go to --output-dir (default: results/dataset_audit).

Checks performed (numbers match the audit brief):
  1  image count                      9  blur / low-quality candidates
  2  file formats (+ JPEG quality)    10  black microscope borders + white page padding
  3  dimensions / aspect ratios       11  burned-in text / watermark / banner / scale bar
  4  RGB vs grayscale                 12  outliers (IsolationForest + per-group robust z)
  5  intensity statistics             13  source groups (filename prefix + appearance clusters)
  6  brightness / contrast            14  YOLO annotation audit
  7  corrupted / unreadable files     15  split structure and distribution
  8  exact / near / flipped duplicates 16  data-leakage risks

Watermark detection is heuristic (no OCR engine is installed). An optional manual
visual-review CSV (--manual-review) is merged into the report so human findings are
recorded alongside the automatic flags.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.fft import dctn

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = ["train", "valid", "val", "test"]
WORK_SIDE = 1024  # analysis resolution (longest side)
GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
RF_NAME = re.compile(r"^(?P<stem>.+?)_(?P<ext>png|jpg|jpeg|bmp|tif|tiff|webp)\.rf\.(?P<hash>[0-9a-f]+)$", re.I)

# IJG standard luminance quantisation table (quality 50), for JPEG quality estimation
IJG_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                     14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                     49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)


# --------------------------------------------------------------------------- utils
def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def robust_z(values: np.ndarray) -> np.ndarray:
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med)) * 1.4826
    # when most values are identical (e.g. a shared page size) MAD collapses to ~0;
    # floor it with a fraction of the std so z-scores stay meaningful
    mad = max(float(mad) if np.isfinite(mad) else 0.0, 0.25 * float(np.nanstd(values)), 1e-6)
    return (values - med) / mad


def summarize(values) -> dict:
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return {}
    return {"n": int(a.size), "min": float(a.min()), "p05": float(np.percentile(a, 5)), "median": float(np.median(a)),
            "mean": float(a.mean()), "p95": float(np.percentile(a, 95)), "max": float(a.max()), "std": float(a.std())}


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def location_tag(cx: float, cy: float) -> str:
    v = "top" if cy < 1 / 3 else ("bottom" if cy > 2 / 3 else "middle")
    h = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
    return f"{v}-{h}"


def jpeg_quality(img: Image.Image):
    q = getattr(img, "quantization", None)
    if not q or 0 not in q:
        return None
    table = np.asarray(q[0], dtype=np.float64)
    if table.size != 64:
        return None
    scale = float(np.mean(table / IJG_LUMA) * 100.0)
    quality = (200.0 - scale) / 2.0 if scale <= 100 else 5000.0 / scale
    return float(np.clip(quality, 1, 100))


# ------------------------------------------------------------------ discovery
def discover(root: Path):
    """Find image folders (YOLO style split/images) and their label folders."""
    images = []
    for split in SPLIT_NAMES:
        d = root / split / "images"
        if d.is_dir():
            for p in sorted(d.iterdir()):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    images.append((split, p))
    # loose images anywhere else (excluding derived/output/venv folders) - reported only
    skip = {".venv", "venv", "results", "data", "__pycache__", ".git"}
    stray = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS
             and not set(p.relative_to(root).parts[:1]) & (skip | set(SPLIT_NAMES))]
    return images, stray


def parse_name(name: str):
    stem = Path(name).stem
    m = RF_NAME.match(stem)
    if m:
        src = m.group("stem")
        return {"source_stem": src, "source_ext": m.group("ext").lower(), "rf_hash": m.group("hash"),
                "group": src.split("_")[0] if "_" in src else src}
    return {"source_stem": stem, "source_ext": Path(name).suffix.lower().lstrip("."), "rf_hash": "",
            "group": stem.split("_")[0] if "_" in stem else "?"}


# ---------------------------------------------------------- per-image analysis
def margin_run(mask_rows: np.ndarray) -> int:
    """Number of leading True entries."""
    idx = np.flatnonzero(~mask_rows)
    return int(idx[0]) if idx.size else int(mask_rows.size)


def rect_padding(mask: np.ndarray, frac: float = 0.985):
    rows = mask.mean(axis=1) >= frac
    cols = mask.mean(axis=0) >= frac
    top, bottom = margin_run(rows), margin_run(rows[::-1])
    left, right = margin_run(cols), margin_run(cols[::-1])
    return top, bottom, left, right


def hue_name(deg: float) -> str:
    if deg is None or not np.isfinite(deg):
        return "none"
    if deg < 25 or deg >= 330:
        return "red/pink (orcein/carmine-like)"
    if deg < 60:
        return "orange/brown"
    if deg < 170:
        return "green/yellow"
    if deg < 255:
        return "blue (methylene/toluidine-like)"
    return "purple/magenta (Giemsa/H&E/Feulgen-like)"


def dhash_bits(gray: np.ndarray) -> np.ndarray:
    im = Image.fromarray((gray * 255).astype(np.uint8)).resize((9, 8), Image.Resampling.BOX)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).ravel()


def phash_bits(gray: np.ndarray) -> np.ndarray:
    im = Image.fromarray((gray * 255).astype(np.uint8)).resize((32, 32), Image.Resampling.BOX)
    d = dctn(np.asarray(im, dtype=np.float64), norm="ortho")[:8, :8].ravel()
    return d > np.median(d[1:])


def dihedral(a: np.ndarray):
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        out.extend([r, r[:, ::-1]])
    return out


def transitions_per_height(patch: np.ndarray) -> float:
    """Mean number of off->on transitions per row, per glyph-height of width (text ~1.5-4)."""
    if patch.size == 0:
        return 0.0
    t = (np.diff(patch.astype(np.int8), axis=1) == 1).sum(axis=1)
    rows = t[patch.any(axis=1)]
    return float(rows.mean() / max(patch.shape[1] / patch.shape[0], 1e-6)) if rows.size else 0.0


def text_like_lines(stroke: np.ndarray, gray: np.ndarray):
    """Group glyph- or word-sized connected components into horizontal text lines.

    Bold captions merge letters into word blobs, so word-sized components are accepted
    and a line is confirmed by its horizontal stroke-transition density.
    """
    lab, n = ndi.label(stroke)
    if n == 0:
        return []
    objs = ndi.find_objects(lab)
    areas = ndi.sum(stroke, lab, index=np.arange(1, n + 1))
    chars = []
    for i, sl in enumerate(objs):
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        a = areas[i]
        if 4 <= h <= 40 and 1 <= w <= 14 * h and a >= 5 and 0.10 <= a / (h * w) <= 0.95:
            chars.append((sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))
    if len(chars) < 2:
        return []
    chars.sort(key=lambda c: c[2])
    parent = list(range(len(chars)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(chars):
        ha = a[1] - a[0]
        for j in range(i + 1, len(chars)):
            b = chars[j]
            hb = b[1] - b[0]
            gap = b[2] - a[3]
            if gap > 1.3 * max(ha, hb):
                if b[2] - a[3] > 60:
                    break
                continue
            overlap = min(a[1], b[1]) - max(a[0], b[0])
            if overlap >= 0.55 * min(ha, hb) and max(ha, hb) <= 2.2 * min(ha, hb):
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(len(chars)):
        groups[find(i)].append(chars[i])
    lines = []
    for g in groups.values():
        hs = np.array([c[1] - c[0] for c in g], dtype=float)
        y0, y1 = min(c[0] for c in g), max(c[1] for c in g)
        x0, x1 = min(c[2] for c in g), max(c[3] for c in g)
        mh = float(np.median(hs))
        if len(g) < 2 and (x1 - x0) < 4 * mh:
            continue
        if hs.std() / max(hs.mean(), 1) > 0.45 or (x1 - x0) < 2.5 * mh or (y1 - y0) > 2.2 * mh:
            continue
        patch = stroke[y0:y1, x0:x1]
        fill = patch.mean()
        # bold glyphs fuse in the top-hat mask, so also binarise the raw patch
        gp = gray[y0:y1, x0:x1]
        lo, hi = np.percentile(gp, [5, 95])
        raw = gp > (lo + hi) / 2 if hi - lo > 0.2 else np.zeros_like(gp, bool)
        if raw.mean() > 0.5:
            raw = ~raw  # make glyphs the foreground (dark text on light)
        tph = max(transitions_per_height(patch), transitions_per_height(raw))
        if not (0.12 <= fill <= 0.9 and tph >= 1.0):
            continue
        lines.append({"y0": y0, "y1": y1, "x0": x0, "x1": x1, "n_chars": len(g), "tph": tph})
    return lines


def analyse_image(path: Path) -> dict:
    r: dict = {"status": "ok", "error": ""}
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            r["format"] = im.format
            r["mode"] = im.mode
            r["width"], r["height"] = im.size
            r["dpi"] = str(im.info.get("dpi", ""))
            r["jpeg_quality_est"] = jpeg_quality(im)
            r["has_exif"] = bool(im.info.get("exif"))
            im.load()
            full = np.asarray(im.convert("RGB"))
    except Exception as exc:  # corrupted / unreadable
        r["status"] = "corrupted"
        r["error"] = f"{type(exc).__name__}: {exc}"
        return r

    H, W = full.shape[:2]
    r["aspect_ratio"] = W / H
    r["orientation"] = "square" if abs(W / H - 1) < 0.05 else ("landscape" if W > H else "portrait")
    r["megapixels"] = W * H / 1e6
    g_full = full.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    r["int_min"], r["int_max"] = float(g_full.min()), float(g_full.max())
    r["int_mean"], r["int_std"] = float(g_full.mean()), float(g_full.std())
    r["int_p01"], r["int_p99"] = (float(v) for v in np.percentile(g_full, [1, 99]))
    for c, name in enumerate("rgb"):
        r[f"full_{name}_mean"] = float(full[..., c].mean())

    # working copy
    s = WORK_SIDE / max(H, W)
    if s < 1:
        work = np.asarray(Image.fromarray(full).resize((max(1, round(W * s)), max(1, round(H * s))),
                                                       Image.Resampling.BOX))
    else:
        work = full
    rgb = work.astype(np.float32) / 255.0
    h, w = rgb.shape[:2]
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    sat = np.where(mx > 1e-3, (mx - mn) / np.maximum(mx, 1e-3), 0.0)

    # --- white page padding (rectangular, uniform) ---------------------------
    white = (gray > 0.93) & (sat < 0.08)
    wt, wb, wl, wr = rect_padding(white)
    if wt + wb >= h or wl + wr >= w:
        wt = wb = wl = wr = 0
    r["white_pad_top"], r["white_pad_bottom"], r["white_pad_left"], r["white_pad_right"] = (
        round(v / s if s < 1 else v) for v in (wt, wb, wl, wr))
    inner = np.zeros((h, w), bool)
    inner[wt:h - wb, wl:w - wr] = True
    r["white_pad_frac"] = float(1 - inner.mean())

    # --- black border (rectangular bars + edge-connected dark, e.g. circular FOV)
    dark = (gray < 0.11)
    sub_dark = dark[wt:h - wb, wl:w - wr]
    bt, bb, bl, br = rect_padding(sub_dark, 0.97)
    r["black_bar_frac"] = float(1 - (max(0, sub_dark.shape[0] - bt - bb) * max(0, sub_dark.shape[1] - bl - br))
                                / max(1, sub_dark.size)) * inner.mean()
    lab, n = ndi.label(dark & inner)
    edge_ids = set()
    if n:
        ib = np.zeros_like(inner)
        ib[wt:h - wb, wl:w - wr] = True
        ring = ib & ~ndi.binary_erosion(ib, iterations=2)
        edge_ids = set(np.unique(lab[ring & (lab > 0)]).tolist())
    border_dark = np.isin(lab, list(edge_ids)) if edge_ids else np.zeros_like(dark)
    # a dark component that is most of the image is "very dark image", not a border
    r["dark_border_frac"] = float(border_dark.mean())
    content = inner & ~border_dark
    content = ndi.binary_opening(content, iterations=2)
    r["usable_frac"] = float(content.mean())
    # circular field of view: corners of inner box dark, center bright
    ys, xs = np.nonzero(inner)
    if ys.size:
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        ch, cw = max(2, (y1 - y0) // 10), max(2, (x1 - x0) // 10)
        corners = [border_dark[y0:y0 + ch, x0:x0 + cw], border_dark[y0:y0 + ch, x1 - cw:x1],
                   border_dark[y1 - ch:y1, x0:x0 + cw], border_dark[y1 - ch:y1, x1 - cw:x1]]
        corner_dark = float(np.mean([c.mean() for c in corners]))
        center = content[(y0 + y1) // 2 - ch:(y0 + y1) // 2 + ch, (x0 + x1) // 2 - cw:(x0 + x1) // 2 + cw]
        r["corner_dark_frac"] = corner_dark
        r["circular_fov"] = bool(corner_dark > 0.75 and center.mean() > 0.9)
    else:
        r["corner_dark_frac"], r["circular_fov"] = 0.0, False
    cy, cx = np.nonzero(content)
    if cy.size == 0:
        content = inner.copy()
        cy, cx = np.nonzero(content)
    by0, by1, bx0, bx1 = cy.min(), cy.max() + 1, cx.min(), cx.max() + 1
    inv = (1 / s) if s < 1 else 1
    r["content_bbox_xyxy"] = f"{round(bx0 * inv)},{round(by0 * inv)},{round(bx1 * inv)},{round(by1 * inv)}"
    r["content_w"], r["content_h"] = round((bx1 - bx0) * inv), round((by1 - by0) * inv)

    # --- content intensity / colour ----------------------------------------
    cg = gray[content]
    crgb = rgb[content]
    r["content_mean"], r["content_std"] = float(cg.mean()), float(cg.std())
    r["content_p01"], r["content_p99"] = (float(v) for v in np.percentile(cg, [1, 99]))
    r["content_dynamic_range"] = r["content_p99"] - r["content_p01"]
    r["clip_high_frac"] = float((crgb.max(axis=1) >= 0.985).mean())
    r["clip_low_frac"] = float((crgb.max(axis=1) <= 0.02).mean())
    for c, name in enumerate("rgb"):
        r[f"{name}_mean"] = float(crgb[:, c].mean())
    r["chroma_diff"] = float(np.mean(np.abs(crgb[:, 0] - crgb[:, 1]) + np.abs(crgb[:, 1] - crgb[:, 2])) / 2)
    r["is_grayscale"] = bool(r["mode"] in ("L", "LA", "I", "I;16", "F", "1") or r["chroma_diff"] < 0.01)
    r["sat_mean"] = float(sat[content].mean())
    rg = crgb[:, 0] - crgb[:, 1]
    yb = 0.5 * (crgb[:, 0] + crgb[:, 1]) - crgb[:, 2]
    r["colorfulness"] = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    stained = content & (sat > 0.25) & (gray > 0.08)
    if stained.sum() > 50:
        px = rgb[stained]
        mxp, mnp = px.max(1), px.min(1)
        d = np.maximum(mxp - mnp, 1e-6)
        hue = np.where(mxp == px[:, 0], ((px[:, 1] - px[:, 2]) / d) % 6,
                       np.where(mxp == px[:, 1], (px[:, 2] - px[:, 0]) / d + 2, (px[:, 0] - px[:, 1]) / d + 4)) * 60
        ang = np.deg2rad(hue)
        r["stain_hue_deg"] = float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360)
        r["stained_frac"] = float(stained.sum() / max(1, content.sum()))
    else:
        r["stain_hue_deg"], r["stained_frac"] = float("nan"), 0.0
    r["stain_family"] = hue_name(r["stain_hue_deg"])

    # --- sharpness on content crop at normalised scale ------------------------
    crop = gray[by0:by1, bx0:bx1]
    cmask = content[by0:by1, bx0:bx1]
    ns = 512 / max(crop.shape)
    cimg = Image.fromarray((crop * 255).astype(np.uint8)).resize(
        (max(8, round(crop.shape[1] * ns)), max(8, round(crop.shape[0] * ns))), Image.Resampling.BOX)
    cm = np.asarray(Image.fromarray(cmask.astype(np.uint8) * 255).resize(cimg.size, Image.Resampling.NEAREST)) > 0
    cm = ndi.binary_erosion(cm, iterations=4)
    c = np.asarray(cimg, dtype=np.float32) / 255.0
    lap = ndi.laplace(c)
    sx, sy = ndi.sobel(c, 1), ndi.sobel(c, 0)
    if cm.sum() > 100:
        r["lap_var"] = float(lap[cm].var() * 1e4)
        r["tenengrad"] = float(np.mean((sx ** 2 + sy ** 2)[cm]) * 1e2)
    else:
        r["lap_var"], r["tenengrad"] = float(lap.var() * 1e4), float(np.mean(sx ** 2 + sy ** 2) * 1e2)
    # contrast-normalised sharpness (blur independent of stain contrast)
    r["sharpness_norm"] = float(r["tenengrad"] / max(r["content_std"] ** 2 * 1e2, 1e-4))

    # --- overlays: text / watermark / banner / scale bar / drawn lines --------
    wth = gray - ndi.grey_opening(gray, size=(7, 7))
    bth = ndi.grey_closing(gray, size=(7, 7)) - gray
    achrom = sat < 0.18
    stroke = ((wth > 0.22) | (bth > 0.22)) & achrom
    lines = text_like_lines(stroke, gray)
    near_content = ndi.binary_erosion(content, iterations=3)
    outside = ~inner | border_dark
    tl_content, tl_border, locs = 0, 0, []
    for ln in lines:
        ccy, ccx = (ln["y0"] + ln["y1"]) // 2, (ln["x0"] + ln["x1"]) // 2
        m = max(4, (ln["y1"] - ln["y0"]) // 2)
        ring = outside[max(0, ln["y0"] - m):ln["y1"] + m, max(0, ln["x0"] - m):ln["x1"] + m]
        inside = bool(near_content[ccy, ccx]) and ring.mean() < 0.35
        tl_content += inside
        tl_border += not inside
        locs.append(f"{location_tag(ccx / w, ccy / h)}{'' if inside else '(border)'}")
    r["text_lines_in_content"] = tl_content
    r["text_lines_in_border"] = tl_border
    r["text_locations"] = ";".join(sorted(set(locs)))
    # gray semi-transparent banner (e.g. Getty credit box)
    banner = (sat < 0.10) & (gray > 0.28) & (gray < 0.80) & content
    banner = ndi.binary_closing(ndi.binary_opening(banner, iterations=2), iterations=3)
    lb, nb = ndi.label(banner)
    n_banner, banner_locs = 0, []
    if nb:
        for i, sl in enumerate(ndi.find_objects(lb)):
            bh, bw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            area = (lb[sl] == i + 1).sum()
            if area > 0.004 * h * w and bw >= 2.2 * bh and area / (bh * bw) > 0.85 and bh >= 10:
                n_banner += 1
                banner_locs.append(location_tag((sl[1].start + bw / 2) / w, (sl[0].start + bh / 2) / h))
    r["gray_banner_count"] = n_banner
    r["gray_banner_locations"] = ";".join(banner_locs)
    # thin straight horizontal bars (scale bars) and long thin curves (hand-drawn lines)
    thin_dark = (bth > 0.20) & (gray < 0.40) & near_content  # ink lines may be coloured (e.g. navy)
    thin_any = ((bth > 0.25) | (wth > 0.25)) & achrom & near_content
    lt, nt = ndi.label(thin_any)
    n_scale = 0
    if nt:
        for sl in ndi.find_objects(lt):
            bh, bw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            if bh <= 6 and bw >= 25 and bw / bh >= 8:
                n_scale += 1
    r["scale_bar_candidates"] = n_scale
    ld, nd = ndi.label(thin_dark)
    n_draw = 0
    if nd:
        areas = ndi.sum(thin_dark, ld, index=np.arange(1, nd + 1))
        for i, sl in enumerate(ndi.find_objects(ld)):
            bh, bw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            if (max(bh, bw) >= 0.15 * max(h, w) and min(bh, bw) >= 4 and areas[i] >= 150
                    and areas[i] / (bh * bw) < 0.12):
                n_draw += 1
    r["drawn_line_candidates"] = n_draw

    # --- hashes / thumbnails for duplicate search ----------------------------
    # fill non-specimen pixels with the specimen mean so borders don't dominate hashes
    cgray = np.where(content, gray, float(cg.mean()))[by0:by1, bx0:bx1]
    r["_dhash"] = dhash_bits(cgray)
    r["_phash_variants"] = np.stack([phash_bits(v) for v in dihedral(cgray)])
    r["_dhash_full"] = dhash_bits(gray)
    th = np.asarray(Image.fromarray((cgray * 255).astype(np.uint8)).resize((32, 32), Image.Resampling.BOX),
                    dtype=np.float32).ravel()
    r["_thumb"] = (th - th.mean()) / (th.std() + 1e-6)
    r["_thumb_variants"] = np.stack([v.ravel() for v in dihedral(r["_thumb"].reshape(32, 32))])
    r["_pixel_md5"] = hashlib.md5(full.tobytes()).hexdigest()
    r["_content_mask_small"] = content  # used for label/padding overlap (working resolution)
    r["_scale"] = s if s < 1 else 1.0
    return r


# ------------------------------------------------------------------- labels
def audit_labels(img_path: Path, split_dir: Path, rec: dict, nc: int):
    lbl = split_dir / "labels" / (img_path.stem + ".txt")
    out = {"label_file": str(lbl.relative_to(split_dir.parent)) if lbl.exists() else "", "label_exists": lbl.exists(),
           "n_objects": 0, "n_bbox_rows": 0, "n_polygon_rows": 0, "n_invalid_rows": 0, "n_out_of_range": 0,
           "class_counts": {}, "boxes_in_padding": 0, "tiny_boxes": 0, "dup_boxes": 0, "_boxes": []}
    if not lbl.exists():
        return out
    boxes = []
    for line in lbl.read_text(encoding="utf-8", errors="replace").splitlines():
        t = line.split()
        if not t:
            continue
        try:
            cls = int(float(t[0]))
            vals = [float(v) for v in t[1:]]
        except ValueError:
            out["n_invalid_rows"] += 1
            continue
        if len(vals) == 4:
            out["n_bbox_rows"] += 1
            xc, yc, bw, bh = vals
            x0, y0, x1, y1 = xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2
            coords = vals
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            out["n_polygon_rows"] += 1
            xs, ys = vals[0::2], vals[1::2]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            coords = vals
        else:
            out["n_invalid_rows"] += 1
            continue
        if cls < 0 or cls >= nc:
            out["n_invalid_rows"] += 1
        if any(v < -1e-6 or v > 1 + 1e-6 for v in coords):
            out["n_out_of_range"] += 1
        out["class_counts"][cls] = out["class_counts"].get(cls, 0) + 1
        boxes.append((cls, x0, y0, x1, y1))
    out["n_objects"] = len(boxes)
    out["_boxes"] = boxes
    if rec.get("status") == "ok":
        W, H = rec["width"], rec["height"]
        cmask = rec["_content_mask_small"]
        mh, mw = cmask.shape
        cw, chh = rec["content_w"], rec["content_h"]
        for i, (cls, x0, y0, x1, y1) in enumerate(boxes):
            px, py = int(np.clip((x0 + x1) / 2 * mw, 0, mw - 1)), int(np.clip((y0 + y1) / 2 * mh, 0, mh - 1))
            if not cmask[py, px]:
                out["boxes_in_padding"] += 1
            # tiny relative to content after a 640px resize of the content region
            if min((x1 - x0) * W, (y1 - y0) * H) * 640 / max(cw, chh, 1) < 8:
                out["tiny_boxes"] += 1
            for (_, a0, b0, a1, b1) in boxes[i + 1:]:
                iw, ih = max(0, min(x1, a1) - max(x0, a0)), max(0, min(y1, b1) - max(y0, b0))
                inter = iw * ih
                union = (x1 - x0) * (y1 - y0) + (a1 - a0) * (b1 - b0) - inter
                if union > 0 and inter / union > 0.9:
                    out["dup_boxes"] += 1
        out["box_w_px"] = [(b[3] - b[1]) * W for b in boxes]
        out["box_h_px"] = [(b[4] - b[2]) * H for b in boxes]
    return out


def analyse_job(job):
    k, split, path, root, md5, nc = job
    p, root = Path(path), Path(root)
    r = {"idx": k, "split": split, "file": p.name, "path": str(p), "rel_path": str(p.relative_to(root)),
         "size_bytes": p.stat().st_size, "md5": md5}
    r.update(parse_name(p.name))
    r.update(analyse_image(p))
    r.update(audit_labels(p, root / split, r, nc))
    return r


# ---------------------------------------------------------------- duplicates
def find_duplicates(recs):
    ok = [i for i, r in enumerate(recs) if r["status"] == "ok"]
    n = len(ok)
    D = np.stack([recs[i]["_dhash"] for i in ok])
    DF = np.stack([recs[i]["_dhash_full"] for i in ok])
    P = np.stack([recs[i]["_phash_variants"] for i in ok])  # (n, 8, 64)
    T = np.stack([recs[i]["_thumb"] for i in ok])
    ncc = (T @ T.T) / T.shape[1]
    dd = (D[:, None, :] != D[None, :, :]).sum(-1)
    ddf = (DF[:, None, :] != DF[None, :, :]).sum(-1)
    canon = P[:, 0, :]
    pd = np.full((n, n), 64, dtype=np.int16)
    best_t = np.zeros((n, n), dtype=np.int8)
    for t in range(8):
        dt = (P[:, t, None, :] != canon[None, :, :]).sum(-1)
        better = dt < pd
        pd = np.where(better, dt, pd)
        best_t = np.where(better, t, best_t)
    tnames = ["identity", "hflip", "rot90", "rot90+flip", "rot180", "rot180+flip", "rot270", "rot270+flip"]
    pairs = []
    for a in range(n):
        for b in range(a + 1, n):
            i, j = ok[a], ok[b]
            exact_file = recs[i]["md5"] == recs[j]["md5"]
            exact_px = recs[i]["_pixel_md5"] == recs[j]["_pixel_md5"]
            p_id = int(min(pd[a, b], pd[b, a]))
            kind = None
            t = int(best_t[a, b])
            ncc_t = float(recs[i]["_thumb_variants"][t] @ recs[j]["_thumb"]) / T.shape[1]
            if exact_file or exact_px:
                kind = "exact"
            elif (p_id <= 6 and dd[a, b] <= 10 and ncc[a, b] >= 0.80) or (p_id <= 10 and ncc[a, b] >= 0.95):
                kind = "near-duplicate"
            elif p_id <= 6 and t != 0 and ncc_t >= 0.90:
                kind = "flipped/rotated copy"
            elif (p_id <= 10 and ncc[a, b] >= 0.80) or ncc[a, b] >= 0.90:
                kind = "similar (review)"
            if kind:
                pairs.append({"a": i, "b": j, "kind": kind, "phash_dist": p_id, "dhash_dist": int(dd[a, b]),
                              "dhash_full_dist": int(ddf[a, b]), "ncc32": float(ncc[a, b]),
                              "transform": tnames[int(best_t[a, b])]})
    # clusters over strong links (exact/near/flipped)
    parent = {i: i for i in ok}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        if p["kind"] != "similar (review)":
            parent[find(p["a"])] = find(p["b"])
    clusters = defaultdict(list)
    for i in ok:
        clusters[find(i)].append(i)
    clusters = [c for c in clusters.values() if len(c) > 1]
    return pairs, clusters


# ---------------------------------------------------------- contact sheets
def tile_image(path: Path, size: int, boxes=None, names=None):
    with Image.open(path) as im:
        im = im.convert("RGB")
        W, H = im.size
        if boxes:
            d = ImageDraw.Draw(im)
            lw = max(2, int(max(W, H) / 400))
            for cls, x0, y0, x1, y1 in boxes:
                col = (255, 40, 40) if cls == 0 else (40, 220, 255)
                d.rectangle([x0 * W, y0 * H, x1 * W, y1 * H], outline=col, width=lw)
        im.thumbnail((size, size), Image.Resampling.LANCZOS)
        return im


def contact_sheet(items, out_path: Path, title: str, cols: int = 6, size: int = 260, note: str = ""):
    """items: list of (path, [caption lines], border_color|None, boxes|None)."""
    if not items:
        return None
    f_cap, f_title = font(13), font(20)
    cap_h = 18 * max(len(it[1]) for it in items) + 6
    rows = math.ceil(len(items) / cols)
    head = 56 if note else 34
    pad = 6
    sheet = Image.new("RGB", (cols * (size + pad) + pad, head + rows * (size + cap_h + pad) + pad), (250, 250, 248))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 6), title, fill=(15, 15, 15), font=f_title)
    if note:
        d.text((pad, 32), note, fill=(80, 80, 78), font=f_cap)
    for k, (p, caps, border, boxes) in enumerate(items):
        x = pad + (k % cols) * (size + pad)
        y = head + (k // cols) * (size + cap_h + pad)
        d.rectangle([x, y, x + size - 1, y + size - 1], fill=(225, 225, 222))
        try:
            t = tile_image(p, size, boxes)
            sheet.paste(t, (x + (size - t.width) // 2, y + (size - t.height) // 2))
        except Exception as exc:
            d.text((x + 6, y + size // 2), f"UNREADABLE: {exc}"[:40], fill=(200, 0, 0), font=f_cap)
        if border:
            d.rectangle([x, y, x + size - 1, y + size - 1], outline=border, width=4)
        for li, line in enumerate(caps):
            d.text((x + 2, y + size + 3 + 18 * li), line[:40], fill=(20, 20, 20) if li == 0 else (90, 90, 88),
                   font=f_cap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, optimize=True)
    return out_path


# ------------------------------------------------------------------- plots
def make_plots(recs, groups, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("matplotlib unavailable, skipping plots:", exc)
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = [r for r in recs if r["status"] == "ok"]
    cmap = {g: GROUP_COLORS[i % len(GROUP_COLORS)] for i, g in enumerate(groups)}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#b5b4ae", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.titleweight": "bold",
                         "axes.titlesize": 10, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    metrics = [("content_mean", "Brightness (content mean, 0-1)", False),
               ("content_std", "Contrast (content std, 0-1)", False),
               ("lap_var", "Sharpness (Laplacian var, log)", True),
               ("usable_frac", "Usable specimen area (fraction)", False),
               ("dark_border_frac", "Black border (fraction)", False),
               ("white_pad_frac", "White page padding (fraction)", False)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, (key, label, logy) in zip(axes.ravel(), metrics):
        for gi, g in enumerate(groups):
            v = np.array([r[key] for r in ok if r["group"] == g], dtype=float)
            if v.size == 0:
                continue
            xj = gi + rng.uniform(-0.28, 0.28, v.size)
            ax.scatter(xj, v, s=9, color=cmap[g], alpha=0.65, linewidths=0)
            med = np.median(v)
            ax.plot([gi - 0.34, gi + 0.34], [med, med], color="#0b0b0b", lw=2, solid_capstyle="round")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{g}\n(n={sum(r['group'] == g for r in ok)})" for g in groups])
        if logy:
            ax.set_yscale("log")
        ax.set_title(label, loc="left")
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Per-source-group distributions (dot = image, black bar = median)", x=0.01, ha="left",
                 fontweight="bold")
    p1 = out_dir / "group_distributions.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
    for g in groups:
        ww = [r["width"] for r in ok if r["group"] == g]
        hh = [r["height"] for r in ok if r["group"] == g]
        ax.scatter(ww, hh, s=22, color=cmap[g], alpha=0.75, label=f"{g} (n={len(ww)})", linewidths=0.5,
                   edgecolors="#fcfcfb")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Image dimensions by source group", loc="left")
    ax.grid(color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, title="Group")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    p2 = out_dir / "dimensions_by_group.png"
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    return [p1, p2]


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--output-dir", default="results/dataset_audit")
    ap.add_argument("--manual-review", default="results/dataset_audit/manual_visual_review.csv",
                    help="optional CSV: filename (or Roboflow source key like E_9_jpg),category,finding")
    ap.add_argument("--catalog-per-sheet", type=int, default=40)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (__import__("os").cpu_count() or 2) - 1)))
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    out = Path(args.output_dir)
    out = (root / out).resolve() if not out.is_absolute() else out.resolve()
    for split in SPLIT_NAMES:
        if (root / split).exists() and (root / split).resolve() in out.parents or out == (root / split).resolve():
            sys.exit(f"Refusing to write output inside dataset folder {root / split}")
    out.mkdir(parents=True, exist_ok=True)
    sheets_dir, plots_dir = out / "contact_sheets", out / "plots"

    images, stray = discover(root)
    print(f"Found {len(images)} images in split folders; {len(stray)} stray images elsewhere")

    # data.yaml
    yaml_info = {}
    yp = root / "data.yaml"
    if yp.exists():
        txt = yp.read_text(encoding="utf-8", errors="replace")
        yaml_info["raw"] = txt
        m = re.search(r"names:\s*\[(.*?)\]", txt)
        yaml_info["names"] = [x.strip().strip("'\"") for x in m.group(1).split(",")] if m else []
        m = re.search(r"^nc:\s*(\d+)", txt, re.M)
        yaml_info["nc"] = int(m.group(1)) if m else len(yaml_info["names"])
        for k in ("train", "val", "test"):
            m = re.search(rf"^{k}:\s*(\S+)", txt, re.M)
            if m:
                yaml_info[k] = m.group(1)
                yaml_info[f"{k}_exists"] = (yp.parent / m.group(1)).resolve().exists()
    nc = yaml_info.get("nc", 1000)
    class_names = yaml_info.get("names", [])

    # integrity manifest BEFORE
    all_label_files = sorted(p for s in SPLIT_NAMES for p in (root / s / "labels").glob("*.txt")) \
        if True else []
    before = {str(p): md5_file(p) for _, p in images}
    before.update({str(p): md5_file(p) for p in all_label_files})

    jobs = [(k, split, str(p), str(root), before[str(p)], nc) for k, (split, p) in enumerate(images)]
    recs = []
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for k, r in enumerate(ex.map(analyse_job, jobs, chunksize=4)):
                recs.append(r)
                if (k + 1) % 100 == 0:
                    print(f"  analysed {k + 1}/{len(images)}")
    else:
        for k, job in enumerate(jobs):
            recs.append(analyse_job(job))
            if (k + 1) % 100 == 0:
                print(f"  analysed {k + 1}/{len(images)}")
    ok = [r for r in recs if r["status"] == "ok"]
    groups = sorted({r["group"] for r in recs})

    # orphan labels
    orphan_labels = []
    for s in SPLIT_NAMES:
        ld, idir = root / s / "labels", root / s / "images"
        if ld.is_dir():
            stems = {p.stem for p in idir.glob("*")} if idir.is_dir() else set()
            orphan_labels += [str(p.relative_to(root)) for p in ld.glob("*.txt") if p.stem not in stems]

    # ---- blur / quality / exposure flags (global + within group) -----------
    lv = np.log10(np.array([r["lap_var"] for r in ok]) + 1e-6)
    sn = np.log10(np.array([r["sharpness_norm"] for r in ok]) + 1e-6)
    p_lv = np.percentile(lv, 5)
    for gname in groups:
        idx = [i for i, r in enumerate(ok) if r["group"] == gname]
        if len(idx) >= 5:
            z1 = robust_z(lv[idx])
            z2 = robust_z(sn[idx])
            for j, i in enumerate(idx):
                ok[i]["sharp_z_group"] = float(z1[j])
                ok[i]["sharpnorm_z_group"] = float(z2[j])
        else:
            for i in idx:
                ok[i]["sharp_z_group"] = ok[i]["sharpnorm_z_group"] = 0.0
    for i, r in enumerate(ok):
        fl = []
        if (r["sharp_z_group"] < -2.0 and r["sharpnorm_z_group"] < -1.5) or lv[i] < p_lv - 0.3:
            fl.append("blur")
        if r["content_mean"] < 0.25:
            fl.append("underexposed")
        if r["content_mean"] > 0.85 or r["clip_high_frac"] > 0.10:
            fl.append("overexposed/washed-out")
        if r["content_std"] < 0.06:
            fl.append("low-contrast")
        if r.get("jpeg_quality_est") is not None and r["jpeg_quality_est"] < 60:
            fl.append("low JPEG quality")
        if r["usable_frac"] < 0.35:
            fl.append("small usable area")
        if r["dark_border_frac"] >= 0.20:
            fl.append("large black border")
        if r["white_pad_frac"] >= 0.20:
            fl.append("large white padding")
        ov = []
        if r["gray_banner_count"]:
            ov.append("gray banner (watermark-like)")
        if r["text_lines_in_content"]:
            ov.append("text over specimen")
        if r["text_lines_in_border"]:
            ov.append("text in border/margin")
        if r["scale_bar_candidates"]:
            ov.append("scale-bar-like line")
        if r["drawn_line_candidates"]:
            ov.append("drawn line/annotation")
        r["quality_flags"] = ";".join(fl)
        r["overlay_flags"] = ";".join(ov)
        r["watermark_suspect"] = bool(r["gray_banner_count"] or r["text_lines_in_content"])

    # ---- manual visual review --------------------------------------------
    manual = {}
    if args.manual_review and Path(args.manual_review).exists():
        with open(args.manual_review, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                manual.setdefault(row["filename"].strip(), []).append((row["category"].strip(), row["finding"].strip()))
    for r in recs:
        # rows may give the full filename or the Roboflow source key ("E_9_jpg")
        mf = manual.get(r["file"], []) + manual.get(r["file"].split(".rf.")[0], [])
        r["manual_categories"] = ";".join(sorted({c for c, _ in mf}))
        r["manual_findings"] = " | ".join(f for _, f in mf)
        r["watermark_confirmed"] = "WATERMARK" in r["manual_categories"]
    keys = {r["file"] for r in recs} | {r["file"].split(".rf.")[0] for r in recs}
    unmatched = [k for k in manual if k not in keys]
    if unmatched:
        print("WARNING: manual-review rows that match no image:", unmatched)

    # ---- duplicates ----------------------------------------------------------
    print("Searching duplicates ...")
    pairs, clusters = find_duplicates(recs)
    for p in pairs:
        p["file_a"], p["file_b"] = recs[p["a"]]["file"], recs[p["b"]]["file"]
        p["split_a"], p["split_b"] = recs[p["a"]]["split"], recs[p["b"]]["split"]
        p["cross_split"] = p["split_a"] != p["split_b"]
    stem_counts = Counter((r["source_stem"]) for r in recs)
    stem_splits = defaultdict(set)
    for r in recs:
        stem_splits[r["source_stem"]].add(r["split"])

    # ---- outliers ------------------------------------------------------------
    feat_keys = ["aspect_ratio", "white_pad_frac", "dark_border_frac", "usable_frac", "content_mean", "content_std",
                 "r_mean", "g_mean", "b_mean", "sat_mean", "colorfulness", "lap_var", "tenengrad",
                 "clip_high_frac", "content_dynamic_range"]
    X = np.array([[r[k] for k in feat_keys] for r in ok], dtype=float)
    X[:, feat_keys.index("aspect_ratio")] = np.log(X[:, feat_keys.index("aspect_ratio")])
    X[:, feat_keys.index("lap_var")] = np.log10(X[:, feat_keys.index("lap_var")] + 1e-6)
    X[:, feat_keys.index("tenengrad")] = np.log10(X[:, feat_keys.index("tenengrad")] + 1e-6)
    Xz = np.column_stack([robust_z(X[:, j]) for j in range(X.shape[1])])
    Xz = np.clip(np.nan_to_num(Xz), -15, 15)
    cluster_info = {}
    try:
        from sklearn.cluster import KMeans
        from sklearn.ensemble import IsolationForest
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        iso = IsolationForest(n_estimators=400, contamination=0.03, random_state=0).fit(Xz)
        scores = -iso.score_samples(Xz)
        is_out = iso.predict(Xz) == -1
        Xs = StandardScaler().fit_transform(Xz)
        best = None
        for k in range(2, 9):
            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
            sil = silhouette_score(Xs, km.labels_)
            cluster_info[k] = float(sil)
            if best is None or sil > best[0]:
                best = (sil, k, km.labels_)
        for r, lab in zip(ok, best[2]):
            r["appearance_cluster"] = int(lab)
        cluster_info = {"silhouette_by_k": cluster_info, "best_k": best[1]}
    except Exception as exc:  # sklearn missing
        print("sklearn unavailable, falling back to robust-z outliers:", exc)
        scores = np.abs(Xz).max(axis=1)
        is_out = scores > 6
    for i, r in enumerate(ok):
        r["outlier_score"] = float(scores[i])
        r["outlier_global"] = bool(is_out[i])
        top = np.argsort(-np.abs(Xz[i]))[:3]
        r["outlier_reason"] = ", ".join(f"{feat_keys[j]} z={Xz[i, j]:+.1f}" for j in top)
    # within-group outliers
    # appearance only: geometry (page size / crop) subgroups are reported in section 13 instead
    gkeys = ["content_mean", "content_std", "colorfulness", "lap_var", "sat_mean"]
    for gname in groups:
        gi = [i for i, r in enumerate(ok) if r["group"] == gname]
        if len(gi) < 8:
            for i in gi:
                ok[i]["outlier_in_group"] = False
                ok[i]["outlier_in_group_reason"] = ""
            continue
        Z = np.column_stack([robust_z(np.log10(np.array([ok[i][k] for i in gi]) + 1e-6)) for k in gkeys])
        Z = np.nan_to_num(Z)
        for j, i in enumerate(gi):
            bad = [f"{gkeys[c]} z={Z[j, c]:+.1f}" for c in range(len(gkeys)) if abs(Z[j, c]) > 4.0]
            ok[i]["outlier_in_group"] = bool(bad)
            ok[i]["outlier_in_group_reason"] = ", ".join(bad)

    # ---- source-group signatures -------------------------------------------
    group_summary = {}
    for gname in groups:
        g = [r for r in ok if r["group"] == gname]
        if not g:
            continue
        group_summary[gname] = {
            "n": len(g),
            "splits": dict(Counter(r["split"] for r in g)),
            "source_ext": dict(Counter(r["source_ext"] for r in g)),
            "top_sizes": Counter(f"{r['width']}x{r['height']}" for r in g).most_common(4),
            "n_distinct_sizes": len({(r["width"], r["height"]) for r in g}),
            "layout_subgroups": dict(Counter(
                ("A4 page" if (r["width"], r["height"]) in {(1653, 2338), (1653, 2339), (2339, 1653),
                                                            (2339, 1651), (2339, 1650)}
                 else "square crop ~1:1" if r["orientation"] == "square" else "other crop")
                for r in g)),
            "orientation": dict(Counter(r["orientation"] for r in g)),
            "circular_fov": sum(r["circular_fov"] for r in g),
            "median_white_pad": float(np.median([r["white_pad_frac"] for r in g])),
            "median_dark_border": float(np.median([r["dark_border_frac"] for r in g])),
            "median_usable": float(np.median([r["usable_frac"] for r in g])),
            "median_content_w": float(np.median([r["content_w"] for r in g])),
            "median_content_h": float(np.median([r["content_h"] for r in g])),
            "stain_family": dict(Counter(r["stain_family"] for r in g)),
            "median_brightness": float(np.median([r["content_mean"] for r in g])),
            "median_contrast": float(np.median([r["content_std"] for r in g])),
            "median_lap_var": float(np.median([r["lap_var"] for r in g])),
            "overlay_any": sum(bool(r["overlay_flags"]) for r in g),
            "watermark_suspect": sum(r["watermark_suspect"] for r in g),
            "manual_flagged": sum(bool(r["manual_categories"]) for r in g),
            "appearance_clusters": dict(Counter(r.get("appearance_cluster", -1) for r in g)),
            "objects": sum(r["n_objects"] for r in g),
            "class_counts": dict(sum((Counter(r["class_counts"]) for r in g), Counter())),
        }

    # ---- label / split summary -----------------------------------------------
    split_summary = {}
    for s in sorted({r["split"] for r in recs}):
        rs = [r for r in recs if r["split"] == s]
        cc = sum((Counter(r["class_counts"]) for r in rs), Counter())
        split_summary[s] = {"images": len(rs), "with_label_file": sum(r["label_exists"] for r in rs),
                            "empty_label_files": sum(r["label_exists"] and r["n_objects"] == 0 for r in rs),
                            "objects": sum(r["n_objects"] for r in rs),
                            "class_counts": {class_names[c] if c < len(class_names) else str(c): v
                                             for c, v in sorted(cc.items())},
                            "groups": dict(sorted(Counter(r["group"] for r in rs).items()))}
    bw = [v for r in recs for v in r.get("box_w_px", [])]
    bh = [v for r in recs for v in r.get("box_h_px", [])]
    label_summary = {
        "images_with_labels": sum(r["label_exists"] for r in recs),
        "images_without_label_file": [r["rel_path"] for r in recs if not r["label_exists"]],
        "empty_label_files": [r["rel_path"] for r in recs if r["label_exists"] and r["n_objects"] == 0],
        "orphan_label_files": orphan_labels,
        "total_objects": sum(r["n_objects"] for r in recs),
        "bbox_rows": sum(r["n_bbox_rows"] for r in recs),
        "polygon_rows": sum(r["n_polygon_rows"] for r in recs),
        "images_with_polygons": sum(r["n_polygon_rows"] > 0 for r in recs),
        "invalid_rows": sum(r["n_invalid_rows"] for r in recs),
        "out_of_range_rows": sum(r["n_out_of_range"] for r in recs),
        "boxes_centered_in_padding_or_border": sum(r["boxes_in_padding"] for r in recs),
        "tiny_boxes_lt8px_at640": sum(r["tiny_boxes"] for r in recs),
        "near_identical_box_pairs": sum(r["dup_boxes"] for r in recs),
        "objects_per_image": summarize([r["n_objects"] for r in recs]),
        "box_w_px": summarize(bw), "box_h_px": summarize(bh),
        "class_names": class_names,
    }

    # derived folders already present (read-only inspection)
    derived = {}
    for d in [root / "data" / "processed", root / "data" / "splits"]:
        if d.exists():
            for sub in sorted(p for p in d.rglob("*") if p.is_dir()):
                files = [p for p in sub.iterdir() if p.is_file()]
                if files:
                    derived[str(sub.relative_to(root))] = len(files)

    # ---- write per-image CSV -------------------------------------------------
    skip_cols = {"_boxes", "class_counts", "box_w_px", "box_h_px"}
    cols = [k for k in recs[0].keys() if not k.startswith("_") and k not in skip_cols]
    for r in recs:
        for k in r:
            if not k.startswith("_") and k not in skip_cols and k not in cols:
                cols.append(k)
    cols.append("class_counts")
    with open(out / "per_image_audit.csv", "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wri.writeheader()
        for r in recs:
            row = {k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items() if k in cols}
            row["class_counts"] = json.dumps({class_names[c] if c < len(class_names) else str(c): v
                                              for c, v in r["class_counts"].items()})
            wri.writerow(row)
    with open(out / "duplicate_pairs.csv", "w", newline="", encoding="utf-8") as f:
        keys = ["kind", "file_a", "split_a", "file_b", "split_b", "cross_split", "phash_dist", "dhash_dist",
                "dhash_full_dist", "ncc32", "transform"]
        wri = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wri.writeheader()
        for p in sorted(pairs, key=lambda p: (p["kind"], p["phash_dist"])):
            wri.writerow({k: (f"{p[k]:.4f}" if isinstance(p[k], float) else p[k]) for k in keys})

    # ---- contact sheets ------------------------------------------------------
    print("Rendering contact sheets ...")
    gcol = {g: GROUP_COLORS[i % len(GROUP_COLORS)] for i, g in enumerate(groups)}
    sheets = []

    def cap(r, extra=""):
        lines = [f"#{r['idx']} {r['source_stem']} [{r['split']}]", f"{r['width']}x{r['height']}"]
        if extra:
            lines.append(extra)
        return lines

    # representative: per group, 8 images spread across brightness quantiles
    rep = []
    for gname in groups:
        g = sorted([r for r in ok if r["group"] == gname], key=lambda r: r["content_mean"])
        pick = sorted({int(round(q * (len(g) - 1))) for q in np.linspace(0, 1, min(8, len(g)))})
        rep += [(Path(g[i]["path"]), cap(g[i], f"group {gname} | {g[i]['stain_family'].split(' ')[0]}"),
                 gcol[gname], None) for i in pick]
    sheets.append(contact_sheet(rep, sheets_dir / "01_representative_by_group.png",
                                "Representative images: 8 per source group (sorted dark to bright)", cols=8, size=230,
                                note="Frame colour = source group (filename prefix). Original files, no preprocessing."))
    # annotations overlay
    ann = []
    for gname in groups:
        g = [r for r in ok if r["group"] == gname and r["n_objects"]]
        rng = np.random.default_rng(1)
        for r in rng.choice(g, size=min(4, len(g)), replace=False) if g else []:
            ann.append((Path(r["path"]), cap(r, f"{r['n_objects']} obj {json.dumps(r['class_counts'])}"),
                        gcol[gname], r["_boxes"]))
    sheets.append(contact_sheet(ann, sheets_dir / "02_annotation_overlay_samples.png",
                                "YOLO annotations drawn on originals (4 per group)", cols=6, size=300,
                                note=f"red = class 0 '{class_names[0] if class_names else 0}', cyan = class 1 "
                                     f"'{class_names[1] if len(class_names) > 1 else 1}'. Polygons shown as bounding boxes."))

    def flagged_sheet(name, title, sel, note="", extra=lambda r: "", cols=6, size=260, limit=120):
        items = [(Path(r["path"]), cap(r, extra(r)), "#d03b3b", None) for r in sel[:limit]]
        if items:
            sheets.append(contact_sheet(items, sheets_dir / name, f"{title} ({len(sel)})", cols=cols, size=size,
                                        note=note))

    wm = sorted([r for r in ok if r["watermark_suspect"] or r["overlay_flags"]],
                key=lambda r: (not r["watermark_suspect"], r["group"], r["idx"]))
    flagged_sheet("03_auto_text_watermark_overlay_candidates.png", "AUTO: text / watermark / overlay candidates", wm,
                  note="Heuristic detector (no OCR). Includes captions, magnification labels, scale bars, credit banners.",
                  extra=lambda r: r["overlay_flags"][:40])
    man = [r for r in recs if r["manual_categories"]]
    flagged_sheet("04_manual_review_flags.png", "MANUAL visual review: flagged images", man, size=300, cols=5,
                  note="Findings from looking at every image in the catalog sheets (see manual_visual_review.csv).",
                  extra=lambda r: r["manual_categories"][:40])
    flagged_sheet("05_blur_candidates.png", "Blur / low-sharpness candidates",
                  sorted([r for r in ok if "blur" in r["quality_flags"]], key=lambda r: r["sharp_z_group"]),
                  extra=lambda r: f"lapvar={r['lap_var']:.1f} zgrp={r['sharp_z_group']:+.1f}")
    flagged_sheet("06_outliers.png", "Outliers (IsolationForest 3% + within-group |z|>4)",
                  sorted([r for r in ok if r["outlier_global"] or r["outlier_in_group"]],
                         key=lambda r: -r["outlier_score"]),
                  extra=lambda r: (r["outlier_in_group_reason"] or r["outlier_reason"])[:40])
    flagged_sheet("07_large_black_border.png", "Large black microscope border (>=20% of image)",
                  sorted([r for r in ok if r["dark_border_frac"] >= 0.20], key=lambda r: -r["dark_border_frac"]),
                  extra=lambda r: f"black={r['dark_border_frac']:.0%} usable={r['usable_frac']:.0%}", limit=60)
    flagged_sheet("08_large_white_padding.png", "Large white page padding (>=20% of image)",
                  sorted([r for r in ok if r["white_pad_frac"] >= 0.20], key=lambda r: -r["white_pad_frac"]),
                  extra=lambda r: f"white={r['white_pad_frac']:.0%} usable={r['usable_frac']:.0%}", limit=60)
    flagged_sheet("09_exposure_contrast_issues.png", "Exposure / contrast issues",
                  [r for r in ok if re.search("exposed|contrast", r["quality_flags"])],
                  extra=lambda r: f"mean={r['content_mean']:.2f} std={r['content_std']:.2f}")
    # duplicate pairs side by side
    dup_items = []
    for p in sorted(pairs, key=lambda p: (p["kind"] == "similar (review)", p["phash_dist"]))[:60]:
        a, b = recs[p["a"]], recs[p["b"]]
        col = "#d03b3b" if p["kind"] != "similar (review)" else "#eda100"
        dup_items.append((Path(a["path"]), cap(a, f"{p['kind']} d={p['phash_dist']} ncc={p['ncc32']:.2f}"), col, None))
        dup_items.append((Path(b["path"]), cap(b, f"pair of #{a['idx']} {p['transform']}"), col, None))
    if dup_items:
        sheets.append(contact_sheet(dup_items, sheets_dir / "10_duplicate_pairs.png",
                                    f"Duplicate / near-duplicate pairs (side by side, {len(pairs)} pairs)", cols=6,
                                    size=230, note="Red = exact/near/flipped copy; amber = similar, needs human review."))
    # full catalog for manual review
    per = args.catalog_per_sheet
    for s0 in range(0, len(recs), per):
        chunk = recs[s0:s0 + per]
        items = [(Path(r["path"]), cap(r), gcol.get(r["group"]), None) for r in chunk]
        sheets.append(contact_sheet(items, sheets_dir / "catalog" / f"catalog_{s0 // per + 1:02d}.png",
                                    f"Full catalog {s0}-{s0 + len(chunk) - 1} of {len(recs)}", cols=8, size=240))
    sheets = [s for s in sheets if s]
    plots = make_plots(recs, groups, plots_dir)

    # ---- integrity check AFTER -----------------------------------------------
    after = {p: md5_file(Path(p)) for p in before}
    changed = [p for p in before if before[p] != after[p]]
    if changed:
        print("ERROR: original files changed during audit:", changed[:5])

    # ---- JSON summary --------------------------------------------------------
    def frac_flag(word):
        return [r["rel_path"] for r in ok if word in r["quality_flags"]]

    summary = {
        "dataset_root": str(root),
        "data_yaml": {k: v for k, v in yaml_info.items() if k != "raw"},
        "total_images": len(recs),
        "stray_images_outside_splits": [str(p.relative_to(root)) for p in stray],
        "readable": len(ok),
        "corrupted": [{"path": r["rel_path"], "error": r["error"]} for r in recs if r["status"] != "ok"],
        "formats_on_disk": dict(Counter(r.get("format") for r in recs)),
        "original_source_ext_from_filename": dict(Counter(r["source_ext"] for r in recs)),
        "modes": dict(Counter(r.get("mode") for r in recs)),
        "grayscale_images": [r["rel_path"] for r in ok if r["is_grayscale"]],
        "jpeg_quality_est": summarize([r["jpeg_quality_est"] for r in ok if r["jpeg_quality_est"] is not None]),
        "distinct_sizes": len({(r["width"], r["height"]) for r in ok}),
        "top_sizes": Counter(f"{r['width']}x{r['height']}" for r in ok).most_common(10),
        "width": summarize([r["width"] for r in ok]), "height": summarize([r["height"] for r in ok]),
        "aspect_ratio": summarize([r["aspect_ratio"] for r in ok]),
        "orientation": dict(Counter(r["orientation"] for r in ok)),
        "content_w": summarize([r["content_w"] for r in ok]), "content_h": summarize([r["content_h"] for r in ok]),
        "intensity_full": {k: summarize([r[k] for r in ok]) for k in ["int_min", "int_max", "int_mean", "int_std"]},
        "brightness_content": summarize([r["content_mean"] for r in ok]),
        "contrast_content": summarize([r["content_std"] for r in ok]),
        "dynamic_range_content": summarize([r["content_dynamic_range"] for r in ok]),
        "sharpness_lap_var": summarize([r["lap_var"] for r in ok]),
        "channel_means_content": {c: summarize([r[f"{c}_mean"] for r in ok]) for c in "rgb"},
        "stain_family": dict(Counter(r["stain_family"] for r in ok)),
        "blur_candidates": frac_flag("blur"),
        "underexposed": frac_flag("underexposed"), "overexposed": frac_flag("overexposed"),
        "low_contrast": frac_flag("low-contrast"), "low_jpeg_quality": frac_flag("low JPEG"),
        "large_black_border": frac_flag("large black border"),
        "large_white_padding": frac_flag("large white padding"),
        "circular_fov": sum(r["circular_fov"] for r in ok),
        "small_usable_area": frac_flag("small usable area"),
        "auto_watermark_suspects": [r["rel_path"] for r in ok if r["watermark_suspect"]],
        "auto_overlay_any": [{"path": r["rel_path"], "flags": r["overlay_flags"], "where": r["text_locations"]}
                             for r in ok if r["overlay_flags"]],
        "watermarks_confirmed_by_visual_review": [r["rel_path"] for r in recs if r["watermark_confirmed"]],
        "manual_review": {r["rel_path"]: {"categories": r["manual_categories"], "findings": r["manual_findings"]}
                          for r in recs if r["manual_categories"]},
        "duplicates": {"pairs_by_kind": dict(Counter(p["kind"] for p in pairs)),
                       "cross_split_pairs": [{k: p[k] for k in ("kind", "file_a", "split_a", "file_b", "split_b",
                                                                "phash_dist", "ncc32")} for p in pairs if p["cross_split"]],
                       "clusters": [[recs[i]["rel_path"] for i in c] for c in clusters],
                       "same_source_stem_multiple_files": {s: n for s, n in stem_counts.items() if n > 1},
                       "same_source_stem_across_splits": [s for s, v in stem_splits.items() if len(v) > 1]},
        "outliers_global": [{"path": r["rel_path"], "score": round(r["outlier_score"], 3), "why": r["outlier_reason"]}
                            for r in sorted(ok, key=lambda r: -r["outlier_score"]) if r["outlier_global"]],
        "outliers_in_group": [{"path": r["rel_path"], "why": r["outlier_in_group_reason"]}
                              for r in ok if r["outlier_in_group"]],
        "appearance_clustering": cluster_info,
        "groups": group_summary,
        "splits": split_summary,
        "labels": label_summary,
        "derived_folders_present_not_audited": derived,
        "integrity": {"files_hashed": len(before), "changed_during_audit": changed},
        "outputs": {"sheets": [str(Path(s).relative_to(root)) for s in sheets],
                    "plots": [str(Path(p).relative_to(root)) for p in plots]},
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_markdown(summary, recs, pairs, out, root)
    print(json.dumps({k: summary[k] for k in ("total_images", "readable", "formats_on_disk", "distinct_sizes")}))
    print("blur", len(summary["blur_candidates"]), "| watermark suspects", len(summary["auto_watermark_suspects"]),
          "| overlays", len(summary["auto_overlay_any"]), "| dup pairs", summary["duplicates"]["pairs_by_kind"],
          "| global outliers", len(summary["outliers_global"]), "| integrity changed", len(changed))
    print("Report:", out / "AUDIT_REPORT.md")


# --------------------------------------------------------------- markdown
def write_markdown(S, recs, pairs, out: Path, root: Path):
    def f(x, d=3):
        return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))

    def srow(name, s, d=3):
        return f"| {name} | {f(s.get('min'), d)} | {f(s.get('p05'), d)} | {f(s.get('median'), d)} | " \
               f"{f(s.get('mean'), d)} | {f(s.get('p95'), d)} | {f(s.get('max'), d)} |"

    L = []
    A = L.append
    A("# Dataset audit: onion root-tip microscopy (STEP 2)\n")
    A(f"- Dataset root: `{S['dataset_root']}`")
    A("- Generated by `src/dataset_audit_full.py`. Originals were opened read-only; "
      f"{S['integrity']['files_hashed']} image+label files were MD5-hashed before and after, "
      f"**{len(S['integrity']['changed_during_audit'])} changed**.")
    A("- Per-image metrics: `per_image_audit.csv`; machine-readable summary: `audit_summary.json`; "
      "duplicate pairs: `duplicate_pairs.csv`; contact sheets: `contact_sheets/`; plots: `plots/`.\n")
    A("## 1-2. Count and formats\n")
    A(f"- **Total images: {S['total_images']}** (readable {S['readable']}). Stray images outside split folders: "
      f"{len(S['stray_images_outside_splits'])}.")
    A(f"- Format on disk: {S['formats_on_disk']}. Original upload extension (from Roboflow filename): "
      f"{S['original_source_ext_from_filename']}.")
    jq = S["jpeg_quality_est"]
    A(f"- Estimated JPEG quality (from quantisation tables): median {f(jq.get('median'), 0)}, "
      f"range {f(jq.get('min'), 0)}-{f(jq.get('max'), 0)}. All images went through at least one lossy JPEG "
      "re-encode by Roboflow; the PNG-origin ones are no longer lossless.\n")
    A("## 3. Dimensions and aspect ratios\n")
    A(f"- {S['distinct_sizes']} distinct sizes. Orientation: {S['orientation']}.")
    A("- Most common sizes: " + ", ".join(f"{s} ({n})" for s, n in S["top_sizes"]))
    A("\n| metric | min | p05 | median | mean | p95 | max |\n|---|---|---|---|---|---|---|")
    for k, n in [("width", "width px"), ("height", "height px"), ("aspect_ratio", "aspect W/H"),
                 ("content_w", "specimen content width px"), ("content_h", "specimen content height px")]:
        A(srow(n, S[k], 2 if k == "aspect_ratio" else 0))
    A("\n## 4. RGB vs grayscale\n")
    A(f"- Modes: {S['modes']}. Effectively grayscale (channels nearly equal): {len(S['grayscale_images'])}.")
    A("- Stain colour family (hue of stained pixels): " + ", ".join(f"{k}: {v}" for k, v in S["stain_family"].items()))
    A("\n## 5-6. Intensity, brightness and contrast\n")
    A("Full-image values (0-255 luminance; include borders/padding):\n")
    A("| metric | min | p05 | median | mean | p95 | max |\n|---|---|---|---|---|---|---|")
    for k in ["int_min", "int_max", "int_mean", "int_std"]:
        A(srow(k, S["intensity_full"][k], 1))
    A("\nSpecimen-only values (0-1; borders and padding excluded):\n")
    A("| metric | min | p05 | median | mean | p95 | max |\n|---|---|---|---|---|---|---|")
    A(srow("brightness (mean)", S["brightness_content"]))
    A(srow("contrast (std)", S["contrast_content"]))
    A(srow("dynamic range p1-p99", S["dynamic_range_content"]))
    for c in "rgb":
        A(srow(f"{c.upper()} channel mean", S["channel_means_content"][c]))
    A(f"\n- Under-exposed: {len(S['underexposed'])}; over-exposed/washed-out: {len(S['overexposed'])}; "
      f"low contrast: {len(S['low_contrast'])}. See `plots/group_distributions.png`.\n")
    A("## 7. Corrupted / unreadable\n")
    A(f"- {len(S['corrupted'])} corrupted." + ("" if not S["corrupted"] else " " + str(S["corrupted"])))
    A("\n## 8. Duplicates\n")
    d = S["duplicates"]
    A(f"- Pairs by kind: {d['pairs_by_kind'] or 'none'}; strong-link clusters: {len(d['clusters'])}; "
      f"cross-split pairs: {len(d['cross_split_pairs'])}.")
    A(f"- Same Roboflow source stem used by >1 file: {len(d['same_source_stem_multiple_files'])}; "
      f"same stem in >1 split: {len(d['same_source_stem_across_splits'])}.")
    if pairs:
        A("\n| kind | A | B | splits | pHash d | NCC |\n|---|---|---|---|---|---|")
        for p in sorted(pairs, key=lambda p: (p["kind"] == "similar (review)", p["phash_dist"]))[:40]:
            A(f"| {p['kind']} | {recs[p['a']]['source_stem']} | {recs[p['b']]['source_stem']} | "
              f"{p['split_a']}/{p['split_b']} | {p['phash_dist']} | {p['ncc32']:.2f} |")
    A("\n## 9. Blur / low quality\n")
    A(f"- Blur candidates (sharpness far below own group, or global bottom tail): {len(S['blur_candidates'])}.")
    for pth in S["blur_candidates"][:30]:
        A(f"  - `{pth}`")
    A(f"- Low JPEG quality (<60): {len(S['low_jpeg_quality'])}. Small usable area (<35%): {len(S['small_usable_area'])}.")
    A("\n## 10. Black microscope borders and white page padding\n")
    A(f"- Large black border (>=20% of pixels): **{len(S['large_black_border'])}**; circular field of view detected: "
      f"{S['circular_fov']}.")
    A(f"- Large white page padding (>=20%): **{len(S['large_white_padding'])}**.")
    A("\n## 11. Watermarks, credits, burned-in text\n")
    wc = S["watermarks_confirmed_by_visual_review"]
    A(f"- **CONFIRMED stock-agency watermarks (visual review): {len(wc)}** - "
      + ", ".join(f"`{Path(p).name.split('.rf.')[0]}` ({Path(p).parts[0]})" for p in wc))
    A(f"- AUTO watermark suspects (gray banner or text over specimen): {len(S['auto_watermark_suspects'])}.")
    A(f"- AUTO any overlay (incl. captions in borders, scale bars, drawn lines): {len(S['auto_overlay_any'])}.")
    if S["manual_review"]:
        A(f"- **MANUAL visual review flagged {len(S['manual_review'])} images:**\n")
        A("| image | categories | finding |\n|---|---|---|")
        for pth, v in sorted(S["manual_review"].items()):
            A(f"| `{Path(pth).name.split('.rf.')[0]}` ({Path(pth).parts[0]}) | {v['categories']} | {v['findings']} |")
    A("\n## 12. Outliers\n")
    A(f"- Global (IsolationForest, 3%): {len(S['outliers_global'])}; within-group |robust z|>4: "
      f"{len(S['outliers_in_group'])}.")
    for o in S["outliers_global"]:
        A(f"  - `{o['path']}` score {o['score']}: {o['why']}")
    for o in S["outliers_in_group"][:25]:
        A(f"  - (in-group) `{o['path']}`: {o['why']}")
    A("\n## 13. Source groups\n")
    A("| group | n | splits | src ext | main sizes | #sizes | circ. FOV | white pad | black border | usable | "
      "stain | brightness | contrast | sharp (lapvar) | overlays |\n|" + "---|" * 15)
    for g, v in S["groups"].items():
        A(f"| {g} | {v['n']} | {v['splits']} | {v['source_ext']} | "
          f"{', '.join(f'{s}({n})' for s, n in v['top_sizes'][:2])} | {v['n_distinct_sizes']} | {v['circular_fov']} | "
          f"{v['median_white_pad']:.0%} | {v['median_dark_border']:.0%} | {v['median_usable']:.0%} | "
          f"{max(v['stain_family'], key=v['stain_family'].get).split(' (')[0]} | {v['median_brightness']:.2f} | "
          f"{v['median_contrast']:.2f} | {v['median_lap_var']:.0f} | {v['overlay_any']} |")
    A("\n- Layout sub-groups (A4 page render vs crop) per filename group:")
    for g, v in S["groups"].items():
        A(f"  - {g}: {v['layout_subgroups']}")
    ac = S["appearance_clustering"]
    if ac:
        A(f"\n- Unsupervised appearance clustering (KMeans on 15 appearance features, best k by silhouette = "
          f"{ac.get('best_k')}). Cluster membership per filename group:")
        for g, v in S["groups"].items():
            A(f"  - {g}: {v['appearance_clusters']}")
    A("\n## 14. Annotations (YOLO)\n")
    lb = S["labels"]
    A(f"- Classes from data.yaml: {lb['class_names']} (nc={S['data_yaml'].get('nc')}).")
    A(f"- Images with label file: {lb['images_with_labels']}/{S['total_images']}; missing: "
      f"{len(lb['images_without_label_file'])}; empty: {len(lb['empty_label_files'])}; orphan labels: "
      f"{len(lb['orphan_label_files'])}.")
    A(f"- Objects: {lb['total_objects']} = {lb['bbox_rows']} bbox rows + **{lb['polygon_rows']} polygon rows** "
      f"(in {lb['images_with_polygons']} images). Invalid rows: {lb['invalid_rows']}; out-of-range coords: "
      f"{lb['out_of_range_rows']}; near-identical box pairs: {lb['near_identical_box_pairs']}.")
    A(f"- Objects/image: median {f(lb['objects_per_image'].get('median'), 0)}, max "
      f"{f(lb['objects_per_image'].get('max'), 0)}. Box width px: median {f(lb['box_w_px'].get('median'), 0)}; "
      f"height px median {f(lb['box_h_px'].get('median'), 0)}.")
    A(f"- Boxes whose centre lies in padding/black border: {lb['boxes_centered_in_padding_or_border']}; "
      f"boxes <8 px if the specimen region is resized to 640: {lb['tiny_boxes_lt8px_at640']}.")
    A("\n## 15. Splits\n")
    dy = S["data_yaml"]
    A(f"- data.yaml: train=`{dy.get('train')}`, val=`{dy.get('val')}`, test=`{dy.get('test')}`. Resolved relative "
      f"to data.yaml these exist: train={dy.get('train_exists')}, val={dy.get('val_exists')}, "
      f"test={dy.get('test_exists')}. Split folders actually present: "
      f"{', '.join(f'`{s}/`' for s in S['splits'])}; **no test split exists**.")
    A("\n| split | images | labelled | objects | classes | groups |\n|---|---|---|---|---|---|")
    for s, v in S["splits"].items():
        A(f"| {s} | {v['images']} | {v['with_label_file']} | {v['objects']} | {v['class_counts']} | {v['groups']} |")
    if S["derived_folders_present_not_audited"]:
        A("\n- Pre-existing derived folders found (created before this step; not modified):")
        for k, v in S["derived_folders_present_not_audited"].items():
            A(f"  - `{k}`: {v} files")
    A("\n## 16. Leakage risks\n")
    A(f"- Cross-split duplicate/similar pairs: {len(d['cross_split_pairs'])}.")
    for p in d["cross_split_pairs"][:20]:
        A(f"  - {p['kind']}: `{p['file_a']}` ({p['split_a']}) vs `{p['file_b']}` ({p['split_b']}), d={p['phash_dist']}")
    A("- See summary in the chat / recommendations file for interpretation (source-group shortcuts, sequential "
      "frames from the same slide, burned-in text).")
    A("\n## Outputs\n")
    for s in S["outputs"]["sheets"] + S["outputs"]["plots"]:
        A(f"- `{s}`")
    (out / "AUDIT_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
