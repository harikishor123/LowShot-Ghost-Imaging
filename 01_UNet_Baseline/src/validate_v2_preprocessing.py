#!/usr/bin/env python3
"""Validate data/v2 after STEP 3: contact sheets, statistics and information-loss checks.

Read-only on originals and on data/v2. Writes to results/preprocessing_v2/.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from color_space import load_reference_linear, self_test, linear_to_srgb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data" / "v2"
OUT = ROOT / "results" / "preprocessing_v2"
GROUP_COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100", "E": "#e87ba4", "F": "#008300"}
SPLIT_ORDER = ["train", "val", "test"]


def font(n):
    try:
        return ImageFont.load_default(size=n)
    except TypeError:
        return ImageFont.load_default()


def thumb(img: Image.Image, size: int) -> Image.Image:
    img = img.copy()
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    return img


def checker_bg(w, h, sq=12):
    """Neutral checkerboard to make masked (zeroed) pixels visible as 'outside FOV'."""
    a = (np.indices((h, w)).sum(0) // sq) % 2
    v = np.where(a[..., None] == 1, 205, 230).astype(np.uint8)
    return np.repeat(v, 3, axis=2)


def processed_view(stem, size, outline=True, boxes=False):
    img = np.asarray(Image.open(V2 / "images" / f"{stem}.png").convert("RGB"))
    m = np.asarray(Image.open(V2 / "masks" / f"{stem}.png")) > 0
    bg = checker_bg(img.shape[1], img.shape[0], sq=max(6, img.shape[1] // 60))
    comp = np.where(m[..., None], img, bg)
    im = Image.fromarray(comp)
    d = ImageDraw.Draw(im)
    if outline:
        edge = m & ~ndi.binary_erosion(m, iterations=2, border_value=1)
        arr = np.asarray(im).copy()
        arr[ndi.binary_dilation(edge, iterations=1)] = (255, 0, 200)
        im = Image.fromarray(arr)
        d = ImageDraw.Draw(im)
    if boxes:
        W, H = im.size
        lw = max(2, W // 300)
        for line in (V2 / "labels" / f"{stem}.txt").read_text().splitlines():
            c, xc, yc, bw, bh = line.split()
            xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
            col = (230, 30, 30) if c == "0" else (0, 200, 255)
            d.rectangle([(xc - bw / 2) * W, (yc - bh / 2) * H, (xc + bw / 2) * W, (yc + bh / 2) * H], outline=col,
                        width=lw)
    return thumb(im, size)


def original_view(path, size, crop_box=None, color=(255, 0, 0)):
    im = Image.open(path).convert("RGB")
    if crop_box:
        d = ImageDraw.Draw(im)
        d.rectangle(crop_box, outline=color, width=max(4, im.size[0] // 200))
    return thumb(im, size)


def sheet(rows, out_path, title, note="", size=300, header_cols=None):
    """rows: list of (label_lines, [PIL images])"""
    f_t, f_c = font(20), font(13)
    ncol = max(len(r[1]) for r in rows)
    cap_h = 18 * max(len(r[0]) for r in rows) + 4
    pad = 8
    head = 60 if note else 36
    if header_cols:
        head += 20
    W = pad + ncol * (size + pad)
    Hh = head + len(rows) * (size + cap_h + pad) + pad
    S = Image.new("RGB", (W, Hh), (250, 250, 248))
    d = ImageDraw.Draw(S)
    d.text((pad, 6), title, fill=(10, 10, 10), font=f_t)
    if note:
        d.text((pad, 34), note, fill=(82, 81, 78), font=f_c)
    if header_cols:
        for j, hc in enumerate(header_cols):
            d.text((pad + j * (size + pad), head - 18), hc, fill=(40, 40, 40), font=f_c)
    for i, (labels, ims) in enumerate(rows):
        y = head + i * (size + cap_h + pad)
        for j, im in enumerate(ims):
            x = pad + j * (size + pad)
            d.rectangle([x, y, x + size - 1, y + size - 1], fill=(228, 228, 225))
            S.paste(im, (x + (size - im.width) // 2, y + (size - im.height) // 2))
        for k, l in enumerate(labels):
            d.text((pad, y + size + 2 + 18 * k), l[:160], fill=(20, 20, 20) if k == 0 else (90, 90, 88), font=f_c)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    S.save(out_path, optimize=True)
    return out_path


def grid(items, out_path, title, cols=8, size=230, note=""):
    """items: list of (PIL image, [caption lines], frame colour)"""
    f_t, f_c = font(20), font(13)
    cap_h = 18 * max(len(c) for _, c, _ in items) + 4
    pad = 6
    head = 58 if note else 34
    rows = math.ceil(len(items) / cols)
    S = Image.new("RGB", (pad + cols * (size + pad), head + rows * (size + cap_h + pad) + pad), (250, 250, 248))
    d = ImageDraw.Draw(S)
    d.text((pad, 6), title, fill=(10, 10, 10), font=f_t)
    if note:
        d.text((pad, 32), note, fill=(82, 81, 78), font=f_c)
    for k, (im, caps, col) in enumerate(items):
        x = pad + (k % cols) * (size + pad)
        y = head + (k // cols) * (size + cap_h + pad)
        d.rectangle([x, y, x + size - 1, y + size - 1], fill=(228, 228, 225))
        S.paste(im, (x + (size - im.width) // 2, y + (size - im.height) // 2))
        if col:
            d.rectangle([x, y, x + size - 1, y + size - 1], outline=col, width=4)
        for i, c in enumerate(caps):
            d.text((x + 2, y + size + 2 + 18 * i), c[:38], fill=(20, 20, 20) if i == 0 else (90, 90, 88), font=f_c)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    S.save(out_path, optimize=True)
    return out_path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    M = list(csv.DictReader(open(V2 / "metadata" / "manifest.csv", encoding="utf-8")))
    by = {r["stem"]: r for r in M}
    kept = [r for r in M if r["split"] != "excluded"]
    excl = [r for r in M if r["split"] == "excluded"]
    orig_path = {r["stem"]: ROOT / r["original_split"] / "images" / r["original_filename"] for r in M}
    rng = np.random.default_rng(7)
    made = []

    def pick(cond, n):
        c = [r for r in kept if cond(r)]
        idx = rng.choice(len(c), size=min(n, len(c)), replace=False) if c else []
        return [c[i] for i in sorted(idx)]

    def crop_box(r):
        return [int(r["crop_x0"]), int(r["crop_y0"]), int(r["crop_x1"]), int(r["crop_y1"])]

    # 01 before / after, representative per group/layout
    rows = []
    layouts = [("A", lambda r: r["group"] == "A" and r["had_white_padding"] == "True", 3, "A (A4 page, circle)"),
               ("A", lambda r: r["group"] == "A" and r["had_white_padding"] == "False", 2, "A (square crop)"),
               ("B", lambda r: r["group"] == "B", 1, "B (only B kept)"),
               ("C", lambda r: r["group"] == "C", 2, "C (zoom crop)"),
               ("D", lambda r: r["group"] == "D", 2, "D (blue stain)"),
               ("F", lambda r: r["group"] == "F", 2, "F (phone, caption)")]
    for g, cond, n, lab in layouts:
        for r in pick(cond, n):
            rows.append(([f"{r['stem']} [{r['split']}] {lab}: {r['orig_w']}x{r['orig_h']} -> {r['proc_w']}x{r['proc_h']} "
                          f"(scale {r['resize_scale']}, FOV {r['fov_type']}, valid {float(r['fov_valid_frac']):.0%})"],
                         [original_view(orig_path[r["stem"]], 330, crop_box(r)), processed_view(r["stem"], 330)]))
    made.append(sheet(rows, OUT / "01_before_after.png", "Before / after: original (red = crop box) vs processed",
                      note="Processed: masked pixels shown as checkerboard (stored as 0); magenta = FOV mask edge.",
                      size=330, header_cols=["original", "processed (data/v2/images)"]))

    # 02 specimen cropping (white padding removal)
    rows = []
    for r in sorted(pick(lambda r: float(r["white_pad_frac_removed"]) > 0.15, 6),
                    key=lambda r: -float(r["white_pad_frac_removed"])):
        rows.append(([f"{r['stem']} ({r['group']}): white padding removed {float(r['white_pad_frac_removed']):.0%}; "
                      f"crop {r['crop_w']}x{r['crop_h']} -> {r['proc_w']}x{r['proc_h']}"],
                     [original_view(orig_path[r["stem"]], 300, crop_box(r)), processed_view(r["stem"], 300)]))
    made.append(sheet(rows, OUT / "02_specimen_cropping.png", "Specimen cropping: white page padding removed",
                      size=300, header_cols=["original + crop box", "processed"]))

    # 03 black border masking
    rows = []
    for r in pick(lambda r: r["fov_type"] == "circular" and r["group"] in "AF", 6):
        mask = Image.open(V2 / "masks" / f"{r['stem']}.png").convert("RGB")
        rows.append(([f"{r['stem']} ({r['group']}): black surround {float(r['black_border_frac_orig']):.0%} of original; "
                      f"valid FOV {float(r['fov_valid_frac']):.0%} of processed frame"],
                     [original_view(orig_path[r["stem"]], 280, crop_box(r)), thumb(mask, 280),
                      processed_view(r["stem"], 280)]))
    made.append(sheet(rows, OUT / "03_black_border_masking.png", "Black microscope border: FOV mask",
                      note="Pixels outside the circular field are set to 0 and excluded via the mask.", size=280,
                      header_cols=["original + crop box", "FOV mask", "processed"]))

    # 04 text removal (F caption, B_27 X1000)
    rows = []
    for r in pick(lambda r: r["group"] == "F", 4) + [by["B_27"]]:
        o = Image.open(orig_path[r["stem"]]).convert("RGB")
        W, H = o.size
        if r["group"] == "F":
            region = (int(W * 0.2), 0, int(W * 0.8), int(H * 0.2))
        else:
            cx0, cy0, cx1, cy1 = crop_box(r)
            region = (cx0, cy0 + int((cy1 - cy0) * 0.7), cx0 + int((cx1 - cx0) * 0.45), cy1)
        o_reg = thumb(o.crop(region), 420)
        # same region in processed frame
        p = processed_view(r["stem"], 10_000, outline=False)
        s = float(r["resize_scale"])
        cx0, cy0 = int(r["crop_x0"]), int(r["crop_y0"])
        pr = tuple(int(round(v)) for v in ((region[0] - cx0) * s, (region[1] - cy0) * s,
                                           (region[2] - cx0) * s, (region[3] - cy0) * s))
        pr = (max(0, pr[0]), max(0, pr[1]), min(p.width, pr[2]), min(p.height, pr[3]))
        p_reg = thumb(p.crop(pr), 420) if pr[2] > pr[0] and pr[3] > pr[1] else Image.new("RGB", (420, 100), (228, 228, 225))
        rows.append(([f"{r['stem']}: {r['text_detail']}",
                      "region outside crop/FOV -> removed" if pr[3] <= 0 or pr[2] <= pr[0] else
                      "same region after processing (checkerboard = masked)"],
                     [o_reg, p_reg]))
    made.append(sheet(rows, OUT / "04_text_removal.png", "Burned-in text removal (captions lie outside the FOV)",
                      size=420, header_cols=["original region", "processed, same region"]))

    # 05 retained difficult images
    diff = [r for r in kept if r["flag_blur"] == "True" or r["flag_artefact"] == "True"
            or r["flag_low_contrast"] == "True" or r["flag_overexposed"] == "True"]
    items = []
    for r in diff:
        fl = [k.replace("flag_", "") for k in ("flag_blur", "flag_artefact", "flag_low_contrast", "flag_overexposed")
              if r[k] == "True"]
        items.append((processed_view(r["stem"], 230), [f"{r['stem']} [{r['split']}]", ",".join(fl)],
                      GROUP_COLORS[r["group"]]))
    made.append(grid(items, OUT / "05_retained_difficult.png", f"Retained difficult images, flagged in manifest ({len(items)})",
                     note="Kept as realistic microscopy variation: blur, fibres/debris, low contrast, washed-out."))

    # 06 excluded (non-duplicate)
    items = []
    for r in [x for x in excl if x["dup_role"] != "removed"]:
        why = "group E" if r["group"] == "E" else r["exclusion_reason"].split(":")[0]
        extra = r["exclusion_reason"].split("[this image: ")[-1].rstrip("]") if "[this image" in r["exclusion_reason"] else ""
        items.append((original_view(orig_path[r["stem"]], 230), [f"{r['stem']} [{r['original_split']}]", why, extra],
                      "#d03b3b"))
    made.append(grid(items, OUT / "06_excluded_images.png", f"Excluded images (not duplicates): {len(items)}",
                     cols=6, note="All of group E (provenance/watermarks) + A_384 (foreign stain) + A_134 (drawn pointer)."))

    # 07 duplicates removed: kept vs removed
    groups = list(csv.DictReader(open(V2 / "metadata" / "duplicate_groups.csv", encoding="utf-8")))
    sel = [g for g in groups if "B_" not in g["members"]] + [g for g in groups if "B_" in g["members"]][:10]
    items = []
    for g in sel:
        keep = g["kept"]
        items.append((original_view(orig_path[keep], 210), [f"KEPT {keep}", by[keep]["split"]], "#0ca30c"))
        for m in g["members"].split("|"):
            if m != keep:
                items.append((original_view(orig_path[m], 210), [f"removed {m}", f"dup of {keep}"], "#d03b3b"))
        while len(items) % 4:
            items.append((Image.new("RGB", (10, 10), (250, 250, 248)), [""], None))
    made.append(grid(items, OUT / "07_duplicates_removed_examples.png",
                     f"Duplicates: kept (green) vs removed (red) - all {len([g for g in groups if 'B_' not in g['members']])} "
                     "non-B groups + 10 of the A/B groups", cols=8, size=210))

    # 08 labels transformed onto processed frame
    items = []
    for g in "ABCDF":
        for r in pick(lambda r, g=g: r["group"] == g, 3 if g != "B" else 1):
            items.append((processed_view(r["stem"], 300, outline=False, boxes=True),
                          [f"{r['stem']} kept {r['n_boxes_kept']}/{r['n_boxes_orig']} boxes"], GROUP_COLORS[g]))
    made.append(grid(items, OUT / "08_labels_on_processed.png", "YOLO boxes after transform (auxiliary labels)",
                     cols=5, size=300, note="red = d (dividing), cyan = nd (non-dividing). Boxes follow crop + scale."))

    # 09 full processed catalog for visual information-loss inspection
    order = sorted(kept, key=lambda r: (r["group"], r["stem"]))
    for k in range(0, len(order), 48):
        items = [(processed_view(r["stem"], 230), [f"{r['stem']} [{r['split']}]", f"{r['proc_w']}x{r['proc_h']}"],
                  GROUP_COLORS[r["group"]]) for r in order[k:k + 48]]
        made.append(grid(items, OUT / "catalog" / f"processed_{k // 48 + 1:02d}.png",
                         f"Processed catalog {k + 1}-{k + len(items)} of {len(order)}"))

    # ---- quantitative checks ------------------------------------------------------------
    stats = {"color_self_test": self_test()}
    # rim erosion loss estimate + reference/png consistency + intensity preservation
    rim, pngdiff = [], []
    for r in kept:
        lin, m = load_reference_linear(V2 / "reference" / f"{r['stem']}.npz")
        if r["fov_type"] == "circular":
            dil = ndi.binary_dilation(m, iterations=4) & ~m
            rim.append(dil.sum() / (m.sum() + dil.sum()))
        png = np.asarray(Image.open(V2 / "images" / f"{r['stem']}.png").convert("RGB")).astype(np.float32) / 255
        pngdiff.append(float(np.abs(linear_to_srgb(lin) - png).max()))
    stats["rim_erosion_area_loss_circular_fov"] = {"median": float(np.median(rim)), "max": float(np.max(rim))}
    stats["max_abs_diff_png_vs_npz_srgb"] = float(np.max(pngdiff))
    audit = {r["file"]: r for r in csv.DictReader(open(ROOT / "results/dataset_audit/per_image_audit.csv",
                                                      encoding="utf-8"))}
    o_mean, p_mean = [], []
    for r in kept:
        a = audit[r["original_filename"]]
        o_mean.append((float(a["r_mean"]) + float(a["g_mean"]) + float(a["b_mean"])) / 3)
        p_mean.append(float(r["mean_srgb_in_fov"]))
    o_mean, p_mean = np.array(o_mean), np.array(p_mean)
    stats["intensity_preservation"] = {
        "pearson_r_orig_vs_processed_mean_srgb": float(np.corrcoef(o_mean, p_mean)[0, 1]),
        "median_abs_diff": float(np.median(np.abs(o_mean - p_mean))),
        "note": "original = audit specimen-region mean (sRGB, 0-1); processed = mean inside FOV mask"}
    pw = np.array([int(r["proc_w"]) for r in kept])
    ph = np.array([int(r["proc_h"]) for r in kept])
    stats["final_dims"] = {
        "width": {"min": int(pw.min()), "median": float(np.median(pw)), "max": int(pw.max())},
        "height": {"min": int(ph.min()), "median": float(np.median(ph)), "max": int(ph.max())},
        "aspect_w_over_h": {"min": float((pw / ph).min()), "median": float(np.median(pw / ph)),
                            "max": float((pw / ph).max())},
        "long_side_eq_1024": int(((np.maximum(pw, ph)) == 1024).sum()),
        "long_side_lt_1024_not_upscaled": int(((np.maximum(pw, ph)) < 1024).sum()),
        "top_sizes": Counter(f"{w}x{h}" for w, h in zip(pw, ph)).most_common(8),
        "by_group": {g: Counter(f"{r['proc_w']}x{r['proc_h']}" for r in kept if r["group"] == g).most_common(3)
                     for g in "ABCDF"}}
    stats["fov_valid_frac"] = {g: float(np.median([float(r["fov_valid_frac"]) for r in kept if r["group"] == g]))
                               for g in "ABCDF"}
    stats["boxes"] = {"orig_in_kept_images": sum(int(r["n_boxes_orig"]) for r in kept),
                      "kept": sum(int(r["n_boxes_kept"]) for r in kept),
                      "dropped": {r["stem"]: r["boxes_dropped"] for r in kept if r["n_boxes_dropped"] not in ("0", "")}}
    stats["split_counts"] = dict(Counter(r["split"] for r in kept))
    stats["split_by_group"] = {s: dict(sorted(Counter(r["group"] for r in kept if r["split"] == s).items()))
                               for s in SPLIT_ORDER}
    stats["split_classes"] = {s: {"d": sum(int(r["n_d_kept"]) for r in kept if r["split"] == s),
                                  "nd": sum(int(r["n_nd_kept"]) for r in kept if r["split"] == s)} for s in SPLIT_ORDER}
    stats["split_flags"] = {s: {f: sum(r[f] == "True" for r in kept if r["split"] == s)
                                for f in ("flag_blur", "flag_artefact", "flag_low_contrast", "flag_overexposed")}
                            for s in SPLIT_ORDER}
    # leakage re-check: no duplicate group spans splits; original-valid images spread
    dg = defaultdict(set)
    for r in M:
        if r["dup_group"] and r["split"] != "excluded":
            dg[r["dup_group"]].add(r["split"])
    stats["dup_groups_with_more_than_one_kept"] = sum(len(v) > 1 for v in dg.values())
    stats["kept_from_original_split"] = dict(Counter(f"{r['original_split']}->{r['split']}" for r in kept).most_common())
    (OUT / "validation_stats.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")

    # ---- plots ---------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#b5b4ae", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "axes.titleweight": "bold", "axes.titlesize": 10})

    def clean(ax):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for g in "ABCDF":
        rs = [r for r in kept if r["group"] == g]
        axes[0].scatter([int(r["proc_w"]) for r in rs], [int(r["proc_h"]) for r in rs], s=20, alpha=0.75,
                        color=GROUP_COLORS[g], label=f"{g} (n={len(rs)})", edgecolors="#fcfcfb", linewidths=0.5)
    axes[0].set_xlabel("processed width (px)")
    axes[0].set_ylabel("processed height (px)")
    axes[0].set_title("Final image dimensions (long side capped at 1024, no upscaling)", loc="left")
    axes[0].legend(frameon=False, title="group")
    clean(axes[0])
    asp = np.array([int(r["proc_w"]) / int(r["proc_h"]) for r in kept])
    axes[1].hist(asp, bins=30, color="#2a78d6", edgecolor="#fcfcfb")
    axes[1].set_xlabel("aspect ratio (W/H)")
    axes[1].set_ylabel("images")
    axes[1].set_title("Final aspect-ratio distribution", loc="left")
    clean(axes[1])
    p = OUT / "plots" / "final_dimensions.png"
    p.parent.mkdir(exist_ok=True)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(p)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True, sharey=False)
    groups = "ABCDF"
    for ax, s in zip(axes, SPLIT_ORDER):
        cnt = [stats["split_by_group"][s].get(g, 0) for g in groups]
        tot = sum(cnt)
        bars = ax.bar(list(groups), cnt, color=[GROUP_COLORS[g] for g in groups], edgecolor="#fcfcfb", linewidth=2)
        for b, c in zip(bars, cnt):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{c}\n{c / tot:.0%}", ha="center", va="bottom",
                    fontsize=8, color="#0b0b0b")
        cl = stats["split_classes"][s]
        ax.set_title(f"{s}: {tot} images | d={cl['d']} nd={cl['nd']} ({cl['d'] / (cl['d'] + cl['nd']):.1%} d)",
                     loc="left")
        ax.set_ylim(0, max(cnt) * 1.25)
        ax.set_xlabel("source group")
        clean(ax)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("images")
    p = OUT / "plots" / "split_by_source_group.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(p)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    allg = "ABCDEF"
    orig = Counter(r["group"] for r in M)
    fin = Counter(r["group"] for r in kept)
    x = np.arange(len(allg))
    ax.bar(x - 0.2, [orig[g] for g in allg], 0.38, color="#b5b4ae", label="original (644)", edgecolor="#fcfcfb")
    ax.bar(x + 0.2, [fin[g] for g in allg], 0.38, color="#2a78d6", label=f"data/v2 kept ({len(kept)})",
           edgecolor="#fcfcfb")
    for i, g in enumerate(allg):
        ax.text(i - 0.2, orig[g], str(orig[g]), ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, fin[g], str(fin[g]), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, list(allg))
    ax.set_xlabel("source group")
    ax.set_ylabel("images")
    ax.set_title("Source-group distribution: original vs final", loc="left")
    ax.legend(frameon=False)
    clean(ax)
    ax.grid(axis="x", visible=False)
    p = OUT / "plots" / "source_groups_original_vs_final.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(p)

    print(json.dumps({k: stats[k] for k in ("color_self_test", "rim_erosion_area_loss_circular_fov",
                                            "max_abs_diff_png_vs_npz_srgb", "intensity_preservation", "final_dims",
                                            "fov_valid_frac", "split_counts", "split_classes", "split_flags",
                                            "dup_groups_with_more_than_one_kept", "kept_from_original_split")},
                     indent=1, default=str))
    print("outputs:", len(made))


if __name__ == "__main__":
    main()
