#!/usr/bin/env python3
"""STEP 3: build the clean, deduplicated, leakage-free data/v2 dataset.

Reads (never modifies) the original Roboflow export in train/ and valid/, plus the STEP 2
audit outputs in results/dataset_audit/. Writes only to data/v2/.

Per retained image:
  1. find the specimen region: strip uniform white page padding, find the black
     microscope surround (edge-connected dark pixels), keep the largest specimen
     component and take its convex hull as the field-of-view (FOV) mask
     (circular eyepiece fields stay circular; rectangular crops stay rectangular);
  2. blank known burned-in captions that sit outside the FOV (group F banner);
  3. crop to the FOV bounding box (no centre-crop, aspect ratio preserved);
  4. sRGB -> linear light, area-average DOWNscale only if the long side > MAX_SIDE
     (never upscale), mask out pixels outside the FOV (set to 0), linear -> sRGB;
  5. save: uint16 sRGB reference + mask (.npz, lossless), 8-bit PNG preview, mask PNG,
     and YOLO boxes transformed into the new pixel frame (auxiliary only).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parent))
from color_space import encode_uint8, encode_uint16, linear_to_srgb, srgb_to_linear  # noqa: E402

MAX_SIDE = 1024          # cap on the long side of the processed image (downscale only)
WORK_SIDE = 1024         # analysis resolution for mask detection
RIM_ERODE_PX = 4         # erode FOV mask at output resolution to drop the coloured eyepiece rim
SEED = 42
SPLIT_FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}
RELATED_NCC = 0.80       # processed-thumbnail correlation that forces two kept images into one split
EXCLUDE_GROUPS = {"E": "source group E excluded: web/stock images of heterogeneous, uncertain provenance; "
                       "9/15 carry stock-agency watermarks (Getty Images, Shutterstock, Science Photo Library)"}
EXCLUDE_IMAGES = {
    "A_384": "severe artefact/source mismatch: violet Giemsa-like stain unlike the rest of group A (likely foreign source)",
    "A_134": "severe artefact: black eyepiece pointer arrow drawn over the specimen",
}
# group F: phone screenshots with a fixed caption banner in the black margin (2048x2048 frame)
F_CAPTION_BOX_2048 = (500, 0, 1550, 195)  # x0, y0, x1, y1 ; caption pixels found at rows 31-166, cols 542-1503


def stem_of(fname: str) -> str:
    """'A_100_png.rf.<hash>.jpg' -> 'A_100'"""
    s = fname.split(".rf.")[0]
    for ext in ("_png", "_jpg", "_jpeg"):
        if s.endswith(ext):
            return s[: -len(ext)]
    return s


def rect_padding(mask: np.ndarray, frac: float = 0.985):
    def run(v):
        idx = np.flatnonzero(~v)
        return int(idx[0]) if idx.size else int(v.size)
    rows, cols = mask.mean(1) >= frac, mask.mean(0) >= frac
    return run(rows), run(rows[::-1]), run(cols), run(cols[::-1])


def fov_mask_working(rgb_u8: np.ndarray, group: str):
    """Return (scale, fov mask at working res, info) for one original image."""
    H, W = rgb_u8.shape[:2]
    s = min(1.0, WORK_SIDE / max(H, W))
    work = np.asarray(Image.fromarray(rgb_u8).resize((max(1, round(W * s)), max(1, round(H * s))),
                                                     Image.Resampling.BOX)) if s < 1 else rgb_u8
    rgb = work.astype(np.float32) / 255
    h, w = rgb.shape[:2]
    gray = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    mx, mn = rgb.max(2), rgb.min(2)
    sat = np.where(mx > 1e-3, (mx - mn) / np.maximum(mx, 1e-3), 0)
    info = {}

    # white page padding (uniform rows/cols at the edges)
    wt, wb, wl, wr = rect_padding((gray > 0.93) & (sat < 0.08))
    if wt + wb >= h or wl + wr >= w:
        wt = wb = wl = wr = 0
    inner = np.zeros((h, w), bool)
    inner[wt:h - wb, wl:w - wr] = True
    info["white_pad_frac"] = float(1 - inner.mean())

    dark = gray < 0.11
    caption = np.zeros((h, w), bool)
    if group == "F":
        x0, y0, x1, y1 = (round(v * w / 2048) for v in F_CAPTION_BOX_2048)
        caption[y0:y1, x0:x1] = True
        dark = dark | caption  # treat caption as part of the black surround

    lab, n = ndi.label(dark & inner)
    border = np.zeros((h, w), bool)
    if n:
        ring = inner & ~ndi.binary_erosion(inner, iterations=2)
        ids = np.unique(lab[ring & (lab > 0)])
        border = np.isin(lab, ids[ids > 0])
    info["black_border_frac"] = float(border.mean())

    content = ndi.binary_opening(inner & ~border, iterations=2)
    lab, n = ndi.label(content)
    if n == 0:
        content = inner.copy()
    else:
        sizes = ndi.sum(content, lab, index=np.arange(1, n + 1))
        content = lab == (int(np.argmax(sizes)) + 1)
    # convex hull of the specimen component = field of view (restores dark debris bites at the rim)
    ys, xs = np.nonzero(content)
    pts = np.column_stack([xs, ys])
    hull_img = Image.new("L", (w, h), 0)
    try:
        hull = ConvexHull(pts)
        poly = [tuple(map(float, pts[v])) for v in hull.vertices]
        ImageDraw.Draw(hull_img).polygon(poly, fill=255, outline=255)
        fov = np.asarray(hull_img) > 0
    except Exception:
        fov = content
    fov &= inner
    fov &= ~caption
    bys, bxs = np.nonzero(fov)
    by0, by1, bx0, bx1 = bys.min(), bys.max() + 1, bxs.min(), bxs.max() + 1
    fill = fov[by0:by1, bx0:bx1].mean()
    info["fov_type"] = "circular" if fill < 0.9 else "rectangular"
    info["fov_fill_of_bbox"] = float(fill)
    info["caption_overlap_px"] = int((caption & content).sum())
    return s, fov, (bx0, by0, bx1, by1), info


def process_one(job):
    rec, out_root = job
    out_root = Path(out_root)
    src = Path(rec["src_path"])
    with Image.open(src) as im:
        rgb_u8 = np.asarray(im.convert("RGB"))
    H, W = rgb_u8.shape[:2]
    s, fov_w, (bx0, by0, bx1, by1), info = fov_mask_working(rgb_u8, rec["group"])

    # crop box in original pixel coordinates
    x0, y0 = max(0, math.floor(bx0 / s)), max(0, math.floor(by0 / s))
    x1, y1 = min(W, math.ceil(bx1 / s)), min(H, math.ceil(by1 / s))
    crop = rgb_u8[y0:y1, x0:x1]
    cw, ch = x1 - x0, y1 - y0
    scale = min(1.0, MAX_SIDE / max(cw, ch))  # downscale only
    ow, oh = max(1, round(cw * scale)), max(1, round(ch * scale))

    # sRGB -> linear, area-average resample in linear light, back to sRGB
    lin = srgb_to_linear(crop.astype(np.float32) / 255.0)
    if scale < 1:
        lin = np.stack([np.asarray(Image.fromarray(lin[..., c], mode="F").resize((ow, oh), Image.Resampling.BOX))
                        for c in range(3)], axis=-1)
    # FOV mask at output resolution
    mcrop = fov_w[by0:by1, bx0:bx1].astype(np.float32)
    m = np.asarray(Image.fromarray(mcrop, mode="F").resize((ow, oh), Image.Resampling.BILINEAR)) > 0.5
    if info["fov_type"] == "circular":
        # the eyepiece rim also touches the crop edge at the circle's extremes: erode against the frame too
        m = ndi.binary_erosion(m, iterations=RIM_ERODE_PX, border_value=0)
    elif not m.all():
        m = ndi.binary_erosion(m, iterations=RIM_ERODE_PX, border_value=1)
    lin = lin * m[..., None]
    srgb = linear_to_srgb(lin)

    sid = rec["stem"]
    np.savez_compressed(out_root / "reference" / f"{sid}.npz", srgb16=encode_uint16(srgb), mask=m.astype(np.uint8))
    Image.fromarray(encode_uint8(srgb)).save(out_root / "images" / f"{sid}.png", optimize=True)
    Image.fromarray((m * 255).astype(np.uint8)).save(out_root / "masks" / f"{sid}.png", optimize=True)

    # YOLO boxes -> new frame
    kept, dropped = [], Counter()
    for cls, xc, yc, bw, bh in rec["boxes"]:
        bx_0, by_0 = (xc - bw / 2) * W, (yc - bh / 2) * H
        bx_1, by_1 = (xc + bw / 2) * W, (yc + bh / 2) * H
        area = max(1e-9, (bx_1 - bx_0) * (by_1 - by_0))
        nx0, ny0 = (max(bx_0, x0) - x0) * scale, (max(by_0, y0) - y0) * scale
        nx1, ny1 = (min(bx_1, x1) - x0) * scale, (min(by_1, y1) - y0) * scale
        if nx1 <= nx0 or ny1 <= ny0:
            dropped["outside_crop"] += 1
            continue
        if (nx1 - nx0) * (ny1 - ny0) / (scale * scale) < 0.5 * area:
            dropped["mostly_cropped"] += 1
            continue
        cx, cy = int(np.clip((nx0 + nx1) / 2, 0, ow - 1)), int(np.clip((ny0 + ny1) / 2, 0, oh - 1))
        if not m[cy, cx]:
            dropped["centre_outside_fov_mask"] += 1
            continue
        kept.append((cls, (nx0 + nx1) / 2 / ow, (ny0 + ny1) / 2 / oh, (nx1 - nx0) / ow, (ny1 - ny0) / oh))
    with open(out_root / "labels" / f"{sid}.txt", "w", encoding="utf-8") as f:
        for cls, a, b, c, d in kept:
            f.write(f"{cls} {a:.6f} {b:.6f} {c:.6f} {d:.6f}\n")

    # 32x32 gray thumbnail (mask-filled) for related-image linking
    g = srgb @ np.array([0.299, 0.587, 0.114], np.float32)
    g = np.where(m, g, g[m].mean() if m.any() else 0)
    th = np.asarray(Image.fromarray(g.astype(np.float32), mode="F").resize((32, 32), Image.Resampling.BOX)).ravel()
    th = (th - th.mean()) / (th.std() + 1e-6)

    out = {
        "stem": sid, "crop_x0": x0, "crop_y0": y0, "crop_x1": x1, "crop_y1": y1,
        "crop_w": cw, "crop_h": ch, "resize_scale": round(scale, 5), "proc_w": ow, "proc_h": oh,
        "proc_aspect": round(ow / oh, 4), "fov_type": info["fov_type"], "fov_valid_frac": round(float(m.mean()), 4),
        "white_pad_frac_removed": round(info["white_pad_frac"], 4),
        "black_border_frac_orig": round(info["black_border_frac"], 4),
        "caption_overlap_px": info["caption_overlap_px"],
        "n_boxes_kept": len(kept), "n_d_kept": sum(k[0] == 0 for k in kept), "n_nd_kept": sum(k[0] == 1 for k in kept),
        "boxes_dropped": json.dumps(dict(dropped)), "n_boxes_dropped": sum(dropped.values()),
        "mean_linear_in_fov": round(float(lin[m].mean()) if m.any() else 0.0, 5),
        "mean_srgb_in_fov": round(float(srgb[m].mean()) if m.any() else 0.0, 5),
        "_thumb": th,
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="data/v2")
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    out = (root / args.out).resolve()
    for d in ["images", "labels", "masks", "reference", "metadata", "splits/train", "splits/val", "splits/test"]:
        (out / d).mkdir(parents=True, exist_ok=True)

    audit = {r["file"]: r for r in csv.DictReader(open(root / "results/dataset_audit/per_image_audit.csv",
                                                       encoding="utf-8"))}
    pairs = list(csv.DictReader(open(root / "results/dataset_audit/duplicate_pairs.csv", encoding="utf-8")))

    # ---- inventory of originals ---------------------------------------------
    recs = {}
    for split_dir in ["train", "valid"]:
        for p in sorted((root / split_dir / "images").glob("*.jpg")):
            a = audit[p.name]
            stem = stem_of(p.name)
            lbl = root / split_dir / "labels" / (p.stem + ".txt")
            boxes = []
            for line in lbl.read_text(encoding="utf-8").splitlines():
                t = line.split()
                if len(t) == 5:
                    boxes.append((int(t[0]), *map(float, t[1:])))
            manual = a["manual_categories"]
            recs[stem] = {
                "stem": stem, "original_filename": p.name, "original_split": split_dir, "group": a["group"],
                "src_path": str(p), "md5": a["md5"], "orig_w": int(a["width"]), "orig_h": int(a["height"]),
                "boxes": boxes, "n_boxes_orig": len(boxes),
                "n_d_orig": sum(b[0] == 0 for b in boxes), "n_nd_orig": sum(b[0] == 1 for b in boxes),
                "flag_blur": "blur" in a["quality_flags"],
                "flag_low_contrast": "low-contrast" in a["quality_flags"],
                "flag_overexposed": "overexposed" in a["quality_flags"],
                "flag_artefact": ("artefact" in manual) or ("source-mismatch" in manual),
                "artefact_detail": a["manual_findings"],
                "had_white_padding": float(a["white_pad_frac"]) >= 0.05,
                "had_black_border": float(a["dark_border_frac"]) >= 0.05,
                "had_circular_fov": a["circular_fov"] == "True",
                "lap_var": float(a["lap_var"]),
                "content_area": int(a["content_w"]) * int(a["content_h"]),
                "manual_categories": manual,
                "auto_overlay_flags": a["overlay_flags"],
            }
    # text present in original + how it is handled
    for r in recs.values():
        txt = []
        if r["group"] == "F":
            txt.append("caption banner 'ONION ROOT TIP SQUASH VIEW (X675) / STAIN- 2% ACETIC-ORCEIN' (top black margin)")
        if r["group"] == "B":
            txt.append("'X1000' magnification label (bottom-left black corner)")
        if "WATERMARK" in r["manual_categories"]:
            txt.append("stock-agency watermark")
        if r["group"] == "E" and "scale-bar" in r["manual_categories"]:
            txt.append("scale bar")
        if "pointer" in r["manual_categories"]:
            txt.append("drawn pointer")
        r["had_text"] = bool(txt)
        r["text_detail"] = "; ".join(txt)

    # ---- exclusions ------------------------------------------------------------
    for r in recs.values():
        r["exclusion_reason"] = ""
        if r["group"] in EXCLUDE_GROUPS:
            extra = f" [this image: {r['manual_categories']}]" if r["manual_categories"] else ""
            r["exclusion_reason"] = EXCLUDE_GROUPS[r["group"]] + extra
        elif r["stem"] in EXCLUDE_IMAGES:
            r["exclusion_reason"] = EXCLUDE_IMAGES[r["stem"]]

    # ---- deduplication over the audit's duplicate graph -------------------------
    parent = {s: s for s in recs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pair_rows = []
    for p in pairs:
        a, b = stem_of(p["file_a"]), stem_of(p["file_b"])
        parent[find(a)] = find(b)
        pair_rows.append((a, b, p["kind"], p["phash_dist"], p["ncc32"], p["transform"]))
    clusters = defaultdict(list)
    for s in recs:
        clusters[find(s)].append(s)
    dup_groups = {k: v for k, v in clusters.items() if len(v) > 1}
    dup_log = []
    for gid, (k, members) in enumerate(sorted(dup_groups.items(), key=lambda kv: sorted(kv[1])[0]), start=1):
        cand = [m for m in members if not recs[m]["exclusion_reason"]]
        # keeper: prefer group A over B (B copies carry the burned-in X1000 label), then the larger
        # specimen area in the original, then the sharper frame
        keeper = max(cand, key=lambda m: (recs[m]["group"] != "B", recs[m]["content_area"], recs[m]["lap_var"])) \
            if cand else None
        for m in sorted(members):
            recs[m]["dup_group"] = f"DUP{gid:03d}"
            recs[m]["dup_members"] = "|".join(sorted(members))
            if m == keeper:
                recs[m]["dup_role"] = "kept"
            elif not recs[m]["exclusion_reason"]:
                recs[m]["dup_role"] = "removed"
                recs[m]["exclusion_reason"] = f"duplicate: same underlying frame as {keeper} (kept)"
            else:
                recs[m]["dup_role"] = "excluded_other_reason"
        dup_log.append({"dup_group": f"DUP{gid:03d}", "members": "|".join(sorted(members)), "kept": keeper or "",
                        "kinds": ";".join(sorted({k for a, b, k, *_ in pair_rows if a in members}))})
    for r in recs.values():
        r.setdefault("dup_group", "")
        r.setdefault("dup_members", "")
        r.setdefault("dup_role", "")

    kept = [r for r in recs.values() if not r["exclusion_reason"]]
    print(f"originals {len(recs)} | excluded {len(recs) - len(kept)} | kept {len(kept)}")

    # ---- process images -----------------------------------------------------------
    jobs = [({k: r[k] for k in ("stem", "src_path", "group", "boxes")}, str(out)) for r in kept]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(process_one, jobs, chunksize=4))
    for res in results:
        recs[res["stem"]].update(res)

    # ---- related-image linking for the split (belt and braces beyond the audit graph)
    ks = [r["stem"] for r in kept]
    T = np.stack([recs[s]["_thumb"] for s in ks])
    ncc = T @ T.T / T.shape[1]
    sp = {s: s for s in ks}

    def f2(x):
        while sp[x] != x:
            sp[x] = sp[sp[x]]
            x = sp[x]
        return x

    related = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            if ncc[i, j] >= RELATED_NCC:
                sp[f2(ks[i])] = f2(ks[j])
                related.append((ks[i], ks[j], float(ncc[i, j])))
    sgroups = defaultdict(list)
    for s in ks:
        sgroups[f2(s)].append(s)
    for gi, (k, members) in enumerate(sorted(sgroups.items(), key=lambda kv: sorted(kv[1])[0])):
        for m in members:
            recs[m]["split_group"] = f"SG{gi:04d}"
            recs[m]["split_group_size"] = len(members)

    # ---- 70/15/15 split at split-group level, stratified by source group --------------
    rng = random.Random(SEED)
    assign = {}
    for src in sorted({recs[s]["group"] for s in ks}):
        groups = sorted({recs[s]["split_group"] for s in ks if recs[s]["group"] == src})
        rng.shuffle(groups)
        groups.sort(key=lambda g: -sum(recs[s]["split_group"] == g for s in ks))  # big groups first (stable)
        n_src = sum(recs[s]["group"] == src for s in ks)
        target = {k: v * n_src for k, v in SPLIT_FRACS.items()}
        have = Counter()
        for g in groups:
            size = sum(recs[s]["split_group"] == g for s in ks)
            split = max(SPLIT_FRACS, key=lambda k: (target[k] - have[k]) / target[k])
            assign[g] = split
            have[split] += size
    for s in ks:
        recs[s]["split"] = assign[recs[s]["split_group"]]
    for r in recs.values():
        if r["exclusion_reason"]:
            r["split"] = "excluded"

    # ---- write split lists ---------------------------------------------------------------
    for split in SPLIT_FRACS:
        members = sorted(s for s in ks if recs[s]["split"] == split)
        d = out / "splits" / split
        (d / "ids.txt").write_text("\n".join(members) + "\n", encoding="utf-8")
        (d / "images.txt").write_text("\n".join(f"../../images/{s}.png" for s in members) + "\n", encoding="utf-8")
        (d / "references.txt").write_text("\n".join(f"../../reference/{s}.npz" for s in members) + "\n",
                                          encoding="utf-8")
    (out / "splits" / "test" / "HELD_OUT_README.txt").write_text(
        "TEST SPLIT - HELD OUT.\nDo not use for training, model selection, early stopping, hyper-parameter tuning\n"
        "or photon-level calibration. Evaluate on it exactly once, at final evaluation.\n", encoding="utf-8")

    # ---- manifest -----------------------------------------------------------------------------
    cols = ["stem", "original_filename", "original_split", "group", "md5", "orig_w", "orig_h",
            "split", "exclusion_reason", "dup_group", "dup_role", "dup_members", "split_group", "split_group_size",
            "crop_x0", "crop_y0", "crop_x1", "crop_y1", "crop_w", "crop_h", "resize_scale", "proc_w", "proc_h",
            "proc_aspect", "fov_type", "fov_valid_frac", "white_pad_frac_removed", "black_border_frac_orig",
            "had_white_padding", "had_black_border", "had_circular_fov", "had_text", "text_detail",
            "caption_overlap_px", "flag_blur", "flag_low_contrast", "flag_overexposed", "flag_artefact",
            "artefact_detail", "auto_overlay_flags", "n_boxes_orig", "n_d_orig", "n_nd_orig", "n_boxes_kept",
            "n_d_kept", "n_nd_kept", "n_boxes_dropped", "boxes_dropped", "mean_srgb_in_fov", "mean_linear_in_fov"]
    rows = sorted(recs.values(), key=lambda r: (r["split"] == "excluded", r["group"], r["stem"]))
    with open(out / "metadata" / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    with open(out / "metadata" / "exclusions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "original_filename", "group", "original_split", "exclusion_reason"])
        for r in rows:
            if r["exclusion_reason"]:
                w.writerow([r["stem"], r["original_filename"], r["group"], r["original_split"], r["exclusion_reason"]])
    with open(out / "metadata" / "duplicate_groups.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dup_group", "members", "kept", "kinds"])
        w.writeheader()
        w.writerows(dup_log)
    with open(out / "metadata" / "split_related_links.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem_a", "stem_b", "ncc32_processed", "split_a", "split_b"])
        for a, b, c in related:
            w.writerow([a, b, f"{c:.4f}", recs[a]["split"], recs[b]["split"]])
    cfg = {"MAX_SIDE": MAX_SIDE, "WORK_SIDE": WORK_SIDE, "RIM_ERODE_PX": RIM_ERODE_PX, "SEED": SEED,
           "SPLIT_FRACS": SPLIT_FRACS, "RELATED_NCC": RELATED_NCC, "EXCLUDE_GROUPS": EXCLUDE_GROUPS,
           "EXCLUDE_IMAGES": EXCLUDE_IMAGES, "F_CAPTION_BOX_2048": F_CAPTION_BOX_2048,
           "colour": "IEC 61966-2-1 sRGB transfer assumed; resampling done in linear light (area average)",
           "reference_format": "npz: srgb16 = uint16 sRGB-encoded RGB (HxWx3), mask = uint8 FOV mask (HxW)",
           "audit_inputs": ["results/dataset_audit/per_image_audit.csv", "results/dataset_audit/duplicate_pairs.csv",
                            "results/dataset_audit/manual_visual_review.csv (via per_image_audit.csv)"]}
    (out / "metadata" / "preprocessing_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    # auxiliary YOLO dataset description (labels are NOT a dependency of reconstruction training)
    (out / "data_yolo_aux.yaml").write_text(
        "# AUXILIARY: YOLO view of data/v2 (boxes transformed to the processed frame). Not used by the\n"
        "# reconstruction pipeline. Test split is held out until final evaluation.\n"
        f"path: {out.as_posix()}\ntrain: splits/train/images.txt\nval: splits/val/images.txt\n"
        "test: splits/test/images.txt\nnc: 2\nnames: ['d', 'nd']  # d = dividing (mitotic), nd = non-dividing\n",
        encoding="utf-8")

    # ---- summary ---------------------------------------------------------------------------------
    summ = {
        "originals": len(recs), "kept": len(ks), "excluded": len(recs) - len(ks),
        "excluded_by_reason": dict(Counter(
            "group E (provenance/watermarks)" if r["group"] == "E" else
            ("duplicate" if r["exclusion_reason"].startswith("duplicate") else r["stem"])
            for r in recs.values() if r["exclusion_reason"])),
        "duplicate_groups": len(dup_groups), "duplicates_removed": sum(r["dup_role"] == "removed" for r in recs.values()),
        "split_counts": dict(Counter(recs[s]["split"] for s in ks)),
        "split_by_group": {sp_: dict(sorted(Counter(recs[s]["group"] for s in ks if recs[s]["split"] == sp_).items()))
                           for sp_ in SPLIT_FRACS},
        "split_classes": {sp_: {"d": sum(recs[s]["n_d_kept"] for s in ks if recs[s]["split"] == sp_),
                                "nd": sum(recs[s]["n_nd_kept"] for s in ks if recs[s]["split"] == sp_)}
                          for sp_ in SPLIT_FRACS},
        "related_links": len(related),
        "related_cross_split": sum(recs[a]["split"] != recs[b]["split"] for a, b, _ in related),
        "boxes_orig_kept_images": sum(recs[s]["n_boxes_orig"] for s in ks),
        "boxes_kept": sum(recs[s]["n_boxes_kept"] for s in ks),
        "max_caption_overlap_px": max(recs[s]["caption_overlap_px"] for s in ks),
    }
    (out / "metadata" / "build_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
