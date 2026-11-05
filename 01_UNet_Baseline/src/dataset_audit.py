#!/usr/bin/env python3
"""Audit a microscopy image dataset and generate a human-readable report.

This script is intentionally conservative: it reports suspicious images without deleting
anything. It scans the dataset for basic quality issues, duplicates, aspect-ratio
heterogeneity, and likely outliers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def safe_image_open(path: Path):
    try:
        with Image.open(path) as img:
            img.load()
        return Image.open(path), None
    except Exception as exc:  # pragma: no cover - used for file validation
        return None, str(exc)


def perceptual_hash(img: Image.Image, hash_size: int = 8) -> str:
    """Simple dHash implementation for duplicate and near-duplicate detection."""
    gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.uint8)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = np.packbits(diff.astype(np.uint8), axis=1)
    return hashlib.sha1(bits.tobytes()).hexdigest()


def hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        return 9999
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


def compute_border_score(gray: np.ndarray, border_fraction: float = 0.08) -> Dict[str, float]:
    h, w = gray.shape
    border = max(2, int(min(h, w) * border_fraction))
    slices = []
    if h > border:
        slices.append(gray[:border, :])
        slices.append(gray[-border:, :])
    if w > border:
        slices.append(gray[:, :border])
        slices.append(gray[:, -border:])
    if not slices:
        arr = gray.astype(np.float32).ravel()
    else:
        arr = np.concatenate([s.ravel() for s in slices], axis=0).astype(np.float32)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "ratio_low": float(np.mean(arr < 0.15)) if arr.size else 0.0,
    }


def compute_stats_for_image(path: Path) -> Dict[str, object]:
    result: Dict[str, object] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "status": "ok",
        "issues": [],
    }

    if not path.exists():
        result["status"] = "missing"
        return result

    try:
        with Image.open(path) as img:
            img.load()
            width, height = img.size
            mode = img.mode
            rgb_img = img.convert("RGB")
            thumb = rgb_img.copy()
            thumb.thumbnail((512, 512), Image.Resampling.LANCZOS)
            arr = np.asarray(thumb)
            gray = np.asarray(ImageOps.grayscale(thumb), dtype=np.float32) / 255.0
    except Exception as exc:
        result["status"] = "corrupted"
        result["issues"].append(f"Could not open image: {exc}")
        return result

    result["width"] = width
    result["height"] = height
    result["aspect_ratio"] = width / max(height, 1)
    result["mode"] = mode

    if len(arr.shape) == 2:
        result["channels"] = 1
        rgb = np.stack([arr.astype(np.float32), arr.astype(np.float32), arr.astype(np.float32)], axis=-1) / 255.0
    elif arr.shape[-1] in (3, 4):
        result["channels"] = arr.shape[-1]
        rgb = arr.astype(np.float32) / 255.0
    else:
        result["channels"] = 0
        rgb = np.zeros((height, width, 3), dtype=np.float32)

    result["brightness_mean"] = float(gray.mean())
    result["brightness_std"] = float(gray.std())
    result["contrast"] = float(gray.std())
    result["mean_rgb"] = {"r": float(rgb[..., 0].mean()), "g": float(rgb[..., 1].mean()), "b": float(rgb[..., 2].mean())}
    result["std_rgb"] = {"r": float(rgb[..., 0].std()), "g": float(rgb[..., 1].std()), "b": float(rgb[..., 2].std())}

    border = compute_border_score(gray)
    result["border_mean"] = border["mean"]
    result["border_std"] = border["std"]
    result["border_low_ratio"] = border["ratio_low"]

    if result["brightness_mean"] < 0.08 or result["brightness_std"] < 0.02:
        result["issues"].append("very dark or low-contrast image")
    if result["contrast"] < 0.02:
        result["issues"].append("extremely low contrast")
    if result["border_low_ratio"] > 0.4:
        result["issues"].append("large dark border / irrelevant background candidate")

    if result["channels"] == 3:
        if rgb[..., 0].mean() > 0.8 and rgb[..., 1].mean() > 0.8 and rgb[..., 2].mean() > 0.8:
            result["issues"].append("very bright RGB image")

    return result


def find_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    paths: List[Path] = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTS:
            paths.append(candidate)
    return paths


def collect_dataset_report(root: Path) -> Dict[str, object]:
    roots = [root / "train" / "images", root / "valid" / "images", root / "test" / "images", root / "images"]
    image_paths = []
    seen = set()
    for search_root in roots:
        for p in find_images(search_root):
            if p not in seen:
                seen.add(p)
                image_paths.append(p)
    image_paths = sorted(image_paths)

    reports = []
    for p in image_paths:
        r = compute_stats_for_image(p)
        reports.append(r)

    duplicate_groups: Dict[str, List[str]] = defaultdict(list)
    hashes = {}
    for rep in reports:
        if rep["status"] != "ok":
            continue
        p = Path(rep["path"])
        try:
            with Image.open(p) as img:
                key = perceptual_hash(img)
        except Exception:
            continue
        hashes[str(p)] = key
        duplicate_groups[key].append(str(p))

    duplicate_clusters = []
    for key, group in duplicate_groups.items():
        if len(group) > 1:
            duplicate_clusters.append({"hash": key, "paths": group})

    # near-duplicate detection using hash distance on all unique hashes
    near_duplicate_pairs = []
    unique_hashes = list(hashes.values())
    unique_paths = list(hashes.keys())
    for i in range(len(unique_hashes)):
        for j in range(i + 1, len(unique_hashes)):
            if hamming_distance(unique_hashes[i], unique_hashes[j]) <= 4:
                near_duplicate_pairs.append({
                    "path_a": unique_paths[i],
                    "path_b": unique_paths[j],
                    "distance": hamming_distance(unique_hashes[i], unique_hashes[j]),
                })

    widths = [float(r["width"]) for r in reports if r.get("status") == "ok"]
    heights = [float(r["height"]) for r in reports if r.get("status") == "ok"]
    aspect_ratios = [float(r["aspect_ratio"]) for r in reports if r.get("status") == "ok"]
    brightness = [float(r["brightness_mean"]) for r in reports if r.get("status") == "ok"]
    contrast_vals = [float(r["contrast"]) for r in reports if r.get("status") == "ok"]
    modes = Counter(str(r["mode"]) for r in reports if r.get("status") == "ok")

    report = {
        "dataset_root": str(root),
        "total_images": len(image_paths),
        "total_valid_images": sum(1 for r in reports if r.get("status") == "ok"),
        "corrupted_images": [r["path"] for r in reports if r.get("status") == "corrupted"],
        "image_formats": dict(Counter(p.suffix.lower() for p in image_paths)),
        "channels": dict(modes),
        "width_summary": summarize_values(widths),
        "height_summary": summarize_values(heights),
        "aspect_ratio_summary": summarize_values(aspect_ratios),
        "brightness_summary": summarize_values(brightness),
        "contrast_summary": summarize_values(contrast_vals),
        "possible_border_issue_images": [
            r["path"] for r in reports if r.get("status") == "ok" and "large dark border / irrelevant background candidate" in r.get("issues", [])
        ],
        "low_quality_candidates": [
            r["path"] for r in reports if r.get("status") == "ok" and ("very dark or low-contrast image" in r.get("issues", []) or "extremely low contrast" in r.get("issues", []))
        ],
        "duplicate_groups": duplicate_clusters,
        "near_duplicate_pairs": near_duplicate_pairs,
        "all_reports": reports,
    }
    return report


def summarize_values(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def choose_representative_images(reports: List[Dict[str, object]], n: int = 16) -> List[str]:
    ok = [r for r in reports if r.get("status") == "ok"]
    ranked = sorted(ok, key=lambda r: (float(r["brightness_mean"]), float(r["contrast"]), float(r["aspect_ratio"])))
    selected: List[str] = []
    indices = []
    for q in np.linspace(0, 1, min(n, len(ranked))):
        idx = min(len(ranked) - 1, max(0, int(round(q * (len(ranked) - 1)))))
        if idx not in indices:
            indices.append(idx)
    for idx in indices:
        selected.append(ranked[idx]["path"])
    if len(selected) < n:
        for rep in ok:
            if rep["path"] not in selected:
                selected.append(rep["path"])
            if len(selected) >= n:
                break
    return selected[:n]


def montage_images(paths, output_path: Path, grid_size: int = 4, max_images: int = 16):
    selected = paths[:max_images]
    if not selected:
        return
    cols = grid_size
    rows = math.ceil(len(selected) / cols)
    sample = []
    for p in selected:
        with Image.open(p) as img:
            rgb = img.convert("RGB")
            rgb.thumbnail((256, 256), Image.Resampling.LANCZOS)
            sample.append(rgb)
    tile_w = max(img.size[0] for img in sample)
    tile_h = max(img.size[1] for img in sample)
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(245, 245, 245))
    for idx, img in enumerate(sample):
        x = (idx % cols) * tile_w
        y = (idx // cols) * tile_h
        canvas.paste(img, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Audit an onion microscopy dataset and generate a report.")
    parser.add_argument("--data-root", type=str, default=".", help="Dataset root directory. Defaults to current workspace.")
    parser.add_argument("--output-dir", type=str, default="results/audit", help="Where to save JSON/figure outputs.")
    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = collect_dataset_report(root)
    (output_dir / "dataset_audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    selected = choose_representative_images(report["all_reports"], n=16)
    montage_images(selected, output_dir / "contact_sheet.png", grid_size=4, max_images=16)

    print(json.dumps({
        "dataset_root": str(root),
        "total_images": report["total_images"],
        "corrupted": len(report["corrupted_images"]),
        "formats": report["image_formats"],
        "channels": report["channels"],
        "width_range": [report["width_summary"]["min"], report["width_summary"]["max"]],
        "height_range": [report["height_summary"]["min"], report["height_summary"]["max"]],
        "aspect_ratio_range": [report["aspect_ratio_summary"]["min"], report["aspect_ratio_summary"]["max"]],
        "candidate_border_images": len(report["possible_border_issue_images"]),
        "duplicate_clusters": len(report["duplicate_groups"]),
        "report_path": str(output_dir / "dataset_audit_report.json"),
        "contact_sheet_path": str(output_dir / "contact_sheet.png"),
    }, indent=2))


if __name__ == "__main__":
    main()
