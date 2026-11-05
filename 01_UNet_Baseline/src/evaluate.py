#!/usr/bin/env python3
"""Full-image evaluation of a trained reconstruction model, per photon level.

    python src/evaluate.py --model models/model_A_baseline_unet/best.keras --split test

For every sample of the split (all 6 levels), computed in LINEAR LIGHT inside the FOV mask:
  raw          : I_noisy = N/P as stored (unclipped, the unbiased photon estimate)
  raw_clipped  : clip(I_noisy, 0, 1) (a trivial, fairer baseline)
  model        : U-Net reconstruction (pixels outside the mask set to 0)
Metrics: MSE, MAE, PSNR (data range 1), SSIM (see losses.image_metrics).
Full images are zero-padded into a fixed 1024x1024 canvas for inference and cropped back.
Model A receives only the noisy image; P is used here only to group the results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import keras  # noqa: E402

from color_space import encode_uint16, encode_uint8, linear_to_srgb  # noqa: E402
from dataset_loader import load_pair, read_samples  # noqa: E402
from losses import image_metrics, masked_mae, masked_mse, masked_psnr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LEVELS = [10.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
# chosen from the STEP 3 catalog before any model results existed (one+ per source group,
# incl. a debris image A_265 and a blur-flagged image D_23)
REPRESENTATIVE = {"test": ["A_22", "A_3", "A_265", "C_14", "C_3", "D_12", "D_23", "F_58"]}


CANVAS = 1024  # every processed image has H, W <= 1024 (long side = 1024)


def predict_full(model, x: np.ndarray, canvas: int = CANVAS) -> np.ndarray:
    """Zero-pad into one fixed canvas (single input shape -> no per-size retracing / memory growth),
    run the model, crop back. Zero padding matches the masked (zero) region outside the FOV."""
    H, W, _ = x.shape
    if H > canvas or W > canvas:
        raise ValueError(f"image {H}x{W} larger than canvas {canvas}")
    xp = np.zeros((1, canvas, canvas, 3), np.float32)
    xp[0, :H, :W] = x
    return np.asarray(model(xp, training=False))[0, :H, :W]


def summarize(vals):
    a = np.asarray(vals, float)
    n = len(a)
    se = a.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if n > 1 else 0.0, "median": float(np.median(a)),
            "ci95_low": float(a.mean() - 1.96 * se), "ci95_high": float(a.mean() + 1.96 * se), "n": n}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # table contains 'Δ'; Windows console default is cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/model_A_baseline_unet/final_restored_best.keras")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out-dir", default="results/evaluation/model_A_test")
    ap.add_argument("--save-png", action="store_true", help="save 8-bit sRGB previews of every reconstruction")
    ap.add_argument("--limit", type=int, default=0, help="debug only: evaluate the first N samples (val split only)")
    a = ap.parse_args()
    if a.limit and a.split == "test":
        sys.exit("--limit is for debugging on val only; the test split is evaluated in full, once")
    model_path = ROOT / a.model
    model = keras.models.load_model(model_path, custom_objects={"masked_mse": masked_mse, "masked_mae": masked_mae,
                                                               "masked_psnr": masked_psnr}, compile=False)
    model_md5 = hashlib.md5(model_path.read_bytes()).hexdigest()
    odir = ROOT / a.out_dir
    rdir = odir / "reconstructions"
    odir.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)

    rows = read_samples(a.split)
    if a.limit:
        rows = [r for r in rows if r["photon_level"] in ("1%", "100%")][: a.limit]
    manifest = {r["stem"]: r for r in csv.DictReader(open(ROOT / "data/v2/metadata/manifest.csv", encoding="utf-8"))}
    reps = set(REPRESENTATIVE.get(a.split, []))
    out, t0 = [], time.time()
    for k, r in enumerate(rows):
        x, y, m = load_pair(r)
        t = time.time()
        pred = predict_full(model, x)
        if pred.shape != x.shape:
            raise RuntimeError(f"output shape {pred.shape} != input shape {x.shape} for {r['sample_id']}")
        pred = pred * m[..., None]
        infer_s = time.time() - t
        variants = {"raw": x, "raw_clipped": np.clip(x, 0, 1), "model": pred}
        res = {k_: image_metrics(v, y, m) for k_, v in variants.items()}  # sequential: bounded memory
        row = {"sample_id": r["sample_id"], "image_id": r["image_id"], "group": manifest[r["image_id"]]["group"],
               "split": a.split, "P": r["photon_budget_P"], "level": r["photon_level"],
               "height": x.shape[0], "width": x.shape[1], "fov_pixels": int(m.sum()), "inference_s": infer_s,
               "flag_blur": manifest[r["image_id"]]["flag_blur"], "flag_artefact": manifest[r["image_id"]]["flag_artefact"]}
        for v, d in res.items():
            for mname, val in d.items():
                row[f"{v}_{mname}"] = val
        row["delta_psnr"] = row["model_psnr"] - row["raw_psnr"]
        row["delta_ssim"] = row["model_ssim"] - row["raw_ssim"]
        row["mse_reduction_pct"] = 100 * (1 - row["model_mse"] / row["raw_mse"])
        row["mae_reduction_pct"] = 100 * (1 - row["model_mae"] / row["raw_mae"])
        out.append(row)
        if r["image_id"] in reps:
            np.savez_compressed(rdir / f"{r['image_id']}__P{int(r['photon_budget_P']):04d}.npz",
                                recon_srgb16=encode_uint16(linear_to_srgb(pred)), mask=m.astype(np.uint8))
        if a.save_png:
            from PIL import Image
            Image.fromarray(encode_uint8(linear_to_srgb(pred))).save(
                rdir / f"{r['image_id']}__P{int(r['photon_budget_P']):04d}.png")
        if (k + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {k + 1}/{len(rows)}  {el:.0f}s  eta {el / (k + 1) * (len(rows) - k - 1):.0f}s", flush=True)

    keys = list(out[0].keys())
    with open(odir / f"{a.split}_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    V = ["raw", "raw_clipped", "model"]
    M = ["mse", "mae", "psnr", "ssim"]

    def block(rr):
        b = {v: {mm: summarize([r[f"{v}_{mm}"] for r in rr]) for mm in M} for v in V}
        b["delta_psnr_model_vs_raw"] = summarize([r["delta_psnr"] for r in rr])
        b["delta_ssim_model_vs_raw"] = summarize([r["delta_ssim"] for r in rr])
        b["delta_psnr_model_vs_raw_clipped"] = summarize([r["model_psnr"] - r["raw_clipped_psnr"] for r in rr])
        b["delta_ssim_model_vs_raw_clipped"] = summarize([r["model_ssim"] - r["raw_clipped_ssim"] for r in rr])
        b["mse_reduction_pct_of_mean_mse"] = 100 * (1 - b["model"]["mse"]["mean"] / b["raw"]["mse"]["mean"])
        b["mae_reduction_pct_of_mean_mae"] = 100 * (1 - b["model"]["mae"]["mean"] / b["raw"]["mae"]["mean"])
        b["mse_reduction_pct_per_image"] = summarize([r["mse_reduction_pct"] for r in rr])
        b["frac_images_model_better_psnr_than_raw"] = float(np.mean([r["delta_psnr"] > 0 for r in rr]))
        b["frac_images_model_better_ssim_than_raw"] = float(np.mean([r["delta_ssim"] > 0 for r in rr]))
        b["frac_images_model_better_psnr_than_raw_clipped"] = float(
            np.mean([r["model_psnr"] > r["raw_clipped_psnr"] for r in rr]))
        return b

    summary = {"model": str(a.model), "model_md5": model_md5, "split": a.split, "n_samples": len(out),
               "n_images": len({r["image_id"] for r in out}), "levels_P": sorted({r["P"] for r in out}),
               "evaluated_unix_time": time.time(), "eval_seconds": round(time.time() - t0),
               "metric_notes": {"domain": "linear light, inside FOV mask; PSNR data range 1",
                                "raw": "stored N/P, unclipped (primary raw baseline)",
                                "raw_clipped": "clip(N/P, 0, 1), secondary baseline",
                                "ssim": "Gaussian 11x11 sigma 1.5, K1=.01 K2=.03, L=1, mask eroded 5 px, mean of RGB",
                                "overall": "mean over all samples; each photon level weighted equally"},
               "per_level": {f"P{int(P)}": block([r for r in out if r["P"] == P]) for P in LEVELS},
               "overall": block(out), "per_level_group": {}}
    for P in LEVELS:
        for g in sorted({r["group"] for r in out}):
            gg = [r for r in out if r["P"] == P and r["group"] == g]
            if gg:
                summary["per_level_group"][f"P{int(P)}_{g}"] = {
                    "n": len(gg), **{f"{v}_{mm}": float(np.mean([r[f'{v}_{mm}'] for r in gg])) for v in V for mm in M}}
    summary["mean_inference_s_per_image"] = float(np.mean([r["inference_s"] for r in out]))
    (odir / f"{a.split}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["| Photon P | Raw PSNR | U-Net PSNR | ΔPSNR | Raw SSIM | U-Net SSIM | ΔSSIM | Raw MAE | U-Net MAE | "
             "Raw MSE | U-Net MSE | MSE reduction |", "|" + "---|" * 12]
    for key, s in list(summary["per_level"].items()) + [("All", summary["overall"])]:
        g = lambda v, mm: s[v][mm]["mean"]
        lines.append(f"| {key.lstrip('P')} | {g('raw', 'psnr'):.2f} | **{g('model', 'psnr'):.2f}** | "
                     f"{s['delta_psnr_model_vs_raw']['mean']:+.2f} | {g('raw', 'ssim'):.3f} | "
                     f"**{g('model', 'ssim'):.3f}** | {s['delta_ssim_model_vs_raw']['mean']:+.3f} | "
                     f"{g('raw', 'mae'):.4f} | {g('model', 'mae'):.4f} | {g('raw', 'mse'):.3e} | "
                     f"{g('model', 'mse'):.3e} | {s['mse_reduction_pct_of_mean_mse']:.1f}% |")
    (odir / f"{a.split}_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
