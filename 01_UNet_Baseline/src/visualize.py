#!/usr/bin/env python3
"""Figures for a trained model: training curves, metric-vs-photon-level plots and
representative reconstructions (clean / raw photon-limited / U-Net / |error|).

    python src/visualize.py --tag model_A --model-dir models/model_A_baseline_unet --split test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from color_space import decode_uint16, linear_to_srgb, srgb_to_linear  # noqa: E402
from dataset_loader import load_pair, read_samples  # noqa: E402
from evaluate import REPRESENTATIVE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LEVELS = [10, 50, 100, 250, 500, 1000]
NAMES = {10: "1%", 50: "5%", 100: "10%", 250: "25%", 500: "50%", 1000: "100%"}
C_RAW, C_CLIP, C_MODEL = "#b5b4ae", "#eb6834", "#2a78d6"


def style(plt):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#b5b4ae", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "axes.titleweight": "bold", "axes.titlesize": 10})


def clean_ax(a):
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.grid(color="#e6e5e0", lw=0.8)
    a.set_axisbelow(True)


def disp(lin, m):
    d = linear_to_srgb(np.clip(lin, 0, 1))
    d[~m] = 0.5
    return d


def structured_window(y, m, size=256, step=32):
    """Window fully inside the mask with the most stained structure (high local contrast)."""
    lum = y @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    H, W = lum.shape
    best, arg = -1, (max(0, H // 2 - size // 2), max(0, W // 2 - size // 2))
    for yy in range(0, H - size + 1, step):
        for xx in range(0, W - size + 1, step):
            if m[yy:yy + size, xx:xx + size].mean() < 0.999:
                continue
            w = lum[yy:yy + size, xx:xx + size]
            score = np.percentile(w, 95) - np.percentile(w, 5)
            if score > best:
                best, arg = score, (yy, xx)
    return slice(arg[0], arg[0] + size), slice(arg[1], arg[1] + size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="model_A")
    ap.add_argument("--model-dir", default="models/model_A_baseline_unet")
    ap.add_argument("--split", default="test")
    ap.add_argument("--eval-dir", default="results/evaluation/model_A_test")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    edir = ROOT / a.eval_dir
    fdir = edir / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    mdir = ROOT / a.model_dir

    # ---- training curves -------------------------------------------------------------
    h = list(csv.DictReader(open(mdir / "history.csv", encoding="utf-8")))
    ep = np.array([int(r["epoch"]) + 1 for r in h])
    ts = json.load(open(mdir / "training_summary.json", encoding="utf-8"))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
    ax[0].plot(ep, [float(r["loss"]) for r in h], color="#eb6834", lw=2, label="train (patches)")
    ax[0].plot(ep, [float(r["val_loss"]) for r in h], color=C_MODEL, lw=2, label="validation (fixed patches)")
    ax[0].axvline(ts["best_epoch"], color="#52514e", ls=":", lw=1)
    ax[0].text(ts["best_epoch"], ax[0].get_ylim()[1], " best", va="top", fontsize=8, color="#52514e")
    ax[0].set_yscale("log")
    ax[0].set_title("Masked MSE loss (linear light)", loc="left")
    ax[0].set_xlabel("epoch")
    ax[0].legend(frameon=False)
    ax[1].plot(ep, [float(r["masked_psnr"]) for r in h], color="#eb6834", lw=2, label="train")
    ax[1].plot(ep, [float(r["val_masked_psnr"]) for r in h], color=C_MODEL, lw=2, label="validation")
    ax[1].set_title("PSNR on patches, all levels pooled (dB)", loc="left")
    ax[1].set_xlabel("epoch")
    ax[1].legend(frameon=False)
    hist = ts["per_level_val_psnr_history"]
    ramp = ["#86b6ef", "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    for P, c in zip(LEVELS, ramp):
        ax[2].plot([r["epoch"] for r in hist], [r[f"val_psnr_P{P}"] for r in hist], marker="o", ms=4, color=c, lw=1.8,
                   label=f"P={P}")
    ax[2].set_title("Validation PSNR per photon level (dB)", loc="left")
    ax[2].set_xlabel("epoch")
    ax[2].legend(frameon=False, ncol=2, fontsize=8)
    for x in ax:
        clean_ax(x)
    fig.savefig(fdir / "training_curves.png", dpi=130)
    plt.close(fig)

    # ---- metrics vs photon level -------------------------------------------------------
    S = json.load(open(edir / f"{a.split}_summary.json", encoding="utf-8"))
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.6), constrained_layout=True)
    for axx, (mm, lab, logy) in zip(ax.ravel(), [("psnr", "PSNR (dB, linear light)", False), ("ssim", "SSIM", False),
                                                 ("mse", "MSE (linear light)", True), ("mae", "MAE (linear light)", True)]):
        for v, c, lab_v, mk in [("raw", C_RAW, "raw photon-limited N/P", "s"),
                                ("raw_clipped", C_CLIP, "raw, clipped to [0,1]", "^"),
                                ("model", C_MODEL, "U-Net (Model A)", "o")]:
            mean = [S["per_level"][f"P{P}"][v][mm]["mean"] for P in LEVELS]
            lo = [S["per_level"][f"P{P}"][v][mm]["ci95_low"] for P in LEVELS]
            hi = [S["per_level"][f"P{P}"][v][mm]["ci95_high"] for P in LEVELS]
            axx.plot(LEVELS, mean, marker=mk, ms=7, lw=2, color=c, label=lab_v)
            axx.fill_between(LEVELS, lo, hi, color=c, alpha=0.18, lw=0)
        axx.set_xscale("log")
        if logy:
            axx.set_yscale("log")
        axx.set_xticks(LEVELS, [f"{P}\n({NAMES[P]})" for P in LEVELS])
        axx.set_xlabel("photon budget P (photons at I=1)")
        axx.set_title(f"{lab} vs photon level, {a.split} set (n={S['n_images']} images/level)", loc="left")
        clean_ax(axx)
    ax[0, 0].legend(frameon=False)
    fig.savefig(fdir / f"metrics_vs_photon_level_{a.split}.png", dpi=130)
    plt.close(fig)

    # ---- representative reconstructions -------------------------------------------------
    rows = {(r["image_id"], int(r["photon_budget_P"])): r for r in read_samples(a.split)}
    rdir = edir / "reconstructions"
    per = {r["sample_id"]: r for r in csv.DictReader(open(edir / f"{a.split}_metrics.csv", encoding="utf-8"))}
    for sid in REPRESENTATIVE.get(a.split, []):
        _, y, m = load_pair(rows[(sid, 10)])
        ys, xs = structured_window(y, m)
        fig, ax = plt.subplots(4, 6, figsize=(17, 11.8), constrained_layout=True)
        for j, P in enumerate(LEVELS):
            r = rows[(sid, P)]
            x, _, _ = load_pair(r)
            with np.load(rdir / f"{sid}__P{P:04d}.npz") as z:
                pred = srgb_to_linear(decode_uint16(z["recon_srgb16"]))
            met = per[r["sample_id"]]
            err = np.abs(pred - y).mean(2)
            ax[0, j].imshow(disp(y[ys, xs], m[ys, xs]))
            ax[0, j].set_title(f"P={P} ({NAMES[P]})", fontsize=11)
            ax[1, j].imshow(disp(x[ys, xs], m[ys, xs]), interpolation="nearest")
            ax[1, j].set_xlabel(f"PSNR {float(met['raw_clipped_psnr']):.1f} dB  SSIM {float(met['raw_clipped_ssim']):.3f}",
                                fontsize=8)
            ax[2, j].imshow(disp(pred[ys, xs], m[ys, xs]), interpolation="nearest")
            ax[2, j].set_xlabel(f"PSNR {float(met['model_psnr']):.1f} dB  SSIM {float(met['model_ssim']):.3f}", fontsize=8)
            im = ax[3, j].imshow(err[ys, xs], cmap="magma", vmin=0, vmax=0.1)
            for i in range(4):
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
        for i, lab in enumerate(["clean reference", "raw photon-limited", "U-Net reconstruction", "|U-Net - clean|"]):
            ax[i, 0].set_ylabel(lab, fontsize=11)
        fig.colorbar(im, ax=ax[3, :], fraction=0.02, label="abs error, linear (0-0.1)")
        fig.suptitle(f"{sid} [{a.split}] - 256 px region (full-image metrics in labels; raw metrics = clipped to [0,1])",
                     x=0.01, ha="left", fontweight="bold")
        fig.savefig(fdir / f"recon_{sid}.png", dpi=100)
        plt.close(fig)

        # full field of view, lowest and moderate photon level
        fig, ax = plt.subplots(2, 3, figsize=(14, 9.6), constrained_layout=True)
        for i, P in enumerate([10, 100]):
            x, _, _ = load_pair(rows[(sid, P)])
            with np.load(rdir / f"{sid}__P{P:04d}.npz") as z:
                pred = srgb_to_linear(decode_uint16(z["recon_srgb16"]))
            for j, (t, im_) in enumerate([("clean", y), (f"raw P={P}", x), (f"U-Net P={P}", pred)]):
                ax[i, j].imshow(disp(im_, m))
                ax[i, j].set_title(t)
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                if j == 0:
                    ax[i, j].add_patch(plt.Rectangle((xs.start, ys.start), 256, 256, fill=False, ec="#e34948", lw=1.2))
        fig.suptitle(f"{sid} [{a.split}] full field of view", x=0.01, ha="left", fontweight="bold")
        fig.savefig(fdir / f"recon_full_{sid}.png", dpi=90)
        plt.close(fig)
    print("figures in", fdir)


if __name__ == "__main__":
    main()
