"""PART D: verified result figures. Original [F] | noisy DGI-1000 (exact recomputation, float) | trained U-Net-1000
(frozen test output, decoded 8-bit) | noisy DGI-10000 (exact) | |U-Net - original|. Metrics under each image are
computed here (SSIM, PSNR, NCC, MSE; [0,1] range, 64x64, data_range 1); U-Net values also cross-checked vs frozen CSV."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from skimage.metrics import structural_similarity
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; OUT = HERE.parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "audit/poisson_visual_analysis/scripts")]
import ext_poisson_common as C
from panel_decode import load_all, ROW_IDS
P = load_all(); E = np.load(OUT / "exact_test_dgi_six_images.npz")
ids = list(E["ids"]); lv = list(E["levels"]); sh = list(E["shots"]); dg = E["dgi"]
te = C.load_objects(("test",)); obj = dict(zip(te["image_id"], te["object"]))
LN = {"p00": "0%", "p05": "5%", "p10": "10%", "p30": "30%"}
def M(ref, x):
    ref = ref.astype(np.float64); x = x.astype(np.float64); mse = ((ref - x) ** 2).mean()
    a = ref.ravel() - ref.mean(); b = x.ravel() - x.mean()
    return f"SSIM {structural_similarity(ref, x, data_range=1.0):.3f}  PSNR {10*np.log10(1/mse):.2f} dB\nNCC {a@b/np.sqrt((a@a)*(b@b)):.3f}  MSE {mse:.4f}"
rows_out = []
for nm, sel in (("blood", ids[:3]), ("tissue", ids[3:])):
    rows = [(i, l) for i in sel for l in lv]
    fig, axes = plt.subplots(len(rows), 5, figsize=(5 * 2.7, len(rows) * 3.75), layout="constrained")
    for r, (i, l) in enumerate(rows):
        k, li = ids.index(i), lv.index(l); o = obj[i]
        d1, d10, u = dg[li, k, sh.index(1000)], dg[li, k, sh.index(10000)], P[(l, i, "AI-original-1000")]
        pre = "Noisy " if l != "p00" else ""
        panels = [(o, "Original [F]", "reference (vs itself: SSIM 1, NCC 1, MSE 0)"),
                  (d1, f"{pre}DGI-1000 [exact recomputation]", "vs original:\n" + M(o, d1)),
                  (u, "Trained U-Net-1000 [frozen output, 8-bit]", "vs original:\n" + M(o, u)),
                  (d10, f"{pre}DGI-10000 [exact recomputation]", "vs original:\n" + M(o, d10)),
                  (np.abs(u - o), "|U-Net-1000 − original|  (0–0.5)", "U-Net-1000 vs DGI-10000:\n" + M(d10, u))]
        for c, (a, t, x) in enumerate(panels):
            ax = axes[r, c]; err = c == 4
            ax.imshow(a, cmap="magma" if err else "gray", vmin=0, vmax=0.5 if err else 1, interpolation="nearest")
            ax.set_title(t, fontsize=7.6); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(x, fontsize=7.2, linespacing=1.15)
        axes[r, 0].set_ylabel(f"{i}\n{LN[l]} noise · 1,000 shots", fontsize=8.5)
    fig.suptitle(f"Verified results, {'blood smears' if nm == 'blood' else 'H&E tissue'} (test set, sequence 2026). Metrics computed independently on 64 × 64 images in [0, 1] "
                 "(SSIM data_range 1). Grey 0–1; error map 0–0.5.", fontsize=10)
    fig.savefig(OUT / f"figures/D_verified_{nm}.png", dpi=150, facecolor="white"); plt.close(fig)
print("ok")
