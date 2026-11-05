"""Report-only figures for Phase 2, built from EXISTING outputs (no model inference, no recomputation).
- Fig A: panels cropped from figures/test/comparison_all_examples.png; scores read from per_image_results.csv.
- Fig B: metrics vs shots re-plotted from results/test/aggregate_metrics.csv (all seeds, 296 pairs).
Writes only to thesis/figures/."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]          # project root (this file is thesis/figures/make_report_figs.py)
P2 = ROOT / "outputs" / "phase_final_original_target"
OUT = ROOT / "thesis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- Fig A -------------------------------------------------------------
if not (P2 / "figures/test/comparison_all_examples.png").exists():   # onion figures are withheld from the repository
    raise SystemExit("missing outputs/phase_final_original_target/figures/test/comparison_all_examples.png "
                     "(onion image, withheld pending permission; see 07_Reproducibility_and_Audit/EXCLUDED_PENDING_PERMISSION.md)")
src = np.asarray(Image.open(P2 / "figures/test/comparison_all_examples.png").convert("RGB"))
cols = [(32, 410), (421, 799), (810, 1188), (1200, 1578), (1589, 1967), (1979, 2357),
        (2368, 2746), (2758, 3136), (3147, 3525), (3536, 3914)]
rows = [(81, 459), (530, 908), (979, 1356), (1427, 1805), (1876, 2254)]
ids = ["A_115", "A_258", "A_78", "D_23", "F_9"]
pick = [(0, "Original clean object", None), (9, "DGI-10000", "DGI-10000"),
        (4, "AI-original-1000", "AI-original-1000"), (6, "AI-original-2000", "AI-original-2000"),
        (8, "AI-original-5000", "AI-original-5000")]
pi = pd.read_csv(P2 / "results/test/per_image_results.csv")
pi = pi[pi.seed == 2026].set_index(["image_id", "label"])

fig, axes = plt.subplots(len(ids), len(pick), figsize=(12.5, 14.2))
for r, iid in enumerate(ids):
    r0, r1 = rows[r]
    for c, (ci, head, lab) in enumerate(pick):
        c0, c1 = cols[ci]
        ax = axes[r, c]
        ax.imshow(src[r0 + 2:r1 - 2, c0 + 2:c1 - 2])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if lab is None:
            sub = "ideal match: SSIM 1, r 1\nPSNR infinite"
        else:
            row = pi.loc[(iid, lab)]
            sub = f"SSIM {row.SSIM:.3f} | r {row.correlation:.3f}\nPSNR {row.PSNR:.2f} dB"
        ax.set_title(f"{head}\n{sub}" if r == 0 else sub, fontsize=9.5)
        if c == 0:
            ax.set_ylabel(iid, fontsize=11)
fig.tight_layout(h_pad=1.2, w_pad=0.6)
fig.savefig(OUT / "phase2_clean_object_comparison.png", dpi=150)
plt.close(fig)

# ---- Fig B -------------------------------------------------------------
ag = pd.read_csv(P2 / "results/test/aggregate_metrics.csv")
ag = ag[ag.seed_group.str.startswith("all seeds")]
shots = [500, 1000, 2000, 5000]
mets = [("SSIM", "SSIM"), ("correlation", "Pearson r"), ("PSNR", "PSNR (dB)"), ("edge_r", "Edge r")]
fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))
for ax, (k, name) in zip(axes, mets):
    for meth, lab, col, ls in [("DGI", "Conventional DGI-N", "#4a4a4a", "--"), ("AI-original", "AI-original-N", "#e8613c", "-")]:
        d = ag[(ag.method == meth) & (ag.shot_count.isin(shots))].sort_values("shot_count")
        ax.errorbar(d.shot_count, d[f"{k}_mean"], yerr=d[f"{k}_std"], color=col, ls=ls, marker="o", lw=2, capsize=4, label=lab)
    ref = ag[ag.label == "DGI-10000"][f"{k}_mean"].item()
    ax.axhline(ref, color="#2a6fdb", ls=":", lw=1.8, label=f"DGI-10000 = {ref:.3f}" if k != "PSNR" else f"DGI-10000 = {ref:.2f} dB")
    ax.set_xscale("log"); ax.set_xticks(shots); ax.set_xticklabels(["500", "1,000", "2,000", "5,000"]); ax.minorticks_off()
    ax.set_xlabel("number of measurement shots N (log scale)")
    ax.set_title(f"{name} vs original clean object", loc="left", fontsize=11)
    ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "phase2_metrics_vs_shots.png", dpi=150)
plt.close(fig)
print("ok")
