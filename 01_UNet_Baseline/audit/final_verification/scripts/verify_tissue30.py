"""PART C: tissue (path) at 30% noise, AI-original-1000 vs noisy DGI-10000, recomputed from the frozen test per-image CSV."""
import json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]; OUT = Path(__file__).resolve().parents[1]
d = ROOT / "outputs/extension_medmnist_microscopy_poisson/results/level_p30/test"
per = pd.read_csv(d / "per_image_results.csv"); comp = pd.read_csv(d / "comparisons.csv")
p = per[per.subset == "path"]
ids = list(dict.fromkeys(p.image_id)); R = {"n_images": len(ids), "n_pairs": int((p.label == "AI-original-1000").sum())}
for lab in ("AI-original-1000", "AI-original-2000", "DGI-10000", "noise-free DGI-10000"):
    R[f"mean SSIM {lab}"] = float(p[p.label == lab].SSIM.mean()); R[f"mean r {lab}"] = float(p[p.label == lab].correlation.mean())
def paired(a, b, m):
    A = p[p.label == a].pivot(index="image_id", columns="seed", values=m).loc[ids]; B = p[p.label == b].pivot(index="image_id", columns="seed", values=m).loc[ids]
    return (A - B).to_numpy()
out = {}
for a, b in (("AI-original-1000", "DGI-10000"), ("AI-original-2000", "DGI-10000"), ("AI-original-1000", "noise-free DGI-10000")):
    res = {}
    for m, thr in (("SSIM", -0.02), ("correlation", -0.01)):
        dd = paired(a, b, m); rng = np.random.default_rng(12345)          # independent RNG: CI should agree closely, not bit-exactly
        bm = np.array([np.nanmean(dd[rng.integers(0, len(dd), len(dd))]) for _ in range(2000)])
        st = comp[(comp.group == "path") & (comp.A == a) & (comp.B == b)].iloc[0]
        res[m] = {"mean_diff": float(np.nanmean(dd)), "ci_low_independent": float(np.percentile(bm, 2.5)), "ci_high_independent": float(np.percentile(bm, 97.5)),
                  "ci_low_frozen": float(st[f"{m}_ci_low"]), "pct_pairs_AI_better": float((dd > 0).mean() * 100), "margin": thr}
    res["passes (lo>=-0.02 SSIM and lo>=-0.01 r)"] = bool(res["SSIM"]["ci_low_independent"] >= -0.02 and res["correlation"]["ci_low_independent"] >= -0.01)
    res["frozen_criterion_pass"] = bool(str(comp[(comp.group == "path") & (comp.A == a) & (comp.B == b)].iloc[0].criterion_pass) == "True")
    out[f"{a} minus {b}"] = res
R["comparisons"] = out
(OUT / "C_tissue30.json").write_text(json.dumps(R, indent=1)); print(json.dumps(R, indent=1))
