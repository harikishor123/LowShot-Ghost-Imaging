#!/usr/bin/env python3
"""Phase 2 (original clean object target), step 2: evaluate against the ORIGINAL CLEAN object.

    python orig_evaluate.py                              # validation (75 images); freezes models for test
    python orig_evaluate.py --split test --final-test    # 74 test images, once, after validation froze everything

For each image and evaluation sequence (2026-2029, same as Phase 1):
    original clean object -> simulated buckets -> DGI-N -> AI-original-N
and, as conventional baselines, DGI-500/1000/2000/5000/10000. Every reconstruction is scored against the
original clean object, which is never an input. Metrics: SSIM, Pearson r, PSNR, MSE, edge r (Sobel).
Fidelity criterion (protocol.json, fixed before training): AI-original-N is non-inferior to DGI-10000
(paired 95% bootstrap lower bound >= -0.02 SSIM and >= -0.01 r).
Outputs only under outputs/phase_final_original_target/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.ghost_imaging import load_object  # noqa: E402
from src.gi_unet_data import DGI_NPZ, read_split_ids  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.unet_gi import UNetGI  # noqa: E402
from src.visualization import GRID, INK, INK_MUTED, save_panel_grid  # noqa: E402
from final_build_pool import SHOTS, dgi_readouts  # noqa: E402
from final_evaluate import edge_r  # noqa: E402
from train_unet_gi import predict  # noqa: E402

OUT = ROOT / "outputs" / "phase_final_original_target"
LOW = [500, 1000, 2000, 5000]
EVAL_SEEDS = [2026, 2027, 2028, 2029]
N_BOOT = 2000
MET = ["SSIM", "correlation", "PSNR", "MSE", "edge_r"]


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def score(img: np.ndarray, clean: np.ndarray) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = evaluate(clean.astype(np.float64), img.astype(np.float64))
    return {"SSIM": m["SSIM"], "correlation": m["Pearson_r"], "PSNR": m["PSNR"], "MSE": m["MSE"],
            "edge_r": edge_r(img, clean)}


def load_models():
    models, info = {}, {}
    for n in LOW:
        p = OUT / "models" / f"model_{n}" / "checkpoints" / "best.pt"
        ck = torch.load(p, weights_only=False)
        m = UNetGI(base=ck["config"]["base"])
        m.load_state_dict(ck["model"])
        models[n], info[n] = m, {"checkpoint": str(p.relative_to(ROOT)), "best_epoch": int(ck["epoch"]), "md5": md5(p)}
    return models, info


def paired(per, a, b, ids, rng):
    """Paired differences a - b per (image, seed); bootstrap over images (seeds kept together)."""
    out = {}
    A = per[per.label == a].set_index(["image_id", "seed"])
    B = per[per.label == b].set_index(["image_id", "seed"])
    for k in MET:
        d = (A[k] - B[k]).unstack("seed").loc[ids].to_numpy()
        boots = np.array([np.nanmean(d[rng.integers(0, len(ids), len(ids))]) for _ in range(N_BOOT)])
        better = (d < 0) if k == "MSE" else (d > 0)
        out[k] = {"mean": float(np.nanmean(d)), "std": float(np.nanstd(d, ddof=1)), "ci_low": float(np.percentile(boots, 2.5)),
                  "ci_high": float(np.percentile(boots, 97.5)), "pct_A_better": float(better.mean() * 100)}
    return out


def plot_curves(summ, path, title):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    d10 = summ[summ.label == "DGI-10000"].iloc[0]
    for ax, (k, lab) in zip(axes, [("SSIM", "SSIM"), ("correlation", "Pearson r"), ("PSNR", "PSNR (dB)"), ("edge_r", "Edge r")]):
        for meth, col in (("DGI", INK_MUTED), ("AI-original", "#eb6834")):
            g = summ[(summ.method == meth) & summ.shot_count.isin(LOW)].sort_values("shot_count")
            ax.errorbar(g.shot_count, g[f"{k}_mean"], yerr=g[f"{k}_std"], color=col, lw=2, marker="o", ms=7, mec="white",
                        capsize=3, ls="--" if meth == "DGI" else "-", label=("DGI-N" if meth == "DGI" else "AI-original-N"))
        ax.axhline(d10[f"{k}_mean"], color="#2a78d6", lw=1.6, ls=":", label=f"DGI-10000 = {d10[f'{k}_mean']:.3f}")
        ax.set_xscale("log")
        ax.set_xticks(LOW, [f"{n:,}" for n in LOW])
        ax.minorticks_off()
        ax.set_xlabel("number of measurements N (log)", color=INK_MUTED)
        ax.set_title(f"{lab} vs original clean object", fontsize=10.5, color=INK, loc="left")
        ax.grid(color=GRID, lw=0.8)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final-test", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(8)
    protocol = json.loads((OUT / "reports" / "protocol.json").read_text())
    frozen_path = OUT / "reports" / "frozen_for_test.json"
    ids_all = read_split_ids()
    models, info = load_models()
    if args.split == "test":
        if not args.final_test or not frozen_path.exists():
            sys.exit("Test evaluation requires --final-test AND reports/frozen_for_test.json from the validation run.")
        frozen = json.loads(frozen_path.read_text())
        for n in LOW:
            assert info[n]["md5"] == frozen["models"][str(n)]["md5"], f"checkpoint for N={n} changed after freezing"
        ids = ids_all["test"]
        ex_ids = [ids[i] for i in np.linspace(0, len(ids) - 1, 5).astype(int)]        # same rule as Phase 1 test figures
    else:
        ids = ids_all["val"]
        ex_ids = json.loads((ROOT / "outputs" / "phase2b_unet" / "model_A_dgi500" / "full" / "training_report.json")
                            .read_text())["val_example_ids"]
    res_dir, fig_dir = OUT / "results" / args.split, OUT / "figures" / args.split
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    clean = np.stack([load_object(ROOT / "data" / "v2" / "reference" / f"{i}.npz", 64)[0] for i in ids])
    checks = {"split": args.split, "n_images": len(ids), "models": info}

    with np.load(DGI_NPZ, allow_pickle=True) as z:      # consistency with the existing seed-2026 DGI file
        pos = {i: j for j, i in enumerate(z["image_id"].astype(str))}
        existing = {n: np.clip(z[f"estimate_unclipped_{n}"][[pos[i] for i in ids]], 0, 1) for n in (500, 1000, 5000, 10000)}

    rows, keep = [], {}
    for seed in EVAL_SEEDS:
        acq = dgi_readouts(clean, seed)                                   # clean object -> buckets -> DGI (no AI input of clean)
        if seed == 2026:
            checks["seed2026_dgi_vs_existing_file_max_abs_diff"] = {n: float(np.abs(acq[:, SHOTS.index(n)] - existing[n]).max())
                                                                   for n in existing}
        imgs = {f"DGI-{n}": acq[:, SHOTS.index(n)] for n in SHOTS}
        for n in LOW:
            imgs[f"AI-original-{n}"] = np.clip(predict(models[n], acq[:, SHOTS.index(n)][:, None].astype(np.float32)), 0, 1)[:, 0]
        for label, arr in imgs.items():
            meth, n = label.rsplit("-", 1)
            for j, image_id in enumerate(ids):
                rows.append({"image_id": image_id, "seed": seed, "label": label, "method": meth, "shot_count": int(n),
                             **score(arr[j], clean[j])})
        if seed == 2026:
            keep = imgs
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(res_dir / "per_image_results.csv", index=False)

    # aggregate
    def agg(df, group):
        out = []
        for (lab, meth, n), g in df.groupby(["label", "method", "shot_count"]):
            r = {"seed_group": group, "label": lab, "method": meth, "shot_count": n, "n_pairs": len(g)}
            for k in MET:
                r[f"{k}_mean"], r[f"{k}_std"], r[f"{k}_median"] = g[k].mean(), g[k].std(), g[k].median()
            out.append(r)
        return pd.DataFrame(out).sort_values(["method", "shot_count"])
    summ = agg(per, "all seeds (2026-2029)")
    summ_all = pd.concat([summ, agg(per[per.seed == 2026], "seed 2026"), agg(per[per.seed != 2026], "unseen 2027-2029")])
    summ_all.to_csv(res_dir / "aggregate_metrics.csv", index=False)
    summ[summ.method == "DGI"].to_csv(res_dir / "dgi_baseline_results.csv", index=False)

    rng = np.random.default_rng(0)
    crit = protocol["fidelity_criterion"]
    comp_rows = []
    for n in LOW:
        same = paired(per, f"AI-original-{n}", f"DGI-{n}", ids, rng)
        vs10 = paired(per, f"AI-original-{n}", "DGI-10000", ids, rng)
        for name, d in (("AI-original-N minus DGI-N (same shots)", same), ("AI-original-N minus DGI-10000", vs10)):
            r = {"comparison": name, "shot_count": n}
            for k in MET:
                for s_ in ("mean", "std", "ci_low", "ci_high", "pct_A_better"):
                    r[f"{k}_{s_}"] = d[k][s_]
            if name.endswith("DGI-10000"):
                r["high_fidelity_criterion_pass"] = bool(d["SSIM"]["ci_low"] >= -0.02 and d["correlation"]["ci_low"] >= -0.01)
            comp_rows.append(r)
    for a, b in ((500, 1000), (1000, 2000), (2000, 5000)):
        d = paired(per, f"AI-original-{b}", f"AI-original-{a}", ids, rng)
        r = {"comparison": f"AI-original-{b} minus AI-original-{a}", "shot_count": b}
        for k in MET:
            for s_ in ("mean", "std", "ci_low", "ci_high", "pct_A_better"):
                r[f"{k}_{s_}"] = d[k][s_]
        comp_rows.append(r)
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(res_dir / "ai_vs_dgi_comparison.csv", index=False)
    passing = comp[(comp.comparison == "AI-original-N minus DGI-10000") & (comp.high_fidelity_criterion_pass == True)]  # noqa: E712
    min_n = int(passing.shot_count.min()) if len(passing) else None

    # figures
    title = (f"Phase 2 ({args.split}): original clean object -> simulated GI -> DGI-N -> AI-original-N | seed 2026 | "
             "all scores vs the original clean object; identical grey window [0, 1]")
    plot_curves(summ, fig_dir / "metrics_vs_shots_vs_clean.png", title)
    pv = per[per.seed == 2026].set_index(["image_id", "label"])
    order = ["DGI-500", "AI-original-500", "DGI-1000", "AI-original-1000", "DGI-2000", "AI-original-2000",
             "DGI-5000", "AI-original-5000", "DGI-10000"]
    rws = []
    for image_id in ex_ids:
        j = ids.index(image_id)
        panels = [("Original clean object", clean[j])]
        for lab in order:
            q = pv.loc[(image_id, lab)]
            panels.append((f"{lab}\n{q.PSNR:.2f} dB | SSIM {q.SSIM:.3f}", np.clip(keep[lab][j], 0, 1)))
        rws.append({"label": image_id, "target": clean[j], "panels": panels})
        save_panel_grid([rws[-1]], fig_dir / f"comparison_{image_id}.png", title)
    save_panel_grid(rws, fig_dir / "comparison_all_examples.png", title)

    result = {"split": args.split, "checks": checks, "criterion": crit, "minimum_high_fidelity_shot_count": min_n,
              "reduction_factor_vs_10000": (10000 / min_n) if min_n else None, "example_ids": ex_ids,
              "runtime_s": round(time.time() - t0, 1)}
    (res_dir / "evaluation_report.json").write_text(json.dumps(result, indent=2, default=float))
    if args.split == "val":
        frozen_path.write_text(json.dumps({"frozen_after_validation": True, "models": {str(n): info[n] for n in LOW},
                                           "eval_seeds": EVAL_SEEDS, "criterion": crit,
                                           "validation_minimum_high_fidelity_shot_count": min_n}, indent=2))
    pd.set_option("display.width", 230)
    cols = ["label"] + [f"{k}_{s}" for k in MET for s in ("mean", "std")]
    print(summ[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    show = ["comparison", "shot_count"] + [f"{k}_{s}" for k in ("SSIM", "correlation", "PSNR", "edge_r") for s in ("mean", "ci_low", "pct_A_better")] + ["high_fidelity_criterion_pass"]
    print(comp[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(json.dumps({k: v for k, v in checks.items() if k != "models"}, indent=1))
    print(f"minimum high-fidelity N: {min_n} | {result['runtime_s']} s -> {res_dir}")


if __name__ == "__main__":
    main()
