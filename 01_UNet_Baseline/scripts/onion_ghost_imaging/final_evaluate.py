#!/usr/bin/env python3
"""Final experiment, step 3: evaluate AI-N against the existing 10,000-shot reference.

    python final_evaluate.py                          # validation (75 images) - development decisions
    python final_evaluate.py --split test --final-test   # ONLY after validation froze everything

Per image and evaluation seed (2026 = existing reference file; 2027-2029 = unseen sequences with their
own 10k reconstruction from the identical pipeline), against the 10k reference of the SAME acquisition:
    conventional DGI-N, smoothed DGI-N (sigma tuned on training sequences), AI-N, and an independent
    10k acquisition (retest seed = eval seed + 7000) as the comparability yardstick.
Metrics: PSNR, SSIM, MSE, Pearson r, edge r (Pearson r of Sobel gradient-magnitude maps); plus the
same image scored against the clean onion image (context only).
The comparability rule is read from protocol.json (fixed before training).
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
from skimage.filters import sobel  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.ghost_imaging import load_object  # noqa: E402
from src.gi_unet_data import DGI_NPZ, load_pairs, read_split_ids  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.unet_gi import UNetGI  # noqa: E402
from src.visualization import GRID, INK, INK_MUTED, save_panel_grid  # noqa: E402
from final_build_pool import SHOTS, dgi_readouts  # noqa: E402
from run_generalization_validation import smooth  # noqa: E402
from train_unet_gi import predict  # noqa: E402

OUT = ROOT / "outputs" / "phase_final_low_shot_to_10k_reference"
LOW = [500, 1000, 2000, 5000]
EVAL_SEEDS = [2026, 2027, 2028, 2029]
RETEST_OFFSET = 7000
SIGMAS = [0.0] + [0.25 + 0.125 * k for k in range(15)] + [2.25, 2.5, 3.0, 3.5, 4.0]
N_BOOT = 2000
COL = {"conventional DGI": INK_MUTED, "smoothed DGI": "#2a78d6", "AI": "#eb6834"}


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def edge_r(a: np.ndarray, b: np.ndarray) -> float:
    ea, eb = sobel(a.astype(np.float64)).ravel(), sobel(b.astype(np.float64)).ravel()
    if ea.std() == 0 or eb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ea, eb)[0, 1])


def score(img: np.ndarray, ref: np.ndarray, clean: np.ndarray) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = evaluate(ref.astype(np.float64), img.astype(np.float64))
        c = evaluate(clean.astype(np.float64), img.astype(np.float64))
    return {"PSNR": m["PSNR"], "SSIM": m["SSIM"], "MSE": m["MSE"], "correlation": m["Pearson_r"],
            "edge_r": edge_r(img, ref), "vs_clean_PSNR": c["PSNR"], "vs_clean_SSIM": c["SSIM"],
            "vs_clean_correlation": c["Pearson_r"]}


def tune_sigmas() -> dict:
    """sigma per N minimising MSE(smooth(DGI-N), DGI-10000 same sequence) on training realisations k=0,1."""
    pool = np.load(OUT / "pool_train.npy", mmap_mode="r")
    rows, best = [], {}
    for n in LOW:
        x = np.ascontiguousarray(pool[:, :2, SHOTS.index(n)]).reshape(-1, 1, 64, 64)
        y = np.ascontiguousarray(pool[:, :2, SHOTS.index(10000)]).reshape(-1, 1, 64, 64)
        m = [float(((smooth(x, s) - y) ** 2).mean()) for s in SIGMAS]
        rows += [{"shots": n, "sigma": s, "train_MSE_vs_10k": v} for s, v in zip(SIGMAS, m)]
        best[n] = SIGMAS[int(np.argmin(m))]
    pd.DataFrame(rows).to_csv(OUT / "smoothing_sigma_tuning_train.csv", index=False)
    return best


def load_models():
    models, info = {}, {}
    for n in LOW:
        p = OUT / "models" / f"model_{n}" / "checkpoints" / "best.pt"
        ck = torch.load(p, weights_only=False)
        m = UNetGI(base=ck["config"]["base"])
        m.load_state_dict(ck["model"])
        models[n] = m
        info[n] = {"checkpoint": str(p.relative_to(ROOT)), "best_epoch": int(ck["epoch"]), "md5": md5(p)}
    return models, info


def bootstrap_diff(diff: np.ndarray, rng) -> tuple[float, float, float]:
    """diff: (images, seeds). Mean and 95% CI over images (seeds kept together)."""
    n = diff.shape[0]
    means = np.array([np.nanmean(diff[rng.integers(0, n, n)]) for _ in range(N_BOOT)])
    return float(np.nanmean(diff)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def plot_curve(summ, key, label, path, title, retest_val):
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for m in ("conventional DGI", "smoothed DGI", "AI"):
        g = summ[summ.method == m].sort_values("shot_count")
        ax.errorbar(g.shot_count, g[f"{key}_mean"], yerr=g[f"{key}_std"], color=COL[m], lw=2, marker="o", ms=7,
                    mec="white", capsize=3, label=m + (" (AI-N)" if m == "AI" else " N-shot"))
    ax.axhline(retest_val, color=INK, lw=1.4, ls="--", label=f"independent 10k acquisition = {retest_val:.3f}")
    ax.set_xscale("log")
    ax.set_xticks(LOW + [10000], [f"{n:,}" for n in LOW + [10000]])
    ax.minorticks_off()
    ax.set_xlabel("number of measurements N (log)", color=INK_MUTED)
    ax.set_title(f"{label} vs the 10,000-shot reference (mean ± std)", fontsize=10.5, color=INK, loc="left")
    ax.grid(color=GRID, lw=0.8)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(colors=INK_MUTED)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(title, fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final-test", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    protocol = json.loads((OUT / "protocol.json").read_text())
    crit = protocol["comparable_criterion"]["delta"]
    torch.set_num_threads(8)
    frozen_path = OUT / "frozen_for_test.json"

    if args.split == "test":
        if not args.final_test or not frozen_path.exists():
            sys.exit("Test evaluation requires --final-test AND frozen_for_test.json from the completed validation run.")
        frozen = json.loads(frozen_path.read_text())
        models, info = load_models()
        for n in LOW:
            assert info[n]["md5"] == frozen["models"][str(n)]["md5"], f"checkpoint for N={n} changed after freezing"
        sigma = {int(k): v for k, v in frozen["smoothing_sigma"].items()}
        x0, clean, ids = load_pairs("test", 1000, allow_test=True)
    else:
        models, info = load_models()
        sigma = tune_sigmas()
        x0, clean, ids = load_pairs("val", 1000)
    out = OUT / ("test" if args.split == "test" else "validation")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    clean = clean[:, 0]
    obj = np.stack([load_object(ROOT / "data" / "v2" / "reference" / f"{i}.npz", 64)[0] for i in ids])
    checks = {"split": args.split, "n_images": len(ids), "sigma": sigma, "models": info}

    # existing reference file (seed 2026) for these images
    with np.load(DGI_NPZ, allow_pickle=True) as z:
        pos = {i: j for j, i in enumerate(z["image_id"].astype(str))}
        idx = [pos[i] for i in ids]
        existing = {n: np.clip(z[f"estimate_unclipped_{n}"][idx], 0, 1) for n in (500, 1000, 5000, 10000)}

    rows, keep = [], {}
    for seed in EVAL_SEEDS:
        acq = dgi_readouts(obj, seed)                                   # (M, 5, 64, 64)
        if seed == 2026:
            checks["seed2026_recomputed_vs_existing_file_max_abs_diff"] = {
                n: float(np.abs(acq[:, SHOTS.index(n)] - existing[n]).max()) for n in existing}
            ref = existing[10000]                                       # the existing reference itself
        else:
            ref = acq[:, SHOTS.index(10000)]
        retest = dgi_readouts(obj, seed + RETEST_OFFSET)[:, SHOTS.index(10000)]
        for j, image_id in enumerate(ids):
            rows.append({"image_id": image_id, "seed": seed, "shot_count": 10000, "method": "independent 10k (retest)",
                         **score(retest[j], ref[j], clean[j])})
            rows.append({"image_id": image_id, "seed": seed, "shot_count": 10000, "method": "10k reference (vs itself)",
                         **score(ref[j], ref[j], clean[j])})
        for n in LOW:
            x = acq[:, SHOTS.index(n)][:, None]
            methods = {"conventional DGI": x[:, 0], "smoothed DGI": smooth(x, sigma[n])[:, 0],
                       "AI": np.clip(predict(models[n], x.astype(np.float32)), 0, 1)[:, 0]}
            for mth, imgs in methods.items():
                for j, image_id in enumerate(ids):
                    rows.append({"image_id": image_id, "seed": seed, "shot_count": n, "method": mth,
                                 **score(imgs[j], ref[j], clean[j])})
                if seed == 2026:
                    keep[(n, mth)] = imgs
        if seed == 2026:
            keep["ref"], keep["retest"] = ref, retest
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(out / "per_image_results.csv", index=False)

    # ---- summaries -------------------------------------------------------------------------
    mets = ["PSNR", "SSIM", "MSE", "correlation", "edge_r", "vs_clean_PSNR", "vs_clean_SSIM", "vs_clean_correlation"]
    groups = {"all eval seeds (2026-2029)": EVAL_SEEDS, "existing reference (2026)": [2026], "unseen sequences (2027-2029)": [2027, 2028, 2029]}
    srows = []
    for gname, seeds in groups.items():
        g0 = per[per.seed.isin(seeds) & (per.method != "10k reference (vs itself)")]
        for (n, m), g in g0.groupby(["shot_count", "method"]):
            r = {"seed_group": gname, "shot_count": n, "method": m, "n_pairs": len(g)}
            for k in mets:
                r[f"{k}_mean"], r[f"{k}_std"], r[f"{k}_median"] = g[k].mean(), g[k].std(), g[k].median()
            srows.append(r)
    summ = pd.DataFrame(srows)
    summ.to_csv(out / "summary.csv", index=False)

    # ---- comparability rule (protocol) -------------------------------------------------------
    rng = np.random.default_rng(0)
    comp_rows = []
    ret = per[per.method == "independent 10k (retest)"].pivot(index="image_id", columns="seed", values=["SSIM", "correlation", "PSNR", "edge_r"])
    for n in LOW:
        for m in ("conventional DGI", "smoothed DGI", "AI"):
            cur = per[(per.method == m) & (per.shot_count == n)].pivot(index="image_id", columns="seed",
                                                                         values=["SSIM", "correlation", "PSNR", "edge_r"])
            r = {"shot_count": n, "method": m}
            for k, key in (("SSIM", "SSIM"), ("correlation", "Pearson_r"), ("PSNR", None), ("edge_r", None)):
                d = (cur[k] - ret[k]).loc[ids].to_numpy()
                mean, lo, hi = bootstrap_diff(d, rng)
                r[f"{k}_diff_vs_retest_mean"], r[f"{k}_ci_low"], r[f"{k}_ci_high"] = mean, lo, hi
                r[f"{k}_pct_pairs_ge_retest"] = float((d >= 0).mean() * 100)
            r["comparable"] = bool(r["SSIM_ci_low"] >= -crit["SSIM"] and r["correlation_ci_low"] >= -crit["Pearson_r"])
            comp_rows.append(r)
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(out / "comparability.csv", index=False)
    ok_ai = comp[(comp.method == "AI") & comp.comparable].shot_count.tolist()
    min_n = int(min(ok_ai)) if ok_ai else None
    ok_conv = comp[(comp.method != "AI") & comp.comparable][["method", "shot_count"]].to_dict("records")

    # ---- plots ---------------------------------------------------------------------------------
    sa = summ[summ.seed_group == "all eval seeds (2026-2029)"]
    rt = sa[sa.method == "independent 10k (retest)"].iloc[0]
    sub = f"{args.split} images x seeds 2026-2029; reference = 10k-shot DGI of the same acquisition (2026 = existing file)"
    for k, f, lab in (("SSIM", "ssim_vs_shots.png", "SSIM"), ("PSNR", "psnr_vs_shots.png", "PSNR (dB)"),
                      ("correlation", "correlation_vs_shots.png", "Pearson r"), ("edge_r", "edge_correlation_vs_shots.png", "Edge r (Sobel)")):
        plot_curve(sa, k, lab, out / f, sub, float(rt[f"{k}_mean"]))
    ex = json.loads((ROOT / "outputs" / "phase2b_unet" / "model_A_dgi500" / "full" / "training_report.json").read_text())["val_example_ids"] \
        if args.split == "val" else list(np.array(ids)[np.linspace(0, len(ids) - 1, 5).astype(int)])
    p26 = per[per.seed == 2026].set_index(["image_id", "shot_count", "method"])
    for n in LOW:
        rws = []
        for image_id in ex:
            j = ids.index(image_id)
            c = p26.loc[(image_id, n, "conventional DGI")]
            a = p26.loc[(image_id, n, "AI")]
            rws.append({"label": f"{image_id}", "target": keep["ref"][j], "panels": [
                ("10k reference (existing, seed 2026)", keep["ref"][j]),
                (f"conventional DGI-{n}\nvs ref {c.PSNR:.2f} dB | SSIM {c.SSIM:.3f}", keep[(n, "conventional DGI")][j]),
                (f"AI-{n}\nvs ref {a.PSNR:.2f} dB | SSIM {a.SSIM:.3f}", keep[(n, "AI")][j])]})
        save_panel_grid(rws, out / "figures" / f"comparison_N{n}.png",
                        f"N = {n:,}: 10k reference | conventional N-shot | AI-N  ({args.split}, seed 2026 = existing reference)\n"
                        "identical gray window [0, 1] for every panel; no enhancement")

    result = {"split": args.split, "protocol_criterion": protocol["comparable_criterion"], "checks": checks,
              "minimum_successful_shot_count_AI": min_n,
              "measurement_reduction_factor": (10000 / min_n) if min_n else None,
              "conventional_methods_meeting_criterion": ok_conv, "comparability": comp.to_dict("records"),
              "runtime_s": round(time.time() - t0, 1)}
    (out / "evaluation_report.json").write_text(json.dumps(result, indent=2, default=float))
    if args.split == "val":
        frozen = {"frozen_after_validation": True, "models": {str(n): info[n] for n in LOW},
                  "smoothing_sigma": {str(n): sigma[n] for n in LOW}, "eval_seeds": EVAL_SEEDS,
                  "retest_offset": RETEST_OFFSET, "criterion": protocol["comparable_criterion"],
                  "validation_minimum_successful_shot_count": min_n}
        frozen_path.write_text(json.dumps(frozen, indent=2))

    pd.set_option("display.width", 220)
    cols = ["shot_count", "method", "PSNR_mean", "PSNR_std", "SSIM_mean", "SSIM_std", "MSE_mean", "correlation_mean",
            "edge_r_mean", "vs_clean_PSNR_mean", "vs_clean_SSIM_mean"]
    for gname in groups:
        print(f"\n{gname}:\n" + summ[summ.seed_group == gname][cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\ncomparability (paired vs independent 10k; 95% bootstrap CI):\n" + comp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nchecks:", json.dumps({k: v for k, v in checks.items() if k != "models"}, indent=1, default=float))
    print(f"\nminimum successful AI shot count: {min_n}  -> reduction {result['measurement_reduction_factor']}")
    print(f"{result['runtime_s']} s -> {out}")


if __name__ == "__main__":
    main()
