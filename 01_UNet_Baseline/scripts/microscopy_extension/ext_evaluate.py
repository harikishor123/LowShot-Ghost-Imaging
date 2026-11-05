#!/usr/bin/env python3
"""Extension (broader microscopy dataset), step 4: evaluate (validation first, then the test ONCE).

    python ext_evaluate.py --target clean                              # validation; freezes checkpoints (MD5)
    python ext_evaluate.py --target clean --split test --final-test    # once, after validation
    (same with --target dgi10k)

target clean  (as onion Phase 2, orig_evaluate.py): DGI-N and AI-original-N scored against the original clean
              object; criterion: AI-original-N non-inferior to DGI-10000.
target dgi10k (as onion Phase 1, final_evaluate.py): conventional DGI-N, smoothed DGI-N and AI-N scored against
              the 10k DGI of the same acquisition; criterion: non-inferior to an independent 10k acquisition
              (seed + 7000). Also scored against the clean object (context only).
Criterion (protocol.json): lower bound of the paired 95% bootstrap CI (2,000 resamples over images, seeds kept
together) >= -0.02 SSIM and >= -0.01 Pearson r. Applied to all images (primary) and to each subset (secondary).
"""

from __future__ import annotations

import argparse
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

from ext_common import (EVAL_SEEDS, LOW, OUT, POOL_NPY, PROTOCOL, RETEST_OFFSET, SHOTS, SUBSETS, TARGETS,  # noqa: E402
                        load_objects, md5, safe_path, write_json)
from ext_train import predict  # noqa: E402
from final_build_pool import dgi_readouts  # noqa: E402
from final_evaluate import SIGMAS, edge_r  # noqa: E402
from run_generalization_validation import smooth  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.unet_gi import UNetGI  # noqa: E402
from src.visualization import GRID, INK, INK_MUTED, save_panel_grid  # noqa: E402

N_BOOT = 2000
GROUPS = ["all"] + list(SUBSETS)
COL = {"DGI": INK_MUTED, "smoothed DGI": "#2a78d6", "AI": "#eb6834", "AI-original": "#eb6834"}


def metrics(img: np.ndarray, ref: np.ndarray) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = evaluate(ref.astype(np.float64), img.astype(np.float64))
    return {"SSIM": m["SSIM"], "correlation": m["Pearson_r"], "PSNR": m["PSNR"], "MSE": m["MSE"], "edge_r": edge_r(img, ref)}


def load_models(target: str, device: str):
    models, info = {}, {}
    for n in LOW:
        p = OUT / "models" / f"target_{target}" / f"model_{n}" / "checkpoints" / "best.pt"
        ck = torch.load(p, weights_only=False)
        m = UNetGI(base=ck["config"]["base"]).to(device)
        m.load_state_dict(ck["model"])
        models[n], info[n] = m, {"checkpoint": str(p.relative_to(ROOT)), "best_epoch": int(ck["epoch"]), "md5": md5(p)}
    return models, info


def tune_sigmas(train_n: int) -> tuple[dict, pd.DataFrame]:
    """sigma per N minimising MSE(smooth(DGI-N), DGI-10000 same sequence) on training realisations k = 0, 1 (as onion)."""
    pool = np.load(POOL_NPY, mmap_mode="r")
    rows, best = [], {}
    for n in LOW:
        x = np.ascontiguousarray(pool[:train_n, :2, SHOTS.index(n)]).reshape(-1, 1, 64, 64)
        y = np.ascontiguousarray(pool[:train_n, :2, SHOTS.index(10000)]).reshape(-1, 1, 64, 64)
        m = [float(((smooth(x, s) - y) ** 2).mean()) for s in SIGMAS]
        rows += [{"shots": n, "sigma": s, "train_MSE_vs_10k": v} for s, v in zip(SIGMAS, m)]
        best[n] = SIGMAS[int(np.argmin(m))]
    return best, pd.DataFrame(rows)


def boot(d: np.ndarray, rng) -> tuple[float, float, float]:
    """d: (images, seeds). Mean and 95% CI over images (seeds kept together), as onion bootstrap_diff."""
    n = d.shape[0]
    means = np.array([np.nanmean(d[rng.integers(0, n, n)]) for _ in range(N_BOOT)])
    return float(np.nanmean(d)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(per: pd.DataFrame, a: str, b: str, ids, rng, mets) -> dict:
    A = per[per.label == a].set_index(["image_id", "seed"])
    B = per[per.label == b].set_index(["image_id", "seed"])
    out = {}
    for k in mets:
        d = (A[k] - B[k]).unstack("seed").loc[ids].to_numpy()
        mean, lo, hi = boot(d, rng)
        better = (d < 0) if k == "MSE" else (d > 0)
        out[k] = {"mean": mean, "std": float(np.nanstd(d, ddof=1)), "ci_low": lo, "ci_high": hi,
                  "pct_A_better": float(better.mean() * 100), "pct_A_ge": float(((d <= 0) if k == "MSE" else (d >= 0)).mean() * 100)}
    return out


def plot_curves(summ: pd.DataFrame, comparator: str, methods: list[str], path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    c = summ[summ.label == comparator].iloc[0]
    for ax, (k, lab) in zip(axes, [("SSIM", "SSIM"), ("correlation", "Pearson r"), ("PSNR", "PSNR (dB)"), ("edge_r", "Edge r")]):
        for meth in methods:
            g = summ[(summ.method == meth) & summ.shot_count.isin(LOW)].sort_values("shot_count")
            ax.errorbar(g.shot_count, g[f"{k}_mean"], yerr=g[f"{k}_std"], color=COL[meth], lw=2, marker="o", ms=7,
                        mec="white", capsize=3, ls="--" if meth != methods[-1] else "-", label=f"{meth}-N")
        ax.axhline(c[f"{k}_mean"], color=INK if comparator.startswith("independent") else "#2a78d6", lw=1.6, ls=":",
                   label=f"{comparator} = {c[f'{k}_mean']:.3f}")
        ax.set_xscale("log")
        ax.set_xticks(LOW, [f"{n:,}" for n in LOW])
        ax.minorticks_off()
        ax.set_xlabel("number of measurement shots N (log)", color=INK_MUTED)
        ax.set_title(lab, fontsize=10.5, color=INK, loc="left")
        ax.grid(color=GRID, lw=0.8)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(safe_path(path), dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, choices=list(TARGETS))
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(8)
    protocol = json.loads(PROTOCOL.read_text())
    crit = protocol["criterion"]["delta"]
    pre = TARGETS[args.target]
    base = OUT / "results" / f"target_{args.target}"
    frozen_path = base / "frozen_for_test.json"
    res = base / args.split
    models, info = load_models(args.target, args.device)
    if args.split == "test":
        if not args.final_test or not frozen_path.exists():
            sys.exit("Test evaluation requires --final-test AND frozen_for_test.json from the validation run.")
        if (res / "evaluation_report.json").exists():
            sys.exit(f"The test set was already evaluated for target {args.target}; it is run once only.")
        frozen = json.loads(frozen_path.read_text())
        for n in LOW:
            assert info[n]["md5"] == frozen["models"][str(n)]["md5"], f"checkpoint for N={n} changed after freezing"
        sigma = {int(k): v for k, v in frozen.get("smoothing_sigma", {}).items()}
        d = load_objects(("test",))
    else:
        d = load_objects(("val",))
        sigma = {}
        if args.target == "dgi10k":
            sigma, tun = tune_sigmas(len(json.loads((OUT / "pool" / "pool_meta.json").read_text())["train_ids"]))
            tun.to_csv(safe_path(base / "smoothing_sigma_tuning_train.csv"), index=False)
    ids, subset, clean = d["image_id"].tolist(), d["subset"], d["object"]
    sub_of = dict(zip(ids, subset))
    ex_ids = [i for s in SUBSETS for i in np.array(ids)[subset == s][np.linspace(0, (subset == s).sum() - 1, 3).astype(int)]]
    checks = {"split": args.split, "n_images": len(ids), "n_by_subset": {s: int((subset == s).sum()) for s in SUBSETS},
              "models": info, "smoothing_sigma": sigma}

    rows, keep = [], {}
    for seed in EVAL_SEEDS:
        acq = dgi_readouts(clean, seed)                                    # clean object -> buckets -> DGI
        imgs = {}
        if args.target == "clean":
            ref = clean
            imgs.update({f"DGI-{n}": acq[:, SHOTS.index(n)] for n in SHOTS})
        else:
            ref = acq[:, SHOTS.index(10000)]
            imgs["independent 10k-10000"] = dgi_readouts(clean, seed + RETEST_OFFSET)[:, SHOTS.index(10000)]
            for n in LOW:
                imgs[f"DGI-{n}"] = acq[:, SHOTS.index(n)]
                imgs[f"smoothed DGI-{n}"] = smooth(acq[:, SHOTS.index(n)][:, None], sigma[n])[:, 0]
        for n in LOW:
            x = acq[:, SHOTS.index(n)][:, None].astype(np.float32)
            imgs[f"{pre}-{n}"] = np.clip(predict(models[n], x, args.device), 0, 1)[:, 0]
        for lab, arr in imgs.items():
            meth, n = lab.rsplit("-", 1)
            for j, image_id in enumerate(ids):
                r = {"image_id": image_id, "subset": subset[j], "seed": seed, "label": lab, "method": meth, "shot_count": int(n),
                     **metrics(arr[j], ref[j])}
                if args.target == "dgi10k":
                    r.update({f"vs_clean_{k}": v for k, v in metrics(arr[j], clean[j]).items()})
                rows.append(r)
        if seed == 2026:
            keep = imgs | {"_ref": ref}
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(safe_path(res / "per_image_results.csv"), index=False)

    mets = ["SSIM", "correlation", "PSNR", "MSE", "edge_r"] + ([f"vs_clean_{k}" for k in ("SSIM", "correlation", "PSNR", "MSE", "edge_r")]
                                                              if args.target == "dgi10k" else [])
    srows = []
    for grp in GROUPS:
        for sg, seeds in (("all seeds (2026-2029)", EVAL_SEEDS), ("seed 2026", [2026]), ("seeds 2027-2029", [2027, 2028, 2029])):
            g0 = per[per.seed.isin(seeds) & ((per.subset == grp) if grp != "all" else True)]
            for (lab, meth, n), g in g0.groupby(["label", "method", "shot_count"]):
                r = {"group": grp, "seed_group": sg, "label": lab, "method": meth, "shot_count": n, "n_pairs": len(g)}
                for k in mets:
                    r[f"{k}_mean"], r[f"{k}_std"], r[f"{k}_median"] = g[k].mean(), g[k].std(), g[k].median()
                srows.append(r)
    summ = pd.DataFrame(srows).sort_values(["group", "seed_group", "method", "shot_count"])
    summ.to_csv(safe_path(res / "aggregate_metrics.csv"), index=False)

    comparator = "DGI-10000" if args.target == "clean" else "independent 10k-10000"
    if args.target == "clean":
        cand = [f"{pre}-{n}" for n in LOW]
        extra = [(f"{pre}-{n}", f"DGI-{n}") for n in LOW] + [(f"{pre}-{b}", f"{pre}-{a}") for a, b in zip(LOW, LOW[1:])]
    else:
        cand = [f"{m}-{n}" for n in LOW for m in ("DGI", "smoothed DGI", pre)]
        extra = [(f"{pre}-{n}", f"DGI-{n}") for n in LOW]
    comp_rows, minimum = [], {}
    for grp in GROUPS:
        gids = [i for i in ids if grp == "all" or sub_of[i] == grp]
        rng = np.random.default_rng(0)
        for a, b in [(c, comparator) for c in cand] + extra:
            dd = paired(per, a, b, gids, rng, ["SSIM", "correlation", "PSNR", "MSE", "edge_r"])
            r = {"group": grp, "n_images": len(gids), "comparison": f"{a} minus {b}", "A": a, "B": b,
                 "shot_count": int(a.rsplit("-", 1)[1])}
            for k, v in dd.items():
                for s_, val in v.items():
                    r[f"{k}_{s_}"] = val
            if b == comparator:
                r["criterion_pass"] = bool(dd["SSIM"]["ci_low"] >= -crit["SSIM"] and dd["correlation"]["ci_low"] >= -crit["Pearson_r"])
            comp_rows.append(r)
        cg = pd.DataFrame([r for r in comp_rows if r["group"] == grp and r["B"] == comparator])
        minimum[grp] = {}
        for meth in sorted({a.rsplit("-", 1)[0] for a in cand}):
            ok = cg[cg.A.str.rsplit("-", n=1).str[0].eq(meth) & cg.criterion_pass].shot_count
            minimum[grp][meth] = int(ok.min()) if len(ok) else None
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(safe_path(res / "comparisons.csv"), index=False)

    methods = ["DGI", pre] if args.target == "clean" else ["DGI", "smoothed DGI", pre]
    for grp in GROUPS:
        s = summ[(summ.group == grp) & (summ.seed_group == "all seeds (2026-2029)")]
        plot_curves(s, comparator, methods, res / "figures" / f"metrics_vs_shots_{grp}.png",
                    f"Extension ({args.split}, {grp}): scored against "
                    f"{'the original clean object' if args.target == 'clean' else 'the 10k DGI of the same acquisition'}; "
                    "mean ± std over all image-sequence pairs")
    p26 = per[per.seed == 2026].set_index(["image_id", "label"])
    order = [f"{m}-{n}" for n in LOW for m in (["DGI", pre] if args.target == "clean" else ["DGI", "smoothed DGI", pre])]
    order += ["DGI-10000"] if args.target == "clean" else ["independent 10k-10000"]
    rws = []
    for image_id in ex_ids:
        j = ids.index(image_id)
        first = ("Original clean object" if args.target == "clean" else "10k DGI reference (seed 2026)", keep["_ref"][j])
        panels = [first] + [(f"{lab}\n{p26.loc[(image_id, lab)].PSNR:.2f} dB | SSIM {p26.loc[(image_id, lab)].SSIM:.3f}",
                             keep[lab][j]) for lab in order]
        rws.append({"label": f"{image_id}", "target": keep["_ref"][j], "panels": panels})
    save_panel_grid(rws, safe_path(res / "figures" / "comparison_examples.png"),
                    f"Extension ({args.split}), seed 2026; identical grey window [0, 1]; no enhancement")

    report = {"target": args.target, "split": args.split, "criterion": protocol["criterion"], "checks": checks,
              "minimum_passing_shot_count": minimum,
              "reduction_factor_vs_10000": {g: {m: (10000 / v if v else None) for m, v in mm.items()} for g, mm in minimum.items()},
              "example_ids": ex_ids, "runtime_s": round(time.time() - t0, 1)}
    write_json(res / "evaluation_report.json", report)
    if args.split == "val":
        write_json(frozen_path, {"frozen_after_validation": True, "models": {str(n): info[n] for n in LOW},
                                 "smoothing_sigma": {str(n): v for n, v in sigma.items()}, "eval_seeds": EVAL_SEEDS,
                                 "criterion": protocol["criterion"], "validation_minimum_passing_shot_count": minimum,
                                 "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    pd.set_option("display.width", 240)
    s = summ[summ.seed_group == "all seeds (2026-2029)"]
    print(s[["group", "label", "n_pairs"] + [f"{k}_mean" for k in ("SSIM", "correlation", "PSNR", "edge_r")]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    show = ["group", "comparison"] + [f"{k}_{x}" for k in ("SSIM", "correlation") for x in ("mean", "ci_low")] + ["criterion_pass"]
    print(comp[comp.B == comparator][show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("minimum passing N:", json.dumps(minimum), f"| {report['runtime_s']} s -> {res}")


if __name__ == "__main__":
    main()
