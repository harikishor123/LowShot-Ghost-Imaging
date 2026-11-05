#!/usr/bin/env python3
"""Extension with Poisson bucket noise, step 4: evaluate per noise level (validation first, then the test ONCE).

    python ext_poisson_evaluate.py --level p10                              # validation; freezes checkpoints + sigmas
    python ext_poisson_evaluate.py --level p10 --split test --final-test    # once, after validation

Everything is scored against the original clean object. Methods: noisy DGI-N (N = 500..10000), smoothed noisy
DGI-N, AI-original-N, and the noise-free DGI-10000 of the same pattern sequence (context).
Criterion (protocol.json, unchanged): paired 95% bootstrap CI lower bound >= -0.02 SSIM and >= -0.01 Pearson r
vs the noisy DGI-10000 of the same acquisition; candidates AI-original-N and smoothed DGI-N. Secondary: AI-original-N
vs the noise-free DGI-10000. All images (primary), blood and path (secondary).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import ext_evaluate as E  # noqa: E402
from ext_poisson_common import (EVAL_SEEDS, LEVELS, LOW, OUT, PROTOCOL, SHOTS, SUBSETS, load_objects, md5,  # noqa: E402
                                noisy_dgi_readouts, pool_npy, safe_path, write_json)
from ext_train import predict  # noqa: E402
from final_evaluate import SIGMAS  # noqa: E402
from run_generalization_validation import smooth  # noqa: E402
from src.unet_gi import UNetGI  # noqa: E402
from src.visualization import save_panel_grid  # noqa: E402

E.safe_path = safe_path            # E.plot_curves writes through this module's guard (only under the Poisson OUT)
GROUPS = ["all"] + list(SUBSETS)
PRE = "AI-original"
COMP = "DGI-10000"                 # noisy, same acquisition
NF = "noise-free DGI-10000"
METS = ["SSIM", "correlation", "PSNR", "MSE", "edge_r"]


def load_models(level: str, device: str):
    models, info = {}, {}
    for n in LOW:
        p = OUT / "models" / f"level_{level}" / f"model_{n}" / "checkpoints" / "best.pt"
        ck = torch.load(p, weights_only=False)
        m = UNetGI(base=ck["config"]["base"]).to(device)
        m.load_state_dict(ck["model"])
        models[n], info[n] = m, {"checkpoint": str(p.relative_to(ROOT)), "best_epoch": int(ck["epoch"]), "md5": md5(p)}
    return models, info


def tune_sigmas(level: str, clean_tr: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """sigma per N minimising MSE(smooth(noisy DGI-N), clean object) on training realisations k = 0, 1."""
    pool = np.load(pool_npy(level), mmap_mode="r")
    y = np.repeat(clean_tr[:, None], 2, axis=1).reshape(-1, 1, 64, 64)
    rows, best = [], {}
    for n in LOW:
        x = np.ascontiguousarray(pool[:, :2, LOW.index(n)]).reshape(-1, 1, 64, 64)
        m = [float(((smooth(x, s) - y) ** 2).mean()) for s in SIGMAS]
        rows += [{"shots": n, "sigma": s, "train_MSE_vs_clean": v} for s, v in zip(SIGMAS, m)]
        best[n] = SIGMAS[int(np.argmin(m))]
    return best, pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", required=True, choices=list(LEVELS))
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(8)
    protocol = json.loads(PROTOCOL.read_text())
    crit = protocol["criterion"]["delta"]
    f = LEVELS[args.level]
    base = OUT / "results" / f"level_{args.level}"
    frozen_path = base / "frozen_for_test.json"
    res = base / args.split
    models, info = load_models(args.level, args.device)
    if args.split == "test":
        if not args.final_test or not frozen_path.exists():
            sys.exit("Test evaluation requires --final-test AND frozen_for_test.json from the validation run.")
        if (res / "evaluation_report.json").exists():
            sys.exit(f"The test set was already evaluated for level {args.level}; it is run once only.")
        frozen = json.loads(frozen_path.read_text())
        for n in LOW:
            assert info[n]["md5"] == frozen["models"][str(n)]["md5"], f"checkpoint for N={n} changed after freezing"
        sigma = {int(k): v for k, v in frozen["smoothing_sigma"].items()}
        d = load_objects(("test",))
    else:
        d = load_objects(("val",))
        tv = load_objects(("train",))
        sigma, tun = tune_sigmas(args.level, tv["object"].astype(np.float32))
        tun.to_csv(safe_path(base / "smoothing_sigma_tuning_train.csv"), index=False)
    ids, subset, clean = d["image_id"].tolist(), d["subset"], d["object"]
    sub_of = dict(zip(ids, subset))
    ex_ids = [i for s in SUBSETS for i in np.array(ids)[subset == s][np.linspace(0, (subset == s).sum() - 1, 3).astype(int)]]
    checks = {"split": args.split, "level": args.level, "noise_f": f, "n_images": len(ids),
              "n_by_subset": {s: int((subset == s).sum()) for s in SUBSETS}, "models": info, "smoothing_sigma": sigma}

    rows, keep = [], {}
    for seed in EVAL_SEEDS:
        acq0, acq = noisy_dgi_readouts(clean, seed, [0.0, f])               # noise-free and noisy, same patterns
        imgs = {f"DGI-{n}": acq[:, SHOTS.index(n)] for n in SHOTS}
        imgs[NF] = acq0[:, SHOTS.index(10000)]
        for n in LOW:
            imgs[f"smoothed DGI-{n}"] = smooth(acq[:, SHOTS.index(n)][:, None], sigma[n])[:, 0]
            imgs[f"{PRE}-{n}"] = np.clip(predict(models[n], acq[:, SHOTS.index(n)][:, None].astype(np.float32), args.device), 0, 1)[:, 0]
        for lab, arr in imgs.items():
            meth, n = lab.rsplit("-", 1)
            for j, image_id in enumerate(ids):
                rows.append({"image_id": image_id, "subset": subset[j], "seed": seed, "label": lab, "method": meth,
                             "shot_count": int(n), **E.metrics(arr[j], clean[j])})
        if seed == 2026:
            keep = imgs
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(safe_path(res / "per_image_results.csv"), index=False)

    srows = []
    for grp in GROUPS:
        for sg, seeds in (("all seeds (2026-2029)", EVAL_SEEDS), ("seed 2026", [2026]), ("seeds 2027-2029", [2027, 2028, 2029])):
            g0 = per[per.seed.isin(seeds) & ((per.subset == grp) if grp != "all" else True)]
            for (lab, meth, n), g in g0.groupby(["label", "method", "shot_count"]):
                r = {"group": grp, "seed_group": sg, "label": lab, "method": meth, "shot_count": n, "n_pairs": len(g)}
                for k in METS:
                    r[f"{k}_mean"], r[f"{k}_std"], r[f"{k}_median"] = g[k].mean(), g[k].std(), g[k].median()
                srows.append(r)
    summ = pd.DataFrame(srows).sort_values(["group", "seed_group", "method", "shot_count"])
    summ.to_csv(safe_path(res / "aggregate_metrics.csv"), index=False)

    cand = [f"{m}-{n}" for n in LOW for m in ("smoothed DGI", PRE)]
    pairs = ([(c, COMP) for c in cand] + [(f"{PRE}-{n}", NF) for n in LOW]
             + [(f"{PRE}-{n}", f"DGI-{n}") for n in LOW] + [(f"{PRE}-{n}", f"smoothed DGI-{n}") for n in LOW]
             + [(f"{PRE}-{b}", f"{PRE}-{a}") for a, b in zip(LOW, LOW[1:])] + [(COMP, NF)])
    comp_rows, minimum, minimum_nf = [], {}, {}
    for grp in GROUPS:
        gids = [i for i in ids if grp == "all" or sub_of[i] == grp]
        rng = np.random.default_rng(0)
        for a, b in pairs:
            dd = E.paired(per, a, b, gids, rng, METS)
            r = {"group": grp, "n_images": len(gids), "comparison": f"{a} minus {b}", "A": a, "B": b,
                 "shot_count": int(a.rsplit("-", 1)[1])}
            for k, v in dd.items():
                for s_, val in v.items():
                    r[f"{k}_{s_}"] = val
            if b in (COMP, NF) and a != COMP:
                r["criterion_pass"] = bool(dd["SSIM"]["ci_low"] >= -crit["SSIM"] and dd["correlation"]["ci_low"] >= -crit["Pearson_r"])
            comp_rows.append(r)
        cg = pd.DataFrame([r for r in comp_rows if r["group"] == grp and "criterion_pass" in r])
        minimum[grp] = {}
        for meth in ("smoothed DGI", PRE):
            ok = cg[(cg.B == COMP) & cg.A.str.rsplit("-", n=1).str[0].eq(meth) & cg.criterion_pass].shot_count
            minimum[grp][meth] = int(ok.min()) if len(ok) else None
        ok = cg[(cg.B == NF) & cg.criterion_pass.astype(bool)].shot_count
        minimum_nf[grp] = {PRE: int(ok.min()) if len(ok) else None}
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(safe_path(res / "comparisons.csv"), index=False)

    for grp in GROUPS:
        s = summ[(summ.group == grp) & (summ.seed_group == "all seeds (2026-2029)")]
        E.plot_curves(s, COMP, ["DGI", "smoothed DGI", PRE], res / "figures" / f"metrics_vs_shots_{grp}.png",
                      f"Poisson bucket noise f = {f} ({args.split}, {grp}): scored against the original clean object; "
                      f"dotted = noisy DGI-10000; mean ± std over all image-sequence pairs")
    p26 = per[per.seed == 2026].set_index(["image_id", "label"])
    order = [f"{m}-{n}" for n in LOW for m in ("DGI", "smoothed DGI", PRE)] + [COMP, NF]
    rws = []
    for image_id in ex_ids:
        j = ids.index(image_id)
        panels = [("Original clean object", clean[j])] + [
            (f"{lab}\n{p26.loc[(image_id, lab)].PSNR:.2f} dB | SSIM {p26.loc[(image_id, lab)].SSIM:.3f}", keep[lab][j]) for lab in order]
        rws.append({"label": f"{image_id}", "target": clean[j], "panels": panels})
    save_panel_grid(rws, safe_path(res / "figures" / "comparison_examples.png"),
                    f"Poisson bucket noise f = {f} ({args.split}), seed 2026; identical grey window [0, 1]; no enhancement")

    report = {"level": args.level, "noise_f": f, "split": args.split, "criterion": protocol["criterion"], "checks": checks,
              "minimum_passing_shot_count": minimum, "minimum_passing_shot_count_vs_noise_free_DGI10000": minimum_nf,
              "reduction_factor_vs_10000": {g: {m: (10000 / v if v else None) for m, v in mm.items()} for g, mm in minimum.items()},
              "example_ids": ex_ids, "runtime_s": round(time.time() - t0, 1)}
    write_json(res / "evaluation_report.json", report)
    if args.split == "val":
        write_json(frozen_path, {"frozen_after_validation": True, "level": args.level, "noise_f": f,
                                 "models": {str(n): info[n] for n in LOW}, "smoothing_sigma": {str(n): v for n, v in sigma.items()},
                                 "eval_seeds": EVAL_SEEDS, "criterion": protocol["criterion"],
                                 "validation_minimum_passing_shot_count": minimum,
                                 "validation_minimum_vs_noise_free_DGI10000": minimum_nf,
                                 "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    pd.set_option("display.width", 240)
    s = summ[summ.seed_group == "all seeds (2026-2029)"]
    print(s[["group", "label", "n_pairs"] + [f"{k}_mean" for k in ("SSIM", "correlation", "PSNR", "edge_r")]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    show = ["group", "comparison"] + [f"{k}_{x}" for k in ("SSIM", "correlation") for x in ("mean", "ci_low")] + ["criterion_pass"]
    print(comp[comp.B.isin([COMP, NF])][show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("minimum passing N (vs noisy DGI-10000):", json.dumps(minimum), "| vs noise-free:", json.dumps(minimum_nf),
          f"| {report['runtime_s']} s -> {res}")


if __name__ == "__main__":
    main()
