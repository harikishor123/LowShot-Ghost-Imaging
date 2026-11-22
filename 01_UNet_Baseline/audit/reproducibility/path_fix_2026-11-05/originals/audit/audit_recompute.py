"""Independent, read-only recomputation of the extension and Poisson results from per_image_results.csv.

Does not import the project's evaluation code. For every results folder (val + test) it recomputes:
  1. aggregate mean/std/median per group x seed-group x label          -> vs aggregate_metrics.csv
  2. paired-difference mean/std/%better per comparison                    -> vs comparisons.csv
  3. bootstrap 95% CI (same documented rule: 2,000 resamples over images, seeds kept together,
     default_rng(0) per group, comparisons in file order)                 -> vs comparisons.csv
  4. criterion pass (lo >= -0.02 SSIM and lo >= -0.01 r) and minimum N   -> vs evaluation_report.json
Prints max absolute differences and any mismatching decisions. Writes nothing to the project.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/ss_students/Downloads/Onion-CNN-Final.v1-onion-original-v1.yolov8")
DIRS = [ROOT / f"outputs/extension_medmnist_microscopy/results/target_{t}/{s}" for t in ("clean", "dgi10k") for s in ("val", "test")] + \
       [ROOT / f"outputs/extension_medmnist_microscopy_poisson/results/level_{l}/{s}" for l in ("p05", "p10", "p30") for s in ("val", "test")]
METS = ["SSIM", "correlation", "PSNR", "MSE", "edge_r"]
SGS = {"all seeds (2026-2029)": [2026, 2027, 2028, 2029], "seed 2026": [2026], "seeds 2027-2029": [2027, 2028, 2029]}
out = {}

for d in DIRS:
    key = f"{d.parent.name}/{d.name}"
    per = pd.read_csv(d / "per_image_results.csv")
    agg = pd.read_csv(d / "aggregate_metrics.csv")
    comp = pd.read_csv(d / "comparisons.csv")
    rep = json.loads((d / "evaluation_report.json").read_text())
    res = {"n_rows": len(per), "n_images": per.image_id.nunique(), "labels": per.label.nunique()}

    # structural checks
    cnt = per.groupby(["label"]).size()
    res["pairs_per_label"] = sorted(set(cnt.tolist()))
    res["dup_rows"] = int(per.duplicated(["image_id", "seed", "label"]).sum())
    res["nonfinite"] = int((~np.isfinite(per[[c for c in per.columns if c in METS]].to_numpy())).sum())

    # 1. aggregates
    md = 0.0
    for _, r in agg.iterrows():
        g = per[per.seed.isin(SGS[r.seed_group]) & (per.label == r.label)]
        if r.group != "all":
            g = g[g.subset == r.group]
        assert len(g) == r.n_pairs, (key, r.label, r.group, r.seed_group)
        for m in METS:
            md = max(md, abs(g[m].mean() - r[f"{m}_mean"]), abs(g[m].std() - r[f"{m}_std"]), abs(g[m].median() - r[f"{m}_median"]))
    res["aggregate_max_abs_diff"] = md

    # 2-4. comparisons
    ids_order = list(dict.fromkeys(per.image_id))
    sub_of = per.drop_duplicates("image_id").set_index("image_id").subset
    piv = {m: per.pivot_table(index=["image_id", "label"], columns="seed", values=m) for m in METS}
    mdiff = {"mean": 0.0, "std": 0.0, "pct": 0.0, "ci": 0.0}
    decision_mismatch = []
    recomputed_pass = {}
    for grp in comp.group.unique():
        rng = np.random.default_rng(0)
        gids = [i for i in ids_order if grp == "all" or sub_of[i] == grp]
        for _, r in comp[comp.group == grp].iterrows():
            ok = {}
            for m in METS:
                A = piv[m].xs(r.A, level="label").loc[gids].to_numpy()
                B = piv[m].xs(r.B, level="label").loc[gids].to_numpy()
                dd = A - B
                n = dd.shape[0]
                bm = np.array([np.nanmean(dd[rng.integers(0, n, n)]) for _ in range(2000)])
                lo, hi = np.percentile(bm, 2.5), np.percentile(bm, 97.5)
                better = (dd < 0) if m == "MSE" else (dd > 0)
                mdiff["mean"] = max(mdiff["mean"], abs(np.nanmean(dd) - r[f"{m}_mean"]))
                mdiff["std"] = max(mdiff["std"], abs(np.nanstd(dd, ddof=1) - r[f"{m}_std"]))
                mdiff["pct"] = max(mdiff["pct"], abs(better.mean() * 100 - r[f"{m}_pct_A_better"]))
                mdiff["ci"] = max(mdiff["ci"], abs(lo - r[f"{m}_ci_low"]), abs(hi - r[f"{m}_ci_high"]))
                ok[m] = lo
            if "criterion_pass" in comp.columns and pd.notna(r.get("criterion_pass")) and r.get("criterion_pass") != "":
                p = bool(ok["SSIM"] >= -0.02 and ok["correlation"] >= -0.01)
                stored = str(r.criterion_pass) == "True"
                recomputed_pass[(grp, r.A, r.B)] = p
                if p != stored:
                    decision_mismatch.append((grp, r.comparison, p, stored))
    res["comparison_max_abs_diff"] = mdiff
    res["decision_mismatches"] = decision_mismatch

    # minimum N from recomputed decisions
    def minimum(B_filter):
        outm = {}
        for (grp, a, b), p in recomputed_pass.items():
            if not B_filter(b):
                continue
            meth, n = a.rsplit("-", 1)
            outm.setdefault(grp, {}).setdefault(meth, [])
            if p:
                outm[grp][meth].append(int(n))
        return {g: {m: (min(v) if v else None) for m, v in mm.items()} for g, mm in outm.items()}

    if "poisson" in str(d):
        mn = minimum(lambda b: b == "DGI-10000")
        mnf = minimum(lambda b: b == "noise-free DGI-10000")
        res["minN_match"] = (mn == rep["minimum_passing_shot_count"], mnf == rep["minimum_passing_shot_count_vs_noise_free_DGI10000"])
    else:
        comparator = "DGI-10000" if "clean" in key else "independent 10k-10000"
        mn = minimum(lambda b: b == comparator)
        res["minN_match"] = (mn == rep["minimum_passing_shot_count"],)
    res["minN"] = mn
    # monotonicity: any pass at N followed by a fail at larger N?
    nonmono = []
    for (grp, a, b) in recomputed_pass:
        meth, n = a.rsplit("-", 1)
        larger = [(k, v) for k, v in recomputed_pass.items() if k[0] == grp and k[2] == b and k[1].rsplit("-", 1)[0] == meth and int(k[1].rsplit("-", 1)[1]) > int(n)]
        if recomputed_pass[(grp, a, b)] and any(not v for _, v in larger):
            nonmono.append((grp, a, b))
    res["non_monotone"] = nonmono
    out[key] = res
    print(key, json.dumps(res, default=str), flush=True)

json.dump(out, open(Path(sys.argv[1]), "w"), indent=1, default=str)
