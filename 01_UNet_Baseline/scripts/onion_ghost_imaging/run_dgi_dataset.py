#!/usr/bin/env python3
"""Phase 2A: dataset-scale conventional DGI baseline (no AI).

    python run_dgi_dataset.py --check     # MANDATORY first: 3 images (1 train, 1 val, 1 test)
    python run_dgi_dataset.py             # all 490 images (refuses to run unless --check passed)

Configuration (identical to the validated Phase 1 / 1b C_16 runs):
    object      : linear-light luminance, centre square crop, box-downsampled to 64x64
    patterns    : i.i.d. Bernoulli(0.5) binary, ONE sequence of 10,000, seed 2026,
                  the SAME sequence for every image (like a fixed DMD pattern set)
    buckets     : B_i = sum_xy P_i O,  R_i = sum_xy P_i  (noise-free)
    DGI         : k_N = <B>_N/<R>_N, S_i = B_i - k_N R_i, DGI_N = <S (P - <P>_N)>_N
    estimate    : O_hat = DGI_N / 0.25 + k_N, clipped to [0, 1] for images/metrics
    N           : prefixes 500 / 1,000 / 5,000 / 10,000 of the same sequence

Memory: patterns are generated in batches of 500 and never stored; all images are
updated together through running sums (src.reconstruction.StreamingDGI), read
out at each N. Ground truth is used only to compute metrics.

Primary metrics are on the whole 64x64 image (as in Phase 1). Metrics restricted
to the field of view (FOV) are saved as secondary columns (*_fov).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.ghost_imaging import (bucket_measurements, generate_patterns, load_object,  # noqa: E402
                               pattern_batches, pattern_variance, reference_intensity)
from src.metrics import evaluate, evaluate_in_mask  # noqa: E402
from src.reconstruction import StreamingDGI, dgi_reconstruct, normalize_dgi_calibrated  # noqa: E402
from src.visualization import (save_contrast_diagnostic, save_distribution_boxplots,  # noqa: E402
                               save_sample_grid, save_split_curves)

DATA = ROOT / "data" / "v2"
SPLITS = ("train", "val", "test")
EXPECTED_COUNTS = {"train": 341, "val": 75, "test": 74}
PHASE1B = ROOT / "outputs" / "phase1b_dgi" / "C_16_64px"
METRIC_KEYS = ("PSNR", "SSIM", "MSE", "Pearson_r")


# --------------------------------------------------------------------------- data
def load_split_table() -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for image_id in (DATA / "splits" / split / "ids.txt").read_text().split():
            rows.append({"image_id": image_id, "split": split})
    df = pd.DataFrame(rows)
    counts = df.split.value_counts().to_dict()
    assert counts == EXPECTED_COUNTS, f"split counts changed: {counts}"
    assert df.image_id.is_unique, "an image appears in more than one split"
    man = pd.read_csv(DATA / "metadata" / "manifest.csv")[["stem", "group", "fov_type"]]
    df = df.merge(man.rename(columns={"stem": "image_id"}), on="image_id", how="left", validate="one_to_one")
    assert df.fov_type.notna().all()
    return df


def load_objects(ids, size):
    objs, fovs, infos = [], [], []
    for image_id in ids:
        o, info, fov = load_object(DATA / "reference" / f"{image_id}.npz", size, return_fov=True)
        objs.append(o), fovs.append(fov), infos.append(info)
    return np.stack(objs), np.stack(fovs), infos


# --------------------------------------------------------------------------- core
def run_dgi(objects, shots, seed, size, batch, p_on, s2):
    """Stream the seeded pattern sequence through all objects; return {N: (est (M,H,W), k (M,))}."""
    assert all(n % batch == 0 for n in shots), "every N must be a multiple of the batch size"
    stream = StreamingDGI(objects)
    out = {}
    for pats in pattern_batches(max(shots), (size, size), seed, batch, p_on):
        stream.update(pats)
        if stream.n in shots:
            est, k = stream.estimate(s2)
            out[stream.n] = (est.reshape(-1, size, size), k)
    return out


def evaluate_all(table, objects, fovs, results, s2_unused=None):
    rows = []
    for i, meta in enumerate(table.itertuples(index=False)):
        obj = objects[i]
        fov_mask = fovs[i] >= 0.999                      # output pixels fully inside the FOV
        dev2 = float(((obj - obj.mean()) ** 2).sum())
        for n, (est_all, k_all) in sorted(results.items()):
            est = est_all[i]
            rec = np.clip(est, 0.0, 1.0)
            m = evaluate(obj, rec)
            mf = evaluate_in_mask(obj, rec, fov_mask) if fov_mask.sum() > 50 else dict.fromkeys(METRIC_KEYS, np.nan)
            rows.append({
                "image_id": meta.image_id, "split": meta.split, "group": meta.group, "fov_type": meta.fov_type,
                "measurements": n,
                "psnr": m["PSNR"], "ssim": m["SSIM"], "mse": m["MSE"], "correlation": m["Pearson_r"],
                "psnr_fov": mf["PSNR"], "ssim_fov": mf["SSIM"], "mse_fov": mf["MSE"], "correlation_fov": mf["Pearson_r"],
                "k_mean_estimate": float(k_all[i]),
                "object_mean": float(obj.mean()), "object_std": float(obj.std()),
                "fov_fraction": float(fov_mask.mean()),
                "unclipped_mse": float(((est - obj) ** 2).mean()),
                "theory_mse": dev2 / n,
                "clipped_fraction": float(((est < 0) | (est > 1)).mean()),
            })
    return pd.DataFrame(rows)


def monotonic_report(df):
    """Per image: is quality strictly better at every larger N?"""
    out = {}
    for col, better in (("psnr", 1), ("ssim", 1), ("correlation", 1), ("mse", -1)):
        wide = df.pivot(index="image_id", columns="measurements", values=col).sort_index(axis=1)
        steps = np.sign(np.diff(wide.values, axis=1)) == better
        out[col] = {
            "all_steps_improve_n_images": int(steps.all(axis=1).sum()),
            "n_images": int(len(wide)),
            "per_step_fraction_improved": {f"{a}->{b}": round(float(steps[:, j].mean()), 4)
                                           for j, (a, b) in enumerate(zip(wide.columns[:-1], wide.columns[1:]))},
            "first_vs_last_improved_fraction": round(float((np.sign(wide.values[:, -1] - wide.values[:, 0]) == better).mean()), 4),
            "violating_images": sorted(wide.index[~steps.all(axis=1)].tolist()),
        }
    return out


def summarize(df, suffix=""):
    """Mean/median/std per subset x N. suffix='' (primary) or '_fov' (secondary)."""
    p, s, m, r = (f"psnr{suffix}", f"ssim{suffix}", f"mse{suffix}", f"correlation{suffix}")
    rows = []
    for subset, d in [("all", df)] + [(sp, df[df.split == sp]) for sp in SPLITS]:
        for n, g in d.groupby("measurements"):
            rows.append({"subset": subset, "measurements": n, "n_images": g.image_id.nunique(),
                         "psnr_mean": g[p].mean(), "psnr_median": g[p].median(), "psnr_std": g[p].std(),
                         "ssim_mean": g[s].mean(), "ssim_median": g[s].median(), "ssim_std": g[s].std(),
                         "mse_mean": g[m].mean(), "mse_std": g[m].std(),
                         "correlation_mean": g[r].mean(), "correlation_std": g[r].std()})
    return pd.DataFrame(rows)


def to_plot_frame(df, suffix=""):
    return df.rename(columns={f"psnr{suffix}": "PSNR", f"ssim{suffix}": "SSIM", f"mse{suffix}": "MSE",
                              f"correlation{suffix}": "Pearson_r"})


# --------------------------------------------------------------------------- check stage
def run_check(args, out):
    """3 images (C_16 from train + first id of val and test). All must pass before the full run."""
    table = load_split_table()
    first = {sp: (DATA / "splits" / sp / "ids.txt").read_text().split()[0] for sp in ("val", "test")}
    ids = ["C_16", first["val"], first["test"]]
    sub = table.set_index("image_id").loc[ids].reset_index()
    objects, fovs, _ = load_objects(ids, args.size)
    s2 = pattern_variance(args.p_on)
    results = run_dgi(objects, args.shots, args.seed, args.size, args.batch, args.p_on, s2)
    df = evaluate_all(sub, objects, fovs, results)

    checks = {"images": dict(zip(ids, sub.split))}
    full_pats = generate_patterns(max(args.shots), (args.size, args.size), args.seed, args.p_on)
    batched = np.concatenate(list(pattern_batches(max(args.shots), (args.size, args.size), args.seed, args.batch, args.p_on)))
    checks["batched_patterns_identical_to_single_call"] = bool(np.array_equal(full_pats, batched))

    sim = np.load(PHASE1B.parent.parent / "phase1_single_image" / "C_16_64px" / "simulation.npz")
    checks["C_16_object_identical_to_phase1"] = bool(np.array_equal(sim["object"], objects[0]))
    p1_pats = np.unpackbits(sim["patterns_packed"], axis=-1, count=args.size).reshape(tuple(sim["pattern_shape"]))
    checks["patterns_identical_to_phase1"] = bool(np.array_equal(p1_pats, full_pats))

    # streaming vs the validated batch implementation (dgi_reconstruct + normalize_dgi_calibrated)
    refs = reference_intensity(full_pats)
    max_diff, max_k_diff = 0.0, 0.0
    for i in range(len(ids)):
        bk = bucket_measurements(full_pats, objects[i])
        for n in args.shots:
            raw, k = dgi_reconstruct(full_pats, bk, refs, n)
            est, _ = normalize_dgi_calibrated(raw, s2, k)
            max_diff = max(max_diff, float(np.abs(est - results[n][0][i]).max()))
            max_k_diff = max(max_k_diff, abs(k - float(results[n][1][i])))
    checks["streaming_vs_validated_batch_max_abs_diff_estimate"] = max_diff
    checks["streaming_vs_validated_batch_max_abs_diff_k"] = max_k_diff

    p1b = pd.read_csv(PHASE1B / "metrics_long.csv")
    p1b = p1b[p1b.method == "DGI"].set_index("measurements")
    c16 = df[df.image_id == "C_16"].set_index("measurements")
    checks["C_16_metrics_vs_phase1b_max_abs_diff"] = float(max(
        (p1b.loc[args.shots, a] - c16.loc[args.shots, b]).abs().max()
        for a, b in (("PSNR", "psnr"), ("SSIM", "ssim"), ("MSE", "mse"), ("Pearson_r", "correlation"))))

    order = {}
    for image_id, g in df.groupby("image_id"):
        g = g.set_index("measurements")
        order[image_id] = {
            "500_worse_than_10000": bool(g.psnr[500] < g.psnr[10000] and g.ssim[500] < g.ssim[10000]),
            "1000_better_than_500": bool(g.psnr[1000] > g.psnr[500] and g.ssim[1000] > g.ssim[500]),
            "5000_better_than_1000": bool(g.psnr[5000] > g.psnr[1000] and g.ssim[5000] > g.ssim[1000]),
            "unclipped_mse_over_theory": {int(n): round(v, 3) for n, v in (g.unclipped_mse / g.theory_mse).items()},
        }
    checks["ordering"] = order

    passed = (checks["batched_patterns_identical_to_single_call"] and checks["C_16_object_identical_to_phase1"]
              and checks["patterns_identical_to_phase1"]
              and max_diff < 1e-9 and max_k_diff < 1e-12
              and checks["C_16_metrics_vs_phase1b_max_abs_diff"] < 1e-8
              and all(v[key] for v in order.values()
                      for key in ("500_worse_than_10000", "1000_better_than_500", "5000_better_than_1000")))
    checks["PASSED"] = bool(passed)
    df.to_csv(out / "check_metrics.csv", index=False)
    (out / "check_result.json").write_text(json.dumps(checks, indent=2))
    print(df[["image_id", "split", "measurements", "psnr", "ssim", "mse", "correlation"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print(json.dumps(checks, indent=2))
    return passed


# --------------------------------------------------------------------------- full stage
def pick_samples(df, n_max):
    """Fixed, rule-based subset (no hand-picking): per split, images nearest to set
    percentiles of DGI PSNR at the largest N -> shows typical AND hard/easy cases."""
    pct = {"train": (10, 50, 90), "val": (25, 75), "test": (10, 50, 90)}
    last = df[df.measurements == n_max]
    picks = []
    for split, qs in pct.items():
        d = last[last.split == split].sort_values("psnr").reset_index(drop=True)
        for q in qs:
            target = np.percentile(d.psnr, q)
            cand = d.iloc[(d.psnr - target).abs().argsort()]
            picks.append((split, q, next(i for i in cand.image_id if i not in [p[2] for p in picks])))
    return picks


def run_full(args, out):
    gate = out / "check" / "check_result.json"
    if not gate.exists() or not json.loads(gate.read_text()).get("PASSED"):
        sys.exit("Refusing to run: the 3-image check has not passed. Run with --check first.")
    for sub in ("metrics", "summaries", "sample_reconstructions", "plots", "reconstructions", "robustness"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    table = load_split_table()
    objects, fovs, _ = load_objects(table.image_id, args.size)
    t_load = time.time() - t0
    s2 = pattern_variance(args.p_on)

    t1 = time.time()
    results = run_dgi(objects, args.shots, args.seed, args.size, args.batch, args.p_on, s2)
    t_dgi = time.time() - t1
    df = evaluate_all(table, objects, fovs, results)
    t_eval = time.time() - t1 - t_dgi

    # ---- per-image metrics + reconstructions --------------------------------
    df.to_csv(out / "metrics" / "dgi_per_image_seed2026.csv", index=False)
    np.savez_compressed(
        out / "reconstructions" / f"dgi_estimates_seed{args.seed}_{args.size}px.npz",
        image_id=table.image_id.values.astype(str), split=table.split.values.astype(str), shots=np.array(args.shots),
        object=objects.astype(np.float32), fov_fraction=fovs.astype(np.float32),
        **{f"estimate_unclipped_{n}": results[n][0].astype(np.float32) for n in args.shots},
        **{f"k_{n}": results[n][1] for n in args.shots},
    )

    # ---- summaries -----------------------------------------------------------
    summ = summarize(df)
    summ.to_csv(out / "summaries" / "summary_primary_whole_image.csv", index=False)
    summarize(df, "_fov").to_csv(out / "summaries" / "summary_secondary_fov_only.csv", index=False)
    by_fov = df.groupby(["fov_type", "measurements"]).agg(
        n_images=("image_id", "nunique"), psnr_mean=("psnr", "mean"), psnr_std=("psnr", "std"),
        ssim_mean=("ssim", "mean"), ssim_std=("ssim", "std"), correlation_mean=("correlation", "mean"),
        psnr_fov_mean=("psnr_fov", "mean"), ssim_fov_mean=("ssim_fov", "mean")).reset_index()
    by_fov.to_csv(out / "summaries" / "summary_by_fov_type.csv", index=False)
    mono = monotonic_report(df)
    theory_ratio = (df.unclipped_mse / df.theory_mse).groupby(df.measurements).describe(percentiles=[0.05, 0.5, 0.95])
    theory_ratio.to_csv(out / "summaries" / "unclipped_mse_over_theory.csv")

    c16 = df[df.image_id == "C_16"].set_index("measurements")
    c16_rank = {}
    for n in args.shots:
        d = df[df.measurements == n]
        c16_rank[int(n)] = {m: round(float((d[m] < c16.loc[n, m]).mean() * 100), 1) for m in ("psnr", "ssim", "correlation")}

    # ---- figures -------------------------------------------------------------
    base = f"DGI, {len(table)} onion images, 64x64, same 10,000 binary patterns (seed {args.seed})"
    save_split_curves(to_plot_frame(df), out / "plots" / "metrics_vs_shots_by_split.png",
                      base + " - whole-image metrics")
    save_distribution_boxplots(to_plot_frame(df), out / "plots" / "psnr_ssim_distributions.png",
                               base + " - all images")
    last = to_plot_frame(df[df.measurements == max(args.shots)])
    save_contrast_diagnostic(last, out / "plots" / f"psnr{max(args.shots)}_vs_object_contrast.png",
                             f"DGI-{max(args.shots):,}: why images differ (diagnostic)")
    picks = pick_samples(df, max(args.shots))
    idx = {i: j for j, i in enumerate(table.image_id)}
    rows = []
    for split, q, image_id in picks:
        i = idx[image_id]
        g = df[df.image_id == image_id].set_index("measurements")
        rows.append({"label": f"{image_id}\n{split}, p{q}", "original": objects[i],
                     "recons": {n: np.clip(results[n][0][i], 0, 1) for n in args.shots},
                     "metrics": {n: {"PSNR": g.psnr[n], "SSIM": g.ssim[n]} for n in args.shots}})
    save_sample_grid(rows, args.shots, out / "sample_reconstructions" / "samples_fixed_scale_0to1.png",
                     "DGI reconstructions, fixed gray scale [0, 1] for every panel (scientific figure)\n"
                     "rows chosen by rule: images nearest the p10/p50/p90 (train, test) and p25/p75 (val) "
                     f"of DGI-{max(args.shots):,} PSNR")
    save_sample_grid(rows, args.shots, out / "sample_reconstructions" / "samples_DISPLAY_ONLY_row_window.png",
                     "DISPLAY ONLY: each row uses one fixed window (its original's 1-99th pct) for all panels; "
                     "metrics are from the unstretched [0, 1] images", display_window=True)

    # ---- robustness: independent sequences, separate from main results -------
    rob = []
    t2 = time.time()
    for seed in args.robustness_seeds:
        r = run_dgi(objects, args.shots, seed, args.size, args.batch, args.p_on, s2)
        d = evaluate_all(table, objects, fovs, r)
        d.insert(0, "seed", seed)
        rob.append(d)
    t_rob = time.time() - t2
    if rob:
        rob = pd.concat([df.assign(seed=args.seed)] + rob, ignore_index=True)
        rob.to_csv(out / "robustness" / "robustness_per_image_all_seeds.csv", index=False)
        seed_means = rob.groupby(["seed", "measurements"])[["psnr", "ssim", "mse", "correlation"]].mean()
        seed_means.to_csv(out / "robustness" / "robustness_dataset_mean_per_seed.csv")
        per_img_sd = rob.groupby(["image_id", "measurements"])[["psnr", "ssim"]].std().groupby("measurements").mean()
        per_img_sd.to_csv(out / "robustness" / "robustness_mean_within_image_std_across_seeds.csv")

    run_info = {
        "n_images": int(len(table)), "split_counts": table.split.value_counts().to_dict(),
        "shots": args.shots, "seed": args.seed, "robustness_seeds": args.robustness_seeds,
        "size": args.size, "batch": args.batch,
        "pattern_model": "i.i.d. Bernoulli(0.5) binary; ONE sequence (seed 2026) shared by all images",
        "estimate": "O_hat = DGI_N/0.25 + <B>_N/<R>_N, clip [0,1]",
        "n_reconstructions_evaluated": int(len(df)),
        "nan_or_inf_in_metrics": int(df[["psnr", "ssim", "mse", "correlation"]].isna().sum().sum()
                                     + np.isinf(df[["psnr", "ssim", "mse", "correlation"]].values).sum()),
        "monotonic": mono, "C_16_percentile_rank_in_dataset": c16_rank,
        "sample_images": [{"split": s, "percentile": q, "image_id": i} for s, q, i in picks],
        "timing_s": {"load_objects": round(t_load, 1), "dgi_stream": round(t_dgi, 1),
                     "metrics": round(t_eval, 1), "robustness": round(t_rob, 1),
                     "total": round(time.time() - t0, 1)},
    }
    (out / "summaries" / "run_info.json").write_text(json.dumps(run_info, indent=2))

    pd.set_option("display.width", 200)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nby FOV type:\n" + by_fov.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nunclipped MSE / theory:\n" + theory_ratio.round(3).to_string())
    if len(rob):
        print("\nrobustness, dataset mean per seed:\n" + seed_means.round(4).to_string())
        print("\nmean within-image std across seeds:\n" + per_img_sd.round(4).to_string())
    print("\n" + json.dumps({k: run_info[k] for k in ("monotonic", "C_16_percentile_rank_in_dataset",
                                                     "sample_images", "timing_s", "nan_or_inf_in_metrics")}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="run the mandatory 3-image verification only")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--shots", type=int, nargs="+", default=[500, 1000, 5000, 10000])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--robustness-seeds", type=int, nargs="*", default=[2027, 2028, 2029])
    ap.add_argument("--p-on", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=500)
    args = ap.parse_args()
    args.shots = sorted(args.shots)
    out = ROOT / "outputs" / "phase2a_dgi_dataset"
    if args.check:
        (out / "check").mkdir(parents=True, exist_ok=True)
        ok = run_check(args, out / "check")
        print("\nCHECK", "PASSED" if ok else "FAILED")
        sys.exit(0 if ok else 1)
    run_full(args, out)


if __name__ == "__main__":
    main()
