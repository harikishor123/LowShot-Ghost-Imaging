#!/usr/bin/env python3
"""Phase 1: validate conventional correlation ghost imaging (CGI) on ONE onion image.

    python run_single_image.py                       # defaults: C_16, 64x64, 10,000 shots
    python run_single_image.py --image-id C_16 --size 64 --seed 2026

One sequence of M = max(--shots) binary random patterns is generated once. The
N-shot reconstruction uses the FIRST N patterns and buckets of that sequence.
No deep learning, no photon/Poisson noise: the only variable is N.

Outputs (default outputs/phase1_single_image/<id>_<size>px/):
    comparison_calibrated.png  original + 500/1k/5k/10k reconstructions (primary)
    comparison_minmax.png      same with min-max normalisation (secondary)
    metric_curves.png          PSNR/SSIM/MSE vs N
    metrics.csv                full table (both normalisations)
    validation.json            config, object info, and all sanity checks
    simulation.npz             object, bit-packed patterns, buckets, raw GI images
    images/                    individual PNGs
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

from src.ghost_imaging import bucket_measurements, generate_patterns, load_object, pattern_variance  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.reconstruction import cgi_reconstruct, normalize_calibrated, normalize_minmax  # noqa: E402
from src.visualization import save_comparison, save_gray_png, save_metric_curves  # noqa: E402

REFERENCE_DIR = ROOT / "data" / "v2" / "reference"
SPLIT_DIR = ROOT / "data" / "v2" / "splits"


def find_split(image_id: str) -> str:
    for split in ("train", "val", "test"):
        ids = (SPLIT_DIR / split / "ids.txt").read_text().split()
        if image_id in ids:
            return split
    raise ValueError(f"{image_id} is not in any v2 split")


def is_monotone(values, increasing: bool) -> bool:
    pairs = zip(values[:-1], values[1:])
    return all(b > a for a, b in pairs) if increasing else all(b < a for a, b in pairs)


def seed_robustness(obj, shots, n_seeds, base_seed, p_on, s2):
    """Repeat the whole experiment with independent pattern sequences."""
    rows = []
    for k in range(n_seeds):
        seed = base_seed + 1000 + k
        pats = generate_patterns(max(shots), obj.shape, seed, p_on)
        bk = bucket_measurements(pats, obj)
        for n in shots:
            _, rec = normalize_calibrated(cgi_reconstruct(pats, bk, n), s2)
            rows.append({"seed": seed, "measurements": n, **evaluate(obj, rec)})
    df = pd.DataFrame(rows)
    per_seed_ok = {
        int(seed): bool(is_monotone(g.sort_values("measurements")["PSNR"].tolist(), True)
                        and is_monotone(g.sort_values("measurements")["SSIM"].tolist(), True))
        for seed, g in df.groupby("seed")
    }
    summary = df.groupby("measurements")[["PSNR", "SSIM", "MSE"]].agg(["mean", "std"])
    return per_seed_ok, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-id", default="C_16")
    ap.add_argument("--size", type=int, default=64, help="object side length in pixels")
    ap.add_argument("--shots", type=int, nargs="+", default=[500, 1000, 5000, 10000])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--p-on", type=float, default=0.5, help="Bernoulli probability of a lit pixel")
    ap.add_argument("--robustness-seeds", type=int, default=5, help="extra independent sequences (0 = skip)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    shots = sorted(args.shots)
    n_total = shots[-1]
    out = args.out or ROOT / "outputs" / "phase1_single_image" / f"{args.image_id}_{args.size}px"
    (out / "images").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- 1. object -----------------------------------------------------------
    split = find_split(args.image_id)
    obj, obj_info = load_object(REFERENCE_DIR / f"{args.image_id}.npz", args.size)
    print(f"object {args.image_id} ({split} split): {obj_info['source_shape_hw']} -> {obj.shape}, "
          f"range [{obj.min():.3f}, {obj.max():.3f}], mean {obj.mean():.3f}")

    # ---- 2. patterns + buckets (ONE sequence for every N) -------------------
    s2 = pattern_variance(args.p_on)
    patterns = generate_patterns(n_total, obj.shape, args.seed, args.p_on)
    buckets = bucket_measurements(patterns, obj)
    print(f"{n_total:,} binary patterns {patterns.shape}, buckets range [{buckets.min():.1f}, {buckets.max():.1f}]")

    # ---- 3. reconstructions + metrics ---------------------------------------
    rows, recon_cal, recon_mm, raw_gi = [], {}, {}, {}
    sum_o2 = float((obj ** 2).sum())
    for n in shots:
        gi = cgi_reconstruct(patterns, buckets, n)
        est, rec_c = normalize_calibrated(gi, s2)
        rec_m = normalize_minmax(gi)
        raw_gi[n], recon_cal[n], recon_mm[n] = gi, rec_c, rec_m
        unclipped_mse = float(((est - obj) ** 2).mean())
        for norm, rec in (("calibrated", rec_c), ("minmax", rec_m)):
            rows.append({"normalization": norm, "measurements": n, **evaluate(obj, rec),
                         "unclipped_MSE": unclipped_mse if norm == "calibrated" else np.nan,
                         "theory_MSE_sumO2_over_N": sum_o2 / n if norm == "calibrated" else np.nan})
        save_gray_png(rec_c, out / "images" / f"recon_{n}_calibrated.png")
        save_gray_png(rec_m, out / "images" / f"recon_{n}_minmax.png")
    save_gray_png(obj, out / "images" / "original.png")
    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics.csv", index=False)

    # ---- 4. validation checks -----------------------------------------------
    checks = {}
    # (a) prefix/determinism: regenerating a shorter sequence with the same seed gives the same first patterns
    short = generate_patterns(shots[0], obj.shape, args.seed, args.p_on)
    checks["prefix_patterns_identical"] = bool(np.array_equal(short, patterns[:shots[0]]))
    # (b) algebraic identity: <dB dP> == <B P> - <B><P>
    n = shots[-1]
    p = patterns.reshape(n, -1).astype(np.float64)
    alt = (buckets @ p) / n - buckets.mean() * p.mean(axis=0)
    checks["covariance_identity_max_abs_diff"] = float(np.abs(alt.reshape(obj.shape) - raw_gi[n]).max())
    # (c) patterns are statistically what we claim
    checks["pattern_mean_empirical"] = float(patterns.mean())
    checks["pattern_var_empirical"] = float(patterns.var())
    checks["pattern_var_theory"] = s2
    # (d) CGI noise law: Var[GI/s2 - O] ~= sum(O^2) / N for i.i.d. binary patterns
    cal = table[table.normalization == "calibrated"].sort_values("measurements")
    checks["unclipped_MSE_over_theory"] = {int(r.measurements): round(r.unclipped_MSE / r.theory_MSE_sumO2_over_N, 3)
                                           for r in cal.itertuples()}
    # (e) the ordering we expect, emerging (not forced) from the simulation
    for norm, g in table.groupby("normalization"):
        g = g.sort_values("measurements")
        checks[f"{norm}_PSNR_increasing"] = is_monotone(g["PSNR"].tolist(), True)
        checks[f"{norm}_SSIM_increasing"] = is_monotone(g["SSIM"].tolist(), True)
        checks[f"{norm}_MSE_decreasing"] = is_monotone(g["MSE"].tolist(), False)
    # (f) not a lucky seed
    if args.robustness_seeds:
        per_seed_ok, rob = seed_robustness(obj, shots, args.robustness_seeds, args.seed, args.p_on, s2)
        checks["robustness_monotone_per_seed"] = per_seed_ok
        rob.to_csv(out / "robustness_seeds_calibrated.csv")

    # ---- 5. figures + artefacts ---------------------------------------------
    title = f"Conventional CGI, onion {args.image_id} ({args.size}x{args.size}), binary random patterns"
    met_cal = {r.measurements: r._asdict() for r in cal.itertuples()}
    mm = table[table.normalization == "minmax"]
    met_mm = {r.measurements: r._asdict() for r in mm.itertuples()}
    save_comparison(obj, recon_cal, met_cal, out / "comparison_calibrated.png",
                    title + "  |  normalisation: GI / var(P), clipped to [0,1]")
    save_comparison(obj, recon_mm, met_mm, out / "comparison_minmax.png",
                    title + "  |  normalisation: min-max")
    save_metric_curves(table, out / "metric_curves.png", title)
    np.savez_compressed(
        out / "simulation.npz", object=obj, patterns_packed=np.packbits(patterns, axis=-1),
        pattern_shape=np.array(patterns.shape), buckets=buckets,
        **{f"gi_raw_{k}": v for k, v in raw_gi.items()},
    )
    config = {
        "image_id": args.image_id, "split": split, "size": args.size, "shots": shots, "seed": args.seed,
        "pattern_model": f"i.i.d. Bernoulli(p={args.p_on}) binary, values in {{0,1}}",
        "bucket_model": "B_i = sum_xy P_i(x,y) O(x,y), float64, noise-free",
        "reconstruction": "GI_N(x,y) = (1/N) sum_i (B_i - <B>_N)(P_i(x,y) - <P(x,y)>_N), first N shots",
        "normalization_primary": f"calibrated: GI / {s2} (theoretical pattern variance), clip to [0,1]",
        "normalization_secondary": "min-max of GI",
        "object_info": obj_info, "checks": checks, "runtime_s": round(time.time() - t0, 2),
    }
    (out / "validation.json").write_text(json.dumps(config, indent=2))

    # ---- 6. console report --------------------------------------------------
    pd.set_option("display.width", 140)
    for norm in ("calibrated", "minmax"):
        g = table[table.normalization == norm][["measurements", "PSNR", "SSIM", "MSE", "Pearson_r"]]
        print(f"\n[{norm}]\n" + g.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nchecks:", json.dumps(checks, indent=2))
    if args.robustness_seeds:
        print("\nrobustness over", args.robustness_seeds, "independent pattern sequences (calibrated):")
        print(rob.round(4).to_string())
    print(f"\noutputs -> {out}  ({config['runtime_s']} s)")


if __name__ == "__main__":
    main()
