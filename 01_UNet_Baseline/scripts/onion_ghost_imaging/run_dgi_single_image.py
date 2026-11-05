#!/usr/bin/env python3
"""Phase 1b: differential ghost imaging (DGI) vs conventional CGI on the SAME data.

    python run_dgi_single_image.py      # uses outputs/phase1_single_image/C_16_64px/simulation.npz

Requires the Phase 1 run (run_single_image.py) to exist. This script does NOT
regenerate a new pattern sequence: it loads the saved Phase 1 object, patterns
and buckets, verifies them bit-for-bit against a fresh regeneration with the
same seed, and verifies that its CGI results reproduce the Phase 1 metrics.csv.

Reconstructions (first N of the same 10,000 shots):
    CGI:  GI(x)  = < (B_i - <B>) (P_i(x) - <P(x)>) >,        O_hat = GI / s2
    DGI:  S_i    = B_i - (<B>/<R>) R_i,  R_i = sum_x P_i(x)
          DGI(x) = < S_i (P_i(x) - <P(x)>) >,                O_hat = DGI / s2 + <B>/<R>
Both estimates are clipped to [0, 1] for metrics. No ground truth is used in
reconstruction or normalisation.

Outputs: outputs/phase1b_dgi/<id>_<size>px/  (Phase 1 outputs are left untouched)
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
                               pattern_variance, reference_intensity)
from src.metrics import evaluate  # noqa: E402
from src.reconstruction import (cgi_reconstruct, dgi_reconstruct, normalize_calibrated,  # noqa: E402
                                normalize_dgi_calibrated)
from src.visualization import save_gray_png, save_method_curves, save_method_grid  # noqa: E402


def reconstruct_both(patterns, buckets, refs, n, s2):
    """Return {method: (unclipped estimate, clipped estimate)} plus DGI's k."""
    cgi_est, cgi_rec = normalize_calibrated(cgi_reconstruct(patterns, buckets, n), s2)
    dgi_raw, k = dgi_reconstruct(patterns, buckets, refs, n)
    dgi_est, dgi_rec = normalize_dgi_calibrated(dgi_raw, s2, k)
    return {"CGI": (cgi_est, cgi_rec), "DGI": (dgi_est, dgi_rec)}, k, dgi_raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-id", default="C_16")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--shots", type=int, nargs="+", default=[500, 1000, 5000, 10000])
    ap.add_argument("--seed", type=int, default=2026, help="seed used in Phase 1 (for verification)")
    ap.add_argument("--p-on", type=float, default=0.5)
    ap.add_argument("--robustness-seeds", type=int, default=5, help="extra independent sequences (0 = skip)")
    args = ap.parse_args()

    shots = sorted(args.shots)
    tag = f"{args.image_id}_{args.size}px"
    phase1 = ROOT / "outputs" / "phase1_single_image" / tag
    out = ROOT / "outputs" / "phase1b_dgi" / tag
    (out / "images").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- 1. load the exact Phase 1 data and verify it ------------------------
    sim = np.load(phase1 / "simulation.npz")
    obj = sim["object"]
    shape = tuple(sim["pattern_shape"])
    patterns = np.unpackbits(sim["patterns_packed"], axis=-1, count=shape[-1]).reshape(shape)
    buckets = sim["buckets"]
    refs = reference_intensity(patterns)
    s2 = pattern_variance(args.p_on)

    checks = {}
    obj_fresh, _ = load_object(ROOT / "data" / "v2" / "reference" / f"{args.image_id}.npz", args.size)
    checks["object_identical_to_phase1"] = bool(np.array_equal(obj_fresh, obj))
    checks["patterns_identical_to_seed_regeneration"] = bool(
        np.array_equal(generate_patterns(shape[0], shape[1:], args.seed, args.p_on), patterns))
    checks["buckets_identical_to_recomputation"] = bool(np.array_equal(bucket_measurements(patterns, obj), buckets))
    print(f"loaded Phase 1 data: object {obj.shape}, patterns {patterns.shape}, buckets {buckets.shape}")

    # ---- 2. reconstruct + evaluate -------------------------------------------
    rows, recons, ks = [], {"CGI": {}, "DGI": {}}, {}
    obj_mean, sum_o2 = float(obj.mean()), float((obj ** 2).sum())
    sum_dev2 = float(((obj - obj_mean) ** 2).sum())
    theory = {"CGI": {}, "DGI": {}}
    for n in shots:
        res, k, dgi_raw = reconstruct_both(patterns, buckets, refs, n, s2)
        ks[n] = k
        theory["CGI"][n], theory["DGI"][n] = sum_o2 / n, sum_dev2 / n
        p = patterns[:n].reshape(n, -1).astype(np.float64)
        alt = (buckets[:n] @ p) / n - k * (refs[:n] @ p) / n            # Ferri form <BP> - k<RP>
        checks[f"dgi_form_equivalence_max_abs_diff_N{n}"] = float(np.abs(alt.reshape(obj.shape) - dgi_raw).max())
        for method, (est, rec) in res.items():
            recons[method][n] = rec
            rows.append({"method": method, "measurements": n, **evaluate(obj, rec),
                         "unclipped_MSE": float(((est - obj) ** 2).mean()),
                         "theory_MSE": theory[method][n],
                         "clipped_fraction": float(((est < 0) | (est > 1)).mean())})
            save_gray_png(rec, out / "images" / f"{method.lower()}_{n}_calibrated.png")
    save_gray_png(obj, out / "images" / "original.png")
    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics_long.csv", index=False)

    # ---- 3. checks ------------------------------------------------------------
    p1 = pd.read_csv(phase1 / "metrics.csv")
    p1 = p1[p1.normalization == "calibrated"].set_index("measurements")
    cgi = table[table.method == "CGI"].set_index("measurements")
    checks["cgi_reproduces_phase1_max_abs_diff"] = float(
        max((p1.loc[shots, m] - cgi.loc[shots, m]).abs().max() for m in ("PSNR", "SSIM", "MSE")))
    checks["dgi_k_equals_<B>/<R>"] = {int(n): round(v, 5) for n, v in ks.items()}
    checks["diagnostic_only_true_object_mean"] = round(obj_mean, 5)   # GT, NOT used in reconstruction
    checks["predicted_MSE_gain_CGI_over_DGI"] = round(sum_o2 / sum_dev2, 2)
    checks["bucket_var_B"] = float(buckets.var())
    s_full = buckets - ks[shots[-1]] * refs
    checks["diff_signal_var_S"] = float(s_full.var())
    checks["var_reduction_B_over_S"] = round(float(buckets.var() / s_full.var()), 2)
    for method, g in table.groupby("method"):
        g = g.sort_values("measurements")
        checks[f"{method}_unclipped_MSE_over_theory"] = {int(r.measurements): round(r.unclipped_MSE / r.theory_MSE, 3)
                                                         for r in g.itertuples()}
        checks[f"{method}_PSNR_increasing"] = bool(np.all(np.diff(g["PSNR"]) > 0))
        checks[f"{method}_SSIM_increasing"] = bool(np.all(np.diff(g["SSIM"]) > 0))
        checks[f"{method}_MSE_decreasing"] = bool(np.all(np.diff(g["MSE"]) < 0))
        checks[f"{method}_r_increasing"] = bool(np.all(np.diff(g["Pearson_r"]) > 0))

    # robustness: same object, independent pattern sequences (NOT used for the headline numbers)
    if args.robustness_seeds:
        rob = []
        for j in range(args.robustness_seeds):
            seed = args.seed + 1000 + j
            pats = generate_patterns(shape[0], shape[1:], seed, args.p_on)
            bk, rf = bucket_measurements(pats, obj), reference_intensity(pats)
            for n in shots:
                res, _, _ = reconstruct_both(pats, bk, rf, n, s2)
                for method, (_, rec) in res.items():
                    rob.append({"seed": seed, "method": method, "measurements": n, **evaluate(obj, rec)})
        rob = pd.DataFrame(rob)
        rob_summary = rob.groupby(["method", "measurements"])[["PSNR", "SSIM", "MSE", "Pearson_r"]].agg(["mean", "std"])
        rob_summary.to_csv(out / "robustness_seeds.csv")
        wide_rob = rob.pivot_table(index=["seed", "measurements"], columns="method", values=["PSNR", "SSIM"])
        checks["robustness_DGI_beats_CGI_PSNR_all_runs"] = bool((wide_rob["PSNR"]["DGI"] > wide_rob["PSNR"]["CGI"]).all())
        checks["robustness_DGI_beats_CGI_SSIM_all_runs"] = bool((wide_rob["SSIM"]["DGI"] > wide_rob["SSIM"]["CGI"]).all())

    # ---- 4. side-by-side tables ----------------------------------------------
    wide = table.pivot(index="measurements", columns="method", values=["PSNR", "SSIM", "MSE", "Pearson_r"])
    wide.columns = [f"{m} {k}" for k, m in wide.columns]
    wide.to_csv(out / "cgi_vs_dgi.csv")

    # ---- 5. figures -----------------------------------------------------------
    base = f"Onion {args.image_id}, {args.size}x{args.size}, same 10,000 binary patterns (seed {args.seed})"
    met = {m: {r.measurements: r._asdict() for r in table[table.method == m].itertuples()} for m in ("CGI", "DGI")}
    save_method_grid(obj, recons, met, out / "grid_calibrated_0to1.png",
                     base + "\ncalibrated estimates, gray window [0, 1] (exactly what the metrics see)")
    lo, hi = (float(v) for v in np.percentile(obj, [1, 99]))
    save_method_grid(obj, recons, met, out / "grid_display_window.png",
                     base + f"\nDISPLAY ONLY: same fixed gray window [{lo:.3f}, {hi:.3f}] for every panel "
                            "(original's 1-99th percentile); metrics unchanged", window=(lo, hi))
    save_method_curves(table, out / "metric_curves_cgi_vs_dgi.png",
                       base + " - dashed: predicted MSE (CGI: sum O^2/N, DGI: sum (O-mean)^2/N)", theory)

    config = {
        "image_id": args.image_id, "size": args.size, "shots": shots, "seed": args.seed,
        "data_source": str((phase1 / "simulation.npz").relative_to(ROOT)),
        "cgi": "GI = <(B-<B>)(P-<P>)>, O_hat = GI/s2, clip [0,1]",
        "dgi": "S = B - (<B>/<R>) R, R = sum_x P; DGI = <S (P-<P>)>; O_hat = DGI/s2 + <B>/<R>, clip [0,1]",
        "s2": s2, "reference": "Ferri, Magatti, Lugiato, Gatti, PRL 104, 253603 (2010)",
        "checks": checks, "runtime_s": round(time.time() - t0, 2),
    }
    (out / "validation.json").write_text(json.dumps(config, indent=2))

    pd.set_option("display.width", 160)
    print("\n" + wide.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nclipped fraction:\n" + table.pivot(index="measurements", columns="method",
                                                 values="clipped_fraction").to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nchecks:", json.dumps(checks, indent=2))
    if args.robustness_seeds:
        print("\nrobustness (5 independent sequences):\n" + rob_summary.round(4).to_string())
    print(f"\noutputs -> {out}  ({config['runtime_s']} s)")


if __name__ == "__main__":
    main()
