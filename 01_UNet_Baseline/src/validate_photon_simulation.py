#!/usr/bin/env python3
"""STEP 4 sanity check of the Poisson photon-limited simulator (no dataset generation).

Checks
  1. flat-field counting test: E[N] = lambda, Var(N)/E[N] = 1, relative noise = 1/sqrt(lambda),
     and the empirical histogram matches the Poisson pmf (low and moderate lambda)
  2. real images, K repeated realizations: per-pixel Var(N) vs E[N] slope = 1, I_noisy unbiased
  3. image SNR vs P: log-log slope = 0.5 (SNR proportional to sqrt(P))
  4. FOV mask respected (no photons outside), determinism of seeds, clean references unchanged

Outputs -> data/v2/photon_simulation_preview/ (clean references are never written).
Only train/val images are used for previews; the test split is not touched.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from color_space import linear_to_srgb, load_reference_linear  # noqa: E402
from photon_simulation import (BASE_SEED, P_FULL, PHOTON_LEVELS, describe, rng_for,  # noqa: E402
                               simulate)

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data" / "v2"
OUT = V2 / "photon_simulation_preview"
REPS = ["A_344", "A_82", "C_13", "D_52", "F_49", "B_27"]   # one per source layout; all train split
K_REAL = 100        # repeated realizations for the per-pixel variance test
CROP = 256          # central crop used for the repeated-realization test
LEVEL_COLORS = ["#86b6ef", "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # sequential: light = low P


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def flat_field_test():
    rng = np.random.Generator(np.random.PCG64(BASE_SEED))
    rows = []
    n = 200_000
    for I in [0.01, 0.05, 0.2, 0.5, 1.0]:
        for name, P in PHOTON_LEVELS.items():
            lam = I * P
            N = rng.poisson(np.full(n, lam))
            mean, var = N.mean(), N.var(ddof=1)
            rows.append({"I": I, "level": name, "P": P, "lambda": lam, "mean_N": mean, "var_N": var,
                         "mean_over_lambda": mean / lam, "var_over_mean": var / mean if mean else np.nan,
                         "rel_noise": np.sqrt(var) / mean if mean else np.nan, "rel_noise_theory": 1 / np.sqrt(lam),
                         "se_var_over_mean": np.sqrt((2 * lam ** 2 + lam) / n) / lam})
    # pmf goodness of fit at low and moderate lambda
    gof = {}
    for lam in [0.5, 2.0, 10.0, 100.0]:
        N = rng.poisson(lam, size=n)
        k = np.arange(0, int(lam + 8 * np.sqrt(lam) + 10))
        emp = np.bincount(N, minlength=k.size)[:k.size] / n
        pmf = sstats.poisson.pmf(k, lam)
        gof[str(lam)] = {"total_variation": float(0.5 * np.abs(emp - pmf).sum())}
    return rows, gof


def central_window(mask, size):
    ys, xs = np.nonzero(mask)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    y0, x0 = max(0, cy - size // 2), max(0, cx - size // 2)
    return slice(y0, y0 + size), slice(x0, x0 + size)


def darkest_window(linear, mask, size=200, step=50):
    """Window (inside mask) with lowest mean luminance -> stained nuclei/chromosomes, hardest for shot noise."""
    lum = linear @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    H, W = lum.shape
    best, arg = 9, (0, 0)
    for y in range(0, H - size, step):
        for x in range(0, W - size, step):
            if mask[y:y + size, x:x + size].mean() < 0.999:
                continue
            v = np.percentile(lum[y:y + size, x:x + size], 20)
            if v < best:
                best, arg = v, (y, x)
    return slice(arg[0], arg[0] + size), slice(arg[1], arg[1] + size)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "samples").mkdir(exist_ok=True)
    manifest = {r["stem"]: r for r in csv.DictReader(open(V2 / "metadata" / "manifest.csv", encoding="utf-8"))}
    for s in REPS:
        assert manifest[s]["split"] in ("train", "val"), f"{s} is not train/val"
    ref_hash_before = {p.name: md5(p) for p in sorted((V2 / "reference").glob("*.npz"))}
    levels = list(PHOTON_LEVELS.items())
    report = {"model": describe(), "representatives": {s: manifest[s]["split"] for s in REPS}}

    # ---- 1. flat field -------------------------------------------------------------------------
    ff, gof = flat_field_test()
    report["flat_field"] = {"rows": ff, "pmf_goodness_of_fit": gof}
    ff_ok = all(abs(r["mean_over_lambda"] - 1) < 0.02 and abs(r["var_over_mean"] - 1) < max(0.03, 5 * r["se_var_over_mean"])
                for r in ff) and all(v["total_variation"] < 0.01 for v in gof.values())

    # ---- per-image simulation, stats, repeated realizations ----------------------------------------
    per_image, scatter, snr_rows = [], {name: [] for name, _ in levels}, []
    mask_ok = det_ok = True
    vis = {}
    for sid in REPS:
        split = manifest[sid]["split"]
        lin, m = load_reference_linear(V2 / "reference" / f"{sid}.npz")
        vis[sid] = {"clean": lin, "mask": m, "noisy": {}}
        lum_px = (lin @ np.array([0.2126, 0.7152, 0.0722], np.float32))[m]
        dark_thr, bright_thr = np.percentile(lum_px, 10), np.percentile(lum_px, 50)
        lum = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        dark_m, bright_m = m & (lum <= dark_thr), m & (lum >= bright_thr)
        for name, P in levels:
            noisy, N, lam = simulate(lin, m, P, rng_for(split, sid, P, 0))
            vis[sid]["noisy"][name] = noisy
            # mask + determinism
            mask_ok &= bool((N[~m] == 0).all() and (noisy[~m] == 0).all())
            again, _, _ = simulate(lin, m, P, rng_for(split, sid, P, 0))
            other, _, _ = simulate(lin, m, P, rng_for(split, sid, P, 1))
            det_ok &= bool(np.array_equal(noisy, again) and not np.array_equal(noisy, other))
            np.savez_compressed(OUT / "samples" / f"{sid}__{name.replace('%', 'pct')}.npz",
                                counts=N.astype(np.uint16), photons=np.float32(P), mask=m.astype(np.uint8),
                                seed_key=np.array(f"{split}|{sid}|{P:.6g}|0"))
            Ic = np.clip(lin, 0, 1)[m]
            err = (noisy - lin)[m]
            mse = float(np.mean(err ** 2))
            row = {"image": sid, "group": manifest[sid]["group"], "level": name, "P": P,
                   "mean_lambda_fov": float(lam[m].mean()), "mean_N_fov": float(N[m].mean()),
                   "mean_lambda_nuclei": float(lam[dark_m].mean()), "mean_lambda_background": float(lam[bright_m].mean()),
                   "dispersion_index_single": float(np.mean((N[m] - lam[m]) ** 2) / np.mean(lam[m])),
                   "bias_mean_Inoisy_minus_I": float(err.mean()),
                   "rmse_linear": float(np.sqrt(mse)),
                   "psnr_linear_db": float(10 * np.log10(1.0 / mse)),
                   "snr_image": float(np.sqrt(np.sum(Ic ** 2) / np.sum(err ** 2))),
                   "snr_image_theory": float(np.sqrt(P * np.sum(Ic ** 2) / np.sum(Ic))),
                   "snr_pixel_nuclei_theory": float(np.sqrt(lam[dark_m].mean())),
                   "snr_pixel_background_theory": float(np.sqrt(lam[bright_m].mean())),
                   "frac_pixels_zero_counts": float((N[m] == 0).mean()),
                   "frac_Inoisy_gt_1": float((noisy[m] > 1).mean())}
            per_image.append(row)

            # repeated realizations on a central crop
            ys, xs = central_window(m, CROP)
            lc, mc = lin[ys, xs], m[ys, xs]
            stack = np.stack([simulate(lc, mc, P, rng_for(split, sid, P, 1000 + k))[1] for k in range(K_REAL)])
            px_mean = stack.mean(0)[mc]
            px_var = stack.var(0, ddof=1)[mc]
            lam_c = (np.clip(lc, 0, 1) * P)[mc]
            slope, intercept = np.polyfit(px_mean.ravel(), px_var.ravel(), 1)
            rel = np.sqrt(px_var) / np.maximum(px_mean, 1e-9)
            snr_rows.append({"image": sid, "level": name, "P": P, "var_vs_mean_slope": float(slope),
                             "var_vs_mean_intercept": float(intercept),
                             "mean_var_over_mean": float(px_var.sum() / px_mean.sum()),
                             "mean_rel_bias_E[N]/lambda": float(px_mean.sum() / lam_c.sum()),
                             "median_rel_noise": float(np.median(rel[px_mean > 0])),
                             "median_rel_noise_theory": float(np.median(1 / np.sqrt(lam_c[lam_c > 0])))})
            sel = np.random.default_rng(0).choice(px_mean.size, size=min(3000, px_mean.size), replace=False)
            scatter[name].append((px_mean.ravel()[sel], px_var.ravel()[sel]))
        print(f"  {sid} done")

    report["per_image_level_stats"] = per_image
    report["repeated_realizations"] = {"K": K_REAL, "crop": CROP, "rows": snr_rows}

    # ---- 3. SNR scaling -----------------------------------------------------------------------
    slopes = {}
    for sid in REPS:
        r = [x for x in per_image if x["image"] == sid]
        b, a = np.polyfit(np.log([x["P"] for x in r]), np.log([x["snr_image"] for x in r]), 1)
        slopes[sid] = float(b)
    report["snr_loglog_slope_per_image"] = slopes

    ref_hash_after = {p.name: md5(p) for p in sorted((V2 / "reference").glob("*.npz"))}
    checks = {
        "flat_field_mean_var_pmf": ff_ok,
        "real_image_var_vs_mean_slope_within_0.05_of_1": all(abs(r["var_vs_mean_slope"] - 1) < 0.05 for r in snr_rows),
        "real_image_unbiased_within_1pct": all(abs(r["mean_rel_bias_E[N]/lambda"] - 1) < 0.01 for r in snr_rows),
        "dispersion_index_single_realization_within_0.05": all(abs(r["dispersion_index_single"] - 1) < 0.05
                                                                 for r in per_image),
        "snr_slope_0.5_pm_0.02": all(abs(s - 0.5) < 0.02 for s in slopes.values()),
        "relative_noise_decreases_with_P": all(
            all(np.diff([r["median_rel_noise"] for r in snr_rows if r["image"] == sid]) < 0) for sid in REPS),
        "mask_respected": mask_ok,
        "deterministic_seeds": det_ok,
        "clean_references_unchanged": ref_hash_before == ref_hash_after,
    }
    report["checks"] = checks
    report["ALL_PASSED"] = all(checks.values())
    (OUT / "sanity_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    with open(OUT / "noise_statistics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        w.writeheader()
        for r in per_image:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    make_figures(vis, levels, per_image, snr_rows, scatter, ff)
    print(json.dumps(checks, indent=1))
    print("ALL_PASSED", report["ALL_PASSED"])


# --------------------------------------------------------------------------------------- figures
def to_display(lin, mask):
    rgb = linear_to_srgb(np.clip(lin, 0, 1))
    rgb[~mask] = 0.5
    return rgb


def make_figures(vis, levels, per_image, snr_rows, scatter, ff):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#b5b4ae", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "axes.titlesize": 9})

    for sid, v in vis.items():
        lin, m = v["clean"], v["mask"]
        ys, xs = darkest_window(lin, m)
        ncol = 1 + len(levels)
        fig, ax = plt.subplots(3, ncol, figsize=(2.6 * ncol, 8.4), constrained_layout=True)
        cols = [("clean", lin)] + [(n, v["noisy"][n]) for n, _ in levels]
        for j, (name, img) in enumerate(cols):
            P = dict(levels).get(name)
            title = "clean reference" if P is None else f"{name}  P={P:g} photons"
            ax[0, j].imshow(to_display(img, m))
            ax[0, j].add_patch(plt.Rectangle((xs.start, ys.start), xs.stop - xs.start, ys.stop - ys.start,
                                             fill=False, ec="#e34948", lw=1))
            ax[0, j].set_title(title)
            ax[1, j].imshow(to_display(img[ys, xs], m[ys, xs]), interpolation="nearest")
            if P is None:
                ax[2, j].imshow(np.mean(np.clip(lin[ys, xs], 0, 1), axis=2) * 1000, cmap="viridis", vmin=0, vmax=1000)
                ax[2, j].set_title("expected photons at 100%\n(mean RGB, 0-1000)")
            else:
                errmap = np.abs(img - lin).mean(axis=2)[ys, xs]
                im = ax[2, j].imshow(errmap, cmap="magma", vmin=0, vmax=0.25)
                r = [x for x in per_image if x["image"] == sid and x["level"] == name][0]
                ax[2, j].set_title(f"|I_noisy - I| (linear)\nPSNR {r['psnr_linear_db']:.1f} dB, SNR {r['snr_image']:.1f}")
            for a in ax[:, j]:
                a.set_xticks([])
                a.set_yticks([])
        ax[1, 0].set_ylabel("zoom: darkest (stained) region")
        fig.colorbar(im, ax=ax[2, -1], fraction=0.046, label="abs error")
        fig.suptitle(f"{sid}: clean -> simulated photon-limited acquisition (Poisson, P = photons at I=1)",
                     x=0.01, ha="left", fontweight="bold")
        fig.savefig(OUT / f"levels_{sid}.png", dpi=110)
        plt.close(fig)

    # statistics figure
    fig, ax = plt.subplots(2, 2, figsize=(12, 9.5), constrained_layout=True)
    a = ax[0, 0]
    for (name, P), c in zip(levels, LEVEL_COLORS):
        mu = np.concatenate([s[0] for s in scatter[name]])
        var = np.concatenate([s[1] for s in scatter[name]])
        a.scatter(mu, var, s=3, alpha=0.35, color=c, label=f"{name} (P={P:g})", linewidths=0)
    lim = [0.05, 1500]
    a.plot(lim, lim, color="#0b0b0b", lw=1.2, label="Var = Mean (Poisson)")
    a.set_xscale("log"), a.set_yscale("log"), a.set_xlim(lim), a.set_ylim(lim)
    a.set_xlabel("per-pixel mean count E[N] over 100 realizations")
    a.set_ylabel("per-pixel variance Var(N)")
    a.set_title("Variance vs mean (6 images, central crops)", loc="left", fontweight="bold")
    a.legend(frameon=False, markerscale=4, fontsize=8)

    a = ax[0, 1]
    Ps = np.array([P for _, P in levels])
    for sid in vis:
        r = [x for x in per_image if x["image"] == sid]
        a.plot(Ps, [x["snr_image"] for x in r], marker="o", ms=5, lw=1.5, label=sid)
    ref = [x for x in per_image if x["image"] == list(vis)[0]]
    a.plot(Ps, ref[0]["snr_image"] * np.sqrt(Ps / Ps[0]), color="#0b0b0b", ls="--", lw=1, label="∝ sqrt(P)")
    a.set_xscale("log"), a.set_yscale("log")
    a.set_xlabel("photon budget P (photons at I = 1)")
    a.set_ylabel("image SNR = ||I|| / ||I_noisy - I||")
    a.set_title("SNR scaling", loc="left", fontweight="bold")
    a.legend(frameon=False, fontsize=8)

    a = ax[1, 0]
    lam = np.array([r["lambda"] for r in ff])
    a.scatter(lam, [r["rel_noise"] for r in ff], color="#2a78d6", s=18, zorder=3, label="flat field, 200k draws")
    xx = np.logspace(-1.2, 3.1, 100)
    a.plot(xx, 1 / np.sqrt(xx), color="#0b0b0b", lw=1, label="1/sqrt(lambda)")
    a.set_xscale("log"), a.set_yscale("log")
    a.set_xlabel("expected count lambda")
    a.set_ylabel("relative noise std(N)/E[N]")
    a.set_title("Relative noise vs expected photons", loc="left", fontweight="bold")
    a.legend(frameon=False)

    a = ax[1, 1]
    a.scatter(lam, [r["var_over_mean"] for r in ff], color="#2a78d6", s=18, zorder=3)
    a.axhline(1, color="#0b0b0b", lw=1)
    a.set_xscale("log")
    a.set_ylim(0.9, 1.1)
    a.set_xlabel("expected count lambda")
    a.set_ylabel("Var(N) / E[N]")
    a.set_title("Dispersion index (flat field)", loc="left", fontweight="bold")
    for aa in ax.ravel():
        for sp in ("top", "right"):
            aa.spines[sp].set_visible(False)
        aa.grid(color="#e6e5e0", lw=0.8)
        aa.set_axisbelow(True)
    fig.savefig(OUT / "poisson_statistics.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
