#!/usr/bin/env python3
"""Phase 3 diagnostic: strong conventional baselines vs frozen Models C/D (validation only, noise-free).

    python run_strong_baselines.py

Methods x N in {500, 1000, 2000, 5000, 10000}, on the 75 validation images:
    raw DGI, smoothed DGI (frozen sigma), pseudoinverse / least squares, TV (lambda tuned on
    40 training images, seed 2026), Model D (N=500), Model C (N=1000).
Seeds: 2026 (main sequence; NOT used to train C/D) and unseen 2027-2029.
Every reconstruction uses only the pattern matrix P_N and the buckets B_N = P_N O.
The test split is never loaded. Models C/D are loaded read-only.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.cs_reconstruction import pinv_solve, svd_patterns, tv_reconstruct  # noqa: E402
from src.ghost_imaging import load_object, pattern_batches, pattern_variance  # noqa: E402
from src.gi_unet_data import load_pairs, read_split_ids  # noqa: E402
from src.reconstruction import StreamingDGI  # noqa: E402
from src.unet_gi import UNetGI  # noqa: E402
from src.visualization import GRID, INK, INK_MUTED, save_panel_grid  # noqa: E402
from run_generalization_validation import smooth  # noqa: E402
from train_unet_gi import evaluate_set, predict  # noqa: E402

OUT = ROOT / "outputs" / "phase3_strong_baselines"
HIST = ROOT / "outputs" / "phase2b_unet"
SHOTS = [500, 1000, 2000, 5000, 10000]
SEEDS = [2026, 2027, 2028, 2029]
UNSEEN = [2027, 2028, 2029]
D = 64 * 64
SIGMA = {500: 1.75, 1000: 1.375, 2000: 1.125, 5000: 0.875, 10000: 0.75}   # frozen (training-tuned)
LAMBDAS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
N_TUNE_IMAGES = 40
TV_OUTER, TV_INNER = 300, 10
KEYS = ("PSNR", "SSIM", "MSE", "Pearson_r")
METHODS = ["raw DGI", "smoothed DGI", "pseudoinverse", "TV", "Model D", "Model C"]
COLORS = {"raw DGI": INK_MUTED, "smoothed DGI": "#2a78d6", "pseudoinverse": "#4a3aa7", "TV": "#1baf7a",
          "Model D": "#eb6834", "Model C": "#e87ba4"}


def log(msg):
    print(msg, flush=True)


def patterns_for(seed: int) -> np.ndarray:
    return np.concatenate(list(pattern_batches(max(SHOTS), (64, 64), seed, 500, 0.5))).reshape(max(SHOTS), D)


def dgi_from(p: np.ndarray, objects: np.ndarray) -> dict[int, np.ndarray]:
    """Phase 2A DGI (calibrated, clipped) read out at SHOTS from the given pattern matrix."""
    stream, out = StreamingDGI(objects), {}
    for k in range(0, max(SHOTS), 500):
        stream.update(p[k:k + 500].reshape(-1, 64, 64))
        if stream.n in SHOTS:
            est, _ = stream.estimate(pattern_variance(0.5))
            out[stream.n] = np.clip(est, 0, 1).reshape(-1, 1, 64, 64).astype(np.float32)
    return out


def linear_solvers(p: np.ndarray, b: np.ndarray, n: int):
    """Pseudoinverse solution (M, D), Vt for the TV data term (None if N >= D), conditioning info."""
    u, s, vt = svd_patterns(p[:n])
    x = pinv_solve(u, s, vt, b[:n])
    cond = {"shots": n, "regime": "underdetermined" if n < D else "overdetermined", "rank": int((s > 1e-12 * s[0]).sum()),
            "s_max": float(s[0]), "s_2": float(s[1]), "s_min": float(s[-1]), "condition_number": float(s[0] / s[-1])}
    return x, (vt if n < D else None), cond


def plot_curves(summ, key, label, path, title, log_y=False):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    off_marks = []
    for m in METHODS:
        g = summ[summ.method == m].sort_values("shots")
        if g.empty:
            continue
        if m.startswith("Model"):
            ax.errorbar(g.shots, g[f"{key}_mean"], yerr=g[f"{key}_std"], fmt="*", ms=15, color=COLORS[m],
                        mec="white", capsize=3, label=f"{m} ({int(g.shots.iloc[0])} shots)", zorder=5)
        else:
            y = g[f"{key}_mean"].to_numpy(dtype=float)
            off = ((~np.isfinite(y)) | (y > 100)) if key == "PSNR" else np.zeros(len(y), bool)
            ax.plot(g.shots[~off], y[~off], marker="o", ms=7, lw=2, color=COLORS[m], mec="white", mew=1.5,
                    ls="--" if m == "raw DGI" else "-", label=m)
            off_marks += [(xs, m) for xs in g.shots[off]]
    if off_marks:
        lo, top = ax.get_ylim()
        for xs, m in sorted(set(off_marks)):
            xx = xs * (0.95 if m == "pseudoinverse" else 1.05)
            ax.annotate("", (xx, top), xytext=(xx, top - 0.12 * (top - lo)),
                        arrowprops=dict(arrowstyle="->", color=COLORS[m], lw=2))
        ax.text(7000, top - 0.14 * (top - lo), "pseudoinverse & TV: exact recovery\n(PSNR ≈ 280 dB or ∞, off scale)",
                fontsize=8, color=INK, ha="center", va="top")
    ax.axvline(D, color=INK_MUTED, lw=1, ls=":")
    ax.annotate("N = 4096\n(unknowns)", (D, ax.get_ylim()[0]), xytext=(4, 8), textcoords="offset points",
                fontsize=8, color=INK_MUTED)
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xticks(SHOTS, [f"{n:,}" for n in SHOTS])
    ax.minorticks_off()
    ax.set_xlabel("number of measurements N (log scale)", color=INK_MUTED)
    ax.set_title(label, fontsize=10.5, color=INK, loc="left")
    ax.grid(color=GRID, lw=0.8)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(colors=INK_MUTED)
    ax.legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle(title, fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    t0 = time.time()
    checks, cond_rows = {}, []
    ids = read_split_ids()
    ref = ROOT / "data" / "v2" / "reference"

    _, y_va, id_va = load_pairs("val", 1000)
    _, _, id_tr = load_pairs("train", 1000)
    assert len(id_va) == 75 and not (set(id_va) | set(id_tr)) & set(ids["test"])
    obj_va = np.stack([load_object(ref / f"{i}.npz", 64)[0] for i in id_va])
    tune_ids = sorted(np.random.default_rng(0).choice(id_tr, N_TUNE_IMAGES, replace=False).tolist())
    obj_tune = np.stack([load_object(ref / f"{i}.npz", 64)[0] for i in tune_ids])
    checks["tv_tuning_train_images"] = tune_ids

    # ---- 1. TV lambda selection on 40 TRAINING images (seed 2026) ------------------------
    p26 = patterns_for(2026)
    b_tune = p26.astype(np.float64) @ obj_tune.reshape(len(obj_tune), -1).T          # (10000, 40)
    y_tune = obj_tune[:, None].astype(np.float32)
    lam_rows, lam_star = [], {}
    for n in SHOTS:
        xp, vt, _ = linear_solvers(p26, b_tune, n)
        for lam in LAMBDAS:
            tl = time.time()
            rec, _ = tv_reconstruct(xp, vt, lam, n_outer=TV_OUTER, n_inner=TV_INNER)
            e = evaluate_set(y_tune, rec[:, None].astype(np.float32))
            lam_rows.append({"shots": n, "lambda": lam, **{f"train_{k}": e[k] for k in KEYS},
                             "seconds": round(time.time() - tl, 1)})
            log(f"tune N={n} lam={lam:g}: train MSE {e['MSE']:.6f} PSNR {e['PSNR']:.2f} SSIM {e['SSIM']:.3f}")
        t = pd.DataFrame([r for r in lam_rows if r["shots"] == n])
        lam_star[n] = float(t.loc[t.train_MSE.idxmin(), "lambda"])
    lam_df = pd.DataFrame(lam_rows)
    lam_df.to_csv(OUT / "tv_lambda_selection.csv", index=False)
    checks["tv_lambda_frozen"] = lam_star
    checks["tv_lambda_at_grid_edge"] = {n: (l == LAMBDAS[-1]) or (l == LAMBDAS[0] and n < D) for n, l in lam_star.items()}
    log(f"frozen lambda: {lam_star}")

    # ---- 2. frozen models -------------------------------------------------------------
    models = {}
    for name, sub in (("Model D", "model_D_random_patterns_dgi500"), ("Model C", "model_C_random_patterns")):
        ck = torch.load(HIST / sub / "checkpoints" / "best.pt", weights_only=False)
        m = UNetGI(base=ck["config"]["base"])
        m.load_state_dict(ck["model"])
        models[name] = m

    # ---- 3. validation: all methods, all seeds ---------------------------------------------
    rows, keep, conv = [], {}, []
    for seed in SEEDS:
        p = p26 if seed == 2026 else patterns_for(seed)
        b = p.astype(np.float64) @ obj_va.reshape(75, -1).T                               # (10000, 75)
        dgi = dgi_from(p, obj_va)
        if seed == 2026:
            checks["dgi_seed2026_vs_phase2A_max_abs_diff"] = float(max(
                np.abs(dgi[n] - load_pairs("val", n)[0]).max() for n in (500, 1000, 5000, 10000)))
        for n in SHOTS:
            tn = time.time()
            xp, vt, c = linear_solvers(p, b, n)
            c["seed"] = seed
            exact_err = np.abs(xp - obj_va.reshape(75, -1)).max()
            c["pinv_max_abs_error_vs_truth_diagnostic_only"] = float(exact_err)
            c["pinv_data_residual_rel"] = float(np.linalg.norm(p[:n].astype(np.float64) @ xp.T - b[:n])
                                                / np.linalg.norm(b[:n]))
            c["pinv_fraction_outside_0_1"] = float(((xp < 0) | (xp > 1)).mean())
            cond_rows.append(c)
            track = seed == 2026 and n in (500, 1000)
            tv, trace = tv_reconstruct(xp, vt, lam_star[n], n_outer=TV_OUTER, n_inner=TV_INNER, track=track)
            if track:
                conv += [{"shots": n, **r} for r in trace]
            imgs = {"raw DGI": dgi[n], "smoothed DGI": smooth(dgi[n], SIGMA[n]),
                    "pseudoinverse": np.clip(xp, 0, 1).reshape(75, 1, 64, 64).astype(np.float32),
                    "TV": tv[:, None].astype(np.float32)}
            if n == 500:
                imgs["Model D"] = predict(models["Model D"], dgi[500])
            if n == 1000:
                imgs["Model C"] = predict(models["Model C"], dgi[1000])
            for m, img in imgs.items():
                e = evaluate_set(y_va, img)["per_image"]
                for j, image_id in enumerate(id_va):
                    rows.append({"image_id": image_id, "seed": seed, "method": m, "shots": n,
                                 **{k: float(e[k].iloc[j]) for k in KEYS}})
                if seed == 2026:
                    keep[(m, n)] = np.clip(img, 0, 1)
            if seed == 2026:
                keep[("pinv_unclipped", n)] = xp.reshape(75, 64, 64).astype(np.float32)
            log(f"seed {seed} N={n}: done in {time.time() - tn:.1f}s (pinv max err {exact_err:.2e}, cond {c['condition_number']:.1f})")

    per = pd.DataFrame(rows)
    per.to_csv(OUT / "per_image_metrics.csv", index=False)
    pd.DataFrame(cond_rows).to_csv(OUT / "conditioning_and_exactness.csv", index=False)
    pd.DataFrame(conv).to_csv(OUT / "tv_convergence_trace.csv", index=False)

    # TV convergence check: 3x more iterations on 20 val images, seed 2026, N = 500 / 1000 / 2000
    b26 = p26.astype(np.float64) @ obj_va[:20].reshape(20, -1).T
    conv_chk = {}
    for n in (500, 1000, 2000):
        xp, vt, _ = linear_solvers(p26, b26, n)
        a, _ = tv_reconstruct(xp, vt, lam_star[n], n_outer=TV_OUTER, n_inner=TV_INNER)
        bb, _ = tv_reconstruct(xp, vt, lam_star[n], n_outer=3 * TV_OUTER, n_inner=TV_INNER)
        ea = evaluate_set(y_va[:20], a[:, None].astype(np.float32))
        eb = evaluate_set(y_va[:20], bb[:, None].astype(np.float32))
        conv_chk[n] = {f"PSNR_{TV_OUTER}it": ea["PSNR"], f"PSNR_{3 * TV_OUTER}it": eb["PSNR"],
                       f"SSIM_{TV_OUTER}it": ea["SSIM"], f"SSIM_{3 * TV_OUTER}it": eb["SSIM"],
                       "max_abs_image_change": float(np.abs(a - bb).max())}
    checks["tv_convergence_300_vs_900_iterations"] = conv_chk

    # model consistency with their own reports (unseen seeds)
    rep_d = json.loads((HIST / "model_D_random_patterns_dgi500" / "training_report.json").read_text())
    rep_c = json.loads((HIST / "model_C_random_patterns" / "training_report.json").read_text())
    want_d = next(x for x in rep_d["unseen_table"] if x["method"].startswith("Model D"))["PSNR_mean"]
    want_c = next(x for x in rep_c["unseen_summary"] if x["method"].startswith("Model C"))["PSNR_mean"]
    un = per[per.seed.isin(UNSEEN)]
    checks["model_D_unseen_PSNR_vs_report_abs_diff"] = abs(un[un.method == "Model D"].PSNR.mean() - want_d)
    checks["model_C_unseen_PSNR_vs_report_abs_diff"] = abs(un[un.method == "Model C"].PSNR.mean() - want_c)

    # ---- 4. summaries ------------------------------------------------------------------
    def summarize(df, group):
        out = []
        for (m, n), g in df.groupby(["method", "shots"]):
            r = {"seed_group": group, "method": m, "shots": n, "n_pairs": len(g)}
            for k in KEYS:
                r[f"{k}_mean"], r[f"{k}_std"], r[f"{k}_median"] = g[k].mean(), g[k].std(), g[k].median()
            out.append(r)
        o = pd.DataFrame(out)
        o["order"] = o.method.map({m: i for i, m in enumerate(METHODS)})
        return o.sort_values(["shots", "order"]).drop(columns="order")

    summ_un = summarize(per[per.seed.isin(UNSEEN)], "unseen 2027-2029")
    summ_26 = summarize(per[per.seed == 2026], "main sequence 2026")
    summ = pd.concat([summ_un, summ_26], ignore_index=True)
    summ.to_csv(OUT / "summary.csv", index=False)
    for k in ("PSNR", "SSIM", "Pearson_r"):
        summ_un.pivot(index="method", columns="shots", values=f"{k}_mean").reindex(METHODS).to_csv(
            OUT / f"table_{k}_unseen.csv")

    # paired comparisons (unseen seeds, per image-seed pair)
    wide = per[per.seed.isin(UNSEEN)].pivot_table(index=["image_id", "seed"], columns=["method", "shots"],
                                                    values=["PSNR", "SSIM", "Pearson_r"])
    pairs = [("Model D", 500, "pseudoinverse", 500), ("Model D", 500, "TV", 500), ("Model D", 500, "smoothed DGI", 500),
             ("Model D", 500, "TV", 1000), ("Model C", 1000, "pseudoinverse", 1000), ("Model C", 1000, "TV", 1000),
             ("Model C", 1000, "smoothed DGI", 1000), ("Model C", 1000, "TV", 2000), ("Model C", 1000, "pseudoinverse", 2000),
             ("Model C", 1000, "smoothed DGI", 2000), ("TV", 500, "smoothed DGI", 500), ("TV", 1000, "smoothed DGI", 1000)]
    prow = []
    for a, na, b_, nb in pairs:
        r = {"A": f"{a}-{na}", "B": f"{b_}-{nb}"}
        for k in ("PSNR", "SSIM", "Pearson_r"):
            d = wide[(k, a, na)] - wide[(k, b_, nb)]
            r[f"{k}_mean_diff"], r[f"{k}_median_diff"], r[f"{k}_pct_A_better"] = d.mean(), d.median(), (d > 0).mean() * 100
        prow.append(r)
    paired = pd.DataFrame(prow)
    paired.to_csv(OUT / "paired_comparisons_unseen.csv", index=False)

    # ---- 5. reconstructions + figures -----------------------------------------------------
    np.savez_compressed(OUT / "reconstructions_seed2026_val.npz", image_id=np.array(id_va), shots=np.array(SHOTS),
                        **{f"pinv_unclipped_{n}": keep[("pinv_unclipped", n)] for n in SHOTS},
                        **{f"tv_{n}": keep[("TV", n)][:, 0] for n in SHOTS})
    base = "75 val images x unseen seeds 2027-2029 (mean ± std over 225 pairs); noise-free buckets"
    for k, fname, label, ly in (("PSNR", "psnr_vs_shots.png", "PSNR (dB)", False),
                                ("SSIM", "ssim_vs_shots.png", "SSIM", False),
                                ("MSE", "mse_vs_shots.png", "MSE (log)", True),
                                ("Pearson_r", "correlation_vs_shots.png", "Pearson r", False)):
        plot_curves(summ_un, k, label, OUT / fname, f"Conventional baselines vs AI: {label}\n{base}", log_y=ly)

    ex_ids = json.loads((HIST / "model_A_dgi500" / "full" / "training_report.json").read_text())["val_example_ids"]
    p26v = per[per.seed == 2026].set_index(["image_id", "method", "shots"])
    for n, extra in ((500, "Model D"), (1000, "Model C"), (10000, None)):
        rows_fig = []
        for image_id in ex_ids:
            j = id_va.index(image_id)
            panels = [("Ground truth", y_va[j, 0])]
            for m in ["raw DGI", "smoothed DGI", "pseudoinverse", "TV"] + ([extra] if extra else []):
                r = p26v.loc[(image_id, m, n)]
                panels.append((f"{m}-{n}\n{r.PSNR:.2f} dB | SSIM {r.SSIM:.3f}", keep[(m, n)][j, 0]))
            rows_fig.append({"label": f"{image_id} (val)", "target": y_va[j, 0], "panels": panels})
        save_panel_grid(rows_fig, OUT / f"reconstructions_N{n}.png",
                        f"N = {n:,} measurements ({'under' if n < D else 'over'}determined: {n} equations, {D} unknowns); "
                        "seed 2026, noise-free; gray window [0, 1] for every panel")

    cfg = {"measurement_model": "B = P O, P in {0,1}^(N x 4096) i.i.d. Bernoulli(0.5), noise-free, float64",
           "pseudoinverse": "thin SVD P = U S V^T; O = V S^-1 U^T B (min-norm if N<4096, least squares if N>=4096); "
                            "singular values < 1e-12 s_max dropped; result clipped to [0,1] for metrics",
           "tv": {"objective": "0.5||W(P O - B)||^2 + lambda * TV_iso(O), O in [0,1]; W whitens residuals so the data "
                               "term is 0.5||V^T(O - O_pinv)||^2 (N<4096) or 0.5||O - O_ls||^2 (N>=4096)",
                  "solver": f"FISTA (step 1 = 1/L), {TV_OUTER} iterations; prox = FGP (Beck-Teboulle) with box [0,1], "
                            f"{TV_INNER} inner iterations, warm-started dual",
                  "init": "clip(O_pinv, 0, 1)",
                  "lambda_grid": LAMBDAS, "lambda_selection": f"lowest mean MSE on {N_TUNE_IMAGES} training images "
                                                             "(seed 2026), frozen per N", "lambda_frozen": lam_star},
           "smoothed_dgi_sigma": SIGMA, "seeds": SEEDS, "test_images_used": 0}
    (OUT / "solver_config.json").write_text(json.dumps(cfg, indent=2))
    report = {"checks": checks, "summary_unseen": summ_un.to_dict("records"), "summary_seed2026": summ_26.to_dict("records"),
              "paired_unseen": paired.to_dict("records"), "conditioning": cond_rows,
              "runtime_s": round(time.time() - t0, 1)}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=float))

    pd.set_option("display.width", 230)
    cols = ["method", "shots"] + [f"{k}_{s}" for k in KEYS for s in ("mean", "std", "median")]
    log("\nUNSEEN seeds 2027-2029:\n" + summ_un[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    log("\nseed 2026:\n" + summ_26[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    log("\nconditioning:\n" + pd.DataFrame(cond_rows).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    log("\npaired (unseen):\n" + paired.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    log("\nchecks: " + json.dumps(checks, indent=1, default=float))
    log(f"\n{report['runtime_s']} s -> {OUT}")


if __name__ == "__main__":
    main()
