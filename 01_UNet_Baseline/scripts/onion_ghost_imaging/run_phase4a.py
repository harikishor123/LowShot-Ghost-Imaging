#!/usr/bin/env python3
"""Phase 4A: conventional ghost-image reconstruction under noisy bucket measurements (validation only).

    python phase4a_noise_setup.py     # step 1 (already run): fixes noise_config.json from training statistics
    python run_phase4a.py             # step 2: this script

Noise regimes are read from outputs/phase4a_noisy_baselines/noise_config.json and never changed here.
Methods per (regime, N): raw DGI, smoothed DGI, pseudoinverse, Tikhonov, TV (ADMM).
Every hyperparameter (sigma, Tikhonov lambda, TV lambda) is chosen by lowest mean MSE on TRAINING images
measured with pattern seed 2026 and its own noise draw; then frozen for the 75 validation images measured
with seeds 2027-2029 (independent patterns AND noise). The test split is never loaded. No model is trained.
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.cs_reconstruction import TVADMM, svd_patterns, tikhonov_from_svd  # noqa: E402
from src.ghost_imaging import load_object, pattern_batches  # noqa: E402
from src.gi_unet_data import load_pairs, read_split_ids  # noqa: E402
from src.noise_model import REGIME_ORDER, add_bucket_noise, load_noise_config, noise_rng  # noqa: E402
from src.visualization import GRID, INK, INK_MUTED, save_panel_grid  # noqa: E402
from run_generalization_validation import smooth  # noqa: E402
from train_unet_gi import evaluate_set  # noqa: E402

OUT = ROOT / "outputs" / "phase4a_noisy_baselines"
SHOTS = [500, 1000, 2000, 5000, 10000]
D = 4096
TUNE_SEED = 2026
VAL_SEEDS = [2027, 2028, 2029]
SIGMAS = [0.0] + [0.25 + 0.125 * k for k in range(15)] + [2.25, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
TIK_LAMBDAS = [float(10 ** (k / 2)) for k in range(-2, 15)]            # 0.1 .. 1e7  (0 = pseudoinverse)
TV_LAMBDAS = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
TV_ITERS = 200
N_TV_TUNE = 40
KEYS = ("PSNR", "SSIM", "MSE", "Pearson_r")
METHODS = ["raw DGI", "smoothed DGI", "pseudoinverse", "Tikhonov", "TV"]
COLORS = {"raw DGI": INK_MUTED, "smoothed DGI": "#2a78d6", "pseudoinverse": "#4a3aa7", "Tikhonov": "#eda100",
          "TV": "#1baf7a"}
FIG_SHOTS = [500, 1000, 2000, 10000]
if "--smoke" in sys.argv:                                              # tiny end-to-end code check only
    OUT = Path(sys.argv[sys.argv.index("--smoke") + 1])
    SHOTS, VAL_SEEDS, TV_LAMBDAS, TV_ITERS, N_TV_TUNE = [500, 1000, 10000], [2027], [0.1, 1.0], 10, 5
    SIGMAS, TIK_LAMBDAS = SIGMAS[::6], TIK_LAMBDAS[::4]
# "useful regime" criteria, fixed before any validation result is computed
USEFUL = {"min_gain_db_10000_over_500_and_1000": 3.0, "max_best_psnr_10000_db": 45.0, "min_best_ssim_10000": 0.5}


def log(msg):
    print(msg, flush=True)


def rho_for(lam: float) -> float:
    return 10.0 * lam


def patterns_for(seed: int) -> np.ndarray:
    return np.concatenate(list(pattern_batches(max(SHOTS), (64, 64), seed, 500, 0.5))).reshape(max(SHOTS), D)


def dgi_from_buckets(p_n: np.ndarray, b_n: np.ndarray) -> np.ndarray:
    """Phase 2A calibrated DGI from GIVEN (possibly noisy) buckets: (M, 1, 64, 64) clipped to [0, 1]."""
    pf = p_n.astype(np.float64)
    r = pf.sum(1)
    k = b_n.mean(0) / r.mean()
    s = b_n - r[:, None] * k[None, :]
    dgi = (s.T @ (pf - pf.mean(0))) / len(pf)
    est = dgi / 0.25 + k[:, None]
    return np.clip(est, 0, 1).reshape(-1, 1, 64, 64).astype(np.float32)


def mse_per_image(rec: np.ndarray, obj: np.ndarray) -> np.ndarray:
    return ((rec.reshape(len(obj), -1) - obj.reshape(len(obj), -1)) ** 2).mean(1)


def noisy(b_clean, seed, ri, reg, mu_ref):
    return add_bucket_noise(b_clean, reg["photons_per_mean_bucket"], reg["kappa"], mu_ref, noise_rng(seed, ri))


def bucket_stats(b_clean, b_noisy):
    n = b_noisy - b_clean
    return {"clean_mean": float(b_clean.mean()), "clean_std_over_patterns_median": float(np.median(b_clean.std(0))),
            "noise_std_median": float(np.median(n.std(0))),
            "fluct_snr_median": float(np.median(b_clean.std(0) / n.std(0))),
            "dc_snr_median": float(np.median(b_clean.mean(0) / n.std(0))),
            "negative_readings": int((b_noisy < 0).sum())}


def plot_metric(summ, key, label, path, title, log_y=False):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=not log_y)
    for ax, regime in zip(axes, REGIME_ORDER):
        g0 = summ[summ.regime == regime]
        for m in METHODS:
            g = g0[g0.method == m].sort_values("shots")
            ax.plot(g.shots, g[f"{key}_mean"], marker="o", ms=6, lw=2, color=COLORS[m], mec="white", mew=1.2,
                    ls="--" if m == "raw DGI" else "-", label=m)
        ax.axvline(D, color=INK_MUTED, lw=1, ls=":")
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xticks(SHOTS, [f"{n:,}" for n in SHOTS])
        ax.minorticks_off()
        ax.set_title(f"{regime} noise", fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("number of measurements N (log)", color=INK_MUTED)
        ax.grid(color=GRID, lw=0.8)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.tick_params(colors=INK_MUTED)
    axes[0].set_ylabel(label, color=INK_MUTED)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    cfg = load_noise_config()
    assert cfg.get("fixed_before_validation")
    mu_ref, regimes = cfg["mu_ref"], cfg["regimes"]
    (OUT / "reconstructions").mkdir(parents=True, exist_ok=True)
    checks = {"noise_config_used": {r: regimes[r] for r in REGIME_ORDER}, "useful_regime_criteria": USEFUL}
    ids = read_split_ids()
    ref = ROOT / "data" / "v2" / "reference"
    _, _, id_tr = load_pairs("train", 1000)
    _, y_va, id_va = load_pairs("val", 1000)
    assert len(id_tr) == 341 and len(id_va) == 75 and not (set(id_tr) | set(id_va)) & set(ids["test"])
    obj_tr = np.stack([load_object(ref / f"{i}.npz", 64)[0] for i in id_tr])
    obj_va = np.stack([load_object(ref / f"{i}.npz", 64)[0] for i in id_va])
    tv_idx = np.sort(np.random.default_rng(0).choice(len(id_tr), N_TV_TUNE, replace=False))

    # ================= 1. tuning on TRAINING images (pattern seed 2026, own noise) =================
    p26 = patterns_for(TUNE_SEED)
    btr_clean = p26.astype(np.float64) @ obj_tr.reshape(len(obj_tr), -1).T
    btr = {r: noisy(btr_clean, TUNE_SEED, ri, regimes[r], mu_ref) for ri, r in enumerate(REGIME_ORDER)}
    checks["dgi_from_buckets_matches_phase2A_clean_max_abs_diff"] = float(max(
        np.abs(dgi_from_buckets(p26[:n], btr_clean[:n]) - load_pairs("train", n)[0]).max() for n in (500, 1000)))
    stat_rows = [{"split": "train", "seed": TUNE_SEED, "regime": r, **bucket_stats(btr_clean, btr[r])} for r in REGIME_ORDER]

    sig_rows, tik_rows, tv_rows = [], [], []
    sigma, tik_lam, tv_lam = {}, {}, {}
    for n in SHOTS:
        tn = time.time()
        u, s, vt = svd_patterns(p26[:n])
        tik_init = {}
        for r in REGIME_ORDER:
            b = btr[r][:n]
            dg = dgi_from_buckets(p26[:n], b)
            m = [float(mse_per_image(smooth(dg, sg), obj_tr).mean()) for sg in SIGMAS]
            sig_rows += [{"regime": r, "shots": n, "sigma": sg, "train_MSE": v} for sg, v in zip(SIGMAS, m)]
            sigma[(r, n)] = SIGMAS[int(np.argmin(m))]
            utb = u.T @ b
            pinv_mse = float(mse_per_image(np.clip(tikhonov_from_svd(u, s, vt, utb, 0.0), 0, 1), obj_tr).mean())
            tik_rows.append({"regime": r, "shots": n, "lambda": 0.0, "train_MSE": pinv_mse})
            m = []
            for lam in TIK_LAMBDAS:
                v = float(mse_per_image(np.clip(tikhonov_from_svd(u, s, vt, utb, lam), 0, 1), obj_tr).mean())
                m.append(v)
                tik_rows.append({"regime": r, "shots": n, "lambda": lam, "train_MSE": v})
            tik_lam[(r, n)] = TIK_LAMBDAS[int(np.argmin(m))]
            tik_init[r] = tikhonov_from_svd(u, s, vt, u.T @ btr[r][:n, tv_idx], tik_lam[(r, n)]).T   # (D, 40)
        for lam in TV_LAMBDAS:                       # factorisation shared by the three regimes
            solver = TVADMM(p26[:n], rho_for(lam))
            for r in REGIME_ORDER:
                rec, _ = solver.solve(btr[r][:n, tv_idx], lam, n_iter=TV_ITERS, x0=tik_init[r])
                tv_rows.append({"regime": r, "shots": n, "lambda": lam,
                                "train_MSE": float(mse_per_image(rec, obj_tr[tv_idx]).mean())})
            del solver
        for r in REGIME_ORDER:
            t = pd.DataFrame([x for x in tv_rows if x["regime"] == r and x["shots"] == n])
            tv_lam[(r, n)] = float(t.loc[t.train_MSE.idxmin(), "lambda"])
        log(f"tuned N={n} in {time.time() - tn:.0f}s: sigma " + str({r: sigma[(r, n)] for r in REGIME_ORDER})
            + " tik " + str({r: tik_lam[(r, n)] for r in REGIME_ORDER}) + " tv " + str({r: tv_lam[(r, n)] for r in REGIME_ORDER}))
        del u, s, vt
    pd.DataFrame(sig_rows).to_csv(OUT / "tuning_smoothing_sigma_train.csv", index=False)
    pd.DataFrame(tik_rows).to_csv(OUT / "tuning_tikhonov_lambda_train.csv", index=False)
    pd.DataFrame(tv_rows).to_csv(OUT / "tuning_tv_lambda_train.csv", index=False)
    selected = pd.DataFrame([{"regime": r, "shots": n, "smoothing_sigma": sigma[(r, n)], "tikhonov_lambda": tik_lam[(r, n)],
                              "tv_lambda": tv_lam[(r, n)], "tv_rho": rho_for(tv_lam[(r, n)])}
                             for r in REGIME_ORDER for n in SHOTS])
    selected.to_csv(OUT / "selected_parameters.csv", index=False)
    edge = {"sigma": [f"{r}-{n}" for (r, n), v in sigma.items() if v == SIGMAS[-1]],
            "tikhonov": [f"{r}-{n}" for (r, n), v in tik_lam.items() if v in (TIK_LAMBDAS[0], TIK_LAMBDAS[-1])],
            "tv": [f"{r}-{n}" for (r, n), v in tv_lam.items() if v in (TV_LAMBDAS[0], TV_LAMBDAS[-1])]}
    checks["parameters_at_grid_edge"] = edge

    # ================= 2. validation: seeds 2027-2029, frozen parameters =================
    rows, keep = [], {}
    for seed in VAL_SEEDS:
        p = patterns_for(seed)
        bclean = p.astype(np.float64) @ obj_va.reshape(75, -1).T
        bval = {r: noisy(bclean, seed, ri, regimes[r], mu_ref) for ri, r in enumerate(REGIME_ORDER)}
        stat_rows += [{"split": "val", "seed": seed, "regime": r, **bucket_stats(bclean, bval[r])} for r in REGIME_ORDER]
        for n in SHOTS:
            tn = time.time()
            u, s, vt = svd_patterns(p[:n])
            solvers = {}
            for r in REGIME_ORDER:
                b = bval[r][:n]
                dg = dgi_from_buckets(p[:n], b)
                utb = u.T @ b
                tik = tikhonov_from_svd(u, s, vt, utb, tik_lam[(r, n)])
                lam = tv_lam[(r, n)]
                if lam not in solvers:
                    solvers[lam] = TVADMM(p[:n], rho_for(lam))
                tv, _ = solvers[lam].solve(b, lam, n_iter=TV_ITERS, x0=tik.T)
                imgs = {"raw DGI": dg, "smoothed DGI": smooth(dg, sigma[(r, n)]),
                        "pseudoinverse": np.clip(tikhonov_from_svd(u, s, vt, utb, 0.0), 0, 1).reshape(75, 1, 64, 64),
                        "Tikhonov": np.clip(tik, 0, 1).reshape(75, 1, 64, 64), "TV": tv[:, None]}
                for mth, img in imgs.items():
                    e = evaluate_set(y_va, img.astype(np.float32))["per_image"]
                    for j, image_id in enumerate(id_va):
                        rows.append({"regime": r, "seed": seed, "shots": n, "method": mth, "image_id": image_id,
                                     **{k: float(e[k].iloc[j]) for k in KEYS}})
                    if seed == VAL_SEEDS[0] and n in FIG_SHOTS:
                        keep[(r, n, mth)] = np.clip(img, 0, 1).astype(np.float32)
            del solvers, u, s, vt
            log(f"val seed {seed} N={n} done in {time.time() - tn:.0f}s")
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "per_image_metrics.csv", index=False)
    pd.DataFrame(stat_rows).to_csv(OUT / "bucket_statistics.csv", index=False)

    # TV convergence: 200 vs 600 iterations at the frozen lambda (20 val images, seed 2027)
    p = patterns_for(VAL_SEEDS[0])
    bclean = p.astype(np.float64) @ obj_va[:20].reshape(20, -1).T
    conv = []
    for ri, r in enumerate(REGIME_ORDER):
        b_all = noisy(bclean, VAL_SEEDS[0], ri, regimes[r], mu_ref)
        for n in (1000, 10000):
            lam = tv_lam[(r, n)]
            sol = TVADMM(p[:n], rho_for(lam))
            a, tr = sol.solve(b_all[:n], lam, n_iter=TV_ITERS, track=True)
            bb, _ = sol.solve(b_all[:n], lam, n_iter=3 * TV_ITERS)
            pa = 10 * np.log10(1 / mse_per_image(a, obj_va[:20]).mean())
            pb = 10 * np.log10(1 / mse_per_image(bb, obj_va[:20]).mean())
            conv.append({"regime": r, "shots": n, "lambda": lam, f"PSNR_{TV_ITERS}it_meanMSE": pa,
                         f"PSNR_{3 * TV_ITERS}it_meanMSE": pb, "max_abs_image_change": float(np.abs(a - bb).max()),
                         "trace_first_last_objective": [tr[0]["objective_mean"], tr[-1]["objective_mean"]]})
    pd.DataFrame(conv).to_csv(OUT / "tv_convergence.csv", index=False)
    checks["tv_convergence"] = conv

    # ================= 3. sanity sweeps (validation seed 2027; DGI-smoothing and Tikhonov tuned on train) ======
    sv_cache = {}
    pv_full = patterns_for(VAL_SEEDS[0])
    bv_clean_full = pv_full.astype(np.float64) @ obj_va.reshape(75, -1).T

    def svds(n):
        if n not in sv_cache:
            sv_cache[n] = (svd_patterns(p26[:n]), svd_patterns(pv_full[:n]))
        return sv_cache[n]

    def cheap_eval(n_ph, kappa, n, cond_idx):
        sp_rng_tr = np.random.default_rng(np.random.SeedSequence([TUNE_SEED, 5050, cond_idx]))
        sp_rng_va = np.random.default_rng(np.random.SeedSequence([VAL_SEEDS[0], 5050, cond_idx]))
        btr_n = add_bucket_noise(btr_clean[:n], n_ph, kappa, mu_ref, sp_rng_tr) if n_ph else btr_clean[:n]
        (u, s, vt), (uv, sv, vtv) = svds(n)
        utb = u.T @ btr_n
        lam = TIK_LAMBDAS[int(np.argmin([mse_per_image(np.clip(tikhonov_from_svd(u, s, vt, utb, l), 0, 1), obj_tr).mean()
                                         for l in TIK_LAMBDAS]))] if n_ph else 0.0
        dg = dgi_from_buckets(p26[:n], btr_n)
        sg = SIGMAS[int(np.argmin([mse_per_image(smooth(dg, x), obj_tr).mean() for x in SIGMAS]))]
        pv = pv_full[:n]
        bv_clean = bv_clean_full[:n]
        bv = add_bucket_noise(bv_clean, n_ph, kappa, mu_ref, sp_rng_va) if n_ph else bv_clean
        out = {}
        out["Tikhonov"] = evaluate_set(y_va, np.clip(tikhonov_from_svd(uv, sv, vtv, uv.T @ bv, lam), 0, 1)
                                       .reshape(75, 1, 64, 64).astype(np.float32))
        out["smoothed DGI"] = evaluate_set(y_va, smooth(dgi_from_buckets(pv, bv), sg))
        return {m: {k: out[m][k] for k in ("PSNR", "SSIM", "Pearson_r")} for m in out}, lam, sg

    sweep = []
    budgets = [None, 1e10, 1e8, 1e7, 1e6, 3.16e5, 1e5, 3.16e4, 1e4, 3.16e3, 1e3, 1e2]
    for ci, nph in enumerate(budgets):
        for n in (1000, 10000):
            res, lam, sg = cheap_eval(nph, 0.0, n, 100 + ci)
            for m, v in res.items():
                sweep.append({"sweep": "photon budget (kappa=0)", "photons_per_mean_bucket": nph or np.inf, "kappa": 0.0,
                              "shots": n, "method": m, "tikhonov_lambda": lam, "sigma": sg, **v})
    for ci, kap in enumerate([0.0, 0.5, 1.0, 2.0, 4.0]):
        for n in (1000, 10000):
            res, lam, sg = cheap_eval(regimes["MEDIUM"]["photons_per_mean_bucket"], kap, n, 200 + ci)
            for m, v in res.items():
                sweep.append({"sweep": "detector noise (MEDIUM photon budget)",
                              "photons_per_mean_bucket": regimes["MEDIUM"]["photons_per_mean_bucket"], "kappa": kap,
                              "shots": n, "method": m, "tikhonov_lambda": lam, "sigma": sg, **v})
    sweep = pd.DataFrame(sweep)
    del sv_cache
    sweep.to_csv(OUT / "sanity_sweeps.csv", index=False)
    log("sanity sweeps done")

    # ================= 4. summaries, paired comparisons, sanity verdicts ======================
    def summarize(df):
        out = []
        for (r, n, m), g in df.groupby(["regime", "shots", "method"]):
            rr = {"regime": r, "shots": n, "method": m, "n_pairs": len(g)}
            for k in KEYS:
                rr[f"{k}_mean"], rr[f"{k}_std"], rr[f"{k}_median"] = g[k].mean(), g[k].std(), g[k].median()
                for sd in VAL_SEEDS:
                    rr[f"{k}_seed{sd}"] = g[g.seed == sd][k].mean()
            out.append(rr)
        o = pd.DataFrame(out)
        o["ro"] = o.regime.map({r: i for i, r in enumerate(REGIME_ORDER)})
        o["mo"] = o.method.map({m: i for i, m in enumerate(METHODS)})
        return o.sort_values(["ro", "shots", "mo"]).drop(columns=["ro", "mo"])

    summ = summarize(per)
    summ.to_csv(OUT / "summary.csv", index=False)
    for k in ("PSNR", "SSIM", "Pearson_r", "MSE"):
        t = summ.pivot_table(index=["regime", "shots"], columns="method", values=f"{k}_mean")[METHODS]
        t = t.reindex(REGIME_ORDER, level=0)
        t.to_csv(OUT / f"table_{k}.csv")

    wide = per.pivot_table(index=["regime", "shots", "seed", "image_id"], columns="method", values=["PSNR", "SSIM", "Pearson_r"])
    prow = []
    for a, b in (("TV", "Tikhonov"), ("TV", "smoothed DGI"), ("Tikhonov", "smoothed DGI"), ("Tikhonov", "pseudoinverse"),
                 ("smoothed DGI", "raw DGI")):
        for (r, n), g in wide.groupby(level=[0, 1]):
            row = {"A": a, "B": b, "regime": r, "shots": n}
            for k in ("PSNR", "SSIM", "Pearson_r"):
                d = g[(k, a)] - g[(k, b)]
                row[f"{k}_mean_diff"], row[f"{k}_median_diff"], row[f"{k}_pct_A_better"] = d.mean(), d.median(), (d > 0).mean() * 100
            prow.append(row)
    paired = pd.DataFrame(prow)
    paired.to_csv(OUT / "paired_comparisons.csv", index=False)

    best = summ[summ.method.isin(["smoothed DGI", "Tikhonov", "TV"])].groupby(["regime", "shots"]).agg(
        best_PSNR=("PSNR_mean", "max"), best_SSIM=("SSIM_mean", "max"), best_r=("Pearson_r_mean", "max")).reset_index()
    best.to_csv(OUT / "best_conventional.csv", index=False)
    verdict = {}
    for r in REGIME_ORDER:
        g = best[best.regime == r].set_index("shots")
        gain = float(g.best_PSNR[10000] - max(g.best_PSNR[500], g.best_PSNR[1000]))
        crit = {"gain_10000_vs_500_1000_db": gain, "best_PSNR_10000": float(g.best_PSNR[10000]),
                "best_SSIM_10000": float(g.best_SSIM[10000])}
        crit["meets_gain"] = gain >= USEFUL["min_gain_db_10000_over_500_and_1000"]
        crit["not_trivial"] = crit["best_PSNR_10000"] < USEFUL["max_best_psnr_10000_db"]
        crit["not_impossible"] = crit["best_SSIM_10000"] >= USEFUL["min_best_ssim_10000"]
        crit["useful"] = crit["meets_gain"] and crit["not_trivial"] and crit["not_impossible"]
        verdict[r] = crit

    def mono(vals, inc=True):
        v = np.array(vals, float)
        return bool(np.all(np.diff(v) > 0) if inc else np.all(np.diff(v) < 0))
    sanity = {}
    sanity["A_more_photons_better_regimes"] = {
        f"{m}-{n}": mono([summ[(summ.regime == r) & (summ.shots == n) & (summ.method == m)].PSNR_mean.iloc[0]
                          for r in ["HIGH", "MEDIUM", "LOW"]])
        for m in METHODS for n in SHOTS}
    sanity["B_more_shots_better"] = {
        f"{m}-{r}": mono(summ[(summ.regime == r) & (summ.method == m)].sort_values("shots").PSNR_mean.tolist())
        for m in METHODS for r in REGIME_ORDER}
    det = sweep[sweep.sweep.str.startswith("detector")]
    sanity["C_more_detector_noise_worse"] = {f"{m}-{n}": mono(det[(det.method == m) & (det.shots == n)].sort_values("kappa").PSNR.tolist(), inc=False)
                                            for m in ("Tikhonov", "smoothed DGI") for n in (1000, 10000)}
    ph = sweep[sweep.sweep.str.startswith("photon")]
    sanity["D_high_budget_approaches_noise_free"] = {
        f"{m}-{n}": {"noise_free_PSNR": float(ph[(ph.method == m) & (ph.shots == n) & np.isinf(ph.photons_per_mean_bucket)].PSNR.iloc[0]),
                     "1e10_PSNR": float(ph[(ph.method == m) & (ph.shots == n) & (ph.photons_per_mean_bucket == 1e10)].PSNR.iloc[0]),
                     "1e8_PSNR": float(ph[(ph.method == m) & (ph.shots == n) & (ph.photons_per_mean_bucket == 1e8)].PSNR.iloc[0])}
        for m in ("Tikhonov", "smoothed DGI") for n in (1000, 10000)}
    sanity["E_low_budget_hard"] = {
        f"{m}-{n}": {"1e2_PSNR": float(ph[(ph.method == m) & (ph.shots == n) & (ph.photons_per_mean_bucket == 1e2)].PSNR.iloc[0]),
                     "1e2_SSIM": float(ph[(ph.method == m) & (ph.shots == n) & (ph.photons_per_mean_bucket == 1e2)].SSIM.iloc[0])}
        for m in ("Tikhonov", "smoothed DGI") for n in (1000, 10000)}
    sanity["photon_sweep_monotone_finite_budgets"] = {
        f"{m}-{n}": mono(ph[(ph.method == m) & (ph.shots == n) & np.isfinite(ph.photons_per_mean_bucket)]
                         .sort_values("photons_per_mean_bucket").PSNR.tolist())
        for m in ("Tikhonov", "smoothed DGI") for n in (1000, 10000)}

    # ================= 5. figures ===============================================================
    base = "75 val images x seeds 2027-2029 (225 pairs per point); parameters tuned on training images only"
    for k, f, lab, ly in (("PSNR", "psnr_vs_shots.png", "PSNR (dB)", False), ("SSIM", "ssim_vs_shots.png", "SSIM", False),
                          ("Pearson_r", "correlation_vs_shots.png", "Pearson r", False), ("MSE", "mse_vs_shots.png", "MSE (log)", True)):
        plot_metric(summ, k, lab, OUT / f, f"Noisy buckets: {lab} vs shots, per noise regime\n{base}", log_y=ly)
    ex_ids = json.loads((ROOT / "outputs" / "phase2b_unet" / "model_A_dgi500" / "full" / "training_report.json")
                        .read_text())["val_example_ids"]
    fig_ids = [ex_ids[0], ex_ids[2], ex_ids[4]]
    pv = per[per.seed == VAL_SEEDS[0]].set_index(["regime", "shots", "method", "image_id"])
    for r in REGIME_ORDER:
        for n in [x for x in FIG_SHOTS if x in SHOTS]:
            rws = []
            for image_id in fig_ids:
                j = id_va.index(image_id)
                panels = [("Ground truth", y_va[j, 0])]
                for m in METHODS:
                    q = pv.loc[(r, n, m, image_id)]
                    panels.append((f"{m}\n{q.PSNR:.2f} dB | SSIM {q.SSIM:.3f}", keep[(r, n, m)][j, 0]))
                rws.append({"label": f"{image_id} (val)", "target": y_va[j, 0], "panels": panels})
            save_panel_grid(rws, OUT / "reconstructions" / f"{r}_N{n}.png",
                            f"{r} noise (photons/bucket {regimes[r]['photons_per_mean_bucket']:.3g}, kappa {regimes[r]['kappa']}), "
                            f"N = {n:,}, seed {VAL_SEEDS[0]}; gray window [0, 1] for every panel")

    report = {"noise_config": cfg, "selected_parameters": selected.to_dict("records"), "checks": checks,
              "summary": summ.to_dict("records"), "best_conventional": best.to_dict("records"),
              "useful_regime_verdict": verdict, "sanity": sanity, "test_images_used": 0,
              "runtime_s": round(time.time() - t0, 1)}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=float))
    run_cfg = {"shots": SHOTS, "tune_seed": TUNE_SEED, "val_seeds": VAL_SEEDS, "sigma_grid": SIGMAS,
               "tikhonov_lambda_grid": [0.0] + TIK_LAMBDAS, "tv_lambda_grid": TV_LAMBDAS, "tv_iterations": TV_ITERS,
               "tv_rho_rule": "rho = 10 * lambda", "tv_init": "Tikhonov solution at the frozen lambda",
               "tv_tuning_images": f"{N_TV_TUNE} training images (rng 0)", "selection_metric": "lowest mean training MSE",
               "tikhonov": "argmin ||P O - B||^2 + lambda ||O||^2 via SVD; clipped to [0,1]",
               "tv": "argmin 0.5||A O - b||^2 + lambda TV_iso(O), O in [0,1], A = 2P/sqrt(N), b = 2B/sqrt(N); ADMM",
               "useful_regime_criteria": USEFUL}
    (OUT / "experiment_config.json").write_text(json.dumps(run_cfg, indent=2))

    pd.set_option("display.width", 220)
    for k in ("PSNR", "SSIM", "Pearson_r"):
        log(f"\n{k} (mean):\n" + pd.read_csv(OUT / f"table_{k}.csv").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    log("\nselected parameters:\n" + selected.to_string(index=False))
    log("\nbucket statistics:\n" + pd.DataFrame(stat_rows).groupby(["split", "regime"]).mean(numeric_only=True).round(3).to_string())
    log("\nuseful regime verdict: " + json.dumps(verdict, indent=1))
    log("\nsanity: " + json.dumps(sanity, indent=1, default=float))
    log("\nchecks: " + json.dumps({k: v for k, v in checks.items() if k != "noise_config_used"}, indent=1, default=float))
    log(f"\n{report['runtime_s']} s -> {OUT}")


if __name__ == "__main__":
    main()
