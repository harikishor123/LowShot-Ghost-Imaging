#!/usr/bin/env python3
"""Conventional measurement-equivalence curve (validation only, no training, no test data).

    python run_measurement_curve.py

For N in {500, 1000, 2000, 3000, 5000, 10000}:
  1. conventional DGI-N (Phase 2A pipeline: calibrated, clipped to [0, 1])
  2. Gaussian sigma tuned on the 341 TRAINING images (seed-2026 DGI, lowest mean MSE)
  3. sigma frozen and applied to the 75 VALIDATION images measured with unseen seeds 2027-2029
Frozen Model D (DGI-500) and Model C (DGI-1000) are evaluated on the same validation inputs.
Equivalent conventional shot count = where the smoothed-DGI mean curve reaches the model's mean,
linearly interpolated in log(N) between tested N (an ESTIMATE); 95% interval by bootstrap over images.
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

from src.ghost_imaging import load_object, pattern_batches, pattern_variance  # noqa: E402
from src.gi_unet_data import load_pairs, read_split_ids  # noqa: E402
from src.reconstruction import StreamingDGI  # noqa: E402
from src.unet_gi import UNetGI  # noqa: E402
from src.visualization import GRID, INK, INK_MUTED, save_panel_grid  # noqa: E402
from run_generalization_validation import smooth  # noqa: E402
from train_unet_gi import evaluate_set, predict  # noqa: E402

OUT = ROOT / "outputs" / "phase2b_unet" / "conventional_measurement_curve"
HIST = ROOT / "outputs" / "phase2b_unet"
SHOTS = [500, 1000, 2000, 3000, 5000, 10000]
UNSEEN = [2027, 2028, 2029]
TUNE_SEED = 2026
SIGMAS = [0.0, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875,
          2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 5.0]
KEYS = ("PSNR", "SSIM", "MSE", "Pearson_r")
EQ_KEYS = ("PSNR", "SSIM", "Pearson_r")
N_BOOT = 2000


def dgi_curve(objects: np.ndarray, seed: int) -> dict[int, np.ndarray]:
    stream, out = StreamingDGI(objects), {}
    for pats in pattern_batches(max(SHOTS), objects.shape[1:], seed, 500, 0.5):
        stream.update(pats)
        if stream.n in SHOTS:
            est, _ = stream.estimate(pattern_variance(0.5))
            out[stream.n] = np.clip(est, 0, 1).reshape(-1, 1, *objects.shape[1:]).astype(np.float32)
    return out


def load_model(sub: str):
    ck = torch.load(HIST / sub / "checkpoints" / "best.pt", weights_only=False)
    m = UNetGI(base=ck["config"]["base"])
    m.load_state_dict(ck["model"])
    return m, int(ck["epoch"])


def crossing(curve: np.ndarray, target: float) -> tuple[float | None, str]:
    """Equivalent N where the (monotone increasing) mean curve reaches target; log-N interpolation."""
    logn = np.log(SHOTS)
    if target <= curve[0]:
        return float(SHOTS[0]), f"<= {SHOTS[0]} (at or below the smallest tested N)"
    if target > curve[-1]:
        return None, f"> {SHOTS[-1]} (not reached within tested range)"
    for k in range(len(SHOTS) - 1):
        lo, hi = curve[k], curve[k + 1]
        if lo <= target <= hi and hi > lo:
            t = (target - lo) / (hi - lo)
            return float(np.exp(logn[k] + t * (logn[k + 1] - logn[k]))), \
                f"interpolated between {SHOTS[k]} and {SHOTS[k + 1]} (linear in log N)"
    return None, "curve not monotone at this level"


def plot_metric(curve_s, curve_raw, ai, eq, key, label, path, title):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(SHOTS, curve_raw, color=INK_MUTED, lw=1.4, ls="--", marker="o", ms=5, label="raw DGI (no post-processing)")
    ax.plot(SHOTS, curve_s, color="#2a78d6", lw=2.2, marker="o", ms=8, mec="white", mew=2,
            label="smoothed DGI (sigma tuned on train)")
    for (name, shots, color), val in zip((("Model D", 500, "#eb6834"), ("Model C", 1000, "#1baf7a")), ai):
        ax.axhline(val, color=color, lw=1.6, label=f"{name} ({shots} shots) = {val:.3f}")
        ax.plot([shots], [val], marker="*", ms=15, color=color, mec="white", mew=1, zorder=5)
        n_eq = eq[name]
        if n_eq is not None:
            ax.axvline(n_eq, color=color, lw=1, ls=":")
            ax.annotate(f"≈{n_eq:,.0f} shots (est.)", (n_eq, val), xytext=(6, -14 if name == "Model D" else 6),
                        textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(SHOTS, [f"{n:,}" for n in SHOTS])
    ax.minorticks_off()
    ax.set_xlabel("number of measurements (log scale)", color=INK_MUTED)
    ax.set_title(label, fontsize=10.5, color=INK, loc="left")
    ax.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK_MUTED)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    t0 = time.time()
    checks = {}
    ids = read_split_ids()
    ref_dir = ROOT / "data" / "v2" / "reference"

    # ---- 1. sigma tuning on TRAINING images (seed 2026) -------------------------------
    _, y_tr, id_tr = load_pairs("train", 1000)
    assert len(id_tr) == 341 and not set(id_tr) & set(ids["test"])
    obj_tr = np.stack([load_object(ref_dir / f"{i}.npz", 64)[0] for i in id_tr])
    x_tr = dgi_curve(obj_tr, TUNE_SEED)
    checks["train_dgi_vs_phase2A_saved_max_abs_diff"] = float(max(
        np.abs(x_tr[n] - load_pairs("train", n)[0]).max() for n in (500, 1000, 5000, 10000)))
    tune_rows, sigma = [], {}
    for n in SHOTS:
        for s in SIGMAS:
            e = evaluate_set(y_tr, smooth(x_tr[n], s))
            tune_rows.append({"shots": n, "sigma": s, **{f"train_{k}": e[k] for k in KEYS}})
        t = pd.DataFrame([r for r in tune_rows if r["shots"] == n])
        sigma[n] = float(t.loc[t.train_MSE.idxmin(), "sigma"])
    tune = pd.DataFrame(tune_rows)
    tune.to_csv(OUT / "training_sigma_tuning.csv", index=False)
    checks["sigma_frozen"] = sigma
    checks["sigma_at_grid_edge"] = {n: s in (SIGMAS[0], SIGMAS[-1]) for n, s in sigma.items()}
    prev = {500: 1.75, 1000: 1.5, 10000: 0.75}
    checks["sigma_vs_previous_coarser_grid"] = {n: {"previous": prev[n], "now": sigma[n]} for n in prev}
    print("frozen sigma (train MSE):", sigma, flush=True)

    # ---- 2. validation images, unseen seeds ---------------------------------------------
    _, y_va, id_va = load_pairs("val", 1000)
    assert len(id_va) == 75 and not set(id_va) & set(ids["test"])
    obj_va = np.stack([load_object(ref_dir / f"{i}.npz", 64)[0] for i in id_va])
    x_va = {s: dgi_curve(obj_va, s) for s in UNSEEN}

    model_d, ep_d = load_model("model_D_random_patterns_dgi500")
    model_c, ep_c = load_model("model_C_random_patterns")
    checks["model_epochs"] = {"D": ep_d, "C": ep_c}

    per_rows = []
    keep = {}
    for s in UNSEEN:
        cols = {}
        for n in SHOTS:
            raw = x_va[s][n]
            sm = smooth(raw, sigma[n])
            for tag, img in ((f"raw{n}", raw), (f"smooth{n}", sm)):
                cols[tag] = evaluate_set(y_va, img)["per_image"]
                keep[(s, tag)] = img
        for tag, model, n in (("ModelD", model_d, 500), ("ModelC", model_c, 1000)):
            pred = predict(model, x_va[s][n])
            cols[tag] = evaluate_set(y_va, pred)["per_image"]
            keep[(s, tag)] = np.clip(pred, 0, 1)
        for j, image_id in enumerate(id_va):
            row = {"image_id": image_id, "seed": s}
            for tag, df in cols.items():
                for k in KEYS:
                    row[f"{tag}_{k}"] = float(df[k].iloc[j])
            per_rows.append(row)
    per = pd.DataFrame(per_rows)                         # 225 rows
    per.to_csv(OUT / "validation_per_image_seed.csv", index=False)

    # consistency with Model C / D reports (same frozen models, same unseen inputs)
    for tag, sub, n in (("ModelD", "model_D_random_patterns_dgi500", 500), ("ModelC", "model_C_random_patterns", 1000)):
        rep = json.loads((HIST / sub / "training_report.json").read_text())
        if tag == "ModelD":
            r = next(x for x in rep["unseen_table"] if x["method"].startswith("Model D"))
            want = {k: r[f"{k}_mean"] for k in KEYS}
        else:
            r = next(x for x in rep["unseen_summary"] if x["method"].startswith("Model C"))
            want = {k: r[f"{k}_mean"] for k in KEYS}
        got = per.groupby("seed")[[f"{tag}_{k}" for k in KEYS]].mean().mean()
        checks[f"{tag}_reproduces_report_max_abs_diff"] = float(max(abs(got[f"{tag}_{k}"] - want[k]) for k in KEYS))

    def summary(prefix):
        rows = []
        for n in SHOTS:
            r = {"shots": n, "sigma": sigma[n] if prefix == "smooth" else 0.0}
            for k in KEYS:
                c = per[f"{prefix}{n}_{k}"]
                r[f"{k}_mean"] = c.mean()
                r[f"{k}_std"] = c.std()
                r[f"{k}_median"] = c.median()
                r[f"{k}_std_of_seed_means"] = per.groupby("seed")[f"{prefix}{n}_{k}"].mean().std()
            rows.append(r)
        return pd.DataFrame(rows)

    curve_s, curve_r = summary("smooth"), summary("raw")
    curve_s.to_csv(OUT / "validation_curve.csv", index=False)
    curve_r.to_csv(OUT / "raw_dgi_curve.csv", index=False)
    ai = {tag: {k: per[f"{tag}_{k}"].mean() for k in KEYS} | {f"{k}_std": per[f"{tag}_{k}"].std() for k in KEYS}
          | {f"{k}_median": per[f"{tag}_{k}"].median() for k in KEYS} for tag in ("ModelD", "ModelC")}
    checks["smoothed_curve_monotone_increasing"] = {k: bool(np.all(np.diff(curve_s[f"{k}_mean"]) > 0)) for k in EQ_KEYS}

    # ---- 3. equivalence + bootstrap over images ------------------------------------------
    rng = np.random.default_rng(0)
    img_index = {i: np.where(per.image_id.values == i)[0] for i in id_va}
    boot = {(tag, k): [] for tag in ("ModelD", "ModelC") for k in EQ_KEYS}
    for _ in range(N_BOOT):
        pick = rng.choice(id_va, size=len(id_va), replace=True)
        rows = np.concatenate([img_index[i] for i in pick])
        sub = per.iloc[rows]
        for k in EQ_KEYS:
            cv = np.array([sub[f"smooth{n}_{k}"].mean() for n in SHOTS])
            for tag in ("ModelD", "ModelC"):
                ne, _ = crossing(cv, sub[f"{tag}_{k}"].mean())
                boot[(tag, k)].append(np.nan if ne is None else ne)
    eq_rows, eq_for_plot = [], {k: {} for k in EQ_KEYS}
    for tag, name, n_ai in (("ModelD", "Model D", 500), ("ModelC", "Model C", 1000)):
        for k in EQ_KEYS:
            cv = curve_s[f"{k}_mean"].values
            ne, how = crossing(cv, ai[tag][k])
            b = np.array(boot[(tag, k)])
            frac_nan = float(np.isnan(b).mean())
            lo, hi = (np.nanpercentile(b, [2.5, 97.5]) if frac_nan < 1 else (np.nan, np.nan))
            eq_rows.append({"model": name, "ai_shots": n_ai, "metric": k, "ai_value": ai[tag][k],
                            "equivalent_shots_estimate": ne, "how": how,
                            "reduction_factor": (ne / n_ai) if ne is not None else np.nan,
                            "boot95_equivalent_low": lo, "boot95_equivalent_high": hi,
                            "boot95_reduction_low": lo / n_ai, "boot95_reduction_high": hi / n_ai,
                            "boot_fraction_not_reached": frac_nan})
            eq_for_plot[k][name] = ne
    eq = pd.DataFrame(eq_rows)
    eq.to_csv(OUT / "measurement_equivalence.csv", index=False)
    conservative = {}
    for name in ("Model D", "Model C"):
        g = eq[eq.model == name]
        worst = g.loc[g.reduction_factor.idxmin()]
        conservative[name] = {"weakest_metric": worst.metric, "equivalent_shots": worst.equivalent_shots_estimate,
                              "reduction_factor": worst.reduction_factor,
                              "reduction_factor_boot95_low": float(g.boot95_reduction_low.min())}

    # ---- 4. figures ---------------------------------------------------------------------
    base = "75 val images x unseen seeds 2027-2029 (225 pairs); sigma tuned on 341 training images"
    for k, fname, label in (("PSNR", "psnr_vs_shots.png", "Mean PSNR (dB)"),
                            ("SSIM", "ssim_vs_shots.png", "Mean SSIM"),
                            ("Pearson_r", "correlation_vs_shots.png", "Mean Pearson r")):
        plot_metric(curve_s[f"{k}_mean"].values, curve_r[f"{k}_mean"].values,
                    (ai["ModelD"][k], ai["ModelC"][k]), eq_for_plot[k], k, label, OUT / fname,
                    f"Conventional DGI vs AI: {label}\n{base}")

    ex_ids = json.loads((HIST / "model_A_dgi500" / "full" / "training_report.json").read_text())["val_example_ids"]
    s = UNSEEN[0]
    pr = per[per.seed == s].set_index("image_id")
    fig_rows = []
    for image_id in ex_ids:
        j = id_va.index(image_id)
        r = pr.loc[image_id]

        def ttl(t, tg):
            return f"{t}\n{r[f'{tg}_PSNR']:.2f} dB | SSIM {r[f'{tg}_SSIM']:.3f}"
        panels = [("Original (target)", y_va[j, 0]),
                  (ttl(f"Smoothed DGI-500", "smooth500"), np.clip(keep[(s, "smooth500")][j, 0], 0, 1)),
                  (ttl("Model D (500)", "ModelD"), keep[(s, "ModelD")][j, 0]),
                  (ttl("Smoothed DGI-1000", "smooth1000"), np.clip(keep[(s, "smooth1000")][j, 0], 0, 1)),
                  (ttl("Model C (1000)", "ModelC"), keep[(s, "ModelC")][j, 0])]
        for n in (2000, 3000, 5000, 10000):
            panels.append((ttl(f"Smoothed DGI-{n:,}", f"smooth{n}"), np.clip(keep[(s, f"smooth{n}")][j, 0], 0, 1)))
        fig_rows.append({"label": f"{image_id} (val)\nseed {s} (unseen)", "target": y_va[j, 0], "panels": panels})
    sig_txt = ", ".join(f"{n}:{sigma[n]}" for n in SHOTS)
    head = f"Unseen seed {s}; smoothed DGI uses sigma tuned on training images only (N:sigma = {sig_txt})\n"
    save_panel_grid(fig_rows, OUT / "reconstruction_examples.png",
                    head + "VALIDATION images (same 5 as before), gray window [0, 1] for every panel")
    save_panel_grid(fig_rows, OUT / "reconstruction_examples_DISPLAY_ONLY.png",
                    head + "DISPLAY ONLY: one window per row (target's 1-99th pct); metrics from [0, 1] images",
                    display_window=True)

    report = {"shots": SHOTS, "unseen_seeds": UNSEEN, "sigma_tuning": "lowest mean MSE on 341 training images, "
              "seed-2026 DGI; sigma grid " + str(SIGMAS), "sigma_frozen": sigma,
              "interpolation": "linear in log(N) between tested shot counts (estimate)",
              "bootstrap": f"{N_BOOT} resamples of the 75 validation images (all 3 seeds kept per image)",
              "test_images_used": 0, "checks": checks,
              "ai": ai, "smoothed_curve": curve_s.to_dict("records"), "raw_curve": curve_r.to_dict("records"),
              "equivalence": eq.to_dict("records"), "conservative": conservative,
              "runtime_s": round(time.time() - t0, 1)}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=float))

    pd.set_option("display.width", 220)
    cols = ["shots", "sigma"] + [f"{k}_{s_}" for k in ("PSNR", "SSIM", "MSE", "Pearson_r") for s_ in ("mean", "std", "median")]
    print("\nsmoothed DGI (val, unseen seeds):\n" + curve_s[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nraw DGI:\n" + curve_r[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nAI:\n" + pd.DataFrame(ai).T.round(4).to_string())
    print("\nequivalence:\n" + eq.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nconservative:", json.dumps(conservative, indent=1, default=float))
    print("\nchecks:", json.dumps(checks, indent=1, default=float))
    print(f"\n{report['runtime_s']} s -> {OUT}")


if __name__ == "__main__":
    main()
