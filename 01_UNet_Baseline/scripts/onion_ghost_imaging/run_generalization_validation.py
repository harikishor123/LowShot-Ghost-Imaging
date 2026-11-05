#!/usr/bin/env python3
"""Validation-only checks of the trained Model A / Model B (no retraining, no test data).

    python run_generalization_validation.py

Part 1 - non-learned smoothing baseline
    Gaussian blur of the clipped DGI-N image (the exact image the U-Net receives).
    sigma is chosen per N on the 341 TRAINING images (lowest mean MSE, the U-Net's own
    training objective), then frozen and applied to the 75 validation images.

Part 2 - unseen illumination-pattern sequences
    The 75 validation objects are re-measured with seeds 2027/2028/2029 (same pattern
    model: 10,000 i.i.d. Bernoulli(0.5) 64x64 patterns, prefixes 500/1,000/10,000) using
    the Phase 2A streaming DGI code. Model A gets DGI-500, Model B gets DGI-1000; the
    checkpoints are loaded read-only. Seed 2026 is recomputed the same way as a control.

Outputs: outputs/phase2b_unet/generalization_validation/
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.ghost_imaging import load_object, pattern_batches, pattern_variance  # noqa: E402
from src.gi_unet_data import load_pairs, read_split_ids  # noqa: E402
from src.reconstruction import StreamingDGI  # noqa: E402
from src.unet_gi import UNetGI  # noqa: E402
from src.visualization import save_panel_grid  # noqa: E402
from train_unet_gi import evaluate_set, predict  # noqa: E402

OUT = ROOT / "outputs" / "phase2b_unet" / "generalization_validation"
RUNS = {"A": (500, ROOT / "outputs" / "phase2b_unet" / "model_A_dgi500" / "full"),
        "B": (1000, ROOT / "outputs" / "phase2b_unet" / "model_B_dgi1000" / "full")}
SIGMAS = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 5.0]
SEEDS_UNSEEN = [2027, 2028, 2029]
OFFICIAL_SEED = 2026
SHOTS = [500, 1000, 10000]
KEYS = ("PSNR", "SSIM", "MSE", "Pearson_r")


def smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of each (1, H, W) image; reflect boundary. sigma=0 returns the input."""
    if sigma == 0:
        return x.copy()
    return np.stack([gaussian_filter(im[0], sigma, mode="reflect")[None] for im in x]).astype(np.float32)


def load_model(run_dir: Path):
    ck = torch.load(run_dir / "checkpoints" / "best.pt", weights_only=False)
    m = UNetGI(base=ck["config"]["base"])
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


def dgi_for_seed(objects: np.ndarray, seed: int) -> dict[int, np.ndarray]:
    """Calibrated, clipped DGI estimates (M,1,64,64) for N in SHOTS, one seeded sequence."""
    stream, out = StreamingDGI(objects), {}
    s2 = pattern_variance(0.5)
    for pats in pattern_batches(max(SHOTS), objects.shape[1:], seed, 500, 0.5):
        stream.update(pats)
        if stream.n in SHOTS:
            est, _ = stream.estimate(s2)
            out[stream.n] = np.clip(est, 0, 1).reshape(-1, 1, *objects.shape[1:]).astype(np.float32)
    return out


def summarize(name, shots, seed, ev):
    return {"method": name, "shots": shots, "seed": seed, **{k: ev[k] for k in KEYS},
            **{f"{k}_median": float(ev["per_image"][k].median()) for k in ("PSNR", "SSIM", "Pearson_r")}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    t0 = time.time()
    checks = {}
    ids = read_split_ids()
    models = {k: load_model(d) for k, (_, d) in RUNS.items()}
    checks["best_epochs"] = {k: int(v[1]["epoch"]) for k, v in models.items()}

    # ======================= Part 1: smoothing baseline ===========================
    sigma_star, tuning_rows = {}, []
    for name, (n, _) in RUNS.items():
        x_tr, y_tr, id_tr = load_pairs("train", n)
        assert len(id_tr) == 341 and not set(id_tr) & set(ids["test"])
        for s in SIGMAS:
            e = evaluate_set(y_tr, smooth(x_tr, s))
            tuning_rows.append({"shots": n, "sigma": s, "split": "train",
                                **{f"train_{k}": e[k] for k in KEYS}})
        t = pd.DataFrame([r for r in tuning_rows if r["shots"] == n])
        sigma_star[n] = float(t.loc[t.train_MSE.idxmin(), "sigma"])
        checks[f"sigma_selected_DGI{n}_by_train_MSE"] = sigma_star[n]
        checks[f"sigma_that_would_max_train_SSIM_DGI{n}_info_only"] = float(t.loc[t.train_SSIM.idxmax(), "sigma"])
        checks[f"sigma_DGI{n}_at_grid_edge"] = sigma_star[n] in (SIGMAS[0], SIGMAS[-1])
    tuning = pd.DataFrame(tuning_rows)
    tuning.to_csv(OUT / "smoothing_tuning_results.csv", index=False)
    print("frozen sigma (train-selected):", sigma_star)

    # ======================= validation objects + all seeds ==========================
    x_saved, y_va, id_va = {}, None, None
    for n in SHOTS:
        x, y, i = load_pairs("val", n)
        if y_va is None:
            y_va, id_va = y, i
        assert i == id_va and np.array_equal(y, y_va)
        x_saved[n] = x
    assert len(id_va) == 75 and not set(id_va) & set(ids["test"])
    objects = np.stack([load_object(ROOT / "data" / "v2" / "reference" / f"{i}.npz", 64)[0] for i in id_va])
    checks["val_objects_match_saved_targets_max_abs_diff"] = float(np.abs(objects - y_va[:, 0]).max())

    inputs = {}
    for seed in [OFFICIAL_SEED] + SEEDS_UNSEEN:
        inputs[seed] = dgi_for_seed(objects, seed)
    checks["seed2026_recomputed_vs_phase2A_saved_max_abs_diff"] = {
        n: float(np.abs(inputs[OFFICIAL_SEED][n] - x_saved[n]).max()) for n in SHOTS}

    # cross-check unseen-seed DGI against the Phase 2A robustness run (validation rows only)
    rob = pd.read_csv(ROOT / "outputs" / "phase2a_dgi_dataset" / "robustness" / "robustness_per_image_all_seeds.csv")
    rob = rob[(rob.split == "val") & rob.seed.isin(SEEDS_UNSEEN)].set_index(["seed", "measurements", "image_id"])
    diffs = []
    for seed in SEEDS_UNSEEN:
        for n in SHOTS:
            if n not in rob.index.get_level_values("measurements"):
                continue
            e = evaluate_set(y_va, inputs[seed][n])["per_image"].PSNR.values
            diffs.append(np.abs(e - rob.loc[(seed, n)].loc[id_va, "psnr"].values).max())
    checks["unseen_seed_DGI_vs_phase2A_robustness_max_psnr_diff_db"] = float(max(diffs))

    # ======================= evaluate every method x seed ============================
    summary_rows, per_seed_frames, preds_keep = [], {}, {}
    for seed in [OFFICIAL_SEED] + SEEDS_UNSEEN:
        per = pd.DataFrame({"image_id": id_va, "seed": seed})
        ev_ref = evaluate_set(y_va, inputs[seed][10000])
        summary_rows.append(summarize("DGI", 10000, seed, ev_ref))
        for k in KEYS:
            per[f"DGI10000_{k}"] = ev_ref["per_image"][k].values
        for name, (n, _) in RUNS.items():
            x = inputs[seed][n]
            pred = predict(models[name][0], x)
            preds_keep[(seed, name)] = pred
            for label, tag, img in ((f"DGI", f"DGI{n}", x),
                                    (f"Smoothed DGI (sigma={sigma_star[n]})", f"Smooth{n}", smooth(x, sigma_star[n])),
                                    (f"U-Net {name}", f"UNet{name}", pred)):
                ev = evaluate_set(y_va, img)
                summary_rows.append(summarize(label, n, seed, ev))
                for k in KEYS:
                    per[f"{tag}_{k}"] = ev["per_image"][k].values
        per_seed_frames[seed] = per
        if seed in SEEDS_UNSEEN:
            per.to_csv(OUT / f"unseen_seed_{seed}.csv", index=False)
    per_seed_frames[OFFICIAL_SEED].to_csv(OUT / "seed_2026_control_recomputed.csv", index=False)

    summ = pd.DataFrame(summary_rows)
    # control: seed-2026 U-Net metrics must equal the recorded training reports
    for name, (n, d) in RUNS.items():
        rec = next(r for r in json.loads((d / "training_report.json").read_text())["val_summary"]
                   if r["method"].startswith("U-Net"))
        got = summ[(summ.method == f"U-Net {name}") & (summ.seed == OFFICIAL_SEED)].iloc[0]
        checks[f"model_{name}_seed2026_reproduces_recorded_max_abs_diff"] = float(max(abs(got[k] - rec[k]) for k in KEYS))
    avg = (summ[summ.seed.isin(SEEDS_UNSEEN)].groupby(["method", "shots"], sort=False)
           [list(KEYS) + ["PSNR_median", "SSIM_median", "Pearson_r_median"]].agg(["mean", "std"]))
    avg.columns = [f"{a}_{b}" for a, b in avg.columns]
    avg = avg.reset_index()
    avg.insert(2, "seed", "2027-2029 mean")
    summ_out = pd.concat([summ.assign(seed=summ.seed.astype(str)),
                          avg.rename(columns={f"{k}_mean": k for k in list(KEYS) + ["PSNR_median", "SSIM_median", "Pearson_r_median"]})],
                         ignore_index=True)
    summ_out.to_csv(OUT / "unseen_seed_summary.csv", index=False)

    # ======================= Part 1 tables: smoothing vs U-Net =====================
    smooth_rows, vs_rows = [], []
    for seed_label, seeds in (("2026 (official)", [OFFICIAL_SEED]), ("2027-2029 (unseen)", SEEDS_UNSEEN)):
        cat = pd.concat([per_seed_frames[s] for s in seeds], ignore_index=True)
        for name, (n, _) in RUNS.items():
            for tag, label in ((f"DGI{n}", f"DGI-{n}"), (f"Smooth{n}", f"Smoothed DGI-{n} (sigma={sigma_star[n]})"),
                               (f"UNet{name}", f"U-Net {name}")):
                smooth_rows.append({"seeds": seed_label, "shots": n, "method": label,
                                    **{k: float(cat[f"{tag}_{k}"].mean()) for k in KEYS}})
            row = {"seeds": seed_label, "comparison": f"U-Net {name} vs smoothed DGI-{n}", "n_image_seed_pairs": len(cat)}
            for k in ("PSNR", "SSIM", "Pearson_r"):
                d = cat[f"UNet{name}_{k}"] - cat[f"Smooth{n}_{k}"]
                row[f"{k}_mean_gain"] = float(d.mean())
                row[f"{k}_median_gain"] = float(d.median())
                row[f"{k}_pct_unet_better"] = float((d > 0).mean() * 100)
            d_in = cat[f"Smooth{n}_PSNR"] - cat[f"DGI{n}_PSNR"]
            d_un = cat[f"UNet{name}_PSNR"] - cat[f"DGI{n}_PSNR"]
            row["PSNR_gain_over_DGI_smoothing"] = float(d_in.mean())
            row["PSNR_gain_over_DGI_unet"] = float(d_un.mean())
            row["share_of_unet_PSNR_gain_explained_by_smoothing"] = float(d_in.mean() / d_un.mean())
            for k in ("SSIM", "Pearson_r"):
                gs = (cat[f"Smooth{n}_{k}"] - cat[f"DGI{n}_{k}"]).mean()
                gu = (cat[f"UNet{name}_{k}"] - cat[f"DGI{n}_{k}"]).mean()
                row[f"share_of_unet_{k}_gain_explained_by_smoothing"] = float(gs / gu)
            for k in ("PSNR", "SSIM", "Pearson_r"):
                row[f"{k}_pct_unet_beats_DGI10000"] = float((cat[f"UNet{name}_{k}"] > cat[f"DGI10000_{k}"]).mean() * 100)
                row[f"{k}_pct_smoothing_beats_DGI10000"] = float((cat[f"Smooth{n}_{k}"] > cat[f"DGI10000_{k}"]).mean() * 100)
            vs_rows.append(row)
    pd.DataFrame(smooth_rows).to_csv(OUT / "smoothing_baseline_results.csv", index=False)
    vs = pd.DataFrame(vs_rows)
    vs.to_csv(OUT / "smoothing_vs_unet.csv", index=False)

    # seed-2026 vs unseen: change in mean metric per method
    deltas = {}
    for method, n in (("U-Net A", 500), ("U-Net B", 1000), ("DGI", 500), ("DGI", 1000), ("DGI", 10000)):
        a = summ[(summ.method == method) & (summ.shots == n)]
        base = a[a.seed == OFFICIAL_SEED].iloc[0]
        un = a[a.seed.isin(SEEDS_UNSEEN)]
        deltas[f"{method} ({n})"] = {k: float(un[k].mean() - base[k]) for k in ("PSNR", "SSIM", "Pearson_r")}
    checks["unseen_minus_2026_mean"] = deltas

    # ======================= figures (seed 2027, rule-based picks) ===================
    ex_ids = json.loads((RUNS["A"][1] / "training_report.json").read_text())["val_example_ids"]
    picks = [ex_ids[0], ex_ids[2]]          # Model A's p10 and p50 rows (input-PSNR rule)
    seed = SEEDS_UNSEEN[0]
    per = per_seed_frames[seed]
    rows = []
    for image_id in picks:
        j = id_va.index(image_id)
        r = per.iloc[j]

        def ttl(t, tag):
            return f"{t}\n{r[f'{tag}_PSNR']:.2f} dB | SSIM {r[f'{tag}_SSIM']:.3f}"
        rows.append({"label": f"{image_id} (val)\nseed {seed} (unseen)", "target": y_va[j, 0], "panels": [
            ("Original (target)", y_va[j, 0]),
            (ttl("DGI-500", "DGI500"), inputs[seed][500][j, 0]),
            (ttl("U-Net A", "UNetA"), np.clip(preds_keep[(seed, "A")][j, 0], 0, 1)),
            (ttl("DGI-1000", "DGI1000"), inputs[seed][1000][j, 0]),
            (ttl("U-Net B", "UNetB"), np.clip(preds_keep[(seed, "B")][j, 0], 0, 1)),
            (ttl("DGI-10,000", "DGI10000"), inputs[seed][10000][j, 0])]})
    head = (f"UNSEEN pattern sequence (seed {seed}); models trained on seed 2026 only, no retraining\n")
    save_panel_grid(rows, OUT / "representative_unseen_seed_reconstructions.png",
                    head + "VALIDATION images, gray window [0, 1] for every panel")
    save_panel_grid(rows, OUT / "representative_unseen_seed_reconstructions_DISPLAY_ONLY.png",
                    head + "DISPLAY ONLY: one window per row (target's 1-99th pct); metrics from [0, 1] images",
                    display_window=True)

    report = {"sigma_grid": SIGMAS, "sigma_selection": "lowest mean MSE on the 341 training images, per N",
              "sigma_frozen": sigma_star, "seeds_unseen": SEEDS_UNSEEN, "official_seed": OFFICIAL_SEED,
              "n_val_images": len(id_va), "test_images_used": 0, "checks": checks,
              "summary": summ_out.to_dict("records"), "smoothing_vs_unet": vs.to_dict("records"),
              "figure_images": picks, "runtime_s": round(time.time() - t0, 1)}
    (OUT / "generalization_report.json").write_text(json.dumps(report, indent=2, default=float))

    pd.set_option("display.width", 220)
    print("\nsmoothing tuning (train):\n" + tuning.pivot(index="sigma", columns="shots",
                                                        values=["train_MSE", "train_PSNR", "train_SSIM"]).round(4).to_string())
    print("\nper-seed summary:\n" + summ[["method", "shots", "seed"] + list(KEYS)].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nsmoothing baseline:\n" + pd.DataFrame(smooth_rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nsmoothing vs U-Net:\n" + vs.T.to_string())
    print("\nchecks:", json.dumps(checks, indent=1, default=float))
    print(f"\n{report['runtime_s']} s -> {OUT}")


if __name__ == "__main__":
    main()
