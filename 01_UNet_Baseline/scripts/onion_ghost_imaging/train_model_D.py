#!/usr/bin/env python3
"""Phase 2B Model D: DGI-500 -> U-Net -> original, random-pattern training (the 500-shot twin of Model C).

    python train_model_D.py

Identical to Model C (train_unet_gi_random.train_random_model) except n_shots = 500:
  * training input: DGI-500 = the first 500 patterns of the sequence seeded by
    10,000,000 + 1000*epoch + train_index (the SAME seeds Model C used, so each D input
    is built from the first half of exactly the measurements C saw at 1,000 shots)
  * checkpoint selection: 75 val images x seeds 2040-2042 (DGI-500)
  * final check: seeds 2027-2029, never used for training or selection; seed 2026 also reported
Smoothing baselines: Gaussian sigma per shot count chosen by lowest mean MSE on the 341 training
images (seed-2026 DGI), never on validation. sigma(500) and sigma(1000) are the frozen values from
generalization_validation; sigma(10000) is tuned here with the same rule.
The test split is never loaded. Earlier model directories are only read.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.ghost_imaging import pattern_batches, pattern_variance  # noqa: E402
from src.gi_unet_data import load_pairs  # noqa: E402
from src.reconstruction import StreamingDGI  # noqa: E402
from src.unet_gi import UNetGI  # noqa: E402
from src.visualization import save_loss_curves, save_panel_grid, save_val_metric_curves  # noqa: E402
from run_generalization_validation import SIGMAS, smooth  # noqa: E402
from train_unet_gi import evaluate_set, predict  # noqa: E402
from train_unet_gi_random import (EXTRA_SEED, HIST, KEYS, UNSEEN_SEEDS, VAL_SEEDS,  # noqa: E402
                                  build_parser, train_random_model)

N_SHOTS = 500
OUT_DEFAULT = ROOT / "outputs" / "phase2b_unet" / "model_D_random_patterns_dgi500"
EVAL_SHOTS = [500, 1000, 5000, 10000]


def dgi_multi(objects: np.ndarray, seed: int) -> dict[int, np.ndarray]:
    """Phase 2A DGI (shared sequence for all objects) read out at EVAL_SHOTS, clipped to [0, 1]."""
    stream, out = StreamingDGI(objects), {}
    for pats in pattern_batches(max(EVAL_SHOTS), objects.shape[1:], seed, 500, 0.5):
        stream.update(pats)
        if stream.n in EVAL_SHOTS:
            est, _ = stream.estimate(pattern_variance(0.5))
            out[stream.n] = np.clip(est, 0, 1).reshape(-1, 1, *objects.shape[1:]).astype(np.float32)
    return out


def main() -> None:
    ap = build_parser()
    ap.set_defaults(out=OUT_DEFAULT)
    args = ap.parse_args()
    t0 = time.time()

    d = train_random_model(args, N_SHOTS)              # identical procedure to Model C, 500 shots
    out, checks, id_va, y_va = d.out, d.checks, d.id_va, d.y_va
    model_d = d.model

    # ---- smoothing sigmas (training images only) ------------------------------------
    sigma = dict(d.sigma)                                # frozen 500 / 1000 values
    x_tr10k, y_tr10k, _ = load_pairs("train", 10000)
    tune = []
    for s in SIGMAS:
        e = evaluate_set(y_tr10k, smooth(x_tr10k, s))
        tune.append({"shots": 10000, "sigma": s, **{f"train_{k}": e[k] for k in KEYS}})
    tune = pd.DataFrame(tune)
    sigma[10000] = float(tune.loc[tune.train_MSE.idxmin(), "sigma"])
    tune.to_csv(out / "smoothing_tuning_dgi10000_train.csv", index=False)
    checks["sigma_used"] = sigma
    checks["sigma10000_at_grid_edge"] = sigma[10000] in (SIGMAS[0], SIGMAS[-1])

    # ---- evaluation inputs: val images, shared sequence per seed ---------------------
    inputs = {}
    for s in UNSEEN_SEEDS + VAL_SEEDS + [EXTRA_SEED]:
        inputs[s] = dgi_multi(d.obj_va, s)
    checks["eval_inputs_match_training_pipeline_max_abs_diff"] = float(max(
        np.abs(inputs[s][n] - d.eval_inputs[s][n]).max() for s in UNSEEN_SEEDS + VAL_SEEDS + [EXTRA_SEED]
        for n in (500, 1000, 10000)))

    ck_c = torch.load(HIST / "model_C_random_patterns" / "checkpoints" / "best.pt", weights_only=False)
    model_c = UNetGI(base=ck_c["config"]["base"])
    model_c.load_state_dict(ck_c["model"])
    checks["model_C_best_epoch_loaded"] = int(ck_c["epoch"])

    # ---- evaluate all methods ---------------------------------------------------------
    methods = [("DGI", 500, "DGI500", lambda x: x[500]),
               (f"Smoothed DGI (sigma={sigma[500]})", 500, "Smooth500", lambda x: smooth(x[500], sigma[500])),
               ("Model D (random seeds)", 500, "ModelD", lambda x: predict(model_d, x[500])),
               ("DGI", 1000, "DGI1000", lambda x: x[1000]),
               (f"Smoothed DGI (sigma={sigma[1000]})", 1000, "Smooth1000", lambda x: smooth(x[1000], sigma[1000])),
               ("Model C (random seeds)", 1000, "ModelC", lambda x: predict(model_c, x[1000])),
               ("DGI", 5000, "DGI5000", lambda x: x[5000]),
               ("DGI", 10000, "DGI10000", lambda x: x[10000]),
               (f"Smoothed DGI (sigma={sigma[10000]})", 10000, "Smooth10000", lambda x: smooth(x[10000], sigma[10000]))]
    rows, frames, keep = [], {}, {}
    for s in UNSEEN_SEEDS + [EXTRA_SEED] + VAL_SEEDS:
        per = pd.DataFrame({"image_id": id_va, "seed": s})
        role = "unseen" if s in UNSEEN_SEEDS else ("A/B training seed" if s == EXTRA_SEED else "C/D selection seed")
        for name, shots, tag, fn in methods:
            img = fn(inputs[s])
            e = evaluate_set(y_va, img)
            rows.append({"method": name, "shots": shots, "tag": tag, "seed": s, "seed_role": role,
                         **{k: e[k] for k in KEYS}})
            for k in KEYS:
                per[f"{tag}_{k}"] = e["per_image"][k].values
            keep[(s, tag)] = np.clip(img, 0, 1)
        frames[s] = per
        if s in UNSEEN_SEEDS:
            per.to_csv(out / f"unseen_seed_{s}.csv", index=False)
    pd.concat([frames[s] for s in [EXTRA_SEED] + VAL_SEEDS]).to_csv(out / "other_seeds_per_image.csv", index=False)
    long = pd.DataFrame(rows)
    long.to_csv(out / "all_seeds_summary_long.csv", index=False)

    # ---- main table over unseen seeds -------------------------------------------------
    cat = pd.concat([frames[s] for s in UNSEEN_SEEDS], ignore_index=True)      # 225 image-seed pairs
    un = long[long.seed.isin(UNSEEN_SEEDS)]
    table = []
    for name, shots, tag, _ in methods:
        g = un[un.tag == tag]
        row = {"method": name, "shots": shots}
        for k in KEYS:
            row[f"{k}_mean"] = g[k].mean()
            row[f"{k}_std_over_seeds"] = g[k].std()
            row[f"{k}_median_per_image"] = cat[f"{tag}_{k}"].median()
        for ref in ("DGI10000", "Smooth10000"):
            for k in ("PSNR", "SSIM", "Pearson_r"):
                row[f"win_pct_vs_{ref}_{k}"] = (float((cat[f"{tag}_{k}"] > cat[f"{ref}_{k}"]).mean() * 100)
                                               if tag != ref else np.nan)
        table.append(row)
    table = pd.DataFrame(table)
    table.to_csv(out / "unseen_seed_summary.csv", index=False)

    paired = {}
    for a, b in (("ModelD", "DGI500"), ("ModelD", "Smooth500"), ("ModelD", "DGI1000"), ("ModelD", "DGI5000"),
                 ("ModelD", "DGI10000"), ("ModelD", "Smooth10000"), ("ModelC", "ModelD"),
                 ("ModelC", "DGI10000"), ("ModelC", "Smooth10000"), ("ModelC", "Smooth1000")):
        for k in ("PSNR", "SSIM", "Pearson_r", "MSE"):
            diff = cat[f"{a}_{k}"] - cat[f"{b}_{k}"]
            better = diff < 0 if k == "MSE" else diff > 0
            paired[f"{a}_minus_{b}_{k}"] = {"mean": float(diff.mean()), "median": float(diff.median()),
                                            "std": float(diff.std()), f"pct_{a}_better": float(better.mean() * 100)}
    pd.DataFrame(paired).T.to_csv(out / "paired_unseen_225.csv")

    # other seeds: does D behave the same everywhere?
    seed_view = long[long.tag.isin(["ModelD", "ModelC", "DGI10000"])].pivot_table(
        index="seed", columns="tag", values=["PSNR", "SSIM", "Pearson_r"])
    seed_view.to_csv(out / "per_seed_D_C_DGI10000.csv")

    # overfitting indicator: training images with fresh sequences beyond the training seed range
    from train_unet_gi_random import TRAIN_SEED_BASE, dgi_single
    x_fresh = np.stack([dgi_single(d.obj_tr[i], TRAIN_SEED_BASE + 900_000 + i, N_SHOTS) for i in range(len(d.obj_tr))])
    ev_tr = evaluate_set(d.y_tr, predict(model_d, x_fresh))
    overfit = {"train_images_fresh_seq_PSNR": ev_tr["PSNR"], "train_images_fresh_seq_SSIM": ev_tr["SSIM"],
               "val_PSNR_at_best": d.ev_b["PSNR"], "val_SSIM_at_best": d.ev_b["SSIM"],
               "epochs_after_best": int((d.history.epoch > d.ckb["epoch"]).sum())}

    # ---- figures -----------------------------------------------------------------------
    tag = (f"Model D: DGI-500 (fresh random pattern sequence per image per epoch) -> U-Net | "
           f"best epoch {d.ckb['epoch']} of {int(d.history.epoch.max())}")
    save_loss_curves(d.history, pd.read_csv(out / "batch_losses.csv"), out / "loss_curves.png",
                     tag + "\nvalidation = 75 val images x held-out seeds 2040-2042",
                     {k: v["loss"] for k, v in d.refs.items()})
    save_val_metric_curves(d.history, out / "metric_curves.png", tag + " | validation seeds 2040-2042",
                           {k: {m: v[m] for m in ("PSNR", "SSIM", "Pearson_r")} for k, v in d.refs.items()},
                           int(d.ckb["epoch"]), unet_label="Model D (DGI-500 input)")
    ex_ids = json.loads((HIST / "model_A_dgi500" / "full" / "training_report.json").read_text())["val_example_ids"]
    s = UNSEEN_SEEDS[0]
    per = frames[s]
    fig_rows = []
    for image_id in ex_ids:
        j = id_va.index(image_id)
        r = per.iloc[j]

        def ttl(t, tg):
            return f"{t}\n{r[f'{tg}_PSNR']:.2f} dB | SSIM {r[f'{tg}_SSIM']:.3f}"
        fig_rows.append({"label": f"{image_id} (val)\nseed {s} (unseen)", "target": y_va[j, 0], "panels": [
            ("Original (target)", y_va[j, 0]),
            (ttl("DGI-500", "DGI500"), keep[(s, "DGI500")][j, 0]),
            (ttl(f"Smoothed DGI-500 (σ={sigma[500]})", "Smooth500"), keep[(s, "Smooth500")][j, 0]),
            (ttl("Model D (500 shots)", "ModelD"), keep[(s, "ModelD")][j, 0]),
            (ttl("DGI-1000", "DGI1000"), keep[(s, "DGI1000")][j, 0]),
            (ttl("Model C (1000 shots)", "ModelC"), keep[(s, "ModelC")][j, 0]),
            (ttl("DGI-10,000", "DGI10000"), keep[(s, "DGI10000")][j, 0])]})
    head = f"UNSEEN pattern sequence (seed {s}); neither model saw it during training or checkpoint selection\n"
    save_panel_grid(fig_rows, out / "reconstruction_examples.png",
                    head + "VALIDATION images (same 5 as Models A/B/C), gray window [0, 1] for every panel")
    save_panel_grid(fig_rows, out / "reconstruction_examples_DISPLAY_ONLY.png",
                    head + "DISPLAY ONLY: one window per row (target's 1-99th pct); metrics from [0, 1] images",
                    display_window=True)

    report = {"model": "D", "n_shots": N_SHOTS, "n_params": d.n_params,
              "architecture": "UNetGI(base=32, depth=3), identical to Models B/C",
              "optimizer": f"Adam lr={args.lr} (constant)", "loss": "MSE", "batch_size": args.batch_size,
              "max_epochs": args.epochs, "patience": args.patience, "master_seed": args.seed,
              "epochs_completed": int(d.history.epoch.max()), "stopped_early": bool(d.stopped_early),
              "best_epoch": int(d.ckb["epoch"]), "best_val_loss": float(d.ckb["val_loss"]),
              "seeds": {"training": f"10,000,000 + 1000*epoch + train_index -> {list(d.train_seed_range)} "
                                    "(same seeds as Model C; D uses the first 500 patterns of each)",
                        "checkpoint_selection": VAL_SEEDS, "final_unseen": UNSEEN_SEEDS,
                        "also_reported": [EXTRA_SEED], "test_images_used": 0},
              "smoothing_sigma": sigma, "checks": checks, "overfitting": overfit,
              "unseen_table": table.to_dict("records"), "paired_unseen_225": paired,
              "wall_time_s": round(time.time() - d.t_start, 1)}
    (out / "training_report.json").write_text(json.dumps(report, indent=2, default=float))

    pd.set_option("display.width", 220)
    show = table[["method", "shots"] + [f"{k}_mean" for k in KEYS] + [f"{k}_std_over_seeds" for k in ("PSNR", "SSIM", "Pearson_r")]
                 + [f"{k}_median_per_image" for k in ("PSNR", "SSIM", "Pearson_r")]
                 + [f"win_pct_vs_DGI10000_{k}" for k in ("PSNR", "SSIM", "Pearson_r")]
                 + [f"win_pct_vs_Smooth10000_{k}" for k in ("PSNR", "SSIM", "Pearson_r")]]
    print("\nunseen seeds 2027-2029:\n" + show.T.to_string(header=False, float_format=lambda v: f"{v:.4f}"))
    print("\npaired (225):\n" + pd.DataFrame(paired).T.round(4).to_string())
    print("\nper seed:\n" + seed_view.round(4).to_string())
    print("\noverfitting:", json.dumps(overfit, indent=1, default=float))
    print("\nchecks:", json.dumps(checks, indent=1, default=float))
    print(f"\nepochs {report['epochs_completed']}, best {report['best_epoch']}, {report['wall_time_s']} s -> {out}")


if __name__ == "__main__":
    main()
