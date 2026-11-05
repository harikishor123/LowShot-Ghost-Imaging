#!/usr/bin/env python3
"""Phase 2B Model C: DGI-1000 -> U-Net -> original, trained with pattern-sequence augmentation.

    python train_unet_gi_random.py                      # full run (80 epochs max, patience 10)
    python train_unet_gi_random.py --epochs 1 --out <dir>   # smoke test

Only change from Model B: the DGI-1000 input of every training image is re-simulated
EVERY epoch with its own, never-repeated illumination-pattern sequence:

    train seed(epoch e, train image i) = 10,000,000 + 1000*e + i      (e = 1..80, i = 0..340)

    validation / checkpoint selection : seeds 2040, 2041, 2042 (fixed DGI-1000 inputs, 75 val images)
    final unseen-seed check           : seeds 2027, 2028, 2029 (never used for training or selection)
    also reported                     : seed 2026 (Model A/B training sequence; NOT used to train C)

Pattern model, DGI, calibration and clipping are identical to Phase 2A (src.reconstruction.StreamingDGI):
1,000 i.i.d. Bernoulli(0.5) 64x64 patterns, O_hat = DGI/0.25 + <B>/<R>, clipped to [0, 1].
Architecture, optimizer, lr, batch size, loss, early stopping: identical to Model B.
The test split (74 images) is never loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.ghost_imaging import load_object, pattern_batches, pattern_variance  # noqa: E402
from src.gi_unet_data import PairDataset, load_pairs, read_split_ids  # noqa: E402
from src.reconstruction import StreamingDGI  # noqa: E402
from src.unet_gi import UNetGI, count_parameters  # noqa: E402
from src.visualization import save_loss_curves, save_panel_grid, save_val_metric_curves  # noqa: E402
from run_generalization_validation import dgi_for_seed, smooth  # noqa: E402
from train_unet_gi import evaluate_set, predict, set_determinism  # noqa: E402

N_SHOTS = 1000
TRAIN_SEED_BASE = 10_000_000
VAL_SEEDS = [2040, 2041, 2042]
UNSEEN_SEEDS = [2027, 2028, 2029]
EXTRA_SEED = 2026
KEYS = ("PSNR", "SSIM", "MSE", "Pearson_r")
OUT_DEFAULT = ROOT / "outputs" / "phase2b_unet" / "model_C_random_patterns"
HIST = ROOT / "outputs" / "phase2b_unet"


def train_seed(epoch: int, i: int) -> int:
    return TRAIN_SEED_BASE + 1000 * epoch + i


def dgi_single(obj: np.ndarray, seed: int, n: int = N_SHOTS) -> np.ndarray:
    """Calibrated, clipped DGI-n estimate (1, 64, 64) of one object with its own pattern sequence."""
    stream = StreamingDGI(obj[None])
    for pats in pattern_batches(n, obj.shape, seed, 500, 0.5):
        stream.update(pats)
    est, _ = stream.estimate(pattern_variance(0.5))
    return np.clip(est, 0, 1).reshape(1, *obj.shape).astype(np.float32)


def epoch_inputs(objects: np.ndarray, epoch: int, n: int = N_SHOTS) -> np.ndarray:
    return np.stack([dgi_single(objects[i], train_seed(epoch, i), n) for i in range(len(objects))])


def load_models():
    out = {}
    for name, sub in (("A", "model_A_dgi500/full"), ("B", "model_B_dgi1000/full")):
        ck = torch.load(HIST / sub / "checkpoints" / "best.pt", weights_only=False)
        m = UNetGI(base=ck["config"]["base"])
        m.load_state_dict(ck["model"])
        out[name] = m
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0, help="master seed (init, shuffling)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    return ap


def train_random_model(args, n_shots: int) -> SimpleNamespace:
    """Train the random-pattern U-Net on DGI-n_shots (fresh sequence per image per epoch); reload best.pt."""
    out = args.out
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    gen = set_determinism(args.seed)
    t_start = time.time()
    checks: dict = {}

    # ---- data (train + val only) ------------------------------------------------
    ids = read_split_ids()
    _, y_tr, id_tr = load_pairs("train", n_shots)
    x26_va, y_va, id_va = load_pairs("val", 1000)             # seed-2026 DGI (reporting only)
    x26_va500, _, _ = load_pairs("val", 500)
    x26_va10k, _, _ = load_pairs("val", 10000)
    checks["n_train"], checks["n_val"] = len(id_tr), len(id_va)
    checks["test_ids_loaded"] = len((set(id_tr) | set(id_va)) & set(ids["test"]))
    ref_dir = ROOT / "data" / "v2" / "reference"
    obj_tr = np.stack([load_object(ref_dir / f"{i}.npz", 64)[0] for i in id_tr])
    obj_va = np.stack([load_object(ref_dir / f"{i}.npz", 64)[0] for i in id_va])
    checks["objects_vs_targets_max_abs_diff"] = float(max(np.abs(obj_tr - y_tr[:, 0]).max(),
                                                          np.abs(obj_va - y_va[:, 0]).max()))

    # ---- seed bookkeeping ---------------------------------------------------------
    train_seed_range = (train_seed(1, 0), train_seed(args.epochs, len(id_tr) - 1))
    reserved = set(VAL_SEEDS) | set(UNSEEN_SEEDS) | {EXTRA_SEED}
    checks["train_seed_range"] = list(train_seed_range)
    checks["train_seeds_unique"] = len({train_seed(e, i) for e in range(1, args.epochs + 1)
                                        for i in range(len(id_tr))}) == args.epochs * len(id_tr)
    checks["train_seeds_overlap_reserved"] = any(train_seed_range[0] <= s <= train_seed_range[1] for s in reserved)
    checks["val_unseen_overlap"] = sorted(set(VAL_SEEDS) & set(UNSEEN_SEEDS))
    # the single-image generator must reproduce the validated Phase 2A pipeline exactly
    x26_tr, _, _ = load_pairs("train", n_shots)
    checks["dgi_single_seed2026_vs_phase2A_max_abs_diff"] = float(max(
        np.abs(dgi_single(obj_tr[j], EXTRA_SEED, n_shots) - x26_tr[j]).max() for j in (0, 100, 200, 340)))

    # ---- fixed evaluation inputs (val images, shared sequence per seed, as in Phase 2A) --
    eval_inputs = {s: dgi_for_seed(obj_va, s) for s in VAL_SEEDS + UNSEEN_SEEDS}
    eval_inputs[EXTRA_SEED] = {500: x26_va500, 1000: x26_va, 10000: x26_va10k}
    x_val = np.concatenate([eval_inputs[s][n_shots] for s in VAL_SEEDS])
    y_val = np.concatenate([y_va] * len(VAL_SEEDS))
    sigma = json.loads((HIST / "generalization_validation" / "generalization_report.json").read_text())["sigma_frozen"]
    sigma = {int(k): float(v) for k, v in sigma.items()}
    refs = {f"DGI-{n_shots} input": evaluate_set(y_val, x_val),
            f"smoothed DGI-{n_shots} (sigma={sigma[n_shots]})": evaluate_set(y_val, smooth(x_val, sigma[n_shots])),
            "DGI-10,000": evaluate_set(y_val, np.concatenate([eval_inputs[s][10000] for s in VAL_SEEDS]))}

    # ---- model ----------------------------------------------------------------------
    model = UNetGI(base=args.base)
    n_params = count_parameters(model)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    print(f"U-Net {n_params:,} params | train seeds {train_seed_range} | val seeds {VAL_SEEDS} | "
          f"unseen {UNSEEN_SEEDS}", flush=True)
    print(json.dumps(checks, indent=1), flush=True)

    def val_eval(m):
        pred = predict(m, x_val)
        ev = evaluate_set(y_val, pred)
        per_seed = {}
        for k, s in enumerate(VAL_SEEDS):
            sl = slice(k * len(id_va), (k + 1) * len(id_va))
            e = evaluate_set(y_va, pred[sl])
            per_seed[s] = {m_: e[m_] for m_ in KEYS} | {"loss": e["loss"]}
        return ev, per_seed

    ev0, _ = val_eval(model)
    history = [{"epoch": 0, "train_loss": np.nan, "val_loss": ev0["loss"], "val_psnr": ev0["PSNR"],
                "val_ssim": ev0["SSIM"], "val_correlation": ev0["Pearson_r"], "val_mse_clipped": ev0["MSE"],
                "learning_rate": args.lr, "gen_seconds": 0.0, "train_seconds": 0.0}]
    val_rows, batch_log = [], []
    best, best_epoch, since, stopped_early = np.inf, 0, 0, False
    prev_inputs = None
    for epoch in range(1, args.epochs + 1):
        tg = time.time()
        x_ep = epoch_inputs(obj_tr, epoch, n_shots)
        gen_s = time.time() - tg
        if epoch == 1:
            checks["epoch1_input_differs_from_seed2026_mean_abs"] = float(np.abs(x_ep - x26_tr).mean())
        if prev_inputs is not None and epoch == 2:
            checks["epoch2_vs_epoch1_input_mean_abs_diff"] = float(np.abs(x_ep - prev_inputs).mean())
        prev_inputs = x_ep
        loader = DataLoader(PairDataset(x_ep, y_tr, id_tr), batch_size=args.batch_size, shuffle=True,
                            generator=gen, num_workers=0)
        tt = time.time()
        model.train()
        tot, cnt = 0.0, 0
        for b, (xb, yb, _) in enumerate(loader):
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} batch {b}")
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            cnt += len(xb)
            batch_log.append({"epoch_float": epoch - 1 + (b + 1) / len(loader), "loss": loss.item()})
        train_s = time.time() - tt
        ev, per_seed = val_eval(model)
        row = {"epoch": epoch, "train_loss": tot / cnt, "val_loss": ev["loss"], "val_psnr": ev["PSNR"],
               "val_ssim": ev["SSIM"], "val_correlation": ev["Pearson_r"], "val_mse_clipped": ev["MSE"],
               "learning_rate": opt.param_groups[0]["lr"], "gen_seconds": round(gen_s, 1),
               "train_seconds": round(train_s, 1)}
        history.append(row)
        for s, m in per_seed.items():
            val_rows.append({"epoch": epoch, "val_seed": s, **m})
        ck = {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch, "val_loss": ev["loss"],
              "val_metrics": {k: ev[k] for k in KEYS},
              "config": vars(args) | {"n_shots": n_shots, "n_params": n_params, "val_seeds": VAL_SEEDS,
                                      "train_seed_rule": "10,000,000 + 1000*epoch + train_index"}}
        ck["config"]["out"] = str(ck["config"]["out"])
        torch.save(ck, out / "checkpoints" / "last.pt")
        improved = ev["loss"] < best
        if improved:
            best, best_epoch, since = ev["loss"], epoch, 0
            torch.save(ck, out / "checkpoints" / "best.pt")
        else:
            since += 1
        print(f"epoch {epoch:3d}: train {row['train_loss']:.5f} | val(2040-42) {row['val_loss']:.5f} | "
              f"PSNR {row['val_psnr']:.2f} SSIM {row['val_ssim']:.3f} r {row['val_correlation']:.3f}"
              f"{'  *best' if improved else ''} | gen {gen_s:.1f}s train {train_s:.1f}s", flush=True)
        if since >= args.patience:
            stopped_early = True
            print(f"early stopping: no val-loss improvement for {args.patience} epochs", flush=True)
            break

    history = pd.DataFrame(history)
    history.to_csv(out / "history.csv", index=False)
    pd.DataFrame(batch_log).to_csv(out / "batch_losses.csv", index=False)
    pd.DataFrame(val_rows).to_csv(out / "validation_metrics.csv", index=False)

    # ---- reload best, verify ------------------------------------------------------
    model_c = UNetGI(base=args.base)
    ckb = torch.load(out / "checkpoints" / "best.pt", weights_only=False)
    model_c.load_state_dict(ckb["model"])
    ev_b, _ = val_eval(model_c)
    rec = history.set_index("epoch").loc[ckb["epoch"]]
    checks["best_epoch"] = int(ckb["epoch"])
    checks["best_reload_max_abs_diff"] = float(max(abs(ev_b["loss"] - rec.val_loss), abs(ev_b["PSNR"] - rec.val_psnr),
                                                   abs(ev_b["SSIM"] - rec.val_ssim),
                                                   abs(ev_b["Pearson_r"] - rec.val_correlation)))

    return SimpleNamespace(model=model_c, out=out, checks=checks, t_start=t_start, id_tr=id_tr, y_tr=y_tr, obj_tr=obj_tr, id_va=id_va, y_va=y_va, obj_va=obj_va, eval_inputs=eval_inputs, sigma=sigma, refs=refs, history=history, ckb=ckb, ev_b=ev_b, n_params=n_params, stopped_early=stopped_early, train_seed_range=train_seed_range, val_eval=val_eval)


def main() -> None:
    args = build_parser().parse_args()
    c = train_random_model(args, N_SHOTS)
    model_c = c.model
    out, checks, t_start, id_tr, y_tr, obj_tr, id_va, y_va, obj_va, eval_inputs, sigma, refs, history, ckb, ev_b, n_params, stopped_early, train_seed_range, val_eval = (c.out, c.checks, c.t_start, c.id_tr, c.y_tr, c.obj_tr, c.id_va, c.y_va, c.obj_va, c.eval_inputs, c.sigma, c.refs, c.history, c.ckb, c.ev_b, c.n_params, c.stopped_early, c.train_seed_range, c.val_eval)

    # ---- final evaluation: unseen seeds 2027-2029 (+ 2026, + val seeds for reference) --
    ab = load_models()
    summary_rows, frames, keep = [], {}, {}
    for s in UNSEEN_SEEDS + [EXTRA_SEED] + VAL_SEEDS:
        inp = eval_inputs[s]
        methods = {
            "DGI-1000": inp[1000],
            f"Smoothed DGI-1000 (sigma={sigma[1000]})": smooth(inp[1000], sigma[1000]),
            "Model A (fixed seed 2026, DGI-500)": predict(ab["A"], inp[500]),
            "Model B (fixed seed 2026, DGI-1000)": predict(ab["B"], inp[1000]),
            "Model C (random seeds, DGI-1000)": predict(model_c, inp[1000]),
            "DGI-10000": inp[10000],
        }
        per = pd.DataFrame({"image_id": id_va, "seed": s})
        tags = {"DGI-1000": "DGI1000", f"Smoothed DGI-1000 (sigma={sigma[1000]})": "Smooth1000",
                "Model A (fixed seed 2026, DGI-500)": "ModelA", "Model B (fixed seed 2026, DGI-1000)": "ModelB",
                "Model C (random seeds, DGI-1000)": "ModelC", "DGI-10000": "DGI10000"}
        for name, img in methods.items():
            e = evaluate_set(y_va, img)
            shots = 500 if name.startswith("Model A") else (10000 if name == "DGI-10000" else 1000)
            role = ("unseen" if s in UNSEEN_SEEDS else "A/B training seed" if s == EXTRA_SEED else "C validation seed")
            summary_rows.append({"method": name, "shots": shots, "seed": s, "seed_role": role,
                                 **{k: e[k] for k in KEYS},
                                 **{f"{k}_median": float(e["per_image"][k].median()) for k in ("PSNR", "SSIM", "Pearson_r")}})
            for k in KEYS:
                per[f"{tags[name]}_{k}"] = e["per_image"][k].values
            keep[(s, tags[name])] = np.clip(img, 0, 1)
        frames[s] = per
        if s in UNSEEN_SEEDS:
            per.to_csv(out / f"unseen_seed_{s}.csv", index=False)
    pd.concat([frames[s] for s in [EXTRA_SEED] + VAL_SEEDS]).to_csv(out / "other_seeds_per_image.csv", index=False)

    summ = pd.DataFrame(summary_rows)
    agg = (summ[summ.seed.isin(UNSEEN_SEEDS)].groupby(["method", "shots"], sort=False)[list(KEYS)]
           .agg(["mean", "std"]))
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.insert(2, "seed", "2027-2029 mean")
    summ_out = pd.concat([summ.assign(seed=summ.seed.astype(str)),
                          agg.rename(columns={f"{k}_mean": k for k in KEYS})], ignore_index=True)
    summ_out.to_csv(out / "unseen_seed_summary.csv", index=False)

    strategy = {"Model A (fixed seed 2026, DGI-500)": "fixed seed 2026",
                "Model B (fixed seed 2026, DGI-1000)": "fixed seed 2026",
                "Model C (random seeds, DGI-1000)": "fresh random sequence per image per epoch",
                f"Smoothed DGI-1000 (sigma={sigma[1000]})": "non-learned (sigma tuned on train)",
                "DGI-1000": "conventional", "DGI-10000": "conventional"}
    comp = summ[summ.seed.isin(UNSEEN_SEEDS + [EXTRA_SEED])][["method", "shots", "seed", "seed_role"] + list(KEYS)].copy()
    comp.insert(1, "training_pattern_strategy", comp.method.map(strategy))
    comp_avg = agg.copy()
    comp_avg.insert(1, "training_pattern_strategy", comp_avg.method.map(strategy))
    comp_avg["seed_role"] = "unseen (mean, std over seeds)"
    pd.concat([comp.assign(seed=comp.seed.astype(str)), comp_avg], ignore_index=True).to_csv(
        out / "comparison_A_B_C.csv", index=False)

    # paired per-image statistics over the 225 unseen (image, seed) pairs
    cat = pd.concat([frames[s] for s in UNSEEN_SEEDS], ignore_index=True)
    paired = {}
    for other in ("ModelB", "Smooth1000", "DGI10000", "DGI1000", "ModelA"):
        for k in ("PSNR", "SSIM", "Pearson_r"):
            d = cat[f"ModelC_{k}"] - cat[f"{other}_{k}"]
            paired[f"C_minus_{other}_{k}"] = {"mean": float(d.mean()), "median": float(d.median()),
                                              "std": float(d.std()), "pct_C_better": float((d > 0).mean() * 100)}
    for k in ("PSNR", "SSIM", "Pearson_r"):
        d = cat[f"ModelB_{k}"] - cat[f"Smooth1000_{k}"]
        paired[f"B_minus_Smooth1000_{k}"] = {"mean": float(d.mean()), "pct_B_better": float((d > 0).mean() * 100)}
    c26 = frames[EXTRA_SEED]
    for k in ("PSNR", "SSIM", "Pearson_r"):
        d = c26[f"ModelC_{k}"] - c26[f"ModelB_{k}"]
        paired[f"seed2026_C_minus_B_{k}"] = {"mean": float(d.mean()), "pct_C_better": float((d > 0).mean() * 100)}

    # overfitting indicator on fresh training sequences (one new sequence per image, beyond training range)
    x_tr_fresh = np.stack([dgi_single(obj_tr[i], TRAIN_SEED_BASE + 900_000 + i) for i in range(len(obj_tr))])
    ev_tr = evaluate_set(y_tr, predict(model_c, x_tr_fresh))
    overfit = {"train_images_fresh_seq_PSNR": ev_tr["PSNR"], "train_images_fresh_seq_SSIM": ev_tr["SSIM"],
               "train_images_fresh_seq_loss": ev_tr["loss"], "val_loss_at_best": ev_b["loss"],
               "val_PSNR_at_best": ev_b["PSNR"], "val_SSIM_at_best": ev_b["SSIM"],
               "epochs_after_best": int((history.epoch > ckb["epoch"]).sum())}

    # ---- figures --------------------------------------------------------------------
    tag = (f"Model C: DGI-1000 (fresh random pattern sequence per image per epoch) -> U-Net | "
           f"best epoch {ckb['epoch']} of {int(history.epoch.max())}")
    save_loss_curves(history, pd.read_csv(out / "batch_losses.csv"), out / "loss_curves.png",
                     tag + "\nvalidation = 75 val images x held-out seeds 2040-2042",
                     {k: v["loss"] for k, v in refs.items()})
    save_val_metric_curves(history, out / "metric_curves.png", tag + " | validation seeds 2040-2042",
                           {k: {m: v[m] for m in ("PSNR", "SSIM", "Pearson_r")} for k, v in refs.items()},
                           int(ckb["epoch"]), unet_label="Model C (DGI-1000 input)")
    ex_ids = json.loads((HIST / "model_A_dgi500" / "full" / "training_report.json").read_text())["val_example_ids"]
    s = UNSEEN_SEEDS[0]
    per = frames[s]
    rows = []
    for q, image_id in zip((10, 30, 50, 70, 90), ex_ids):
        j = id_va.index(image_id)
        r = per.iloc[j]

        def ttl(t, tg):
            return f"{t}\n{r[f'{tg}_PSNR']:.2f} dB | SSIM {r[f'{tg}_SSIM']:.3f}"
        rows.append({"label": f"{image_id} (val)\nseed {s} (unseen)", "target": y_va[j, 0], "panels": [
            ("Original (target)", y_va[j, 0]),
            (ttl("DGI-1000", "DGI1000"), keep[(s, "DGI1000")][j, 0]),
            (ttl(f"Smoothed DGI-1000 (σ={sigma[1000]})", "Smooth1000"), keep[(s, "Smooth1000")][j, 0]),
            (ttl("Model B (fixed 2026)", "ModelB"), keep[(s, "ModelB")][j, 0]),
            (ttl("Model C (random seeds)", "ModelC"), keep[(s, "ModelC")][j, 0]),
            (ttl("DGI-10,000", "DGI10000"), keep[(s, "DGI10000")][j, 0])]})
    head = f"UNSEEN pattern sequence (seed {s}); no model saw it during training or checkpoint selection\n"
    save_panel_grid(rows, out / "reconstruction_examples.png",
                    head + "VALIDATION images (same 5 as Models A/B), gray window [0, 1] for every panel")
    save_panel_grid(rows, out / "reconstruction_examples_DISPLAY_ONLY.png",
                    head + "DISPLAY ONLY: one window per row (target's 1-99th pct); metrics from [0, 1] images",
                    display_window=True)

    report = {"model": "C", "n_shots": N_SHOTS, "n_params": n_params,
              "architecture": "UNetGI(base=32, depth=3), identical to Model B",
              "optimizer": f"Adam lr={args.lr} (constant)", "loss": "MSE", "batch_size": args.batch_size,
              "max_epochs": args.epochs, "patience": args.patience, "master_seed": args.seed,
              "epochs_completed": int(history.epoch.max()), "stopped_early": stopped_early,
              "best_epoch": int(ckb["epoch"]), "best_val_loss": float(ckb["val_loss"]),
              "pattern_augmentation": "every training image gets a fresh 1,000-pattern sequence every epoch; "
                                      "seed = 10,000,000 + 1000*epoch + train_index",
              "validation_seeds": VAL_SEEDS, "unseen_seeds": UNSEEN_SEEDS,
              "seed_2026": "not used to train or select Model C; reported as an additional unseen sequence",
              "smoothing_sigma": sigma, "checks": checks, "overfitting": overfit,
              "unseen_summary": agg.to_dict("records"), "paired_unseen": paired,
              "test_images_used": 0, "wall_time_s": round(time.time() - t_start, 1)}
    (out / "training_report.json").write_text(json.dumps(report, indent=2, default=float))

    pd.set_option("display.width", 200)
    print("\nunseen seeds 2027-2029 (mean, std over seeds):\n" + agg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nper seed:\n" + summ[["method", "seed"] + list(KEYS)].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\npaired (225 unseen image-seed pairs):\n" + pd.DataFrame(paired).T.round(4).to_string())
    print("\noverfitting:", json.dumps(overfit, indent=1, default=float))
    print("\nchecks:", json.dumps(checks, indent=1, default=float))
    print(f"\nepochs {report['epochs_completed']}, best {report['best_epoch']}, {report['wall_time_s']} s -> {out}")


if __name__ == "__main__":
    main()
