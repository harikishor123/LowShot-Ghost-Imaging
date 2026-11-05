#!/usr/bin/env python3
"""Phase 2B: U-Net reconstruction  DGI-N (seed 2026) -> original onion object.

    python train_unet_gi.py --model A                                  # SANITY run: DGI-500, 3 epochs, lr 1e-3
    python train_unet_gi.py --model A --full --epochs 80 --lr 1e-4     # full run, early stopping (patience 10)

    Model A: input DGI-500  -> target original
    Model B: input DGI-1000 -> target original

Data: train (341) and val (75) only. The test split (74) is never loaded here.
Loss: MSE between the raw (unclipped) network output and the target, both on the
physical [0, 1] linear-light scale. Metrics use the output clipped to [0, 1].
Model selection: lowest validation loss (best.pt); last.pt is saved every epoch.
Validation references (no learning): the DGI-N input, a flat image at the measured
<B>/<R>, and conventional DGI-10,000 (seed 2026), all against the original object.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.ghost_imaging import load_object  # noqa: E402
from src.gi_unet_data import DGI_NPZ, PairDataset, load_pairs, read_split_ids, verify_alignment  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.unet_gi import UNetGI, count_parameters  # noqa: E402
from src.visualization import save_loss_curves, save_panel_grid, save_val_metric_curves  # noqa: E402

MODELS = {"A": 500, "B": 1000}
REFERENCE_SHOTS = 10000


def set_determinism(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def image_metrics(target: np.ndarray, pred: np.ndarray) -> dict:
    """Metrics of one image (2-D arrays, pred already in [0, 1]). r is NaN for a constant pred."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return evaluate(target, pred)


@torch.no_grad()
def predict(model, x: np.ndarray, batch: int = 64) -> np.ndarray:
    model.eval()
    out = [model(torch.from_numpy(x[i:i + batch])).numpy() for i in range(0, len(x), batch)]
    return np.concatenate(out)


def evaluate_set(y: np.ndarray, pred_raw: np.ndarray) -> dict:
    """Loss on raw prediction (as trained) + mean image metrics on the clipped prediction."""
    loss = float(((pred_raw.astype(np.float64) - y) ** 2).mean())
    pred = np.clip(pred_raw, 0, 1)
    per = [image_metrics(y[i, 0].astype(np.float64), pred[i, 0].astype(np.float64)) for i in range(len(y))]
    df = pd.DataFrame(per)
    return {"loss": loss, "PSNR": df.PSNR.mean(), "SSIM": df.SSIM.mean(), "MSE": df.MSE.mean(),
            "Pearson_r": df.Pearson_r.mean(), "per_image": df,
            "finite": bool(np.isfinite(pred_raw).all()), "shape": list(pred_raw.shape[1:])}


def data_checks(args, n_shots):
    checks: dict = {}
    split_ids = read_split_ids()                        # asserts counts and pairwise disjointness
    checks["split_counts"] = {s: len(v) for s, v in split_ids.items()}
    checks["split_overlap"] = {f"{a}&{b}": len(set(split_ids[a]) & set(split_ids[b]))
                               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    x_tr, y_tr, id_tr = load_pairs("train", n_shots)
    x_va, y_va, id_va = load_pairs("val", n_shots)
    checks["n_train_loaded"], checks["n_val_loaded"] = len(id_tr), len(id_va)
    checks["test_ids_in_loaded_data"] = len((set(id_tr) | set(id_va)) & set(split_ids["test"]))
    checks["alignment_max_psnr_diff_db_train"] = verify_alignment(x_tr, y_tr, id_tr, n_shots)
    checks["alignment_max_psnr_diff_db_val"] = verify_alignment(x_va, y_va, id_va, n_shots)
    rng = np.random.default_rng(args.seed)
    spot = [(id_tr, y_tr, j) for j in rng.choice(len(id_tr), 3, replace=False)] + \
           [(id_va, y_va, j) for j in rng.choice(len(id_va), 2, replace=False)]
    checks["target_vs_fresh_reference_max_abs_diff"] = float(max(
        np.abs(load_object(ROOT / "data" / "v2" / "reference" / f"{ids[j]}.npz", 64)[0] - ys[j, 0]).max()
        for ids, ys, j in spot))
    checks["input_range"] = [float(x_tr.min()), float(x_tr.max())]
    checks["target_range"] = [float(y_tr.min()), float(y_tr.max())]
    checks["data_finite"] = bool(all(np.isfinite(a).all() for a in (x_tr, y_tr, x_va, y_va)))
    return checks, (x_tr, y_tr, id_tr), (x_va, y_va, id_va)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), default="A")
    ap.add_argument("--full", action="store_true", help="full training run (default is the sanity run)")
    ap.add_argument("--epochs", type=int, default=3, help="maximum number of epochs")
    ap.add_argument("--patience", type=int, default=10, help="early stopping on val loss (full runs)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=32, help="U-Net base channel count")
    ap.add_argument("--seed", type=int, default=0, help="training seed (init, shuffling)")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    if not args.full and args.epochs > 3:
        sys.exit("sanity mode allows at most 3 epochs; use --full for longer runs")

    n_shots = MODELS[args.model]
    run = "full" if args.full else "sanity"
    out = ROOT / "outputs" / "phase2b_unet" / f"model_{args.model}_dgi{n_shots}" / run
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    gen = set_determinism(args.seed)
    t_start = time.time()

    # ---- 1. data + leakage checks --------------------------------------------
    checks, (x_tr, y_tr, id_tr), (x_va, y_va, id_va) = data_checks(args, n_shots)
    print(json.dumps(checks, indent=1))

    # ---- 2. non-learned validation references (no ground truth used to form them)
    with np.load(DGI_NPZ, allow_pickle=True) as z:
        pos = {i: j for j, i in enumerate(z["image_id"].astype(str))}
        k_va = z[f"k_{n_shots}"][[pos[i] for i in id_va]]
    flat = np.broadcast_to(k_va[:, None, None, None], y_va.shape).astype(np.float32)
    x_ref, y_ref, id_ref = load_pairs("val", REFERENCE_SHOTS)        # conventional DGI-10,000
    assert id_ref == id_va and np.array_equal(y_ref, y_va), "DGI-10,000 reference not aligned with val targets"
    checks["dgi10000_alignment_max_psnr_diff_db"] = verify_alignment(x_ref, y_ref, id_ref, REFERENCE_SHOTS)
    refs = {f"DGI-{n_shots} input": evaluate_set(y_va, x_va),
            "flat image at <B>/<R>": evaluate_set(y_va, flat),
            f"DGI-{REFERENCE_SHOTS:,} (conventional)": evaluate_set(y_va, x_ref)}

    # ---- 3. model ----------------------------------------------------------------
    model = UNetGI(base=args.base)
    n_params = count_parameters(model)
    with torch.no_grad():
        checks["model_output_shape"] = list(model(torch.from_numpy(x_va[:2])).shape)
    print(f"U-Net base={args.base}: {n_params:,} parameters")
    loader = DataLoader(PairDataset(x_tr, y_tr, id_tr), batch_size=args.batch_size, shuffle=True,
                        generator=gen, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # ---- 4. train with validation every epoch -----------------------------------
    def hist_row(epoch, train_loss, ev, secs):
        return {"epoch": epoch, "train_loss": train_loss, "val_loss": ev["loss"], "val_psnr": ev["PSNR"],
                "val_ssim": ev["SSIM"], "val_correlation": ev["Pearson_r"], "val_mse_clipped": ev["MSE"],
                "learning_rate": opt.param_groups[0]["lr"], "seconds": round(secs, 1)}

    history = [hist_row(0, np.nan, evaluate_set(y_va, predict(model, x_va)), 0.0)]
    batch_log, seen = [], set()
    best_loss, best_epoch, since_best, stopped_early = np.inf, 0, 0, False
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot, cnt = 0.0, 0
        for b, (xb, yb, idx) in enumerate(loader):
            seen.update(idx.tolist())
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} batch {b}")
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            cnt += len(xb)
            batch_log.append({"epoch_float": epoch - 1 + (b + 1) / len(loader), "loss": loss.item()})
        ev = evaluate_set(y_va, predict(model, x_va))
        if not ev["finite"]:
            raise FloatingPointError(f"non-finite validation output at epoch {epoch}")
        row = hist_row(epoch, tot / cnt, ev, time.time() - t0)
        history.append(row)
        ckpt = {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch,
                "val_loss": ev["loss"], "val_metrics": {k: ev[k] for k in ("PSNR", "SSIM", "MSE", "Pearson_r")},
                "config": vars(args) | {"n_shots": n_shots, "n_params": n_params}}
        torch.save(ckpt, out / "checkpoints" / "last.pt")
        improved = ev["loss"] < best_loss
        if improved:
            best_loss, best_epoch, since_best = ev["loss"], epoch, 0
            torch.save(ckpt, out / "checkpoints" / "best.pt")
        else:
            since_best += 1
        print(f"epoch {epoch:3d}: train {row['train_loss']:.5f} | val {row['val_loss']:.5f} | "
              f"PSNR {row['val_psnr']:.2f} SSIM {row['val_ssim']:.3f} r {row['val_correlation']:.3f}"
              f"{'  *best' if improved else ''} | {row['seconds']} s", flush=True)
        if args.full and since_best >= args.patience:
            stopped_early = True
            print(f"early stopping: no val-loss improvement for {args.patience} epochs")
            break

    history = pd.DataFrame(history)
    history.to_csv(out / "history.csv", index=False)
    batch_log = pd.DataFrame(batch_log)
    batch_log.to_csv(out / "batch_losses.csv", index=False)
    checks["all_train_images_seen_each_epoch"] = len(seen) == len(id_tr)

    # ---- 5. reload best.pt, recompute, verify against the recorded epoch ---------
    best_model = UNetGI(base=args.base)
    ck = torch.load(out / "checkpoints" / "best.pt", weights_only=False)
    best_model.load_state_dict(ck["model"])
    pred_va = predict(best_model, x_va)
    ev_best = evaluate_set(y_va, pred_va)
    rec = history.set_index("epoch").loc[ck["epoch"]]
    checks["best_epoch"] = int(ck["epoch"])
    checks["best_reload_reproduces_recorded"] = {
        "val_loss_abs_diff": abs(ev_best["loss"] - rec.val_loss),
        "psnr_abs_diff": abs(ev_best["PSNR"] - rec.val_psnr),
        "ssim_abs_diff": abs(ev_best["SSIM"] - rec.val_ssim),
        "r_abs_diff": abs(ev_best["Pearson_r"] - rec.val_correlation)}
    checks["best_reload_exact"] = bool(max(checks["best_reload_reproduces_recorded"].values()) == 0.0)
    checks["output_finite"] = ev_best["finite"]
    checks["output_shape"] = ev_best["shape"]
    checks["output_raw_range"] = [float(pred_va.min()), float(pred_va.max())]

    # overfitting indicators: same checkpoint on the training images
    ev_train = evaluate_set(y_tr, predict(best_model, x_tr))
    after = history[history.epoch > ck["epoch"]]
    overfit = {"train_loss_at_best_eval_mode": ev_train["loss"], "val_loss_at_best": ev_best["loss"],
               "val_over_train_loss_ratio": ev_best["loss"] / ev_train["loss"],
               "train_PSNR_at_best": ev_train["PSNR"], "val_PSNR_at_best": ev_best["PSNR"],
               "train_SSIM_at_best": ev_train["SSIM"], "val_SSIM_at_best": ev_best["SSIM"],
               "epochs_after_best": int(len(after)),
               "val_loss_last_minus_best": float(history.val_loss.iloc[-1] - ev_best["loss"]),
               "train_loss_last_epoch": float(history.train_loss.iloc[-1])}

    # ---- 6. comparison table + per-image analysis --------------------------------
    unet_name = f"U-Net DGI-{n_shots} (best epoch {ck['epoch']})"
    all_methods = {**refs, unet_name: ev_best}
    order = [f"DGI-{n_shots} input", "flat image at <B>/<R>", unet_name, f"DGI-{REFERENCE_SHOTS:,} (conventional)"]
    summary = pd.DataFrame([{"method": m, "PSNR": all_methods[m]["PSNR"], "SSIM": all_methods[m]["SSIM"],
                             "MSE": all_methods[m]["MSE"], "Pearson_r": all_methods[m]["Pearson_r"]} for m in order])
    summary.to_csv(out / "summary.csv", index=False)

    p_in, p_flat, p_ref, p_un = (refs[f"DGI-{n_shots} input"]["per_image"], refs["flat image at <B>/<R>"]["per_image"],
                                 refs[f"DGI-{REFERENCE_SHOTS:,} (conventional)"]["per_image"], ev_best["per_image"])
    tag_in, tag_ref = f"DGI{n_shots}", f"DGI{REFERENCE_SHOTS}"
    per = pd.DataFrame({
        "image_id": id_va,
        f"{tag_in}_PSNR": p_in.PSNR, f"{tag_in}_SSIM": p_in.SSIM, f"{tag_in}_MSE": p_in.MSE, f"{tag_in}_r": p_in.Pearson_r,
        "UNet_PSNR": p_un.PSNR, "UNet_SSIM": p_un.SSIM, "UNet_MSE": p_un.MSE, "UNet_r": p_un.Pearson_r,
        f"{tag_ref}_PSNR": p_ref.PSNR, f"{tag_ref}_SSIM": p_ref.SSIM, f"{tag_ref}_MSE": p_ref.MSE, f"{tag_ref}_r": p_ref.Pearson_r,
        "flat_PSNR": p_flat.PSNR, "flat_SSIM": p_flat.SSIM, "flat_MSE": p_flat.MSE})
    per[f"UNet_minus_{tag_in}_PSNR"] = per.UNet_PSNR - per[f"{tag_in}_PSNR"]
    per[f"UNet_minus_{tag_in}_SSIM"] = per.UNet_SSIM - per[f"{tag_in}_SSIM"]
    per[f"UNet_minus_{tag_ref}_PSNR"] = per.UNet_PSNR - per[f"{tag_ref}_PSNR"]
    per[f"UNet_minus_{tag_ref}_SSIM"] = per.UNet_SSIM - per[f"{tag_ref}_SSIM"]
    per[f"UNet_minus_{tag_in}_r"] = per.UNet_r - per[f"{tag_in}_r"]
    per[f"UNet_minus_{tag_ref}_r"] = per.UNet_r - per[f"{tag_ref}_r"]
    per["UNet_minus_flat_PSNR"] = per.UNet_PSNR - per.flat_PSNR
    per["UNet_minus_flat_SSIM"] = per.UNet_SSIM - per.flat_SSIM
    per.to_csv(out / "val_per_image_metrics.csv", index=False)

    def stats(col):
        d = per[col]
        return {"mean": float(d.mean()), "median": float(d.median()), "std": float(d.std()),
                "pct_positive": float((d > 0).mean() * 100), "min": float(d.min()), "max": float(d.max())}
    analysis = {f"UNet_vs_{tag_in}_PSNR": stats(f"UNet_minus_{tag_in}_PSNR"),
                f"UNet_vs_{tag_in}_SSIM": stats(f"UNet_minus_{tag_in}_SSIM"),
                f"UNet_vs_{tag_ref}_PSNR": stats(f"UNet_minus_{tag_ref}_PSNR"),
                f"UNet_vs_{tag_ref}_SSIM": stats(f"UNet_minus_{tag_ref}_SSIM"),
                f"UNet_vs_{tag_in}_r": stats(f"UNet_minus_{tag_in}_r"),
                f"UNet_vs_{tag_ref}_r": stats(f"UNet_minus_{tag_ref}_r"),
                "UNet_vs_flat_PSNR": stats("UNet_minus_flat_PSNR"),
                "UNet_vs_flat_SSIM": stats("UNet_minus_flat_SSIM")}

    # ---- 7. figures (validation only) ------------------------------------------
    srt = per.sort_values(f"{tag_in}_PSNR").reset_index(drop=True)
    picks = [srt.image_id.iloc[int(round(q * (len(srt) - 1)))] for q in (0.1, 0.3, 0.5, 0.7, 0.9)]
    rows = []
    for q, image_id in zip((10, 30, 50, 70, 90), picks):
        j = id_va.index(image_id)
        r = per.iloc[j]
        rows.append({"label": f"{image_id} (val)\ninput-PSNR p{q}", "target": y_va[j, 0], "panels": [
            ("Original (target)", y_va[j, 0]),
            (f"DGI-{n_shots}\n{r[f'{tag_in}_PSNR']:.2f} dB | SSIM {r[f'{tag_in}_SSIM']:.3f}", x_va[j, 0]),
            (f"U-Net DGI-{n_shots}\n{r.UNet_PSNR:.2f} dB | SSIM {r.UNet_SSIM:.3f}", np.clip(pred_va[j, 0], 0, 1)),
            (f"DGI-{REFERENCE_SHOTS:,}\n{r[f'{tag_ref}_PSNR']:.2f} dB | SSIM {r[f'{tag_ref}_SSIM']:.3f}", x_ref[j, 0])]})
    tag = (f"Model {args.model} ({run}): DGI-{n_shots} -> U-Net -> original | best epoch {ck['epoch']} "
           f"of {int(history.epoch.max())}")
    save_panel_grid(rows, out / "reconstruction_examples.png",
                    tag + "\nVALIDATION images, gray window [0, 1] for every panel; rows = input-PSNR p10/p30/p50/p70/p90")
    save_panel_grid(rows, out / "reconstruction_examples_DISPLAY_ONLY.png",
                    tag + "\nDISPLAY ONLY: one window per row (target's 1-99th pct); metrics from [0, 1] images",
                    display_window=True)
    save_loss_curves(history.rename(columns={"val_psnr": "val_PSNR"}), batch_log, out / "loss_curves.png", tag,
                     {k: v["loss"] for k, v in refs.items()})
    save_val_metric_curves(history, out / "metric_curves.png", tag,
                           {k: {m: v[m] for m in ("PSNR", "SSIM", "Pearson_r")} for k, v in refs.items()},
                           int(ck["epoch"]), unet_label=f"U-Net (DGI-{n_shots} input)")

    report = {"run": run, "model": args.model, "n_shots": n_shots, "pattern_seed": 2026,
              "architecture": f"UNetGI(base={args.base}, depth=3): 32/64/128 + 256 bottleneck, linear head",
              "n_params": n_params, "optimizer": f"Adam lr={args.lr} (constant)", "loss": "MSE (raw output vs target)",
              "batch_size": args.batch_size, "max_epochs": args.epochs, "patience": args.patience,
              "epochs_completed": int(history.epoch.max()), "stopped_early": stopped_early,
              "best_epoch": int(ck["epoch"]), "best_val_loss": float(ck["val_loss"]),
              "training_seed": args.seed, "torch": torch.__version__, "wall_time_s": round(time.time() - t_start, 1),
              "normalization": {"target": "linear-light luminance object, [0,1], no scaling",
                                "input": f"O_hat = DGI_{n_shots}/0.25 + <B>/<R>, clipped to [0,1], no scaling",
                                "output": "linear; clipped to [0,1] only for metrics/figures"},
              "checks": checks, "overfitting": overfit, "val_summary": summary.to_dict("records"),
              "per_image_analysis": analysis, "val_example_ids": picks}
    (out / ("training_report.json" if args.full else "sanity_report.json")).write_text(
        json.dumps(report, indent=2, default=float))

    pd.set_option("display.width", 170)
    print("\nvalidation (75 images):\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nper-image analysis:\n" + pd.DataFrame(analysis).T.round(4).to_string())
    print("\noverfitting:", json.dumps(overfit, indent=1, default=float))
    print("\nchecks:", json.dumps(checks, indent=1, default=float))
    print(f"\nepochs {report['epochs_completed']}, best {report['best_epoch']}, "
          f"early stop {stopped_early}, {report['wall_time_s']} s -> {out}")


if __name__ == "__main__":
    main()
