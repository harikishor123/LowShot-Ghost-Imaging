#!/usr/bin/env python3
"""Phase 2 (original clean object target), step 1: train AI-original-N.

    python orig_train.py --shots 500        # also 1000, 2000, 5000

Identical to Phase 1 (final_train.py) except for the target:
    input  = DGI-N of a random training sequence (Phase 1 pool_train.npy, read-only)
    target = the ORIGINAL CLEAN preprocessed 64x64 object of that training image
Checkpoint selection: 75 validation images x seeds 2040-2042 (Phase 1 selection_val.npz, read-only),
scored against the clean object. The test split is never loaded. Outputs go only to
outputs/phase_final_original_target/models/model_<N>/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.ghost_imaging import load_object  # noqa: E402
from src.gi_unet_data import PairDataset, read_split_ids  # noqa: E402
from src.unet_gi import UNetGI, count_parameters  # noqa: E402
from src.visualization import save_loss_curves, save_val_metric_curves  # noqa: E402
from train_unet_gi import evaluate_set, predict, set_determinism  # noqa: E402

P1 = ROOT / "outputs" / "phase_final_low_shot_to_10k_reference"     # read-only
OUT = ROOT / "outputs" / "phase_final_original_target"
SHOTS = [500, 1000, 2000, 5000, 10000]
KEYS = ("PSNR", "SSIM", "MSE", "Pearson_r")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", type=int, required=True, choices=[500, 1000, 2000, 5000])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    n, si = args.shots, SHOTS.index(args.shots)
    out = OUT / "models" / f"model_{n}"
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    gen = set_determinism(args.seed)
    t0 = time.time()

    ids = read_split_ids()
    meta = json.loads((P1 / "pool_meta.json").read_text())
    train_ids = meta["train_ids"]
    assert train_ids == ids["train"] and not set(train_ids) & set(ids["test"])
    pool = np.load(P1 / "pool_train.npy", mmap_mode="r")
    x_all = np.ascontiguousarray(pool[:, :, si])                                     # (341, 32, 64, 64)
    n_img, n_real = x_all.shape[:2]
    clean_tr = np.stack([load_object(ROOT / "data" / "v2" / "reference" / f"{i}.npz", 64)[0]
                         for i in train_ids]).astype(np.float32)                    # (341, 64, 64)
    sel = np.load(P1 / "selection_val.npz", allow_pickle=False)
    assert not set(sel["image_id"].tolist()) & set(ids["test"])
    x_sel = sel["dgi"][:, :, si].reshape(-1, 1, 64, 64)                              # (225, 1, 64, 64)
    y_sel = np.repeat(sel["target_clean"][:, None], len(sel["seeds"]), axis=1).reshape(-1, 1, 64, 64).astype(np.float32)
    checks = {"n_train_images": n_img, "realisations_per_image": n_real, "selection_pairs": len(x_sel),
              "test_ids_in_training_or_selection": 0}
    refs = {f"conventional DGI-{n} vs clean (selection)": evaluate_set(y_sel, x_sel)}

    model = UNetGI(base=args.base)
    n_params = count_parameters(model)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    def sel_eval(m):
        return evaluate_set(y_sel, predict(m, x_sel))

    ev0 = sel_eval(model)
    history = [{"epoch": 0, "train_loss": np.nan, "val_loss": ev0["loss"], "val_psnr": ev0["PSNR"], "val_ssim": ev0["SSIM"],
                "val_correlation": ev0["Pearson_r"], "learning_rate": args.lr, "seconds": 0.0}]
    batch_log, used_k = [], []
    best, since, stopped = np.inf, 0, False
    for epoch in range(1, args.epochs + 1):
        te = time.time()
        ks = np.random.default_rng([args.seed, epoch]).integers(0, n_real, n_img)   # same rule as Phase 1
        used_k.append(ks)
        x_ep = x_all[np.arange(n_img), ks][:, None].copy()
        y_ep = clean_tr[:, None].copy()                                              # target: clean object
        loader = DataLoader(PairDataset(x_ep, y_ep, list(range(n_img))), batch_size=args.batch_size, shuffle=True,
                            generator=gen, num_workers=0)
        model.train()
        tot, cnt = 0.0, 0
        for b, (xb, yb, _) in enumerate(loader):
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss epoch {epoch}")
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            cnt += len(xb)
            batch_log.append({"epoch_float": epoch - 1 + (b + 1) / len(loader), "loss": loss.item()})
        ev = sel_eval(model)
        row = {"epoch": epoch, "train_loss": tot / cnt, "val_loss": ev["loss"], "val_psnr": ev["PSNR"], "val_ssim": ev["SSIM"],
               "val_correlation": ev["Pearson_r"], "learning_rate": opt.param_groups[0]["lr"], "seconds": round(time.time() - te, 1)}
        history.append(row)
        ck = {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch, "val_loss": ev["loss"],
              "config": vars(args) | {"n_params": n_params, "target": "original clean object"}}
        torch.save(ck, out / "checkpoints" / "last.pt")
        improved = ev["loss"] < best
        if improved:
            best, since = ev["loss"], 0
            torch.save(ck, out / "checkpoints" / "best.pt")
        else:
            since += 1
        print(f"[AI-original-{n}] epoch {epoch:3d}: train {row['train_loss']:.5f} | sel {row['val_loss']:.5f} | "
              f"PSNR {row['val_psnr']:.2f} SSIM {row['val_ssim']:.3f} r {row['val_correlation']:.3f}"
              f"{'  *best' if improved else ''} | {row['seconds']}s", flush=True)
        if since >= args.patience:
            stopped = True
            print(f"[AI-original-{n}] early stopping", flush=True)
            break

    history = pd.DataFrame(history)
    history.to_csv(out / "history.csv", index=False)
    batch_log = pd.DataFrame(batch_log)
    batch_log.to_csv(out / "batch_losses.csv", index=False)
    np.save(out / "realisations_used.npy", np.array(used_k))
    ckb = torch.load(out / "checkpoints" / "best.pt", weights_only=False)
    mb = UNetGI(base=args.base)
    mb.load_state_dict(ckb["model"])
    evb = sel_eval(mb)
    rec = history.set_index("epoch").loc[ckb["epoch"]]
    checks["best_reload_max_abs_diff"] = float(max(abs(evb["loss"] - rec.val_loss), abs(evb["PSNR"] - rec.val_psnr),
                                                   abs(evb["SSIM"] - rec.val_ssim)))
    tag = f"AI-original-{n}: DGI-{n} -> original clean object | best epoch {ckb['epoch']} of {int(history.epoch.max())}"
    save_loss_curves(history, batch_log, out / "loss_curves.png", tag + "\nselection = 75 val images x seeds 2040-2042",
                     {k: v["loss"] for k, v in refs.items()})
    save_val_metric_curves(history, out / "metric_curves.png", tag,
                           {k: {m: v[m] for m in ("PSNR", "SSIM", "Pearson_r")} for k, v in refs.items()},
                           int(ckb["epoch"]), unet_label=f"AI-original-{n}")
    report = {"shots": n, "target": "original clean preprocessed object", "best_epoch": int(ckb["epoch"]),
              "best_selection_loss": float(ckb["val_loss"]), "epochs_completed": int(history.epoch.max()),
              "stopped_early": stopped, "n_params": n_params,
              "checkpoint": str((out / "checkpoints" / "best.pt").relative_to(ROOT)), "config": vars(args),
              "pool_seed_rule": meta["seed_rule"], "selection_seeds": meta["selection_seeds"],
              "epoch_realisation_rule": "default_rng([master_seed, epoch]).integers(0, 32, 341)",
              "selection_at_best": {k: evb[k] for k in KEYS},
              "conventional_on_selection": {k: refs[f"conventional DGI-{n} vs clean (selection)"][k] for k in KEYS},
              "checks": checks, "wall_time_s": round(time.time() - t0, 1)}
    (out / "training_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"[AI-original-{n}] best epoch {ckb['epoch']}, selection vs clean PSNR {evb['PSNR']:.2f} SSIM {evb['SSIM']:.3f} "
          f"r {evb['Pearson_r']:.3f} (DGI-{n}: {report['conventional_on_selection']['PSNR']:.2f} / "
          f"{report['conventional_on_selection']['SSIM']:.3f}), {report['wall_time_s']} s", flush=True)


if __name__ == "__main__":
    main()
