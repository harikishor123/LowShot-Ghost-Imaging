#!/usr/bin/env python3
"""Extension (broader microscopy dataset), step 3: train one U-Net per shot count and target.

    python ext_train.py --target clean  --shots 1000     # AI-original-N: DGI-N -> original clean object
    python ext_train.py --target dgi10k --shots 1000     # AI-N: DGI-N -> DGI-10000 of the same acquisition

Recipe identical to the onion final models (final_train.py / orig_train.py): UNetGI base 32, Adam 1e-4,
MSE, batch 16, max 80 epochs, patience 10 on the selection loss, master seed 0. Each epoch every training
image gets one of its 32 sequences, drawn with default_rng([0, epoch]). Selection: validation images x seeds
2040-2042. Differences, all recorded in protocol.json: the dataset, and training on CUDA with
torch.use_deterministic_algorithms(True). The test split is never loaded.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ext_common import OUT, POOL_NPY, PROTOCOL, SELECTION_NPZ, SHOTS, TARGETS, load_objects, safe_path, write_json  # noqa: E402
from src.gi_unet_data import PairDataset  # noqa: E402
from src.unet_gi import UNetGI, count_parameters  # noqa: E402
from src.visualization import save_loss_curves, save_val_metric_curves  # noqa: E402
from train_unet_gi import evaluate_set, set_determinism  # noqa: E402

KEYS = ("PSNR", "SSIM", "MSE", "Pearson_r")


@torch.no_grad()
def predict(model, x: np.ndarray, device: str, batch: int = 256) -> np.ndarray:
    model.eval()
    return np.concatenate([model(torch.from_numpy(x[i:i + batch]).to(device)).cpu().numpy()
                           for i in range(0, len(x), batch)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, choices=list(TARGETS))
    ap.add_argument("--shots", type=int, required=True, choices=[500, 1000, 2000, 5000])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    assert PROTOCOL.exists(), "protocol.json must exist before training"
    n, si, ti = args.shots, SHOTS.index(args.shots), SHOTS.index(10000)
    label = f"{TARGETS[args.target]}-{n}"
    out = OUT / "models" / f"target_{args.target}" / f"model_{n}"
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    gen = set_determinism(args.seed)
    dev = args.device
    t0 = time.time()

    meta = json.loads((OUT / "pool" / "pool_meta.json").read_text())
    tv = load_objects(("train", "val"))
    train_ids = tv["image_id"][tv["split"] == "train"].tolist()
    test_ids = set(load_objects(("test",))["image_id"].tolist())           # ids only, for the leakage assertion
    assert train_ids == meta["train_ids"] and not set(train_ids) & test_ids
    pool = np.load(POOL_NPY, mmap_mode="r")
    x_all = np.ascontiguousarray(pool[:, :, si])                            # (n_train, 32, 64, 64)
    if args.target == "dgi10k":
        y_all = np.ascontiguousarray(pool[:, :, ti])
    else:
        clean_tr = tv["object"][tv["split"] == "train"].astype(np.float32)  # (n_train, 64, 64)
    n_img, n_real = x_all.shape[:2]
    sel = np.load(SELECTION_NPZ, allow_pickle=False)
    assert not set(sel["image_id"].tolist()) & test_ids
    x_sel = sel["dgi"][:, :, si].reshape(-1, 1, 64, 64)
    if args.target == "dgi10k":
        y_sel = sel["dgi"][:, :, ti].reshape(-1, 1, 64, 64)
    else:
        y_sel = np.repeat(sel["target_clean"][:, None], len(sel["seeds"]), axis=1).reshape(-1, 1, 64, 64).astype(np.float32)
    checks = {"n_train_images": n_img, "realisations_per_image": n_real, "selection_pairs": len(x_sel),
              "test_ids_in_training_or_selection": 0, "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}
    tgt_name = "DGI-10000 of the same acquisition" if args.target == "dgi10k" else "original clean object"
    refs = {f"conventional DGI-{n} (selection)": evaluate_set(y_sel, x_sel)}

    model = UNetGI(base=args.base).to(dev)
    n_params = count_parameters(model)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    def sel_eval(m):
        return evaluate_set(y_sel, predict(m, x_sel, dev))

    ev0 = sel_eval(model)
    history = [{"epoch": 0, "train_loss": np.nan, "val_loss": ev0["loss"], "val_psnr": ev0["PSNR"], "val_ssim": ev0["SSIM"],
                "val_correlation": ev0["Pearson_r"], "learning_rate": args.lr, "seconds": 0.0}]
    batch_log, used_k = [], []
    best, since, stopped = np.inf, 0, False
    for epoch in range(1, args.epochs + 1):
        te = time.time()
        ks = np.random.default_rng([args.seed, epoch]).integers(0, n_real, n_img)   # same rule as onion
        used_k.append(ks)
        x_ep = x_all[np.arange(n_img), ks][:, None].copy()
        y_ep = (y_all[np.arange(n_img), ks] if args.target == "dgi10k" else clean_tr)[:, None].copy()
        loader = DataLoader(PairDataset(x_ep, y_ep, list(range(n_img))), batch_size=args.batch_size, shuffle=True,
                            generator=gen, num_workers=0)
        model.train()
        tot, cnt, losses = torch.zeros((), device=dev), 0, []
        for xb, yb, _ in loader:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.detach())
            tot += loss.detach() * len(xb)
            cnt += len(xb)
        losses = torch.stack(losses).cpu().numpy()
        if not np.isfinite(losses).all():
            raise FloatingPointError(f"non-finite loss epoch {epoch}")
        batch_log += [{"epoch_float": epoch - 1 + (b + 1) / len(losses), "loss": float(v)} for b, v in enumerate(losses)]
        ev = sel_eval(model)
        row = {"epoch": epoch, "train_loss": float(tot) / cnt, "val_loss": ev["loss"], "val_psnr": ev["PSNR"],
               "val_ssim": ev["SSIM"], "val_correlation": ev["Pearson_r"], "learning_rate": opt.param_groups[0]["lr"],
               "seconds": round(time.time() - te, 1)}
        history.append(row)
        ck = {"model": {k: v.cpu() for k, v in model.state_dict().items()}, "optimizer": opt.state_dict(), "epoch": epoch,
              "val_loss": ev["loss"], "config": vars(args) | {"n_params": n_params, "target": tgt_name}}
        torch.save(ck, safe_path(out / "checkpoints" / "last.pt"))
        improved = ev["loss"] < best
        if improved:
            best, since = ev["loss"], 0
            torch.save(ck, safe_path(out / "checkpoints" / "best.pt"))
        else:
            since += 1
        print(f"[{label}] epoch {epoch:3d}: train {row['train_loss']:.5f} | sel {row['val_loss']:.5f} | "
              f"PSNR {row['val_psnr']:.2f} SSIM {row['val_ssim']:.3f} r {row['val_correlation']:.3f}"
              f"{'  *best' if improved else ''} | {row['seconds']}s", flush=True)
        if since >= args.patience:
            stopped = True
            print(f"[{label}] early stopping", flush=True)
            break

    history = pd.DataFrame(history)
    history.to_csv(safe_path(out / "history.csv"), index=False)
    batch_log = pd.DataFrame(batch_log)
    batch_log.to_csv(safe_path(out / "batch_losses.csv"), index=False)
    np.save(safe_path(out / "realisations_used.npy"), np.array(used_k))
    ckb = torch.load(out / "checkpoints" / "best.pt", weights_only=False)
    mb = UNetGI(base=args.base).to(dev)
    mb.load_state_dict(ckb["model"])
    evb = sel_eval(mb)
    rec = history.set_index("epoch").loc[ckb["epoch"]]
    checks["best_reload_max_abs_diff"] = float(max(abs(evb["loss"] - rec.val_loss), abs(evb["PSNR"] - rec.val_psnr),
                                                   abs(evb["SSIM"] - rec.val_ssim)))
    tag = f"{label}: DGI-{n} -> {tgt_name} | best epoch {ckb['epoch']} of {int(history.epoch.max())}"
    save_loss_curves(history, batch_log, safe_path(out / "loss_curves.png"),
                     tag + f"\nselection = {len(sel['image_id'])} val images x seeds 2040-2042",
                     {k: v["loss"] for k, v in refs.items()})
    save_val_metric_curves(history, safe_path(out / "metric_curves.png"), tag,
                           {k: {m: v[m] for m in ("PSNR", "SSIM", "Pearson_r")} for k, v in refs.items()},
                           int(ckb["epoch"]), unet_label=label)
    report = {"label": label, "shots": n, "target": tgt_name, "best_epoch": int(ckb["epoch"]),
              "best_selection_loss": float(ckb["val_loss"]), "epochs_completed": int(history.epoch.max()),
              "stopped_early": stopped, "n_params": n_params,
              "checkpoint": str((out / "checkpoints" / "best.pt").relative_to(ROOT)), "config": vars(args),
              "pool_seed_rule": meta["seed_rule"], "selection_seeds": meta["selection_seeds"],
              "epoch_realisation_rule": f"default_rng([master_seed, epoch]).integers(0, 32, {n_img})",
              "selection_at_best": {k: evb[k] for k in KEYS},
              "conventional_on_selection": {k: refs[f"conventional DGI-{n} (selection)"][k] for k in KEYS},
              "checks": checks, "wall_time_s": round(time.time() - t0, 1)}
    write_json(out / "training_report.json", report)
    print(f"[{label}] best epoch {ckb['epoch']}, selection PSNR {evb['PSNR']:.2f} SSIM {evb['SSIM']:.3f} "
          f"r {evb['Pearson_r']:.3f} (DGI-{n}: {report['conventional_on_selection']['PSNR']:.2f} / "
          f"{report['conventional_on_selection']['SSIM']:.3f}), {report['wall_time_s']} s", flush=True)


if __name__ == "__main__":
    main()
