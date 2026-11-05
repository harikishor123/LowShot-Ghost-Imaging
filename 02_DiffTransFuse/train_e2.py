#!/usr/bin/env python3
"""
E2: Improved U-Net training — MSE + SSIM + edge loss.

This is the first experiment in ghost_imaging_v2.  It uses the SAME architecture
as the frozen E1 baseline (UNetGI, base=32, ~1.9M params) but adds two new losses:
  L_SSIM — penalises blurry reconstructions
  L_edge — preserves cell boundaries (Sobel-based)

The frozen project is NEVER modified.  New checkpoints go into:
  ghost_imaging_v2/checkpoints/e2_N{shot_count}/best.pt
  ghost_imaging_v2/checkpoints/e2_N{shot_count}/last.pt
  ghost_imaging_v2/checkpoints/e2_N{shot_count}/train_log.csv

Usage:
  cd /home/ss_students/Downloads/MTP_Ghost_Imaging/02_DiffTransFuse
  python train_e2.py --shot-count 1000
  python train_e2.py --shot-count 500 --epochs 100 --lr 5e-5

Compare all shot counts:
  for N in 500 1000 2000 5000; do
      python train_e2.py --shot-count $N
  done
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# ── Project 2 imports only ───────────────────────────────────
from src.dataset  import GIDataset
from src.losses_v2 import combined_loss
from src.metrics  import evaluate_batch
from src.unet_gi  import UNetGI, count_parameters


# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ──────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────

def load_config(config_path: str, overrides: dict) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Apply CLI overrides
    for key, val in overrides.items():
        if val is not None:
            # Navigate nested keys: "training.lr" → cfg['training']['lr']
            parts = key.split(".")
            node  = cfg
            for p in parts[:-1]:
                node = node[p]
            node[parts[-1]] = val
    return cfg


# ──────────────────────────────────────────────────────────────
# Data collation (strip meta dict for DataLoader)
# ──────────────────────────────────────────────────────────────

def collate_fn(batch):
    dgi     = torch.stack([b[0] for b in batch])
    target  = torch.stack([b[1] for b in batch])
    return dgi, target   # meta dropped — not needed for E2


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict,
) -> dict:
    model.train()
    total = {"loss": 0.0, "mse": 0.0, "ssim": 0.0, "edge": 0.0}
    n = 0
    for dgi, target in loader:
        dgi    = dgi.to(device)
        target = target.to(device)
        pred   = model(dgi)
        loss, comps = combined_loss(
            pred, target,
            lambda_ssim=cfg["loss"]["lambda_ssim"],
            lambda_edge=cfg["loss"]["lambda_edge"],
        )
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        total["loss"] += loss.item()
        total["mse"]  += comps["mse"]
        total["ssim"] += comps["ssim"]
        total["edge"] += comps["edge"]
        n += 1
    return {k: v / max(n, 1) for k, v in total.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: dict,
) -> dict:
    model.eval()
    total = {"loss": 0.0, "mse": 0.0, "ssim": 0.0, "edge": 0.0}
    all_refs, all_recs = [], []
    n = 0
    for dgi, target in loader:
        dgi    = dgi.to(device)
        target = target.to(device)
        pred   = model(dgi)
        loss, comps = combined_loss(
            pred, target,
            lambda_ssim=cfg["loss"]["lambda_ssim"],
            lambda_edge=cfg["loss"]["lambda_edge"],
        )
        total["loss"] += loss.item()
        total["mse"]  += comps["mse"]
        total["ssim"] += comps["ssim"]
        total["edge"] += comps["edge"]
        n += 1
        # Collect for image-quality metrics
        pred_np   = pred.clamp(0, 1).squeeze(1).cpu().numpy()
        target_np = target.squeeze(1).cpu().numpy()
        all_refs.append(target_np)
        all_recs.append(pred_np)

    losses = {k: v / max(n, 1) for k, v in total.items()}
    refs   = np.concatenate(all_refs, axis=0)
    recs   = np.concatenate(all_recs, axis=0)
    img_q  = evaluate_batch(refs, recs)
    return {**losses, **img_q}


# ──────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────

def save_checkpoint(path: Path, model, optimiser, epoch, metrics, cfg):
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "metrics":   metrics,
        "config":    cfg,
    }, path)


# ──────────────────────────────────────────────────────────────
# Safety check: frozen project must not be touched
# ──────────────────────────────────────────────────────────────

def _file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def assert_frozen_untouched(frozen_root: str) -> None:
    """
    B1 fix: use subprocess.run(..., cwd=frozen_root) so relative paths in
    manifest files resolve against the frozen project root, not the cwd of
    this training script.  os.system() does not set cwd and silently checks
    the wrong paths.

    Also verifies that our copied src/unet_gi.py is byte-for-byte identical
    to the frozen original — detecting accidental edits to either copy.
    """
    manifests = [
        "FREEZE_MANIFEST_phase1_phase2.md5",
        "outputs/extension_medmnist_microscopy/FREEZE_MANIFEST_extension.md5",
        "outputs/extension_medmnist_microscopy_poisson/FREEZE_MANIFEST_poisson.md5",
    ]
    root   = Path(frozen_root)
    failed = []

    for m in manifests:
        p = root / m
        if not p.exists():
            print(f"  [warn] manifest not found: {p} — skipping")
            continue
        result = subprocess.run(
            ["md5sum", "--check", "--quiet", str(p)],
            cwd=str(root),          # critical: manifests use relative paths
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            failed.append(str(p))
            print(f"  [FAIL] {p}")
            if result.stdout.strip():
                print(result.stdout.strip())
            if result.stderr.strip():
                print(result.stderr.strip())
        else:
            print(f"  [ok] {m}")

    if failed:
        raise RuntimeError(
            f"FROZEN MANIFEST CHECK FAILED for {len(failed)} manifest(s).\n"
            "A frozen file was modified.  Abort."
        )

    # Verify our unet_gi.py is a verbatim copy of the frozen original
    here_v2 = Path(__file__).parent / "src" / "unet_gi.py"
    frozen_unet = root / "src" / "unet_gi.py"
    if here_v2.exists() and frozen_unet.exists():
        h_v2     = _file_md5(here_v2)
        h_frozen = _file_md5(frozen_unet)
        if h_v2 != h_frozen:
            raise RuntimeError(
                f"ghost_imaging_v2/src/unet_gi.py has diverged from frozen original!\n"
                f"  v2 md5:     {h_v2}\n"
                f"  frozen md5: {h_frozen}\n"
                "Restore from the frozen project before continuing."
            )
        print(f"  [ok] src/unet_gi.py matches frozen original (md5={h_v2[:12]}…)")

    print("  [ok] all frozen manifest checks passed")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train E2: Improved U-Net")
    parser.add_argument("--config",      default="configs/e2_unet.yaml")
    parser.add_argument("--shot-count",  type=int,   default=None,
                        help="Override shot_count in config (500/1000/2000/5000)")
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--batch-size",  type=int,   default=None)
    parser.add_argument("--seed",        type=int,   default=None)
    parser.add_argument("--device",      type=str,   default=None)
    parser.add_argument("--no-manifest-check", action="store_true",
                        help="Skip frozen manifest md5 check (not recommended)")
    args = parser.parse_args()

    # Load config with CLI overrides
    overrides = {
        "data.shot_count":    args.shot_count,
        "training.epochs":    args.epochs,
        "training.lr":        args.lr,
        "training.batch_size":args.batch_size,
        "training.seed":      args.seed,
        "training.device":    args.device,
    }
    cfg = load_config(args.config, overrides)

    shot_count = cfg["data"]["shot_count"]
    seed       = cfg["training"]["seed"]
    seed_everything(seed)

    # Device
    dev_str = cfg["training"]["device"]
    if dev_str == "cuda" and not torch.cuda.is_available():
        print("  [warn] CUDA requested but not available — using CPU")
        dev_str = "cpu"
    device = torch.device(dev_str)
    print(f"\n=== E2 Improved U-Net  |  N={shot_count} shots  |  device={device} ===\n")

    # Safety check
    if not args.no_manifest_check:
        print("Checking frozen manifests …")
        assert_frozen_untouched(cfg["data"]["frozen_root"])

    # Output directory — derived from config experiment name for ablation separation
    exp_name  = cfg["experiment"]["name"]
    run_name  = f"{exp_name}_N{shot_count}"
    out_dir   = Path(cfg["output"]["checkpoint_dir"]) / run_name
    log_dir   = Path(cfg["output"]["log_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save full config for reproducibility
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)
    print(f"Config saved to {out_dir / 'config.yaml'}")

    # Datasets
    print("Loading datasets …")
    train_ds = GIDataset(
        cfg["data"]["frozen_root"],
        split="train",
        shot_count=shot_count,
        random_k=cfg["data"]["random_k"],
        seed=seed,
    )
    val_ds = GIDataset(
        cfg["data"]["frozen_root"],
        split="val",
        shot_count=shot_count,
        random_k=False,
    )
    print(f"  train: {len(train_ds)} samples  |  val: {len(val_ds)} samples")

    # B3 fix: generator seeds shuffle order; worker_init_fn seeds each worker's
    # numpy RNG so that torch.randint calls inside __getitem__ are independent.
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        worker_init_fn=GIDataset.worker_init_fn,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=GIDataset.worker_init_fn,
    )

    # Model
    model = UNetGI(
        base=cfg["model"]["base"],
        depth=cfg["model"]["depth"],
    ).to(device)
    print(f"  UNetGI  parameters: {count_parameters(model):,}")

    # Optimiser (same as frozen baseline: Adam 1e-4)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    # CSV log
    log_path = log_dir / f"{run_name}.csv"
    csv_fields = [
        "epoch", "train_loss", "train_mse", "train_ssim", "train_edge",
        "val_loss", "val_mse", "val_ssim", "val_edge",
        "val_PSNR", "val_SSIM", "val_Pearson_r", "val_edge_r", "elapsed_s",
    ]
    csv_file = open(log_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()

    # Training loop
    best_val_loss = float("inf")
    patience_ctr  = 0
    max_epochs    = cfg["training"]["epochs"]
    patience      = cfg["training"]["patience"]
    min_delta     = cfg["training"]["min_delta"]

    print(f"\nTraining for up to {max_epochs} epochs (patience={patience}) …\n")
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        t_ep = time.time()
        train_m = train_one_epoch(model, train_loader, optimiser, device, cfg)
        val_m   = validate(model, val_loader, device, cfg)

        elapsed  = time.time() - t0
        ep_time  = time.time() - t_ep

        print(
            f"Epoch {epoch:03d}/{max_epochs:03d} "
            f"| train_loss={train_m['loss']:.4f} "
            f"| val_loss={val_m['loss']:.4f} "
            f"| val_SSIM={val_m.get('SSIM', 0):.4f} "
            f"| val_edge_r={val_m.get('edge_r', 0):.4f} "
            f"| {ep_time:.0f}s"
        )

        # CSV
        row = {
            "epoch":        epoch,
            "train_loss":   f"{train_m['loss']:.6f}",
            "train_mse":    f"{train_m['mse']:.6f}",
            "train_ssim":   f"{train_m['ssim']:.6f}",
            "train_edge":   f"{train_m['edge']:.6f}",
            "val_loss":     f"{val_m['loss']:.6f}",
            "val_mse":      f"{val_m['mse']:.6f}",
            "val_ssim":     f"{val_m['ssim']:.6f}",
            "val_edge":     f"{val_m['edge']:.6f}",
            "val_PSNR":     f"{val_m.get('PSNR', 0):.4f}",
            "val_SSIM":     f"{val_m.get('SSIM', 0):.4f}",
            "val_Pearson_r":f"{val_m.get('Pearson_r', 0):.4f}",
            "val_edge_r":   f"{val_m.get('edge_r', 0):.4f}",
            "elapsed_s":    f"{elapsed:.0f}",
        }
        writer.writerow(row)
        csv_file.flush()

        # Checkpoints
        save_checkpoint(out_dir / "last.pt", model, optimiser, epoch, val_m, cfg)
        if val_m["loss"] < best_val_loss - min_delta:
            best_val_loss = val_m["loss"]
            patience_ctr  = 0
            save_checkpoint(out_dir / "best.pt", model, optimiser, epoch, val_m, cfg)
            print(f"  → new best checkpoint saved (val_loss={best_val_loss:.6f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    csv_file.close()

    # Final summary
    best_ckpt = torch.load(out_dir / "best.pt", map_location="cpu")
    best_metrics = best_ckpt["metrics"]
    summary = {
        "experiment":  exp_name,
        "shot_count":  shot_count,
        "best_epoch":  best_ckpt["epoch"],
        "best_val_loss": best_val_loss,
        "best_PSNR":   best_metrics.get("PSNR"),
        "best_SSIM":   best_metrics.get("SSIM"),
        "best_Pearson_r": best_metrics.get("Pearson_r"),
        "best_edge_r": best_metrics.get("edge_r"),
        "config":      cfg,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Training complete ===")
    print(f"  Best epoch : {best_ckpt['epoch']}")
    print(f"  Val PSNR   : {best_metrics.get('PSNR', 0):.4f} dB")
    print(f"  Val SSIM   : {best_metrics.get('SSIM', 0):.4f}")
    print(f"  Val Pearson: {best_metrics.get('Pearson_r', 0):.4f}")
    print(f"  Val edge_r : {best_metrics.get('edge_r', 0):.4f}")
    print(f"  Checkpoint : {out_dir / 'best.pt'}")
    print(f"  Log        : {log_path}")


if __name__ == "__main__":
    main()
