#!/usr/bin/env python3
"""
E4: TransFusionNet training — single-pass reconstruction using frozen E3 encoder.

Takes DGI → frozen E3 encoder features → Transformer fusion decoder → O_pred.
No iterative DDIM sampling needed at training or inference time.

Usage:
  cd /home/ss_students/Downloads/MTP_Ghost_Imaging/02_DiffTransFuse
  python3 train_e4.py --config configs/e4a_mse.yaml --shot-count 1000
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

from src.dataset import GIDataset
from src.unet_diffusion import DiffusionUNet
from src.transformer_fusion import TransFusionNet
from src.losses_v2 import ssim_loss
from src.metrics import evaluate_batch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str, overrides: dict) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    for key, val in overrides.items():
        if val is not None:
            parts = key.split(".")
            node = cfg
            for p in parts[:-1]:
                if p not in node:
                    node[p] = {}
                node = node[p]
            node[parts[-1]] = val
    return cfg


def collate_fn(batch):
    dgi = torch.stack([b[0] for b in batch])
    target = torch.stack([b[1] for b in batch])
    return dgi, target


def load_frozen_encoder(cfg: dict, shot_count: int, device: torch.device) -> DiffusionUNet:
    """Load and freeze the E3 encoder checkpoint."""
    e3_cfg_path = cfg["frozen_encoder"]["config"]
    with open(e3_cfg_path) as f:
        e3_cfg = yaml.safe_load(f)

    ckpt_template = cfg["frozen_encoder"]["checkpoint_template"]
    ckpt_path = Path(ckpt_template.format(N=shot_count))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"E3 checkpoint not found: {ckpt_path}")

    encoder = DiffusionUNet(
        in_ch=2, out_ch=1,
        base_ch=e3_cfg["model"]["base_ch"],
        time_dim=e3_cfg["model"]["time_dim"],
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    encoder.load_state_dict(ckpt["model"])
    encoder = encoder.to(device)
    encoder.eval()

    for p in encoder.parameters():
        p.requires_grad = False

    return encoder


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model: TransFusionNet,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict,
) -> dict:
    model.train()
    # Keep frozen encoder in eval mode
    model.encoder.eval()

    total = {"loss": 0.0, "mse": 0.0, "ssim": 0.0}
    n = 0
    lambda_ssim = cfg["loss"]["lambda_ssim"]

    for dgi, target in loader:
        dgi = dgi.to(device)
        target = target.to(device)

        o_pred = model(dgi)

        l_mse = nn.functional.mse_loss(o_pred, target)
        loss = l_mse

        ssim_val = 0.0
        if lambda_ssim > 0:
            o_clamped = o_pred.clamp(0, 1)
            ssim_val = ssim_loss(o_clamped, target.clamp(0, 1))
            loss = loss + lambda_ssim * ssim_val

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        optimiser.step()

        total["loss"] += loss.item()
        total["mse"] += l_mse.item()
        total["ssim"] += (ssim_val.item() if isinstance(ssim_val, torch.Tensor) else ssim_val)
        n += 1

    return {k: v / max(n, 1) for k, v in total.items()}


@torch.no_grad()
def validate(
    model: TransFusionNet,
    loader: DataLoader,
    device: torch.device,
    cfg: dict,
) -> dict:
    model.eval()
    total = {"loss": 0.0, "mse": 0.0, "ssim": 0.0}
    all_refs, all_recs = [], []
    n = 0
    lambda_ssim = cfg["loss"]["lambda_ssim"]

    for dgi, target in loader:
        dgi = dgi.to(device)
        target = target.to(device)

        o_pred = model(dgi)

        l_mse = nn.functional.mse_loss(o_pred, target)
        loss = l_mse

        ssim_val = 0.0
        if lambda_ssim > 0:
            o_clamped = o_pred.clamp(0, 1)
            ssim_val = ssim_loss(o_clamped, target.clamp(0, 1))
            loss = loss + lambda_ssim * ssim_val

        total["loss"] += loss.item()
        total["mse"] += l_mse.item()
        total["ssim"] += (ssim_val.item() if isinstance(ssim_val, torch.Tensor) else ssim_val)
        n += 1

        pred_np = o_pred.clamp(0, 1).squeeze(1).cpu().numpy()
        target_np = target.squeeze(1).cpu().numpy()
        all_refs.append(target_np)
        all_recs.append(pred_np)

    losses = {k: v / max(n, 1) for k, v in total.items()}
    refs = np.concatenate(all_refs, axis=0)
    recs = np.concatenate(all_recs, axis=0)
    img_q = evaluate_batch(refs, recs)
    return {**losses, **img_q}


# ──────────────────────────────────────────────────────────────
# Safety check
# ──────────────────────────────────────────────────────────────

def assert_frozen_untouched(frozen_root: str) -> None:
    manifests = [
        "FREEZE_MANIFEST_phase1_phase2.md5",
        "outputs/extension_medmnist_microscopy/FREEZE_MANIFEST_extension.md5",
        "outputs/extension_medmnist_microscopy_poisson/FREEZE_MANIFEST_poisson.md5",
    ]
    root = Path(frozen_root)
    failed = []

    for m in manifests:
        p = root / m
        if not p.exists():
            print(f"  [warn] manifest not found: {p} — skipping")
            continue
        result = subprocess.run(
            ["md5sum", "--check", "--quiet", str(p)],
            cwd=str(root),
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
    print("  [ok] all frozen manifest checks passed")


# ──────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────

def save_checkpoint(path: Path, model: TransFusionNet, optimiser, epoch, metrics, cfg):
    """Save only trainable parameters (not the frozen encoder)."""
    trainable_state = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith("encoder.")
    }
    torch.save({
        "epoch": epoch,
        "model": trainable_state,
        "optimiser": optimiser.state_dict(),
        "metrics": metrics,
        "config": cfg,
    }, path)


def load_checkpoint(path: Path, model: TransFusionNet, optimiser=None):
    """Load trainable parameters into model (encoder already loaded separately)."""
    ckpt = torch.load(path, map_location="cpu")
    # Load only the trainable parts
    model_state = model.state_dict()
    for k, v in ckpt["model"].items():
        model_state[k] = v
    model.load_state_dict(model_state)
    if optimiser is not None and "optimiser" in ckpt:
        optimiser.load_state_dict(ckpt["optimiser"])
    return ckpt


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train E4: TransFusionNet")
    parser.add_argument("--config", default="configs/e4a_mse.yaml")
    parser.add_argument("--shot-count", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-manifest-check", action="store_true")
    args = parser.parse_args()

    overrides = {
        "training.seed": args.seed,
        "training.lr": args.lr,
        "training.batch_size": args.batch_size,
        "training.max_epochs": args.epochs,
    }
    cfg = load_config(args.config, overrides)

    shot_count = args.shot_count
    if shot_count is None:
        shot_count = cfg.get("data", {}).get("shot_count", 1000)

    seed = cfg["training"]["seed"]
    seed_everything(seed)

    dev_str = args.device or "cuda"
    if dev_str == "cuda" and not torch.cuda.is_available():
        print("  [warn] CUDA not available — using CPU")
        dev_str = "cpu"
    device = torch.device(dev_str)

    exp_name = cfg["experiment"]["name"]
    print(f"\n=== E4 TransFusionNet  |  {exp_name}  |  N={shot_count}  |  device={device} ===\n")

    # Safety check
    frozen_root = "/home/ss_students/Downloads/MTP_Ghost_Imaging/01_UNet_Baseline"
    if not args.no_manifest_check:
        print("Checking frozen manifests …")
        assert_frozen_untouched(frozen_root)

    # Output directory
    run_name = f"{exp_name}_N{shot_count}"
    out_dir = Path("checkpoints") / run_name
    log_dir = Path("logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # Datasets
    print("Loading datasets …")
    train_ds = GIDataset(frozen_root, split="train", shot_count=shot_count, random_k=True, seed=seed)
    val_ds = GIDataset(frozen_root, split="val", shot_count=shot_count, random_k=False)
    print(f"  train: {len(train_ds)} samples  |  val: {len(val_ds)} samples")

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        worker_init_fn=GIDataset.worker_init_fn,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=GIDataset.worker_init_fn,
    )

    # Load frozen E3 encoder
    print("Loading frozen E3 encoder …")
    frozen_encoder = load_frozen_encoder(cfg, shot_count, device)
    e3_params = sum(p.numel() for p in frozen_encoder.parameters())
    print(f"  E3 encoder parameters (frozen): {e3_params:,}")

    # Build TransFusionNet
    model = TransFusionNet(
        frozen_encoder=frozen_encoder,
        embed_dim=cfg["model"]["embed_dim"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        base_ch=cfg["model"]["base_ch"],
    ).to(device)

    n_trainable = model.count_trainable_params()
    n_frozen = model.count_frozen_params()
    print(f"  TransFusionNet trainable params: {n_trainable:,}")
    print(f"  TransFusionNet frozen params:    {n_frozen:,}")
    print(f"  TransFusionNet total params:     {n_trainable + n_frozen:,}")

    # Optimiser — only trainable parameters
    optimiser = torch.optim.AdamW(model.trainable_parameters(), lr=cfg["training"]["lr"])

    # CSV log
    log_path = log_dir / f"{run_name}.csv"
    csv_fields = [
        "epoch", "train_loss", "train_mse", "train_ssim",
        "val_loss", "val_mse", "val_ssim",
        "val_PSNR", "val_SSIM", "val_Pearson_r", "val_edge_r", "elapsed_s",
    ]
    csv_file = open(log_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()

    # Training loop
    best_val_loss = float("inf")
    patience_ctr = 0
    max_epochs = cfg["training"]["max_epochs"]
    patience = cfg["training"]["patience"]

    print(f"\nTraining for up to {max_epochs} epochs (patience={patience}) …\n")
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        t_ep = time.time()
        train_m = train_one_epoch(model, train_loader, optimiser, device, cfg)
        val_m = validate(model, val_loader, device, cfg)

        elapsed = time.time() - t0
        ep_time = time.time() - t_ep

        print(
            f"Epoch {epoch:03d}/{max_epochs:03d} "
            f"| train_loss={train_m['loss']:.4f} "
            f"| val_loss={val_m['loss']:.4f} "
            f"| val_PSNR={val_m.get('PSNR', 0):.2f} "
            f"| val_SSIM={val_m.get('SSIM', 0):.4f} "
            f"| {ep_time:.0f}s"
        )

        row = {
            "epoch": epoch,
            "train_loss": f"{train_m['loss']:.6f}",
            "train_mse": f"{train_m['mse']:.6f}",
            "train_ssim": f"{train_m['ssim']:.6f}",
            "val_loss": f"{val_m['loss']:.6f}",
            "val_mse": f"{val_m['mse']:.6f}",
            "val_ssim": f"{val_m['ssim']:.6f}",
            "val_PSNR": f"{val_m.get('PSNR', 0):.4f}",
            "val_SSIM": f"{val_m.get('SSIM', 0):.4f}",
            "val_Pearson_r": f"{val_m.get('Pearson_r', 0):.4f}",
            "val_edge_r": f"{val_m.get('edge_r', 0):.4f}",
            "elapsed_s": f"{elapsed:.0f}",
        }
        writer.writerow(row)
        csv_file.flush()

        save_checkpoint(out_dir / "last.pt", model, optimiser, epoch, val_m, cfg)
        if val_m["loss"] < best_val_loss - 1e-6:
            best_val_loss = val_m["loss"]
            patience_ctr = 0
            save_checkpoint(out_dir / "best.pt", model, optimiser, epoch, val_m, cfg)
            print(f"  → new best (val_loss={best_val_loss:.6f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    csv_file.close()

    best_ckpt = torch.load(out_dir / "best.pt", map_location="cpu")
    best_metrics = best_ckpt["metrics"]
    summary = {
        "experiment": exp_name,
        "shot_count": shot_count,
        "best_epoch": best_ckpt["epoch"],
        "best_val_loss": best_val_loss,
        "best_PSNR": best_metrics.get("PSNR"),
        "best_SSIM": best_metrics.get("SSIM"),
        "best_Pearson_r": best_metrics.get("Pearson_r"),
        "best_edge_r": best_metrics.get("edge_r"),
        "n_trainable_params": n_trainable,
        "n_frozen_params": n_frozen,
        "config": cfg,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    total_time = time.time() - t0
    print(f"\n=== Training complete ({total_time:.0f}s) ===")
    print(f"  Best epoch : {best_ckpt['epoch']}")
    print(f"  Val PSNR   : {best_metrics.get('PSNR', 0):.4f} dB")
    print(f"  Val SSIM   : {best_metrics.get('SSIM', 0):.4f}")
    print(f"  Val Pearson: {best_metrics.get('Pearson_r', 0):.4f}")
    print(f"  Val edge_r : {best_metrics.get('edge_r', 0):.4f}")
    print(f"  Checkpoint : {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
