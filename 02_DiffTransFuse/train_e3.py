#!/usr/bin/env python3
"""
E3: Residual Diffusion U-Net training.

Trains a conditional diffusion model on the residual r₀ = O − DGI.
  - Forward process: q(r_t | r₀) = N(√ᾱ_t r₀, (1−ᾱ_t) I)
  - Model predicts noise: ε̂_θ(concat(r_t, DGI), t)
  - Loss: L1(ε̂, ε) [E3a] or L1(ε̂, ε) + λ_ssim · L_SSIM(O_hat, O) [E3b]

Usage:
  cd /home/ss_students/Downloads/MTP_Ghost_Imaging/02_DiffTransFuse
  python3 train_e3.py --config configs/e3a_eps_only.yaml --shot-count 1000
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
from src.diffusion import GaussianDiffusion
from src.unet_diffusion import DiffusionUNet
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


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict,
) -> dict:
    model.train()
    total = {"loss": 0.0, "l1_eps": 0.0, "ssim": 0.0}
    n = 0
    lambda_ssim = cfg["loss"]["lambda_ssim"]

    for dgi, target in loader:
        dgi = dgi.to(device)
        target = target.to(device)
        B = dgi.shape[0]

        # Compute residual r₀ = target - DGI
        r0 = target - dgi

        # Sample random timesteps and noise
        t = torch.randint(0, diffusion.T, (B,), device=device)
        noise = torch.randn_like(r0)

        # Forward diffusion: r_t = √ᾱ_t · r₀ + √(1−ᾱ_t) · ε
        r_t = diffusion.q_sample(r0, t, noise)

        # Model input: concat(r_t, DGI) → [B, 2, 64, 64]
        model_input = torch.cat([r_t, dgi], dim=1)
        eps_pred = model(model_input, t)

        # L1 noise loss
        l1_eps = nn.functional.l1_loss(eps_pred, noise)
        loss = l1_eps

        # Optional SSIM loss on reconstructed image
        ssim_val = 0.0
        if lambda_ssim > 0:
            r_hat_0 = diffusion.predict_x0_from_eps(r_t, t, eps_pred)
            o_hat = (dgi + r_hat_0).clamp(0, 1)
            ssim_val = ssim_loss(o_hat, target.clamp(0, 1))
            loss = loss + lambda_ssim * ssim_val

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        total["loss"] += loss.item()
        total["l1_eps"] += l1_eps.item()
        total["ssim"] += (ssim_val.item() if isinstance(ssim_val, torch.Tensor) else ssim_val)
        n += 1

    return {k: v / max(n, 1) for k, v in total.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    device: torch.device,
    cfg: dict,
) -> dict:
    model.eval()
    total = {"loss": 0.0, "l1_eps": 0.0, "ssim": 0.0}
    all_refs, all_recs = [], []
    n = 0
    lambda_ssim = cfg["loss"]["lambda_ssim"]
    ddim_steps = cfg["inference"]["ddim_steps"]

    for dgi, target in loader:
        dgi = dgi.to(device)
        target = target.to(device)
        B = dgi.shape[0]

        # Compute residual and noise loss on a random timestep
        r0 = target - dgi
        t = torch.randint(0, diffusion.T, (B,), device=device)
        noise = torch.randn_like(r0)
        r_t = diffusion.q_sample(r0, t, noise)
        model_input = torch.cat([r_t, dgi], dim=1)
        eps_pred = model(model_input, t)

        l1_eps = nn.functional.l1_loss(eps_pred, noise)
        loss = l1_eps

        ssim_val = 0.0
        if lambda_ssim > 0:
            r_hat_0 = diffusion.predict_x0_from_eps(r_t, t, eps_pred)
            o_hat = (dgi + r_hat_0).clamp(0, 1)
            ssim_val = ssim_loss(o_hat, target.clamp(0, 1))
            loss = loss + lambda_ssim * ssim_val

        total["loss"] += loss.item()
        total["l1_eps"] += l1_eps.item()
        total["ssim"] += (ssim_val.item() if isinstance(ssim_val, torch.Tensor) else ssim_val)
        n += 1

        # DDIM reconstruction for image-quality metrics
        r_hat = diffusion.ddim_sample(model, dgi, num_steps=ddim_steps)
        o_pred = (dgi + r_hat).clamp(0, 1)

        pred_np = o_pred.squeeze(1).cpu().numpy()
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

def _file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


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

def save_checkpoint(path: Path, model, optimiser, epoch, metrics, cfg):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "metrics": metrics,
        "config": cfg,
    }, path)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train E3: Residual Diffusion U-Net")
    parser.add_argument("--config", default="configs/e3a_eps_only.yaml")
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

    # Device
    dev_str = args.device or "cuda"
    if dev_str == "cuda" and not torch.cuda.is_available():
        print("  [warn] CUDA not available — using CPU")
        dev_str = "cpu"
    device = torch.device(dev_str)

    exp_name = cfg["experiment"]["name"]
    print(f"\n=== E3 Residual Diffusion  |  {exp_name}  |  N={shot_count}  |  device={device} ===\n")

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

    # Model
    model = DiffusionUNet(
        in_ch=2, out_ch=1,
        base_ch=cfg["model"]["base_ch"],
        time_dim=cfg["model"]["time_dim"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  DiffusionUNet parameters: {n_params:,}")

    # Diffusion
    diffusion = GaussianDiffusion(
        timesteps=cfg["diffusion"]["timesteps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
        device=device,
    )

    # Optimiser
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])

    # CSV log
    log_path = log_dir / f"{run_name}.csv"
    csv_fields = [
        "epoch", "train_loss", "train_l1_eps", "train_ssim",
        "val_loss", "val_l1_eps", "val_ssim",
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
        train_m = train_one_epoch(model, diffusion, train_loader, optimiser, device, cfg)
        val_m = validate(model, diffusion, val_loader, device, cfg)

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
            "train_l1_eps": f"{train_m['l1_eps']:.6f}",
            "train_ssim": f"{train_m['ssim']:.6f}",
            "val_loss": f"{val_m['loss']:.6f}",
            "val_l1_eps": f"{val_m['l1_eps']:.6f}",
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
        "n_params": n_params,
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
