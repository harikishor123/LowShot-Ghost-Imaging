#!/usr/bin/env python3
"""
Evaluate E4 TransFusionNet against all prior experiments.

Loads E1 (frozen), E2a, E3a, and E4 checkpoints, runs inference on
selection_val.npz, and produces per-image CSV, bootstrap CIs, and
6-panel visual comparisons (DGI, E1, E2a, E3a, E4, GT).

Usage:
  python3 evaluate_e4.py --e4-config configs/e4a_mse.yaml --shot-count 1000
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.dataset import GIDataset, SHOT_IDX
from src.diffusion import GaussianDiffusion
from src.unet_diffusion import DiffusionUNet
from src.unet_gi import UNetGI
from src.transformer_fusion import TransFusionNet
from src.metrics import evaluate, bootstrap_ci


FROZEN_ROOT = "/home/ss_students/Downloads/MTP_Ghost_Imaging/01_UNet_Baseline"

E1_CKPT_TEMPLATE = (
    "outputs/extension_medmnist_microscopy/models/target_clean"
    "/model_{N}/checkpoints/best.pt"
)


def load_e1_model(shot_count: int, device: torch.device) -> UNetGI:
    frozen_root = Path(FROZEN_ROOT)
    ckpt_path = frozen_root / E1_CKPT_TEMPLATE.format(N=shot_count)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"E1 checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = UNetGI(base=cfg.get("base", 32), depth=3).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_e2a_model(shot_count: int, device: torch.device) -> UNetGI:
    ckpt_path = Path("checkpoints") / f"e2a_mse_ssim_N{shot_count}" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"E2a checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    model = UNetGI(base=model_cfg.get("base", 32), depth=model_cfg.get("depth", 3)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_e3a_model(
    shot_count: int, device: torch.device,
) -> tuple[DiffusionUNet, GaussianDiffusion]:
    cfg_path = "configs/e3a_eps_only.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    ckpt_path = Path("checkpoints") / f"e3a_eps_only_N{shot_count}" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"E3a checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = DiffusionUNet(
        in_ch=2, out_ch=1,
        base_ch=cfg["model"]["base_ch"],
        time_dim=cfg["model"]["time_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    diffusion = GaussianDiffusion(
        timesteps=cfg["diffusion"]["timesteps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
        device=device,
    )
    return model, diffusion


def load_e4_model(
    cfg: dict, shot_count: int, device: torch.device,
) -> TransFusionNet:
    # Load frozen encoder first
    e3_cfg_path = cfg["frozen_encoder"]["config"]
    with open(e3_cfg_path) as f:
        e3_cfg = yaml.safe_load(f)

    ckpt_template = cfg["frozen_encoder"]["checkpoint_template"]
    e3_ckpt_path = Path(ckpt_template.format(N=shot_count))
    if not e3_ckpt_path.exists():
        raise FileNotFoundError(f"E3 encoder checkpoint not found: {e3_ckpt_path}")

    encoder = DiffusionUNet(
        in_ch=2, out_ch=1,
        base_ch=e3_cfg["model"]["base_ch"],
        time_dim=e3_cfg["model"]["time_dim"],
    )
    e3_ckpt = torch.load(e3_ckpt_path, map_location="cpu")
    encoder.load_state_dict(e3_ckpt["model"])

    # Build TransFusionNet
    model = TransFusionNet(
        frozen_encoder=encoder,
        embed_dim=cfg["model"]["embed_dim"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        base_ch=cfg["model"]["base_ch"],
    )

    # Load E4 trainable weights
    exp_name = cfg["experiment"]["name"]
    e4_ckpt_path = Path("checkpoints") / f"{exp_name}_N{shot_count}" / "best.pt"
    if not e4_ckpt_path.exists():
        raise FileNotFoundError(f"E4 checkpoint not found: {e4_ckpt_path}")
    e4_ckpt = torch.load(e4_ckpt_path, map_location="cpu")

    model_state = model.state_dict()
    for k, v in e4_ckpt["model"].items():
        model_state[k] = v
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_e1e2_inference(model: UNetGI, dgi_tensor: torch.Tensor,
                       device: torch.device, batch_size: int = 64) -> np.ndarray:
    preds = []
    for i in range(0, len(dgi_tensor), batch_size):
        batch = dgi_tensor[i:i+batch_size].to(device)
        out = model(batch).clamp(0, 1).squeeze(1).cpu().numpy()
        preds.append(out)
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def run_e3_inference(
    model: DiffusionUNet,
    diffusion: GaussianDiffusion,
    dgi_tensor: torch.Tensor,
    device: torch.device,
    ddim_steps: int = 100,
    batch_size: int = 32,
) -> np.ndarray:
    preds = []
    for i in range(0, len(dgi_tensor), batch_size):
        dgi_batch = dgi_tensor[i:i+batch_size].to(device)
        r_hat = diffusion.ddim_sample(model, dgi_batch, num_steps=ddim_steps)
        o_pred = (dgi_batch + r_hat).clamp(0, 1).squeeze(1).cpu().numpy()
        preds.append(o_pred)
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def run_e4_inference(
    model: TransFusionNet,
    dgi_tensor: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    preds = []
    for i in range(0, len(dgi_tensor), batch_size):
        dgi_batch = dgi_tensor[i:i+batch_size].to(device)
        o_pred = model(dgi_batch).clamp(0, 1).squeeze(1).cpu().numpy()
        preds.append(o_pred)
    return np.concatenate(preds, axis=0)


def save_comparison_figures(
    out_dir: Path,
    dgi_imgs: np.ndarray,
    e1_preds: np.ndarray,
    e2a_preds: np.ndarray,
    e3a_preds: np.ndarray,
    e4_preds: np.ndarray,
    targets: np.ndarray,
    image_ids: list,
    e4_label: str = "E4",
    n_figures: int = 10,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib not available — skipping figures")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    n = min(n_figures, len(targets))
    indices = np.linspace(0, len(targets) - 1, n, dtype=int)

    for idx in indices:
        fig, axes = plt.subplots(1, 6, figsize=(24, 4))
        for ax, img, title in zip(
            axes,
            [dgi_imgs[idx], e1_preds[idx], e2a_preds[idx],
             e3a_preds[idx], e4_preds[idx], targets[idx]],
            ["DGI input", "E1 (frozen)", "E2a (MSE+SSIM)",
             "E3a (diffusion)", f"{e4_label} (fusion)", "Ground truth"],
        ):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        iid = image_ids[idx] if idx < len(image_ids) else idx
        fig.suptitle(f"Image {iid}", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(fig_dir / f"compare_{idx:04d}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved {n} figures to {fig_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate E4 vs E1/E2a/E3a")
    parser.add_argument("--e4-config", required=True)
    parser.add_argument("--shot-count", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-figures", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()

    with open(args.e4_config) as f:
        cfg_e4 = yaml.safe_load(f)

    shot_count = args.shot_count
    dev_str = args.device
    if dev_str == "cuda" and not torch.cuda.is_available():
        dev_str = "cpu"
    device = torch.device(dev_str)

    exp_name = cfg_e4["experiment"]["name"]
    print(f"\n=== Evaluation: E4 ({exp_name}) vs E1/E2a/E3a  |  N={shot_count} ===\n")

    out_dir = Path("results") / f"{exp_name}_eval_N{shot_count}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print("Loading models …")
    e1_model = load_e1_model(shot_count, device)
    print("  E1 loaded")

    e2a_model = None
    try:
        e2a_model = load_e2a_model(shot_count, device)
        print("  E2a loaded")
    except FileNotFoundError:
        print("  [warn] E2a checkpoint not found — skipping E2a")

    e3a_model, diffusion = None, None
    try:
        e3a_model, diffusion = load_e3a_model(shot_count, device)
        print("  E3a loaded")
    except FileNotFoundError:
        print("  [warn] E3a checkpoint not found — skipping E3a")

    e4_model = load_e4_model(cfg_e4, shot_count, device)
    print("  E4 loaded")

    # Load val data
    print("Loading validation data …")
    frozen_root = Path(FROZEN_ROOT)
    pool_dir = frozen_root / "outputs" / "extension_medmnist_microscopy" / "pool"
    val_data = np.load(pool_dir / "selection_val.npz")
    dgi_all = val_data["dgi"]
    targets_all = val_data["target_clean"]
    image_ids = val_data["image_id"]

    n_images, n_seeds, _, H, W = dgi_all.shape
    shot_idx = SHOT_IDX[shot_count]

    dgi_flat, target_flat, ids_flat, seeds_flat = [], [], [], []
    for img_i in range(n_images):
        for seed_j in range(n_seeds):
            dgi_flat.append(dgi_all[img_i, seed_j, shot_idx])
            target_flat.append(targets_all[img_i])
            ids_flat.append(str(image_ids[img_i]))
            seeds_flat.append(seed_j)

    dgi_tensor = torch.from_numpy(np.array(dgi_flat)).unsqueeze(1).float()
    target_np = np.array(target_flat, dtype=np.float64)
    dgi_np = dgi_tensor.squeeze(1).numpy().astype(np.float64)

    # Run inference
    print("Running E1 inference …")
    e1_preds = run_e1e2_inference(e1_model, dgi_tensor, device)

    e2a_preds = None
    if e2a_model is not None:
        print("Running E2a inference …")
        e2a_preds = run_e1e2_inference(e2a_model, dgi_tensor, device)

    e3a_preds = None
    if e3a_model is not None:
        print("Running E3a DDIM inference (100 steps) …")
        e3a_preds = run_e3_inference(e3a_model, diffusion, dgi_tensor, device, ddim_steps=100)

    print("Running E4 inference (single forward pass) …")
    e4_preds = run_e4_inference(e4_model, dgi_tensor, device)

    # Per-image CSV
    print("Computing per-image metrics …")
    csv_path = out_dir / "per_image_results.csv"
    fields = ["image_id", "seed", "shot_count", "method",
              "PSNR", "SSIM", "MSE", "Pearson_r", "edge_r"]

    methods_data = [("DGI", dgi_np), ("E1_frozen", e1_preds.astype(np.float64))]
    if e2a_preds is not None:
        methods_data.append(("E2a", e2a_preds.astype(np.float64)))
    if e3a_preds is not None:
        methods_data.append(("E3a_diffusion", e3a_preds.astype(np.float64)))
    methods_data.append((f"E4_{exp_name}", e4_preds.astype(np.float64)))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(len(target_np)):
            ref = target_np[i]
            for method, all_recs in methods_data:
                rec = np.clip(all_recs[i], 0, 1)
                m = evaluate(ref, rec)
                writer.writerow({
                    "image_id": ids_flat[i],
                    "seed": seeds_flat[i],
                    "shot_count": shot_count,
                    "method": method,
                    **{k: f"{v:.6f}" for k, v in m.items()},
                })

    print(f"  CSV: {csv_path}")

    # Bootstrap CIs
    print(f"Computing bootstrap CIs ({args.bootstrap_resamples} resamples) …")
    summary = {}
    for method, all_recs in methods_data:
        recs_clip = np.clip(all_recs, 0, 1)
        ci = bootstrap_ci(target_np, recs_clip,
                          n_resamples=args.bootstrap_resamples,
                          n_per_image=n_seeds)
        summary[method] = {}
        for metric, (mean, lo, hi) in ci.items():
            summary[method][metric] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}

    # Print table
    method_names = [m for m, _ in methods_data]
    print(f"\n{'Method':<20} {'PSNR':>8} {'SSIM':>8} {'Pearson_r':>10} {'edge_r':>8}")
    print("-" * 60)
    for method in method_names:
        s = summary[method]
        print(f"{method:<20} "
              f"{s['PSNR']['mean']:8.2f} "
              f"{s['SSIM']['mean']:8.4f} "
              f"{s['Pearson_r']['mean']:10.4f} "
              f"{s['edge_r']['mean']:8.4f}")
    print()

    for method in method_names:
        s = summary[method]
        print(f"  {method} 95% CI:")
        for metric in ["PSNR", "SSIM", "Pearson_r", "edge_r"]:
            m = s[metric]
            print(f"    {metric:>10}: {m['mean']:.4f}  [{m['ci_lo']:.4f}, {m['ci_hi']:.4f}]")

    # Save summary
    summary_path = out_dir / "summary.json"
    full_summary = {
        "shot_count": shot_count,
        "n_images": n_images,
        "n_seeds": n_seeds,
        "bootstrap_resamples": args.bootstrap_resamples,
        "e4_config": args.e4_config,
        "results": summary,
    }
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2)
    print(f"  Summary: {summary_path}")

    # Visual comparisons (6-panel if all models available)
    if args.save_figures > 0 and e2a_preds is not None and e3a_preds is not None:
        print("Saving visual comparisons …")
        save_comparison_figures(
            out_dir, dgi_np, e1_preds, e2a_preds, e3a_preds, e4_preds,
            target_np, ids_flat, e4_label=f"E4_{exp_name}",
            n_figures=args.save_figures,
        )

    print(f"\n=== Evaluation complete — results in {out_dir}/ ===")


if __name__ == "__main__":
    main()
