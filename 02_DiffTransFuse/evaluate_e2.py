#!/usr/bin/env python3
"""
Evaluate E2 checkpoints against frozen E1 baseline on the validation set.

Loads both E1 (frozen) and E2 (new) checkpoints for a given shot count,
runs inference on selection_val.npz, and produces:
  1. Per-image CSV with metrics for DGI, E1, and E2 methods
  2. Bootstrap 95% CIs (image-level cluster resampling)
  3. Visual comparison grids (DGI input | E1 pred | E2 pred | ground truth)
  4. Summary JSON

Usage:
  python evaluate_e2.py --shot-count 1000
  python evaluate_e2.py --shot-count 500 --e2-config configs/e2a_mse_ssim.yaml
  python evaluate_e2.py --shot-count 1000 --save-figures 20

Compare all shots:
  for N in 500 1000 2000 5000; do python evaluate_e2.py --shot-count $N; done
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# ── Project 2 imports ────────────────────────────────────────
from src.dataset import GIDataset, SHOT_IDX
from src.metrics import evaluate, bootstrap_ci
from src.unet_gi import UNetGI


FROZEN_ROOT_DEFAULT = "/home/ss_students/Downloads/MTP_Ghost_Imaging/01_UNet_Baseline"

# E1 checkpoint layout (frozen project):
#   outputs/extension_medmnist_microscopy/models/target_clean/model_{N}/checkpoints/best.pt
E1_CKPT_TEMPLATE = (
    "outputs/extension_medmnist_microscopy/models/target_clean"
    "/model_{N}/checkpoints/best.pt"
)


def load_e1_model(frozen_root: Path, shot_count: int, device: torch.device) -> UNetGI:
    ckpt_path = frozen_root / E1_CKPT_TEMPLATE.format(N=shot_count)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"E1 checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_e1 = ckpt.get("config", {})
    model = UNetGI(base=cfg_e1.get("base", 32), depth=3).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  E1 loaded: epoch={ckpt.get('epoch', '?')} | {ckpt_path.name}")
    return model


def load_e2_model(ckpt_dir: Path, shot_count: int, device: torch.device,
                  config: dict | None = None, exp_name: str = "e2_improved_unet") -> UNetGI:
    ckpt_path = ckpt_dir / f"{exp_name}_N{shot_count}" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"E2 checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_e2 = ckpt.get("config", config or {})
    model_cfg = cfg_e2.get("model", {})
    model = UNetGI(base=model_cfg.get("base", 32),
                   depth=model_cfg.get("depth", 3)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ep = ckpt.get("epoch", "?")
    mets = ckpt.get("metrics", {})
    print(f"  E2 loaded: epoch={ep} | val_SSIM={mets.get('SSIM', 0):.4f} | {ckpt_path.name}")
    return model


@torch.no_grad()
def run_inference(model: UNetGI, dgi_batch: torch.Tensor, device: torch.device,
                  batch_size: int = 64) -> np.ndarray:
    """Run model on (N, 1, H, W) DGI input, return (N, H, W) predictions in [0,1]."""
    preds = []
    for i in range(0, len(dgi_batch), batch_size):
        batch = dgi_batch[i:i+batch_size].to(device)
        out = model(batch).clamp(0, 1).squeeze(1).cpu().numpy()
        preds.append(out)
    return np.concatenate(preds, axis=0)


def save_comparison_figures(
    out_dir: Path,
    dgi_imgs: np.ndarray,
    e1_preds: np.ndarray,
    e2_preds: np.ndarray,
    targets: np.ndarray,
    image_ids: list,
    n_figures: int = 10,
) -> None:
    """Save side-by-side visual comparison grids (no matplotlib dependency)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib not available — skipping visual comparisons")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    n = min(n_figures, len(targets))
    indices = np.linspace(0, len(targets) - 1, n, dtype=int)

    for idx in indices:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, img, title in zip(
            axes,
            [dgi_imgs[idx], e1_preds[idx], e2_preds[idx], targets[idx]],
            ["DGI input", "E1 (frozen)", "E2 (ours)", "Ground truth"],
        ):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title, fontsize=11)
            ax.axis("off")

        iid = image_ids[idx] if idx < len(image_ids) else idx
        fig.suptitle(f"Image {iid}", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(fig_dir / f"compare_{idx:04d}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved {n} comparison figures to {fig_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate E2 vs E1 baseline")
    parser.add_argument("--shot-count", type=int, default=1000,
                        help="Shot count to evaluate (500/1000/2000/5000)")
    parser.add_argument("--frozen-root", type=str, default=FROZEN_ROOT_DEFAULT)
    parser.add_argument("--e2-config", type=str, default="configs/e2_unet.yaml",
                        help="E2 config (used to find checkpoint_dir)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-figures", type=int, default=10,
                        help="Number of visual comparisons to save (0=none)")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()

    frozen_root = Path(args.frozen_root)
    shot_count  = args.shot_count
    device_str  = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available — using CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    print(f"\n=== Evaluation: E2 vs E1  |  N={shot_count} shots  |  device={device} ===\n")

    # Load E2 config for checkpoint path
    with open(args.e2_config) as f:
        cfg_e2 = yaml.safe_load(f)
    exp_name = cfg_e2["experiment"]["name"]
    ckpt_dir = Path(cfg_e2["output"]["checkpoint_dir"])
    out_dir = Path(cfg_e2["output"]["results_dir"]) / f"{exp_name}_eval_N{shot_count}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print("Loading models …")
    e1_model = load_e1_model(frozen_root, shot_count, device)
    e2_model = load_e2_model(ckpt_dir, shot_count, device, cfg_e2, exp_name=exp_name)

    # Load val data
    print("Loading validation data …")
    pool_dir = frozen_root / "outputs" / "extension_medmnist_microscopy" / "pool"
    val_data = np.load(pool_dir / "selection_val.npz")
    dgi_all    = val_data["dgi"]           # (1000, 3, 5, 64, 64)
    targets_all = val_data["target_clean"] # (1000, 64, 64)
    image_ids   = val_data["image_id"]     # (1000,)

    n_images, n_seeds, _, H, W = dgi_all.shape
    shot_idx = SHOT_IDX[shot_count]
    print(f"  Val set: {n_images} images × {n_seeds} seeds = {n_images * n_seeds} items")

    # Flatten: (1000*3, 1, 64, 64) DGI inputs + (1000*3, 64, 64) targets
    dgi_flat = []
    target_flat = []
    ids_flat = []
    seeds_flat = []
    for img_i in range(n_images):
        for seed_j in range(n_seeds):
            dgi_flat.append(dgi_all[img_i, seed_j, shot_idx])
            target_flat.append(targets_all[img_i])
            ids_flat.append(str(image_ids[img_i]))
            seeds_flat.append(seed_j)

    dgi_tensor = torch.from_numpy(np.array(dgi_flat)).unsqueeze(1).float()  # (3000,1,64,64)
    target_np  = np.array(target_flat, dtype=np.float64)                     # (3000,64,64)
    dgi_np     = dgi_tensor.squeeze(1).numpy().astype(np.float64)            # (3000,64,64)

    # Run inference
    print("Running E1 inference …")
    e1_preds = run_inference(e1_model, dgi_tensor, device)
    print("Running E2 inference …")
    e2_preds = run_inference(e2_model, dgi_tensor, device)

    # Per-image metrics
    print("Computing per-image metrics …")
    csv_path = out_dir / "per_image_results.csv"
    fields = ["image_id", "seed", "shot_count", "method",
              "PSNR", "SSIM", "MSE", "Pearson_r", "edge_r"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i in range(len(target_np)):
            ref = target_np[i]
            for method, rec in [("DGI", dgi_np[i]),
                                 ("E1_frozen", e1_preds[i].astype(np.float64)),
                                 ("E2_ours", e2_preds[i].astype(np.float64))]:
                rec_clip = np.clip(rec, 0, 1)
                m = evaluate(ref, rec_clip)
                writer.writerow({
                    "image_id":   ids_flat[i],
                    "seed":       seeds_flat[i],
                    "shot_count": shot_count,
                    "method":     method,
                    **{k: f"{v:.6f}" for k, v in m.items()},
                })

    print(f"  Per-image CSV: {csv_path}")

    # Aggregate metrics with bootstrap CI (image-level)
    print(f"Computing bootstrap CIs ({args.bootstrap_resamples} resamples, image-level) …")
    summary = {}
    for method, recs in [("DGI", dgi_np),
                          ("E1_frozen", e1_preds.astype(np.float64)),
                          ("E2_ours", e2_preds.astype(np.float64))]:
        recs_clip = np.clip(recs, 0, 1)
        ci = bootstrap_ci(target_np, recs_clip,
                          n_resamples=args.bootstrap_resamples,
                          n_per_image=n_seeds)
        summary[method] = {}
        for metric, (mean, lo, hi) in ci.items():
            summary[method][metric] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}

    # Print summary table
    print(f"\n{'Method':<14} {'PSNR':>8} {'SSIM':>8} {'Pearson_r':>10} {'edge_r':>8}")
    print("-" * 55)
    for method in ["DGI", "E1_frozen", "E2_ours"]:
        s = summary[method]
        print(f"{method:<14} "
              f"{s['PSNR']['mean']:8.2f} "
              f"{s['SSIM']['mean']:8.4f} "
              f"{s['Pearson_r']['mean']:10.4f} "
              f"{s['edge_r']['mean']:8.4f}")
    print()

    for method in ["DGI", "E1_frozen", "E2_ours"]:
        s = summary[method]
        print(f"  {method} 95% CI:")
        for metric in ["PSNR", "SSIM", "Pearson_r", "edge_r"]:
            m = s[metric]
            print(f"    {metric:>10}: {m['mean']:.4f}  [{m['ci_lo']:.4f}, {m['ci_hi']:.4f}]")

    # Save summary
    summary_path = out_dir / "summary.json"
    full_summary = {
        "shot_count":        shot_count,
        "n_images":          n_images,
        "n_seeds":           n_seeds,
        "bootstrap_resamples": args.bootstrap_resamples,
        "e2_config":         args.e2_config,
        "results":           summary,
    }
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2)
    print(f"  Summary JSON: {summary_path}")

    # Visual comparisons
    if args.save_figures > 0:
        print("Saving visual comparisons …")
        save_comparison_figures(
            out_dir, dgi_np, e1_preds, e2_preds, target_np,
            ids_flat, n_figures=args.save_figures,
        )

    print(f"\n=== Evaluation complete — results in {out_dir}/ ===")


if __name__ == "__main__":
    main()
