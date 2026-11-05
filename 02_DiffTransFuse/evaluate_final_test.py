#!/usr/bin/env python3
"""
Final held-out test evaluation for the Ghost Imaging MTP.

Evaluates E1, E2a, E3a, E4a, E4b on the 2000 held-out test images from
objects_64.npz at N = {500, 1000, 2000, 5000}.

DGI is generated on-the-fly using the same physics pipeline as pool generation
(Bernoulli(0.5) patterns, noise-free buckets, calibrated DGI, clipped [0,1]).

Outputs (all under reports/final_test_evaluation/):
  - FINAL_TEST_REPORT.md
  - test_metrics.csv           per-image, per-seed metrics
  - test_summary.csv           aggregate metrics
  - paired_bootstrap.csv       paired CIs for E4a-E1, E4b-E1
  - latency_benchmark.csv      inference latency
  - checkpoint_hashes.txt      SHA256 of all checkpoints
  - test_evaluation_config.yaml
  - figures/                   reconstruction examples
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import hashlib
from pathlib import Path

import numpy as np
import torch
import yaml

# ── Project paths ───────────────────────────────────────────────
FROZEN_ROOT = Path("/home/ss_students/Downloads/MTP_Ghost_Imaging/01_UNet_Baseline")
GI2_ROOT = Path("/home/ss_students/Downloads/MTP_Ghost_Imaging/02_DiffTransFuse")
OUT_DIR = GI2_ROOT / "reports" / "final_test_evaluation"

# Add ghost_imaging_v2 to path for imports
sys.path.insert(0, str(GI2_ROOT))

from src.unet_gi import UNetGI
from src.unet_diffusion import DiffusionUNet
from src.diffusion import GaussianDiffusion
from src.transformer_fusion import TransFusionNet
from src.metrics import evaluate, bootstrap_ci

# DGI generation — reimplemented from frozen project's src/ghost_imaging.py
# and src/reconstruction.py (pure numpy, no cross-imports needed).

def pattern_batches(n_patterns: int, shape: tuple, seed: int,
                    batch: int, p_on: float = 0.5):
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_patterns:
        b = min(batch, n_patterns - done)
        yield (rng.random((b, *shape)) < p_on).astype(np.uint8)
        done += b


def pattern_variance(p_on: float = 0.5) -> float:
    return p_on * (1.0 - p_on)


class StreamingDGI:
    """Streaming DGI accumulator (from frozen project's reconstruction.py)."""
    def __init__(self, objects: np.ndarray):
        self.obj = objects.reshape(objects.shape[0], -1).astype(np.float64)
        m, d = self.obj.shape
        self.n = 0
        self.sum_b = np.zeros(m)
        self.sum_r = 0.0
        self.sum_p = np.zeros(d)
        self.sum_bp = np.zeros((m, d))
        self.sum_rp = np.zeros(d)

    def update(self, pattern_batch: np.ndarray) -> np.ndarray:
        p = pattern_batch.reshape(pattern_batch.shape[0], -1).astype(np.float64)
        b = p @ self.obj.T
        r = p.sum(axis=1)
        self.n += p.shape[0]
        self.sum_b += b.sum(axis=0)
        self.sum_r += r.sum()
        self.sum_p += p.sum(axis=0)
        self.sum_bp += b.T @ p
        self.sum_rp += r @ p
        return b

    def estimate(self, pvar: float) -> tuple[np.ndarray, np.ndarray]:
        n = self.n
        mean_b = self.sum_b / n
        mean_r = self.sum_r / n
        mean_p = self.sum_p / n
        k = mean_b / mean_r
        cov_bp = self.sum_bp / n - mean_b[:, None] * mean_p[None, :]
        cov_rp = self.sum_rp / n - mean_r * mean_p
        dgi = cov_bp - k[:, None] * cov_rp[None, :]
        return dgi / pvar + k[:, None], k

# ── Constants ───────────────────────────────────────────────────
SHOTS = [500, 1000, 2000, 5000, 10000]
SHOT_IDX = {s: i for i, s in enumerate(SHOTS)}
EVAL_SHOTS = [500, 1000, 2000, 5000]
TEST_SEEDS = [3040, 3041, 3042]
N_SEEDS = len(TEST_SEEDS)
BOOTSTRAP_RESAMPLES = 2000

E1_CKPT_TEMPLATE = (
    "outputs/extension_medmnist_microscopy/models/target_clean"
    "/model_{N}/checkpoints/best.pt"
)

METHODS = ["DGI", "E1", "E2a", "E3a", "E4a", "E4b"]


# ── DGI generation ──────────────────────────────────────────────

def generate_test_dgi(objects: np.ndarray, seed: int) -> np.ndarray:
    """Generate calibrated DGI for all objects at all SHOTS, one seed.

    Returns: (n_objects, len(SHOTS), 64, 64) float32, clipped [0,1].
    Uses the identical pipeline as ext_build_pool.py / final_build_pool.py.
    """
    stream = StreamingDGI(objects)
    out = np.empty((len(objects), len(SHOTS), 64, 64), np.float32)
    pvar = pattern_variance(0.5)
    for pats in pattern_batches(max(SHOTS), objects.shape[1:], seed, 500, 0.5):
        stream.update(pats)
        if stream.n in SHOTS:
            est, _ = stream.estimate(pvar)
            out[:, SHOTS.index(stream.n)] = np.clip(est, 0, 1).reshape(-1, 64, 64)
    return out


# ── Model loading ───────────────────────────────────────────────

def load_e1(shot_count: int, device: torch.device) -> UNetGI:
    ckpt_path = FROZEN_ROOT / E1_CKPT_TEMPLATE.format(N=shot_count)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = UNetGI(base=cfg.get("base", 32), depth=3).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_e2a(shot_count: int, device: torch.device) -> UNetGI:
    ckpt_path = GI2_ROOT / "checkpoints" / f"e2a_mse_ssim_N{shot_count}" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    model = UNetGI(base=model_cfg.get("base", 32), depth=model_cfg.get("depth", 3)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_e3a(shot_count: int, device: torch.device) -> tuple[DiffusionUNet, GaussianDiffusion]:
    with open(GI2_ROOT / "configs" / "e3a_eps_only.yaml") as f:
        cfg = yaml.safe_load(f)
    ckpt_path = GI2_ROOT / "checkpoints" / f"e3a_eps_only_N{shot_count}" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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


def load_e4(config_path: str, shot_count: int, device: torch.device) -> TransFusionNet:
    with open(GI2_ROOT / config_path) as f:
        cfg = yaml.safe_load(f)
    e3_cfg_path = GI2_ROOT / cfg["frozen_encoder"]["config"]
    with open(e3_cfg_path) as f:
        e3_cfg = yaml.safe_load(f)
    ckpt_template = cfg["frozen_encoder"]["checkpoint_template"]
    e3_ckpt_path = GI2_ROOT / Path(ckpt_template.format(N=shot_count))
    encoder = DiffusionUNet(
        in_ch=2, out_ch=1,
        base_ch=e3_cfg["model"]["base_ch"],
        time_dim=e3_cfg["model"]["time_dim"],
    )
    e3_ckpt = torch.load(e3_ckpt_path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(e3_ckpt["model"])
    model = TransFusionNet(
        frozen_encoder=encoder,
        embed_dim=cfg["model"]["embed_dim"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        base_ch=cfg["model"]["base_ch"],
    )
    exp_name = cfg["experiment"]["name"]
    e4_ckpt_path = GI2_ROOT / "checkpoints" / f"{exp_name}_N{shot_count}" / "best.pt"
    e4_ckpt = torch.load(e4_ckpt_path, map_location="cpu", weights_only=False)
    model_state = model.state_dict()
    for k, v in e4_ckpt["model"].items():
        model_state[k] = v
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
    return model


# ── Inference ───────────────────────────────────────────────────

@torch.no_grad()
def run_unet_inference(model: UNetGI, dgi_tensor: torch.Tensor,
                       device: torch.device, batch_size: int = 64) -> np.ndarray:
    preds = []
    for i in range(0, len(dgi_tensor), batch_size):
        batch = dgi_tensor[i:i+batch_size].to(device)
        out = model(batch).clamp(0, 1).squeeze(1).cpu().numpy()
        preds.append(out)
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def run_e3_inference(model: DiffusionUNet, diffusion: GaussianDiffusion,
                     dgi_tensor: torch.Tensor, device: torch.device,
                     ddim_steps: int = 100, batch_size: int = 32) -> np.ndarray:
    preds = []
    for i in range(0, len(dgi_tensor), batch_size):
        dgi_batch = dgi_tensor[i:i+batch_size].to(device)
        r_hat = diffusion.ddim_sample(model, dgi_batch, num_steps=ddim_steps)
        o_pred = (dgi_batch + r_hat).clamp(0, 1).squeeze(1).cpu().numpy()
        preds.append(o_pred)
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def run_e4_inference(model: TransFusionNet, dgi_tensor: torch.Tensor,
                     device: torch.device, batch_size: int = 64) -> np.ndarray:
    preds = []
    for i in range(0, len(dgi_tensor), batch_size):
        dgi_batch = dgi_tensor[i:i+batch_size].to(device)
        o_pred = model(dgi_batch).clamp(0, 1).squeeze(1).cpu().numpy()
        preds.append(o_pred)
    return np.concatenate(preds, axis=0)


# ── Latency benchmark ──────────────────────────────────────────

@torch.no_grad()
def benchmark_latency(device: torch.device, shot_count: int = 1000,
                      n_warmup: int = 5, n_reps: int = 20) -> dict:
    dgi_single = torch.randn(1, 1, 64, 64, device=device)

    e1 = load_e1(shot_count, device)
    e3_model, diffusion = load_e3a(shot_count, device)
    e4a = load_e4("configs/e4a_mse.yaml", shot_count, device)
    e4b = load_e4("configs/e4b_mse_ssim.yaml", shot_count, device)

    results = {}
    for name, run_fn in [
        ("E1", lambda: e1(dgi_single).clamp(0, 1)),
        ("E3a_DDIM100", lambda: (dgi_single + diffusion.ddim_sample(
            e3_model, dgi_single, num_steps=100)).clamp(0, 1)),
        ("E4a", lambda: e4a(dgi_single).clamp(0, 1)),
        ("E4b", lambda: e4b(dgi_single).clamp(0, 1)),
    ]:
        for _ in range(n_warmup):
            run_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(n_reps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_fn()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        times_ms = [t * 1000 for t in times]
        results[name] = {
            "mean_ms": float(np.mean(times_ms)),
            "std_ms": float(np.std(times_ms)),
            "min_ms": float(np.min(times_ms)),
            "max_ms": float(np.max(times_ms)),
        }

    del e1, e3_model, diffusion, e4a, e4b
    torch.cuda.empty_cache()
    return results


# ── Paired bootstrap CI ────────────────────────────────────────

def paired_bootstrap_ci(
    refs: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    n_per_image: int = 1,
    n_resamples: int = 2000,
    seed: int = 0,
    metrics: tuple = ("PSNR", "SSIM"),
) -> dict:
    """Paired bootstrap CI for the difference (method_a - method_b).

    Returns {metric: {"mean_diff": ..., "ci_lo": ..., "ci_hi": ..., "significant": bool}}
    """
    rng = np.random.default_rng(seed)
    N = len(refs)
    assert N % n_per_image == 0
    n_images = N // n_per_image

    scores_a = [evaluate(refs[i], np.clip(preds_a[i], 0, 1)) for i in range(N)]
    scores_b = [evaluate(refs[i], np.clip(preds_b[i], 0, 1)) for i in range(N)]

    results = {}
    for metric in metrics:
        vals_a = np.array([s[metric] for s in scores_a])
        vals_b = np.array([s[metric] for s in scores_b])
        diffs = vals_a - vals_b
        mean_diff = float(diffs.mean())

        boot_diffs = np.empty(n_resamples)
        for b in range(n_resamples):
            img_idx = rng.integers(0, n_images, n_images)
            item_idx = (img_idx[:, None] * n_per_image
                        + np.arange(n_per_image)[None, :]).ravel()
            boot_diffs[b] = diffs[item_idx].mean()

        ci_lo = float(np.percentile(boot_diffs, 2.5))
        ci_hi = float(np.percentile(boot_diffs, 97.5))
        significant = (ci_lo > 0) or (ci_hi < 0)

        results[metric] = {
            "mean_diff": mean_diff,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "significant": significant,
        }
    return results


# ── Main evaluation ─────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load test objects ────────────────────────────────────
    print("\n=== Loading test objects ===")
    objects_path = FROZEN_ROOT / "data" / "extension_medmnist" / "processed" / "objects_64.npz"
    z = np.load(objects_path, allow_pickle=False)
    all_split = z["split"].astype(str)
    all_ids = z["image_id"].astype(str)
    all_objects = z["object"]

    test_mask = all_split == "test"
    test_ids = all_ids[test_mask]
    test_objects = all_objects[test_mask].astype(np.float64)
    n_test = len(test_ids)
    print(f"  Test images: {n_test}")
    print(f"  First 5 IDs: {test_ids[:5].tolist()}")

    # Verify no overlap with train/val
    train_ids_set = set(all_ids[all_split == "train"])
    val_ids_set = set(all_ids[all_split == "val"])
    test_id_set = set(test_ids)
    del z, all_split, all_ids, all_objects
    assert len(test_id_set & train_ids_set) == 0, "Test/train overlap!"
    assert len(test_id_set & val_ids_set) == 0, "Test/val overlap!"
    print(f"  No overlap with train ({len(train_ids_set)}) or val ({len(val_ids_set)}) confirmed")

    # ── 2. Generate test DGI ────────────────────────────────────
    print("\n=== Generating test DGI ===")
    print(f"  Seeds: {TEST_SEEDS}")
    print(f"  Generating {n_test} images × {N_SEEDS} seeds × {len(SHOTS)} shots ...")
    t0 = time.time()
    # dgi_all: (n_test, n_seeds, len(SHOTS), 64, 64)
    dgi_all = np.stack(
        [generate_test_dgi(test_objects, seed) for seed in TEST_SEEDS],
        axis=1,
    )
    print(f"  Shape: {dgi_all.shape}, took {time.time()-t0:.1f}s")
    print(f"  Range: [{dgi_all.min():.4f}, {dgi_all.max():.4f}], mean={dgi_all.mean():.4f}")

    # ── 3. Per-shot evaluation ──────────────────────────────────
    all_rows = []
    summary_rows = []
    paired_rows = []

    for shot_count in EVAL_SHOTS:
        shot_idx = SHOT_IDX[shot_count]
        print(f"\n{'='*60}")
        print(f"  N = {shot_count}")
        print(f"{'='*60}")

        # Flatten: (n_test × n_seeds) samples
        dgi_flat = []
        target_flat = []
        ids_flat = []
        seeds_flat = []
        for img_i in range(n_test):
            for seed_j in range(N_SEEDS):
                dgi_flat.append(dgi_all[img_i, seed_j, shot_idx])
                target_flat.append(test_objects[img_i].astype(np.float32))
                ids_flat.append(str(test_ids[img_i]))
                seeds_flat.append(TEST_SEEDS[seed_j])

        dgi_tensor = torch.from_numpy(np.array(dgi_flat)).unsqueeze(1).float()
        target_np = np.array(target_flat, dtype=np.float64)
        dgi_np = dgi_tensor.squeeze(1).numpy().astype(np.float64)
        n_total = len(target_np)
        print(f"  Samples: {n_total} ({n_test} images × {N_SEEDS} seeds)")

        # Load models
        print("  Loading models ...")
        e1_model = load_e1(shot_count, device)
        e2a_model = load_e2a(shot_count, device)
        e3a_model, diffusion = load_e3a(shot_count, device)
        e4a_model = load_e4("configs/e4a_mse.yaml", shot_count, device)
        e4b_model = load_e4("configs/e4b_mse_ssim.yaml", shot_count, device)

        # Run inference
        print("  Running E1 inference ...")
        e1_preds = run_unet_inference(e1_model, dgi_tensor, device)
        print("  Running E2a inference ...")
        e2a_preds = run_unet_inference(e2a_model, dgi_tensor, device)
        print("  Running E3a DDIM inference (100 steps) ...")
        t_e3 = time.time()
        e3a_preds = run_e3_inference(e3a_model, diffusion, dgi_tensor, device, ddim_steps=100)
        print(f"    E3a done in {time.time()-t_e3:.1f}s")
        print("  Running E4a inference ...")
        e4a_preds = run_e4_inference(e4a_model, dgi_tensor, device)
        print("  Running E4b inference ...")
        e4b_preds = run_e4_inference(e4b_model, dgi_tensor, device)

        method_preds = {
            "DGI": dgi_np,
            "E1": e1_preds.astype(np.float64),
            "E2a": e2a_preds.astype(np.float64),
            "E3a": e3a_preds.astype(np.float64),
            "E4a": e4a_preds.astype(np.float64),
            "E4b": e4b_preds.astype(np.float64),
        }

        # Per-image metrics
        print("  Computing per-image metrics ...")
        shot_metrics = {m: [] for m in METHODS}
        for i in range(n_total):
            ref = target_np[i]
            for method in METHODS:
                rec = np.clip(method_preds[method][i], 0, 1)
                m = evaluate(ref, rec)
                all_rows.append({
                    "image_id": ids_flat[i],
                    "seed": seeds_flat[i],
                    "shot_count": shot_count,
                    "method": method,
                    **{k: f"{v:.6f}" for k, v in m.items()},
                })
                shot_metrics[method].append(m)

        # Aggregate summary
        for method in METHODS:
            metrics_list = shot_metrics[method]
            keys = metrics_list[0].keys()
            agg = {}
            for k in keys:
                vals = [m[k] for m in metrics_list]
                agg[k] = float(np.mean(vals))
            summary_rows.append({
                "shot_count": shot_count,
                "method": method,
                **{k: f"{v:.4f}" for k, v in agg.items()},
            })

        # Print summary table
        print(f"\n  {'Method':<8} {'PSNR':>8} {'SSIM':>8} {'Pearson_r':>10} {'edge_r':>8}")
        print(f"  {'-'*46}")
        for method in METHODS:
            s = {k: float(v) for k, v in summary_rows[-len(METHODS) + METHODS.index(method)].items()
                 if k not in ("shot_count", "method")}
            print(f"  {method:<8} {s['PSNR']:8.2f} {s['SSIM']:8.4f} "
                  f"{s['Pearson_r']:10.4f} {s['edge_r']:8.4f}")

        # Paired bootstrap CIs: E4a-E1, E4b-E1 (PSNR and SSIM)
        print(f"\n  Paired bootstrap CIs ({BOOTSTRAP_RESAMPLES} resamples) ...")
        for e4_name, e4_preds_arr in [("E4a", e4a_preds), ("E4b", e4b_preds)]:
            ci = paired_bootstrap_ci(
                target_np,
                e4_preds_arr.astype(np.float64),
                e1_preds.astype(np.float64),
                n_per_image=N_SEEDS,
                n_resamples=BOOTSTRAP_RESAMPLES,
                seed=0,
                metrics=("PSNR", "SSIM"),
            )
            for metric, vals in ci.items():
                sig_str = "YES" if vals["significant"] else "NO"
                print(f"    {e4_name}-E1 {metric}: "
                      f"{vals['mean_diff']:+.4f} [{vals['ci_lo']:+.4f}, {vals['ci_hi']:+.4f}] "
                      f"sig={sig_str}")
                paired_rows.append({
                    "shot_count": shot_count,
                    "comparison": f"{e4_name}-E1",
                    "metric": metric,
                    "mean_diff": f"{vals['mean_diff']:.6f}",
                    "ci_lo": f"{vals['ci_lo']:.6f}",
                    "ci_hi": f"{vals['ci_hi']:.6f}",
                    "significant": sig_str,
                })

        # Save reconstruction figures for this shot count
        print("  Saving reconstruction figures ...")
        _save_figures(
            OUT_DIR, shot_count, dgi_np, e1_preds, e2a_preds,
            e3a_preds, e4a_preds, e4b_preds, target_np, ids_flat,
        )

        # Free models
        del e1_model, e2a_model, e3a_model, diffusion, e4a_model, e4b_model
        torch.cuda.empty_cache()

    # ── 4. Latency benchmark ────────────────────────────────────
    print("\n=== Latency benchmark ===")
    latency = benchmark_latency(device, shot_count=1000)
    latency_rows = []
    for name, vals in latency.items():
        print(f"  {name:>15}: {vals['mean_ms']:.1f} ± {vals['std_ms']:.1f} ms")
        latency_rows.append({
            "method": name,
            "mean_ms": f"{vals['mean_ms']:.2f}",
            "std_ms": f"{vals['std_ms']:.2f}",
            "min_ms": f"{vals['min_ms']:.2f}",
            "max_ms": f"{vals['max_ms']:.2f}",
        })
    if "E3a_DDIM100" in latency and "E4a" in latency:
        speedup = latency["E3a_DDIM100"]["mean_ms"] / latency["E4a"]["mean_ms"]
        print(f"  E3a/E4a speedup: {speedup:.1f}x")

    # ── 5. Write all outputs ────────────────────────────────────
    print("\n=== Writing outputs ===")

    # test_metrics.csv
    csv_path = OUT_DIR / "test_metrics.csv"
    fields = ["image_id", "seed", "shot_count", "method", "PSNR", "SSIM", "MSE", "Pearson_r", "edge_r"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"  {csv_path} ({len(all_rows)} rows)")

    # test_summary.csv
    csv_path = OUT_DIR / "test_summary.csv"
    fields = ["shot_count", "method", "PSNR", "SSIM", "MSE", "Pearson_r", "edge_r"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"  {csv_path} ({len(summary_rows)} rows)")

    # paired_bootstrap.csv
    csv_path = OUT_DIR / "paired_bootstrap.csv"
    fields = ["shot_count", "comparison", "metric", "mean_diff", "ci_lo", "ci_hi", "significant"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(paired_rows)
    print(f"  {csv_path} ({len(paired_rows)} rows)")

    # latency_benchmark.csv
    csv_path = OUT_DIR / "latency_benchmark.csv"
    fields = ["method", "mean_ms", "std_ms", "min_ms", "max_ms"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(latency_rows)
    print(f"  {csv_path}")

    # checkpoint_hashes.txt (copy from scratchpad)
    _write_checkpoint_hashes(OUT_DIR / "checkpoint_hashes.txt")
    print(f"  {OUT_DIR / 'checkpoint_hashes.txt'}")

    # test_evaluation_config.yaml
    config = {
        "evaluation_type": "final_held_out_test",
        "date": time.strftime("%Y-%m-%d"),
        "test_images": n_test,
        "test_seeds": TEST_SEEDS,
        "n_seeds_per_image": N_SEEDS,
        "shot_counts": EVAL_SHOTS,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_confidence": 0.95,
        "bootstrap_seed": 0,
        "ddim_steps_e3": 100,
        "device": str(device),
        "frozen_root": str(FROZEN_ROOT),
        "ghost_imaging_v2_root": str(GI2_ROOT),
        "models_evaluated": METHODS,
        "metrics": ["PSNR", "SSIM", "MSE", "Pearson_r", "edge_r"],
        "dgi_generation": {
            "method": "StreamingDGI (Bernoulli 0.5, noise-free buckets, calibrated DGI, clipped [0,1])",
            "max_patterns": max(SHOTS),
            "pattern_batch_size": 500,
            "pattern_variance": 0.25,
        },
    }
    with open(OUT_DIR / "test_evaluation_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  {OUT_DIR / 'test_evaluation_config.yaml'}")

    print(f"\n=== All outputs saved to {OUT_DIR}/ ===")


# ── Helpers ─────────────────────────────────────────────────────

def _save_figures(out_dir, shot_count, dgi_np, e1_preds, e2a_preds,
                  e3a_preds, e4a_preds, e4b_preds, targets, ids_flat,
                  n_figures=5):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    [warn] matplotlib not available — skipping figures")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    n = min(n_figures, len(targets))
    indices = np.linspace(0, len(targets) - 1, n, dtype=int)

    for idx in indices:
        fig, axes = plt.subplots(1, 7, figsize=(28, 4))
        panels = [
            (dgi_np[idx], "DGI input"),
            (e1_preds[idx], "E1 (baseline)"),
            (e2a_preds[idx], "E2a (MSE+SSIM)"),
            (e3a_preds[idx], "E3a (diffusion)"),
            (e4a_preds[idx], "E4a (MSE)"),
            (e4b_preds[idx], "E4b (MSE+SSIM)"),
            (targets[idx], "Ground truth"),
        ]
        for ax, (img, title) in zip(axes, panels):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        iid = ids_flat[idx] if idx < len(ids_flat) else idx
        fig.suptitle(f"Test image: {iid}  |  N={shot_count}", fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(fig_dir / f"test_N{shot_count}_{idx:04d}.png",
                    dpi=100, bbox_inches="tight")
        plt.close(fig)
    print(f"    Saved {n} figures for N={shot_count}")


def _write_checkpoint_hashes(path: Path):
    lines = ["=== Checkpoint SHA256 Hashes ===",
             f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}", ""]

    ckpts = []
    for N in EVAL_SHOTS:
        ckpts.append(("E1", N, FROZEN_ROOT / E1_CKPT_TEMPLATE.format(N=N)))
    for exp in ["e2a_mse_ssim", "e3a_eps_only", "e4a_mse", "e4b_mse_ssim"]:
        label = exp.split("_")[0].upper()
        for N in EVAL_SHOTS:
            ckpts.append((label, N, GI2_ROOT / "checkpoints" / f"{exp}_N{N}" / "best.pt"))

    for label, N, p in ckpts:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {label}_N{N}  {p}")

    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
