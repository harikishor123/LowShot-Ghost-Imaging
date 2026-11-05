"""
Evaluation metrics for ghost_imaging_v2.

Extends the frozen project's metrics.py with edge_r (Sobel-map Pearson correlation).
All inputs are numpy float64 arrays in [0, 1].
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)


# ──────────────────────────────────────────────────────────────
# Sobel edge helper (NumPy, matches DiffTransFuse Sobelxy kernel)
# ──────────────────────────────────────────────────────────────

def _sobel_edges_np(img: np.ndarray) -> np.ndarray:
    """Sobel edge magnitude on a 2D float image."""
    from scipy.ndimage import convolve
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    gx = convolve(img.astype(np.float64), kx, mode="reflect")
    gy = convolve(img.astype(np.float64), ky, mode="reflect")
    return np.abs(gx) + np.abs(gy)


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def evaluate(reference: np.ndarray, recon: np.ndarray) -> dict:
    """
    Full metric set: PSNR, SSIM, MSE, Pearson_r, edge_r.
    Both arrays: 2D float, range [0, 1].

    edge_r — Pearson correlation on Sobel edge-magnitude maps.
    Low edge_r indicates lost cell boundaries or over-smoothing.
    """
    ref = reference.astype(np.float64)
    rec = recon.astype(np.float64)
    psnr = float(peak_signal_noise_ratio(ref, rec, data_range=1.0))
    ssim = float(structural_similarity(ref, rec, data_range=1.0))
    mse  = float(mean_squared_error(ref, rec))
    r    = float(np.corrcoef(ref.ravel(), rec.ravel())[0, 1])

    edge_ref = _sobel_edges_np(ref).ravel()
    edge_rec = _sobel_edges_np(rec).ravel()
    edge_r   = float(np.corrcoef(edge_ref, edge_rec)[0, 1])

    return {"PSNR": psnr, "SSIM": ssim, "MSE": mse, "Pearson_r": r, "edge_r": edge_r}


def evaluate_batch(
    references: np.ndarray, recons: np.ndarray
) -> dict:
    """
    Mean metrics over a batch of images.
    references, recons: (N, H, W) float, range [0, 1].
    Returns mean of each metric.
    """
    all_metrics: list[dict] = []
    for ref, rec in zip(references, recons):
        all_metrics.append(evaluate(ref, rec))
    keys = all_metrics[0].keys()
    return {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}


def bootstrap_ci(
    references: np.ndarray,
    recons: np.ndarray,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    n_per_image: int = 1,
) -> dict:
    """
    Bootstrap confidence intervals (image-level cluster resampling).

    B5 fix: when n_per_image > 1 (val set has 1000 images × 3 seeds = 3000 items),
    resample at IMAGE level, keeping all seeds for each resampled image together.
    This preserves within-image correlation across seeds.

    Args:
        references:   (N, H, W) float, range [0, 1]
        recons:       (N, H, W) float, range [0, 1]
        n_resamples:  number of bootstrap samples (2000 recommended)
        confidence:   e.g. 0.95 for 95% CI
        seed:         reproducibility seed
        n_per_image:  seeds per image; 1 for flat resampling, 3 for val set.
                      N must be divisible by n_per_image.

    Returns:
        dict: {metric: (mean, lower, upper)}

    Example for val set (1000 images × 3 seeds):
        ci = bootstrap_ci(refs, recs, n_per_image=3)
    """
    rng = np.random.default_rng(seed)
    N   = len(references)
    assert N % n_per_image == 0, f"N={N} not divisible by n_per_image={n_per_image}"
    n_images = N // n_per_image

    per_item = [evaluate(references[i], recons[i]) for i in range(N)]
    keys = per_item[0].keys()

    alpha = 1 - confidence
    results = {}
    for k in keys:
        vals = np.array([m[k] for m in per_item])
        mean = float(vals.mean())

        # Resample at image level: pick n_images image indices, include all seeds
        boot_means = np.empty(n_resamples, dtype=np.float64)
        for b in range(n_resamples):
            img_indices = rng.integers(0, n_images, n_images)   # (n_images,) with replacement
            # Expand each image index to its n_per_image item slots
            item_indices = (img_indices[:, None] * n_per_image
                            + np.arange(n_per_image)[None, :]).ravel()   # (N,)
            boot_means[b] = vals[item_indices].mean()

        lo = float(np.percentile(boot_means, 100 * alpha / 2))
        hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        results[k] = (mean, lo, hi)
    return results
