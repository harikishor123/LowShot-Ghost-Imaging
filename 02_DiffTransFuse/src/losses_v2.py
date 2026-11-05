"""
Loss functions for ghost_imaging_v2.

The frozen E1 baseline trains with nn.MSELoss() on raw (unclamped) predictions
from a linear output head.  We follow the same convention:

  • All losses here accept RAW predictions — values may be slightly outside [0,1].
  • We do NOT clamp predictions before computing losses.
  • The MSE gradient naturally drives predictions toward [0,1] (where targets live).
  • SSIM and edge losses are well-defined for predictions outside [0,1] as long as
    sigma_p^2 is clamped to 0 (floating-point variance can go slightly negative).

Three new losses:
  ssim_loss   — 1 - SSIM; targets over-smoothing (blurry cells).
  edge_loss   — L1 on Sobel edge maps; targets cell boundary preservation.
  meas_loss   — physics-grounded measurement consistency; disabled in E2,
                enabled in E3+ (requires patterns P and bucket values B).

Ablation support via combined_loss():
  E2a: lambda_ssim > 0, lambda_edge = 0
  E2b: lambda_ssim = 0, lambda_edge > 0
  E2c: lambda_ssim > 0, lambda_edge > 0  (default E2 configuration)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# SSIM loss   (B2 fix: clamp variance to 0 before denominator)
# ──────────────────────────────────────────────────────────────

def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def _ssim_window(size: int, channels: int, device: torch.device) -> torch.Tensor:
    k1d = _gaussian_kernel_1d(size, 1.5, device)
    k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)       # [size, size]
    return k2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, size, size).contiguous()


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    K1: float = 0.01,
    K2: float = 0.03,
    data_range: float = 1.0,
) -> torch.Tensor:
    """
    1 − SSIM(pred, target).  Lower is better.

    Inputs:
        pred, target — [B, 1, H, W] float32.
        target is always in [0, 1] (confirmed from data inspection).
        pred may be slightly outside [0, 1] (linear U-Net head).
        data_range = 1.0 because targets are in [0, 1].

    SSIM stability constants are C1 = (K1 * data_range)^2 and
    C2 = (K2 * data_range)^2 — standard values for data_range=1.

    Fix B2: sigma_p_sq and sigma_t_sq are clamped to ≥ 0 before entering
    the denominator.  Floating-point subtraction E[X^2] - E[X]^2 can yield
    small negative values; without clamping this produces NaN in the denom.
    """
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    B, C, H, W = pred.shape
    win = _ssim_window(window_size, C, pred.device)
    pad = window_size // 2

    # Local means
    mu_p = F.conv2d(pred,   win, padding=pad, groups=C)   # E[pred]
    mu_t = F.conv2d(target, win, padding=pad, groups=C)   # E[target]

    # Local variances — clamp negative floating-point artefacts to 0
    sigma_p_sq = (F.conv2d(pred   * pred,   win, padding=pad, groups=C) - mu_p ** 2).clamp_min(0)
    sigma_t_sq = (F.conv2d(target * target, win, padding=pad, groups=C) - mu_t ** 2).clamp_min(0)

    # Local covariance (can legitimately be negative — do NOT clamp)
    sigma_pt   =  F.conv2d(pred   * target, win, padding=pad, groups=C) - mu_p * mu_t

    # SSIM map
    numer = (2.0 * mu_p * mu_t + C1) * (2.0 * sigma_pt + C2)
    denom = (mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p_sq + sigma_t_sq + C2)
    # denom > 0 always because sigma_p_sq ≥ 0, sigma_t_sq ≥ 0, and C2 > 0.
    # The clamp_min(1e-8) is a belt-and-suspenders guard only.
    ssim_map = numer / denom.clamp_min(1e-8)
    return 1.0 - ssim_map.mean()


# ──────────────────────────────────────────────────────────────
# Edge (Sobel) loss   —  adapted from DiffTransFuse/models/losses
# ──────────────────────────────────────────────────────────────

def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Sobel edge magnitude.  Input [B, 1, H, W].  Output [B, 1, H, W].

    Sobel is a linear filter — it does not require [0,1] inputs.
    Applies to raw (unclamped) predictions without issue.
    Kernel matches DiffTransFuse Sobelxy exactly (also used in src/metrics.py).
    """
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=x.dtype, device=x.device,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
        dtype=x.dtype, device=x.device,
    ).view(1, 1, 3, 3)
    return torch.abs(F.conv2d(x, kx, padding=1)) + torch.abs(F.conv2d(x, ky, padding=1))


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    L1 loss on Sobel edge magnitudes.  Both [B, 1, H, W], float32.

    Target edge map is detached (no gradient through target).
    Addresses the edge_r deficit in the frozen E1 baseline at low shot counts
    (edge_r 0.57 vs Pearson_r 0.81 at 500 shots).
    """
    with torch.no_grad():
        edge_target = _sobel_edges(target)
    edge_pred = _sobel_edges(pred)
    return F.l1_loss(edge_pred, edge_target)


# ──────────────────────────────────────────────────────────────
# Measurement consistency loss  (E3+ only — disabled in E2)
# ──────────────────────────────────────────────────────────────

def meas_loss(
    o_pred: torch.Tensor,
    bucket_values: torch.Tensor,
    patterns_flat: torch.Tensor,
) -> torch.Tensor:
    """
    Physics-grounded measurement consistency.

    Forward measurement operator A: O ∈ ℝ^{64×64} → B ∈ ℝ^N
        A(O) = P.reshape(N, 4096) @ O.reshape(4096)
        B_i = P_i · O  (inner product of i-th pattern with object)

    Loss:  L_meas = (1/N) × ||A(O_pred) − B_measured||²

    Args:
        o_pred:         [B, 1, 64, 64]  — predicted object (raw, before clipping)
        bucket_values:  [B, N]          — ground-truth bucket measurements
        patterns_flat:  [B, N, 4096]    — illumination patterns (float32, {0,1})

    Scale note:
        B_i = P_i · O ∈ [0, N_pix * max(O)] ≈ [0, 4096 * 1] = 4096 (unnormalized)
        In the frozen project: B = P @ O where O ∈ [0,1] and P ∈ {0,1}
        → B_i ∈ [0, 4096] for 64×64 images.
        L_meas normalises by N (number of measurements) to give per-measurement MSE.
        At typical values: B_i ~ 2000, MSE ~ 10^4-10^6 without 1/N.
        With 1/N normalisation: L_meas ~ (error/B_i)^2 × B_i ~ order 1.
        However, this is still much larger than MSE on images (order 0.01-0.1).
        Weight λ_meas = 0.001 to 0.005 is recommended when first enabling.

    Not enabled in E2.  Enabled in E3+ where the diffusion model uses
    x0-prediction (predicts O directly, not noise), making L_meas stable.
    """
    B, _, H, W = o_pred.shape
    o_flat = o_pred.reshape(B, H * W, 1)                         # [B, 4096, 1]
    b_pred = torch.bmm(patterns_flat, o_flat).squeeze(-1)        # [B, N]
    return F.mse_loss(b_pred, bucket_values)


# ──────────────────────────────────────────────────────────────
# Combined loss — used by train_e2.py
# Ablation support: set lambda=0 to disable any term.
# ──────────────────────────────────────────────────────────────

def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_ssim: float = 0.1,
    lambda_edge: float = 0.05,
) -> tuple[torch.Tensor, dict]:
    """
    Total = MSE(pred, target)
          + lambda_ssim × ssim_loss(pred, target)
          + lambda_edge × edge_loss(pred, target)

    MSE weight is always 1.0 (not configurable — matches frozen E1 baseline).
    Set lambda_ssim=0 or lambda_edge=0 for ablation experiments E2a/E2b.

    Returns (total_loss, {component_dict}) for logging.
    Components are raw (unweighted) values — multiply by lambda to get contribution.
    """
    l_mse  = F.mse_loss(pred, target)
    l_ssim = ssim_loss(pred, target)
    l_edge = edge_loss(pred, target)

    total = l_mse + lambda_ssim * l_ssim + lambda_edge * l_edge

    return total, {
        "mse":  l_mse.item(),
        "ssim": l_ssim.item(),
        "edge": l_edge.item(),
    }
