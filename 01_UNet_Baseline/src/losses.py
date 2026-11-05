#!/usr/bin/env python3
"""Losses and metrics. All in LINEAR LIGHT, restricted to the field-of-view mask.

Training targets are packed as y_true = [R, G, B, mask] (4 channels) so the mask travels
with the patch through augmentation.

Evaluation metrics (numpy, full images, inside mask):
  MSE, MAE   : mean over valid pixels x 3 channels
  PSNR       : 10 log10(1 / MSE)  (data range 1, linear light)
  SSIM       : Wang et al. 2004, Gaussian window sigma=1.5 (truncate 3.5 -> 11x11),
               K1=0.01, K2=0.03, L=1, per channel, averaged over pixels whose whole
               window lies inside the mask (mask eroded by 5 px), then over channels.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def masked_mse(y_true, y_pred):
    import keras.ops as K
    y, m = y_true[..., :3], y_true[..., 3:4]
    se = K.square(y_pred - y) * m
    return K.sum(se) / (3.0 * K.sum(m) + 1e-8)


def masked_mae(y_true, y_pred):
    import keras.ops as K
    y, m = y_true[..., :3], y_true[..., 3:4]
    return K.sum(K.abs(y_pred - y) * m) / (3.0 * K.sum(m) + 1e-8)


def masked_psnr(y_true, y_pred):
    import keras.ops as K
    return 10.0 * K.log(1.0 / (masked_mse(y_true, y_pred) + 1e-12)) / np.log(10.0)


# ------------------------------------------------------------------------- numpy metrics
def ssim_map(a: np.ndarray, b: np.ndarray, L: float = 1.0, sigma: float = 1.5) -> np.ndarray:
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    f = lambda z: ndi.gaussian_filter(z, sigma, truncate=3.5)
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a, mu_b = f(a), f(b)
    saa = f(a * a) - mu_a ** 2
    sbb = f(b * b) - mu_b ** 2
    sab = f(a * b) - mu_a * mu_b
    return ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))


def image_metrics(pred: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> dict:
    m = mask.astype(bool)
    d = (pred - clean)[m]
    mse = float(np.mean(d ** 2))
    mae = float(np.mean(np.abs(d)))
    core = ndi.binary_erosion(m, iterations=5)
    ssim = float(np.mean([ssim_map(pred[..., c], clean[..., c])[core].mean() for c in range(3)]))
    return {"mse": mse, "mae": mae, "psnr": float(10 * np.log10(1.0 / max(mse, 1e-12))), "ssim": ssim}
