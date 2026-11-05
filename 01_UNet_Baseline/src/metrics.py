"""Image-quality metrics for ghost-imaging reconstructions (images in [0, 1])."""

from __future__ import annotations

import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


def evaluate(reference: np.ndarray, recon: np.ndarray) -> dict:
    """PSNR (dB), SSIM and MSE against the reference, both on a [0, 1] scale.

    Pearson correlation is added as a scale-free sanity check: it does not depend
    on how the reconstruction was normalised.
    """
    ref = reference.astype(np.float64)
    rec = recon.astype(np.float64)
    return {
        "PSNR": float(peak_signal_noise_ratio(ref, rec, data_range=1.0)),
        "SSIM": float(structural_similarity(ref, rec, data_range=1.0)),
        "MSE": float(mean_squared_error(ref, rec)),
        "Pearson_r": float(np.corrcoef(ref.ravel(), rec.ravel())[0, 1]),
    }


def evaluate_in_mask(reference: np.ndarray, recon: np.ndarray, mask: np.ndarray) -> dict:
    """Secondary metrics restricted to pixels where mask is True (e.g. inside the FOV).

    PSNR/MSE/r use only masked pixels; SSIM is the mean of the full SSIM map over
    the mask. Used to check that black out-of-FOV surrounds do not drive the
    whole-image numbers.
    """
    ref = reference.astype(np.float64)
    rec = recon.astype(np.float64)
    _, ssim_map = structural_similarity(ref, rec, data_range=1.0, full=True)
    a, b = ref[mask], rec[mask]
    mse = float(np.mean((a - b) ** 2))
    return {
        "PSNR": float(10 * np.log10(1.0 / mse)) if mse > 0 else float("inf"),
        "SSIM": float(ssim_map[mask].mean()),
        "MSE": mse,
        "Pearson_r": float(np.corrcoef(a, b)[0, 1]),
    }
