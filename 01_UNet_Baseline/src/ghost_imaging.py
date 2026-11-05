"""Forward model for classical / computational ghost imaging (CGI) simulation.

Phase 1 of the AI-assisted Ghost Imaging study. The variable under study is the
NUMBER OF MEASUREMENTS (illumination patterns / bucket values), not the photon
count per measurement: the bucket signal here is noise-free (no Poisson noise,
no detector noise).

Pipeline for one object O(x, y):

    reference .npz (linear-light RGB)  --load_object-->  O, grayscale in [0, 1]
    seeded RNG                         --generate_patterns-->  P_1 .. P_M (binary)
    B_i = sum_{x,y} P_i(x, y) * O(x, y)   (bucket_measurements)

The same pattern sequence is reused for every measurement count: an N-shot
experiment uses patterns/buckets [0:N] of the single M-shot simulation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.color_space import load_reference_linear

# Rec. 709 / sRGB luminance weights, applied to LINEAR-light RGB, so that O is a
# relative transmitted intensity (what a bucket detector would integrate).
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def load_object(npz_path: str | Path, size: int, return_fov: bool = False):
    """Load a v2 reference and turn it into a size x size grayscale GI object.

    Steps (all documented in the returned info dict):
      1. linear-light RGB + FOV mask via color_space.load_reference_linear
      2. luminance Y = 0.2126 R + 0.7152 G + 0.0722 B (linear light)
      3. centre crop to the largest square
      4. area-average (box filter) downsample to size x size
    No contrast stretching or renormalisation: O keeps its physical [0, 1] scale.

    Returns (obj, info), or (obj, info, fov) with return_fov=True, where fov is the
    size x size fraction of each output pixel that lies inside the FOV mask (same
    crop and box filter as the object). fov is only used for secondary metrics.
    """
    rgb_lin, mask = load_reference_linear(npz_path)
    y = rgb_lin.astype(np.float64) @ LUMA_WEIGHTS
    h, w = y.shape
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    y_sq = y[y0:y0 + s, x0:x0 + s]
    m_sq = mask[y0:y0 + s, x0:x0 + s]

    obj = np.asarray(
        Image.fromarray(y_sq.astype(np.float32), mode="F").resize((size, size), Image.BOX),
        dtype=np.float64,
    )
    obj = np.clip(obj, 0.0, 1.0)
    info = {
        "source_shape_hw": [int(h), int(w)],
        "crop_box_y0x0_side": [int(y0), int(x0), int(s)],
        "fov_valid_fraction_in_crop": float((m_sq > 0).mean()),
        "object_shape_hw": [size, size],
        "object_min": float(obj.min()),
        "object_max": float(obj.max()),
        "object_mean": float(obj.mean()),
        "object_std": float(obj.std()),
    }
    if not return_fov:
        return obj, info
    fov = np.asarray(
        Image.fromarray(m_sq.astype(np.float32), mode="F").resize((size, size), Image.BOX),
        dtype=np.float64,
    )
    return obj, info, fov


def generate_patterns(n_patterns: int, shape: tuple[int, int], seed: int, p_on: float = 0.5) -> np.ndarray:
    """Independent binary random illumination patterns, P_i(x, y) in {0, 1}.

    Each pixel of each pattern is i.i.d. Bernoulli(p_on). Returned as uint8 with
    shape (n_patterns, H, W). A single seeded generator produces the whole
    sequence, so pattern i is identical regardless of how many are used later.
    """
    rng = np.random.default_rng(seed)
    return (rng.random((n_patterns, *shape)) < p_on).astype(np.uint8)


def pattern_batches(n_patterns: int, shape: tuple[int, int], seed: int, batch: int, p_on: float = 0.5):
    """Yield the SAME sequence as generate_patterns(n_patterns, ...) in consecutive batches.

    numpy's Generator draws consecutive values across calls, so concatenating the
    batches is bit-identical to one large call (checked in run_dgi_dataset.py).
    """
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_patterns:
        b = min(batch, n_patterns - done)
        yield (rng.random((b, *shape)) < p_on).astype(np.uint8)
        done += b


def pattern_variance(p_on: float = 0.5) -> float:
    """Theoretical per-pixel variance of a Bernoulli(p_on) pattern: p(1 - p)."""
    return p_on * (1.0 - p_on)


def bucket_measurements(patterns: np.ndarray, obj: np.ndarray) -> np.ndarray:
    """Bucket detector signal B_i = sum_{x,y} P_i(x, y) * O(x, y), one per pattern.

    Noise-free and in float64: the only source of reconstruction error is the
    finite number of measurements.
    """
    n = patterns.shape[0]
    return patterns.reshape(n, -1).astype(np.float64) @ obj.reshape(-1).astype(np.float64)


def reference_intensity(patterns: np.ndarray) -> np.ndarray:
    """Total illumination of each pattern, R_i = sum_{x,y} P_i(x, y).

    In computational GI the patterns are known, so R_i is computed rather than
    measured (it is what a reference "bucket" with no object would record).
    Used by differential ghost imaging (DGI).
    """
    n = patterns.shape[0]
    return patterns.reshape(n, -1).sum(axis=1, dtype=np.float64)
