#!/usr/bin/env python3
"""Explicit colour-space conversions for the photon-limited reconstruction pipeline.

Pipeline (each arrow is a named function here or in STEP 4):

    JPEG RGB (uint8, sRGB-encoded)
      -> normalized RGB            : decode_uint8 / decode_uint16     (values in [0, 1], still gamma-encoded)
      -> linear-light RGB          : srgb_to_linear                   (proportional to relative irradiance)
      -> photon simulation         : STEP 4 (Poisson on expected counts = linear * photon budget)
      -> noisy photon-limited image (linear) -> linear_to_srgb for display only

ASSUMPTIONS (documented, not verified - the dataset has no camera metadata):
  * The JPEGs carry no ICC profile/EXIF, so we assume the standard sRGB transfer curve
    (IEC 61966-2-1). Phone/microscope cameras normally encode to sRGB, but their exact
    tone curve, white balance, sharpening and denoising are unknown.
  * Linear values are therefore *relative* linear light, not calibrated radiometric
    intensity or photon counts. A value of 1.0 means "brightest representable", not a
    known number of photons; the photon scale is a simulation parameter chosen in STEP 4.
  * Brightfield images are transmitted light: bright background = high photon flux,
    stained structures absorb light = low flux. Simulated shot noise is therefore
    strongest (relatively) in the dark stained nuclei/chromosomes.
  * The clean references are already JPEG-compressed (q~75) and some were rescaled by the
    original authors, so they contain compression artefacts and are not noise-free.
  * This is a SIMULATION experiment: images produced from these references are synthetic
    photon-limited images, not experimentally acquired low-light measurements.
"""

from __future__ import annotations

import numpy as np

# IEC 61966-2-1 sRGB constants
_A = 0.055
_GAMMA = 2.4
_SRGB_THRESH = 0.04045          # encoded-domain threshold
_LIN_THRESH = 0.0031308         # linear-domain threshold
_SLOPE = 12.92


def decode_uint8(img_u8: np.ndarray) -> np.ndarray:
    """uint8 sRGB-encoded -> float32 normalized sRGB in [0, 1]."""
    return img_u8.astype(np.float32) / 255.0


def decode_uint16(img_u16: np.ndarray) -> np.ndarray:
    """uint16 sRGB-encoded (data/v2/reference/*.npz) -> float32 normalized sRGB in [0, 1]."""
    return img_u16.astype(np.float32) / 65535.0


def encode_uint16(srgb: np.ndarray) -> np.ndarray:
    return np.round(np.clip(srgb, 0.0, 1.0) * 65535.0).astype(np.uint16)


def encode_uint8(srgb: np.ndarray) -> np.ndarray:
    return np.round(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Normalized sRGB-encoded values in [0, 1] -> linear-light values in [0, 1]."""
    x = np.clip(np.asarray(srgb, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= _SRGB_THRESH, x / _SLOPE, ((x + _A) / (1 + _A)) ** _GAMMA).astype(np.float32)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Linear-light values in [0, 1] -> normalized sRGB-encoded values in [0, 1]."""
    x = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= _LIN_THRESH, x * _SLOPE, (1 + _A) * np.power(x, 1 / _GAMMA) - _A).astype(np.float32)


def load_reference_linear(npz_path) -> tuple[np.ndarray, np.ndarray]:
    """Load a data/v2 clean reference: returns (linear RGB float32 HxWx3, valid mask bool HxW)."""
    with np.load(npz_path) as z:
        srgb = decode_uint16(z["srgb16"])
        mask = z["mask"].astype(bool)
    return srgb_to_linear(srgb), mask


def self_test() -> dict:
    """Round-trip checks used by the preprocessing validation report."""
    grid = np.arange(256, dtype=np.float32) / 255.0
    rt8 = linear_to_srgb(srgb_to_linear(grid))
    g16 = np.linspace(0, 1, 65536, dtype=np.float32)
    rt16 = decode_uint16(encode_uint16(linear_to_srgb(srgb_to_linear(g16))))
    lin = srgb_to_linear(grid)
    return {
        "max_abs_roundtrip_err_uint8_grid": float(np.abs(rt8 - grid).max()),
        "max_abs_roundtrip_err_uint16_grid": float(np.abs(rt16 - g16).max()),
        "monotonic": bool(np.all(np.diff(lin) > 0)),
        "linear_of_srgb_0.5": float(srgb_to_linear(np.float32(0.5))),  # ~0.214
        "endpoints": [float(lin[0]), float(lin[-1])],
    }


if __name__ == "__main__":
    print(self_test())
