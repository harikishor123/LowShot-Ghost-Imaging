#!/usr/bin/env python3
"""STEP 4: simulation of photon-limited (shot-noise-limited) brightfield acquisition.

This is a SIMULATION of photon-limited acquisition with classical Poisson counting
statistics. It is not a model of squeezed light, quantum imaging or any sub-shot-noise
technique, and the outputs are synthetic images, not experimental low-light measurements.

Model (per pixel x,y and colour channel c, applied only inside the FOV mask M):

    I(x,y,c)      in [0,1]   clean linear-light reference  (color_space.load_reference_linear)
    P             photons    photon budget = expected detected photons for a pixel with I = 1
    lambda        = I * P                     expected photon count
    N             ~ Poisson(lambda)           detected photon count (integer)
    I_noisy       = N / P                     normalized noisy image (unbiased: E[I_noisy] = I)
    outside M:    lambda = N = I_noisy = 0    (no photons simulated, pixel stays masked)

Consequences: Var(N) = E[N] = lambda;  per-pixel SNR = E[N]/std(N) = sqrt(I * P).

Normalized levels map to absolute budgets via P = level * P_FULL with P_FULL = 1000
photons (see PHOTON_LEVELS). I_noisy is NOT clipped (it can exceed 1 at low P); clip only
for display. Each RGB channel is treated as an independent counting detector (assumption).

Reproducibility: every draw uses a numpy Generator seeded from
(BASE_SEED, split, image id, level, realization) via SeedSequence, so the same
image/level/realization always gives the same noisy image, independent of run order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from color_space import load_reference_linear  # noqa: E402

P_FULL = 1000.0                      # photons for I = 1 at the 100% level
NORMALIZED_LEVELS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
PHOTON_LEVELS = {f"{int(round(l * 100))}%": l * P_FULL for l in NORMALIZED_LEVELS}  # {'1%': 10.0, ...}
BASE_SEED = 20261004


def seed_key(split: str, image_id: str, photons: float, realization: int = 0) -> str:
    return f"{split}|{image_id}|{photons:.6g}|{realization}"


def seed_entropy(split: str, image_id: str, photons: float, realization: int = 0) -> list[int]:
    """The exact SeedSequence entropy words used for this sample: [BASE_SEED, w0, w1, w2, w3]."""
    words = np.frombuffer(hashlib.sha256(seed_key(split, image_id, photons, realization).encode()).digest()[:16],
                          dtype=np.uint32)
    return [BASE_SEED, *[int(w) for w in words]]


def rng_for(split: str, image_id: str, photons: float, realization: int = 0) -> np.random.Generator:
    """Deterministic, order-independent generator for one (split, image, level, realization)."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        seed_entropy(split, image_id, photons, realization))))


def expected_counts(linear: np.ndarray, mask: np.ndarray, photons: float) -> np.ndarray:
    """lambda = I * P inside the mask, 0 outside."""
    if photons <= 0:
        raise ValueError("photon budget must be > 0")
    lam = np.clip(linear, 0.0, 1.0).astype(np.float64) * float(photons)
    return np.where(mask[..., None] if lam.ndim == 3 else mask, lam, 0.0)


def sample_counts(lam: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """N ~ Poisson(lambda); exact Poisson sampling (numpy), int32 counts."""
    return rng.poisson(lam).astype(np.int32)


def counts_to_image(counts: np.ndarray, photons: float) -> np.ndarray:
    """I_noisy = N / P (float32, unclipped)."""
    return (counts.astype(np.float64) / float(photons)).astype(np.float32)


def simulate(linear: np.ndarray, mask: np.ndarray, photons: float, rng: np.random.Generator):
    """Full forward model. Returns (I_noisy float32, N int32, lambda float64)."""
    lam = expected_counts(linear, mask, photons)
    n = sample_counts(lam, rng)
    return counts_to_image(n, photons), n, lam


def simulate_reference(npz_path, photons: float, split: str, realization: int = 0):
    """Load a data/v2 reference and simulate one deterministic realization."""
    image_id = Path(npz_path).stem
    linear, mask = load_reference_linear(npz_path)
    noisy, n, lam = simulate(linear, mask, photons, rng_for(split, image_id, photons, realization))
    return {"clean_linear": linear, "mask": mask, "noisy_linear": noisy, "counts": n, "expected": lam,
            "photons": photons, "image_id": image_id, "split": split, "realization": realization}


def describe() -> dict:
    return {"model": "N ~ Poisson(I*P) inside FOV mask; I_noisy = N/P; outside mask = 0",
            "P_FULL_photons_at_I_eq_1": P_FULL, "levels": PHOTON_LEVELS, "base_seed": BASE_SEED,
            "channels": "R,G,B simulated as independent Poisson counters",
            "noise_sources": "shot noise only (no read noise, dark current, gain, quantization, blur)"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Simulate one photon-limited realization of a data/v2 reference.")
    ap.add_argument("reference", help="path to data/v2/reference/<id>.npz")
    ap.add_argument("--photons", type=float, default=PHOTON_LEVELS["10%"])
    ap.add_argument("--split", default="preview")
    ap.add_argument("--realization", type=int, default=0)
    ap.add_argument("--describe", action="store_true")
    a = ap.parse_args()
    if a.describe:
        print(json.dumps(describe(), indent=2))
    r = simulate_reference(a.reference, a.photons, a.split, a.realization)
    m = r["mask"]
    print(json.dumps({"image": r["image_id"], "photons": a.photons, "mean_lambda_in_fov": float(r["expected"][m].mean()),
                      "mean_counts_in_fov": float(r["counts"][m].mean())}, indent=2))
