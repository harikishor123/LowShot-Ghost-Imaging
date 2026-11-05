"""Bucket-measurement noise for Phase 4A (parameters fixed in outputs/phase4a_noisy_baselines/noise_config.json).

    s       = N_ph / mu_ref                       photons per normalised bucket unit
    K       ~ Poisson(s * B_clean)                photon shot noise
    e       ~ Normal(0, kappa * sqrt(N_ph))       additive detector/readout noise (photo-electrons)
    B_noisy = (K + e) / s                         signed reading; not clipped (negatives are counted)

The noise generator is seeded by SeedSequence([pattern_seed, 4040, regime_index]), independent of the
pattern generator (which keeps the established pattern_batches(seed)). One draw covers all buckets of
an (image set, seed), so shot-count prefixes are nested measurements.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REGIME_ORDER = ["LOW", "MEDIUM", "HIGH"]
CONFIG = Path(__file__).resolve().parents[1] / "outputs" / "phase4a_noisy_baselines" / "noise_config.json"


def load_noise_config() -> dict:
    return json.loads(CONFIG.read_text())


def noise_rng(pattern_seed: int, regime_index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([pattern_seed, 4040, regime_index]))


def add_bucket_noise(b_clean: np.ndarray, n_ph: float, kappa: float, mu_ref: float,
                     rng: np.random.Generator) -> np.ndarray:
    s = n_ph / mu_ref
    k = rng.poisson(b_clean * s).astype(np.float64)
    e = rng.normal(0.0, kappa * np.sqrt(n_ph), size=b_clean.shape)
    return (k + e) / s
