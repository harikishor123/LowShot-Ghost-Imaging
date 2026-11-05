"""Shared code for the extension with Poisson bucket noise (MedMNIST+ BloodMNIST + PathMNIST, 64x64).

Same objects, Bernoulli(0.5) pattern sequences (same seeds) and calibrated DGI as the frozen noise-free
extension (outputs/extension_medmnist_microscopy/); the ONLY change is Poisson shot noise on every bucket value.

Noise model, per object and noise level f (fraction):
    photon scale  s = 2*sum(x) / (f^2 * sum(x^2))
    noisy bucket  B' = Poisson(s * B) / s
so the noise std sqrt(B/s) ~= f * sigma_B, with sigma_B = 0.5*sqrt(sum(x^2)) the spread of the noise-free bucket
signal over Bernoulli(0.5) patterns (mean bucket 0.5*sum(x)). The reference arm R = sum(P) stays noise-free.
Noise stream per acquisition seed and level: default_rng([seed, 1_000_003, round(100*f)]), drawn shot by shot,
so DGI-N is still the first N shots of the same 10,000-shot noisy sequence. f = 0 gives the noise-free DGI.

All writes go through safe_path(): only under OUT, never a file in either freeze manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ext_common import (DATA, EVAL_SEEDS, K, LOW, OBJECTS_NPZ, SELECTION_SEEDS, SHOTS, SUBSETS, load_objects, md5,  # noqa: F401
                        pool_seed)
from src.ghost_imaging import pattern_batches, pattern_variance

ROOT = Path(__file__).resolve().parent
FROZEN_EXT = ROOT / "outputs" / "extension_medmnist_microscopy"
OUT = ROOT / "outputs" / "extension_medmnist_microscopy_poisson"
PROTOCOL = OUT / "protocol.json"
OBJECTS_MD5 = "73e8ce929c5a7c67111e028032e528c1"
LEVELS = {"p05": 0.05, "p10": 0.10, "p30": 0.30}      # noise std as a fraction of the bucket-signal spread
NOISE_TAG = 1_000_003
BATCH = 100                                           # same pattern stream as batch 500 (consecutive draws)


def pool_npy(level: str) -> Path:
    return OUT / "pool" / f"pool_train_{level}.npy"


def selection_npz() -> Path:
    return OUT / "pool" / "selection_val.npz"


def _frozen_paths() -> set[Path]:
    out = set()
    for man in (ROOT / "FREEZE_MANIFEST_phase1_phase2.md5", FROZEN_EXT / "FREEZE_MANIFEST_extension.md5"):
        out |= {(ROOT / line.split(None, 1)[1].strip().lstrip("*")).resolve()
                for line in man.read_text().splitlines() if line.strip()}
    return out


def safe_path(p: Path) -> Path:
    p = Path(p).resolve()
    if not p.is_relative_to(OUT.resolve()):
        raise PermissionError(f"Poisson-noise extension code may only write under {OUT}: {p}")
    if p in _frozen_paths():
        raise PermissionError(f"refusing to write a frozen file: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_json(p: Path, obj) -> None:
    safe_path(p).write_text(json.dumps(obj, indent=2, default=float))


def photon_scale(objects: np.ndarray, f: float) -> np.ndarray:
    x = objects.reshape(len(objects), -1).astype(np.float64)
    return 2.0 * x.sum(1) / (f * f * (x * x).sum(1))


class NoisyStreamingDGI:
    """StreamingDGI (src/reconstruction.py) for several noise levels at once, one pass over each pattern batch.

    Same accumulated sums and estimate formula; the bucket of every object is replaced by its Poisson version
    for each level. objects (M, H, W); fs list of noise fractions (0 = noise-free).
    """

    def __init__(self, objects: np.ndarray, fs: list[float], seed: int):
        self.obj = objects.reshape(objects.shape[0], -1).astype(np.float64)
        m, d = self.obj.shape
        self.fs, self.L, self.M = list(fs), len(fs), m
        self.scale = [photon_scale(objects, f) if f > 0 else None for f in fs]
        self.rngs = [np.random.default_rng([seed, NOISE_TAG, int(round(100 * f))]) if f > 0 else None for f in fs]
        self.n = 0
        self.sum_b = np.zeros((self.L, m))
        self.sum_r = 0.0
        self.acc = np.zeros((self.L * m + 2, d))        # rows: [B'_level,object ..., R, 1] . P

    def update(self, pattern_batch: np.ndarray) -> None:
        p = pattern_batch.reshape(pattern_batch.shape[0], -1).astype(np.float64)
        b = p @ self.obj.T                                        # (batch, M), noise-free buckets
        cols = []
        for li in range(self.L):
            bn = b if self.scale[li] is None else self.rngs[li].poisson(b * self.scale[li]) / self.scale[li]
            self.sum_b[li] += bn.sum(axis=0)
            cols.append(bn)
        r = p.sum(axis=1)
        self.sum_r += r.sum()
        w = np.concatenate(cols + [r[:, None], np.ones((len(p), 1))], axis=1)
        self.acc += w.T @ p
        self.n += len(p)

    def estimate(self, pattern_var: float) -> np.ndarray:
        """(L, M, D) unclipped calibrated estimates, identical formula to StreamingDGI.estimate."""
        n, m = self.n, self.M
        mean_b, mean_r = self.sum_b / n, self.sum_r / n
        mean_p = self.acc[-1] / n
        k = mean_b / mean_r                                        # (L, M)
        cov_bp = self.acc[:-2].reshape(self.L, m, -1) / n - mean_b[..., None] * mean_p
        cov_rp = self.acc[-2] / n - mean_r * mean_p
        dgi = cov_bp - k[..., None] * cov_rp
        return dgi / pattern_var + k[..., None]


def noisy_dgi_readouts(objects: np.ndarray, seed: int, fs: list[float], shots=SHOTS) -> np.ndarray:
    """(L, M, len(shots), 64, 64) float32: DGI of each noise level read out at `shots`, clipped to [0, 1]."""
    shots = list(shots)
    stream = NoisyStreamingDGI(objects, fs, seed)
    out = np.empty((len(fs), len(objects), len(shots), 64, 64), np.float32)
    for pats in pattern_batches(max(shots), objects.shape[1:], seed, BATCH, 0.5):
        stream.update(pats)
        if stream.n in shots:
            out[:, :, shots.index(stream.n)] = np.clip(stream.estimate(pattern_variance(0.5)), 0, 1).reshape(len(fs), -1, 64, 64)
    return out
