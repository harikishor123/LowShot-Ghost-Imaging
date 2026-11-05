#!/usr/bin/env python3
"""Data pipeline for photon-limited reconstruction (STEP 5).

Pairs every stored noisy sample (data/v2/photon_sim/<split>/P<PPPP>/<id>.npz, counts N) with
its own clean reference (data/v2/reference/<id>.npz) via samples_<split>.csv, and checks the
image id, split and shape agree. All tensors are LINEAR LIGHT:

    x = N / P                 noisy input (unclipped)
    y = srgb_to_linear(ref)   clean target in [0,1]
    m = FOV mask              1 = valid specimen pixel

Model A receives ONLY x (3 channels). P / level / metadata are kept for evaluation only.

Patch sampling (training):
  * fixed size PATCH x PATCH (256); every processed image is >= 559 x 579 px, so it fits.
  * candidate top-left corners on a STRIDE-px grid whose window has >= MIN_VALID of its
    pixels inside the FOV mask (computed with an integral image); a corner is drawn
    uniformly from the candidates of the chosen image, then jittered within the stride
    (keeping validity) so all positions are reachable.
  * the loader walks through shuffled permutations of all (image, level) training samples,
    so every sample is visited once per pass; PATCHES_PER_LOAD patches are cut from each
    loaded sample, and a shuffle buffer mixes patches from many images/levels in a batch.
  * augmentation: one of the 8 dihedral transforms (flips / 90-degree rotations), applied
    identically to input, target and mask. No intensity changes, no extra noise.
Validation: a FIXED patch set (VAL_PATCHES_PER_SAMPLE deterministic patches per val sample).
"""

from __future__ import annotations

import csv
import queue
import threading
from pathlib import Path

import numpy as np

from color_space import decode_uint16, srgb_to_linear

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data" / "v2"
SIM = V2 / "photon_sim"


# --------------------------------------------------------------------------------- index
def read_samples(split: str) -> list[dict]:
    rows = list(csv.DictReader(open(SIM / "metadata" / f"samples_{split}.csv", encoding="utf-8")))
    ids = set((V2 / "splits" / split / "ids.txt").read_text(encoding="utf-8").split())
    for r in rows:
        if r["split"] != split or r["image_id"] not in ids:
            raise ValueError(f"split isolation violated: {r['sample_id']}")
        if Path(r["clean_reference_path"]).stem != r["image_id"] or Path(r["noisy_counts_path"]).stem != r["image_id"]:
            raise ValueError(f"noisy/clean pairing mismatch: {r['sample_id']}")
        r["photon_budget_P"] = float(r["photon_budget_P"])
    return rows


def load_reference(image_id: str):
    with np.load(V2 / "reference" / f"{image_id}.npz") as z:
        return z["srgb16"], z["mask"].astype(bool)


def load_noisy(row: dict, shape=None) -> np.ndarray:
    with np.load(ROOT / row["noisy_counts_path"]) as z:
        c = z["counts"]
        P = float(z["photons"])
    if abs(P - row["photon_budget_P"]) > 1e-6:
        raise ValueError(f"P mismatch in {row['sample_id']}")
    if shape is not None and c.shape != shape:
        raise ValueError(f"shape mismatch noisy {c.shape} vs clean {shape} in {row['sample_id']}")
    return c.astype(np.float32) / np.float32(P)


def load_pair(row: dict):
    """Full-image (x_noisy, y_clean, mask) in linear light for evaluation."""
    s16, m = load_reference(row["image_id"])
    y = srgb_to_linear(decode_uint16(s16))
    x = load_noisy(row, shape=s16.shape)
    return x, y, m


# ------------------------------------------------------------------------ patch geometry
def valid_corners(mask: np.ndarray, patch: int, stride: int, min_valid: float) -> np.ndarray:
    H, W = mask.shape
    if H < patch or W < patch:
        return np.zeros((0, 2), int)
    ii = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ys = np.arange(0, H - patch + 1, stride)
    xs = np.arange(0, W - patch + 1, stride)
    if ys[-1] != H - patch:
        ys = np.append(ys, H - patch)
    if xs[-1] != W - patch:
        xs = np.append(xs, W - patch)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    s = ii[Y + patch, X + patch] - ii[Y, X + patch] - ii[Y + patch, X] + ii[Y, X]
    ok = s >= min_valid * patch * patch
    return np.stack([Y[ok], X[ok]], 1)


def dihedral(a: np.ndarray, k: int) -> np.ndarray:
    a = np.rot90(a, k % 4, axes=(0, 1))
    return a[:, ::-1] if k >= 4 else a


# ------------------------------------------------------------------------ training stream
class TrainPatchStream:
    """Infinite, seeded stream of augmented (x, y_with_mask) training patches."""

    def __init__(self, cfg: dict, seed: int):
        self.cfg = cfg
        self.rows = read_samples("train")
        self.patch, self.stride, self.min_valid = cfg["patch_size"], cfg["corner_stride"], cfg["min_valid_fraction"]
        self.k = cfg["patches_per_load"]
        self.rng = np.random.default_rng(seed)
        ids = sorted({r["image_id"] for r in self.rows})
        self.ref = {}
        self.corners = {}
        for i in ids:  # cache clean references (uint16 sRGB) + masks in RAM
            s16, m = load_reference(i)
            self.ref[i] = (s16, m)
            c = valid_corners(m, self.patch, self.stride, self.min_valid)
            if len(c) == 0:
                raise ValueError(f"no valid {self.patch}px patch in {i}")
            self.corners[i] = c
        self.n_corners = {i: len(c) for i, c in self.corners.items()}

    def _patches_from(self, row: dict, rng_state: int):
        local = np.random.default_rng(rng_state)
        s16, m = self.ref[row["image_id"]]
        x_full = load_noisy(row, shape=s16.shape)
        out = []
        for _ in range(self.k):
            y0, x0 = self._corner_local(row["image_id"], m, local)
            sl = (slice(y0, y0 + self.patch), slice(x0, x0 + self.patch))
            xp = x_full[sl]
            yp = srgb_to_linear(decode_uint16(s16[sl]))
            mp = m[sl].astype(np.float32)[..., None]
            t = int(local.integers(8)) if self.cfg["augment_dihedral"] else 0
            xp, yp, mp = dihedral(xp, t), dihedral(yp, t), dihedral(mp, t)
            out.append((np.ascontiguousarray(xp, np.float32),
                        np.ascontiguousarray(np.concatenate([yp, mp], -1), np.float32)))
        return out

    def _corner_local(self, image_id, m, local):
        H, W = m.shape
        c = self.corners[image_id]
        for _ in range(20):
            y, x = c[local.integers(len(c))]
            y = int(np.clip(y + local.integers(-self.stride // 2, self.stride // 2 + 1), 0, H - self.patch))
            x = int(np.clip(x + local.integers(-self.stride // 2, self.stride // 2 + 1), 0, W - self.patch))
            if m[y:y + self.patch, x:x + self.patch].mean() >= self.min_valid:
                return y, x
        return int(c[0][0]), int(c[0][1])

    def generator(self, n_workers: int = 4, shuffle_buffer: int = 256):
        """Yield (x, y4) forever. Loading runs in background threads; order is seeded per load."""
        jobs: queue.Queue = queue.Queue(maxsize=64)
        results: queue.Queue = queue.Queue(maxsize=64)

        def producer():
            while True:
                for idx in self.rng.permutation(len(self.rows)):
                    jobs.put((self.rows[idx], int(self.rng.integers(2 ** 31))))

        def worker():
            while True:
                row, st = jobs.get()
                results.put(self._patches_from(row, st))

        threading.Thread(target=producer, daemon=True).start()
        for _ in range(n_workers):
            threading.Thread(target=worker, daemon=True).start()
        buf, mix = [], np.random.default_rng(int(self.rng.integers(2 ** 31)))
        while True:
            buf.extend(results.get())
            while len(buf) >= shuffle_buffer:
                yield buf.pop(int(mix.integers(len(buf))))


# ------------------------------------------------------------------------ validation set
def fixed_validation_patches(cfg: dict, seed: int):
    """Deterministic patch set from VALIDATION references only. Returns x, y4, P, ids."""
    rows = read_samples("val")
    patch, n = cfg["patch_size"], cfg["val_patches_per_sample"]
    xs, ys, Ps, ids = [], [], [], []
    cache = {}
    for r in rows:
        if r["image_id"] not in cache:
            s16, m = load_reference(r["image_id"])
            cache = {r["image_id"]: (s16, m, valid_corners(m, patch, cfg["corner_stride"], cfg["min_valid_fraction"]))}
        s16, m, c = cache[r["image_id"]]
        # same patch positions for every level of an image -> per-level comparison is like-for-like
        rng = np.random.default_rng([seed, int.from_bytes(r["image_id"].encode(), "little") % (2 ** 32)])
        sel = c[rng.choice(len(c), size=min(n, len(c)), replace=False)]
        x_full = load_noisy(r, shape=s16.shape)
        for y0, x0 in sel:
            sl = (slice(y0, y0 + patch), slice(x0, x0 + patch))
            xs.append(x_full[sl])
            ys.append(np.concatenate([srgb_to_linear(decode_uint16(s16[sl])), m[sl].astype(np.float32)[..., None]], -1))
            Ps.append(r["photon_budget_P"])
            ids.append(r["image_id"])
    return (np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32), np.array(Ps, np.float32), ids)
