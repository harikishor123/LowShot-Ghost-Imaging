"""
Data loader for ghost_imaging_v2.  Reads from the FROZEN extension data pool.

Does NOT use pandas.  Does NOT modify or import any file from the frozen project.
Reads the frozen NPZ/NPY files directly by absolute path.

Data structure (from frozen project):
    pool_train.npy        (7000, 32, 5, 64, 64) float32
                           axes: image × realisation_k × shot_idx × H × W
    pool_meta.json         maps axis 0 → image_id strings, lists shots/seeds
    selection_val.npz      keys: dgi (1000,3,5,64,64), target_clean (1000,64,64),
                                 seeds (3,), shots (5,), image_id (1000,)
    objects_64.npz         keys: image_id (10000,), object (10000,64,64),
                                 split ('train'/'val'/'test'), subset ('blood'/'path')

Usage:
    from src.dataset import GIDataset
    train_ds = GIDataset(frozen_root, split='train', shot_count=1000)
    val_ds   = GIDataset(frozen_root, split='val',   shot_count=1000)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# Shot-count index table (matches frozen pool axes)
SHOTS     = [500, 1000, 2000, 5000, 10000]
SHOT_IDX  = {s: i for i, s in enumerate(SHOTS)}
# Training-pool seed rule (from pool_meta.json)
POOL_SEED_BASE = 30_000_000  # seed = BASE + 10_000*k + train_index


class GIDataset(Dataset):
    """
    Returns (dgi, target, meta) tuples for supervised ghost-imaging reconstruction.

    dgi    — calibrated DGI-N image,  shape [1, 64, 64], float32, range [0, 1]
    target — clean object O,           shape [1, 64, 64], float32, range [0, 1]
    meta   — dict: image_id, seed, shot_count, train_index, k (realisation)

    Training samples:
        len = 7000 items per epoch.  random_k=True picks a different realisation
        k at each __getitem__ call, providing 32× augmentation variety without
        inflating __len__ (which would bloat scheduler step counts).
        To fix k (e.g. for reproducibility) set random_k=False.

    Val samples:
        len = 1000 images × 3 seeds = 3000
        Uses selection_val.npz directly.
    """

    def __init__(
        self,
        frozen_root: str | Path,
        split: str = "train",
        shot_count: int = 1000,
        random_k: bool = True,
        seed: int = 0,
    ) -> None:
        """
        Args:
            frozen_root:  Absolute path to the frozen project root.
                          e.g. /home/ss_students/Downloads/Onion-CNN-Final.v1-...
            split:        'train' or 'val'.  Never pass 'test' here.
            shot_count:   One of [500, 1000, 2000, 5000].
                          (10000 is the upper reference; not used for training.)
            random_k:     If True (train), pick a random realisation k per item.
                          If False, always use k=0.
            seed:         RNG seed for random_k draws.
        """
        assert split in ("train", "val"), "Use split='train' or 'val'. Never use test here."
        assert shot_count in SHOTS, f"shot_count must be one of {SHOTS}"

        self.split      = split
        self.shot_count = shot_count
        self.shot_idx   = SHOT_IDX[shot_count]
        self.random_k   = random_k and (split == "train")
        # seed stored only for reference in meta; k draws use torch.randint (B3 fix)

        root = Path(frozen_root)
        pool_dir = root / "outputs" / "extension_medmnist_microscopy" / "pool"
        data_dir = root / "data"  / "extension_medmnist" / "processed"

        self._check_paths(pool_dir, data_dir)

        if split == "train":
            self._load_train(pool_dir, data_dir)
        else:
            self._load_val(pool_dir)

    # ──────────────────────────────────────────────────────────
    # Loading
    # ──────────────────────────────────────────────────────────

    def _check_paths(self, pool_dir: Path, data_dir: Path) -> None:
        required = [
            pool_dir / "pool_train.npy",
            pool_dir / "pool_meta.json",
            pool_dir / "selection_val.npz",
            data_dir / "objects_64.npz",
        ]
        for p in required:
            if not p.exists():
                raise FileNotFoundError(
                    f"Frozen data file not found: {p}\n"
                    "Make sure frozen_root points to the correct project directory."
                )

    def _load_train(self, pool_dir: Path, data_dir: Path) -> None:
        # DGI pool: (7000, 32, 5, 64, 64)
        self._pool = np.load(pool_dir / "pool_train.npy")          # float32
        assert self._pool.ndim == 5 and self._pool.shape[2] == 5, \
            f"Unexpected pool_train shape: {self._pool.shape}"
        n_images, self._K, _, H, W = self._pool.shape

        # Metadata: image_id order
        with open(pool_dir / "pool_meta.json", "r") as f:
            meta = json.load(f)
        train_ids = meta["train_ids"]
        assert len(train_ids) == n_images, \
            f"pool_meta train_ids ({len(train_ids)}) != pool_train axis-0 ({n_images})"

        # Ground-truth objects: (10000, 64, 64)
        obj_data   = np.load(data_dir / "objects_64.npz")
        all_ids    = obj_data["image_id"]          # shape (10000,), dtype str
        all_objects = obj_data["object"]           # shape (10000, 64, 64), float64

        # Build lookup: image_id string → row index in objects_64.npz
        id_to_idx = {str(img_id): i for i, img_id in enumerate(all_ids)}
        missing = [tid for tid in train_ids if tid not in id_to_idx]
        if missing:
            raise KeyError(f"{len(missing)} train_ids not found in objects_64.npz: {missing[:3]}")

        # Build targets array aligned to pool axis-0
        self._targets = np.empty((n_images, H, W), dtype=np.float32)
        for i, tid in enumerate(train_ids):
            self._targets[i] = all_objects[id_to_idx[tid]].astype(np.float32)

        self._train_ids = train_ids
        self._n = n_images

    def _load_val(self, pool_dir: Path) -> None:
        val = np.load(pool_dir / "selection_val.npz")
        # dgi: (1000, 3, 5, 64, 64)  seeds: (3,)  shots: (5,)
        self._dgi_val     = val["dgi"]           # float32
        self._targets_val = val["target_clean"]  # (1000, 64, 64) float32
        self._ids_val     = val["image_id"]
        n_images, n_seeds, _, H, W = self._dgi_val.shape
        self._n       = n_images * n_seeds
        self._n_seeds = n_seeds
        # Pre-build (image_idx, seed_idx) pairs
        self._pairs = [(i, s) for i in range(n_images) for s in range(n_seeds)]

    # ──────────────────────────────────────────────────────────
    # Dataset protocol
    # ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        if self.split == "train":
            return self._get_train(idx)
        return self._get_val(idx)

    def _get_train(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        # idx = image index (0 to 6999)
        # k drawn randomly (or 0 if fixed)
        image_idx = idx % self._n
        # torch.randint is worker-safe: each DataLoader worker has its own PyTorch
        # RNG seeded by worker_init_fn, so workers don't share RNG state (B3 fix).
        k = int(torch.randint(0, self._K, (1,)).item()) if self.random_k else 0

        dgi    = self._pool[image_idx, k, self.shot_idx]       # (64, 64), float32
        target = self._targets[image_idx]                       # (64, 64), float32

        seed = POOL_SEED_BASE + 10_000 * k + image_idx
        meta = {
            "image_id":    self._train_ids[image_idx],
            "seed":        seed,
            "shot_count":  self.shot_count,
            "train_index": image_idx,
            "k":           k,
        }
        return (
            torch.from_numpy(dgi).unsqueeze(0),     # [1, 64, 64]
            torch.from_numpy(target).unsqueeze(0),  # [1, 64, 64]
            meta,
        )

    def _get_val(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        image_idx, seed_idx = self._pairs[idx]
        dgi    = self._dgi_val[image_idx, seed_idx, self.shot_idx]  # (64, 64)
        target = self._targets_val[image_idx]                        # (64, 64)
        meta = {
            "image_id":   str(self._ids_val[image_idx]),
            "seed":       seed_idx,
            "shot_count": self.shot_count,
        }
        return (
            torch.from_numpy(dgi.copy()).unsqueeze(0),
            torch.from_numpy(target.copy()).unsqueeze(0),
            meta,
        )

    # ──────────────────────────────────────────────────────────
    # DataLoader worker RNG seeding  (B3 fix)
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        """Seed each DataLoader worker's numpy RNG independently.

        PyTorch auto-seeds each worker's torch RNG as initial_seed + worker_id.
        We mirror that for numpy (used in _load_train/_load_val file reads).
        Pass this to DataLoader(..., worker_init_fn=GIDataset.worker_init_fn).
        """
        worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
        np.random.seed(worker_seed)

    # ──────────────────────────────────────────────────────────
    # Optional: build patterns for L_meas (slow — O(N×4096) per image)
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def build_patterns_flat(seed: int, N: int, shape: tuple = (64, 64)) -> np.ndarray:
        """
        Regenerate the first N illumination patterns from a known seed.
        Returns float32 array [N, H*W].

        WARNING: generating 1000×4096 patterns per image per batch step adds
        ~100 ms CPU overhead (batch=16).  Use sparingly in E2.  For E3+ diffusion
        training, pre-generate and cache the bucket values during data prep.

        Seed rule: POOL_SEED_BASE + 10_000 * k + train_index
        """
        H, W  = shape
        rng   = np.random.default_rng(seed)
        P     = (rng.random((N, H * W)) < 0.5).astype(np.float32)
        return P  # [N, H*W]

    @staticmethod
    def compute_buckets(patterns_flat: np.ndarray, obj: np.ndarray) -> np.ndarray:
        """
        B = P @ o.reshape(-1)
        patterns_flat: [N, H*W] float32
        obj:           [H, W]   float32
        Returns B: [N] float32
        """
        return patterns_flat @ obj.reshape(-1).astype(np.float32)
