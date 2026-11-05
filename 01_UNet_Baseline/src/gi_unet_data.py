"""Paired (DGI-N input, original target) data for the Phase 2B U-Net.

Source: outputs/phase2a_dgi_dataset/reconstructions/dgi_estimates_seed2026_64px.npz
(written by run_dgi_dataset.py). Nothing is re-simulated here.

Numerical convention (identical to Phase 2A metrics):
    target : the GI object O, linear-light luminance, 64x64, float32, already in [0, 1]
             (no scaling, no normalisation)
    input  : calibrated DGI estimate O_hat = DGI_N/0.25 + <B>_N/<R>_N, clipped to [0, 1]
             (the exact image Phase 2A evaluated; no per-image min-max, no rescaling)
Both live on the same physical [0, 1] transmission scale, so an MSE between them
is meaningful and the U-Net output is directly comparable to the target.

The test split is not returned unless allow_test=True is passed explicitly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
DGI_NPZ = ROOT / "outputs" / "phase2a_dgi_dataset" / "reconstructions" / "dgi_estimates_seed2026_64px.npz"
DGI_CSV = ROOT / "outputs" / "phase2a_dgi_dataset" / "metrics" / "dgi_per_image_seed2026.csv"
SPLIT_DIR = ROOT / "data" / "v2" / "splits"
EXPECTED = {"train": 341, "val": 75, "test": 74}


def read_split_ids() -> dict[str, list[str]]:
    ids = {s: (SPLIT_DIR / s / "ids.txt").read_text().split() for s in EXPECTED}
    for s, n in EXPECTED.items():
        assert len(ids[s]) == n, f"{s}: {len(ids[s])} ids, expected {n}"
        assert len(set(ids[s])) == n, f"{s}: duplicate ids"
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(ids[a]) & set(ids[b])
        assert not overlap, f"{a}/{b} overlap: {sorted(overlap)[:5]}"
    return ids


def load_pairs(split: str, n_shots: int, allow_test: bool = False, npz_path: Path = DGI_NPZ):
    """Return (inputs (M,1,64,64) float32, targets (M,1,64,64) float32, ids list) for one split.

    Rows are ordered as in data/v2/splits/<split>/ids.txt. The input and the target of
    row j are taken from the same index of the same npz, selected by image_id.
    """
    if split == "test" and not allow_test:
        raise PermissionError("test split is held out; pass allow_test=True only for final evaluation")
    split_ids = read_split_ids()[split]
    # image_id/split were stored as object arrays by an earlier writer; this is our own file.
    with np.load(npz_path, allow_pickle=True) as z:
        npz_ids = z["image_id"].astype(str)
        npz_split = z["split"].astype(str)
        pos = {i: j for j, i in enumerate(npz_ids)}
        missing = [i for i in split_ids if i not in pos]
        assert not missing, f"ids missing from npz: {missing[:5]}"
        idx = np.array([pos[i] for i in split_ids])
        assert (npz_split[idx] == split).all(), "npz split label disagrees with ids.txt"
        assert list(npz_ids[idx]) == split_ids
        x = np.clip(z[f"estimate_unclipped_{n_shots}"][idx], 0.0, 1.0).astype(np.float32)
        y = z["object"][idx].astype(np.float32)
    return x[:, None], y[:, None], split_ids


def verify_alignment(x, y, ids, n_shots: int, tol_db: float = 1e-3) -> float:
    """Recompute PSNR(input, target) per image and compare with the Phase 2A CSV row of the
    SAME image_id. A mismatched pair would differ by whole dB. Returns max |diff| in dB."""
    ref = pd.read_csv(DGI_CSV)
    ref = ref[ref.measurements == n_shots].set_index("image_id")["psnr"]
    mse = ((x.astype(np.float64) - y.astype(np.float64)) ** 2).mean(axis=(1, 2, 3))
    psnr = 10 * np.log10(1.0 / mse)
    diff = np.abs(psnr - ref.loc[ids].values)
    assert diff.max() < tol_db, f"input/target alignment check failed: max diff {diff.max():.4f} dB"
    return float(diff.max())


class PairDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, ids: list[str]):
        assert x.shape == y.shape and x.shape[0] == len(ids)
        self.x, self.y, self.ids = torch.from_numpy(x), torch.from_numpy(y), ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.x[i], self.y[i], i
