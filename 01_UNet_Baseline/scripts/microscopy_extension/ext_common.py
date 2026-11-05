"""Shared paths, guards and helpers for the extension: broader microscopy dataset (MedMNIST+).

All extension scripts (ext_*.py) write ONLY under DATA and OUT below. Every write path goes through
safe_path(), which also refuses any file listed in FREEZE_MANIFEST_phase1_phase2.md5, so the frozen
onion results (Phase 1 and Phase 2) cannot be touched by this code.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "extension_medmnist"
OUT = ROOT / "outputs" / "extension_medmnist_microscopy"
OBJECTS_NPZ = DATA / "processed" / "objects_64.npz"
POOL_NPY = OUT / "pool" / "pool_train.npy"
SELECTION_NPZ = OUT / "pool" / "selection_val.npz"
PROTOCOL = OUT / "protocol.json"

SHOTS = [500, 1000, 2000, 5000, 10000]
LOW = [500, 1000, 2000, 5000]
SUBSETS = {"blood": "bloodmnist", "path": "pathmnist"}
K = 32                                    # independent pattern sequences per training image (as onion)
POOL_SEED_BASE = 30_000_000               # seed = 30,000,000 + 10,000*k + train_index (k < 32, index < 10,000)
SELECTION_SEEDS = [2040, 2041, 2042]
EVAL_SEEDS = [2026, 2027, 2028, 2029]
RETEST_OFFSET = 7000                      # independent 10k acquisition = eval seed + 7000 (as onion)
TARGETS = {"dgi10k": "AI", "clean": "AI-original"}   # target key -> model label prefix


def pool_seed(k: int, i: int) -> int:
    assert 0 <= k < K and 0 <= i < 10_000
    return POOL_SEED_BASE + 10_000 * k + i


def _frozen_paths() -> set[Path]:
    man = ROOT / "FREEZE_MANIFEST_phase1_phase2.md5"
    return {(ROOT / line.split(None, 1)[1].strip().lstrip("*")).resolve()
            for line in man.read_text().splitlines() if line.strip()}


def safe_path(p: Path) -> Path:
    """Return p (resolved) if it lies under DATA or OUT and is not a frozen file; create its parent."""
    p = Path(p).resolve()
    if not any(p.is_relative_to(d.resolve()) for d in (DATA, OUT)):
        raise PermissionError(f"extension code may only write under {DATA} or {OUT}: {p}")
    if p in _frozen_paths():
        raise PermissionError(f"refusing to write a frozen file: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(p: Path, obj) -> None:
    safe_path(p).write_text(json.dumps(obj, indent=2, default=float))


def load_objects(splits=("train", "val")) -> dict:
    """Clean 64x64 GI objects and metadata for the requested splits (test only when asked for)."""
    with np.load(OBJECTS_NPZ, allow_pickle=False) as z:
        split = z["split"].astype(str)
        sel = np.isin(split, list(splits))
        return {"image_id": z["image_id"].astype(str)[sel], "subset": z["subset"].astype(str)[sel],
                "split": split[sel], "label": z["label"][sel], "object": z["object"][sel]}


def set_threads(n: int) -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, str(n))
