#!/usr/bin/env python3
"""Final experiment, step 1: training pool + checkpoint-selection set (no test data).

    python final_build_pool.py

Training pool: for every training image i and realisation k = 0..31, one independent random pattern
sequence (seed 20,000,000 + 1000*k + i) is measured for 10,000 shots and read out with the Phase 2A DGI
pipeline at N = 500, 1000, 2000, 5000, 10000 (calibrated, clipped to [0, 1]). Prefixes are nested.
Stored as pool_train.npy, shape (341, 32, 5, 64, 64) float32.

Selection set: the 75 validation images measured with shared sequences 2040-2042 (as Models C/D),
read out at the same N. Stored as selection_val.npz.

Before building, the pipeline is checked bit-exactly against the existing seed-2026 reference file.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.ghost_imaging import load_object, pattern_batches, pattern_variance  # noqa: E402
from src.gi_unet_data import load_pairs, read_split_ids  # noqa: E402
from src.reconstruction import StreamingDGI  # noqa: E402

OUT = ROOT / "outputs" / "phase_final_low_shot_to_10k_reference"
SHOTS = [500, 1000, 2000, 5000, 10000]
K = 32
POOL_SEED_BASE = 20_000_000
SELECTION_SEEDS = [2040, 2041, 2042]


def pool_seed(k: int, i: int) -> int:
    return POOL_SEED_BASE + 1000 * k + i


def dgi_readouts(objects: np.ndarray, seed: int) -> np.ndarray:
    """(M, 5, 64, 64): Phase 2A DGI of `objects` (shared sequence `seed`) read out at SHOTS, clipped."""
    stream = StreamingDGI(objects)
    out = np.empty((len(objects), len(SHOTS), 64, 64), np.float32)
    for pats in pattern_batches(max(SHOTS), objects.shape[1:], seed, 500, 0.5):
        stream.update(pats)
        if stream.n in SHOTS:
            est, _ = stream.estimate(pattern_variance(0.5))
            out[:, SHOTS.index(stream.n)] = np.clip(est, 0, 1).reshape(-1, 64, 64)
    return out


def _worker(args):
    obj, i, ks = args
    return i, ks, np.stack([dgi_readouts(obj[None], pool_seed(k, i))[0] for k in ks])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ids = read_split_ids()
    _, _, id_tr = load_pairs("train", 1000)
    _, _, id_va = load_pairs("val", 1000)
    assert len(id_tr) == 341 and len(id_va) == 75 and not (set(id_tr) | set(id_va)) & set(ids["test"])
    ref = ROOT / "data" / "v2" / "reference"
    obj_tr = np.stack([load_object(ref / f"{i}.npz", 64)[0] for i in id_tr])
    obj_va = np.stack([load_object(ref / f"{i}.npz", 64)[0] for i in id_va])

    # ---- pipeline == existing reference (seed 2026), bit-exact, train + val images --------------
    checks = {}
    chk = dgi_readouts(obj_va, 2026)
    diffs = {n: float(np.abs(chk[:, SHOTS.index(n)] - load_pairs("val", n)[0][:, 0]).max()) for n in (500, 1000, 5000, 10000)}
    checks["val_seed2026_pipeline_vs_existing_reference_max_abs_diff"] = diffs
    chk_tr = dgi_readouts(obj_tr[:20], 2026)
    checks["train20_seed2026_pipeline_vs_existing_max_abs_diff"] = float(max(
        np.abs(chk_tr[:, SHOTS.index(n)] - load_pairs("train", n)[0][:20, 0]).max() for n in (500, 1000, 5000, 10000)))
    single = dgi_readouts(obj_tr[:1], 2026)[0]
    checks["single_image_sequence_equals_shared_sequence_max_abs_diff"] = float(np.abs(single - chk_tr[0]).max())
    assert max(diffs.values()) <= 1e-9, "pipeline does not reproduce the existing reference"  # fp-rounding level only
    print(json.dumps(checks, indent=1), flush=True)

    # ---- selection set -----------------------------------------------------------------------
    sel = np.stack([dgi_readouts(obj_va, s) for s in SELECTION_SEEDS], axis=1)          # (75, 3, 5, 64, 64)
    np.savez_compressed(OUT / "selection_val.npz", image_id=np.array(id_va), seeds=np.array(SELECTION_SEEDS),
                        shots=np.array(SHOTS), dgi=sel, target_clean=obj_va.astype(np.float32))

    # ---- training pool (parallel, one sequence per image per realisation) ---------------------
    pool = np.lib.format.open_memmap(OUT / "pool_train.npy", mode="w+", dtype=np.float32,
                                     shape=(len(id_tr), K, len(SHOTS), 64, 64))
    jobs = [(obj_tr[i], i, list(range(K))) for i in range(len(id_tr))]
    done = 0
    with ProcessPoolExecutor(max_workers=12) as ex:
        for i, ks, arr in ex.map(_worker, jobs, chunksize=2):
            pool[i, ks] = arr
            done += 1
            if done % 25 == 0:
                print(f"pool: {done}/{len(id_tr)} images, {time.time() - t0:.0f}s", flush=True)
    pool.flush()
    # spot-check a pool entry against an independent recomputation
    i, k = 123, 17
    rec = dgi_readouts(obj_tr[i][None], pool_seed(k, i))[0]
    checks["pool_spot_check_max_abs_diff"] = float(np.abs(rec - pool[i, k]).max())
    checks["pool_finite"] = bool(np.isfinite(pool[::17]).all())
    meta = {"shape": list(pool.shape), "axes": "train_image, realisation k, shots, H, W", "shots": SHOTS, "K": K,
            "seed_rule": "20,000,000 + 1000*k + train_index", "train_ids": id_tr, "selection_seeds": SELECTION_SEEDS,
            "checks": checks, "seconds": round(time.time() - t0, 1)}
    (OUT / "pool_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(checks, indent=1), f"\n{meta['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
