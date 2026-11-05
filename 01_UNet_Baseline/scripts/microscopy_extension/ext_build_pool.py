#!/usr/bin/env python3
"""Extension (broader microscopy dataset), step 2: training pool + checkpoint-selection set (no test data).

    python ext_build_pool.py [--workers 30]

Identical simulation and DGI read-out as the onion study (final_build_pool.dgi_readouts, imported unchanged):
Bernoulli(0.5) 64x64 patterns, noise-free buckets, calibrated DGI read out at N = 500/1000/2000/5000/10000,
clipped to [0, 1], nested prefixes of one 10,000-shot sequence.

Training pool: every training image i gets K = 32 independent sequences, seed 30,000,000 + 10,000*k + i
(the onion rule 20,000,000 + 1000*k + i would collide for i >= 1000). Stored as pool/pool_train.npy,
shape (n_train, 32, 5, 64, 64) float32. Selection set: validation images x shared sequences 2040-2042.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ext_common import (K, OUT, POOL_NPY, PROTOCOL, SELECTION_NPZ, SELECTION_SEEDS, SHOTS, load_objects, md5,  # noqa: E402
                        pool_seed, safe_path, write_json)
from final_build_pool import dgi_readouts  # noqa: E402

_OBJ = None


def _init(obj):
    global _OBJ
    _OBJ = obj


def _worker(i):
    return i, np.stack([dgi_readouts(_OBJ[i][None], pool_seed(k, i))[0] for k in range(K)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=30)
    args = ap.parse_args()
    assert PROTOCOL.exists(), "write protocol.json (ext_write_protocol.py) before building the training pool"
    t0 = time.time()
    d = load_objects(("train", "val"))
    tr, va = d["split"] == "train", d["split"] == "val"
    obj_tr, obj_va = d["object"][tr], d["object"][va]
    ids_tr, ids_va = d["image_id"][tr], d["image_id"][va]
    checks = {"n_train": int(tr.sum()), "n_val": int(va.sum())}

    sel = np.stack([dgi_readouts(obj_va, s) for s in SELECTION_SEEDS], axis=1)          # (n_val, 3, 5, 64, 64)
    np.savez_compressed(safe_path(SELECTION_NPZ), image_id=ids_va, subset=d["subset"][va], seeds=np.array(SELECTION_SEEDS),
                        shots=np.array(SHOTS), dgi=sel, target_clean=obj_va.astype(np.float32))
    print(f"selection set {sel.shape} ({time.time() - t0:.0f}s)", flush=True)

    pool = np.lib.format.open_memmap(safe_path(POOL_NPY), mode="w+", dtype=np.float32,
                                     shape=(len(obj_tr), K, len(SHOTS), 64, 64))
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(obj_tr,)) as ex:
        for i, arr in ex.map(_worker, range(len(obj_tr)), chunksize=4):
            pool[i] = arr
            done += 1
            if done % 500 == 0:
                print(f"pool: {done}/{len(obj_tr)} images, {time.time() - t0:.0f}s", flush=True)
    pool.flush()

    rng = np.random.default_rng(7)
    spots = []
    for i, k in zip(rng.integers(0, len(obj_tr), 3), rng.integers(0, K, 3)):
        rec = dgi_readouts(obj_tr[i][None], pool_seed(int(k), int(i)))[0]
        spots.append(float(np.abs(rec - pool[i, k]).max()))
    checks["pool_spot_checks_max_abs_diff"] = spots
    checks["pool_finite"] = bool(all(np.isfinite(pool[j]).all() for j in range(0, len(obj_tr), 37)))
    checks["selection_finite"] = bool(np.isfinite(sel).all())
    meta = {"shape": list(pool.shape), "axes": "train_image, realisation k, shots, H, W", "shots": SHOTS, "K": K,
            "seed_rule": "30,000,000 + 10,000*k + train_index", "train_ids": ids_tr.tolist(),
            "selection_seeds": SELECTION_SEEDS, "checks": checks, "seconds": round(time.time() - t0, 1),
            "selection_npz_md5": md5(SELECTION_NPZ)}
    write_json(OUT / "pool" / "pool_meta.json", meta)
    print(json.dumps(checks, indent=1), f"\n{meta['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
