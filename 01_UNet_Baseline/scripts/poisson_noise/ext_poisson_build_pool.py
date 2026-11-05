#!/usr/bin/env python3
"""Extension with Poisson bucket noise, step 2: training pools (one per noise level) + selection set. No test data.

    python ext_poisson_build_pool.py [--workers 30]

Same images, pattern seeds (30,000,000 + 10,000*k + i, K = 32) and DGI as the frozen noise-free extension; Poisson
noise on the buckets (ext_poisson_common). All three levels come from one pass over each pattern sequence.
Pools: pool/pool_train_{p05,p10,p30}.npy, shape (n_train, 32, 4, 64, 64) float32 at N = 500/1000/2000/5000
(the clean-target models need no 10k input). Selection set: validation images x seeds 2040-2042, all levels,
N = 500..10000.
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

from ext_poisson_common import (FROZEN_EXT, K, LEVELS, LOW, OBJECTS_MD5, OBJECTS_NPZ, OUT, PROTOCOL, SELECTION_SEEDS,  # noqa: E402
                                SHOTS, load_objects, md5, noisy_dgi_readouts, pool_npy, pool_seed, safe_path,
                                selection_npz, write_json)

FS = list(LEVELS.values())
_OBJ = None


def _init(obj):
    global _OBJ
    _OBJ = obj


def _worker(i):
    return i, np.stack([noisy_dgi_readouts(_OBJ[i][None], pool_seed(k, i), FS, LOW)[:, 0] for k in range(K)], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=30)
    args = ap.parse_args()
    assert PROTOCOL.exists(), "write protocol.json (ext_poisson_write_protocol.py) before building the pools"
    assert md5(OBJECTS_NPZ) == OBJECTS_MD5, "objects npz changed"
    t0 = time.time()
    d = load_objects(("train", "val"))
    tr, va = d["split"] == "train", d["split"] == "val"
    obj_tr, obj_va = d["object"][tr], d["object"][va]
    ids_tr, ids_va = d["image_id"][tr], d["image_id"][va]
    checks = {"n_train": int(tr.sum()), "n_val": int(va.sum())}

    sel = np.stack([noisy_dgi_readouts(obj_va, s, FS) for s in SELECTION_SEEDS], axis=2)     # (L, n_val, 3, 5, 64, 64)
    np.savez_compressed(safe_path(selection_npz()), image_id=ids_va, subset=d["subset"][va], seeds=np.array(SELECTION_SEEDS),
                        shots=np.array(SHOTS), levels=np.array(list(LEVELS)), dgi=sel, target_clean=obj_va.astype(np.float32))
    print(f"selection set {sel.shape} ({time.time() - t0:.0f}s)", flush=True)

    pools = [np.lib.format.open_memmap(safe_path(pool_npy(lv)), mode="w+", dtype=np.float32,
                                       shape=(len(obj_tr), K, len(LOW), 64, 64)) for lv in LEVELS]
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(obj_tr,)) as ex:
        for i, arr in ex.map(_worker, range(len(obj_tr)), chunksize=4):         # arr (L, K, 4, 64, 64)
            for li, pl in enumerate(pools):
                pl[i] = arr[li]
            done += 1
            if done % 500 == 0:
                print(f"pool: {done}/{len(obj_tr)} images, {time.time() - t0:.0f}s", flush=True)
    for pl in pools:
        pl.flush()

    rng = np.random.default_rng(7)
    frozen = np.load(FROZEN_EXT / "pool" / "pool_train.npy", mmap_mode="r")
    spots, spots0 = [], []
    for i, k in zip(rng.integers(0, len(obj_tr), 3), rng.integers(0, K, 3)):
        i, k = int(i), int(k)
        rec = noisy_dgi_readouts(obj_tr[i][None], pool_seed(k, i), FS, LOW)[:, 0]
        spots.append(float(max(np.abs(rec[li] - pools[li][i, k]).max() for li in range(len(FS)))))
        rec0 = noisy_dgi_readouts(obj_tr[i][None], pool_seed(k, i), [0.0], LOW)[0, 0]
        spots0.append(float(np.abs(rec0 - frozen[i, k, :len(LOW)]).max()))
    checks["pool_spot_checks_max_abs_diff"] = spots
    checks["noise_free_path_vs_frozen_extension_pool_max_abs_diff"] = spots0
    checks["pool_finite"] = bool(all(np.isfinite(pl[j]).all() for pl in pools for j in range(0, len(obj_tr), 37)))
    checks["selection_finite"] = bool(np.isfinite(sel).all())
    meta = {"levels": LEVELS, "files": {lv: str(pool_npy(lv).relative_to(ROOT)) for lv in LEVELS},
            "shape": list(pools[0].shape), "axes": "train_image, realisation k, shots, H, W", "shots": LOW, "K": K,
            "seed_rule": "patterns 30,000,000 + 10,000*k + train_index (as noise-free extension); "
                         "noise default_rng([pattern_seed, 1000003, round(100 f)])",
            "train_ids": ids_tr.tolist(), "selection_seeds": SELECTION_SEEDS, "checks": checks,
            "seconds": round(time.time() - t0, 1), "selection_npz_md5": md5(selection_npz())}
    write_json(OUT / "pool" / "pool_meta.json", meta)
    print(json.dumps(checks, indent=1), f"\n{meta['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
