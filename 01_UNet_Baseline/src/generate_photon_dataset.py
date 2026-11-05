#!/usr/bin/env python3
"""STEP 4b: generate the fixed photon-limited dataset from data/v2 clean references.

Shot noise only (Poisson). Fixed photon budget per level for every image (Option A):
P = 10, 50, 100, 250, 500, 1000 photons at I = 1. One deterministic realization
(realization 0) per (image, level) for train, val and test. Each split's noisy samples
are generated only from that split's references (split read from data/v2/splits/<split>/ids.txt).

Stored per sample: data/v2/photon_sim/<split>/P<PPPP>/<id>.npz with
    counts  uint16 HxWx3   detected photon counts N (the noisy image is I_noisy = counts / P)
    photons float32        P
    level   str            e.g. '10%'
The FOV mask and clean target live in data/v2/reference/<id>.npz (not duplicated).

Clean references are opened read-only and MD5-verified before/after.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from color_space import load_reference_linear  # noqa: E402
from photon_simulation import (BASE_SEED, P_FULL, PHOTON_LEVELS, describe, rng_for, seed_entropy,  # noqa: E402
                               seed_key, simulate)

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data" / "v2"
OUT = V2 / "photon_sim"
SPLITS = ["train", "val", "test"]
REALIZATION = 0


def level_dir(P: float) -> str:
    return f"P{int(round(P)):04d}"


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def gen_one(job):
    split, sid = job
    ref = V2 / "reference" / f"{sid}.npz"
    lin, m = load_reference_linear(ref)
    rows = []
    for level, P in PHOTON_LEVELS.items():
        noisy, N, lam = simulate(lin, m, P, rng_for(split, sid, P, REALIZATION))
        if N.max() > np.iinfo(np.uint16).max:
            raise OverflowError(f"{sid} {level}: count {N.max()} exceeds uint16")
        out = OUT / split / level_dir(P) / f"{sid}.npz"
        np.savez_compressed(out, counts=N.astype(np.uint16), photons=np.float32(P), level=np.array(level))
        rows.append({
            "sample_id": f"{split}/{level_dir(P)}/{sid}", "image_id": sid, "split": split,
            "photon_level": level, "photon_level_fraction": P / P_FULL, "photon_budget_P": P,
            "log10_P": float(np.log10(P)), "realization": REALIZATION,
            "seed_key": seed_key(split, sid, P, REALIZATION),
            "seed_entropy": " ".join(map(str, seed_entropy(split, sid, P, REALIZATION))),
            "clean_reference_path": f"data/v2/reference/{sid}.npz",
            "noisy_counts_path": f"data/v2/photon_sim/{split}/{level_dir(P)}/{sid}.npz",
            "height": N.shape[0], "width": N.shape[1], "fov_pixels": int(m.sum()),
            "mean_lambda_fov": float(lam[m].mean()), "mean_counts_fov": float(N[m].mean()),
            "dispersion_index": float(np.mean((N[m] - lam[m]) ** 2) / max(np.mean(lam[m]), 1e-12)),
            "max_count": int(N.max()), "frac_zero_counts_fov": float((N[m] == 0).mean()),
            "counts_md5": hashlib.md5(N.astype(np.uint16).tobytes()).hexdigest(),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if OUT.exists() and any(OUT.rglob("*.npz")) and not a.overwrite:
        sys.exit(f"{OUT} already contains samples; pass --overwrite to regenerate")

    split_ids = {s: (V2 / "splits" / s / "ids.txt").read_text(encoding="utf-8").split() for s in SPLITS}
    manifest = {r["stem"]: r for r in csv.DictReader(open(V2 / "metadata" / "manifest.csv", encoding="utf-8"))}
    # split isolation pre-check
    for s, ids in split_ids.items():
        assert all(manifest[i]["split"] == s for i in ids), f"manifest/split mismatch in {s}"
    assert not (set(split_ids["train"]) & set(split_ids["val"]) or set(split_ids["train"]) & set(split_ids["test"])
                or set(split_ids["val"]) & set(split_ids["test"])), "image in more than one split"

    refs = sorted((V2 / "reference").glob("*.npz"))
    before = {p.name: md5(p) for p in refs}
    for s in SPLITS:
        for P in PHOTON_LEVELS.values():
            (OUT / s / level_dir(P)).mkdir(parents=True, exist_ok=True)
    (OUT / "metadata").mkdir(parents=True, exist_ok=True)

    jobs = [(s, i) for s in SPLITS for i in split_ids[s]]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, r in enumerate(ex.map(gen_one, jobs, chunksize=2)):
            rows.extend(r)
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(jobs)} images ({time.time() - t0:.0f}s)")
    rows.sort(key=lambda r: (SPLITS.index(r["split"]), r["photon_budget_P"], r["image_id"]))

    cols = list(rows[0].keys())
    with open(OUT / "metadata" / "samples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    for s in SPLITS:  # per-split convenience indices
        with open(OUT / "metadata" / f"samples_{s}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                if r["split"] == s:
                    w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    after = {p.name: md5(p) for p in refs}
    cfg = {**describe(), "fixed_budget_option": "A (same P for every image at a given level)",
           "realization": REALIZATION, "storage": "uint16 photon counts N; I_noisy = N / P (linear light, unclipped)",
           "mask_and_clean_target": "data/v2/reference/<id>.npz (srgb16 -> linear via color_space, mask)",
           "generated_unix_time": time.time(), "n_samples": len(rows),
           "reference_md5": before}
    (OUT / "metadata" / "generation_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(rows), "per_split": dict(Counter(r["split"] for r in rows)),
                      "per_level": dict(Counter(r["photon_level"] for r in rows)),
                      "references_unchanged": before == after, "seconds": round(time.time() - t0)}, indent=1))


if __name__ == "__main__":
    main()
