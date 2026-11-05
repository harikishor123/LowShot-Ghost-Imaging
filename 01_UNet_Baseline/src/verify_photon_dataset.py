#!/usr/bin/env python3
"""Independent verification of data/v2/photon_sim (read-only). Writes data/v2/photon_sim/metadata/verification.json
and example figures to data/v2/photon_sim/examples/."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from color_space import linear_to_srgb, load_reference_linear  # noqa: E402
from photon_simulation import PHOTON_LEVELS, rng_for, simulate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data" / "v2"
SIM = V2 / "photon_sim"
SPLITS = ["train", "val", "test"]


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def dir_size(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())


def main():
    rows = list(csv.DictReader(open(SIM / "metadata" / "samples.csv", encoding="utf-8")))
    cfg = json.load(open(SIM / "metadata" / "generation_config.json", encoding="utf-8"))
    manifest = {r["stem"]: r for r in csv.DictReader(open(V2 / "metadata" / "manifest.csv", encoding="utf-8"))}
    split_ids = {s: set((V2 / "splits" / s / "ids.txt").read_text().split()) for s in SPLITS}
    kept = {k for k, r in manifest.items() if r["split"] != "excluded"}
    V = {}

    # counts
    V["n_clean_references"] = len(list((V2 / "reference").glob("*.npz")))
    V["n_noisy_samples_metadata"] = len(rows)
    V["n_noisy_files_on_disk"] = len(list(SIM.glob("*/P*/*.npz")))
    V["per_split"] = dict(Counter(r["split"] for r in rows))
    V["per_level"] = dict(Counter(r["photon_level"] for r in rows))
    V["per_split_level"] = {s: dict(Counter(r["photon_level"] for r in rows if r["split"] == s)) for s in SPLITS}
    V["images_per_split"] = {s: len({r["image_id"] for r in rows if r["split"] == s}) for s in SPLITS}

    # split isolation
    folder_ids = defaultdict(set)
    for f in SIM.glob("*/P*/*.npz"):
        folder_ids[f.parts[-3]].add(f.stem)
    leak = {
        "metadata_split_matches_manifest": all(manifest[r["image_id"]]["split"] == r["split"] for r in rows),
        "metadata_split_matches_ids_txt": all(r["image_id"] in split_ids[r["split"]] for r in rows),
        "folder_split_matches_ids_txt": all(folder_ids[s] == split_ids[s] for s in SPLITS),
        "no_image_in_two_splits": not (folder_ids["train"] & folder_ids["val"] or folder_ids["train"] & folder_ids["test"]
                                       or folder_ids["val"] & folder_ids["test"]),
        "every_kept_image_has_all_6_levels": all(
            sum(1 for _ in SIM.glob(f"{manifest[i]['split']}/P*/{i}.npz")) == 6 for i in kept),
        "no_excluded_image_simulated": not ({r["image_id"] for r in rows} - kept),
        "duplicate_groups_single_split": all(
            len({manifest[m]["split"] for m in manifest[k]["dup_members"].split("|") if manifest[m]["split"] != "excluded"}) <= 1
            for k in kept if manifest[k]["dup_members"]),
    }
    V["split_isolation"] = leak

    # reference integrity
    now = {p.name: md5(p) for p in sorted((V2 / "reference").glob("*.npz"))}
    V["references_unchanged_since_generation_start"] = now == cfg["reference_md5"]

    # file integrity + exact reproducibility on a random subset (all splits, all levels)
    random.seed(0)
    sub = random.sample(rows, 60) + [r for r in rows if r["split"] in ("val", "test") and r["image_id"] in
                                     (sorted(split_ids["val"])[0], sorted(split_ids["test"])[0])]
    ok_file = ok_regen = 0
    for r in sub:
        z = np.load(ROOT / r["noisy_counts_path"])
        c = z["counts"]
        ok_file += hashlib.md5(c.tobytes()).hexdigest() == r["counts_md5"] and float(z["photons"]) == float(r["photon_budget_P"])
        lin, m = load_reference_linear(ROOT / r["clean_reference_path"])
        P = float(r["photon_budget_P"])
        _, N, _ = simulate(lin, m, P, rng_for(r["split"], r["image_id"], P, int(r["realization"])))
        ok_regen += np.array_equal(N.astype(np.uint16), c) and bool((c[~m] == 0).all())
    V["reproducibility"] = {"checked": len(sub), "stored_md5_match": ok_file, "regenerated_identical": ok_regen}

    # Poisson sanity on the whole dataset
    lev = {}
    for name, P in PHOTON_LEVELS.items():
        rr = [r for r in rows if r["photon_level"] == name]
        d = np.array([float(r["dispersion_index"]) for r in rr])
        lam = np.array([float(r["mean_lambda_fov"]) for r in rr])
        cnt = np.array([float(r["mean_counts_fov"]) for r in rr])
        lev[name] = {"P": P, "n": len(rr), "dispersion_index_mean": float(d.mean()), "dispersion_index_min": float(d.min()),
                     "dispersion_index_max": float(d.max()), "mean_lambda_fov_median": float(np.median(lam)),
                     "mean_lambda_fov_range": [float(lam.min()), float(lam.max())],
                     "counts_over_lambda": float(cnt.sum() / lam.sum()),
                     "max_count": int(max(int(r["max_count"]) for r in rr))}
    V["poisson_per_level"] = lev

    # storage
    V["storage_bytes"] = {"photon_sim_total": dir_size(SIM),
                          **{f"photon_sim/{s}": dir_size(SIM / s) for s in SPLITS},
                          **{f"photon_sim/*/{d.name}": sum(dir_size(SIM / s / d.name) for s in SPLITS)
                             for d in sorted((SIM / "train").glob("P*"))},
                          "reference_clean": dir_size(V2 / "reference")}
    V["ALL_OK"] = (all(leak.values()) and V["references_unchanged_since_generation_start"]
                   and ok_file == len(sub) and ok_regen == len(sub)
                   and V["n_noisy_files_on_disk"] == V["n_noisy_samples_metadata"] == 6 * len(kept)
                   and all(abs(v["dispersion_index_mean"] - 1) < 0.01 for v in lev.values()))
    (SIM / "metadata" / "verification.json").write_text(json.dumps(V, indent=2), encoding="utf-8")
    make_examples()
    print(json.dumps(V, indent=1))


def make_examples():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = SIM / "examples"
    out.mkdir(exist_ok=True)
    picks = [("train", "F_49"), ("train", "D_52"), ("val", "C_5"), ("val", "A_8")]
    fig, ax = plt.subplots(2 * len(picks), 7, figsize=(19, 5.6 * len(picks)), constrained_layout=True)
    for i, (split, sid) in enumerate(picks):
        lin, m = load_reference_linear(V2 / "reference" / f"{sid}.npz")
        H, W = m.shape
        ys, xs = slice(H // 2 - 128, H // 2 + 128), slice(W // 2 - 128, W // 2 + 128)
        imgs = [("clean", lin)]
        for name, P in PHOTON_LEVELS.items():
            c = np.load(SIM / split / f"P{int(P):04d}" / f"{sid}.npz")["counts"]
            imgs.append((f"{name}  P={P:g}", c.astype(np.float32) / P))
        for j, (t, im) in enumerate(imgs):
            d = linear_to_srgb(np.clip(im, 0, 1))
            d[~m] = 0.5
            ax[2 * i, j].imshow(d)
            ax[2 * i, j].set_title(f"{sid} [{split}] {t}" if j == 0 else t, fontsize=9)
            ax[2 * i + 1, j].imshow(d[ys, xs], interpolation="nearest")
            for a in (ax[2 * i, j], ax[2 * i + 1, j]):
                a.set_xticks([])
                a.set_yticks([])
        ax[2 * i + 1, 0].set_ylabel("central 256 px zoom")
    fig.suptitle("Stored photon-limited samples (I_noisy = counts / P, shown in sRGB, clipped for display)",
                 x=0.01, ha="left", fontweight="bold")
    fig.savefig(out / "examples_all_levels.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
