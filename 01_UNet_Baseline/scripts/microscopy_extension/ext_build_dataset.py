#!/usr/bin/env python3
"""Extension (broader microscopy dataset), step 1: build the GI objects from MedMNIST+ (64 x 64).

    python ext_build_dataset.py

Sources (CC BY 4.0, Zenodo 10.5281/zenodo.10519652, MD5-checked): bloodmnist_64.npz and pathmnist_64.npz
in data/extension_medmnist/raw/. Per subset 5,000 images are drawn from MedMNIST's OFFICIAL splits
(train 3,500 / val 500 / test 1,000), stratified by class (largest-remainder quotas), seed 42.

Conversion is the onion conversion without the crop/resize (MedMNIST+ 64 px images are already 64 x 64):
uint8 sRGB -> [0,1] -> linear light (IEC 61966-2-1, src/color_space.py) -> Y = 0.2126 R + 0.7152 G + 0.0722 B
-> clip [0, 1]; no per-image normalisation.

Screening (before the final draw, on an over-sampled candidate set):
  * quality: object std < 0.005 (near-blank) is excluded;
  * duplicates: the onion audit rules (src/dataset_audit_full.py), vectorised: exact pixel MD5; pHash (min over
    the 8 dihedral variants), dHash and 32x32 NCC thresholds for near-duplicates and flipped/rotated copies.
    From each strong-link cluster one image is kept (train before val before test, then draw order).
Outputs: data/extension_medmnist/processed/objects_64.npz, outputs/extension_medmnist_microscopy/dataset/*.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ext_common import DATA, OBJECTS_NPZ, OUT, SUBSETS, safe_path, write_json  # noqa: E402
from src.color_space import decode_uint8, linear_to_srgb, srgb_to_linear  # noqa: E402
from src.dataset_audit_full import dhash_bits, dihedral, phash_bits  # noqa: E402
from src.ghost_imaging import LUMA_WEIGHTS  # noqa: E402

RAW_MD5 = {"bloodmnist_64.npz": "2b94928a2ae4916078ca51e05b6b800b", "pathmnist_64.npz": "55aa9c1e0525abe5a6b9d8343a507616"}
QUOTA = {"train": 3500, "val": 500, "test": 1000}
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}
OVERSAMPLE = 0.15
MIN_STD = 0.005
SEED = 42
DS = OUT / "dataset"


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def quotas(labels: np.ndarray, total: int) -> dict[int, int]:
    """Largest-remainder allocation of `total` over classes, proportional to class frequency."""
    cls, cnt = np.unique(labels, return_counts=True)
    exact = cnt / cnt.sum() * total
    q = np.floor(exact).astype(int)
    for j in np.argsort(-(exact - q), kind="stable")[: total - q.sum()]:
        q[j] += 1
    return {int(c): int(n) for c, n in zip(cls, q)}


def to_object(rgb_u8: np.ndarray) -> np.ndarray:
    """(64, 64, 3) uint8 sRGB -> (64, 64) float64 linear luminance in [0, 1] (as src.ghost_imaging.load_object)."""
    lin = srgb_to_linear(decode_uint8(rgb_u8))
    y = lin.astype(np.float64) @ LUMA_WEIGHTS
    return np.clip(y.astype(np.float32).astype(np.float64), 0.0, 1.0)


def fingerprints(gray: np.ndarray) -> dict:
    th = np.asarray(Image.fromarray((gray * 255).astype(np.uint8)).resize((32, 32), Image.Resampling.BOX),
                    dtype=np.float32).ravel()
    th = (th - th.mean()) / (th.std() + 1e-6)
    return {"dhash": dhash_bits(gray), "phash_variants": np.stack([phash_bits(v) for v in dihedral(gray)]),
            "thumb": th, "thumb_variants": np.stack([v.ravel() for v in dihedral(th.reshape(32, 32))])}


def pack(bits: np.ndarray) -> np.ndarray:
    """(..., 64) bool -> (...) uint64."""
    return np.packbits(bits.astype(np.uint8), axis=-1).view(np.uint64)[..., 0]


def find_pairs(pix_md5, D, P, T, TV, chunk: int = 512) -> pd.DataFrame:
    """All pairs a < b classified with the onion audit rules (src/dataset_audit_full.find_duplicates)."""
    n = len(D)
    Dp, Pp, canon = pack(D), pack(P), pack(P[:, 0])
    rows = []
    for s in range(0, n, chunk):
        a = np.arange(s, min(s + chunk, n))
        dd = np.bitwise_count(Dp[a, None] ^ Dp[None, :]).astype(np.int16)
        dist_t = np.stack([np.bitwise_count(Pp[a, t, None] ^ canon[None, :]) for t in range(8)]).astype(np.int16)
        best_t = dist_t.argmin(0)
        p_ab = dist_t.min(0)
        p_ba = np.stack([np.bitwise_count(Pp[None, :, t] ^ canon[a, None]) for t in range(8)]).min(0).astype(np.int16)
        p_id = np.minimum(p_ab, p_ba)
        ncc = (T[a] @ T.T) / T.shape[1]
        ncc_t = np.take_along_axis(np.einsum("atk,bk->tab", TV[a], T) / T.shape[1], best_t[None], 0)[0]
        exact = pix_md5[a, None] == pix_md5[None, :]
        near = ((p_id <= 6) & (dd <= 10) & (ncc >= 0.80)) | ((p_id <= 10) & (ncc >= 0.95))
        flip = (p_id <= 6) & (best_t != 0) & (ncc_t >= 0.90)
        similar = ((p_id <= 10) & (ncc >= 0.80)) | (ncc >= 0.90)
        upper = np.arange(n)[None, :] > a[:, None]
        hit = upper & (exact | near | flip | similar)
        for i, j in zip(*np.nonzero(hit)):
            kind = ("exact" if exact[i, j] else "near-duplicate" if near[i, j] else
                    "flipped/rotated copy" if flip[i, j] else "similar (review)")
            rows.append({"a": int(a[i]), "b": int(j), "kind": kind, "phash_dist": int(p_id[i, j]),
                         "dhash_dist": int(dd[i, j]), "ncc32": float(ncc[i, j])})
    return pd.DataFrame(rows, columns=["a", "b", "kind", "phash_dist", "dhash_dist", "ncc32"])


def main() -> None:
    t0 = time.time()
    raw = DATA / "raw"
    checks = {"raw_md5": {}}
    for f, want in RAW_MD5.items():
        got = md5_file(raw / f)
        checks["raw_md5"][f] = {"md5": got, "matches_zenodo": got == want}
        assert got == want, f"{f}: MD5 {got} != {want}"

    # ---- candidates: per subset / official split / class, shuffled with seed 42 ----------------
    cand, rgb, official = [], [], {}
    for sub, name in SUBSETS.items():
        with np.load(raw / f"{name}_64.npz") as z:
            official[sub] = {s: int(len(z[f"{s}_labels"])) for s in QUOTA}
            assert z["train_images"].shape[1:] == (64, 64, 3) and z["train_images"].dtype == np.uint8
            for s, total in QUOTA.items():
                imgs, lab = z[f"{s}_images"], z[f"{s}_labels"][:, 0]
                rng = np.random.default_rng([SEED, list(SUBSETS).index(sub), SPLIT_ORDER[s]])
                for c, q in quotas(lab, total).items():
                    idx = np.flatnonzero(lab == c)
                    idx = idx[rng.permutation(len(idx))][: q + max(3, int(np.ceil(OVERSAMPLE * q)))]
                    for rank, i in enumerate(idx):
                        cand.append({"subset": sub, "split": s, "label": int(c), "quota": q, "rank": rank,
                                     "orig_index": int(i), "image_id": f"{sub}_{s}_{int(i):06d}"})
                        rgb.append(imgs[i])
    cand = pd.DataFrame(cand)
    rgb = np.stack(rgb)
    checks["official_split_sizes"] = official
    print(f"{len(cand)} candidates ({time.time() - t0:.0f}s)", flush=True)

    obj = np.stack([to_object(im) for im in rgb])
    cand["pixel_md5"] = [hashlib.md5(im.tobytes()).hexdigest() for im in rgb]
    cand["obj_mean"], cand["obj_std"] = obj.mean((1, 2)), obj.std((1, 2))
    cand["obj_min"], cand["obj_max"] = obj.min((1, 2)), obj.max((1, 2))
    cand["low_information"] = cand.obj_std < MIN_STD

    fp = [fingerprints(linear_to_srgb(o).astype(np.float64)) for o in obj]
    D = np.stack([f["dhash"] for f in fp])
    P = np.stack([f["phash_variants"] for f in fp])
    T = np.stack([f["thumb"] for f in fp])
    TV = np.stack([f["thumb_variants"] for f in fp])
    pairs = find_pairs(cand.pixel_md5.to_numpy(), D, P, T, TV)
    print(f"pairs flagged among candidates: {pairs.kind.value_counts().to_dict()} ({time.time() - t0:.0f}s)", flush=True)

    # ---- remove: one image kept per strong-link cluster; low-information images ------------------
    parent = list(range(len(cand)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    strong = pairs[pairs.kind != "similar (review)"]
    for a, b in zip(strong.a, strong.b):
        parent[find(a)] = find(b)
    cand["cluster"] = [find(i) for i in range(len(cand))]
    cand["priority"] = cand.split.map(SPLIT_ORDER) * 100000 + cand["rank"]
    keep_rep = cand.sort_values("priority").groupby("cluster").head(1).index
    cand["duplicate_removed"] = ~cand.index.isin(keep_rep)
    cand["eligible"] = ~cand.duplicate_removed & ~cand.low_information

    # ---- final draw: first `quota` eligible candidates per stratum, in shuffled order --------------
    chosen = []
    for (sub, s, c), g in cand.groupby(["subset", "split", "label"]):
        e = g[g.eligible].sort_values("rank")
        q = int(g.quota.iloc[0])
        assert len(e) >= q, f"{sub}/{s}/class {c}: only {len(e)} eligible for quota {q}"
        chosen.extend(e.index[:q])
    cand["selected"] = cand.index.isin(chosen)
    sel = cand[cand.selected].sort_values(["subset", "split", "label", "rank"])
    order = sel.index.to_numpy()
    remaining = pairs[pairs.a.isin(order) & pairs.b.isin(order)].copy()
    split_of = cand.split.to_dict()
    remaining["cross_split"] = [split_of[a] != split_of[b] for a, b in zip(remaining.a, remaining.b)]
    checks["selected_strong_pairs"] = int((remaining.kind != "similar (review)").sum())
    assert checks["selected_strong_pairs"] == 0
    checks["selected_similar_review_pairs"] = int(len(remaining))
    checks["selected_similar_review_cross_split"] = int(remaining.cross_split.sum())
    checks["selected_ids_unique"] = bool(sel.image_id.is_unique)

    # ---- save -----------------------------------------------------------------------------------
    np.savez_compressed(safe_path(OBJECTS_NPZ), image_id=sel.image_id.to_numpy().astype("U32"),
                        subset=sel.subset.to_numpy().astype("U8"), split=sel.split.to_numpy().astype("U8"),
                        label=sel.label.to_numpy().astype(np.int16), orig_index=sel.orig_index.to_numpy().astype(np.int64),
                        object=obj[order], rgb_u8=rgb[order])
    cols = ["image_id", "subset", "split", "label", "orig_index", "pixel_md5", "obj_mean", "obj_std", "obj_min", "obj_max"]
    sel[cols].to_csv(safe_path(DS / "manifest.csv"), index=False)
    cand.drop(columns=["cluster", "priority"]).to_csv(safe_path(DS / "candidates_screening.csv"), index=False)
    for df, f in ((pairs, "flagged_pairs_candidates.csv"), (remaining, "similar_pairs_in_final_set.csv")):
        out = df.copy()
        out["id_a"], out["id_b"] = cand.image_id.to_numpy()[out.a], cand.image_id.to_numpy()[out.b]
        out.drop(columns=["a", "b"]).to_csv(safe_path(DS / f), index=False)

    counts = sel.groupby(["subset", "split"]).size().unstack().to_dict()
    cls = {sub: {s: sel[(sel.subset == sub) & (sel.split == s)].label.value_counts().sort_index().to_dict()
                 for s in QUOTA} for sub in SUBSETS}
    stats = {sub: {k: float(getattr(sel[sel.subset == sub][c], f)()) for k, c, f in
                   (("mean_of_object_mean", "obj_mean", "mean"), ("mean_of_object_std", "obj_std", "mean"),
                    ("min_object_std", "obj_std", "min"))} for sub in SUBSETS}
    report = {"sources": {"zenodo_doi": "10.5281/zenodo.10519652", "record_version": "3.0", "license": "CC BY 4.0",
                          "files": checks["raw_md5"]},
              "sampling": {"per_subset": QUOTA, "stratified_by": "class label within each official split",
                           "seed": SEED, "oversample_for_screening": OVERSAMPLE},
              "conversion": "uint8 sRGB -> linear (IEC 61966-2-1) -> Y = 0.2126R + 0.7152G + 0.0722B -> clip [0,1]; 64x64 as provided",
              "screening": {"n_candidates": int(len(cand)), "pairs_flagged_among_candidates": pairs.kind.value_counts().to_dict(),
                            "removed_as_duplicates": int(cand.duplicate_removed.sum()),
                            "removed_duplicates_by_subset": cand[cand.duplicate_removed].subset.value_counts().to_dict(),
                            "low_information_flagged": int(cand.low_information.sum()), "min_std_rule": MIN_STD},
              "final_counts": counts, "class_counts": cls, "object_stats": stats, "checks": checks,
              "objects_npz": str(OBJECTS_NPZ.relative_to(ROOT)), "objects_npz_md5": md5_file(OBJECTS_NPZ),
              "runtime_s": round(time.time() - t0, 1)}
    write_json(DS / "dataset_report.json", report)

    # ---- contact sheet: RGB source and GI object for 8 images per subset ---------------------------
    fig, axes = plt.subplots(4, 8, figsize=(13, 7))
    for r, sub in enumerate(SUBSETS):
        g = sel[(sel.subset == sub) & (sel.split == "train")]
        pick = g.groupby("label").head(1).index[:8]
        for c, i in enumerate(pick):
            j = int(np.flatnonzero(order == i)[0])
            axes[2 * r, c].imshow(rgb[order][j])
            axes[2 * r, c].set_title(f"{sub} class {cand.label[i]}", fontsize=8)
            axes[2 * r + 1, c].imshow(obj[order][j], cmap="gray", vmin=0, vmax=1)
            axes[2 * r + 1, c].set_title("GI object (linear Y)", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Extension dataset: MedMNIST+ 64 px source (sRGB) and GI object (linear luminance, grey window [0,1])", fontsize=10)
    fig.tight_layout()
    fig.savefig(safe_path(DS / "contact_sheet.png"), dpi=130, facecolor="white")
    plt.close(fig)
    print(json.dumps({k: report[k] for k in ("screening", "final_counts", "object_stats", "checks")}, indent=1, default=str))
    print(f"{report['runtime_s']} s")


if __name__ == "__main__":
    main()
