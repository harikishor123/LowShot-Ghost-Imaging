#!/usr/bin/env python3
"""Read-only checks before the official held-out test evaluation of Model A.
Writes results/evaluation/model_A_test/preflight.json."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import keras  # noqa: E402

from dataset_loader import load_pair, read_samples  # noqa: E402
from losses import masked_mae, masked_mse, masked_psnr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data" / "v2"
OUT = ROOT / "results" / "evaluation" / "model_A_test"
CKPT = "models/model_A_baseline_unet/final_restored_best.keras"
BEST = "models/model_A_baseline_unet/best.keras"


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = {}
    ids = {s: set((V2 / "splits" / s / "ids.txt").read_text().split()) for s in ["train", "val", "test"]}
    rows = read_samples("test")  # raises on any split/pairing mismatch
    manifest = {r["stem"]: r for r in csv.DictReader(open(V2 / "metadata/manifest.csv", encoding="utf-8"))}
    R["n_test_images"] = len(ids["test"])
    R["n_test_samples"] = len(rows)
    R["photon_levels_P"] = sorted({r["photon_budget_P"] for r in rows})
    R["samples_per_level"] = {str(int(P)): sum(r["photon_budget_P"] == P for r in rows) for P in R["photon_levels_P"]}
    R["test_ids_in_samples_equal_ids_txt"] = {r["image_id"] for r in rows} == ids["test"]
    R["test_overlap_train"] = sorted(ids["test"] & ids["train"])
    R["test_overlap_val"] = sorted(ids["test"] & ids["val"])
    R["manifest_split_is_test_for_all"] = all(manifest[i]["split"] == "test" for i in ids["test"])
    # near-duplicate groups: any kept member of a test image's duplicate group must also be test
    R["dup_group_members_outside_test"] = sorted({m for i in ids["test"] for m in manifest[i]["dup_members"].split("|")
                                                  if m and manifest[m]["split"] not in ("test", "excluded")})
    # sample-level leakage (the training stream / validation set read only these files)
    tr = {r["image_id"] for r in read_samples("train")}
    va = {r["image_id"] for r in read_samples("val")}
    R["test_ids_in_train_samples"] = sorted(ids["test"] & tr)
    R["test_ids_in_val_samples"] = sorted(ids["test"] & va)

    # test data integrity vs generation-time checksums
    bad = []
    for r in rows:
        with np.load(ROOT / r["noisy_counts_path"]) as z:
            if hashlib.md5(z["counts"].tobytes()).hexdigest() != r["counts_md5"]:
                bad.append(r["sample_id"])
    R["test_noisy_files_match_generation_md5"] = f"{len(rows) - len(bad)}/{len(rows)}"
    gen = json.load(open(V2 / "photon_sim/metadata/generation_config.json", encoding="utf-8"))["reference_md5"]
    refbad = [i for i in ids["test"] if md5(V2 / "reference" / f"{i}.npz") != gen[f"{i}.npz"]]
    R["test_references_match_generation_md5"] = f"{len(ids['test']) - len(refbad)}/{len(ids['test'])}"

    # checkpoint
    co = {"masked_mse": masked_mse, "masked_mae": masked_mae, "masked_psnr": masked_psnr}
    m1 = keras.models.load_model(ROOT / CKPT, custom_objects=co, compile=False)
    m2 = keras.models.load_model(ROOT / BEST, custom_objects=co, compile=False)
    R["checkpoint"] = CKPT
    R["checkpoint_md5"] = md5(ROOT / CKPT)
    R["checkpoint_weights_identical_to_best_keras"] = all(np.array_equal(a, b) for a, b in
                                                          zip(m1.get_weights(), m2.get_weights()))
    R["checkpoint_params"] = int(m1.count_params())
    ts = json.load(open(ROOT / "models/model_A_baseline_unet/training_summary.json", encoding="utf-8"))
    R["checkpoint_epoch"] = ts["best_epoch"]
    R["model_input_shape"] = str(tuple(m1.input.shape))
    R["model_output_shape"] = str(tuple(m1.output.shape))
    R["model_input_names"] = [t.name for t in m1.inputs]
    R["model_n_inputs"] = len(m1.inputs)
    # shape check on one test sample (no metrics computed)
    x, y, m = load_pair(rows[0])
    H, W = x.shape[:2]
    Hp, Wp = -(-H // 16) * 16, -(-W // 16) * 16
    xp = np.zeros((1, Hp, Wp, 3), np.float32)
    xp[0, :H, :W] = x
    out = m1.predict(xp, verbose=0)
    R["example_full_image"] = {"sample": rows[0]["sample_id"], "input_hw": [H, W], "padded_input": list(xp.shape),
                               "output": list(out.shape), "output_range": [float(out.min()), float(out.max())]}
    R["ALL_OK"] = (R["n_test_images"] == 74 and R["n_test_samples"] == 444
                   and R["photon_levels_P"] == [10.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
                   and R["test_ids_in_samples_equal_ids_txt"] and not R["test_overlap_train"]
                   and not R["test_overlap_val"] and R["manifest_split_is_test_for_all"]
                   and not R["dup_group_members_outside_test"] and not R["test_ids_in_train_samples"]
                   and not R["test_ids_in_val_samples"] and not bad and not refbad
                   and R["checkpoint_weights_identical_to_best_keras"] and R["model_n_inputs"] == 1
                   and list(out.shape) == list(xp.shape))
    (OUT / "preflight.json").write_text(json.dumps(R, indent=2), encoding="utf-8")
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
