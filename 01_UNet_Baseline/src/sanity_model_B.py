#!/usr/bin/env python3
"""Sanity checks + CPU benchmark for Model B before full training (no files written to models/).
Writes results/logs/sanity_model_B.json."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import keras  # noqa: E402

from dataset_loader import TrainPatchStream, fixed_validation_patches, read_samples  # noqa: E402
from photon_condition import (ALLOWED_P, ConditionedTrainPatchStream, condition_validation_inputs,  # noqa: E402
                              photon_condition)
from train_model_B import build_model_B, compile_like_model_A  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    cfg = json.load(open(ROOT / "configs/model_B_photon_conditioned.json", encoding="utf-8"))
    cfgA = json.load(open(ROOT / "configs/model_A_baseline.json", encoding="utf-8"))
    R, checks = {}, {}
    a_md5 = {p.name: md5(p) for p in sorted((ROOT / "models/model_A_baseline_unet").glob("*.keras"))}
    R["model_A_checkpoint_md5"] = a_md5

    # condition mapping
    R["condition_table"] = {str(int(P)): round(photon_condition(P), 6) for P in ALLOWED_P}
    checks["P10_gives_0.0"] = photon_condition(10) == 0.0
    checks["P1000_gives_1.0"] = abs(photon_condition(1000) - 1.0) < 1e-12
    checks["condition_monotonic"] = all(np.diff([photon_condition(P) for P in ALLOWED_P]) > 0)

    # streams: B vs A on identical (row, rng_state)
    sB = ConditionedTrainPatchStream(cfg, seed=cfg["seeds"]["train_stream"])
    sA = TrainPatchStream(cfgA, seed=cfgA["seeds"]["train_stream"])
    test_ids = set((ROOT / "data/v2/splits/test/ids.txt").read_text().split())
    val_ids = set((ROOT / "data/v2/splits/val/ids.txt").read_text().split())
    checks["train_stream_rows_all_train_split"] = all(r["split"] == "train" for r in sB.rows)
    checks["no_test_image_in_train_stream"] = not ({r["image_id"] for r in sB.rows} | set(sB.ref)) & test_ids
    checks["no_val_image_in_train_stream"] = not ({r["image_id"] for r in sB.rows} | set(sB.ref)) & val_ids
    rng = np.random.default_rng(0)
    per_level = {}
    same_rgb = same_target = const_ok = value_ok = shape_ok = True
    for P in ALLOWED_P:
        rows = [r for r in sB.rows if r["photon_budget_P"] == P]
        for r in [rows[i] for i in rng.choice(len(rows), 3, replace=False)]:
            st = int(rng.integers(2 ** 31))
            pb, pa = sB._patches_from(r, st), sA._patches_from(r, st)
            for (xb, yb), (xa, ya) in zip(pb, pa):
                shape_ok &= xb.shape == (256, 256, 4) and yb.shape == (256, 256, 4)
                same_rgb &= np.array_equal(xb[..., :3], xa)
                same_target &= np.array_equal(yb, ya)
                c = xb[..., 3]
                const_ok &= float(c.max() - c.min()) == 0.0
                value_ok &= abs(float(c[0, 0]) - photon_condition(P)) < 1e-7
                per_level[str(int(P))] = round(float(c[0, 0]), 6)
    checks["input_patch_shape_256x256x4"] = shape_ok
    checks["rgb_channels_identical_to_model_A_noisy_input"] = same_rgb
    checks["target_identical_to_model_A_target"] = same_target
    checks["condition_channel_spatially_constant"] = const_ok
    checks["condition_value_matches_P"] = value_ok
    R["observed_condition_per_P"] = per_level

    # generator: condition values in allowed set, RGB statistics consistent with N/P
    g = sB.generator(4, 64)
    seen = []
    for _ in range(300):
        x, y = next(g)
        seen.append(round(float(x[0, 0, 3]), 6))
        assert float(x[..., 3].max() - x[..., 3].min()) == 0.0
    allowed = {round(photon_condition(P), 6) for P in ALLOWED_P}
    checks["generator_condition_values_in_allowed_set"] = set(seen) <= allowed
    R["generator_condition_counts"] = {str(v): seen.count(v) for v in sorted(set(seen))}

    # validation patches: same as Model A's + condition
    xv, yv, Pv, idv = fixed_validation_patches(cfg, seed=cfg["seeds"]["val_patches"])
    xv4 = condition_validation_inputs(xv, Pv)
    checks["val_input_shape_Nx256x256x4"] = xv4.shape[1:] == (256, 256, 4)
    checks["val_rgb_unchanged"] = np.array_equal(xv4[..., :3], xv)
    checks["val_condition_matches_P"] = all(abs(float(xv4[i, 0, 0, 3]) - photon_condition(Pv[i])) < 1e-7
                                            for i in range(len(Pv)))
    checks["val_images_are_val_split_only"] = set(idv) <= val_ids and not set(idv) & test_ids
    R["n_val_patches"] = len(xv4)

    # model
    model = build_model_B(cfg)
    compile_like_model_A(model, cfg)
    out = model.predict(xv4[:2], verbose=0)
    R["model_params"] = int(model.count_params())
    R["model_input_shape"] = str(tuple(model.input.shape))
    R["model_output_shape"] = str(tuple(model.output.shape))
    R["first_conv_kernel_shape"] = str(tuple(model.get_layer("enc0_conv1").kernel.shape))
    checks["output_shape_256x256x3"] = out.shape[1:] == (256, 256, 3)
    checks["params_equal_model_A_plus_144"] = R["model_params"] == 1_941_139 + 3 * 3 * 1 * 16

    # CPU benchmark: a few real batches from the stream (throwaway model, nothing saved)
    bs = cfg["batch_size"]
    batches = []
    for _ in range(12):
        xb, yb = zip(*[next(g) for _ in range(bs)])
        batches.append((np.stack(xb), np.stack(yb)))
    model.train_on_batch(*batches[0])
    model.train_on_batch(*batches[1])
    t = time.time()
    for xb, yb in batches[2:]:
        model.train_on_batch(xb, yb)
    step = (time.time() - t) / (len(batches) - 2)
    R["benchmark_s_per_step"] = round(step, 3)
    R["estimated_epoch_train_s"] = round(step * cfg["steps_per_epoch"])
    R["estimated_60_epochs_h"] = round(step * cfg["steps_per_epoch"] * cfg["max_epochs"] / 3600, 2)

    R["model_A_checkpoints_unchanged"] = a_md5 == {p.name: md5(p) for p in
                                                   sorted((ROOT / "models/model_A_baseline_unet").glob("*.keras"))}
    checks["model_A_checkpoints_unchanged"] = R["model_A_checkpoints_unchanged"]
    R["checks"] = checks
    R["ALL_PASSED"] = all(checks.values())
    (ROOT / "results/logs").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/logs/sanity_model_B.json").write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    print(json.dumps(R, indent=1, default=str))
    os._exit(0)  # loader threads are daemons; exit without waiting


if __name__ == "__main__":
    main()
