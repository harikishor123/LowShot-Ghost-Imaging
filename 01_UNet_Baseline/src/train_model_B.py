#!/usr/bin/env python3
"""Train Model B: photon-conditioned U-Net (noisy RGB + photon-condition map -> clean RGB).

Identical to Model A's src/train.py (same loader, patches, augmentation, validation patches,
loss, optimizer, LR schedule, early stopping, checkpointing, seeds) except:
  * input has 4 channels: [noisy R, G, B, c(P)], c(P) = (log10 P - 1) / 2 (photon_condition.py)
  * outputs go to models/model_B_photon_conditioned_unet/
Uses ONLY train references for training and ONLY val references for validation. Test is never read.

    python src/train_model_B.py --config configs/model_B_photon_conditioned.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import keras  # noqa: E402
import tensorflow as tf  # noqa: E402

from dataset_loader import fixed_validation_patches  # noqa: E402
from losses import masked_mae, masked_mse, masked_psnr  # noqa: E402
from photon_condition import ConditionedTrainPatchStream, condition_validation_inputs  # noqa: E402
from train import PerLevelVal  # noqa: E402  (Model A's per-level validation callback, reused)
from unet import build_unet, describe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_A_DIR = ROOT / "models" / "model_A_baseline_unet"


def build_model_B(cfg: dict) -> keras.Model:
    arch = cfg["architecture"]
    return build_unet(arch["in_channels"], 3, arch["base_filters"], arch["depth"], name=cfg["model_name"],
                      input_name=arch["input_name"])


def compile_like_model_A(model: keras.Model, cfg: dict) -> None:
    o = cfg["optimizer"]
    model.compile(optimizer=keras.optimizers.Adam(o["learning_rate"], beta_1=o["beta_1"], beta_2=o["beta_2"],
                                                  clipnorm=o["clipnorm"]),
                  loss=masked_mse, metrics=[masked_mae, masked_psnr])


def load_data(cfg: dict):
    stream = ConditionedTrainPatchStream(cfg, seed=cfg["seeds"]["train_stream"])
    xv, yv, Pv, idv = fixed_validation_patches(cfg, seed=cfg["seeds"]["val_patches"])
    xv4 = condition_validation_inputs(xv, Pv)
    return stream, xv4, yv, Pv, idv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model_B_photon_conditioned.json")
    a = ap.parse_args()
    cfg = json.load(open(ROOT / a.config, encoding="utf-8"))
    name = cfg["model_name"]
    mdir = ROOT / "models" / name
    if mdir.resolve() == MODEL_A_DIR.resolve():
        sys.exit("refusing to write Model B into the Model A directory")
    if (mdir / "best.keras").exists():
        sys.exit(f"{mdir}/best.keras already exists; not overwriting")
    mdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / a.config, mdir / "training_config.json")

    keras.utils.set_random_seed(cfg["seeds"]["global"])
    if cfg.get("deterministic_ops"):
        tf.config.experimental.enable_op_determinism()

    t0 = time.time()
    stream, xv, yv, Pv, idv = load_data(cfg)
    print(f"train samples (image x level): {len(stream.rows)}, train images: {len(stream.ref)}, "
          f"min valid corners/image: {min(stream.n_corners.values())}", flush=True)
    print(f"val patches: {len(xv)} from {len(set(idv))} val images, input shape {xv.shape[1:]} "
          f"({time.time() - t0:.0f}s)", flush=True)
    assert not set(idv) & set(stream.ref), "validation image found in training set"
    ps = cfg["patch_size"]
    cin = cfg["architecture"]["in_channels"]
    ds = tf.data.Dataset.from_generator(
        lambda: stream.generator(cfg["loader_threads"], cfg["shuffle_buffer"]),
        output_signature=(tf.TensorSpec((ps, ps, cin), tf.float32), tf.TensorSpec((ps, ps, 4), tf.float32)))
    ds = ds.batch(cfg["batch_size"], drop_remainder=True).prefetch(4)
    val_ds = tf.data.Dataset.from_tensor_slices((xv, yv)).batch(16)

    model = build_model_B(cfg)
    compile_like_model_A(model, cfg)
    n_params = model.count_params()
    print(f"parameters: {n_params:,}", flush=True)
    with open(mdir / "architecture.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda s, **k: f.write(s + "\n"))
    (mdir / "architecture_layers.json").write_text(json.dumps(describe(model), indent=1), encoding="utf-8")

    per_level = PerLevelVal(xv, yv, Pv, every=5)
    es, lr = cfg["early_stopping"], cfg["lr_schedule"]
    cbs = [
        keras.callbacks.ModelCheckpoint(str(mdir / "best.keras"), monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor=es["monitor"], patience=es["patience"],
                                      restore_best_weights=es["restore_best_weights"]),
        keras.callbacks.ReduceLROnPlateau(monitor=lr["monitor"], factor=lr["factor"], patience=lr["patience"],
                                          min_lr=lr["min_lr"], verbose=1),
        keras.callbacks.CSVLogger(str(mdir / "history.csv")),
        per_level,
    ]
    t_train = time.time()
    hist = model.fit(ds, epochs=cfg["max_epochs"], steps_per_epoch=cfg["steps_per_epoch"], validation_data=val_ds,
                     callbacks=cbs, verbose=2)
    train_s = time.time() - t_train
    model.save(mdir / "final_restored_best.keras")

    best_epoch = int(np.argmin(hist.history["val_loss"])) + 1
    info = {
        "model": name, "parameters": int(n_params), "epochs_run": len(hist.history["loss"]), "best_epoch": best_epoch,
        "best_val_loss": float(min(hist.history["val_loss"])),
        "best_val_psnr_patches": float(hist.history["val_masked_psnr"][best_epoch - 1]),
        "final_val_mae": float(hist.history["val_masked_mae"][-1]),
        "final_learning_rate": float(hist.history["learning_rate"][-1]) if "learning_rate" in hist.history else None,
        "early_stopped": len(hist.history["loss"]) < cfg["max_epochs"],
        "train_seconds": round(train_s), "train_hours": round(train_s / 3600, 2),
        "patches_seen": len(hist.history["loss"]) * cfg["steps_per_epoch"] * cfg["batch_size"],
        "per_level_val_psnr_history": per_level.history,
        "hardware": {"cpu": platform.processor(), "machine": platform.machine(), "os": platform.platform(),
                     "gpus_visible_to_tf": [d.name for d in tf.config.list_physical_devices("GPU")],
                     "tensorflow": tf.__version__, "keras": keras.__version__, "python": platform.python_version()},
        "train_images": len(stream.ref), "val_images": len(set(idv)), "val_patches": len(xv),
    }
    (mdir / "training_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "per_level_val_psnr_history"}, indent=1))


if __name__ == "__main__":
    main()
