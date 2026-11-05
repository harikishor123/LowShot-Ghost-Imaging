#!/usr/bin/env python3
"""Train Model A (baseline U-Net, noisy image only). Uses ONLY train references for training
and ONLY val references for validation/early stopping. The test split is never read here.

    python src/train.py --config configs/model_A_baseline.json
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

from dataset_loader import TrainPatchStream, fixed_validation_patches  # noqa: E402
from losses import masked_mae, masked_mse, masked_psnr  # noqa: E402
from unet import build_unet, describe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class PerLevelVal(keras.callbacks.Callback):
    """Per-photon-level masked PSNR on the fixed validation patches (logged, not used to train)."""

    def __init__(self, xv, yv, Pv, every=5):
        super().__init__()
        self.xv, self.yv, self.Pv, self.every = xv, yv, Pv, every
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every and epoch != 0:
            return
        pred = self.model.predict(self.xv, batch_size=16, verbose=0)
        row = {"epoch": epoch + 1}
        for P in sorted(set(self.Pv.tolist())):
            s = self.Pv == P
            m = self.yv[s, ..., 3:4]
            mse = float(np.sum((pred[s] - self.yv[s, ..., :3]) ** 2 * m) / (3 * m.sum()))
            row[f"val_psnr_P{int(P)}"] = 10 * np.log10(1 / mse)
        self.history.append(row)
        print("  per-level val PSNR:", {k: round(v, 2) for k, v in row.items() if k != "epoch"}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model_A_baseline.json")
    a = ap.parse_args()
    cfg = json.load(open(ROOT / a.config, encoding="utf-8"))
    name = cfg["model_name"]
    mdir = ROOT / "models" / name
    mdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / a.config, mdir / "training_config.json")

    keras.utils.set_random_seed(cfg["seeds"]["global"])
    if cfg.get("deterministic_ops"):
        tf.config.experimental.enable_op_determinism()

    # ---- data -------------------------------------------------------------------------------
    t0 = time.time()
    stream = TrainPatchStream(cfg, seed=cfg["seeds"]["train_stream"])
    print(f"train samples (image x level): {len(stream.rows)}, train images: {len(stream.ref)}, "
          f"min valid corners/image: {min(stream.n_corners.values())}", flush=True)
    xv, yv, Pv, idv = fixed_validation_patches(cfg, seed=cfg["seeds"]["val_patches"])
    print(f"val patches: {len(xv)} from {len(set(idv))} val images ({time.time() - t0:.0f}s)", flush=True)
    assert not set(idv) & set(stream.ref), "validation image found in training set"
    ps = cfg["patch_size"]
    ds = tf.data.Dataset.from_generator(
        lambda: stream.generator(cfg["loader_threads"], cfg["shuffle_buffer"]),
        output_signature=(tf.TensorSpec((ps, ps, 3), tf.float32), tf.TensorSpec((ps, ps, 4), tf.float32)))
    ds = ds.batch(cfg["batch_size"], drop_remainder=True).prefetch(4)
    val_ds = tf.data.Dataset.from_tensor_slices((xv, yv)).batch(16)

    # ---- model -------------------------------------------------------------------------------
    arch = cfg["architecture"]
    model = build_unet(3, 3, arch["base_filters"], arch["depth"], name=name)
    o = cfg["optimizer"]
    model.compile(optimizer=keras.optimizers.Adam(o["learning_rate"], beta_1=o["beta_1"], beta_2=o["beta_2"],
                                                  clipnorm=o["clipnorm"]),
                  loss=masked_mse, metrics=[masked_mae, masked_psnr])
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
