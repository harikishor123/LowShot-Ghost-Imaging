#!/usr/bin/env python3
"""Lightweight U-Net for photon-limited reconstruction (Model A baseline).

Fully convolutional, so it runs on any H, W divisible by 2**depth (pad otherwise).

  input  : noisy linear-light RGB, 3 channels (Model A gets NOTHING else)
  encoder: depth x [Conv3x3-ReLU, Conv3x3-ReLU] -> MaxPool2x2      (filters base*2^d)
  bottleneck: Conv3x3-ReLU x2                                       (base*2^depth)
  decoder: depth x [ConvTranspose2x2 stride 2 -> concat skip -> Conv3x3-ReLU x2]
  head   : Conv1x1 -> sigmoid  => reconstruction in (0, 1), linear light

No normalization layers: BatchNorm statistics would mix photon levels within a batch and
couple the output to batch composition; the network is small enough to train without it.
"""

from __future__ import annotations

import keras
from keras import layers


def conv_block(x, filters: int, name: str):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", kernel_initializer="he_normal",
                      name=f"{name}_conv1")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", kernel_initializer="he_normal",
                      name=f"{name}_conv2")(x)
    return x


def build_unet(in_channels: int = 3, out_channels: int = 3, base_filters: int = 16, depth: int = 4,
               name: str = "unet", input_name: str = "noisy_linear") -> keras.Model:
    inp = keras.Input(shape=(None, None, in_channels), name=input_name)
    x, skips = inp, []
    for d in range(depth):
        x = conv_block(x, base_filters * 2 ** d, f"enc{d}")
        skips.append(x)
        x = layers.MaxPool2D(2, name=f"down{d}")(x)
    x = conv_block(x, base_filters * 2 ** depth, "bottleneck")
    for d in reversed(range(depth)):
        x = layers.Conv2DTranspose(base_filters * 2 ** d, 2, strides=2, padding="same", name=f"up{d}")(x)
        x = layers.Concatenate(name=f"skip{d}")([x, skips[d]])
        x = conv_block(x, base_filters * 2 ** d, f"dec{d}")
    out = layers.Conv2D(out_channels, 1, activation="sigmoid", name="clean_linear")(x)
    return keras.Model(inp, out, name=name)


def describe(model: keras.Model) -> list[dict]:
    rows = []
    for l in model.layers:
        try:
            shape = tuple(l.output.shape)
        except Exception:
            shape = None
        rows.append({"layer": l.name, "type": l.__class__.__name__, "output_shape": str(shape),
                     "params": int(l.count_params())})
    return rows
