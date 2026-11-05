#!/usr/bin/env python3
"""Extension with Poisson bucket noise: write protocol.json BEFORE the pools, training or evaluation.

    python ext_poisson_write_protocol.py

Refuses to overwrite an existing protocol.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ext_poisson_common import (EVAL_SEEDS, FROZEN_EXT, K, LEVELS, LOW, NOISE_TAG, OBJECTS_MD5, OBJECTS_NPZ, OUT,  # noqa: E402
                                PROTOCOL, SELECTION_SEEDS, md5, write_json)


def main() -> None:
    if PROTOCOL.exists() or (OUT / "pool").exists() or (OUT / "models").exists():
        sys.exit("protocol.json, pools or models already exist: the protocol must precede them and is never rewritten")
    assert md5(OBJECTS_NPZ) == OBJECTS_MD5
    base = json.loads((FROZEN_EXT / "protocol.json").read_text())
    protocol = {
        "experiment": "Extension: broader microscopy dataset with Poisson bucket noise (MedMNIST+ BloodMNIST and PathMNIST, 64x64)",
        "written_before_pool_training_or_evaluation": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "relation_to_other_work": "separate experiment. The onion study (FREEZE_MANIFEST_phase1_phase2.md5) and the noise-free "
                                  "extension (outputs/extension_medmnist_microscopy/FREEZE_MANIFEST_extension.md5) are "
                                  "read-only and checked before/after. Same images, pattern seeds, DGI, U-Net, recipe and pass "
                                  "rule as the noise-free extension; the only change is Poisson noise on the bucket values.",
        "claim_scope": "fewer simulated measurement shots under simulated Poisson bucket noise at three relative levels; "
                       "the photon scale is set per image to fix the RELATIVE noise level, so no photon-dose, "
                       "biological-effect, hardware or quantum claim. Noisy DGI-10000 is a study-specific benchmark, not ground truth.",
        "data": {"objects_npz": str(OBJECTS_NPZ.relative_to(ROOT)), "objects_npz_md5": OBJECTS_MD5,
                 "final_counts": base["data"]["final_counts"], "splits": base["data"]["splits"],
                 "sources": base["data"]["sources"]},
        "noise": {
            "model": "B' = Poisson(s * B) / s on every bucket value B = sum(P * x); reference arm R = sum(P) noise-free",
            "photon_scale": "s = 2 sum(x) / (f^2 sum(x^2)) per image, so noise std ~= f * sigma_B, "
                            "sigma_B = 0.5 sqrt(sum(x^2)) = spread of the noise-free bucket over Bernoulli(0.5) patterns",
            "levels_f": LEVELS,
            "noise_rng": f"default_rng([acquisition seed, {NOISE_TAG}, round(100 f)]), drawn shot by shot (nested prefixes)",
            "pre_check": "f = 0 reproduces the frozen noise-free extension pool exactly (max abs diff 0.0, 3 spot checks); "
                         "measured noise std / sigma_B = 0.0499 / 0.0997 / 0.2994 (blood image) and "
                         "0.0498 / 0.0997 / 0.2993 (path image)"},
        "simulation": base["simulation"] | {"buckets": "Poisson-noisy (see noise), float64"},
        "training": {
            "images": "7,000 training images (3,500 blood + 3,500 path); one model per noise level and N (12 models)",
            "target": "original clean object (models AI-original-N, as the noise-free extension's clean target). "
                      "No DGI-10000-target models: with noise the 10k DGI is itself noisy.",
            "input": "noisy DGI-N of the same level",
            "sequences": f"pool of K = {K} sequences per training image, pattern seeds as the noise-free extension; "
                         "one realisation per image per epoch via default_rng([0, epoch])",
            "architecture": "UNetGI(base=32, depth=3), 1,925,025 parameters",
            "optimiser": "Adam lr 1e-4", "loss": "MSE", "batch_size": 16, "max_epochs": 80, "patience": 10, "master_seed": 0,
            "selection": f"1,000 validation images x shared sequences {SELECTION_SEEDS} at the same noise level; "
                         "checkpoint = lowest selection MSE vs clean object",
            "hardware": "NVIDIA RTX A5000 (CUDA), torch.use_deterministic_algorithms(True), CUBLAS_WORKSPACE_CONFIG=:4096:8"},
        "evaluation": {
            "seeds": EVAL_SEEDS, "test_pairs": "2,000 test images x 4 sequences = 8,000 per noise level",
            "reference": "original clean object",
            "comparator": "noisy DGI-10000 of the same acquisition and noise level",
            "methods": ["noisy DGI-500..10000", "smoothed noisy DGI-N (Gaussian sigma per level and N minimising MSE vs "
                        "the clean object on training pool k = 0, 1; grid final_evaluate.SIGMAS)", "AI-original-N"],
            "context": "noise-free DGI-10000 of the same pattern sequence, and AI-original-N minus noise-free DGI-10000 "
                       "(secondary, reported with the same rule)",
            "metrics": base["evaluation"]["metrics"]},
        "criterion": base["criterion"] | {
            "candidates": "AI-original-N and smoothed DGI-N, each vs noisy DGI-10000, separately per noise level"},
        "order_of_work": ["this protocol", "pools + selection set", "train 12 models",
                          "validation per level -> freeze checkpoints (MD5) and sigmas in results/level_*/frozen_for_test.json",
                          "single test evaluation per level (script refuses a second run)", "freeze manifest"],
        "deviations_from_noise_free_extension": [
            "Poisson bucket noise at three relative levels (the purpose of this experiment)",
            "clean-target models only",
            "pools store N = 500..5000 only (10k not needed as a training input)",
            "pattern batches of 100 instead of 500 (identical pattern stream; consecutive generator draws)",
            "smoothed-DGI baseline tuned against the clean object (not the 10k DGI)"],
        "seeds": {"patterns": "identical to the noise-free extension (pool, selection, evaluation) so the noise is the only change",
                  "noise": f"[pattern seed, {NOISE_TAG}, 5 | 10 | 30]"},
    }
    write_json(PROTOCOL, protocol)
    print(f"wrote {PROTOCOL.relative_to(ROOT)} ({protocol['written_before_pool_training_or_evaluation']})")


if __name__ == "__main__":
    main()
