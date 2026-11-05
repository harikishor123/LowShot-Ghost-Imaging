#!/usr/bin/env python3
"""Extension (broader microscopy dataset): write protocol.json BEFORE the training pool, training or evaluation.

    python ext_write_protocol.py

Refuses to overwrite an existing protocol.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ext_common import (EVAL_SEEDS, K, LOW, OBJECTS_NPZ, OUT, POOL_NPY, PROTOCOL, RETEST_OFFSET, SELECTION_SEEDS,  # noqa: E402
                        md5, write_json)


def main() -> None:
    if PROTOCOL.exists() or POOL_NPY.exists() or (OUT / "models").exists():
        sys.exit("protocol.json, the pool or models already exist: the protocol must precede them and is never rewritten")
    ds = json.loads((OUT / "dataset" / "dataset_report.json").read_text())
    protocol = {
        "experiment": "Extension: broader microscopy dataset (MedMNIST+ BloodMNIST and PathMNIST, 64x64)",
        "written_before_pool_training_or_evaluation": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "relation_to_onion_study": "separate experiment; reuses the onion simulation, DGI, U-Net, training recipe, seeds "
                                   "policy and pass rule unchanged except where listed under deviations. The frozen onion "
                                   "Phase 1 / Phase 2 files are read-only (FREEZE_MANIFEST_phase1_phase2.md5 checked before/after).",
        "claim_scope": "fewer simulated, noise-free measurement shots; no photon-dose, biological-effect, hardware or quantum claim. "
                       "DGI-10000 is a study-specific benchmark, not ground truth.",
        "data": {"sources": ds["sources"], "objects_npz": ds["objects_npz"], "objects_npz_md5": md5(OBJECTS_NPZ),
                 "sampling": ds["sampling"], "conversion": ds["conversion"], "final_counts": ds["final_counts"],
                 "screening": ds["screening"],
                 "splits": "MedMNIST official train / val / test (PathMNIST test = CRC-VAL-HE-7K, a different centre "
                           "from its train/val NCT-CRC-HE-100K)"},
        "simulation": {"patterns": "i.i.d. Bernoulli(0.5) binary 64x64", "buckets": "noise-free, float64",
                       "reconstruction": "calibrated DGI (final_build_pool.dgi_readouts, unchanged), clipped to [0,1]",
                       "shots": LOW + [10000], "nesting": "DGI-N uses the first N shots of the same 10,000-shot sequence"},
        "training": {
            "images": "7,000 training images (3,500 blood + 3,500 path); one model per N and target, trained on both subsets",
            "sequences": f"pool of K = {K} independent sequences per training image; seed = 30,000,000 + 10,000*k + train_index; "
                         "one realisation per image per epoch via default_rng([0, epoch])",
            "targets": {"clean": "original clean object (models AI-original-N; as onion Phase 2)",
                        "dgi10k": "DGI-10000 of the same acquisition (models AI-N; as onion Phase 1)"},
            "architecture": "UNetGI(base=32, depth=3), 1,925,025 parameters",
            "optimiser": "Adam lr 1e-4", "loss": "MSE", "batch_size": 16, "max_epochs": 80, "patience": 10, "master_seed": 0,
            "selection": f"1,000 validation images x shared sequences {SELECTION_SEEDS}; checkpoint = lowest selection MSE",
            "hardware": "NVIDIA RTX A5000 (CUDA), torch.use_deterministic_algorithms(True), CUBLAS_WORKSPACE_CONFIG=:4096:8"},
        "evaluation": {
            "seeds": EVAL_SEEDS, "test_pairs": "2,000 test images x 4 sequences = 8,000 (4,000 per subset)",
            "clean_target": {"reference": "original clean object", "comparator": "conventional DGI-10000 of the same sequence",
                             "methods": ["DGI-500..10000", "AI-original-500..5000"]},
            "dgi10k_target": {"reference": "DGI-10000 of the same acquisition",
                              "comparator": f"independent 10k acquisition (seed + {RETEST_OFFSET})",
                              "methods": ["conventional DGI-N", "smoothed DGI-N (Gaussian sigma tuned on training pool k=0,1)", "AI-N"],
                              "context": "every image also scored against the clean object"},
            "metrics": ["SSIM (skimage, data_range 1)", "Pearson r", "PSNR", "MSE", "edge r (Pearson r of Sobel magnitude maps)"]},
        "criterion": {
            "definition": "on paired image-level differences [metric(method) - metric(comparator)] pooled over seeds 2026-2029, "
                          "the lower bound of the 95% bootstrap CI (2,000 resamples over images, seeds kept together, "
                          "rng default_rng(0) per group) is >= -0.02 for SSIM AND >= -0.01 for Pearson r",
            "delta": {"SSIM": 0.02, "Pearson_r": 0.01},
            "minimum_N": "smallest N in {500, 1000, 2000, 5000} meeting the criterion; reduction factor = 10000 / N",
            "primary": "all 2,000 test images (blood + path)",
            "secondary": "blood and path separately (same rule, images of that subset only)"},
        "order_of_work": ["build dataset (done)", "this protocol", "training pool + selection set", "train 8 models",
                          "validation evaluation -> freeze checkpoints (MD5) in results/target_*/frozen_for_test.json",
                          "single test evaluation per target (script refuses a second run)", "extension freeze manifest"],
        "deviations_from_onion": [
            "dataset: MedMNIST+ blood smears and H&E tissue instead of onion; images are already 64x64, so no centre crop or area downsampling",
            "pool seed rule 30,000,000 + 10,000*k + i (onion 20,000,000 + 1000*k + i would collide for i >= 1000)",
            "training on GPU with deterministic algorithms (onion: CPU); not bit-identical to CPU training",
            "an epoch has 7,000 images (onion 341), so 80 epochs is ~20x more gradient steps",
            "duplicate screening vectorised; same thresholds as src/dataset_audit_full.py, NCC for flips taken at the best pHash transform",
            "near-blank images (object std < 0.005) excluded before the final draw"],
        "seed_ranges_reserved": {"pool": "30,000,000-30,316,999", "selection": SELECTION_SEEDS, "evaluation": EVAL_SEEDS,
                                 "retest": [s + RETEST_OFFSET for s in EVAL_SEEDS]},
    }
    write_json(PROTOCOL, protocol)
    print(f"wrote {PROTOCOL.relative_to(ROOT)} ({protocol['written_before_pool_training_or_evaluation']})")


if __name__ == "__main__":
    main()
