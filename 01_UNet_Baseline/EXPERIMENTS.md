# Experiments

All three experiments use the same simulation and evaluation:
- **Simulation:** binary Bernoulli(0.5) illumination patterns on a 64 × 64 object, single-pixel bucket values, and calibrated DGI read out after N = 500, 1,000, 2,000, 5,000 and 10,000 shots (nested prefixes of one 10,000-shot sequence), clipped to [0, 1].
- **Model and training:** a U-Net (UNetGI, base 32, 1,925,025 parameters) maps DGI-N to a target. Training uses Adam 1e-4, MSE loss, batch 16, at most 80 epochs, patience 10 and master seed 0. Each training image has 32 independent pattern sequences.
- **Evaluation:** test-time illumination sequences 2026–2029, never used in training.
- **Pass rule (fixed before training):** the paired 95% bootstrap lower bound (2,000 resamples over images, sequences kept together) must be ≥ −0.02 for SSIM **and** ≥ −0.01 for Pearson r.
- **Order of work:** validation first, then checkpoints frozen with MD5 hashes, then **one** test run.

## 1 · Original onion study (frozen; `FREEZE_MANIFEST_phase1_phase2.md5`, 134 files)

**Data:** onion microscopy images from Roboflow (`train/`, `valid/`). After cleaning, duplicate removal and field-of-view cropping (`src/build_v2_dataset.py`), the set in `data/v2/` has 341 training, 75 validation and 74 test images.

**Phase 1** (`final_build_pool.py`, `final_train.py`, `final_evaluate.py` → `outputs/phase_final_low_shot_to_10k_reference/`):
- **Design:** AI-N is trained towards the 10,000-shot DGI of the same acquisition and compared with an independent 10,000-shot acquisition.
- **Result:** AI-1000 passes (10×). Gaussian-smoothed DGI first passes at 2,000 shots and raw DGI at 5,000.

**Phase 2** (`orig_train.py`, `orig_evaluate.py` → `outputs/phase_final_original_target/`):
- **Design:** AI-original-N is trained towards the original clean object and compared with conventional DGI-10000.
- **Result:** AI-original-1000 passes (10×), and AI-original-2000 matches or exceeds DGI-10000 on all five metrics.

**Earlier exploratory phases** (history, kept; superseded by Phases 1 and 2):

| Phase | Script(s) | Output |
|---|---|---|
| 1, 1b: correlation GI and DGI on one image | `run_single_image.py`, `run_dgi_single_image.py` | `outputs/phase1_single_image/`, `outputs/phase1b_dgi/` |
| 2A: dataset-scale DGI baseline; **holds the Phase 1 seed-2026 reference** | `run_dgi_dataset.py` | `outputs/phase2a_dgi_dataset/` |
| 2B: first U-Net models A–D | `train_unet_gi.py`, `train_unet_gi_random.py`, `train_model_D.py`, `compare_models_AB.py`, `run_generalization_validation.py`, `run_measurement_curve.py` | `outputs/phase2b_unet/` |
| 3: TV / compressed-sensing baselines | `run_strong_baselines.py` | `outputs/phase3_strong_baselines/` |
| 4A: early noisy-bucket baselines (onion, fixed photon scale) | `phase4a_noise_setup.py`, `run_phase4a.py` | `outputs/phase4a_noisy_baselines/` (not in the thesis) |
| earliest: photon-conditioned image-domain Models A/B | `src/train.py`, `src/train_model_B.py`, `configs/` | `models/`, `results/evaluation/` |

## 2 · Microscopy extension, no added noise (frozen; `FREEZE_MANIFEST_extension.md5`, 131 files)

**Data:** MedMNIST+ 64 × 64 (Zenodo 10.5281/zenodo.10519652, CC BY 4.0), drawn from the official splits:
- **BloodMNIST:** 3,500 / 500 / 1,000 images.
- **PathMNIST:** 3,500 / 500 / 1,000 images. Its test split (CRC-VAL-HE-7K) comes from a different clinical centre.

Images were converted to luminance exactly as for onion and screened for duplicates: 1 exact duplicate removed, 47 near-blank images excluded. The data are in `data/extension_medmnist/`.

**Design:** the same two Phase 1 / Phase 2 designs, one model per N trained on both collections. Training was on GPU with deterministic algorithms. The protocol was written before training.

**Result (test, 2,000 images × 4 sequences):**

| Method | All images | Blood | Tissue |
|---|---|---|---|
| U-Net, clean-object design | 1,000 | 500 | 5,000 |
| U-Net, 10k-DGI design | 1,000 | 500 | 2,000 |
| Smoothed DGI | 2,000 | 1,000 | 2,000 |
| Raw DGI | 5,000 | 5,000 | 5,000 |

Validation and test agree in all 12 cases.

## 3 · Poisson bucket-noise experiments (frozen; `FREEZE_MANIFEST_poisson.md5`, 180 files)

**Noise model** (`ext_poisson_common.py:96`): every bucket value B becomes B' = Poisson(s·B)/s, with s = 2ΣO / (f²ΣO²). This sets the noise standard deviation to f × the shot-to-shot spread of the bucket signal, for f = 5, 10 and 30%. The reference arm is noise-free.

**Design:**
- **Pairing:** the same images and pattern seeds as the extension, so the noise is the only change.
- **Models:** 12 clean-target models, one per noise level and shot count.
- **Comparator:** noisy DGI-10000 of the same acquisition.
- **Secondary checks:** a noise-free DGI-10000 comparison and a smoothed-DGI baseline.

**Result (test):**
- **All images:** passes at 1,000 shots at every noise level.
- **Blood:** passes at 500.
- **Tissue:** needs 5,000 / 2,000 / 2,000 shots at 5 / 10 / 30% noise. The lower requirement at 10–30% reflects a weaker noisy comparator, not a better reconstruction.
- **Against noise-free DGI-10000:** the minimum is 1,000 / 1,000 / 2,000 shots. Tissue at 30% never passes.
- **Borderline:** validation and test differ in 2 of 27 cases, both borderline at 10% noise.

## Known limitations (see the audit reports)

- **Simulation only:** no hardware, photon-dose or quantum result.
- **Over-smoothing:** fine texture and small structures are partly lost, especially in tissue and at 500–1,000 shots.
- **Training:** one training run per model, and most best checkpoints fall near the 80-epoch cap.
- **Smoke test:** before training, a smoke test ran the evaluation code on 24 test images with untrained networks. It influenced no decision.
- **Numerics:** GPU results reproduce exactly only with the same batch layout (TF32 rounding).
