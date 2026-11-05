# Phase 2: Original clean object target - summary report

**Phase 2 design:** original clean object → simulated GI (noise-free) → DGI-N → U-Net (AI-original-N) → compared with the original clean object.

**Independence from Phase 1:** Phase 2 is separate from Phase 1, whose target is the 10k DGI reconstruction. All 72 Phase 1 output files were verified byte-identical before and after this phase.

## Setup (identical to Phase 1 except for the target)

- **Input:** clipped calibrated DGI-N, N = 500 / 1,000 / 2,000 / 5,000. Training inputs come from Phase 1's `pool_train.npy` (read-only): 341 training images × 32 random sequences.
- **Target:** the original clean preprocessed 64×64 object. It is never a network input.
- **Models:** UNetGI (1,925,025 parameters); Adam, learning rate 1e-4; MSE loss; batch 16; 80 epochs; patience 10; seed 0. Best epochs: 80, 80, 80 and 79 (all at or near the cap).
- **Checkpoint selection:** 75 validation images × seeds 2040-2042, scored against the clean object.
- **Evaluation:** seeds 2026-2029. The seed-2026 DGI reconstructions were checked against the existing Phase 1 reference file (identical, max difference 0.0).
- **Test:** 74 images × 4 sequences = 296 pairs, run once after validation froze the checkpoints (MD5s recorded in `frozen_for_test.json`).
- **Fidelity criterion (pre-registered before training):** AI-original-N must be non-inferior to conventional DGI-10000 against the clean object. The lower bound of the paired 95% bootstrap CI must be ≥ -0.02 for SSIM and ≥ -0.01 for Pearson r.

## Test results against the original clean object (mean ± std over 296 pairs)

| Method | SSIM | Pearson r | PSNR (dB) | MSE | Edge r |
|---|---|---|---|---|---|
| DGI-500 | 0.117 ± 0.020 | 0.306 ± 0.010 | 11.92 ± 3.53 | 0.0838 ± 0.0506 | 0.083 ± 0.036 |
| **AI-original-500** | 0.545 ± 0.114 | 0.821 ± 0.039 | 23.58 ± 4.59 | 0.0070 ± 0.0058 | 0.548 ± 0.112 |
| DGI-1000 | 0.188 ± 0.030 | 0.421 ± 0.012 | 14.02 ± 3.91 | 0.0549 ± 0.0373 | 0.169 ± 0.041 |
| **AI-original-1000** | 0.640 ± 0.089 | 0.867 ± 0.027 | 24.80 ± 4.66 | 0.0053 ± 0.0044 | 0.640 ± 0.092 |
| DGI-2000 | 0.293 ± 0.044 | 0.556 ± 0.012 | 16.51 ± 4.16 | 0.0325 ± 0.0240 | 0.303 ± 0.048 |
| **AI-original-2000** | 0.731 ± 0.064 | 0.907 ± 0.019 | 26.30 ± 4.61 | 0.0037 ± 0.0031 | 0.729 ± 0.077 |
| DGI-5000 | 0.469 ± 0.064 | 0.735 ± 0.007 | 20.10 ± 4.34 | 0.0147 ± 0.0115 | 0.540 ± 0.059 |
| **AI-original-5000** | 0.817 ± 0.045 | 0.941 ± 0.014 | 28.22 ± 4.40 | 0.0023 ± 0.0019 | 0.820 ± 0.060 |
| DGI-10000 (conventional baseline, not ground truth) | 0.611 ± 0.071 | 0.842 ± 0.003 | 23.00 ± 4.40 | 0.0076 ± 0.0061 | 0.713 ± 0.051 |

## AI-original-N minus DGI-10000 (paired; mean [95% CI low]; % of pairs where AI is better)

| N | ΔSSIM | Δr | ΔPSNR (dB) | ΔEdge r | Criterion |
|---|---|---|---|---|---|
| 500 | -0.066 [-0.093] (35%) | -0.021 [-0.030] (29%) | +0.58 [+0.40] (76%) | -0.166 [-0.188] (6%) | **Fail** |
| 1,000 | +0.029 [+0.008] (58%) | +0.025 [+0.019] (82%) | +1.79 [+1.62] (100%) | -0.073 [-0.091] (17%) | **Pass** |
| 2,000 | +0.121 [+0.104] (97%) | +0.065 [+0.061] (100%) | +3.30 [+3.12] (100%) | +0.015 [+0.002] (61%) | **Pass** |
| 5,000 | +0.206 [+0.193] (100%) | +0.099 [+0.096] (100%) | +5.22 [+5.03] (100%) | +0.107 [+0.097] (97%) | **Pass** |

**Minimum count meeting the pre-registered criterion: 1,000 measurements**, a **10× reduction** relative to 10,000-shot DGI, in the simulated noise-free setting. Validation gave the same result (1,000).

**AI against DGI at the same shot count:** AI-original-N exceeds DGI-N on all five metrics for 100% of pairs at every N. For example, at 2,000 shots: +0.439 SSIM, +0.352 r, +9.79 dB.

**Shot-count increments (paired, all improvements on 100% of pairs):**

| Step | ΔSSIM | Δr | ΔPSNR (dB) |
|---|---|---|---|
| 500 → 1,000 | +0.095 | +0.046 | +1.21 |
| 1,000 → 2,000 | +0.092 | +0.040 | +1.50 |
| 2,000 → 5,000 | +0.086 | +0.034 | +1.92 |

## Interpretation

- **1,000 shots** is the minimum count whose fidelity to the clean object is non-inferior to conventional 10,000-shot DGI on SSIM and correlation. It is clearly better on PSNR (100% of pairs). It is still below DGI-10000 on edge correlation (-0.073), so fine edges and texture are not fully recovered.
- **2,000 shots** is the first count that matches or exceeds DGI-10000 on all five metrics, including edge correlation (+0.015, CI lower bound +0.002). It is the practical trade-off when fine structure matters (5× reduction).
- **5,000 shots** gives the highest fidelity tested (SSIM 0.817, r 0.941) but only a 2× reduction.
- **Fidelity keeps rising** with shot count throughout 500 → 5,000; no plateau within the tested range.
- **No result reaches identity with the clean object.** The best case is SSIM 0.817, r 0.941, MSE 0.0023. Statements such as "identical copy" or "100% reconstruction" are not supported.

## Caveats

1. **The test images are not untouched for this question.** The same 74 images were already evaluated in Phase 1 and in the supplementary clean-object analysis. No Phase 2 training or model selection used them, and the test was run once after freezing.
2. **The clean object is the simulation's source image**, not an independently acquired ground truth. The measurements are noise-free simulations.
3. **The models were probably still improving:** best epochs were 79-80 of 80.
4. **Figure layout:** in the per-image comparison figures, the overall figure title partly overlaps the panel titles. The panels and their labels are correct, and the problem is cosmetic. Redrawing the figures would mean regenerating the reconstructions, so they were left as they are.

## Files

- `models/model_{500,1000,2000,5000}/`: `checkpoints/best.pt` and `last.pt`, `history.csv`, `batch_losses.csv`, `training_report.json`, loss and metric curves
- `results/test/` and `results/val/`:
  - `per_image_results.csv`
  - `aggregate_metrics.csv` (all seeds, seed 2026, unseen 2027-2029)
  - `dgi_baseline_results.csv`
  - `ai_vs_dgi_comparison.csv`
  - `evaluation_report.json`
- `figures/test/` and `figures/val/`:
  - `metrics_vs_shots_vs_clean.png`
  - `comparison_<image>.png` (Original | DGI-N | AI-original-N for N = 500 / 1,000 / 2,000 / 5,000 | DGI-10000)
  - `comparison_all_examples.png`
- `reports/`: `protocol.json`, `frozen_for_test.json`, training, validation and test console logs, this report
- Code: `orig_train.py`, `orig_evaluate.py` (Phase 1 code unchanged)
