# FINAL TEST RESULT: fewer GI shots vs the existing 10,000-shot reference

**Run:**
- Command `py -3.13 final_evaluate.py --split test --final-test`, executed once.
- Started 2026-10-06T15:31:53+05:30, completed 15:32:33, exit code 0.
- Provenance (`run_record.json`):
  - checkpoint MD5s match `frozen_for_test.json`;
  - protocol MD5 `fa82061d462ff01a70c7c37354d81d5a`;
  - test split `data/v2/splits/test/ids.txt` (74 images, MD5 `41235d66b43e2e6cf754fc0dbbaf9440`).

**Frozen inputs:**
- Nothing was retrained, retuned or modified.
- Models: AI-500/1000/2000/5000, best epochs 80/79/80/78.
- Smoothing sigma: 1.5 / 1.125 / 0.75 / 0.5.
- Evaluation seeds: 2026 (existing reference file) and 2027-2029 (unseen sequences).
- Retest = independent 10k acquisition, seed + 7000.
- Criterion from `protocol.json`: lower bound of the 95% bootstrap CI of the paired difference vs the
  independent 10k acquisition >= -0.02 (SSIM) and >= -0.01 (Pearson r).

**Integrity:** the recomputed seed-2026 acquisition matches the existing reference file exactly on the
test images (max diff 0.0 at 500 / 1000 / 5000 / 10000).

## 1. Metrics vs the 10k reference (74 test images x 4 seeds = 296 pairs; mean ± std)

| Shots | Method | PSNR | SSIM | r | edge r |
|---:|---|---|---|---|---|
| 500 | conventional DGI | 12.13 ± 3.53 | 0.200 ± 0.021 | 0.363 | 0.120 |
| 500 | smoothed DGI | 19.51 ± 4.51 | 0.348 ± 0.092 | 0.658 | 0.261 |
| 500 | AI-500 | 20.62 ± 4.41 | 0.394 ± 0.084 | 0.701 | 0.391 |
| 1000 | conventional DGI | 14.49 ± 3.91 | 0.325 ± 0.024 | 0.503 | 0.231 |
| 1000 | smoothed DGI | 20.67 ± 4.48 | 0.451 ± 0.070 | 0.725 | 0.381 |
| 1000 | AI-1000 | 21.44 ± 4.43 | 0.494 ± 0.064 | 0.759 | 0.492 |
| 2000 | conventional DGI | 17.48 ± 4.17 | 0.501 ± 0.028 | 0.664 | 0.404 |
| 2000 | smoothed DGI | 21.94 ± 4.41 | 0.594 ± 0.048 | 0.794 | 0.519 |
| 2000 | AI-2000 | 22.57 ± 4.41 | 0.617 ± 0.046 | 0.820 | 0.607 |
| 5000 | conventional DGI | 23.10 ± 4.35 | 0.787 ± 0.021 | 0.875 | 0.731 |
| 5000 | smoothed DGI | 25.13 ± 4.40 | 0.815 ± 0.022 | 0.905 | 0.766 |
| 5000 | AI-5000 | 25.27 ± 4.38 | 0.814 ± 0.023 | 0.908 | 0.786 |
| 10000 | independent 10k acquisition | 20.10 ± 4.36 | 0.469 ± 0.065 | 0.710 | 0.530 |

## 2. Pre-registered comparability (paired vs independent 10k acquisition; 95% bootstrap CI over images)

| Shots | Method | SSIM diff [CI] | r diff [CI] | Pass |
|---:|---|---|---|---|
| 500 | AI | -0.075 [-0.092, -0.058] | -0.008 [-0.015, -0.003] | **FAIL** (SSIM and r) |
| 1000 | AI | +0.025 [+0.013, +0.037] | +0.050 [+0.046, +0.053] | **PASS** |
| 2000 | AI | +0.147 [+0.139, +0.155] | +0.110 [+0.108, +0.113] | **PASS** |
| 5000 | AI | +0.344 [+0.335, +0.354] | +0.198 [+0.197, +0.199] | **PASS** |
| 500 / 1000 | smoothed DGI | -0.121 / -0.018 [CI low -0.031] | -0.052 / +0.015 | FAIL / FAIL |
| 2000 / 5000 | smoothed DGI | +0.124 / +0.346 | +0.084 / +0.195 | PASS / PASS |
| 500 / 1000 / 2000 | conventional DGI | negative except SSIM at 2000 | -0.347 / -0.206 / -0.046 | FAIL / FAIL / FAIL |
| 5000 | conventional DGI | +0.318 | +0.165 | PASS |

**Minimum successful AI shot count on the untouched test set: 1000. Measurement reduction 10000 / 1000 = 10x**
(relative to the existing 10,000-shot reference, under the frozen criterion).

## 3. Structural detail (edge r, pre-defined metric)

| Shots | AI edge r minus independent 10k [95% CI] | % pairs AI >= independent 10k |
|---:|---|---:|
| 500 | -0.139 [-0.151, -0.127] | 1% |
| 1000 | -0.038 [-0.047, -0.029] | 20% |
| 2000 | +0.077 [+0.070, +0.083] | 98% |
| 5000 | +0.256 [+0.248, +0.265] | 100% |

AI-1000 passes the frozen SSIM/r criterion but its edge correlation is below that of an independent 10k
acquisition. Edge r is not part of the pass/fail criterion. AI-2000 exceeds it.

Visually (figures, five test images chosen by a fixed evenly-spaced rule in the code): AI-1000 reproduces
nuclei positions, cell outlines and the field of view, but is smoother than the 10k reference and does not
reproduce its fine intranuclear texture.

## 4. Generalisation

- Unseen images: test results match validation to within 0.6 dB PSNR, 0.005 SSIM and 0.002 r for every AI model.
- Unseen illumination sequences: seed 2026 (existing reference) and unseen seeds 2027-2029 agree within
  0.08 dB / 0.004 SSIM for every AI model.
- The validation decision (AI-1000 = minimum passing N) is reproduced on the test set.

## 5. Scope of the result

- The comparison is against the established noise-free 10k-shot DGI reference.
- "Comparable" means: agrees with that reference at least as well as an independent 10k acquisition does
  (SSIM and r, pre-registered margins).
- No claims are made about photon dose, biological damage, other samples or noisy acquisitions.

## Files

`summary.csv`, `per_image_results.csv`, `comparability.csv`, `evaluation_report.json`, `run_record.json`,
`console.log`, `ssim_vs_shots.png`, `psnr_vs_shots.png`, `correlation_vs_shots.png`,
`edge_correlation_vs_shots.png`, `figures/comparison_N{500,1000,2000,5000}.png`
