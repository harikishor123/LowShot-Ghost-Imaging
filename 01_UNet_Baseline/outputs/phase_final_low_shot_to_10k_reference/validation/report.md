# Final experiment: fewer GI shots vs the existing 10,000-shot reference (VALIDATION)

Status: validation complete; models, preprocessing, smoothing sigma and the comparability rule are frozen
(`frozen_for_test.json`, with checkpoint MD5s). The 74-image test set has NOT been opened.
All 223 earlier output files are byte-identical before and after this experiment.

## Setup

- **Reference (target):** existing `phase2a_dgi_dataset/reconstructions/dgi_estimates_seed2026_64px.npz`,
  `estimate_unclipped_10000`, clipped to [0, 1]. This is calibrated DGI of a 10,000-shot noise-free
  acquisition with pattern seed 2026. The recomputed pipeline reproduces it exactly (max diff 0.0 on the
  validation images).
- **Inputs:** the first N = 500 / 1000 / 2000 / 5000 shots of the same acquisition, same DGI method, clipped.
- **Training:** 341 training images x 32 independent random sequences each (seeds 20,000,000 + 1000k + i).
  Input = DGI-N of a sequence; target = DGI-10000 of the same sequence. One realisation per image per
  epoch. UNetGI (1.93 M parameters), Adam 1e-4, MSE loss, batch 16, 80 epochs, patience 10, seed 0.
  Checkpoints selected on validation images x seeds 2040-2042. No noise: the reference acquisition is noise-free.
- **Evaluation:** 75 validation images x seed 2026 (existing reference) + unseen sequences 2027-2029, each
  against the 10k reference of the same acquisition (300 image-seed pairs).
- **Comparison methods:** conventional DGI-N (the reference's own method); smoothed DGI-N (Gaussian sigma
  tuned on training sequences to match their 10k: 1.5 / 1.125 / 0.75 / 0.5); AI-N.
- **Yardstick:** an independent 10,000-shot acquisition of the same image (seed + 7000) scored against the
  reference. This shows how closely a genuine second 10k measurement matches the reference.

## Results vs the 10k reference (validation, all 300 pairs; mean ± std)

| Shots | Method | PSNR | SSIM | MSE | r | edge r |
|---:|---|---|---|---:|---:|---:|
| 500 | conventional DGI | 12.59 ± 3.76 | 0.198 ± 0.023 | 0.0748 | 0.363 | 0.120 |
| 500 | smoothed DGI | 20.01 ± 4.68 | 0.355 ± 0.092 | 0.0159 | 0.659 | 0.253 |
| 500 | **AI-500** | 21.20 ± 4.62 | 0.404 ± 0.085 | 0.0121 | 0.705 | 0.395 |
| 1000 | conventional DGI | 14.96 ± 4.12 | 0.323 ± 0.027 | 0.0459 | 0.502 | 0.230 |
| 1000 | smoothed DGI | 21.17 ± 4.63 | 0.453 ± 0.068 | 0.0122 | 0.723 | 0.369 |
| 1000 | **AI-1000** | 21.97 ± 4.62 | 0.499 ± 0.063 | 0.0101 | 0.760 | 0.492 |
| 2000 | conventional DGI | 17.98 ± 4.35 | 0.499 ± 0.029 | 0.0239 | 0.663 | 0.403 |
| 2000 | smoothed DGI | 22.43 ± 4.56 | 0.593 ± 0.046 | 0.0090 | 0.792 | 0.512 |
| 2000 | **AI-2000** | 23.08 ± 4.59 | 0.618 ± 0.044 | 0.0078 | 0.820 | 0.604 |
| 5000 | conventional DGI | 23.63 ± 4.49 | 0.787 ± 0.021 | 0.0067 | 0.875 | 0.730 |
| 5000 | smoothed DGI | 25.63 ± 4.54 | 0.815 ± 0.021 | 0.0043 | 0.905 | 0.763 |
| 5000 | **AI-5000** | 25.78 ± 4.52 | 0.814 ± 0.021 | 0.0041 | 0.908 | 0.783 |
| 10000 | independent 10k acquisition | 20.61 ± 4.51 | 0.469 ± 0.061 | 0.0135 | 0.709 | 0.529 |

Seed 2026 alone (the existing reference file) and unseen sequences 2027-2029 agree to within
0.1 dB / 0.006 SSIM for every method (`summary.csv`).

## Pre-registered comparability rule (protocol.json, fixed before training)

AI-N is comparable if the lower bound of the 95% bootstrap CI of [metric(AI-N, ref) - metric(independent 10k, ref)]
is >= -0.02 for SSIM and >= -0.01 for r.

| Shots | AI: SSIM diff (95% CI) | AI: r diff (95% CI) | AI comparable? | Smoothed DGI comparable? | Conventional DGI comparable? |
|---:|---|---|---|---|---|
| 500 | -0.066 (-0.084, -0.048) | -0.004 (-0.011, +0.003) | **no** (SSIM) | no | no |
| 1000 | **+0.029 (+0.017, +0.043)** | **+0.051 (+0.046, +0.056)** | **yes** | no (SSIM CI low -0.030) | no |
| 2000 | +0.149 | +0.111 | yes | yes | no (r) |
| 5000 | +0.344 | +0.199 | yes | yes | yes |

**Minimum successful AI shot count on validation: 1000, a 10x measurement reduction** relative to the
10,000-shot reference.

## Interpretation and caveats (read before quoting the 10x)

1. **What "comparable" means here.** The reference is itself a noisy single acquisition. Two independent
   10k acquisitions agree only moderately (SSIM 0.47, r 0.71, PSNR 20.6 dB). AI-1000 agrees with the
   reference at least as well as a second 10k acquisition would. Because it outputs a denoised estimate,
   it beats the retest easily on PSNR (+1.36 dB, 100% of pairs) and modestly on SSIM (70% of pairs).
2. **Fine structure is not fully preserved at 1000.**
   - Edge r of AI-1000 is 0.037 below the independent 10k (only 24% of pairs at or above it).
   - Visually, AI-1000 recovers nuclei, cell walls and the field of view. Fine detail visible in the 10k
     reference (chromosome arms in A_383, texture inside the nucleus in A_358) is smoothed away.
   - AI-2000 exceeds the independent 10k on edge r as well (+0.076, 97% of pairs).
   - If edge preservation were required, the supported minimum would be **2000 shots (5x)**.
3. **Fairer conventional baselines.**
   - Simple smoothing already makes DGI-2000 meet the criterion.
   - Compared with smoothing, the AI's advantage is therefore about 2x (1000 vs 2000 shots).
   - Compared with unprocessed DGI (which needs 5000), it is 5x.
   - The 10x figure is relative to the 10,000-shot reference itself.
4. **Shared measurements.** Low-shot inputs are prefixes of the reference acquisition, so they share
   N/10000 of its measurements. This is why conventional DGI-5000 (50% overlap) scores well against the
   reference. At N = 1000 the overlap is 10%.
5. **Context against the clean image** (not the target): AI-1000 scores 24.6 dB / SSIM 0.60 / r 0.88
   against the clean onion image. The 10k reference itself scores 23.5 dB / SSIM 0.61.
6. **Noise-free measurements.** Phase 3/4A showed that noise-free DGI-10k is a weak conventional
   reconstruction (TV and least squares do far better) and that realistic bucket noise changes the
   picture. These results apply to the fixed noise-free 10k-DGI reference only.
7. **Training budget.** All four models' best epochs were 78-80 of 80, so they were probably still improving.
8. **Validation reuse.** Validation images have been reused across many phases. The test set gives the unbiased check.

## Generalisation

Results on unseen sequences 2027-2029 match the seed-2026 existing-reference results (AI-1000 SSIM 0.499
vs 0.499; PSNR 21.95 vs 22.03). Training used none of these sequences.

## Ready for test?

Yes. Everything is frozen in `frozen_for_test.json`:
- the four checkpoints (with MD5s);
- the sigmas;
- the seeds;
- the criterion.

The test run is `python final_evaluate.py --split test --final-test`. It refuses to run if any checkpoint
changed, and it applies the same rule once without changing N.
