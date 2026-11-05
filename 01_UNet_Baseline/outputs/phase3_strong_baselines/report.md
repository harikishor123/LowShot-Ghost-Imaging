# Phase 3 diagnostic: strong conventional baselines (validation only, noise-free)

Diagnostic experiment, not a final result. The 74-image test set was not loaded. Models C and D
were loaded read-only from their frozen best checkpoints. All earlier output folders were verified
byte-identical before and after this run.

## 1. What was implemented

Measurement model: `B = P O` with `P` in {0,1}^(N x 4096), i.i.d. Bernoulli(0.5), noise-free, float64.
Every reconstruction uses only `P_N` and `B_N`.

| Method | Formulation | Tuning |
|---|---|---|
| Raw DGI | Phase 2A DGI, calibrated, clipped to [0,1] | none |
| Smoothed DGI | Gaussian blur of raw DGI | sigma frozen from training-set tuning (500:1.75, 1000:1.375, 2000:1.125, 5000:0.875, 10000:0.75) |
| Pseudoinverse | thin SVD `P = U S V^T`, `O = V S^-1 U^T B`: minimum-norm solution if N < 4096, least squares if N >= 4096; clipped to [0,1] | none |
| TV | `min 0.5 ||W(P O - B)||^2 + lambda TV_iso(O)`, O in [0,1]; W whitens residuals, so the data term is `0.5 ||V^T(O - O_pinv)||^2` (N < 4096) or `0.5 ||O - O_ls||^2` (N >= 4096). FISTA, 300 iterations; prox by FGP (Beck-Teboulle), 10 inner iterations, warm-started | lambda = lowest mean MSE on 40 training images (seed 2026), frozen per N: 500: 1e-4, 1000: 1e-4, 2000: 3e-4, 5000: 0, 10000: 0 |
| Model D / Model C | frozen U-Nets on DGI-500 / DGI-1000 | none (no retraining) |

Seeds: 2026 is the main sequence (not used to train C/D). 2027-2029 are unseen. 75 validation images per seed.
Main numbers below are on the unseen seeds (225 image-seed pairs).

With noise-free data, the whitened TV objective has exactly the same set of data-consistent images as the
unweighted `0.5||PO - B||^2`; only the scale of lambda differs. The whitening was needed because the 0/1
patterns have one dominant singular value (the mean-intensity direction), which makes plain gradient steps
converge very slowly.

Checks:
- DGI reproduces Phase 2A exactly (max diff 0).
- Model C/D numbers reproduce their reports (difference 4e-15 dB).
- TV at 300 vs 900 iterations differs by at most 0.009 dB PSNR and 0.0012 SSIM (N = 500/1000/2000), so it has converged.
- No selected lambda is on a grid edge.

## 2. Numerical results (unseen seeds 2027-2029, mean ± std, median in brackets)

| Method | Shots | PSNR | SSIM | MSE | r |
|---|---:|---|---|---:|---|
| raw DGI | 500 | 12.33 ± 3.75 (11.25) | 0.116 ± 0.022 (0.115) | 0.0788 | 0.307 ± 0.011 |
| smoothed DGI | 500 | 21.70 ± 5.19 (21.85) | 0.497 ± 0.136 (0.492) | 0.0118 | 0.768 ± 0.033 |
| pseudoinverse | 500 | 19.78 ± 4.67 (19.43) | 0.330 ± 0.137 (0.290) | 0.0169 | 0.338 ± 0.019 |
| **TV** | 500 | **26.84 ± 5.76 (27.55)** | **0.702 ± 0.136 (0.719)** | **0.0043** | **0.900 ± 0.044** |
| Model D | 500 | 24.22 ± 4.84 (24.06) | 0.553 ± 0.123 (0.562) | 0.0064 | 0.822 ± 0.047 |
| raw DGI | 1000 | 14.48 ± 4.12 (13.65) | 0.186 ± 0.033 (0.183) | 0.0513 | 0.421 ± 0.012 |
| smoothed DGI | 1000 | 23.45 ± 5.09 (23.64) | 0.584 ± 0.097 (0.572) | 0.0079 | 0.828 ± 0.023 |
| pseudoinverse | 1000 | 20.48 ± 4.68 (20.15) | 0.394 ± 0.107 (0.353) | 0.0144 | 0.495 ± 0.013 |
| **TV** | 1000 | **30.25 ± 5.91 (31.38)** | **0.848 ± 0.074 (0.858)** | **0.0021** | **0.953 ± 0.026** |
| Model C | 1000 | 25.38 ± 4.91 (25.26) | 0.649 ± 0.094 (0.649) | 0.0049 | 0.868 ± 0.034 |
| raw DGI | 2000 | 17.00 ± 4.35 (16.32) | 0.291 ± 0.046 (0.291) | 0.0300 | 0.554 ± 0.013 |
| smoothed DGI | 2000 | 25.12 ± 4.99 (25.41) | 0.669 ± 0.070 (0.649) | 0.0053 | 0.876 ± 0.016 |
| pseudoinverse | 2000 | 22.18 ± 4.68 (21.95) | 0.521 ± 0.079 (0.500) | 0.0097 | 0.700 ± 0.008 |
| **TV** | 2000 | **35.84 ± 5.89 (35.68)** | **0.950 ± 0.027 (0.953)** | **0.00058** | **0.986 ± 0.010** |
| raw DGI | 5000 | 20.63 ± 4.50 (20.33) | 0.468 ± 0.064 (0.466) | 0.0134 | 0.735 ± 0.007 |
| smoothed DGI | 5000 | 27.30 ± 4.83 (27.65) | 0.762 ± 0.048 (0.750) | 0.0031 | 0.923 ± 0.011 |
| pseudoinverse | 5000 | exact: median 279 dB, max |error| <= 5e-12 | 1.000 | 3e-28 | 1.000 |
| TV (lambda = 0) | 5000 | exact (identical to pseudoinverse) | 1.000 | 3e-28 | 1.000 |
| raw DGI | 10000 | 23.52 ± 4.55 (23.18) | 0.611 ± 0.069 (0.609) | 0.0070 | 0.842 ± 0.003 |
| smoothed DGI | 10000 | 28.96 ± 4.77 (29.30) | 0.820 ± 0.038 (0.812) | 0.0021 | 0.948 ± 0.009 |
| pseudoinverse | 10000 | exact: median 281 dB, max |error| <= 3e-12 | 1.000 | 2e-28 | 1.000 |
| TV (lambda = 0) | 10000 | exact (identical to pseudoinverse) | 1.000 | 2e-28 | 1.000 |

For pseudoinverse at N >= 5000, the mean PSNR is reported as infinite: 208 of 600 pairs reconstruct
bit-exactly (MSE = 0). Every other pair is above 268 dB. Seed 2026 gives the same picture (summary.csv).

Paired comparisons (225 unseen image-seed pairs):

| A vs B | PSNR diff (A better) | SSIM diff (A better) | r diff (A better) |
|---|---|---|---|
| Model D-500 vs pseudoinverse-500 | +4.44 dB (100%) | +0.222 (100%) | +0.485 (100%) |
| Model D-500 vs TV-500 | **-2.63 dB (0%)** | **-0.150 (0%)** | **-0.078 (0%)** |
| Model D-500 vs smoothed DGI-500 | +2.52 dB (96%) | +0.055 (93%) | +0.054 (93%) |
| Model C-1000 vs pseudoinverse-1000 | +4.90 dB (100%) | +0.255 (100%) | +0.374 (100%) |
| Model C-1000 vs TV-1000 | **-4.87 dB (0%)** | **-0.200 (0%)** | **-0.085 (0%)** |
| Model C-1000 vs smoothed DGI-1000 | +1.93 dB (95%) | +0.065 (95%) | +0.041 (96%) |
| Model C-1000 vs TV-500 | (TV-500 alone already beats Model C: 26.84 vs 25.38 dB, SSIM 0.702 vs 0.649) | | |
| Model C-1000 vs smoothed DGI-2000 | +0.26 dB (56%) | -0.020 (32%) | -0.008 (37%) |

## 3. Does the pseudoinverse recover the image exactly at 10k?

Yes, as measured:

| N | Regime | Rank | sigma_min | Condition number | Max abs error | Relative data residual |
|---:|---|---:|---:|---:|---:|---:|
| 500 | underdetermined | 500 | 20.9 | 34 | 0.68 | 1.5e-15 |
| 1000 | underdetermined | 1000 | 16.2 | 62 | 0.72 | 1.5e-15 |
| 2000 | underdetermined | 2000 | 9.8 | 147 | 0.66 | 1.7e-15 |
| 5000 | overdetermined | 4096 | 3.4 | 665 | 3.4e-12 | 2.7e-15 |
| 10000 | overdetermined | 4096 | 18.1 | 177 | 1.9e-12 | 2.3e-15 |

(seed 2026; seeds 2027-2029 are the same to 3 significant figures)

Once N > 4096, P has full column rank and is well conditioned. Noise-free least squares therefore returns
the object to about 1e-12, i.e. exact recovery already at **5,000** shots, not just at 10,000. Conditioning
is worst just above 4096 (cond 665 at N = 5000), but that still does not matter without noise.

Below 4096 the minimum-norm pseudoinverse is a poor reconstruction (19.8-22.2 dB). It is worse than
smoothed DGI, because the minimum-norm solution keeps only the row-space component of the image.

## 4. Does TV improve reconstruction?

Strongly. TV beats every other method on every image:

| N | TV over smoothed DGI | TV over pseudoinverse |
|---:|---|---|
| 500 | +5.1 dB (100%) | +7.1 dB |
| 1000 | +6.8 dB (100%) | +9.8 dB |
| 2000 | +10.7 dB | +13.7 dB |

TV-1000 (30.3 dB, SSIM 0.85) is better than smoothed DGI-10,000 (29.0 dB, SSIM 0.82).
TV-2000 reaches SSIM 0.95.

Visually, TV-500 is piecewise-flat (cartoon-like) but locates nuclei and cell walls sharply. Fine
chromosome texture is not recovered at 500 shots.

## 5. How do Models C and D compare?

Both beat raw DGI, smoothed DGI and the pseudoinverse at the same N. Both lose to TV at the same N on
every one of the 225 pairs, on PSNR, SSIM and r. TV with only 500 shots beats Model C at 1000 shots.

The U-Nets only see the DGI image, which has already discarded information that the measurements contain.
TV works directly on the measurement constraints `P O = B`.

## 6. Is the current simulation adequate?

Not for claiming a measurement reduction:
- With noise-free buckets, any N > 4096 gives exact recovery, so "10,000 shots" is not a meaningful
  quality ceiling.
- Below 4096, noise-free compressed sensing (TV) is very strong, because the measurements are exact
  linear constraints. This flatters TV and makes the 64x64 problem much easier than a real ghost-imaging
  experiment.

## 7. Should realistic measurement noise be added?

Yes. Before any final experiment, add bucket-detector noise and/or photon shot noise at stated SNR levels
(for example several signal-to-noise ratios or photons per pattern). With noise:
- least squares is no longer exact (noise amplification grows as conditioning worsens near N = 4096);
- TV needs a noise-aware lambda;
- the trade-off between measurement count and quality becomes real.

## 8. Recommended next experiment

1. Add a realistic noise model to the buckets (validation and training only), with 2-3 noise levels.
2. Recompute the conventional curve (raw/smoothed DGI, pseudoinverse with Tikhonov regularisation, TV),
   with all hyperparameters tuned on training images per noise level and N.
3. Replace the DGI -> U-Net input with a physics-aware one: for example the TV or regularised
   least-squares reconstruction as U-Net input (a learned refinement of TV), or an unrolled /
   measurement-consistent network that uses P and B directly. Train with random pattern sequences, as in
   Models C/D.
4. Repeat the measurement-equivalence analysis against the **best** conventional method (TV), not DGI.
   Only then run the single test-set evaluation.

## Answers to the key questions

- **A. Is DGI a weak baseline?** Yes. DGI is weaker than TV at every N, by 5-11 dB even after smoothing.
  The minimum-norm pseudoinverse, however, is weaker than smoothed DGI below 4096 shots.
- **B. Pseudoinverse at 10,000 shots?** Exact (max error about 2e-12, PSNR about 280 dB or infinite).
  It is already exact at 5,000.
- **C. Pseudoinverse at 500/1000/2000?** 19.8 / 20.5 / 22.2 dB, SSIM 0.33 / 0.39 / 0.52. Weak.
- **D. Does Model D beat strong methods at 500?** It beats the pseudoinverse and smoothed DGI. It does
  not beat TV-500 (-2.6 dB, -0.15 SSIM, 0% of pairs).
- **E. Does Model C beat strong methods at 1000?** It beats the pseudoinverse and smoothed DGI. It does
  not beat TV-1000 (-4.9 dB, -0.20 SSIM, 0% of pairs), and it is below TV-500.
- **F. Is 2000-shot conventional reconstruction already strong enough?** Yes. TV-2000 reaches 35.8 dB,
  SSIM 0.95, r 0.986, far above both U-Nets. Even smoothed DGI-2000 roughly matches Model C.
- **G. Does the noise-free simulation support a measurement-reduction claim?** No. It makes N > 4096
  exact and inflates compressed sensing below 4096.
- **H. Should realistic noise be added before the final experiment?** Yes.

**Outcome 3**: conventional TV strongly beats the current DGI -> U-Net models in this noise-free setting.
