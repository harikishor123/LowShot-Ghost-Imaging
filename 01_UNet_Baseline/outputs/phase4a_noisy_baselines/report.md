# Phase 4A: conventional reconstruction under noisy bucket measurements (validation only)

Diagnostic validation experiment, not a final result.

- No test images were loaded.
- No model was trained or retrained.
- All 185 earlier output files are byte-identical before and after this run.

Main metrics: 75 validation images x pattern/noise seeds 2027-2029 = 225 image-seed pairs per value.

## 1. Noise model

The forward model is unchanged: `B_clean,i = sum_xy P_i(x,y) O(x,y)`, with Bernoulli(0.5) patterns from the established generator.

Noise is added to each bucket value:

| Step | Formula |
|---|---|
| Photon scale | `s = N_ph / mu_ref`, where `mu_ref = 515.26` is the mean clean bucket over the 341 training images (seed-2026 patterns); one constant for all images |
| Shot noise | `K ~ Poisson(s * B_clean)` |
| Detector / readout noise | `e ~ Normal(0, kappa * sqrt(N_ph))` photo-electrons, independent per bucket |
| Reading | `B_noisy = (K + e) / s`, a signed reading. **Not clipped**: a detector with an offset can report negatives. Their number is counted; it was 0 in every condition. |

`N_ph` is the number of photons collected for a bucket of average brightness.

Seeding:
- Patterns come from `pattern_batches(seed)`, as before.
- Noise comes from `default_rng(SeedSequence([seed, 4040, regime_index]))`, independent of the patterns.
- One noise draw covers all 10,000 buckets, so the 500/1000/2000/5000-shot measurements are nested prefixes of the 10,000.
- Tuning uses seed 2026 (training images). Validation uses seeds 2027-2029 (new patterns and new noise).

## 2. Noise regimes (fixed before any reconstruction; `phase4a_noise_setup.py`)

Selection rule:
- kappa (detector noise relative to shot noise at the mean bucket) is fixed per regime: LOW 0.1, MEDIUM 0.5, HIGH 1.0.
- N_ph is the half-decade value whose median **training** fluctuation SNR is closest to the target: 10 / 3 / 1.

The first version of the rule used whole decades only. It put LOW and MEDIUM on the same budget (1e5), so it was refined to half-decades before any reconstruction was run. This revision is recorded in `noise_config.json`.

| Regime | N_ph (photons per mean bucket) | kappa | Detector sigma (e-) | Photon scale s |
|---|---:|---:|---:|---:|
| LOW | 3.16e5 | 0.1 | 56 | 613.7 |
| MEDIUM | 3.16e4 | 0.5 | 89 | 61.4 |
| HIGH | 3.16e3 | 1.0 | 56 | 6.1 |

**About the suggested 1e4 / 1e3 / 1e2 budgets.** The image information sits in the pattern-to-pattern fluctuation of the bucket signal, which is only 1.8% of its mean (median over training images; p10-p90 1.6-1.9%). At kappa = 0.5 those budgets give fluctuation SNRs of 1.7 / 0.53 / 0.17. 1e2 is essentially information-free per measurement, so the budgets were chosen by fluctuation SNR instead.

## 3. Bucket SNR (measured on noisy buckets)

| Regime | Clean mean | Clean std over patterns (median) | Noise std (median) | **Fluctuation SNR** (median) | Mean-based SNR (median) | Negative readings |
|---|---:|---:|---:|---:|---:|---:|
| LOW (train / val) | 515 / 534 | 9.9 / 10.4 | 0.94 / 0.97 | **10.4 / 10.6** | 574 / 590 | 0 |
| MEDIUM | 515 / 534 | 9.9 / 10.4 | 3.31 / 3.38 | **3.0 / 3.1** | 164 / 169 | 0 |
| HIGH | 515 / 534 | 9.9 / 10.4 | 13.1 / 13.3 | **0.75 / 0.78** | 41 / 43 | 0 |

Fluctuation SNR = std of the clean bucket over patterns / noise std. It is the quantity that limits reconstruction.

## 4. Conventional reconstruction results (mean over 225 pairs)

All parameters were chosen by lowest mean training MSE, then frozen (`selected_parameters.csv`; none at a grid edge):
- Gaussian sigma for smoothed DGI;
- Tikhonov lambda;
- TV lambda (ADMM with rho = 10*lambda, 200 iterations, Tikhonov initialisation; tuned on 40 training images).

**PSNR (dB)**

| Noise | Shots | DGI | Smoothed DGI | Pseudoinverse | Tikhonov | TV |
|---|---:|---:|---:|---:|---:|---:|
| LOW | 500 | 12.04 | 21.43 | 19.71 | 19.72 | **25.45** |
| LOW | 1000 | 14.14 | 23.20 | 20.30 | 20.32 | **27.57** |
| LOW | 2000 | 16.64 | 24.88 | 21.50 | 21.59 | **30.10** |
| LOW | 5000 | 20.25 | 27.06 | 24.80 | 27.02 | **33.99** |
| LOW | 10000 | 23.14 | 28.73 | 32.89 | 33.01 | **36.56** |
| MEDIUM | 500 | 10.15 | 19.26 | 19.04 | 19.32 | **22.84** |
| MEDIUM | 1000 | 12.08 | 21.43 | 18.81 | 19.48 | **23.95** |
| MEDIUM | 2000 | 14.34 | 23.36 | 17.84 | 19.74 | **25.73** |
| MEDIUM | 5000 | 17.73 | 25.49 | 14.58 | 20.98 | **27.15** |
| MEDIUM | 10000 | 20.53 | 27.19 | 21.98 | 23.46 | **29.44** |
| HIGH | 500 | 6.45 | 13.88 | 14.79 | 19.19 | **19.21** |
| HIGH | 1000 | 7.25 | 15.09 | 12.84 | 19.23 | **20.84** |
| HIGH | 2000 | 8.43 | 16.75 | 10.10 | 19.25 | **20.73** |
| HIGH | 5000 | 10.68 | 19.59 | 7.23 | 19.31 | **22.62** |
| HIGH | 10000 | 12.85 | 21.72 | 11.45 | 19.49 | **23.62** |

**SSIM**

| Noise | Shots | DGI | Smoothed DGI | Pseudoinverse | Tikhonov | TV |
|---|---:|---:|---:|---:|---:|---:|
| LOW | 500 / 1000 / 2000 / 5000 / 10000 | 0.11 / 0.18 / 0.28 / 0.45 / 0.59 | 0.49 / 0.57 / 0.66 / 0.75 / 0.81 | 0.32 / 0.37 / 0.47 / 0.65 / 0.89 | 0.32 / 0.38 / 0.48 / 0.74 / 0.89 | **0.65 / 0.77 / 0.86 / 0.93 / 0.96** |
| MEDIUM | same | 0.07 / 0.12 / 0.19 / 0.33 / 0.47 | 0.42 / 0.50 / 0.59 / 0.69 / 0.76 | 0.25 / 0.26 / 0.27 / 0.22 / 0.53 | 0.27 / 0.29 / 0.34 / 0.45 / 0.59 | **0.47 / 0.58 / 0.68 / 0.74 / 0.83** |
| HIGH | same | 0.02 / 0.03 / 0.05 / 0.10 / 0.16 | 0.26 / 0.29 / 0.34 / 0.44 / 0.52 | 0.08 / 0.07 / 0.05 / 0.04 / 0.12 | 0.27 / 0.27 / 0.26 / 0.27 / 0.30 | **0.31 / 0.36 / 0.35 / 0.49 / 0.52** |

**Pearson r**

| Noise | Shots | DGI | Smoothed DGI | Pseudoinverse | Tikhonov | TV |
|---|---:|---:|---:|---:|---:|---:|
| LOW | 500 / 1000 / 2000 / 5000 / 10000 | 0.29 / 0.41 / 0.54 / 0.72 / 0.83 | 0.76 / 0.82 / 0.87 / 0.92 / 0.95 | 0.32 / 0.47 / 0.65 / 0.85 / 0.97 | 0.32 / 0.47 / 0.65 / 0.90 / 0.97 | **0.87 / 0.92 / 0.95 / 0.98 / 0.99** |
| MEDIUM | same | 0.22 / 0.31 / 0.42 / 0.60 / 0.73 | 0.68 / 0.75 / 0.82 / 0.88 / 0.92 | 0.24 / 0.34 / 0.42 / 0.45 / 0.77 | 0.24 / 0.35 / 0.47 / 0.66 / 0.79 | **0.74 / 0.80 / 0.87 / 0.92 / 0.94** |
| HIGH | same | 0.09 / 0.12 / 0.17 / 0.27 / 0.36 | 0.42 / 0.51 / 0.59 / 0.69 / 0.75 | 0.09 / 0.12 / 0.14 / 0.12 / 0.31 | 0.08 / 0.13 / 0.19 / 0.29 / 0.39 | **0.44 / 0.53 / 0.53 / 0.73 / 0.79** |

Std, median and per-seed values are in `summary.csv`. The three seeds agree to within about 0.1 dB.

Paired comparisons (`paired_comparisons.csv`):
- TV beats Tikhonov on 100% of pairs in LOW and MEDIUM.
- TV beats smoothed DGI in MEDIUM by +1.7 to +3.6 dB (on 96-100% of pairs) and by +0.05 to +0.09 SSIM (on 79-97% of pairs).

Observations:
- **TV is the strongest conventional method in every regime.** Noise costs it a lot: TV-1000 falls from 30.3 dB (noise-free, Phase 3) to 27.6 (LOW), 23.9 (MEDIUM) and 20.8 (HIGH).
- **Smoothed DGI is close to TV in MEDIUM** (within about 2-3.5 dB). It is a reasonable second baseline under noise.
- **Plain pseudoinverse is unstable.** In MEDIUM and HIGH it gets worse as N approaches 4096 (14.6 dB at N = 5000 in MEDIUM), because noise is amplified along the poorly conditioned directions (condition number about 665 at N = 5000). This is the expected behaviour.
- **Tikhonov is weak below 4096 shots.** In HIGH it is flat at about 19.2-19.5 dB because heavy regularisation pushes it toward a near-uniform image (a flat mean image scores about 19.3 dB on these images).

## 5. Effect of shot count (best conventional method, TV)

| Regime | 500 -> 1000 -> 2000 -> 5000 -> 10000 (PSNR dB) | SSIM | Gain 10000 vs max(500, 1000) |
|---|---|---|---:|
| LOW | 25.5 -> 27.6 -> 30.1 -> 34.0 -> 36.6 | 0.65 -> 0.96 | +9.0 dB |
| MEDIUM | 22.8 -> 23.9 -> 25.7 -> 27.2 -> 29.4 | 0.47 -> 0.83 | +5.5 dB |
| HIGH | 19.2 -> 20.8 -> 20.7 -> 22.6 -> 23.6 | 0.31 -> 0.52 | +2.8 dB |

With noise, more measurements now genuinely help. Nothing becomes exact above 4096 shots: in LOW, Tikhonov at 10,000 shots is 33.0 dB, not about 280 dB.

## 6. Sanity checks

| Check | Result |
|---|---|
| A. More photons -> better | **Pass**: HIGH < MEDIUM < LOW for every method and every N |
| B. More shots -> better | **Pass** for raw DGI, smoothed DGI, Tikhonov and TV in LOW/MEDIUM. Exceptions: (i) pseudoinverse in MEDIUM and HIGH (noise amplification near N = 4096, as expected); (ii) TV in HIGH, 1000 -> 2000 (20.84 -> 20.73 dB, a 0.1 dB dip near the noise floor) |
| C. More detector noise -> worse | **Pass**: at the MEDIUM photon budget, kappa 0 -> 4 lowers PSNR monotonically (Tikhonov-10000: 24.4 -> 19.6 dB; smoothed DGI-10000: 27.6 -> 22.0 dB) |
| D. High budget approaches noise-free | **Pass**: at 1e10 photons, smoothed DGI-10000 = 28.99 dB (noise-free 28.99) and Tikhonov-1000 = 20.50 (noise-free 20.50). Tikhonov-10000 rises 33 -> 38 -> 48 -> 58 -> 78 dB from 3.2e5 to 1e10 photons, towards exact recovery |
| E. Low budget is hard | **Pass**: at 1e2 photons, smoothed DGI-10000 = 14.6 dB and Tikhonov collapses to about 19 dB (the near-uniform image), with SSIM 0.23-0.28 |
| Photon sweep monotone (finite budgets) | Pass for smoothed DGI. Tikhonov shows differences of 0.01-0.1 dB between 1e2 and 3.2e3 photons, at the near-uniform-image floor; these are not real reversals |
| TV convergence (200 vs 600 iterations) | <= 0.001 dB change in all regimes at N = 1000 and 10000 |
| DGI-from-buckets matches Phase 2A on clean data | max diff 6e-8 |

## 7. Recommended primary noise regime: **MEDIUM**

Based only on the criteria fixed before validation and on physical reasonableness (no AI model was evaluated):

| Regime | 10k gain >= 3 dB | Not trivial (best 10k < 45 dB) | Not impossible (best 10k SSIM >= 0.5) | Useful |
|---|---|---|---|---|
| LOW | +9.0 | 36.6 | 0.96 | yes |
| MEDIUM | +5.5 | 29.4 | 0.83 | yes |
| HIGH | **+2.8** | 23.6 | 0.52 | **no** |

HIGH is excluded: each measurement carries little information (fluctuation SNR 0.75), and the best 10,000-shot reconstruction is only about 4 dB above a flat image.

MEDIUM is recommended as the primary regime:
- photon budget **3.16e4 photons per mean bucket**;
- detector noise **kappa = 0.5, about 89 e- rms**;
- effective **fluctuation SNR about 3.0** (mean-based SNR about 165).

TV / smoothed DGI performance in MEDIUM:

| Shots | TV (PSNR / SSIM / r) | Smoothed DGI (PSNR / SSIM) |
|---:|---|---|
| 500 | 22.8 / 0.47 / 0.74 | 19.3 / 0.42 |
| 1000 | 23.9 / 0.58 / 0.80 | 21.4 / 0.50 |
| 2000 | 25.7 / 0.68 / 0.87 | 23.4 / 0.59 |
| 5000 | 27.2 / 0.74 / 0.92 | 25.5 / 0.69 |
| 10000 | 29.4 / 0.83 / 0.94 | 27.2 / 0.76 |

Why MEDIUM:
- Noise clearly limits every method.
- 10,000 shots are much better than 500 or 1,000 (+5.5 dB), so measurement count really matters.
- Nothing is trivial or exact.
- The images at 10,000 shots are clearly usable (SSIM 0.83).
- A few times 1e4 photons per bucket is a plausible low-light single-pixel regime (low illumination dose, short integration).

LOW (fluctuation SNR about 10) is recommended as a secondary robustness regime: it is closer to the noise-free behaviour.

## 8. Limitations

- Noise is i.i.d. per bucket: no source-intensity drift, pattern (DMD) errors, background light, quantisation or detector non-linearity.
- Gaussian read noise is modelled in photo-electrons with kappa tied to sqrt(N_ph) per regime. A real detector has a fixed read noise; the kappa sweep partly covers this.
- One photon scale for all images, so darker images get fewer photons (physical, but image-dependent SNR).
- Hyperparameters are chosen by training MSE; SSIM-optimal choices may differ. TV lambda was tuned on 40 of the 341 training images.
- 64x64, binary patterns, one specimen type. Metrics are on the whole 64x64 image.
- Validation images have been reused across many experiments; the test set is still untouched.
