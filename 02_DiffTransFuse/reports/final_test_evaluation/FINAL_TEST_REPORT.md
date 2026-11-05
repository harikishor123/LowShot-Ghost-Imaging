# Final Held-Out Test Evaluation

**Project:** Computational Ghost Imaging via DiffTransFuse
**Team:** B23147 / B23139 / B23235
**Guide:** Amit Dattatraya Lad, CQST, IIT Mandi
**Date:** 2026-11-22
**Split:** Held-out test set (2000 images, never used for training or validation)

---

## 1. Methodology

### 1.1 Test Data

- **Source:** `objects_64.npz` (MedMNIST extension), split='test'
- **Images:** 2000 (blood + pathology microscopy, 64×64, grayscale)
- **Overlap check:** Zero overlap with train (7000) or val (1000) — verified
- **DGI generation:** On-the-fly using StreamingDGI with Bernoulli(0.5)
  patterns, noise-free buckets, calibrated DGI clipped to [0,1]
- **Seeds:** [3040, 3041, 3042] — 3 independent pattern sequences per image,
  distinct from training (30M+) and validation (2040–2042) seeds
- **Total samples per shot count:** 2000 images × 3 seeds = 6000

### 1.2 Models Evaluated

| Model | Description | Checkpoint source |
|-------|-------------|-------------------|
| DGI | Calibrated DGI reconstruction (no ML) | — |
| E1 | U-Net baseline (frozen project) | extension models/target_clean/ |
| E2a | U-Net + MSE+SSIM loss | ghost_imaging_v2/checkpoints/ |
| E3a | Residual diffusion U-Net (ε-prediction) | ghost_imaging_v2/checkpoints/ |
| E4a | TransFusionNet, MSE loss | ghost_imaging_v2/checkpoints/ |
| E4b | TransFusionNet, MSE+0.1×SSIM loss | ghost_imaging_v2/checkpoints/ |

All checkpoints are the `best.pt` files selected on the **validation set**.
No checkpoint selection, hyperparameter tuning, or model modification was
performed using test data. SHA256 hashes recorded in `checkpoint_hashes.txt`.

### 1.3 Shot Counts

N ∈ {500, 1000, 2000, 5000}

### 1.4 Metrics

PSNR, SSIM, Pearson correlation (r), edge correlation (edge_r, Sobel-map
Pearson), MSE. All computed on float64, data range [0, 1].

### 1.5 Statistical Analysis

Paired image-level bootstrap 95% CIs (2000 resamples, seed=0, clustered by
image with n_per_image=3). The CI is on the **difference** between methods
on matched test images, not on marginal CIs.

---

## 2. Results — PSNR Comparison (dB)

| N | DGI | E1 | E2a | E3a | **E4a** | **E4b** |
|------|-------|-------|-------|-------|---------|---------|
| 500 | 9.99 | 21.21 | 21.13 | 18.49 | **21.35** | 21.31 |
| 1000 | 11.66 | 22.80 | 22.71 | 20.05 | **22.85** | 22.79 |
| 2000 | 13.76 | 24.45 | 24.41 | 22.13 | **24.60** | 24.50 |
| 5000 | 17.03 | 26.65 | 26.56 | 24.45 | **26.78** | 26.59 |

**E4a achieves the highest PSNR at all four shot counts on the held-out test set.**

---

## 3. Results — SSIM Comparison

| N | DGI | E1 | E2a | E3a | E4a | **E4b** |
|------|-------|-------|-------|-------|-------|---------|
| 500 | 0.139 | 0.579 | 0.605 | 0.484 | 0.592 | **0.612** |
| 1000 | 0.204 | 0.669 | 0.686 | 0.571 | 0.678 | **0.692** |
| 2000 | 0.291 | 0.741 | 0.755 | 0.662 | 0.748 | **0.759** |
| 5000 | 0.428 | 0.820 | 0.826 | 0.759 | 0.823 | **0.830** |

**E4b achieves the highest SSIM at all four shot counts.**

---

## 4. Results — Edge Correlation (edge_r)

| N | E1 | E2a | E3a | E4a | **E4b** |
|------|-------|-------|-------|---------|---------|
| 500 | 0.565 | 0.573 | 0.429 | 0.581 | **0.584** |
| 1000 | 0.673 | 0.678 | 0.528 | 0.680 | **0.683** |
| 2000 | 0.763 | 0.765 | 0.650 | **0.771** | **0.770** |
| 5000 | 0.853 | 0.854 | 0.773 | **0.856** | **0.856** |

---

## 5. Results — Pearson Correlation (r)

| N | E1 | E2a | E3a | E4a | E4b |
|------|-------|-------|-------|-------|-------|
| 500 | 0.805 | 0.807 | 0.710 | 0.811 | **0.813** |
| 1000 | 0.861 | 0.863 | 0.770 | 0.864 | **0.865** |
| 2000 | 0.903 | 0.905 | 0.843 | 0.905 | **0.906** |
| 5000 | 0.942 | 0.943 | 0.900 | 0.943 | **0.944** |

---

## 6. Paired Bootstrap CIs — E4 vs E1

### 6.1 E4a − E1 (PSNR)

| N | Mean Δ (dB) | 95% CI | Significant? |
|------|-------------|-------------------------|--------------|
| 500 | +0.1423 | [+0.1296, +0.1553] | **YES** |
| 1000 | +0.0494 | [+0.0289, +0.0675] | **YES** |
| 2000 | +0.1462 | [+0.1350, +0.1584] | **YES** |
| 5000 | +0.1284 | [+0.1198, +0.1364] | **YES** |

**E4a significantly outperforms E1 for PSNR at all shot counts on the test set.**

### 6.2 E4b − E1 (PSNR)

| N | Mean Δ (dB) | 95% CI | Significant? |
|------|-------------|-------------------------|--------------|
| 500 | +0.0981 | [+0.0813, +0.1139] | **YES** |
| 1000 | −0.0089 | [−0.0234, +0.0072] | **NO** |
| 2000 | +0.0436 | [+0.0293, +0.0578] | **YES** |
| 5000 | −0.0603 | [−0.0789, −0.0416] | **YES** (E1 better) |

E4b trades PSNR for SSIM. At N=1000, E4b is statistically indistinguishable
from E1 for PSNR. At N=5000, E1 significantly outperforms E4b for PSNR
(the SSIM loss hurts PSNR at high shot counts).

### 6.3 E4a − E1 (SSIM)

| N | Mean Δ | 95% CI | Significant? |
|------|--------|-------------------------|--------------|
| 500 | +0.0132 | [+0.0126, +0.0139] | **YES** |
| 1000 | +0.0090 | [+0.0085, +0.0096] | **YES** |
| 2000 | +0.0068 | [+0.0064, +0.0072] | **YES** |
| 5000 | +0.0027 | [+0.0024, +0.0030] | **YES** |

### 6.4 E4b − E1 (SSIM)

| N | Mean Δ | 95% CI | Significant? |
|------|--------|-------------------------|--------------|
| 500 | +0.0333 | [+0.0322, +0.0344] | **YES** |
| 1000 | +0.0239 | [+0.0230, +0.0247] | **YES** |
| 2000 | +0.0181 | [+0.0175, +0.0188] | **YES** |
| 5000 | +0.0101 | [+0.0097, +0.0104] | **YES** |

**Both E4a and E4b significantly outperform E1 for SSIM at all shot counts.**

---

## 7. Inference Latency

Measured on CUDA, single image, 20 repetitions after 5 warmup passes (N=1000).

| Method | Mean (ms) | Std (ms) |
|--------|-----------|----------|
| E1 | 1.1 | 0.0 |
| E3a (DDIM 100 steps) | 275.3 | 0.5 |
| E4a (single pass) | 4.5 | 0.0 |
| E4b (single pass) | 4.5 | 0.0 |

- **E3a → E4 speedup: 61.7×**
- E4a/E4b are 4× slower than E1 (due to the frozen encoder + transformer),
  but 62× faster than E3a's iterative DDIM sampling.

---

## 8. PSNR vs Shot Count Analysis

E4a's improvement over E1 is consistent across shot counts (+0.05 to +0.15 dB),
with the largest gains at N=500 (+0.14 dB) and N=2000 (+0.15 dB), and the
smallest at N=1000 (+0.05 dB). The improvement is genuine but modest.

E3a (standalone diffusion) underperforms E1 by 2–3 dB at all shot counts,
consistent with validation results. E4 fully recovers this deficit and
exceeds E1.

The experiment chain on the test set:
```
DGI → E1 → E2a → E3a → E4a
 ↓      ↓     ↓      ↓      ↓
9.99  21.21 21.13  18.49  21.35   (N=500)
11.66 22.80 22.71  20.05  22.85   (N=1000)
13.76 24.45 24.41  22.13  24.60   (N=2000)
17.03 26.65 26.56  24.45  26.78   (N=5000)
```

---

## 9. SSIM vs Shot Count Analysis

E4b consistently achieves the highest SSIM, confirming that the SSIM
auxiliary loss transfers to held-out data. The gap narrows with increasing N:
+0.033 at N=500 → +0.010 at N=5000 (relative to E1).

E4a also improves SSIM over E1 (+0.003 to +0.013), but less than E4b.

---

## 10. Validation-to-Test Generalization

| Method | N | Val PSNR | Test PSNR | Δ |
|--------|------|----------|-----------|-------|
| E4a | 500 | 21.10 | 21.35 | +0.25 |
| E4a | 1000 | 22.63 | 22.85 | +0.22 |
| E4a | 2000 | 24.27 | 24.60 | +0.33 |
| E4a | 5000 | 26.45 | 26.78 | +0.33 |
| E1 | 500 | 20.93 | 21.21 | +0.28 |
| E1 | 1000 | 22.52 | 22.80 | +0.28 |
| E1 | 2000 | 24.12 | 24.45 | +0.33 |
| E1 | 5000 | 26.31 | 26.65 | +0.34 |

Test PSNR is consistently 0.2–0.35 dB higher than validation for all methods.
This is not a sign of test leakage — the difference is uniform across all
methods (including E1 from the frozen project) and likely reflects a slight
difference in image difficulty between the val and test partitions.

The E4a−E1 improvement is preserved on the test set (0.05–0.15 dB test vs
0.09–0.16 dB val), confirming generalization.

---

## 11. Limitations

1. **Input distribution mismatch** (see E4 audit): The frozen E3 encoder
   operates on out-of-distribution inputs (DGI+DGI at t=0 vs residual+DGI
   during E3 training). The decoder compensates, but the "pretrained feature"
   narrative is qualified.

2. **E4b PSNR trade-off:** E4b's SSIM loss hurts PSNR — at N=5000, E1
   significantly outperforms E4b for PSNR (−0.06 dB, CI excludes zero).
   E4b is recommended only when SSIM is the primary metric.

3. **Small improvements:** E4a's PSNR gain over E1 is 0.05–0.15 dB — real
   and statistically significant, but modest. This may not be perceptually
   meaningful for all applications.

4. **Single dataset:** Results are for MedMNIST microscopy images only.
   Generalization to other object types is not tested.

5. **No failed evaluations.** All 5 models × 4 shot counts × 3 seeds ran
   to completion with no errors.

---

## 12. Integrity Checks

- [x] No training or validation data entered the test evaluation
- [x] No gradients or training updates occurred (all inference under `@torch.no_grad()`)
- [x] No checkpoints were modified (SHA256 hashes recorded before evaluation)
- [x] All models used identical test objects and measurement seeds
- [x] Results are reproducible from `test_evaluation_config.yaml`
- [x] Test/train/val split overlap: 0 images in common (verified programmatically)
- [x] DGI seeds [3040, 3041, 3042] do not overlap with train (30M+) or val (2040–2042)
- [x] Three frozen-project MD5 manifests pass (verified in prior audit)

---

## 13. Output Files

| File | Description |
|------|-------------|
| `FINAL_TEST_REPORT.md` | This report |
| `test_metrics.csv` | 144,000 rows: per-image, per-seed metrics |
| `test_summary.csv` | 24 rows: aggregate metrics per method per N |
| `paired_bootstrap.csv` | 16 rows: paired CIs for E4a/E4b vs E1 |
| `latency_benchmark.csv` | Inference latency measurements |
| `checkpoint_hashes.txt` | SHA256 hashes of all 20 checkpoints |
| `test_evaluation_config.yaml` | Exact evaluation configuration |
| `figures/` | 20 reconstruction comparisons (5 per shot count) |

---

## 14. Conclusion

**E4a (TransFusionNet, MSE loss) generalizes to the held-out test set.**

- **PSNR:** E4a is the best method at all 4 shot counts, significantly
  outperforming E1 (paired bootstrap CIs exclude zero at every N).
- **SSIM:** E4b is the best method at all 4 shot counts, significantly
  outperforming E1.
- **Speed:** E4 is 61.7× faster than E3a (4.5 ms vs 275 ms per image).
- **Generalization:** The validation-to-test gap is uniform across methods,
  confirming no overfitting specific to E4.

The DiffTransFuse two-stage paradigm — freeze the diffusion encoder, attach
a Transformer fusion decoder — produces a model that is both more accurate
and dramatically faster than iterative diffusion sampling, and that
consistently (if modestly) outperforms the supervised baseline on
genuinely held-out data.
