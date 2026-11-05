# E4: Transformer Fusion — Design, Validation, and Results

**Experiment:** E4 — TransFusionNet (DiffTransFuse Stage 2)
**Team:** B23147 / B23139 / B23235
**Guide:** Amit Dattatraya Lad, CQST, IIT Mandi
**Date:** 2026-11-22

---

## 1. Motivation

E3 established that a residual diffusion U-Net learns rich multi-scale features
for ghost-image reconstruction, but its iterative DDIM sampling (100 steps) is
slow and its reconstruction quality fell 2–3 dB below the E1 baseline. The
DiffTransFuse two-stage paradigm addresses both issues:

- **Stage 1 (E3):** Train a diffusion U-Net to learn denoising features.
- **Stage 2 (E4):** Freeze the E3 encoder; attach a Transformer fusion decoder
  that reconstructs the object in a **single forward pass** — no iterative sampling.

E4 asks: can the frozen diffusion encoder features, when fused by a Transformer
decoder with channel attention, surpass the E1 baseline?

---

## 2. Architecture

### 2.1 Frozen E3 Encoder

The DiffusionUNet encoder from E3a (ε-prediction, L1 loss) is loaded per shot
count and completely frozen. At inference, we feed `concat(DGI, DGI)` with
timestep `t=0`:

| Level | Shape | Channels |
|-------|-------|----------|
| e1 | [B, 32, 64, 64] | 32 |
| e2 | [B, 64, 32, 32] | 64 |
| e3 | [B, 128, 16, 16] | 128 |
| mid | [B, 256, 8, 8] | 256 |

**Input strategy selection:** Six candidates were evaluated by feature variance:

| Strategy | e1 var | mid var |
|----------|--------|---------|
| DGI+DGI, t=0 | **0.8591** | 0.3017 |
| zeros+DGI, t=0 | 0.5103 | 0.2734 |
| noise+DGI, t=500 | 0.7545 | 0.2601 |

`DGI+DGI at t=0` was selected for highest feature activation variance.

### 2.2 TransFusionNet (Trainable Decoder)

```
Frozen E3 Encoder (2,999,393 params — frozen)
         ↓
Bottleneck Transformer (8×8 = 64 tokens, dim=256, 4 layers, 4 heads)
         ↓
CNN Decoder with Channel-Attention Fusion:
  Level 3: up(256→128) ⊕ e3(128) ⊕ DGI_16(1) → FusionBlock → 128
  Level 2: up(128→64) ⊕ e2(64) ⊕ DGI_32(1) → FusionBlock → 64
  Level 1: up(64→32) ⊕ e1(32) ⊕ DGI(1) → FusionBlock → 32
  Output:  Conv(32→1) — linear head
```

Each **FusionBlock** contains:
- Concatenation of upsampled features + encoder skip + DGI at that scale
- Two Conv2d-GroupNorm-SiLU blocks
- Squeeze-and-excitation channel attention (reduction=4)

Each **TransformerBlock** contains:
- Pre-norm LayerNorm → Multi-head self-attention (4 heads)
- Pre-norm LayerNorm → FFN (dim × 4 expansion, GELU)
- Residual connections around both

**Parameter counts:**

| Component | Parameters |
|-----------|-----------|
| Frozen E3 encoder | 2,999,393 |
| Transformer (4L, 4H, dim=256) | ~3,175,000 |
| CNN decoder + channel attention | ~1,285,000 |
| **Total trainable** | **4,460,097** |
| **Total (all)** | **7,459,490** |

### 2.3 Key Design Decisions

1. **Single forward pass:** No DDIM sampling — O_pred = TransFusionNet(DGI)
2. **Per-N checkpoints:** Each shot count uses its own matched E3a encoder
3. **DGI conditioning at every scale:** DGI is downsampled and concatenated
   at each decoder level, providing spatial detail throughout
4. **train() override:** `model.train()` keeps the frozen encoder in `eval()`
   mode to prevent BatchNorm/Dropout drift

---

## 3. Training Configuration

### 3.1 Ablation Configs

| Config | Loss | Description |
|--------|------|-------------|
| E4a | MSE | MSE(O_pred, O_target) |
| E4b | MSE + 0.1×SSIM | MSE + 0.1 × (1 − SSIM) |

### 3.2 Shared Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Batch size | 16 |
| Max epochs | 100 |
| Early stopping patience | 15 |
| Gradient clipping | max_norm=1.0 |
| Seed | 0 |

### 3.3 Data

- **Train:** 7,000 samples from pool_train.npy (random_k=True)
- **Val:** 3,000 samples from selection_val.npz (1,000 images × 3 seeds)
- Shot counts: N ∈ {500, 1000, 2000, 5000}

---

## 4. Training Results

### 4.1 Convergence Summary

| Config | N | Best Epoch | Total Epochs | Time (s) | Val PSNR | Val SSIM |
|--------|------|-----------|-------------|----------|----------|----------|
| E4a | 500 | 76 | 91 | 1145 | 21.10 | 0.5731 |
| E4a | 1000 | 44 | 59 | 751 | 22.63 | 0.6620 |
| E4a | 2000 | 46 | 61 | 754 | 24.27 | 0.7354 |
| E4a | 5000 | 34 | 49 | 662 | 26.45 | 0.8156 |
| E4b | 500 | 71 | 86 | 1167 | 21.03 | 0.5954 |
| E4b | 1000 | 44 | 59 | 799 | 22.51 | 0.6780 |
| E4b | 2000 | 32 | 47 | 659 | 24.16 | 0.7481 |
| E4b | 5000 | 35 | 50 | 685 | 26.29 | 0.8240 |

**8/8 runs completed, 0 failures.**

### 4.2 Timing Comparison

E4 is dramatically faster than E3 at inference:

| Method | Val inference | Mechanism |
|--------|--------------|-----------|
| E3 (DDIM 100 steps) | ~90s/epoch | 100 sequential forward passes |
| **E4 (single pass)** | **~3s/epoch** | 1 forward pass |

---

## 5. Evaluation Results (Bootstrap CIs)

### 5.1 PSNR Comparison (dB) — All Methods

| N | DGI | E1 (frozen) | E2a | E3a (diffusion) | E4a (MSE) | E4b (MSE+SSIM) |
|------|-------|-------------|-------|------------------|-----------|----------------|
| 500 | 9.71 | 20.93 | 20.82 | 18.14 | **21.10** | 21.03 |
| 1000 | 11.39 | 22.52 | 22.41 | 19.88 | **22.63** | 22.51 |
| 2000 | 13.50 | 24.12 | 24.03 | 21.77 | **24.27** | 24.16 |
| 5000 | 16.77 | 26.31 | 26.20 | 24.13 | **26.45** | 26.29 |

### 5.2 SSIM Comparison

| N | DGI | E1 | E2a | E3a | E4a | E4b |
|------|-------|-------|-------|-------|-------|-------|
| 500 | 0.132 | 0.560 | 0.588 | 0.458 | 0.573 | **0.595** |
| 1000 | 0.197 | 0.653 | 0.672 | 0.551 | 0.662 | **0.678** |
| 2000 | 0.286 | 0.728 | 0.742 | 0.643 | 0.735 | **0.748** |
| 5000 | 0.429 | 0.813 | 0.820 | 0.747 | 0.816 | **0.824** |

### 5.3 Edge Preservation (edge_r)

| N | E1 | E2a | E3a | E4a | E4b |
|------|-------|-------|-------|-------|-------|
| 500 | 0.555 | 0.559 | 0.412 | **0.573** | **0.573** |
| 1000 | 0.666 | 0.665 | 0.517 | **0.673** | **0.673** |
| 2000 | 0.756 | 0.754 | 0.642 | **0.764** | 0.762 |
| 5000 | 0.848 | 0.847 | 0.768 | **0.852** | 0.850 |

### 5.4 95% Bootstrap Confidence Intervals (PSNR)

| Method | N=500 | N=1000 | N=2000 | N=5000 |
|--------|-------|--------|--------|--------|
| E1 | [20.66, 21.21] | [22.26, 22.81] | [23.85, 24.40] | [26.04, 26.59] |
| E3a | [17.88, 18.41] | [19.64, 20.11] | [21.51, 22.05] | [23.89, 24.38] |
| **E4a** | **[20.83, 21.38]** | **[22.37, 22.91]** | **[24.00, 24.56]** | **[26.18, 26.73]** |
| **E4b** | **[20.75, 21.32]** | **[22.23, 22.79]** | **[23.89, 24.44]** | **[26.03, 26.57]** |

---

## 6. Statistical Analysis

### 6.1 Paired Bootstrap CIs: E4a vs E1 (PSNR)

Paired image-level bootstrap (2000 resamples, clustered by image with 3
seeds per image). The CI is on the **difference** E4a − E1, not on each
method separately — overlapping marginal CIs do not determine significance;
only paired CIs that exclude zero do.

| N | E4a − E1 (dB) | 95% Paired CI | Significant? |
|------|---------------|---------------------------|--------------|
| 500 | +0.1622 | [+0.1448, +0.1801] | **YES** |
| 1000 | +0.1091 | [+0.0929, +0.1259] | **YES** |
| 2000 | +0.1545 | [+0.1376, +0.1723] | **YES** |
| 5000 | +0.1351 | [+0.1208, +0.1509] | **YES** |

E4a significantly outperforms E1 at all four shot counts. The paired CIs
exclude zero everywhere, confirming a genuine 0.09–0.18 dB improvement.

### 6.1b Paired Bootstrap CIs: E4b vs E1 (PSNR)

| N | E4b − E1 (dB) | 95% Paired CI | Significant? |
|------|---------------|---------------------------|--------------|
| 500 | +0.0917 | [+0.0682, +0.1156] | **YES** |
| 1000 | −0.0184 | [−0.0396, +0.0032] | **NO** |
| 2000 | +0.0407 | [+0.0195, +0.0628] | **YES** |
| 5000 | −0.0215 | [−0.0440, +0.0016] | **NO** |

E4b's SSIM auxiliary loss trades PSNR for structural similarity. At N=1000
and N=5000, E4b is **not** statistically better than E1 for PSNR — the
paired CI includes zero. E4b's advantage is in SSIM, not PSNR.

### 6.2 E4 vs E3a (PSNR)

| N | E4a − E3a | Overlap? |
|------|-----------|----------|
| 500 | +2.96 dB | **No** — statistically significant |
| 1000 | +2.75 dB | **No** — statistically significant |
| 2000 | +2.50 dB | **No** — statistically significant |
| 5000 | +2.32 dB | **No** — statistically significant |

E4 completely recovers the 2–3 dB deficit that E3 had relative to E1,
and exceeds E1 at every shot count.

### 6.3 E4a vs E4b (Loss Ablation)

| N | E4a PSNR | E4b PSNR | E4a SSIM | E4b SSIM |
|------|----------|----------|----------|----------|
| 500 | **21.10** | 21.03 | 0.573 | **0.595** |
| 1000 | **22.63** | 22.51 | 0.662 | **0.678** |
| 2000 | **24.27** | 24.16 | 0.735 | **0.748** |
| 5000 | **26.45** | 26.29 | 0.816 | **0.824** |

The SSIM auxiliary loss (E4b) trades ~0.1–0.15 dB PSNR for improved structural
similarity (+0.01–0.02 SSIM). This follows the same pattern observed in E2.

---

## 7. Key Findings

### 7.1 DiffTransFuse Two-Stage Paradigm Validated

The frozen diffusion encoder provides multi-scale features that the
Transformer fusion decoder exploits for single-pass reconstruction. E4a
significantly outperforms E1 at all shot counts (paired bootstrap CIs
exclude zero). However, see §12 (Limitations) for an important caveat
about input distribution mismatch: the encoder operates on out-of-distribution
inputs, so the "pretrained feature" narrative requires qualification.

### 7.2 E4 Surpasses All Prior Experiments

| Metric | Best Method | Runner-up |
|--------|------------|-----------|
| PSNR | **E4a** (MSE) | E1 (frozen baseline) |
| SSIM | **E4b** (MSE+SSIM) | E2a (MSE+SSIM) |
| edge_r | **E4a** (MSE) | E1 (frozen baseline) |
| Pearson_r | **E4a/E4b** | E1/E2a |

### 7.3 Inference Speed

E4 replaces E3's 100-step DDIM sampling with a single forward pass.
Measured on CUDA (single image, 20 repetitions after warmup):

| Method | Latency (ms) | Mechanism |
|--------|-------------|-----------|
| E3 (DDIM 100 steps) | 301.9 ± 0.4 | 100 sequential forward passes |
| **E4 (single pass)** | **4.8 ± 0.1** | 1 forward pass |
| **Speedup** | **62.3×** | |

Batch=16: E3=289.4 ms, E4=5.1 ms (56.7× speedup).

### 7.4 Experiment Chain Summary (PSNR at N=1000)

```
DGI (11.39) → E1 (22.52) → E2a (22.41) → E3a (19.88) → E4a (22.63)
                                                              ↑
                                              Best overall: +0.11 dB over E1
```

The full chain E1→E2→E3→E4 demonstrates that:
1. E1 baseline sets a strong floor
2. E2 loss ablations provided marginal changes
3. E3 diffusion underperformed as a standalone reconstructor
4. **E4 leverages E3's learned features to exceed E1**

---

## 8. Ablation: E4a vs E4b

| Aspect | E4a (MSE only) | E4b (MSE + SSIM) |
|--------|---------------|-------------------|
| PSNR | **Higher** (+0.1–0.15 dB) | Lower |
| SSIM | Lower | **Higher** (+0.01–0.02) |
| edge_r | Comparable | Comparable |
| Convergence | Similar epochs | Slightly faster |

**Recommendation:** E4a for PSNR-critical applications; E4b when structural
fidelity matters more than peak SNR.

---

## 9. Files Created

| File | Description |
|------|-------------|
| `src/transformer_fusion.py` | TransFusionNet architecture |
| `configs/e4a_mse.yaml` | E4a config (MSE loss) |
| `configs/e4b_mse_ssim.yaml` | E4b config (MSE + SSIM loss) |
| `train_e4.py` | Training script |
| `evaluate_e4.py` | Evaluation script (5-method comparison) |
| `tests/test_transformer_fusion.py` | 20 unit tests (all passing) |
| `checkpoints/e4{a,b}_*_N{500,1000,2000,5000}/` | 8 trained checkpoints |
| `results/e4{a,b}_*_eval_N{500,1000,2000,5000}/` | 8 evaluation results |

---

## 10. Reproducibility

- All 57 tests pass (20 diffusion + 17 U-Net + 20 TransFusionNet)
- Seed=0 for all runs
- Frozen project manifests verified before every training run
- E1/E2 source files, checkpoints, and results unmodified
- E3 source files and checkpoints unmodified (used as frozen encoder)
- Test set not used for model selection (val set only)

---

## 11. Conclusion

E4 (TransFusionNet) successfully demonstrates the DiffTransFuse two-stage
paradigm for computational ghost imaging:

1. **Quality:** E4a achieves the best PSNR at all shot counts, surpassing
   the E1 frozen baseline by 0.1–0.2 dB — a consistent improvement confirmed
   across N=500 to N=5000.

2. **Speed:** Single-pass inference eliminates the 100-step DDIM bottleneck,
   achieving 62.3× faster reconstruction than E3.

3. **Validation:** The 2–3 dB E3 deficit relative to E1 is entirely recovered
   by the Transformer fusion decoder. However, the frozen encoder operates
   on out-of-distribution inputs (see §12), so the decoder is partly
   learning to interpret mismatched features rather than purely exploiting
   pretrained representations.

4. **Architecture:** The combination of bottleneck self-attention Transformer
   (64 tokens at 8×8) with CNN channel-attention skip fusion provides an
   effective and parameter-efficient fusion strategy (4.5M trainable params).
   Note: the Transformer uses self-attention (Q=K=V from the same token
   sequence), not cross-attention between modalities. Cross-modal fusion
   is handled by the CNN FusionBlocks.

5. **Statistical caveat:** E4a is significantly better than E1 at all shot
   counts (paired bootstrap). E4b (MSE+SSIM) is NOT significantly better
   than E1 for PSNR at N=1000 and N=5000.

---

## 12. Limitations and Audit Corrections

This section documents findings from a post-hoc scientific audit of E4.

### 12.1 Input Distribution Mismatch (Critical)

The E3 encoder was trained on `concat(r_t, DGI)` where `r_t` is the noisy
residual (mean ≈ 0, range [−1, 1]). E4 feeds `concat(DGI, DGI)` where DGI
has mean ≈ 0.49, range [0, 1]. The first channel's distribution is
fundamentally different:

| Statistic | E3 training (r₀ at t=0) | E4 inference (DGI) |
|-----------|------------------------|---------------------|
| Mean | −0.013 | +0.491 |
| Std | 0.380 | 0.398 |
| Range | [−0.971, 0.839] | [0.000, 1.000] |

Feature-level cosine similarity between E3-training and E4-inference inputs:
- **e1 (shallow):** 0.81 — moderate agreement
- **mid (bottleneck):** 0.38 — substantial disagreement

**Consequence:** The frozen encoder operates substantially out-of-distribution
at the bottleneck level. The E4 decoder learns to interpret these mismatched
features during its own training, rather than exploiting semantically meaningful
pretrained representations. The encoder functions more as a fixed nonlinear
transform than a traditional pretrained feature extractor.

**Implication for the DiffTransFuse narrative:** The claim that "diffusion
pretraining learns superior features" is only partially supported. An ablation
comparing E4 with a randomly-initialized (untrained) frozen encoder would
be needed to quantify how much of E4's performance derives from the specific
E3 weights versus the architecture acting as a generic nonlinear projection.

### 12.2 Attention Mechanism Terminology

The original report described the Transformer as using "cross-attention."
This is incorrect. The TransformerBlock uses **self-attention** (Q=K=V from
the same 64-token sequence):

```python
# src/transformer_fusion.py, TransformerBlock.forward()
normed = self.norm1(x)
x = x + self.attn(normed, normed, normed)[0]  # Q=K=V — self-attention
```

Cross-modal fusion (encoder skips + DGI + upsampled features) is performed
by the **CNN FusionBlocks** via concatenation and channel attention, not
by the Transformer. This distinction matters: the Transformer refines
spatial relationships within the bottleneck tokens, while the FusionBlocks
handle multi-source integration at each scale.

### 12.3 E4b Statistical Significance

E4b (MSE + 0.1×SSIM) does not significantly outperform E1 for PSNR at
N=1000 (CI: [−0.040, +0.003]) and N=5000 (CI: [−0.044, +0.002]). The
paired bootstrap CIs include zero at these shot counts. E4b's advantage
is in SSIM, not PSNR.

### 12.4 Speedup Correction

The original report stated "~30× faster." Measured CUDA latency benchmarking
(single image, 20 repetitions with warmup) gives **62.3×** (E3: 301.9 ms vs
E4: 4.8 ms). The original estimate likely compared per-epoch validation times
rather than per-image latency.

### 12.5 Data Integrity Verification

Verified during audit:
- `validate()` runs under `@torch.no_grad()` — no gradient leakage
- Model selection uses `val_loss` only — test set never referenced
- All three frozen-project MD5 manifests pass:
  - `FREEZE_MANIFEST_phase1_phase2.md5` ✓
  - `outputs/extension_medmnist_microscopy/FREEZE_MANIFEST_extension.md5` ✓
  - `outputs/extension_medmnist_microscopy_poisson/FREEZE_MANIFEST_poisson.md5` ✓
- E1, E2, E3 source files, checkpoints, and results unmodified
