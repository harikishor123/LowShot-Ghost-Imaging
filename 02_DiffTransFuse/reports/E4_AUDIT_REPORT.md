# E4 Scientific Audit Report

**Experiment:** E4 — TransFusionNet (DiffTransFuse Stage 2)
**Audit date:** 2026-11-22
**Scope:** 8-point audit of E4 implementation, results, and report accuracy

---

## Audit Point 1: E3 Encoder Input Pathway

**Question:** How does the frozen E3 encoder receive inputs during E4?

**Finding:** `src/transformer_fusion.py`, `_extract_frozen_features()` (lines 141–157):

```python
x_input = torch.cat([dgi, dgi], dim=1)   # [B, 2, 64, 64]
t_zero = torch.zeros(B, device=dgi.device, dtype=torch.long)
time_emb = self.encoder.time_mlp(t_zero)
e1 = self.encoder.enc1(x_input, time_emb)
...
```

The frozen encoder receives `concat(DGI, DGI)` with timestep `t=0`. This
was selected by feature-variance analysis (e1_var=0.86, highest of 6
candidates). The encoder runs under `torch.no_grad()` and `self.encoder.eval()`.

**Verdict:** ✓ Implementation matches reported design.

---

## Audit Point 2: Input Distribution Mismatch

**Question:** Is there a distribution mismatch between E3 training inputs
and E4 inference inputs?

**Finding:** **YES — significant mismatch.**

E3 was trained on `concat(r_t, DGI)` where `r_t` is the noisy residual at
diffusion timestep `t`. At `t=0`, `r_t = r₀ = O − DGI` (the clean
residual), which has:
- mean ≈ −0.013, std ≈ 0.380, range [−0.971, 0.839]

E4 feeds `concat(DGI, DGI)` where both channels are DGI:
- mean ≈ 0.491, std ≈ 0.398, range [0.000, 1.000]

Measured feature-level cosine similarity (DGI vs residual through frozen encoder):
- **e1 (32ch, 64×64):** cos_sim = 0.8118 — moderate agreement
- **mid (256ch, 8×8):** cos_sim = 0.3822 — substantial disagreement

**Interpretation:** The mismatch grows deeper in the network. At the
bottleneck level (where the Transformer operates), features are substantially
out-of-distribution. The E4 decoder compensates by learning to interpret
these mismatched features during its own training. The encoder therefore
acts more as a fixed nonlinear transform than a semantically meaningful
pretrained feature extractor.

**Impact on claims:** The "diffusion pretraining learns superior features"
narrative is only partially supported. An ablation with a randomly-initialized
frozen encoder (same architecture, untrained weights) would be needed to
isolate how much of E4's improvement comes from the specific E3 weights
versus the encoder architecture alone.

**Severity:** Medium — does not invalidate results (E4a still significantly
outperforms E1), but qualifies the theoretical interpretation.

---

## Audit Point 3: Attention Mechanism Verification

**Question:** Does TransFusionNet use cross-attention or self-attention?

**Finding:** **Self-attention only.** The original report's use of
"cross-attention" was a terminology error.

Source: `src/transformer_fusion.py`, `TransformerBlock.forward()` (lines 82–85):

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    normed = self.norm1(x)
    x = x + self.attn(normed, normed, normed)[0]  # Q=K=V
    x = x + self.ff(self.norm2(x))
    return x
```

`self.attn(normed, normed, normed)` passes the same tensor as Q, K, and V —
this is self-attention by definition.

**Fusion mechanism:** Cross-modal fusion (encoder skips + DGI + decoder
features) is performed by the CNN `FusionBlock` modules via concatenation
followed by convolution and squeeze-and-excitation channel attention. This
is a CNN-based fusion, not attention-based cross-modality fusion.

**Correction applied:** Report §11 and §12.2 updated.

---

## Audit Point 4: Paired Bootstrap Confidence Intervals

**Question:** Are E4's improvements over E1 statistically significant using
proper paired testing?

**Finding:** Paired image-level bootstrap (2000 resamples, clustered by
image with n_per_image=3 for the val set).

### E4a vs E1 (PSNR) — ALL SIGNIFICANT

| N | Δ PSNR (dB) | 95% Paired CI | Zero excluded? |
|------|-------------|-------------------------|----------------|
| 500 | +0.1622 | [+0.1448, +0.1801] | YES |
| 1000 | +0.1091 | [+0.0929, +0.1259] | YES |
| 2000 | +0.1545 | [+0.1376, +0.1723] | YES |
| 5000 | +0.1351 | [+0.1208, +0.1509] | YES |

### E4b vs E1 (PSNR) — MIXED

| N | Δ PSNR (dB) | 95% Paired CI | Zero excluded? |
|------|-------------|-------------------------|----------------|
| 500 | +0.0917 | [+0.0682, +0.1156] | YES |
| 1000 | −0.0184 | [−0.0396, +0.0032] | **NO** |
| 2000 | +0.0407 | [+0.0195, +0.0628] | YES |
| 5000 | −0.0215 | [−0.0440, +0.0016] | **NO** |

**Note:** The original report used separate (marginal) CIs and noted
"overlapping." Overlapping marginal CIs do not determine significance for
paired comparisons. The paired bootstrap correctly shows that E4a is
significant everywhere, while E4b is not significant at N=1000 and N=5000
for PSNR.

**Correction applied:** Report §6.1 and §6.1b replaced with paired CIs.

---

## Audit Point 5: Inference Latency Benchmark

**Question:** What is the actual single-image inference speedup?

**Finding:** Measured on CUDA with 20 repetitions after 5 warmup passes.

| Method | Single image | Batch=16 |
|--------|-------------|----------|
| E3 (DDIM 100 steps) | 301.9 ± 0.4 ms | 289.4 ms |
| E4 (single pass) | 4.8 ± 0.1 ms | 5.1 ms |
| **Speedup** | **62.3×** | **56.7×** |

The original report claimed "~30×." The actual speedup is **62.3×** for
single-image inference. The discrepancy likely arose from comparing
per-epoch validation times (which include data loading overhead) rather
than pure per-image forward-pass latency.

**Correction applied:** Report §7.3, §11, and §4.2 updated.

---

## Audit Point 6: Data Leakage Check

**Question:** Is there any data leakage — test set usage or gradient flow
through validation?

**Finding:** **No leakage detected.**

Evidence from source code:

1. **`train_e4.py`, `validate()` function:** Runs under `@torch.no_grad()`.
   Model selection is based on `val_loss` only. No test set references
   exist in the training script.

2. **`evaluate_e4.py`:** Uses `selection_val.npz` (1000 images × 3 seeds =
   3000 samples). No test set file is loaded.

3. **grep for test set references:**
   ```
   grep -rn "test" train_e4.py evaluate_e4.py
   ```
   Returns only references to `pytest`-style test files, not to any
   test data split.

**Verdict:** ✓ Clean — no data leakage.

---

## Audit Point 7: MD5 Manifest Verification

**Question:** Are all three frozen-project manifests intact?

**Finding:** All three pass.

```
$ cd /home/ss_students/Downloads/Onion-CNN-Final.v1-onion-original-v1.yolov8
$ md5sum --check --quiet FREEZE_MANIFEST_phase1_phase2.md5
(exit 0 — all OK)
$ md5sum --check --quiet outputs/extension_medmnist_microscopy/FREEZE_MANIFEST_extension.md5
(exit 0 — all OK)
$ md5sum --check --quiet outputs/extension_medmnist_microscopy_poisson/FREEZE_MANIFEST_poisson.md5
(exit 0 — all OK)
```

**Verdict:** ✓ Frozen project completely unmodified.

---

## Audit Point 8: Report Corrections Applied

The following corrections were applied to `reports/E4_TRANSFORMER_FUSION.md`:

| Issue | Original | Corrected |
|-------|----------|-----------|
| Attention type | "cross-attention" (implied) | Self-attention (Q=K=V); fusion via CNN FusionBlocks |
| Speedup | "~30×" | 62.3× (measured CUDA latency) |
| Statistical test | Separate marginal CIs | Paired image-level bootstrap CIs |
| E4b significance | Implicitly all better | NOT significant at N=1000, N=5000 for PSNR |
| Feature claim | "superior pretrained features" | Qualified: distribution mismatch, encoder as nonlinear transform |
| New §12 | (absent) | Limitations and Audit Corrections section added |

---

## Summary

| Point | Status | Severity |
|-------|--------|----------|
| 1. E3 input pathway | ✓ Matches design | — |
| 2. Distribution mismatch | ⚠ Real and significant | Medium |
| 3. Attention terminology | ✗ Error corrected | Low (report-only) |
| 4. Paired bootstrap CIs | ⚠ E4a significant; E4b mixed | Medium (E4b) |
| 5. Latency benchmark | ✗ Corrected 30× → 62.3× | Low (conservative error) |
| 6. Data leakage | ✓ Clean | — |
| 7. MD5 manifests | ✓ All pass | — |
| 8. Report updated | ✓ All corrections applied | — |

**Overall assessment:** E4's core result — that TransFusionNet (E4a, MSE
loss) significantly outperforms the E1 baseline at all shot counts — is
valid and confirmed by paired bootstrap CIs. The main theoretical caveat
is the input distribution mismatch, which means the frozen encoder is not
being used as a conventional pretrained feature extractor. The E4 decoder
compensates for this during training, so the practical result stands but
the DiffTransFuse narrative about "superior pretrained features" needs
qualification.
