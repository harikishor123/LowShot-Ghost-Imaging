# Scientific Audit — Ghost Imaging U-Net MTP (onion study, MedMNIST extension, Poisson-noise extension)

**Project:** AI-DRIVEN Reconstruction of Quantum-Enhanced Low-Photon Bio-Imaging Datasets
**Audit date:** 2026-11-05 · **Mode:** read-only. No existing file was modified, overwritten, retrained or regenerated. All recomputation was done in memory or in a scratch directory; this folder (`audit/`) is new.

## 0. Summary

| Area | Verdict |
|---|---|
| MD5 manifests (onion 134, extension 131, Poisson 180 files) | **All pass**; no unlisted files inside frozen folders |
| Onion hashes recorded in the thesis appendix (checkpoints, protocols, frozen configs, scripts, test split) | **All match** |
| Reported metrics vs per-image CSVs (all three experiments) | **Exact agreement** (max difference ≈ 1e-14, floating-point level) |
| Bootstrap CIs, pass/fail decisions, minimum shot counts (extension and Poisson, val + test) | **Independently reproduced exactly**; 0 decision mismatches |
| Onion thesis tables (4.1–4.5, 4.7, 4.8) and abstract vs CSVs | **All agree** at the printed precision (one cosmetic sign-of-zero) |
| Thesis draft (`thesis/draft_extension_sections.md`) vs frozen results | **Byte-identical** when regenerated from the frozen CSVs |
| Validation vs test | Same minimum shot count in 25 of 27 extension/Poisson cases and both onion phases; **2 borderline flips** at 10% noise |
| Protocol order (protocol → pool → training → freeze → single test) | **Correct** for both extensions (timestamps in §2.6) |
| Test leakage | **None found** in training/selection; one minor pre-training smoke test touched 24 test images with untrained networks (§3.1) |

No numerical error was found in any reported result. The findings that need action concern **disclosure and wording** (§3, §6), not the numbers.

---

## 1. Inventory of frozen material

| Experiment | Output folder | Protocol | Freeze record | Checkpoints | Manifest |
|---|---|---|---|---|---|
| Onion Phase 1 (target: 10k DGI) | `outputs/phase_final_low_shot_to_10k_reference/` | `protocol.json` | `frozen_for_test.json`, `test/run_record.json` | `models/model_{500,1000,2000,5000}/checkpoints/best.pt` | `FREEZE_MANIFEST_phase1_phase2.md5` (72 files) |
| Onion Phase 2 (target: clean object) | `outputs/phase_final_original_target/` | `reports/protocol.json` | `reports/frozen_for_test.json` | `models/model_*/checkpoints/best.pt` | same manifest (62 files) |
| Extension (BloodMNIST + PathMNIST) | `outputs/extension_medmnist_microscopy/` + `data/extension_medmnist/` | `protocol.json` | `results/target_{clean,dgi10k}/frozen_for_test.json` | `models/target_{clean,dgi10k}/model_*/checkpoints/best.pt` (8) | `FREEZE_MANIFEST_extension.md5` (131 files, incl. 6 `ext_*.py`) |
| Poisson-noise extension (5/10/30%) | `outputs/extension_medmnist_microscopy_poisson/` | `protocol.json` | `results/level_{p05,p10,p30}/frozen_for_test.json` | `models/level_*/model_*/checkpoints/best.pt` (12) | `FREEZE_MANIFEST_poisson.md5` (180 files, incl. 5 `ext_poisson_*.py`) |

Per-image result files: onion Phase 1 4,144 rows; Phase 2 2,665 rows; extension and Poisson 36,000–112,000 rows per folder (10 folders, about 770,000 rows in total). Earlier exploratory folders (`phase1_single_image`, `phase1b_dgi`, `phase2a_dgi_dataset`, `phase2b_unet`, `phase3_strong_baselines`, `phase4a_noisy_baselines`) are **not** covered by any manifest (see §4).

## 2. Verified findings

### 2.1 File integrity
- `md5sum -c` on all three manifests: **134/134, 131/131, 180/180 OK** (run 2026-11-05).
- A directory walk found **no file** inside the frozen folders that is missing from its manifest.
- Onion values in thesis Appendix A.4/A.6 all match the files on disk: checkpoint MD5s (AI-500 … AI-5000 and AI-original-500 … 5000), protocol MD5s (`fa82061d…`, `c9d4dd91…`), frozen-config MD5s (`0f5199e8…`, `5bc703fe…`), script MD5s (`final_evaluate.py` `25717d6c…`, `orig_train.py` `f71dddad…`, `orig_evaluate.py` `a7655732…`) and the test-split file (`41235d66…`).
- Extension and Poisson: every checkpoint MD5 equals its `frozen_for_test.json` entry, the test evaluations loaded exactly those checkpoints, and the smoothing σ used at test equals the frozen σ (5 of 5 freeze records).

### 2.2 Metrics (PSNR, SSIM, Pearson *r*, MSE, edge *r*)
- **Onion Phase 1 and Phase 2:** `summary.csv` / `aggregate_metrics.csv` / `dgi_baseline_results.csv` recomputed from `per_image_results.csv`: max difference **3.6e-15**. Phase 2 paired SSIM differences (AI-original-*N* − DGI-10000) recomputed exactly.
- **Extension and Poisson (10 folders, val + test):** every `aggregate_metrics.csv` entry (mean, std, median; three seed groups; all, blood, tissue) recomputed: max difference **1.1e-14**. Structure: 4,000 (val) or 8,000 (test) pairs for every label, no duplicate rows, no non-finite values.
- Per-image metric values were **not** recomputed from images. Doing so would mean re-running the frozen models on the test set, which the single-test rule forbids. A noise-free cross-check is available instead: the noise-free DGI-10000 evaluated inside the Poisson experiment agrees with the extension's DGI-10000 for all 8,000 test pairs at every noise level to ≤ 3.8e-9 (PSNR) and ≤ 1.1e-10 (SSIM, *r*). The tiny difference comes from float rounding of a different pattern batch size; the pools themselves agreed exactly (0.0).

### 2.3 Criterion, confidence intervals and minimum shot counts
An independent reimplementation (`audit/audit_recompute.py`, which does not import project evaluation code) re-derived every paired comparison from the per-image rows:
- paired mean, std and % of pairs better: max difference ≤ 1.4e-14;
- **bootstrap 95% CI bounds: reproduced exactly** (≤ 3.6e-15) under the documented rule (2,000 resamples over images, sequences kept together, `default_rng(0)` per group);
- **pass/fail decisions: 0 mismatches**; **minimum shot counts: all match** the evaluation reports (primary, per collection, and the Poisson noise-free secondary);
- **no non-monotone cases**: no method passes at some *N* and then fails at a larger *N*.

Onion: the thesis Table 4.2 CIs and pass/fail outcomes match `comparability.csv`, and Table 4.8 matches `ai_vs_dgi_comparison.csv`. The onion CIs were taken from the frozen CSVs, not re-bootstrapped.

### 2.4 Minimum successful shot count — validation → test (from the frozen evaluation reports)

| Experiment | Method / comparator | All (val → test) | Blood (val → test) | Tissue (val → test) |
|---|---|---|---|---|
| Onion Phase 1 | AI vs independent 10k | 1,000 → 1,000 | — | — |
| Onion Phase 2 | AI-original vs DGI-10000 | 1,000 → 1,000 | — | — |
| Extension, clean target | AI-original vs DGI-10000 | 1,000 → 1,000 | 500 → 500 | 5,000 → 5,000 |
| Extension, 10k-DGI target | AI vs independent 10k | 1,000 → 1,000 | 500 → 500 | 2,000 → 2,000 |
| Extension, 10k-DGI target | smoothed DGI vs independent 10k | 2,000 → 2,000 | 1,000 → 1,000 | 2,000 → 2,000 |
| Extension, 10k-DGI target | conventional DGI vs independent 10k | 5,000 → 5,000 | 5,000 → 5,000 | 5,000 → 5,000 |
| Poisson 5% | AI-original vs noisy DGI-10000 | 1,000 → 1,000 | 500 → 500 | 5,000 → 5,000 |
| Poisson 5% | smoothed DGI vs noisy DGI-10000 | 2,000 → 2,000 | 500 → 500 | 5,000 → 5,000 |
| Poisson 5% | AI-original vs noise-free DGI-10000 | 1,000 → 1,000 | 500 → 500 | 5,000 → 5,000 |
| Poisson 10% | AI-original vs noisy DGI-10000 | 1,000 → 1,000 | 500 → 500 | 5,000 → 2,000 **(differs)** |
| Poisson 10% | smoothed DGI vs noisy DGI-10000 | 2,000 → 2,000 | 500 → 500 | 5,000 → 5,000 |
| Poisson 10% | AI-original vs noise-free DGI-10000 | 2,000 → 1,000 **(differs)** | 500 → 500 | 5,000 → 5,000 |
| Poisson 30% | AI-original vs noisy DGI-10000 | 1,000 → 1,000 | 500 → 500 | 2,000 → 2,000 |
| Poisson 30% | smoothed DGI vs noisy DGI-10000 | 2,000 → 2,000 | 500 → 500 | 5,000 → 5,000 |
| Poisson 30% | AI-original vs noise-free DGI-10000 | 2,000 → 2,000 | 500 → 500 | none ≤ 5,000 → none ≤ 5,000 |

### 2.5 Thesis numbers
- **Main thesis (onion):** 39 table rows checked automatically against the CSVs (Tables 4.1, 4.2, 4.3, 4.4, 4.5, 4.7, 4.8) plus the abstract values. **All agree** at the printed precision. The only difference is cosmetic: Table 4.4 AI-5000 SSIM difference is printed `0.000`, while the CSV value is a tiny negative (prints `−0.000`). The quoted smoothed-DGI-1000 SSIM lower bound (−0.031) also matches.
- **Thesis draft:** regenerating `thesis/draft_extension_sections.md` from the frozen CSVs with its own script gives a **byte-identical** file, so every number in the draft equals the frozen results. Prose claims were also checked by hand (§6 lists the wording that needs correction).

### 2.6 Protocol order and leakage
| Experiment | Protocol written | Pools finished | Last best checkpoint | Frozen after validation | Test report written |
|---|---|---|---|---|---|
| Extension | 2026-11-02 03:04:06 | 05:19:43 | 05:52:50 | 05:54:13 (clean), 05:57:43 (10k DGI) | 05:59:38, 06:03:19 |
| Poisson | 2026-11-02 12:49:52 | 13:13:15 | 14:01:56 | 14:07:31 / 14:07:34 / 14:07:35 | 14:12:51–14:12:54 |

- 10,000 extension images, all IDs unique; source indices unique within each official split; **0 pixel-identical images** in the final set; screening found 0 exact / near-duplicate / flipped pairs among selected images.
- All 20 extension/Poisson training reports: **0 test IDs** in training or selection; reloading the best checkpoint reproduces the logged selection metrics exactly (difference 0.0).

## 3. Numerical inconsistencies and protocol deviations

No reported number was found to disagree with the frozen data. Items to record:

1. **Pre-training smoke test touched test images (minor deviation, undisclosed).** Before any extension model was trained, the evaluation script was smoke-tested on the clean-target **test** split using 24 of the 2,000 test images (12 blood, 12 tissue) and **randomly initialised, untrained** placeholder networks, in a scratch directory. The run was repeated once to confirm the test-once guard. Nothing from it was used for any design, tuning or selection decision, and the conventional-DGI values it printed do not depend on any choice. It is nevertheless a contact with test images before the official single run and should be disclosed. The Poisson smoke test did **not** load test images (its guard exited first).
2. **Cosmetic:** Table 4.4 "0.000" vs CSV −0.0004 (rounds to −0.000).
3. **Environment differs from thesis A.1:** the extensions ran on Linux, Python 3.10, PyTorch 2.14.1+cu126 (GPU), NumPy 2.2.6, scikit-image **0.25.2**, pandas 2.3.2. Thesis A.1 lists CPU PyTorch 2.8.0 and scikit-image **0.26.0**. Onion and extension metrics therefore come from different scikit-image versions. The SSIM implementation is believed stable between these versions, but this was not verified.

## 4. Missing information

1. **No run record for the extension/Poisson test runs.** Onion Phase 1 has `test/run_record.json` (command, start/end, exit code, script MD5 at test time). The extensions only have `evaluation_report.json` (runtime, checkpoint MD5s) and console logs. The evaluation-script MD5 was not recorded at validation or test time; it is recorded only afterwards, in the manifest.
2. **Code not frozen for the onion study.** `FREEZE_MANIFEST_phase1_phase2.md5` covers outputs only. `src/*.py`, `final_*.py`, `orig_*.py` and `data/v2/` are not hashed there (some script MD5s appear only in the thesis appendix). The extensions import `src/` modules, which are also not hashed by any manifest.
3. **No environment lock file** for the Linux runs (no `pip freeze` stored with the outputs).
4. **Earlier onion noise experiment not reported.** `outputs/phase4a_noisy_baselines/` contains an earlier onion experiment with Poisson shot noise plus Gaussian detector noise at a fixed photon scale, and Tikhonov/TV baselines. It is not mentioned in the thesis (which states that no noise was modelled), not frozen, and uses a different noise model from the new Poisson extension (fixed versus per-image relative photon scale).
5. **Similar-pair screening:** 2,183 "similar (review)" pairs remain in the extension set (1,068 across splits). Only the **12 most similar pairs** were inspected visually.
6. **Training-seed variability not measured:** every model was trained once (master seed 0). Run-to-run variation from training randomness is unknown, which matters for the borderline cases. Bit-reproducibility of the GPU runs (deterministic algorithms enabled) was not confirmed by a repeat run.

## 5. Borderline cases and failures

**Validation/test flips (both at 10% noise):**

| Case | Validation lower bounds (SSIM / *r*) | Test lower bounds (SSIM / *r*) | Result |
|---|---|---|---|
| All images, AI-original-1000 vs **noise-free** DGI-10000 | +0.0893 / **−0.0101** | +0.1185 / −0.0042 | fail → pass (validation missed by 0.0001) |
| Tissue, AI-original-2000 vs noisy DGI-10000 | −0.0068 / **−0.0151** | +0.0249 / **−0.0095** | fail → pass (test passes by 0.0005) |

**Other test results within 0.01 (SSIM) or 0.005 (*r*) of a margin:**

| Case | Test lower bounds (SSIM / *r*) | Result |
|---|---|---|
| Extension 10k-DGI, all, AI-500 vs independent 10k | −0.0554 / −0.0139 | fail (*r* misses by 0.004) |
| Extension 10k-DGI, all, smoothed DGI-1000 | −0.0462 / −0.0085 | fail on SSIM |
| Extension 10k-DGI, tissue, AI-1000 | −0.0370 / −0.0128 | fail |
| Poisson 5% and 10%, blood, smoothed DGI-500 | −0.014 / −0.004 | pass (narrow) |
| Poisson 30%, all, AI-original-500 vs noise-free | −0.0263 / −0.1144 | fail |
| Extension clean, all, AI-original-1000 (headline result) | +0.126 / **+0.007** | pass; *r* margin 0.017 |

**Failures:** 60 of 156 test-set criterion decisions fail. They follow the expected pattern: low *N*, tissue, raw DGI, and the stricter noise-free comparator. In particular, **tissue at 30% noise never reaches the noise-free DGI-10000 within 5,000 shots**. Under the clean-object design, tissue needs 5,000 shots even without noise.

## 6. Scientific limitations (for the thesis)

1. **Simulation only.** Bucket values are simulated from the images themselves; the clean object is the simulation source, not an independent ground truth.
2. **Relative Poisson noise.** The photon scale is set per image (about 35,000 to 1.6 million mean photons per bucket for the two pre-check images), so noise levels are relative, not a physical photon budget. Detector readout noise, background light and pattern errors are absent. Photon-dose claims are not supported.
3. **Comparator strength varies.** DGI-10000 is a weak comparator on blood (SSIM 0.396 against the clean object) and a strong one on tissue (0.672), and it degrades with noise. Pass/fail against a noisy comparator therefore partly reflects comparator degradation (for example tissue at 10% and 30% noise passes earlier than at 5%).
4. **Single training run per model;** no estimate of training variance.
5. **Training at the epoch cap.** Most best checkpoints were at epochs 71–80 of 80 (onion and noise-free extension). Results are not converged optima.
6. **Validation reuse.** The same 1,000 validation images (different sequences, seeds 2040–2042) were used for checkpoint selection and for the validation-stage criterion, so validation results may be slightly optimistic. The test set is unaffected.
7. **Multiple comparisons.** 156 test decisions were made without multiplicity correction. The primary endpoint (all images, AI vs comparator) was pre-specified; per-collection and noise-free comparisons should be presented as secondary.
8. **Dataset scope.** MedMNIST+ 64 × 64 images are downsized by the dataset authors (BloodMNIST source images are 360 × 363) and converted to luminance; only blood smears and H&E colon tissue were tested. MedMNIST is stated as **not intended for clinical use**.
9. **Missed structure at low *N* and high noise** (for example blood_test_003004 at 30% noise, 500 shots: three small dark bodies are absent).

## 7. Recommended corrections

**Thesis draft (`thesis/draft_extension_sections.md`):** wording only; no numbers change.
1. §3.14.2: replace "Pairs flagged only as 'similar' … were inspected visually and are distinct images" with: "The 12 most similar of the 2,183 remaining 'similar' pairs were inspected visually and are distinct images; the rest were not individually inspected."
2. §3.14.3: replace "GPU training is deterministic but not bit-identical to CPU training" with "GPU training used deterministic algorithms (not re-verified by a repeat run) and is not bit-identical to CPU training."
3. §3.14.4: add one sentence disclosing the pre-training smoke test (§3.1 above).
4. A.7: add the Linux environment versions (§3.3), including scikit-image 0.25.2.
5. Limitations: add training-seed variability (single run per model) and validation reuse (§6.4, §6.6).

**Main thesis (when merging; existing numbers unchanged):**
6. Optionally print the Table 4.4 AI-5000 SSIM difference as "−0.000" or "0.000 (rounded)". This is cosmetic and can be left as is.
7. Mention `phase4a_noisy_baselines` either as an earlier exploratory noise experiment or explicitly as out of scope, so that "no noise was modelled" is not contradicted by files in the project.

**Record-keeping (new files only, frozen files untouched):**
8. Add a code manifest (for example `FREEZE_MANIFEST_code.md5`) hashing `src/*.py`, `final_*.py`, `orig_*.py`, `train_unet_gi.py`, `run_generalization_validation.py` and `data/v2/splits/`.
9. Store `pip freeze` output for `.venv-linux` beside the extension outputs, as a new file.
10. Add a short `RUN_LOG_extension.md` recording, for each test run, the command, start and end times, exit code (all 0) and evaluation-script MD5 (those in the manifests), mirroring onion `run_record.json`.

**Optional, scientifically useful (new experiment, needs approval):**
11. Retrain the borderline models (10% noise: AI-original-1000 and AI-original-2000) with 2–3 extra training seeds on validation only, to quantify training variance. This would be a new protocol and must not touch the frozen test results.

## 8. How this audit was done (reproducible, read-only)

- `audit/audit_recompute.py` — independent recomputation of aggregates, paired differences, bootstrap CIs, decisions and minimum *N* for the 10 extension/Poisson result folders; output `audit/audit_recompute.json`.
- `audit/audit_onion_thesis.py` — checks onion thesis tables and abstract against the frozen onion CSVs.
- `audit/audit_tables.py` — prints the validation → test minimum-shot-count table (§2.4).
- MD5 checks: `md5sum -c` on the three manifests plus a directory walk for unlisted files.
- Smoke-test facts (§3.1) and the similar-pair inspection count (§4.5) come from the session transcript of the extension work.
