# README Update Report

**Date:** 2026-11-22  
**Scope:** Complete rewrite of root `README.md` from a minimal directory listing to a professional research project homepage.

---

## Changes Made

### 1. README.md — Complete Rewrite

**Before:** 65-line placeholder with directory descriptions and venv troubleshooting notes.  
**After:** Comprehensive research homepage (~280 lines) with:

- Professional header with badges (Python, PyTorch, License, Data)
- Hero image showing U-Net reconstruction quality across shot counts
- Project overview with two-stage experiment summary table
- Pipeline section with figure and step-by-step explanation
- Full experiment descriptions (E1–E4) with architecture details
- Results tables (PSNR, SSIM, latency, statistical significance)
- Dataset section with contact sheet
- Repository structure guide
- Reproducibility section with commands and verification steps
- Limitations section (7 items)
- Data attribution with required citations
- License status notice
- Reports and documentation index

### 2. docs/images/ — New Directory (9 figures)

All figures copied from existing project outputs. No images were fabricated, altered, or sourced externally.

| File | Source | Description |
|------|--------|-------------|
| `pipeline_overview.png` | `01_UNet_Baseline/research_report/figures/fig01_pipeline.png` | CGI simulation pipeline flowchart |
| `unet_architecture.png` | `01_UNet_Baseline/research_report/figures/fig02_unet.png` | U-Net architecture diagram |
| `ssim_vs_shots_extension.png` | `01_UNet_Baseline/research_report/figures/fig04_ext_ssim.png` | SSIM vs shots, MedMNIST extension |
| `ssim_vs_shots_poisson.png` | `01_UNet_Baseline/research_report/figures/fig05_poisson_ssim.png` | SSIM vs shots, Poisson noise |
| `medmnist_reconstruction_examples.png` | `01_UNet_Baseline/research_report/figures/fig07_medmnist_examples.png` | Blood + tissue reconstructions with SSIM |
| `unet_comparison_grid.png` | `01_UNet_Baseline/outputs/extension_medmnist_microscopy/results/target_clean/test/figures/comparison_examples.png` | Full comparison grid (hero image) |
| `dataset_contact_sheet.png` | `01_UNet_Baseline/outputs/extension_medmnist_microscopy/dataset/contact_sheet.png` | MedMNIST+ sRGB + GI object pairs |
| `difftransfuse_N1000_blood.png` | `02_DiffTransFuse/reports/final_test_evaluation/figures/test_N1000_0000.png` | DiffTransFuse comparison strip, blood |
| `difftransfuse_N1000_tissue.png` | `02_DiffTransFuse/reports/final_test_evaluation/figures/test_N1000_1499.png` | DiffTransFuse comparison strip, tissue |

**Note:** The `research_report/` directory is in `.gitignore` (not tracked on GitHub). Copying selected figures to `docs/images/` makes them available on GitHub without un-ignoring the full report directory.

### 3. Files NOT Changed

- No experiment code, configs, or outputs were modified
- No `.gitignore` rules were changed
- No model checkpoints were touched
- No existing documentation in subprojects was altered

---

## Metrics Verification

All numerical values in the README were cross-checked against:

| Metric | Source file | Verified |
|--------|------------|----------|
| PSNR table (E1–E4) | `02_DiffTransFuse/reports/final_test_evaluation/FINAL_TEST_REPORT.md` §2 | ✓ |
| SSIM table (E1–E4) | `02_DiffTransFuse/reports/final_test_evaluation/FINAL_TEST_REPORT.md` §3 | ✓ |
| Latency (E1, E3a, E4) | `02_DiffTransFuse/reports/final_test_evaluation/FINAL_TEST_REPORT.md` §7 | ✓ |
| Bootstrap CIs (E4a−E1, E4b−E1) | `02_DiffTransFuse/reports/final_test_evaluation/FINAL_TEST_REPORT.md` §6 | ✓ |
| U-Net parameters (1,925,025) | `01_UNet_Baseline/EXPERIMENTS.md` | ✓ |
| TransFusionNet parameters | `02_DiffTransFuse/reports/E4_TRANSFORMER_FUSION.md` §2.2 | ✓ |
| Dataset split (7000/1000/2000) | `01_UNet_Baseline/EXPERIMENTS.md` §2 | ✓ |
| Pass rule (−0.02 SSIM, −0.01 r) | `01_UNet_Baseline/EXPERIMENTS.md` header | ✓ |
| Poisson noise levels (5/10/30%) | `01_UNet_Baseline/EXPERIMENTS.md` §3 | ✓ |
| Test set size (2000 images × 3 seeds) | `02_DiffTransFuse/reports/final_test_evaluation/FINAL_TEST_REPORT.md` §1.1 | ✓ |

**No results were fabricated.** Every number traces to a verified source document.

---

## Image Verification

All images are copies of existing project outputs:
- **Hero image** (`unet_comparison_grid.png`): already tracked in git at the source path
- **DiffTransFuse strips**: already tracked in git at the source path
- **Dataset contact sheet**: already tracked in git at the source path
- **Research report figures** (pipeline, U-Net, SSIM charts, examples): on disk but not in git (the `research_report/` directory is gitignored); now accessible via `docs/images/`

No images were altered, cropped, or modified in any way that changes their scientific meaning. All copies are byte-identical to their sources.
