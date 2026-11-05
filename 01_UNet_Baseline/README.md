# AI-DRIVEN Reconstruction of Quantum-Enhanced Low-Photon Bio-Imaging Datasets

**LowShot Ghost Imaging: AI-assisted image reconstruction using reduced measurements.** AI-based reconstruction for measurement-shot reduction in simulated differential ghost imaging.
Repository: [`harikishor123/LowShot-Ghost-Imaging`](https://github.com/harikishor123/LowShot-Ghost-Imaging) (public). Some project material is deliberately **not** in this repository; see [What is not included](#what-is-not-included).
MTP project, IIT Mandi.

> **Simulation only.** All measurements in this repository are computer-simulated: noise-free, or with a simplified relative Poisson noise model. The claims concern the **number of measurement shots** only. No hardware experiment was performed, and no photon-dose, biological-effect or quantum-advantage claim is made. The 10,000-shot DGI reconstruction is a study-specific benchmark, not ground truth.

---

## 1 · Overview and research objectives

Ghost imaging (GI) builds an image from many structured illumination patterns and a single-pixel ("bucket") detector. Quality improves with the number of measurement shots. This project asks:

> Can a trained U-Net reconstruct an image from **far fewer shots** and still match a conventional **10,000-shot differential ghost imaging (DGI)** reconstruction?

Objectives:
1. Build a computational GI pipeline: patterns, buckets, DGI.
2. Train U-Nets that map reduced-shot DGI (500 / 1,000 / 2,000 / 5,000 shots) to a better image.
3. Decide with a **pre-registered statistical criterion** the minimum number of shots that is non-inferior to the 10,000-shot comparator.
4. Test the result on unseen images and unseen illumination sequences.
5. Repeat it on two further microscopy collections and under simulated Poisson bucket noise.

### Results (frozen test sets, each evaluated once)

| Experiment | Test data | Minimum shots meeting the criterion |
|---|---|---|
| Onion Phase 1 (target: 10k DGI) and Phase 2 (target: clean object) | 74 onion images × 4 sequences | **1,000 (10×)** in both phases |
| Extension, noise-free (MedMNIST+) | 1,000 blood + 1,000 tissue images × 4 sequences | all images **1,000 (10×)**; blood 500 (20×); tissue 2,000–5,000 |
| Poisson bucket noise 5 / 10 / 30% | same 2,000 images | all images **1,000** at every level (vs noisy DGI-10000); blood 500; tissue 5,000 / 2,000 (borderline) / 2,000 |

**Pass rule:** the lower bound of the paired 95% bootstrap confidence interval of (method − comparator) is ≥ −0.02 for SSIM **and** ≥ −0.01 for Pearson *r*. The bootstrap uses 2,000 resamples over images, with sequences kept together.

The onion results and the BloodMNIST / PathMNIST results are separate experiments and are **never pooled**. Details are in `EXPERIMENTS.md` and the thesis.

## 2 · Pipeline

```
clean 64×64 object O ──► Bernoulli(0.5) patterns P_i (seeded) ──► bucket B_i = Σ P_i·O
                                                                   (+ Poisson noise in the noise extension)
                                       reference arm R_i = Σ P_i
          calibrated DGI from the first N shots ─► DGI-N  (N = 500 … 10,000, nested prefixes, clipped to [0,1])
          DGI-N ─► U-Net (UNetGI, base 32, 1.93 M parameters, no normalisation layers) ─► reconstruction
          score vs clean object (or vs the 10k DGI of the same acquisition) ─► SSIM, Pearson r, PSNR, MSE, edge r
```

| Step | Code |
|---|---|
| Patterns, bucket model | `src/ghost_imaging.py` |
| Streaming DGI | `src/reconstruction.py` (`StreamingDGI`); readouts in `final_build_pool.py` (`dgi_readouts`) |
| Poisson bucket noise | `ext_poisson_common.py` (`NoisyStreamingDGI`): B′ = Poisson(sB)/s, s = 2ΣO/(f²ΣO²) |
| U-Net | `src/unet_gi.py` |
| Training | `final_train.py`, `orig_train.py`, `ext_train.py`, `ext_poisson_train.py` (Adam 1e-4, MSE, batch 16, ≤ 80 epochs, patience 10, seed 0) |
| Evaluation and criterion | `final_evaluate.py`, `orig_evaluate.py`, `ext_evaluate.py`, `ext_poisson_evaluate.py`; metrics in `src/metrics.py` |

## 3 · Repository structure

The experiment files stay **where they were created**. Freeze manifests, `protocol.json`, `frozen_for_test.json`, the thesis figure links and the scripts' imports all use these paths, so the folders were not rearranged. The numbered `01_…`–`08_…` folders are guides that point to the real files. `FILE_INDEX.md` lists every important file, and `07_Reproducibility_and_Audit/REPO_INVENTORY.csv` lists every file.

```
README.md  EXPERIMENTS.md  REPRODUCIBILITY.md  FILE_INDEX.md  FINAL_GITHUB_READINESS_REPORT.md
requirements.txt                       Windows/CPU environment (onion study)
08_Environment_and_Dependencies/       Linux/GPU environment (extensions)
src/                                   shared library (GI simulation, DGI, U-Net, metrics, data)
final_*.py  orig_*.py                  onion Phase 1 / Phase 2 pipelines
ext_*.py                               MedMNIST+ extension
ext_poisson_*.py                       Poisson-noise extension
infer_example.py                       run a frozen checkpoint on a few images
train_unet_gi.py, run_generalization_validation.py   imported by the final pipelines
run_*.py, phase4a_*.py, train_model_D.py, …          earlier exploratory phases (preliminary)
data/v2/splits/, data/v2/metadata/     onion split lists (images themselves not included)
outputs/
  phase_final_low_shot_to_10k_reference/   onion Phase 1   FROZEN
  phase_final_original_target/             onion Phase 2   FROZEN
  extension_medmnist_microscopy/           extension       FROZEN
  extension_medmnist_microscopy_poisson/   Poisson noise   FROZEN
  phase1_single_image/ … phase4a_noisy_baselines/   preliminary, not reported results
results/  models/  configs/            earliest exploratory work and the onion dataset audit
audit/                                 scientific audit, independent recomputation, analyses, galleries
final_comparison_gallery/              main image gallery (open index.html)
thesis/                                thesis-support scripts and the claims check (the thesis text itself is not published, see below)
presentation/                          deck notes and review list (the .pptx decks are withheld: they contain onion images)
01_… – 07_…                            guide folders; 07 also holds manifests, inventory, release plan
```

Not in Git: datasets, training pools (62 GB), model checkpoints, virtual environments, and the files withheld pending permission (onion micrographs, slide decks, processed MedMNIST+ data). See §4–§5.

## 4 · Datasets

| Dataset | Used for | Licence | How to obtain |
|---|---|---|---|
| **MedMNIST+ 64×64** BloodMNIST, PathMNIST (Yang et al. 2023; sources Acevedo et al. 2020, Kather et al. 2019) | extension, Poisson | CC BY 4.0 | download `bloodmnist_64.npz` and `pathmnist_64.npz` from Zenodo record **10.5281/zenodo.10519652** (v3.0) into `data/extension_medmnist/raw/` |
| Processed objects `objects_64.npz` | extension inference / evaluation | derivative of MedMNIST+; redistribution licence **being verified** | **not distributed** for now; rebuild it from the raw files (below) |
| **Onion microscopy**, Roboflow export "Onion-CNN-Final" v1 (644 images) | onion study | marked CC BY 4.0 in `data.yaml`; **origin and redistribution permission not yet confirmed** | not distributed; Roboflow project `onion-cnn-final-flpoy` v1 (workspace as recorded in `data.yaml`), YOLOv8 export, into `train/` and `valid/` (access to be confirmed) |

**Withheld pending permission.** Until permission is confirmed, the repository contains **no onion micrograph** and **no processed MedMNIST+ data**. That means 186 onion figures and examples, all slide decks, and one MedMNIST-derived array; the 60 metric plots in those folders stay. Nothing was deleted locally. The list and the effect on each script are in `07_Reproducibility_and_Audit/EXCLUDED_PENDING_PERMISSION.md`.

Expected MD5s:
- `bloodmnist_64.npz`: `2b94928a2ae4916078ca51e05b6b800b`
- `pathmnist_64.npz`: `55aa9c1e0525abe5a6b9d8343a507616`
- `objects_64.npz`: `73e8ce929c5a7c67111e028032e528c1`

```
md5sum data/extension_medmnist/raw/*.npz data/extension_medmnist/processed/objects_64.npz
```

**Rebuilding derived data.** The original outputs are frozen; after any rebuild, check that it reproduced them exactly.
- **MedMNIST objects:** `python ext_build_dataset.py`. It also rewrites `outputs/extension_medmnist_microscopy/dataset/*`. Run `md5sum -c outputs/extension_medmnist_microscopy/FREEZE_MANIFEST_extension.md5 --ignore-missing` afterwards, and if anything differs, restore it with `git checkout -- outputs/`.
- **Onion data** (`data/v2/reference/`, needed for onion runs): `python src/build_v2_dataset.py`, from the Roboflow export and the committed audit outputs in `results/dataset_audit/`. Then check `md5sum -c 07_Reproducibility_and_Audit/FREEZE_MANIFEST_code_2026-11-05.md5`, which covers the split lists.

## 5 · Checkpoints

**No checkpoint is distributed with this repository, and no Release exists.** The 28 evaluated `best.pt` files (0.65 GB) stay with the authors:
- **Onion checkpoints (8):** withheld. They were trained on the onion micrographs, whose redistribution permission is unresolved; excluding the images does not by itself make the weights redistributable (`DECISIONS_REQUIRED.md` §3.2).
- **MedMNIST+ checkpoints (20; extension and Poisson):** trained only on CC BY 4.0 data, but not distributed until the authors approve a Release (plan only: `07_Reproducibility_and_Audit/release_plan/RELEASE_PLAN.md`).

**What a researcher can do now:**
- **Verify** any copy obtained from the authors: the MD5 and SHA-256 values of all 28 checkpoints, as frozen before each single test run, are in `07_Reproducibility_and_Audit/release_plan/MD5SUMS_checkpoints.txt` and `SHA256SUMS_checkpoints.txt`, and in each `frozen_for_test.json`.
- **Retrain** the MedMNIST+ models with the committed scripts (`ext_train.py`, `ext_poisson_train.py`; seeds and settings in each `protocol.json`; see `REPRODUCIBILITY.md` §3). Retrained weights reproduce the published decisions up to GPU/TF32 rounding but will not be bit-identical to the frozen files. Training into a new output folder is required: the frozen folders must not be overwritten.
- **Onion models** can be retrained only with access to the onion images (§4).

## 6 · Environment

Two environments were used; versions are recorded in `08_Environment_and_Dependencies/README.md`.

| Part | Environment | Install |
|---|---|---|
| Onion study | Windows, Python 3.13, PyTorch 2.8.0 **CPU** | `py -3.13 -m pip install -r requirements.txt` |
| Extensions, audits, galleries | Linux, Python 3.10, PyTorch 2.14.1 + CUDA 12.6 | `python3.10 -m venv .venv-linux && .venv-linux/bin/pip install -r 08_Environment_and_Dependencies/requirements-linux-minimal.txt --extra-index-url https://download.pytorch.org/whl/cu126` |

The full Linux `pip freeze` (`requirements-linux-freeze.txt`) includes Ubuntu system packages and is a record, not an install file. Inference also works on CPU, with small rounding differences (§7).

## 7 · Run inference and reproduce the reported metrics

**Check reported numbers without any computation.** These read only the frozen CSVs:
```
python audit/audit_recompute.py /tmp/recompute.json      # all extension + Poisson aggregates, CIs, decisions, minimum shots
python audit/audit_onion_thesis.py                       # onion thesis tables vs CSVs
python thesis/merge_proposal/verify_claims.py            # 29 headline claims -> CLAIMS_VERIFICATION.md
```

**Run a checkpoint (inference only).** Needs `objects_64.npz` (rebuild from MedMNIST+, §4) and a checkpoint, which is not distributed here (§5). Uses validation images by default:
```
python infer_example.py \
  --checkpoint outputs/extension_medmnist_microscopy/models/target_clean/model_1000/checkpoints/best.pt \
  --shots 1000 --n-images 4 --out demo_out/
```
It prints SSIM / *r* / PSNR of DGI-1000, U-Net-1000 and DGI-10000 against the clean object. On 4 validation images these matched the frozen `per_image_results.csv` to within 6 × 10⁻⁵.

**Re-running full evaluations or training.** See `REPRODUCIBILITY.md`. The evaluation scripts refuse a second test run and must not write into the frozen folders; copy the scripts and change their output folder first. GPU results are bit-exact only with the original batch layout (TF32 rounding ≤ 0.0008 per pixel, ≤ 4 × 10⁻⁵ SSIM).

## 8 · Verify the frozen files

In the original working copy, every manifest passes completely:
```
md5sum -c FREEZE_MANIFEST_phase1_phase2.md5                                          # 134 files
md5sum -c outputs/extension_medmnist_microscopy/FREEZE_MANIFEST_extension.md5        # 131 files
md5sum -c outputs/extension_medmnist_microscopy_poisson/FREEZE_MANIFEST_poisson.md5  # 180 files
md5sum -c 07_Reproducibility_and_Audit/FREEZE_MANIFEST_code_2026-11-05.md5           # 69 code files
```

In a **clone**, 95 manifest entries are not in Git: 28 evaluated checkpoints and 28 `last.pt` files (not distributed), training pools and datasets, and 20 onion figures withheld pending permission. Add `--ignore-missing` so that only the files present are checked:
```
md5sum -c --ignore-missing FREEZE_MANIFEST_phase1_phase2.md5
md5sum -c --ignore-missing outputs/extension_medmnist_microscopy/FREEZE_MANIFEST_extension.md5
md5sum -c --ignore-missing outputs/extension_medmnist_microscopy_poisson/FREEZE_MANIFEST_poisson.md5
```
If you obtain checkpoints from the authors, place them at their original paths and the same commands check them too.

## 9 · Limitations

- **Simulation only:** noise-free buckets, or a relative Poisson model. The 5 / 10 / 30% levels are relative to each image's bucket-signal spread, not physical photon budgets. There is no detector readout noise, background light or pattern error, and no hardware.
- **Not ground truth:** the 10,000-shot DGI is a study-specific benchmark. On blood smears it is weak (SSIM 0.396), which inflates the blood margin.
- **Over-smoothing:** the U-Net preserves structure but smooths fine detail. At 1,000 shots it keeps about one third (blood) and one sixth (tissue) of the fine-detail content, at up to 10% noise.
- **Tissue at 30% noise:** 1,000 shots fails; 2,000 passes against noisy DGI-10000. Tissue never reaches noise-free DGI-10000 within 5,000 shots.
- **Borderline results:** two results at 10% noise differ between validation and test by margins of 0.0001–0.0005.
- **Training and data reuse:** each model was trained once (seed 0), and training was not fully converged at the 80-epoch cap. Validation images were reused for model selection. Phase 2 reuses the Phase 1 test images. A 24-image smoke test with untrained networks touched the extension test set before training; no decision used it.
- **Scope of data:** 64 × 64 images of onion, blood smears and H&E colorectal tissue only.

The full list is in thesis Section 4.12; the thesis is not published here (see [What is not included](#what-is-not-included)).

## 10 · Thesis, results and galleries

| What | Where |
|---|---|
| Thesis | **not published** (see [What is not included](#what-is-not-included)); headline claims are checked against the frozen results by `thesis/merge_proposal/verify_claims.py` → `CLAIMS_VERIFICATION.md` |
| Result CSVs | `outputs/*/…/{val,test}/` (`per_image_results.csv`, `aggregate_metrics.csv`, `comparisons.csv`, `evaluation_report.json`) |
| Audit | `audit/SCIENTIFIC_AUDIT_2026-11-05.md`, `audit/final_verification/`, `FINAL_GITHUB_READINESS_REPORT.md` |
| Galleries | `final_comparison_gallery/index.html`, `audit/test40_gallery/index.html`, `audit/visual_gallery/index.html` |

## 11 · Licence and credits

- **Licence:** **no licence has been granted yet.** The decision is pending approval by the authors and their institute (options: `07_Reproducibility_and_Audit/LICENSING_RECOMMENDATION.md`). Until a `LICENSE` file is added, default copyright applies: you may view the code, but no permission to reuse or redistribute it is given.
- **Data credits** (full notice, licence terms and the list of changes: [`ATTRIBUTION.md`](ATTRIBUTION.md)):
  - **MedMNIST+** (Zenodo doi:10.5281/zenodo.10519652; CC BY 4.0; not for clinical use). Cite J. Yang et al., *Sci. Data* 10, 41 (2023) **and** J. Yang, R. Shi, B. Ni, *IEEE ISBI* 2021, pp. 191–195.
  - **BloodMNIST source:** A. Acevedo et al., *Data in Brief* 30, 105474 (2020).
  - **PathMNIST source:** J. N. Kather et al., *PLOS Med.* 16, e1002730 (2019).
  - **Onion images:** Roboflow "Onion-CNN-Final" v1. **Not included**: origin and permission unconfirmed.

## What is not included

| Material | Why | How to obtain or rebuild |
|---|---|---|
| Onion micrographs, onion figures and examples (186 files), raw Roboflow export, derived onion data | redistribution permission not confirmed | ask the authors; onion results are available as CSV/JSON numbers and metric plots |
| All 28 model checkpoints | onion: permission unresolved; MedMNIST+: not yet approved for release | retrain with the committed scripts (§5); hashes of the frozen checkpoints are in `07_Reproducibility_and_Audit/release_plan/` |
| Slide decks | contain onion micrographs | — |
| Processed MedMNIST+ objects (`objects_64.npz`) and a derived array | redistribution of the processed data not yet approved | download MedMNIST+ from Zenodo and rebuild (§4) |
| Training pools and selection sets (62 GB) | size | rebuild with `*_build_pool.py` (§4, `REPRODUCIBILITY.md`) |
| Thesis text (original, review copy with extensions, drafts) | not yet approved for publication by the authors and the guide | — |
| Internal review, viva and decision notes | internal; some collect personal details | — |

The exact lists are in `07_Reproducibility_and_Audit/EXCLUDED_PENDING_PERMISSION.md` and `07_Reproducibility_and_Audit/EXCLUDED_FROM_PUBLIC_RELEASE.md`. The freeze manifests still list the excluded frozen files; check a clone with `md5sum -c --ignore-missing` (§8).

## Rules for this repository

- Never edit, regenerate or delete anything listed in the freeze manifests.
- Each test set was evaluated once; the evaluation scripts refuse a second test run.
- Keep onion results separate from the BloodMNIST / PathMNIST results.
