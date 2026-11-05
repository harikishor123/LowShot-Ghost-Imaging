# Data attribution and licence notices

This repository contains **figures and galleries** showing images derived from MedMNIST+ (BloodMNIST and PathMNIST). It contains **no dataset files**: no raw MedMNIST+ archives, no processed objects (`objects_64.npz`) and no onion micrographs. Source licences were checked against the public records on 2026-11-05.

## MedMNIST+ (BloodMNIST, PathMNIST)

- **Dataset:** "[MedMNIST+] 18x Standardized Datasets for 2D and 3D Biomedical Image Classification with Multiple Size Options: 28 (MNIST-Like), 64, 128, and 224", version 3.0, Zenodo, doi:[10.5281/zenodo.10519652](https://doi.org/10.5281/zenodo.10519652). The 64 × 64 files `bloodmnist_64.npz` and `pathmnist_64.npz` were used.
- **Licence:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), as stated by the dataset authors. DermaMNIST is the one exception (CC BY-NC 4.0); it was not used.
- **Not for clinical use:** "this dataset is NOT intended for clinical use" (MedMNIST authors). Nothing in this repository is a clinical or diagnostic tool.
- **Required citations.** The MedMNIST authors ask users to cite both MedMNIST papers and "the corresponding paper(s) of source data":
  1. J. Yang, R. Shi, D. Wei, Z. Liu, L. Zhao, B. Ke, H. Pfister and B. Ni, "MedMNIST v2 – A large-scale lightweight benchmark for 2D and 3D biomedical image classification," *Scientific Data*, vol. 10, no. 1, p. 41, 2023, doi:10.1038/s41597-022-01721-8.
  2. J. Yang, R. Shi and B. Ni, "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis," in *IEEE 18th International Symposium on Biomedical Imaging (ISBI)*, 2021, pp. 191–195, doi:10.1109/ISBI48211.2021.9434062.
  3. **BloodMNIST source:** A. Acevedo, A. Merino, S. Alférez, Á. Molina, L. Boldú and J. Rodellar, "A dataset of microscopic peripheral blood cell images for development of automatic recognition systems," *Data in Brief*, vol. 30, Art. no. 105474, 2020, doi:10.1016/j.dib.2020.105474. The dataset is on Mendeley Data (`snkd93bnjr`, v1, CC BY 4.0).
  4. **PathMNIST source:** J. N. Kather et al., "Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study," *PLOS Medicine*, vol. 16, no. 1, e1002730, 2019, doi:10.1371/journal.pmed.1002730. The dataset is J. N. Kather, N. Halama and A. Marx, "100,000 histological images of human colorectal cancer and healthy tissue," Zenodo, doi:10.5281/zenodo.1214456 (CC BY 4.0).

**Changes made to the original images** (CC BY 4.0 §3(a)(1)(B)):
- 10,000 images (5,000 per collection) were drawn from the official splits, stratified by class (seed 42).
- They were screened for exact duplicates and near-blank images (one duplicate and 47 near-blank images removed).
- They were converted from sRGB to linear-light luminance Y = 0.2126 R + 0.7152 G + 0.0722 B, at the original 64 × 64 size.
- Every image shown in this repository is either that luminance image or a **simulated** ghost-imaging reconstruction of it: DGI from simulated measurements, optionally with simulated Poisson noise, or a U-Net output. None is an unmodified original.

**Where MedMNIST-derived images appear:**
- `final_comparison_gallery/`, `audit/test40_gallery/`, `audit/visual_gallery/`, `audit/poisson_visual_analysis/figures/`, `audit/final_verification/figures/`;
- the figures inside `outputs/extension_medmnist_microscopy*/` (frozen experiment outputs, including `dataset/contact_sheet.png`).

The per-image CSV files contain metric values and image identifiers only.

**Disclaimer** (as in CC BY 4.0): the licensors offer the material as-is and make no warranties. Credit to the dataset authors does not imply their endorsement of this work.

## Onion micrographs (Roboflow export "Onion-CNN-Final", v1)

The onion study used 644 brightfield micrographs from a Roboflow export, marked CC BY 4.0 in `data.yaml`. **The images are not included in this repository**, and neither are figures showing them, slide decks, onion model checkpoints or derived onion data. The origin of the original micrographs, and therefore the permission to redistribute them or works derived from them, has not yet been confirmed. See `07_Reproducibility_and_Audit/EXCLUDED_PENDING_PERMISSION.md`. The committed onion results are numbers only (CSV and JSON) plus metric plots.

## This repository's own code and text

**No licence has been granted yet.** The licence decision is pending approval by the authors and their institute. Until a `LICENSE` file is added, the default copyright rules apply: you may view the code, but no permission to reuse or redistribute it is given. The options under consideration are in `07_Reproducibility_and_Audit/LICENSING_RECOMMENDATION.md`.
