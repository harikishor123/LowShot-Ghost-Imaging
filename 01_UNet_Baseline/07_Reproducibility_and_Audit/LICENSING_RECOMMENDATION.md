# Licensing recommendation (for your decision)

This is not legal advice. **No `LICENSE` file has been created.** Before choosing, check your institute's policy on student project code and theses. Some institutes claim rights, or require a specific licence or approval before publication.

The repository contains three kinds of material with different owners. They need separate treatment.

## 1 · Original work by the team: code, documentation, figures, thesis text

| Material | Recommendation | Notes |
|---|---|---|
| Code (`*.py`, `src/`, audit and gallery scripts) | **MIT** (short, permissive), or **BSD-3-Clause** if you want to forbid use of your names for endorsement | needs the agreement of all three authors (and the guide, if the institute requires it) |
| Documentation, result tables, figures, thesis text, slides | **CC BY 4.0** | lets others reuse with attribution. The thesis may be subject to institute rules: you may prefer to keep the thesis "all rights reserved" until it is submitted or accepted |
| Frozen results (CSV, JSON, logs) and checkpoints | CC BY 4.0, matching the data they were derived from | the checkpoints were trained on CC BY 4.0 data (MedMNIST+) and on the onion images (see §3) |

**Licence text.** A typical `LICENSE` would hold the MIT text for the code, plus a short section stating that documentation, figures and results are under CC BY 4.0, and that third-party data keep their own licences (§2, §3). The copyright holder line ("Copyright (c) 2026 …") needs your confirmed names.

## 2 · Third-party datasets: MedMNIST+ (BloodMNIST, PathMNIST)

| Item | Status |
|---|---|
| Licence | CC BY 4.0, as recorded in `outputs/extension_medmnist_microscopy/protocol.json` from the Zenodo record 10.5281/zenodo.10519652 (v3.0). The MedMNIST documentation says each subset keeps the licence of its source; for BloodMNIST and PathMNIST this is CC BY 4.0. **Please re-confirm on the Zenodo page before publishing.** |
| Raw files (`bloodmnist_64.npz`, `pathmnist_64.npz`, 1.2 GB) | **Do not commit.** Users download them from Zenodo; the MD5s are recorded and checked by `ext_build_dataset.py`. |
| Derived objects (`objects_64.npz`, 229 MB) | CC BY 4.0 allows redistribution of derivatives with attribution and an indication of changes. It can be offered as an optional Release asset (see the release plan) with credits [25]–[27] and the statement "converted to linear-light luminance, 10,000 images selected". **Your decision.** |
| Figures and galleries showing blood or tissue images | allowed under CC BY 4.0 with attribution; add a credits line to the README (done) |
| Required credits | Yang et al. 2023 (MedMNIST v2); Acevedo et al. 2020 (blood); Kather et al. 2019 (tissue): thesis references [25]–[27] |

## 3 · Onion microscopy images (Roboflow "Onion-CNN-Final")

| Item | Status |
|---|---|
| Licence as recorded | `data.yaml` says `license: CC BY 4.0`, for the Roboflow project `onion-cnn-final-flpoy`, version 1 (workspace as recorded in `data.yaml`) |
| Origin of the images | **UNVERIFIED.** The workspace appears to belong to a team member. Who acquired the 644 original micrographs is not documented: the team, a lab, or a third-party source. A CC BY 4.0 tag set by an uploader grants no rights the uploader did not have. |
| Raw images (`train/`, `valid/`, 142 MB) | **Do not commit** until the origin is confirmed. The proposed `.gitignore` already excludes them. |
| Derived onion data (`data/v2/`, 7.9 GB) | not committed (size); same permission question |
| Onion images inside committed figures (`outputs/phase_final_*/…/figures`, `final_comparison_gallery`, `results/`, the thesis) | these are reproductions of the onion images, so they need the same permission. **If the team acquired the images itself**, there is no issue: state "images acquired by the authors" in the README. **If they come from a third party**, confirm their licence first, or keep onion figures out of the public repository |

**What I need from you:** who took or provided the onion micrographs, and under what terms.

## 4 · Summary of decisions

1. Code licence: MIT or BSD-3-Clause (recommended), or another.
2. Licence for documentation, figures and results: CC BY 4.0 (recommended), or another.
3. Is the thesis included in the public repository now, or only after acceptance?
4. Origin and permission of the onion images (blocks the public release of onion figures).
5. Is `objects_64.npz` published as a Release asset?
6. The copyright holder names for `LICENSE`.
