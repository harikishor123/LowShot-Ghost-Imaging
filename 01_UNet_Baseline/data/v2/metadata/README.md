# data/v2: clean reference dataset for photon-limited reconstruction (STEP 3)

Built by `src/build_v2_dataset.py` from the original Roboflow export in `train/` and `valid/`, which is unmodified. Validated by `src/validate_v2_preprocessing.py`; the outputs are in `results/preprocessing_v2/`.

## Contents
| path | what |
|---|---|
| `reference/<id>.npz` | **Clean reference for the experiment.** `srgb16`: uint16 sRGB-encoded RGB (H×W×3). `mask`: uint8 field-of-view (FOV) mask (H×W). Load it with `color_space.load_reference_linear()`, which returns linear-light RGB and the mask. |
| `images/<id>.png` | 8-bit sRGB preview of the same data. Pixels outside the FOV are 0. For viewing only. |
| `masks/<id>.png` | FOV mask (255 = valid specimen pixel). |
| `labels/<id>.txt` | YOLO boxes (`0` = d, dividing; `1` = nd, non-dividing), transformed to the processed frame. **Auxiliary only**: the reconstruction pipeline does not depend on them. |
| `splits/{train,val,test}/ids.txt` | Split membership. `images.txt` and `references.txt` give relative paths. |
| `metadata/manifest.csv` | One row per **original** image (644 rows): source, dimensions, crop box, scale, flags, exclusion reason, duplicate group, split. |
| `metadata/exclusions.csv`, `duplicate_groups.csv`, `preprocessing_config.json`, `build_summary.json` | Provenance of every decision. |
| `data_yolo_aux.yaml` | Optional YOLO view of the dataset (auxiliary). |

**The test split is held out.** Do not use it for training, model selection, early stopping, tuning or photon-level calibration.

## Processing for each retained image
1. Strip uniform white A4 page padding.
2. Detect the black microscope surround: dark pixels connected to the image edge.
   - Group F's caption banner is treated as part of that surround.
3. Take the convex hull of the largest specimen component as the FOV.
   - Circular eyepiece fields become a circular mask.
   - Plain crops become a rectangle covering about 100% of the frame.
4. Crop to the FOV bounding box. There is no centre-crop and no padding to square, so the aspect ratio is preserved.
5. Convert sRGB to linear light. Area-average downscale (in linear light) so that the long side is 1024 px; images are **never upscaled**. Convert back to sRGB for storage.
6. Erode the circular FOV mask by 4 px to drop the coloured eyepiece rim, then set pixels outside the mask to 0.
7. Transform the YOLO boxes by the same crop and scale. A box is dropped (and recorded) only if its centre falls outside the FOV mask; 11 of 15,340 were dropped.

There is no histogram equalisation, no contrast stretching and no per-image normalisation. The intensities are the original JPEG values, apart from resampling.

## Colour / photon model (implemented in `src/color_space.py`)
JPEG RGB (uint8, sRGB-encoded) → normalised RGB [0,1] → **linear light** (inverse IEC 61966-2-1 sRGB curve) → *(STEP 4)* Poisson photon simulation → noisy photon-limited image.

Assumptions:
- No ICC profile or EXIF data exists, so the standard sRGB curve is assumed. The real camera tone curve, white balance, sharpening and denoising are unknown.
- Linear values are *relative* light levels, not calibrated photon counts. The photon scale is a free simulation parameter.
- Brightfield is transmitted light: stained structures receive fewer photons, so they will be relatively noisier.
- The references were already JPEG-compressed (quality ~75) and partly rescaled by the original authors. They are therefore "clean" only relative to the simulated noise.
- **This is a simulation study.** Noisy images derived from these references are synthetic, not experimentally acquired low-photon measurements.
