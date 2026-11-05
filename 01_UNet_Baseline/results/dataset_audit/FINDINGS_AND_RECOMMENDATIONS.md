# STEP 2 audit: key findings and proposed preprocessing

Full numbers are in `AUDIT_REPORT.md`; per-image data is in `per_image_audit.csv`. Nothing in `train/` or `valid/` was modified: all 1,288 image and label files were MD5-hashed before and after the run, and none changed.

## Dataset

- Location: `train/images` (605) + `valid/images` (39) = **644 JPEGs**, each with a matching YOLO label file in `*/labels`.
- Export: Roboflow, project `onion-cnn-final-flpoy` v1, licence CC BY 4.0.
- Classes: `d` (dividing/mitotic, 2,913 boxes) and `nd` (non-dividing, 17,605 boxes). That is about 1:6 class imbalance.
- Labels: all 20,518 rows are valid axis-aligned bounding boxes. There are no polygons, no out-of-range coordinates, no empty label files and no orphan labels.
- `data.yaml`: points to `../test/images`, but **no test split exists**. Its relative paths also don't resolve from the yaml's own folder.

## Source groups (filename prefix, confirmed by appearance clustering)

| group | n | what it is |
|---|---|---|
| A | 276 | Orcein/carmine stain, mixed layouts: 162 A4 page renders (1653x2338, i.e. 200 dpi, with white page padding and a black circular eyepiece field), 98 square crops (~1653x1650), 16 other crops. Contains the square-cropped twins of the B frames. |
| B | 123 | The same frames as the A square crops, re-exported on A4 pages with an **"X1000" label** burned into the black corner. 121/123 have a near-identical A twin. |
| C | 86 | High-zoom crops on A4 pages, about 40% white padding, low contrast, soft focus. |
| D | 55 | Blue (methylene/toluidine-like) stain, landscape A4 pages, 20% white padding. Several have thick dark fibre/hair lines. |
| E | 15 | **Web/stock images**: 13 sizes, mixed stains, scale bars. **9 carry stock-agency watermarks.** |
| F | 89 | 2048x2048 phone-through-eyepiece shots with a burned-in caption ("ONION ROOT TIP SQUASH VIEW (X675) STAIN- 2% ACETIC-ORCEIN"), a black circular border (35% of pixels) and heavy debris. |

## Problems found

1. **Watermarked stock images (confirmed visually):**
   - Getty Images: E_9, E_10, E_11, E_12 ("Credit: Ed Reschke") and E_13 ("Credit: Clouds Hill Imaging Ltd.").
   - Shutterstock: E_16.
   - Science Photo Library: E_2, E_8, and **E_7 (in valid)**.

   The other six E images have unknown provenance. E_14 may not be an onion root tip at all.
2. **Massive duplication / leakage:**
   - 134 duplicate pairs: 121 A↔B twins, 11 A↔A near-duplicates and 2 rotated copies (A_215/A_216, A_218/A_219).
   - Only about 510 of the 644 images are unique fields.
   - **16 of the 39 validation images (41%) have a near-duplicate in train**, so current validation scores would be inflated.
3. **Burned-in text in 220 images:** 89 F captions, 123 B "X1000" labels, and 8 E images (5 Getty credit banners with ID strips, and 3 scale bars on E_3, E_4, E_5). A model can use these as source shortcuts, and a denoiser would learn to reproduce them.
4. **Heterogeneous geometry:** 77 image sizes. 396 images have ≥20% white A4 padding, and 189 have ≥20% black border; 461 have a circular field of view. Usable specimen area is 43–100% (median 64%). Magnification differs by source (X675, X1000, and high-zoom crops).
5. **Not truly "clean":**
   - Every image was re-encoded as JPEG at quality ~75 by Roboflow.
   - Many A, C and D frames look enlarged from lower-resolution sources, so they are softer than their pixel count suggests.
   - 7 blur candidates: A_126, A_358, A_367, A_371, A_373, D_23, D_25.
   - 17 low-contrast and 4 washed-out images.
6. **Artefacts:**
   - Eyepiece pointer drawn over the specimen: A_134.
   - Fibres/hairs: A_60, A_339, A_373, B_107, D_15–D_18, D_20, D_49, D_54.
   - Large debris clumps: A_263/B_52, A_265/B_53, many F images.
   - Source mismatches: A_384 (violet stain) and A_383.
7. **Stale derived data from before this audit:**
   - `data/splits/` has 644 images and **0 labels**. The split script looked for `.txt` files next to the images.
   - `data/processed/` was made with a centre crop to square. That discards specimen and does not transform the YOLO boxes.
   - Neither folder should be used. Nothing was deleted.

## Proposed STEP 3 preprocessing (awaiting approval)

1. **Exclude** all 15 E images (9 watermarked, 6 with unknown licence or provenance). Also exclude A_384 (foreign stain) and A_134 (drawn pointer). Keep a written exclusion list, and copy nothing over the originals.
2. **Deduplicate before splitting:**
   - Collapse each duplicate pair to one copy. For A↔B twins, keep the A square crop, which has no "X1000" text.
   - Drop the second copy of the A↔A and rotated pairs.
   - This leaves about 490 unique images. Twin labels should be compared during this step.
3. **Fresh, group-aware split, 70/15/15, with a real test set:**
   - Any surviving near-duplicate or consecutive-frame clusters (e.g. A_208–A_210) go to the same split.
   - Stratify by source group and by `d`/`nd` ratio.
   - Optionally hold out one whole source (e.g. F) as an out-of-distribution test.
4. **Geometry:**
   - Crop each image to its specimen bounding box, which removes the white A4 padding and the black bars.
   - Build a per-image field-of-view mask (circle) instead of cropping into the circle. Pixels outside the mask are zeroed and excluded from loss and noise statistics. This also removes the F caption and the B "X1000" label, which sit outside the circle.
   - No centre-crop to square and no upscaling.
   - Transform the YOLO boxes with every crop or resize, and drop boxes that fall outside the mask.
5. **Intensity:**
   - Keep RGB (every image is stained colour) and store float [0,1].
   - Don't apply per-image histogram equalisation, because that would break the link to photon counts.
   - Record each image's linear-light intensity (inverse sRGB gamma), because photon-noise simulation in STEP 4 must use linear intensities.
6. **Flag, don't drop:**
   - Keep the blur, low-contrast and artefact images, but tag them in a metadata CSV so they can be excluded from the "clean target" set or from evaluation if needed.
7. **Versioned output:** write everything to a new folder (e.g. `data/v2/`) with a manifest (source file, MD5, crop box, mask, split, flags), plus a corrected `data.yaml`.
