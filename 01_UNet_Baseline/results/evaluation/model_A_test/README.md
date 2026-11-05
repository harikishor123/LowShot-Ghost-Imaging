# Model A (baseline U-Net): official held-out test evaluation

This is a simulation of photon-limited acquisition, using Poisson shot noise only. It is not real experimental low-light data, and it is not quantum or squeezed-light imaging.

- **Checkpoint:** `models/model_A_baseline_unet/final_restored_best.keras` (MD5 015f12b1…, epoch 60). Its weights are identical to `best.keras`.
- **Test set:** 74 images × 6 levels (P = 10, 50, 100, 250, 500, 1000) = 444 samples. The checks before evaluation are in `preflight.json`.
- **Metrics:** computed in linear light inside the FOV mask, as in training. PSNR uses data range 1. SSIM uses an 11×11 Gaussian window with σ = 1.5, averaged over the mask eroded by 5 px and over R, G, B.
- **Raw baseline:** the stored N/P, unclipped. A clipped [0, 1] variant is also reported.
- **Inference:** each image is zero-padded into a 1024×1024 canvas, run through the model, cropped back and multiplied by the mask. No per-image normalisation is applied.

## Files
| file | content |
|---|---|
| `test_table.md` | per-level table plus an overall row |
| `test_summary.json` | per-level and overall mean/std/median/95% CI for raw, raw_clipped and model; ΔPSNR, ΔSSIM, error reductions; per-group breakdown |
| `test_metrics.csv` | one row per sample (444): MSE, MAE, PSNR, SSIM for each variant, plus Δ values and flags |
| `preflight.json` | split isolation, data checksums, checkpoint and shape checks |
| `evaluation.log` | run log |
| `reconstructions/<id>__P<PPPP>.npz` | 16-bit sRGB reconstructions plus mask for the 8 representative test images at all 6 levels |
| `figures/recon_<id>.png` | clean / raw / U-Net / \|error\| on the same 256 px region at all 6 levels |
| `figures/recon_full_<id>.png` | full field of view at P = 10 and P = 100 |
| `figures/metrics_vs_photon_level_test.png` | PSNR, SSIM, MSE and MAE against P, with 95% CIs |
| `figures/training_curves.png` | training and validation curves (from the training history; no test data) |

## Run history
The first evaluation run ran out of memory during SSIM after about 50 samples. No metrics were written and no results were inspected. Two changes followed:
- inference uses a fixed 1024×1024 canvas, so TensorFlow does not retrace for each image size;
- SSIM is computed sequentially.

The model and the metric definitions were not changed. The second run evaluated all 444 samples once. The final table print then hit a Windows console encoding error, after every file had been written; that print has since been fixed.
