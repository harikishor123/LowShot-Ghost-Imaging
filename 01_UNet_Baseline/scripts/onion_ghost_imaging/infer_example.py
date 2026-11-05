#!/usr/bin/env python3
"""Run one frozen U-Net checkpoint on MedMNIST+ extension images (inference only, read-only).

    python infer_example.py --checkpoint outputs/extension_medmnist_microscopy/models/target_clean/model_1000/checkpoints/best.pt \
                            --shots 1000 --n-images 4 --out demo_out/

Simulates the noise-free GI acquisition of each image with illumination sequence --seed (Bernoulli(0.5) patterns,
calibrated DGI, the same functions as ext_evaluate.py), feeds DGI-N to the checkpoint, and prints SSIM / Pearson r / PSNR
of DGI-N, U-Net-N and DGI-10000 against the clean object. Optionally saves a comparison PNG.

* Uses the validation split by default. The frozen test sets were evaluated once; --split test needs --allow-test.
* Writes nothing except the optional --out folder, which must be outside outputs/.
* Needs data/extension_medmnist/processed/objects_64.npz (rebuild with ext_build_dataset.py) and a checkpoint
  (not distributed with the repository; see README section 5). The checkpoint must match --shots (model_1000 -> --shots 1000).
* For Poisson-noise checkpoints the input must be noisy DGI; use ext_poisson_common.noisy_dgi_readouts instead.
* Metrics agree with the frozen per_image_results.csv up to TF32 / batch-layout rounding (REPRODUCIBILITY.md section 4).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ext_common import OBJECTS_NPZ, SHOTS, load_objects  # noqa: E402
from final_build_pool import dgi_readouts  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.unet_gi import UNetGI  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shots", type=int, required=True, choices=[500, 1000, 2000, 5000])
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--allow-test", action="store_true", help="required for --split test")
    ap.add_argument("--image-ids", nargs="*", help="image ids (default: the first --n-images of the split)")
    ap.add_argument("--n-images", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026, help="illumination sequence (evaluation seeds 2026-2029)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", help="optional folder for a comparison PNG (outside outputs/)")
    a = ap.parse_args()
    if a.split == "test" and not a.allow_test:
        sys.exit("refusing to load test images without --allow-test (the frozen test sets were evaluated once)")
    if a.out and (ROOT / "outputs") in [*(Path(a.out).resolve().parents), Path(a.out).resolve()]:
        sys.exit("--out must be outside outputs/ (frozen experiment folders)")

    for need, how in ((OBJECTS_NPZ, "the processed MedMNIST+ objects are not distributed with the repository; "
                                    "see README section 4 (download the raw MedMNIST+ files and rebuild with ext_build_dataset.py)"),
                      (Path(a.checkpoint), "checkpoints are not distributed with the repository; see README section 5")):
        if not Path(need).exists():
            sys.exit(f"missing {need}\n  {how}")
    d = load_objects((a.split,))
    ids = list(d["image_id"])
    pick = [ids.index(i) for i in a.image_ids] if a.image_ids else list(range(min(a.n_images, len(ids))))
    clean = d["object"][pick].astype(np.float64)
    names = [ids[i] for i in pick]

    ck = torch.load(a.checkpoint, map_location=a.device, weights_only=False)
    model = UNetGI(base=ck["config"]["base"]).to(a.device).eval()
    model.load_state_dict(ck["model"])

    acq = dgi_readouts(clean, a.seed)                       # (M, 5, 64, 64) DGI at 500..10000 shots, clipped
    x = acq[:, SHOTS.index(a.shots)][:, None].astype(np.float32)
    with torch.no_grad():
        unet = np.clip(model(torch.from_numpy(x).to(a.device)).cpu().numpy()[:, 0], 0, 1)
    rows = {f"DGI-{a.shots}": acq[:, SHOTS.index(a.shots)], f"U-Net-{a.shots}": unet, "DGI-10000": acq[:, SHOTS.index(10000)]}

    print(f"checkpoint {a.checkpoint} (best epoch {ck.get('epoch')}), split {a.split}, sequence {a.seed}, device {a.device}")
    print(f"{'image':22s} {'method':12s} {'SSIM':>7s} {'r':>7s} {'PSNR':>7s}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for j, name in enumerate(names):
            for lab, arr in rows.items():
                m = evaluate(clean[j], arr[j].astype(np.float64))
                print(f"{name:22s} {lab:12s} {m['SSIM']:7.4f} {m['Pearson_r']:7.4f} {m['PSNR']:7.2f}")

    if a.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        cols = {"original": clean, **rows}
        fig, ax = plt.subplots(len(names), len(cols), figsize=(2.2 * len(cols), 2.3 * len(names)), squeeze=False)
        for j in range(len(names)):
            for c, (lab, arr) in enumerate(cols.items()):
                ax[j, c].imshow(arr[j], cmap="gray", vmin=0, vmax=1)
                ax[j, c].set_xticks([]); ax[j, c].set_yticks([])
                if j == 0:
                    ax[j, c].set_title(lab, fontsize=9)
            ax[j, 0].set_ylabel(names[j], fontsize=7)
        fig.tight_layout()
        p = out / f"infer_{a.split}_N{a.shots}_seed{a.seed}.png"
        fig.savefig(p, dpi=120)
        print("saved", p)


if __name__ == "__main__":
    main()
