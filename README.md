# MTP Ghost Imaging Project

**Team:** B23147 / B23139 / B23235
**Guide:** Amit Dattatraya Lad, CQST, IIT Mandi
**Project:** Computational Ghost Imaging via DiffTransFuse

## Directory Structure

```
MTP_Ghost_Imaging/
├── 01_UNet_Baseline/   Original LowShot Ghost Imaging project (FROZEN)
├── 02_DiffTransFuse/      E2–E4 DiffTransFuse experiments
├── 03_Final_MTP_Report/          Final MTP thesis/report (placeholder)
├── 04_Presentation/              Final presentation materials (placeholder)
└── README.md                     This file
```

## 01_UNet_Baseline

The original project covering onion cell ghost imaging (Phase 1–2), MedMNIST
microscopy extension, and Poisson noise sweep. Contains the E1 baseline
U-Net models, all datasets, and frozen evaluation outputs.

**Status:** FROZEN — do not modify any files. Three MD5 manifests verify
integrity:
- `FREEZE_MANIFEST_phase1_phase2.md5`
- `outputs/extension_medmnist_microscopy/FREEZE_MANIFEST_extension.md5`
- `outputs/extension_medmnist_microscopy_poisson/FREEZE_MANIFEST_poisson.md5`

**Note on `.venv-linux`:** The Linux virtual environment inside the frozen
project contains hardcoded absolute paths from the original location
(`/home/ss_students/Downloads/Onion-CNN-Final.v1-onion-original-v1.yolov8/`).
After the reorganization move, `pip`/`pip3` shebangs and `activate` scripts
will reference nonexistent paths. The Python symlink chain still resolves to
`/usr/bin/python3`, so the interpreter itself works, but pip and activation
are broken. **Do not modify the frozen venv.** To create a replacement:
```bash
cd 01_UNet_Baseline
python3 -m venv .venv-replacement --system-site-packages
source .venv-replacement/bin/activate
pip install -r requirements.txt
```

## 02_DiffTransFuse

DiffTransFuse experiments building on the frozen E1 baseline:
- **E2:** U-Net loss ablation (MSE+SSIM, MSE+edge, MSE+SSIM+edge)
- **E3:** Residual diffusion U-Net (epsilon-prediction, DDIM sampling)
- **E4:** TransFusionNet — frozen E3 encoder + bottleneck Transformer
  + CNN decoder with channel-attention skip fusion

Key results on 2000 held-out test images:
- E4a (MSE): best PSNR at all shot counts, 61.7x faster than E3
- E4b (MSE+SSIM): best SSIM at all shot counts

Reports: `reports/FINAL_TEST_REPORT.md`, `reports/E4_AUDIT_REPORT.md`

## 03_Final_MTP_Report

Placeholder for the final MTP thesis document.

## 04_Presentation

Placeholder for final presentation materials.
