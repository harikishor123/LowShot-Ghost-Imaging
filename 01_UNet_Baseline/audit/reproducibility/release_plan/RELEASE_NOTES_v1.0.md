# MTP thesis: frozen checkpoints v1.0

These are the frozen U-Net checkpoints used for the reported results of "AI-DRIVEN Reconstruction of Quantum-Enhanced Low-Photon Bio-Imaging Datasets" (computational study of measurement-shot reduction in simulated differential ghost imaging).

- **Contents:** 28 `best.pt` files, from four experiments:
  - onion Phase 1 (target: 10,000-shot DGI);
  - onion Phase 2 (target: clean object);
  - MedMNIST+ extension (BloodMNIST + PathMNIST, two targets);
  - simulated Poisson bucket noise at 5 / 10 / 30%.
- **Selection:** every checkpoint was selected on validation data and frozen (MD5 recorded) before the single test evaluation. The hashes match `frozen_for_test.json` in the repository.
- **Architecture:** `src/unet_gi.py` (`UNetGI`, base 32, 1,925,025 parameters). Load with `torch.load(path, weights_only=False)` and then `UNetGI(base=ck["config"]["base"]).load_state_dict(ck["model"])`; see `infer_example.py`.
- **Extraction:** extract each archive in the repository root; the paths are preserved. Then verify with `sha256sum -c SHA256SUMS_checkpoints.txt`.

**Scope.** All measurements are simulated: noise-free, or with a simplified relative Poisson noise model. No hardware, photon-dose, biological or quantum claim is made. The 10,000-shot DGI is a study-specific benchmark, not ground truth.

**Data credits.**
- **Onion images:** Roboflow export "Onion-CNN-Final" v1, marked CC BY 4.0.
- **Blood and tissue images:** MedMNIST+ (Yang et al., *Sci. Data* 10, 41, 2023; Zenodo 10.5281/zenodo.10519652, CC BY 4.0).
  - BloodMNIST is derived from Acevedo et al., *Data Brief* 30, 105474 (2020).
  - PathMNIST is derived from Kather et al., *PLOS Med.* 16, e1002730 (2019).

**Licence of these files:** [to be decided; see `07_Reproducibility_and_Audit/LICENSING_RECOMMENDATION.md`]
