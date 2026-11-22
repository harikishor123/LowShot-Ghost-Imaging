# 07 · Reproducibility and audit

| Item | Where |
|---|---|
| Frozen manifests (do not edit) | `FREEZE_MANIFEST_phase1_phase2.md5` (134), `outputs/extension_medmnist_microscopy/FREEZE_MANIFEST_extension.md5` (131), `outputs/extension_medmnist_microscopy_poisson/FREEZE_MANIFEST_poisson.md5` (180) |
| Code manifest (new, 2026-11-05) | `07_Reproducibility_and_Audit/FREEZE_MANIFEST_code_2026-11-05.md5`: 69 files (all root and `src/` scripts, `configs/`, `data.yaml`, `requirements.txt`, `data/v2/splits/`) |
| Scientific audit (all results re-derived) | `audit/SCIENTIFIC_AUDIT_2026-11-05.md`, scripts `audit/audit_recompute.py`, `audit/audit_onion_thesis.py`, `audit/audit_tables.py` |
| Final verification | `audit/final_verification/` |
| How to reproduce and verify | `REPRODUCIBILITY.md` (project root) |

Verify from the project root:
```
md5sum -c FREEZE_MANIFEST_phase1_phase2.md5
md5sum -c outputs/extension_medmnist_microscopy/FREEZE_MANIFEST_extension.md5
md5sum -c outputs/extension_medmnist_microscopy_poisson/FREEZE_MANIFEST_poisson.md5
md5sum -c 07_Reproducibility_and_Audit/FREEZE_MANIFEST_code_2026-11-05.md5
```
