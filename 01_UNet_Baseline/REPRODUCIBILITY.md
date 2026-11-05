# Reproducibility and verification

Run every command from the project root.

## 1 · Verify the frozen results (no computation)

```
md5sum -c FREEZE_MANIFEST_phase1_phase2.md5                                          # onion: 134 OK
md5sum -c outputs/extension_medmnist_microscopy/FREEZE_MANIFEST_extension.md5        # extension: 131 OK
md5sum -c outputs/extension_medmnist_microscopy_poisson/FREEZE_MANIFEST_poisson.md5  # Poisson: 180 OK
md5sum -c 07_Reproducibility_and_Audit/FREEZE_MANIFEST_code_2026-11-05.md5           # code: 69 OK
```

The code manifest was written on 2026-11-05, after all test runs. For the onion study, the code state at test time is documented by the script MD5s recorded at the time (thesis Appendix A.4 and A.6; they match the current files) and by the file dates (4–6 October 2026). The extension and Poisson manifests already contained their scripts when they were frozen. In a clone, add `--ignore-missing` to the commands above (see README §8).

Re-derive every reported metric, confidence interval, pass/fail decision and minimum shot count from the frozen per-image CSVs. These scripts are read-only:

```
.venv-linux/bin/python audit/audit_recompute.py /tmp/audit_recompute.json   # extension + Poisson (about 20 min)
.venv-linux/bin/python audit/audit_onion_thesis.py                          # onion thesis tables vs CSVs
.venv-linux/bin/python audit/audit_tables.py                                # validation -> test minimum shot counts
```

The results of these checks are in `audit/SCIENTIFIC_AUDIT_2026-11-05.md` and `audit/final_verification/FINAL_VERIFICATION_REPORT.md`.

## 2 · Environments

See `08_Environment_and_Dependencies/README.md`:
- **Onion study:** Windows, Python 3.13, CPU (`requirements.txt`).
- **Extensions:** Linux, Python 3.10, CUDA (`.venv-linux/`, `requirements-linux-freeze.txt`).

## 3 · Re-running experiments (do NOT run into the frozen folders)

Every evaluation script refuses a second test run, and the frozen output folders must not be overwritten. To re-run an experiment, first copy the scripts and point their output folder somewhere new.

**Onion Phase 1** (Windows, `py -3.13`; see thesis Appendix A.5):
```
py -3.13 final_build_pool.py
py -3.13 final_train.py --shots 500        # likewise 1000, 2000, 5000
py -3.13 final_evaluate.py                 # validation; writes frozen_for_test.json
py -3.13 final_evaluate.py --split test --final-test   # once
```

**Onion Phase 2** (Appendix A.6):
```
py -3.13 orig_train.py --shots 500         # likewise 1000, 2000, 5000
py -3.13 orig_evaluate.py
py -3.13 orig_evaluate.py --split test --final-test   # once
```

**Microscopy extension** (Linux):
```
.venv-linux/bin/python ext_build_dataset.py
.venv-linux/bin/python ext_write_protocol.py          # refuses if the protocol already exists
.venv-linux/bin/python ext_build_pool.py --workers 30
.venv-linux/bin/python ext_train.py --target clean --shots 1000     # targets clean|dgi10k, shots 500|1000|2000|5000
.venv-linux/bin/python ext_evaluate.py --target clean               # validation; freezes checkpoints
.venv-linux/bin/python ext_evaluate.py --target clean --split test --final-test   # once
```

**Poisson noise** (Linux):
```
.venv-linux/bin/python ext_poisson_write_protocol.py
.venv-linux/bin/python ext_poisson_build_pool.py --workers 30
.venv-linux/bin/python ext_poisson_train.py --level p10 --shots 1000  # levels p05|p10|p30
.venv-linux/bin/python ext_poisson_evaluate.py --level p10
.venv-linux/bin/python ext_poisson_evaluate.py --level p10 --split test --final-test   # once
```

## 4 · Numerical notes

- **Data checks:** dataset files are MD5-checked by the scripts, and pattern and noise streams are seeded (see `EXPERIMENTS.md`).
- **GPU determinism:** runs use deterministic algorithms with cuDNN TF32 convolutions. Re-running the frozen checkpoints reproduces the frozen metrics **exactly only with the same batch layout** (the whole split, batches of 256). Other layouts differ by TF32 rounding (≤ 0.0008 per output pixel; ≤ 4e-5 SSIM).
- **Onion vs extension:** the onion study ran on CPU, and its SSIM came from scikit-image 0.26.0; the extensions used 0.25.2.

## 5 · Galleries

- **Main gallery:** open `final_comparison_gallery/index.html` in a browser. Rebuild with the three scripts in `final_comparison_gallery/scripts/`; see its README. The rebuild uses inference only.
- **Other galleries:** `audit/test40_gallery/` and `audit/visual_gallery/` have their own build scripts.
