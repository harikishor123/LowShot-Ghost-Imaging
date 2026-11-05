# 08 · Environment and dependencies

Two environments were used. Neither is needed to *view* results; both are needed to *re-run* code.

| Experiments | Environment | Record |
|---|---|---|
| Onion study (Phases 1, 2 and earlier phases) | Windows, Python 3.13.7, PyTorch 2.8.0 **CPU**, NumPy 2.2.6, SciPy 1.16.2, scikit-image 0.26.0, pandas 2.3.2, Matplotlib 3.10.6, Pillow 11.3.0 | `requirements.txt` (project root) |
| MedMNIST extension, Poisson experiments, audits and galleries | Linux (Ubuntu, glibc 2.35), Python 3.10.12, PyTorch 2.14.1+cu126 (CUDA 12.6, cuDNN 9.10.02) on an NVIDIA RTX A5000, NumPy 2.2.6, SciPy 1.15.3, scikit-image 0.25.2, pandas 2.3.2, Matplotlib 3.10.9, Pillow 12.3.0 | `08_Environment_and_Dependencies/requirements-linux-freeze.txt` (full `pip freeze`, 189 packages) |

Notes:
- The Linux virtual environment is `.venv-linux/` (project root). Run scripts with `.venv-linux/bin/python <script>` from the project root.
- `.venv/` (project root) is a **Windows** Python 3.14 environment created on another computer (`c:\Users\vishn\…`). It was not the experiment environment and does not run on Linux. It is listed in `CLEANUP_REPORT.md` for a decision; it was not removed.
- GPU runs used `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` and cuDNN's default TF32 convolutions. Results are reproduced bit-exactly only with the same batch layout; other layouts differ by TF32 rounding (≤ 0.0008 per output pixel, no effect on any reported decision).
- Different scikit-image versions (0.26.0 for onion, 0.25.2 for the extensions) computed SSIM in the two parts of the study.
