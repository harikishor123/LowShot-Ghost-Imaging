"""Read-only inventory of every project file: phase, purpose, manifest coverage and GitHub recommendation.

    .venv-linux/bin/python 07_Reproducibility_and_Audit/make_repo_inventory.py
Writes 07_Reproducibility_and_Audit/REPO_INVENTORY.csv (one row per file) and prints a summary.
Recommendations: COMMIT, RELEASE (GitHub Release / Zenodo asset, not in Git), EXCLUDE-DOWNLOAD (public source),
EXCLUDE-REGENERABLE (rebuilt by a script; hash recorded), EXCLUDE (environment, history or pending decision).
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {"onion": "FREEZE_MANIFEST_phase1_phase2.md5",
             "extension": "outputs/extension_medmnist_microscopy/FREEZE_MANIFEST_extension.md5",
             "poisson": "outputs/extension_medmnist_microscopy_poisson/FREEZE_MANIFEST_poisson.md5",
             "code": "07_Reproducibility_and_Audit/FREEZE_MANIFEST_code_2026-11-05.md5"}
covered = defaultdict(list)
for k, m in MANIFESTS.items():
    for line in (ROOT / m).read_text().splitlines():
        if line.strip():
            covered[re.sub(r"^[0-9a-f]{32} [ *]", "", line.strip())].append(k)

FINAL_OUT = ("outputs/phase_final_low_shot_to_10k_reference", "outputs/phase_final_original_target")
EARLY_OUT = ("outputs/phase1_single_image", "outputs/phase1b_dgi", "outputs/phase2a_dgi_dataset", "outputs/phase2b_unet",
             "outputs/phase3_strong_baselines", "outputs/phase4a_noisy_baselines")
EARLY_PY = {"run_single_image.py", "run_dgi_single_image.py", "run_dgi_dataset.py", "train_unet_gi_random.py",
            "train_model_D.py", "compare_models_AB.py", "run_measurement_curve.py", "run_strong_baselines.py",
            "phase4a_noise_setup.py", "run_phase4a.py", "train_unet_gi.py", "run_generalization_validation.py"}


def classify(p: str, size: int):
    """(phase, purpose, recommendation, note)."""
    name = p.rsplit("/", 1)[-1]
    top = p.split("/", 1)[0]
    if top in (".venv", ".venv-linux"):
        return "environment", "Python virtual environment", "EXCLUDE", "recreate from requirements files"
    if "__pycache__/" in p or p.endswith(".pyc"):
        return "environment", "Python bytecode cache (regenerated automatically)", "EXCLUDE", ""
    if top == "archive_review_required":
        return "history", "superseded pre-audit material", "EXCLUDE", "kept locally until thesis submission (your decision)"
    if p.startswith("data/extension_medmnist/raw/"):
        return "extension", "MedMNIST+ raw download", "EXCLUDE-DOWNLOAD", "Zenodo 10.5281/zenodo.10519652 (CC BY 4.0); MD5 in protocol.json"
    if p.startswith("data/extension_medmnist/processed/"):
        return "extension", "processed 64x64 luminance objects", "EXCLUDE-REGENERABLE", "ext_build_dataset.py; MD5 in dataset_report.json"
    if p.startswith("data/v2/splits/") or p.startswith("data/v2/metadata/"):
        return "onion", "onion split lists / metadata", "COMMIT" if size < 20e6 else "EXCLUDE-REGENERABLE", ""
    if p.startswith("data/"):
        return "onion", "derived onion images / simulations", "EXCLUDE-REGENERABLE", "src/build_v2_dataset.py from train/ and valid/"
    if top in ("train", "valid") or name in ("data.yaml", "README.roboflow.txt"):
        rec = "COMMIT" if name in ("data.yaml", "README.roboflow.txt") else "EXCLUDE-DOWNLOAD"
        return "onion", "raw Roboflow onion export (CC BY 4.0)", rec, "confirm public access to the Roboflow project before relying on download"
    if re.search(r"(pool_train[^/]*\.npy|selection_val\.npz)$", p) or (p.startswith(FINAL_OUT) and p.endswith((".npy", ".npz"))):
        return phase_of(p), "training pool / selection set", "EXCLUDE-REGENERABLE", "rebuilt by *_build_pool.py; MD5 in freeze manifest"
    if p.endswith(".pt"):
        if p.startswith(EARLY_OUT):
            return "preliminary", "exploratory checkpoint", "EXCLUDE", "not used for reported results"
        if name == "best.pt":
            return phase_of(p), "frozen evaluated checkpoint", "RELEASE", "MD5 in frozen_for_test.json; attach to a Release"
        return phase_of(p), "last-epoch checkpoint (not evaluated)", "EXCLUDE", "optional Release asset"
    if p.endswith(".keras"):
        return "preliminary", "earliest Keras Models A/B", "EXCLUDE", "history; optional Release asset"
    if p.endswith((".npz", ".npy")) and (p.startswith(EARLY_OUT) or top == "results"):
        return "preliminary", "exploratory reconstruction arrays", "EXCLUDE", "history; regenerable by the exploratory scripts"
    if size > 50e6:
        return phase_of(p), "large file", "EXCLUDE", "over GitHub's 50 MB warning size"
    if p.startswith(EARLY_OUT) or name in EARLY_PY and "/" not in p:
        ph = "preliminary"
        if name in ("train_unet_gi.py", "run_generalization_validation.py"):
            return ph, "exploratory script, imported by the final pipelines", "COMMIT", "dependency of final_*.py / orig_*.py"
        return ph, "exploratory phase output / script", "COMMIT", "label as preliminary"
    if top == "outputs":
        return phase_of(p), "frozen result / log / figure", "COMMIT", ""
    if top == "src":
        return "shared", "library module", "COMMIT", ""
    if top in ("results", "models", "configs"):
        return "preliminary", "earliest work / onion dataset audit", "COMMIT", "contains Windows user paths in JSON"
    if top == "thesis":
        return "thesis", "thesis, draft and merge material", "COMMIT", ""
    if top == "presentation":
        return "thesis", "slide decks", "COMMIT", "final deck not yet chosen"
    if top == "audit":
        return "audit", "audit, verification, galleries", "COMMIT", ""
    if top == "final_comparison_gallery":
        return "galleries", "final comparison gallery", "COMMIT", ""
    if re.match(r"0[1-8]_", top):
        return "docs", "guide folder / records", "COMMIT", ""
    if p.endswith(".py"):
        return phase_of(p), "experiment entry point", "COMMIT", ""
    return "docs", "project documentation / manifest", "COMMIT", ""


def phase_of(p: str) -> str:
    if "extension_medmnist_microscopy_poisson" in p or p.startswith("ext_poisson"):
        return "poisson"
    if "extension_medmnist" in p or p.startswith("ext_"):
        return "extension"
    if p.startswith(FINAL_OUT) or p.startswith(("final_", "orig_")):
        return "onion"
    return "other"


_wl = ROOT / "07_Reproducibility_and_Audit/EXCLUDED_PENDING_PERMISSION.txt"
WITHHELD = {l.strip() for l in _wl.read_text().splitlines() if l.strip() and not l.startswith("#")} if _wl.exists() else set()

_np = ROOT / "07_Reproducibility_and_Audit/EXCLUDED_FROM_PUBLIC_RELEASE.txt"
NOT_PUBLIC = {l.strip() for l in _np.read_text().splitlines() if l.strip() and not l.startswith("#")} if _np.exists() else set()

rows = []
for f in sorted(ROOT.rglob("*")):
    if not f.is_file():
        continue
    p = f.relative_to(ROOT).as_posix()
    if p.startswith((".venv", ".git/", "local_records/")):
        continue                      # counted below, not listed file by file
    size = f.stat().st_size
    ph, purpose, rec, note = classify(p, size)
    if p in WITHHELD:
        rec, note = "WITHHELD-PERMISSION", "excluded until permission is confirmed (EXCLUDED_PENDING_PERMISSION.md)"
    elif p in NOT_PUBLIC:
        rec, note = "NOT-FOR-PUBLIC-RELEASE", "kept out of the public repository (EXCLUDED_FROM_PUBLIC_RELEASE.md)"
    rows.append({"path": p, "size_bytes": size, "phase": ph, "purpose": purpose,
                 "freeze_manifest": "+".join(covered.get(p, [])), "github": rec, "note": note})
venv = {d: (sum(1 for x in (ROOT / d).rglob("*") if x.is_file()), sum(x.stat().st_size for x in (ROOT / d).rglob("*") if x.is_file()))
        for d in (".venv", ".venv-linux")}
for d, (n, s) in venv.items():
    rows.append({"path": d + "/", "size_bytes": s, "phase": "environment", "purpose": f"virtual environment ({n} files)",
                 "freeze_manifest": "", "github": "EXCLUDE", "note": "recreate from requirements files"})

out = ROOT / "07_Reproducibility_and_Audit/REPO_INVENTORY.csv"
with out.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

agg = defaultdict(lambda: [0, 0])
for r in rows:
    agg[r["github"]][0] += 1
    agg[r["github"]][1] += r["size_bytes"]
print(f"wrote {out.relative_to(ROOT)}: {len(rows)} rows")
for k, (n, s) in sorted(agg.items()):
    print(f"  {k:20s} {n:6d} files {s / 1e9:8.2f} GB")
big = [r for r in rows if r["github"] == "COMMIT" and r["size_bytes"] > 20e6]
print("  COMMIT files > 20 MB:", len(big), [r["path"] for r in big])
miss = [k for k in covered if not (ROOT / k).exists()]
print("  manifest entries missing on disk:", len(miss))
excl = [r["path"] for r in rows if r["freeze_manifest"] and r["github"] != "COMMIT"]
print("  manifest-covered files not committed (need md5sum --ignore-missing on a clone):", len(excl))
