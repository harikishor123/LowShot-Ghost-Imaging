#!/usr/bin/env bash
# Build the GitHub Release assets from the frozen checkpoints (read-only on the project; writes only to dist/).
# PROPOSED ONLY; NOT RUN. Run from the project root only after explicit approval:
#     bash 07_Reproducibility_and_Audit/release_plan/make_release_assets.sh
# Each archive keeps the original project-relative paths, so extracting it in a clone restores the files
# exactly where the freeze manifests and frozen_for_test.json expect them.
set -euo pipefail
cd "$(dirname "$0")/../.."
P=07_Reproducibility_and_Audit/release_plan
md5sum -c "$P/MD5SUMS_checkpoints.txt" --quiet          # stop if any checkpoint differs from its frozen MD5
mkdir -p dist
TAR=(tar --sort=name --mtime='2026-11-05 00:00:00Z' --owner=0 --group=0 --numeric-owner -cf)   # reproducible archives
grep -E 'phase_final_'                 "$P/MD5SUMS_checkpoints.txt" | awk '{print $2}' > dist/onion.list
grep -E 'extension_medmnist_microscopy/' "$P/MD5SUMS_checkpoints.txt" | awk '{print $2}' > dist/extension.list
grep -E 'microscopy_poisson/'          "$P/MD5SUMS_checkpoints.txt" | awk '{print $2}' > dist/poisson.list
"${TAR[@]}" dist/mtp-gi-checkpoints-onion-v1.0.tar      -T dist/onion.list
"${TAR[@]}" dist/mtp-gi-checkpoints-extension-v1.0.tar  -T dist/extension.list
"${TAR[@]}" dist/mtp-gi-checkpoints-poisson-v1.0.tar    -T dist/poisson.list
# The processed MedMNIST+ objects (objects_64.npz) are deliberately NOT packaged: kept separate until their
# redistribution licence is verified (decision 2026-11-05).
cp "$P/SHA256SUMS_checkpoints.txt" "$P/MD5SUMS_checkpoints.txt" dist/
(cd dist && sha256sum ./*.tar > SHA256SUMS_archives.txt && cat SHA256SUMS_archives.txt)
wc -l dist/*.list
