# GitHub Release plan: frozen checkpoints (v1.0)

**Status: PROPOSED PLAN ONLY (decision 2026-11-05).** No archive has been built and no Release published. Build and publish only after your explicit approval. The Release would belong to the private repository and so be visible only to its collaborators.

## Why a Release and not Git

- **Checkpoints:** the 28 evaluated checkpoints total 0.65 GB, 23 MB each.
- **Repository size:** committing them would more than double the repository (0.57 GB) and lock them into the Git history for good.
- **Release assets:** these sit outside the history, and each may be up to 2 GB.
- **Not included:** the 28 `last.pt` files (not evaluated) and the exploratory checkpoints.

## Assets

| Asset (built by `make_release_assets.sh`) | Contents | Size |
|---|---|---|
| `mtp-gi-checkpoints-onion-v1.0.tar` | onion Phase 1 (AI-*N*) and Phase 2 (AI-original-*N*), 8 × `best.pt` | 185 MB |
| `mtp-gi-checkpoints-extension-v1.0.tar` | extension, clean target (AI-original-*N*) and 10k-DGI target (AI-*N*), 8 × `best.pt` | 185 MB |
| `mtp-gi-checkpoints-poisson-v1.0.tar` | Poisson 5 / 10 / 30%, AI-original-*N*, 12 × `best.pt` | 278 MB |
| ~~`mtp-gi-medmnist-objects64-v1.0.tar`~~ **not part of the Release**: the processed dataset is kept separate until its redistribution licence is verified | `data/extension_medmnist/processed/objects_64.npz`: the 10,000 selected MedMNIST+ images converted to luminance (CC BY 4.0 derivative; attribution in the release notes) | 229 MB |
| `SHA256SUMS_checkpoints.txt`, `MD5SUMS_checkpoints.txt`, `SHA256SUMS_archives.txt` | per-file and per-archive hashes | — |

Each archive keeps the project-relative paths. After `tar -xf …` in the clone's root, every file is where `frozen_for_test.json` and the freeze manifests expect it.

**About `objects_64.npz` (deferred):** without it, the extension can only be used by rebuilding the objects with `ext_build_dataset.py`. That script also rewrites `outputs/extension_medmnist_microscopy/dataset/*`, which is covered by the extension freeze manifest (the rebuild should be identical, and `md5sum -c` would confirm it).

## The 28 checkpoints

Each MD5 below was checked against the value recorded in the frozen `frozen_for_test.json` before the test run. Full SHA-256 values are in `SHA256SUMS_checkpoints.txt`; machine-readable details are in `release_assets.json`.

| Group | Model | Path | Best epoch | MD5 | SHA-256 (start) |
|---|---|---|---|---|---|
| onion-phase1 | AI-500 | `outputs/phase_final_low_shot_to_10k_reference/models/model_500/checkpoints/best.pt` | 80 | `ec72468607dc43b98ae2b1899fccd597` | `3bd76a2005abe317…` |
| onion-phase1 | AI-1000 | `outputs/phase_final_low_shot_to_10k_reference/models/model_1000/checkpoints/best.pt` | 79 | `26efe9ef2a69cd813b16cc65aedbaa60` | `ced61b75bca3fef9…` |
| onion-phase1 | AI-2000 | `outputs/phase_final_low_shot_to_10k_reference/models/model_2000/checkpoints/best.pt` | 80 | `dc2ff919034623c4b92ea0a6fcf6fffe` | `aa43cc9ce0354dbc…` |
| onion-phase1 | AI-5000 | `outputs/phase_final_low_shot_to_10k_reference/models/model_5000/checkpoints/best.pt` | 78 | `ac0d35c0aa86747012118411b94eb535` | `3c884b93eb2b4aa4…` |
| onion-phase2 | AI-original-500 | `outputs/phase_final_original_target/models/model_500/checkpoints/best.pt` | 80 | `2af10174c4b8d267abccc9aacbfc0038` | `1c76521b434cbf74…` |
| onion-phase2 | AI-original-1000 | `outputs/phase_final_original_target/models/model_1000/checkpoints/best.pt` | 80 | `1eec40bff638b0169ef9a0d1ec7bfd9a` | `d62a34af117c0664…` |
| onion-phase2 | AI-original-2000 | `outputs/phase_final_original_target/models/model_2000/checkpoints/best.pt` | 80 | `3aeda148fefc79a914b5ef6dfd0558a2` | `cee13f590e6ede2e…` |
| onion-phase2 | AI-original-5000 | `outputs/phase_final_original_target/models/model_5000/checkpoints/best.pt` | 79 | `8f0fc0d5e301a03f1615f19987f2b074` | `0a948fa5b7afec5d…` |
| extension-clean | AI-original-500 | `outputs/extension_medmnist_microscopy/models/target_clean/model_500/checkpoints/best.pt` | 79 | `14292ff849960596941c8fe50a1c7b99` | `e3c1468a55b6a3ed…` |
| extension-clean | AI-original-1000 | `outputs/extension_medmnist_microscopy/models/target_clean/model_1000/checkpoints/best.pt` | 80 | `25784148b599d3a967a2042e57d3eef8` | `49d8a5dcfa258908…` |
| extension-clean | AI-original-2000 | `outputs/extension_medmnist_microscopy/models/target_clean/model_2000/checkpoints/best.pt` | 59 | `bf4a264dd6df34a9936e6367f614d57c` | `226d15356d2ff31d…` |
| extension-clean | AI-original-5000 | `outputs/extension_medmnist_microscopy/models/target_clean/model_5000/checkpoints/best.pt` | 79 | `103f5c4eda854fc27f9a26fb62967612` | `0cded24ba38ae3e2…` |
| extension-dgi10k | AI-500 | `outputs/extension_medmnist_microscopy/models/target_dgi10k/model_500/checkpoints/best.pt` | 71 | `35e24d52246c038ea35e2d39126e7ee3` | `e96a27d599f93c08…` |
| extension-dgi10k | AI-1000 | `outputs/extension_medmnist_microscopy/models/target_dgi10k/model_1000/checkpoints/best.pt` | 80 | `4c898c20dd9571a1d12c25e750ad0135` | `b738d83ea3f81e1a…` |
| extension-dgi10k | AI-2000 | `outputs/extension_medmnist_microscopy/models/target_dgi10k/model_2000/checkpoints/best.pt` | 77 | `d236a647450f52c3c06c1a5c64253990` | `55d2d23e713dd20a…` |
| extension-dgi10k | AI-5000 | `outputs/extension_medmnist_microscopy/models/target_dgi10k/model_5000/checkpoints/best.pt` | 79 | `383805e9ada96289ce0075feaa5e1f3f` | `e979d5a726d7ae28…` |
| poisson-p05 | AI-original-500 | `outputs/extension_medmnist_microscopy_poisson/models/level_p05/model_500/checkpoints/best.pt` | 79 | `93b3363387f2cdb4f2dfc947f572ab7a` | `91e05c0bccbc6f91…` |
| poisson-p05 | AI-original-1000 | `outputs/extension_medmnist_microscopy_poisson/models/level_p05/model_1000/checkpoints/best.pt` | 61 | `cf0ed7ed7d421cf1fae59cf7e6362d88` | `dc863256900a19b0…` |
| poisson-p05 | AI-original-2000 | `outputs/extension_medmnist_microscopy_poisson/models/level_p05/model_2000/checkpoints/best.pt` | 68 | `0eed438062db1597fe6fc70ac36a9002` | `978732a7b3aa037f…` |
| poisson-p05 | AI-original-5000 | `outputs/extension_medmnist_microscopy_poisson/models/level_p05/model_5000/checkpoints/best.pt` | 76 | `af61fb9a8bb400dcf488e5c56ca4fab4` | `b52d78698e705df8…` |
| poisson-p10 | AI-original-500 | `outputs/extension_medmnist_microscopy_poisson/models/level_p10/model_500/checkpoints/best.pt` | 74 | `b418c2a900a9c836563ca10dbaeb60b2` | `fd0d0a86d0503528…` |
| poisson-p10 | AI-original-1000 | `outputs/extension_medmnist_microscopy_poisson/models/level_p10/model_1000/checkpoints/best.pt` | 80 | `47441495a1a4a88222054bc6c263def7` | `7a1fbefcd7afc666…` |
| poisson-p10 | AI-original-2000 | `outputs/extension_medmnist_microscopy_poisson/models/level_p10/model_2000/checkpoints/best.pt` | 79 | `e029d735f17f558f579865fed5203a09` | `5c131f7350796e6a…` |
| poisson-p10 | AI-original-5000 | `outputs/extension_medmnist_microscopy_poisson/models/level_p10/model_5000/checkpoints/best.pt` | 68 | `47ce62e27ad9acb7f49a3565407fd593` | `9aa7d00ef3bbaef0…` |
| poisson-p30 | AI-original-500 | `outputs/extension_medmnist_microscopy_poisson/models/level_p30/model_500/checkpoints/best.pt` | 79 | `07ad4d2ef18ede395896b432ccbbd8e8` | `6d391b7c7764b617…` |
| poisson-p30 | AI-original-1000 | `outputs/extension_medmnist_microscopy_poisson/models/level_p30/model_1000/checkpoints/best.pt` | 61 | `c510add10a66b7f4bdf104ae38fd4889` | `666220a9142897b5…` |
| poisson-p30 | AI-original-2000 | `outputs/extension_medmnist_microscopy_poisson/models/level_p30/model_2000/checkpoints/best.pt` | 59 | `329fd9c2df76f0f3c42306c8c5148b62` | `b31c6a1ef5c295dc…` |
| poisson-p30 | AI-original-5000 | `outputs/extension_medmnist_microscopy_poisson/models/level_p30/model_5000/checkpoints/best.pt` | 69 | `2747b0dfe3392b34e23032d76d1b9a0d` | `63d3f6ee8b2c6d0b…` |
| extension-data | objects_64 | `data/extension_medmnist/processed/objects_64.npz` |  | `73e8ce929c5a7c67111e028032e528c1` | `a749310bd2e1161e…` |

The last row (`objects_64.npz`) is listed for reference only; it is **not** part of the proposed Release.

## Steps (after approval)

1. **Build the archives (after approval):** `bash 07_Reproducibility_and_Audit/release_plan/make_release_assets.sh`. The script first checks every checkpoint's MD5 and writes only to `dist/`, which is ignored by Git.
2. **Push the repository** (see `07_Reproducibility_and_Audit/git_plan/GIT_PLAN.md`).
3. **Create the Release:** on GitHub, *Releases → Draft a new release*. Tag `v1.0-thesis`, title "MTP thesis: frozen checkpoints v1.0". Paste the release notes below and upload the files from `dist/`. Or use `gh release create v1.0-thesis dist/*.tar dist/*SUMS*.txt --notes-file 07_Reproducibility_and_Audit/release_plan/RELEASE_NOTES_v1.0.md`.
4. **Point the README at it:** replace `<RELEASE-URL>` in `README.md` with the release URL.
5. **Optional long-term archive:** also deposit the archives on Zenodo for a DOI.

## Verifying a download (in a clone)

```
sha256sum -c SHA256SUMS_archives.txt
tar -xf mtp-gi-checkpoints-onion-v1.0.tar        # likewise the other archives
sha256sum -c SHA256SUMS_checkpoints.txt
md5sum -c FREEZE_MANIFEST_phase1_phase2.md5 --ignore-missing
```
