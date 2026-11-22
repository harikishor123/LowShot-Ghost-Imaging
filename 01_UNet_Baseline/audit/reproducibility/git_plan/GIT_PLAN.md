# Proposed Git initialisation and first commit (for your review)

> **Update 2026-11-05:** the authors decided to make the GitHub repository **public**. The permission-pending exclusions stay in force, and a further 16 files are kept out of the public repository (`07_Reproducibility_and_Audit/EXCLUDED_FROM_PUBLIC_RELEASE.md`). The first commit contains 3,066 files.

**Status: PROPOSAL ONLY. Not run; awaiting your explicit approval** (decision 2026-11-05: no `git init`, no staging, no commit, no push until you approve). There is no `.git` folder in the project. The repository is to be **private** (`PRIVATE_REPO_CONFIG.md`).

## 1 · What the first commit would contain

- **File list:** `FIRST_COMMIT_FILELIST.txt`, **3,066 files, 392 MB, largest file 25.5 MB** (after the public-release exclusions). No file is over GitHub's 50 MB warning size, so Git LFS is not needed.
- **How the list was made:** `git add --dry-run -A` with the proposed ignore rules (`../gitignore_PROPOSED.txt`), run from a throw-away Git directory outside the project. It matches the COMMIT rows of `REPO_INVENTORY.csv` exactly.
- **Folder summary:** `first_commit_summary.md`.

**Withheld pending permission** (`../EXCLUDED_PENDING_PERMISSION.md`):
- all 186 files showing onion micrographs, including all slide decks;
- the MedMNIST-derived array `audit/final_verification/exact_test_dgi_six_images.npz`.

The 60 metric plots in the onion folders stay.

**Manifest coverage.** Committed: 408 of the 503 distinct files in the four freeze manifests. Not in Git:
- 28 evaluated checkpoints (proposed Release, not yet created);
- 20 onion figures (withheld);
- 47 pools, datasets and unevaluated `last.pt` files.

A clone therefore uses `md5sum -c --ignore-missing`.

**Also excluded (by `.gitignore`):**
- virtual environments, `__pycache__`;
- `archive_review_required/`;
- raw onion images (`/train/`, `/valid/`) and derived `data/v2/*`, except splits and metadata;
- MedMNIST raw and processed files;
- all training pools and selection sets (62 GB);
- all `.pt` and `.keras` files;
- exploratory `.npz`/`.npy` arrays;
- `results/audit/contact_sheet.png` (50.6 MB);
- `dist/` and `demo_out/`.

**Verification before committing:** `PRECOMMIT_VERIFICATION.md` (automated in `precommit_verify.sh`).

## 2 · Commands (after approval)

```bash
cd /home/ss_students/Downloads/Onion-CNN-Final.v1-onion-original-v1.yolov8

# 0. integrity before
md5sum -c FREEZE_MANIFEST_phase1_phase2.md5 --quiet && \
md5sum -c outputs/extension_medmnist_microscopy/FREEZE_MANIFEST_extension.md5 --quiet && \
md5sum -c outputs/extension_medmnist_microscopy_poisson/FREEZE_MANIFEST_poisson.md5 --quiet && \
md5sum -c 07_Reproducibility_and_Audit/FREEZE_MANIFEST_code_2026-11-05.md5 --quiet && echo ALL FROZEN FILES OK

# 1. initialise (branch main) and activate the reviewed ignore rules
git init -b main
cp 07_Reproducibility_and_Audit/gitignore_PROPOSED.txt .gitignore
#    (no LICENSE yet: the licence remains a proposal until ownership and permissions are confirmed)

# 2. stage and REVIEW before committing
git add -A
git status --short | wc -l                      # expect 3,066 (including .gitignore)
git diff --cached --name-only | sort > /tmp/staged.txt
diff /tmp/staged.txt 07_Reproducibility_and_Audit/git_plan/FIRST_COMMIT_FILELIST.txt   # only the new files should differ
git diff --cached --stat | tail -1
find . -path ./.git -prune -o -type f -size +50M -print | git check-ignore --stdin -v --non-matching | grep '^::'   # must print nothing

# 3. commit (only after the review above)
git commit -m "Initial commit: LowShot Ghost Imaging research project"   # only after explicit approval

# 4. push: needs your GitHub repository URL; not part of this plan until you create the repository
# git remote add origin https://github.com/harikishor123/LowShot-Ghost-Imaging.git
# git push -u origin main
```

**Commit message (proposed by the authors, 2026-11-05):**
```
Initial commit: LowShot Ghost Imaging research project
```
No `Co-Authored-By` line is added unless authorship is explicitly confirmed.

## 3 · After the push

1. Create the Release (`../release_plan/RELEASE_PLAN.md`) and put its URL into README §5.
2. Clone into a fresh folder and run the README §8 commands with `--ignore-missing`. Then extract the Release archives and run them again; every checkpoint must pass.
3. Optionally tag the commit: `git tag -a v1.0-thesis -m "thesis submission state"` and `git push --tags`.
