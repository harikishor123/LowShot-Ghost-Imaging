# Final pre-commit verification procedure

> **Update 2026-11-05:** the authors decided to make the GitHub repository **public**. The permission-pending exclusions stay in force, and a further 16 files are kept out of the public repository (`07_Reproducibility_and_Audit/EXCLUDED_FROM_PUBLIC_RELEASE.md`). The first commit contains 3,066 files.

Do these steps **in order**, after your approval of the Git initialisation and before the first commit. Every check is automated in `precommit_verify.sh` (read-only; it writes only to a temporary folder). Stop at the first FAIL.

## Step 1 · Decisions recorded

Confirm in writing (the checklist in `FINAL_GITHUB_READINESS_REPORT.md` §H):
- the repository stays **private**;
- no licence file is added yet (or the one you approved);
- the "PERMISSION PENDING" block of `gitignore_PROPOSED.txt` is still in place;
- no Release is created.

## Step 2 · Verify the working copy and simulate the commit (before `git init`)

```bash
cd /home/ss_students/Downloads/Onion-CNN-Final.v1-onion-original-v1.yolov8
FULL=1 bash 07_Reproducibility_and_Audit/git_plan/precommit_verify.sh
```

Section 1 of the script checks the working copy; every line must be PASS:

| Check | Expected |
|---|---|
| Onion / extension / Poisson / code manifests | 134 / 131 / 180 / 69 OK |
| `PROTECTED_ORIGINALS.md5` (original thesis + 3 original decks) | 4 / 4 OK |
| `release_plan/MD5SUMS_checkpoints.txt` | 28 / 28 OK |

Section 2 checks the files that would be committed:
- none of the 187 withheld files (`EXCLUDED_PENDING_PERMISSION.txt`);
- no `.pt`, `.keras` or `.pptx` file;
- no pools, datasets, environments, archive or caches;
- in the onion folders, only the approved metric plots;
- no file over 50 MB;
- no secrets.

Section 3 copies only those files to a temporary "clone" and runs:
- `md5sum -c --ignore-missing` on the four manifests (no failures);
- compiling every Python file;
- `verify_claims.py` (29 of 29);
- `audit_onion_thesis.py` (39 rows; only the two known checker artefacts);
- `check_merged_thesis.py` (withheld figure links reported, not failed);
- `infer_example.py`, which must stop with a clear "missing" message;
- with `FULL=1`, `audit_recompute.py`, whose result must be identical to the stored one.

Section 4 re-checks that nothing in the project changed. The script must end with `RESULT: ALL CHECKS PASSED`.

## Step 3 · Initialise and stage (only after explicit approval)

```bash
git init -b main
cp 07_Reproducibility_and_Audit/gitignore_PROPOSED.txt .gitignore
git add -A
```

## Step 4 · Verify the real staging area

```bash
bash 07_Reproducibility_and_Audit/git_plan/precommit_verify.sh       # now reads `git diff --cached`
git diff --cached --name-only | sort | diff - 07_Reproducibility_and_Audit/git_plan/FIRST_COMMIT_FILELIST.txt
git diff --cached --stat | tail -1
git status --short | grep -v '^A ' || true                           # nothing should be unstaged or untracked-and-unignored
```

- **File list:** the only differences from `FIRST_COMMIT_FILELIST.txt` should be `.gitignore`, plus any file you deliberately changed after the list was made. Investigate anything else.
- **Content:** open `git diff --cached -- README.md FINAL_GITHUB_READINESS_REPORT.md` and skim a few staged files (for example `git show :thesis/MTP_Thesis_Final_with_extension.md | head`).
- **Check for images:** `git diff --cached --name-only | grep -iE '\.(png|jpe?g|pptx)$' | grep -E '^(outputs/phase|results/|thesis/figures|presentation/)'` must print only metric plots.

## Step 5 · Commit (only after Step 4 passes)

Use the message in `GIT_PLAN.md` §2. Then run `git log --stat -1 | tail -3` and keep the output with the review records.

## Step 6 · Push (separate approval)

Push only after the private repository exists with the settings in `PRIVATE_REPO_CONFIG.md`:
```bash
git remote add origin <PRIVATE-REPO-URL>
git push -u origin main
```

Then clone into a fresh folder and run `bash 07_Reproducibility_and_Audit/git_plan/precommit_verify.sh` there. Its Section 1 will report missing files, because the checkpoints and withheld files are not in Git. Run Section 3's commands by hand instead: `md5sum -c --ignore-missing …`, `verify_claims.py`, `check_merged_thesis.py`.

**If something wrong was committed but not pushed:** `git reset --soft HEAD~1`, fix, and repeat Step 4. **If it was already pushed,** tell your collaborators immediately: removing material from the history needs a rewrite, and copies may already exist.
