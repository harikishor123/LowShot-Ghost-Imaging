#!/usr/bin/env bash
# Pre-commit verification (read-only on the project). Run from anywhere:
#     bash 07_Reproducibility_and_Audit/git_plan/precommit_verify.sh            # quick (skips the 10-20 min recomputation)
#     FULL=1 bash 07_Reproducibility_and_Audit/git_plan/precommit_verify.sh     # also audit_recompute.py in the clone
# Before `git init` it SIMULATES staging with the proposed ignore rules (a throw-away Git directory in a temp folder,
# never inside the project). After `git init` + `git add -A` it checks the real staging area instead.
# Exit code 0 only if every check passes. Writes only to a temporary folder (removed at the end unless KEEP=1).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
PY="${PY:-$ROOT/.venv-linux/bin/python}"
A=07_Reproducibility_and_Audit
TMP="$(mktemp -d)"; [ "${KEEP:-0}" = 1 ] || trap 'rm -rf "$TMP"' EXIT
fail=0; ok(){ echo "PASS  $*"; }; bad(){ echo "FAIL  $*"; fail=1; }

echo "== 1 · frozen files and protected originals (working copy)"
for m in FREEZE_MANIFEST_phase1_phase2.md5 outputs/extension_medmnist_microscopy/FREEZE_MANIFEST_extension.md5 \
         outputs/extension_medmnist_microscopy_poisson/FREEZE_MANIFEST_poisson.md5 $A/FREEZE_MANIFEST_code_2026-11-05.md5 \
         $A/PROTECTED_ORIGINALS.md5 $A/release_plan/MD5SUMS_checkpoints.txt; do
  n=$(grep -c . "$m"); r=$(md5sum -c "$m" 2>/dev/null | grep -c ': OK$')
  [ "$r" = "$n" ] && ok "$m: $r/$n" || bad "$m: $r/$n"
done

r=$("$PY" audit/audit_onion_thesis.py 2>&1 | grep 'rows checked'); [ "$r" = "onion thesis rows checked: 39, mismatching rows: 2" ] \
  && ok "working copy: audit_onion_thesis.py (39 rows; the 2 known checker artefacts)" || bad "working copy: audit_onion_thesis.py: $r"
"$PY" thesis/merge_proposal/check_merged_thesis.py >"$TMP/cmt0.txt" 2>&1 && ok "working copy: check_merged_thesis.py ($(grep -c '^PASS' "$TMP/cmt0.txt") checks)" || bad "working copy: check_merged_thesis.py: $(grep FAIL "$TMP/cmt0.txt")"

echo "== 2 · what would be committed"
if [ -d .git ]; then
  git diff --cached --name-only | sort > "$TMP/staged.txt"; echo "INFO  using the real staging area"
else
  git init -q --bare "$TMP/sim.git"
  git --git-dir="$TMP/sim.git" --work-tree="$ROOT" -c core.excludesFile="$ROOT/$A/gitignore_PROPOSED.txt" add --dry-run -A \
    | sed -E "s/^add '(.*)'$/\1/" | sort > "$TMP/staged.txt"; echo "INFO  simulated with $A/gitignore_PROPOSED.txt (no .git yet)"
fi
N=$(wc -l < "$TMP/staged.txt"); echo "INFO  $N files"
if [ -d .git ]; then [ -f .gitignore ] && cmp -s .gitignore $A/gitignore_PROPOSED.txt && ok ".gitignore equals the reviewed proposal" || bad ".gitignore missing or differs from $A/gitignore_PROPOSED.txt"; fi
grep -v '^#' $A/EXCLUDED_PENDING_PERMISSION.txt | sort > "$TMP/withheld.txt"
w=$(comm -12 "$TMP/staged.txt" "$TMP/withheld.txt" | wc -l); [ "$w" = 0 ] && ok "no file withheld pending permission is staged ($(grep -c . "$TMP/withheld.txt") listed)" || bad "$w withheld files staged"
if [ -f $A/EXCLUDED_FROM_PUBLIC_RELEASE.txt ]; then
  grep -v '^#' $A/EXCLUDED_FROM_PUBLIC_RELEASE.txt | sort > "$TMP/notpublic.txt"
  w=$(comm -12 "$TMP/staged.txt" "$TMP/notpublic.txt" | wc -l); [ "$w" = 0 ] && ok "no file excluded from public release is staged ($(grep -c . "$TMP/notpublic.txt") listed)" || bad "$w not-for-public files staged"
fi
p=$(grep -cE '\.(pt|keras|pptx)$|(^|/)(pool_train[^/]*\.npy|selection_val\.npz|objects_64\.npz)$|^(train|valid|archive_review_required|\.venv|\.venv-linux|dist|demo_out)/|^data/extension_medmnist/|__pycache__/' "$TMP/staged.txt")
[ "$p" = 0 ] && ok "no checkpoints, decks, pools, datasets, environments or caches staged" || bad "$p forbidden files staged"
o=$(grep -E '^(outputs/phase|results/|thesis/figures/|presentation/)' "$TMP/staged.txt" | grep -iE '\.(png|jpe?g)$' \
    | grep -vE '(_vs_shots[^/]*|metrics_vs_shots_by_split|loss_curves|metric_curves[^/]*|psnr_ssim_distributions|psnr10000_vs_object_contrast|training_curves|metrics_vs_photon_level_test|group_distributions|dimensions_by_group|final_dimensions|source_groups_original_vs_final|split_by_source_group|phase2_metrics_vs_shots)\.png$' | wc -l)
[ "$o" = 0 ] && ok "only approved metric plots among images in onion folders" || bad "$o unexpected images in onion folders"
big=$(tr '\n' '\0' < "$TMP/staged.txt" | xargs -0 -r stat -c '%s %n' 2>/dev/null | awk '$1>50e6' | wc -l)
[ "$big" = 0 ] && ok "no staged file over 50 MB" || bad "$big staged files over 50 MB"
sec=$(tr '\n' '\0' < "$TMP/staged.txt" | xargs -0 -r grep -IlE '(AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|BEGIN [A-Z ]*PRIVATE KEY|api[_-]?key\s*[:=]|password\s*[:=])' 2>/dev/null | wc -l)
[ "$sec" = 0 ] && ok "no secrets found" || bad "$sec files look like they contain secrets"
pi=$(tr '\n' '\0' < "$TMP/staged.txt" | xargs -0 -r grep -IlE 'B231[0-9]{2}|B232[0-9]{2}' 2>/dev/null | sort | tr '\n' ' ')
echo "INFO  files with roll-number patterns (B23xxx): ${pi:-none}"
# personal names are read from a LOCAL file (never committed), so this script itself contains none
PP="$ROOT/local_records/personal_patterns.txt"
if [ -s "$PP" ]; then
  nm=$(tr '\n' '\0' < "$TMP/staged.txt" | xargs -0 -r grep -IliF -f "$PP" 2>/dev/null | sort | tr '\n' ' ')
  echo "INFO  files containing a listed personal name or roll number: ${nm:-none}"
else echo "INFO  no local personal-pattern list (local_records/personal_patterns.txt); name scan skipped"; fi
em=$(tr '\n' '\0' < "$TMP/staged.txt" | xargs -0 -r grep -IhoE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(com|in|org|edu|net)' 2>/dev/null | sort -u | tr '\n' ' ')
[ -z "$em" ] && ok "no e-mail addresses in staged files" || echo "INFO  e-mail addresses found: $em"
lic=$(grep -cE '^(LICENSE|LICENCE)(\.|$)' "$TMP/staged.txt"); [ "$lic" = 0 ] && echo "INFO  no LICENSE staged (licence still a proposal)" || echo "INFO  LICENSE staged: confirm it was approved"

echo "== 3 · simulated clone (only the files that would be committed)"
mkdir -p "$TMP/clone"; (cd "$ROOT" && tr '\n' '\0' < "$TMP/staged.txt" | xargs -0 cp --parents -t "$TMP/clone")
cd "$TMP/clone"
for m in FREEZE_MANIFEST_phase1_phase2.md5 outputs/extension_medmnist_microscopy/FREEZE_MANIFEST_extension.md5 \
         outputs/extension_medmnist_microscopy_poisson/FREEZE_MANIFEST_poisson.md5 $A/FREEZE_MANIFEST_code_2026-11-05.md5; do
  out=$(md5sum -c --ignore-missing --quiet "$m" 2>&1); [ -z "$out" ] && ok "clone: $m (--ignore-missing)" || bad "clone: $m: $out"
done
find . -name '*.py' -print0 | xargs -0 "$PY" -m py_compile 2>"$TMP/pyc.err" && ok "clone: all Python files compile" || bad "clone: compile errors ($(head -c 300 "$TMP/pyc.err"))"
"$PY" thesis/merge_proposal/verify_claims.py >/dev/null 2>&1 && grep -q '29 of 29 claims confirmed' thesis/merge_proposal/CLAIMS_VERIFICATION.md \
  && ok "clone: verify_claims.py 29/29" || bad "clone: verify_claims.py"
if [ -f thesis/MTP_Thesis_Final.md ]; then
  r=$("$PY" audit/audit_onion_thesis.py 2>&1 | grep 'rows checked'); [ "$r" = "onion thesis rows checked: 39, mismatching rows: 2" ] \
    && ok "clone: audit_onion_thesis.py (39 rows; the 2 known checker artefacts)" || bad "clone: audit_onion_thesis.py: $r"
else echo "INFO  clone: thesis not committed; thesis checks ran on the working copy (section 1)"; fi
if [ -f thesis/MTP_Thesis_Final_with_extension.md ]; then
  "$PY" thesis/merge_proposal/check_merged_thesis.py >"$TMP/cmt.txt" 2>&1 && ok "clone: check_merged_thesis.py ($(grep -c '^PASS' "$TMP/cmt.txt") checks)" || bad "clone: check_merged_thesis.py: $(grep FAIL "$TMP/cmt.txt")"
fi
"$PY" infer_example.py --checkpoint outputs/extension_medmnist_microscopy/models/target_clean/model_1000/checkpoints/best.pt --shots 1000 >"$TMP/inf.txt" 2>&1
grep -q '^missing ' "$TMP/inf.txt" && ok "clone: infer_example.py stops with a clear 'missing' message" || bad "clone: infer_example.py: $(tail -2 "$TMP/inf.txt")"
if [ "${FULL:-0}" = 1 ]; then
  "$PY" audit/audit_recompute.py "$TMP/rc.json" >/dev/null 2>&1 && "$PY" -c "import json,sys; sys.exit(json.load(open('$TMP/rc.json'))!=json.load(open('audit/audit_recompute.json')))" \
    && ok "clone: audit_recompute.py identical to the stored result" || bad "clone: audit_recompute.py"
else echo "INFO  audit_recompute.py skipped (set FULL=1)"; fi
cd "$ROOT"

echo "== 4 · nothing in the project changed during this check"
for m in FREEZE_MANIFEST_phase1_phase2.md5 $A/PROTECTED_ORIGINALS.md5; do md5sum -c --quiet "$m" >/dev/null 2>&1 && ok "$m still OK" || bad "$m changed"; done
[ "$fail" = 0 ] && echo "RESULT: ALL CHECKS PASSED ($N files would be committed)" || echo "RESULT: FAILED (see FAIL lines)"
exit $fail
