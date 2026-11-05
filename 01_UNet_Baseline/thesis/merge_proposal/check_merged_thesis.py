"""Read-only structure checks of thesis/MTP_Thesis_Final_with_extension.md against the original thesis.

    .venv-linux/bin/python thesis/merge_proposal/check_merged_thesis.py
Prints one line per check; exit code 1 if any structural check fails (placeholders are reported, not failed).
"""
import collections
import glob
import os
import re
import sys
from pathlib import Path

T = Path(__file__).resolve().parents[1]
O = (T / "MTP_Thesis_Final.md").read_text(encoding="utf-8").replace("\r", "")
M = (T / "MTP_Thesis_Final_with_extension.md").read_text(encoding="utf-8").replace("\r", "")
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{(': ' + str(detail)) if detail else ''}")
    if not ok:
        fails.append(name)


num = lambda s: collections.Counter(re.findall(r"\d[\d,]*\.\d+|\d[\d,]*", s))
lost = num(O) - num(M)
check("every number of the original is still present", not lost, dict(lost) or "")
secs = re.findall(r"^#{1,3} ((?:[1-6]|A)\.\d+(?:\.\d+)?)\s", M, re.M)
check("no duplicate section numbers", len(secs) == len(set(secs)), f"{len(secs)} numbered headings")
bad = []
for ch in "123456A":
    top = [int(s.split(".")[1]) for s in secs if s.split(".")[0] == ch and s.count(".") == 1]
    if top != list(range(1, len(top) + 1)):
        bad.append((ch, top))
sub = collections.defaultdict(list)
for s in secs:
    if s.count(".") == 2:
        sub[s.rsplit(".", 1)[0]].append(int(s.rsplit(".", 1)[1]))
bad += [(k, v) for k, v in sub.items() if v != list(range(1, len(v) + 1))]
check("sections and subsections sequential", not bad, bad or "")
toc = re.findall(r"^  - (\d\.\d+) ", M.split("## TABLE OF CONTENTS")[1].split("## LIST OF FIGURES")[0], re.M)
check("table of contents matches chapter sections", toc == [s for s in secs if s.count(".") == 1 and s[0] != "A"])
tabs = re.findall(r"^\*\*Table (\d\.\d+) –", M, re.M)
figs = re.findall(r"^\*\*Figure (\d\.\d+) –", M, re.M)
lot = re.findall(r"^\| (\d\.\d+) \| ", M.split("## LIST OF TABLES")[1].split("\n---\n")[0], re.M)
lof = re.findall(r"^\| (\d\.\d+) \| ", M.split("## LIST OF FIGURES")[1].split("\n---\n")[0], re.M)
check("list of tables matches body", lot == tabs, f"{len(tabs)} tables")
check("list of figures matches body", lof == figs, f"{len(figs)} figures")
refs = set(re.findall(r"Sections? ((?:\d|A)\.\d+(?:\.\d+)?)", M))
refs |= {x for a in re.findall(r"Sections ((?:\d|A)\.\d+(?:\.\d+)?)[–-]((?:\d|A)\.\d+(?:\.\d+)?)", M) for x in a}
check("section references resolve", all(r in secs for r in refs), sorted(r for r in refs if r not in secs) or "")
check("table references resolve", all(t in tabs for t in re.findall(r"Tables? (\d\.\d+)", M)))
check("figure references resolve", all(f in figs for f in re.findall(r"Figures? (\d\.\d+)", M)))
body, reflist = M.split("# REFERENCES")[0], re.findall(r"^\[(\d+)\] ", M.split("# REFERENCES")[1], re.M)
cited = set(re.findall(r"\[(\d+)\]", body))
check("references: all cited and all listed", cited == set(reflist) and reflist == [str(i) for i in range(1, len(reflist) + 1)],
      f"[1]–[{reflist[-1]}]")
imgs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", M)
wl = T.parent / "07_Reproducibility_and_Audit/EXCLUDED_PENDING_PERMISSION.txt"   # files withheld from the repository
withheld = {l.strip() for l in wl.read_text().splitlines() if l.strip() and not l.startswith("#")} if wl.exists() else set()
rel = lambda i: os.path.normpath(os.path.join("thesis", i))
missing = [i for i in imgs if not os.path.exists(T / i)]
held = [i for i in missing if rel(i) in withheld]
check("figure image files exist (or are withheld pending permission)", len(missing) == len(held),
      f"{len(imgs)} links, {len(held)} withheld" + (f", missing: {[i for i in missing if i not in held]}" if len(missing) > len(held) else ""))
print(f"INFO  figure links to files withheld pending permission: {sum(rel(i) in withheld for i in imgs)}")
paths = set(re.findall(r"`((?:outputs|audit|thesis|data|0[1-8]_)[^`\s]*)`", M))
inv = T.parent / "07_Reproducibility_and_Audit/REPO_INVENTORY.csv"        # files deliberately not in Git (data, withheld, …)
not_in_git = withheld | ({r["path"] for r in __import__("csv").DictReader(inv.open()) if r["github"] != "COMMIT"} if inv.exists() else set())
miss = [p for p in paths if "{" not in p and "..." not in p and "…" not in p and not glob.glob(str(T.parent / p))]
excluded = [p for p in miss if p in not_in_git]
check("cited project paths exist (or are deliberately not in Git)", len(miss) == len(excluded),
      [p for p in miss if p not in excluded] or (f"{len(excluded)} deliberately not in Git" if excluded else ""))
check("no leftover draft markers", not any(m in M for m in ("DRAFT", "Delete this comment", "Additions to Section", "{{")))
ph = [(i + 1, x) for i, l in enumerate(M.split("\n")) for x in
      re.findall(r"\[(?:Student Name|Roll[^\]]*|Supervisor Name[^\]]*|Department[^\]]*|Institute[^\]]*|Month Year|Date|Programme[^\]]*|Designation)\]|\[PLACEHOLDER[^\]]*\]", l)]
print(f"INFO  front-matter placeholders remaining: {len(ph)} (lines {sorted({n for n, _ in ph})})")
sys.exit(1 if fails else 0)
