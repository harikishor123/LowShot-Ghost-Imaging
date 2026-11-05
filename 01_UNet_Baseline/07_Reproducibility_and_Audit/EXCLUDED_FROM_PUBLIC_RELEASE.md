# Files kept out of the public repository (2026-11-05)

The GitHub repository `harikishor123/LowShot-Ghost-Imaging` is **public** by the authors' decision. In addition to the permission-pending exclusions (`EXCLUDED_PENDING_PERMISSION.md`), these 16 files are kept out of Git. **They stay on disk unchanged**; only the "NOT FOR PUBLIC RELEASE" block of `gitignore_PROPOSED.txt` / `.gitignore` excludes them. The exact list is in `EXCLUDED_FROM_PUBLIC_RELEASE.txt`.

| Files | Reason | Released when |
|---|---|---|
| `thesis/MTP_Thesis_Final_with_extension.md` | thesis review copy, not approved | the authors and the guide approve it |
| `thesis/merge_proposal/MERGE_DIFF.txt`, `THESIS_MERGE_PROPOSAL.md`, `draft_extension_sections_BEFORE_corrections.md`, `thesis/draft_extension_sections.md`, `thesis/ext_draft_template.md` | contain the review copy's new text, in diff, draft or template form | same |
| `thesis/MTP_Thesis_Final.md` | original thesis: 24 front-matter placeholders; publication not yet approved by the guide or institute (decision G6) | the authors and the guide approve publication |
| `thesis/review/GUIDE_REVIEW_PACKAGE.md`, `PLACEHOLDERS_TO_PROVIDE.md`, `REQUIRED_INFORMATION.md` | internal review material; collect personal details (names, roll numbers) | not intended for publication |
| `thesis/merge_proposal/PRESENTATION_COMPARISON.md`, `presentation/VIVA_QA.md`, `PPT_FINAL_CHECK.md`, `REVIEW_DRAFT_CORRECTIONS.md`, `build_deck.js` | internal presentation and viva preparation; the decks themselves are withheld (onion images); `build_deck.js` and two notes contain names and roll numbers | authors' decision |
| `DECISIONS_REQUIRED.md` | internal decision record; lists personal and institutional details still to be provided | not intended for publication |

## Effect on the committed scripts

| Script | Without the excluded files |
|---|---|
| `thesis/merge_proposal/verify_claims.py` (29 claims) | works: reads frozen CSV and JSON only |
| `audit/audit_onion_thesis.py`, `thesis/merge_proposal/check_merged_thesis.py`, `build_merged_thesis.py`, `thesis/make_ext_draft.py` | need the thesis or draft files; they run in the authors' working copy only. `precommit_verify.sh` runs them there before every commit |
| All experiment, audit and gallery scripts | unaffected |
