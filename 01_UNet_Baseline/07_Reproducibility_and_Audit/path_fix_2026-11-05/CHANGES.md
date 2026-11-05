# Machine-specific path fix (2026-11-05)

Three scripts had a hard-coded project root. Each now derives the root from its own location, so it works from any clone and any working directory. **No other line was changed.** None of the three files is listed in any freeze manifest. Byte-for-byte copies of the originals are in `originals/`.

| File | Old line | New line | MD5 before → after |
|---|---|---|---|
| `audit/audit_recompute.py:18` | `ROOT = Path("/home/ss_students/Downloads/Onion-CNN-Final.v1-onion-original-v1.yolov8")` | `ROOT = Path(__file__).resolve().parents[1]` | `5179efc37548c08287c5997d11300212` → see `FINAL_GITHUB_READINESS_REPORT.md` |
| `thesis/make_ext_draft.py:13` | `ROOT = Path("/home/ss_students/Downloads/Onion-CNN-Final.v1-onion-original-v1.yolov8")` | `ROOT = Path(__file__).resolve().parents[1]` | `4a678458c947bf69e7e70879bded9d50` → see report |
| `thesis/figures/make_report_figs.py:12` | `ROOT = Path(r"C:\Users\vishn\Downloads\Onion-CNN-Final.v1-onion-original-v1.yolov8")` | `ROOT = Path(__file__).resolve().parents[2]` | `1169403afdf6a4b98d43016f6cb170b6` → see report |

## Checks that behaviour is unchanged

- All three new expressions resolve to exactly the old Linux root on this machine.
- `make_ext_draft.py`, run from the project root and from `/tmp`: output byte-identical to `thesis/draft_extension_sections.md`.
- `audit_recompute.py`, run from `/tmp`: output JSON identical to the stored `audit/audit_recompute.json`.
- `make_report_figs.py`: the old Windows path could not run on Linux. A copy with its output folder redirected to a scratch folder ran successfully and produced both figures. The figures in `thesis/figures/` were **not** regenerated or replaced; outputs differ in bytes because of a different Matplotlib version and fonts.
