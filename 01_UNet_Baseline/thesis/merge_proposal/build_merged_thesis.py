"""Build thesis/MTP_Thesis_Final_with_extension.md from the unchanged original and the corrected extension draft.

Applies changes 1-22 of THESIS_MERGE_PROPOSAL.md. Every replacement must match exactly once, or the script stops.
The original thesis/MTP_Thesis_Final.md is only read. Run from the project root:
    .venv-linux/bin/python thesis/merge_proposal/build_merged_thesis.py
"""
import hashlib
import re
from pathlib import Path

T = Path(__file__).resolve().parents[1]
SRC, DRAFT, OUT = T / "MTP_Thesis_Final.md", T / "draft_extension_sections.md", T / "MTP_Thesis_Final_with_extension.md"
ORIG_MD5 = "926f7ab493a8bac6ce060975cbc544e1"
assert hashlib.md5(SRC.read_bytes()).hexdigest() == ORIG_MD5, "original thesis changed"

txt = SRC.read_text()
dr = DRAFT.read_text()
log = []


def rep(old, new, tag):
    global txt
    n = txt.count(old)
    assert n == 1, f"{tag}: expected 1 match, found {n}"
    txt = txt.replace(old, new)
    log.append(tag)


def cut(start, end):
    """Draft text from the line starting `start` up to (not including) the line starting `end`."""
    a = dr.index("\n" + start) + 1
    b = dr.index("\n" + end, a) + 1 if end else len(dr)
    body = dr[a:b].rstrip()
    body = re.sub(r"\n---\s*$", "", body).rstrip()
    return body


ch3 = cut("## 3.14 ", "## 4.10 ")
ch4 = cut("## 4.10 ", "## Additions to Section 4.12")
lim = "\n".join(l for l in dr.splitlines() if re.match(r"^(1[4-9]|2[0-3])\. \*\*", l))
refs = "\n\n".join(l for l in dr.splitlines() if re.match(r"^\[2[5-7]\] ", l))
a7 = cut("## A.7 ", None)
assert lim.count("\n") == 9 and refs.count("[2") == 3

# change 22: phase4a sentence at the end of 3.15.1
P4A = ("An earlier exploratory onion experiment with Poisson and detector noise at a fixed photon scale and Tikhonov / TV "
       "baselines (`outputs/phase4a_noisy_baselines/`) is not part of the reported results.")
i = ch3.index("\n### 3.15.2")
ch3 = ch3[:i].rstrip() + "\n\n" + P4A + "\n" + ch3[i:]

# 0 · title-page note (found in the final audit: it still described the study as noise-free and onion-only)
rep("The work reported here is a computational study of measurement-shot reduction in simulated, noise-free classical Differential Ghost Imaging of onion microscopy images, as described by the subtitle.",
    "The work reported here is a computational study of measurement-shot reduction in simulated classical Differential Ghost Imaging, as described by the subtitle: noise-free measurements of onion microscopy images (main study), extended to BloodMNIST and PathMNIST images and to simulated Poisson bucket noise (Sections 3.14–3.15 and 4.10–4.11).",
    "0 title note")

# 0b · whole-study statements that would otherwise exclude the extension (final audit; wording only)
rep("Reconstructions using 500, 1,000, 2,000 and 5,000 measurements are evaluated against the 10,000-shot reference.\n",
    "Reconstructions using 500, 1,000, 2,000 and 5,000 measurements are evaluated against the 10,000-shot reference. A separate extension repeats the study on BloodMNIST and PathMNIST images and with simulated Poisson bucket noise (Sections 3.14–3.15).\n",
    "0b 1.6 problem statement")
rep("This work reduces the **number of measurement shots** in a simulated, noise-free setting.",
    "This work reduces the **number of measurement shots** in a simulated setting (noise-free, with a simplified Poisson-noise extension).",
    "0b 2.5")
rep("The present study addresses this question in a controlled, simulated, noise-free setting.",
    "The present study addresses this question in a controlled, simulated setting (noise-free in the main study; simplified Poisson bucket noise in an extension).",
    "0b 2.11")
rep("It does not reduce or quantify the number of photons per measurement or the illumination dose.\n",
    "It does not reduce or quantify the number of photons per measurement or the illumination dose. The extensions of Sections 3.14–3.15 are likewise computational, using MedMNIST+ images and, in Section 3.15, simulated Poisson bucket noise.\n",
    "0b ch3 intro")
rep("The most important next step is to move from simulated, noise-free measurements to a physical GI/DGI system.",
    "The most important next step is to move from simulated measurements (noise-free, or with the simplified Poisson noise of Section 4.11) to a physical GI/DGI system.",
    "0b 6.1")

# 1 · Abstract
rep("""Within the simulated conditions studied, AI-based reconstruction reduced the number of measurement shots needed for reference-comparable reconstruction by up to 10×, with 2,000 shots preferable when fine detail matters. The measurements were simulated and noise-free on one specimen type, and the study does not experimentally demonstrate quantum-enhanced or low-photon imaging, nor evaluate photon dose or biological effects.""",
    """A separate extension repeated both designs on 2,000 previously unseen test images from MedMNIST+ (1,000 BloodMNIST blood smears and 1,000 PathMNIST colorectal-tissue patches). The same pre-registered criterion was again met at 1,000 shots for all images (10×), with AI-original-1000 reaching SSIM 0.671 against the original image compared with 0.534 for conventional 10,000-shot DGI. The reduction depended on image content: blood smears passed at 500 shots (20×), while textured tissue needed 2,000–5,000 shots (2–5×). With simulated Poisson noise on the bucket measurements at 5%, 10% and 30% of the bucket-signal spread, 1,000 shots still met the criterion for all images against 10,000-shot DGI of the same noisy acquisition at every level (at 30%: SSIM 0.606 versus 0.448). Tissue alone, however, required 2,000–5,000 shots, and the reconstructions preserved overall structure while smoothing fine texture and some small structures.

Within the simulated conditions studied, AI-based reconstruction reduced the number of measurement shots needed for reference-comparable reconstruction by up to 10× for the combined datasets (20× for blood smears), with 2,000 shots preferable when fine detail matters. The measurements were simulated, either noise-free or with a simplified relative Poisson noise model. The study does not experimentally demonstrate quantum-enhanced or low-photon imaging, nor evaluate photon dose or biological effects.""",
    "1 Abstract")

# 2 · Table of contents
rep("  - 3.13 Phase 2: Direct Reconstruction Against the Original Clean Object\n",
    "  - 3.13 Phase 2: Direct Reconstruction Against the Original Clean Object\n"
    "  - 3.14 Extension: Broader Microscopy Dataset\n  - 3.15 Extension with Poisson Bucket Noise\n", "2 TOC ch3")
rep("  - 4.10 Limitations\n  - 4.11 Overall Finding\n",
    "  - 4.10 Extension: Broader Microscopy Dataset\n  - 4.11 Extension with Poisson Bucket Noise\n"
    "  - 4.12 Limitations\n  - 4.13 Overall Finding\n", "2 TOC ch4")
rep("  - 5.7 Main Contribution\n  - 5.8 Overall Conclusion\n",
    "  - 5.7 Extension to Other Microscopy Images and to Measurement Noise\n  - 5.8 Main Contribution\n  - 5.9 Overall Conclusion\n",
    "2 TOC ch5")
rep("- Appendix A – Computational Reproducibility\n\n---\n\n## LIST OF FIGURES",
    "- Appendix A – Computational Reproducibility (A.1–A.7)\n\n---\n\n## LIST OF FIGURES", "2 TOC appendix")

# 3 · List of figures
E, P = "outputs/extension_medmnist_microscopy/results/target_clean/test/figures/", "outputs/extension_medmnist_microscopy_poisson/results/level_p30/test/figures/"
rep("(panels from `outputs/phase_final_original_target/figures/test/comparison_all_examples.png`) |\n",
    "(panels from `outputs/phase_final_original_target/figures/test/comparison_all_examples.png`) |\n"
    f"| 4.11 | Extension, clean-object target: SSIM, Pearson r, PSNR and edge correlation versus number of shots (test set) | `{E}metrics_vs_shots_all.png` |\n"
    f"| 4.12 | Extension test examples: clean object, DGI-N, AI-original-N and DGI-10000 (three blood, three tissue) | `{E}comparison_examples.png` |\n"
    f"| 4.13 | Poisson noise f = 0.30: metrics versus number of shots (test set) | `{P}metrics_vs_shots_all.png` |\n"
    f"| 4.14 | Poisson noise f = 0.30: test examples | `{P}comparison_examples.png` |\n"
    "| 4.15 | Tissue test images at 1,000 shots and 0, 5, 10 and 30% noise, with difference maps | `audit/final_verification/figures/D_verified_tissue.png` |\n",
    "3 LoF")

# 4 · List of tables
rep("| 3.8 | Phase 1 and Phase 2 experimental designs |\n",
    "| 3.8 | Phase 1 and Phase 2 experimental designs |\n| 3.9 | Onion study and broader-microscopy extension |\n", "4 LoT 3.9")
newt = [t for t in re.findall(r"^\*\*Table (4\.(?:9|1[0-8])) – (.+?)\*\*$", ch4, re.M)]
assert [n for n, _ in newt] == [f"4.{k}" for k in range(9, 19)], newt
TRAIL = re.compile(r"\s*\((?:[^()]|\([^()]*\))*\)$")          # final bracketed note only
rows = "".join("| " + n + " | " + TRAIL.sub("", t).replace("*", "") + " |\n" for n, t in newt)
rep("| 4.8 | Phase 2: AI-original-N minus DGI-10000 (paired; test set) |\n",
    "| 4.8 | Phase 2: AI-original-N minus DGI-10000 (paired; test set) |\n" + rows, "4 LoT 4.9-4.18")
rep("| 5.4 | Phase 2 summary against the original clean object |\n",
    "| 5.4 | Phase 2 summary against the original clean object |\n| 5.5 | Extension: minimum successful shot count |\n", "4 LoT 5.5")

# 5 · Objectives
rep("using SSIM, Pearson correlation, PSNR, MSE and edge correlation.\n\n## 1.10 Scope of the Study",
    "using SSIM, Pearson correlation, PSNR, MSE and edge correlation.\n"
    "11. (Extension) Repeat both phases on blood-smear and colorectal-tissue images from MedMNIST+.\n"
    "12. (Extension) Test robustness to simulated Poisson bucket noise at three relative levels.\n\n## 1.10 Scope of the Study",
    "5 Objectives")

# 6, 7 · Scope
rep("- noise-free bucket measurements;\n- onion microscopy images;\n",
    "- noise-free bucket measurements (main study) and Poisson bucket noise at three relative levels (extension);\n"
    "- onion microscopy images (main study) and BloodMNIST / PathMNIST images (extension);\n", "6 Scope within")
rep("- real detector and measurement noise;\n- generalisation to other biological specimens;\n",
    "- detector readout noise, background light, pattern errors and physically calibrated photon budgets;\n"
    "- specimens other than onion, blood smears and H&E colorectal tissue;\n", "7 Scope outside")
rep("in classical, simulated, noise-free GI only.", "in classical, simulated GI only.", "7 Scope last bullet")

# 8 · 1.11 Main Contribution
rep("No reconstruction was identical to the clean object.\n\nThese results demonstrate the feasibility of using AI-based reconstruction",
    "No reconstruction was identical to the clean object.\n\n"
    "A separate extension to 2,000 blood-smear and colorectal-tissue test images (MedMNIST+) reached the same criterion at **1,000 shots (10×)** for all images under both designs, and kept this result under simulated Poisson bucket noise at 5%, 10% and 30%. Blood smears passed at 500 shots, while textured tissue needed 2,000–5,000 shots, and the reconstructions smooth fine detail.\n\n"
    "These results demonstrate the feasibility of using AI-based reconstruction", "8 1.11")

# 9 · Chapter 3 insertion (after 3.13.4, before Chapter 4)
rep("All 72 Phase 1 output files were verified to be byte-identical before and after Phase 2.\n\n---\n\n# CHAPTER 4",
    "All 72 Phase 1 output files were verified to be byte-identical before and after Phase 2.\n\n" + ch3 + "\n\n---\n\n# CHAPTER 4",
    "9 insert 3.14-3.15")

# 12 · cross-reference
rep("(Section 4.10, items 10–12)", "(Section 4.12, items 10–12)", "12 cross-ref")

# 10 · Chapter 4 insertion; 11 · Limitations
rep("In particular, it reuses the Phase 1 test images.\n\n## 4.10 Limitations",
    "In particular, it reuses the Phase 1 test images.\n\n" + ch4 + "\n\n## 4.12 Limitations", "10 insert 4.10-4.11")
rep("1. **Simulated, noise-free measurements.** All measurements were generated computationally, and the bucket values contain no noise.",
    "1. **Simulated measurements.** All measurements were generated computationally. The main study uses noise-free bucket values; the extension (Section 4.11) adds only a simplified relative Poisson noise model.",
    "11 item 1")
rep("2. **No experimental imperfections.** Detector noise, photon (shot) statistics, illumination-pattern errors, source fluctuations and other experimental effects were not modelled.",
    "2. **No experimental imperfections.** Detector readout noise, illumination-pattern errors, source fluctuations, background light and physically calibrated photon statistics were not modelled.",
    "11 item 2")
rep("8. **One specimen type.** The dataset consists of onion microscopy images at 64 × 64 resolution. Whether the results extend to other biological specimens, higher resolutions or other imaging modalities remains to be established.",
    "8. **Limited specimen types.** The main study uses onion images; the extension adds blood smears and H&E colorectal tissue (MedMNIST+), all at 64 × 64. Other specimens, higher resolutions and other modalities remain untested.",
    "11 item 8")
rep("The reported results should not be described as fully converged optimum performance.\n13.",
    "The reported results should not be described as fully converged optimum performance. In the noise-free extension, seven of eight best checkpoints also fell at epochs 71–80 of 80; with Poisson noise, six of the 12 models stopped early under the patience rule (Section 4.11.3).\n13.",
    "11 item 12")
rep("the study concerns measurement-shot reduction in simulated classical GI.\n\nThese limitations do not undermine",
    "the study concerns measurement-shot reduction in simulated classical GI.\n" + lim + "\n\nThese limitations do not undermine",
    "11 items 14-23")

# 13 · Overall Finding
rep("## 4.11 Overall Finding", "## 4.13 Overall Finding", "13 renumber")
rep("They do not show any reduction in biological damage, which was not measured.\n\n---\n\n# CHAPTER 5",
    "They do not show any reduction in biological damage, which was not measured.\n\n"
    "In a separate extension to 2,000 blood-smear and colorectal-tissue test images, the same pipeline again met the pre-registered criterion at **1,000 shots (10×)** for all images under both designs. Blood smears passed at 500 shots and textured tissue needed 2,000–5,000. With simulated Poisson bucket noise at 5%, 10% and 30%, AI-original-1000 remained non-inferior to 10,000-shot DGI of the same noisy acquisition at every level. The 1,000-shot result holds for the combined test set, not for tissue alone, and the U-Net reconstructions are structure-preserving but detail-smoothing.\n\n---\n\n# CHAPTER 5",
    "13 paragraph")

# 14 · 5.1
rep("with conventional 10,000-shot DGI as the baseline (Section 5.6).\n",
    "with conventional 10,000-shot DGI as the baseline (Section 5.6). Separate extensions (Section 5.7) repeated the study on 2,000 blood-smear and colorectal-tissue test images and under simulated Poisson bucket noise.\n",
    "14 5.1")

# 15 · 5.5
rep("It does not establish generalisation to other biological specimens, to real hardware, or to measurements with experimental noise.",
    "It does not establish generalisation to real hardware or to measurements with experimental noise. Generalisation to two further microscopy collections and to simulated Poisson bucket noise was tested separately (Section 5.7).",
    "15 5.5")

# 16 · Chapter 5: new 5.7, renumber 5.7 -> 5.8 and 5.8 -> 5.9
S57 = """## 5.7 Extension to Other Microscopy Images and to Measurement Noise

The pipeline was repeated, with an unchanged recipe and a protocol written before training, on 7,000 training, 1,000 validation and 2,000 test images from MedMNIST+ (half BloodMNIST, half PathMNIST). It was then repeated again with simulated Poisson noise on every bucket value. Table 5.5 gives the minimum shot counts that met the pre-registered criterion on the test set.

**Table 5.5 – Extension: minimum successful shot count (test set; validation agreed in 25 of 27 noise cases and in all 12 noise-free cases)**

| Setting | All images | Blood | Tissue |
|---|---|---|---|
| Noise-free, AI-original vs DGI-10000 | 1,000 | 500 | 5,000 |
| Noise-free, AI vs independent 10k acquisition | 1,000 | 500 | 2,000 |
| 5% Poisson noise, AI-original vs noisy DGI-10000 | 1,000 | 500 | 5,000 |
| 10% Poisson noise | 1,000 | 500 | 2,000 (borderline) |
| 30% Poisson noise | 1,000 | 500 | 2,000 |

- **Generalisation.** The 10× result of the onion study held for the combined blood-and-tissue test set, with and without noise. Gaussian-smoothed DGI needed 2,000 shots and raw DGI 5,000.
- **Image content matters.** Blood smears (a few large objects on a smooth background) passed at 500 shots. Textured tissue needed 2,000–5,000 shots. At 30% noise, tissue did not reach noise-free 10,000-shot DGI within 5,000 shots. Its earlier pass at 10–30% noise against the noisy comparator reflects a weaker comparator, not better reconstruction.
- **Fine detail.** The reconstructions preserve structure but smooth detail. At 1,000 shots and up to 10% noise they keep about a third (blood) and a sixth (tissue) of the original's fine-detail content (29% and 9% at 30% noise; validation images, Section 4.11.4), and some small structures fade at 500 shots and 30% noise.

"""
rep("## 5.7 Main Contribution\n", S57 + "## 5.8 Main Contribution\n", "16 insert 5.7")
rep("within the simulated conditions studied.\n\nThe study does not directly measure photon dose, illumination dose,",
    "within the simulated conditions studied.\n\n"
    "The extension showed that the 10× reduction carries over to blood-smear and colorectal-tissue images and survives simulated Poisson bucket noise up to 30% of the bucket-signal spread for the combined test set, while making clear that the achievable reduction depends on image content (20× for blood smears, 2–5× for textured tissue).\n\n"
    "The study does not directly measure photon dose, illumination dose,", "16 5.8 paragraph")
rep("## 5.8 Overall Conclusion", "## 5.9 Overall Conclusion", "16 renumber 5.9")
rep("- **AI-original-5000** achieves the highest tested clean-object fidelity, although it is not identical to the clean object.\n\nOverall,",
    "- **AI-original-5000** achieves the highest tested clean-object fidelity, although it is not identical to the clean object.\n"
    "- **Extension:** on 2,000 blood-smear and tissue test images the minimum was again **1,000 shots (10×)** for all images, with and without simulated Poisson bucket noise up to 30%; blood smears needed 500 shots, textured tissue 2,000–5,000.\n\nOverall,",
    "16 5.9 bullet")
rep("- It was obtained with simulated, noise-free measurements of one specimen type, and the Phase 2 test images had already been used in Phase 1.",
    "- It was obtained with simulated measurements (noise-free, or with a simplified relative Poisson noise model) of onion, blood-smear and colorectal-tissue images at 64 × 64; the Phase 2 test images had already been used in Phase 1, and every model was trained once.",
    "16 5.9 limit bullet")

# 17 · Chapter 6
rep("The final experiment used noise-free bucket measurements. Future work should",
    "A first step was taken in Section 4.11 with Poisson bucket noise at three relative levels. Detector readout noise, background light, pattern errors and physically calibrated photon budgets remain to be studied. The main experiment used noise-free bucket measurements. Future work should",
    "17 6.2")
rep("The present results are limited to onion microscopy images. Possible future datasets include:",
    "Section 4.10 extended the study to blood smears and H&E colorectal tissue. Higher-resolution images, other modalities (for example fluorescence) and other plant and animal tissues remain untested. Possible future datasets include:",
    "17 6.3")

# 18 · 6.8
rep("- longer training, since the best checkpoints of both phases occurred at or near the 80-epoch cap.\n",
    "- longer training, since the best checkpoints of both phases occurred at or near the 80-epoch cap;\n"
    "- repeated training with several random seeds, especially for borderline results;\n"
    "- losses or architectures that preserve fine texture and small structures (the extension's U-Net outputs over-smooth).\n",
    "18 6.8")

# 19 · References
rep("Available: arXiv:2209.14687.\n\n---\n\n# APPENDIX A",
    "Available: arXiv:2209.14687.\n\n" + refs + "\n\n---\n\n# APPENDIX A", "19 references")

# 20 · A.1
rep("File hashes of both frozen experiment folders are recorded in `FREEZE_MANIFEST_phase1_phase2.md5`.\n",
    "File hashes of both frozen experiment folders are recorded in `FREEZE_MANIFEST_phase1_phase2.md5`. The extensions used a separate Linux/GPU environment; see Section A.7.\n",
    "20 A.1")

# 21 · A.7 at the end
txt = txt.rstrip() + "\n\n" + a7 + "\n"
log.append("21 A.7")

# review banner (decision 2026-11-05): remove only after the authors and the guide approve this copy
BANNER = ("> **REVIEW COPY (not approved).** This merged version adds the MedMNIST+ and Poisson-noise extensions to "
          "`thesis/MTP_Thesis_Final.md`, which remains the unchanged original. It is not the final accepted thesis "
          "until the authors and the guide approve it. Front-matter placeholders are still to be filled in.\n\n")
txt = BANNER + txt
log.append("review banner")

OUT.write_bytes(txt.replace("\n", "\r\n").encode("utf-8"))      # keep the original's CRLF line endings
print(f"wrote {OUT.relative_to(T.parent)}: {len(log)} edits applied")
for t in log:
    print("  ", t)
