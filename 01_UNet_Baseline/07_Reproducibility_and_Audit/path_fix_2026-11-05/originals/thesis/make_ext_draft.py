"""Fill the thesis-draft template with numbers read directly from the frozen extension CSV/JSON outputs.

Every {{expression}} in the template is evaluated against the helpers below; nothing is typed by hand.
Usage: python make_ext_draft.py TEMPLATE OUT
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/home/ss_students/Downloads/Onion-CNN-Final.v1-onion-original-v1.yolov8")
EXT = ROOT / "outputs/extension_medmnist_microscopy"
POI = ROOT / "outputs/extension_medmnist_microscopy_poisson"
LOW = [500, 1000, 2000, 5000]
LV = {"p05": "5%", "p10": "10%", "p30": "30%"}
SG = "all seeds (2026-2029)"


def D(kind, split="test"):
    """kind: 'clean' | 'dgi10k' | 'p05' | 'p10' | 'p30'"""
    if kind in ("clean", "dgi10k"):
        return EXT / "results" / f"target_{kind}" / split
    return POI / "results" / f"level_{kind}" / split


_cache = {}


def _csv(p):
    if p not in _cache:
        _cache[p] = pd.read_csv(p)
    return _cache[p]


def _json(p):
    return json.loads(Path(p).read_text())


def A(kind, grp, label, met="SSIM", stat="mean", split="test"):
    t = _csv(D(kind, split) / "aggregate_metrics.csv")
    r = t[(t.group == grp) & (t.seed_group == SG) & (t.label == label)]
    assert len(r) == 1, (kind, grp, label)
    return float(r.iloc[0][f"{met}_{stat}"])


def C(kind, grp, a, b, met="SSIM", f="mean", split="test"):
    t = _csv(D(kind, split) / "comparisons.csv")
    r = t[(t.group == grp) & (t.A == a) & (t.B == b)]
    assert len(r) == 1, (kind, grp, a, b)
    return r.iloc[0][f"{met}_{f}"] if f != "pass" else r.iloc[0]["criterion_pass"]


def P(kind, grp, a, b, split="test"):
    v = C(kind, grp, a, b, f="pass", split=split)
    return "**Pass**" if str(v) == "True" else "Fail"


def MINN(kind, grp, meth, split="test", key="minimum_passing_shot_count"):
    return _json(D(kind, split) / "evaluation_report.json")[key][grp][meth]


def n(x):
    return "none ≤ 5,000" if x is None else f"{x:,}"


def f(x, dp=3):
    return f"{x:.{dp}f}"


def s(x, dp=3):
    return f"{x:+.{dp}f}".replace("-", "−")


def ms(kind, grp, label, met, dp=3):
    return f"{f(A(kind, grp, label, met), dp)} ± {f(A(kind, grp, label, met, 'std'), dp)}"


def ci(kind, grp, a, b, met="SSIM", dp=3, pct=True):
    m, lo, hi = (C(kind, grp, a, b, met, k) for k in ("mean", "ci_low", "ci_high"))
    out = f"{s(m, dp)} [{s(lo, dp)}, {s(hi, dp)}]"
    if pct:
        out += f" ({C(kind, grp, a, b, met, 'pct_A_better'):.0f}%)"
    return out


def B4(level, N, col, dp=2, pct=False):
    """Validation-set smoothness / small-structure statistics (audit/final_verification, frozen checkpoints, read-only)."""
    t = _csv(ROOT / "audit/final_verification/B4_smoothness_specificity_val.csv")
    v = float(t[(t.level == level) & (t.N == N)][col].iloc[0])
    return f"{v * 100:.0f}%" if pct else f"{v:.{dp}f}"


def MINS(kind, grp, meth="AI-original", key="minimum_passing_shot_count"):
    return n(MINN(kind, grp, meth, key=key))


def TR(path):
    return _json(path)


def best_epochs(models_dir, pattern):
    return " / ".join(str(_json(models_dir / pattern.format(N=N))["best_epoch"]) for N in LOW)


GN = {"all": "All", "blood": "Blood", "path": "Path"}


# ---------------------------------------------------------------- tables
def T_EXT_CLEAN():
    rows = ["| Method | Shots | SSIM | Pearson *r* | PSNR (dB) | MSE | Edge *r* |", "|---|---:|---|---|---|---|---|"]
    for N in LOW:
        for lab, nm in ((f"DGI-{N}", f"Conventional DGI-{N}"), (f"AI-original-{N}", f"**AI-original-{N}**")):
            rows.append(f"| {nm} | {N:,} | {ms('clean','all',lab,'SSIM')} | {ms('clean','all',lab,'correlation')} | "
                        f"{ms('clean','all',lab,'PSNR',2)} | {ms('clean','all',lab,'MSE',4)} | {ms('clean','all',lab,'edge_r')} |")
    lab = "DGI-10000"
    rows.append(f"| Conventional 10k DGI baseline (DGI-10000) | 10,000 | {ms('clean','all',lab,'SSIM')} | {ms('clean','all',lab,'correlation')} | "
                f"{ms('clean','all',lab,'PSNR',2)} | {ms('clean','all',lab,'MSE',4)} | {ms('clean','all',lab,'edge_r')} |")
    return "\n".join(rows)


def T_EXT_SUB():
    rows = ["| Method | Blood SSIM | Blood *r* | Blood PSNR (dB) | Path SSIM | Path *r* | Path PSNR (dB) |", "|---|---|---|---|---|---|---|"]
    for lab in ["DGI-10000"] + [f"AI-original-{N}" for N in LOW]:
        c = [f(A("clean", g, lab, m), 2 if m == "PSNR" else 3) for g in ("blood", "path") for m in ("SSIM", "correlation", "PSNR")]
        rows.append(f"| {'Conventional DGI-10000' if lab == 'DGI-10000' else lab} | " + " | ".join(c) + " |")
    return "\n".join(rows)


def T_EXT_CRIT():
    rows = ["| Images | *N* | ΔSSIM | Δ*r* | ΔPSNR (dB) | ΔEdge *r* | Criterion |", "|---|---:|---|---|---|---|---|"]
    for g in ("all", "blood", "path"):
        for N in LOW:
            a = f"AI-original-{N}"
            rows.append(f"| {GN[g]} | {N:,} | {ci('clean',g,a,'DGI-10000','SSIM')} | {ci('clean',g,a,'DGI-10000','correlation')} | "
                        f"{ci('clean',g,a,'DGI-10000','PSNR',2)} | {ci('clean',g,a,'DGI-10000','edge_r')} | {P('clean',g,a,'DGI-10000')} |")
    return "\n".join(rows)


def T_EXT_D10():
    B = "independent 10k-10000"
    rows = ["| Method | *N* | ΔSSIM (all) | Δ*r* (all) | All | Blood | Path |", "|---|---:|---|---|---|---|---|"]
    for N in LOW:
        for m, nm in (("DGI", "Conventional DGI-{}"), ("smoothed DGI", "Gaussian-smoothed DGI-{}"), ("AI", "**AI-{}**")):
            a = f"{m}-{N}"
            rows.append(f"| {nm.format(N)} | {N:,} | {ci('dgi10k','all',a,B,'SSIM',pct=False)} | {ci('dgi10k','all',a,B,'correlation',pct=False)} | "
                        f"{P('dgi10k','all',a,B)} | {P('dgi10k','blood',a,B)} | {P('dgi10k','path',a,B)} |")
    return "\n".join(rows)


def T_EXT_MIN():
    rows = ["| Training target / comparator | Method | All images | Blood | Path |", "|---|---|---|---|---|"]
    spec = [("clean", "Clean object / conventional DGI-10000", "AI-original"),
            ("dgi10k", "10k DGI / independent 10k acquisition", "AI"),
            ("dgi10k", "— / independent 10k acquisition", "smoothed DGI"),
            ("dgi10k", "— / independent 10k acquisition", "DGI")]
    for k, t, m in spec:
        cells = [f"{n(MINN(k, g, m))} ({n(MINN(k, g, m, 'val'))})" for g in ("all", "blood", "path")]
        rows.append(f"| {t} | {m if m != 'DGI' else 'conventional DGI'} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def T_P_AGG():
    hdr = "| Method | 0% (noise-free extension) | " + " | ".join(LV.values()) + " |"
    rows = [hdr, "|---|---|---|---|---|"]

    def cell(kind, lab):
        try:
            return f"{f(A(kind,'all',lab,'SSIM'))} / {f(A(kind,'all',lab,'correlation'))}"
        except AssertionError:
            return "—"
    spec = [("noise-free DGI-10000", "DGI-10000", "Noise-free DGI-10000"),
            ("DGI-10000", "DGI-10000", "Noisy DGI-10000 (comparator)"),
            ("DGI-1000", "DGI-1000", "Noisy DGI-1000"),
            ("smoothed DGI-1000", None, "Smoothed noisy DGI-1000"),
            ("smoothed DGI-2000", None, "Smoothed noisy DGI-2000")] + \
           [(f"AI-original-{N}", f"AI-original-{N}", f"**AI-original-{N}**") for N in LOW]
    for plab, clab, nm in spec:
        c0 = cell("clean", clab) if clab else "—"
        rows.append(f"| {nm} | {c0} | " + " | ".join(cell(k, plab) for k in LV) + " |")
    return "\n".join(rows)


def T_P_DEG():
    rows = ["| Noise level | ΔSSIM all | Δ*r* all | ΔSSIM blood | Δ*r* blood | ΔSSIM path | Δ*r* path |", "|---|---|---|---|---|---|---|"]
    for k, nm in LV.items():
        c = [s(C(k, g, "DGI-10000", "noise-free DGI-10000", m)) for g in ("all", "blood", "path") for m in ("SSIM", "correlation")]
        rows.append(f"| {nm} | " + " | ".join(c) + " |")
    return "\n".join(rows)


def T_P_CRIT():
    rows = ["| Noise | *N* | ΔSSIM vs noisy DGI-10000 | Δ*r* vs noisy DGI-10000 | Criterion | ΔSSIM vs noise-free DGI-10000 | Δ*r* vs noise-free DGI-10000 | Criterion |",
            "|---|---:|---|---|---|---|---|---|"]
    for k, nm in LV.items():
        for N in LOW:
            a = f"AI-original-{N}"
            rows.append(f"| {nm} | {N:,} | {ci(k,'all',a,'DGI-10000','SSIM',pct=False)} | {ci(k,'all',a,'DGI-10000','correlation',pct=False)} | {P(k,'all',a,'DGI-10000')} | "
                        f"{ci(k,'all',a,'noise-free DGI-10000','SSIM',pct=False)} | {ci(k,'all',a,'noise-free DGI-10000','correlation',pct=False)} | {P(k,'all',a,'noise-free DGI-10000')} |")
    return "\n".join(rows)


def T_P_MIN():
    rows = ["| Noise | Method / comparator | All images | Blood | Path |", "|---|---|---|---|---|"]
    for k, nm in LV.items():
        for m, key, t in (("AI-original", "minimum_passing_shot_count", "AI-original / noisy DGI-10000"),
                          ("smoothed DGI", "minimum_passing_shot_count", "Smoothed DGI / noisy DGI-10000"),
                          ("AI-original", "minimum_passing_shot_count_vs_noise_free_DGI10000", "AI-original / noise-free DGI-10000")):
            vkey = key if key != "minimum_passing_shot_count_vs_noise_free_DGI10000" else key
            cells = [f"{n(MINN(k, g, m, key=key))} ({n(MINN(k, g, m, 'val', key=vkey))})" for g in ("all", "blood", "path")]
            rows.append(f"| {nm} | {t} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def T_P_TRAIN():
    rows = ["| Noise | Best epoch (500 / 1,000 / 2,000 / 5,000) | Stopped early | Selection SSIM of AI-original-1000 | Selection SSIM of noisy DGI-1000 |", "|---|---|---|---|---|"]
    for k, nm in LV.items():
        reps = [_json(POI / f"models/level_{k}/model_{N}/training_report.json") for N in LOW]
        se = sum(r["stopped_early"] for r in reps)
        rows.append(f"| {nm} | {' / '.join(str(r['best_epoch']) for r in reps)} | {se} of 4 | "
                    f"{f(reps[1]['selection_at_best']['SSIM'])} | {f(reps[1]['noisy_dgi_on_selection']['SSIM'])} |")
    return "\n".join(rows)


def T_PROV():
    fe_c, fe_d = _json(EXT / "results/target_clean/frozen_for_test.json"), _json(EXT / "results/target_dgi10k/frozen_for_test.json")
    rows = ["| Item | Value |", "|---|---|",
            f"| Protocol (written before pool and training) | `outputs/extension_medmnist_microscopy/protocol.json`, {_json(EXT/'protocol.json')['written_before_pool_training_or_evaluation']} |",
            f"| Data | MedMNIST+ 64 × 64, Zenodo DOI {_json(EXT/'dataset/dataset_report.json')['sources']['zenodo_doi']} (v{_json(EXT/'dataset/dataset_report.json')['sources']['record_version']}, CC BY 4.0); objects file MD5 `{_json(EXT/'dataset/dataset_report.json')['objects_npz_md5']}` |",
            "| Checkpoint MD5, clean target (500 / 1000 / 2000 / 5000) | " + " / ".join(f"`{fe_c['models'][str(N)]['md5']}`" for N in LOW) + f" (best epochs {' / '.join(str(fe_c['models'][str(N)]['best_epoch']) for N in LOW)}) |",
            "| Checkpoint MD5, 10k-DGI target (500 / 1000 / 2000 / 5000) | " + " / ".join(f"`{fe_d['models'][str(N)]['md5']}`" for N in LOW) + f" (best epochs {' / '.join(str(fe_d['models'][str(N)]['best_epoch']) for N in LOW)}) |",
            f"| Frozen after validation | {fe_c['frozen_at']} (clean), {fe_d['frozen_at']} (10k DGI); smoothing σ {' / '.join(str(fe_d['smoothing_sigma'][str(N)]) for N in LOW)} |",
            "| Test runs | 1 per target (`ext_evaluate.py --target {clean,dgi10k} --split test --final-test`; the script refuses a second run) |",
            "| Freeze manifest | `outputs/extension_medmnist_microscopy/FREEZE_MANIFEST_extension.md5` (131 files) |",
            f"| Poisson protocol | `outputs/extension_medmnist_microscopy_poisson/protocol.json`, {_json(POI/'protocol.json')['written_before_pool_training_or_evaluation']} |"]
    for k, nm in LV.items():
        fr = _json(POI / f"results/level_{k}/frozen_for_test.json")
        rows.append(f"| Poisson {nm}: checkpoint MD5 (500 / 1000 / 2000 / 5000) | " + " / ".join(f"`{fr['models'][str(N)]['md5']}`" for N in LOW)
                    + f"; σ {' / '.join(str(fr['smoothing_sigma'][str(N)]) for N in LOW)}; frozen {fr['frozen_at']} |")
    rows += ["| Poisson test runs | 1 per noise level (`ext_poisson_evaluate.py --level {p05,p10,p30} --split test --final-test`) |",
             "| Poisson freeze manifest | `outputs/extension_medmnist_microscopy_poisson/FREEZE_MANIFEST_poisson.md5` (180 files) |",
             f"| Integrity checks | noise-free path of the Poisson code reproduced the frozen extension pool exactly (max difference {max(_json(POI/'pool/pool_meta.json')['checks']['noise_free_path_vs_frozen_extension_pool_max_abs_diff'])}); "
             f"pool spot checks {max(_json(EXT/'pool/pool_meta.json')['checks']['pool_spot_checks_max_abs_diff'])} / {max(_json(POI/'pool/pool_meta.json')['checks']['pool_spot_checks_max_abs_diff'])}; "
             "no test image in training or selection; onion (134 files) and extension (131 files) manifests verified before and after |"]
    return "\n".join(rows)


def main():
    tpl, out = Path(sys.argv[1]).read_text(), Path(sys.argv[2])
    ns = dict(globals())
    txt = re.sub(r"\{\{(.+?)\}\}", lambda m: str(eval(m.group(1), ns)), tpl, flags=re.S)
    assert "{{" not in txt
    out.write_text(txt)
    print(f"wrote {out} ({len(txt.splitlines())} lines)")


if __name__ == "__main__":
    main()
