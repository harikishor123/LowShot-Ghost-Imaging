"""Check every headline claim of the extension draft / proposed abstract and conclusion against frozen results (read-only)."""
import json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "outputs/extension_medmnist_microscopy/results"; P = ROOT / "outputs/extension_medmnist_microscopy_poisson/results"
J = lambda p: json.loads(Path(p).read_text())
rows = []
def claim(text, ok, evidence):
    rows.append((text, "PASS" if ok else "**FAIL**", evidence))
def mn(base, sp, grp, meth, key="minimum_passing_shot_count"):
    return J(f"{base}/{sp}/evaluation_report.json")[key][grp][meth]
def comp(base, sp, grp, a, b):
    c = pd.read_csv(f"{base}/{sp}/comparisons.csv"); return c[(c.group == grp) & (c.A == a) & (c.B == b)].iloc[0]
O1 = ROOT / "outputs/phase_final_low_shot_to_10k_reference"; O2 = ROOT / "outputs/phase_final_original_target/results"
for sp in ("validation", "test"):
    v = J(O1 / sp / "evaluation_report.json")["minimum_successful_shot_count_AI"]; claim(f"Onion Phase 1 minimum = 1,000 ({sp})", v == 1000, f"evaluation_report: {v}")
for sp in ("val", "test"):
    v = J(O2 / sp / "evaluation_report.json")["minimum_high_fidelity_shot_count"]; claim(f"Onion Phase 2 minimum = 1,000 ({sp})", v == 1000, f"evaluation_report: {v}")
exp = {("target_clean", "AI-original"): {"all": 1000, "blood": 500, "path": 5000},
       ("target_dgi10k", "AI"): {"all": 1000, "blood": 500, "path": 2000},
       ("target_dgi10k", "smoothed DGI"): {"all": 2000, "blood": 1000, "path": 2000},
       ("target_dgi10k", "DGI"): {"all": 5000, "blood": 5000, "path": 5000}}
for (t, m), d in exp.items():
    for sp in ("val", "test"):
        got = {g: mn(E / t, sp, g, m) for g in d}; claim(f"Extension {t}, {m}: all/blood/tissue = {d['all']}/{d['blood']}/{d['path']} ({sp})", got == d, str(got))
pn = {"p05": {"all": 1000, "blood": 500, "path": 5000}, "p10": {"all": 1000, "blood": 500, "path": 2000}, "p30": {"all": 1000, "blood": 500, "path": 2000}}
for l, d in pn.items():
    got = {g: mn(P / f"level_{l}", "test", g, "AI-original") for g in d}; claim(f"Poisson {l}: U-Net vs noisy DGI-10000, all/blood/tissue = {d['all']}/{d['blood']}/{d['path']} (test)", got == d, str(got))
    s = {g: mn(P / f"level_{l}", "test", g, "smoothed DGI") for g in ("all",)}; claim(f"Poisson {l}: smoothed DGI needs 2,000 for all images (test)", s["all"] == 2000, str(s))
nf = {"p05": {"all": 1000, "path": 5000}, "p10": {"all": 1000, "path": 5000}, "p30": {"all": 2000, "path": None}}
for l, d in nf.items():
    got = {g: mn(P / f"level_{l}", "test", g, "AI-original", "minimum_passing_shot_count_vs_noise_free_DGI10000") for g in d}
    claim(f"Poisson {l}: U-Net vs noise-free DGI-10000, all = {d['all']}, tissue = {d['path'] or 'never'} (test)", got == d, str(got))
r = comp(P / "level_p30", "test", "path", "AI-original-1000", "DGI-10000")
claim("Tissue 30%: U-Net-1000 FAILS vs noisy DGI-10000", str(r.criterion_pass) == "False" and r.SSIM_ci_low < -0.02 and r.correlation_ci_low < -0.01,
      f"ΔSSIM {r.SSIM_mean:+.4f} [lo {r.SSIM_ci_low:+.4f}], Δr {r.correlation_mean:+.4f} [lo {r.correlation_ci_low:+.4f}]")
r = comp(P / "level_p30", "test", "path", "AI-original-2000", "DGI-10000")
claim("Tissue 30%: U-Net-2000 PASSES vs noisy DGI-10000", str(r.criterion_pass) == "True", f"lo SSIM {r.SSIM_ci_low:+.4f}, lo r {r.correlation_ci_low:+.4f}")
a = pd.read_csv(P / "level_p30/test/aggregate_metrics.csv").query("group=='path' and seed_group=='all seeds (2026-2029)'").set_index("label")
claim("Tissue 30% means: U-Net-1000 SSIM 0.529 < noisy DGI-10000 0.557", round(a.loc["AI-original-1000"].SSIM_mean, 3) == 0.529 and round(a.loc["DGI-10000"].SSIM_mean, 3) == 0.557,
      f"{a.loc['AI-original-1000'].SSIM_mean:.4f} vs {a.loc['DGI-10000'].SSIM_mean:.4f}")
claim("Reduction factors: 10,000/1,000 = 10×, /500 = 20×, /2,000 = 5×, /5,000 = 2×", (10000/1000, 10000/500, 10000/2000, 10000/5000) == (10, 20, 5, 2), "arithmetic")
agree = tot = 0
for l in ("p05", "p10", "p30"):
    for key in ("minimum_passing_shot_count", "minimum_passing_shot_count_vs_noise_free_DGI10000"):
        v, t = J(P / f"level_{l}/val/evaluation_report.json")[key], J(P / f"level_{l}/test/evaluation_report.json")[key]
        for g in ("all", "blood", "path"):
            for m in t[g]:
                tot += 1; agree += v[g][m] == t[g][m]
claim("Poisson validation vs test minimum shots agree in 25 of 27 cases", (agree, tot) == (25, 27), f"{agree} of {tot}")
for l, g, a_, b_, val_lo, test_lo in (("p10", "all", "AI-original-1000", "noise-free DGI-10000", -0.0101, -0.0042), ("p10", "path", "AI-original-2000", "DGI-10000", -0.0151, -0.0095)):
    v, t = comp(P / f"level_{l}", "val", g, a_, b_).correlation_ci_low, comp(P / f"level_{l}", "test", g, a_, b_).correlation_ci_low
    claim(f"Borderline {l} {g} {a_} vs {b_}: r lower bound val {val_lo} / test {test_lo}", round(v, 4) == val_lo and round(t, 4) == test_lo, f"val {v:+.4f}, test {t:+.4f}")
b4 = pd.read_csv(ROOT / "audit/final_verification/B4_smoothness_specificity_val.csv")
g = lambda l, n, c: float(b4[(b4.level == l) & (b4.N == n)][c].iloc[0])
claim("Over-smoothing: fine-detail power kept at 1,000 shots ≈ 1/3 blood, ≈ 1/6 tissue (≤ 10% noise)",
      all(0.30 <= g(l, 1000, "blood_HF_power_ratio_16_31") <= 0.37 and 0.13 <= g(l, 1000, "path_HF_power_ratio_16_31") <= 0.17 for l in ("p00", "p05", "p10")),
      ", ".join(f"{l}: {g(l,1000,'blood_HF_power_ratio_16_31'):.3f}/{g(l,1000,'path_HF_power_ratio_16_31'):.3f}" for l in ("p00", "p05", "p10")))
out = ["# Claims verification (2026-11-05)", "", "Every headline claim of the corrected extension draft and of the proposed Abstract / Conclusion, checked against the frozen evaluation reports and comparison CSVs by `thesis/merge_proposal/verify_claims.py` (read-only).", "",
       "| # | Claim | Result | Evidence from frozen files |", "|---|---|---|---|"] + [f"| {k} | {c} | {r} | {e} |" for k, (c, r, e) in enumerate(rows, 1)]
out += ["", f"**{sum(r == 'PASS' for _, r, _ in rows)} of {len(rows)} claims confirmed.**"]
(Path(__file__).parent / "CLAIMS_VERIFICATION.md").write_text("\n".join(out) + "\n"); print("\n".join(out[-1:])); print("\n".join(f"{r}: {c}" for c, r, e in rows if r != "PASS"))
