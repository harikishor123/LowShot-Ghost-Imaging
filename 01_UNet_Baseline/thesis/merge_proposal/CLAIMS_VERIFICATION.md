# Claims verification (2026-11-05)

Every headline claim of the corrected extension draft and of the proposed Abstract / Conclusion, checked against the frozen evaluation reports and comparison CSVs by `thesis/merge_proposal/verify_claims.py` (read-only).

| # | Claim | Result | Evidence from frozen files |
|---|---|---|---|
| 1 | Onion Phase 1 minimum = 1,000 (validation) | PASS | evaluation_report: 1000 |
| 2 | Onion Phase 1 minimum = 1,000 (test) | PASS | evaluation_report: 1000 |
| 3 | Onion Phase 2 minimum = 1,000 (val) | PASS | evaluation_report: 1000 |
| 4 | Onion Phase 2 minimum = 1,000 (test) | PASS | evaluation_report: 1000 |
| 5 | Extension target_clean, AI-original: all/blood/tissue = 1000/500/5000 (val) | PASS | {'all': 1000, 'blood': 500, 'path': 5000} |
| 6 | Extension target_clean, AI-original: all/blood/tissue = 1000/500/5000 (test) | PASS | {'all': 1000, 'blood': 500, 'path': 5000} |
| 7 | Extension target_dgi10k, AI: all/blood/tissue = 1000/500/2000 (val) | PASS | {'all': 1000, 'blood': 500, 'path': 2000} |
| 8 | Extension target_dgi10k, AI: all/blood/tissue = 1000/500/2000 (test) | PASS | {'all': 1000, 'blood': 500, 'path': 2000} |
| 9 | Extension target_dgi10k, smoothed DGI: all/blood/tissue = 2000/1000/2000 (val) | PASS | {'all': 2000, 'blood': 1000, 'path': 2000} |
| 10 | Extension target_dgi10k, smoothed DGI: all/blood/tissue = 2000/1000/2000 (test) | PASS | {'all': 2000, 'blood': 1000, 'path': 2000} |
| 11 | Extension target_dgi10k, DGI: all/blood/tissue = 5000/5000/5000 (val) | PASS | {'all': 5000, 'blood': 5000, 'path': 5000} |
| 12 | Extension target_dgi10k, DGI: all/blood/tissue = 5000/5000/5000 (test) | PASS | {'all': 5000, 'blood': 5000, 'path': 5000} |
| 13 | Poisson p05: U-Net vs noisy DGI-10000, all/blood/tissue = 1000/500/5000 (test) | PASS | {'all': 1000, 'blood': 500, 'path': 5000} |
| 14 | Poisson p05: smoothed DGI needs 2,000 for all images (test) | PASS | {'all': 2000} |
| 15 | Poisson p10: U-Net vs noisy DGI-10000, all/blood/tissue = 1000/500/2000 (test) | PASS | {'all': 1000, 'blood': 500, 'path': 2000} |
| 16 | Poisson p10: smoothed DGI needs 2,000 for all images (test) | PASS | {'all': 2000} |
| 17 | Poisson p30: U-Net vs noisy DGI-10000, all/blood/tissue = 1000/500/2000 (test) | PASS | {'all': 1000, 'blood': 500, 'path': 2000} |
| 18 | Poisson p30: smoothed DGI needs 2,000 for all images (test) | PASS | {'all': 2000} |
| 19 | Poisson p05: U-Net vs noise-free DGI-10000, all = 1000, tissue = 5000 (test) | PASS | {'all': 1000, 'path': 5000} |
| 20 | Poisson p10: U-Net vs noise-free DGI-10000, all = 1000, tissue = 5000 (test) | PASS | {'all': 1000, 'path': 5000} |
| 21 | Poisson p30: U-Net vs noise-free DGI-10000, all = 2000, tissue = never (test) | PASS | {'all': 2000, 'path': None} |
| 22 | Tissue 30%: U-Net-1000 FAILS vs noisy DGI-10000 | PASS | ΔSSIM -0.0278 [lo -0.0423], Δr -0.0606 [lo -0.0679] |
| 23 | Tissue 30%: U-Net-2000 PASSES vs noisy DGI-10000 | PASS | lo SSIM +0.0602, lo r +0.0152 |
| 24 | Tissue 30% means: U-Net-1000 SSIM 0.529 < noisy DGI-10000 0.557 | PASS | 0.5290 vs 0.5568 |
| 25 | Reduction factors: 10,000/1,000 = 10×, /500 = 20×, /2,000 = 5×, /5,000 = 2× | PASS | arithmetic |
| 26 | Poisson validation vs test minimum shots agree in 25 of 27 cases | PASS | 25 of 27 |
| 27 | Borderline p10 all AI-original-1000 vs noise-free DGI-10000: r lower bound val -0.0101 / test -0.0042 | PASS | val -0.0101, test -0.0042 |
| 28 | Borderline p10 path AI-original-2000 vs DGI-10000: r lower bound val -0.0151 / test -0.0095 | PASS | val -0.0151, test -0.0095 |
| 29 | Over-smoothing: fine-detail power kept at 1,000 shots ≈ 1/3 blood, ≈ 1/6 tissue (≤ 10% noise) | PASS | p00: 0.352/0.152, p05: 0.347/0.149, p10: 0.344/0.150 |

**29 of 29 claims confirmed.**
