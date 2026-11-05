#!/usr/bin/env python3
"""Phase 4A step 1: fix the bucket-noise model and regimes from TRAINING-image bucket statistics only.

    python phase4a_noise_setup.py

Runs BEFORE any reconstruction or validation evaluation and writes
outputs/phase4a_noisy_baselines/noise_config.json, which the main experiment only reads.

Noise model (per bucket value, normalised units):
    photon scale  s = N_ph / mu_ref           mu_ref = mean clean bucket over training images (seed 2026)
    photons       K ~ Poisson(s * B_clean)
    detector      e ~ Normal(0, sigma_det),   sigma_det = kappa * sqrt(N_ph)  photo-electrons (read noise)
    reading       B_noisy = (K + e) / s       signed reading, NOT clipped (an ADC with offset can report
                                              negative values; their count is recorded)
N_ph = photons collected for a bucket of average brightness ("photons per representative bucket").

Regime rule (fixed here, before any reconstruction is seen):
    kappa: LOW 0.1, MEDIUM 0.5, HIGH 1.0  (detector noise relative to shot noise at the mean bucket)
    N_ph : the half-decade value in {1e2, 3.16e2, ..., 1e8} whose median training-image FLUCTUATION SNR
           (std of clean bucket over patterns / noise std) is closest in log to the target
           LOW 10, MEDIUM 3, HIGH 1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.ghost_imaging import load_object, pattern_batches  # noqa: E402
from src.gi_unet_data import load_pairs, read_split_ids  # noqa: E402

OUT = ROOT / "outputs" / "phase4a_noisy_baselines"
SEED_STATS = 2026
N_PATTERNS = 10000
BUDGETS = [float(10 ** (k / 2)) for k in range(4, 17)]       # half-decades 1e2 .. 1e8
REGIMES = {"LOW": {"kappa": 0.1, "target_fluct_snr": 10.0},
           "MEDIUM": {"kappa": 0.5, "target_fluct_snr": 3.0},
           "HIGH": {"kappa": 1.0, "target_fluct_snr": 1.0}}
USER_SUGGESTED = [1e4, 1e3, 1e2]


def add_noise(b_clean: np.ndarray, n_ph: float, kappa: float, mu_ref: float, rng: np.random.Generator):
    s = n_ph / mu_ref
    k = rng.poisson(b_clean * s).astype(np.float64)
    e = rng.normal(0.0, kappa * np.sqrt(n_ph), size=b_clean.shape)
    return (k + e) / s


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ids = read_split_ids()
    _, _, id_tr = load_pairs("train", 1000)
    assert len(id_tr) == 341 and not set(id_tr) & set(ids["test"])
    obj = np.stack([load_object(ROOT / "data" / "v2" / "reference" / f"{i}.npz", 64)[0] for i in id_tr])
    p = np.concatenate(list(pattern_batches(N_PATTERNS, (64, 64), SEED_STATS, 500, 0.5))).reshape(N_PATTERNS, -1)
    b = p.astype(np.float64) @ obj.reshape(len(obj), -1).T                      # (10000, 341)
    mu_img, sd_img = b.mean(0), b.std(0)
    mu_ref = float(b.mean())
    r_img = sd_img / mu_img

    per_img = pd.DataFrame({"image_id": id_tr, "bucket_mean": mu_img, "bucket_std_over_patterns": sd_img,
                            "relative_fluctuation": r_img})
    per_img.to_csv(OUT / "bucket_statistics_train_per_image.csv", index=False)

    # analytic + empirical SNR for every candidate (budget, kappa)
    rng = np.random.default_rng(12345)
    rows = []
    for name, reg in REGIMES.items():
        for n_ph in BUDGETS:
            kappa = reg["kappa"]
            noise_var = mu_img * mu_ref / n_ph + (kappa ** 2) * mu_ref ** 2 / n_ph   # normalised units
            fl_snr = sd_img / np.sqrt(noise_var)
            dc_snr = mu_img / np.sqrt(noise_var)
            bn = add_noise(b[:2000, :60], n_ph, kappa, mu_ref, rng)                  # empirical check
            emp_sd = (bn - b[:2000, :60]).std(0)
            rows.append({"kappa": kappa, "regime_kappa_of": name, "photons_per_mean_bucket": n_ph,
                         "noise_std_norm_median": float(np.median(np.sqrt(noise_var))),
                         "noise_std_norm_empirical_median": float(np.median(emp_sd)),
                         "dc_snr_median": float(np.median(dc_snr)),
                         "fluct_snr_median": float(np.median(fl_snr)),
                         "fluct_snr_p10": float(np.percentile(fl_snr, 10)),
                         "fluct_snr_p90": float(np.percentile(fl_snr, 90)),
                         "negative_readings_fraction": float((bn < 0).mean())})
    cand = pd.DataFrame(rows)
    cand.to_csv(OUT / "noise_candidates_train.csv", index=False)

    regimes = {}
    for name, reg in REGIMES.items():
        c = cand[cand.regime_kappa_of == name].copy()
        c["dist"] = np.abs(np.log10(c.fluct_snr_median) - np.log10(reg["target_fluct_snr"]))
        best = c.loc[c.dist.idxmin()]
        regimes[name] = {"photons_per_mean_bucket": float(best.photons_per_mean_bucket), "kappa": reg["kappa"],
                         "detector_sigma_photoelectrons": float(reg["kappa"] * np.sqrt(best.photons_per_mean_bucket)),
                         "photon_scale_s": float(best.photons_per_mean_bucket / mu_ref),
                         "target_fluct_snr": reg["target_fluct_snr"],
                         "fluct_snr_median_train": float(best.fluct_snr_median),
                         "fluct_snr_p10_p90_train": [float(best.fluct_snr_p10), float(best.fluct_snr_p90)],
                         "dc_snr_median_train": float(best.dc_snr_median),
                         "noise_std_normalised_median": float(best.noise_std_norm_median)}
    suggested = cand[(cand.kappa == 0.5) & cand.photons_per_mean_bucket.round(6).isin(USER_SUGGESTED)][
        ["photons_per_mean_bucket", "fluct_snr_median", "dc_snr_median"]].to_dict("records")

    cfg = {"forward_model": "B_clean = sum_xy P_i(x,y) O(x,y) (unchanged Phase 2A physics)",
           "noise_model": {"photon_scale": "s = N_ph / mu_ref; mu_ref = mean clean bucket over the 341 training "
                                           "images, seed-2026 patterns (a fixed constant for all images)",
                           "shot_noise": "K ~ Poisson(s * B_clean)",
                           "detector_noise": "e ~ Normal(0, kappa * sqrt(N_ph)) photo-electrons, independent per bucket",
                           "reading": "B_noisy = (K + e) / s, signed, not clipped",
                           "rng": "noise generator = numpy default_rng(SeedSequence([pattern_seed, 4040, regime_index])); "
                                  "patterns from the established pattern_batches(seed); one noise draw for all 10,000 "
                                  "buckets of each (image, seed), so shot-count prefixes are nested"},
           "mu_ref": mu_ref,
           "train_bucket_stats": {"mean_of_image_means": float(mu_img.mean()),
                                  "median_relative_fluctuation": float(np.median(r_img)),
                                  "relative_fluctuation_p10_p90": [float(np.percentile(r_img, 10)),
                                                                   float(np.percentile(r_img, 90))],
                                  "median_bucket_std_over_patterns": float(np.median(sd_img))},
           "regime_rule": {k: v for k, v in REGIMES.items()} | {
               "selection": "half-decade photon budget whose median training fluctuation SNR is closest in log10 to target",
               "revision": "first version used whole decades only; it mapped LOW and MEDIUM onto the same budget (1e5; fluct SNR 5.9 vs 5.3), so the grid was refined to half-decades BEFORE any reconstruction was run"},
           "regimes": regimes,
           "user_suggested_budgets_at_kappa_0.5": suggested,
           "fixed_before_validation": True}
    (OUT / "noise_config.json").write_text(json.dumps(cfg, indent=2))
    pd.set_option("display.width", 200)
    print(cand.round(4).to_string(index=False))
    print(json.dumps({k: cfg[k] for k in ("mu_ref", "train_bucket_stats", "regimes")}, indent=1))


if __name__ == "__main__":
    main()
