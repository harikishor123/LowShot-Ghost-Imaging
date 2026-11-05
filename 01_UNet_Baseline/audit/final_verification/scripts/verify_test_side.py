"""PART A (test images; NO model is run): IDs, exact DGI recomputation, panel-label identification, copy checks,
independent metrics. Read-only on frozen data; writes only audit/final_verification/."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from skimage.metrics import structural_similarity

HERE = Path(__file__).resolve().parent; OUT = HERE.parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "audit/poisson_visual_analysis/scripts")]
import ext_poisson_common as C
from run_generalization_validation import smooth
from panel_decode import load_all, ROW_IDS, LOW
t0 = time.time(); R = {}
SH = [500, 1000, 2000, 5000, 10000]
LV = {"p00": 0.0, "p05": 0.05, "p10": 0.10, "p30": 0.30}
P = load_all()
allobj = {s: C.load_objects((s,)) for s in ("train", "val", "test")}
te = allobj["test"]; tid = list(te["image_id"]); tobj = te["object"]
everything = np.concatenate([allobj[s]["object"] for s in ("train", "val", "test")]).reshape(-1, 4096)
every_id = sum([list(allobj[s]["image_id"]) for s in ("train", "val", "test")], [])

def ncc(a, b):   # zero-mean normalised cross-correlation (= Pearson r)
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean(); return float((a @ b) / np.sqrt((a @ a) * (b @ b)))
def mets(ref, x):
    ref = ref.astype(np.float64); x = x.astype(np.float64); mse = float(((ref - x) ** 2).mean())
    return {"SSIM": float(structural_similarity(ref, x, data_range=1.0)), "PSNR": 10 * np.log10(1.0 / mse), "MSE": mse, "NCC": ncc(ref, x)}

# A1: which stored image does each decoded "Original" panel match?
a1 = {}
for l in LV:
    for i in ROW_IDS:
        d = np.abs(everything - P[(l, i, "Original clean object")].ravel()).max(1)
        k = int(np.argmin(d)); a1[f"{l} {i}"] = {"nearest_id": every_id[k], "max_abs_diff": float(d[k]),
                                                 "second_best_max_abs_diff": float(np.partition(d, 1)[1])}
R["A1_original_panel_identity"] = {"all_match_own_id": all(v["nearest_id"] == k.split()[1] for k, v in a1.items()),
                                   "worst_own_diff": max(v["max_abs_diff"] for v in a1.values()),
                                   "smallest_runner_up_diff": min(v["second_best_max_abs_diff"] for v in a1.values())}
print("A1", R["A1_original_panel_identity"], flush=True)

# A2: exact DGI for all 2000 test images (seed 2026, all levels jointly as in evaluation)
dg = C.noisy_dgi_readouts(tobj, 2026, list(LV.values()))           # (4, 2000, 5, 64, 64) float32
print(f"A2 recomputed test DGI ({time.time()-t0:.0f}s)", flush=True)
rows = [i for i in ROW_IDS]; J = [tid.index(i) for i in rows]
cand = {(l, n): dg[li, :, SH.index(n)] for li, l in enumerate(LV) for n in SH}
lab_ok, worst = [], 0.0
for l in LV:
    for i, j in zip(rows, J):
        labs = [(f"DGI-{n}", l, n) for n in SH if not (l != "p00" and n == 10000 and False)]
        for lab, ll, n in labs:
            if (l, i, lab) not in P: continue
            panel = P[(l, i, lab)]
            d = {k: float(np.abs(v[j] - panel).max()) for k, v in cand.items()}
            best = min(d, key=d.get); lab_ok.append(best == (ll, n)); worst = max(worst, d[(ll, n)])
        if l != "p00":
            panel = P[(l, i, "noise-free DGI-10000")]
            d = {k: float(np.abs(v[j] - panel).max()) for k, v in cand.items()}
            best = min(d, key=d.get); lab_ok.append(best == ("p00", 10000)); worst = max(worst, d[("p00", 10000)])
R["A2_DGI_panels"] = {"panels_checked": len(lab_ok), "best_match_is_labelled_level_and_shots": int(sum(lab_ok)),
                      "max_abs_diff_to_exact_recomputation": worst}
print("A2", R["A2_DGI_panels"], flush=True)

# smoothed panels: recompute with frozen sigma
sm_ok, sm_worst = [], 0.0
for l in ("p05", "p10", "p30"):
    sig = {int(k): v for k, v in json.loads((ROOT / f"outputs/extension_medmnist_microscopy_poisson/results/level_{l}/frozen_for_test.json").read_text())["smoothing_sigma"].items()}
    for i, j in zip(rows, J):
        for n in LOW:
            s = smooth(cand[(l, n)][j][None, None], sig[n])[0, 0]
            dd = float(np.abs(s - P[(l, i, f"smoothed DGI-{n}")]).max()); sm_worst = max(sm_worst, dd); sm_ok.append(dd < 0.01)
R["A2_smoothed_panels"] = {"panels_checked": len(sm_ok), "match_within_8bit": int(sum(sm_ok)), "max_abs_diff": sm_worst}
print("A2s", R["A2_smoothed_panels"], flush=True)

# A3: U-Net panel label identification via frozen CSV metrics
CSV = {"p00": ROOT / "outputs/extension_medmnist_microscopy/results/target_clean/test"}
CSV.update({l: ROOT / f"outputs/extension_medmnist_microscopy_poisson/results/level_{l}/test" for l in ("p05", "p10", "p30")})
per = {l: pd.read_csv(d / "per_image_results.csv").query("seed == 2026").set_index(["image_id", "label"]) for l, d in CSV.items()}
a3, a3d = [], []
for l in LV:
    for i, j in zip(rows, J):
        for n in LOW:
            m = mets(tobj[j], P[(l, i, f"AI-original-{n}")])
            # candidates: every AI-original-N of this image at every level
            cands = {(ll, nn): abs(per[ll].loc[(i, f"AI-original-{nn}")].SSIM - m["SSIM"]) + abs(per[ll].loc[(i, f"AI-original-{nn}")].PSNR - m["PSNR"]) / 50
                     for ll in LV for nn in LOW}
            best = min(cands, key=cands.get); a3.append(best == (l, n))
            r = per[l].loc[(i, f"AI-original-{n}")]
            a3d.append({"level": l, "image": i, "N": n, "SSIM_decoded": m["SSIM"], "SSIM_csv": r.SSIM, "PSNR_decoded": m["PSNR"], "PSNR_csv": r.PSNR,
                        "NCC_decoded": m["NCC"], "r_csv": r.correlation, "best_match": f"{best[0]} N={best[1]}"})
a3d = pd.DataFrame(a3d); a3d.to_csv(OUT / "A3_unet_panels_decoded_vs_csv.csv", index=False)
R["A3_unet_panel_labels"] = {"panels": len(a3), "uniquely_matched_to_labelled_level_and_N": int(sum(a3)),
                             "max_abs_SSIM_diff": float((a3d.SSIM_decoded - a3d.SSIM_csv).abs().max()),
                             "max_abs_PSNR_diff": float((a3d.PSNR_decoded - a3d.PSNR_csv).abs().max()),
                             "max_abs_NCC_vs_r_diff": float((a3d.NCC_decoded - a3d.r_csv).abs().max())}
print("A3", R["A3_unet_panel_labels"], flush=True)

# A4: U-Net output is not a copy of any stored image
a4 = []
for l in LV:
    for i, j in zip(rows, J):
        for n in LOW:
            u = P[(l, i, f"AI-original-{n}")].ravel()
            d = np.abs(everything - u).max(1); c = (everything - everything.mean(1, keepdims=True)) @ (u - u.mean())
            c /= np.linalg.norm(everything - everything.mean(1, keepdims=True), axis=1) * np.linalg.norm(u - u.mean())
            a4.append({"own_corr": float(c[every_id.index(i)]), "best_corr_id": every_id[int(np.argmax(c))],
                       "best_other_corr": float(np.max(np.delete(c, every_id.index(i)))), "min_max_abs_diff_any_image": float(d.min())})
a4 = pd.DataFrame(a4)
R["A4_copy_check"] = {"outputs": len(a4), "most_correlated_stored_image_is_own_original": int((a4.best_corr_id == [i for l in LV for i in rows for n in LOW]).sum()),
                      "min_over_outputs_of_max_abs_diff_to_ANY_of_10000_images": float(a4.min_max_abs_diff_any_image.min()),
                      "mean_own_corr": float(a4.own_corr.mean()), "mean_best_other_corr": float(a4.best_other_corr.mean())}
print("A4", R["A4_copy_check"], flush=True)

# A5: independent metrics on exact float DGI vs frozen CSV, plus U-Net-1000 vs DGI-10000
a5 = []
for l in LV:
    li = list(LV).index(l)
    for i, j in zip(rows, J):
        for lab, arr in (("DGI-1000", dg[li, j, 1]), ("DGI-10000", dg[li, j, 4])):
            m = mets(tobj[j], arr); r = per[l].loc[(i, lab)]
            a5.append({"level": l, "image": i, "pair": f"original vs {lab}", **{k: m[k] for k in m}, "SSIM_csv": r.SSIM, "PSNR_csv": r.PSNR, "MSE_csv": r.MSE, "r_csv": r.correlation})
        u = P[(l, i, "AI-original-1000")]; m = mets(tobj[j], u); r = per[l].loc[(i, "AI-original-1000")]
        a5.append({"level": l, "image": i, "pair": "original vs U-Net-1000 (decoded)", **m, "SSIM_csv": r.SSIM, "PSNR_csv": r.PSNR, "MSE_csv": r.MSE, "r_csv": r.correlation})
        m = mets(dg[li, j, 4], u)
        a5.append({"level": l, "image": i, "pair": "DGI-10000 (ref) vs U-Net-1000 (decoded)", **m})
a5 = pd.DataFrame(a5); a5.to_csv(OUT / "A5_independent_metrics.csv", index=False)
ex = a5[a5.pair.str.startswith("original vs DGI")]
R["A5_exact_DGI_metrics_vs_csv_max_abs_diff"] = {k: float((ex[k] - ex[c]).abs().max()) for k, c in (("SSIM", "SSIM_csv"), ("PSNR", "PSNR_csv"), ("MSE", "MSE_csv"), ("NCC", "r_csv"))}
un = a5[a5.pair.str.startswith("original vs U-Net")]
R["A5_decoded_unet_metrics_vs_csv_max_abs_diff"] = {k: float((un[k] - un[c]).abs().max()) for k, c in (("SSIM", "SSIM_csv"), ("PSNR", "PSNR_csv"), ("MSE", "MSE_csv"), ("NCC", "r_csv"))}
print("A5", R["A5_exact_DGI_metrics_vs_csv_max_abs_diff"], R["A5_decoded_unet_metrics_vs_csv_max_abs_diff"], flush=True)
np.savez_compressed(OUT / "exact_test_dgi_six_images.npz", ids=np.array(rows), levels=np.array(list(LV)), shots=np.array(SH), dgi=dg[:, J])
(OUT / "A_test_side.json").write_text(json.dumps(R, indent=1))
print(f"done {time.time()-t0:.0f}s")
