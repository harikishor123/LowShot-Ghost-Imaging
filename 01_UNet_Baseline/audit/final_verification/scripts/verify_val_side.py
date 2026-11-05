"""PART B (VALIDATION images only; frozen checkpoints run read-only, nothing frozen is written).
B1 decoded val-figure panels == model(recomputed DGI input); B2 model metrics == frozen val CSV for all 1,000 images;
B3 wrong-checkpoint control; B4 smoothness / input-dependence / small-structure analysis."""
import os; os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import ndimage
from skimage.metrics import structural_similarity
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; OUT = HERE.parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "audit/poisson_visual_analysis/scripts")]
import ext_poisson_common as C
import panel_decode as PD
from src.unet_gi import UNetGI
t0 = time.time(); R = {}
torch.backends.cudnn.benchmark = False; torch.use_deterministic_algorithms(True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
LV = {"p00": 0.0, "p05": 0.05, "p10": 0.10, "p30": 0.30}; LOW = [500, 1000, 2000, 5000]; SH = LOW + [10000]
va = C.load_objects(("val",)); vid = list(va["image_id"]); vobj = va["object"]; sub = va["subset"]
tr = C.load_objects(("train",))
dg = C.noisy_dgi_readouts(vobj, 2026, list(LV.values()))            # (4, 1000, 5, 64, 64)
print(f"val DGI recomputed {time.time()-t0:.0f}s", flush=True)

def ckpt(l, n):
    return (ROOT / "outputs/extension_medmnist_microscopy/models/target_clean" if l == "p00" else
            ROOT / f"outputs/extension_medmnist_microscopy_poisson/models/level_{l}") / f"model_{n}/checkpoints/best.pt"

@torch.no_grad()
def run(l, n, x):
    ck = torch.load(ckpt(l, n), weights_only=False, map_location=dev)
    m = UNetGI(base=ck["config"]["base"]).to(dev); m.load_state_dict(ck["model"]); m.eval()
    out = np.concatenate([m(torch.from_numpy(x[i:i + 250, None].astype(np.float32)).to(dev)).cpu().numpy() for i in range(0, len(x), 250)])
    return np.clip(out, 0, 1)[:, 0]

U = {(l, n): run(l, n, dg[li, :, SH.index(n)]) for li, l in enumerate(LV) for n in LOW}
print(f"inference done {time.time()-t0:.0f}s", flush=True)

# B1: decoded validation figures vs fresh model outputs
VAL_IDS = json.loads((ROOT / "outputs/extension_medmnist_microscopy_poisson/results/level_p05/val/evaluation_report.json").read_text())["example_ids"]
PD.ROW_IDS = VAL_IDS
dec = {}
for l in ("p05", "p10", "p30"):
    for k, v in PD.decode(ROOT / f"outputs/extension_medmnist_microscopy_poisson/results/level_{l}/val/figures/comparison_examples.png", PD.POI_COLS).items():
        dec[(l,) + k] = v
for k, v in PD.decode(ROOT / "outputs/extension_medmnist_microscopy/results/target_clean/val/figures/comparison_examples.png", PD.EXT_COLS).items():
    dec[("p00",) + k] = v
b1 = max(float(np.abs(U[(l, n)][vid.index(i)] - dec[(l, i, f"AI-original-{n}")]).max()) for l in LV for i in VAL_IDS for n in LOW)
# cross-check: is each decoded val U-Net panel closest to ITS model's output (not another model / image)?
b1_id = []
for l in LV:
    for i in VAL_IDS:
        for n in LOW:
            d = {(ll, nn): float(np.abs(U[(ll, nn)][vid.index(i)] - dec[(l, i, f"AI-original-{n}")]).mean()) for ll in LV for nn in LOW}
            b1_id.append(min(d, key=d.get) == (l, n))
R["B1_val_figure_panels_vs_model_outputs"] = {"panels": len(b1_id), "max_abs_diff": b1, "closest_model_is_labelled_model": int(sum(b1_id))}
print("B1", R["B1_val_figure_panels_vs_model_outputs"], flush=True)

# B2: metrics of fresh outputs vs frozen val CSV (all 1000 images, sequence 2026)
def m4(ref, x):
    ref = ref.astype(np.float64); x = x.astype(np.float64); mse = ((ref - x) ** 2).mean()
    return structural_similarity(ref, x, data_range=1.0), 10 * np.log10(1 / mse), np.corrcoef(ref.ravel(), x.ravel())[0, 1]
b2 = {}
for l in LV:
    csv = (ROOT / "outputs/extension_medmnist_microscopy/results/target_clean/val" if l == "p00" else ROOT / f"outputs/extension_medmnist_microscopy_poisson/results/level_{l}/val") / "per_image_results.csv"
    per = pd.read_csv(csv).query("seed == 2026").set_index(["image_id", "label"])
    for n in LOW:
        s = np.array([m4(vobj[j], U[(l, n)][j]) for j in range(1000)])
        c = per.loc[[(i, f"AI-original-{n}") for i in vid], ["SSIM", "PSNR", "correlation"]].to_numpy()
        b2[f"{l} N={n}"] = float(np.abs(s - c).max())
        if l == "p05" and n == 1000:     # B3 control: wrong checkpoint on the same input
            w = np.array([m4(vobj[j], x) for j, x in enumerate(run("p30", 500, dg[1, :, 1]))])
            R["B3_wrong_checkpoint_control (p30 AI-500 weights on p05 DGI-1000 input)"] = float(np.abs(w - c).max())
R["B2_fresh_val_metrics_vs_frozen_csv_max_abs_diff (SSIM,PSNR,r)"] = b2
print("B2", b2, flush=True)

# B4: smoothness, specificity, input dependence, small structures
def corrmat(A, B):
    A = A.reshape(len(A), -1); B = B.reshape(len(B), -1)
    A = (A - A.mean(1, keepdims=True)) / A.std(1, keepdims=True); B = (B - B.mean(1, keepdims=True)) / B.std(1, keepdims=True)
    return A @ B.T / A.shape[1]
def radial(img):
    F = np.abs(np.fft.fftshift(np.fft.fft2(img - img.mean()))) ** 2
    yy, xx = np.indices(F.shape); r = np.hypot(yy - 32, xx - 32).astype(int)
    return np.bincount(r.ravel(), F.ravel())[:32] / np.bincount(r.ravel())[:32]
mean_img = {s: tr["object"][tr["subset"] == s].mean(0) for s in ("blood", "path")}
base = {s: float(np.mean([structural_similarity(vobj[j].astype(np.float64), mean_img[s].astype(np.float64), data_range=1.0) for j in np.flatnonzero(sub == s)])) for s in ("blood", "path")}
R["B4_mean_image_baseline_SSIM (predict the training-set mean image)"] = base
rc = {s: np.mean([radial(vobj[j]) for j in np.flatnonzero(sub == s)], 0) for s in ("blood", "path")}
cc_clean = {s: corrmat(vobj[sub == s], vobj[sub == s]) for s in ("blood", "path")}

def small_bodies(img, thr=0.35, amin=3, amax=60):
    lab, n = ndimage.label(img < thr); out = []
    for k in range(1, n + 1):
        msk = lab == k; a = msk.sum()
        if amin <= a <= amax: out.append(msk)
    return out
blood_idx = np.flatnonzero(sub == "blood")
bodies = {j: small_bodies(vobj[j]) for j in blood_idx}
nb = sum(len(v) for v in bodies.values())
rows = []
for (l, n), out in U.items():
    row = {"level": l, "N": n}
    for s in ("blood", "path"):
        J = np.flatnonzero(sub == s); cm = corrmat(out[J], vobj[J])
        row[f"{s}_identification_top1"] = float((cm.argmax(1) == np.arange(len(J))).mean())
        row[f"{s}_corr_own"] = float(np.diag(cm).mean())
        row[f"{s}_corr_best_other"] = float(np.where(np.eye(len(J), dtype=bool), -9, cm).max(1).mean())
        oo = corrmat(out[J], out[J]); iu = np.triu_indices(len(J), 1)
        row[f"{s}_output_pairwise_corr"] = float(oo[iu].mean()); row[f"{s}_clean_pairwise_corr"] = float(cc_clean[s][iu].mean())
        row[f"{s}_corr_with_train_mean_output"] = float(np.mean([np.corrcoef(o.ravel(), mean_img[s].ravel())[0, 1] for o in out[J]]))
        row[f"{s}_corr_with_train_mean_clean"] = float(np.mean([np.corrcoef(o.ravel(), mean_img[s].ravel())[0, 1] for o in vobj[J]]))
        ro = np.mean([radial(o) for o in out[J]], 0)
        row[f"{s}_HF_power_ratio_16_31"] = float((ro[16:32] / rc[s][16:32]).mean()); row[f"{s}_MF_power_ratio_8_15"] = float((ro[8:16] / rc[s][8:16]).mean())
        row[f"{s}_output_std_over_clean_std"] = float(out[J].std(axis=(1, 2)).mean() / vobj[J].std(axis=(1, 2)).mean())
    det = sum(int(out[j][np.asarray(ndimage.binary_dilation(m, iterations=1))].min() < 0.45) for j in blood_idx for m in bodies[j])
    row["blood_small_body_recall"] = det / nb
    rows.append(row)
b4 = pd.DataFrame(rows); b4.to_csv(OUT / "B4_smoothness_specificity_val.csv", index=False)
R["B4_small_bodies_in_val_blood_clean"] = nb
print(b4[["level", "N", "blood_identification_top1", "path_identification_top1", "blood_HF_power_ratio_16_31", "path_HF_power_ratio_16_31", "blood_small_body_recall"]].round(3).to_string(), flush=True)
(OUT / "B_val_side.json").write_text(json.dumps(R, indent=1))
# figure
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), layout="constrained")
for l, c in zip(LV, ("#444", "#2a78d6", "#1baf7a", "#d6452a")):
    g = b4[b4.level == l]
    ax[0].plot(g.N, g.blood_HF_power_ratio_16_31, "-o", color=c, label=f"blood {l[1:]}%"); ax[0].plot(g.N, g.path_HF_power_ratio_16_31, "--s", color=c, label=f"tissue {l[1:]}%")
    ax[1].plot(g.N, g.blood_small_body_recall, "-o", color=c, label=f"{l[1:]}% noise")
    ax[2].plot(g.N, g.path_identification_top1, "--s", color=c, label=f"tissue {l[1:]}%"); ax[2].plot(g.N, g.blood_identification_top1, "-o", color=c, label=f"blood {l[1:]}%")
for a, t in zip(ax, ("Fine-detail power kept (16–31 cycles), U-Net / clean", f"Small dark bodies recovered (blood, {nb} bodies)", "Output identifies its own original (top-1 of 500)")):
    a.set_xscale("log"); a.set_xticks(LOW, [f"{n:,}" for n in LOW]); a.minorticks_off(); a.set_title(t, loc="left", fontsize=10); a.grid(alpha=0.3); a.set_xlabel("shots")
ax[0].legend(fontsize=7, ncol=2, frameon=False); ax[1].legend(fontsize=8, frameon=False); ax[2].legend(fontsize=7, ncol=2, frameon=False)
fig.suptitle("Validation set (1,000 images, sequence 2026): frozen U-Net checkpoints run on recomputed DGI inputs", fontsize=11)
fig.savefig(OUT / "figures/B4_smoothness_specificity_val.png", dpi=150, facecolor="white"); plt.close(fig)
print(f"done {time.time()-t0:.0f}s")
