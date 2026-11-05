#!/usr/bin/env python3
"""Phase 2B: Model A (DGI-500 -> U-Net) vs Model B (DGI-1000 -> U-Net) on VALIDATION only.

    python compare_models_AB.py

Loads both best checkpoints read-only (Model A is not modified), re-checks that each
reproduces its own recorded validation metrics, and writes into
outputs/phase2b_unet/model_B_dgi1000/full/:
    comparison_A_vs_B.csv                       5-row table (DGI-500, U-Net-500, DGI-1000, U-Net-1000, DGI-10,000)
    per_image_A_vs_B.csv                        per-image metrics + paired B - A differences
    reconstruction_examples.png                 6 panels, same 5 val images as Model A, fixed [0, 1] scale
    reconstruction_examples_DISPLAY_ONLY.png    same, display-only contrast window
    comparison_A_vs_B.json
The test split is never loaded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.gi_unet_data import load_pairs  # noqa: E402
from src.unet_gi import UNetGI  # noqa: E402
from src.visualization import save_panel_grid  # noqa: E402
from train_unet_gi import evaluate_set, predict  # noqa: E402

OUT = ROOT / "outputs" / "phase2b_unet"
RUNS = {"A": (500, OUT / "model_A_dgi500" / "full"), "B": (1000, OUT / "model_B_dgi1000" / "full")}
KEYS = ("PSNR", "SSIM", "MSE", "Pearson_r")


def load_best(run_dir: Path):
    ck = torch.load(run_dir / "checkpoints" / "best.pt", weights_only=False)
    model = UNetGI(base=ck["config"]["base"])
    model.load_state_dict(ck["model"])
    return model, ck


def main() -> None:
    torch.set_num_threads(8)
    data, preds, ev, checks = {}, {}, {}, {}
    for shots in (500, 1000, 10000):
        x, y, ids = load_pairs("val", shots)
        data[shots] = x
        if shots == 500:
            y_va, id_va = y, ids
        assert ids == id_va and np.array_equal(y, y_va), f"val targets differ for DGI-{shots}"
        ev[f"DGI-{shots}"] = evaluate_set(y_va, x)

    for name, (shots, run_dir) in RUNS.items():
        model, ck = load_best(run_dir)
        preds[name] = predict(model, data[shots])
        e = evaluate_set(y_va, preds[name])
        ev[f"U-Net DGI-{shots}"] = e
        rec = json.loads((run_dir / "training_report.json").read_text())
        recorded = next(r for r in rec["val_summary"] if r["method"].startswith("U-Net"))
        checks[f"model_{name}_best_epoch"] = int(ck["epoch"])
        checks[f"model_{name}_reproduces_recorded_max_abs_diff"] = float(max(abs(e[k] - recorded[k]) for k in KEYS))

    order = [("DGI-500", 500), ("U-Net DGI-500", 500), ("DGI-1000", 1000), ("U-Net DGI-1000", 1000),
             ("DGI-10000", 10000)]
    table = pd.DataFrame([{"method": m, "shots": s,
                           **{k: ev[m][k] for k in KEYS},
                           **{f"{k}_median": ev[m]["per_image"][k].median() for k in ("PSNR", "SSIM", "Pearson_r")}}
                          for m, s in order])
    table.to_csv(RUNS["B"][1] / "comparison_A_vs_B.csv", index=False)

    per = pd.DataFrame({"image_id": id_va})
    for m, _ in order:
        tag = m.replace("U-Net DGI-", "UNet").replace("DGI-", "DGI")
        for k in KEYS:
            per[f"{tag}_{k}"] = ev[m]["per_image"][k].values
    paired = {}
    for k in ("PSNR", "SSIM", "Pearson_r", "MSE"):
        d = per[f"UNet1000_{k}"] - per[f"UNet500_{k}"]
        per[f"B_minus_A_{k}"] = d
        better = d < 0 if k == "MSE" else d > 0
        paired[k] = {"mean": float(d.mean()), "median": float(d.median()), "pct_B_better": float(better.mean() * 100)}
    for name, tag in (("A", "UNet500"), ("B", "UNet1000")):
        for k in ("PSNR", "SSIM", "Pearson_r"):
            paired[f"{name}_beats_DGI10000_{k}_pct"] = float((per[f"{tag}_{k}"] > per[f"DGI10000_{k}"]).mean() * 100)
    per.to_csv(RUNS["B"][1] / "per_image_A_vs_B.csv", index=False)

    # same 5 validation images as Model A's figures
    ids_a = json.loads((RUNS["A"][1] / "training_report.json").read_text())["val_example_ids"]
    quant = (10, 30, 50, 70, 90)
    rows = []
    for q, image_id in zip(quant, ids_a):
        j = id_va.index(image_id)
        r = per.iloc[j]

        def ttl(name, tag):
            return f"{name}\n{r[f'{tag}_PSNR']:.2f} dB | SSIM {r[f'{tag}_SSIM']:.3f}"
        rows.append({"label": f"{image_id} (val)\nDGI-500 PSNR p{q}", "target": y_va[j, 0], "panels": [
            ("Original (target)", y_va[j, 0]),
            (ttl("DGI-500", "DGI500"), data[500][j, 0]),
            (ttl("U-Net DGI-500 (A)", "UNet500"), np.clip(preds["A"][j, 0], 0, 1)),
            (ttl("DGI-1000", "DGI1000"), data[1000][j, 0]),
            (ttl("U-Net DGI-1000 (B)", "UNet1000"), np.clip(preds["B"][j, 0], 0, 1)),
            (ttl("DGI-10,000", "DGI10000"), data[10000][j, 0])]})
    head = (f"Model A (best epoch {checks['model_A_best_epoch']}) vs Model B (best epoch "
            f"{checks['model_B_best_epoch']}), same U-Net/training, only the input changes")
    save_panel_grid(rows, RUNS["B"][1] / "reconstruction_examples.png",
                    head + "\nVALIDATION images (same 5 as Model A), gray window [0, 1] for every panel")
    save_panel_grid(rows, RUNS["B"][1] / "reconstruction_examples_DISPLAY_ONLY.png",
                    head + "\nDISPLAY ONLY: one window per row (target's 1-99th pct); metrics from [0, 1] images",
                    display_window=True)

    result = {"checks": checks, "table": table.to_dict("records"), "paired_B_minus_A": paired,
              "example_ids": ids_a}
    (RUNS["B"][1] / "comparison_A_vs_B.json").write_text(json.dumps(result, indent=2, default=float))
    pd.set_option("display.width", 170)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(json.dumps({"checks": checks, "paired": paired}, indent=1))


if __name__ == "__main__":
    main()
