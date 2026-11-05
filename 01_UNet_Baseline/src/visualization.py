"""Figures for the single-image ghost-imaging validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_comparison(original: np.ndarray, recons: dict[int, np.ndarray], metrics: dict[int, dict],
                    path: Path, title: str) -> None:
    """Original + one panel per measurement count, all on a fixed [0, 1] gray scale."""
    n = 1 + len(recons)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.9))
    axes[0].imshow(original, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("Original (ground truth)", fontsize=10)
    for ax, (shots, img) in zip(axes[1:], recons.items()):
        m = metrics[shots]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{shots:,} shots\nPSNR {m['PSNR']:.2f} dB | SSIM {m['SSIM']:.3f}", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_metric_curves(table, path: Path, title: str) -> None:
    """PSNR / SSIM / MSE versus measurement count, one line per normalisation."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, key in zip(axes, ["PSNR", "SSIM", "MSE"]):
        for norm, grp in table.groupby("normalization"):
            ax.plot(grp["measurements"], grp[key], marker="o", label=norm)
        ax.set_xscale("log")
        if key == "MSE":
            ax.set_yscale("log")
        ax.set_xlabel("number of measurements (log)")
        ax.set_ylabel(key)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


METHOD_COLORS = {"CGI": "#2a78d6", "DGI": "#eb6834"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def save_method_grid(original: np.ndarray, recons: dict[str, dict[int, np.ndarray]],
                     metrics: dict[str, dict[int, dict]], path: Path, title: str,
                     window: tuple[float, float] = (0.0, 1.0)) -> None:
    """Rows = methods, columns = original + measurement counts; one shared gray window."""
    methods = list(recons)
    shots = list(next(iter(recons.values())))
    ncol = 1 + len(shots)
    fig, axes = plt.subplots(len(methods), ncol, figsize=(3.0 * ncol, 3.7 * len(methods)), squeeze=False)
    lo, hi = window
    for r, method in enumerate(methods):
        axes[r, 0].imshow(original, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        axes[r, 0].set_title("Original", fontsize=10, color=INK)
        axes[r, 0].set_ylabel(method, fontsize=12, color=INK)
        for c, n in enumerate(shots, start=1):
            m = metrics[method][n]
            axes[r, c].imshow(recons[method][n], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            axes[r, c].set_title(f"{method} {n:,}\nPSNR {m['PSNR']:.2f} | SSIM {m['SSIM']:.3f}",
                                 fontsize=9.5, color=INK)
    for ax in axes.flat:
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(title, fontsize=10.5, color=INK)
    fig.tight_layout(h_pad=2.5)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def save_method_curves(table, path: Path, title: str, theory: dict[str, dict[int, float]] | None = None) -> None:
    """PSNR / SSIM / MSE / Pearson r vs measurement count, one line per method.

    `theory` (optional) maps method -> {N: predicted MSE}, drawn dashed on the MSE panel
    together with the measured UNCLIPPED MSE (hollow markers), which is what the theory
    describes; the solid line is the clipped [0, 1] estimate used for all metrics.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    for ax, key, label in zip(axes, ["PSNR", "SSIM", "MSE", "Pearson_r"],
                              ["PSNR (dB)", "SSIM", "MSE (log)", "Pearson r"]):
        for method, grp in table.groupby("method"):
            grp = grp.sort_values("measurements")
            color = METHOD_COLORS.get(method, INK_MUTED)
            ax.plot(grp["measurements"], grp[key], color=color, lw=2, marker="o", ms=8,
                    mec="white", mew=2, label=method, zorder=3)
            x, y = grp["measurements"].iloc[-1], grp[key].iloc[-1]
            ax.annotate(method, (x, y), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
            if key == "MSE" and theory and method in theory:
                ns = sorted(theory[method])
                ax.plot(ns, [theory[method][n] for n in ns], color=color, lw=1.2, ls="--", alpha=0.8,
                        label=f"{method} theory (unclipped)", zorder=2)
                ax.plot(grp["measurements"], grp["unclipped_MSE"], color=color, ls="none", marker="o", ms=8,
                        mfc="white", mew=1.5, label=f"{method} measured, unclipped", zorder=2)
        ax.set_xscale("log")
        if key == "MSE":
            ax.set_yscale("log")
        ax.set_xlabel("number of measurements (log)", color=INK_MUTED)
        ax.set_title(label, fontsize=10.5, color=INK, loc="left")
        ax.grid(color=GRID, lw=0.8, which="both")
        ax.tick_params(colors=INK_MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlim(right=ax.get_xlim()[1] * 1.6)
    axes[2].legend(fontsize=8.5, frameon=False)
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def save_gray_png(img: np.ndarray, path: Path) -> None:
    """Save a [0, 1] image as an 8-bit grayscale PNG at native resolution."""
    plt.imsave(path, np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)


SPLIT_COLORS = {"all": INK, "train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}


def _clean_axes(ax, title: str) -> None:
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.grid(color=GRID, lw=0.8, which="both")
    ax.tick_params(colors=INK_MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save_split_curves(per_image, path: Path, title: str) -> None:
    """Mean +/- 1 std of PSNR/SSIM/MSE/r vs N, for all images and each split.

    per_image: long DataFrame with columns split, measurements, PSNR, SSIM, MSE, Pearson_r.
    Splits are offset slightly along x so error bars do not overlap.
    """
    groups = [("all", per_image)] + [(s, per_image[per_image.split == s]) for s in ("train", "val", "test")]
    offsets = {"all": 1.0, "train": 0.93, "val": 1.07, "test": 1.14}
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.1))
    for ax, key, label in zip(axes, ["PSNR", "SSIM", "MSE", "Pearson_r"],
                              ["PSNR (dB), mean ± 1 std", "SSIM, mean ± 1 std", "MSE (log), mean ± 1 std",
                               "Pearson r, mean ± 1 std"]):
        for name, df in groups:
            st = df.groupby("measurements")[key].agg(["mean", "std"]).reset_index()
            x = st["measurements"] * offsets[name]
            ax.errorbar(x, st["mean"], yerr=st["std"], color=SPLIT_COLORS[name], lw=2.6 if name == "all" else 1.6,
                        marker="o", ms=7, mec="white", mew=1.5, capsize=3, elinewidth=1,
                        label=f"{name} (n={df.image_id.nunique()})", zorder=4 if name == "all" else 3)
        ax.set_xscale("log")
        if key == "MSE":
            ax.set_yscale("log")
        ax.set_xlabel("number of measurements (log)", color=INK_MUTED)
        _clean_axes(ax, label)
    axes[0].legend(fontsize=8.5, frameon=False, loc="upper left")
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def save_distribution_boxplots(per_image, path: Path, title: str) -> None:
    """Per-image PSNR and SSIM distributions for each measurement count (all images)."""
    shots = sorted(per_image["measurements"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key in zip(axes, ["PSNR", "SSIM"]):
        data = [per_image.loc[per_image.measurements == n, key].values for n in shots]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, whis=(5, 95), showfliers=True,
                        medianprops=dict(color=INK, lw=1.6),
                        flierprops=dict(marker="o", ms=3, mfc=INK_MUTED, mec="none", alpha=0.6))
        for box in bp["boxes"]:
            box.set(facecolor="#cde2fb", edgecolor="#2a78d6", lw=1.2)
        ax.set_xticks(range(1, len(shots) + 1), [f"{n:,}" for n in shots])
        ax.set_xlabel("number of measurements", color=INK_MUTED)
        _clean_axes(ax, f"{key} per image (box: IQR, whiskers: 5th-95th pct)")
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def save_contrast_diagnostic(per_image_n, path: Path, title: str) -> None:
    """DGI PSNR at the largest N vs object contrast (std), coloured by FOV type."""
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for fov, color in (("rectangular", "#2a78d6"), ("circular", "#eb6834")):
        d = per_image_n[per_image_n.fov_type == fov]
        ax.scatter(d["object_std"], d["PSNR"], s=16, color=color, alpha=0.75, edgecolor="white", lw=0.4,
                   label=f"{fov} FOV (n={len(d)})")
    ax.set_xscale("log")
    ax.set_xlabel("object std σ_O (log)", color=INK_MUTED)
    ax.legend(fontsize=8.5, frameon=False)
    _clean_axes(ax, "PSNR (dB) vs object contrast")
    fig.suptitle(title, fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def save_sample_grid(rows: list[dict], shots: list[int], path: Path, title: str, display_window: bool = False) -> None:
    """One row per image: original + reconstruction per N.

    rows: dicts with keys label, original, recons {N: img}, metrics {N: dict}.
    display_window=False: every panel on the fixed [0, 1] scale (scientific figure).
    display_window=True: DISPLAY ONLY, each row shares the window of its original's
    1-99th percentile (same window for all panels in that row).
    """
    ncol = 1 + len(shots)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.5 * ncol, 2.75 * len(rows)), squeeze=False)
    for r, row in enumerate(rows):
        lo, hi = (np.percentile(row["original"], [1, 99]) if display_window else (0.0, 1.0))
        axes[r, 0].imshow(row["original"], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        axes[r, 0].set_title("Original", fontsize=8.5, color=INK)
        axes[r, 0].set_ylabel(row["label"], fontsize=8.5, color=INK)
        for c, n in enumerate(shots, start=1):
            m = row["metrics"][n]
            axes[r, c].imshow(row["recons"][n], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            axes[r, c].set_title(f"DGI-{n:,}\n{m['PSNR']:.1f} dB | SSIM {m['SSIM']:.2f}", fontsize=8.5, color=INK)
    for ax in axes.flat:
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout(h_pad=1.5, rect=(0, 0, 1, 1 - 0.5 / len(rows) / 2.75))
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def save_triplet_grid(rows: list[dict], path: Path, title: str, input_label: str = "DGI-500 input",
                      display_window: bool = False) -> None:
    """One row per image: original target | DGI input | U-Net output, ONE gray window per row.

    rows: dicts with label, target, input, output (2-D arrays in [0, 1]) and
    m_in / m_out (metric dicts with PSNR, SSIM). display_window=False uses [0, 1] for
    every panel; True (DISPLAY ONLY) uses the target's 1-99th percentile per row.
    """
    fig, axes = plt.subplots(len(rows), 3, figsize=(8.4, 2.95 * len(rows)), squeeze=False)
    for r, row in enumerate(rows):
        lo, hi = (np.percentile(row["target"], [1, 99]) if display_window else (0.0, 1.0))
        panels = [("Original (target)", row["target"]),
                  (f"{input_label}\n{row['m_in']['PSNR']:.2f} dB | SSIM {row['m_in']['SSIM']:.3f}", row["input"]),
                  (f"U-Net output\n{row['m_out']['PSNR']:.2f} dB | SSIM {row['m_out']['SSIM']:.3f}", row["output"])]
        for c, (t, img) in enumerate(panels):
            axes[r, c].imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            axes[r, c].set_title(t, fontsize=9, color=INK)
        axes[r, 0].set_ylabel(row["label"], fontsize=9, color=INK)
    for ax in axes.flat:
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout(h_pad=1.2, rect=(0, 0, 1, 1 - 0.55 / len(rows) / 2.95))
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)


def save_loss_curves(history, batch_log, path: Path, title: str, refs: dict[str, float]) -> None:
    """Train (per-batch faint + per-epoch mean) and validation MSE vs epoch, log y.

    refs: name -> validation MSE of a non-learned reference, drawn as dashed lines.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    if len(batch_log):
        ax.plot(batch_log["epoch_float"], batch_log["loss"], color="#2a78d6", lw=0.8, alpha=0.35,
                label="train MSE (per batch)")
    h = history[history.epoch > 0]
    many = len(history) > 12
    mk = dict(marker=None) if many else dict(marker="o", ms=8, mec="white", mew=2)
    ax.plot(h["epoch"], h["train_loss"], color="#2a78d6", lw=2, label="train MSE (epoch mean)", **mk)
    ax.plot(history["epoch"], history["val_loss"], color="#eb6834", lw=2,
            label="validation MSE (epoch 0 = untrained)", **mk)
    for (name, v), ls in zip(refs.items(), ("--", ":", "-.")):
        ax.axhline(v, color=INK_MUTED, lw=1.2, ls=ls, label=f"{name} (val) = {v:.4f}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch", color=INK_MUTED)
    if not many:
        ax.set_xticks(history["epoch"])
    _clean_axes(ax, "MSE loss (log)")
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def save_panel_grid(rows: list[dict], path: Path, title: str, display_window: bool = False) -> None:
    """One row per image, any number of panels, ONE gray window per row.

    rows: dicts with label, target (2-D, used for the display window) and panels
    [(title, image)]. display_window=False: [0, 1] for every panel (scientific
    figure); True (DISPLAY ONLY): the target's 1-99th percentile per row.
    """
    ncol = len(rows[0]["panels"])
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.8 * ncol, 3.25 * len(rows)), squeeze=False,
                             layout="constrained")
    for r, row in enumerate(rows):
        lo, hi = (np.percentile(row["target"], [1, 99]) if display_window else (0.0, 1.0))
        for c, (t, img) in enumerate(row["panels"]):
            axes[r, c].imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            axes[r, c].set_title(t, fontsize=8.8, color=INK)
        axes[r, 0].set_ylabel(row["label"], fontsize=9, color=INK)
    for ax in axes.flat:
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(title, fontsize=10, color=INK)
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)


def save_val_metric_curves(history, path: Path, title: str, refs: dict[str, dict], best_epoch: int,
                           unet_label: str = "U-Net") -> None:
    """Validation PSNR / SSIM / r vs epoch, with non-learned references as horizontal lines.

    refs: name -> {"PSNR":..., "SSIM":..., "Pearson_r":...} (NaN entries are skipped).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    h = history[history.epoch > 0]
    for ax, (col, key, label) in zip(axes, [("val_psnr", "PSNR", "Validation PSNR (dB)"),
                                            ("val_ssim", "SSIM", "Validation SSIM"),
                                            ("val_correlation", "Pearson_r", "Validation Pearson r")]):
        ax.plot(h["epoch"], h[col], color="#eb6834", lw=2, label=unet_label, zorder=3)
        ax.axvline(best_epoch, color="#eb6834", lw=1, ls=":", alpha=0.8, label=f"best epoch {best_epoch}")
        for (name, vals), ls in zip(refs.items(), ("--", ":", "-.")):
            v = vals.get(key, np.nan)
            if np.isfinite(v):
                ax.axhline(v, color=INK_MUTED, lw=1.2, ls=ls, label=f"{name} = {v:.3f}")
        ax.set_xlabel("epoch", color=INK_MUTED)
        _clean_axes(ax, label)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle(title, fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
