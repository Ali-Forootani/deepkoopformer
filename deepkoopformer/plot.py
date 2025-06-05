import os
import argparse
from pathlib import Path

import matplotlib
if os.getenv("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
_STYLES = ["-", "--", "-.", ":"]


def _save(fig, base):
    fig.savefig(f"{base}.png", dpi=600, bbox_inches="tight",
                transparent=True)
    fig.savefig(f"{base}.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 11)  Saving                                                                 #
# --------------------------------------------------------------------------- #
def save_results(res, prefix, metrics_file, patch_len, horizon, d_model):
    pred, x_out = res["pred"], res["x_out"]
    err = x_out - pred
    np.save(f"{prefix}_predictions.npy", pred)
    np.save(f"{prefix}_errors.npy", err)
    mse = float((err**2).mean()); mae = float(np.abs(err).mean())
    pd.DataFrame({"Model":[Path(prefix).stem],"PatchLen":[patch_len],
                  "Horizon":[horizon],"d_model":[d_model],"MSE":[mse],"MAE":[mae]}
        ).to_csv(metrics_file, mode="a", header=False, index=False)




def plot_training(res, path=None):
    if not res:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for i, (n, r) in enumerate(res.items()):
        ax1.plot(r["loss"], label=n,
                 color=_COLORS[i % 10], linestyle=_STYLES[i % 4], lw=3)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_title("Loss"); ax1.grid(True, ls="--", alpha=.6); ax1.legend()

    k = 0
    for n, r in res.items():
        if np.allclose(r["eig"], 0):
            continue
        ax2.plot(r["eig"], label=n,
                 color=_COLORS[k % 10], linestyle=_STYLES[k % 4], lw=3)
        k += 1
    ax2.set_title("Koopman eig max"); ax2.grid(True, ls="--", alpha=.6)
    ax2.legend()
    plt.tight_layout()
    if path:
        _save(fig, path)



def plot_per_feature(x_out: np.ndarray, preds: dict, F: int, H: int, save_path: str | None = None):
    N = x_out.shape[0]
    gt = x_out.reshape(N, F, H)[:, :, 0]
    preds_plot = {n: p.reshape(N, F, H)[:, :, 0] for n, p in preds.items()}
    t = np.arange(N)
    fig, axes = plt.subplots(F, 1, figsize=(10, 3 * F), sharex=True)
    axes = [axes] if F == 1 else axes
    for f in range(F):
        ax = axes[f]
        ax.plot(t, gt[:, f], lw=3, label="Ground Truth", color="black")
        for i, (n, p) in enumerate(preds_plot.items()):
            ax.plot(t, p[:, f], label=n, color=_COLORS[i % len(_COLORS)], linestyle=_STYLES[i % len(_STYLES)], linewidth=3)
        ax.set_ylabel("Value"); ax.grid(True, which="both", linestyle="--", alpha=0.6)
        if f == 0: ax.legend(ncol=3)
    axes[-1].set_xlabel("Sample")
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
