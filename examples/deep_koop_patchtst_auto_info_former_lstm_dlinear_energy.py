#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Koopformer-PRO benchmark on Wind-Speed data (HPC-ready)

Sweeps
  • patch lengths    [80, 90, 100, 110, 120, 130]
  • forecast horizons [10, 15, 20, 25, 30]

Each Koopformer variant accepts `patch_len`:

  · Koopformer_PatchTST   – real patch embedding
  · Koopformer_Autoformer – trend-kernel length = patch_len
  · Koopformer_Informer   – optional patch embedding

2025-05-20  – Ali Forootani
Modified: DLinear baseline added (2025-07-08)
"""
# --------------------------------------------------------------------------- #
# 0)  HPC-safe backend & imports                                              #
# --------------------------------------------------------------------------- #
import os
import argparse
from pathlib import Path

import matplotlib
if os.getenv("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------------------------------- #
# 1)  Reproducibility                                                         #
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(7)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------- #
# 2)  Koopman operator                                                        #
# --------------------------------------------------------------------------- #
def _orth(w: torch.Tensor) -> torch.Tensor:
    return torch.linalg.qr(w)[0]

class StrictStableKoopmanOperator(nn.Module):
    def __init__(self, latent_dim: int, ρ_max: float = 0.99):
        super().__init__()
        self.U_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.V_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.S_raw = nn.Parameter(torch.randn(latent_dim))
        self.ρ_max = ρ_max

    def forward(self, z: torch.Tensor):
        U, V = _orth(self.U_raw), _orth(self.V_raw)
        Σ = torch.sigmoid(self.S_raw) * self.ρ_max
        K = U @ torch.diag(Σ) @ V.T
        return z @ K.T, K, Σ

# --------------------------------------------------------------------------- #
# 3)  Positional encodings & patch embed                                      #
# --------------------------------------------------------------------------- #
class SinCosPosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10_000):
        super().__init__()
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-np.log(10_000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]

class PatchEmbed1D(nn.Module):
    def __init__(self, in_ch: int, d_model: int,
                 patch_len: int, stride: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, d_model,
                              kernel_size=patch_len, stride=stride)

    def forward(self, x):
        x = x.permute(0, 2, 1)     # (B, C, T)
        x = self.conv(x)           # (B, d, T//patch)
        return x.permute(0, 2, 1)  # (B, P, d)

# --------------------------------------------------------------------------- #
# 4)  PatchTST backbone & Koopformer                                          #
# --------------------------------------------------------------------------- #
class PatchTST_Backbone(nn.Module):
    def __init__(self, input_dim: int, seq_len: int,
                 patch_len: int, d_model: int = 64,
                 num_layers: int = 3, num_heads: int = 4,
                 dim_ff: int = 96):
        super().__init__()
        self.patch = PatchEmbed1D(input_dim, d_model,
                                  patch_len, patch_len)
        n_patches = int(np.ceil(seq_len / patch_len))
        self.pos  = SinCosPosEnc(d_model, max_len=n_patches + 1)
        enc = nn.TransformerEncoderLayer(d_model, num_heads,
                                         dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        x   = self.patch(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.encoder(self.pos(x))
        return x[:, 0]

class Koopformer_PatchTST(nn.Module):
    def __init__(self, input_dim, seq_len, horizon,
                 patch_len, d_model=64):
        super().__init__()
        self.backbone = PatchTST_Backbone(input_dim, seq_len,
                                          patch_len=patch_len,
                                          d_model=d_model)
        self.koop = StrictStableKoopmanOperator(d_model)
        self.fc   = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z       = self.backbone(x)
        z_next, _, _ = self.koop(z)
        pred    = self.fc(z_next)
        if return_latents:
            return pred, z, z_next
        return pred

# --------------------------------------------------------------------------- #
# 5)  Autoformer backbone & Koopformer                                        #
# --------------------------------------------------------------------------- #
class SeriesDecomp(nn.Module):
    def __init__(self, k: int = 3):
        super().__init__()
        self.avg = nn.AvgPool1d(k, stride=1, padding=k // 2)

    def forward(self, x):
        trend = self.avg(x.transpose(1, 2)).transpose(1, 2)
        if trend.size(1) != x.size(1):
            trend = trend[:, : x.size(1)]
        return trend, x - trend

class SimpleAutoformer(nn.Module):
    def __init__(self, input_len, horizon, input_dim,
                 patch_len,                # used as MA window
                 d_model=64, num_heads=4,
                 dim_ff=64, num_layers=3):
        super().__init__()
        self.dec   = SeriesDecomp(k=patch_len)
        self.embed = nn.Linear(input_dim, d_model)
        self.pos   = SinCosPosEnc(d_model, max_len=input_len)
        enc = nn.TransformerEncoderLayer(d_model, num_heads,
                                         dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        self.fc_seas  = nn.Linear(d_model, horizon)
        self.fc_trend = nn.Linear(input_len * input_dim, horizon)

    def forward(self, x):
        trend, seas = self.dec(x)
        seas   = self.encoder(self.pos(self.embed(seas)))
        seas_o = self.fc_seas(seas.mean(1))
        trend_o= self.fc_trend(trend.reshape(trend.size(0), -1))
        return seas_o + trend_o

class Koopformer_Autoformer(nn.Module):
    def __init__(self, input_dim, seq_len, horizon,
                 patch_len, d_model=16):
        super().__init__()
        self.backbone = SimpleAutoformer(seq_len, horizon, input_dim,
                                         patch_len=patch_len,
                                         d_model=d_model)
        self.koop = StrictStableKoopmanOperator(horizon)
        self.fc   = nn.Linear(horizon, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z       = self.backbone(x)
        z_next, _, _ = self.koop(z)
        pred    = self.fc(z_next)
        if return_latents:
            return pred, z, z_next
        return pred

# --------------------------------------------------------------------------- #
# 6)  Informer backbone & Koopformer                                          #
# --------------------------------------------------------------------------- #
class InformerSparse(nn.Module):
    def __init__(self, input_dim, d_model=64, num_heads=4,
                 dim_ff=96, num_layers=3,
                 seq_len=120, patch_len=1):
        super().__init__()
        self.use_patch = patch_len > 1
        if self.use_patch:
            self.patch = PatchEmbed1D(input_dim, d_model,
                                      patch_len, patch_len)
            n_tokens = int(np.ceil(seq_len / patch_len))
        else:
            self.embed = nn.Linear(input_dim, d_model)
            n_tokens = seq_len
        self.pos  = SinCosPosEnc(d_model, max_len=n_tokens)
        enc = nn.TransformerEncoderLayer(d_model, num_heads,
                                         dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)

    def forward(self, x):
        if self.use_patch:
            x = self.patch(x)
        else:
            x = self.embed(x)
        x = self.encoder(self.pos(x))
        return x.mean(dim=1)

class Koopformer_Informer(nn.Module):
    def __init__(self, input_dim, seq_len, horizon,
                 patch_len, d_model=64):
        super().__init__()
        self.backbone = InformerSparse(input_dim,
                                       d_model=d_model,
                                       seq_len=seq_len,
                                       patch_len=patch_len)
        self.koop = StrictStableKoopmanOperator(d_model)
        self.fc   = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z       = self.backbone(x)
        z_next, _, _ = self.koop(z)
        pred    = self.fc(z_next)
        if return_latents:
            return pred, z, z_next
        return pred

# --------------------------------------------------------------------------- #
# 7)  LSTM baseline                                                           #
# --------------------------------------------------------------------------- #
class SimpleLSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, horizon):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc   = nn.Linear(hidden_dim, horizon * input_dim)

    def forward(self, x, return_latents=False):
        _, (h_n, _) = self.lstm(x)
        z = h_n[-1]
        out = self.fc(z)
        if return_latents:
            return out, z, z
        return out

# --------------------------------------------------------------------------- #
# 8)  DLinear baseline                                                        #
# --------------------------------------------------------------------------- #
class DLinearForecaster(nn.Module):
    """
    DLinear: Decomposition-Linear baseline (Zeng et al., 2022).
    For each feature, applies a 1-D Conv from input window (seq_len) to forecast horizon.
    """
    def __init__(self, input_dim, seq_len, horizon):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Conv1d(1, 1, kernel_size=seq_len, bias=True)
             for _ in range(input_dim)]
        )
        self.horizon   = horizon
        self.seq_len   = seq_len
        self.input_dim = input_dim

    def forward(self, x, return_latents=False):
        # x: (B, seq_len, input_dim)
        # ► FIX START ────────────────────────────────────────────────────────
        outs = []
        for i, linear in enumerate(self.linears):
            xi = x[:, :, i].unsqueeze(1)          # (B, 1, seq_len)
            if xi.shape[-1] < self.seq_len:       # left-pad if needed
                pad = self.seq_len - xi.shape[-1]
                xi = torch.nn.functional.pad(xi, (pad, 0))
            yi = linear(xi)                       # (B, 1, 1) or (B, 1, H)
            if yi.shape[-1] == 1 and self.horizon > 1:
                yi = yi.expand(-1, -1, self.horizon)
            outs.append(yi)                       # keep 3-D (B, 1, H)
        out = torch.cat(outs, dim=1)              # (B, F, H)
        out = out.permute(0, 2, 1).reshape(x.size(0), -1)  # (B, F·H)
        # ► FIX END ──────────────────────────────────────────────────────────
        if return_latents:
            return out, out, out
        return out


# --------------------------------------------------------------------------- #
# 9)  Dataset util                                                            #
# --------------------------------------------------------------------------- #
def build_dataset(data: np.ndarray, indices: list[int],
                  seq_len: int, horizon: int):
    sub = data[0:500, indices]
    scaler = MinMaxScaler().fit(sub)
    sub = scaler.transform(sub)
    X, Y = [], []
    for i in range(len(sub) - seq_len - horizon):
        X.append(sub[i: i + seq_len])
        y = sub[i + seq_len: i + seq_len + horizon].T.reshape(-1)
        Y.append(y)
    x_in  = torch.tensor(np.stack(X), dtype=torch.float32)
    x_out = torch.tensor(np.stack(Y), dtype=torch.float32)
    return x_in, x_out, scaler

# --------------------------------------------------------------------------- #
# 10)  Training helper                                                         #
# --------------------------------------------------------------------------- #
def _nested_getattr(obj, path: str):
    for p in path.split('.'):
        obj = getattr(obj, p, None)
        if obj is None:
            return None
    return obj

def train(model, x_in, x_out, epochs=4000, lr=3e-4,
          koop_attr="koop", lyap_weight=0.1):
    model.to(DEV)
    x_in, x_out = x_in.to(DEV), x_out.to(DEV)
    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    losses, eigs = [], []

    with torch.no_grad():
        try:
            _, z, z_next = model(x_in[:1], return_latents=True)
        except ValueError:
            z = z_next = torch.zeros(1,
                    getattr(model, "fc").in_features, device=DEV)
    P = torch.eye(z.shape[1], device=DEV)

    for ep in range(epochs):
        model.train(); opt.zero_grad()
        pred, z, z_next = model(x_in, return_latents=True)
        pred_loss = mse(pred, x_out)
        if lyap_weight:
            zp  = torch.einsum("bi,ij->bj", z, P)
            znp = torch.einsum("bi,ij->bj", z_next, P)
            lyap = torch.relu((znp * z_next).sum(1)
                              - (zp * z).sum(1)).mean()
        else: lyap = torch.zeros(1, device=DEV)
        loss = pred_loss + lyap_weight * lyap
        loss.backward(); opt.step()
        losses.append(loss.item())

        obj = _nested_getattr(model, koop_attr) if koop_attr else None
        eig_raw = (obj.S_raw if isinstance(obj,
                   StrictStableKoopmanOperator) else None)
        eigs.append(0.0 if eig_raw is None
                    else torch.sigmoid(eig_raw).abs().max().item())

        if ep % max(1, epochs//10) == 0 or ep == epochs-1:
            print(f"Epoch {ep:>4}/{epochs}  "
                  f"MSE {pred_loss.item():.5f}  "
                  f"Lyap {(lyap_weight*lyap).item():.5f}")

    model.eval()
    with torch.no_grad():
        preds = model(x_in).cpu().numpy()
    return preds, np.array(losses), np.array(eigs)

# --------------------------------------------------------------------------- #
# 11)  Plot helpers                                                           #
# --------------------------------------------------------------------------- #
_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
           "#bcbd22", "#17becf"]
_STYLES = ["-", "--", "-.", ":"]

def _save(fig, base):
    fig.savefig(f"{base}.png", dpi=600, bbox_inches="tight",
                transparent=True)
    fig.savefig(f"{base}.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)

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

# --------------------------------------------------------------------------- #
# 12)  Saving                                                                 #
# --------------------------------------------------------------------------- #
def save_results(res, prefix, metrics_file, patch_len, horizon):
    pred, x_out = res["pred"], res["x_out"]
    err = x_out - pred
    np.save(f"{prefix}_predictions.npy", pred)
    np.save(f"{prefix}_errors.npy", err)
    mse = float((err**2).mean()); mae = float(np.abs(err).mean())
    pd.DataFrame({"Model":[Path(prefix).stem],"PatchLen":[patch_len],
                  "Horizon":[horizon],"MSE":[mse],"MAE":[mae]}
        ).to_csv(metrics_file, mode="a", header=False, index=False)

# --------------------------------------------------------------------------- #
# 13)  Main loop                                                              #
# --------------------------------------------------------------------------- #
def main(file="bitstamp_windows.npy", epochs=200, seq_len=120,
         patch_lens=None, horizons=None, indices=None,
         save_dir="results"):

    patch_lens = patch_lens or [120]
    horizons   = horizons   or [2]
    indices    = indices    or [2]

    save_dir   = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = save_dir / "metrics.csv"
    pd.DataFrame(columns=["Model", "PatchLen", "Horizon", "MSE", "MAE"]
        ).to_csv(metrics_file, index=False)

    raw = np.load(file)

    for patch_len in patch_lens:
        for horizon in horizons:
            print(f"\n>>> patch_len={patch_len}, horizon={horizon}\n")
            # ------------------------------------------------------------ #
            # keep seq_len = patch_len for variable-length experiments
            # If you need a *fixed* seq_len, replace `patch_len` below
            # with `seq_len`.
            # ------------------------------------------------------------ #
            x_in, x_out, _ = build_dataset(raw, indices,
                                           patch_len, horizon)
            F, H = len(indices), horizon
            res_plot = {}
            x_np = x_out.numpy()
            """
            # ----- LSTM -----
            print("== LSTM ==")
            lstm = SimpleLSTMForecaster(F, 48, 2, horizon)
            pred, loss, eig = train(lstm, x_in, x_out,
                                    epochs=epochs, koop_attr=None, lyap_weight=0)
            res_plot["LSTM"] = {"pred": pred, "loss": loss,
                                "eig": eig, "x_out": x_np}
            prefix = save_dir / f"LSTM_{patch_len}_{horizon}"
            save_results(res_plot["LSTM"], prefix, metrics_file,
                         patch_len, horizon)
            torch.save(lstm.cpu().state_dict(), f"{prefix}.pt")
            """
            
            # ----- DLinear -----
            print("== DLinear ==")
            dlinear = DLinearForecaster(F, patch_len, horizon)
            pred, loss, eig = train(dlinear, x_in, x_out,
                                    epochs=epochs, koop_attr=None, lyap_weight=0)
            res_plot["DLinear"] = {"pred": pred, "loss": loss,
                                   "eig": eig, "x_out": x_np}
            prefix = save_dir / f"DLinear_{patch_len}_{horizon}"
            save_results(res_plot["DLinear"], prefix, metrics_file,
                         patch_len, horizon)
            torch.save(dlinear.cpu().state_dict(), f"{prefix}.pt")


            # ----- PatchTST -----
            print("== PatchTST ==")
            m = Koopformer_PatchTST(F, patch_len, horizon,
                                    patch_len=patch_len, d_model= 96)
            pred, loss, eig = train(m, x_in, x_out, epochs=epochs, koop_attr="koop")
            res_plot["Koop-PatchTST"] = {"pred": pred, "loss": loss,
                                         "eig": eig, "x_out": x_np}
            prefix = save_dir / f"PatchTST_{patch_len}_{horizon}"
            save_results(res_plot["Koop-PatchTST"], prefix, metrics_file,
                         patch_len, horizon)
            torch.save(m.cpu().state_dict(), f"{prefix}.pt")
            """
            # ----- Autoformer -----
            print("== Autoformer ==")
            m = Koopformer_Autoformer(F, patch_len, horizon,
                                      patch_len=patch_len, d_model=48)
            pred, loss, eig = train(m, x_in, x_out, epochs=epochs, koop_attr="koop")
            res_plot["Koop-Autoformer"] = {"pred": pred, "loss": loss,
                                           "eig": eig, "x_out": x_np}
            prefix = save_dir / f"Autoformer_{patch_len}_{horizon}"
            save_results(res_plot["Koop-Autoformer"], prefix, metrics_file,
                         patch_len, horizon)
            torch.save(m.cpu().state_dict(), f"{prefix}.pt")

            # ----- Informer -----
            print("== Informer ==")
            m = Koopformer_Informer(F, patch_len, horizon,
                                    patch_len=patch_len, d_model=48)
            pred, loss, eig = train(m, x_in, x_out, epochs=epochs, koop_attr="koop")
            res_plot["Koop-Informer"] = {"pred": pred, "loss": loss,
                                         "eig": eig, "x_out": x_np}
            prefix = save_dir / f"Informer_{patch_len}_{horizon}"
            save_results(res_plot["Koop-Informer"], prefix, metrics_file,
                         patch_len, horizon)
            torch.save(m.cpu().state_dict(), f"{prefix}.pt")
            """
            # ----- Figures -----
            plot_training(res_plot,
                save_dir / f"train_patch{patch_len}_h{horizon}")
            plot_per_feature(x_np,
                {n: r["pred"] for n, r in res_plot.items()},
                F, H, save_dir / f"feat_patch{patch_len}_h{horizon}")

            # Console summary
            print("\nFinal metrics (train set):")
            for n, r in res_plot.items():
                e = r["x_out"] - r["pred"]
                print(f"{n:<18s} MSE {(e**2).mean():.4e} "
                      f"MAE {np.abs(e).mean():.4e}")

# --------------------------------------------------------------------------- #
# 14)  CLI                                                                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Koopformer-PRO benchmark (HPC-ready)")
    p.add_argument("--file", type=str, default="./df_cleaned_numeric_2.npy")
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--seq_len", type=int, default=120)
    p.add_argument("--save_dir", type=str, default="results")
    args = p.parse_args()
    main(file=args.file, epochs=args.epochs,
         seq_len=args.seq_len, save_dir=args.save_dir)
