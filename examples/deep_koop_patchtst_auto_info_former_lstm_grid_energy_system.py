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
        # --- make sure trend has the same length as x ---------------------- #
        if trend.size(1) != x.size(1):              # ### FIX 1 ###
            trend = trend[:, : x.size(1)]           # ### FIX 2 ###
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
            x = self.patch(x)          # already d_model
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
# 8)  Dataset util                                                            #
# --------------------------------------------------------------------------- #

# 5) Dataset util: accept optional scaler and allow arbitrary-length data
def build_dataset(data: np.ndarray, indices: list[int],
                  seq_len: int, horizon: int, scaler=None):
    sub = data[:, indices]
    if scaler is None:
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
# 9)  Training helper                                                         #
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

    # size for Lyapunov
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
# 10)  Plot helpers                                                           #
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
# 11)  Saving                                                                 #
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
# 12)  Main loop                                                              #
# --------------------------------------------------------------------------- #


def main(file="bitstamp_windows.npy", epochs=200, seq_len=120,
         patch_lens=None, horizons=None, indices=None,
         save_dir="results", train_frac=0.8):

    patch_lens = patch_lens or [70, 80, 90, 100, 110, 120, 130]
    horizons   = horizons   or [2, 4, 6, 8, 10, 12, 14, 16]
    indices    = indices    or [0, 1, 2, 3, 4, 5]

    save_dir   = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = save_dir / "metrics.csv"
    pd.DataFrame(columns=["Model", "PatchLen", "Horizon", "Set", "MSE", "MAE"]
        ).to_csv(metrics_file, index=False)

    raw = np.load(file)
    N = raw.shape[0]
    split = int(train_frac * N)
    train_data = raw[:split]
    test_data  = raw[split:]

    for patch_len in patch_lens:
        for horizon in horizons:
            print(f"\n>>> patch_len={patch_len}, horizon={horizon}\n")
            if len(test_data) < (patch_len + horizon):
                print("Not enough data for test window. Skipping.")
                continue

            x_in_train, x_out_train, scaler = build_dataset(
                train_data, indices, patch_len, horizon)
            x_in_test,  x_out_test, _ = build_dataset(
                test_data, indices, patch_len, horizon, scaler=scaler)

            F, H = len(indices), horizon
            x_np_train = x_out_train.numpy()
            x_np_test  = x_out_test.numpy()
            res_plot = {}

            # === Model Variants ===
            model_variants = {
                "LSTM": SimpleLSTMForecaster(F, 48, 2, H),
                "Koop-PatchTST": Koopformer_PatchTST(F, patch_len, H, patch_len, d_model= 96),
                "Koop-Autoformer": Koopformer_Autoformer(F, patch_len, H, patch_len, d_model= 96),
                "Koop-Informer": Koopformer_Informer(F, patch_len, H, patch_len, d_model= 96),
            }

            for name, model in model_variants.items():
                print(f"== {name} ==")
                pred_train, loss, eig = train(model, x_in_train, x_out_train,
                                              epochs=epochs,
                                              koop_attr="koop" if "Koop" in name else None,
                                              lyap_weight=0.1 if "Koop" in name else 0.0)

                # Save model
                prefix = save_dir / f"{name.replace('Koop-', '')}_{patch_len}_{horizon}"
                torch.save(model.cpu().state_dict(), f"{prefix}.pt")
                
                # Save training metrics
                err_train = x_np_train - pred_train
                mse_train = float((err_train ** 2).mean())
                mae_train = float(np.abs(err_train).mean())
                pd.DataFrame({
                    "Model": [name],
                    "PatchLen": [patch_len],
                    "Horizon": [horizon],
                    "Set": ["Train"],
                    "MSE": [mse_train],
                    "MAE": [mae_train],
                }).to_csv(metrics_file, mode="a", header=False, index=False)
                
                # Put model back to correct device for test
                model.to(DEV)
                # Save test metrics
                model.eval()
                with torch.no_grad():
                    pred_test = model(x_in_test.to(DEV)).cpu().numpy()
                    err_test = x_np_test - pred_test
                    mse_test = float((err_test ** 2).mean())
                    mae_test = float(np.abs(err_test).mean())

                pd.DataFrame({
                    "Model": [name],
                    "PatchLen": [patch_len],
                    "Horizon": [horizon],
                    "Set": ["Test"],
                    "MSE": [mse_test],
                    "MAE": [mae_test],
                }).to_csv(metrics_file, mode="a", header=False, index=False)

                np.save(f"{prefix}_test_predictions.npy", pred_test)
                np.save(f"{prefix}_test_errors.npy", err_test)

                # Store for plotting
                res_plot[name] = {
                    "pred": pred_train,
                    "loss": loss,
                    "eig": eig,
                    "x_out": x_np_train,
                }

            # === Plot results ===
            plot_training(res_plot,
                save_dir / f"train_patch{patch_len}_h{horizon}")
            plot_per_feature(x_np_train,
                {n: r["pred"] for n, r in res_plot.items()},
                F, H, save_dir / f"feat_patch{patch_len}_h{horizon}_train")
            plot_per_feature(x_np_test,
                {n.replace("Koop-", ""): np.load(save_dir / f"{n.replace('Koop-', '')}_{patch_len}_{horizon}_test_predictions.npy")
                 for n in model_variants},
                F, H, save_dir / f"feat_patch{patch_len}_h{horizon}_test")

            print("\nFinal metrics (train set):")
            for n, r in res_plot.items():
                e = r["x_out"] - r["pred"]
                print(f"{n:<15s} MSE {(e**2).mean():.4e}  MAE {np.abs(e).mean():.4e}")


# --------------------------------------------------------------------------- #
# 13)  CLI                                                                    #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Koopformer-PRO benchmark (HPC-ready, with test eval)")
    p.add_argument("--file", type=str, default="./df_cleaned_numeric.npy")
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--seq_len", type=int, default=120)
    p.add_argument("--save_dir", type=str, default="results")
    p.add_argument("--train_frac", type=float, default=0.8,
                   help="Fraction of data to use for training (rest for test)")
    args = p.parse_args()
    main(file=args.file, epochs=args.epochs,
         seq_len=args.seq_len, save_dir=args.save_dir, train_frac=args.train_frac)
