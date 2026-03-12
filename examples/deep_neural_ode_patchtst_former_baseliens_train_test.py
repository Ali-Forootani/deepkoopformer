#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 08:34:19 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Koopformer-PRO benchmark (PatchTST backbone) — but replacing the Koopman latent
propagator with a NeuralODE latent propagator.

Key change:
  z_next = K z     (Koopman / linear latent)
becomes
  z_next = ODEsolve( dz/dt = f_theta(z,t), t in [0,1] )

No external deps (no torchdiffeq). Uses a simple fixed-step RK4 solver.

Ablations (PatchTST backbone):
  1) PatchTSTBaseline (pure Transformer, no latent propagator)
  2) PatchTST+LinearLatent (Transformer + unconstrained linear latent layer)
  3) PatchTST+NeuralODE_NoLyap (Transformer + NeuralODE latent, no Lyapunov)
  4) PatchTST+NeuralODE_Full   (Transformer + NeuralODE latent, with Lyapunov)

Additional baselines:
  - DLinearForecaster
  - SimpleLSTMForecaster
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
# 2)  NeuralODE latent propagator (RK4)                                       #
# --------------------------------------------------------------------------- #
class ODEFuncMLP(nn.Module):
    """dz/dt = f(z,t). We include t optionally by concatenation."""
    def __init__(self, latent_dim: int, hidden_dim: int = 128, use_time: bool = True):
        super().__init__()
        self.use_time = use_time
        inp = latent_dim + (1 if use_time else 0)
        self.net = nn.Sequential(
            nn.Linear(inp, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # t: scalar tensor or (B,1); z: (B,D)
        if self.use_time:
            if t.ndim == 0:
                tt = t.expand(z.shape[0], 1)
            elif t.ndim == 1:
                tt = t.view(-1, 1)
            else:
                tt = t
            x = torch.cat([z, tt], dim=-1)
        else:
            x = z
        return self.net(x)

class NeuralODELatentPropagator(nn.Module):
    """
    z_next = ODESolve(f, z0, t0=0, t1=1) using fixed-step RK4.
    Interface matches your Koopman modules: returns (z_next, K, Sigma),
    but K/Sigma are None here.
    """
    def __init__(self, latent_dim: int, hidden_dim: int = 128,
                 n_steps: int = 8, use_time: bool = True):
        super().__init__()
        self.f = ODEFuncMLP(latent_dim, hidden_dim=hidden_dim, use_time=use_time)
        self.n_steps = int(n_steps)

    def rk4_step(self, z, t, h):
        k1 = self.f(t, z)
        k2 = self.f(t + 0.5*h, z + 0.5*h*k1)
        k3 = self.f(t + 0.5*h, z + 0.5*h*k2)
        k4 = self.f(t + h,     z + h*k3)
        return z + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    def forward(self, z: torch.Tensor):
        # Integrate from t=0 to t=1
        h = 1.0 / self.n_steps
        t = torch.zeros((), device=z.device, dtype=z.dtype)
        zt = z
        for _ in range(self.n_steps):
            zt = self.rk4_step(zt, t, h)
            t = t + h
        return zt, None, None

class LinearLatentPropagator(nn.Module):
    """Unconstrained linear latent layer: z_next = W z."""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.W = nn.Linear(latent_dim, latent_dim, bias=False)

    def forward(self, z: torch.Tensor):
        K = self.W.weight.T
        return self.W(z), K, None

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
    def __init__(self, in_ch: int, d_model: int, patch_len: int, stride: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, d_model, kernel_size=patch_len, stride=stride)

    def forward(self, x):
        x = x.permute(0, 2, 1)     # (B, C, T)
        x = self.conv(x)           # (B, d, P)
        return x.permute(0, 2, 1)  # (B, P, d)

# --------------------------------------------------------------------------- #
# 4)  PatchTST backbone & variants                                            #
# --------------------------------------------------------------------------- #
class PatchTST_Backbone(nn.Module):
    def __init__(self, input_dim: int, seq_len: int,
                 patch_len: int, d_model: int = 64,
                 num_layers: int = 3, num_heads: int = 4,
                 dim_ff: int = 96):
        super().__init__()
        self.patch = PatchEmbed1D(input_dim, d_model, patch_len, patch_len)
        n_patches = int(np.ceil(seq_len / patch_len))
        self.pos  = SinCosPosEnc(d_model, max_len=n_patches + 1)
        enc = nn.TransformerEncoderLayer(d_model, num_heads, dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        x   = self.patch(x)                     # (B, P, d)
        cls = self.cls.expand(x.size(0), -1, -1)
        x   = torch.cat([cls, x], dim=1)        # (B, 1+P, d)
        x   = self.encoder(self.pos(x))
        return x[:, 0]                          # (B, d_model)

class PatchTSTBaseline(nn.Module):
    """Pure PatchTST + direct head."""
    def __init__(self, input_dim, seq_len, horizon, patch_len, d_model=64):
        super().__init__()
        self.backbone = PatchTST_Backbone(input_dim, seq_len, patch_len=patch_len, d_model=d_model)
        self.fc = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z = self.backbone(x)
        pred = self.fc(z)
        if return_latents:
            return pred, z, z
        return pred

class PatchTST_UnconstrainedLatent(nn.Module):
    """PatchTST + unconstrained linear latent propagation."""
    def __init__(self, input_dim, seq_len, horizon, patch_len, d_model=64):
        super().__init__()
        self.backbone = PatchTST_Backbone(input_dim, seq_len, patch_len=patch_len, d_model=d_model)
        self.latent   = LinearLatentPropagator(d_model)
        self.fc       = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z = self.backbone(x)
        z_next, _, _ = self.latent(z)
        pred = self.fc(z_next)
        if return_latents:
            return pred, z, z_next
        return pred

class NeuralODE_PatchTST(nn.Module):
    """PatchTST + NeuralODE latent propagation (+ optional Lyapunov loss in training)."""
    def __init__(self, input_dim, seq_len, horizon, patch_len, d_model=64,
                 ode_hidden=128, ode_steps=8, ode_use_time=True):
        super().__init__()
        self.backbone = PatchTST_Backbone(input_dim, seq_len, patch_len=patch_len, d_model=d_model)
        self.latent   = NeuralODELatentPropagator(
            latent_dim=d_model, hidden_dim=ode_hidden, n_steps=ode_steps, use_time=ode_use_time
        )
        self.fc       = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z = self.backbone(x)
        z_next, _, _ = self.latent(z)
        pred = self.fc(z_next)
        if return_latents:
            return pred, z, z_next
        return pred

# --------------------------------------------------------------------------- #
# 5)  LSTM baseline                                                           #
# --------------------------------------------------------------------------- #
class SimpleLSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, horizon):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, horizon * input_dim)

    def forward(self, x, return_latents=False):
        _, (h_n, _) = self.lstm(x)
        z = h_n[-1]
        out = self.fc(z)
        if return_latents:
            return out, z, z
        return out

# --------------------------------------------------------------------------- #
# 6)  DLinear baseline                                                        #
# --------------------------------------------------------------------------- #
class DLinearForecaster(nn.Module):
    """DLinear: Decomposition-Linear style baseline."""
    def __init__(self, input_dim, seq_len, horizon):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Conv1d(1, 1, kernel_size=seq_len, bias=True) for _ in range(input_dim)]
        )
        self.horizon   = horizon
        self.seq_len   = seq_len
        self.input_dim = input_dim

    def forward(self, x, return_latents=False):
        outs = []
        for i, linear in enumerate(self.linears):
            xi = x[:, :, i].unsqueeze(1)          # (B, 1, seq_len)
            if xi.shape[-1] < self.seq_len:
                pad = self.seq_len - xi.shape[-1]
                xi = torch.nn.functional.pad(xi, (pad, 0))
            yi = linear(xi)                       # (B, 1, 1) or (B, 1, H)
            if yi.shape[-1] == 1 and self.horizon > 1:
                yi = yi.expand(-1, -1, self.horizon)
            outs.append(yi)
        out = torch.cat(outs, dim=1)              # (B, F, H)
        out = out.permute(0, 2, 1).reshape(x.size(0), -1)  # (B, F·H)
        if return_latents:
            return out, out, out
        return out

# --------------------------------------------------------------------------- #
# 7)  Dataset util                                                            #
# --------------------------------------------------------------------------- #
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
# 8)  Training helper (Lyapunov optional; eig logging disabled for NeuralODE) #
# --------------------------------------------------------------------------- #
def train(model, x_in, x_out, epochs=4000, lr=3e-4, lyap_weight=0.1):
    model.to(DEV)
    x_in, x_out = x_in.to(DEV), x_out.to(DEV)
    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    losses = []

    with torch.no_grad():
        pred0, z0, z1 = model(x_in[:1], return_latents=True)
    P = torch.eye(z0.shape[1], device=DEV)

    for ep in range(epochs):
        model.train(); opt.zero_grad()
        pred, z, z_next = model(x_in, return_latents=True)
        pred_loss = mse(pred, x_out)

        if lyap_weight:
            zp  = torch.einsum("bi,ij->bj", z, P)
            znp = torch.einsum("bi,ij->bj", z_next, P)
            lyap = torch.relu((znp * z_next).sum(1) - (zp * z).sum(1)).mean()
        else:
            lyap = torch.zeros((), device=DEV)

        loss = pred_loss + lyap_weight * lyap
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

        if ep % max(1, epochs//10) == 0 or ep == epochs - 1:
            print(f"Epoch {ep:>4}/{epochs}  "
                  f"MSE {pred_loss.item():.5e}  "
                  f"Lyap {(lyap_weight*lyap).item():.5e}")

    model.eval()
    with torch.no_grad():
        preds = model(x_in).cpu().numpy()
    return preds, np.array(losses)

# --------------------------------------------------------------------------- #
# 9)  Plot helpers                                                            #
# --------------------------------------------------------------------------- #
_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
           "#bcbd22", "#17becf"]
_STYLES = ["-", "--", "-.", ":"]

def _save(fig, base):
    fig.savefig(f"{base}.png", dpi=600, bbox_inches="tight", transparent=True)
    fig.savefig(f"{base}.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)

def plot_training(res, path=None):
    if not res:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
    for i, (n, r) in enumerate(res.items()):
        ax.plot(r["loss"], label=n, color=_COLORS[i % 10],
                linestyle=_STYLES[i % 4], lw=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Loss")
    ax.grid(True, ls="--", alpha=.6)
    ax.legend()
    plt.tight_layout()
    if path:
        _save(fig, path)

def plot_per_feature(x_out: np.ndarray, preds: dict, F: int, H: int,
                     save_path: str | None = None):
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
            ax.plot(t, p[:, f], label=n,
                    color=_COLORS[i % len(_COLORS)],
                    linestyle=_STYLES[i % len(_STYLES)], linewidth=3)
        ax.set_ylabel("Value")
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        if f == 0:
            ax.legend(ncol=3)
    axes[-1].set_xlabel("Sample")
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)

# --------------------------------------------------------------------------- #
# 10)  Saving (Set column; optional test npy files)                           #
# --------------------------------------------------------------------------- #
def save_results(model_name: str, set_name: str,
                 prefix: Path, x_out: np.ndarray, pred: np.ndarray,
                 metrics_file: Path, patch_len: int, horizon: int,
                 save_npy: bool = False):
    err = x_out - pred
    mse = float((err**2).mean())
    mae = float(np.abs(err).mean())

    if save_npy:
        suffix = "_test" if set_name.lower() == "test" else "_train"
        np.save(f"{prefix}{suffix}_predictions.npy", pred)
        np.save(f"{prefix}{suffix}_errors.npy", err)

    pd.DataFrame({
        "Model":   [model_name],
        "PatchLen":[patch_len],
        "Horizon":[horizon],
        "Set":    [set_name],
        "MSE":    [mse],
        "MAE":    [mae],
    }).to_csv(metrics_file, mode="a", header=False, index=False)

# --------------------------------------------------------------------------- #
# 11)  Main loop with Train/Test split                                        #
# --------------------------------------------------------------------------- #
def main(file="./df_cleaned_numeric_2.npy", epochs=4000, seq_len=120,
         patch_lens=None, horizons=None, indices=None,
         save_dir="results_neuralode", train_frac: float = 0.8):

    patch_lens = patch_lens or [70, 80, 90, 100, 110, 120, 130]
    horizons   = horizons   or [2, 4, 6, 8, 10, 12, 14, 16]
    indices    = indices    or [0, 1, 2, 3, 4, 5]

    save_dir   = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = save_dir / "metrics.csv"
    pd.DataFrame(columns=["Model", "PatchLen", "Horizon", "Set", "MSE", "MAE"]).to_csv(
        metrics_file, index=False
    )

    raw = np.load(file)
    N = raw.shape[0]
    split = int(train_frac * N)
    train_data = raw[:split]
    test_data  = raw[split:]

    for patch_len in patch_lens:
        for horizon in horizons:
            print(f"\n>>> patch_len={patch_len}, horizon={horizon}\n")

            if len(train_data) < (patch_len + horizon) or len(test_data) < (patch_len + horizon):
                print("Not enough data for train/test window. Skipping.")
                continue

            x_in_train, x_out_train, scaler = build_dataset(train_data, indices, patch_len, horizon)
            x_in_test,  x_out_test,  _      = build_dataset(test_data,  indices, patch_len, horizon, scaler=scaler)

            F, H = len(indices), horizon
            x_np_train = x_out_train.numpy()
            x_np_test  = x_out_test.numpy()

            # --- Define variants (Koopman removed; NeuralODE added) -------- #
            model_variants = [
                ("DLinear",
                 DLinearForecaster(F, patch_len, horizon),
                 0.0),

                ("LSTM",
                 SimpleLSTMForecaster(F, hidden_dim=96, num_layers=2, horizon=horizon),
                 0.0),

                ("PatchTST",
                 PatchTSTBaseline(F, patch_len, horizon, patch_len=patch_len, d_model=96),
                 0.0),

                ("PatchTST+LinearLatent",
                 PatchTST_UnconstrainedLatent(F, patch_len, horizon, patch_len=patch_len, d_model=96),
                 0.0),

                ("PatchTST+NeuralODE_NoLyap",
                 NeuralODE_PatchTST(F, patch_len, horizon, patch_len=patch_len, d_model=96,
                                    ode_hidden=128, ode_steps=8, ode_use_time=True),
                 0.0),

                ("PatchTST+NeuralODE_Full",
                 NeuralODE_PatchTST(F, patch_len, horizon, patch_len=patch_len, d_model=96,
                                    ode_hidden=128, ode_steps=8, ode_use_time=True),
                 0.1),
            ]

            res_plot = {}
            for name, model, lyap_w in model_variants:
                print(f"== {name} ==")

                # ---- Train on train set ----------------------------------- #
                pred_train, loss = train(model, x_in_train, x_out_train, epochs=epochs, lyap_weight=lyap_w)

                # Save model
                prefix = save_dir / f"{name}_{patch_len}_{horizon}"
                torch.save(model.cpu().state_dict(), f"{prefix}.pt")

                # ---- Train metrics ---------------------------------------- #
                save_results(name, "Train", prefix, x_np_train, pred_train,
                             metrics_file, patch_len, horizon, save_npy=False)

                # ---- Test metrics & predictions --------------------------- #
                model.to(DEV)
                model.eval()
                with torch.no_grad():
                    pred_test = model(x_in_test.to(DEV)).cpu().numpy()

                save_results(name, "Test", prefix, x_np_test, pred_test,
                             metrics_file, patch_len, horizon, save_npy=True)

                res_plot[name] = {
                    "pred_train":  pred_train,
                    "pred_test":   pred_test,
                    "loss":        loss,
                    "x_out_train": x_np_train,
                    "x_out_test":  x_np_test,
                }

            # ----- Figures: training loss (train only) --------------------- #
            plot_training(res_plot, save_dir / f"train_patch{patch_len}_h{horizon}")

            # Per-feature plots: train/test
            plot_per_feature(
                x_np_train,
                {n: r["pred_train"] for n, r in res_plot.items()},
                F, H,
                save_dir / f"feat_patch{patch_len}_h{horizon}_train"
            )
            plot_per_feature(
                x_np_test,
                {n: r["pred_test"] for n, r in res_plot.items()},
                F, H,
                save_dir / f"feat_patch{patch_len}_h{horizon}_test"
            )

            print("\nFinal metrics (train set):")
            for n, r in res_plot.items():
                e = r["x_out_train"] - r["pred_train"]
                print(f"{n:<28s} MSE {(e**2).mean():.4e}  MAE {np.abs(e).mean():.4e}")

# --------------------------------------------------------------------------- #
# 12)  CLI                                                                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="PatchTST benchmark with NeuralODE latent propagator (HPC-ready)"
    )
    p.add_argument("--file", type=str, default="./df_cleaned_numeric_2.npy")
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--seq_len", type=int, default=120)  # kept for compatibility
    p.add_argument("--save_dir", type=str, default="results_neuralode")
    p.add_argument("--train_frac", type=float, default=0.8)
    args = p.parse_args()

    main(file=args.file, epochs=args.epochs, seq_len=args.seq_len,
         save_dir=args.save_dir, train_frac=args.train_frac)
