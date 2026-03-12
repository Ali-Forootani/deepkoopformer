#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 13:46:27 2026

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_liquid_all_variants.py

Code-1 procedure, but replacing NeuralODE with Liquid Neural Network (LNN)
latent propagators (Euler integration, no external deps).

Compare ALL variants together:
  - Baselines: DLinear, LSTM, PatchTST
  - Linear latent: PatchTST+LinearLatent
  - LNN (original): PatchTST+LNN_Euler_{NoLyap,Full}
  - LNN (augmented solutions):
      * Stable linear+residual Liquid dynamics (A = -B^T B)
      * + Jacobian regularization
      * + Multi-step latent rollout loss
      * (Euler steps orig vs aug)

HPC-safe matplotlib backend.

Author: Ali Forootani
Adapted: 2026-01-03
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
# 2)  Liquid Neural Network latent dynamics (Code-2 style)                     #
# --------------------------------------------------------------------------- #
class LiquidCell(nn.Module):
    """
    LTC-inspired latent dynamics:
        dz/dt = -(1/tau(z,t))*z + (1/tau(z,t))*drive(z,t)
    where tau(z,t) > 0 is learnable (bounded), drive is a small MLP of z.
    """
    def __init__(self, latent_dim: int, hidden_dim: int = 128, use_time: bool = True,
                 tau_min: float = 0.05, tau_max: float = 5.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_time = use_time
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)

        inp = latent_dim + (1 if use_time else 0)

        self.drive = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.tau_net = nn.Sequential(
            nn.Linear(inp, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def _tau(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
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

        tau_raw = self.tau_net(x)
        tau_pos = torch.nn.functional.softplus(tau_raw) + 1e-6
        tau = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(tau_pos)
        return tau

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        tau = self._tau(t, z)               # (B, d)
        inv_tau = 1.0 / tau
        drive = self.drive(z)               # (B, d)
        dz = -inv_tau * z + inv_tau * drive
        return dz


class LiquidLatentPropagator(nn.Module):
    """
    Integrate liquid dynamics from t0=0 to t1=1 with fixed Euler steps.
    forward(z) -> (z_next, None, None)  (Koopman-compatible interface)
    """
    def __init__(self, latent_dim: int, hidden_dim: int = 128,
                 n_steps: int = 16, use_time: bool = True,
                 tau_min: float = 0.05, tau_max: float = 5.0):
        super().__init__()
        self.cell = LiquidCell(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            use_time=use_time,
            tau_min=tau_min,
            tau_max=tau_max,
        )
        self.n_steps = int(n_steps)

    def forward(self, z: torch.Tensor):
        h = 1.0 / self.n_steps
        t = torch.zeros((), device=z.device, dtype=z.dtype)
        zt = z
        for _ in range(self.n_steps):
            dz = self.cell(t, zt)
            zt = zt + h * dz
            t = t + h
        return zt, None, None


class StableResidualLiquidCell(nn.Module):
    r"""
    Augmented liquid dynamics with a *stable linear* term:
        dz/dt = A z + liquid(z,t)
    where A = -B^T B  (negative semidefinite, stable in continuous time).
    """
    def __init__(self, latent_dim: int, hidden_dim: int = 128, use_time: bool = True,
                 stable_rank: int | None = None,
                 tau_min: float = 0.05, tau_max: float = 5.0):
        super().__init__()
        self.latent_dim = latent_dim
        r = stable_rank if stable_rank is not None else latent_dim
        self.B = nn.Parameter(0.02 * torch.randn(r, latent_dim))

        self.liquid = LiquidCell(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            use_time=use_time,
            tau_min=tau_min,
            tau_max=tau_max,
        )

    def A_mat(self):
        return -(self.B.transpose(0, 1) @ self.B)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        A = self.A_mat()  # (d,d)
        lin = z @ A.transpose(0, 1)
        return lin + self.liquid(t, z)


class LiquidLatentPropagator_StableResidual(nn.Module):
    """
    Stable linear + residual liquid dynamics integrated with Euler steps.
    Returns (z_next, A, None) for logging/regularization hooks (like Code-1).
    """
    def __init__(self, latent_dim: int, hidden_dim: int = 128,
                 n_steps: int = 32, use_time: bool = True,
                 stable_rank: int | None = None,
                 tau_min: float = 0.05, tau_max: float = 5.0):
        super().__init__()
        self.f = StableResidualLiquidCell(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            use_time=use_time,
            stable_rank=stable_rank,
            tau_min=tau_min,
            tau_max=tau_max,
        )
        self.n_steps = int(n_steps)

    def forward(self, z: torch.Tensor):
        h = 1.0 / self.n_steps
        t = torch.zeros((), device=z.device, dtype=z.dtype)
        zt = z
        for _ in range(self.n_steps):
            dz = self.f(t, zt)
            zt = zt + h * dz
            t = t + h
        A = self.f.A_mat()
        return zt, A, None


class LinearLatentPropagator(nn.Module):
    """Unconstrained linear latent layer: z_next = W z."""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.W = nn.Linear(latent_dim, latent_dim, bias=False)

    def forward(self, z: torch.Tensor):
        K = self.W.weight.T
        return self.W(z), K, None


def jacobian_proxy_sq(func, z: torch.Tensor, create_graph: bool = True):
    """Cheap Jacobian/Lipschitz proxy: ||d sum(func(z))/dz||^2."""
    z = z.requires_grad_(True)
    out = func(z)
    s = out.sum()
    grad = torch.autograd.grad(s, z, create_graph=create_graph, retain_graph=True)[0]
    return (grad**2).mean()

# --------------------------------------------------------------------------- #
# 3)  Positional encodings & patch embed                                      #
# --------------------------------------------------------------------------- #
class SinCosPosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10_000):
        super().__init__()
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10_000.0) / d_model))
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


class LNN_PatchTST(nn.Module):
    """PatchTST + LNN latent propagation (Euler)."""
    def __init__(
        self,
        input_dim,
        seq_len,
        horizon,
        patch_len,
        d_model=64,
        lnn_hidden=128,
        lnn_use_time=True,
        lnn_steps=16,
        tau_min=0.05,
        tau_max=5.0,
    ):
        super().__init__()
        self.backbone = PatchTST_Backbone(input_dim, seq_len, patch_len=patch_len, d_model=d_model)
        self.latent = LiquidLatentPropagator(
            latent_dim=d_model,
            hidden_dim=lnn_hidden,
            n_steps=lnn_steps,
            use_time=lnn_use_time,
            tau_min=tau_min,
            tau_max=tau_max,
        )
        self.fc = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z = self.backbone(x)
        z_next, _, _ = self.latent(z)
        pred = self.fc(z_next)
        if return_latents:
            return pred, z, z_next
        return pred


class LNNStableResidual_PatchTST(nn.Module):
    """PatchTST + stable linear + residual LNN latent propagation (Euler)."""
    def __init__(
        self,
        input_dim,
        seq_len,
        horizon,
        patch_len,
        d_model=64,
        lnn_hidden=128,
        lnn_use_time=True,
        stable_rank=None,
        lnn_steps=32,
        tau_min=0.05,
        tau_max=5.0,
    ):
        super().__init__()
        self.backbone = PatchTST_Backbone(input_dim, seq_len, patch_len=patch_len, d_model=d_model)
        self.latent = LiquidLatentPropagator_StableResidual(
            latent_dim=d_model,
            hidden_dim=lnn_hidden,
            n_steps=lnn_steps,
            use_time=lnn_use_time,
            stable_rank=stable_rank,
            tau_min=tau_min,
            tau_max=tau_max,
        )
        self.fc = nn.Linear(d_model, horizon * input_dim)

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
            yi = linear(xi)
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
# 8)  Training helpers (same structure as Code-1)                             #
# --------------------------------------------------------------------------- #
def train_one(model, x_in, x_out,
              epochs=4000, lr=3e-4,
              lyap_weight=0.0,
              jac_weight=0.0,
              rollout_steps: int = 1,
              grad_clip: float = 1.0):
    """Train with optional Lyapunov, Jacobian reg, and multi-step latent rollout."""
    model.to(DEV)
    x_in, x_out = x_in.to(DEV), x_out.to(DEV)

    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    losses = []

    def _s(x):
        if torch.is_tensor(x):
            return float(x.detach().cpu().item())
        return float(x)

    with torch.no_grad():
        _, z0, _ = model(x_in[:1], return_latents=True)
    P = torch.eye(z0.shape[1], device=DEV)

    latent_module = getattr(model, "latent", None)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()

        pred0, z, z_next = model(x_in, return_latents=True)

        if rollout_steps <= 1 or latent_module is None:
            pred_loss = mse(pred0, x_out)
            lyap_loss = torch.zeros((), device=DEV)
            jac_loss  = torch.zeros((), device=DEV)

            if lyap_weight and latent_module is not None:
                zp  = torch.einsum("bi,ij->bj", z, P)
                znp = torch.einsum("bi,ij->bj", z_next, P)
                lyap_loss = torch.relu((znp * z_next).sum(1) - (zp * z).sum(1)).mean()

            # Jacobian proxy: depends on latent_module having "cell" or "f"
            if jac_weight and latent_module is not None:
                if hasattr(latent_module, "cell"):  # LiquidLatentPropagator
                    cell = latent_module.cell
                    jac_loss = jacobian_proxy_sq(
                        lambda zz: cell(torch.zeros((), device=zz.device, dtype=zz.dtype), zz),
                        z,
                        create_graph=True
                    )
                elif hasattr(latent_module, "f"):   # StableResidual version
                    f = latent_module.f
                    jac_loss = jacobian_proxy_sq(
                        lambda zz: f(torch.zeros((), device=zz.device, dtype=zz.dtype), zz),
                        z,
                        create_graph=True
                    )

            loss = pred_loss + lyap_weight * lyap_loss + jac_weight * jac_loss

        else:
            zt = z
            pred_loss = torch.zeros((), device=DEV)
            lyap_loss = torch.zeros((), device=DEV)
            jac_loss  = torch.zeros((), device=DEV)

            for _k in range(rollout_steps):
                zt_next, _, _ = latent_module(zt)
                pred_k = model.fc(zt_next)
                pred_loss = pred_loss + mse(pred_k, x_out)

                if lyap_weight:
                    zp  = torch.einsum("bi,ij->bj", zt, P)
                    znp = torch.einsum("bi,ij->bj", zt_next, P)
                    lyap_k = torch.relu((znp * zt_next).sum(1) - (zp * zt).sum(1)).mean()
                    lyap_loss = lyap_loss + lyap_k

                if jac_weight:
                    if hasattr(latent_module, "cell"):
                        cell = latent_module.cell
                        jac_k = jacobian_proxy_sq(
                            lambda zz: cell(torch.zeros((), device=zz.device, dtype=zz.dtype), zz),
                            zt,
                            create_graph=True
                        )
                        jac_loss = jac_loss + jac_k
                    elif hasattr(latent_module, "f"):
                        f = latent_module.f
                        jac_k = jacobian_proxy_sq(
                            lambda zz: f(torch.zeros((), device=zz.device, dtype=zz.dtype), zz),
                            zt,
                            create_graph=True
                        )
                        jac_loss = jac_loss + jac_k

                zt = zt_next

            pred_loss = pred_loss / rollout_steps
            lyap_loss = lyap_loss / rollout_steps
            jac_loss  = jac_loss / rollout_steps
            loss = pred_loss + lyap_weight * lyap_loss + jac_weight * jac_loss

        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        losses.append(_s(loss))

        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            print(f"Epoch {ep:>4}/{epochs}  "
                  f"MSE {_s(pred_loss):.5e}  "
                  f"Lyap {_s(lyap_weight * lyap_loss):.5e}  "
                  f"Jac {_s(jac_weight * jac_loss):.5e}")

    model.eval()
    with torch.no_grad():
        preds_train = model(x_in).cpu().numpy()
    return preds_train, np.array(losses)

# --------------------------------------------------------------------------- #
# 9) Plot helpers                                                             #
# --------------------------------------------------------------------------- #
_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
           "#bcbd22", "#17becf"]
_STYLES = ["-", "--", "-.", ":"]

def _save(fig, base: Path):
    fig.savefig(f"{base}.png", dpi=600, bbox_inches="tight", transparent=True)
    fig.savefig(f"{base}.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)

def plot_training(res, path: Path | None = None):
    if not res:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
    for i, (n, r) in enumerate(res.items()):
        ax.plot(r["loss"], label=n, color=_COLORS[i % 10], linestyle=_STYLES[i % 4], lw=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Loss")
    ax.grid(True, ls="--", alpha=.6)
    ax.legend()
    plt.tight_layout()
    if path:
        _save(fig, path)

def plot_per_feature(x_out: np.ndarray, preds: dict, F: int, H: int,
                     save_path: Path | None = None):
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
            ax.plot(t, p[:, f], label=n, color=_COLORS[i % len(_COLORS)],
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
# 10) Saving metrics                                                          #
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
# 11) Main loop (Code-1 structure, LNN variants instead of NeuralODE)         #
# --------------------------------------------------------------------------- #
def main(file="./df_cleaned_numeric_2.npy", epochs=4000,
         patch_lens=None, horizons=None, indices=None,
         save_dir="results_all_liquid_variants", train_frac: float = 0.8,
         rollout_steps_aug: int = 5, jac_weight_aug: float = 1e-4,
         # LNN steps
         lnn_steps_orig: int = 16, lnn_steps_aug: int = 32,
         # LNN tau bounds
         tau_min: float = 0.05, tau_max: float = 5.0):

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

            # -------------------- Define ALL variants -------------------- #
            model_variants = [
                ("DLinear",
                 DLinearForecaster(F, patch_len, horizon),
                 dict(lyap=0.0, jac=0.0, roll=1)),

                ("LSTM",
                 SimpleLSTMForecaster(F, hidden_dim=96, num_layers=2, horizon=horizon),
                 dict(lyap=0.0, jac=0.0, roll=1)),

                ("PatchTST",
                 PatchTSTBaseline(F, patch_len, horizon, patch_len=patch_len, d_model=96),
                 dict(lyap=0.0, jac=0.0, roll=1)),

                ("PatchTST+LinearLatent",
                 PatchTST_UnconstrainedLatent(F, patch_len, horizon, patch_len=patch_len, d_model=96),
                 dict(lyap=0.0, jac=0.0, roll=1)),

                # -------- Original LNN: Euler steps (orig) --------
                ("PatchTST+LNN_Euler_NoLyap",
                 LNN_PatchTST(
                     F, patch_len, horizon, patch_len=patch_len, d_model=96,
                     lnn_hidden=128, lnn_use_time=True,
                     lnn_steps=lnn_steps_orig, tau_min=tau_min, tau_max=tau_max
                 ),
                 dict(lyap=0.0, jac=0.0, roll=1)),

                ("PatchTST+LNN_Euler_Full",
                 LNN_PatchTST(
                     F, patch_len, horizon, patch_len=patch_len, d_model=96,
                     lnn_hidden=128, lnn_use_time=True,
                     lnn_steps=lnn_steps_orig, tau_min=tau_min, tau_max=tau_max
                 ),
                 dict(lyap=0.1, jac=0.0, roll=1)),

                # -------- StableResidual LNN (aug): Euler steps (aug) + regs --------
                ("PatchTST+LNN_StableResidual_Euler",
                 LNNStableResidual_PatchTST(
                     F, patch_len, horizon, patch_len=patch_len, d_model=96,
                     lnn_hidden=128, lnn_use_time=True, stable_rank=None,
                     lnn_steps=lnn_steps_aug, tau_min=tau_min, tau_max=tau_max
                 ),
                 dict(lyap=0.0, jac=jac_weight_aug, roll=rollout_steps_aug)),

                ("PatchTST+LNN_StableResidual_Euler+Lyap",
                 LNNStableResidual_PatchTST(
                     F, patch_len, horizon, patch_len=patch_len, d_model=96,
                     lnn_hidden=128, lnn_use_time=True, stable_rank=None,
                     lnn_steps=lnn_steps_aug, tau_min=tau_min, tau_max=tau_max
                 ),
                 dict(lyap=0.1, jac=jac_weight_aug, roll=rollout_steps_aug)),
            ]

            res_plot = {}
            for name, model, cfg in model_variants:
                print(f"== {name} ==")

                pred_train, loss = train_one(
                    model,
                    x_in_train,
                    x_out_train,
                    epochs=epochs,
                    lr=3e-4,
                    lyap_weight=float(cfg["lyap"]),
                    jac_weight=float(cfg["jac"]),
                    rollout_steps=int(cfg["roll"]),
                    grad_clip=1.0,
                )

                prefix = save_dir / f"{name}_{patch_len}_{horizon}"
                torch.save(model.cpu().state_dict(), f"{prefix}.pt")

                save_results(name, "Train", prefix, x_np_train, pred_train,
                             metrics_file, patch_len, horizon, save_npy=False)

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

            plot_training(res_plot, save_dir / f"train_patch{patch_len}_h{horizon}")

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
                print(f"{n:<48s} MSE {(e**2).mean():.4e}  MAE {np.abs(e).mean():.4e}")

# --------------------------------------------------------------------------- #
# 12)  CLI                                                                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="PatchTST benchmark: compare ALL variants incl. Liquid (LNN) families (HPC-ready)"
    )
    p.add_argument("--file", type=str, default="./df_cleaned_numeric_2.npy")
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--save_dir", type=str, default="results/results_all_liquid_variants")
    p.add_argument("--train_frac", type=float, default=0.8)

    # Augmented training knobs
    p.add_argument("--rollout_steps_aug", type=int, default=5)
    p.add_argument("--jac_weight_aug", type=float, default=1e-4)

    # LNN integration knobs
    p.add_argument("--lnn_steps_orig", type=int, default=16)
    p.add_argument("--lnn_steps_aug", type=int, default=32)
    p.add_argument("--tau_min", type=float, default=0.05)
    p.add_argument("--tau_max", type=float, default=5.0)

    args = p.parse_args()

    main(
        file=args.file,
        epochs=args.epochs,
        save_dir=args.save_dir,
        train_frac=args.train_frac,
        rollout_steps_aug=args.rollout_steps_aug,
        jac_weight_aug=args.jac_weight_aug,
        lnn_steps_orig=args.lnn_steps_orig,
        lnn_steps_aug=args.lnn_steps_aug,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
    )
