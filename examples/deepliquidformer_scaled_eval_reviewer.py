#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLiquidFormer/Koopformer benchmark on Wind-Speed (or similar) data (HPC-ready)

This version AUGMENTS your benchmark with a NeuralODE latent propagator family,
in addition to:
  - constrained Koopman
  - learnable Koopman family
  - unconstrained Koopman
  - baselines (LSTM, DLinear)

NeuralODE family (per-backbone):
  - NeuralODE_PatchTST
  - NeuralODE_Autoformer
  - NeuralODE_Informer

Notes:
  • No torchdiffeq dependency: fixed-step RK4 ODE solver implemented here.
  • Lyapunov regularization can also be used for NeuralODE (on (z, z_next)).
  • Evaluation is performed ONLY in the MinMax-scaled target space; no inverse-transformed/original-scale metrics are reported.

2025-11-29  – Ali Forootani
2025-12-17  – NeuralODE latent propagator added
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
# 2)  Koopman operators                                                       #
# --------------------------------------------------------------------------- #
def _orth(w: torch.Tensor) -> torch.Tensor:
    return torch.linalg.qr(w)[0]

class StrictStableKoopmanOperator(nn.Module):
    """ODO-style Koopman parameterisation with spectral radius < ρ_max."""
    def __init__(self, latent_dim: int, ρ_max: float = 0.99):
        super().__init__()
        self.U_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.V_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.S_raw = nn.Parameter(torch.randn(latent_dim))
        self.ρ_max = ρ_max

    def _sigma(self) -> torch.Tensor:
        return torch.sigmoid(self.S_raw) * self.ρ_max

    def forward(self, z: torch.Tensor):
        U, V = _orth(self.U_raw), _orth(self.V_raw)
        Σ = self._sigma()
        K = U @ torch.diag(Σ) @ V.T
        z_next = z @ K.T
        return z_next, K, Σ

class LearnableKoopmanBase(nn.Module):
    def _sigma(self) -> torch.Tensor:
        raise NotImplementedError

class LearnableKoopmanOperatorScalar(LearnableKoopmanBase):
    def __init__(self, latent_dim: int, ρ_max: float = 0.99):
        super().__init__()
        self.U_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.V_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.S_raw = nn.Parameter(torch.randn(latent_dim))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(0.0))
        self.ρ_max = ρ_max

    def _sigma(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha * self.S_raw + self.beta) * self.ρ_max

    def forward(self, z: torch.Tensor):
        U, V = _orth(self.U_raw), _orth(self.V_raw)
        Σ = self._sigma()
        K = U @ torch.diag(Σ) @ V.T
        return z @ K.T, K, Σ

class LearnableKoopmanOperatorPerMode(LearnableKoopmanBase):
    def __init__(self, latent_dim: int, ρ_max: float = 0.99):
        super().__init__()
        self.U_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.V_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.S_raw = nn.Parameter(torch.randn(latent_dim))
        self.alpha = nn.Parameter(torch.ones(latent_dim))
        self.beta  = nn.Parameter(torch.zeros(latent_dim))
        self.ρ_max = ρ_max

    def _sigma(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha * self.S_raw + self.beta) * self.ρ_max

    def forward(self, z: torch.Tensor):
        U, V = _orth(self.U_raw), _orth(self.V_raw)
        Σ = self._sigma()
        K = U @ torch.diag(Σ) @ V.T
        return z @ K.T, K, Σ

class LearnableKoopmanOperatorMLP(LearnableKoopmanBase):
    def __init__(self, latent_dim: int, ρ_max: float = 0.99, hidden_dim: int = 16):
        super().__init__()
        self.U_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.V_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.S_raw = nn.Parameter(torch.randn(latent_dim))
        self.ρ_max = ρ_max
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def _sigma(self) -> torch.Tensor:
        s = self.S_raw.view(-1, 1)
        g_s = self.mlp(s).view(-1)
        return torch.sigmoid(g_s) * self.ρ_max

    def forward(self, z: torch.Tensor):
        U, V = _orth(self.U_raw), _orth(self.V_raw)
        Σ = self._sigma()
        K = U @ torch.diag(Σ) @ V.T
        return z @ K.T, K, Σ

class LearnableKoopmanOperatorLowRank(LearnableKoopmanBase):
    def __init__(self, latent_dim: int, rank: int, ρ_max: float = 0.99):
        super().__init__()
        self.d = latent_dim
        self.r = rank
        self.U_raw = nn.Parameter(torch.randn(latent_dim, rank))
        self.V_raw = nn.Parameter(torch.randn(latent_dim, rank))
        self.S_raw = nn.Parameter(torch.randn(rank))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(0.0))
        self.ρ_max = ρ_max

    def _sigma(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha * self.S_raw + self.beta) * self.ρ_max

    def forward(self, z: torch.Tensor):
        U, _ = torch.linalg.qr(self.U_raw, mode="reduced")
        V, _ = torch.linalg.qr(self.V_raw, mode="reduced")
        Σ = self._sigma()
        K = U @ torch.diag(Σ) @ V.T
        return z @ K.T, K, Σ

class UnconstrainedKoopmanOperator(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.K_raw = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.01)

    def forward(self, z: torch.Tensor):
        K = self.K_raw
        return z @ K.T, K, None

def make_learnable_koopman(kind: str, latent_dim: int, ρ_max: float = 0.99):
    key = kind.lower()
    if key == "scalar":
        return LearnableKoopmanOperatorScalar(latent_dim, ρ_max=ρ_max)
    if key == "permode":
        return LearnableKoopmanOperatorPerMode(latent_dim, ρ_max=ρ_max)
    if key == "mlp":
        return LearnableKoopmanOperatorMLP(latent_dim, ρ_max=ρ_max)
    if key.startswith("lowrank"):
        r = latent_dim // 2
        suffix = key.replace("lowrank", "")
        if suffix.strip():
            try:
                r = int(suffix)
            except ValueError:
                pass
        r = max(1, min(latent_dim, r))
        return LearnableKoopmanOperatorLowRank(latent_dim, rank=r, ρ_max=ρ_max)
    raise ValueError(f"Unknown learnable Koopman kind: {kind}")

# --------------------------------------------------------------------------- #
# 3)  NeuralODE latent propagator (RK4, fixed steps; no torchdiffeq)          #
# --------------------------------------------------------------------------- #
class ODEFuncMLP(nn.Module):
    """dz/dt = f(z,t) (optionally time-augmented)."""
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
    z_next = ODESolve(f, z0, t0=0, t1=1) via fixed-step RK4.
    Interface matches Koopman: forward(z) -> (z_next, K, Σ) with K/Σ=None.
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
        h = 1.0 / self.n_steps
        t = torch.zeros((), device=z.device, dtype=z.dtype)
        zt = z
        for _ in range(self.n_steps):
            zt = self.rk4_step(zt, t, h)
            t = t + h
        return zt, None, None

# --------------------------------------------------------------------------- #
# 3b) Liquid Neural Network latent propagator (LTC-style; fixed Euler steps)  #
# --------------------------------------------------------------------------- #
class LiquidCell(nn.Module):
    """
    Continuous-time liquid dynamics (LTC-inspired):
        dz/dt = -(1/tau(z,t)) * z + (1/tau(z,t)) * g(Wz*z + b)
    where tau(z,t) is positive and learnable (optionally state/time dependent).

    This is NOT meant to replicate any specific paper verbatim; it's a robust,
    dependency-free "liquid" continuous-time recurrent dynamic for latent z.
    """
    def __init__(self, latent_dim: int, hidden_dim: int = 128, use_time: bool = True,
                 tau_min: float = 0.05, tau_max: float = 5.0,
                 tau_mode: str = "state_time"): # state_time | constant
        super().__init__()
        self.latent_dim = latent_dim
        self.use_time = use_time
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.tau_mode = tau_mode.lower()
        if self.tau_mode not in {"state_time", "constant"}:
            raise ValueError(f"Unsupported tau_mode={tau_mode}. Use state_time or constant.")

        inp = latent_dim + (1 if use_time else 0)

        # "drive" term
        self.drive = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # state/time dependent time-constant (positive via softplus).
        # For the reviewer ablation tau_mode="constant", this network is replaced
        # by one learnable vector shared across samples and integration time.
        if self.tau_mode == "constant":
            self.tau_raw = nn.Parameter(torch.zeros(latent_dim))
            self.tau_net = None
        else:
            self.tau_net = nn.Sequential(
                nn.Linear(inp, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, latent_dim),
            )

    def _tau(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.tau_mode == "constant":
            tau_pos = torch.nn.functional.softplus(self.tau_raw) + 1e-6
            tau = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(tau_pos)
            return tau.view(1, -1).expand_as(z)

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

        # tau in (tau_min, tau_max)
        tau_raw = self.tau_net(x)
        tau_pos = torch.nn.functional.softplus(tau_raw) + 1e-6
        tau = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(tau_pos)
        return tau

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        tau = self._tau(t, z)               # (B, d)
        inv_tau = 1.0 / tau                 # (B, d)
        drive = self.drive(z)               # (B, d)
        dz = -inv_tau * z + inv_tau * drive
        return dz


class LiquidLatentPropagator(nn.Module):
    """
    z_next = integrate liquid dynamics from t0=0 to t1=1 using fixed Euler steps.
    Interface matches Koopman/NeuralODE: forward(z) -> (z_next, None, None).
    """
    def __init__(self, latent_dim: int, hidden_dim: int = 128,
                 n_steps: int = 16, use_time: bool = True,
                 tau_min: float = 0.05, tau_max: float = 5.0,
                 tau_mode: str = "state_time"):
        super().__init__()
        self.cell = LiquidCell(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            use_time=use_time,
            tau_min=tau_min,
            tau_max=tau_max,
            tau_mode=tau_mode,
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



# --------------------------------------------------------------------------- #
# 4)  Positional encodings & patch embed                                      #
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]

class PatchEmbed1D(nn.Module):
    def __init__(self, in_ch: int, d_model: int, patch_len: int, stride: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, d_model, kernel_size=patch_len, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)     # (B, C, T)
        x = self.conv(x)           # (B, d, P)
        return x.permute(0, 2, 1)  # (B, P, d)

# --------------------------------------------------------------------------- #
# 5)  Backbones + (Koopman / NeuralODE) wrappers                              #
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = self.patch(x)                  # (B, P, d)
        cls = self.cls.expand(x.size(0), -1, -1)
        x   = torch.cat([cls, x], dim=1)     # (B, P+1, d)
        x   = self.encoder(self.pos(x))
        return x[:, 0]                       # (B, d)

# -------------------- PatchTST family -------------------- #
class Koopformer_PatchTST_Constrained(nn.Module):
    def __init__(self, input_dim, seq_len, horizon, patch_len, d_model=64, ρ_max: float = 0.99):
        super().__init__()
        self.backbone = PatchTST_Backbone(input_dim, seq_len, patch_len=patch_len, d_model=d_model)
        self.koop = StrictStableKoopmanOperator(d_model, ρ_max=ρ_max)
        self.fc   = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z = self.backbone(x)
        z_next, _, _ = self.koop(z)
        pred = self.fc(z_next)
        return (pred, z, z_next) if return_latents else pred

class Koopformer_PatchTST_Learnable(nn.Module):
    def __init__(self, input_dim, seq_len, horizon, patch_len, d_model=64, ρ_max: float = 0.99, koop_kind: str = "scalar"):
        super().__init__()
        self.backbone = PatchTST_Backbone(input_dim, seq_len, patch_len=patch_len, d_model=d_model)
        self.koop = make_learnable_koopman(koop_kind, d_model, ρ_max=ρ_max)
        self.fc   = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z = self.backbone(x)
        z_next, _, _ = self.koop(z)
        pred = self.fc(z_next)
        return (pred, z, z_next) if return_latents else pred

class Koopformer_PatchTST_Unconstrained(nn.Module):
    def __init__(self, input_dim, seq_len, horizon, patch_len, d_model=64):
        super().__init__()
        self.backbone = PatchTST_Backbone(input_dim, seq_len, patch_len=patch_len, d_model=d_model)
        self.koop = UnconstrainedKoopmanOperator(d_model)
        self.fc   = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z = self.backbone(x)
        z_next, _, _ = self.koop(z)
        pred = self.fc(z_next)
        return (pred, z, z_next) if return_latents else pred

class NeuralODE_PatchTST(nn.Module):
    def __init__(self, input_dim, seq_len, horizon, patch_len, d_model=64,
                 ode_hidden=128, ode_steps=8, ode_use_time=True):
        super().__init__()
        self.backbone = PatchTST_Backbone(input_dim, seq_len, patch_len=patch_len, d_model=d_model)
        self.ode  = NeuralODELatentPropagator(d_model, hidden_dim=ode_hidden, n_steps=ode_steps, use_time=ode_use_time)
        self.fc   = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z = self.backbone(x)
        z_next, _, _ = self.ode(z)
        pred = self.fc(z_next)
        return (pred, z, z_next) if return_latents else pred


class LNN_PatchTST(nn.Module):
    def __init__(self, input_dim, seq_len, horizon, patch_len, d_model=64,
                 lnn_hidden=128, lnn_steps=16, lnn_use_time=True,
                 tau_min=0.05, tau_max=5.0, lnn_tau_mode: str = "state_time"):
        super().__init__()
        self.backbone = PatchTST_Backbone(input_dim, seq_len, patch_len=patch_len, d_model=d_model)
        self.lnn = LiquidLatentPropagator(
            d_model, hidden_dim=lnn_hidden, n_steps=lnn_steps, use_time=lnn_use_time,
            tau_min=tau_min, tau_max=tau_max, tau_mode=lnn_tau_mode
        )
        self.fc = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z = self.backbone(x)
        z_next, _, _ = self.lnn(z)
        pred = self.fc(z_next)
        return (pred, z, z_next) if return_latents else pred





# -------------------- Autoformer family -------------------- #
class SeriesDecomp(nn.Module):
    def __init__(self, k: int = 3):
        super().__init__()
        self.avg = nn.AvgPool1d(k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor):
        trend = self.avg(x.transpose(1, 2)).transpose(1, 2)
        if trend.size(1) != x.size(1):
            trend = trend[:, : x.size(1)]
        return trend, x - trend

class SimpleAutoformer(nn.Module):
    def __init__(self, input_len, horizon, input_dim, patch_len,
                 d_model=64, num_heads=4, dim_ff=64, num_layers=3):
        super().__init__()
        self.dec   = SeriesDecomp(k=patch_len)
        self.embed = nn.Linear(input_dim, d_model)
        self.pos   = SinCosPosEnc(d_model, max_len=input_len)
        enc = nn.TransformerEncoderLayer(d_model, num_heads, dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        self.fc_seas  = nn.Linear(d_model, horizon)
        self.fc_trend = nn.Linear(input_len * input_dim, horizon)

    def forward(self, x: torch.Tensor):
        trend, seas = self.dec(x)
        seas = self.encoder(self.pos(self.embed(seas)))
        seas_o = self.fc_seas(seas.mean(1))
        trend_o = self.fc_trend(trend.reshape(trend.size(0), -1))
        return seas_o + trend_o  # (B, H)

class Koopformer_Autoformer_Constrained(nn.Module):
    def __init__(self, input_dim, seq_len, horizon, patch_len, d_model=16, ρ_max: float = 0.99):
        super().__init__()
        self.backbone = SimpleAutoformer(seq_len, horizon, input_dim, patch_len=patch_len, d_model=d_model)
        self.koop = StrictStableKoopmanOperator(horizon, ρ_max=ρ_max)
        self.fc   = nn.Linear(horizon, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z = self.backbone(x)         # (B, H)
        z_next, _, _ = self.koop(z)  # (B, H)
        pred = self.fc(z_next)
        return (pred, z, z_next) if return_latents else pred

class Koopformer_Autoformer_Learnable(nn.Module):
    def __init__(self, input_dim, seq_len, horizon, patch_len, d_model=16, ρ_max: float = 0.99, koop_kind: str = "scalar"):
        super().__init__()
        self.backbone = SimpleAutoformer(seq_len, horizon, input_dim, patch_len=patch_len, d_model=d_model)
        self.koop = make_learnable_koopman(koop_kind, horizon, ρ_max=ρ_max)
        self.fc   = nn.Linear(horizon, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z = self.backbone(x)
        z_next, _, _ = self.koop(z)
        pred = self.fc(z_next)
        return (pred, z, z_next) if return_latents else pred

class Koopformer_Autoformer_Unconstrained(nn.Module):
    def __init__(self, input_dim, seq_len, horizon, patch_len, d_model=16):
        super().__init__()
        self.backbone = SimpleAutoformer(seq_len, horizon, input_dim, patch_len=patch_len, d_model=d_model)
        self.koop = UnconstrainedKoopmanOperator(horizon)
        self.fc   = nn.Linear(horizon, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z = self.backbone(x)
        z_next, _, _ = self.koop(z)
        pred = self.fc(z_next)
        return (pred, z, z_next) if return_latents else pred

class NeuralODE_Autoformer(nn.Module):
    """Autoformer backbone output z∈R^H; propagate in R^H with NeuralODE; then map to F*H."""
    def __init__(self, input_dim, seq_len, horizon, patch_len, d_model=16,
                 ode_hidden=128, ode_steps=8, ode_use_time=True):
        super().__init__()
        self.backbone = SimpleAutoformer(seq_len, horizon, input_dim, patch_len=patch_len, d_model=d_model)
        self.ode = NeuralODELatentPropagator(horizon, hidden_dim=ode_hidden, n_steps=ode_steps, use_time=ode_use_time)
        self.fc  = nn.Linear(horizon, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z = self.backbone(x)          # (B, H)
        z_next, _, _ = self.ode(z)    # (B, H)
        pred = self.fc(z_next)
        return (pred, z, z_next) if return_latents else pred

class LNN_Autoformer(nn.Module):
    """Autoformer backbone output z∈R^H; propagate in R^H with LNN; then map to F*H."""
    def __init__(self, input_dim, seq_len, horizon, patch_len, d_model=16,
                 lnn_hidden=128, lnn_steps=16, lnn_use_time=True,
                 tau_min=0.05, tau_max=5.0, lnn_tau_mode: str = "state_time"):
        super().__init__()
        self.backbone = SimpleAutoformer(seq_len, horizon, input_dim, patch_len=patch_len, d_model=d_model)
        self.lnn = LiquidLatentPropagator(
            horizon, hidden_dim=lnn_hidden, n_steps=lnn_steps, use_time=lnn_use_time,
            tau_min=tau_min, tau_max=tau_max, tau_mode=lnn_tau_mode
        )
        self.fc = nn.Linear(horizon, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z = self.backbone(x)
        z_next, _, _ = self.lnn(z)
        pred = self.fc(z_next)
        return (pred, z, z_next) if return_latents else pred



# -------------------- Informer family -------------------- #
class InformerSparse(nn.Module):
    def __init__(self, input_dim, d_model=64, num_heads=4, dim_ff=96, num_layers=3,
                 seq_len=120, patch_len=1):
        super().__init__()
        self.use_patch = patch_len > 1
        if self.use_patch:
            self.patch = PatchEmbed1D(input_dim, d_model, patch_len, patch_len)
            n_tokens = int(np.ceil(seq_len / patch_len))
        else:
            self.embed = nn.Linear(input_dim, d_model)
            n_tokens = seq_len
        self.pos  = SinCosPosEnc(d_model, max_len=n_tokens)
        enc = nn.TransformerEncoderLayer(d_model, num_heads, dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x) if self.use_patch else self.embed(x)
        x = self.encoder(self.pos(x))
        return x.mean(dim=1)

class Koopformer_Informer_Constrained(nn.Module):
    def __init__(self, input_dim, seq_len, horizon, patch_len, d_model=64, ρ_max: float = 0.99):
        super().__init__()
        self.backbone = InformerSparse(input_dim, d_model=d_model, seq_len=seq_len, patch_len=patch_len)
        self.koop = StrictStableKoopmanOperator(d_model, ρ_max=ρ_max)
        self.fc   = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z = self.backbone(x)
        z_next, _, _ = self.koop(z)
        pred = self.fc(z_next)
        return (pred, z, z_next) if return_latents else pred

class Koopformer_Informer_Learnable(nn.Module):
    def __init__(self, input_dim, seq_len, horizon, patch_len, d_model=64, ρ_max: float = 0.99, koop_kind: str = "scalar"):
        super().__init__()
        self.backbone = InformerSparse(input_dim, d_model=d_model, seq_len=seq_len, patch_len=patch_len)
        self.koop = make_learnable_koopman(koop_kind, d_model, ρ_max=ρ_max)
        self.fc   = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z = self.backbone(x)
        z_next, _, _ = self.koop(z)
        pred = self.fc(z_next)
        return (pred, z, z_next) if return_latents else pred

class Koopformer_Informer_Unconstrained(nn.Module):
    def __init__(self, input_dim, seq_len, horizon, patch_len, d_model=64):
        super().__init__()
        self.backbone = InformerSparse(input_dim, d_model=d_model, seq_len=seq_len, patch_len=patch_len)
        self.koop = UnconstrainedKoopmanOperator(d_model)
        self.fc   = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z = self.backbone(x)
        z_next, _, _ = self.koop(z)
        pred = self.fc(z_next)
        return (pred, z, z_next) if return_latents else pred

class NeuralODE_Informer(nn.Module):
    def __init__(self, input_dim, seq_len, horizon, patch_len, d_model=64,
                 ode_hidden=128, ode_steps=8, ode_use_time=True):
        super().__init__()
        self.backbone = InformerSparse(input_dim, d_model=d_model, seq_len=seq_len, patch_len=patch_len)
        self.ode = NeuralODELatentPropagator(d_model, hidden_dim=ode_hidden, n_steps=ode_steps, use_time=ode_use_time)
        self.fc  = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z = self.backbone(x)
        z_next, _, _ = self.ode(z)
        pred = self.fc(z_next)
        return (pred, z, z_next) if return_latents else pred

class LNN_Informer(nn.Module):
    def __init__(self, input_dim, seq_len, horizon, patch_len, d_model=64,
                 lnn_hidden=128, lnn_steps=16, lnn_use_time=True,
                 tau_min=0.05, tau_max=5.0, lnn_tau_mode: str = "state_time"):
        super().__init__()
        self.backbone = InformerSparse(input_dim, d_model=d_model, seq_len=seq_len, patch_len=patch_len)
        self.lnn = LiquidLatentPropagator(
            d_model, hidden_dim=lnn_hidden, n_steps=lnn_steps, use_time=lnn_use_time,
            tau_min=tau_min, tau_max=tau_max, tau_mode=lnn_tau_mode
        )
        self.fc = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z = self.backbone(x)
        z_next, _, _ = self.lnn(z)
        pred = self.fc(z_next)
        return (pred, z, z_next) if return_latents else pred



# --------------------------------------------------------------------------- #
# 6)  Baselines: LSTM, DLinear, SSM                                           #
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
        return (out, z, z) if return_latents else out

class DLinearForecaster(nn.Module):
    def __init__(self, input_dim, seq_len, horizon):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Conv1d(1, 1, kernel_size=seq_len, bias=True) for _ in range(input_dim)]
        )
        self.horizon = horizon
        self.seq_len = seq_len
        self.input_dim = input_dim

    def forward(self, x, return_latents=False):
        outs = []
        for i, linear in enumerate(self.linears):
            xi = x[:, :, i].unsqueeze(1)
            if xi.shape[-1] < self.seq_len:
                pad = self.seq_len - xi.shape[-1]
                xi = torch.nn.functional.pad(xi, (pad, 0))
            yi = linear(xi)
            if yi.shape[-1] == 1 and self.horizon > 1:
                yi = yi.expand(-1, -1, self.horizon)
            outs.append(yi)
        out = torch.cat(outs, dim=1)
        out = out.permute(0, 2, 1).reshape(x.size(0), -1)
        return (out, out, out) if return_latents else out

class SimpleSSMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, horizon: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.01)
        self.C = nn.Linear(hidden_dim, input_dim * horizon)

    def forward(self, x, return_latents: bool = False):
        Bsz, T, F = x.shape
        h = torch.zeros(Bsz, self.hidden_dim, device=x.device)
        for t in range(T):
            x_t = x[:, t, :]
            h = h @ self.A.T + x_t @ self.B.T
        z = h
        out = self.C(z)
        return (out, z, z) if return_latents else out

# --------------------------------------------------------------------------- #
# 7)  Dataset util                                                            #
# --------------------------------------------------------------------------- #
def build_dataset_fit_scaler(data: np.ndarray, indices: list[int], seq_len: int, horizon: int):
    sub = data[:, indices]
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

def build_dataset_with_scaler(data: np.ndarray, indices: list[int], seq_len: int, horizon: int, scaler: MinMaxScaler):
    sub = data[:, indices]
    sub = scaler.transform(sub)
    X, Y = [], []
    for i in range(len(sub) - seq_len - horizon):
        X.append(sub[i: i + seq_len])
        y = sub[i + seq_len: i + seq_len + horizon].T.reshape(-1)
        Y.append(y)
    x_in  = torch.tensor(np.stack(X), dtype=torch.float32)
    x_out = torch.tensor(np.stack(Y), dtype=torch.float32)
    return x_in, x_out

# --------------------------------------------------------------------------- #
# 8)  Training helper & spectrum logging                                      #
# --------------------------------------------------------------------------- #
def _nested_getattr(obj, path: str):
    for p in path.split('.'):
        obj = getattr(obj, p, None)
        if obj is None:
            return None
    return obj

def get_koopman_spectrum(model: nn.Module, koop_attr: str = "koop"):
    obj = _nested_getattr(model, koop_attr) if koop_attr else None
    if isinstance(obj, StrictStableKoopmanOperator):
        with torch.no_grad():
            return obj._sigma().detach().cpu().numpy()
    if isinstance(obj, LearnableKoopmanBase):
        with torch.no_grad():
            return obj._sigma().detach().cpu().numpy()
    if isinstance(obj, UnconstrainedKoopmanOperator):
        with torch.no_grad():
            vals = torch.linalg.eigvals(obj.K_raw)
            return vals.detach().cpu().numpy()
    return None

def train(model, x_in, x_out, epochs=4000, lr=3e-4, koop_attr="koop", lyap_weight=0.1):
    model.to(DEV)
    x_in, x_out = x_in.to(DEV), x_out.to(DEV)
    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    losses, eigs = [], []

    with torch.no_grad():
        pred0, z0, z1 = model(x_in[:1], return_latents=True)
    P = torch.eye(z0.shape[1], device=DEV)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()

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

        # spectral proxy only for Koopman-ish layers
        obj = _nested_getattr(model, koop_attr) if koop_attr else None
        if isinstance(obj, StrictStableKoopmanOperator):
            with torch.no_grad():
                eigs.append(float(obj._sigma().abs().max().item()))
        elif isinstance(obj, LearnableKoopmanBase):
            with torch.no_grad():
                eigs.append(float(obj._sigma().abs().max().item()))
        elif isinstance(obj, UnconstrainedKoopmanOperator):
            with torch.no_grad():
                try:
                    vals = torch.linalg.eigvals(obj.K_raw)
                    eigs.append(float(vals.abs().max().item()))
                except RuntimeError:
                    eigs.append(0.0)
        else:
            eigs.append(0.0)

        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            print(f"Epoch {ep:>4}/{epochs}  MSE {pred_loss.item():.6e}  Lyap {(lyap_weight*lyap).item():.6e}")

    model.eval()
    with torch.no_grad():
        pred_train = model(x_in).detach().cpu().numpy()
    return pred_train, np.array(losses), np.array(eigs)

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

def plot_training(res_loss, path=None):
    if not res_loss:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for i, (n, r) in enumerate(res_loss.items()):
        ax1.plot(r["loss"], label=n, color=_COLORS[i % 10], linestyle=_STYLES[i % 4], lw=3)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_title("Loss"); ax1.grid(True, ls="--", alpha=.6); ax1.legend()

    k = 0
    for n, r in res_loss.items():
        if np.allclose(r["eig"], 0):
            continue
        ax2.plot(r["eig"], label=n, color=_COLORS[k % 10], linestyle=_STYLES[k % 4], lw=3)
        k += 1
    ax2.set_title("Spectral proxy (max)"); ax2.grid(True, ls="--", alpha=.6); ax2.legend()
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
            ax.plot(t, p[:, f], label=n, color=_COLORS[i % len(_COLORS)],
                    linestyle=_STYLES[i % len(_STYLES)], linewidth=3)
        ax.set_ylabel("Value"); ax.grid(True, which="both", linestyle="--", alpha=0.6)
        if f == 0:
            ax.legend(ncol=3)
    axes[-1].set_xlabel("Sample")
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)

# --------------------------------------------------------------------------- #
# 10)  Saving                                                                 #
# --------------------------------------------------------------------------- #
def save_results(pred: np.ndarray, x_out: np.ndarray,
                 model_name: str, prefix: Path, metrics_file: Path,
                 patch_len: int, horizon: int, set_name: str):
    err = x_out - pred
    set_tag = set_name.lower()
    np.save(f"{prefix}_{set_tag}_predictions.npy", pred)
    np.save(f"{prefix}_{set_tag}_errors.npy", err)

    mse = float((err ** 2).mean())
    mae = float(np.abs(err).mean())
    pd.DataFrame({
        "Model": [model_name],
        "PatchLen": [patch_len],
        "Horizon": [horizon],
        "Set": [set_name],
        "MSE": [mse],
        "MAE": [mae],
    }).to_csv(metrics_file, mode="a", header=False, index=False)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 11)  Reviewer-ready benchmark loop                                          #
# --------------------------------------------------------------------------- #
# This replacement main loop directly addresses the TNNLS reviewer concerns:
#   (R1/R2) repeated runs + mean/std tables, not only KDE plots
#   (R2-10) Friedman test over model ranks
#   (R2-12) extra metrics: RMSE, MAPE, SMAPE, WAPE, R2
#   (R1/R2-14) fair setup: same seq_len/horizon/training settings for all models
#   (R1) PatchTST fairness: patch_len is no longer forced to equal seq_len
#   (R2-15/16) Koopman spectral logs + perturbation stability evaluation
#   (R2-13) full reproducibility metadata saved as JSON

import json
import time
import math
import random
from typing import Dict, List, Tuple, Optional

try:
    from scipy.stats import friedmanchisquare, rankdata, norm, wilcoxon
except Exception:  # keeps the code runnable on minimal HPC nodes
    friedmanchisquare = None
    rankdata = None
    norm = None
    wilcoxon = None


def sanitize_name(name: str) -> str:
    return (name.replace(" ", "_").replace(":", "-").replace("(", "")
                .replace(")", "").replace("/", "_").replace(".", ""))


def set_seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def make_windows_from_scaled(sub_scaled: np.ndarray, seq_len: int, horizon: int):
    X, Y = [], []
    for i in range(len(sub_scaled) - seq_len - horizon + 1):
        X.append(sub_scaled[i: i + seq_len])
        y = sub_scaled[i + seq_len: i + seq_len + horizon].T.reshape(-1)
        Y.append(y)
    if not X:
        raise ValueError(
            f"Not enough samples: got len={len(sub_scaled)}, seq_len={seq_len}, horizon={horizon}."
        )
    return (torch.tensor(np.stack(X), dtype=torch.float32),
            torch.tensor(np.stack(Y), dtype=torch.float32))


def build_dataset_fit_scaler_v2(data: np.ndarray, indices: list[int], seq_len: int, horizon: int):
    sub = data[:, indices]
    scaler = MinMaxScaler().fit(sub)
    sub_scaled = scaler.transform(sub)
    x_in, x_out = make_windows_from_scaled(sub_scaled, seq_len, horizon)
    return x_in, x_out, scaler


def build_dataset_with_scaler_v2(data: np.ndarray, indices: list[int], seq_len: int, horizon: int, scaler: MinMaxScaler):
    sub = data[:, indices]
    sub_scaled = scaler.transform(sub)
    return make_windows_from_scaled(sub_scaled, seq_len, horizon)


# IMPORTANT: We intentionally do NOT inverse-transform predictions or targets for evaluation.
# All reported metrics are computed in the MinMax-scaled target space.

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    err = y_true - y_pred
    abs_err = np.abs(err)
    sq_err = err ** 2
    denom = np.maximum(np.abs(y_true), eps)
    denom_s = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    ss_res = float(np.sum(sq_err))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "MSE": float(np.mean(sq_err)),
        "RMSE": float(np.sqrt(np.mean(sq_err))),
        "MAE": float(np.mean(abs_err)),
        "MAPE": float(100.0 * np.mean(abs_err / denom)),
        "SMAPE": float(100.0 * np.mean(2.0 * abs_err / denom_s)),
        "WAPE": float(100.0 * np.sum(abs_err) / max(float(np.sum(np.abs(y_true))), eps)),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > eps else float("nan"),
    }


def per_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray, F: int, H: int) -> List[Dict[str, float]]:
    yt = y_true.reshape(-1, F, H)
    yp = y_pred.reshape(-1, F, H)
    rows = []
    for h in range(H):
        m = regression_metrics(yt[:, :, h].reshape(len(yt), F), yp[:, :, h].reshape(len(yp), F))
        m["HorizonStep"] = h + 1
        rows.append(m)
    return rows


def perturbation_stability(model: nn.Module, x: torch.Tensor, noise_std: float = 1e-3,
                           batches: int = 4, batch_size: int = 256) -> Dict[str, float]:
    """Empirical Lipschitz-style sensitivity: ||f(x+δ)-f(x)|| / ||δ||."""
    model.eval().to(DEV)
    ratios = []
    n = min(x.shape[0], batches * batch_size)
    if n == 0:
        return {"PerturbRatioMean": float("nan"), "PerturbRatioMax": float("nan")}
    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = x[start:start + batch_size].to(DEV)
            delta = noise_std * torch.randn_like(xb)
            y0 = model(xb)
            y1 = model(xb + delta)
            num = torch.norm((y1 - y0).reshape(y0.shape[0], -1), dim=1)
            den = torch.norm(delta.reshape(delta.shape[0], -1), dim=1).clamp_min(1e-12)
            ratios.extend((num / den).detach().cpu().numpy().tolist())
    return {"PerturbRatioMean": float(np.mean(ratios)), "PerturbRatioMax": float(np.max(ratios))}


def train_reviewer_ready(model, x_in, x_out, epochs=4000, lr=3e-4, weight_decay=0.0,
                         koop_attr="koop", lyap_weight=0.1, batch_size: int = 0,
                         patience: int = 0, scheduler: str = "cosine"):
    model.to(DEV)
    x_in, x_out = x_in.to(DEV), x_out.to(DEV)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler == "cosine":
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 0.05)
    elif scheduler == "plateau":
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=max(5, patience // 4))
    else:
        sch = None
    mse = nn.MSELoss()
    losses, pred_losses, lyap_losses, eigs, lrs = [], [], [], [], []

    with torch.no_grad():
        _, z0, _ = model(x_in[:1], return_latents=True)
    P = torch.eye(z0.shape[1], device=DEV)

    best_state, best_loss, bad = None, float("inf"), 0
    n = x_in.shape[0]
    use_minibatch = batch_size and batch_size > 0 and batch_size < n

    for ep in range(epochs):
        model.train()
        if use_minibatch:
            perm = torch.randperm(n, device=DEV)
            epoch_losses, epoch_pred, epoch_lyap = [], [], []
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                xb, yb = x_in[idx], x_out[idx]
                opt.zero_grad()
                pred, z, z_next = model(xb, return_latents=True)
                pred_loss = mse(pred, yb)
                if lyap_weight:
                    zp = torch.einsum("bi,ij->bj", z, P)
                    znp = torch.einsum("bi,ij->bj", z_next, P)
                    lyap = torch.relu((znp * z_next).sum(1) - (zp * z).sum(1)).mean()
                else:
                    lyap = torch.zeros((), device=DEV)
                loss = pred_loss + lyap_weight * lyap
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                epoch_losses.append(float(loss.item()))
                epoch_pred.append(float(pred_loss.item()))
                epoch_lyap.append(float((lyap_weight * lyap).item()))
            loss_value = float(np.mean(epoch_losses))
            pred_value = float(np.mean(epoch_pred))
            lyap_value = float(np.mean(epoch_lyap))
        else:
            opt.zero_grad()
            pred, z, z_next = model(x_in, return_latents=True)
            pred_loss = mse(pred, x_out)
            if lyap_weight:
                zp = torch.einsum("bi,ij->bj", z, P)
                znp = torch.einsum("bi,ij->bj", z_next, P)
                lyap = torch.relu((znp * z_next).sum(1) - (zp * z).sum(1)).mean()
            else:
                lyap = torch.zeros((), device=DEV)
            loss = pred_loss + lyap_weight * lyap
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_value = float(loss.item())
            pred_value = float(pred_loss.item())
            lyap_value = float((lyap_weight * lyap).item())

        if sch is not None:
            if scheduler == "plateau":
                sch.step(loss_value)
            else:
                sch.step()

        losses.append(loss_value)
        pred_losses.append(pred_value)
        lyap_losses.append(lyap_value)
        lrs.append(float(opt.param_groups[0]["lr"]))

        obj = _nested_getattr(model, koop_attr) if koop_attr else None
        if isinstance(obj, (StrictStableKoopmanOperator, LearnableKoopmanBase)):
            with torch.no_grad():
                eigs.append(float(obj._sigma().abs().max().item()))
        elif isinstance(obj, UnconstrainedKoopmanOperator):
            with torch.no_grad():
                try:
                    eigs.append(float(torch.linalg.eigvals(obj.K_raw).abs().max().item()))
                except RuntimeError:
                    eigs.append(float("nan"))
        else:
            eigs.append(0.0)

        if patience and loss_value < best_loss - 1e-10:
            best_loss = loss_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        elif patience:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {ep}; best loss={best_loss:.6e}")
                break

        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            print(f"Epoch {ep:>4}/{epochs}  MSE {pred_value:.6e}  Lyap {lyap_value:.6e}  lr {lrs[-1]:.2e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_train = model(x_in).detach().cpu().numpy()
    history = {
        "loss": np.array(losses),
        "pred_loss": np.array(pred_losses),
        "lyap_loss": np.array(lyap_losses),
        "eig": np.array(eigs),
        "lr": np.array(lrs),
    }
    return pred_train, history


def make_model_variants(F: int, seq_len: int, horizon: int, patch_len: int,
                        d_model: int, rho_max: float, learnable_kinds: List[str],
                        lyap_weight_constr: float, lyap_weight_learn: float,
                        lyap_weight_unconstr: float, lyap_weight_ode: float,
                        lyap_weight_lnn: float, ode_hidden: int, ode_steps: int,
                        ode_use_time: bool, lnn_hidden: int, lnn_steps: int,
                        lnn_use_time: bool, lnn_tau_min: float, lnn_tau_max: float,
                        include_lnn_ablations: bool = True):
    baseline_variants = [
        ("LSTM", SimpleLSTMForecaster(F, hidden_dim=d_model, num_layers=2, horizon=horizon), None, 0.0),
        ("DLinear", DLinearForecaster(F, seq_len=seq_len, horizon=horizon), None, 0.0),
    ]

    patchtst_variants = [
        ("Koop-PatchTST (constr.)", Koopformer_PatchTST_Constrained(F, seq_len, horizon, patch_len=patch_len, d_model=d_model, ρ_max=rho_max), "koop", lyap_weight_constr),
        ("Koop-PatchTST (unconstr.)", Koopformer_PatchTST_Unconstrained(F, seq_len, horizon, patch_len=patch_len, d_model=d_model), "koop", lyap_weight_unconstr),
        ("ODE-PatchTST", NeuralODE_PatchTST(F, seq_len, horizon, patch_len=patch_len, d_model=d_model, ode_hidden=ode_hidden, ode_steps=ode_steps, ode_use_time=ode_use_time), None, lyap_weight_ode),
        ("LNN-PatchTST", LNN_PatchTST(F, seq_len, horizon, patch_len=patch_len, d_model=d_model, lnn_hidden=lnn_hidden, lnn_steps=lnn_steps, lnn_use_time=lnn_use_time, tau_min=lnn_tau_min, tau_max=lnn_tau_max, lnn_tau_mode="state_time"), None, lyap_weight_lnn),
    ]
    for kind in learnable_kinds:
        patchtst_variants.append((f"Koop-PatchTST (learn:{kind})", Koopformer_PatchTST_Learnable(F, seq_len, horizon, patch_len=patch_len, d_model=d_model, ρ_max=rho_max, koop_kind=kind), "koop", lyap_weight_learn))

    autoformer_variants = [
        ("Koop-Autoformer (constr.)", Koopformer_Autoformer_Constrained(F, seq_len, horizon, patch_len=patch_len, d_model=d_model, ρ_max=rho_max), "koop", lyap_weight_constr),
        ("Koop-Autoformer (unconstr.)", Koopformer_Autoformer_Unconstrained(F, seq_len, horizon, patch_len=patch_len, d_model=d_model), "koop", lyap_weight_unconstr),
        ("ODE-Autoformer", NeuralODE_Autoformer(F, seq_len, horizon, patch_len=patch_len, d_model=d_model, ode_hidden=ode_hidden, ode_steps=ode_steps, ode_use_time=ode_use_time), None, lyap_weight_ode),
        ("LNN-Autoformer", LNN_Autoformer(F, seq_len, horizon, patch_len=patch_len, d_model=d_model, lnn_hidden=lnn_hidden, lnn_steps=lnn_steps, lnn_use_time=lnn_use_time, tau_min=lnn_tau_min, tau_max=lnn_tau_max, lnn_tau_mode="state_time"), None, lyap_weight_lnn),
    ]
    for kind in learnable_kinds:
        autoformer_variants.append((f"Koop-Autoformer (learn:{kind})", Koopformer_Autoformer_Learnable(F, seq_len, horizon, patch_len=patch_len, d_model=d_model, ρ_max=rho_max, koop_kind=kind), "koop", lyap_weight_learn))

    informer_variants = [
        ("Koop-Informer (constr.)", Koopformer_Informer_Constrained(F, seq_len, horizon, patch_len=patch_len, d_model=d_model, ρ_max=rho_max), "koop", lyap_weight_constr),
        ("Koop-Informer (unconstr.)", Koopformer_Informer_Unconstrained(F, seq_len, horizon, patch_len=patch_len, d_model=d_model), "koop", lyap_weight_unconstr),
        ("ODE-Informer", NeuralODE_Informer(F, seq_len, horizon, patch_len=patch_len, d_model=d_model, ode_hidden=ode_hidden, ode_steps=ode_steps, ode_use_time=ode_use_time), None, lyap_weight_ode),
        ("LNN-Informer", LNN_Informer(F, seq_len, horizon, patch_len=patch_len, d_model=d_model, lnn_hidden=lnn_hidden, lnn_steps=lnn_steps, lnn_use_time=lnn_use_time, tau_min=lnn_tau_min, tau_max=lnn_tau_max, lnn_tau_mode="state_time"), None, lyap_weight_lnn),
    ]
    for kind in learnable_kinds:
        informer_variants.append((f"Koop-Informer (learn:{kind})", Koopformer_Informer_Learnable(F, seq_len, horizon, patch_len=patch_len, d_model=d_model, ρ_max=rho_max, koop_kind=kind), "koop", lyap_weight_learn))

    if include_lnn_ablations:
        # Reviewer-requested explicit liquid dynamics ablations.
        # 1) No Lyapunov regularization: same LNN, lyap_weight=0.
        # 2) Constant tau: tau is a learned vector independent of z and t.
        # 3) No time input: tau depends on z only, not the integration time.
        lnn_ablation_variants = [
            ("LNN-PatchTST (no Lyap)", LNN_PatchTST(F, seq_len, horizon, patch_len=patch_len, d_model=d_model, lnn_hidden=lnn_hidden, lnn_steps=lnn_steps, lnn_use_time=lnn_use_time, tau_min=lnn_tau_min, tau_max=lnn_tau_max, lnn_tau_mode="state_time"), None, 0.0),
            ("LNN-PatchTST (constant tau)", LNN_PatchTST(F, seq_len, horizon, patch_len=patch_len, d_model=d_model, lnn_hidden=lnn_hidden, lnn_steps=lnn_steps, lnn_use_time=lnn_use_time, tau_min=lnn_tau_min, tau_max=lnn_tau_max, lnn_tau_mode="constant"), None, lyap_weight_lnn),
            ("LNN-PatchTST (no time input)", LNN_PatchTST(F, seq_len, horizon, patch_len=patch_len, d_model=d_model, lnn_hidden=lnn_hidden, lnn_steps=lnn_steps, lnn_use_time=False, tau_min=lnn_tau_min, tau_max=lnn_tau_max, lnn_tau_mode="state_time"), None, lyap_weight_lnn),
            ("LNN-Autoformer (no Lyap)", LNN_Autoformer(F, seq_len, horizon, patch_len=patch_len, d_model=d_model, lnn_hidden=lnn_hidden, lnn_steps=lnn_steps, lnn_use_time=lnn_use_time, tau_min=lnn_tau_min, tau_max=lnn_tau_max, lnn_tau_mode="state_time"), None, 0.0),
            ("LNN-Autoformer (constant tau)", LNN_Autoformer(F, seq_len, horizon, patch_len=patch_len, d_model=d_model, lnn_hidden=lnn_hidden, lnn_steps=lnn_steps, lnn_use_time=lnn_use_time, tau_min=lnn_tau_min, tau_max=lnn_tau_max, lnn_tau_mode="constant"), None, lyap_weight_lnn),
            ("LNN-Autoformer (no time input)", LNN_Autoformer(F, seq_len, horizon, patch_len=patch_len, d_model=d_model, lnn_hidden=lnn_hidden, lnn_steps=lnn_steps, lnn_use_time=False, tau_min=lnn_tau_min, tau_max=lnn_tau_max, lnn_tau_mode="state_time"), None, lyap_weight_lnn),
            ("LNN-Informer (no Lyap)", LNN_Informer(F, seq_len, horizon, patch_len=patch_len, d_model=d_model, lnn_hidden=lnn_hidden, lnn_steps=lnn_steps, lnn_use_time=lnn_use_time, tau_min=lnn_tau_min, tau_max=lnn_tau_max, lnn_tau_mode="state_time"), None, 0.0),
            ("LNN-Informer (constant tau)", LNN_Informer(F, seq_len, horizon, patch_len=patch_len, d_model=d_model, lnn_hidden=lnn_hidden, lnn_steps=lnn_steps, lnn_use_time=lnn_use_time, tau_min=lnn_tau_min, tau_max=lnn_tau_max, lnn_tau_mode="constant"), None, lyap_weight_lnn),
            ("LNN-Informer (no time input)", LNN_Informer(F, seq_len, horizon, patch_len=patch_len, d_model=d_model, lnn_hidden=lnn_hidden, lnn_steps=lnn_steps, lnn_use_time=False, tau_min=lnn_tau_min, tau_max=lnn_tau_max, lnn_tau_mode="state_time"), None, lyap_weight_lnn),
        ]
    else:
        lnn_ablation_variants = []

    return baseline_variants + patchtst_variants + autoformer_variants + informer_variants + lnn_ablation_variants



def nemenyi_posthoc_from_ranks(avg_ranks: np.ndarray, model_names: List[str], num_blocks: int,
                               alpha: float = 0.05) -> pd.DataFrame:
    """
    Pairwise Nemenyi post-hoc comparison after Friedman.
    Uses the normal approximation to the Studentized range distribution:
        z = |R_i - R_j| / sqrt(k(k+1)/(6N))
        p ~= 2 * (1 - Phi(z / sqrt(2)))
    This avoids adding a hard dependency on scikit-posthocs/statsmodels.
    """
    rows = []
    k = len(model_names)
    se = math.sqrt(k * (k + 1) / (6.0 * max(num_blocks, 1)))
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(float(avg_ranks[i]) - float(avg_ranks[j]))
            z = diff / se if se > 0 else float("nan")
            if norm is not None and np.isfinite(z):
                p_approx = float(2.0 * (1.0 - norm.cdf(z / math.sqrt(2.0))))
                p_approx = min(max(p_approx, 0.0), 1.0)
            else:
                p_approx = float("nan")
            rows.append({
                "PostHoc": "Nemenyi_normal_approx",
                "ModelA": model_names[i],
                "ModelB": model_names[j],
                "AverageRankA": float(avg_ranks[i]),
                "AverageRankB": float(avg_ranks[j]),
                "RankDiff": diff,
                "SE": se,
                "z": z,
                "p_value": p_approx,
                "alpha": alpha,
                "significant": bool(p_approx < alpha) if np.isfinite(p_approx) else False,
            })
    return pd.DataFrame(rows)


def wilcoxon_holm_posthoc(arr: np.ndarray, model_names: List[str], alpha: float = 0.05) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests with Holm correction as a robust fallback/supplement."""
    rows = []
    k = len(model_names)
    for i in range(k):
        for j in range(i + 1, k):
            try:
                if wilcoxon is None:
                    raise RuntimeError("scipy.stats.wilcoxon unavailable")
                stat, p = wilcoxon(arr[:, i], arr[:, j], zero_method="wilcox", alternative="two-sided")
                stat, p = float(stat), float(p)
            except Exception:
                stat, p = float("nan"), float("nan")
            rows.append({
                "PostHoc": "Wilcoxon-Holm",
                "ModelA": model_names[i],
                "ModelB": model_names[j],
                "Statistic": stat,
                "raw_p_value": p,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    valid = df["raw_p_value"].notna()
    m = int(valid.sum())
    df["holm_adjusted_p_value"] = np.nan
    df["holm_significant"] = False
    if m > 0:
        ordered = df[valid].sort_values("raw_p_value").index.tolist()
        adjusted_vals = []
        running_max = 0.0
        for rank_idx, idx in enumerate(ordered):
            raw_p = float(df.loc[idx, "raw_p_value"])
            adj = min(1.0, (m - rank_idx) * raw_p)
            running_max = max(running_max, adj)
            adjusted_vals.append((idx, running_max))
        for idx, adj in adjusted_vals:
            df.loc[idx, "holm_adjusted_p_value"] = adj
            df.loc[idx, "holm_significant"] = bool(adj < alpha)
    df["alpha"] = alpha
    return df

def write_summary_tables(save_dir: Path):
    metrics_path = save_dir / "metrics_long.csv"
    if not metrics_path.exists():
        return
    df = pd.read_csv(metrics_path)
    test = df[df["Set"] == "Test"].copy()
    if test.empty:
        return

    group_cols = ["Model", "PatchLen", "Horizon"]
    metric_cols = ["MSE", "RMSE", "MAE", "MAPE", "SMAPE", "WAPE", "R2", "PerturbRatioMean", "SpectralRadius"]
    agg = test.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join([c for c in col if c]) for col in agg.columns.values]
    agg.to_csv(save_dir / "summary_mean_std_by_setting.csv", index=False)

    overall = test.groupby("Model")[metric_cols].agg(["mean", "std"]).reset_index()
    overall.columns = ["_".join([c for c in col if c]) for col in overall.columns.values]
    overall.sort_values("MSE_mean").to_csv(save_dir / "summary_mean_std_overall.csv", index=False)

    # Friedman test: each dataset/patch/horizon/seed block gives one error value per model.
    # Post-hoc comparisons are saved both as a Nemenyi table and a Wilcoxon-Holm table.
    blocks = []
    for key, g in test.groupby(["Dataset", "Seed", "PatchLen", "Horizon"]):
        pivot = g.pivot_table(index=None, columns="Model", values="MSE", aggfunc="mean")
        if pivot.shape[1] >= 3:
            blocks.append(pivot)
    if blocks:
        mat_df = pd.concat(blocks, ignore_index=True).dropna(axis=1)
        mat_df.to_csv(save_dir / "friedman_blocks_mse.csv", index=False)
        if mat_df.shape[0] >= 2 and mat_df.shape[1] >= 3:
            arr = mat_df.to_numpy(dtype=float)
            models = list(mat_df.columns)
            out = {"metric": "MSE", "num_blocks": int(arr.shape[0]), "num_models": int(arr.shape[1]), "models": models}
            if friedmanchisquare is not None:
                stat, pval = friedmanchisquare(*[arr[:, j] for j in range(arr.shape[1])])
                out.update({"friedman_statistic": float(stat), "p_value": float(pval)})
            else:
                out.update({"friedman_statistic": None, "p_value": None, "note": "scipy not installed"})
            if rankdata is not None:
                ranks = np.vstack([rankdata(row, method="average") for row in arr])  # lower MSE => smaller rank
                avg_ranks = ranks.mean(axis=0)
                rank_summary = pd.DataFrame({"Model": models, "AverageRank": avg_ranks})
                rank_summary.sort_values("AverageRank").to_csv(save_dir / "friedman_average_ranks.csv", index=False)
                nemenyi = nemenyi_posthoc_from_ranks(avg_ranks, models, num_blocks=arr.shape[0], alpha=0.05)
                wilcoxon_holm = wilcoxon_holm_posthoc(arr, models, alpha=0.05)
                nemenyi.to_csv(save_dir / "nemenyi_posthoc.csv", index=False)
                wilcoxon_holm.to_csv(save_dir / "wilcoxon_holm_posthoc.csv", index=False)
                # The exact requested filename: contains Nemenyi rows first, then Wilcoxon-Holm rows.
                combined = pd.concat([nemenyi, wilcoxon_holm], ignore_index=True, sort=False)
                combined.to_csv(save_dir / "nemenyi_or_wilcoxon_posthoc.csv", index=False)
            with open(save_dir / "friedman_test_mse.json", "w") as f:
                json.dump(out, f, indent=2)


def main(file="bitstamp_windows.npy", epochs=200, seq_len=120,
         patch_lens=None, horizons=None, indices=None, save_dir="results",
         train_frac: float = 0.8, seeds=None, d_model: int = 96, lr: float = 3e-4,
         weight_decay: float = 0.0, batch_size: int = 0, patience: int = 0,
         scheduler: str = "cosine", rho_max: float = 0.99,
         lyap_weight_constr: float = 0.1, lyap_weight_learn: float = 0.1,
         lyap_weight_unconstr: float = 0.1, lyap_weight_ode: float = 0.1,
         learnable_kinds=None, ode_steps: int = 8, ode_hidden: int = 128,
         ode_use_time: bool = True, lyap_weight_lnn: float = 0.1,
         lnn_steps: int = 16, lnn_hidden: int = 128, lnn_use_time: bool = True,
         lnn_tau_min: float = 0.05, lnn_tau_max: float = 5.0,
         dataset_name: str = "dataset", save_models: bool = False,
         perturb_noise_std: float = 1e-3, include_lnn_ablations: bool = True):

    patch_lens = patch_lens or [16, 24, 32, 40]
    horizons = horizons or [4, 8, 12, 16]
    indices = indices or [0, 1, 2, 3, 4, 5]
    seeds = seeds or [7, 42, 123, 2025, 2026]
    learnable_kinds = learnable_kinds or ["scalar", "permode", "mlp", "lowrank16"]

    save_dir = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(file)
    if raw.ndim != 2:
        raise ValueError(f"Expected a 2D array (time, features), got shape={raw.shape}")
    max_index = max(indices)
    if max_index >= raw.shape[1]:
        raise ValueError(f"indices={indices} exceeds number of features={raw.shape[1]}")

    N_raw = raw.shape[0]
    split = int(train_frac * N_raw)
    train_raw = raw[:split]
    test_raw = raw[split:]

    config = dict(
        file=file, dataset_name=dataset_name, raw_shape=list(raw.shape), train_frac=train_frac,
        seq_len=seq_len, patch_lens=patch_lens, horizons=horizons, indices=indices,
        seeds=seeds, epochs=epochs, d_model=d_model, lr=lr, weight_decay=weight_decay,
        batch_size=batch_size, patience=patience, scheduler=scheduler, rho_max=rho_max,
        lyap_weight_constr=lyap_weight_constr, lyap_weight_learn=lyap_weight_learn,
        lyap_weight_unconstr=lyap_weight_unconstr, lyap_weight_ode=lyap_weight_ode,
        lyap_weight_lnn=lyap_weight_lnn, learnable_kinds=learnable_kinds,
        ode_steps=ode_steps, ode_hidden=ode_hidden, ode_use_time=ode_use_time,
        lnn_steps=lnn_steps, lnn_hidden=lnn_hidden, lnn_use_time=lnn_use_time,
        lnn_tau_min=lnn_tau_min, lnn_tau_max=lnn_tau_max,
        perturb_noise_std=perturb_noise_std, include_lnn_ablations=include_lnn_ablations,
        evaluation_scale="MinMax-scaled target space only; no inverse transform",
        device=str(DEV), torch_version=torch.__version__,
    )
    with open(save_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    metrics_file = save_dir / "metrics_long.csv"
    per_horizon_file = save_dir / "metrics_per_horizon.csv"
    train_history_dir = save_dir / "train_histories"
    pred_dir = save_dir / "predictions"
    spec_dir = save_dir / "spectra"
    for d in [train_history_dir, pred_dir, spec_dir]:
        d.mkdir(exist_ok=True)

    metric_cols = ["Dataset", "Seed", "Model", "PatchLen", "SeqLen", "Horizon", "NumPatches", "Set",
                   "MSE", "RMSE", "MAE", "MAPE", "SMAPE", "WAPE", "R2",
                   "TrainSeconds", "FinalTrainLoss", "FinalPredLoss", "FinalLyapLoss",
                   "SpectralRadius", "PerturbRatioMean", "PerturbRatioMax"]
    pd.DataFrame(columns=metric_cols).to_csv(metrics_file, index=False)
    pd.DataFrame(columns=metric_cols + ["HorizonStep"]).to_csv(per_horizon_file, index=False)

    F = len(indices)
    for patch_len in patch_lens:
        if patch_len >= seq_len:
            print(f"WARNING: patch_len={patch_len} >= seq_len={seq_len}. This creates one/few tokens and may be unfair to PatchTST.")
        num_patches = max(1, 1 + (seq_len - patch_len) // patch_len) if seq_len >= patch_len else 1
        for horizon in horizons:
            print(f"\n>>> seq_len={seq_len}, patch_len={patch_len}, horizon={horizon}, approx_num_patches={num_patches}\n")
            if len(train_raw) < (seq_len + horizon + 2) or len(test_raw) < (seq_len + horizon + 2):
                print("Not enough raw data for train/test windows. Skipping.")
                continue

            x_train, y_train, scaler = build_dataset_fit_scaler_v2(train_raw, indices, seq_len, horizon)
            x_test, y_test = build_dataset_with_scaler_v2(test_raw, indices, seq_len, horizon, scaler)
            # Evaluation targets remain in the scaled space.
            # Do NOT inverse-transform: reviewer tables should report scaled metrics.
            y_train_np_scaled, y_test_np_scaled = y_train.numpy(), y_test.numpy()
            y_train_np, y_test_np = y_train_np_scaled, y_test_np_scaled

            # only for readable figures: keep last seed predictions for plotting
            last_seed_losses, last_seed_test_preds = {}, {}

            for seed in seeds:
                print(f"\n--- Seed {seed} ---")
                set_seed_all(seed)
                variants = make_model_variants(
                    F, seq_len, horizon, patch_len, d_model, rho_max, learnable_kinds,
                    lyap_weight_constr, lyap_weight_learn, lyap_weight_unconstr, lyap_weight_ode,
                    lyap_weight_lnn, ode_hidden, ode_steps, ode_use_time,
                    lnn_hidden, lnn_steps, lnn_use_time, lnn_tau_min, lnn_tau_max,
                    include_lnn_ablations=include_lnn_ablations
                )

                for name, model, koop_attr, lyap_w in variants:
                    print(f"== {name} ==")
                    t0 = time.time()
                    pred_train_scaled, hist = train_reviewer_ready(
                        model, x_train, y_train, epochs=epochs, lr=lr, weight_decay=weight_decay,
                        koop_attr=koop_attr, lyap_weight=lyap_w, batch_size=batch_size,
                        patience=patience, scheduler=scheduler
                    )
                    train_seconds = time.time() - t0

                    model.to(DEV).eval()
                    with torch.no_grad():
                        pred_test_scaled = model(x_test.to(DEV)).cpu().numpy()

                    # Keep predictions in scaled space for metrics and saved arrays.
                    pred_train = pred_train_scaled
                    pred_test = pred_test_scaled
                    spec = get_koopman_spectrum(model, koop_attr=koop_attr) if koop_attr else None
                    spectral_radius = float(np.max(np.abs(spec))) if spec is not None else float("nan")
                    stability = perturbation_stability(model, x_test, noise_std=perturb_noise_std)

                    tag = f"{sanitize_name(name)}_seed{seed}_seq{seq_len}_patch{patch_len}_h{horizon}"
                    np.save(pred_dir / f"{tag}_test_predictions_scaled.npy", pred_test_scaled)
                    np.save(train_history_dir / f"{tag}_loss.npy", hist["loss"])
                    if spec is not None:
                        np.save(spec_dir / f"{tag}_spectrum.npy", spec)
                    if save_models:
                        torch.save(model.cpu().state_dict(), save_dir / f"{tag}.pt")

                    last_seed_losses[name] = {"loss": hist["loss"], "eig": hist["eig"]}
                    last_seed_test_preds[name] = pred_test_scaled

                    for set_name, yt, yp in [("Train", y_train_np, pred_train), ("Test", y_test_np, pred_test)]:
                        m = regression_metrics(yt, yp)
                        row = {
                            "Dataset": dataset_name, "Seed": seed, "Model": name,
                            "PatchLen": patch_len, "SeqLen": seq_len, "Horizon": horizon,
                            "NumPatches": num_patches, "Set": set_name,
                            **m,
                            "TrainSeconds": train_seconds,
                            "FinalTrainLoss": float(hist["loss"][-1]) if len(hist["loss"]) else float("nan"),
                            "FinalPredLoss": float(hist["pred_loss"][-1]) if len(hist["pred_loss"]) else float("nan"),
                            "FinalLyapLoss": float(hist["lyap_loss"][-1]) if len(hist["lyap_loss"]) else float("nan"),
                            "SpectralRadius": spectral_radius,
                            **stability,
                        }
                        pd.DataFrame([row], columns=metric_cols).to_csv(metrics_file, mode="a", header=False, index=False)

                        ph_rows = []
                        for hm in per_horizon_metrics(yt, yp, F, horizon):
                            ph_rows.append({**row, **hm})
                        pd.DataFrame(ph_rows).to_csv(per_horizon_file, mode="a", header=False, index=False)

                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Reviewer-friendly figures remain, but tables are now primary evidence.
            if last_seed_losses:
                plot_training(last_seed_losses, save_dir / f"train_seed{seeds[-1]}_seq{seq_len}_patch{patch_len}_h{horizon}")
            if last_seed_test_preds:
                plot_per_feature(y_test_np_scaled, last_seed_test_preds, F, horizon,
                                 save_dir / f"feat_seed{seeds[-1]}_seq{seq_len}_patch{patch_len}_h{horizon}_test_scaled")

            write_summary_tables(save_dir)
            print("\nCurrent best overall models by mean Test MSE:")
            overall_path = save_dir / "summary_mean_std_overall.csv"
            if overall_path.exists():
                print(pd.read_csv(overall_path).head(10).to_string(index=False))

    write_summary_tables(save_dir)
    print(f"\nDone. Reviewer-ready outputs saved in: {save_dir}")
    print("Main tables use SCALED metrics only: metrics_long.csv, metrics_per_horizon.csv, summary_mean_std_by_setting.csv, summary_mean_std_overall.csv")
    print("Statistics: friedman_test_mse.json, friedman_average_ranks.csv, nemenyi_or_wilcoxon_posthoc.csv")


# --------------------------------------------------------------------------- #
# 12)  CLI                                                                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DeepLiquidFormer reviewer-ready benchmark with scaled-space evaluation only")
    p.add_argument("--file", type=str, default="./df_cleaned_numeric_2.npy")
    p.add_argument("--dataset_name", type=str, default="dataset")
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--seq_len", type=int, default=120)
    p.add_argument("--patch_lens", type=str, default="16,24,32,40",
                   help="Comma-separated patch lengths. Keep < seq_len for fair PatchTST.")
    p.add_argument("--horizons", type=str, default="4,8,12,16")
    p.add_argument("--indices", type=str, default="0,1,2,3,4,5")
    p.add_argument("--seeds", type=str, default="7,42,123,2025,2026")
    p.add_argument("--save_dir", type=str, default="results/reviewer_ready_dlf_scal_energy")
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--d_model", type=int, default=96)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=0, help="0 means full-batch training as in the original script.")
    p.add_argument("--patience", type=int, default=0, help="0 disables early stopping.")
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "plateau"])
    p.add_argument("--rho_max", type=float, default=0.99)
    p.add_argument("--save_models", action="store_true")

    p.add_argument("--lyap_weight_constr", type=float, default=0.1)
    p.add_argument("--lyap_weight_learn", type=float, default=0.1)
    p.add_argument("--lyap_weight_unconstr", type=float, default=0.1)
    p.add_argument("--lyap_weight_ode", type=float, default=0.1)
    p.add_argument("--lyap_weight_lnn", type=float, default=0.1)
    p.add_argument("--learnable_kinds", type=str, default="scalar,permode,mlp,lowrank16")

    p.add_argument("--ode_steps", type=int, default=8)
    p.add_argument("--ode_hidden", type=int, default=128)
    p.add_argument("--ode_use_time", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--lnn_steps", type=int, default=16)
    p.add_argument("--lnn_hidden", type=int, default=128)
    p.add_argument("--lnn_use_time", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lnn_tau_min", type=float, default=0.05)
    p.add_argument("--lnn_tau_max", type=float, default=5.0)
    p.add_argument("--perturb_noise_std", type=float, default=1e-3)
    p.add_argument("--include_lnn_ablations", action=argparse.BooleanOptionalAction, default=True)

    args = p.parse_args()
    parse_ints = lambda s: [int(x.strip()) for x in s.split(",") if x.strip()]
    kinds_list = [s.strip() for s in args.learnable_kinds.split(",") if s.strip()] if args.learnable_kinds else []

    main(file=args.file,
         dataset_name=args.dataset_name,
         epochs=args.epochs,
         seq_len=args.seq_len,
         patch_lens=parse_ints(args.patch_lens),
         horizons=parse_ints(args.horizons),
         indices=parse_ints(args.indices),
         seeds=parse_ints(args.seeds),
         save_dir=args.save_dir,
         train_frac=args.train_frac,
         d_model=args.d_model,
         lr=args.lr,
         weight_decay=args.weight_decay,
         batch_size=args.batch_size,
         patience=args.patience,
         scheduler=args.scheduler,
         rho_max=args.rho_max,
         save_models=args.save_models,
         lyap_weight_constr=args.lyap_weight_constr,
         lyap_weight_learn=args.lyap_weight_learn,
         lyap_weight_unconstr=args.lyap_weight_unconstr,
         lyap_weight_ode=args.lyap_weight_ode,
         lyap_weight_lnn=args.lyap_weight_lnn,
         learnable_kinds=kinds_list,
         ode_steps=args.ode_steps,
         ode_hidden=args.ode_hidden,
         ode_use_time=args.ode_use_time,
         lnn_steps=args.lnn_steps,
         lnn_hidden=args.lnn_hidden,
         lnn_use_time=args.lnn_use_time,
         lnn_tau_min=args.lnn_tau_min,
         lnn_tau_max=args.lnn_tau_max,
         perturb_noise_std=args.perturb_noise_std,
         include_lnn_ablations=args.include_lnn_ablations)
