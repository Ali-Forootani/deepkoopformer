#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Koopformer-PRO benchmark on Wind-Speed (or similar) data (HPC-ready)

This version AUGMENTS your benchmark with a NeuralODE latent propagator family,
in addition to:
  - constrained Koopman
  - learnable Koopman family
  - unconstrained Koopman
  - baselines (LSTM, DLinear, SSM)

NeuralODE family (per-backbone):
  - NeuralODE_PatchTST
  - NeuralODE_Autoformer
  - NeuralODE_Informer

Notes:
  • No torchdiffeq dependency: fixed-step RK4 ODE solver implemented here.
  • Lyapunov regularization can also be used for NeuralODE (on (z, z_next)).
  • Spectrum logging: Koopman variants save spectrum; NeuralODE does not.

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
# 11)  Main loop                                                              #
# --------------------------------------------------------------------------- #
def main(file="bitstamp_windows.npy", epochs=200, seq_len=120,
         patch_lens=None, horizons=None, indices=None,
         save_dir="results", train_frac: float = 0.8,
         lyap_weight_constr: float = 0.1,
         lyap_weight_learn: float = 0.1,
         lyap_weight_unconstr: float = 0.1,
         lyap_weight_ode: float = 0.1,
         learnable_kinds=None,
         ode_steps: int = 8,
         ode_hidden: int = 128,
         ode_use_time: bool = True):

    patch_lens = patch_lens or [80, 100, 120, 140]
    horizons   = horizons   or [4, 8, 12, 16]
    indices    = indices    or [0, 1, 2, 3, 4, 5]
    learnable_kinds = learnable_kinds or ["scalar", "permode", "mlp", "lowrank16"]

    save_dir = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = save_dir / "metrics.csv"
    pd.DataFrame(columns=["Model", "PatchLen", "Horizon", "Set", "MSE", "MAE"]).to_csv(metrics_file, index=False)

    raw = np.load(file)

    # time split (as in your earlier script)
    N_raw = raw.shape[0]
    split = int(train_frac * N_raw)
    train_raw = raw[:split]
    test_raw  = raw[split:]

    for patch_len in patch_lens:
        for horizon in horizons:
            print(f"\n>>> patch_len={patch_len}, horizon={horizon}\n")

            # Ensure enough samples
            if len(train_raw) < (patch_len + horizon + 2) or len(test_raw) < (patch_len + horizon + 2):
                print("Not enough raw data for train/test windows. Skipping.")
                continue

            # Fit scaler on train only, reuse for test
            x_train, y_train, scaler = build_dataset_fit_scaler(train_raw, indices, patch_len, horizon)
            x_test,  y_test          = build_dataset_with_scaler(test_raw, indices, patch_len, horizon, scaler)

            F, H = len(indices), horizon
            y_train_np = y_train.numpy()
            y_test_np  = y_test.numpy()

            res_loss = {}
            test_preds = {}

            # ---------------- Baselines ---------------- #
            baseline_variants = [
                ("LSTM",   SimpleLSTMForecaster(F, hidden_dim=96, num_layers=2, horizon=horizon), None, 0.0),
                ("DLinear",DLinearForecaster(F, seq_len=patch_len, horizon=horizon), None, 0.0),
                ("SSM",    SimpleSSMForecaster(F, hidden_dim=96, horizon=horizon), None, 0.0),
            ]

            # ---------------- PatchTST families ---------------- #
            patchtst_variants = [
                ("Koop-PatchTST (constr.)",
                 Koopformer_PatchTST_Constrained(F, patch_len, horizon, patch_len=patch_len, d_model=96, ρ_max=0.99),
                 "koop", lyap_weight_constr),

                ("Koop-PatchTST (unconstr.)",
                 Koopformer_PatchTST_Unconstrained(F, patch_len, horizon, patch_len=patch_len, d_model=96),
                 "koop", lyap_weight_unconstr),

                ("ODE-PatchTST",
                 NeuralODE_PatchTST(F, patch_len, horizon, patch_len=patch_len, d_model=96,
                                    ode_hidden=ode_hidden, ode_steps=ode_steps, ode_use_time=ode_use_time),
                 None, lyap_weight_ode),
            ]

            learn_patchtst = []
            for kind in learnable_kinds:
                learn_patchtst.append((
                    f"Koop-PatchTST (learn:{kind})",
                    Koopformer_PatchTST_Learnable(F, patch_len, horizon, patch_len=patch_len, d_model=96, ρ_max=0.99, koop_kind=kind),
                    "koop", lyap_weight_learn
                ))

            # ---------------- Autoformer families ---------------- #
            autoformer_variants = [
                ("Koop-Autoformer (constr.)",
                 Koopformer_Autoformer_Constrained(F, patch_len, horizon, patch_len=patch_len, d_model=96, ρ_max=0.99),
                 "koop", lyap_weight_constr),

                ("Koop-Autoformer (unconstr.)",
                 Koopformer_Autoformer_Unconstrained(F, patch_len, horizon, patch_len=patch_len, d_model=96),
                 "koop", lyap_weight_unconstr),

                ("ODE-Autoformer",
                 NeuralODE_Autoformer(F, patch_len, horizon, patch_len=patch_len, d_model=96,
                                      ode_hidden=ode_hidden, ode_steps=ode_steps, ode_use_time=ode_use_time),
                 None, lyap_weight_ode),
            ]

            learn_autoformer = []
            for kind in learnable_kinds:
                learn_autoformer.append((
                    f"Koop-Autoformer (learn:{kind})",
                    Koopformer_Autoformer_Learnable(F, patch_len, horizon, patch_len=patch_len, d_model=96, ρ_max=0.99, koop_kind=kind),
                    "koop", lyap_weight_learn
                ))

            # ---------------- Informer families ---------------- #
            informer_variants = [
                ("Koop-Informer (constr.)",
                 Koopformer_Informer_Constrained(F, patch_len, horizon, patch_len=patch_len, d_model=96, ρ_max=0.99),
                 "koop", lyap_weight_constr),

                ("Koop-Informer (unconstr.)",
                 Koopformer_Informer_Unconstrained(F, patch_len, horizon, patch_len=patch_len, d_model=96),
                 "koop", lyap_weight_unconstr),

                ("ODE-Informer",
                 NeuralODE_Informer(F, patch_len, horizon, patch_len=patch_len, d_model=96,
                                    ode_hidden=ode_hidden, ode_steps=ode_steps, ode_use_time=ode_use_time),
                 None, lyap_weight_ode),
            ]

            learn_informer = []
            for kind in learnable_kinds:
                learn_informer.append((
                    f"Koop-Informer (learn:{kind})",
                    Koopformer_Informer_Learnable(F, patch_len, horizon, patch_len=patch_len, d_model=96, ρ_max=0.99, koop_kind=kind),
                    "koop", lyap_weight_learn
                ))

            all_variants = baseline_variants + patchtst_variants + learn_patchtst + autoformer_variants + learn_autoformer + informer_variants + learn_informer

            # ---------------- Train/Eval loop ---------------- #
            for name, model, koop_attr, lyap_w in all_variants:
                print(f"== {name} ==")

                # train on train set
                pred_train, loss, eig = train(model, x_train, y_train, epochs=epochs, lr=3e-4, koop_attr=koop_attr, lyap_weight=lyap_w)
                res_loss[name] = {"loss": loss, "eig": eig}

                # predictions on test set
                model.to(DEV).eval()
                with torch.no_grad():
                    pred_test = model(x_test.to(DEV)).cpu().numpy()
                test_preds[name] = pred_test

                # save model + metrics + npy (train/test)
                prefix = save_dir / f"{name.replace(' ', '_').replace(':','-').replace('(','').replace(')','')}_{patch_len}_{horizon}"
                torch.save(model.cpu().state_dict(), f"{prefix}.pt")

                save_results(pred_train, y_train_np, name, prefix, metrics_file, patch_len, horizon, set_name="Train")
                save_results(pred_test,  y_test_np,  name, prefix, metrics_file, patch_len, horizon, set_name="Test")

                # save spectrum only for Koopman layers
                if koop_attr:
                    spec = get_koopman_spectrum(model, koop_attr=koop_attr)
                    if spec is not None:
                        np.save(f"{prefix}_spectrum.npy", spec)

            # ---------------- Figures ---------------- #
            plot_training(res_loss, save_dir / f"train_patch{patch_len}_h{horizon}")
            plot_per_feature(y_test_np, test_preds, F, H, save_dir / f"feat_patch{patch_len}_h{horizon}_test")

            # Console summary (TEST)
            print("\nFinal TEST metrics:")
            for n, p in test_preds.items():
                e = y_test_np - p
                print(f"{n:<40s} MSE {(e**2).mean():.4e}  MAE {np.abs(e).mean():.4e}")

# --------------------------------------------------------------------------- #
# 12)  CLI                                                                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Koopformer-PRO benchmark + NeuralODE family (HPC-ready)"
    )
    p.add_argument("--file", type=str, default="./df_cleaned_numeric_2.npy")
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--seq_len", type=int, default=120)
    p.add_argument("--save_dir", type=str, default="results/results_dkf_plus_neuralode")
    p.add_argument("--train_frac", type=float, default=0.8)

    p.add_argument("--lyap_weight_constr", type=float, default=0.1)
    p.add_argument("--lyap_weight_learn", type=float, default=0.1)
    p.add_argument("--lyap_weight_unconstr", type=float, default=0.1)
    p.add_argument("--lyap_weight_ode", type=float, default=0.1)

    p.add_argument("--learnable_kinds", type=str, default="scalar,permode,mlp,lowrank16")

    p.add_argument("--ode_steps", type=int, default=8)
    p.add_argument("--ode_hidden", type=int, default=128)
    p.add_argument("--ode_use_time", action="store_true", default=True)

    args = p.parse_args()

    kinds_list = [s.strip() for s in args.learnable_kinds.split(",") if s.strip()] if args.learnable_kinds else None

    main(file=args.file,
         epochs=args.epochs,
         seq_len=args.seq_len,
         save_dir=args.save_dir,
         train_frac=args.train_frac,
         lyap_weight_constr=args.lyap_weight_constr,
         lyap_weight_learn=args.lyap_weight_learn,
         lyap_weight_unconstr=args.lyap_weight_unconstr,
         lyap_weight_ode=args.lyap_weight_ode,
         learnable_kinds=kinds_list,
         ode_steps=args.ode_steps,
         ode_hidden=args.ode_hidden,
         ode_use_time=args.ode_use_time)
