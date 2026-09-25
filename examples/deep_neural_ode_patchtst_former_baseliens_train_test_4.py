#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 09:05:03 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_neuralode_all_variants.py

Koopformer-PRO benchmark (PatchTST backbone) — compare ALL variants together:
  - Baselines: DLinear, LSTM, PatchTST
  - Linear latent: PatchTST+LinearLatent
  - NeuralODE (original): PatchTST+NeuralODE_{RK4,RK45}_{NoLyap,Full}
  - NeuralODE (augmented solutions):
      * Stable linear+residual ODE (A = -B^T B)
      * + Jacobian regularization
      * + Multi-step latent rollout loss
      * RK4 vs Adaptive RK45 (Dormand–Prince 5(4)) comparison

No external deps (no torchdiffeq). HPC-safe matplotlib backend.

Author: Ali Forootani
Updated: 2025-12-19
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
# 2)  ODE solvers: Fixed-step RK4 + Adaptive RK45 (Dormand–Prince 5(4))       #
# --------------------------------------------------------------------------- #
def rk4_integrate(f, z0: torch.Tensor, t0: float = 0.0, t1: float = 1.0, n_steps: int = 8):
    """Fixed-step RK4 for z' = f(t,z) from t0 to t1."""
    n_steps = int(n_steps)
    h = (t1 - t0) / float(n_steps)
    z = z0
    t = torch.tensor(t0, device=z0.device, dtype=z0.dtype)
    hh = torch.tensor(h, device=z0.device, dtype=z0.dtype)
    for _ in range(n_steps):
        k1 = f(t, z)
        k2 = f(t + 0.5*hh, z + 0.5*hh*k1)
        k3 = f(t + 0.5*hh, z + 0.5*hh*k2)
        k4 = f(t + hh,     z + hh*k3)
        z = z + (hh/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        t = t + hh
    return z


class AdaptiveRK45:
    """
    Adaptive RK45 (Dormand–Prince 5(4)) integrator for z' = f(t, z).

    Notes:
      - Error norm uses max over batch and state dims.
      - Step-size adaptation uses detached scalar error for stability.
      - Accepted-step dynamics remain differentiable w.r.t. model parameters.
    """
    def __init__(
        self,
        rtol: float = 1e-3,
        atol: float = 1e-5,
        h0: float = 1.0 / 8.0,
        h_min: float = 1e-6,
        h_max: float = 0.5,
        safety: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        max_steps: int = 1000,
    ):
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.h0 = float(h0)
        self.h_min = float(h_min)
        self.h_max = float(h_max)
        self.safety = float(safety)
        self.min_factor = float(min_factor)
        self.max_factor = float(max_factor)
        self.max_steps = int(max_steps)

        # Dormand–Prince tableau
        self.c2 = 1/5
        self.c3 = 3/10
        self.c4 = 4/5
        self.c5 = 8/9
        self.c6 = 1.0
        self.c7 = 1.0

        self.a21 = 1/5

        self.a31 = 3/40
        self.a32 = 9/40

        self.a41 = 44/45
        self.a42 = -56/15
        self.a43 = 32/9

        self.a51 = 19372/6561
        self.a52 = -25360/2187
        self.a53 = 64448/6561
        self.a54 = -212/729

        self.a61 = 9017/3168
        self.a62 = -355/33
        self.a63 = 46732/5247
        self.a64 = 49/176
        self.a65 = -5103/18656

        self.a71 = 35/384
        self.a72 = 0.0
        self.a73 = 500/1113
        self.a74 = 125/192
        self.a75 = -2187/6784
        self.a76 = 11/84

        # 5th order weights
        self.b1 = 35/384
        self.b2 = 0.0
        self.b3 = 500/1113
        self.b4 = 125/192
        self.b5 = -2187/6784
        self.b6 = 11/84
        self.b7 = 0.0

        # embedded 4th order weights
        self.bs1 = 5179/57600
        self.bs2 = 0.0
        self.bs3 = 7571/16695
        self.bs4 = 393/640
        self.bs5 = -92097/339200
        self.bs6 = 187/2100
        self.bs7 = 1/40

    @torch.no_grad()
    def _clip_h(self, h: float) -> float:
        return max(self.h_min, min(self.h_max, h))

    def integrate(self, f, z0: torch.Tensor, t0: float = 0.0, t1: float = 1.0) -> torch.Tensor:
        t = float(t0)
        tf = float(t1)
        z = z0
        h = self._clip_h(self.h0)

        def t_tensor(val: float, like: torch.Tensor) -> torch.Tensor:
            return torch.tensor(val, device=like.device, dtype=like.dtype)

        def err_norm(e: torch.Tensor, z_ref: torch.Tensor, z_new: torch.Tensor) -> torch.Tensor:
            denom = self.atol + self.rtol * torch.maximum(z_ref.abs(), z_new.abs())
            r = (e.abs() / denom)
            return r.reshape(r.shape[0], -1).max(dim=1).values.max()

        steps = 0
        while t < tf and steps < self.max_steps:
            if t + h > tf:
                h = tf - t

            tt = t_tensor(t, z)
            hh = float(h)

            k1 = f(tt, z)
            k2 = f(t_tensor(t + self.c2*hh, z), z + hh * (self.a21 * k1))
            k3 = f(t_tensor(t + self.c3*hh, z), z + hh * (self.a31 * k1 + self.a32 * k2))
            k4 = f(t_tensor(t + self.c4*hh, z), z + hh * (self.a41 * k1 + self.a42 * k2 + self.a43 * k3))
            k5 = f(t_tensor(t + self.c5*hh, z), z + hh * (self.a51 * k1 + self.a52 * k2 + self.a53 * k3 + self.a54 * k4))
            k6 = f(t_tensor(t + self.c6*hh, z), z + hh * (self.a61 * k1 + self.a62 * k2 + self.a63 * k3 + self.a64 * k4 + self.a65 * k5))
            z5 = z + hh * (self.b1 * k1 + self.b2 * k2 + self.b3 * k3 + self.b4 * k4 + self.b5 * k5 + self.b6 * k6)
            k7 = f(t_tensor(t + self.c7*hh, z), z5)

            z4 = z + hh * (self.bs1 * k1 + self.bs2 * k2 + self.bs3 * k3 + self.bs4 * k4 + self.bs5 * k5 + self.bs6 * k6 + self.bs7 * k7)
            e = (z5 - z4)

            en = err_norm(e, z, z5)
            en_val = float(en.detach().cpu().item())

            if en_val <= 1.0:
                z = z5
                t = t + hh
                steps += 1

            # update step size (DP5 => exponent -1/5)
            if en_val == 0.0:
                factor = self.max_factor
            else:
                factor = self.safety * (en_val ** (-0.2))
                factor = max(self.min_factor, min(self.max_factor, factor))
            h = self._clip_h(h * factor)

            if h <= self.h_min and en_val > 1.0:
                # force accept minimal progress
                z = z5
                t = t + hh
                steps += 1
                h = self._clip_h(self.h0)

        return z

# --------------------------------------------------------------------------- #
# 3)  NeuralODE latent propagators (solver selectable)                         #
# --------------------------------------------------------------------------- #
class ODEFuncMLP(nn.Module):
    """Original: dz/dt = f(z,t)."""
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
    Original NeuralODE latent propagator on [0,1], solver ∈ {rk4, rk45}.
    """
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        use_time: bool = True,
        solver: str = "rk45",
        rk4_steps: int = 8,
        rtol: float = 1e-3,
        atol: float = 1e-5,
        h0: float = 1.0 / 8.0,
        max_steps: int = 1000,
    ):
        super().__init__()
        self.f = ODEFuncMLP(latent_dim, hidden_dim=hidden_dim, use_time=use_time)
        self.solver = str(solver).lower()
        self.rk4_steps = int(rk4_steps)
        self.rk45 = AdaptiveRK45(rtol=rtol, atol=atol, h0=h0, max_steps=max_steps)

        if self.solver not in ("rk4", "rk45"):
            raise ValueError(f"Unknown solver='{solver}'. Use 'rk4' or 'rk45'.")

    def forward(self, z: torch.Tensor):
        if self.solver == "rk4":
            zt = rk4_integrate(self.f, z, t0=0.0, t1=1.0, n_steps=self.rk4_steps)
        else:
            zt = self.rk45.integrate(self.f, z, t0=0.0, t1=1.0)
        return zt, None, None


class StableResidualODEFunc(nn.Module):
    r"""Augmented: dz/dt = A z + g_theta(z,t), with stable A = -B^T B."""
    def __init__(self, latent_dim: int, hidden_dim: int = 128, use_time: bool = True,
                 stable_rank: int | None = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_time = use_time
        r = stable_rank if stable_rank is not None else latent_dim
        self.B = nn.Parameter(0.02 * torch.randn(r, latent_dim))

        inp = latent_dim + (1 if use_time else 0)
        self.g = nn.Sequential(
            nn.Linear(inp, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def A_mat(self):
        return -(self.B.transpose(0, 1) @ self.B)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        A = self.A_mat()
        lin = z @ A.transpose(0, 1)
        if self.use_time:
            if t.ndim == 0:
                tt = t.expand(z.shape[0], 1)
            elif t.ndim == 1:
                tt = t.view(-1, 1)
            else:
                tt = t
            zz = torch.cat([z, tt], dim=-1)
        else:
            zz = z
        return lin + self.g(zz)


class NeuralODELatentPropagator_StableResidual(nn.Module):
    """
    Stable linear+residual ODE on [0,1], solver ∈ {rk4, rk45}.
    Returns (z_next, A, None).
    """
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        use_time: bool = True,
        stable_rank: int | None = None,
        solver: str = "rk45",
        rk4_steps: int = 16,
        rtol: float = 1e-3,
        atol: float = 1e-5,
        h0: float = 1.0 / 16.0,
        max_steps: int = 1500,
    ):
        super().__init__()
        self.f = StableResidualODEFunc(latent_dim, hidden_dim=hidden_dim, use_time=use_time,
                                       stable_rank=stable_rank)
        self.solver = str(solver).lower()
        self.rk4_steps = int(rk4_steps)
        self.rk45 = AdaptiveRK45(rtol=rtol, atol=atol, h0=h0, max_steps=max_steps)

        if self.solver not in ("rk4", "rk45"):
            raise ValueError(f"Unknown solver='{solver}'. Use 'rk4' or 'rk45'.")

    def forward(self, z: torch.Tensor):
        if self.solver == "rk4":
            zt = rk4_integrate(self.f, z, t0=0.0, t1=1.0, n_steps=self.rk4_steps)
        else:
            zt = self.rk45.integrate(self.f, z, t0=0.0, t1=1.0)
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
# 4)  Positional encodings & patch embed                                      #
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
# 5)  PatchTST backbone & variants                                            #
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
    """PatchTST + NeuralODE latent propagation with solver ∈ {rk4,rk45}."""
    def __init__(
        self,
        input_dim,
        seq_len,
        horizon,
        patch_len,
        d_model=64,
        ode_hidden=128,
        ode_use_time=True,
        ode_solver="rk45",
        rk4_steps=8,
        ode_rtol=1e-3,
        ode_atol=1e-5,
        ode_h0=None,
        ode_max_steps=1000,
    ):
        super().__init__()
        self.backbone = PatchTST_Backbone(input_dim, seq_len, patch_len=patch_len, d_model=d_model)
        h0 = (1.0 / 8.0) if ode_h0 is None else float(ode_h0)
        self.latent   = NeuralODELatentPropagator(
            latent_dim=d_model, hidden_dim=ode_hidden, use_time=ode_use_time,
            solver=ode_solver, rk4_steps=rk4_steps,
            rtol=ode_rtol, atol=ode_atol, h0=h0, max_steps=ode_max_steps
        )
        self.fc       = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x, return_latents=False):
        z = self.backbone(x)
        z_next, _, _ = self.latent(z)
        pred = self.fc(z_next)
        if return_latents:
            return pred, z, z_next
        return pred


class NeuralODEStableResidual_PatchTST(nn.Module):
    """PatchTST + stable-residual NeuralODE latent propagation with solver ∈ {rk4,rk45}."""
    def __init__(
        self,
        input_dim,
        seq_len,
        horizon,
        patch_len,
        d_model=64,
        ode_hidden=128,
        ode_use_time=True,
        stable_rank=None,
        ode_solver="rk45",
        rk4_steps=16,
        ode_rtol=1e-3,
        ode_atol=1e-5,
        ode_h0=None,
        ode_max_steps=1500,
    ):
        super().__init__()
        self.backbone = PatchTST_Backbone(input_dim, seq_len, patch_len=patch_len, d_model=d_model)
        h0 = (1.0 / 16.0) if ode_h0 is None else float(ode_h0)
        self.latent   = NeuralODELatentPropagator_StableResidual(
            latent_dim=d_model, hidden_dim=ode_hidden, use_time=ode_use_time,
            stable_rank=stable_rank, solver=ode_solver, rk4_steps=rk4_steps,
            rtol=ode_rtol, atol=ode_atol, h0=h0, max_steps=ode_max_steps
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
# 6)  LSTM baseline                                                           #
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
# 7)  DLinear baseline                                                        #
# --------------------------------------------------------------------------- #
class DLinearForecaster(nn.Module):
    """Simple DLinear-like baseline."""
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
# 8)  Dataset util                                                            #
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
# 9)  Training helpers                                                        #
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

            if jac_weight and latent_module is not None and hasattr(latent_module, "f"):
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

                if jac_weight and hasattr(latent_module, "f"):
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
# 10) Plot helpers                                                            #
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
# 11) Saving metrics                                                          #
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
# 12) Main loop                                                               #
# --------------------------------------------------------------------------- #
def main(file="./df_cleaned_numeric_2.npy", epochs=4000,
         patch_lens=None, horizons=None, indices=None,
         save_dir="results_all_variants", train_frac: float = 0.8,
         rollout_steps_aug: int = 5, jac_weight_aug: float = 1e-4,
         # RK4 steps
         rk4_steps_orig: int = 8, rk4_steps_aug: int = 16,
         # RK45 knobs
         ode_rtol: float = 1e-3, ode_atol: float = 1e-5,
         ode_h0_orig: float | None = None, ode_h0_aug: float | None = None,
         ode_max_steps_orig: int = 1000, ode_max_steps_aug: int = 1500):

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

                # -------- Original NeuralODE: RK4 vs RK45 --------
                ("PatchTST+NeuralODE_RK4_NoLyap",
                 NeuralODE_PatchTST(
                     F, patch_len, horizon, patch_len=patch_len, d_model=96,
                     ode_hidden=128, ode_use_time=True,
                     ode_solver="rk4", rk4_steps=rk4_steps_orig,
                     ode_rtol=ode_rtol, ode_atol=ode_atol,
                     ode_h0=ode_h0_orig, ode_max_steps=ode_max_steps_orig
                 ),
                 dict(lyap=0.0, jac=0.0, roll=1)),

                ("PatchTST+NeuralODE_RK4_Full",
                 NeuralODE_PatchTST(
                     F, patch_len, horizon, patch_len=patch_len, d_model=96,
                     ode_hidden=128, ode_use_time=True,
                     ode_solver="rk4", rk4_steps=rk4_steps_orig,
                     ode_rtol=ode_rtol, ode_atol=ode_atol,
                     ode_h0=ode_h0_orig, ode_max_steps=ode_max_steps_orig
                 ),
                 dict(lyap=0.1, jac=0.0, roll=1)),

                ("PatchTST+NeuralODE_RK45_NoLyap",
                 NeuralODE_PatchTST(
                     F, patch_len, horizon, patch_len=patch_len, d_model=96,
                     ode_hidden=128, ode_use_time=True,
                     ode_solver="rk45", rk4_steps=rk4_steps_orig,
                     ode_rtol=ode_rtol, ode_atol=ode_atol,
                     ode_h0=ode_h0_orig, ode_max_steps=ode_max_steps_orig
                 ),
                 dict(lyap=0.0, jac=0.0, roll=1)),

                ("PatchTST+NeuralODE_RK45_Full",
                 NeuralODE_PatchTST(
                     F, patch_len, horizon, patch_len=patch_len, d_model=96,
                     ode_hidden=128, ode_use_time=True,
                     ode_solver="rk45", rk4_steps=rk4_steps_orig,
                     ode_rtol=ode_rtol, ode_atol=ode_atol,
                     ode_h0=ode_h0_orig, ode_max_steps=ode_max_steps_orig
                 ),
                 dict(lyap=0.1, jac=0.0, roll=1)),

                # -------- StableResidual NeuralODE: RK4 vs RK45 (+ regs) --------
                ("PatchTST+NeuralODE_StableResidual_RK4",
                 NeuralODEStableResidual_PatchTST(
                     F, patch_len, horizon, patch_len=patch_len, d_model=96,
                     ode_hidden=128, ode_use_time=True, stable_rank=None,
                     ode_solver="rk4", rk4_steps=rk4_steps_aug,
                     ode_rtol=ode_rtol, ode_atol=ode_atol,
                     ode_h0=ode_h0_aug, ode_max_steps=ode_max_steps_aug
                 ),
                 dict(lyap=0.0, jac=jac_weight_aug, roll=rollout_steps_aug)),

                ("PatchTST+NeuralODE_StableResidual_RK4+Lyap",
                 NeuralODEStableResidual_PatchTST(
                     F, patch_len, horizon, patch_len=patch_len, d_model=96,
                     ode_hidden=128, ode_use_time=True, stable_rank=None,
                     ode_solver="rk4", rk4_steps=rk4_steps_aug,
                     ode_rtol=ode_rtol, ode_atol=ode_atol,
                     ode_h0=ode_h0_aug, ode_max_steps=ode_max_steps_aug
                 ),
                 dict(lyap=0.1, jac=jac_weight_aug, roll=rollout_steps_aug)),

                ("PatchTST+NeuralODE_StableResidual_RK45",
                 NeuralODEStableResidual_PatchTST(
                     F, patch_len, horizon, patch_len=patch_len, d_model=96,
                     ode_hidden=128, ode_use_time=True, stable_rank=None,
                     ode_solver="rk45", rk4_steps=rk4_steps_aug,
                     ode_rtol=ode_rtol, ode_atol=ode_atol,
                     ode_h0=ode_h0_aug, ode_max_steps=ode_max_steps_aug
                 ),
                 dict(lyap=0.0, jac=jac_weight_aug, roll=rollout_steps_aug)),

                ("PatchTST+NeuralODE_StableResidual_RK45+Lyap",
                 NeuralODEStableResidual_PatchTST(
                     F, patch_len, horizon, patch_len=patch_len, d_model=96,
                     ode_hidden=128, ode_use_time=True, stable_rank=None,
                     ode_solver="rk45", rk4_steps=rk4_steps_aug,
                     ode_rtol=ode_rtol, ode_atol=ode_atol,
                     ode_h0=ode_h0_aug, ode_max_steps=ode_max_steps_aug
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
# 13)  CLI                                                                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="PatchTST benchmark: compare ALL variants incl. NeuralODE RK4 vs RK45 (HPC-ready)"
    )
    p.add_argument("--file", type=str, default="./df_cleaned_numeric_2.npy")
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--save_dir", type=str, default="results/results_all_variants")
    p.add_argument("--train_frac", type=float, default=0.8)

    # Augmented training knobs
    p.add_argument("--rollout_steps_aug", type=int, default=5)
    p.add_argument("--jac_weight_aug", type=float, default=1e-4)

    # RK4 knobs
    p.add_argument("--rk4_steps_orig", type=int, default=8)
    p.add_argument("--rk4_steps_aug", type=int, default=16)

    # RK45 knobs
    p.add_argument("--ode_rtol", type=float, default=1e-3)
    p.add_argument("--ode_atol", type=float, default=1e-5)
    p.add_argument("--ode_h0_orig", type=float, default=None)
    p.add_argument("--ode_h0_aug", type=float, default=None)
    p.add_argument("--ode_max_steps_orig", type=int, default=1000)
    p.add_argument("--ode_max_steps_aug", type=int, default=1500)

    args = p.parse_args()

    main(
        file=args.file,
        epochs=args.epochs,
        save_dir=args.save_dir,
        train_frac=args.train_frac,
        rollout_steps_aug=args.rollout_steps_aug,
        jac_weight_aug=args.jac_weight_aug,
        rk4_steps_orig=args.rk4_steps_orig,
        rk4_steps_aug=args.rk4_steps_aug,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
        ode_h0_orig=args.ode_h0_orig,
        ode_h0_aug=args.ode_h0_aug,
        ode_max_steps_orig=args.ode_max_steps_orig,
        ode_max_steps_aug=args.ode_max_steps_aug,
    )
