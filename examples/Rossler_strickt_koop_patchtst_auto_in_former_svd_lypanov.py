#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 19:05:03 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Koopformer PRO: Rössler System Benchmarking with PatchTST, Autoformer, InformerSparse Backbones
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==== Rössler 4-State System Simulator ====
def simulate_rossler_system(a=0.2, b=0.2, c=5.7, gamma=0.1, delta=0.2, dt=0.01, T=20.0):
    num_steps = int(T / dt)
    x = torch.zeros((num_steps, 4))
    x[0] = torch.tensor([1.0, 1.0, 1.0, 0.5])

    for t in range(num_steps - 1):
        x1, x2, x3, x4 = x[t]
        dx1 = -x2 - x3
        dx2 = x1 + a * x2
        dx3 = b + x3 * (x1 - c)
        dx4 = -gamma * x4 + delta * x3

        x[t + 1, 0] = x1 + dt * dx1 + 0.1 * torch.randn(1)
        x[t + 1, 1] = x2 + dt * dx2 + 0.1 * torch.randn(1)
        x[t + 1, 2] = x3 + dt * dx3 + 0.01 * torch.randn(1)
        x[t + 1, 3] = x4 + dt * dx4 + 0.01 * torch.randn(1)
    return x

# ==== Householder Orthogonalization ====
def householder_orthogonalization(W):
    Q, _ = torch.linalg.qr(W)
    return Q

# ==== Strict Stable Koopman Operator ====
class StrictStableKoopmanOperator(nn.Module):
    def __init__(self, latent_dim, max_spectral_radius=0.99):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_spectral_radius = max_spectral_radius
        self.U_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.V_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.S_unconstrained = nn.Parameter(torch.randn(latent_dim))

    def forward(self, z):
        U = householder_orthogonalization(self.U_raw)
        V = householder_orthogonalization(self.V_raw)
        S = torch.sigmoid(self.S_unconstrained) * self.max_spectral_radius
        K = U @ torch.diag(S) @ V.T
        z_next = z @ K.T
        return z_next, K, S

# ==== Positional Encoding ====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        assert x.size(1) <= self.pe.size(1), f"Input sequence length {x.size(1)} exceeds max_len={self.pe.size(1)}"
        return x + self.pe[:, :x.size(1)].to(x.device)

# ==== PatchTST Backbone ====
class PatchTSTBackbone(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, dim_feedforward, num_layers):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding
        x = self.encoder(x)
        return x

# ==== Autoformer Backbone ====
class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        trend = self.moving_avg(x.transpose(1, 2)).transpose(1, 2)
        return trend, x - trend

class AutoformerBackbone(nn.Module):
    def __init__(self, input_len, d_model):
        super().__init__()
        self.decomp = SeriesDecomposition(3)
        self.embed = nn.Linear(1, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=500)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 2, 32, batch_first=True), 2)

    def forward(self, x):
        trend, seasonal = self.decomp(x)
        seasonal = self.pos_encoding(self.embed(seasonal))
        latent = self.encoder(seasonal).mean(dim=1)
        return latent

# ==== Informer Backbone ====
class InformerSparse(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=500)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 2, 32, batch_first=True), 2)

    def forward(self, x):
        x = self.pos_encoding(self.embed(x))
        return self.encoder(x).mean(dim=1)

# ==== Koopformer PRO Wrappers ====
class PatchTSTWrapper(nn.Module):
    def __init__(self, patch_length, horizon, state_dim=4):
        super().__init__()
        self.backbone = PatchTSTBackbone(patch_length * state_dim, 16, 2, 64, 2)
        self.koopman = StrictStableKoopmanOperator(16)
        self.fc = nn.Linear(16, horizon * state_dim)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, -1)
        latent = self.backbone(x).mean(dim=1)
        latent_next, K, eigvals = self.koopman(latent)
        out = self.fc(latent_next)
        return out, latent, latent_next, K, eigvals

class AutoformerWrapper(nn.Module):
    def __init__(self, patch_length, horizon, state_dim=4):
        super().__init__()
        self.backbone = AutoformerBackbone(patch_length * state_dim, 16)
        self.koopman = StrictStableKoopmanOperator(16)
        self.fc = nn.Linear(16, horizon * state_dim)

    def forward(self, x):
        # x: [B, patch_length, state_dim]
        x = x.reshape(x.shape[0], -1, 1)  # (B, seq_len, 1)
        latent = self.backbone(x)
        latent_next, K, eigvals = self.koopman(latent)
        out = self.fc(latent_next)
        return out, latent, latent_next, K, eigvals

class InformerWrapper(nn.Module):
    def __init__(self, patch_length, horizon, state_dim=4):
        super().__init__()
        self.backbone = InformerSparse(state_dim, 16)
        self.koopman = StrictStableKoopmanOperator(16)
        self.fc = nn.Linear(16, horizon * state_dim)

    def forward(self, x):
        # x: [B, patch_length, state_dim]
        latent = self.backbone(x)
        latent_next, K, eigvals = self.koopman(latent)
        out = self.fc(latent_next)
        return out, latent, latent_next, K, eigvals

# ==== Training and Evaluation ====
def train_and_eval(model, x_in, x_out, epochs=3000):
    opt = optim.Adam(model.parameters(), lr=0.001)
    mse = nn.MSELoss()
    eig_hist = []

    for e in range(epochs):
        model.train()
        opt.zero_grad()
        preds, latent, latent_next, _, eigvals = model(x_in)
        loss = mse(preds, x_out) + 0.1 * torch.relu((latent_next.norm(dim=1) ** 2 - latent.norm(dim=1) ** 2)).mean()
        loss.backward()
        opt.step()
        if e % 20 == 0:
            eig_hist.append(eigvals.detach().cpu().numpy())

    model.eval()
    with torch.no_grad():
        preds, _, _, _, _ = model(x_in)
    return preds.cpu().numpy(), x_out.cpu().numpy(), np.array(eig_hist)

# ==== MAIN ====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = simulate_rossler_system()
patch_length, horizon = 100, 5   # patch_length*state_dim=400 < 500, fits in positional encoding
state_dim = data.shape[1]

x_in, x_out = [], []
for i in range(len(data) - patch_length - horizon):
    x_in.append(data[i:i+patch_length])
    x_out.append(data[i+patch_length:i+patch_length+horizon].reshape(-1))

x_in = torch.stack(x_in).to(device)
x_out = torch.stack(x_out).to(device)

models = {
    "PatchTST": PatchTSTWrapper(patch_length, horizon, state_dim).to(device),
    "Autoformer": AutoformerWrapper(patch_length, horizon, state_dim).to(device),
    "Informer": InformerWrapper(patch_length, horizon, state_dim).to(device),
}

results = {}
for name, model in models.items():
    print(f"Training {name} ...")
    preds, true_vals, eigvals = train_and_eval(model, x_in, x_out)
    results[name] = (preds, true_vals, eigvals)

# === Save results ===
import os
import pickle

results_dir = "koopformer_rossler_results"
os.makedirs(results_dir, exist_ok=True)

with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
    pickle.dump(results, f)
print(f"Results saved to {os.path.join(results_dir, 'results.pkl')}")


# ==== Plot Predictions ====


# --- Set consistent font sizes ---
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 12
SUPTITLE_FONTSIZE = 16
LEGEND_FONTSIZE = 12

# ==== Plot Predictions ====
fig, axes = plt.subplots(state_dim, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
var_labels = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$"]

for idx, ax in enumerate(axes):
    ax.plot(results["PatchTST"][1][:, idx], label=f"True {var_labels[idx]}", color='black', linewidth=2)
    for name, (preds, _, _) in results.items():
        ax.plot(preds[:, idx], '--', label=f"{name} Pred {var_labels[idx]}", alpha=0.7)
    ax.set_ylabel(var_labels[idx], fontsize=LABEL_FONTSIZE)
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, ncol=2)  # <--- legend in each subplot
    if idx == 0:
        pass
        #ax.set_title("Koopformer PRO Variants - Rössler System State Predictions", fontsize=TITLE_FONTSIZE, pad=12)
    if idx == state_dim - 1:
        ax.set_xlabel("Time Step", fontsize=LABEL_FONTSIZE)

# Optionally, add a suptitle if you want a figure-wide title
# fig.suptitle("Koopformer PRO Variants - Rössler System State Predictions", fontsize=SUPTITLE_FONTSIZE, y=1.03)

plt.savefig(os.path.join(results_dir, "koopformer_rossler_variants_prediction.png"), dpi=600, transparent=True)
plt.savefig(os.path.join(results_dir, "koopformer_rossler_variants_prediction.pdf"), dpi=600)
plt.show()

# ==== Plot Koopman Eigenvalue Evolution ====


import os
import matplotlib.pyplot as plt
import numpy as np

# Define your font sizes as before
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 12

# Set line styles and colors for each model
model_styles = {
    "PatchTST":  {'linestyle': '-',  'color': '#1f77b4'},   # blue, solid
    "Autoformer":{'linestyle': '--', 'color': '#ff7f0e'},   # orange, dashed
    "Informer":  {'linestyle': '-.', 'color': '#2ca02c'},   # green, dash-dot
    # add more if you have more models
}

plt.figure(figsize=(8, 6))
for name, (_, _, eigvals_hist) in results.items():
    style = model_styles.get(name, {})
    plt.plot(
        np.arange(eigvals_hist.shape[0]),
        eigvals_hist.max(axis=1),
        label=name,
        linestyle=style.get('linestyle', '-'),
        color=style.get('color', None),
        linewidth=2,
    )
plt.xlabel("Training Step (x20 epochs)", fontsize=LABEL_FONTSIZE)
plt.ylabel("Max Eigenvalue (Spectral Radius)", fontsize=LABEL_FONTSIZE)
plt.title("Deep Koopman Operator Stability Evolution", fontsize=TITLE_FONTSIZE, pad=12)
plt.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
plt.legend(fontsize=LEGEND_FONTSIZE)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "koopformer_rossler_variants_eigenvalues.png"), dpi=600, transparent=True)
plt.savefig(os.path.join(results_dir, "koopformer_rossler_variants_eigenvalues.pdf"), dpi=600)
plt.show()





