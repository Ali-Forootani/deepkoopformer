#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 10 2025
@author: forootan
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
        return x + self.pe[:, :x.size(1)].to(x.device)

# ==== Householder Orthogonalization ====
def householder_orthogonalization(W):
    Q, _ = torch.linalg.qr(W)
    return Q

# ==== Strict Stable Koopman Operator ====
class StrictStableKoopmanOperator(nn.Module):
    def __init__(self, latent_dim, max_spectral_radius=0.99, use_householder=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_spectral_radius = max_spectral_radius
        self.use_householder = use_householder

        self.U_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.V_raw = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.S_unconstrained = nn.Parameter(torch.randn(latent_dim))

    def orthogonalize(self, W):
        if self.use_householder:
            return householder_orthogonalization(W)
        else:
            Q, _ = torch.linalg.qr(W)
            return Q

    def forward(self, z):
        U = self.orthogonalize(self.U_raw)
        V = self.orthogonalize(self.V_raw)
        S = torch.sigmoid(self.S_unconstrained) * self.max_spectral_radius
        K = U @ torch.diag(S) @ V.T
        z_next = torch.matmul(z, K.T)
        return z_next, K, S

# ==== PatchTST Backbone ====
class PatchTSTBackbone(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, dim_feedforward, num_layers):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = self.encoder(x)
        return x

# ==== Koopformer PRO ====
class KoopformerPRO(nn.Module):
    def __init__(self, input_dim, horizon, d_model, num_heads, dim_feedforward, num_layers):
        super().__init__()
        self.output_dim = 4  # Rössler system: 4-state
        self.backbone = PatchTSTBackbone(input_dim, d_model, num_heads, dim_feedforward, num_layers)
        self.koopman = StrictStableKoopmanOperator(d_model)
        self.fc = nn.Linear(d_model, horizon * self.output_dim)

    def forward(self, x):
        latent = self.backbone(x)
        latent = latent.mean(dim=1)
        latent_next, K, eigvals = self.koopman(latent)
        out = self.fc(latent_next)
        return out, latent, latent_next, K, eigvals

# ==== Training ====
def train_and_evaluate(model, x_in, x_out, name, epochs=3000):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()
    eigvals_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds, latent, latent_next, K, eigvals = model(x_in)
        pred_loss = mse_loss(preds, x_out)
        diff = (latent_next.norm(dim=1) ** 2) - (latent.norm(dim=1) ** 2)
        lyapunov_loss = torch.relu(diff).mean()
        loss = pred_loss + 0.1 * lyapunov_loss
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            eigvals_history.append(eigvals.detach().cpu().numpy())

    model.eval()
    with torch.no_grad():
        preds, _, _, _, _ = model(x_in)
        preds = preds.cpu().numpy()
        true_vals = x_out.cpu().numpy()

    return preds, true_vals, eigvals_history

# ==== MAIN ====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = simulate_rossler_system()
patch_length = 200
horizon = 5
output_dim = data.shape[1]

x_in, x_out = [], []
for i in range(len(data) - patch_length - horizon):
    x_in.append(data[i:i+patch_length])
    x_out.append(data[i+patch_length:i+patch_length+horizon].reshape(-1))

x_in = torch.stack(x_in).to(device)
x_out = torch.stack(x_out).to(device)

models = {
    "KoopformerPRO (Rössler)": KoopformerPRO(patch_length * output_dim, horizon, 16, 2, 64, 2).to(device),
}

results = {}
for name, model in models.items():
    preds, true_vals, eigvals_hist = train_and_evaluate(model, x_in.reshape(x_in.shape[0], 1, -1), x_out, name)
    results[name] = (preds, true_vals, eigvals_hist)

# ==== Plot Predictions ====
plt.figure(figsize=(10, 14))
model_name, (preds, true_vals, _) = list(results.items())[0]
var_labels = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$"]

for i in range(4):
    ax = plt.subplot(4, 1, i + 1)
    ax.plot(true_vals[:, i], label=fr"True {var_labels[i]}", linewidth=2)
    ax.plot(preds[:, i], '--', label=fr"Predicted {var_labels[i]}", linewidth=3)
    ax.set_ylabel(var_labels[i], fontsize=12)
    if i == 3:
        ax.set_xlabel("Time Step", fontsize=12)
    ax.legend()
    ax.grid(True)

plt.suptitle(f"{model_name} - Prediction Results", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("koopformer_rossler_prediction.png", dpi=300, transparent=True)
plt.savefig("koopformer_rossler_prediction.pdf", dpi=300)
plt.show()

# ==== Plot Eigenvalue Evolution ====
plt.figure(figsize=(8, 6))
for name, (_, _, eigvals_hist) in results.items():
    eigvals_hist = np.array(eigvals_hist)
    plt.plot(np.arange(eigvals_hist.shape[0]), eigvals_hist.max(axis=1), label=name, linestyle='--')

plt.xlabel(r"Training Step ($\times$10 epochs)", fontsize=14)
plt.ylabel(r"Max Eigenvalue (Spectral Radius)", fontsize=14)
plt.title(r"Koopman Operator Stability Evolution", fontsize=15)
plt.legend()
plt.grid()
plt.savefig("koopformer_rossler_eigenvalues.png", dpi=300, transparent=True)
plt.savefig("koopformer_rossler_eigenvalues.pdf", dpi=300)
plt.show()
