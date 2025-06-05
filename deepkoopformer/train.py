import torch
import numpy as np

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
from .koopman import StrictStableKoopmanOperator


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
