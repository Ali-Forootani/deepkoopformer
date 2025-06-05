import torch
from torch import nn


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
    



