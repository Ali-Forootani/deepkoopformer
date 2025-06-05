import torch
from torch import nn
import numpy as np
from .layers import PatchEmbed1D, SinCosPosEnc, SeriesDecomp
from .koopman import StrictStableKoopmanOperator


# --------------------------------------------------------------------------- #
# 4)  PatchTST backbone & Koopformer                                          #
# --------------------------------------------------------------------------- #
class PatchTST_Backbone(nn.Module):
    def __init__(self, input_dim: int, seq_len: int,
                 patch_len: int, d_model: int = 48,
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
                 patch_len, d_model=48):
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
                 d_model=48, num_heads=4,
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
    def __init__(self, input_dim, d_model=48, num_heads=4,
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
                 patch_len, d_model=48):
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

