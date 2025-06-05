import torch
from torch import nn
import numpy as np



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
    def __init__(self, in_ch: int, d_model: int,
                 patch_len: int, stride: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, d_model,
                              kernel_size=patch_len, stride=stride)

    def forward(self, x):
        x = x.permute(0, 2, 1)     # (B, C, T)
        x = self.conv(x)           # (B, d, T//patch)
        return x.permute(0, 2, 1)  # (B, P, d)
    


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
