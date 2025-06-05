import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


# --------------------------------------------------------------------------- #
# 8)  Dataset util                                                            #
# --------------------------------------------------------------------------- #
def build_dataset(data: np.ndarray, indices: list[int],
                  seq_len: int, horizon: int):
    sub = data[:400, indices]
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
