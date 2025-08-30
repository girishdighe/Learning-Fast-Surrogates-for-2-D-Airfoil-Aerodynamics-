# src/utils/normalization.py
import numpy as np
import torch
from dataclasses import dataclass

@dataclass
class NormState:
    mean: np.ndarray  # (C,)
    std:  np.ndarray  # (C,)

class Normalizer:
    """
    Channel-wise z-score normalizer with explicit (C,) mean/std.
    - Works on numpy arrays [C,H,W] or torch tensors [B,C,H,W].
    - Always uses stats saved at TRAIN TIME (no refit at eval!).
    """
    def __init__(self, state: NormState | None = None):
        self.state = state

    @staticmethod
    def fit(x_np: np.ndarray, eps: float = 1e-8) -> "Normalizer":
        # x_np: [N,C,H,W] or [C,H,W] aggregated manually
        x = x_np.reshape(-1, x_np.shape[-3], x_np.shape[-2], x_np.shape[-1]) if x_np.ndim == 4 else x_np[None]
        mean = x.mean(axis=(0,2,3))  # (C,)
        std  = x.std(axis=(0,2,3)) + eps
        return Normalizer(NormState(mean=mean.astype(np.float32), std=std.astype(np.float32)))

    def transform(self, arr):
        if self.state is None: return arr
        m = torch.as_tensor(self.state.mean) if torch.is_tensor(arr) else self.state.mean
        s = torch.as_tensor(self.state.std)  if torch.is_tensor(arr) else self.state.std
        if torch.is_tensor(arr):
            return (arr - m[None,:,None,None].to(arr)) / s[None,:,None,None].to(arr)
        else:
            return (arr - m[:,None,None]) / s[:,None,None]

    def inv_transform(self, arr):
        if self.state is None: return arr
        m = torch.as_tensor(self.state.mean) if torch.is_tensor(arr) else self.state.mean
        s = torch.as_tensor(self.state.std)  if torch.is_tensor(arr) else self.state.std
        if torch.is_tensor(arr):
            return arr * s[None,:,None,None].to(arr) + m[None,:,None,None].to(arr)
        else:
            return arr * s[:,None,None] + m[:,None,None]

    # — persistence —
    def save(self, path: str):
        np.savez(path, mean=self.state.mean, std=self.state.std)

    @staticmethod
    def load(path: str) -> "Normalizer":
        z = np.load(path)
        return Normalizer(NormState(mean=z["mean"].astype(np.float32), std=z["std"].astype(np.float32)))
