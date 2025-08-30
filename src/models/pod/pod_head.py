# src/models/pod/pod_head.py
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np

def _load_basis_any(path: str, H: int, W: int, modes: int) -> torch.Tensor:
    npz = np.load(path, allow_pickle=True)
    keys = set(npz.files)
    HW = H * W

    def block_diag(Pu, Pv, Pp):
        r = Pu.shape[1] + Pv.shape[1] + Pp.shape[1]
        B = np.zeros((3*HW, r), dtype=np.float32)
        B[0*HW:1*HW, 0:Pu.shape[1]] = Pu
        B[1*HW:2*HW, Pu.shape[1]:Pu.shape[1]+Pv.shape[1]] = Pv
        B[2*HW:3*HW, Pu.shape[1]+Pv.shape[1]:] = Pp
        return B

    # NEW format
    if "basis" in keys:
        B = npz["basis"].astype(np.float32)   # (3HW, r)
        return torch.from_numpy(B[:, :modes] if modes < B.shape[1] else B)

    # Legacy guess 1: per-channel Phi_u/Phi_v/Phi_p
    if {"Phi_u","Phi_v","Phi_p"}.issubset(keys):
        Pu = npz["Phi_u"].astype(np.float32)
        Pv = npz["Phi_v"].astype(np.float32)
        Pp = npz["Phi_p"].astype(np.float32)
        B = block_diag(Pu, Pv, Pp)
        return torch.from_numpy(B[:, :modes] if modes < B.shape[1] else B)

    # Legacy guess 2: monolithic Vt (HW or 3HW) — try to interpret
    if "Vt" in keys:
        Vt = npz["Vt"].astype(np.float32)  # (r, D)
        if Vt.shape[1] == 3*HW:
            B = Vt[:modes].T
            return torch.from_numpy(B)
        elif Vt.shape[1] == HW:
            # same basis for all 3 channels (fallback)
            Phi = Vt[: (modes//3 or 1)].T  # (HW, r1)
            Pu = Phi; Pv = Phi; Pp = Phi
            B = block_diag(Pu, Pv, Pp)
            return torch.from_numpy(B[:, :modes] if modes < B.shape[1] else B)

    raise KeyError(
        f"POD basis file '{path}' has keys {sorted(keys)}, "
        f"but expected 'basis' (new) or ('Phi_u','Phi_v','Phi_p') / 'Vt' (legacy). "
        f"Rebuild with scripts/build_pod_basis.py."
    )

class PODRegressor(nn.Module):
    """
    Predict POD coefficients (r) from grid inputs, then reconstruct y ∈ R^{3×H×W} via basis (3HW×r).
    Baseline: small conv encoder + MLP head that emits r coeffs.
    """
    def __init__(self, in_channels: int, H: int, W: int, modes: int, basis_path: str,
                 hidden: int = 128, mlp_layers=(128,128)):
        super().__init__()
        self.in_channels = in_channels
        self.H, self.W, self.r = int(H), int(W), int(modes)

        # Load / normalize basis
        B = _load_basis_any(basis_path, self.H, self.W, modes)  # (3HW, r)
        assert B.shape[0] == 3*self.H*self.W, f"basis rows {B.shape[0]} != 3HW {3*self.H*self.W}"
        assert B.shape[1] >= self.r, f"basis cols {B.shape[1]} < requested modes {self.r}"
        self.register_buffer("basis", B[:, :self.r].contiguous())

        # Tiny encoder + head
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        mlp = []
        in_dim = hidden
        for h in mlp_layers:
            mlp += [nn.Linear(in_dim, h), nn.GELU()]
            in_dim = h
        mlp += [nn.Linear(in_dim, self.r)]
        self.head = nn.Sequential(*mlp)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        z = self.enc(x)
        z = self.pool(z).view(B, -1)
        coeffs = self.head(z)                   # [B, r]
        yvec = torch.matmul(coeffs, self.basis.T)  # [B, 3HW]
        return yvec.view(B, 3, self.H, self.W)
