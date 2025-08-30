# src/models/heads/metrics_head.py
from __future__ import annotations
import torch
import torch.nn as nn


class MetricsHead(nn.Module):
    """
    Optional head to map a latent vector to integrated metrics
    (e.g., Cl, Cd, and sampled Cp stations).
    Configure out_dim accordingly (e.g., 2 + Ns for Cl, Cd, Cp_curve).
    """
    def __init__(self, latent_dim: int, out_dim: int, hidden=(128, 128)):
        super().__init__()
        layers = []
        d = latent_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(True)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
