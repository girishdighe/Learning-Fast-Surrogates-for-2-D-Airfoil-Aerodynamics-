# src/models/unet/unet2d.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch: int, out_ch: int, dropout: float = 0.0):
    layers = [
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout and dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = conv_block(in_ch, out_ch, dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.block(x)
        x_down = self.pool(x)
        return x, x_down


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.block = conv_block(in_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class UNet2D(nn.Module):
    """
    Simple 2D U-Net for dense regression (u, v, p).
    """

    def __init__(self, in_channels: int, out_channels: int = 3, base_channels: int = 64,
                 depth: int = 4, dropout: float = 0.0):
        super().__init__()
        assert depth >= 3, "depth should be >= 3"

        chs = [base_channels * (2 ** i) for i in range(depth)]
        self.inc = conv_block(in_channels, chs[0], dropout)
        self.downs = nn.ModuleList()
        for i in range(depth - 1):
            self.downs.append(Down(chs[i], chs[i + 1], dropout))

        self.bottom = conv_block(chs[-1], chs[-1], dropout)

        ups = []
        for i in range(depth - 1, 0, -1):
            ups.append(Up(chs[i], chs[i - 1], dropout))
        self.ups = nn.ModuleList(ups)

        self.out_conv = nn.Conv2d(chs[0], out_channels, kernel_size=1)

    def forward(self, x):
        xs = []
        x0 = self.inc(x)
        skip = x0
        xs.append(skip)
        x = x0
        for d in self.downs:
            skip, x = d(x)
            xs.append(skip)
        x = self.bottom(x)
        for up, sk in zip(self.ups, reversed(xs[:-1])):
            x = up(x, sk)
        out = self.out_conv(x)
        return out
