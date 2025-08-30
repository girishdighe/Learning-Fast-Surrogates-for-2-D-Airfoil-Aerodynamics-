# src/models/fno/fno2d.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    """
    2D spectral convolution ala FNO.
    - Uses CPU for FFT/complex math when input is on MPS (Apple GPU lacks complex support).
    - Parameters are real (weight_real, weight_imag); complex kernel is formed on the fly.
    """
    def __init__(self, in_channels: int, out_channels: int, modes=(16,16), scale: float = 0.01):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = int(modes[0])
        self.modes2 = int(modes[1])
        # Real/imag parts as learnable params (float32)
        self.weight_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2))
        self.weight_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2))

    def _spectral_op(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C_in, H, W]  (will run on CPU if src is MPS)
        returns: [B, C_out, H, W]
        """
        on_mps = (x.device.type == "mps")
        dev_x = x.device
        cpu = torch.device("cpu")

        # Run FFT on CPU if needed
        x_ = x.to(cpu) if on_mps else x
        x_ft = torch.fft.rfft2(x_, norm="ortho")  # [B, C_in, H, W/2+1] (complex64)

        H, W = x_.shape[-2], x_.shape[-1]
        mx = min(self.modes1, x_ft.size(-2))     # freq in y (height)
        my = min(self.modes2, x_ft.size(-1))     # freq in x (width/2+1)

        # Complex weights on same device as x_ft
        w = torch.complex(
            (self.weight_real.to(x_ft.device))[:, :, :mx, :my],
            (self.weight_imag.to(x_ft.device))[:, :, :mx, :my]
        )  # [C_in, C_out, mx, my]

        out_ft = torch.zeros(x_ft.shape[0], self.out_channels, x_ft.size(-2), x_ft.size(-1),
                             dtype=x_ft.dtype, device=x_ft.device)
        # Complex multiply + channel mix: (B, C_in, mx, my) x (C_in, C_out, mx, my) -> (B, C_out, mx, my)
        out_ft[:, :, :mx, :my] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :mx, :my], w)

        y_ = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho").real  # [B, C_out, H, W]
        return y_.to(dev_x) if on_mps else y_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._spectral_op(x)


class FNOBlock(nn.Module):
    def __init__(self, channels: int, modes=(16,16)):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes=modes)
        self.w = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.spectral(x) + self.w(x)
        return self.act(y)


class FNO2D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 modes=(16,16),
                 layers: int = 4,
                 hidden_channels: int = 64):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.Sequential(*[FNOBlock(hidden_channels, modes=modes) for _ in range(layers)])
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.lift(x)
        x = self.blocks(x)
        x = self.proj(x)
        return x
