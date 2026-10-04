"""
2D Fourier Neural Operator (FNO) for Kirchhoff potential prediction.

Predicts ``theta(x,y)`` from leakage-free chip thermal inputs.

Input:  [B, C_in, H, W]  (default C_in=23)
Output: [B, 1, H, W]     (theta)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2D spectral convolution retaining the lowest ``(modes1, modes2)`` Fourier modes."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def _compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # (B, in, x, y) x (in, out, x, y) -> (B, out, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(
            b,
            self.out_channels,
            h,
            w // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        m1 = min(self.modes1, h)
        m2 = min(self.modes2, w // 2 + 1)

        out_ft[:, :, :m1, :m2] = self._compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = self._compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")


class FNOBlock(nn.Module):
    """Spectral convolution + pointwise residual branch + activation."""

    def __init__(self, width: int, modes1: int, modes2: int, activation: str = "gelu"):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.activation = _get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.spectral(x) + self.pointwise(x))


class FNO2d(nn.Module):
    """
    Baseline 2D FNO.

    Pipeline
    --------
    concat(x, grid) -> lift Conv2d(C+2, width, 1)
                    -> N FNO blocks
                    -> project width -> mid -> 1
    """

    def __init__(
        self,
        in_channels: int = 23,
        out_channels: int = 1,
        width: int = 32,
        modes1: int = 8,
        modes2: int = 8,
        n_layers: int = 4,
        mlp_ratio: float = 1.0,
        activation: str = "gelu",
        use_grid: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.n_layers = n_layers
        self.use_grid = use_grid

        lift_in = in_channels + (2 if use_grid else 0)
        self.lift = nn.Conv2d(lift_in, width, kernel_size=1)

        self.blocks = nn.ModuleList(
            [FNOBlock(width, modes1, modes2, activation=activation) for _ in range(n_layers)]
        )

        mid = max(out_channels, int(width * mlp_ratio))
        self.project = nn.Sequential(
            nn.Conv2d(width, mid, kernel_size=1),
            _get_activation(activation),
            nn.Conv2d(mid, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shape ``[B, in_channels, H, W]``.

        Returns
        -------
        Tensor
            Predicted theta, shape ``[B, out_channels, H, W]``.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )

        if self.use_grid:
            x = torch.cat([x, self._get_grid(x)], dim=1)

        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        return self.project(h)

    @staticmethod
    def _get_grid(x: torch.Tensor) -> torch.Tensor:
        """Normalized spatial coordinates in ``[-1, 1]``, shape ``[B, 2, H, W]``."""
        b, _, h, w = x.shape
        device, dtype = x.device, x.dtype
        # Channel 0 varies along H (dim -2); channel 1 along W (dim -1)
        g0, g1 = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack([g0, g1], dim=0).unsqueeze(0).expand(b, 2, h, w)

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def build_fno_from_config(cfg: dict) -> FNO2d:
    """Build ``FNO2d`` from a config dict (e.g. loaded YAML ``model:`` section)."""
    m = cfg.get("model", cfg)
    return FNO2d(
        in_channels=int(m.get("in_channels", 23)),
        out_channels=int(m.get("out_channels", 1)),
        width=int(m.get("width", 32)),
        modes1=int(m.get("modes1", 8)),
        modes2=int(m.get("modes2", 8)),
        n_layers=int(m.get("n_layers", 4)),
        mlp_ratio=float(m.get("mlp_ratio", 1.0)),
        activation=str(m.get("activation", "gelu")),
        use_grid=bool(m.get("use_grid", True)),
    )
