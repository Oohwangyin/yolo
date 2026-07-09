# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""CENet-inspired detail and context enhancement blocks for YOLO neck features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv, LightConv

__all__ = ("CENetBlock", "CENetLiteBlock")


class CENetBlock(nn.Module):
    """Lightweight CENet-style feature enhancement block.

    The block keeps the transferable parts of CENet for detection: feature edge
    amplification from DSEB and lightweight channel/multi-scale context
    calibration from CFAM. It avoids segmentation-only dependencies such as
    PVT backbones, non-local operators, MONAI, Apex, and TIMM.
    """

    def __init__(
        self,
        c1: int,
        c2: int | None = None,
        edge_gain: float = 0.10,
        reduction: int = 4,
        dilations: tuple[int, ...] = (1, 3, 5),
        shortcut: bool = True,
    ):
        """Initialize the CENet-inspired block.

        Args:
            c1 (int): Input channels.
            c2 (int, optional): Output channels. Defaults to ``c1``.
            edge_gain (float): Initial strength of the FEA-style edge residual.
            reduction (int): Channel reduction ratio for context calibration.
            dilations (tuple[int, ...]): Multi-scale depthwise context dilations.
            shortcut (bool): Add input shortcut when channel counts match.
        """
        super().__init__()
        c2 = c1 if c2 is None else c2
        c_mid = max(c2 // reduction, 16)
        self.shortcut = shortcut and c1 == c2
        self.edge_gain = nn.Parameter(torch.tensor(float(edge_gain)))

        self.in_proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.edge_proj = Conv(c2, c2, 3, 1)

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c_mid, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, c2, 1, bias=True),
        )
        self.context = nn.ModuleList(DWConv(c2, c2, 3, 1, d=d) for d in dilations)
        self.context_proj = Conv(c2 * len(dilations), c2, 1, 1)
        self.out = Conv(c2, c2, 3, 1)

    @staticmethod
    def _edge_residual(x: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale feature edge residuals in the spirit of CENet FEA."""
        edges = []
        for scale in (0.5, 0.25):
            pooled = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
            restored = F.interpolate(pooled, size=x.shape[2:], mode="bilinear", align_corners=False)
            edges.append((x - restored).abs())
        return 0.5 * (edges[0] + edges[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhance a single YOLO neck feature map."""
        identity = x
        x = self.in_proj(x)
        edge = self.edge_proj(self._edge_residual(x))
        x = x + self.edge_gain.to(dtype=x.dtype, device=x.device) * edge

        channel_weight = torch.sigmoid(self.channel_attn(x))
        context = self.context_proj(torch.cat([branch(x) for branch in self.context], 1))
        out = self.out(context * channel_weight)
        return out + identity if self.shortcut else out


class CENetLiteBlock(nn.Module):
    """Compute-friendly CENet block for high-resolution YOLO neck features.

    This variant preserves the CENetBlock data path used by FAFM-Lite-CENet
    but replaces the expensive full 3x3 projections with depthwise/light
    convolutions. It is meant for the P2/P3 neck features where spatial
    resolution dominates FLOPs.
    """

    def __init__(
        self,
        c1: int,
        c2: int | None = None,
        edge_gain: float = 0.10,
        reduction: int = 4,
        dilations: tuple[int, ...] = (1, 3, 5),
        shortcut: bool = True,
    ):
        """Initialize the lightweight CENet block."""
        super().__init__()
        c2 = c1 if c2 is None else c2
        c_mid = max(c2 // reduction, 16)
        self.shortcut = shortcut and c1 == c2
        self.edge_gain = nn.Parameter(torch.tensor(float(edge_gain)))

        self.in_proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.edge_proj = DWConv(c2, c2, 3, 1)

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c_mid, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, c2, 1, bias=True),
        )
        self.context = nn.ModuleList(DWConv(c2, c2, 3, 1, d=d) for d in dilations)
        self.context_proj = Conv(c2 * len(dilations), c2, 1, 1)
        self.out = LightConv(c2, c2, 3, act=nn.SiLU())

    @staticmethod
    def _edge_residual(x: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale feature edge residuals in the spirit of CENet FEA."""
        edges = []
        for scale in (0.5, 0.25):
            pooled = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
            restored = F.interpolate(pooled, size=x.shape[2:], mode="bilinear", align_corners=False)
            edges.append((x - restored).abs())
        return 0.5 * (edges[0] + edges[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhance a single YOLO neck feature map with reduced convolution cost."""
        identity = x
        x = self.in_proj(x)
        edge = self.edge_proj(self._edge_residual(x))
        x = x + self.edge_gain.to(dtype=x.dtype, device=x.device) * edge

        channel_weight = torch.sigmoid(self.channel_attn(x))
        context = self.context_proj(torch.cat([branch(x) for branch in self.context], 1))
        out = self.out(context * channel_weight)
        return out + identity if self.shortcut else out
