# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Lightweight boundary-aware feature attention fusion modules for YOLO."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv

__all__ = ("FAFM",)


class FAFM(nn.Module):
    """Lightweight Feature Attention Fusion Module for two YOLO neck features.

    This adapts BaAFN's feature alignment and attention fusion idea to detection:
    a high-level feature is projected, upsampled, lightly flow-aligned to a
    low-level feature, and then fused with local and global attention weights.
    It intentionally avoids DCN, NATTEN, and KAN dependencies for a first-pass
    YOLO small-object variant.
    """

    def __init__(self, c_low, c_high, c_out=None, reduction=4, offset_scale=2.0, shortcut=True):
        """Initialize FAFM.

        Args:
            c_low (int): Channels of the lower-level, higher-resolution feature.
            c_high (int): Channels of the higher-level, lower-resolution feature.
            c_out (int, optional): Output channels. Defaults to c_low.
            reduction (int): Reduction ratio for attention hidden channels.
            offset_scale (float): Maximum learned flow offset in pixels.
            shortcut (bool): Add a low-feature shortcut when channels match.
        """
        super().__init__()
        c_out = c_low if c_out is None else c_out
        c_mid = max(c_out // reduction, 16)
        self.offset_scale = float(offset_scale)
        self.shortcut = shortcut and c_low == c_out

        self.low_proj = Conv(c_low, c_out, 1, 1)
        self.high_proj = Conv(c_high, c_out, 1, 1)

        self.offset = nn.Sequential(
            Conv(c_out * 2, c_mid, 3, 1),
            nn.Conv2d(c_mid, 2, 3, padding=1),
        )
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)

        self.local_attn = nn.Sequential(
            DWConv(c_out, c_out, 3, 1),
            Conv(c_out, c_out, 1, 1, act=False),
        )
        self.global_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_out, c_mid, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 1, bias=True),
        )
        self.out = Conv(c_out, c_out, 3, 1)

    @staticmethod
    def _make_base_grid(x):
        """Create a normalized sampling grid matching BCHW tensor x."""
        b, _, h, w = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        return torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(b, h, w, 2)

    def _align(self, high, low):
        """Align high-level feature to low-level coordinates with learned flow."""
        flow = torch.tanh(self.offset(torch.cat((low, high), 1))) * self.offset_scale
        _, _, h, w = high.shape
        norm = high.new_tensor([max(w - 1, 1) / 2, max(h - 1, 1) / 2]).view(1, 2, 1, 1)
        grid = self._make_base_grid(high) + (flow / norm).permute(0, 2, 3, 1)
        return F.grid_sample(high, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, x):
        """Fuse [low_feature, high_feature] into an aligned attention-weighted output."""
        low, high = x
        low = self.low_proj(low)
        high = self.high_proj(high)
        high = F.interpolate(high, size=low.shape[2:], mode="bilinear", align_corners=False)
        high = self._align(high, low)

        merged = low + high
        weight = torch.sigmoid(self.local_attn(merged) + self.global_attn(merged))
        fused = high * weight + low * (1.0 - weight)
        fused = self.out(fused)
        return fused + low if self.shortcut else fused
