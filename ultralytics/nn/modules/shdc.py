# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""YOLO-friendly SHDCBlock adapted from DyGLNet."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("SHDCBlock",)


class DyT(nn.Module):
    """Dynamic Tanh normalization used by DyGLNet."""

    def __init__(self, c: int, alpha: float = 0.5):
        """Initialize learnable channel-wise Dynamic Tanh parameters."""
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha)
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Dynamic Tanh to a BCHW feature map."""
        x = torch.tanh(self.alpha * x)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SEBlock(nn.Module):
    """Small squeeze-excitation block without external timm dependency."""

    def __init__(self, c: int, reduction: int = 4):
        """Initialize squeeze-excitation with a conservative hidden width."""
        super().__init__()
        c_mid = max(c // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, c_mid, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, c, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel reweighting."""
        return x * self.fc(self.pool(x))


class DilatedDWBranch(nn.Module):
    """Multi-scale depthwise dilated convolution branch."""

    def __init__(self, c: int, dilations: tuple[int, ...] = (1, 2, 3), use_se: bool = True):
        """Initialize parallel depthwise branches with residual fusion."""
        super().__init__()
        self.branches = nn.ModuleList(Conv(c, c, 3, 1, g=c, d=d) for d in dilations)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)
        self.se = SEBlock(c) if use_se else nn.Identity()
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse multi-scale local context."""
        y = self.act(self.bn(x + sum(branch(x) for branch in self.branches)))
        y = self.se(y)
        return y + self.ffn(y)


class SHDCBlock(nn.Module):
    """Single-head attention and dilated depthwise convolution block for YOLO.

    This keeps DyGLNet's global-local split while making the module safe for
    YOLO detection backbones/necks: the YAML parser supplies c1/c2, the output
    can preserve channels, and attention is capped for accidental high-resolution
    placements. Use it as an in-scale enhancer, not as a replacement for FAFM's
    cross-scale feature fusion.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        attn_ratio: float = 0.25,
        qk_ratio: float = 0.25,
        dilations=(1, 2, 3),
        shortcut: bool = True,
        max_attn_tokens: int = 1024,
        use_se: bool = True,
    ):
        """Initialize a YOLO-compatible SHDCBlock.

        Args:
            c1: Input channels.
            c2: Output channels.
            attn_ratio: Fraction of channels assigned to the attention branch.
            qk_ratio: Query/key width relative to attention-branch channels.
            dilations: Dilation rates for the local depthwise branch.
            shortcut: Add an output residual when input and output channels match.
            max_attn_tokens: Cap spatial attention tokens by pooling if needed.
            use_se: Use a lightweight SE block on the local branch.
        """
        super().__init__()
        if not 0 < attn_ratio < 1:
            raise ValueError("SHDCBlock attn_ratio must be in (0, 1).")
        c_attn = max(8, int(c2 * attn_ratio))
        c_attn = min(c_attn, c2 - 8)
        c_local = c2 - c_attn
        qk_dim = max(8, int(c_attn * qk_ratio))
        qk_dim = min(qk_dim, c_attn)

        self.cv1 = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.shortcut = shortcut and c1 == c2
        self.c_attn = c_attn
        self.c_local = c_local
        self.qk_dim = qk_dim
        self.scale = qk_dim**-0.5
        self.max_attn_tokens = int(max_attn_tokens)

        self.pre_norm = DyT(c_attn)
        self.qkv = Conv(c_attn, qk_dim * 2 + c_attn, 1, 1, act=False)
        self.local = DilatedDWBranch(c_local, tuple(dilations), use_se=use_se)
        self.proj = Conv(c2, c2, 1, 1)

    def _attention_core(self, x: torch.Tensor) -> torch.Tensor:
        """Apply original DyGLNet-style single-head spatial attention."""
        b, _, h, w = x.shape
        q, k, v = self.qkv(self.pre_norm(x)).split((self.qk_dim, self.qk_dim, self.c_attn), dim=1)
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2)
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        y = v.flatten(2) @ attn.transpose(1, 2)
        return y.view(b, self.c_attn, h, w)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        """Run attention directly or on a pooled map when resolution is too high."""
        _, _, h, w = x.shape
        tokens = h * w
        if self.max_attn_tokens > 0 and tokens > self.max_attn_tokens:
            side = max(1, int(math.sqrt(self.max_attn_tokens)))
            pooled = F.adaptive_avg_pool2d(x, (side, side))
            return F.interpolate(self._attention_core(pooled), size=(h, w), mode="bilinear", align_corners=False)
        return self._attention_core(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhance a single-scale YOLO feature map with global-local context."""
        y = self.cv1(x)
        x_attn, x_local = y.split((self.c_attn, self.c_local), dim=1)
        y = self.proj(torch.cat((self._attention(x_attn), self.local(x_local)), dim=1))
        return x + y if self.shortcut else y
