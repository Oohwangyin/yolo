# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""YOLO-friendly selective rank-aware attention modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("SeRankLite",)


class SeRankLite(nn.Module):
    """Lightweight Top-K rank-aware channel attention for YOLO feature maps.

    The module adapts SeRankDet's "pick of the bunch" idea to detection neck
    features: each channel is described by its most salient spatial responses,
    then channel-channel attention is computed from those sparse descriptors and
    applied back to the full feature map. A zero-initialized residual gate keeps
    the initial behavior close to the original YOLO feature.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        topk: int = 256,
        reduction: int = 4,
        use_pos: bool = True,
        shortcut: bool = True,
    ):
        """Initialize SeRankLite.

        Args:
            c1: Input channels.
            c2: Output channels.
            topk: Number of salient spatial responses kept per channel.
            reduction: Reduction ratio for the Top-K descriptor projection.
            use_pos: Add a small coordinate embedding to selected responses.
            shortcut: Add residual gated attention to the projected input.
        """
        super().__init__()
        if topk < 1:
            raise ValueError("SeRankLite topk must be >= 1.")
        if reduction < 1:
            raise ValueError("SeRankLite reduction must be >= 1.")

        self.topk = int(topk)
        self.use_pos = bool(use_pos)
        self.shortcut = bool(shortcut)
        self.cv1 = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        attn_dim = max(16, min(self.topk, c2 // reduction))
        self.norm = nn.LayerNorm(self.topk)
        self.q = nn.Linear(self.topk, attn_dim, bias=False)
        self.k = nn.Linear(self.topk, attn_dim, bias=False)
        self.scale = attn_dim**-0.5

        if self.use_pos:
            pos_hidden = max(8, attn_dim // 2)
            self.pos_mlp = nn.Sequential(
                nn.Linear(2, pos_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(pos_hidden, 1, bias=False),
            )
        else:
            self.pos_mlp = None

        self.proj = Conv(c2, c2, 1, 1, act=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def _topk_descriptor(self, x: torch.Tensor) -> torch.Tensor:
        """Build fixed-length Top-K descriptors for each channel."""
        b, c, h, w = x.shape
        flat = x.flatten(2)
        k_eff = min(self.topk, flat.shape[-1])

        _, topk_indices = torch.topk(flat, k=k_eff, dim=-1)
        topk_indices, _ = torch.sort(topk_indices, dim=-1)
        desc = torch.gather(flat, 2, topk_indices)

        if self.pos_mlp is not None:
            dtype = desc.dtype
            y = (topk_indices // w).to(dtype)
            x_pos = (topk_indices % w).to(dtype)
            y = y / max(h - 1, 1) * 2.0 - 1.0
            x_pos = x_pos / max(w - 1, 1) * 2.0 - 1.0
            coords = torch.stack((x_pos, y), dim=-1)
            desc = desc + self.pos_mlp(coords).squeeze(-1)

        if k_eff < self.topk:
            desc = F.pad(desc, (0, self.topk - k_eff))
        return desc.view(b, c, self.topk)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhance a single-scale YOLO feature map with rank-aware channel attention."""
        x = self.cv1(x)
        b, c, h, w = x.shape

        desc = self.norm(self._topk_descriptor(x))
        q = self.q(desc)
        k = self.k(desc)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        y = torch.matmul(attn, x.flatten(2)).view(b, c, h, w)
        y = self.proj(y)
        return x + self.gamma * y if self.shortcut else y
