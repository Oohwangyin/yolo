# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Feature enhancement modules adapted from FFCA-YOLO for YOLOv8."""

import torch
import torch.nn as nn

from .conv import Conv

__all__ = ("FEM",)


class FEM(nn.Module):
    """Multi-branch feature enhancement module for YOLOv8 lateral features.

    The original FFCA-YOLO FEM was designed around YOLOv5 C3 stages. This
    adaptation keeps the residual multi-scale context idea, but uses Ultralytics
    Conv blocks and SiLU activation so it follows YOLOv8 feature statistics.
    """

    def __init__(self, c1, c2, reduction=8, scale=0.1, act=True):
        """Initialize FEM.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            reduction (int): Channel reduction ratio inside context branches.
            scale (float): Residual branch scaling factor.
            act (bool | nn.Module): Output activation.
        """
        super().__init__()
        hidden = max(c2 // reduction, 4)
        asymmetric = max((hidden // 2) * 3, hidden)
        branch_channels = 2 * hidden

        self.branch0 = nn.Sequential(
            Conv(c1, branch_channels, 1, 1),
            Conv(branch_channels, branch_channels, 3, 1, act=False),
        )
        self.branch1 = nn.Sequential(
            Conv(c1, hidden, 1, 1),
            Conv(hidden, asymmetric, (1, 3), 1),
            Conv(asymmetric, branch_channels, (3, 1), 1),
            Conv(branch_channels, branch_channels, 3, 1, d=5, act=False),
        )
        self.branch2 = nn.Sequential(
            Conv(c1, hidden, 1, 1),
            Conv(hidden, asymmetric, (3, 1), 1),
            Conv(asymmetric, branch_channels, (1, 3), 1),
            Conv(branch_channels, branch_channels, 3, 1, d=5, act=False),
        )
        self.fuse = Conv(branch_channels * 3, c2, 1, 1, act=False)
        self.shortcut = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1, act=False)
        self.scale = scale
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Enhance lateral feature maps with residual multi-scale context."""
        y = torch.cat((self.branch0(x), self.branch1(x), self.branch2(x)), 1)
        return self.act(self.fuse(y) * self.scale + self.shortcut(x))
