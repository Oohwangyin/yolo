# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Receptive-field attention convolution modules."""

import torch
import torch.nn as nn

from .conv import Conv, autopad

__all__ = ("RFAConv",)


class RFAConv(nn.Module):
    """Group-convolution RFAConv adapted to the YOLO Conv interface."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize RFAConv with Conv-compatible arguments.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Receptive-field/convolution kernel size.
            s (int): Stride used when generating receptive-field features.
            p (int, optional): Padding for receptive-field generation.
            g (int): Groups for the final convolution.
            d (int): Dilation for receptive-field generation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        if not isinstance(k, int):
            raise TypeError("RFAConv currently expects an integer kernel size.")
        if k % 2 == 0:
            raise ValueError("RFAConv expects an odd kernel size.")

        self.kernel_size = k
        padding = autopad(k, p, d)
        pool_padding = k // 2 if p is None else p

        self.get_weight = nn.Sequential(
            nn.AvgPool2d(kernel_size=k, padding=pool_padding, stride=s),
            nn.Conv2d(c1, c1 * k * k, kernel_size=1, groups=c1, bias=False),
        )
        self.generate_feature = nn.Sequential(
            nn.Conv2d(c1, c1 * k * k, kernel_size=k, padding=padding, stride=s, groups=c1, dilation=d, bias=False),
            nn.BatchNorm2d(c1 * k * k),
            Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity(),
        )
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=k, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply receptive-field attention and aggregate each local field."""
        b, c = x.shape[:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weight = weight.view(b, c, self.kernel_size * self.kernel_size, h, w).softmax(2)
        feature = self.generate_feature(x).view(b, c, self.kernel_size * self.kernel_size, h, w)
        data = feature * weight
        data = data.view(b, c, self.kernel_size, self.kernel_size, h, w)
        data = data.permute(0, 1, 4, 2, 5, 3).contiguous().view(b, c, h * self.kernel_size, w * self.kernel_size)
        return self.act(self.bn(self.conv(data)))

    def forward_fuse(self, x):
        """RFAConv keeps dynamic attention, so fused inference uses the regular forward path."""
        return self.forward(x)
