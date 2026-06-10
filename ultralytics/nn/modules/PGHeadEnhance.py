# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""
Positional Guidance Head (PGM) adapted from PG-DRFNet.

This module keeps the PGHead-style 3x3 convolutional position predictor and
uses it as a residual spatial feature enhancer. During forward it also caches
the pre-sigmoid position logits, allowing the detection loss to apply an
auxiliary small-object position-map focal loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("PGHeadEnhance",)


class PGHeadEnhance(nn.Module):
    """PGHead-style spatial attention gate for YOLOv8.

    Args:
        c1 (int): Input channels, auto-filled from YAML parse_model.
        conv_channels (int | None): Intermediate conv channels. Defaults to c1.
        num_convs (int): Number of 3x3 Conv layers before prediction.

    Shape:
        Input:  (B, C, H, W)
        Output: (B, C, H, W)
    """

    def __init__(self, c1, conv_channels=None, num_convs=2):
        super().__init__()
        if conv_channels is None:
            conv_channels = c1
        self.num_convs = num_convs
        self.guidance_logits = None

        self.subnet = nn.ModuleList()
        ch = c1
        for _ in range(num_convs):
            self.subnet.append(nn.Conv2d(ch, conv_channels, 3, 1, 1))
            ch = conv_channels

        self.pred_net = nn.Conv2d(ch, 1, 3, 1, 1)

        for layer in self.subnet:
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        nn.init.xavier_normal_(self.pred_net.weight)
        nn.init.constant_(self.pred_net.bias, 0)

    def forward(self, x):
        """Enhance features and cache the logits used for position supervision."""
        feat = x
        for conv in self.subnet:
            feat = F.relu(conv(feat))
        self.guidance_logits = self.pred_net(feat)
        attn = torch.sigmoid(self.guidance_logits)
        return x * (1.0 + attn)
