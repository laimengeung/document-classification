"""M1: a small CNN trained from scratch, the baseline of the comparison (PRD 5.3).

Architecture for a 1 x 224 x 224 input (shapes are C x H x W):

    Block 1  [Conv3x3(1->32)   BN ReLU] [Conv3x3(32->32)   BN ReLU] MaxPool2   ->  32 x 112 x 112
    Block 2  [Conv3x3(32->64)  BN ReLU] [Conv3x3(64->64)   BN ReLU] MaxPool2   ->  64 x  56 x  56
    Block 3  [Conv3x3(64->128) BN ReLU] [Conv3x3(128->128) BN ReLU] MaxPool2   -> 128 x  28 x  28
    Block 4  [Conv3x3(128->256)BN ReLU] [Conv3x3(256->256) BN ReLU] MaxPool2   -> 256 x  14 x  14
    Block 5  [Conv3x3(256->256)BN ReLU] [Conv3x3(256->256) BN ReLU] MaxPool2   -> 256 x   7 x   7
    AdaptiveAvgPool(4x4)                                                       -> 256 x   4 x   4
    Flatten -> Dropout -> Linear(4096->256) ReLU -> Dropout -> Linear(256->16)

Design notes for the report:
    * Two 3x3 convs per block (VGG style) see a 5x5 area with fewer
      parameters than one 5x5 conv, plus an extra non-linearity.
    * BatchNorm after every conv makes training from scratch stable at lr 1e-3.
    * The head pools to a **4x4 grid** instead of a single global average.
      Document classes differ in *where* things are (letterhead at the top,
      totals at the bottom right, form fields in a grid), and a 4x4 grid keeps
      that coarse position information. Global average pooling would discard it.
    * About 3.4M parameters, roughly 7x fewer than ResNet-50.
"""

from __future__ import annotations

import torch
from torch import nn


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two Conv3x3-BN-ReLU layers followed by 2x2 max pooling (halves H and W)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class CustomCNN(nn.Module):
    """Five conv blocks + a spatially aware classifier head (see module docstring)."""

    CHANNELS = (32, 64, 128, 256, 256)
    POOL_GRID = 4
    HIDDEN = 256

    def __init__(self, num_classes: int = 16, in_chans: int = 1, dropout: float = 0.3):
        super().__init__()
        blocks, channels = [], in_chans
        for out_channels in self.CHANNELS:
            blocks.append(conv_block(channels, out_channels))
            channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(self.POOL_GRID)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels * self.POOL_GRID ** 2, self.HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.HIDDEN, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """He initialisation for convs (suits ReLU); standard init for BN and linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))
