"""CIFAR-style ResNet-20 with GroupNorm.

Exact architecture
------------------
Following He et al.'s CIFAR ResNet, with every BatchNorm replaced by GroupNorm:

* stem: ``3x3`` conv, ``3 -> 16`` channels, stride 1, no bias, then GN + ReLU;
* three stages of ``n = 3`` basic blocks each, widths ``16, 32, 64``, first block
  of stages 2 and 3 using stride 2 (so ``32x32 -> 16x16 -> 8x8``);
* each basic block: ``conv3x3 -> GN -> ReLU -> conv3x3 -> GN``, added to the
  shortcut, then ReLU;
* global average pooling over ``8x8``, then a linear layer to ``num_classes``
  (10 for CIFAR-10, 100 for CIFAR-100);
* total depth ``6n + 2 = 20`` weight layers; ~0.27 M parameters.

Shortcut options: ``"A"`` (default, as in the original CIFAR ResNet) is
parameter-free -- stride-2 subsampling plus zero-padding of the new channels;
``"B"`` uses a ``1x1`` stride-2 convolution followed by GroupNorm.

GroupNorm configuration
-----------------------
``num_groups = channels // channels_per_group`` with ``channels_per_group = 8``
by default, i.e. 2 / 4 / 8 groups at widths 16 / 32 / 64, with the standard
per-channel affine parameters and ``eps = 1e-5``.  ``channels_per_group`` must
divide every stage width.

GroupNorm is used because it carries **no running statistics**, so a run is not
contaminated by BatchNorm buffers estimated under one input distribution and
reused under another.  It does *not* remove the effect of the input
transformation itself.

The network is otherwise unmodified: there is **no** feature-map smoothing and
no activation homotopy here.  The feature-smoothing intervention of *Curriculum
by Smoothing* is a different method and must stay a separate, controlled
experiment rather than being combined with this input-only baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int, channels_per_group: int) -> nn.GroupNorm:
    if channels % channels_per_group != 0:
        raise ValueError("channels_per_group=%d does not divide channels=%d"
                         % (channels_per_group, channels))
    return nn.GroupNorm(channels // channels_per_group, channels, eps=1e-5, affine=True)


class _PadShortcut(nn.Module):
    """Option-A shortcut: stride subsample + zero-pad the extra channels."""

    def __init__(self, stride: int, pad: int):
        super().__init__()
        self.stride, self.pad = stride, pad

    def forward(self, x):
        x = x[:, :, ::self.stride, ::self.stride]
        return F.pad(x, (0, 0, 0, 0, self.pad // 2, self.pad - self.pad // 2))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int,
                 channels_per_group: int, shortcut: str):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn1 = _gn(out_ch, channels_per_group)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.gn2 = _gn(out_ch, channels_per_group)

        if stride == 1 and in_ch == out_ch:
            self.shortcut = nn.Identity()
        elif shortcut == "A":
            self.shortcut = _PadShortcut(stride, out_ch - in_ch)
        elif shortcut == "B":
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                _gn(out_ch, channels_per_group),
            )
        else:
            raise ValueError("shortcut must be 'A' or 'B', got %r" % (shortcut,))

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)), inplace=True)
        out = self.gn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNetCifarGN(nn.Module):
    def __init__(self, num_classes: int = 10, num_blocks: int = 3, width: int = 16,
                 channels_per_group: int = 8, shortcut: str = "A",
                 zero_init_residual: bool = False):
        super().__init__()
        widths = [width, width * 2, width * 4]
        self.conv1 = nn.Conv2d(3, widths[0], 3, stride=1, padding=1, bias=False)
        self.gn1 = _gn(widths[0], channels_per_group)

        layers = []
        in_ch = widths[0]
        for stage, out_ch in enumerate(widths):
            for b in range(num_blocks):
                stride = 2 if (stage > 0 and b == 0) else 1
                layers.append(BasicBlock(in_ch, out_ch, stride, channels_per_group, shortcut))
                in_ch = out_ch
        self.blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(in_ch, num_classes)
        self.num_classes = num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.zeros_(m.gn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.gn1(self.conv1(x)), inplace=True)
        out = self.blocks(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def resnet20_gn(num_classes: int = 10, **kwargs) -> ResNetCifarGN:
    return ResNetCifarGN(num_classes=num_classes, num_blocks=3, **kwargs)
