"""CIFAR ResNet-20 with BatchNorm and reference-style initialization.

Same structure as :mod:`continuation.models.resnet_gn`: widths 16/32/64, three
BasicBlocks per stage, **parameter-free option-A shortcuts** (stride subsample +
zero-pad), global average pooling, 64 -> num_classes classifier.  Only the
normalization and the initialization differ:

* ``BatchNorm2d`` in place of GroupNorm, affine scale 1 / bias 0, standard
  running-statistic initialization, ``momentum=0.1``;
* convolutions ``kaiming_normal_(mode="fan_in", nonlinearity="relu")``;
* classifier ``normal_(0, 0.01)`` with zero bias.

Insertion points for Gaussian filtering are the 19 main-path 3x3 convolutions
(1 stem + 2 per block x 9 blocks).  Option-A shortcuts contain no convolution, so
they are never filtered.  Ordering, with the hook on each conv's output:

    stem / first block conv:  conv -> Gaussian -> BN -> ReLU
    second block conv:        conv -> Gaussian -> BN -> add shortcut -> ReLU

This is added alongside ``resnet20_gn`` and ``resnet18_bn_cifar``; it replaces
neither.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_gn import _PadShortcut


class BasicBlockBN20(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int, shortcut: str = "A"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch, momentum=0.1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, momentum=0.1)

        if stride == 1 and in_ch == out_ch:
            self.shortcut = nn.Identity()
        elif shortcut == "A":
            self.shortcut = _PadShortcut(stride, out_ch - in_ch)
        else:
            raise ValueError("only option-A shortcuts are used here, got %r" % (shortcut,))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet20BNCifar(nn.Module):
    def __init__(self, num_classes: int = 10, num_blocks: int = 3, width: int = 16):
        super().__init__()
        widths = [width, width * 2, width * 4]
        self.conv1 = nn.Conv2d(3, widths[0], 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0], momentum=0.1)
        layers, in_ch = [], widths[0]
        for stage, out_ch in enumerate(widths):
            for b in range(num_blocks):
                layers.append(BasicBlockBN20(in_ch, out_ch,
                                             2 if (stage > 0 and b == 0) else 1))
                in_ch = out_ch
        self.blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(in_ch, num_classes)
        self.num_classes = num_classes
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.blocks(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def resnet20_bn_cifar(num_classes: int = 10, **_ignored) -> ResNet20BNCifar:
    """Ignores the GroupNorm-specific ModelConfig fields."""
    return ResNet20BNCifar(num_classes=num_classes)
