"""Reference-style CIFAR ResNet-18 with BatchNorm and projection shortcuts.

Mirrors the architecture in ``pairlab/CBS`` at commit ``5f62e7da`` (``resnet.py``):

* CIFAR stem: ``3x3`` stride-1 conv, 3 -> 64, no bias, **no ImageNet max-pool**;
* four stages of two ``BasicBlock`` s, widths 64 / 128 / 256 / 512, stride 1/2/2/2;
* ``BatchNorm2d`` throughout;
* learned ``1x1`` conv + BatchNorm **projection shortcuts** wherever the stride or
  the channel count changes;
* global average pooling, then a ``512 -> num_classes`` linear classifier.

Initialization follows the reference: convolutions ``kaiming_normal_`` with
``mode="fan_in", nonlinearity="relu"``; BatchNorm scale 1, bias 0; classifier
``normal_(0, 0.01)`` with zero bias.

This is added **alongside** ``resnet20_gn``; it does not replace it.  Only the
3x3 main-path convolutions carry Gaussian insertions -- the ``1x1`` projection
shortcuts are excluded by the ``kernel_size == (3, 3)`` filter used to attach
hooks, so a ResNet-18 has 17 insertion points (1 stem + 2 per block x 8 blocks).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlockBN(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        # conv -> [Gaussian hook] -> BN -> ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        # conv -> [Gaussian hook] -> BN -> add shortcut -> ReLU
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18BNCifar(nn.Module):
    def __init__(self, num_classes: int = 10, num_blocks=(2, 2, 2, 2), width: int = 64):
        super().__init__()
        self.in_planes = width
        self.conv1 = nn.Conv2d(3, width, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.layer1 = self._make_layer(width, num_blocks[0], 1)
        self.layer2 = self._make_layer(width * 2, num_blocks[1], 2)
        self.layer3 = self._make_layer(width * 4, num_blocks[2], 2)
        self.layer4 = self._make_layer(width * 8, num_blocks[3], 2)
        self.linear = nn.Linear(width * 8 * BasicBlockBN.expansion, num_classes)
        self.num_classes = num_classes
        self._initialize_weights()

    def _make_layer(self, planes, blocks, stride):
        layers = []
        for s in [stride] + [1] * (blocks - 1):
            layers.append(BasicBlockBN(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlockBN.expansion
        return nn.Sequential(*layers)

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
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


def resnet18_bn_cifar(num_classes: int = 10, **_ignored) -> ResNet18BNCifar:
    """Ignores the ResNet-20/GroupNorm-specific ModelConfig fields."""
    return ResNet18BNCifar(num_classes=num_classes)
