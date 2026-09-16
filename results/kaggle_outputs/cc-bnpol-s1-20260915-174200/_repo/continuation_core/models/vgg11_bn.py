"""VGG-11 with BatchNorm for small images -- the non-ResNet example.

Ported from ``src/cifarbase/model.py`` on branch ``tests/homotopy_archi``
(commit f81ff55): the eight 3x3 convolutions of vgg11_bn, five 2x2 max pools,
global average pooling and one linear layer.  The activation homotopy of that
file is not ported; ReLUs are plain modules so they can carry hooks.

**Unvalidated here**: no run of any retained method exists for this model.

Site map
--------
* ``conv_out``  -- the 8 convolution outputs (before BN), in forward order;
* ``post_relu`` -- the 8 ReLU outputs, in forward order;
* ``reduction`` -- **empty**.  "block1" is a ResNet-20 location; a VGG has no
  residual blocks, so ``resolution_max_b1`` fails with
  ``UnsupportedInsertionError`` instead of being mapped to an arbitrary pool.
  To study a resolution intervention here, add an explicitly named point
  (e.g. ``"after_pool1": ("input", "features.4")``) and a new method preset.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PLAN = (64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M")


class VGG11BN(nn.Module):
    def __init__(self, num_classes: int = 10, width: int = 64, in_channels: int = 3):
        super().__init__()
        scale = width / 64
        layers, in_ch = [], in_channels
        for item in PLAN:
            if item == "M":
                layers.append(nn.MaxPool2d(2, 2))
                continue
            out_ch = max(1, round(item * scale))
            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                       nn.BatchNorm2d(out_ch), nn.ReLU()]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(in_ch, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(F.adaptive_avg_pool2d(self.features(x), 1).flatten(1))


def _site_map() -> dict:
    conv, relu, i = [], [], 0
    for item in PLAN:
        if item == "M":
            i += 1
            continue
        conv.append("features.%d" % i)
        relu.append(("output", "features.%d" % (i + 2)))
        i += 3
    return {"conv_out": conv, "post_relu": relu, "reduction": {},
            "reference_resolution": None}


SITE_MAP = _site_map()


def build(num_classes: int = 10, in_channels: int = 3, **kwargs) -> VGG11BN:
    unknown = sorted(set(kwargs) - {"width"})
    if unknown:
        raise TypeError("vgg11_bn: unknown options %s" % unknown)
    return VGG11BN(num_classes=num_classes, in_channels=in_channels, **kwargs)
