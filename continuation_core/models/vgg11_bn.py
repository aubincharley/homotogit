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
* ``reduction`` -- ``"after_pool1"`` only.  "block1" is a ResNet-20 location and
  is **not** mapped here: ``resolution_max_b1`` still fails with
  ``UnsupportedInsertionError`` rather than being silently redirected.  A method
  that wants this point has to name it.

``after_pool1`` is the input of ``features.4``, i.e. the tensor leaving the first
max pool, and it is the closest structural analogue of ResNet-20's ``block1``:
one convolution upstream, the other seven downstream, so every later layer
genuinely sees the smaller grid.  Its **native size is half the input** (16 for a
32x32 image, where ResNet-20's block1 runs at the full 32), so a schedule here
must use ``reference_resolution=16`` and values scaled to match -- ``(8, 12, 16)``
keeps the reference ratios 1/2, 3/4, 1.

**Spatial headroom, and why the pools use ceil_mode.** Five pools on a 32x32
image leave none: 32 / 2^5 = 1, so the network already ends at 1x1.  Four of
those pools follow ``after_pool1`` (ResNet-20 has two stride-2 steps after
``block1``), and under any ``r < 16`` the tail reaches 1x1 with a pool still to
go -- which with the default floor rounding is 0x0 and raises "Output size is too
small".  ``ceil_mode=True`` maps 1 -> 1 instead.  It changes nothing about the
network as published, because every native pool input (32, 16, 8, 4, 2) is even
and ceil and floor coincide there; it only makes the reduced states expressible:

    r     features.4-6   8-13   15-20   22-27   final pool
    8         8            4      2       1         1
    12       12            6      3       2         1
    16       16            8      4       2         1   (native, exact bypass)

The last convolutions therefore run at 1x1 or 2x2 while the reduction is active,
where a 3x3 convolution and a Gaussian are both close to degenerate.  That is a
fact about this architecture on small images, not about the methods, and it
should be the first suspect if the resolution-bearing methods underperform here.
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
                # ceil_mode is a no-op for this network as published: on a 32x32
                # image the five pools see 32, 16, 8, 4, 2, all even, so ceil and
                # floor agree and the plain model is bitwise unchanged (asserted
                # in tests/test_vgg11_transfer.py).  It matters only under a
                # resolution intervention, where floor sends a 1x1 map to 0x0 and
                # max_pool2d raises "Output size is too small".  See the reduction
                # note in the module docstring.
                layers.append(nn.MaxPool2d(2, 2, ceil_mode=True))
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
    # the first "M" is at index 3, so features.4 is the convolution that receives
    # the pooled tensor; reducing its input puts 7 of the 8 convolutions
    # downstream of the reduction
    return {"conv_out": conv, "post_relu": relu,
            "reduction": {"after_pool1": ("input", "features.4")},
            "reference_resolution": None}


SITE_MAP = _site_map()


def build(num_classes: int = 10, in_channels: int = 3, **kwargs) -> VGG11BN:
    unknown = sorted(set(kwargs) - {"width"})
    if unknown:
        raise TypeError("vgg11_bn: unknown options %s" % unknown)
    return VGG11BN(num_classes=num_classes, in_channels=in_channels, **kwargs)
