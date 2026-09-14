"""CIFAR ResNet-20 with GroupNorm -- the mechanism control for BatchNorm.

Identical to ``resnet20_bn`` in every respect except the normalisation layer:
same widths 16/32/64, same three BasicBlocks per stage, same option-A shortcuts,
same initialization rule, same forward, and the **same site map**, which never
names a normalisation layer (it maps convolution outputs, block outputs and the
input of ``blocks.2``).  So the four methods attach at exactly the same tensors.

Why this model exists
---------------------
``resolution_max_b1_gaussian_conv`` filters all 19 convolution outputs *before*
normalisation.  Under BatchNorm that means BN computes its batch statistics on
blurred activations and accumulates its **running buffers** on them too, so the
network's normalisation is fitted to a distribution that the schedule then
removes.  That interaction is the leading explanation for why the combined method
behaves as it does, and on STL-10 it is what left the target path at chance for
42 epochs while the blur was active.

GroupNorm removes exactly that coupling and nothing else: it normalises each
sample over channel groups, has no batch dependence and no running statistics, so
there is no fitted state for the blur to corrupt and nothing to re-estimate when
the blur is annealed away.  The prediction, if the interaction is the mechanism:

* ``resolution_max_b1_gaussian_conv`` loses most of its advantage (it blurs
  pre-normalisation at 19 sites);
* ``gaussian_postrelu`` is less affected (it blurs *after* normalisation);
* ``resolution_max_b1`` is unaffected (it does not blur at all).

Groups
------
``groups=8`` divides all three widths evenly (16/8, 32/8, 64/8 = 2, 4, 8 channels
per group).  Wu & He's default of 32 cannot be used here: the first stage has 16
channels.  The count is a build option so it can be varied without a new module.

Evaluation note
---------------
``bn_policy`` is meaningless for this model.  GroupNorm has no running buffers
and no train/eval difference, so ``running_stats`` and ``fixed_batch_stats``
produce identical numbers; the field is still recorded for schema compatibility.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet20_bn import PadShortcut, SITE_MAP as _BN_SITE_MAP


def _norm(channels: int, groups: int) -> nn.GroupNorm:
    if channels % groups:
        raise ValueError("groups=%d does not divide %d channels" % (groups, channels))
    return nn.GroupNorm(groups, channels)


class BasicBlockGN20(nn.Module):
    """``resnet20_bn.BasicBlockBN20`` with GroupNorm; the norms are named
    ``norm1``/``norm2`` rather than ``bn1``/``bn2`` so nothing reads as a
    BatchNorm that is not one.  No site map entry names them."""

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int, groups: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = _norm(out_ch, groups)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.norm2 = _norm(out_ch, groups)
        if stride == 1 and in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = PadShortcut(stride, out_ch - in_ch)

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet20GN(nn.Module):
    def __init__(self, num_classes: int = 10, width: int = 16, in_channels: int = 3,
                 groups: int = 8):
        super().__init__()
        widths = [width, width * 2, width * 4]
        self.conv1 = nn.Conv2d(in_channels, widths[0], 3, stride=1, padding=1, bias=False)
        self.norm1 = _norm(widths[0], groups)
        layers, in_ch = [], widths[0]
        for stage, out_ch in enumerate(widths):
            for b in range(3):
                layers.append(BasicBlockGN20(in_ch, out_ch,
                                             2 if (stage > 0 and b == 0) else 1, groups))
                in_ch = out_ch
        self.blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(in_ch, num_classes)
        self.num_classes = num_classes
        self.groups = groups
        self._initialize_weights()

    def _initialize_weights(self):
        # the rule from resnet20_bn, unchanged
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.blocks(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


#: byte-for-byte the ResNet-20-BN map: it names convolutions, block outputs and
#: the input of blocks.2, never a normalisation layer, so the four methods attach
#: at exactly the same tensors in both models.
SITE_MAP = dict(_BN_SITE_MAP)


def build(num_classes: int = 10, in_channels: int = 3, **kwargs) -> ResNet20GN:
    unknown = sorted(set(kwargs) - {"width", "groups"})
    if unknown:
        raise TypeError("resnet20_gn_cifar: unknown options %s" % unknown)
    return ResNet20GN(num_classes=num_classes, in_channels=in_channels, **kwargs)
