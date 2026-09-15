"""CIFAR ResNet-20-BN with a configurable activation -- the ReLU control.

``resnet20_bn`` with ``F.relu`` replaced by a choice of ``relu``, ``gelu`` or
``silu``, and **nothing else changed**: same widths, same blocks, same option-A
shortcuts, same BatchNorm, same initialization rule, same module names, same
site map.

Why the comparison is unusually tight
-------------------------------------
Activations here are functional, not modules, so they carry no parameters and the
state dict is **identical in keys and shapes** to ``resnet20_bn``'s.  That means
these runs can load ``assets/cifar10_resnet20bn`` directly -- the pinned initial
weights *and* the pinned data order of the original reference campaign -- with no
new asset set at all.  The only difference between a run here and the
corresponding reference run is which function is applied after each BatchNorm.

What it tests
-------------
The ``post_relu`` placement is named after ReLU, and ReLU is the reason the
Gaussian interacts with it the way it does: ReLU is exactly zero on half its
domain, so blurring before it mixes dead and live units, and blurring after it
smooths a non-negative, sparse signal.  GELU and SiLU are smooth, non-monotone
near zero and never exactly zero, so both of those properties go away.  If the
methods depend on the rectifier's sparsity, the gaps should shrink here.

The site map is activation-agnostic, which is why it transfers unchanged: it
names ``blocks`` and ``blocks.<i>`` (whose outputs are post-activation tensors by
construction) and the convolutions, never an activation module.  So all four
methods attach at exactly the same tensors whichever function is chosen.

Initialization is deliberately not retuned: ``kaiming_normal_(mode="fan_in",
nonlinearity="relu")`` is kept for every activation, because the gain is part of
the frozen recipe and changing it alongside the activation would confound the
comparison.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet20_bn import PadShortcut, SITE_MAP as _BN_SITE_MAP

ACTIVATIONS = {"relu": F.relu, "gelu": F.gelu, "silu": F.silu}


def _act(name: str):
    if name not in ACTIVATIONS:
        raise ValueError("unknown activation %r; available: %s"
                         % (name, sorted(ACTIVATIONS)))
    return ACTIVATIONS[name]


class BasicBlockAct20(nn.Module):
    """``resnet20_bn.BasicBlockBN20`` with the activation injected.

    Module names are unchanged (``conv1``, ``bn1``, ``conv2``, ``bn2``,
    ``shortcut``) so the state dict stays interchangeable with the BatchNorm
    model's and the site map keeps resolving.
    """

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int, activation: str):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch, momentum=0.1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, momentum=0.1)
        if stride == 1 and in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = PadShortcut(stride, out_ch - in_ch)
        self.activation = activation
        self._f = _act(activation)

    def forward(self, x):
        out = self._f(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self._f(out)


class ResNet20Act(nn.Module):
    def __init__(self, num_classes: int = 10, width: int = 16, in_channels: int = 3,
                 activation: str = "gelu"):
        super().__init__()
        self._f = _act(activation)
        self.activation = activation
        widths = [width, width * 2, width * 4]
        self.conv1 = nn.Conv2d(in_channels, widths[0], 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0], momentum=0.1)
        layers, in_ch = [], widths[0]
        for stage, out_ch in enumerate(widths):
            for b in range(3):
                layers.append(BasicBlockAct20(in_ch, out_ch,
                                              2 if (stage > 0 and b == 0) else 1,
                                              activation))
                in_ch = out_ch
        self.blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(in_ch, num_classes)
        self.num_classes = num_classes
        self._initialize_weights()

    def _initialize_weights(self):
        # the rule from resnet20_bn, unchanged for every activation on purpose
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
        out = self._f(self.bn1(self.conv1(x)))
        out = self.blocks(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


#: the ResNet-20-BN map verbatim; it names convolutions, block outputs and the
#: input of blocks.2, never an activation, so it is activation-agnostic
SITE_MAP = dict(_BN_SITE_MAP)


def build(num_classes: int = 10, in_channels: int = 3, **kwargs) -> ResNet20Act:
    unknown = sorted(set(kwargs) - {"width", "activation"})
    if unknown:
        raise TypeError("resnet20_act_cifar: unknown options %s" % unknown)
    return ResNet20Act(num_classes=num_classes, in_channels=in_channels, **kwargs)
