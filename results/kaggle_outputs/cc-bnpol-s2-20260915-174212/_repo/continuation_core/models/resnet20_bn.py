"""CIFAR ResNet-20 with BatchNorm and option-A shortcuts.

Numerically identical to the benchmark's ``continuation.models.resnet20_bn``:
same module names (so the pinned initial states load strictly), same
initialization, same forward.  Widths 16/32/64, three BasicBlocks per stage.

Forward, with the module names the site maps refer to::

    x -> conv1 -> bn1 -> relu                                   (stem)
      -> blocks[0..8]                                           (nn.Sequential)
           block: conv1 -> bn1 -> relu -> conv2 -> bn2
                  -> (+ shortcut(x)) -> relu
      -> adaptive_avg_pool2d(1) -> flatten -> fc

``blocks[3]`` and ``blocks[6]`` have ``stride=2`` on ``conv1`` and an option-A
shortcut (``x[:, :, ::2, ::2]`` then zero-pad the new channels).  The global
average pool maps any spatial size to one value per channel, so the classifier
always receives 64 features whatever the feature-map resolution.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PadShortcut(nn.Module):
    """Option-A shortcut: stride subsample, then zero-pad the extra channels."""

    def __init__(self, stride: int, pad: int):
        super().__init__()
        self.stride, self.pad = stride, pad

    def forward(self, x):
        x = x[:, :, ::self.stride, ::self.stride]
        return F.pad(x, (0, 0, 0, 0, self.pad // 2, self.pad - self.pad // 2))


class BasicBlockBN20(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch, momentum=0.1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, momentum=0.1)
        if stride == 1 and in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = PadShortcut(stride, out_ch - in_ch)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet20BN(nn.Module):
    def __init__(self, num_classes: int = 10, width: int = 16, in_channels: int = 3):
        super().__init__()
        widths = [width, width * 2, width * 4]
        self.conv1 = nn.Conv2d(in_channels, widths[0], 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0], momentum=0.1)
        layers, in_ch = [], widths[0]
        for stage, out_ch in enumerate(widths):
            for b in range(3):
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


#: Where each retained intervention attaches, by module name.  Order matters:
#: position ``i`` of a list is site ``i`` of that placement.
SITE_MAP = {
    # the 19 main-path 3x3 convolutions in model.modules() order; the hook
    # replaces the convolution's output, i.e. BEFORE its BatchNorm and ReLU
    "conv_out": ["conv1"] + ["blocks.%d.conv%d" % (b, c)
                             for b in range(9) for c in (1, 2)],
    # 10 positions after a ReLU: the stem's post-ReLU tensor (the input of
    # ``blocks``) and the output of each of the 9 blocks (after the residual
    # addition and the block's final ReLU).  The ReLU *inside* each block
    # (after bn1) is deliberately not a site.
    "post_relu": [("input", "blocks")] + [("output", "blocks.%d" % b) for b in range(9)],
    # reduction points, named by the block whose complete output is reduced:
    # "block1" = the tensor entering blocks[2] = the output of blocks[1]
    "reduction": {"block1": ("input", "blocks.2")},
    "reference_resolution": 32,
}


def build(num_classes: int = 10, in_channels: int = 3, **kwargs) -> ResNet20BN:
    unknown = sorted(set(kwargs) - {"width"})
    if unknown:
        raise TypeError("resnet20_bn_cifar: unknown options %s" % unknown)
    return ResNet20BN(num_classes=num_classes, in_channels=in_channels, **kwargs)
