"""ResNet for 32x32 inputs, in both shapes CIFAR papers use.

The difference from a torchvision ResNet is two lines, and it is the whole
adaptation: a 3x3 stride-1 stem instead of 7x7 stride-2, and no maxpool. An
ImageNet stem downsamples 224 -> 56 before the first block, which on a 32x32
image would hand stage 1 an 8x8 feature map and throw away most of the picture.
Keeping 32x32 into stage 1 is what makes the standard ~95% CIFAR-10 number
reachable; the ImageNet stem costs several points.

Two families live in ARCHS and they differ by their number of STAGES, not by a
depth constant. resnet18/34 are the four-stage ImageNet topology at 32/16/8/4
resolution with width 64. resnet20/32/44 are He et al. section 4.2 -- three
stages at 32/16/8, width 16, 0.27M parameters for resnet20 against 11M for
resnet18. A stage count is therefore read off `len(blocks)`, and `width` has to
match the family: 64 for the four-stage one, 16 for the CIFAR one. _validate
refuses the mismatch rather than quietly training a 16x larger model under the
name resnet20.

The blocks are post-activation (He et al. 2015), i.e. the residual add happens
before the last ReLU. layer-l2-homotopy uses pre-activation blocks instead --
that is a different model, not a refactor of this one.
"""
import torch
import torch.nn.functional as F
from torch import nn


def conv3x3(in_ch, out_ch, stride=1):
    # bias=False because BatchNorm's beta immediately follows and would make the
    # conv bias an unused duplicate parameter.
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """conv-bn-relu-conv-bn, plus the identity, then relu."""

    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # A projection only where the identity cannot be added as-is: a stride
        # change or a channel change. Everywhere else the shortcut is free.
        if stride != 1 or in_ch != out_ch * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * self.expansion))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class ResNet(nn.Module):
    """`len(blocks)` stages, each halving resolution after the first, global
    average pool, linear head. Four stages is the ImageNet topology (32/16/8/4),
    three is the CIFAR one (32/16/8)."""

    def __init__(self, blocks, num_classes=10, width=64, zero_init_residual=True):
        super().__init__()
        widths = [width * 2 ** i for i in range(len(blocks))]
        self.in_ch = widths[0]

        # The CIFAR stem: 3x3, stride 1, and no maxpool after it.
        self.conv1 = conv3x3(3, widths[0])
        self.bn1 = nn.BatchNorm2d(widths[0])
        # Stage 1 keeps the resolution; every later stage halves it. Built in a
        # loop rather than as layer1..layer4 so the stage count is a property of
        # ARCHS instead of a hard-coded four.
        self.stages = nn.Sequential(*[
            self._stage(ch, count, stride=1 if i == 0 else 2)
            for i, (ch, count) in enumerate(zip(widths, blocks, strict=True))])
        self.fc = nn.Linear(widths[-1] * BasicBlock.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Zeroing the last BN gamma in each block starts every residual branch at
        # exactly the identity, so a deep stack begins well-conditioned instead of
        # relying on the init to be small. Worth a few tenths of a point, and it
        # has to happen after the loop above resets every gamma to one.
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, BasicBlock):
                    nn.init.zeros_(module.bn2.weight)

    def _stage(self, out_ch, count, stride):
        layers = []
        for block_stride in [stride] + [1] * (count - 1):
            layers.append(BasicBlock(self.in_ch, out_ch, block_stride))
            self.in_ch = out_ch * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.stages(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


# name -> blocks per stage. Length is the stage count, so it also selects the
# family; WIDTHS below records the width each family is defined at.
ARCHS = {
    "resnet18": (2, 2, 2, 2),
    "resnet34": (3, 4, 6, 3),
    "resnet20": (3, 3, 3),
    "resnet32": (5, 5, 5),
    "resnet44": (7, 7, 7),
}

# The width each arch is defined at, which _validate holds you to. A resnet20 at
# width 64 is a 4.3M-parameter model wearing a 0.27M model's name, and every
# runtime estimate made from the name would be wrong by 16x.
WIDTHS = {name: (16 if len(blocks) == 3 else 64)
          for name, blocks in ARCHS.items()}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg, device, quiet=False):
    arch = cfg["arch"]
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}: pick one of {', '.join(ARCHS)}")
    model = ResNet(ARCHS[arch], num_classes=10, width=cfg.get("width", 64),
                   zero_init_residual=bool(cfg.get("zero_init_residual", True)))
    model = model.to(device)
    if not quiet:
        print(f"model: {arch}, {count_params(model) / 1e6:.2f}M params")
    return model
