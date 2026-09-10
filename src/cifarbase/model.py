"""ResNet for 32x32 inputs.

The difference from a torchvision ResNet is two lines, and it is the whole
adaptation: a 3x3 stride-1 stem instead of 7x7 stride-2, and no maxpool. An
ImageNet stem downsamples 224 -> 56 before the first block, which on a 32x32
image would hand stage 1 an 8x8 feature map and throw away most of the picture.
Keeping 32x32 into stage 1 is what makes the standard ~95% CIFAR-10 number
reachable; the ImageNet stem costs several points.

The blocks are post-activation (He et al. 2015), i.e. the residual add happens
before the last ReLU. layer-l2-homotopy uses pre-activation blocks instead --
that is a different model, not a refactor of this one.

Every ReLU is an `Activation` submodule rather than an `F.relu` call. That is
the one concession this file makes to the activation homotopy, and it is what
makes the homotopy reachable at all: `ResidualGate` can hook `bn2` because bn2
is a module, but a functional call has nothing to attach to. The submodules
carry no state, so the model is otherwise exactly what it was.
"""
import torch
import torch.nn.functional as F
from torch import nn


def conv3x3(in_ch, out_ch, stride=1):
    # bias=False because BatchNorm's beta immediately follows and would make the
    # conv bias an unused duplicate parameter.
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)


class Activation(nn.Module):
    """phi_alpha(x) = max(x, alpha*x): a LeakyReLU whose slope is the homotopy.

        alpha = 0  ->  ReLU, the target network
        alpha = 1  ->  the identity, which makes the whole ResNet affine

    alpha is a plain float, not a parameter and not a buffer, so state_dict()
    is unchanged by this class existing: a checkpoint written before it does
    still loads, and a homotopy model and a baseline model share a state_dict.
    That is the same invariant ResidualGate maintains for s, and it is what
    makes an accuracy measured here comparable to a baseline number.

    alpha == 0 dispatches to F.relu rather than to F.leaky_relu(x, 0.0). The
    two agree mathematically, but leaky_relu returns -0.0 on negative inputs
    and runs a different kernel; the branch is what makes alpha=0 bit-identical
    to the baseline rather than merely equal to it.
    """

    def __init__(self, inplace=True):
        super().__init__()
        self.alpha = 0.0
        self.inplace = inplace

    def forward(self, x):
        if self.alpha == 0.0:
            return F.relu(x, inplace=self.inplace)
        return F.leaky_relu(x, self.alpha, inplace=self.inplace)

    def extra_repr(self):
        return f"alpha={self.alpha}"


class BasicBlock(nn.Module):
    """conv-bn-act-conv-bn, plus the identity, then act.

    `use_residual=False` drops the `+ x` and the projection that exists only to
    make x addable, turning the stack into a PlainNet block. Nothing else
    changes: same channels, same strides, same BatchNorm, same activations, so
    the presence of the skip is the single variable of the ablation.

    That matters because residual connections are what make a deep network's
    loss surface look convex (Li et al. 2018). A homotopy that starts from a
    near-linear network has little left to fix on a ResNet; on a PlainNet the
    surface is chaotic again, and that is where it has to prove itself.
    """

    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        # act1 is the residual branch's nonlinearity, act2 the main path's.
        # They are separable on purpose: alpha on act1 alone leaves the block
        # nonlinear, alpha on act2 alone leaves F nonlinear, and only both at
        # once make the network affine.
        self.act1 = Activation()
        self.act2 = Activation()

        # A projection only where the identity cannot be added as-is: a stride
        # change or a channel change. Everywhere else the shortcut is free.
        # With no residual there is nothing to add, so the attribute is absent
        # rather than an unused Identity -- a shortcut left in place would hold
        # parameters, take gradient, and show up in the count.
        if not use_residual:
            pass
        elif stride != 1 or in_ch != out_ch * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * self.expansion))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.use_residual:
            return self.act2(out)
        return self.act2(out + self.shortcut(x))


class ResNet(nn.Module):
    """Four stages at 32/16/8/4 resolution, global average pool, linear head."""

    def __init__(self, blocks, num_classes=10, width=64, zero_init_residual=True,
                 use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        widths = [width * 2 ** i for i in range(4)]
        self.in_ch = widths[0]

        # The CIFAR stem: 3x3, stride 1, and no maxpool after it.
        self.conv1 = conv3x3(3, widths[0])
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.act0 = Activation()
        self.layer1 = self._stage(widths[0], blocks[0], stride=1)
        self.layer2 = self._stage(widths[1], blocks[1], stride=2)
        self.layer3 = self._stage(widths[2], blocks[2], stride=2)
        self.layer4 = self._stage(widths[3], blocks[3], stride=2)
        self.fc = nn.Linear(widths[3] * BasicBlock.expansion, num_classes)

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
        #
        # It is fatal without the skips: with no x to add back, gamma_bn2 = 0
        # makes every block output exactly zero and the network collapses to a
        # constant. config._validate refuses the combination rather than letting
        # a run spend an hour producing 10% accuracy.
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, BasicBlock):
                    nn.init.zeros_(module.bn2.weight)

    def _stage(self, out_ch, count, stride):
        layers = []
        for block_stride in [stride] + [1] * (count - 1):
            layers.append(BasicBlock(self.in_ch, out_ch, block_stride,
                                     use_residual=self.use_residual))
            self.in_ch = out_ch * BasicBlock.expansion
        return nn.Sequential(*layers)

    def activation_sites(self, which="all"):
        """Every Activation in forward order, as (module, stage, block).

        The model owns this because it is a fact about its own topology.
        stage is 0 for the stem and 1..4 for layer1..layer4; block is a running
        index over BasicBlocks, -1 for the stem. `which` selects act1 (inside
        the residual branch), act2 (the main path, after the add) or both.
        """
        sites = []
        if which == "all":
            sites.append((self.act0, 0, -1))
        block_index = 0
        stages = (self.layer1, self.layer2, self.layer3, self.layer4)
        for stage_index, stage in enumerate(stages, start=1):
            for block in stage:
                if which in ("all", "act1"):
                    sites.append((block.act1, stage_index, block_index))
                if which in ("all", "act2"):
                    sites.append((block.act2, stage_index, block_index))
                block_index += 1
        return sites

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act0(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


class VGG(nn.Module):
    """VGG-11 with BatchNorm, adapted to 32x32 inputs.

    A network designed without skip connections, rather than a ResNet with the
    skips deleted. That distinction is the reason this class exists: removing
    the `+ x` from a ResNet-18 leaves a model He et al. (2015) found roughly as
    trainable as the residual one -- the degradation that motivates skips
    appears at thirty-four layers -- so a PlainNet-18 may not be the rough
    landscape an activation homotopy was meant for. VGG is.

    The CIFAR adaptation is the usual one: the eight convolutions of the
    ImageNet vgg11_bn, then a global average pool and a single linear layer
    instead of the three 4096-wide fully connected layers, which exist for
    224x224 inputs and would be most of the parameters here.

    Each convolution is followed by BatchNorm and an `Activation`, so every
    diagnostic written for the ResNet -- linear_gap, the alpha=0 readout with
    its BatchNorm re-estimation, the anchor over conv and linear weights --
    applies unchanged.
    """

    def __init__(self, plan, num_classes=10, width=64):
        super().__init__()
        # width scales the whole plan, so --width 16 gives a runnable smoke test
        # exactly as it does for the ResNet.
        scale = width / 64
        layers, in_ch = [], 3
        for item in plan:
            if item == "M":
                layers.append(nn.MaxPool2d(2, 2))
                continue
            out_ch = max(1, round(item * scale))
            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                       nn.BatchNorm2d(out_ch), Activation()]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(in_ch, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def activation_sites(self, which="all"):
        """The eight Activations in forward order, as (module, stage, block).

        A VGG stage is a run of convolutions between two maxpools -- 1, 1, 2, 2,
        2 for vgg11 -- which is what a stage-wise schedule should group. block
        is the running convolution index.

        act1 and act2 name the two activations of a residual block and have no
        counterpart here; returning all eight for them would turn an ablation
        config into a silent no-op.
        """
        if which != "all":
            raise ValueError(
                f"a_sites={which!r} names an activation of a residual block; "
                f"this network has none. Use a_sites=all.")
        sites, stage, block = [], 0, 0
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                stage += 1
            elif isinstance(layer, Activation):
                sites.append((layer, stage, block))
                block += 1
        return sites

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


ARCHS = {
    "resnet18": (2, 2, 2, 2),
    "resnet34": (3, 4, 6, 3),
}

VGG_PLANS = {
    "vgg11": (64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"),
}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg, device, quiet=False):
    arch = cfg["arch"]
    if arch in VGG_PLANS:
        model = VGG(VGG_PLANS[arch], num_classes=10, width=cfg.get("width", 64))
        kind = f"{arch}-bn"
    elif arch in ARCHS:
        model = ResNet(ARCHS[arch], num_classes=10, width=cfg.get("width", 64),
                       zero_init_residual=bool(cfg.get("zero_init_residual", True)),
                       use_residual=bool(cfg.get("use_residual", True)))
        kind = arch if model.use_residual else f"plain-{arch} (no skips)"
    else:
        known = ", ".join(list(ARCHS) + list(VGG_PLANS))
        raise ValueError(f"unknown arch {arch!r}: pick one of {known}")
    model = model.to(device)
    if not quiet:
        print(f"model: {kind}, {count_params(model) / 1e6:.2f}M params")
    return model
