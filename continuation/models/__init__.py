"""Model registry.

Future network-space interventions (feature-map smoothing, activation
homotopies, parameter-space homotopies) are *not* data transformations and would
be added here as separate architectures/objectives with their own controlled
experiments -- a data-transform interface alone does not implement them.
"""
from __future__ import annotations

import torch

from ..config import ModelConfig
from ..seeding import derive_seed
from .resnet18_bn import ResNet18BNCifar, resnet18_bn_cifar
from .resnet20_bn import ResNet20BNCifar, resnet20_bn_cifar
from .resnet_gn import ResNetCifarGN, resnet20_gn

_BUILDERS = {
    "resnet20_gn": resnet20_gn,
    "resnet18_bn_cifar": resnet18_bn_cifar,
    "resnet20_bn_cifar": resnet20_bn_cifar,
}


def available_models():
    return sorted(_BUILDERS)


def build_model(cfg: ModelConfig, num_classes: int, seed: int) -> torch.nn.Module:
    """Build a model with initialization drawn from the dedicated ``init`` stream."""
    if cfg.arch not in _BUILDERS:
        raise KeyError("unknown arch %r; available: %s" % (cfg.arch, available_models()))
    # Seeding the global torch RNG immediately before construction keeps parameter
    # initialization identical across conditions with the same run seed; nothing
    # else in the training loop draws from the global torch RNG.
    torch.manual_seed(derive_seed(seed, "init"))
    torch.cuda.manual_seed_all(derive_seed(seed, "init"))
    return _BUILDERS[cfg.arch](
        num_classes=num_classes,
        width=cfg.width,
        channels_per_group=cfg.channels_per_group,
        shortcut=cfg.shortcut,
        zero_init_residual=cfg.zero_init_residual,
    )


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


__all__ = ["ResNetCifarGN", "resnet20_gn", "ResNet18BNCifar", "resnet18_bn_cifar",
           "build_model", "available_models", "count_parameters"]
