"""Architecture registry: a builder and an explicit intervention-site map.

To add an architecture, add one module with ``build(num_classes, in_channels,
**options)`` and a ``SITE_MAP`` naming modules for each placement it supports,
then register it below.  A placement or reduction point that is not in the map
raises ``UnsupportedInsertionError``; nothing is inferred.  See docs/EXTENDING.md.
"""
from __future__ import annotations

from . import resnet20_act, resnet20_bn, resnet20_gn, vgg11_bn

REGISTRY = {
    "resnet20_bn_cifar": {"module": resnet20_bn, "validated": True,
                          "note": "reference architecture of every recorded result"},
    "resnet20_act_cifar": {"module": resnet20_act, "validated": False,
                           "note": "ResNet-20 BN with a choice of relu/gelu/silu; "
                                   "the ReLU control, state dict interchangeable "
                                   "with resnet20_bn_cifar"},
    "resnet20_gn_cifar": {"module": resnet20_gn, "validated": False,
                          "note": "ResNet-20 with GroupNorm; the BatchNorm control, "
                                  "same site map, no running statistics"},
    "vgg11_bn": {"module": vgg11_bn, "validated": False,
                 "note": "non-ResNet adapter; no retained method has been run on it"},
}


def available():
    return sorted(REGISTRY)


def _entry(arch: str):
    if arch not in REGISTRY:
        raise KeyError("unknown architecture %r; registered: %s" % (arch, available()))
    return REGISTRY[arch]


def build_model(arch: str, num_classes: int, in_channels: int = 3, **options):
    return _entry(arch)["module"].build(num_classes=num_classes,
                                        in_channels=in_channels, **options)


def site_map(arch: str) -> dict:
    return _entry(arch)["module"].SITE_MAP


def is_validated(arch: str) -> bool:
    return bool(_entry(arch)["validated"])
