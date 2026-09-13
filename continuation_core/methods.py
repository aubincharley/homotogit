"""The four frozen presets, recovered from the executed ``unified`` batch.

Source: ``experiments/index.json`` on branch ``benchmark-organized``, experiment
``unified_selected``, driver ``scripts/unified_driver.py`` with controller
``continuation.ablation_ops.build_from_cell``.  Every number below was read
from that code and checked against the per-epoch ``current`` state recorded in
the runs' ``metrics.json`` (record ``k`` holds the state of epoch ``k - 1``).

These are frozen representatives of their families, not optimal methods, and
the four do **not** form a factorial design: the Gaussian-only method filters
at 10 post-ReLU positions, the combined method at 19 convolution outputs, and
the combined method's per-site sigma is additionally scaled by the resolution
ratio.  ``resolution_max_b1_gaussian_conv`` is therefore *not*
``resolution_max_b1`` composed with ``gaussian_postrelu``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .schedules import EpochSchedule

#: Executed Gaussian level G(e): 1.00 / 0.85 / 0.70 / 0.60 / 0.50 / 0.40 / 0.30
#: for three epochs each, then exactly 0 from epoch 21 (``ablation_manifest``:
#: ``_PLATEAU[min(e // 3, 6)] if e < 21 else 0``).
PLATEAU_G = EpochSchedule((0, 3, 6, 9, 12, 15, 18, 21),
                          (1.00, 0.85, 0.70, 0.60, 0.50, 0.40, 0.30, 0.0))

#: Executed resolution r(e) at the reduction point: 16 for epochs 0-5, 24 for
#: 6-11, 32 (exact bypass) from 12 (``ablation2_manifest.RESOLUTIONS_ABL2``).
RPROG = EpochSchedule((0, 6, 12), (16, 24, 32))

PLACEMENTS = ("conv_out", "post_relu")
SIGMA_SCALE_RULES = {
    "none": "sigma_site = G(e) at every site",
    "resolution_ratio_all_sites": "sigma_site = (r(e) / reference_resolution) * G(e) "
                                  "at EVERY site, including sites upstream of the "
                                  "reduction point that still run at full resolution "
                                  "(executed behaviour of SiteController.q_for for a "
                                  "block reduction)",
}


@dataclass(frozen=True)
class GaussianSpec:
    placement: str
    schedule: EpochSchedule
    sigma_scale: str
    sigma_max: float = 1.0
    truncate: float = 4.0
    units: str = "pixels of the feature map at the site"

    def __post_init__(self):
        if self.placement not in PLACEMENTS:
            raise ValueError("placement must be one of %s" % (PLACEMENTS,))
        if self.sigma_scale not in SIGMA_SCALE_RULES:
            raise ValueError("sigma_scale must be one of %s" % sorted(SIGMA_SCALE_RULES))


@dataclass(frozen=True)
class ResolutionSpec:
    point: str
    schedule: EpochSchedule
    reference_resolution: int = 32
    operator: str = "adaptive_max"

    def __post_init__(self):
        if self.operator != "adaptive_max":
            raise ValueError("only adaptive_max is retained in continuation-core")


@dataclass(frozen=True)
class MethodSpec:
    id: str
    description: str
    gaussian: GaussianSpec | None = None
    resolution: ResolutionSpec | None = None
    source: dict = field(default_factory=dict)
    validated_on: tuple = ("cifar10", "resnet20_bn_cifar")

    def __post_init__(self):
        if (self.gaussian and self.gaussian.sigma_scale == "resolution_ratio_all_sites"
                and self.resolution is None):
            raise ValueError("resolution_ratio_all_sites needs a resolution schedule")

    def to_dict(self) -> dict:
        d = asdict(self)
        for part in ("gaussian", "resolution"):
            if getattr(self, part) is not None:
                d[part]["schedule"] = getattr(self, part).schedule.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MethodSpec":
        g = r = None
        if d.get("gaussian"):
            g = GaussianSpec(**{**d["gaussian"],
                                "schedule": EpochSchedule.from_dict(d["gaussian"]["schedule"])})
        if d.get("resolution"):
            r = ResolutionSpec(**{**d["resolution"],
                                  "schedule": EpochSchedule.from_dict(d["resolution"]["schedule"])})
        return cls(id=d["id"], description=d["description"], gaussian=g, resolution=r,
                   source=d.get("source", {}), validated_on=tuple(d.get("validated_on", ())))


def _src(config_id, label, per_seed):
    vals = list(per_seed.values())
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return {"experiment": "unified_selected", "branch": "benchmark-organized",
            "config_id": config_id, "label": label,
            "cells": ["%s__seed%s" % (config_id, s) for s in per_seed],
            "final_test_acc_per_seed": per_seed,
            "final_test_acc_mean": round(m, 6), "final_test_acc_sd": round(sd, 6),
            "final_path": "target (equal to current at epoch 30)",
            "asset_set": "r20bn-campaign-assets"}


METHODS = {
    "plain": MethodSpec(
        id="plain",
        description="ResNet-20 BN with no intervention (control).",
        source=_src("plain", "Baseline: no blur, full 32x32 throughout",
                    {"0": 0.7458, "1": 0.7566, "2": 0.7605})),
    "resolution_max_b1": MethodSpec(
        id="resolution_max_b1",
        description="Adaptive max pooling of the complete output of blocks[1] "
                    "(input of blocks[2]) to 16 / 24 / 32; no Gaussian.",
        resolution=ResolutionSpec(point="block1", schedule=RPROG),
        source=_src("shrink_b1", "Shrink block 1 with max-pool, 16 to 24 to 32",
                    {"0": 0.7997, "1": 0.8006, "2": 0.8107})),
    "gaussian_postrelu": MethodSpec(
        id="gaussian_postrelu",
        description="Gaussian at the 10 post-ReLU positions (stem output + 9 "
                    "block outputs), plateau schedule, no resolution change.",
        gaussian=GaussianSpec(placement="post_relu", schedule=PLATEAU_G,
                              sigma_scale="none"),
        source=_src("blur_relu", "Blur after every ReLU, annealed to zero",
                    {"0": 0.8006, "1": 0.8006, "2": 0.7960})),
    "resolution_max_b1_gaussian_conv": MethodSpec(
        id="resolution_max_b1_gaussian_conv",
        description="Adaptive max pooling after blocks[1] (16 / 24 / 32) plus "
                    "the Gaussian at all 19 convolution outputs, sigma scaled "
                    "by r(e)/32 at every site.",
        resolution=ResolutionSpec(point="block1", schedule=RPROG),
        gaussian=GaussianSpec(placement="conv_out", schedule=PLATEAU_G,
                              sigma_scale="resolution_ratio_all_sites"),
        source=_src("shrink_b1_conv", "Shrink block 1, plus blur every conv layer",
                    {"0": 0.8130, "1": 0.8150, "2": 0.8173})),
}


def get_method(method_id: str) -> MethodSpec:
    if method_id not in METHODS:
        raise KeyError("unknown method %r; available: %s" % (method_id, sorted(METHODS)))
    return METHODS[method_id]
