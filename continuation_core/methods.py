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

import math
from fractions import Fraction
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
class CBSSpec:
    """Curriculum By Smoothing (Sinha et al., 2020) as a comparator.

    Operator: 3x3 normalised Gaussian, zero padding 1, depthwise, between every
    main-path convolution and its BatchNorm (``conv_out``, 19 sites in
    ResNet-20).  ``schedule``:

    ``epoch_decay``      sigma(e) = sigma0 * factor ** floor(e / every_epochs)
                         (paper Sec. 4 and App. C; ``ResNet.get_new_kernels``)
    ``update_plateaus``  u_off = floor(off_fraction * U), K = ceil(log(0.1) / log(factor));
                         sigma(u) = sigma0 * factor ** floor(K * u / u_off) for u < u_off,
                         exact bypass (0) for u >= u_off.  A predetermined adaptation
                         to our horizon, not an author-published schedule.

    Native inference keeps the filter of the last training update.
    """
    schedule: str
    sigma0: float = 1.0
    factor: float = 0.9
    every_epochs: int = 5
    off_fraction: float = 0.7
    placement: str = "conv_out"
    kernel_size: int = 3
    source: str = "github.com/pairlab/CBS@5f62e7da5b290e8f62f408c6f89146f3f361cc2e"

    def __post_init__(self):
        if self.schedule not in ("epoch_decay", "update_plateaus"):
            raise ValueError("unknown CBS schedule %r" % self.schedule)
        if self.placement != "conv_out" or self.kernel_size != 3:
            raise ValueError("the CBS comparator is fixed to 3x3 kernels at conv outputs")

    def n_plateaus(self) -> int:
        return int(math.ceil(math.log(0.1) / math.log(self.factor)))

    def sigma_at(self, update: int, updates_per_epoch: int, total_updates: int) -> float:
        u = int(update)
        if self.schedule == "epoch_decay":
            return float(self.sigma0 * self.factor ** ((u // int(updates_per_epoch)) // int(self.every_epochs)))
        u_off = self.off_update(total_updates)
        if u >= u_off:
            return 0.0
        return float(self.sigma0 * self.factor ** ((self.n_plateaus() * u) // u_off))

    def off_update(self, total_updates: int) -> int:
        """``floor(off_fraction * U)`` in exact rational arithmetic (0.7 * 62560 is 43791.999...
        in binary floating point; the decimal fraction 7/10 gives 43792)."""
        f = Fraction(str(self.off_fraction))
        return int(f.numerator * int(total_updates) // f.denominator)

    def table(self, updates_per_epoch: int, total_updates: int) -> list:
        """(first update, epoch, sigma) of every constant segment."""
        out, prev = [], None
        for u in range(int(total_updates)):
            v = self.sigma_at(u, updates_per_epoch, total_updates)
            if v != prev:
                out.append({"first_update": u, "epoch": u // int(updates_per_epoch), "sigma": v})
                prev = v
        return out


@dataclass(frozen=True)
class SDPointSpec:
    """Stochastic Downsampling Point (Kuen et al., 2018) as a comparator.

    Per logical optimizer batch: point p ~ U{0..n_points}, ratio ~ U(ratios),
    independently; p = 0 is the unmodified network, p = k > 0 applies adaptive
    average pooling to ``int(round(n * ratio))`` after the residual addition of
    residual block k and before its final ReLU (``post_add`` sites).  Draws come
    from a counter-based stream ``derive_seed(seed, stream % u)``: independent of
    the data order and of every global torch/numpy/python RNG, and identical
    after resumption.  Inference instance: full resolution (p = 0).
    """
    n_points: int = 9
    ratios: tuple = (0.5, 0.75)
    placement: str = "post_add"
    operator: str = "adaptive_avg"
    inference_instance: str = "full_resolution"
    stream: str = "sdpoint::update::%d"
    source: str = "github.com/xternalz/SDPoint@0013c5dafe80780ea749198ebc42824c3ed41e6c"


@dataclass(frozen=True)
class MethodSpec:
    id: str
    description: str
    gaussian: GaussianSpec | None = None
    resolution: ResolutionSpec | None = None
    source: dict = field(default_factory=dict)
    validated_on: tuple = ("cifar10", "resnet20_bn_cifar")
    cbs: CBSSpec | None = None
    sdpoint: SDPointSpec | None = None

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
        c = CBSSpec(**d["cbs"]) if d.get("cbs") else None
        sp = (SDPointSpec(**{**d["sdpoint"], "ratios": tuple(d["sdpoint"]["ratios"])})
              if d.get("sdpoint") else None)
        return cls(id=d["id"], description=d["description"], gaussian=g, resolution=r,
                   source=d.get("source", {}), validated_on=tuple(d.get("validated_on", ())),
                   cbs=c, sdpoint=sp)


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
    # ---- comparators (branch comparison-cbs-sdpoint); not frozen representatives ----
    "cbs_published_schedule": MethodSpec(
        id="cbs_published_schedule",
        description="Curriculum By Smoothing: 3x3 Gaussian (zero padding) between each of the 19 "
                    "main-path convolutions and its BatchNorm; sigma(e) = 0.9 ** floor(e / 5); "
                    "native inference keeps the epoch-29 filter (sigma 0.59049).",
        cbs=CBSSpec(schedule="epoch_decay"),
        source={"paper": "Sinha, Garg, Larochelle, NeurIPS 2020, arXiv:2003.01367v5, Sec. 3-4, App. C",
                "code": "github.com/pairlab/CBS@5f62e7da5b290e8f62f408c6f89146f3f361cc2e"},
        validated_on=()),
    "cbs_budget_matched": MethodSpec(
        id="cbs_budget_matched",
        description="Same CBS operator and sites; 22 update-level plateaus sigma = 0.9 ** "
                    "floor(22 u / u_off) for u < u_off = floor(0.7 U), then exact bypass. "
                    "CBS-derived adaptation to our horizon, not an author schedule.",
        cbs=CBSSpec(schedule="update_plateaus"),
        source={"adaptation": "predetermined before any result; see comparison/METHOD_MAPPING.md"},
        validated_on=()),
    "sdpoint": MethodSpec(
        id="sdpoint",
        description="SDPoint: per batch, p ~ U{0..9}, ratio ~ U{0.5, 0.75}; adaptive average "
                    "pooling after the residual addition, before the final ReLU, of block p; "
                    "inference on the full-resolution instance.",
        sdpoint=SDPointSpec(),
        source={"paper": "Kuen et al., CVPR 2018, arXiv:1801.09335v1, Sec. 4-6, Alg. 1",
                "code": "github.com/xternalz/SDPoint@0013c5dafe80780ea749198ebc42824c3ed41e6c"},
        validated_on=()),
}


def get_method(method_id: str) -> MethodSpec:
    if method_id not in METHODS:
        raise KeyError("unknown method %r; available: %s" % (method_id, sorted(METHODS)))
    return METHODS[method_id]
