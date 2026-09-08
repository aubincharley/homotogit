r"""Transformation-family registry.

Adding a family means adding one entry here and one module implementing
:class:`~continuation.transforms.base.ImageTransform`.  Planned but *not
implemented in this phase*:

``rate``  -- an operational codec budget; native parameter is a *bit budget*,
             and the achieved rate must be measured, never assumed.

Implemented TV-budget families (native parameter ``t in [0,1]``, target ``t=1``):
``tv_l2`` and ``tv_hminus1`` share the constraint set
``{z in [0,1]: TV(z) <= t TV(x), per-channel means preserved}`` and differ only
in the reconstruction fidelity (squared ``L^2`` vs homogeneous ``\dot H^{-1}``).

See ``docs/extensions.md`` for the specification each of those must satisfy
before it is added (domain, minimum, feasible budgets, reconstruction fidelity,
tie-breaking, achieved-vs-requested reporting, and cost).
"""
from __future__ import annotations

from ..config import TransformConfig
from .base import ImageTransform, TransformResult, check_image_batch
from .gaussian import (
    GaussianSmoothing,
    gaussian_kernel_1d,
    heat_time_from_sigma,
    sigma_from_heat_time,
)
from .tv import TVBudgetTransform, TVHminus1, TVL2, solve_tv_budget, tv_value
from .wavelet import WaveletShrinkage, wavelet_shrink, swt2_analysis, swt2_synthesis
from . import gaussian as _gaussian
from . import tv as _tv
from . import wavelet as _wavelet

_BUILDERS = {
    "gaussian": _gaussian.build,
    "tv_l2": _tv.build_l2,
    "tv_hminus1": _tv.build_hminus1,
    "wavelet": _wavelet.build,
}


def available_families():
    return sorted(_BUILDERS)


def build_transform(cfg: TransformConfig) -> ImageTransform:
    if cfg.family not in _BUILDERS:
        raise KeyError(
            "unknown transformation family %r; implemented: %s "
            "(the operational rate family is specified in docs/extensions.md but "
            "is not implemented)" % (cfg.family, available_families())
        )
    return _BUILDERS[cfg.family](dict(cfg.params or {}))


__all__ = [
    "ImageTransform", "TransformResult", "check_image_batch",
    "GaussianSmoothing", "gaussian_kernel_1d",
    "heat_time_from_sigma", "sigma_from_heat_time",
    "TVBudgetTransform", "TVL2", "TVHminus1", "solve_tv_budget", "tv_value",
    "WaveletShrinkage", "wavelet_shrink", "swt2_analysis", "swt2_synthesis",
    "build_transform", "available_families",
]
