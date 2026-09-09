"""Operators for the 21-configuration CIFAR-10 exploratory campaign.

Three things live here, all built on the already-verified pieces:

**Reduction paths.**  Four ways of realising a resolution schedule ``r(e)``.
``input_bilinear`` / ``input_max`` shrink the *original float image* before
normalization; ``stem_bilinear`` / ``stem_max`` keep the input at 32x32 and
shrink the stem output after its BN and ReLU, immediately before the first
residual block.  Every path is a real spatial-size change -- nothing is
upsampled back to 32 -- and every path is an exact bypass at ``r = 32``, so the
final unfiltered phase runs the ordinary ResNet-20 forward path.

Adaptive max pooling is *not* disjoint 2x2 pooling in general.  PyTorch's
adaptive rule takes, for output index ``i`` of ``out`` from input length ``in``,
the window ``[floor(i*in/out), ceil((i+1)*in/out))``.  At 32 -> 16 that happens
to give 16 disjoint windows of width 2.  At 32 -> 24 it gives a mixture of
width-2 and width-1 windows and **consecutive windows overlap**; the exact
window list is emitted by :func:`adaptive_windows` and recorded in the checks.

**Per-site Gaussian scaling.**  At insertion point ``l`` the width follows the
existing relative-resolution convention

    sigma_l(e) = q_l(e) * G(e),
    q_l(e) = current spatial width at l / width at l in the ordinary 32x32 net.

So an input reduction gives every one of the 19 sites ``q = r/32``; a reduction
after the stem leaves the stem at ``q = 1`` and gives the other 18 ``q = r/32``;
``R32`` gives ``q = 1`` everywhere.  This is a convention, not a claim of exact
spectral equivalence between filters at different resolutions.

**Gmix.**  An identity-to-Gaussian mixture at all 19 sites,

    T_alpha(h) = (1 - alpha) h + alpha * G_{q_l(e)}(h),

with the reference Gaussian width fixed at 1 so ``q_l(e)`` accounts only for the
current resolution.  ``alpha = 0`` returns ``h`` without evaluating the Gaussian;
``alpha = 1`` reproduces the corresponding Gaussian output exactly.  It replaces
the usual Gaussian at each site -- it does not add a second filter, duplicate the
network, or mix logits.  Its alpha levels are **not** claimed to be
strength-matched to the sigma levels of Gplateau.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transforms.gaussian import GaussianSmoothing

N_SITES = 19                      # stem + 2 per BasicBlock x 9 blocks
EARLY7 = tuple(range(7))          # stem + the six stage-1 main-path convolutions
REDUCTIONS = ("input_bilinear", "input_max", "stem_bilinear", "stem_max")


# --------------------------------------------------------------------------
# reductions
# --------------------------------------------------------------------------

def adaptive_windows(n_in: int, n_out: int):
    """The 1-D windows PyTorch's adaptive pooling uses, as ``(start, stop)``."""
    return [(int(math.floor(i * n_in / n_out)),
             int(math.ceil((i + 1) * n_in / n_out))) for i in range(n_out)]


def describe_adaptive(n_in: int, n_out: int) -> dict:
    w = adaptive_windows(n_in, n_out)
    widths = sorted({b - a for a, b in w})
    overlaps = sum(1 for j in range(1, len(w)) if w[j][0] < w[j - 1][1])
    return {"windows": w, "distinct_widths": widths,
            "n_overlapping_pairs": overlaps,
            "disjoint_uniform": overlaps == 0 and len(widths) == 1}


def reduce_spatial(x: torch.Tensor, r, how: str) -> torch.Tensor:
    """Resize ``x`` to ``r x r``.  Exact bypass when ``r`` already matches.

    Autograd flows through both branches, so the stem paths stay differentiable.
    """
    if r is None or int(r) == int(x.shape[-1]):
        return x
    r = int(r)
    if how.endswith("bilinear"):
        return F.interpolate(x, size=(r, r), mode="bilinear",
                             align_corners=False, antialias=True)
    if how.endswith("max"):
        # adaptive max pooling acts per channel already
        return F.adaptive_max_pool2d(x, (r, r))
    raise ValueError("unknown reduction %r" % (how,))


# --------------------------------------------------------------------------
# per-site controller
# --------------------------------------------------------------------------

class SiteController:
    """Applies the scheduled operator, with its own width, at each site.

    ``operator`` is ``"none"``, ``"gaussian"`` or ``"gmix"``.  ``levels[e]`` is
    the schedule value held for the whole of epoch ``e`` (sigma for gaussian,
    alpha for gmix).  ``q_by_site`` gives the per-site resolution ratio for the
    current epoch and is refreshed by :meth:`set_epoch`.
    """

    def __init__(self, operator: str, levels=None, sites=None,
                 resolution_by_epoch=None, reduction="input_bilinear"):
        if operator not in ("none", "gaussian", "gmix"):
            raise ValueError("unknown operator %r" % (operator,))
        self.operator = operator
        self.levels = [float(v) for v in levels] if levels else None
        self.sites = tuple(range(N_SITES)) if sites is None else tuple(sites)
        self.resolution_by_epoch = ([int(v) for v in resolution_by_epoch]
                                    if resolution_by_epoch else None)
        self.reduction = reduction
        self.gauss = (GaussianSmoothing(sigma_max=1.0, truncate=4.0)
                      if operator in ("gaussian", "gmix") else None)
        self.value = None                 # sigma or alpha for the current epoch
        self.resolution = None
        self.bypass_all = False           # target-path evaluation
        self.q = [1.0] * N_SITES

    # -- schedule ---------------------------------------------------------

    def q_for(self, r) -> list:
        """Per-site resolution ratio at input resolution ``r``."""
        if r is None:
            return [1.0] * N_SITES
        ratio = float(r) / 32.0
        if self.reduction.startswith("stem"):
            # the stem still sees a full 32x32 input; everything after it is reduced
            return [1.0] + [ratio] * (N_SITES - 1)
        return [ratio] * N_SITES

    def set_epoch(self, e: int):
        if self.levels is not None:
            i = max(0, min(int(e), len(self.levels) - 1))
            self.value = self.levels[i]
        self.resolution = (self.resolution_by_epoch[
            max(0, min(int(e), len(self.resolution_by_epoch) - 1))]
            if self.resolution_by_epoch else None)
        self.q = self.q_for(self.resolution)
        return self.value

    def set_state(self, level, resolution):
        """Force an explicit (level, resolution) -- used by evaluation."""
        self.value = level
        self.resolution = resolution
        self.q = self.q_for(resolution)
        return self.value

    def is_active(self) -> bool:
        """True when the operator is not an exact identity right now."""
        if self.bypass_all or self.operator == "none" or self.value is None:
            return False
        return float(self.value) > 0.0

    # -- application ------------------------------------------------------

    def apply_at(self, site: int, h: torch.Tensor) -> torch.Tensor:
        if not self.is_active() or site not in self.sites:
            return h                                  # exact bypass
        q = self.q[site]
        if self.operator == "gaussian":
            return self.gauss(h, q * float(self.value))
        alpha = float(self.value)                     # gmix, reference width 1
        return (1.0 - alpha) * h + alpha * self.gauss(h, q)

    def input_resolution(self):
        """Resolution to feed the network (None when the stem paths reduce).

        Deliberately independent of bypass_all: that flag disables the
        *filters* only.  An unfiltered arm still trains and is evaluated at its
        scheduled resolution, and the target path bypasses the reduction by
        asking for 32 -- which reduce_spatial returns unchanged.
        """
        if self.resolution is None:
            return None
        return None if self.reduction.startswith("stem") else self.resolution

    def stem_resolution(self):
        """Resolution to apply after the stem, or None."""
        if self.resolution is None:
            return None
        return self.resolution if self.reduction.startswith("stem") else None

    def describe(self) -> dict:
        return {"operator": self.operator, "levels": self.levels,
                "n_sites": len(self.sites), "sites": list(self.sites),
                "insertion_mask": "all19" if len(self.sites) == N_SITES
                else ("early7" if tuple(self.sites) == EARLY7 else "custom"),
                "reduction": self.reduction,
                "resolution_by_epoch": self.resolution_by_epoch,
                "sigma_rule": "sigma_l(e) = q_l(e) * G(e); q_l = width_l / width_l@32",
                "gmix_rule": ("T_alpha(h) = (1-alpha) h + alpha G_{q_l}(h), "
                              "reference width 1; alpha=0 returns h unevaluated"),
                "gaussian_kernel": (None if self.gauss is None else
                                    {"kernel_size": self.gauss.kernel_size,
                                     "radius": self.gauss.radius})}


def attach_sites(model: nn.Module, ctrl: SiteController):
    """Hook the 19 main-path 3x3 convolutions, plus the stem reduction.

    Convolution order from ``model.modules()`` is stem, then ``conv1``/``conv2``
    of each BasicBlock in depth order, so site 0 is the stem and sites 1..6 are
    stage 1 -- which is exactly the ``early7`` set.  Option-A shortcuts hold no
    convolution and are never hooked.
    """
    handles, site = [], 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
            handles.append(m.register_forward_hook(
                lambda _m, _i, out, s=site: ctrl.apply_at(s, out)))
            site += 1
    if site != N_SITES:
        for h in handles:
            h.remove()
        raise RuntimeError("expected %d insertion sites, found %d" % (N_SITES, site))

    # stem reduction: after stem BN + ReLU, before the first residual block
    def _pre(_mod, args):
        r = ctrl.stem_resolution()
        if r is None:
            return None
        return (reduce_spatial(args[0], r, ctrl.reduction),)

    handles.append(model.blocks.register_forward_pre_hook(_pre))
    return handles
