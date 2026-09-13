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

**Per-site level table.**  The controller's state is a table ``L[e][l]`` giving
the pre-resolution level held at site ``l`` throughout epoch ``e``.  At
insertion point ``l`` the width then follows the existing relative-resolution
convention

    sigma_l(e) = q_l(e) * L[e][l],
    q_l(e) = current spatial width at l / width at l in the ordinary 32x32 net.

So an input reduction gives every one of the 19 sites ``q = r/32``; a reduction
after the stem leaves the stem at ``q = 1`` and gives the other 18 ``q = r/32``;
``R32`` gives ``q = 1`` everywhere.  This is a convention, not a claim of exact
spectral equivalence between filters at different resolutions.

``q_l`` is deliberately **not** folded into the table: it depends on the
resolution schedule, which is a separate experimental axis, and the ``stem_*``
reductions make it site-dependent in a way a table author should not have to
replicate.  Keeping it at call time makes one table portable across resolution
arms.

The table has three constructors, and they are mutually exclusive:

``levels``
    the historical per-epoch scalar; every site gets ``L[e][l] = levels[e]``.
``site_coeffs``
    a length-19 vector ``c_l``, giving ``L[e][l] = c_l * levels[e]``.  This is
    the separable per-layer homotopy: one shared time schedule, fixed per-layer
    weights.  ``c_l = 1`` reproduces the ``levels`` path bitwise.
``sigma_table``
    an explicit ``[epochs][19]`` table, for schedules that are *not* separable
    (a per-layer annealing front, say).  Same internal representation, so this
    is a configuration change rather than a second code path.

Whichever constructor is used, ``apply_at`` reads the table.  There is exactly
one code path and no scalar fast path to diverge from it.

**Gmix.**  An identity-to-Gaussian mixture at all 19 sites,

    T_alpha(h) = (1 - alpha) h + alpha * G_{q_l(e)}(h),

with the reference Gaussian width fixed at 1 so ``q_l(e)`` accounts only for the
current resolution.  ``alpha = 0`` returns ``h`` without evaluating the Gaussian;
``alpha = 1`` reproduces the corresponding Gaussian output exactly.  It replaces
the usual Gaussian at each site -- it does not add a second filter, duplicate the
network, or mix logits.  Its alpha levels are **not** claimed to be
strength-matched to the sigma levels of Gplateau.  For gmix the table entries are
mixture weights, so a per-site coefficient scales ``alpha`` and not a width.
"""
from __future__ import annotations

import hashlib
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transforms.gaussian import GaussianSmoothing

N_SITES = 19                      # stem + 2 per BasicBlock x 9 blocks
EARLY7 = tuple(range(7))          # stem + the six stage-1 main-path convolutions
REDUCTIONS = ("input_bilinear", "input_max", "stem_bilinear", "stem_max")

#: Residual stage owning each site, in the hook order established by
#: :func:`attach_sites`: site 0 is the stem and sites 1..6 are stage 1, which
#: share a spatial width, so they share a stage index here.
STAGE_OF_SITE = (0,) * 7 + (1,) * 6 + (2,) * 6

#: Spatial width each site sees in the ordinary 32x32 ResNet-20.
SITE_WIDTH_AT_32 = (32,) * 7 + (16,) * 6 + (8,) * 6

#: Table entries are rounded to this many decimals at construction.  The
#: Gaussian kernel cache is keyed on sigma, so an unrounded free-form table
#: would grow that cache without bound across a run; rounding also makes the
#: table digest stable across processes.
LEVEL_DECIMALS = 6


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
# per-site coefficient profiles
# --------------------------------------------------------------------------

def per_stage_coeffs(rho: float, normalize: bool = True) -> list:
    """Per-site coefficients ``c_l = rho ** stage(l)``, optionally normalized.

    Normalizing so ``max_l c_l == 1`` keeps every scheduled width inside the
    configured ``sigma_max`` without enlarging the kernel: the fixed support is
    ``radius = ceil(truncate * sigma_max)``, so raising ``sigma_max`` to make room
    for a coefficient above 1 would grow the kernel, the cost and the border
    effects, and invalidate the recorded operator checks.

    ``rho = 1`` returns all-ones and therefore reproduces the uniform schedule.
    ``rho = 0.5`` returns ``(1, 1/2, 1/4)`` across stem+stage1 / stage2 / stage3,
    which is exactly ``width_l / 32`` -- the constant-physical-scale profile.
    """
    rho = float(rho)
    if rho < 0 or not math.isfinite(rho):
        raise ValueError("rho must be a finite value >= 0, got %r" % (rho,))
    raw = [rho ** STAGE_OF_SITE[l] for l in range(N_SITES)]
    top = max(raw)
    if normalize and top > 0:
        raw = [v / top for v in raw]
    return [round(float(v), LEVEL_DECIMALS) for v in raw]


def build_level_table(levels=None, site_coeffs=None, sigma_table=None) -> list:
    """Assemble the ``[epochs][19]`` level table from exactly one constructor."""
    given = [k for k, v in (("levels", levels), ("site_coeffs", site_coeffs),
                            ("sigma_table", sigma_table)) if v is not None]
    if sigma_table is not None and given != ["sigma_table"]:
        raise ValueError("sigma_table is exclusive with levels/site_coeffs, got %s"
                         % (given,))
    if site_coeffs is not None and levels is None:
        raise ValueError("site_coeffs needs levels: it scales a shared schedule")

    if sigma_table is not None:
        rows = [list(r) for r in sigma_table]
        if not rows:
            raise ValueError("sigma_table must have at least one epoch row")
        for e, row in enumerate(rows):
            if len(row) != N_SITES:
                raise ValueError("sigma_table row %d has %d entries, need %d"
                                 % (e, len(row), N_SITES))
        table = [[float(v) for v in row] for row in rows]
    elif levels is not None:
        base = [float(v) for v in levels]
        if not base:
            raise ValueError("levels must have at least one epoch")
        c = ([1.0] * N_SITES if site_coeffs is None
             else [float(v) for v in site_coeffs])
        if len(c) != N_SITES:
            raise ValueError("site_coeffs has %d entries, need %d"
                             % (len(c), N_SITES))
        table = [[c[l] * g for l in range(N_SITES)] for g in base]
    else:
        return None

    for e, row in enumerate(table):
        for l, v in enumerate(row):
            if not math.isfinite(v) or v < 0:
                raise ValueError("level at epoch %d site %d must be finite and "
                                 ">= 0, got %r" % (e, l, v))
    return [[round(v, LEVEL_DECIMALS) for v in row] for row in table]


def table_digest(table) -> str | None:
    """sha256 of the rounded table, for the pairing digest set."""
    if table is None:
        return None
    blob = json.dumps(table, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# per-site controller
# --------------------------------------------------------------------------

class SiteController:
    """Applies the scheduled operator, with its own width, at each site.

    ``operator`` is ``"none"``, ``"gaussian"`` or ``"gmix"``.  The authoritative
    state is :attr:`row`, the length-19 vector of levels held for the current
    epoch; :attr:`value` is a **derived** scalar kept for logging and is ``None``
    whenever the row is not uniform.  Anything that saves and restores controller
    state must save :attr:`row`, not :attr:`value`.

    ``require_terminal_identity`` asserts that the last ``k`` epoch rows are
    exactly zero at every filtered site, i.e. that the run really does end on the
    target objective.  It defaults to 0 because timing probes legitimately hold a
    constant nonzero level for their whole (never-trained) schedule; every real
    training run should set it to at least 1.
    """

    def __init__(self, operator: str, levels=None, sites=None,
                 resolution_by_epoch=None, reduction="input_bilinear",
                 site_coeffs=None, sigma_table=None, profile=None,
                 require_terminal_identity: int = 0):
        if operator not in ("none", "gaussian", "gmix"):
            raise ValueError("unknown operator %r" % (operator,))
        self.operator = operator
        self.sites = tuple(range(N_SITES)) if sites is None else tuple(sites)
        for s in self.sites:
            if not (0 <= int(s) < N_SITES):
                raise ValueError("site %r outside 0..%d" % (s, N_SITES - 1))
        self.resolution_by_epoch = ([int(v) for v in resolution_by_epoch]
                                    if resolution_by_epoch else None)
        self.reduction = reduction
        self.gauss = (GaussianSmoothing(sigma_max=1.0, truncate=4.0)
                      if operator in ("gaussian", "gmix") else None)

        self.L = build_level_table(levels, site_coeffs, sigma_table)
        self.site_coeffs = (None if site_coeffs is None
                            else [round(float(v), LEVEL_DECIMALS) for v in site_coeffs])
        self.profile = profile
        self.require_terminal_identity = int(require_terminal_identity)

        # historical attribute, still read by callers that only ever used scalars
        self.levels = [float(v) for v in levels] if levels else None

        self.row = None                   # per-site levels for the current epoch
        self.resolution = None
        self.bypass_all = False           # target-path evaluation
        self.q = [1.0] * N_SITES

        self._check_terminal_identity()
        self._check_parameter_bounds()

    # -- validation -------------------------------------------------------

    def _epoch_span(self) -> int:
        n = len(self.L) if self.L else 0
        if self.resolution_by_epoch:
            n = max(n, len(self.resolution_by_epoch))
        return n

    def _check_terminal_identity(self):
        k = self.require_terminal_identity
        if k <= 0 or self.L is None or self.operator == "none":
            return
        if k > len(self.L):
            raise ValueError("require_terminal_identity=%d exceeds the %d "
                             "scheduled epochs" % (k, len(self.L)))
        for e in range(len(self.L) - k, len(self.L)):
            for l in self.sites:
                if self.L[e][l] != 0.0:
                    raise ValueError(
                        "require_terminal_identity=%d: epoch %d site %d holds "
                        "level %g, so the run never reaches the target objective. "
                        "A schedule that only decays does not arrive at the "
                        "endpoint by itself -- add an explicit final identity "
                        "phase." % (k, e, l, self.L[e][l]))

    def _check_parameter_bounds(self):
        """Reject an out-of-range schedule at build time, not at first forward.

        ``GaussianSmoothing.validate_parameter`` would raise on the first batch,
        which on a remote worker means the run is already scheduled and paid for.
        """
        if self.L is None or self.operator == "none":
            return
        span = self._epoch_span()
        if self.operator == "gmix":
            worst = (0.0, None, None)
            for e in range(span):
                row = self.L[min(e, len(self.L) - 1)]
                for l in self.sites:
                    if row[l] > worst[0]:
                        worst = (row[l], e, l)
            if worst[0] > 1.0 + 1e-12:
                raise ValueError("gmix alpha must lie in [0,1]; epoch %d site %d "
                                 "holds %g" % (worst[1], worst[2], worst[0]))
            return
        smax = self.gauss.sigma_max
        worst = (0.0, None, None)
        for e in range(span):
            row = self.L[min(e, len(self.L) - 1)]
            q = self.q_for(self._resolution_at(e))
            for l in self.sites:
                v = q[l] * row[l]
                if v > worst[0]:
                    worst = (v, e, l)
        if worst[0] > smax + 1e-12:
            raise ValueError(
                "scheduled sigma %g at epoch %d site %d exceeds sigma_max=%g. "
                "The fixed support (radius=%d) was sized for sigma_max; normalize "
                "the per-site coefficients so max_l c_l == 1 rather than enlarging "
                "the kernel." % (worst[0], worst[1], worst[2], smax,
                                 self.gauss.radius))

    # -- schedule ---------------------------------------------------------

    def _resolution_at(self, e: int):
        if not self.resolution_by_epoch:
            return None
        return self.resolution_by_epoch[
            max(0, min(int(e), len(self.resolution_by_epoch) - 1))]

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
        if self.L is not None:
            i = max(0, min(int(e), len(self.L) - 1))
            self.row = list(self.L[i])
        self.resolution = self._resolution_at(e)
        self.q = self.q_for(self.resolution)
        return self.value

    def set_state(self, level, resolution):
        """Force an explicit uniform (level, resolution) -- used by evaluation."""
        self.row = None if level is None else [float(level)] * N_SITES
        self.resolution = resolution
        self.q = self.q_for(resolution)
        return self.value

    def set_state_row(self, row, resolution):
        """Force an explicit per-site row -- for per-site evaluation."""
        if row is not None and len(row) != N_SITES:
            raise ValueError("row has %d entries, need %d" % (len(row), N_SITES))
        self.row = None if row is None else [float(v) for v in row]
        self.resolution = resolution
        self.q = self.q_for(resolution)
        return self.value

    # -- derived scalars --------------------------------------------------

    @property
    def value(self):
        """The current level when it is uniform across filtered sites, else None."""
        if self.row is None:
            return None
        vals = {self.row[l] for l in self.sites}
        return vals.pop() if len(vals) == 1 else None

    @property
    def max_level(self):
        if self.row is None:
            return None
        return max((self.row[l] for l in self.sites), default=0.0)

    def is_active(self) -> bool:
        """True when the operator is not an exact identity at any filtered site."""
        if self.bypass_all or self.operator == "none" or self.row is None:
            return False
        return any(self.row[l] > 0.0 for l in self.sites)

    # -- application ------------------------------------------------------

    def apply_at(self, site: int, h: torch.Tensor) -> torch.Tensor:
        if not self.is_active() or site not in self.sites:
            return h                                  # exact bypass
        level = self.row[site]
        if level <= 0.0:
            return h                                  # per-site exact bypass
        q = self.q[site]
        if self.operator == "gaussian":
            return self.gauss(h, q * level)
        alpha = float(level)                          # gmix, reference width 1
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
                "sigma_rule": "sigma_l(e) = q_l(e) * L[e][l]; q_l = width_l / width_l@32",
                "gmix_rule": ("T_alpha(h) = (1-alpha) h + alpha G_{q_l}(h), "
                              "reference width 1; alpha=0 returns h unevaluated"),
                "site_coeffs": self.site_coeffs,
                "profile": self.profile,
                "level_table": self.L,
                "level_table_sha256": table_digest(self.L),
                "level_decimals": LEVEL_DECIMALS,
                "require_terminal_identity": self.require_terminal_identity,
                "stage_of_site": list(STAGE_OF_SITE),
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
