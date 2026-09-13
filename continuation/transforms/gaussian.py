r"""Gaussian input smoothing: the first transformation family.

Mathematics
-----------
``T_sigma x = G_sigma * x`` with ``T_0 x = x``.  The normalized isotropic
Gaussian acts spatially and independently on each colour channel.  ``sigma`` is
reported in **pixel units**.

For the continuous Fourier transform in angular frequency ``w``,

    F[T_sigma x](w) = exp(-sigma^2 |w|^2 / 2) * F[x](w),

and the associated heat time is ``a = sigma^2 / 2``, giving the multiplier
``exp(-a |w|^2)``.  The *continuous* heat operators satisfy the semigroup
identity ``T_{a1} T_{a2} = T_{a1+a2}``.  **This identity is not claimed for the
finite, truncated, reflection-padded discrete filter implemented here**; it only
motivates the alternative heat-time schedule parameterization in
``continuation.schedules``.

Discrete implementation and its conventions
-------------------------------------------
* **Separable**: a 1-D kernel is applied along width then height, each as a
  depthwise convolution (``groups = C``).  Channels never mix.
* **One fixed spatial support** for the whole configured parameter range:
  ``radius = ceil(truncate * sigma_max)`` and ``K = 2*radius + 1`` taps, for
  every ``sigma`` in ``[0, sigma_max]``.  A per-sigma kernel-size rounding rule
  would inject extra discontinuities along the parameter path, so it is not
  used.  Asking for ``sigma > sigma_max`` is an error, not a silent retruncation.
* **Truncation renormalization**: taps are ``exp(-d^2 / (2 sigma^2))`` for
  ``d = -radius..radius``, divided by their sum.  The discrete kernel therefore
  sums to exactly 1 (up to float rounding) and preserves constant images.
* **Reflection padding** by ``radius`` on each side, so the output has the
  input's spatial size.  ``torch.nn.functional.pad(mode="reflect")`` is used when
  it applies; it requires ``pad < axis length``, so for small feature maps (e.g.
  a radius-4 kernel on a 4x4 map) an explicit whole-sample reflection gather is
  used instead -- see :func:`reflected_indices`.  Both give identical values
  where both are defined, and both are differentiable.
  Reflection is exact for constant images and introduces the usual mirror-symmetry
  bias near borders: structure within ``radius`` pixels of an edge is smoothed
  against its own mirror image rather than against unseen content.  With
  ``sigma_max = 3`` and ``truncate = 4`` the radius is 12 on 32x32 CIFAR images,
  so border effects are *not* negligible and are documented rather than hidden.
  PyTorch's reflect mode additionally requires ``radius < H`` and ``radius < W``.
* ``sigma == 0`` is handled as an **exact identity** (the tensor is returned
  unchanged), avoiding division by zero.  The family is also continuous there:
  as ``sigma -> 0+`` the normalized taps tend to a discrete delta.
* Images stay in floating point in ``[0,1]`` space.  Filtered training images are
  **never** requantized to 8-bit values.
* No hard Fourier cutoff or projection is used anywhere.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .base import ImageTransform, TransformResult, check_image_batch


def heat_time_from_sigma(sigma: float) -> float:
    """``a = sigma^2 / 2``."""
    return float(sigma) ** 2 / 2.0


def sigma_from_heat_time(a: float) -> float:
    """``sigma = sqrt(2a)``."""
    if a < 0:
        raise ValueError("heat time a must be >= 0, got %r" % (a,))
    return math.sqrt(2.0 * float(a))


def gaussian_kernel_1d(sigma: float, radius: int, dtype=torch.float32, device=None) -> torch.Tensor:
    """Normalized 1-D Gaussian taps on the fixed support ``[-radius, radius]``.

    ``sigma == 0`` returns a discrete delta.
    """
    if radius < 0:
        raise ValueError("radius must be >= 0")
    d = torch.arange(-radius, radius + 1, dtype=torch.float64, device=device)
    if sigma == 0:
        k = (d == 0).to(torch.float64)
    else:
        k = torch.exp(-(d ** 2) / (2.0 * float(sigma) ** 2))
    k = k / k.sum()
    return k.to(dtype)


def reflected_indices(n: int, pad: int, device=None) -> torch.Tensor:
    """Whole-sample reflection indices for ``i = -pad .. n+pad-1``.

    With ``P = 2(n-1)`` and ``m = i mod P``, the reflected index is
    ``r_n(i) = min(m, P - m)``: reflection *without* repeating the boundary
    sample.  For ``n = 4, pad = 4`` this yields
    ``[2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1]``.

    PyTorch's native ``mode="reflect"`` requires ``pad < n``; this gather has no
    such limit, so a radius-4 kernel remains well defined on a 4x4 feature map.
    """
    if n < 1:
        raise ValueError("axis length must be >= 1, got %d" % n)
    i = torch.arange(-pad, n + pad, device=device)
    if n == 1:
        return torch.zeros_like(i)
    period = 2 * (n - 1)
    m = torch.remainder(i, period)
    return torch.minimum(m, period - m)


def reflect_pad_axis(x: torch.Tensor, pad: int, dim: int) -> torch.Tensor:
    """Reflection-pad one spatial axis by ``pad`` on both sides.

    Uses PyTorch's native reflect padding when it is supported (``pad < n``) and
    an explicit ``index_select`` gather otherwise.  Both paths are differentiable
    and produce the same values wherever both are available.
    """
    if pad == 0:
        return x
    n = x.shape[dim]
    if pad < n:
        pad4 = (pad, pad, 0, 0) if dim in (-1, x.dim() - 1) else (0, 0, pad, pad)
        return F.pad(x, pad4, mode="reflect")
    idx = reflected_indices(n, pad, device=x.device)
    return torch.index_select(x, dim, idx)


# --------------------------------------------------------------------------
# Differentiable-in-sigma path (sensitivity probe only)
# --------------------------------------------------------------------------
#
# The float path above is the hot path and is deliberately left untouched: it
# caches kernels, coerces sigma with ``float()`` and short-circuits sigma == 0 to
# an exact identity object, all of which the recorded bitwise-identity audits
# depend on.  The two functions here are a *parallel* path used only to measure
# ``dL/dsigma``.  They never cache -- a cached kernel would carry a graph from a
# previous iteration -- and they never call ``float()``.
#
# Differentiating this operator is unusually well conditioned, for a reason that
# predates any interest in autograd: the support is fixed by ``sigma_max`` and
# not by the current sigma (see ``GaussianSmoothing.__init__``), so ``radius``
# and ``kernel_size`` do not depend on sigma and there is no ``ceil()`` step
# discontinuity along the parameter path.  Two consequences worth asserting:
#
# * because the taps are renormalized, ``sum(k) == 1`` identically in sigma, so
#   ``sum(dk/dsigma) == 0`` exactly -- the derivative can only redistribute mass,
#   never change the DC gain;
# * the gradient is that of the **truncated, renormalized** kernel actually
#   implemented here, which differs from the ideal Gaussian's by O(tail).  Check
#   it against finite differences of this operator, never against a closed form
#   for the ideal Gaussian.
#
# Not valid at sigma = 0: the off-centre taps underflow to exactly zero well
# before that (below sigma ~ 0.075 in float32, ~0.026 in float64), leaving the
# kernel and its derivative identically constant.  Callers must keep sigma
# strictly inside that flat region's upper edge; ``continuation.adaptive``
# enforces a floor and snaps to the exact identity below it.


def gaussian_kernel_1d_tensor(sigma: torch.Tensor, radius: int) -> torch.Tensor:
    """Normalized taps on the fixed support, differentiable w.r.t. ``sigma``.

    ``sigma`` is a 0-dim tensor and must be strictly positive; there is no
    ``sigma == 0`` branch here precisely because that branch is what breaks the
    graph in the float path.
    """
    if radius < 0:
        raise ValueError("radius must be >= 0")
    if not torch.is_tensor(sigma):
        raise TypeError("gaussian_kernel_1d_tensor needs a tensor sigma; the "
                        "float path is GaussianSmoothing.kernel")
    d = torch.arange(-radius, radius + 1, dtype=sigma.dtype, device=sigma.device)
    k = torch.exp(-(d ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def blur_with_sigma_grad(x: torch.Tensor, sigma: torch.Tensor,
                         radius: int) -> torch.Tensor:
    """Separable reflection-padded blur carrying a gradient into ``sigma``.

    Same arithmetic as :meth:`GaussianSmoothing.apply`, minus the cache and the
    identity short-circuit.  The kernel is cast to ``x``'s dtype through a
    differentiable cast, so a float64 probe against a float32 activation still
    propagates.
    """
    check_image_batch(x)
    c = x.shape[1]
    ks = 2 * radius + 1
    k = gaussian_kernel_1d_tensor(sigma, radius).to(x.dtype)
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, ks)
    ky = k.view(1, 1, -1, 1).expand(c, 1, ks, 1)
    y = F.conv2d(reflect_pad_axis(x, radius, -1), kx, groups=c)
    return F.conv2d(reflect_pad_axis(y, radius, -2), ky, groups=c)


class GaussianSmoothing(ImageTransform):
    """Deterministic separable Gaussian blur with a fixed support.

    Parameters
    ----------
    sigma_max : float
        Largest ``sigma`` the configured support must accommodate, in pixels.
    truncate : float
        Support is ``radius = ceil(truncate * sigma_max)`` standard deviations.
    padding : str
        Currently ``"reflect"`` only (documented above).
    """

    name = "gaussian"
    parameter_name = "sigma"
    parameter_units = "pixels"

    def __init__(self, sigma_max: float = 3.0, truncate: float = 4.0,
                 padding: str = "reflect"):
        if sigma_max < 0:
            raise ValueError("sigma_max must be >= 0, got %r" % (sigma_max,))
        if truncate <= 0:
            raise ValueError("truncate must be > 0, got %r" % (truncate,))
        if padding != "reflect":
            raise ValueError("only padding='reflect' is implemented, got %r" % (padding,))
        self.sigma_max = float(sigma_max)
        self.truncate = float(truncate)
        self.padding = padding
        self.radius = int(math.ceil(self.truncate * self.sigma_max))
        self.kernel_size = 2 * self.radius + 1
        self._cache: dict = {}

    # -- interface ----------------------------------------------------------

    @property
    def target_parameter(self) -> float:
        return 0.0

    def validate_parameter(self, eta: float) -> float:
        sigma = float(eta)
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be a finite value >= 0, got %r" % (eta,))
        if sigma > self.sigma_max + 1e-12:
            raise ValueError(
                "sigma=%g exceeds the configured sigma_max=%g; the fixed support "
                "(radius=%d) was sized for sigma_max. Raise sigma_max in the transform "
                "config rather than silently retruncating the kernel."
                % (sigma, self.sigma_max, self.radius)
            )
        return sigma

    def config_signature(self) -> dict:
        return {
            "sigma_max": self.sigma_max,
            "truncate": self.truncate,
            "padding": self.padding,
            "radius": self.radius,
            "kernel_size": self.kernel_size,
            "separable": True,
            "normalized": "taps renormalized to sum 1 on the fixed support",
            "identity_at": "sigma == 0 (exact passthrough)",
            "value_space": "float [0,1], pre-normalization, no requantization",
        }

    # -- computation --------------------------------------------------------

    def kernel(self, sigma: float, dtype=torch.float32, device=None) -> torch.Tensor:
        key = (float(sigma), str(dtype), str(device))
        k = self._cache.get(key)
        if k is None:
            k = gaussian_kernel_1d(sigma, self.radius, dtype=dtype, device=device)
            self._cache[key] = k
        return k

    def apply(self, x: torch.Tensor, eta: float, meta: dict | None = None) -> TransformResult:
        check_image_batch(x)
        sigma = self.validate_parameter(eta)
        info = {
            "family": self.name,
            "sigma": sigma,
            "heat_time": heat_time_from_sigma(sigma),
            "radius": self.radius,
            "kernel_size": self.kernel_size,
            "padding": self.padding,
            "identity": sigma == 0.0,
        }
        if sigma == 0.0:
            # Exact identity endpoint: no kernel, no padding, no residual blur.
            return TransformResult(x, info)

        n, c, h, w = x.shape
        k = self.kernel(sigma, dtype=x.dtype, device=x.device)
        kx = k.view(1, 1, 1, -1).expand(c, 1, 1, self.kernel_size)
        ky = k.view(1, 1, -1, 1).expand(c, 1, self.kernel_size, 1)
        # Pad each axis separately: native reflect where supported, explicit
        # whole-sample reflection indexing where the radius reaches or exceeds
        # the axis length (small feature maps).  Convolution adds no padding.
        y = F.conv2d(reflect_pad_axis(x, self.radius, -1), kx, groups=c)
        y = F.conv2d(reflect_pad_axis(y, self.radius, -2), ky, groups=c)
        info["padding_path"] = {
            "w": "native" if self.radius < w else "explicit_reflect_index",
            "h": "native" if self.radius < h else "explicit_reflect_index"}
        return TransformResult(y, info)


def build(params: dict) -> GaussianSmoothing:
    known = {"sigma_max", "truncate", "padding"}
    unknown = sorted(set(params or {}) - known)
    if unknown:
        raise KeyError("unknown gaussian transform params %s; known: %s"
                       % (unknown, sorted(known)))
    return GaussianSmoothing(**(params or {}))
