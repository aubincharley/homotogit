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
* **Reflection padding** (``torch.nn.functional.pad(mode="reflect")``) by
  ``radius`` on each side, so the output has the input's spatial size.
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
        if self.radius >= h or self.radius >= w:
            raise ValueError(
                "reflection padding radius %d must be smaller than the image size "
                "(H=%d, W=%d); reduce sigma_max or truncate." % (self.radius, h, w)
            )
        k = self.kernel(sigma, dtype=x.dtype, device=x.device)
        kx = k.view(1, 1, 1, -1).expand(c, 1, 1, self.kernel_size)
        ky = k.view(1, 1, -1, 1).expand(c, 1, self.kernel_size, 1)
        y = F.pad(x, (self.radius, self.radius, 0, 0), mode="reflect")
        y = F.conv2d(y, kx, groups=c)
        y = F.pad(y, (0, 0, self.radius, self.radius), mode="reflect")
        y = F.conv2d(y, ky, groups=c)
        return TransformResult(y, info)


def build(params: dict) -> GaussianSmoothing:
    known = {"sigma_max", "truncate", "padding"}
    unknown = sorted(set(params or {}) - known)
    if unknown:
        raise KeyError("unknown gaussian transform params %s; known: %s"
                       % (unknown, sorted(known)))
    return GaussianSmoothing(**(params or {}))
