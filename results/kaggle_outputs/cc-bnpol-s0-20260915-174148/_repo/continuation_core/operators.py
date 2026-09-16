"""The two numerical operators the retained methods use.

Both reproduce the executed benchmark code bit for bit (see
``verification/parity.py``):

* ``FixedSupportGaussian``  <- ``continuation.transforms.gaussian.GaussianSmoothing``
  as built by ``campaign_ops.SiteController``: ``sigma_max = 1.0``,
  ``truncate = 4.0``;
* ``adaptive_max_reduce``   <- ``campaign_ops.reduce_spatial(..., "max")``.

Gaussian conventions
--------------------
* one fixed support for every sigma: ``radius = ceil(truncate * sigma_max) = 4``,
  ``K = 9`` taps;
* taps ``exp(-d^2 / (2 sigma^2))`` for ``d = -4..4``, computed in float64,
  divided by their sum, then cast to the tensor dtype;
* separable depthwise convolution, **width first, then height**, channels never
  mix;
* whole-sample reflection padding by ``radius`` on each axis: PyTorch's native
  ``reflect`` when ``radius < n``, otherwise an explicit index gather
  ``r_n(i) = min(m, P - m)``, ``m = i mod P``, ``P = 2(n-1)``.  With radius 4 the
  explicit path is taken on 4x4 maps (stage 3 when the reduction is at 16);
* ``sigma == 0`` returns the input object itself: no kernel, no padding;
* ``sigma > sigma_max`` is an error rather than a silently re-truncated kernel.

Adaptive max pooling
--------------------
PyTorch's windows are ``[floor(i*n/r), ceil((i+1)*n/r))``.  32 -> 16 gives 16
disjoint width-2 windows; 32 -> 24 gives width-2 windows of which consecutive
ones **overlap** (16 overlapping pairs).  ``r == n`` returns the input object.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gaussian_kernel_1d(sigma: float, radius: int, dtype=torch.float32, device=None):
    d = torch.arange(-radius, radius + 1, dtype=torch.float64, device=device)
    if sigma == 0:
        k = (d == 0).to(torch.float64)
    else:
        k = torch.exp(-(d ** 2) / (2.0 * float(sigma) ** 2))
    k = k / k.sum()
    return k.to(dtype)


def reflected_indices(n: int, pad: int, device=None) -> torch.Tensor:
    i = torch.arange(-pad, n + pad, device=device)
    if n == 1:
        return torch.zeros_like(i)
    period = 2 * (n - 1)
    m = torch.remainder(i, period)
    return torch.minimum(m, period - m)


def reflect_pad_axis(x: torch.Tensor, pad: int, dim: int) -> torch.Tensor:
    if pad == 0:
        return x
    n = x.shape[dim]
    if pad < n:
        pad4 = (pad, pad, 0, 0) if dim in (-1, x.dim() - 1) else (0, 0, pad, pad)
        return F.pad(x, pad4, mode="reflect")
    return torch.index_select(x, dim, reflected_indices(n, pad, device=x.device))


class FixedSupportGaussian:
    """Separable Gaussian on ``[N, C, H, W]`` float tensors, fixed support."""

    def __init__(self, sigma_max: float = 1.0, truncate: float = 4.0):
        if sigma_max < 0 or truncate <= 0:
            raise ValueError("need sigma_max >= 0 and truncate > 0")
        self.sigma_max = float(sigma_max)
        self.truncate = float(truncate)
        self.radius = int(math.ceil(self.truncate * self.sigma_max))
        self.kernel_size = 2 * self.radius + 1
        self._cache: dict = {}

    def kernel(self, sigma: float, dtype, device):
        key = (float(sigma), str(dtype), str(device))
        k = self._cache.get(key)
        if k is None:
            k = gaussian_kernel_1d(sigma, self.radius, dtype=dtype, device=device)
            self._cache[key] = k
        return k

    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        if x.dim() != 4 or not x.is_floating_point():
            raise ValueError("expected a float [N,C,H,W] tensor, got %s %s"
                             % (tuple(x.shape), x.dtype))
        sigma = float(sigma)
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be finite and >= 0, got %r" % sigma)
        if sigma > self.sigma_max + 1e-12:
            raise ValueError("sigma=%g exceeds sigma_max=%g (fixed radius %d)"
                             % (sigma, self.sigma_max, self.radius))
        if sigma == 0.0:
            return x
        c = x.shape[1]
        k = self.kernel(sigma, x.dtype, x.device)
        kx = k.view(1, 1, 1, -1).expand(c, 1, 1, self.kernel_size)
        ky = k.view(1, 1, -1, 1).expand(c, 1, self.kernel_size, 1)
        y = F.conv2d(reflect_pad_axis(x, self.radius, -1), kx, groups=c)
        return F.conv2d(reflect_pad_axis(y, self.radius, -2), ky, groups=c)

    def describe(self) -> dict:
        return {"sigma_max": self.sigma_max, "truncate": self.truncate,
                "radius": self.radius, "kernel_size": self.kernel_size,
                "taps": "exp(-d^2/(2 sigma^2)) in float64, normalised to sum 1, "
                        "cast to tensor dtype",
                "order": "depthwise width pass then height pass",
                "padding": "whole-sample reflection; native when radius < n, "
                           "explicit index gather otherwise",
                "identity": "sigma == 0 returns the input unchanged"}


def adaptive_windows(n_in: int, n_out: int):
    return [(int(math.floor(i * n_in / n_out)), int(math.ceil((i + 1) * n_in / n_out)))
            for i in range(n_out)]


def adaptive_max_reduce(x: torch.Tensor, r) -> torch.Tensor:
    """``r x r`` adaptive max pooling; exact bypass when ``r`` is None or equal."""
    if r is None or int(r) == int(x.shape[-1]):
        return x
    r = int(r)
    return F.adaptive_max_pool2d(x, (r, r))
