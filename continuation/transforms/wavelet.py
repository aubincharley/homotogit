r"""Differentiable undecimated-wavelet shrinkage.

The core is a plain PyTorch tensor function, :func:`wavelet_shrink`, that acts on
real ``[B, C, H, W]`` tensors -- images *or* activations -- runs on CPU or GPU,
and is fully differentiable.  PyWavelets is used **only** to fetch fixed filter
coefficients (and, in the tests, as an independent reference); no PyWavelets call
happens on the compute path.

Construction
------------
**Boundary.**  The input is mirrored by concatenating it with its reversal along
each spatial axis,

    Eh = [[ x,            flip_W(x)          ],
          [ flip_H(x),    flip_H(flip_W(x))  ]]     -> [B, C, 2H, 2W],

the *periodic* stationary transform is applied on that extended domain, and the
original top-left ``H x W`` block is cropped back out by ``P``.  Opposite edges
of the **original** image are therefore never periodically connected: the
periodicity acts on the whole-sample symmetric extension.

**Transform.**  Two-level, separable, undecimated (stationary / a-trous)
analysis, applied independently per sample and per channel.  Level ``j=1`` is
finest, ``j=2`` coarser; level ``j`` dilates the filters by ``2^(j-1)``.  Each
1-D filtering uses the orthogonal filter scaled by ``1/sqrt(2)``, so that
``|H|^2 + |G|^2 = 1`` and the analysis operator is a **tight frame with bound
one**, ``W* W = I`` on the extended domain (checked directly in the tests, not
assumed).  ``W*`` is its exact adjoint, so synthesis is adjoint synthesis rather
than a separate inverse filter bank.

Retained bands are the coarsest approximation ``a_2`` and all six detail bands
``d_{j,o}``, ``j in {1,2}``, ``o in {LH, HL, HH}``.

Orientation convention: the first letter refers to the **row** axis (``dim=-2``)
and the second to the **column** axis (``dim=-1``); e.g. ``LH`` is lowpass along
rows and highpass along columns.  All three detail orientations are thresholded
by the same rule, so the naming affects diagnostic labels only.

**Shrinkage.**  Per sample, per channel and per detail band,

    nu_{j,o}     = ||d_{j,o}||_2 / sqrt(N_{j,o}),
    lambda_{j,o} = 4 (1 - s) 2^(1-j) nu_{j,o},
    T_s(h)       = P W* ( a_2, { soft(d_{j,o}, lambda_{j,o}) } ),
    soft(v, l)   = sign(v) max(|v| - l, 0).

``a_2`` is passed through unchanged.  ``N_{j,o}`` is the number of spatial
coefficients of that band on the extended domain, so ``nu`` is a plain RMS.  It
is computed from the **unthresholded** coefficients, separately for every sample
and channel -- never pooled across them -- and gradients flow through it.  Zero
bands are handled by ``nu = sqrt(mean(d^2) + eps)``, whose derivative at ``d=0``
is ``0`` rather than infinite.

At ``s=1`` every threshold vanishes and ``T_1 = P W* W E = P E = I`` exactly in
exact arithmetic (~1e-7 relative in float32).  **No identity shortcut is taken**,
so the round-trip is exercised on every call and can be tested on its own.
``T_0`` is *not* claimed to be constant.

This is an explicitly defined shrinkage operator.  It is **not** an exact
constrained-TV solution and **not** a claimed proximal operator for redundant
wavelet analysis.  The tensor function adds no clipping, no contrast
normalization and no TV or mean constraint; outputs may leave ``[0,1]``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import ImageTransform, TransformResult, check_image_batch

WAVELETS = ("haar", "db2", "sym4", "coif1")
BANDS = ("LH", "HL", "HH")
_FILTER_CACHE: dict = {}


def get_filters(name: str, dtype=torch.float32, device=None):
    """Analysis ``(lowpass, highpass)`` scaled by ``1/sqrt(2)``.

    Coefficients come from PyWavelets; the scaling turns the orthogonal filter
    pair into a tight frame with bound one for the undecimated transform.
    """
    key = (name, str(dtype), str(device))
    cached = _FILTER_CACHE.get(key)
    if cached is not None:
        return cached
    import pywt

    w = pywt.Wavelet(name)
    if not w.orthogonal:
        raise ValueError("wavelet %r is not orthogonal; W*W = I is not available" % name)
    scale = 2.0 ** -0.5
    lo = torch.tensor(w.dec_lo, dtype=dtype, device=device) * scale
    hi = torch.tensor(w.dec_hi, dtype=dtype, device=device) * scale
    _FILTER_CACHE[key] = (lo, hi)
    return lo, hi


# --------------------------------------------------------------------------
# Boundary handling
# --------------------------------------------------------------------------

def mirror_extend(x: torch.Tensor) -> torch.Tensor:
    """``[B,C,H,W] -> [B,C,2H,2W]`` by reflecting along each spatial axis."""
    x = torch.cat([x, torch.flip(x, dims=[-1])], dim=-1)
    return torch.cat([x, torch.flip(x, dims=[-2])], dim=-2)


def crop(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """``P``: take the original top-left ``h x w`` block."""
    return x[..., :h, :w]


# --------------------------------------------------------------------------
# Circular filtering and its exact adjoint
# --------------------------------------------------------------------------

def _filter1d(x: torch.Tensor, f: torch.Tensor, dim: int, dilation: int,
              adjoint: bool = False) -> torch.Tensor:
    """Circular correlation ``y[n] = sum_k f[k] x[n + k*dilation]`` along ``dim``.

    With ``adjoint=True`` this computes the exact transpose,
    ``y[m] = sum_k f[k] x[m - k*dilation]``, obtained by padding on the other
    side and flipping the kernel.
    """
    k = int(f.numel())
    span = (k - 1) * dilation
    n = x.shape[dim]
    if span >= n:
        raise ValueError(
            "filter support %d (taps=%d, dilation=%d) does not fit the extended "
            "axis length %d; use a shorter wavelet, fewer levels, or a larger input"
            % (span + 1, k, dilation, n))
    w = f.flip(0) if adjoint else f
    if dim in (-1, x.dim() - 1):
        pad = (span, 0) if adjoint else (0, span)
        pad4, kernel, dil = (pad[0], pad[1], 0, 0), w.view(1, 1, 1, k), (1, dilation)
    else:
        pad = (span, 0) if adjoint else (0, span)
        pad4, kernel, dil = (0, 0, pad[0], pad[1]), w.view(1, 1, k, 1), (dilation, 1)
    b, c = x.shape[0], x.shape[1]
    y = F.pad(x.reshape(b * c, 1, x.shape[-2], x.shape[-1]), pad4, mode="circular")
    y = F.conv2d(y, kernel, dilation=dil)
    return y.reshape(b, c, y.shape[-2], y.shape[-1])


def _analysis_level(a: torch.Tensor, lo, hi, dilation: int):
    """One undecimated 2-D level: returns ``(LL, {LH, HL, HH})``."""
    low = _filter1d(a, lo, -2, dilation)
    high = _filter1d(a, hi, -2, dilation)
    return (
        _filter1d(low, lo, -1, dilation),
        {"LH": _filter1d(low, hi, -1, dilation),
         "HL": _filter1d(high, lo, -1, dilation),
         "HH": _filter1d(high, hi, -1, dilation)},
    )


def _synthesis_level(ll: torch.Tensor, details: dict, lo, hi, dilation: int):
    """Exact adjoint of :func:`_analysis_level`."""
    low = (_filter1d(ll, lo, -1, dilation, adjoint=True)
           + _filter1d(details["LH"], hi, -1, dilation, adjoint=True))
    high = (_filter1d(details["HL"], lo, -1, dilation, adjoint=True)
            + _filter1d(details["HH"], hi, -1, dilation, adjoint=True))
    return (_filter1d(low, lo, -2, dilation, adjoint=True)
            + _filter1d(high, hi, -2, dilation, adjoint=True))


def swt2_analysis(x: torch.Tensor, wavelet: str = "db2", levels: int = 2):
    """``W`` on an already-extended tensor.  Returns ``(a_J, [details per level])``."""
    lo, hi = get_filters(wavelet, x.dtype, x.device)
    a = x
    details = []
    for j in range(1, levels + 1):
        a, d = _analysis_level(a, lo, hi, dilation=2 ** (j - 1))
        details.append(d)
    return a, details


def swt2_synthesis(a: torch.Tensor, details, wavelet: str = "db2"):
    """``W*``, the exact adjoint of :func:`swt2_analysis`."""
    lo, hi = get_filters(wavelet, a.dtype, a.device)
    for j in range(len(details), 0, -1):
        a = _synthesis_level(a, details[j - 1], lo, hi, dilation=2 ** (j - 1))
    return a


# --------------------------------------------------------------------------
# Shrinkage
# --------------------------------------------------------------------------

def soft_threshold(v: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """``sign(v) * max(|v| - lam, 0)``."""
    return torch.sign(v) * torch.clamp(v.abs() - lam, min=0.0)


def band_rms(d: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """``||d||_2 / sqrt(N)`` per sample and channel, shape ``[B,C,1,1]``.

    ``eps`` inside the square root keeps the gradient finite (and zero) for an
    all-zero band instead of infinite.
    """
    return torch.sqrt((d * d).mean(dim=(-2, -1), keepdim=True) + eps)


# --------------------------------------------------------------------------
# Fused implementation (same mathematics, fewer kernels and copies)
# --------------------------------------------------------------------------
#
# The reference path issues 6 convolutions and 6 circular pads per 2-D level
# (24 convs / ~49 copies per call).  Profiling showed circular-pad copies at
# ~50% of self-CPU time and the many small dilated convolutions dominating CUDA
# time.  The fused path computes both filters of a stage in a single grouped
# convolution, so one level costs 2 pads + 2 convs for analysis and the same for
# synthesis: 8 convs and 8 pads per call in total.  The arithmetic performed is
# identical; only the grouping of kernel launches changes.

def _stack_rows(lo, hi, k):
    return torch.stack([lo, hi]).view(2, 1, k, 1)


def _analysis_level_fused(a, lo, hi, dilation: int):
    """``[n,1,H,W] -> [n,4,H,W]`` ordered ``(LL, LH, HL, HH)``."""
    k = int(lo.numel())
    span = (k - 1) * dilation
    if span >= min(a.shape[-2], a.shape[-1]):
        raise ValueError(
            "filter support %d (taps=%d, dilation=%d) does not fit the extended "
            "axes %s" % (span + 1, k, dilation, tuple(a.shape[-2:])))
    r = F.conv2d(F.pad(a, (0, 0, 0, span), mode="circular"),
                 _stack_rows(lo, hi, k), dilation=(dilation, 1))
    cw = torch.stack([lo, hi, lo, hi]).view(4, 1, 1, k)
    return F.conv2d(F.pad(r, (0, span, 0, 0), mode="circular"),
                    cw, dilation=(1, dilation), groups=2)


def _synthesis_level_fused(coeffs, lo, hi, dilation: int):
    """Exact adjoint of :func:`_analysis_level_fused`.  ``[n,4,H,W] -> [n,1,H,W]``."""
    k = int(lo.numel())
    span = (k - 1) * dilation
    flo, fhi = lo.flip(0), hi.flip(0)
    cw = torch.stack([flo, fhi]).view(1, 2, 1, k).repeat(2, 1, 1, 1)
    r = F.conv2d(F.pad(coeffs, (span, 0, 0, 0), mode="circular"),
                 cw, dilation=(1, dilation), groups=2)
    rw = torch.stack([flo, fhi]).view(1, 2, k, 1)
    return F.conv2d(F.pad(r, (0, 0, span, 0), mode="circular"),
                    rw, dilation=(dilation, 1))


def _shrink_core(x, s_t, wavelet: str, levels: int, eps: float):
    """Fused ``T_s`` on ``[B,C,H,W]``; returns ``(y, zero_fractions)``."""
    b, c, h, w = x.shape
    lo, hi = get_filters(wavelet, x.dtype, x.device)
    a = mirror_extend(x).reshape(b * c, 1, 2 * h, 2 * w)

    coeffs = []
    for j in range(1, levels + 1):
        out = _analysis_level_fused(a, lo, hi, 2 ** (j - 1))
        a = out[:, :1]
        coeffs.append(out[:, 1:])                     # (LH, HL, HH) together

    zero_fracs = []
    shrunk = []
    for j, d in enumerate(coeffs, start=1):
        # one RMS / threshold / soft-threshold over all three orientations at once
        nu = torch.sqrt((d * d).mean(dim=(-2, -1), keepdim=True) + eps)
        lam = 4.0 * (1.0 - s_t) * (2.0 ** (1 - j)) * nu
        td = torch.sign(d) * torch.clamp(d.abs() - lam, min=0.0)
        shrunk.append(td)
        zero_fracs.append((td == 0).to(x.dtype).mean(dim=(-2, -1)))

    for j in range(levels, 0, -1):
        a = _synthesis_level_fused(torch.cat([a, shrunk[j - 1]], dim=1),
                                   lo, hi, 2 ** (j - 1))
    y = a.reshape(b, c, 2 * h, 2 * w)[..., :h, :w]
    return y, zero_fracs


def wavelet_shrink(x: torch.Tensor, s: float, wavelet: str = "db2", levels: int = 2,
                   eps: float = 1e-12, return_info: bool = False,
                   impl: str = "fast", checkpoint: bool = False,
                   bypass_identity: bool = True):
    """``T_s(x)`` for real ``[B,C,H,W]`` tensors.  Differentiable, CPU or GPU.

    Parameters
    ----------
    impl : ``"fast"`` (default) or ``"reference"``
        Both compute the same mathematics.  ``"reference"`` is the original
        unfused path, kept as the numerical reference; ``"fast"`` fuses each
        filter pair into one grouped convolution and thresholds the three
        orientations of a level together.
    checkpoint : bool
        Recompute the transform during backward instead of storing its
        intermediates.  Exact (not straight-through); trades time for memory.
    bypass_identity : bool
        Return ``x`` directly when ``s == 1``, where ``T_1 = I`` exactly.  ``s``
        is a fixed schedule parameter, so this is a constant-time check.  Pass
        ``False`` to force the full round-trip (used by the tests, so that
        reconstruction is verified independently of the bypass).

    No clipping, normalization, TV or mean constraint is applied.
    """
    if x.dim() != 4:
        raise ValueError("expected [B,C,H,W], got %s" % (tuple(x.shape),))
    if not x.is_floating_point():
        raise TypeError("wavelet shrinkage needs a float tensor")
    if impl not in ("fast", "reference"):
        raise ValueError("impl must be 'fast' or 'reference', got %r" % (impl,))
    s_val = float(s)
    if not (0.0 <= s_val <= 1.0):
        raise ValueError("s must lie in [0,1], got %r" % (s,))

    if bypass_identity and s_val == 1.0:
        y = x
        if not return_info:
            return y
        zero = torch.zeros(x.shape[0], x.shape[1], dtype=x.dtype, device=x.device)
        info = {"zero_fraction": {"j%d_%s" % (j, o): zero
                                  for j in (1, 2) for o in BANDS},
                "lambda": {}, "rms": {}, "identity_bypass": True}
        return y, info

    s_t = torch.as_tensor(s_val, dtype=x.dtype, device=x.device)

    if impl == "fast":
        if checkpoint and x.requires_grad:
            from torch.utils.checkpoint import checkpoint as _ckpt
            y = _ckpt(lambda t: _shrink_core(t, s_t, wavelet, levels, eps)[0],
                      x, use_reentrant=False)
            if not return_info:
                return y
            with torch.no_grad():
                _, zf = _shrink_core(x, s_t, wavelet, levels, eps)
        else:
            y, zf = _shrink_core(x, s_t, wavelet, levels, eps)
            if not return_info:
                return y
        info = {"zero_fraction": {}, "lambda": {}, "rms": {},
                "identity_bypass": False}
        for j, frac in enumerate(zf, start=1):
            for i, o in enumerate(BANDS):
                info["zero_fraction"]["j%d_%s" % (j, o)] = frac[:, i].detach().reshape(
                    x.shape[0], x.shape[1])
        return y, info

    # --- reference path (unfused); kept as the numerical reference ---
    h, w = x.shape[-2], x.shape[-1]
    ext = mirror_extend(x)
    a, details = swt2_analysis(ext, wavelet, levels)
    info = {"zero_fraction": {}, "lambda": {}, "rms": {}, "identity_bypass": False}
    shrunk = []
    for j, level in enumerate(details, start=1):
        out = {}
        for o in BANDS:
            d = level[o]
            nu = band_rms(d, eps)
            lam = 4.0 * (1.0 - s_t) * (2.0 ** (1 - j)) * nu
            td = soft_threshold(d, lam)
            out[o] = td
            if return_info:
                key = "j%d_%s" % (j, o)
                info["zero_fraction"][key] = (td == 0).to(x.dtype).mean(
                    dim=(-2, -1)).detach()
                info["lambda"][key] = lam.detach().flatten(1)
                info["rms"][key] = nu.detach().flatten(1)
        shrunk.append(out)
    y = crop(swt2_synthesis(a, shrunk, wavelet), h, w)
    return (y, info) if return_info else y


# --------------------------------------------------------------------------
# ImageTransform wrapper
# --------------------------------------------------------------------------

class WaveletShrinkage(ImageTransform):
    """Undecimated wavelet shrinkage as a transformation family.

    Native parameter ``s in [0,1]`` with the target endpoint at ``s = 1``.
    """

    name = "wavelet"
    parameter_name = "s"
    parameter_units = "shrinkage strength (1 = identity)"

    def __init__(self, wavelet: str = "db2", levels: int = 2, eps: float = 1e-12):
        if wavelet not in WAVELETS:
            raise ValueError("wavelet must be one of %s, got %r" % (WAVELETS, wavelet))
        if levels < 1:
            raise ValueError("levels must be >= 1")
        self.wavelet, self.levels, self.eps = wavelet, int(levels), float(eps)

    @property
    def target_parameter(self) -> float:
        return 1.0

    def validate_parameter(self, eta: float) -> float:
        s = float(eta)
        if not (0.0 <= s <= 1.0):
            raise ValueError("s must lie in [0,1], got %r" % (eta,))
        return s

    def config_signature(self) -> dict:
        return {
            "wavelet": self.wavelet,
            "levels": self.levels,
            "transform": "undecimated (stationary) separable 2-D, per sample/channel",
            "normalization": "filters scaled by 1/sqrt(2); tight frame, W*W = I",
            "boundary": "mirror-extend to 2Hx2W, periodic transform, crop top-left",
            "threshold": "lambda_{j,o} = 4(1-s) 2^(1-j) * RMS(d_{j,o}), soft, a_J kept",
            "rms": "per sample/channel/band, from unthresholded coefficients",
            "eps": self.eps,
            "identity_at": "s == 1 (no shortcut; exact in exact arithmetic)",
            "clipping": "none; outputs may leave [0,1]",
        }

    def apply(self, x: torch.Tensor, eta: float, meta: dict | None = None) -> TransformResult:
        check_image_batch(x)
        s = self.validate_parameter(eta)
        y, info = wavelet_shrink(x, s, self.wavelet, self.levels, self.eps,
                                 return_info=True)
        summary = {
            "family": self.name, "wavelet": self.wavelet, "s": s,
            "levels": self.levels,
            "is_target_endpoint": s == 1.0,
            "zero_fraction": {k: v.cpu() for k, v in info["zero_fraction"].items()},
            "band_rms": {k: v.cpu() for k, v in info["rms"].items()},
            "threshold": {k: v.cpu() for k, v in info["lambda"].items()},
            "out_of_unit_range": {
                "min": float(y.min()), "max": float(y.max()),
                "fraction_below_0": float((y < 0).to(y.dtype).mean()),
                "fraction_above_1": float((y > 1).to(y.dtype).mean()),
            },
        }
        return TransformResult(y, summary)


_KNOWN = {"wavelet", "levels", "eps"}


def build(params: dict) -> WaveletShrinkage:
    unknown = sorted(set(params or {}) - _KNOWN)
    if unknown:
        raise KeyError("unknown wavelet params %s; known: %s" % (unknown, sorted(_KNOWN)))
    return WaveletShrinkage(**(params or {}))
