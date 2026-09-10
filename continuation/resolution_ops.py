r"""Seven resolution-reduction operators, and the single-site insertion machinery.

Purely additive: nothing here modifies an existing operator or schedule.  Every
internal Gaussian is off in this benchmark -- these arms change *resolution* and
nothing else.  Anti-aliasing that belongs to an operator stays part of it
(MaxBlur keeps its binomial kernel, bilinear keeps ``antialias=True``).

Insertion points, recovered from ``continuation.ablation_ops`` rather than from
display names.  A ``forward_pre_hook`` reduces the tensor *entering* a module:

===========  =============================  ===============================
name         hook target                    tensor reduced
===========  =============================  ===============================
``input``    the image pipeline             RGB before channel normalization
``stem``     ``model.blocks``               output of conv1 -> bn1 -> ReLU
``D0``       ``model.blocks[1]``            complete output of ``blocks[0]``
``D1``       ``model.blocks[2]``            complete output of ``blocks[1]``
``D2``       ``model.blocks[3]``            complete output of ``blocks[2]``
===========  =============================  ===============================

"Complete output" means after the residual addition *and* the block's final
ReLU, so both branches of every later block see the same grid and stay
compatible.  In ResNet-20 all five sites carry a **32x32** map (stage 1 runs
32x32 for blocks 0-2), which is why the absolute schedule 16/24/32 is well
defined at each of them without the relative-target rescaling the earlier
ablation needed at deeper sites.  ``blocks[8]`` is never a site: its output
feeds global average pooling.

``O_ref`` -- the operator the historical D0/D1/D2 arms used -- is
**adaptive max pooling** (``F.adaptive_max_pool2d``), read from
``campaign_ops.reduce_spatial``.  It is one of the seven operators below, so it
needs no separate reference arm.

At ``r == H`` every operator returns the input tensor object unchanged: the
bypass is exact, not merely numerically close.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

OPERATORS = ("bilinear", "max", "maxblur", "softpool", "l2", "hminus1",
             "perceptual")
O_REF = "max"
LOCATIONS = ("input", "stem", "D0", "D1", "D2")
#: location -> index into ``model.blocks`` whose *input* is reduced.
#: ``None`` means the whole ``blocks`` sequential (i.e. just after the stem).
BLOCK_TARGET = {"stem": None, "D0": 1, "D1": 2, "D2": 3}

_SOLVE_CACHE: dict = {}
_PERC_EPS = 1e-6


# --------------------------------------------------------------------------
# adaptive windows (shared by max, maxblur, softpool)
# --------------------------------------------------------------------------

def window_starts(n_in: int, n_out: int):
    """``floor(i * n_in / n_out)`` -- the start of PyTorch's adaptive window."""
    return [int(math.floor(i * n_in / n_out)) for i in range(n_out)]


def window_widths(n_in: int, n_out: int):
    return [int(math.ceil((i + 1) * n_in / n_out)) - int(math.floor(i * n_in / n_out))
            for i in range(n_out)]


# --------------------------------------------------------------------------
# 1. bilinear
# --------------------------------------------------------------------------

def op_bilinear(h, r):
    return F.interpolate(h, size=(r, r), mode="bilinear", align_corners=False,
                         antialias=True)


# --------------------------------------------------------------------------
# 2. adaptive max  (O_ref)
# --------------------------------------------------------------------------

def op_max(h, r):
    return F.adaptive_max_pool2d(h, (r, r))


# --------------------------------------------------------------------------
# 3. MaxBlur
# --------------------------------------------------------------------------

_BINOMIAL = (1.0, 2.0, 1.0)


def op_maxblur(h, r, blur: bool = True):
    """Dense 2x2 max (stride 1) -> separable binomial blur -> gather.

    With ``blur=False`` this must reproduce :func:`op_max` exactly, because the
    adaptive windows at 32->16 and 32->24 are all two wide: a dense stride-1
    maximum sampled at ``floor(32 i / r)`` is the max over that same window.
    ``verify_resolution_ops`` checks it rather than trusting the argument.
    """
    n = h.shape[-1]
    d = F.max_pool2d(h, kernel_size=2, stride=1)              # n-1
    if blur:
        k = torch.tensor(_BINOMIAL, dtype=h.dtype, device=h.device) / 4.0
        c = d.shape[1]
        d = F.conv2d(F.pad(d, (0, 0, 1, 1), mode="reflect"),
                     k.view(1, 1, 3, 1).expand(c, 1, 3, 1), groups=c)
        d = F.conv2d(F.pad(d, (1, 1, 0, 0), mode="reflect"),
                     k.view(1, 1, 1, 3).expand(c, 1, 1, 3), groups=c)
    idx = torch.as_tensor(window_starts(n, r), device=h.device, dtype=torch.long)
    return d.index_select(-2, idx).index_select(-1, idx)


# --------------------------------------------------------------------------
# 4. SoftPool
# --------------------------------------------------------------------------

def op_softpool(h, r):
    r"""Softmax-weighted average inside each adaptive-max window.

    ``tau = 0.5 * sqrt(mean[(h - mean h)^2] + 1e-12)``, one value per sample and
    channel taken from the **whole incoming map**, identical at the input and at
    every internal location.  The temperature is differentiated through: it is a
    function of ``h`` and is deliberately not detached.
    """
    b, c, n, _ = h.shape
    widths = set(window_widths(n, r))
    if widths != {2}:
        raise ValueError("SoftPool assumes width-2 adaptive windows, got %s"
                         % sorted(widths))
    idx = torch.as_tensor(window_starts(n, r), device=h.device, dtype=torch.long)
    # [b, c, r, 2, r, 2] -> stack the four window members
    rows = torch.stack([h.index_select(-2, idx + o) for o in (0, 1)], dim=-3)
    win = torch.stack([rows.index_select(-1, idx + o) for o in (0, 1)], dim=-1)
    win = win.reshape(b, c, 2, r, r, 2).permute(0, 1, 3, 4, 2, 5).reshape(b, c, r, r, 4)

    mean = h.mean(dim=(-2, -1), keepdim=True)
    tau = 0.5 * torch.sqrt(((h - mean) ** 2).mean(dim=(-2, -1), keepdim=True) + 1e-12)
    tau = tau.unsqueeze(-1)                                    # [b,c,1,1,1]
    z = win / tau
    z = z - z.amax(dim=-1, keepdim=True)                       # stable softmax
    w = torch.softmax(z, dim=-1)
    return (w * win).sum(dim=-1)


# --------------------------------------------------------------------------
# 5/6. quadratic reconstruction: L2 and homogeneous H^-1
# --------------------------------------------------------------------------

def _upsample_matrix(n: int, r: int, dtype=torch.float64):
    """``U_r``: r*r coefficients -> n*n, bilinear, align_corners=False, no AA."""
    eye = torch.eye(r * r, dtype=dtype).reshape(r * r, 1, r, r)
    up = F.interpolate(eye, size=(n, n), mode="bilinear", align_corners=False,
                       antialias=False)
    return up.reshape(r * r, n * n).t().contiguous()            # [n*n, r*r]


def _laplacian(n: int, dtype=torch.float64):
    """``L = D1^T D1 + D2^T D2``; forward differences, zero at bottom/right."""
    N = n * n
    d1 = torch.zeros(N, N, dtype=dtype)
    d2 = torch.zeros(N, N, dtype=dtype)
    for i in range(n):
        for j in range(n):
            k = i * n + j
            if i < n - 1:
                d1[k, (i + 1) * n + j] = 1.0
                d1[k, k] -= 1.0
            if j < n - 1:
                d2[k, i * n + (j + 1)] = 1.0
                d2[k, k] -= 1.0
    return d1.t() @ d1 + d2.t() @ d2


def laplacian_pinv(n: int, dtype=torch.float64):
    """``L^+`` with the constant null space removed exactly.

    ``torch.linalg.pinv`` at its default tolerance leaves the null eigenvalues
    near 1e-16 instead of at zero; inverting those gave a KKT system with
    condition 2e17, i.e. numerically singular.  Truncating the spectrum
    explicitly at ``max(eig) * n^2 * eps`` keeps every genuine mode (the
    smallest nonzero eigenvalue of ``L`` on a 32x32 grid is ~4e-2) and zeroes
    the rest.
    """
    L = _laplacian(n, dtype)
    w, V = torch.linalg.eigh(L)
    tol = float(w.max()) * (n * n) * torch.finfo(dtype).eps
    inv = torch.where(w > tol, 1.0 / w, torch.zeros_like(w))
    return (V * inv) @ V.t()


def solution_map(n: int, r: int, kind: str, device=None, dtype=torch.float32):
    r"""``M`` with ``z* = M h``, for the constrained quadratic problem

        min_z  1/2 (U_r z - h)^T Q (U_r z - h)   s.t.  1^T (U_r z - h) = 0

    solved through its KKT system in float64 and cached.  ``Q = I`` for ``l2``;
    ``Q = L^+`` (Moore-Penrose, zero on constants) for ``hminus1``.
    """
    key = (n, r, kind)
    M = _SOLVE_CACHE.get(key)
    if M is None:
        U = _upsample_matrix(n, r)                              # [N, m]
        N, m = U.shape
        one = torch.ones(N, 1, dtype=torch.float64)
        Q = (torch.eye(N, dtype=torch.float64) if kind == "l2"
             else laplacian_pinv(n))
        A = U.t() @ Q                                           # [m, N]
        K = torch.zeros(m + 1, m + 1, dtype=torch.float64)
        K[:m, :m] = A @ U
        K[:m, m:] = U.t() @ one
        K[m:, :m] = (U.t() @ one).t()
        rhs = torch.zeros(m + 1, N, dtype=torch.float64)
        rhs[:m] = A
        rhs[m:] = one.t()
        M = torch.linalg.solve(K, rhs)[:m]                      # [m, N]
        _SOLVE_CACHE[key] = M
    return M.to(device=device, dtype=dtype)


def op_quadratic(h, r, kind: str):
    n = h.shape[-1]
    M = solution_map(n, r, kind, device=h.device, dtype=h.dtype)
    b, c = h.shape[0], h.shape[1]
    z = h.reshape(b * c, n * n) @ M.t()
    return z.reshape(b, c, r, r)


def op_l2(h, r):
    return op_quadratic(h, r, "l2")


def op_hminus1(h, r):
    return op_quadratic(h, r, "hminus1")


# --------------------------------------------------------------------------
# 7. perceptual (Oztireli & Gross 2015, Algorithm 1)
# --------------------------------------------------------------------------

def _box_down(h, s: int):
    return F.avg_pool2d(h, kernel_size=s, stride=s)


def _patch_mean(x, np_: int, adjoint: bool = False):
    """Uniform ``np_ x np_`` patch average, stride 1, reflect padded.

    Patch ``p`` covers pixels ``p .. p + np_ - 1``.  ``adjoint=False`` computes
    a quantity *of* patch ``p`` from the pixels it covers; ``adjoint=True``
    averages patch quantities back onto pixel ``k``, over the patches that
    *contain* ``k`` (indices ``k - np_ + 1 .. k``).  The two differ by the side
    the window is padded on, and using the forward window for both is exactly
    the error the naive reference catches.
    """
    c = x.shape[1]
    if not adjoint:
        k = torch.ones(c, 1, np_, np_, dtype=x.dtype, device=x.device) / (np_ * np_)
        return F.conv2d(F.pad(x, (0, np_ - 1, 0, np_ - 1), mode="reflect"),
                        k, groups=c)
    # Scatter: only patches that actually exist contribute, and the divisor is
    # how many of them cover this pixel -- so the border averages over fewer
    # patches rather than over reflected ones.  Zero padding plus an explicit
    # count is what makes this agree with the naive per-patch reference.
    k = torch.ones(c, 1, np_, np_, dtype=x.dtype, device=x.device)
    num = F.conv2d(F.pad(x, (np_ - 1, 0, np_ - 1, 0)), k, groups=c)
    den = F.conv2d(F.pad(torch.ones_like(x), (np_ - 1, 0, np_ - 1, 0)), k,
                   groups=c)
    return num / den


def perceptual_factor2(h, np_: int = 2, eps: float = _PERC_EPS):
    r"""Factor-2 perceptual downscale, vectorised.

    Per ``np_ x np_`` patch of the low-resolution grid the algorithm fits the
    local linear model that a perceptual (SSIM-like) criterion prescribes,

        out = mu + rho * (L - mu),      rho = sigma_H / sigma_L,

    with ``L`` the uniformly box-downsampled image, ``mu`` the patch mean of
    ``L``, ``sigma_L`` its patch standard deviation, and ``sigma_H`` the
    standard deviation the high-resolution content carries into that patch
    (obtained from the box-downsampled *squares*).  Overlapping patch solutions
    are averaged, which is what the stride-1 patch means implement.

    The low-variance branch sets ``rho = 0`` where ``sigma_L^2 <= eps``, so a
    flat patch reduces to its mean.  Divisions and square roots are masked so
    the backward pass stays finite on degenerate (constant) inputs.
    """
    L = _box_down(h, 2)
    L2 = _box_down(h * h, 2)
    mu = _patch_mean(L, np_)
    var_l = _patch_mean(L * L, np_) - mu * mu
    var_h = _patch_mean(L2, np_) - mu * mu
    ok = var_l > eps
    safe_l = torch.where(ok, var_l, torch.ones_like(var_l))
    safe_h = torch.where(ok, var_h.clamp_min(0.0), torch.zeros_like(var_h))
    rho = torch.where(ok, torch.sqrt(safe_h) / torch.sqrt(safe_l),
                      torch.zeros_like(var_l))
    # average the overlapping patch solutions over the patches CONTAINING each
    # pixel:  mean_p[ mu_p + rho_p (L - mu_p) ]
    return (_patch_mean(mu, np_, adjoint=True)
            + L * _patch_mean(rho, np_, adjoint=True)
            - _patch_mean(rho * mu, np_, adjoint=True))


def perceptual_reference(h, np_: int = 2, eps: float = _PERC_EPS):
    """Naive per-patch loop, used only to verify :func:`perceptual_factor2`."""
    L = _box_down(h, 2)
    L2 = _box_down(h * h, 2)
    b, c, n, _ = L.shape
    acc = torch.zeros_like(L)
    cnt = torch.zeros_like(L)
    Lp = F.pad(L, (0, np_ - 1) * 2, mode="reflect")
    L2p = F.pad(L2, (0, np_ - 1) * 2, mode="reflect")
    for i in range(n):
        for j in range(n):
            pl = Lp[:, :, i:i + np_, j:j + np_]
            p2 = L2p[:, :, i:i + np_, j:j + np_]
            mu = pl.mean(dim=(-2, -1), keepdim=True)
            vl = (pl * pl).mean(dim=(-2, -1), keepdim=True) - mu * mu
            vh = p2.mean(dim=(-2, -1), keepdim=True) - mu * mu
            ok = vl > eps
            rho = torch.where(ok, torch.sqrt(vh.clamp_min(0)) /
                              torch.sqrt(torch.where(ok, vl, torch.ones_like(vl))),
                              torch.zeros_like(vl))
            out = mu + rho * (pl - mu)
            for a in range(np_):
                for bb in range(np_):
                    ii, jj = i + a, j + bb
                    if 0 <= ii < n and 0 <= jj < n:
                        acc[:, :, ii, jj] += out[:, :, a, bb]
                        cnt[:, :, ii, jj] += 1
    return acc / cnt


def op_perceptual(h, r):
    """32 -> r.  Integer factor 2 for r=16; via a 48x48 bicubic step for r=24."""
    n = h.shape[-1]
    if n == 2 * r:
        return perceptual_factor2(h)
    if n * r % 2:
        raise ValueError("perceptual needs an even 2r target, got n=%d r=%d" % (n, r))
    up = F.interpolate(h, size=(2 * r, 2 * r), mode="bicubic",
                       align_corners=False, antialias=False)
    return perceptual_factor2(up)


# --------------------------------------------------------------------------

_DISPATCH = {"bilinear": op_bilinear, "max": op_max, "maxblur": op_maxblur,
             "softpool": op_softpool, "l2": op_l2, "hminus1": op_hminus1,
             "perceptual": op_perceptual}


def reduce_with(h, r, operator: str):
    """Apply ``operator``; an exact identity (same tensor object) when r == H."""
    if r is None or int(r) == int(h.shape[-1]):
        return h
    fn = _DISPATCH.get(operator)
    if fn is None:
        raise ValueError("unknown operator %r; expected one of %s"
                         % (operator, ", ".join(OPERATORS)))
    return fn(h, int(r))


# --------------------------------------------------------------------------
# schedules and the controller
# --------------------------------------------------------------------------

PATHS = {
    "Rprog":    [16] * 6 + [24] * 6 + [32] * 18,
    "Rgentle":  [24] * 6 + [24] * 6 + [32] * 18,
    "Rreverse": [24] * 6 + [16] * 6 + [32] * 18,
    "Rlate":    [16] * 9 + [24] * 9 + [32] * 12,
    "fixed16":  [16] * 30,
    "fixed24":  [24] * 30,
    "none":     [32] * 30,
}


class ResolutionController:
    """One reduction site, one operator, one per-epoch resolution path."""

    def __init__(self, operator: str, location: str, path: str):
        if operator not in OPERATORS and operator != "none":
            raise ValueError("unknown operator %r" % (operator,))
        if location not in LOCATIONS and location != "none":
            raise ValueError("unknown location %r" % (location,))
        if path not in PATHS:
            raise ValueError("unknown path %r" % (path,))
        self.operator, self.location, self.path = operator, location, path
        self.by_epoch = PATHS[path]
        self.resolution = None
        self.bypass_all = False

    def set_epoch(self, e: int):
        e = max(0, min(int(e), len(self.by_epoch) - 1))
        self.resolution = self.by_epoch[e]
        return self.resolution

    def set_state(self, resolution):
        self.resolution = resolution
        return resolution

    def active(self) -> bool:
        return (not self.bypass_all and self.operator != "none"
                and self.location != "none" and self.resolution is not None
                and int(self.resolution) != 32)

    def apply(self, h):
        if not self.active():
            return h
        return reduce_with(h, self.resolution, self.operator)

    def input_resolution(self):
        """Resolution for the image pipeline, or None when the site is internal."""
        if self.location != "input" or not self.active():
            return None
        return self.resolution

    def describe(self) -> dict:
        return {"operator": self.operator, "location": self.location,
                "path": self.path, "resolution_by_epoch": list(self.by_epoch),
                "o_ref": O_REF,
                "note": ("single additional reduction site; internal Gaussian "
                         "filters are entirely disabled in this benchmark")}


def attach_resolution(model, ctrl: ResolutionController):
    """Register the single internal reduction hook, if the site is internal."""
    if ctrl.location in ("input", "none") or ctrl.operator == "none":
        return []

    def _reduce(_mod, args):
        if not ctrl.active():
            return None
        return (ctrl.apply(args[0]),)

    tgt = BLOCK_TARGET[ctrl.location]
    target = model.blocks if tgt is None else model.blocks[tgt]
    return [target.register_forward_pre_hook(_reduce)]
