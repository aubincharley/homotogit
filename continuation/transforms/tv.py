r"""TV-budget transformation families: TV-:math:`L^2` and TV-:math:`\dot H^{-1}`.

Both families solve, on the original float image :math:`x\in[0,1]^{3\times H\times W}`
*before* network normalization,

.. math::
    T_t(x) = \operatorname*{arg\,min}_{z\in\mathcal K_t(x)} f(z-x),

over the common constraint set

.. math::
    \mathcal K_t(x) = \{z\in[0,1]^{3\times H\times W} :
        \operatorname{TV}(z)\le t\operatorname{TV}(x),\;
        \bar z_c=\bar x_c\ \forall c\}.

The TV budget is **global across channels**; each channel's mean is conserved
**separately**.  No variance conservation is imposed.

Only the fidelity differs:

``l2``
    :math:`f(r)=\tfrac12\sum_c\|r_c\|_2^2`.

``hminus1``
    :math:`f(r)=\tfrac12\sum_c r_c^\top\mathsf L^\dagger r_c` with
    :math:`\mathsf L=D_1^\top D_1+D_2^\top D_2` the positive semidefinite discrete
    Laplacian for the boundary convention below, and :math:`\mathsf L^\dagger` its
    Moore-Penrose pseudoinverse (zero on the constant mode).  This is the
    **homogeneous** :math:`\dot H^{-1}` form: there is no :math:`\alpha`, and
    :math:`(I+\alpha\mathsf L)^{-1}` is *not* used anywhere.

    Mean preservation makes every residual :math:`r_c` zero-mean, so the fidelity
    equals :math:`\tfrac12\sum_c\langle r_c,p_c\rangle` where
    :math:`\mathsf Lp_c=r_c` with :math:`\bar p_c=0`.

Difference and TV convention
----------------------------
Forward differences with unit pixel spacing and a **zero difference at the far
boundary** (no periodic wrapping):

    (D_1 z)[c,p,q] = z[c,p+1,q] - z[c,p,q] for p < H, and 0 at p = H-1,
    (D_2 z)[c,p,q] = z[c,p,q+1] - z[c,p,q] for q < W, and 0 at q = W-1,

with channelwise isotropic TV
``TV(z) = sum_{c,p,q} sqrt((D_1 z)^2 + (D_2 z)^2)``.
This is exactly the convention already used for *measurement* in
:mod:`continuation.diagnostics`.

Solver
------
Chambolle-Pock primal-dual (PDHG) on

    min_z  f(z-x) + iota_B(D z) + iota_C(z),

with ``B`` the global isotropic-TV ball of radius ``t*TV(x)`` and ``C`` the
intersection of the box ``[0,1]`` with the per-channel mean hyperplanes.  Both
indicators are handled in the dual via ``K = [D; I]`` (so ``||K||^2 <= 9``),
which means the **relative budget is enforced as a constraint**: no penalty
coefficient appears anywhere.  A common fixed TV penalty would not be an
equivalent implementation across images, and is deliberately not used.

``prox_{tau f}`` is available in closed form for both fidelities -- for
``hminus1`` via the DCT-II diagonalization of the Neumann Laplacian -- so the
two methods share one solver and differ in exactly one function.

Stopping criterion: the mean-absolute PDHG primal and dual residuals,

    r_p = || (z_k - z_{k+1})/tau - K^T (y_k - y_{k+1}) ||_1 / n,
    r_d = || (y_k - y_{k+1})/sigma - K (z_k - z_{k+1}) ||_1 / m,

both below ``tol`` (default 1e-7), checked every ``check_every`` iterations, up
to ``max_iters`` (default 40000 -- correctness first; tight budgets on natural
images need ~10^4 iterations, and a lower cap silently returns infeasible
iterates).  The returned ``info`` reports the achieved TV ratio, budget
violation, box violation, mean drift, iterations and whether the tolerance was
actually met -- an unconverged solve is never reported as exact.

Endpoints are handled exactly, without invoking the solver:
``T_1(x) = x``, ``T_0(x) = mean(x)`` per channel, and ``T_t(x) = x`` for every
``t`` when ``TV(x) = 0``.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy.fft import dctn, idctn

from .base import ImageTransform, TransformResult, check_image_batch

FIDELITIES = ("l2", "hminus1")


# --------------------------------------------------------------------------
# Difference operators (numpy, single image [C,H,W])
# --------------------------------------------------------------------------

def forward_diff(z):
    """``(D1 z, D2 z)`` with zero difference at the far boundary."""
    d1 = np.zeros_like(z)
    d2 = np.zeros_like(z)
    d1[:, :-1, :] = z[:, 1:, :] - z[:, :-1, :]
    d2[:, :, :-1] = z[:, :, 1:] - z[:, :, :-1]
    return d1, d2


def dt_apply(d1, d2):
    """``D^T (d1, d2) = D1^T d1 + D2^T d2``, the exact adjoint of :func:`forward_diff`.

    From ``<D1 z, g> = sum_{p<H-1} (z[p+1]-z[p]) g[p]`` one gets
    ``(D1^T g)[q] = g[q-1]*[q>=1] - g[q]*[q<=H-2]``, which is what is assembled
    below (and likewise for ``D2``).  Verified against the definition in
    ``tests/test_tv_budget.py::test_adjoint_identity``.
    """
    out = np.zeros_like(d1)
    out[:, :-1, :] -= d1[:, :-1, :]
    out[:, 1:, :] += d1[:, :-1, :]
    out[:, :, :-1] -= d2[:, :, :-1]
    out[:, :, 1:] += d2[:, :, :-1]
    return out


def divergence(d1, d2):
    """``-D^T``, kept for readability where the divergence sign is natural."""
    return -dt_apply(d1, d2)


def tv_value(z) -> float:
    d1, d2 = forward_diff(z)
    return float(np.sqrt(d1 * d1 + d2 * d2).sum())


# --------------------------------------------------------------------------
# Laplacian pseudoinverse via DCT-II (Neumann boundary)
# --------------------------------------------------------------------------

def laplacian_eigenvalues(h: int, w: int):
    """Eigenvalues of ``L = D1^T D1 + D2^T D2`` in the DCT-II basis."""
    lp = 2.0 - 2.0 * np.cos(np.pi * np.arange(h) / h)
    lq = 2.0 - 2.0 * np.cos(np.pi * np.arange(w) / w)
    return lp[:, None] + lq[None, :]


def _dct(a):
    return dctn(a, type=2, norm="ortho", axes=(-2, -1))


def _idct(a):
    return idctn(a, type=2, norm="ortho", axes=(-2, -1))


def laplacian_pinv(r, eig):
    """``L^+ r``: zero on the constant mode, so the result is mean-free."""
    rh = _dct(r)
    inv = np.zeros_like(eig)
    nz = eig > 1e-12
    inv[nz] = 1.0 / eig[nz]
    return _idct(rh * inv[None, :, :])


def laplacian_apply(p):
    """``L p`` computed directly from the difference operators."""
    d1, d2 = forward_diff(p)
    return dt_apply(d1, d2)


# --------------------------------------------------------------------------
# Projections
# --------------------------------------------------------------------------

def project_l1_ball_nonneg(v, radius: float):
    """Project a non-negative vector onto ``{a >= 0, sum(a) <= radius}``."""
    if radius <= 0:
        return np.zeros_like(v)
    if v.sum() <= radius:
        return v
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - radius
    idx = np.arange(1, v.size + 1)
    cond = u - css / idx > 0
    rho = int(np.nonzero(cond)[0][-1]) + 1
    theta = css[rho - 1] / rho
    return np.maximum(v - theta, 0.0)


def project_tv_ball(g1, g2, radius: float):
    """Project onto the **global** isotropic-TV ball of the given radius.

    The budget couples all channels and pixels: the per-pixel gradient norms
    form one vector that is projected onto an l1 ball, then each 2-vector is
    rescaled.
    """
    norms = np.sqrt(g1 * g1 + g2 * g2)
    flat = norms.reshape(-1)
    if flat.sum() <= radius:
        return g1, g2
    proj = project_l1_ball_nonneg(flat, radius).reshape(norms.shape)
    scale = np.zeros_like(norms)
    nz = norms > 1e-15
    scale[nz] = proj[nz] / norms[nz]
    return g1 * scale, g2 * scale


def project_box_mean(z, means, lo=0.0, hi=1.0, iters: int = 50):
    """Project each channel onto ``[lo,hi]`` intersected with its mean hyperplane.

    ``mu -> mean(clip(z + mu, lo, hi))`` is non-decreasing, so the per-channel
    shift is found by bisection.  All channels are bisected simultaneously; 50
    halvings of a bracket of width <= 2 give ~1e-15 accuracy in double precision.
    """
    c = z.shape[0]
    flat = z.reshape(c, -1)
    target = np.asarray(means).reshape(c, 1)
    lo_mu = (lo - flat.max(axis=1)).reshape(c, 1)
    hi_mu = (hi - flat.min(axis=1)).reshape(c, 1)
    for _ in range(iters):
        mid = 0.5 * (lo_mu + hi_mu)
        below = np.clip(flat + mid, lo, hi).mean(axis=1, keepdims=True) < target
        lo_mu = np.where(below, mid, lo_mu)
        hi_mu = np.where(below, hi_mu, mid)
    return np.clip(flat + 0.5 * (lo_mu + hi_mu), lo, hi).reshape(z.shape)


# --------------------------------------------------------------------------
# The solver
# --------------------------------------------------------------------------

def solve_tv_budget(x, t: float, fidelity: str = "l2", max_iters: int = 40000,
                    tol: float = 1e-7, check_every: int = 25,
                    tau: float = 0.3, sigma: float = 0.3) -> tuple:
    """Solve one image.  ``x`` is ``[C,H,W]`` float64 in ``[0,1]``.

    Returns ``(z, info)``.  ``tau*sigma*||K||^2 = 0.81 < 1`` with ``||K||^2 <= 9``.
    """
    if fidelity not in FIDELITIES:
        raise ValueError("fidelity must be one of %s, got %r" % (FIDELITIES, fidelity))
    c, h, w = x.shape
    means = x.reshape(c, -1).mean(axis=1)
    tv_x = tv_value(x)
    t0 = time.perf_counter()

    base = {"requested_t": float(t), "tv_original": tv_x, "fidelity": fidelity}

    # --- exact endpoints, no solver ---
    if tv_x <= 0.0:
        z = x.copy()
        return z, dict(base, endpoint="tv_original_is_zero", iterations=0,
                       converged=True, solver_used=False)
    if t >= 1.0:
        return x.copy(), dict(base, endpoint="t=1 identity", iterations=0,
                              converged=True, solver_used=False)
    if t <= 0.0:
        z = np.broadcast_to(means[:, None, None], x.shape).copy()
        return z, dict(base, endpoint="t=0 per-channel mean", iterations=0,
                       converged=True, solver_used=False)

    radius = float(t) * tv_x
    eig = laplacian_eigenvalues(h, w) if fidelity == "hminus1" else None

    def prox_f(v, step):
        """argmin_z 0.5||z-v||^2 + step * f(z-x)."""
        if fidelity == "l2":
            return (v + step * x) / (1.0 + step)
        # hminus1: (I + step * L^+) r = w, diagonal in the DCT basis.
        wv = v - x
        wh = _dct(wv)
        fac = np.ones_like(eig)
        nz = eig > 1e-12
        fac[nz] = eig[nz] / (eig[nz] + step)
        return x + _idct(wh * fac[None, :, :])

    z = x.copy()
    zbar = z.copy()
    y1 = np.zeros_like(x)
    y2 = np.zeros_like(x)
    y3 = np.zeros_like(x)          # dual for the box+mean indicator
    converged, iterations = False, 0
    res_p = res_d = float("nan")

    for k in range(1, max_iters + 1):
        iterations = k
        # dual ascent on the overrelaxed primal point
        g1, g2 = forward_diff(zbar)
        y1_new = y1 + sigma * g1
        y2_new = y2 + sigma * g2
        p1, p2 = project_tv_ball(y1_new / sigma, y2_new / sigma, radius)
        y1_new -= sigma * p1
        y2_new -= sigma * p2
        y3_new = y3 + sigma * zbar
        y3_new -= sigma * project_box_mean(y3_new / sigma, means)

        # primal descent
        z_new = prox_f(z - tau * (dt_apply(y1_new, y2_new) + y3_new), tau)
        zbar = 2.0 * z_new - z

        if k % check_every == 0 or k == max_iters:
            dz, dy1, dy2, dy3 = z - z_new, y1 - y1_new, y2 - y2_new, y3 - y3_new
            kt_dy = dt_apply(dy1, dy2) + dy3
            res_p = float(np.abs(dz / tau - kt_dy).mean())
            k_dz1, k_dz2 = forward_diff(dz)
            res_d = float((np.abs(dy1 / sigma - k_dz1).mean()
                           + np.abs(dy2 / sigma - k_dz2).mean()
                           + np.abs(dy3 / sigma - dz).mean()) / 3.0)
            z, y1, y2, y3 = z_new, y1_new, y2_new, y3_new
            if res_p < tol and res_d < tol:
                converged = True
                break
        else:
            z, y1, y2, y3 = z_new, y1_new, y2_new, y3_new

    # The box and mean constraints are part of K_t, so enforce them exactly on
    # the returned iterate; the residual TV budget violation is then reported
    # rather than hidden.
    z = project_box_mean(z, means)

    tv_z = tv_value(z)
    r = z - x
    if fidelity == "l2":
        fid = 0.5 * float((r * r).sum())
    else:
        p = laplacian_pinv(r, eig)
        fid = 0.5 * float((r * p).sum())

    info = dict(
        base,
        solver="chambolle-pock (PDHG), K=[D; I], ||K||^2<=9, tau=sigma=%.3g" % tau,
        stopping_criterion="mean-abs primal and dual PDHG residuals < tol",
        tol=tol, max_iters=max_iters, iterations=iterations, converged=converged,
        solver_used=True,
        primal_residual=res_p, dual_residual=res_d,
        achieved_tv=tv_z,
        achieved_tv_ratio=tv_z / tv_x,
        tv_budget=radius,
        tv_budget_violation=max(0.0, tv_z - radius),
        tv_budget_rel_violation=max(0.0, tv_z - radius) / max(radius, 1e-30),
        fidelity_value=fid,
        box_violation=float(max(0.0, z.min() * -1.0, z.max() - 1.0)),
        mean_abs_drift=float(np.abs(z.reshape(c, -1).mean(axis=1) - means).max()),
        solve_seconds=round(time.perf_counter() - t0, 4),
    )
    return z, info


# --------------------------------------------------------------------------
# Transform classes
# --------------------------------------------------------------------------

class TVBudgetTransform(ImageTransform):
    """Relative-TV-budget family.  Native parameter ``t in [0,1]``, target ``t=1``."""

    parameter_name = "t"
    parameter_units = "relative TV budget"

    def __init__(self, fidelity: str = "l2", max_iters: int = 40000, tol: float = 1e-7,
                 check_every: int = 25, tau: float = 0.3, sigma: float = 0.3):
        if fidelity not in FIDELITIES:
            raise ValueError("fidelity must be one of %s, got %r" % (FIDELITIES, fidelity))
        if tau * sigma * 9.0 >= 1.0:
            raise ValueError("PDHG needs tau*sigma*||K||^2 < 1 with ||K||^2 <= 9; "
                             "got %g" % (tau * sigma * 9.0))
        self.fidelity = fidelity
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.check_every = int(check_every)
        self.tau, self.sigma = float(tau), float(sigma)

    @property
    def target_parameter(self) -> float:
        return 1.0

    def validate_parameter(self, eta: float) -> float:
        t = float(eta)
        if not (0.0 <= t <= 1.0) or not np.isfinite(t):
            raise ValueError("relative TV budget t must lie in [0,1], got %r" % (eta,))
        return t

    def config_signature(self) -> dict:
        return {
            "fidelity": self.fidelity,
            "constraint": "TV(z) <= t*TV(x) (global across channels)",
            "mean_conservation": "per channel, exact; no variance conservation",
            "box": "[0,1]",
            "differences": "forward, unit spacing, zero difference at far boundary",
            "tv": "channelwise isotropic",
            "solver": "chambolle-pock PDHG",
            "tol": self.tol,
            "max_iters": self.max_iters,
            "tau": self.tau,
            "sigma": self.sigma,
            "budget_enforcement": "hard constraint (no TV penalty coefficient)",
            "value_space": "float [0,1], pre-normalization",
        }

    def apply(self, x: torch.Tensor, eta: float, meta: dict | None = None) -> TransformResult:
        check_image_batch(x)
        t = self.validate_parameter(eta)
        if t == 1.0:
            return TransformResult(x, {"family": self.name, "t": 1.0, "identity": True,
                                       "solver_used": False})
        device, dtype = x.device, x.dtype
        arr = x.detach().to("cpu", torch.float64).numpy()   # solver runs on CPU
        out = np.empty_like(arr)
        infos = []
        for i in range(arr.shape[0]):
            z, info = solve_tv_budget(arr[i], t, self.fidelity, self.max_iters,
                                      self.tol, self.check_every, self.tau, self.sigma)
            out[i] = z
            infos.append(info)
        y = torch.from_numpy(out).to(device=device, dtype=dtype)
        summary = {
            "family": self.name, "t": t, "identity": False,
            "n_images": len(infos),
            "all_converged": all(i.get("converged") for i in infos),
            "max_tv_budget_rel_violation": max(i.get("tv_budget_rel_violation", 0.0)
                                               for i in infos),
            "max_box_violation": max(i.get("box_violation", 0.0) for i in infos),
            "max_mean_abs_drift": max(i.get("mean_abs_drift", 0.0) for i in infos),
            "per_image": infos,
        }
        return TransformResult(y, summary)


class TVL2(TVBudgetTransform):
    name = "tv_l2"

    def __init__(self, **kw):
        kw.pop("fidelity", None)
        super().__init__(fidelity="l2", **kw)


class TVHminus1(TVBudgetTransform):
    name = "tv_hminus1"

    def __init__(self, **kw):
        kw.pop("fidelity", None)
        super().__init__(fidelity="hminus1", **kw)


_KNOWN = {"max_iters", "tol", "check_every", "tau", "sigma"}


def build_l2(params: dict) -> TVL2:
    unknown = sorted(set(params or {}) - _KNOWN)
    if unknown:
        raise KeyError("unknown tv_l2 params %s; known: %s" % (unknown, sorted(_KNOWN)))
    return TVL2(**(params or {}))


def build_hminus1(params: dict) -> TVHminus1:
    unknown = sorted(set(params or {}) - _KNOWN)
    if unknown:
        raise KeyError("unknown tv_hminus1 params %s; known: %s" % (unknown, sorted(_KNOWN)))
    return TVHminus1(**(params or {}))
