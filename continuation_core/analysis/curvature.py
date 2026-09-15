r"""Curvature of the empirical loss at a finished solution.

Why not read it off a perturbation curve
----------------------------------------
For an isotropic perturbation,

    E_{U ~ N(0, tau^2 I)} [ L(W + U) ] = L(W) + (tau^2 / 2) tr H + O(tau^4),

so a random-perturbation curve is, in its quadratic regime, measuring nothing but
``tr H`` -- through a handful of random draws per point, which is why its
seed-to-seed spread is of the same order as the effect it reports.  Estimating
``tr H`` directly removes that noise and yields the quantity the bounds actually
contain.  ``perturbation_sensitivity`` in :mod:`.landscape` remains the right
tool for the *finite-radius* question; this is the infinitesimal one.

It also answers what a scalar cannot: whether the flatness is **diffuse or
concentrated**.  A solution can have small ``tr H`` because the whole spectrum is
low, or because a few sharp directions disappeared, and those call for different
explanations.

Estimators
----------
``H`` is 269,722 x 269,722 here and is never formed.  Access is through
Hessian-vector products by double differentiation, accumulated over minibatches
with the correct size weights so the result does not depend on the batch size.

*Trace* -- Hutchinson with Rademacher probes.  The same draws give the
**per-block** traces for free: for Rademacher ``v`` and any index set ``B``,
``E[<v_B, (H v)_B>] = sum_{i in B} H_ii``, the cross terms having zero mean.

*Deflation* -- the raw estimator's variance grows with ``||H||_F`` and the top
eigenvalue alone carries about a tenth of the trace.  If the ``u_i`` are
eigenvectors then ``P H P = H - sum_i lambda_i u_i u_i^T``, so

    v^T P H P v = v^T H v - sum_i lambda_i <v, u_i>^2

is computable from the *same* product and ``tr H = tr(P H P) + sum_i lambda_i``.
Both estimates are returned: they agree only to the extent that the power
iteration converged, so a disagreement is information rather than something to
hide.

*Top spectrum* -- power iteration with Gram-Schmidt deflation.  The **signed**
Rayleigh quotient is reported, so a dominant negative direction would be visible
rather than silently made positive.  Each value carries its iteration count and
last relative change.

A measured caution.  On the exploratory branch the run-to-run noise was 1.7 % on
``tr H`` -- at the Hutchinson estimator's own precision -- but **8.2 % on
lambda_max**, which is five times larger.  Fine comparisons on the top eigenvalue
alone are below that floor; the share of the trace carried by the top few is
stable at 2.2 % and is the statistic to use instead.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _params(model, names):
    d = dict(model.named_parameters())
    return [d[n] for n in names]


def _batches(ev, n_images: int, batch_size: int):
    n = min(n_images, int(ev.images.shape[0]))
    return ((ev.images[i:i + batch_size], ev.labels[i:i + batch_size])
            for i in range(0, n, batch_size))


def hessian_vector_product(ev, names, batches, v):
    """``H v`` for the mean cross-entropy over ``batches``.

    The caller sets the BatchNorm mode; see :func:`bn_mode`.
    """
    params = _params(ev.model, names)
    out = [torch.zeros_like(p) for p in params]
    total = 0
    for imgs, lbls in batches:
        loss = F.cross_entropy(ev.model(ev.pipeline(imgs)), lbls, reduction="sum")
        g = torch.autograd.grad(loss, params, create_graph=True)
        gv = sum((gi * vi).sum() for gi, vi in zip(g, v))
        hv = torch.autograd.grad(gv, params)
        for o, h in zip(out, hv):
            o.add_(h.detach())
        total += int(imgs.shape[0])
        del loss, g, gv, hv
    return [o / total for o in out]


def _dot(a, b):
    return float(sum((x * y).to(torch.float64).sum() for x, y in zip(a, b)))


def _stats(vals):
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return {"mean": m, "sd": None, "sem": None, "n": n}
    sd = (sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5
    return {"mean": m, "sd": sd, "sem": sd / n ** 0.5, "n": n}


def top_eigenvalues(ev, names, batch_fn, k: int, iters: int, gen, tol: float = 1e-4):
    params = _params(ev.model, names)
    found, out = [], []
    for _ in range(k):
        v = [torch.randn(p.shape, generator=gen, dtype=torch.float32).to(p.device).to(p.dtype)
             for p in params]
        for u in found:
            c = _dot(v, u)
            v = [x - c * y for x, y in zip(v, u)]
        nv = _dot(v, v) ** 0.5
        v = [x / nv for x in v]
        lam, used, delta = 0.0, 0, None
        for it in range(iters):
            hv = hessian_vector_product(ev, names, batch_fn(), v)
            for u in found:
                c = _dot(hv, u)
                hv = [x - c * y for x, y in zip(hv, u)]
            new = _dot(v, hv)                       # signed Rayleigh quotient
            nh = _dot(hv, hv) ** 0.5
            used = it + 1
            if nh == 0:
                break
            delta = abs(new - lam) / max(abs(new), 1e-12)
            v, lam = [x / nh for x in hv], new
            if delta < tol:
                break
        out.append({"eigenvalue": lam, "iterations": used, "last_rel_change": delta,
                    "converged": bool(delta is not None and delta < tol)})
        found.append(v)
    rank = sorted(range(len(out)), key=lambda i: -out[i]["eigenvalue"])
    return [out[i] for i in rank], [found[i] for i in rank]


class bn_mode:
    """Context manager selecting the BatchNorm policy, restoring buffers after.

    ``running_stats`` evaluates with the stored statistics (``eval`` mode).
    ``fixed_batch_stats`` normalises each batch with its own statistics
    (``train`` mode), which is what "the network after its normalisation has
    adapted" means -- and which would otherwise overwrite the checkpoint's
    running buffers on every forward pass, so they are saved and put back.
    """

    def __init__(self, model, policy: str):
        if policy not in ("running_stats", "fixed_batch_stats"):
            raise ValueError("unknown bn_policy %r" % (policy,))
        self.model, self.policy = model, policy

    def __enter__(self):
        self.was = self.model.training
        self.buffers = {n: b.detach().clone() for n, b in self.model.named_buffers()}
        self.model.train(self.policy == "fixed_batch_stats")
        return self.model

    def __exit__(self, *exc):
        with torch.no_grad():
            for n, b in self.model.named_buffers():
                b.copy_(self.buffers[n])
        self.model.train(self.was)
        return False


def block_of(name: str) -> str:
    if name.startswith("blocks."):
        return "block%s" % name.split(".")[1]
    return "fc" if name.startswith("fc.") else "stem"


def probe(ev, *, n_images: int = 5000, batch_size: int = 500, draws: int = 64,
          top_k: int = 5, power_iters: int = 40, seed: int = 0,
          state="target", bn_policy: str = "running_stats") -> dict:
    """Trace, per-block traces and top spectrum for one checkpoint.

    ``bn_policy`` selects which function is differentiated; see the module
    docstring.  Under ``fixed_batch_stats`` the loss depends on how the images
    are batched, so the result is tied to ``batch_size`` -- it is reported with
    the result rather than left implicit.
    """
    st = ev.state(state) if isinstance(state, str) else state
    previous = ev.controller.state
    ev.controller.set_state(st)
    ctx = bn_mode(ev.model, bn_policy)
    ctx.__enter__()
    for p in ev.model.parameters():
        p.requires_grad_(True)
    try:
        names = ev.names
        blocks = {}
        for i, n in enumerate(names):
            blocks.setdefault(block_of(n), []).append(i)

        def batch_fn():
            return _batches(ev, n_images, batch_size)

        gen = torch.Generator().manual_seed(int(seed))
        ev_pairs, vecs = top_eigenvalues(ev, names, batch_fn, top_k, power_iters, gen)
        lams = [e["eigenvalue"] for e in ev_pairs]

        params = _params(ev.model, names)
        totals, deflated = [], []
        per_block = {b: [] for b in blocks}
        for _ in range(draws):
            v = [(torch.randint(0, 2, p.shape, generator=gen, dtype=torch.int8)
                  .to(p.device).to(p.dtype) * 2 - 1) for p in params]
            hv = hessian_vector_product(ev, names, batch_fn(), v)
            raw = _dot(v, hv)
            totals.append(raw)
            deflated.append(raw - sum(l * _dot(v, u) ** 2 for l, u in zip(lams, vecs)))
            for b, idx in blocks.items():
                per_block[b].append(sum(float((v[i] * hv[i]).to(torch.float64).sum())
                                        for i in idx))
        d = _stats(deflated)
        trace = {"mean": d["mean"] + sum(lams), "sd": d["sd"], "sem": d["sem"],
                 "n": d["n"], "eigen_sum": sum(lams), "residual_mean": d["mean"]}
        raw_tr = _stats(totals)
        n_param = sum(int(p.numel()) for p in params)
        top_share = sum(lams) / trace["mean"] if trace["mean"] else None
        return {"checkpoint": ev.path, "split": ev.split,
                "bn_policy": bn_policy, "batch_size": batch_size,
                "n_images": min(n_images, int(ev.images.shape[0])),
                "n_parameters": n_param, "draws": draws,
                "trace": trace, "trace_raw": raw_tr,
                "per_block": {b: _stats(v) for b, v in per_block.items()},
                "top_eigenvalues": ev_pairs,
                "top_share_of_trace": top_share,
                "note": ("run-to-run noise measured elsewhere: 1.7 % on the trace "
                         "(the estimator's own precision), 8.2 % on lambda_max; "
                         "use top_share_of_trace, not lambda_max, for fine "
                         "comparisons")}
    finally:
        ctx.__exit__(None, None, None)
        if previous is not None:
            ev.controller.set_state(previous)
