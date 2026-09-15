"""Cross-entropy Hessian restricted to the perturbed parameter subspace.

Objective.  L(w) = mean cross-entropy over a fixed probe (no weight decay), with
the network in inference mode and BatchNorm running statistics held fixed
(either the checkpoint's saved statistics or statistics recalibrated once at the
centre).  Variables w = every convolution weight and the final linear weight
(the v2 perturbation mask); biases, BatchNorm affine parameters and running
statistics are constants.  H = d^2 L / dw^2 computed by double back-propagation
(exact Hessian-vector products; no dense Hessian, no Fisher, no Gauss-Newton).
ReLU and max-pooling are piecewise linear: the Hessian is the almost-everywhere
second derivative, and finite differences can cross kinks.

Relative coordinates.  For block g (one conv output filter or one fc row),
A_g = ||theta_g||_2 I_g frozen at the reference checkpoint, theta(q) = theta + A q,
H_rel = A^T H A.  Blocks with ||theta_g|| = 0 get A_g = 0 (their coordinates are
exactly zero in every Krylov vector) and are counted.

Lanczos.  Full reorthogonalisation (two classical Gram-Schmidt passes) in
float64; HVPs in float32 (float64 in the verification routines).  Ritz values
and residual estimates |beta_j s_ji| are recorded every ``check_every`` steps;
final Ritz vectors get explicit residuals ||H y - theta y|| with fresh HVPs.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from landscape_study.directions import in_mask


class FrozenBNObjective:
    """L(w) for a model whose buffers and non-masked parameters are already set."""

    def __init__(self, model, pipeline, x, y, batch_size=500, dtype=torch.float32):
        self.model, self.pipeline, self.dtype = model, pipeline, dtype
        self.model.eval()
        self.x, self.y, self.bs = x, y, int(batch_size)
        self.params = [(n, p) for n, p in model.named_parameters() if in_mask(n, p)]
        self.names = [n for n, _ in self.params]
        self.shapes = [p.shape for _, p in self.params]
        self.sizes = [p.numel() for _, p in self.params]
        self.n = sum(self.sizes)
        for n, p in model.named_parameters():
            p.requires_grad_(in_mask(n, p))
        self.N = int(x.shape[0])
        self.hvp_calls = 0

    def _flat(self, grads):
        return torch.cat([g.reshape(-1) for g in grads])

    def _split(self, v):
        out, i = [], 0
        for sh, k in zip(self.shapes, self.sizes):
            out.append(v[i:i + k].view(sh))
            i += k
        return out

    def vector(self):
        return torch.cat([p.detach().reshape(-1) for _, p in self.params]).to(torch.float64)

    def set_vector(self, v):
        with torch.no_grad():
            for (_, p), t in zip(self.params, self._split(v)):
                p.copy_(t.to(p.dtype))

    def batches(self):
        for i in range(0, self.N, self.bs):
            yield self.pipeline(self.x[i:i + self.bs]).to(self.dtype), self.y[i:i + self.bs]

    def loss_grad(self, need_grad=True):
        tot, g = 0.0, None
        ps = [p for _, p in self.params]
        for xb, yb in self.batches():
            w = xb.shape[0] / self.N
            loss = F.cross_entropy(self.model(xb), yb)
            tot += float(loss) * w
            if need_grad:
                gb = torch.autograd.grad(loss, ps)
                gb = self._flat(gb).to(torch.float64) * w
                g = gb if g is None else g + gb
        return tot, g

    def hvp(self, v):
        """H v for a float64 vector v; returns float64."""
        ps = [p for _, p in self.params]
        vs = [t.to(p.dtype) for t, p in zip(self._split(v), ps)]
        out = None
        for xb, yb in self.batches():
            w = xb.shape[0] / self.N
            loss = F.cross_entropy(self.model(xb), yb)
            gb = torch.autograd.grad(loss, ps, create_graph=True)
            dot = sum((a * b).sum() for a, b in zip(gb, vs))
            hb = torch.autograd.grad(dot, ps)
            hb = self._flat(hb).to(torch.float64) * w
            out = hb if out is None else out + hb
        self.hvp_calls += 1
        return out


def make_objective(ev, method, model_state, state, policy, probe, prec="f32", cache_key=None):
    """Frozen-BN objective on ``ev``'s model: saved statistics or centre-recalibrated ones."""
    from .evaluator import as_state
    from . import common as C
    dtype = torch.float64 if prec == "f64" else torch.float32
    model, ctrl = ev.model64(method) if prec == "f64" else ev.model(method)
    with torch.no_grad():
        if policy == C.CFROZEN:
            stats = ev.centre_stats(cache_key, method, model_state, state, prec)
        model.load_state_dict({k: v.to(ev.device) for k, v in model_state.items()}, strict=True)
        model.to(dtype)
        if policy == C.CFROZEN:
            for k, v in stats.items():
                model.get_buffer(k).copy_(v)
        elif policy != C.SAVED:
            raise ValueError("Hessian is defined here only for frozen BN policies, not %s" % policy)
    ctrl.set_state(as_state(state))
    x, y = ev.splits[probe]
    return FrozenBNObjective(model, ev.pipeline, x, y, ev.bs, dtype)


def block_scales(obj, center_state: dict):
    """Per-coordinate A diagonal (float64, on the objective's device) and block counts."""
    parts, n_blocks, n_zero, per_layer = [], 0, 0, []
    for name, sh in zip(obj.names, obj.shapes):
        w = center_state[name].to(torch.float64)
        bn = w.reshape(w.shape[0], -1).norm(dim=1)
        n_blocks += int((bn > 0).sum())
        n_zero += int((bn == 0).sum())
        per_layer.append({"name": name, "blocks": int(w.shape[0]), "zero_blocks": int((bn == 0).sum())})
        parts.append(bn.view(-1, *([1] * (w.dim() - 1))).expand_as(w).reshape(-1))
    a = torch.cat(parts).to(obj.x.device)
    return a, {"n_nonzero_blocks": n_blocks, "n_zero_norm_blocks": n_zero, "layers": per_layer}


def block_index(obj):
    """Block id per coordinate (for the r(delta) metric)."""
    ids, off = [], 0
    for sh, k in zip(obj.shapes, obj.sizes):
        per = k // sh[0]
        ids.append(torch.arange(sh[0]).repeat_interleave(per) + off)
        off += sh[0]
    return torch.cat(ids), off


def rms_relative(delta, center_vec, bid, n_blocks):
    """r(delta) over nonzero blocks and the count G used."""
    dn = torch.zeros(n_blocks, dtype=torch.float64).index_add_(0, bid, delta.cpu().double() ** 2)
    tn = torch.zeros(n_blocks, dtype=torch.float64).index_add_(0, bid, center_vec.cpu().double() ** 2)
    nz = tn > 0
    return float(torch.sqrt((dn[nz] / tn[nz]).mean())), int(nz.sum())


def lanczos(op, n, v0, m_max, check_every=10, tol=1e-3, floor=1e-2, device="cpu", mask=None,
            log=None, min_iter=20):
    """Extreme eigenpairs (two largest, one smallest algebraic) of a symmetric op.

    Converged when for each target Ritz pair the residual estimate
    |beta_j * s_{j,i}| <= tol * max(|theta_i|, floor * |theta_max|).
    Returns a dict with values, vectors (float64), histories, iteration count.
    """
    t0 = time.time()
    Q = torch.zeros(m_max + 1, n, dtype=torch.float64, device=device)
    alpha, beta = [], []
    q = v0.to(device, torch.float64)
    if mask is not None:
        q = q * mask
    q = q / q.norm()
    Q[0] = q
    hist, converged, j = [], False, 0
    b_prev = 0.0
    for j in range(m_max):
        w = op(Q[j])
        if mask is not None:
            w = w * mask
        a = float(Q[j] @ w)
        w = w - a * Q[j] - (b_prev * Q[j - 1] if j > 0 else 0.0)
        for _ in range(2):
            w = w - Q[:j + 1].T @ (Q[:j + 1] @ w)
        b = float(w.norm())
        alpha.append(a)
        beta.append(b)
        if b < 1e-12:
            j += 1
            break
        Q[j + 1] = w / b
        b_prev = b
        k = j + 1
        if (k >= min_iter and k % check_every == 0) or k == m_max:
            T = np.diag(alpha) + np.diag(beta[:-1], 1) + np.diag(beta[:-1], -1)
            th, S = np.linalg.eigh(T)
            est = np.abs(b * S[-1, :])
            scale = np.maximum(np.abs(th), floor * np.abs(th[-1]))
            idx = {"top1": k - 1, "top2": k - 2, "min": 0}
            rec = {"iter": k, "theta": {kk: float(th[i]) for kk, i in idx.items()},
                   "resid_est": {kk: float(est[i]) for kk, i in idx.items()},
                   "rel_resid_est": {kk: float(est[i] / scale[i]) for kk, i in idx.items()},
                   "seconds": time.time() - t0}
            hist.append(rec)
            if log:
                log("lanczos %d: top1 %.5g (%.1e) top2 %.5g (%.1e) min %.5g (%.1e)" % (
                    k, rec["theta"]["top1"], rec["rel_resid_est"]["top1"], rec["theta"]["top2"],
                    rec["rel_resid_est"]["top2"], rec["theta"]["min"], rec["rel_resid_est"]["min"]))
            if all(v <= tol for v in rec["rel_resid_est"].values()):
                converged = True
                j = k
                break
    else:
        j = m_max
    k = len(alpha)
    T = np.diag(alpha) + np.diag(beta[:k - 1], 1) + np.diag(beta[:k - 1], -1)
    th, S = np.linalg.eigh(T)
    idx = {"top1": k - 1, "top2": k - 2, "min": 0}
    Sg = torch.as_tensor(S, dtype=torch.float64, device=device)
    vecs, vals, resid = {}, {}, {}
    for kk, i in idx.items():
        y = Q[:k].T @ Sg[:, i]
        y = y / y.norm()
        Hy = op(y)
        if mask is not None:
            Hy = Hy * mask
        rq = float(y @ Hy)
        vecs[kk] = y
        vals[kk] = float(th[i])
        resid[kk] = {"ritz": float(th[i]), "rayleigh_quotient": rq, "resid_norm": float((Hy - th[i] * y).norm()),
                     "resid_est": float(abs(beta[k - 1] * S[-1, i])),
                     "scale": float(max(abs(th[i]), floor * abs(th[-1])))}
        resid[kk]["rel_resid"] = resid[kk]["resid_norm"] / resid[kk]["scale"]
    ortho = float((Q[:k] @ Q[:k].T - torch.eye(k, dtype=torch.float64, device=device)).abs().max())
    ok = {kk: resid[kk]["rel_resid"] <= tol for kk in idx}
    return {"values": vals, "vectors": vecs, "residuals": resid, "history": hist, "iterations": k,
            "converged_estimate": converged, "converged_explicit": ok, "orthogonality_error": ortho,
            "tol": tol, "floor": floor, "seconds": time.time() - t0,
            "ritz_values_all": th.tolist()[:5] + th.tolist()[-5:]}
