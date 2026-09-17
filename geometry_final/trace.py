"""Block-resolved Hutchinson estimates of three signed traces from one HVP per draw.

For the frozen-BN objective of landscape_v3 (``hessian.FrozenBNObjective``: exact
double back-propagation, ordinary coordinates over the 268,336 masked weights) and
a Rademacher vector z, one product Hz gives, for every block g (conv output filter
or fc row),

    q_g = z_g^T (Hz)_g ,        E[q_g] = tr(H[g,g]).

Aggregates per draw:

    ordinary    sum_g q_g                         -> tr(H)
    relative    sum_g ||theta_g||^2 q_g           -> tr(A^T H A),  A_g = ||theta_g|| I
    covariance  sum_g ||theta_g||^2 / p_g  q_g    -> tr(H C) = E_d[d^T H d]

where C is the covariance of the filter-normalised random directions of v2/v3
(d_g = ||theta_g|| u_g / ||u_g||, u Gaussian, so Cov(d_g) = ||theta_g||^2 / p_g I).
Each is unbiased because E[z z^T] = I within and across blocks; the cross-block
terms vanish in expectation, not per draw.  All three are signed: H is indefinite.

Control variate (optional, offline).  For any fixed unit vector u with scalar c,
E[(z^T W u)(u^T z)] = u^T W u for diagonal W, so

    zWHz - c (z^T W u)(u^T z) + c u^T W u

is unbiased for tr(WH) whatever u and c are.  With u an eigenvector of H and
c its eigenvalue it removes that direction's contribution to the variance.
"""
from __future__ import annotations

import numpy as np
import torch


def block_ids_and_weights(names, shapes, center_state):
    """Block id per coordinate, ||theta_g||^2 and p_g per block (float64 numpy)."""
    ids, norm2, sizes, off = [], [], [], 0
    for name, sh in zip(names, shapes):
        w = center_state[name].detach().to("cpu", torch.float64)
        flat = w.reshape(w.shape[0], -1)
        per = flat.shape[1]
        ids.append(torch.arange(w.shape[0]).repeat_interleave(per) + off)
        norm2.append((flat ** 2).sum(1).numpy())
        sizes.append(np.full(w.shape[0], per, dtype=np.int64))
        off += w.shape[0]
    return torch.cat(ids), np.concatenate(norm2), np.concatenate(sizes)


def weights(norm2, sizes):
    return {"ordinary": np.ones_like(norm2), "relative": norm2, "covariance": norm2 / sizes}


def per_block_quadratic(z, hz, bid, n_blocks):
    """q_g = z_g^T (Hz)_g, float64 on CPU."""
    prod = (z.to(torch.float64) * hz.to(torch.float64))
    return torch.zeros(n_blocks, dtype=torch.float64, device=prod.device).index_add_(0, bid.to(prod.device), prod).cpu().numpy()


def aggregates(q, w):
    return {k: float(np.dot(wk, q)) for k, wk in w.items()}


def summary(values):
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    sem = sd / np.sqrt(n) if n > 1 else float("nan")
    return {"n": int(n), "mean": mean, "sd": sd, "sem": float(sem),
            "sem_rel": float(sem / abs(mean)) if mean != 0 else float("inf")}


# ---- dense reference ---------------------------------------------------------------

class _Tiny(torch.nn.Module):
    """conv + BN + ReLU + max-pool + conv + BN + ReLU + fc: the operator family of ResNet-20-BN."""

    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.b1 = torch.nn.BatchNorm2d(4)
        self.c2 = torch.nn.Conv2d(4, 5, 3, padding=1, bias=False)
        self.b2 = torch.nn.BatchNorm2d(5)
        self.fc = torch.nn.Linear(5 * 4 * 4, 3)

    def forward(self, x):
        x = torch.relu(self.b1(self.c1(x)))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.relu(self.b2(self.c2(x)))
        return self.fc(x.flatten(1))


def dense_reference(n_draws=20000, seed=0):
    """Compare the block aggregation with the dense Hessian of a small frozen-BN network.

    Returns exact traces, the maximum per-draw error of the block formula against
    the dense quadratic form, the Monte Carlo estimates with their SEM, and whether
    BatchNorm buffers stayed bitwise unchanged."""
    from landscape_study.directions import in_mask
    from landscape_v3.hessian import FrozenBNObjective

    torch.manual_seed(seed)
    net = _Tiny().double()
    x = torch.randn(64, 3, 8, 8, dtype=torch.float64)
    y = torch.randint(0, 3, (64,))
    net.train()
    with torch.no_grad():
        net(x)                                     # non-trivial running statistics
    net.eval()
    buffers0 = {k: v.clone() for k, v in net.named_buffers()}
    obj = FrozenBNObjective(net, lambda t: t, x, y, batch_size=16, dtype=torch.float64)
    names, shapes = obj.names, obj.shapes
    state = {n: p.detach() for n, p in net.named_parameters()}
    bid, norm2, sizes = block_ids_and_weights(names, shapes, state)
    G = len(norm2)
    n = obj.n
    H = torch.stack([obj.hvp(e) for e in torch.eye(n, dtype=torch.float64)], 1)
    sym = float((H - H.T).abs().max() / H.abs().max())
    H = 0.5 * (H + H.T)
    wv = {k: torch.as_tensor(v)[bid] for k, v in weights(norm2, sizes).items()}
    exact = {k: float((wv[k] * torch.diagonal(H)).sum()) for k in wv}
    rng = np.random.default_rng(seed)
    est = {k: [] for k in wv}
    worst = 0.0
    for j in range(n_draws):
        z = torch.as_tensor(rng.integers(0, 2, n) * 2 - 1, dtype=torch.float64)
        hz = H @ z if j >= 50 else obj.hvp(z)       # the objective's HVP for the first 50 draws
        q = per_block_quadratic(z, hz, bid, G)
        agg = aggregates(q, weights(norm2, sizes))
        for k in wv:
            if j < 50:
                dense_q = float(z @ (wv[k] * (H @ z)))
                worst = max(worst, abs(agg[k] - dense_q) / max(abs(dense_q), 1e-12))
            est[k].append(agg[k])
    mc = {k: summary(v) for k, v in est.items()}
    z_scores = {k: (mc[k]["mean"] - exact[k]) / mc[k]["sem"] for k in wv}
    unchanged = all(torch.equal(buffers0[k], v) for k, v in net.named_buffers())
    masked = sum(int(p.numel()) for nm, p in net.named_parameters() if in_mask(nm, p))
    return {"n_params": n, "masked_params_check": masked == n, "n_blocks": G,
            "dense_hessian_asymmetry_rel": sym, "exact_traces": exact,
            "block_formula_vs_dense_quadratic_max_rel_err": worst,
            "monte_carlo": mc, "z_score_vs_exact": z_scores, "n_draws": n_draws,
            "bn_buffers_bitwise_unchanged": unchanged,
            "min_eigenvalue": float(torch.linalg.eigvalsh(H)[0]),
            "max_eigenvalue": float(torch.linalg.eigvalsh(H)[-1])}
