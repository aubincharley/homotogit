r"""What the finished function looks at, and how hard it moves when pushed.

Everything here is a property of the **learned function**, not of the weights,
so none of it depends on a parametrisation.  That matters: a BatchNorm network's
weights can be rescaled filter by filter without changing the function, which
makes every weight-space quantity defined only up to a gauge (see
:mod:`continuation_core.analysis.gauge`).  The quantities below have no such
freedom.

Three groups, answering three different questions.

**Which input frequencies does the function respond to?**  The Jacobian of the
logits with respect to the image, ``J = df/dx`` in ``[0,1]`` intensity units with
the channel normalisation *inside* ``f``, is an image-shaped object per logit.
Its 2-D DFT says which input frequencies each logit responds to.  This is exact
and needs no sampling of directions: by Parseval, the sum of ``||J e||^2`` over
any orthonormal family spanning a radial ring **is** that ring's DFT energy, so
ten backward passes give the whole curve.

A ring holds very different numbers of modes (roughly ``2 pi k``), so the raw
ring sum rises with ``k`` for a purely geometric reason.  The per-mode density is
the quantity that answers "does this function prefer high frequencies?", and both
are returned.

**How large is the sensitivity, and is it concentrated?**  The network has ten
outputs, so ``J`` is ``10 x 3072`` of rank at most ten and ``J J^T`` is a ``10 x
10`` matrix: the **entire** singular spectrum is exact from the same ten backward
passes.  No power iteration, no truncation.  ``||J||_F``, ``||J||_2`` and the
participation ratio of the singular values follow.

One warning carried over from the exploratory work: ``||J||_F`` correlated
``-0.83`` with the *training* error across conditions there, so it is largely a
proxy for how hard a network fitted its training set.  Any contrast on it must
be read next to the training error, never alone.

**How far does the function actually move?**  ``||J||`` bounds
``||f(x+d) - f(x)||`` only to first order, and the exploratory measurements found
no scale at which first order describes this function: the ratio ``S(eps) / (eps
sigma_max)`` was already 0.85 at the smallest measurable amplitude and fell to
0.03.  The amplitude curve

    S(eps) = E_x || f(x + eps v_1(x)) - f(x) ||_2 ,   v_1(x) = top right singular
                                                     vector of J(x)

is therefore measured over a wide grid rather than at one point.  ``v_1`` is
recomputed **per image**: it is a field of directions on the data manifold, not
one global direction.  The amplitude is shared, so every network is compared at
equal perturbation energy.
"""
from __future__ import annotations

import torch

from ..controller import InterventionState


def _unit(v: torch.Tensor) -> torch.Tensor:
    n = v.reshape(v.shape[0], -1).norm(dim=1).clamp_min(1e-12)
    return v / n.view(-1, *([1] * (v.dim() - 1)))


def _normalize(ev, x_float: torch.Tensor) -> torch.Tensor:
    """The pipeline's channel normalisation, applied to a ``[0,1]`` float batch.

    ``InputPipeline`` takes uint8; the Jacobian is wanted with respect to the
    ``[0,1]`` image, with the normalisation inside the differentiated map, which
    is exactly what the deployed network computes.
    """
    return (x_float - ev.pipeline.mean.to(x_float.dtype)) / ev.pipeline.std.to(x_float.dtype)


def _forward(ev, x_float: torch.Tensor) -> torch.Tensor:
    return ev.model(_normalize(ev, x_float))


def float_images(ev, n: int | None = None) -> torch.Tensor:
    x = ev.images if n is None else ev.images[:n]
    return x.to(torch.float32) / 255.0


def input_jacobian(ev, x_float: torch.Tensor, n_classes: int | None = None):
    """``J`` for one batch, shape ``[B, K, C, H, W]``, one Jacobian per image."""
    k = int(n_classes or ev.model(_normalize(ev, x_float[:1])).shape[1])
    x = x_float.detach().clone().requires_grad_(True)
    logits = _forward(ev, x)
    rows = []
    for c in range(k):
        g, = torch.autograd.grad(logits[:, c].sum(), x, retain_graph=(c < k - 1))
        rows.append(g.detach())
    return torch.stack(rows, dim=1), logits.detach()


def singular_spectrum(J: torch.Tensor) -> torch.Tensor:
    """All singular values per image, from the ``K x K`` Gram matrix, in float64."""
    flat = J.reshape(J.shape[0], J.shape[1], -1).to(torch.float64)
    gram = torch.einsum("bki,bli->bkl", flat, flat)
    gram = 0.5 * (gram + gram.transpose(1, 2))
    return torch.linalg.eigvalsh(gram).clamp_min(0.0).flip(-1).sqrt()


def top_singular_direction(J: torch.Tensor) -> torch.Tensor:
    """Unit input direction of the largest singular value, per image.

    From the Gram matrix's top eigenvector ``u``: the matching right singular
    vector is ``J^T u`` renormalised, which needs no large SVD.
    """
    shape = J.shape
    flat = J.reshape(shape[0], shape[1], -1).to(torch.float64)
    gram = torch.einsum("bki,bli->bkl", flat, flat)
    gram = 0.5 * (gram + gram.transpose(1, 2))
    u = torch.linalg.eigh(gram)[1][..., -1]
    v = torch.einsum("bk,bki->bi", u, flat)
    return _unit(v.reshape(shape[0], *shape[2:]).to(J.dtype))


def radial_bins(n: int, device):
    f = torch.fft.fftfreq(n, d=1.0 / n, device=device)
    r = (f.view(-1, 1) ** 2 + f.view(1, -1) ** 2).sqrt()
    idx = r.round().to(torch.long)
    return idx, int(idx.max()) + 1


def frequency_profile(J: torch.Tensor, cutoff: int = 8) -> dict:
    """Radial DFT energy of the Jacobian: fractions, per-mode density, summaries."""
    dev, n = J.device, int(J.shape[-1])
    idx, n_bins = radial_bins(n, dev)
    spec = torch.fft.fft2(J.to(torch.float32), norm="ortho")
    e = (spec.real ** 2 + spec.imag ** 2).sum(dim=(0, 1, 2)).to(torch.float64)
    ring = torch.zeros(n_bins, dtype=torch.float64, device=dev)
    ring.scatter_add_(0, idx.reshape(-1), e.reshape(-1))
    counts = torch.zeros(n_bins, dtype=torch.float64, device=dev)
    counts.scatter_add_(0, idx.reshape(-1), torch.ones_like(e).reshape(-1))
    total = float(ring.sum())
    frac = [float(v) / total for v in ring]
    return {"ring_energy_fraction": frac,
            "ring_mode_count": [int(c) for c in counts],
            "ring_energy_per_mode": [float(r) / max(float(c), 1.0) / total
                                     for r, c in zip(ring, counts)],
            "mean_radius": sum(k * f for k, f in enumerate(frac)),
            "fraction_above": {str(k): 1.0 - sum(frac[:k]) for k in (4, cutoff, 12)}}


@torch.no_grad()
def amplitude_curve(ev, x_float, direction, epsilons, labels=None) -> list:
    """``S(eps)`` along a fixed unit direction, with the accuracy alongside.

    The accuracy is reported but is **not** the quantity of interest: on the
    exploratory branch the accuracy under perturbation predicted nothing while
    the displacement predicted strongly, so conflating them hides the effect.
    """
    base = _forward(ev, x_float)
    out = []
    for e in epsilons:
        z = _forward(ev, x_float + float(e) * direction)
        row = {"epsilon": float(e), "displacement": float((z - base).norm(dim=1).mean())}
        if labels is not None:
            row["accuracy"] = float((z.argmax(1) == labels).float().mean())
        out.append(row)
    return out


def probe(ev, *, n_images: int = 1000, batch_size: int = 250,
          epsilons=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 6.0, 12.0),
          state="target") -> dict:
    """Every function-side quantity for one checkpoint, in one pass.

    Evaluated at the **target** intervention state by default: at the last epoch
    the operator is the exact identity in every method, so all arms are probed on
    the same network and the comparison needs no correction.
    """
    st = ev.state(state) if isinstance(state, str) else state
    previous = ev.controller.state
    ev.controller.set_state(st)
    was = ev.model.training
    ev.model.eval()
    try:
        n = min(n_images, int(ev.images.shape[0]))
        svs, freqs, curves, fro2 = [], [], [], 0.0
        for i in range(0, n, batch_size):
            xb = float_images(ev)[i:i + batch_size]
            yb = ev.labels[i:i + batch_size]
            J, _ = input_jacobian(ev, xb)
            svs.append(singular_spectrum(J))
            freqs.append(frequency_profile(J))
            fro2 += float((J.to(torch.float64) ** 2).sum())
            curves.append(amplitude_curve(ev, xb, top_singular_direction(J),
                                          epsilons, yb))
            del J
        sv = torch.cat(svs, dim=0)
        s2 = sv ** 2
        nb = len(curves)
        nring = len(freqs[0]["ring_energy_fraction"])
        curve = [{"epsilon": epsilons[j],
                  "displacement": sum(c[j]["displacement"] for c in curves) / nb,
                  "accuracy": sum(c[j]["accuracy"] for c in curves) / nb}
                 for j in range(len(epsilons))]
        smax = float(sv[:, 0].mean())
        return {
            "checkpoint": ev.path, "split": ev.split, "n_images": n,
            "state": st.to_dict() if hasattr(st, "to_dict") else str(st),
            "jacobian": {
                "frobenius_mean": float(s2.sum(dim=1).sqrt().mean()),
                "frobenius_per_image_rms": (fro2 / n) ** 0.5,
                "spectral_mean": smax,
                "ratio_spectral_over_frobenius":
                    float((sv[:, 0] / s2.sum(dim=1).sqrt().clamp_min(1e-30)).mean()),
                "participation_mean":
                    float((s2.sum(dim=1) ** 2 / (s2 ** 2).sum(dim=1).clamp_min(1e-30)).mean()),
                "singular_values_mean": [float(v) for v in sv.mean(dim=0)]},
            "frequency": {
                "ring_energy_per_mode": [sum(f["ring_energy_per_mode"][j] for f in freqs) / nb
                                         for j in range(nring)],
                "ring_energy_fraction": [sum(f["ring_energy_fraction"][j] for f in freqs) / nb
                                         for j in range(nring)],
                "mean_radius": sum(f["mean_radius"] for f in freqs) / nb,
                "fraction_above": {k: sum(f["fraction_above"][k] for f in freqs) / nb
                                   for k in freqs[0]["fraction_above"]}},
            "amplitude_curve": curve,
            # 1 would mean the first-order description holds at that amplitude
            "linearity_ratio": [{"epsilon": p["epsilon"],
                                 "ratio": p["displacement"] / (p["epsilon"] * smax)}
                                for p in curve],
        }
    finally:
        ev.model.train(was)
        if previous is not None:
            ev.controller.set_state(previous)
