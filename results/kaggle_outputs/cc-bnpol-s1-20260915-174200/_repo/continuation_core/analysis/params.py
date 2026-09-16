"""Parameter vectors and filter-normalised directions.

Vectors cover **trainable parameters only**, in ``model.named_parameters()``
order.  BatchNorm running statistics are buffers, not coordinates: they are
handled by the evaluation's BN policy (see ``continuation_core.evaluate``).
"""
from __future__ import annotations

import torch


def parameter_names(model: torch.nn.Module) -> list:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def to_vector(state: dict, names: list, dtype=torch.float64) -> torch.Tensor:
    return torch.cat([state[n].detach().reshape(-1).to(dtype).cpu() for n in names])


def from_vector(vec: torch.Tensor, template: dict, names: list, device=None) -> dict:
    out, i = {}, 0
    for n in names:
        t = template[n]
        k = t.numel()
        out[n] = vec[i:i + k].reshape(t.shape).to(dtype=t.dtype,
                                                  device=device or t.device)
        i += k
    if i != vec.numel():
        raise ValueError("vector has %d entries, names cover %d" % (vec.numel(), i))
    return out


def filter_normalized_direction(state: dict, names: list, generator: torch.Generator,
                                ignore_1d: bool = True) -> dict:
    """Random direction scaled per output filter to the weights' filter norms.

    Li et al. (2018): for each weight tensor with more than one dimension, draw
    ``d ~ N(0, 1)`` and rescale each output-filter slice so that
    ``||d_f|| = ||w_f||``.  One-dimensional tensors (BatchNorm scale/shift and
    biases) get a zero direction when ``ignore_1d`` -- the convention used by
    that paper and by the benchmark's ``scripts/loss_landscape_blur.py``.
    """
    out = {}
    for n in names:
        w = state[n].detach().cpu().to(torch.float64)
        if w.dim() <= 1:
            out[n] = torch.zeros_like(w) if ignore_1d else torch.randn(
                w.shape, generator=generator, dtype=torch.float64)
            continue
        d = torch.randn(w.shape, generator=generator, dtype=torch.float64)
        wf = w.reshape(w.shape[0], -1).norm(dim=1)
        df = d.reshape(d.shape[0], -1).norm(dim=1).clamp_min(1e-12)
        out[n] = d * (wf / df).view(-1, *([1] * (w.dim() - 1)))
    return out
