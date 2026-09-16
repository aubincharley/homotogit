"""Filter-wise normalised random directions (Li et al., 2018).

Mask: every convolution weight ``[C_out, C_in, kH, kW]`` and the linear weight
``[out, in]``.  One block = one output channel ``W[j]`` (conv) or one output row
(linear).  Biases and BatchNorm affine parameters get a zero direction.

The *underlying draws* ``V`` (float32, one tensor per masked parameter) are
drawn once from a named seed and saved.  A direction for a given centre is
``D_j = ||W_j|| V_j / ||V_j||`` computed in float64; a block with
``||W_j|| = 0`` gets ``D_j = 0``.  Reusing one ``V`` across paired methods gives
*matched relative perturbations*: the same random pattern scaled by each
solution's own filter norms, not one physical plane in parameter space.
No orthogonalisation is applied.
"""
from __future__ import annotations

import torch


def in_mask(name: str, tensor: torch.Tensor) -> bool:
    return name.endswith("weight") and tensor.dim() in (2, 4)


def draw_underlying(template: dict, names: list, seed: int) -> dict:
    g = torch.Generator().manual_seed(int(seed))
    return {n: torch.randn(template[n].shape, generator=g, dtype=torch.float32)
            for n in names if in_mask(n, template[n])}


def scale_to(center: dict, underlying: dict, names: list) -> tuple:
    """Direction for ``center`` (float64 dict over all names) and zero-block count."""
    out, zero_blocks = {}, 0
    for n in names:
        w = center[n].detach().to("cpu", torch.float64)
        if n not in underlying:
            out[n] = torch.zeros_like(w)
            continue
        v = underlying[n].to(torch.float64)
        wn = w.reshape(w.shape[0], -1).norm(dim=1)
        vn = v.reshape(v.shape[0], -1).norm(dim=1)
        factor = torch.where(wn > 0, wn / vn, torch.zeros_like(wn))
        zero_blocks += int((wn == 0).sum())
        out[n] = v * factor.view(-1, *([1] * (w.dim() - 1)))
    return out, zero_blocks


def normalization_report(center: dict, direction: dict, names: list) -> dict:
    worst, n_blocks, n_zero, zero_ok = 0.0, 0, 0, True
    for n in names:
        w = center[n].detach().to("cpu", torch.float64)
        d = direction[n]
        if not in_mask(n, w):
            zero_ok &= bool((d == 0).all())
            continue
        wn = w.reshape(w.shape[0], -1).norm(dim=1)
        dn = d.reshape(d.shape[0], -1).norm(dim=1)
        nz = wn > 0
        n_blocks += int(nz.sum())
        n_zero += int((~nz).sum())
        zero_ok &= bool((dn[~nz] == 0).all())
        if nz.any():
            worst = max(worst, float((dn[nz] / wn[nz] - 1).abs().max()))
    return {"max_abs_relative_norm_error": worst, "n_nonzero_blocks": n_blocks,
            "n_zero_norm_blocks": n_zero, "unmasked_and_zero_blocks_have_zero_direction": zero_ok,
            "masked_tensors": [n for n in names if in_mask(n, center[n])]}


def cosine(a: dict, b: dict, names: list) -> float:
    va = torch.cat([a[n].reshape(-1) for n in names])
    vb = torch.cat([b[n].reshape(-1) for n in names])
    return float(va @ vb / (va.norm() * vb.norm()))
