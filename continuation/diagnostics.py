r"""Complexity / distortion diagnostics, computed independently of any family.

These are *measurements*, not certificates.  A family is never required to have
an exact scalar complexity certificate, and equal TV or equal MSE between two
transformations does **not** mean equal semantic information or equal learning
difficulty.  Gaussian ``sigma`` in particular is not an exact per-image TV budget.

Total variation convention
--------------------------
Channelwise discrete **isotropic** total variation on pixel values,

    TV(z) = sum_{c,p} sqrt( (D1 z_c)_p^2 + (D2 z_c)_p^2 ),

with:

* **forward differences**, ``(D1 z)[i,j] = z[i+1,j] - z[i,j]`` along height and
  ``(D2 z)[i,j] = z[i,j+1] - z[i,j]`` along width;
* **unit pixel spacing** (``h = 1``), so values are in image-intensity units per
  pixel, with intensities in ``[0,1]``;
* a **zero normal difference** (Neumann) boundary treatment: the forward
  difference is defined to be 0 in the last row / last column, so no wrap-around
  or phantom edge is created on the connected image grid;
* channels are summed, never coupled inside the square root.

This TV is a *roughness functional on pixel values*.  It is not a
total-variation distance between probability distributions.
"""
from __future__ import annotations

import math

import torch

TV_CONVENTION = {
    "kind": "channelwise discrete isotropic total variation",
    "differences": "forward",
    "pixel_spacing": 1.0,
    "boundary": "zero normal difference (Neumann) at last row/column",
    "channel_coupling": "none; channels summed after the per-pixel square root",
    "value_range": "[0,1] float image intensities",
}


def forward_differences(x: torch.Tensor):
    """Return ``(D1 x, D2 x)`` for ``[N,C,H,W]`` with zero normal difference."""
    if x.dim() != 4:
        raise ValueError("expected [N,C,H,W], got %s" % (tuple(x.shape),))
    d1 = torch.zeros_like(x)
    d2 = torch.zeros_like(x)
    d1[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    d2[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]
    return d1, d2


def total_variation(x: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Per-image isotropic TV, shape ``[N]``.

    ``eps`` is a smoothing constant inside the square root; it defaults to 0
    (the exact non-smooth functional) and is exposed only for future solvers
    that need a differentiable surrogate.
    """
    d1, d2 = forward_differences(x.to(torch.float64))
    mag = torch.sqrt(d1 * d1 + d2 * d2 + eps)
    return mag.sum(dim=(1, 2, 3))


def per_image_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-image mean squared error, shape ``[N]``, in ``[0,1]`` intensity units."""
    if a.shape != b.shape:
        raise ValueError("shape mismatch %s vs %s" % (tuple(a.shape), tuple(b.shape)))
    diff = (a.to(torch.float64) - b.to(torch.float64)) ** 2
    return diff.mean(dim=(1, 2, 3))


def retained_tv_ratio(transformed: torch.Tensor, original: torch.Tensor):
    """``TV(T_eta x) / TV(x)`` per image, with explicit zero-TV handling.

    Returns ``(ratio, valid_mask)``.  Images with ``TV(x) == 0`` (spatially
    constant in every channel) have an undefined ratio: they are marked invalid
    and their ratio is ``nan`` rather than being silently set to 1.
    """
    tv_x = total_variation(original)
    tv_t = total_variation(transformed)
    valid = tv_x > 0
    ratio = torch.full_like(tv_x, float("nan"))
    ratio[valid] = tv_t[valid] / tv_x[valid]
    return ratio, valid


def retained_contrast(transformed: torch.Tensor, original: torch.Tensor):
    r"""``rho = ||z - mean(x)||_2 / ||x - mean(x)||_2`` per image, with the
    per-channel mean of the **original** image removed from both.

    Distinguishes spatial simplification from a plain loss of contrast: a
    transformation can keep TV low by flattening structure (rho falls) or by
    reorganising it (rho stays near 1).  Returns ``(rho, valid_mask)``; images
    that are spatially constant in every channel have a zero denominator, so
    their ratio is ``nan`` and they are marked invalid rather than being given a
    fabricated value.
    """
    x = original.to(torch.float64)
    z = transformed.to(torch.float64)
    mean = x.mean(dim=(2, 3), keepdim=True)
    denom = (x - mean).flatten(1).norm(dim=1)
    numer = (z - mean).flatten(1).norm(dim=1)
    valid = denom > 0
    rho = torch.full_like(denom, float("nan"))
    rho[valid] = numer[valid] / denom[valid]
    return rho, valid


def _summ(v: torch.Tensor) -> dict:
    v = v[torch.isfinite(v)]
    if v.numel() == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": int(v.numel()),
        "mean": float(v.mean()),
        "std": float(v.std(unbiased=True)) if v.numel() > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
    }


def transform_statistics(transform, eta: float, images: torch.Tensor,
                         meta: dict | None = None) -> dict:
    """Reconstruction MSE and retained TV ratio on a fixed image subset.

    ``images`` are the *original* float images in ``[0,1]``.  The transformation
    is always applied to these originals, never to an already-transformed batch.
    """
    result = transform.apply(images, eta, meta)
    y = result.images
    mse = per_image_mse(y, images)
    ratio, valid = retained_tv_ratio(y, images)
    tv_x = total_variation(images)
    tv_y = total_variation(y)
    rho, rho_valid = retained_contrast(y, images)
    return {
        "family": transform.name,
        "parameter_name": transform.parameter_name,
        "parameter": float(eta),
        "is_target_endpoint": bool(transform.is_target(eta)),
        "n_images": int(images.shape[0]),
        "reconstruction_mse": _summ(mse),
        "retained_tv_ratio": _summ(ratio),
        "retained_contrast": _summ(rho),
        "n_zero_tv_images": int((~valid).sum()),
        "n_zero_contrast_images": int((~rho_valid).sum()),
        "tv_original": _summ(tv_x),
        "tv_transformed": _summ(tv_y),
        "tv_convention": TV_CONVENTION,
        "transform_info": {k: v for k, v in result.info.items()},
    }


# --------------------------------------------------------------------------
# Transition diagnostics for *future* continuation experiments.
# Implemented and unit-tested here; not exercised by Experiment 0.
# --------------------------------------------------------------------------

def gradient_at(model, loss_fn, transform, eta: float, images: torch.Tensor,
                labels: torch.Tensor, normalize=None):
    """Flat gradient of ``L_eta`` at the current weights, on fixed probe examples.

    The caller is responsible for controlling model state (eval/train mode);
    ``continuation.engine`` always evaluates these in eval mode with the same
    probe batch for both objectives.
    """
    model.zero_grad(set_to_none=True)
    x = transform(images, eta)
    if normalize is not None:
        x = normalize(x)
    loss = loss_fn(model(x), labels)
    grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad])
    flat = torch.cat([g.reshape(-1) for g in grads])
    model.zero_grad(set_to_none=True)
    return flat.detach(), float(loss.detach())


def compare_gradients(g_old: torch.Tensor, g_new: torch.Tensor,
                      norm_threshold: float = 1e-8) -> dict:
    """Norms, difference norm and cosine similarity between two objectives' gradients.

    Near a stationary point of the old objective the cosine is unreliable or
    undefined.  When either norm falls below ``norm_threshold`` the cosine is
    reported as ``None`` with ``cosine_available = False`` and a stated reason --
    it is never rescued by a silent denominator adjustment.  Norms are always
    reported alongside, because a cosine on its own means little.
    """
    n_old = float(g_old.norm())
    n_new = float(g_new.norm())
    out = {
        "grad_norm_old": n_old,
        "grad_norm_new": n_new,
        "grad_diff_norm": float((g_new - g_old).norm()),
        "norm_threshold": norm_threshold,
        "cosine": None,
        "cosine_available": False,
        "cosine_unavailable_reason": None,
    }
    if n_old < norm_threshold or n_new < norm_threshold:
        out["cosine_unavailable_reason"] = (
            "gradient norm below threshold (old=%.3e, new=%.3e < %.3e); "
            "cosine is undefined/unreliable near a stationary point" % (n_old, n_new, norm_threshold)
        )
        return out
    cos = float(torch.dot(g_old, g_new) / (n_old * n_new))
    out["cosine"] = max(-1.0, min(1.0, cos))
    out["cosine_available"] = True
    return out


def summarize_across_seeds(values) -> dict:
    """Mean, sample std and standard error across paired seeds."""
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "sem": None, "values": []}
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        std = math.sqrt(var)
        sem = std / math.sqrt(n)
    else:
        std, sem = 0.0, None
    return {"n": n, "mean": mean, "std": std, "sem": sem, "values": vals}
