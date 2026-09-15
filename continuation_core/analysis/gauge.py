r"""BatchNorm gauge fixing, and the gauge-minimal distance between two solutions.

Why any weight-space distance needs this
----------------------------------------
Every main-path convolution here is immediately followed by a BatchNorm, and
BatchNorm divides out any positive per-output-filter rescaling of the
convolution before it.  A fraction of ``||W* - W0||`` is therefore pure gauge:
movement along directions that leave the represented function *exactly*
unchanged.  Reporting a distance, a KL term or a curvature without bounding that
fraction leaves the objection of Dinh et al. (2017) open.

The transformation
------------------
For one output filter ``f`` and any ``alpha > 0``,

    W_f            -> alpha * W_f
    running_mean_f -> alpha * running_mean_f
    running_var_f  -> alpha^2 * running_var_f

leaves the layer output identical in **both** modes.  In train mode the batch
statistics rescale on their own; in eval mode the stored statistics must be
rescaled explicitly, which is what the two buffer lines do.  ``gamma`` and
``beta`` are untouched, and the running statistics are *buffers*, not learnable
parameters, so the whole compensation falls outside any KL.

``alpha <= 0`` is **not** a symmetry: a negative scale survives as a sign on the
output, because BatchNorm divides by a standard deviation and not by the scale.

The gauge-minimal distance
--------------------------
The prior must be fixed before the data are seen, so ``W0`` stays in its
initialisation gauge.  The posterior may be placed anywhere, and every gauge
image of ``W*`` represents the same function, so the closest one is taken:

    d_min^2 = sum_f min_{alpha>0} ||alpha W*_f - W0_f||^2
            = sum_f ||W0_f||^2 sin^2(theta_f)        when cos theta_f > 0

with minimiser ``alpha_f = <W*_f, W0_f> / ||W*_f||^2``.  Filters with
``cos theta_f <= 0`` have no positive optimum; they are counted and reported
rather than being given the open infimum.

Measured on the exploratory branch: the gauge accounts for **3 %** of the
distance and the curvature's gauge sensitivity for at most **2 %**, against
effects of 35 to 40 %.  The construction is therefore a control that passes, not
a correction that rescues anything -- but it has to be run to say so.
"""
from __future__ import annotations

import torch


def conv_bn_pairs(state: dict) -> list:
    """``(conv_weight, bn_prefix)`` for every gauge-free convolution.

    Derived from the naming rule and then **verified**: the BatchNorm must exist
    and its channel count must match the convolution's output channels, or the
    pair is rejected.  A convolution with no BatchNorm behind it has no gauge
    freedom and must not be rescaled.
    """
    pairs = []
    for k, w in state.items():
        if not k.endswith(".weight") or not torch.is_tensor(w) or w.dim() != 4:
            continue
        stem = k[: -len(".weight")]
        last = stem.rsplit(".", 1)[-1]
        if not last.startswith("conv"):
            continue
        head = stem.rsplit(".", 1)
        prefix = (head[0] + ".bn" + last[4:]) if len(head) == 2 else "bn" + last[4:]
        need = [prefix + s for s in (".weight", ".running_mean", ".running_var")]
        if any(n not in state for n in need):
            continue
        if state[need[0]].numel() != w.shape[0]:
            continue
        pairs.append((k, prefix))
    return pairs


def _filter_norms(w: torch.Tensor) -> torch.Tensor:
    return w.to(torch.float64).reshape(w.shape[0], -1).norm(dim=1)


def rescale_(state: dict, conv_key: str, bn_prefix: str, alpha: torch.Tensor) -> None:
    a = alpha.to(torch.float64)
    if not bool((a > 0).all()):
        raise ValueError("gauge scale must be strictly positive")
    w = state[conv_key]
    state[conv_key] = (w.to(torch.float64) * a.reshape(-1, *([1] * (w.dim() - 1)))).to(w.dtype)
    for suffix, power in ((".running_mean", 1), (".running_var", 2)):
        key = bn_prefix + suffix
        state[key] = (state[key].to(torch.float64) * a ** power).to(state[key].dtype)


def align_to(final: dict, init: dict, eps: float = 1e-12):
    """Regauge a copy of ``final`` to the image closest to ``init``."""
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in final.items()}
    pairs = conv_bn_pairs(final)
    n_deg = n_filt = 0
    for ck, bn in pairs:
        wf = final[ck].to(torch.float64).reshape(final[ck].shape[0], -1)
        w0 = init[ck].to(torch.float64).reshape(init[ck].shape[0], -1)
        a = (wf * w0).sum(dim=1) / (wf * wf).sum(dim=1).clamp_min(eps)
        bad = a <= 0
        n_deg += int(bad.sum())
        n_filt += a.numel()
        rescale_(out, ck, bn, torch.where(bad, torch.ones_like(a), a))
    return out, {"n_gauge_free_tensors": len(pairs), "n_filters": n_filt,
                 "n_filters_without_positive_optimum": n_deg}


def distance_breakdown(final: dict, init: dict, learnable) -> dict:
    """Split ``||W* - W0||`` into its gauge-free and gauge-fixed parts."""
    pairs = {k for k, _ in conv_bn_pairs(final)}

    def split(a):
        d = {"conv_gauge_free": 0.0, "other": 0.0}
        for k in learnable:
            v = float(((a[k].to(torch.float64) - init[k].to(torch.float64)) ** 2).sum())
            d["conv_gauge_free" if k in pairs else "other"] += v
        return d

    raw = split(final)
    aligned, report = align_to(final, init)
    mn = split(aligned)
    tot_raw, tot_min = sum(raw.values()) ** 0.5, sum(mn.values()) ** 0.5
    return {"distance_raw": tot_raw, "distance_gauge_minimal": tot_min,
            "ratio": tot_min / tot_raw if tot_raw else None,
            "raw_conv": raw["conv_gauge_free"] ** 0.5, "raw_other": raw["other"] ** 0.5,
            "min_conv": mn["conv_gauge_free"] ** 0.5, "min_other": mn["other"] ** 0.5,
            "gauge": report}


@torch.no_grad()
def verify_equivalence(model, state_a: dict, state_b: dict, x: torch.Tensor) -> dict:
    """Max logit difference between two states -- measured, never assumed."""
    model.load_state_dict(state_a)
    model.eval()
    ya = model(x).to(torch.float64)
    model.load_state_dict(state_b)
    model.eval()
    yb = model(x).to(torch.float64)
    return {"max_abs_logit_diff": float((ya - yb).abs().max()),
            "logit_scale": float(ya.abs().max()),
            "max_abs_prob_diff": float((ya.softmax(1) - yb.softmax(1)).abs().max())}
