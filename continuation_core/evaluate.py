"""Deterministic evaluation that never changes the model, controller or RNG.

BatchNorm policies
------------------
``running_stats``      ``model.eval()``: BN uses the running buffers stored with
                       the weights.  Use it for historical checkpoint evaluation;
                       it is what every recorded benchmark metric used.
``fixed_batch_stats``  BN normalises each evaluation batch with that batch's own
                       statistics (training-mode BN), on **cloned** buffers that
                       are discarded, so the checkpoint's running statistics are
                       never updated.  Batches are consecutive slices of a fixed
                       image tensor, so the result is deterministic for a fixed
                       ``batch_size`` -- and depends on it.  Use it for weights
                       that have no matching running statistics (points on a
                       PCA plane, perturbed weights, a different intervention
                       state than the one the buffers were estimated under).

Every result carries the path label, the intervention state, the per-site
sigma and the BN policy, so current-path, target-path and policy variants stay
distinguishable in outputs.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.func import functional_call

from .controller import InterventionController, InterventionState

BN_POLICIES = ("running_stats", "fixed_batch_stats")


def _devices(model):
    dev = next(model.parameters()).device
    return [dev] if dev.type == "cuda" else []


@torch.no_grad()
def evaluate(model: torch.nn.Module, controller: InterventionController, pipeline,
             images: torch.Tensor, labels: torch.Tensor, state: InterventionState,
             *, bn_policy: str = "running_stats", batch_size: int = 500,
             params: dict | None = None, buffers: dict | None = None,
             split: str = "unnamed", objective=None) -> dict:
    """Mean CE and accuracy of ``model`` under ``state``.

    ``params`` / ``buffers`` optionally override tensors by name (e.g. a point
    on a loss plane); the model's own tensors are left untouched either way.

    ``ce`` is reported whatever was trained, so runs with different objectives
    stay comparable on one scale.  Passing ``objective`` -- the per-sample-mean
    callable the run optimises -- adds ``obj`` beside it; without it the result
    keeps exactly the keys every recorded evaluation has.  ``obj`` is weighted by
    batch size, so it is exact for unequal final batches.
    """
    if bn_policy not in BN_POLICIES:
        raise ValueError("bn_policy must be one of %s" % (BN_POLICIES,))
    was_training = model.training
    previous = controller.state
    controller.set_state(state)
    per_site = controller.per_site_sigma()
    direct = params is None and buffers is None and bn_policy == "running_stats"
    if not direct:
        tensors = {n: p for n, p in model.named_parameters()}
        tensors.update(params or {})
        bufs = {n: b for n, b in model.named_buffers()}
        bufs.update(buffers or {})
        if bn_policy == "fixed_batch_stats":
            bufs = {n: b.clone() for n, b in bufs.items()}
        tensors.update(bufs)
    total_ce, total_obj, correct, n = 0.0, 0.0, 0, int(images.shape[0])
    try:
        with torch.random.fork_rng(devices=_devices(model)):
            model.train(bn_policy == "fixed_batch_stats")
            for i in range(0, n, batch_size):
                x = pipeline(images[i:i + batch_size])
                logits = model(x) if direct else functional_call(model, tensors, (x,))
                y = labels[i:i + batch_size]
                total_ce += float(F.cross_entropy(logits, y, reduction="sum"))
                if objective is not None:
                    total_obj += float(objective(logits, y)) * int(y.numel())
                correct += int((logits.argmax(1) == y).sum())
    finally:
        controller.set_state(previous)
        model.train(was_training)
    out = {"split": split, "n": n, "ce": total_ce / n, "acc": correct / n,
           "path": state.label, "state": state.to_dict(),
           "per_site_sigma": per_site, "bn_policy": bn_policy,
           "batch_size": batch_size}
    if objective is not None:
        out["obj"] = total_obj / n
    return out
