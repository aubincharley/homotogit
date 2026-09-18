"""Measurement-only signals for resolution-continuation controllers.

Everything here *reads* the model and returns numbers; nothing trains.  The
purity requirements of ``docs/adaptive_resolution_plan.md`` section 4 are met
by construction:

* evaluation mode is used unless a train-mode BatchNorm view is explicitly
  requested, and the previous mode is restored afterwards;
* BatchNorm running statistics are never left modified -- the recalibration
  helper works on buffers that :class:`BNState` restores bitwise;
* gradients go through :func:`torch.autograd.grad` and never touch ``.grad``;
* no RNG is consumed: fixed ordered subsets, no augmentation, no dropout.

The signals are the candidates for an adaptive resolution trigger:

``full_gradient``   the corrector residual (plan section 2), at the current or
                    at another resolution, so that gradient *alignment* between
                    the surrogate and the target objective can be measured;
``recalibrate_bn``  Q-04 of OPEN_QUESTIONS: how much of the target-path collapse
                    is BatchNorm statistics rather than weights, and a
                    look-ahead signal for the *next* stage at fixed weights.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def bn_modules(model: nn.Module) -> list:
    return [m for m in model.modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm)]


class BNState:
    """Snapshot of every BatchNorm buffer and momentum.

    ``restore`` copies the saved buffers back, so any measurement that runs the
    network in train mode or recalibrates the statistics leaves the training
    trajectory untouched.  ``differs`` is the corresponding self-check.
    """

    def __init__(self, model: nn.Module):
        self.mods = bn_modules(model)
        with torch.no_grad():
            self.saved = [(m.running_mean.detach().clone(),
                           m.running_var.detach().clone(),
                           m.num_batches_tracked.detach().clone(),
                           m.momentum) for m in self.mods]

    def restore(self) -> None:
        with torch.no_grad():
            for m, (rm, rv, nb, mom) in zip(self.mods, self.saved):
                m.running_mean.copy_(rm)
                m.running_var.copy_(rv)
                m.num_batches_tracked.copy_(nb)
                m.momentum = mom

    def differs(self) -> bool:
        with torch.no_grad():
            for m, (rm, rv, nb, mom) in zip(self.mods, self.saved):
                if (not torch.equal(m.running_mean, rm)
                        or not torch.equal(m.running_var, rv)
                        or not torch.equal(m.num_batches_tracked, nb)
                        or m.momentum != mom):
                    return True
        return False


@torch.no_grad()
def recalibrate_bn(model: nn.Module, forward, images: torch.Tensor,
                   batch: int = 500) -> None:
    """Re-estimate BatchNorm running statistics at *fixed weights*.

    Statistics are reset and rebuilt as the exact cumulative average over
    ``images`` (``momentum=None``), in train mode, without gradients.  The
    caller must hold a :class:`BNState` and restore it afterwards; nothing here
    changes a parameter.
    """
    prev = model.training
    for m in bn_modules(model):
        m.reset_running_stats()
        m.momentum = None
    model.train()
    for i in range(0, int(images.shape[0]), batch):
        forward(images[i:i + batch])
    model.train(prev)


@torch.no_grad()
def ce_acc(model: nn.Module, forward, images: torch.Tensor, labels: torch.Tensor,
           batch: int = 500) -> tuple:
    """Mean cross-entropy and accuracy in evaluation mode (running statistics)."""
    prev = model.training
    model.eval()
    n, tot, correct = int(images.shape[0]), 0.0, 0
    for i in range(0, n, batch):
        logits = forward(images[i:i + batch])
        yb = labels[i:i + batch]
        tot += float(F.cross_entropy(logits, yb, reduction="sum"))
        correct += int((logits.argmax(1) == yb).sum())
    model.train(prev)
    return tot / n, correct / n


def trainable_parameters(model: nn.Module) -> list:
    return [p for p in model.parameters() if p.requires_grad]


def full_gradient(model: nn.Module, forward, images: torch.Tensor,
                  labels: torch.Tensor, batch: int = 250,
                  train_mode_bn: bool = False) -> torch.Tensor:
    """Flat gradient of the mean cross-entropy over ``images``.

    Batches are weighted by ``n_b / N`` so the sum is exactly the full-set mean
    gradient -- no sampling noise, only trajectory noise.  Weight decay is not
    included, matching the reported objective.  By default the forward pass
    uses evaluation-mode BatchNorm (a deterministic function of the weights);
    with ``train_mode_bn`` it uses batch statistics -- the training objective's
    own view -- on buffers that are restored bitwise afterwards.  ``.grad`` is
    never written.
    """
    params = trainable_parameters(model)
    prev = model.training
    state = BNState(model) if train_mode_bn else None
    model.train(bool(train_mode_bn))
    n = int(images.shape[0])
    acc = None
    for i in range(0, n, batch):
        xb, yb = images[i:i + batch], labels[i:i + batch]
        loss = F.cross_entropy(forward(xb), yb) * (float(xb.shape[0]) / n)
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        flat = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                          for g, p in zip(grads, params)])
        acc = flat if acc is None else acc.add_(flat)
    if state is not None:
        state.restore()
    model.train(prev)
    return acc.detach()


def param_groups(model: nn.Module) -> dict:
    """Slices of the flat parameter vector, by role: conv / bn / fc.

    Follows ``model.parameters()`` order restricted to trainable parameters,
    which is the order :func:`full_gradient` flattens.
    """
    groups = {"conv": [], "bn": [], "fc": []}
    offset = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        if p.dim() == 4:
            g = "conv"
        elif name.startswith("fc.") or ".fc." in name:
            g = "fc"
        else:
            g = "bn"
        groups[g].append((offset, offset + n))
        offset += n
    return groups


def group_norms(flat: torch.Tensor, groups: dict) -> dict:
    out = {"total": float(flat.norm())}
    for g, slices in groups.items():
        s = sum(float(flat[a:b].pow(2).sum()) for a, b in slices)
        out[g] = math.sqrt(s)
    return out


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    den = float(a.norm()) * float(b.norm())
    if den == 0.0:
        return float("nan")
    return float(torch.dot(a, b)) / den


@torch.no_grad()
def weight_norms(model: nn.Module) -> dict:
    """L2 norm of the parameters, total and per role (BN makes ||g|| scale-dependent)."""
    flat = torch.cat([p.detach().reshape(-1) for p in trainable_parameters(model)])
    return group_norms(flat, param_groups(model))


@torch.no_grad()
def parameter_checksum(model: nn.Module) -> list:
    """Clones of every trainable parameter, for an exact before/after comparison."""
    return [p.detach().clone() for p in trainable_parameters(model)]


@torch.no_grad()
def parameters_equal(model: nn.Module, ref: list) -> bool:
    return all(torch.equal(p.detach(), r) for p, r in zip(trainable_parameters(model), ref))
