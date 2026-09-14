"""SGD, Adam, AdamW and RAdam, and a learning-rate schedule indexed by global update.

The schedule never restarts at an intervention transition.  ``lr_at`` is the
benchmark's formula unchanged::

    update < warmup:  lr * (update + 1) / warmup
    otherwise:        min_lr + (lr - min_lr) * 0.5 * (1 + cos(pi * p)),
                      p = clip((update - warmup) / (total - warmup), 0, 1)

No borrowed defaults
--------------------
Every optimizer needs an explicit ``lr`` and ``weight_decay``.  Nothing about
the SGD reference values (0.005, 5e-4) transfers to an adaptive optimizer, so
the config refuses to guess one from the other.

Weight decay, which is the axis the Adam/AdamW pair measures
------------------------------------------------------------
The reference recipe applies weight decay to **every** parameter, BatchNorm
weights and biases included, so the coupled/decoupled distinction actually
bites here.

======  ==================================================
sgd     coupled L2, folded into the gradient
adam    coupled L2 (``decoupled_weight_decay=False``)
adamw   decoupled
radam   coupled L2 (``decoupled_weight_decay=False``)
======  ==================================================

``torch.optim.Adam`` and ``torch.optim.RAdam`` both expose a
``decoupled_weight_decay`` flag whose default could move between torch
releases, and the torch on Kaggle is not the torch used here.  It is therefore
passed **explicitly** rather than left to the default: Adam and AdamW then
differ by that one flag and nothing else, which is what makes the pair
informative.
"""
from __future__ import annotations

import math

import torch

from .config import OptimizerConfig


def build_optimizer(params, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    if cfg.lr is None or cfg.weight_decay is None:
        raise ValueError("optimizer %r needs explicit lr and weight_decay" % cfg.name)
    if cfg.name == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                               weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)
    if cfg.name == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, betas=tuple(cfg.betas),
                                eps=cfg.eps, weight_decay=cfg.weight_decay,
                                decoupled_weight_decay=False)
    if cfg.name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, betas=tuple(cfg.betas),
                                 eps=cfg.eps, weight_decay=cfg.weight_decay)
    if cfg.name == "radam":
        return torch.optim.RAdam(params, lr=cfg.lr, betas=tuple(cfg.betas),
                                 eps=cfg.eps, weight_decay=cfg.weight_decay,
                                 decoupled_weight_decay=False)
    raise KeyError("unknown optimizer %r (available: %s)"
                   % (cfg.name, ", ".join(OPTIMIZERS)))


#: registered names, in the order they are reported
OPTIMIZERS = ("sgd", "adam", "adamw", "radam")


def optimizer_tag(cfg: OptimizerConfig) -> str:
    """Short filesystem-safe identity of an optimizer setting, e.g. ``adam_lr0.001``.

    It is the **only** place a run directory name is decided.  It covers the
    optimizer name and the learning rate, which are the two axes this study
    varies: the learning rate is in it because the sweep runs the same
    optimizer, method and seed at several learning rates, and those would
    otherwise collide in one directory.  Varying anything else (weight decay,
    betas) needs this function extended first.
    """
    if cfg.lr is None:
        raise ValueError("optimizer_tag needs an explicit lr")
    return "%s_lr%g" % (cfg.name, float(cfg.lr))


def lr_at(update: int, cfg: OptimizerConfig, total_updates: int) -> float:
    total = max(int(total_updates), 1)
    warm = max(int(cfg.warmup_updates), 0)
    if warm > 0 and update < warm:
        return cfg.lr * float(update + 1) / float(warm)
    if cfg.schedule == "constant":
        return cfg.lr
    if cfg.schedule == "warmup_cosine":
        denom = max(total - warm, 1)
        p = min(max((update - warm) / float(denom), 0.0), 1.0)
        return cfg.min_lr + (cfg.lr - cfg.min_lr) * 0.5 * (1.0 + math.cos(math.pi * p))
    raise KeyError("unknown schedule %r" % cfg.schedule)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
