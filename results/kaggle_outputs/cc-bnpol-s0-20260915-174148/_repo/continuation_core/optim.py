"""SGD and AdamW, and a learning-rate schedule indexed by global update.

The schedule never restarts at an intervention transition.  ``lr_at`` is the
benchmark's formula unchanged::

    update < warmup:  lr * (update + 1) / warmup
    otherwise:        min_lr + (lr - min_lr) * 0.5 * (1 + cos(pi * p)),
                      p = clip((update - warmup) / (total - warmup), 0, 1)

AdamW is available but has **no borrowed defaults**: ``lr`` and
``weight_decay`` must be set explicitly, because nothing about the SGD values
(0.005, 5e-4) transfers to AdamW.
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
    if cfg.name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, betas=tuple(cfg.betas),
                                 eps=cfg.eps, weight_decay=cfg.weight_decay)
    raise KeyError("unknown optimizer %r (available: sgd, adamw)" % cfg.name)


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
