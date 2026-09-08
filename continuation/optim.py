"""Optimizer and learning-rate schedule.

The learning-rate schedule is **indexed by the global optimization step** and
knows nothing about transformation stages, so it is never implicitly restarted
at a stage boundary in a future continuation run.

Momentum at stage boundaries is an explicit, separate decision: carrying the
SGD momentum buffers across a transformation change and resetting them are
different procedures.  :func:`reset_momentum` implements the reset; the default
is to carry.  Experiment 0 has no stage transitions, so neither path is taken.
"""
from __future__ import annotations

import math

import torch

from .config import OptimConfig


def build_optimizer(model: torch.nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    if cfg.optimizer != "sgd":
        raise KeyError("only 'sgd' is implemented, got %r" % (cfg.optimizer,))
    return torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=cfg.nesterov,
    )


def lr_at(step: int, cfg: OptimConfig) -> float:
    """Learning rate at global step ``step`` (0-based).

    Linear warmup over ``warmup_steps`` updates, then the configured decay over
    the remaining budget.  Held fixed across every condition of a comparison.
    """
    total = max(int(cfg.total_steps), 1)
    warm = max(int(cfg.warmup_steps), 0)
    if warm > 0 and step < warm:
        return cfg.lr * float(step + 1) / float(warm)

    if cfg.lr_schedule == "constant":
        return cfg.lr
    if cfg.lr_schedule == "multistep":
        lr = cfg.lr
        for m in cfg.milestones:
            if step >= int(m):
                lr *= cfg.gamma
        return lr
    if cfg.lr_schedule == "cosine":
        denom = max(total - warm, 1)
        p = min(max((step - warm) / float(denom), 0.0), 1.0)
        return cfg.min_lr + (cfg.lr - cfg.min_lr) * 0.5 * (1.0 + math.cos(math.pi * p))
    raise KeyError("unknown lr_schedule %r" % (cfg.lr_schedule,))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def reset_momentum(optimizer: torch.optim.Optimizer) -> None:
    """Drop SGD momentum buffers (the explicit 'reset' policy at a stage boundary)."""
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p)
            if state is not None:
                state.pop("momentum_buffer", None)


def describe(cfg: OptimConfig) -> dict:
    return {
        "optimizer": cfg.optimizer, "lr": cfg.lr, "momentum": cfg.momentum,
        "weight_decay": cfg.weight_decay, "nesterov": cfg.nesterov,
        "batch_size": cfg.batch_size, "total_steps": cfg.total_steps,
        "lr_schedule": cfg.lr_schedule, "warmup_steps": cfg.warmup_steps,
        "min_lr": cfg.min_lr, "milestones": list(cfg.milestones), "gamma": cfg.gamma,
        "grad_clip": cfg.grad_clip,
        "lr_indexing": "global optimization step (never restarted per stage)",
        "weight_decay_note": ("weight decay is fixed across all conditions and is NOT "
                              "part of the reported cross-entropy objective"),
    }
