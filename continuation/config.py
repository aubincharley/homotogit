"""Typed, strict configuration objects loaded from YAML.

Unknown keys are an error rather than a silent no-op: a misspelled key in a
research config is a reproducibility bug, not a convenience.
"""
from __future__ import annotations

import copy
import dataclasses
import typing
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dataset: str = "cifar10"          # cifar10 | cifar100
    root: str = "data"
    num_val: int = 5000               # held out from the official train set
    split_seed: int = 12345           # controls the train/val split only
    stratified: bool = True
    download: bool = True


@dataclass
class ModelConfig:
    arch: str = "resnet20_gn"
    width: int = 16                   # channels in stage 1
    channels_per_group: int = 8       # GroupNorm: num_groups = channels // this
    shortcut: str = "A"               # "A" = parameter-free pad, "B" = 1x1 conv
    zero_init_residual: bool = False


@dataclass
class OptimConfig:
    optimizer: str = "sgd"
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4        # fixed across all conditions
    nesterov: bool = False
    batch_size: int = 128
    total_steps: int = 10560          # the optimization budget (gradient updates)
    lr_schedule: str = "cosine"       # cosine | multistep | constant
    warmup_steps: int = 400
    min_lr: float = 0.0
    milestones: list[int] = field(default_factory=list)   # for multistep
    gamma: float = 0.1
    grad_clip: float | None = None


@dataclass
class TransformConfig:
    """Transformation family and its *configuration* (not its parameter value).

    ``params`` is family-specific and is part of the cache/identity key.
    """
    family: str = "gaussian"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduleConfig:
    """Controller that maps global step -> native transformation parameter."""
    kind: str = "constant"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfig:
    eval_every: int = 500             # shared checkpoints, in gradient updates
    eval_at_step_zero: bool = True
    train_probe_size: int = 5000      # fixed documented subset of the train split
    probe_seed: int = 777             # independent of split/init/batch-order seeds
    eval_batch_size: int = 500
    transform_stats_size: int = 512   # fixed image subset for MSE / TV ratio
    save_final_checkpoint: bool = True
    checkpoint_every: int | None = None
    # Denser evaluations just after a transformation change, for adaptation
    # curves.  Descriptive only: paired claims use the common eval_every grid.
    dense_every: int = 100
    dense_window: int = 1000


@dataclass
class RunConfig:
    name: str = "run"
    seed: int = 0                     # paired seed: init + batch order
    device: str = "auto"              # auto | cuda | cpu
    out_dir: str = "results"
    deterministic: bool = True
    log_every: int = 50
    num_workers: int = 0


@dataclass
class ExperimentConfig:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    # Experiment-0 sweep axes. Empty means "single run at the schedule's value".
    sweep: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def _resolved_fields(cls) -> dict[str, Any]:
    """Field-name -> resolved type. ``from __future__ import annotations`` makes
    ``dataclasses.fields(...).type`` a string, so resolve it explicitly."""
    hints = typing.get_type_hints(cls)
    return {f.name: hints.get(f.name, Any) for f in dataclasses.fields(cls)}


def _build(cls, data: dict[str, Any], path: str):
    if not isinstance(data, dict):
        raise TypeError(f"config section '{path}' must be a mapping, got {type(data).__name__}")
    fields = _resolved_fields(cls)
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise KeyError(
            f"unknown config key(s) {unknown} in section '{path}'; "
            f"known keys are {sorted(fields)}"
        )
    kwargs = {}
    for name, typ in fields.items():
        if name not in data:
            continue
        value = data[name]
        if isinstance(typ, type) and is_dataclass(typ):
            kwargs[name] = _build(typ, value, f"{path}.{name}")
        else:
            kwargs[name] = value
    return cls(**kwargs)


def from_dict(data: dict[str, Any]) -> ExperimentConfig:
    data = copy.deepcopy(data or {})
    return _build(ExperimentConfig, data, "<root>")


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if overrides:
        raw = deep_update(raw, overrides)
    return from_dict(raw)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def to_dict(cfg: Any) -> Any:
    if is_dataclass(cfg):
        return {f.name: to_dict(getattr(cfg, f.name)) for f in dataclasses.fields(cfg)}
    if isinstance(cfg, dict):
        return {k: to_dict(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [to_dict(v) for v in cfg]
    return cfg


def parse_override(text: str) -> dict[str, Any]:
    """Parse ``a.b=value`` into a nested dict, with YAML scalar parsing."""
    if "=" not in text:
        raise ValueError(f"override must look like 'section.key=value', got {text!r}")
    key, _, value = text.partition("=")
    parsed = yaml.safe_load(value)
    node: dict[str, Any] = {}
    cur = node
    parts = key.strip().split(".")
    for p in parts[:-1]:
        cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = parsed
    return node
