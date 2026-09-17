"""Run configuration: independent sections, serialisable to JSON.

Each section can be changed without touching the others::

    data        dataset, root, preprocessing
    model       architecture (and therefore its intervention-site map)
    optimizer   optimizer and learning-rate schedule
    method      intervention preset or an explicit MethodSpec
    budget      epochs, effective batch, microbatch
    evaluation  cadence, what is evaluated, BN policy
    checkpoint  cadence, including windows around intervention transitions
    run         seed, device, output directory, determinism flags
    assets      pinned initial state and data order

Nothing is inferred between sections.  In particular an optimizer change never
changes the learning rate, a dataset change never rescales sigma or the
resolution schedule, and a model change never relocates an insertion site.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

from .methods import MethodSpec, get_method


@dataclass
class DataConfig:
    name: str = "cifar10"
    root: str = "data"
    #: "fit_on_train": per-channel mean/std of the unfiltered training images
    #: (float64, population std) -- the reference setting.
    normalization: str = "fit_on_train"
    #: optional pinned statistics; when given they are checked against the fit
    expected_mean: list | None = None
    expected_std: list | None = None
    stats_atol: float = 1e-7
    #: "none" reproduces every recorded run; "crop_flip" is the standard CIFAR
    #: recipe (pad 4, random 32x32 crop, horizontal flip).  Training only --
    #: evaluation never augments (see :mod:`continuation_core.augment`).
    augmentation: str = "none"
    augment_pad: int = 4
    augment_flip_p: float = 0.5


@dataclass
class ModelConfig:
    arch: str = "resnet20_bn_cifar"
    options: dict = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    name: str = "sgd"
    #: required; never defaulted from another optimizer
    lr: float | None = None
    weight_decay: float | None = None
    momentum: float = 0.9               # sgd only
    nesterov: bool = False              # sgd only
    betas: tuple = (0.9, 0.999)         # adamw only
    eps: float = 1e-8                   # adamw only
    schedule: str = "warmup_cosine"     # warmup_cosine | constant
    warmup_updates: int = 0
    min_lr: float = 0.0


@dataclass
class LossConfig:
    """The training objective.  ``cross_entropy`` reproduces every recorded run.

    Fields not used by the selected ``name`` are still recorded, so a run's
    config always states the full setting (see :mod:`continuation_core.losses`).
    """
    name: str = "cross_entropy"
    label_smoothing: float = 0.1        # label_smoothing only
    gamma: float = 2.0                  # focal only
    normalise_by_classes: bool = True   # square only; divides by C


@dataclass
class BudgetConfig:
    epochs: int = 30
    effective_batch: int = 128
    microbatch: int = 32


@dataclass
class EvaluationConfig:
    every_epochs: int = 1
    at_epoch_zero: bool = True
    batch_size: int = 500
    #: evaluate the scheduled (current) state and the target state
    paths: tuple = ("current", "target")
    splits: tuple = ("train_probe", "test")
    bn_policy: str = "running_stats"


@dataclass
class CheckpointConfig:
    every_epoch: bool = True
    every_updates: int | None = None
    #: extra checkpoints at ``transition_update + offset`` for every update
    #: where the intervention state changes; offset 0 is the epoch boundary
    #: (weights produced by the old state, next update uses the new one)
    transition_offsets: tuple = ()
    keep_rolling: bool = True


@dataclass
class RunConfig:
    seed: int = 0
    device: str = "auto"
    out_dir: str = "runs"
    name: str | None = None
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = False


@dataclass
class AssetsConfig:
    #: directory holding assets_manifest.json, init_seed<k>.pt, shared_indices.npz
    dir: str | None = "assets/cifar10_resnet20bn"
    verify: bool = True


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    method: dict = field(default_factory=lambda: {"id": "plain"})
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    run: RunConfig = field(default_factory=RunConfig)
    assets: AssetsConfig = field(default_factory=AssetsConfig)
    #: "reference" only for the unchanged CIFAR-10 / ResNet-20 preset
    validation_status: str = "unvalidated"
    notes: list = field(default_factory=list)

    def method_spec(self) -> MethodSpec:
        if set(self.method) == {"id"}:
            return get_method(self.method["id"])
        return MethodSpec.from_dict(self.method)

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return _build(cls, d)

    @classmethod
    def load(cls, path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _build(tp, d):
    if not is_dataclass(tp):
        return d
    known = {f.name: f for f in fields(tp)}
    unknown = sorted(set(d) - set(known))
    if unknown:
        raise KeyError("%s: unknown keys %s" % (tp.__name__, unknown))
    kwargs = {}
    for name, f in known.items():
        if name not in d:
            continue
        sub = f.default_factory() if callable(getattr(f, "default_factory", None)) else None
        v = d[name]
        if is_dataclass(sub):
            kwargs[name] = _build(type(sub), v)
        elif isinstance(v, list) and name in ("paths", "splits", "transition_offsets", "betas"):
            kwargs[name] = tuple(v)
        else:
            kwargs[name] = v
    return tp(**kwargs)
