"""The reference preset and explicit transfer templates.

``reference(method_id, seed)`` is the unchanged CIFAR-10 / ResNet-20 recipe the
four frozen methods were run with.  Do not edit it to try something new: build
a new config instead, which is marked ``unvalidated``.

``transfer(...)`` builds a config for another dataset / architecture /
optimizer.  It has **no defaults for the decisions that matter**: resolution
schedule and its reference size, Gaussian schedule and units, schedule
duration, insertion mapping, and the optimizer's learning rate and weight decay
must all be passed.  Nothing is scaled or chosen for you.
"""
from __future__ import annotations

from .config import (AssetsConfig, BudgetConfig, CheckpointConfig, DataConfig,
                     EvaluationConfig, ExperimentConfig, ModelConfig,
                     OptimizerConfig, RunConfig)
from .methods import GaussianSpec, MethodSpec, ResolutionSpec, get_method
from .optim import optimizer_tag
from .schedules import EpochSchedule

#: normalisation statistics of the 50,000 CIFAR-10 training images, as checked
#: by every reference run (unified_driver.EXPECTED_STATS)
CIFAR10_MEAN = [0.4913996756076813, 0.4821584224700928, 0.44653090834617615]
CIFAR10_STD = [0.24703222513198853, 0.24348512291908264, 0.26158782839775085]


def reference_optimizer() -> OptimizerConfig:
    """The executed SGD setting.  A fresh object each call: callers mutate configs."""
    return OptimizerConfig(name="sgd", lr=0.005, weight_decay=5e-4, momentum=0.9,
                           nesterov=False, schedule="warmup_cosine",
                           warmup_updates=60, min_lr=0.0)


def reference(method_id: str, seed: int = 0, *, data_root: str = "data",
              assets_dir: str = "assets/cifar10_resnet20bn", out_dir: str = "runs",
              device: str = "auto",
              optimizer: OptimizerConfig | None = None) -> ExperimentConfig:
    """The reference recipe, optionally with another optimizer.

    ``optimizer=None`` is the unchanged recipe and keeps
    ``validation_status="reference"``.  Any other optimizer setting is a
    variant: the status becomes ``"optimizer-variant"``, because ``"reference"``
    is reserved for the configuration the recorded results were produced with.
    Everything else -- data, model, budget, evaluation, assets -- is untouched,
    so the optimizer is the only thing that differs from the recorded runs.
    """
    get_method(method_id)
    opt = optimizer or reference_optimizer()
    is_reference = opt == reference_optimizer()
    return ExperimentConfig(
        data=DataConfig(name="cifar10", root=data_root, normalization="fit_on_train",
                        expected_mean=CIFAR10_MEAN, expected_std=CIFAR10_STD),
        model=ModelConfig(arch="resnet20_bn_cifar"),
        optimizer=opt,
        method={"id": method_id},
        budget=BudgetConfig(epochs=30, effective_batch=128, microbatch=32),
        evaluation=EvaluationConfig(every_epochs=1, at_epoch_zero=True, batch_size=500,
                                    paths=("current", "target"),
                                    splits=("train_probe", "test"),
                                    bn_policy="running_stats"),
        checkpoint=CheckpointConfig(every_epoch=True, every_updates=None,
                                    transition_offsets=(), keep_rolling=True),
        run=RunConfig(seed=seed, device=device, out_dir=out_dir,
                      name="%s__%s__seed%d" % (method_id, optimizer_tag(opt), seed)),
        assets=AssetsConfig(dir=assets_dir, verify=True),
        validation_status="reference" if is_reference else "optimizer-variant",
        notes=["unified_selected batch recipe; evaluation after every epoch"]
              + ([] if is_reference else
                 ["optimizer replaced: %s; every other section is the reference recipe"
                  % optimizer_tag(opt)]))


def transfer(method_id: str, *, dataset: str, data_root: str, arch: str,
             optimizer: OptimizerConfig, epochs: int,
             resolution_schedule: EpochSchedule | None,
             reference_resolution: int | None,
             gaussian_schedule: EpochSchedule | None,
             gaussian_units: str | None,
             insertion_mapping_note: str,
             assets_dir: str, seed: int = 0, effective_batch: int = 128,
             microbatch: int = 32, out_dir: str = "runs") -> ExperimentConfig:
    """A config for a new setting.  Every transferred decision is explicit."""
    base = get_method(method_id)
    if base.resolution is not None and (resolution_schedule is None
                                        or reference_resolution is None):
        raise ValueError("%s has a resolution intervention: pass resolution_schedule "
                         "and reference_resolution explicitly" % method_id)
    if base.gaussian is not None and (gaussian_schedule is None or not gaussian_units):
        raise ValueError("%s has a Gaussian intervention: pass gaussian_schedule and "
                         "gaussian_units explicitly" % method_id)
    method = MethodSpec(
        id=method_id + "__transfer", description=base.description,
        resolution=(ResolutionSpec(point=base.resolution.point,
                                   schedule=resolution_schedule,
                                   reference_resolution=int(reference_resolution))
                    if base.resolution else None),
        gaussian=(GaussianSpec(placement=base.gaussian.placement,
                               schedule=gaussian_schedule,
                               sigma_scale=base.gaussian.sigma_scale,
                               sigma_max=max(1.0, max(gaussian_schedule.values)),
                               truncate=base.gaussian.truncate, units=gaussian_units)
                  if base.gaussian else None),
        source={"transferred_from": method_id, **base.source},
        validated_on=())
    return ExperimentConfig(
        data=DataConfig(name=dataset, root=data_root),
        model=ModelConfig(arch=arch), optimizer=optimizer,
        method=method.to_dict(),
        budget=BudgetConfig(epochs=epochs, effective_batch=effective_batch,
                            microbatch=microbatch),
        run=RunConfig(seed=seed, out_dir=out_dir,
                      name="%s__%s__%s__seed%d" % (method_id, dataset, arch, seed)),
        assets=AssetsConfig(dir=assets_dir),
        validation_status="unvalidated",
        notes=["transfer preset: not trained, not validated",
               "insertion mapping: " + insertion_mapping_note])
