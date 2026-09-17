"""The frozen 63-run follow-up matrix: three 160-epoch regimes x seven arms x seeds 0-2.

Regimes (CIFAR-10 full 50,000 / 10,000, ResNet-20 BN/ReLU, cross-entropy, float32, no autocast):

* ``sgd_standard_aug_160``  SGD lr 0.1, momentum 0.9, no Nesterov, coupled weight decay 1e-4 on
  every learned parameter; lr x 0.1 at the starts of epochs 80 and 120 (updates 31,280 and
  46,920), no warm-up; physical batch 128 (one microbatch, BN sees all 128); pad-4/crop/flip.
* ``adamw_long_noaug_160``  AdamW lr 0.02, betas (0.9, 0.999), eps 1e-8, decoupled weight decay
  5e-4 on every parameter; 60-update linear warm-up then the same cosine formula to 0 over the
  full 62,560 updates; logical batch 128 as four microbatches of 32 (historical accumulation and
  partial-batch weighting); no augmentation.
* ``adamw_long_aug_160``    the same AdamW recipe with pad-4/crop/flip.

Historical conventions kept: pinned initial states ``init_seed<k>.pt`` (parameters and BN buffers),
input normalisation fitted on the unfiltered training images (``presets.CIFAR10_MEAN/STD``), the
first 30 epochs of ``perm_seed<k>`` unchanged (epochs 30-159 from ``perm_extension``), last partial
batch kept (391 updates per epoch; U = 62,560).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from continuation_core.config import (AssetsConfig, BudgetConfig, CheckpointConfig, DataConfig,
                                      EvaluationConfig, ExperimentConfig, ModelConfig,
                                      OptimizerConfig, RunConfig)
from continuation_core.methods import GaussianSpec, MethodSpec, ResolutionSpec, get_method
from continuation_core.presets import CIFAR10_MEAN, CIFAR10_STD
from continuation_core.schedules import EpochSchedule
from continuation_core.seeding import derive_seed

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies" / "overnight_long_aug_bn"
PREV_STUDY_WORKTREE = ROOT.parent / "cbs-sdpoint" / "studies" / "comparison_cbs_sdpoint"

ARMS = ("plain", "resolution_max_b1", "gaussian_postrelu", "resolution_max_b1_gaussian_conv",
        "cbs_published_schedule", "cbs_budget_matched", "sdpoint")
SHORT = {"plain": "Plain", "resolution_max_b1": "R", "gaussian_postrelu": "G",
         "resolution_max_b1_gaussian_conv": "RG", "cbs_published_schedule": "CBS-pub",
         "cbs_budget_matched": "CBS-bm", "sdpoint": "SDPoint"}
REGIMES = ("sgd_standard_aug_160", "adamw_long_noaug_160", "adamw_long_aug_160")
PRIORITY = {r: i + 1 for i, r in enumerate(REGIMES)}
SEEDS = (0, 1, 2)
EPOCHS = 160
UPE = 391
U = EPOCHS * UPE                                   # 62,560
BASE_EPOCHS = 30                                   # epochs covered by the pinned perm_seed<k>
CHECKPOINT_EPOCHS = (30, 60, 120, 160)
P1_CURVE_EPOCHS = (40, 80, 120, 160)               # 160 reuses the final endpoint's P1
AUGMENTATION = {"sgd_standard_aug_160": "pad4_crop32_hflip", "adamw_long_noaug_160": "none",
                "adamw_long_aug_160": "pad4_crop32_hflip"}
PERM_EXTENSION_STREAM = "perm_extension::epoch::%d"

# ---- schedules ---------------------------------------------------------------------------

#: r = 16 for u < 0.2U, 24 for 0.2U <= u < 0.4U, then 32 (exact bypass).  0.2U = 12,512 = 32 x 391,
#: so the update-fraction rule coincides with epoch boundaries 32 and 64.
R_LONG = EpochSchedule((0, 32, 64), (16, 24, 32))
#: g = 1/.85/.70/.60/.50/.40/.30 on [k/10, (k+1)/10) of u/U, k = 0..6, then 0 (exact bypass) for the
#: final 30%.  0.1U = 6,256 = 16 x 391: 16-epoch plateaus, filtering ends at epoch 112.
G_LONG = EpochSchedule((0, 16, 32, 48, 64, 80, 96, 112), (1.00, 0.85, 0.70, 0.60, 0.50, 0.40, 0.30, 0.0))


def r_by_update(u: int) -> int:
    """Specification in update fractions, integer arithmetic (used to check ``R_LONG``)."""
    return 16 if 5 * u < U else 24 if 5 * u < 2 * U else 32


def g_by_update(u: int) -> float:
    k = (10 * u) // U
    return (1.00, 0.85, 0.70, 0.60, 0.50, 0.40, 0.30)[k] if k < 7 else 0.0


def method_spec(arm: str) -> MethodSpec:
    base = get_method(arm)
    long_note = " [160-epoch follow-up schedule]"
    if arm == "resolution_max_b1":
        return MethodSpec(id=arm, description=base.description.replace("16 / 24 / 32", "16 / 24 / 32 at epochs 0 / 32 / 64") + long_note,
                          resolution=ResolutionSpec(point="block1", schedule=R_LONG), source=base.source)
    if arm == "gaussian_postrelu":
        return MethodSpec(id=arm, description=base.description + " 16-epoch plateaus, bypass from epoch 112." + long_note,
                          gaussian=GaussianSpec(placement="post_relu", schedule=G_LONG, sigma_scale="none"),
                          source=base.source)
    if arm == "resolution_max_b1_gaussian_conv":
        return MethodSpec(id=arm, description=base.description + " r at epochs 0/32/64, g plateaus of 16 epochs." + long_note,
                          resolution=ResolutionSpec(point="block1", schedule=R_LONG),
                          gaussian=GaussianSpec(placement="conv_out", schedule=G_LONG,
                                                sigma_scale="resolution_ratio_all_sites"),
                          source=base.source)
    return base          # plain, both CBS arms and SDPoint are horizon-agnostic specs


# ---- optimizer and config -----------------------------------------------------------------

def optimizer(regime: str) -> OptimizerConfig:
    if regime == "sgd_standard_aug_160":
        return OptimizerConfig(name="sgd", lr=0.1, weight_decay=1e-4, momentum=0.9, nesterov=False,
                               schedule="multistep", warmup_updates=0, min_lr=0.0,
                               milestones=(80 * UPE, 120 * UPE), gamma=0.1)
    if regime in ("adamw_long_noaug_160", "adamw_long_aug_160"):
        return OptimizerConfig(name="adamw", lr=0.02, weight_decay=5e-4, betas=(0.9, 0.999), eps=1e-8,
                               schedule="warmup_cosine", warmup_updates=60, min_lr=0.0)
    raise KeyError(regime)


def microbatch(regime: str) -> int:
    return 128 if regime == "sgd_standard_aug_160" else 32


def run_name(regime: str, arm: str, seed: int) -> str:
    return "%s__%s__seed%d" % (regime, arm, seed)


def config(regime: str, arm: str, seed: int, data_root: str = "data",
           assets_dir: str = "assets/cifar10_resnet20bn", out_dir: str = "runs") -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(name="cifar10", root=data_root, normalization="fit_on_train",
                        expected_mean=CIFAR10_MEAN, expected_std=CIFAR10_STD,
                        augmentation=AUGMENTATION[regime]),
        model=ModelConfig(arch="resnet20_bn_cifar"),
        optimizer=optimizer(regime),
        method=method_spec(arm).to_dict(),
        budget=BudgetConfig(epochs=EPOCHS, effective_batch=128, microbatch=microbatch(regime)),
        evaluation=EvaluationConfig(every_epochs=1, at_epoch_zero=True, batch_size=500,
                                    paths=("current", "target"), splits=("train_probe", "test"),
                                    bn_policy="running_stats"),
        checkpoint=CheckpointConfig(every_epoch=False, every_updates=None, transition_offsets=(),
                                    keep_rolling=True, at_epochs=CHECKPOINT_EPOCHS),
        run=RunConfig(seed=seed, device="auto", out_dir=out_dir, name=run_name(regime, arm, seed)),
        assets=AssetsConfig(dir=assets_dir, verify=True),
        validation_status="overnight-follow-up",
        notes=["overnight_long_aug_bn: regime %s (priority %d)" % (regime, PRIORITY[regime]),
               "data order: perm_seed%d epochs 0-29 pinned, 30-159 from %s" % (seed, PERM_EXTENSION_STREAM)])


#: fields that locate files or pick a device; excluded from the configuration digest
ENVIRONMENT_FIELDS = (("data", "root"), ("assets", "dir"), ("run", "out_dir"), ("run", "device"))


def config_sha(cfg: ExperimentConfig) -> str:
    """sha256 of the configuration without its environment fields (paths, device)."""
    d = cfg.to_dict()
    for sec, key in ENVIRONMENT_FIELDS:
        d[sec].pop(key, None)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def cells() -> list:
    return [{"regime": r, "arm": a, "seed": s, "priority": PRIORITY[r], "run": run_name(r, a, s)}
            for r in REGIMES for s in SEEDS for a in ARMS]


# ---- data-order extension ---------------------------------------------------------------------

def extended_perms(base_perms: np.ndarray, seed: int) -> np.ndarray:
    """Pinned epochs 0-29 unchanged, then one permutation per epoch 30..159 from a named stream."""
    if base_perms.shape[0] != BASE_EPOCHS:
        raise ValueError("expected the 30-epoch pinned order, got %s" % (base_perms.shape,))
    n = base_perms.shape[1]
    ext = [np.random.Generator(np.random.PCG64(derive_seed(seed, PERM_EXTENSION_STREAM % e)))
           .permutation(n).astype(np.int32) for e in range(BASE_EPOCHS, EPOCHS)]
    return np.concatenate([base_perms, np.stack(ext)])


def sha_array(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
