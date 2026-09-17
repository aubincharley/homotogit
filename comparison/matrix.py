"""The frozen 42-cell comparison matrix: settings, arms, seeds, reuse decisions, configs.

Settings (recipes recovered from versioned code, not re-chosen):

* ``sgd``    ``continuation_core.presets.reference`` unchanged: SGD lr 0.005, momentum 0.9,
             no Nesterov, coupled weight decay 5e-4 on every parameter, 60 warm-up updates then
             cosine to 0, 30 epochs x 391 updates, effective batch 128 in microbatches of 32,
             float32, no augmentation, per-epoch evaluation (current/target paths, 500-image
             training probe + full test set, saved BN statistics).
* ``adamw``  ``scripts/job_optimizer_benchmark.config_for`` at ``CHOSEN_LR['adamw']``: identical
             except AdamW lr 0.02, betas (0.9, 0.999), eps 1e-8, decoupled weight decay 5e-4.

Checkpoint cadence is the only recorded setting changed: final checkpoint at update 11,730
plus ``rolling.pt`` (the AdamW benchmark already disabled per-epoch checkpoints; the SGD
reference wrote one per epoch).  Checkpoint writes do not enter the training computation.
"""
from __future__ import annotations

import json
from pathlib import Path

from continuation_core.config import OptimizerConfig
from continuation_core.presets import reference, reference_optimizer

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies" / "comparison_cbs_sdpoint"

HISTORICAL = ("plain", "resolution_max_b1", "gaussian_postrelu", "resolution_max_b1_gaussian_conv")
NEW = ("cbs_published_schedule", "cbs_budget_matched", "sdpoint")
ARMS = HISTORICAL + NEW
SETTINGS = ("sgd", "adamw")
SEEDS = (0, 1, 2)
TOTAL_UPDATES = 11730

#: SGD historical cells: landscape_v2 runs (visualization branch raw outputs), final checkpoints.
#: Verified before freezing (studies/comparison_cbs_sdpoint/protocol/reuse_audit.json).
V2_ACCOUNT = {0: "maxnicaise", 1: "maxlefrr", 2: "maxnikezz"}

#: Shards (setting, seeds) per account.  maxmonstre is excluded: it holds the running
#: geometry_final job.  maxfrrsava was allocated SGD seed 2 but its pilot kernel was granted no
#: GPU (torch 2.10.0+cpu, 2026-09-17 10:44 UTC, raw/pilot/maxfrrsava), so that seed moved to
#: maxlebossdu91, which runs SGD seeds 1 and 2 sequentially in one kernel.
SHARDS = {
    "maximemonstrenikez": ("sgd", (0,)),
    "maxlebossdu91": ("sgd", (1, 2)),
    "maxnicaise": ("adamw", (0,)),
    "maxlefrr": ("adamw", (1,)),
    "maxnikezz": ("adamw", (2,)),
}
NO_GPU_ACCOUNTS = ("maxfrrsava",)
RESERVED_ACCOUNT = "maxmonstre"


def optimizer(setting: str) -> OptimizerConfig:
    ref = reference_optimizer()
    if setting == "sgd":
        return ref
    if setting == "adamw":
        return OptimizerConfig(name="adamw", lr=0.02, weight_decay=5e-4, schedule=ref.schedule,
                               warmup_updates=ref.warmup_updates, min_lr=ref.min_lr)
    raise KeyError(setting)


def config(setting: str, arm: str, seed: int, data_root: str, assets_dir: str, out_dir: str):
    cfg = reference(arm, seed, data_root=data_root, assets_dir=assets_dir, out_dir=out_dir,
                    optimizer=None if setting == "sgd" else optimizer(setting))
    cfg.checkpoint.every_epoch = False
    cfg.checkpoint.every_updates = TOTAL_UPDATES          # the final checkpoint only
    cfg.checkpoint.keep_rolling = True
    cfg.run.name = "%s__%s__seed%d" % (setting, arm, seed)
    cfg.notes.append("comparison-cbs-sdpoint: final checkpoint + rolling only; training unchanged")
    if arm in NEW:
        cfg.validation_status = "comparator"
    return cfg


def cells() -> list:
    out = []
    for setting in SETTINGS:
        for seed in SEEDS:
            for arm in ARMS:
                if arm in NEW:
                    action = "train_new"
                elif setting == "sgd":
                    action = "reuse_landscape_v2_checkpoint"
                else:
                    action = "rerun_historical_no_checkpoint"
                out.append({"setting": setting, "arm": arm, "seed": seed, "action": action,
                            "run": "%s__%s__seed%d" % (setting, arm, seed)})
    return out


def shard_cells(setting: str, seeds) -> list:
    return [c for c in cells() if c["setting"] == setting and c["seed"] in tuple(seeds)]


if __name__ == "__main__":
    cs = cells()
    print(json.dumps({"n_cells": len(cs), "by_action": {a: sum(c["action"] == a for c in cs)
                                                        for a in sorted({c["action"] for c in cs})}}, indent=1))
