"""Checkpoints available to the study, their provenance, and what is missing.

The reference runs of the four selected methods (experiment
``unified_selected`` on branch ``benchmark-organized``) kept **no checkpoint**:
``scripts/unified_driver.py`` saves only ``rolling.pt`` and deletes it when a
cell finishes.  The only saved weights for a selected configuration come from
the resolution-only batch ``resbench`` (``scripts/resbench_driver.py``), which
kept ``checkpoint_ep06/12/18/30.pt`` for every cell:

* ``none__none__none``  -> preset ``plain``
* ``max__D1__Rprog``    -> preset ``resolution_max_b1``: ``F.adaptive_max_pool2d``
  in a forward pre-hook on ``model.blocks[2]``, resolution 16/24/32 for epochs
  0-5/6-11/12-29, identity at 32 (``continuation.resolution_ops``: ``op_max``,
  ``BLOCK_TARGET["D1"] = 2``, ``PATHS["Rprog"]``, ``active()``).

Same pinned asset set as the unified batch (``r20bn-campaign-assets``; digests
checked in ``verify_sources``), same recipe (``campaign_driver.PROTOCOL``).
These are **different runs** of the same configurations, not the unified runs:
their final accuracies differ (GPU non-determinism, see
``docs/VERIFICATION.md``).  No checkpoint exists for ``gaussian_postrelu`` or
``resolution_max_b1_gaussian_conv`` in any batch.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path

import torch

from continuation_core.controller import InterventionController
from continuation_core.methods import get_method
from continuation_core.models import site_map

KAGGLE_OUTPUTS = Path(os.environ.get(
    "LANDSCAPE_KAGGLE_OUTPUTS",
    "C:/Users/mnica/Documents/Projet_filiere/results/kaggle_outputs"))
RESBENCH_JOBS = ("resbench-j0-20260910-214422", "resbench-j1-20260910-214450",
                 "resbench-j2-20260910-214517")
CKPT_EPOCHS = (6, 12, 18, 30)
UPDATES_PER_EPOCH = 391
TOTAL_UPDATES = 11730
SEEDS = (0, 1, 2)

AVAILABLE = {
    "plain": {"batch": "resbench", "cell": "none__none__none"},
    "resolution_max_b1": {"batch": "resbench", "cell": "max__D1__Rprog"},
}

MISSING = [
    {"method": m, "batch": "unified_selected", "cells": ["%s__seed%d" % (c, s) for s in SEEDS],
     "reason": "unified_driver.py keeps only rolling.pt and deletes it at the end of "
               "each cell; no epoch checkpoint was ever written"}
    for m, c in (("plain", "plain"), ("resolution_max_b1", "shrink_b1"),
                 ("gaussian_postrelu", "blur_relu"),
                 ("resolution_max_b1_gaussian_conv", "shrink_b1_conv"))
] + [
    {"method": m, "batch": "any", "cells": [],
     "reason": "no batch with saved checkpoints ran this configuration (campaign "
               "filtered conv outputs with input/stem reductions only; resbench has no "
               "Gaussian)"}
    for m in ("gaussian_postrelu", "resolution_max_b1_gaussian_conv")
]


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_dir(method: str, seed: int) -> Path:
    cell = AVAILABLE[method]["cell"]
    hits = []
    for job in RESBENCH_JOBS:
        hits += glob.glob(str(KAGGLE_OUTPUTS / job / "resbench_job*" / ("%s__seed%d" % (cell, seed))))
    if len(hits) != 1:
        raise FileNotFoundError("expected one run dir for %s seed %d, found %s" % (method, seed, hits))
    return Path(hits[0])


def controller(method: str) -> InterventionController:
    return InterventionController(get_method(method), site_map("resnet20_bn_cifar"),
                                  "resnet20_bn_cifar")


def states_at(method: str, epochs_completed: int) -> dict:
    """State used by the last update, and state of the next update."""
    c = controller(method)
    used = None if epochs_completed == 0 else c.state_for_epoch(epochs_completed - 1).to_dict()
    nxt = (None if epochs_completed >= TOTAL_UPDATES // UPDATES_PER_EPOCH
           else c.state_for_epoch(epochs_completed).to_dict())
    return {"used_for_last_update": used, "next_update": nxt,
            "target": c.target_state().to_dict()}


def verify_sources(core_assets: Path) -> dict:
    """Asset digests, recipe, recorded per-epoch states and checkpoint inventory."""
    core = json.loads((core_assets / "assets_manifest.json").read_text())
    report = {"jobs": {}, "runs": {}}
    for job in RESBENCH_JOBS:
        for d in glob.glob(str(KAGGLE_OUTPUTS / job / "resbench_job*")):
            av = json.loads(Path(d, "assets_verification.json").read_text())
            report["jobs"][Path(d).name] = {
                "all_match": bool(av["all_match"]),
                "same_files_as_core_assets": all(
                    av["manifest"]["files"][f]["sha256"] == core["files"][f]["sha256"]
                    for f in core["files"]),
                "environment": json.loads(Path(d, "environment.json").read_text())}
    for method in AVAILABLE:
        c = controller(method)
        for seed in SEEDS:
            d = run_dir(method, seed)
            summary = json.loads((d / "summary.json").read_text())
            metrics = json.loads((d / "metrics.json").read_text())
            # recorded state of record k is the state of epoch k-1 (record 0: epoch 0)
            state_ok = True
            for rec in metrics:
                e = max(rec["epoch"] - 1, 0)
                want = c.state_for_epoch(e).resolution
                got = rec["current"]["resolution"]
                if method == "plain":
                    state_ok &= want is None and got in (32, None)
                else:
                    state_ok &= int(got) == int(want)
            ckpts = {}
            for ep in CKPT_EPOCHS:
                p = d / ("checkpoint_ep%02d.pt" % ep)
                ckpts[ep] = {"path": str(p), "exists": p.is_file(),
                             "bytes": p.stat().st_size if p.is_file() else None}
            report["runs"]["%s__seed%d" % (method, seed)] = {
                "run_dir": str(d), "cell": summary["cell_id"],
                "recipe": {k: summary[k] for k in ("epochs", "updates", "updates_per_epoch",
                                                   "n_train", "n_test")},
                "controller": summary["controller"],
                "recorded_states_match_core_schedule": bool(state_ok),
                "final_test_acc_recorded": summary["final_test_acc"],
                "checkpoints": ckpts}
    return report


def load_resbench_checkpoint(method: str, seed: int, epoch: int) -> dict:
    p = run_dir(method, seed) / ("checkpoint_ep%02d.pt" % epoch)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if int(ck["epoch"]) != epoch or int(ck["update"]) != epoch * UPDATES_PER_EPOCH:
        raise ValueError("%s: epoch/update %s/%s inconsistent with its name"
                         % (p, ck["epoch"], ck["update"]))
    if int(ck["cell"]["seed"]) != seed or ck["cell"]["id"] != AVAILABLE[method]["cell"]:
        raise ValueError("%s: cell %s does not match %s seed %d" % (p, ck["cell"], method, seed))
    return {"path": p, "ckpt": ck}
