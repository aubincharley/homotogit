"""The experiment grid: four frozen methods, three seeds, four recipe variants.

Why a grid and not the twelve reference cells
--------------------------------------------
Four methods give four *conditions*.  Every contrast against the plain control
is well served by that -- the effects measured on the exploratory branch were
twenty to sixty times the run-to-run noise -- but no **correlation across
conditions** can be established from four points, and the temptation to read one
anyway is exactly the failure mode that wasted two passes there.

The grid therefore keeps the four frozen methods untouched and varies the
*training recipe* around them, which the configuration already supports:

    4 methods x 3 seeds                                      12 cells
    4 methods x {lr 0.0025, lr 0.01, wd 1e-4, wd 2e-3}       16 cells
                                                       ---------------
                                                             28 cells
                                                             20 conditions

Twenty conditions detect a partial correlation of 0.70 at the five per cent
level, which is where the exploratory estimate sat.  It is tight, and the
analysis reports the detection ceiling alongside every coefficient rather than
leaving the reader to assume it.

The variants also answer a question the reference cells cannot: if a quantity
tracks generalisation just as well when the *learning rate* moves as when the
curriculum does, it is not a signature of continuation but a general
phenomenon.  Those are two different papers, and the grid distinguishes them at
the cost of one GPU-hour.

Every variant is marked ``unvalidated`` by ``ExperimentConfig``; only the twelve
reference cells carry ``reference``.  That distinction is deliberate and is
carried into the results.
"""
from __future__ import annotations

from dataclasses import replace

from .presets import reference

#: the four frozen methods, unchanged
METHOD_IDS = ("plain", "resolution_max_b1", "gaussian_postrelu",
              "resolution_max_b1_gaussian_conv")

#: seeds for the reference cells; the recipe variants use seed 0 only
SEEDS = (0, 1, 2)

#: recipe variants, applied to every method at seed 0.  Each changes exactly one
#: knob of the reference optimiser so the axis is unambiguous.
VARIANTS = (
    {"id": "lr_low", "lr": 0.0025},
    {"id": "lr_high", "lr": 0.01},
    {"id": "wd_low", "weight_decay": 1e-4},
    {"id": "wd_high", "weight_decay": 2e-3},
)


def cell_id(method_id: str, seed: int, variant: str | None) -> str:
    return "%s__%s__seed%d" % (method_id, variant or "reference", seed)


def build_grid(*, data_root: str = "data", assets_dir: str = "assets/cifar10_resnet20bn",
               out_dir: str = "runs", device: str = "auto") -> list:
    """Every cell as ``(cell_id, ExperimentConfig)``, reference cells first."""
    cells = []
    for m in METHOD_IDS:
        for s in SEEDS:
            cfg = reference(m, s, data_root=data_root, assets_dir=assets_dir,
                            out_dir=out_dir, device=device)
            cfg.run.name = cell_id(m, s, None)
            cells.append((cfg.run.name, cfg))
    for m in METHOD_IDS:
        for v in VARIANTS:
            cfg = reference(m, 0, data_root=data_root, assets_dir=assets_dir,
                            out_dir=out_dir, device=device)
            knob = {k: val for k, val in v.items() if k != "id"}
            cfg.optimizer = replace(cfg.optimizer, **knob)
            cfg.validation_status = "unvalidated"
            cfg.notes.append("recipe variant %r: %s" % (v["id"], knob))
            cfg.run.name = cell_id(m, 0, v["id"])
            cells.append((cfg.run.name, cfg))
    return cells


def summary() -> dict:
    cells = build_grid()
    ref = [c for c, _ in cells if c.endswith("__reference__seed0")
           or "__reference__" in c]
    return {"n_cells": len(cells), "n_conditions": len(METHOD_IDS) * (1 + len(VARIANTS)),
            "methods": list(METHOD_IDS), "seeds": list(SEEDS),
            "variants": [v["id"] for v in VARIANTS],
            "n_reference_cells": len(ref),
            "detection_ceiling_note": ("20 conditions detect |r| >= 0.70 at p < 0.05; "
                                       "the exploratory estimate was 0.72"),
            "cells": [c for c, _ in cells]}
