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
from .schedules import EpochSchedule

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


# ---------------------------------------------------------------------------
# The objective grid
# ---------------------------------------------------------------------------
#
# The question is whether the curriculum gain is a property of the intervention
# or of the pair (intervention, objective).  The four frozen methods are kept
# untouched and the *loss* is varied around them, exactly as the recipe variants
# vary the optimiser.
#
# Learning rate is swept for ``square`` alone, and that asymmetry is deliberate.
# Label smoothing and focal are reweightings of cross-entropy: their gradient
# with respect to the logits stays bounded in the same way, so the reference step
# size remains appropriate.  The square loss on logits does not share that scale
# -- its gradient is ``2(z - y)/C`` with ``z`` unbounded -- so a poor result at a
# single learning rate would not distinguish the objective from the step size.
# Comparing each objective at its own best learning rate is the only reading of
# that column that means anything.

#: reference optimiser learning rate, and the sweep used for ``square``
REFERENCE_LR = 0.005
#: The first attempt swept 0.0025 / 0.005 / 0.01 -- the reference and its two
#: neighbours -- and every cell failed to train: test accuracy 41.5 / 47.8 /
#: 57.1 %, training cross-entropy 2.17 / 2.11 / 2.02 against ln(10) = 2.30, and
#: **monotone increasing across the whole range**, so the optimum lay above it.
#: That range was the mistake it was meant to guard against: dividing the square
#: loss by ``C`` divides its gradient by ten, so its scale sits an order of
#: magnitude above cross-entropy's, not beside it.  The sweep is now centred
#: there.  Keep the two ranges disjoint -- if the best is again at an endpoint,
#: the sweep is still wrong.
SQUARE_LRS = (0.05, 0.1, 0.5)

#: ``(arm id, LossConfig fields, learning rates)``
LOSS_ARMS = (
    ("ce", {"name": "cross_entropy"}, (REFERENCE_LR,)),
    ("ls", {"name": "label_smoothing", "label_smoothing": 0.1}, (REFERENCE_LR,)),
    ("focal", {"name": "focal", "gamma": 2.0}, (REFERENCE_LR,)),
    ("square", {"name": "square", "normalise_by_classes": True}, SQUARE_LRS),
)


def loss_cell_id(method_id: str, arm: str, lr: float, seed: int) -> str:
    return "%s__%s__lr%g__seed%d" % (method_id, arm, lr, seed)


def build_loss_grid(*, arms=None, seeds=SEEDS, data_root: str = "data",
                    assets_dir: str = "assets/cifar10_resnet20bn",
                    out_dir: str = "runs", device: str = "auto") -> list:
    """Every ``(cell_id, ExperimentConfig)`` of the objective grid.

    ``arms`` selects a subset of :data:`LOSS_ARMS` by id, so the cross-entropy
    arm can be dropped when the recorded reference cells are being reused: this
    module's change is a no-op for ``cross_entropy`` (``losses.build`` returns
    ``F.cross_entropy`` unchanged), so those runs remain valid as the control.
    """
    from dataclasses import replace as _replace

    from .config import LossConfig

    wanted = set(arms) if arms is not None else {a for a, _, _ in LOSS_ARMS}
    unknown = wanted - {a for a, _, _ in LOSS_ARMS}
    if unknown:
        raise ValueError("unknown loss arm(s) %s" % sorted(unknown))

    cells = []
    for arm, fields_, lrs in LOSS_ARMS:
        if arm not in wanted:
            continue
        for m in METHOD_IDS:
            for lr in lrs:
                for s in seeds:
                    cfg = reference(m, s, data_root=data_root, assets_dir=assets_dir,
                                    out_dir=out_dir, device=device)
                    cfg.loss = LossConfig(**fields_)
                    if lr != REFERENCE_LR:
                        cfg.optimizer = _replace(cfg.optimizer, lr=lr)
                    if arm != "ce" or lr != REFERENCE_LR:
                        cfg.validation_status = "unvalidated"
                        cfg.notes.append("objective grid: loss=%s lr=%g" % (arm, lr))
                    cfg.run.name = loss_cell_id(m, arm, lr, s)
                    cells.append((cfg.run.name, cfg))
    return cells


def loss_summary(**kw) -> dict:
    cells = build_loss_grid(**kw)
    per_arm = {}
    for arm, fields_, lrs in LOSS_ARMS:
        per_arm[arm] = {"loss": fields_, "learning_rates": list(lrs),
                        "n_cells": sum(1 for c, _ in cells if "__%s__" % arm in c)}
    return {"n_cells": len(cells), "methods": list(METHOD_IDS),
            "seeds": list(SEEDS), "arms": per_arm,
            "note": ("test accuracy is the objective-agnostic comparator; every run "
                     "also reports cross-entropy, so columns share one scale")}


# ---------------------------------------------------------------------------
# The augmentation grid
# ---------------------------------------------------------------------------
#
# The recorded benchmark trains with **no augmentation at all** -- ``data.py``
# states it -- and the control reaches 75.9 % where a ResNet-20 on CIFAR-10 with
# the standard recipe reaches about 91-92 %.  So the first question anyone will
# ask is whether the curriculum gain is recovering part of what random crop and
# horizontal flip give for free.  It is not a rhetorical worry: crop and flip are
# input-space interventions, exactly like the blur and the resolution reduction.
#
# The budget axis is here rather than left implicit because augmentation pays off
# over long schedules.  Comparing an augmented run to an unaugmented one at 30
# epochs alone would confound "the gain survives augmentation" with "30 epochs is
# too short for augmentation to matter"; adding 60 epochs on **both** sides
# separates them.  The unaugmented 30-epoch cells already exist -- they are the
# reference cells of the 28-cell grid -- so only three of the four corners are
# paid for.

#: ``(augmentation, epochs, lr)`` corners.  ``("none", 30, 0.005)`` is the
#: recorded one.  The learning rate is part of a corner rather than a separate
#: axis because 0.005 is demonstrably below the optimum -- cross-entropy is still
#: improving at the top of the 28-cell grid's own sweep, and the square loss needs
#: twenty times it -- so a corner run at 0.005 alone can answer "the gain survives
#: augmentation" with a baseline that never got strong enough for the question to
#: mean anything.
AUGMENT_CORNERS = (("none", 30, 0.005),
                   ("crop_flip", 30, 0.005), ("crop_flip", 30, 0.01),
                   ("none", 60, 0.005), ("none", 60, 0.01),
                   ("crop_flip", 60, 0.005), ("crop_flip", 60, 0.01))


def stretch_method(spec, factor: int):
    """The same method over ``factor`` times as many epochs.

    Schedules are indexed by **absolute** epoch, so doubling the budget without
    touching them would leave the operator off for 39 of 60 epochs instead of 9
    of 30 -- the curriculum would fall from 70 % of training to 35 %, and its
    contribution would shrink for a reason that has nothing to do with the
    question being asked.  Multiplying the boundaries keeps the method's shape
    and makes the 30-versus-60 comparison a comparison of budget alone.

    The stretched schedule is written into the run's config like any other, so
    what ran stays recoverable from the record.
    """
    from dataclasses import replace as _replace

    def _sched(sc):
        return EpochSchedule(tuple(int(v) * int(factor) for v in sc.starts), sc.values)

    out = spec
    if out.gaussian is not None:
        out = _replace(out, gaussian=_replace(out.gaussian,
                                              schedule=_sched(out.gaussian.schedule)))
    if out.resolution is not None:
        out = _replace(out, resolution=_replace(out.resolution,
                                                schedule=_sched(out.resolution.schedule)))
    return out


def augment_cell_id(method_id: str, aug: str, epochs: int, lr: float, seed: int) -> str:
    return "%s__%s__e%d__lr%g__seed%d" % (method_id, aug, epochs, lr, seed)


def build_augment_grid(*, corners=None, seeds=SEEDS, stretch: bool = False,
                       data_root: str = "data",
                       assets_dir: str = "assets/cifar10_resnet20bn",
                       out_dir: str = "runs", device: str = "auto") -> list:
    """Every ``(cell_id, ExperimentConfig)`` of the augmentation grid.

    ``corners`` selects a subset of :data:`AUGMENT_CORNERS`; the default drops
    ``("none", 30, 0.005)`` because those runs are the recorded reference cells
    and ``analyze_augment.py`` reads them from the grid shards as that corner.

    ``stretch`` decides what a longer budget means, and the two readings are
    different experiments rather than one right and one wrong.

    Left ``False`` (the default), schedules keep their **absolute** epochs: the
    resolution reaches 32 at epoch 12 and the blur switches off at 21 whatever the
    budget, so a 60-epoch run is *the method as defined* followed by 39 epochs of
    bare training.  It asks whether the head start the curriculum buys survives
    being trained past.

    Set ``True``, the boundaries are multiplied by the budget ratio, so the
    operator keeps the same share of training and the comparison isolates budget
    alone.  Use it when the question is about the budget rather than about what
    the curriculum leaves behind.
    """
    from dataclasses import replace as _replace

    wanted = (tuple(tuple(c) for c in corners) if corners is not None
              else tuple(c for c in AUGMENT_CORNERS if c != ("none", 30, 0.005)))
    unknown = [c for c in wanted if c not in AUGMENT_CORNERS]
    if unknown:
        raise ValueError("unknown corner(s) %s" % (unknown,))

    cells = []
    for aug, epochs, lr in wanted:
        for m in METHOD_IDS:
            for s in seeds:
                cfg = reference(m, s, data_root=data_root, assets_dir=assets_dir,
                                out_dir=out_dir, device=device)
                cfg.data = _replace(cfg.data, augmentation=aug)
                cfg.budget = _replace(cfg.budget, epochs=int(epochs))
                cfg.optimizer = _replace(cfg.optimizer, lr=float(lr))
                if stretch and epochs != 30:
                    factor = int(epochs) // 30
                    if factor * 30 != int(epochs):
                        raise ValueError("epochs must be a multiple of 30 to stretch "
                                         "the schedule, got %r" % (epochs,))
                    cfg.method = stretch_method(cfg.method_spec(), factor).to_dict()
                    cfg.notes.append("schedules stretched x%d so the operator keeps "
                                     "the same share of training" % factor)
                elif epochs != 30:
                    cfg.notes.append(
                        "schedules left at absolute epochs: the operator reaches its "
                        "target state at the same epoch as at 30, and the remaining "
                        "%d epochs train the bare network" % (int(epochs) - 21))
                if (aug, epochs, lr) != ("none", 30, 0.005):
                    cfg.validation_status = "unvalidated"
                    cfg.notes.append("augmentation grid: %s, %d epochs, lr %g"
                                     % (aug, epochs, lr))
                cfg.run.name = augment_cell_id(m, aug, epochs, lr, s)
                cells.append((cfg.run.name, cfg))
    return cells


def augment_summary(**kw) -> dict:
    cells = build_augment_grid(**kw)
    return {"n_cells": len(cells), "methods": list(METHOD_IDS), "seeds": list(SEEDS),
            "corners": [list(c) for c in AUGMENT_CORNERS],
            "schedule_note": ("schedules are indexed by absolute epoch, so a 60-epoch "
                              "corner leaves the operator off for 39 of 60 epochs "
                              "instead of 9 of 30; decide whether to stretch them "
                              "before reading that corner"),
            "recipe": "pad 4 zeros, random 32x32 crop, horizontal flip p=0.5; training only",
            "note": ("the (none, 30) corner is the recorded reference cells and is "
                     "not re-run; evaluation never augments, so the train probe "
                     "measures clean images and the train/test gap stays readable")}
