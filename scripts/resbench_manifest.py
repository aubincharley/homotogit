"""The frozen resolution-only grid: 29 configurations x 3 seeds = 87 runs.

Built from three questions, deduplicated by the **complete executed
configuration** (operator, location, path) rather than by display name:

* operator x location : 7 operators at ``input`` and at ``D1``, path Rprog  -> 14
* depth               : ``O_ref`` at all five locations, Rprog              -> +3
                        (``input`` and ``D1`` are already in the 14)
* path                : {O_ref, maxblur, hminus1} x {Rgentle, Rreverse, Rlate}
                        at ``D1``  (their Rprog cells already exist)         -> +9
* controls            : plain, fixed D1-16, fixed D1-24                      -> +3

``O_ref`` is adaptive max pooling -- recovered from the historical D0/D1/D2
implementation, not guessed -- and it is one of the seven operators, so no extra
reference arm is needed and the grid stays at 29 configurations.

Priority order for the budget, applied to whole three-seed configurations only:

1. every operator at D1; O_ref at all five depths; plain and the fixed controls;
   all four paths for O_ref at D1
2. the remaining input-location operator comparisons
3. the extra path crossings for maxblur and hminus1
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from continuation.resolution_ops import LOCATIONS, O_REF, OPERATORS

SEEDS = (0, 1, 2)
PATH_SET = ("Rprog", "Rgentle", "Rreverse", "Rlate")
PATH_OPERATORS = (O_REF, "maxblur", "hminus1")


def _cid(operator, location, path):
    return "%s__%s__%s" % (operator, location, path)


def build_configs() -> list:
    cfgs, seen = [], {}

    def add(operator, location, path, group, priority):
        cid = _cid(operator, location, path)
        if cid in seen:                       # dedup on the executed config
            seen[cid]["groups"].append(group)
            seen[cid]["priority"] = min(seen[cid]["priority"], priority)
            return
        rec = {"id": cid, "operator": operator, "location": location,
               "path": path, "groups": [group], "priority": priority}
        seen[cid] = rec
        cfgs.append(rec)

    # 1. operator x location
    for op in OPERATORS:
        add(op, "D1", "Rprog", "operator_x_location", 1)
    for op in OPERATORS:
        add(op, "input", "Rprog", "operator_x_location",
            1 if op == O_REF else 2)
    # 2. depth, O_ref
    for loc in LOCATIONS:
        add(O_REF, loc, "Rprog", "depth", 1)
    # 3. paths
    for op in PATH_OPERATORS:
        for path in PATH_SET:
            add(op, "D1", path, "path", 1 if op == O_REF else 3)
    # 4. controls
    add("none", "none", "none", "control", 1)
    add(O_REF, "D1", "fixed16", "control", 1)
    add(O_REF, "D1", "fixed24", "control", 1)
    return cfgs


def build_cells(configs=None) -> list:
    configs = configs or build_configs()
    return [{**c, "seed": s, "cell_id": "%s__seed%d" % (c["id"], s)}
            for c in configs for s in SEEDS]


def select(configs, budget_seconds, per_cell_seconds, workers=6, reserve=0.12):
    """Keep whole three-seed configurations in priority order within budget."""
    usable = budget_seconds * (1.0 - reserve) * workers
    chosen, deferred, total = [], [], 0.0
    for c in sorted(configs, key=lambda c: (c["priority"], c["id"])):
        cost = 3 * per_cell_seconds(c)
        if total + cost <= usable:
            chosen.append(c)
            total += cost
        else:
            deferred.append(c)
    return chosen, deferred, total


if __name__ == "__main__":
    cfgs = build_configs()
    cells = build_cells(cfgs)
    print("configurations: %d (unique %d)" % (len(cfgs), len({c['id'] for c in cfgs})))
    print("cells:          %d (unique %d)" % (len(cells), len({c['cell_id'] for c in cells})))
    from collections import Counter
    print("by priority:", dict(Counter(c["priority"] for c in cfgs)))
    for g in ("operator_x_location", "depth", "path", "control"):
        n = len([c for c in cfgs if g in c["groups"]])
        print("  %-20s %d configurations" % (g, n))
