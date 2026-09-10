"""Fourth wave: seeds 1 and 2 for the four arms that finished at the top.

Waves 1-3 ran one seed.  The six best arms finished inside 0.47 point -- 47 test
images out of 10,000 -- while their runtimes spanned 32 %.  A single run defines
no distribution (errata C-12/C-13), so that ranking is not a result; it is an
ordering of numbers that may be noise.

This wave adds the two missing seeds to the four arms that matter for the
decision:

    D0    reduction after blocks[0], no blur
    D1    reduction after blocks[1], no blur   <- the practical candidate
    D2    reduction after blocks[2], no blur
    D2G   reduction after blocks[2] + blur     <- the nominal first place

Seed 0 already exists for all four, so only seeds 1 and 2 are run: 8 cells.
Their configurations are imported unchanged from ``ablation3_manifest`` -- this
file adds seeds, never a variant -- and they reuse the same pinned assets, whose
``init_seed1``/``init_seed2`` and ``perm_seed1``/``perm_seed2`` were staged with
seed 0 from the start.

What three seeds buy: a descriptive spread across seeds and a paired per-seed
sign for each comparison.  What they do not buy: a confidence statement.  Three
runs remain a small sample and the test set has prior exposure.
"""
from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from scripts.ablation3_manifest import build_configs as _w3

WANTED = ("D0", "D1", "D2", "D2G")
NEW_SEEDS = (1, 2)


def build_configs() -> list:
    cfgs = [c for c in _w3() if c["id"] in WANTED]
    missing = set(WANTED) - {c["id"] for c in cfgs}
    if missing:
        raise KeyError("configurations not found in wave 3: %s" % sorted(missing))
    return cfgs


def build_cells() -> list:
    return [{**c, "seed": s, "cell_id": "%s__seed%d" % (c["id"], s)}
            for c in build_configs() for s in NEW_SEEDS]


def assign(cells, n_jobs=4) -> list:
    queues, loads = [[] for _ in range(n_jobs)], [0] * n_jobs
    for c in sorted(cells, key=lambda c: (-c["est_seconds"], c["cell_id"])):
        j = loads.index(min(loads))
        queues[j].append(c)
        loads[j] += c["est_seconds"]
    return queues


def summary() -> dict:
    cells = build_cells()
    qs = assign(cells)
    return {"wave": 4, "configs": list(WANTED), "new_seeds": list(NEW_SEEDS),
            "n_cells": len(cells),
            "jobs": [{"job": i, "cells": [c["cell_id"] for c in q],
                      "est_seconds": sum(c["est_seconds"] for c in q)}
                     for i, q in enumerate(qs)]}


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=2))
