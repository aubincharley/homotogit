"""Third wave: the depth-of-reduction axis, crossed with the blur. 6 cells.

The question this closes: **can anything beat R3** (reduction on the image +
Gaussian after the ReLU, 0.8014), the best arm of waves 1-2.

Waves 1 and 2 measured two depths for the single resolution reduction -- on the
image and just after the stem -- and going one step deeper was worth +1.0 point
without the blur.  This wave adds the three deeper points and, at each of them,
runs **both** with and without the blur, so the grid is complete:

    depth of the reduction        no blur      blur after the ReLU
    on the image                  R1 0.7892    R3 0.8014   <- current champion
    after the stem                R4 0.7995    R5 0.7965
    after blocks[0]               D0           D0G
    after blocks[1]               D1           D1G
    after blocks[2]               D2           D2G

Both columns are run rather than the no-blur column alone: the only evidence that
the blur stops helping at depth was ``R5 - R4 = -0.30 pp``, which is **inside the
undecidable band at one seed**.  Reading a trend from it would be exactly the
over-reading the repository's errata list exists to prevent, and it would also
fail to answer whether a deeper reduction *plus* the blur beats R3.

Reduction operator is **max-pooling** at every depth: ``stem_max`` beat
``stem_bilinear`` by 1.06 points without the blur and 0.51 with it, at the one
depth where both were measured.

Everything else is frozen and identical to waves 1-2, including the pinned
initial weights, so all 24 cells across the three waves are mutually paired.
"""
from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from scripts.ablation_manifest import BUILDER, EPOCHS, SEEDS
from scripts.ablation2_manifest import RESOLUTIONS_ABL2  # noqa: F401  (Rprog)

DEPTHS = [("D0", "block0_max", "apres blocks[0]"),
          ("D1", "block1_max", "apres blocks[1]"),
          ("D2", "block2_max", "apres blocks[2]")]

# rough per-run seconds on a T4: each block left at full resolution during the
# first twelve epochs adds work, and the blur adds roughly 100 s on top
_BASE = {"D0": 465, "D1": 490, "D2": 520}


def build_configs() -> list:
    cfgs = []
    for cid, reduction, where in DEPTHS:
        for suffix, levels, placement, extra in (
                ("", "none", "conv_out", 0),
                ("G", "plateau", "post_block", 105)):
            cfgs.append({
                "id": cid + suffix, "group": "wave3",
                "test": "profondeur de la reduction",
                "operator": "none" if levels == "none" else "gaussian",
                "levels": levels, "resolution": "Rprog", "reduction": reduction,
                "placement": placement, "mask": "all19", "blurpool_sigma": None,
                "constant": False, "primary_path": "target",
                "controller_builder": BUILDER, "diagnostics": True,
                "rationale": "single resolution reduction %s, %s"
                             % (where, "no blur" if levels == "none"
                                else "Gaussian plateau after the ReLU"),
                "est_seconds": _BASE[cid] + extra,
            })
    return cfgs


def build_cells() -> list:
    return [{**c, "seed": s, "cell_id": "%s__seed%d" % (c["id"], s)}
            for c in build_configs() for s in SEEDS]


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
    return {"wave": 3, "n_cells": len(cells), "seeds": list(SEEDS),
            "epochs": EPOCHS, "depths": [d[1] for d in DEPTHS],
            "jobs": [{"job": i, "cells": [c["cell_id"] for c in q],
                      "est_seconds": sum(c["est_seconds"] for c in q)}
                     for i, q in enumerate(qs)]}


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=2))
    for c in build_configs():
        print("%-4s red=%-12s lv=%-8s place=%-11s est=%ds  %s"
              % (c["id"], c["reduction"], c["levels"], c["placement"],
                 c["est_seconds"], c["rationale"]))
