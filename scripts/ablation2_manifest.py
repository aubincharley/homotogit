"""Second wave of the anti-aliasing ablation: the resolution row, 6 configurations.

Wave 1 (``scripts/ablation_manifest.py``, 12 cells) measured the filter question
at **constant 32x32**.  It is frozen and untouched; this file adds the missing
row -- the same placement question crossed with the repo's progressive-resolution
schedule -- plus the one control wave 1 could not provide.

Every cell reuses wave 1's **pinned assets**, so all 18 cells (12 + 6) share
initial weights, BatchNorm buffers, per-epoch permutations and probe indices and
are paired with each other.  Nothing here is comparable to
``results/campaign_results.json``: those runs used initial weights that no longer
exist (see ``scripts/stage_ablation_assets.py``).  That is exactly why ``R2``
re-runs "yesterday's best" inside this batch instead of quoting its old number.

Schedules are taken verbatim from the campaign that produced them
(``scripts/campaign_manifest.py``): ``Rprog`` = 16x16 for six epochs, 24x24 for
six, 32x32 for eighteen; the plateau levels in three-epoch blocks with an exact
bypass from epoch 21; and the historical per-site rescaling
``sigma_l(e) = (r(e)/32) * G(e)``, whose effective schedule is deliberately
non-monotone at the two resolution transitions.
"""
from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from scripts.ablation_manifest import BUILDER, EPOCHS, BYPASS_FROM, LEVELS, SEEDS

#: resolution schedules, verbatim from scripts/campaign_manifest.py
RESOLUTIONS_ABL2 = {
    "R32":   [32] * EPOCHS,
    "Rprog": [16] * 6 + [24] * 6 + [32] * 18,
}

# rough per-run seconds on a T4, for queue balancing only
_COST = {"R1": 495, "R2": 705, "R3": 575, "R4": 495, "R5": 605, "R6": 625}


def build_configs() -> list:
    """The 6 frozen configurations of wave 2, in manifest order."""
    cfgs = []

    def add(cid, resolution, levels, placement, reduction, *,
            constant=False, primary_path="target", question="", rationale=""):
        cfgs.append({
            "id": cid, "group": "wave2", "test": question,
            "operator": "none" if levels == "none" else "gaussian",
            "levels": levels, "resolution": resolution, "reduction": reduction,
            "placement": placement, "mask": "all19", "blurpool_sigma": None,
            "constant": constant, "primary_path": primary_path,
            "controller_builder": BUILDER, "diagnostics": True,
            "rationale": rationale, "est_seconds": _COST[cid],
        })

    add("R1", "Rprog", "none", "conv_out", "input_bilinear",
        question="resolution seule",
        rationale="progressive resolution with no filter at all: the term every "
                  "other resolution arm has to be decomposed against")
    add("R2", "Rprog", "plateau", "conv_out", "input_bilinear",
        question="temoin = hier",
        rationale="the campaign's best configuration, re-run on this batch's "
                  "pinned weights so it can actually be compared to R3")
    add("R3", "Rprog", "plateau", "post_block", "input_bilinear",
        question="test principal",
        rationale="does the post-ReLU placement's advantage survive on top of "
                  "progressive resolution")
    add("R4", "Rprog", "none", "conv_out", "stem_max",
        question="reduction post-ReLU",
        rationale="reduce after the stem's ReLU instead of on the input image, "
                  "no filter; re-run fresh so it is comparable to R1")
    add("R5", "Rprog", "plateau", "post_block", "stem_max",
        question="reduction + flou tous deux post-ReLU",
        rationale="both interventions on the post-ReLU side")
    add("R6", "R32", "const0.5", "post_block", "input_bilinear",
        constant=True, primary_path="current",
        question="l'annelage sert-il au meilleur placement",
        rationale="wave 1 showed a constant sigma matches the annealed schedule "
                  "at conv_out; this asks the same question at post_block, the "
                  "placement that won. Not a continuation: read the current path")
    return cfgs


def build_cells() -> list:
    return [{**c, "seed": s, "cell_id": "%s__seed%d" % (c["id"], s)}
            for c in build_configs() for s in SEEDS]


def assign(cells, n_jobs=4) -> list:
    """Deterministic longest-processing-time-first split, as in wave 1."""
    queues, loads = [[] for _ in range(n_jobs)], [0] * n_jobs
    for c in sorted(cells, key=lambda c: (-c["est_seconds"], c["cell_id"])):
        j = loads.index(min(loads))
        queues[j].append(c)
        loads[j] += c["est_seconds"]
    return queues


def summary() -> dict:
    cells = build_cells()
    qs = assign(cells)
    return {"wave": 2, "n_configs": len(build_configs()), "n_cells": len(cells),
            "seeds": list(SEEDS), "epochs": EPOCHS, "bypass_from": BYPASS_FROM,
            "resolutions": {k: [v[0], v[6], v[12]] for k, v in RESOLUTIONS_ABL2.items()},
            "jobs": [{"job": i, "cells": [c["cell_id"] for c in q],
                      "est_seconds": sum(c["est_seconds"] for c in q)}
                     for i, q in enumerate(qs)]}


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=2))
    for c in build_configs():
        print("%-4s res=%-6s lv=%-9s place=%-11s red=%-15s primary=%-8s %s"
              % (c["id"], c["resolution"], c["levels"], c["placement"],
                 c["reduction"], c["primary_path"], c["test"]))
