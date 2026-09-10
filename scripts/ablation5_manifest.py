"""Fifth wave: a depth prior on sigma, on the architecture that won. 9 cells.

Base architecture is ``D2G``'s, unchanged: single resolution reduction after
``blocks[2]`` by max-pooling, Gaussian after the ReLU at the 10 post-block
positions, plateau schedule annealed to an exact zero from epoch 21.  The only
thing that varies is a **per-position multiplier on sigma**.

Why a depth prior at all
------------------------
Sigma is measured in pixels of the *local* feature map, and those maps shrink
with depth: 32 -> 16 -> 8, and 16 -> 8 -> 4 during the reduced phase.  A uniform
sigma is therefore the same brush on a poster and on a stamp.  Measured on this
network at sigma = 1: the 9-tap kernel spans 0.28 of a 32x32 map but **1.12** of
an 8x8 one and 2.25 of a 4x4 one, and reflection padding supplies 2 % of each
output at 32x32 against 9 % at 8x8 and **18 %** at 4x4.

What the argument is *not*: the per-pixel energy attenuation is identical at all
three stages (0.080 at sigma = 1, measured).  Uniform sigma does not remove more
information at depth; it stops being a well-posed local filter and becomes a
boundary-dominated global average.

Profiles, normalised on the MAXIMUM
-----------------------------------
Peak-normalisation rather than budget-matching, for two reasons.  It keeps every
effective sigma at or below ``G(e) <= 1``, so ``sigma_max`` stays 1.0, the kernel
stays at 9 taps, and **the completed 3-seed ``D2G`` runs remain a valid control**
with no support confound.  And matching the budget would have forced A1 up to
1.60 on the shallow positions to compensate its own reduction at depth --
defeating the very thing it tests.

The multiplier applies **on top of** ``q``.  With the profile computed on the
natural map sizes, ``m_l * q_l`` makes sigma track the *current* map size, which
is what "constant relative width" actually means once a resolution schedule is
running.

    position          0     1     2     3     4     5     6     7     8     9
    natural map      32    32    32    32    16    16    16     8     8     8
    A1  ~ H        1.00  1.00  1.00  1.00  0.50  0.50  0.50  0.25  0.25  0.25
    A3  ~ 1/H      0.25  0.25  0.25  0.25  0.50  0.50  0.50  1.00  1.00  1.00
    A4  ~ RF       0.09  0.22  0.34  0.47  0.66  0.91  1.00  1.00  1.00  1.00

``A3`` is not decorative symmetry.  All three profiles carry **less total blur**
than uniform (6.25, 5.50, 6.69 against 10), so if A1 wins, "better distributed"
and "simply less" are not separable without it: A3 has an even lower budget while
blurring *more* at depth.  Separating them completely would need a
budget-matched A1, which requires raising ``sigma_max`` and re-running the
control -- kept for a second round, and only if A1 wins.

Three seeds from the start: wave 4 measured an unpaired seed spread of 0.36 point
against 0.26 on paired differences, so one seed per profile could not rank them.
"""
from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from scripts.ablation_manifest import BUILDER

SEEDS = (0, 1, 2)

#: natural spatial size at each of the 10 post_block positions
H = (32, 32, 32, 32, 16, 16, 16, 8, 8, 8)
#: receptive field in input pixels at each position
RF = (3, 7, 11, 15, 21, 29, 37, 49, 65, 81)


def _peak(v):
    m = max(v)
    return [round(x / m, 5) for x in v]


PROFILES = {
    "A2": (_peak([(h / 32) ** 0.5 for h in H]),
           "sigma proportional to sqrt(map size): the same direction as A1 but "
           "damped, so the deepest positions land near 0.5 -- the value wave 1's "
           "dose-response peaked at -- instead of at 0.25 and 0.125, where the "
           "kernel is numerically a delta and the blur is simply off"),
    "A1": (_peak([h / 32 for h in H]),
           "sigma proportional to map size: less blur where the kernel already "
           "exceeds the map"),
    "A3": (_peak([32 / h for h in H]),
           "the exact opposite: more blur at depth. Direction control -- without "
           "it, 'better distributed' and 'simply less' are indistinguishable"),
    "A4": (_peak([min(r, 32) / 32 for r in RF]),
           "sigma proportional to receptive field: the only profile that varies "
           "*within* a stage rather than in three plateaus"),
}


#: The two families.  ``P`` keeps D2G's architecture -- the resolution reduction
#: is active, which is where the prior has to survive; ``Q`` runs at a constant
#: 32x32, which is where the prior can be tested *alone*.  The reduction already
#: coarsens depth, so ``P`` on its own could not separate "the prior is useless"
#: from "the prior is redundant with the reduction".
FAMILIES = {
    "P": {"resolution": "Rprog", "reduction": "block2_max",
          "control": "D2G", "est": 575,
          "what": "avec la reduction apres blocks[2] (architecture D2G)"},
    "Q": {"resolution": "R32", "reduction": "input_bilinear",
          "control": "P_postblock", "est": 520,
          "what": "a 32x32 constant, sans aucune reduction"},
}


def build_configs() -> list:
    cfgs = []
    for fam, spec in FAMILIES.items():
        for name, (profile, why) in PROFILES.items():
            cfgs.append({
                "id": "%s_%s" % (fam, name), "group": "wave5",
                "test": "a priori sur sigma, %s" % spec["what"],
                "operator": "gaussian", "levels": "plateau",
                "resolution": spec["resolution"], "reduction": spec["reduction"],
                "placement": "post_block", "mask": "all19",
                "blurpool_sigma": None, "constant": False,
                "primary_path": "target", "sigma_profile": profile,
                "controller_builder": BUILDER, "diagnostics": True,
                "rationale": why, "est_seconds": spec["est"],
            })
    return cfgs


def build_control_cells() -> list:
    """Seeds 1-2 of the R32 uniform control, which only has seed 0 so far.

    The ``P`` family's control (``D2G``) already has three seeds; the ``Q``
    family's (``P_postblock``) does not, and comparing a 3-seed mean against a
    single draw is exactly what wave 4 showed to be misleading.
    """
    from scripts.ablation_manifest import build_configs as _w1
    base = next(c for c in _w1() if c["id"] == "P_postblock")
    return [{**base, "seed": s, "cell_id": "P_postblock__seed%d" % s}
            for s in (1, 2)]


def build_cells() -> list:
    return ([{**c, "seed": s, "cell_id": "%s__seed%d" % (c["id"], s)}
             for c in build_configs() for s in SEEDS]
            + build_control_cells())


def assign(cells, n_jobs=4) -> list:
    queues, loads = [[] for _ in range(n_jobs)], [0] * n_jobs
    for c in sorted(cells, key=lambda c: (-c["est_seconds"], c["cell_id"])):
        j = loads.index(min(loads))
        queues[j].append(c)
        loads[j] += c["est_seconds"]
    return queues


def summary() -> dict:
    cells = build_cells()
    return {"wave": 5, "n_cells": len(cells), "seeds": list(SEEDS),
            "controls": {"P": "D2G (3 graines, deja mesure)",
                         "Q": "P_postblock (graine 0 deja mesuree, 1-2 ajoutees ici)"},
            "profiles": {k: v[0] for k, v in PROFILES.items()},
            "budgets": {k: round(sum(v[0]), 2) for k, v in PROFILES.items()},
            "jobs": [{"job": i, "cells": [c["cell_id"] for c in q]}
                     for i, q in enumerate(assign(cells))]}


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=2))
