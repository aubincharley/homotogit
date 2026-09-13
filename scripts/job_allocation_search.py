"""Allocation search: is uniform dwell optimal, or is there a better shape?

Three fixed schedules, seed 0, 30 epochs, constant resolution 32, Gaussian at
all 19 sites.  **No controller anywhere** -- these are plain level tables.

Every schedule visits the SAME seven levels in the SAME order and ends with the
same nine terminal epochs at the exact target.  The only thing that differs is
how the 21 pre-bypass epochs are allocated across levels:

    uniform       3,3,3,3,3,3,3     (already recorded: this IS Gplateau, 0.7791)
    front-loaded  6,5,4,2,2,1,1     dwell where the deformation is strongest
    back-loaded   1,1,2,2,4,5,6     dwell near the target
    hourglass     5,3,2,1,2,3,5     dwell at both ends, rush the middle

Why this and not another controller
-----------------------------------
Six adaptive runs across two axes all landed strictly between plain and the
fixed schedule.  Each of them changed *two* things at once -- which levels were
visited and how long each was held -- because the trigger set the timing while
the stepper set the levels and nothing enforced

    (number of steps) x (mean step) ~ (path length).

Fixing the level set removes that confound, leaving allocation as the single
free variable.

The decisive logic: **the best any trigger can do is find a good allocation.**
A realised adaptive schedule has to be frozen and re-run to be reportable
anyway, so it is ultimately a fixed allocation.  If none of these beats uniform,
no trigger can, and the adaptive line closes.  If one does, its shape is the
target for a controller to aim at.

Paired controls, recorded at seed 0:

    R32__Gnone__input_bilinear__all19       plain      0.7513
    R32__Gplateau__input_bilinear__all19    uniform    0.7791
                                            (reproduced 0.7797 / 0.7794)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.campaign_driver import log, run_job
from scripts.job_adaptive_sigma import assert_paired, build_assets

SEED, EPOCHS, BYPASS_FROM = 0, 30, 21
LEVELS = [1.00, 0.85, 0.70, 0.60, 0.50, 0.40, 0.30]

ALLOCATIONS = {
    "front": [6, 5, 4, 2, 2, 1, 1],
    "back":  [1, 1, 2, 2, 4, 5, 6],
    "hour":  [5, 3, 2, 1, 2, 3, 5],
}


def table(dwell):
    """Expand a dwell allocation into a 30-entry per-epoch level table."""
    assert sum(dwell) == BYPASS_FROM, (dwell, sum(dwell))
    assert len(dwell) == len(LEVELS)
    t = []
    for lvl, d in zip(LEVELS, dwell):
        t += [lvl] * d
    t += [0.0] * (EPOCHS - BYPASS_FROM)
    assert len(t) == EPOCHS
    return t


# Which allocations to run in this invocation.  Two cells fit two workers in a
# single round, so on a tight quota the run cannot be killed part-way through a
# second round and lose everything.  "front" and "back" are the decisive
# opposites; "hour" is a third shape, deferred to an unconstrained account.
RUN = ("front", "back")

CELLS = [
    {"id": f"alloc_{name}", "cell_id": f"alloc_{name}__seed0", "group": "ALLOC",
     "seed": SEED, "resolution": "R32", "gaussian": "Gplateau",
     "operator": "gaussian", "reduction": "input_bilinear", "mask": "all19",
     "adaptive": False, "gaussian_table": table(d), "dwell": d}
    for name, d in ALLOCATIONS.items() if name in RUN
]


if __name__ == "__main__":
    work = Path(os.environ.get("CAMPAIGN_WORK", "/kaggle/working"))
    assets = build_assets(work / "_assets_local")
    assert_paired(assets, work / "_pairing")
    os.environ["CAMPAIGN_ASSETS"] = str(assets)
    for c in CELLS:
        log("%-22s dwell %s  sum %d" % (c["cell_id"], c["dwell"], sum(c["dwell"])))
    log("uniform (3x7) is already recorded as Gplateau = 0.7791")
    run_job(0, CELLS)
