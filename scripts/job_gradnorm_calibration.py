"""Phase 0 -- characterise the gradient-norm signal the trigger fires on.

**Instrumentation only. No controller, no adaptivity, no behaviour change.**

One cell, seed 0: the recorded fixed-schedule plateau arm
``R32__Gplateau__input_bilinear__all19``, run exactly as the campaign ran it,
with the global gradient-norm logged at **every** optimizer update alongside the
learning rate.  Because nothing is adaptive, this reproduces the recorded arm
(test acc 0.7791 at seed 0) and the trace can be trusted as the signal a
controller would have seen on that trajectory.

Why this run exists
-------------------
The first adaptive run (``adaptive-sigma-20260910-093945``) collapsed sigma to
the floor by epoch 5 and finished at 0.7634, below the 0.7791 fixed schedule it
was meant to improve on.  The cause was the trigger, not the stepper: it fired
141 times out of 142 at a **median gap of exactly 40 updates**, which is
``min_steps``.  ``GradNormTracker`` only resets ``stale`` when the EMA sets a new
best at least ``tol = 1%`` lower; a gradient norm evaluated every update does not
fall 1% per update, so ``stale`` saturates immediately and ``min_steps`` becomes
the entire schedule.

Choosing ``tol``, ``patience``, a window and a check cadence needs the signal
itself, which has never been measured in this project.  This job measures it
once; every candidate setting is then replayed **offline for free** against the
recorded trace, so no further GPU time is spent searching.

What to do with the output
--------------------------
``grad_norm_trace.json`` holds ``[update, epoch, grad_norm, lr]`` for all 11,730
updates.  Replay candidate trigger settings against it and look for a
configuration whose step count and spacing are set by the *plateau condition*
rather than by ``min_steps`` -- the diagnostic being the distribution of gaps
between triggers.  A setting whose median gap equals ``min_steps`` has not
detected anything.

PAIRING
-------
Shared assets are regenerated in-job and re-checked against the campaign's four
pinned sha256 digests; the job aborts before training if any differs.  So this
trace belongs to the same trajectory family as the recorded arms.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.campaign_driver import log, run_job
from scripts.job_adaptive_sigma import assert_paired, build_assets

SEED = 0

CELLS = [
    {"id": "Gplateau_cal", "cell_id": "Gplateau_cal3__seed0", "group": "CAL",
     "seed": SEED, "resolution": "R32", "gaussian": "Gplateau",
     "reduction": "input_bilinear", "mask": "all19", "operator": "gaussian",
     "adaptive": False, "log_grad_norm": True},
]


if __name__ == "__main__":
    work = Path(os.environ.get("CAMPAIGN_WORK", "/kaggle/working"))
    assets = build_assets(work / "_assets_local")
    assert_paired(assets, work / "_pairing")
    os.environ["CAMPAIGN_ASSETS"] = str(assets)
    log("cells: %s  (instrumentation only, no controller)"
        % [c["cell_id"] for c in CELLS])
    run_job(0, CELLS)
