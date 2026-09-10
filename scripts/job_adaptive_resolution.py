"""Adaptive RESOLUTION driven by the transfer gap, seed 0, 30 epochs.

This is the original question: choose the resolution transition points from the
data rather than from a fixed epoch table.

One cell only.  Both controls already exist in the recorded campaign at seed 0
and the assets here reproduce its pinned digests exactly, so they are valid
paired controls and retraining them would waste GPU:

    R32__Gnone__input_bilinear__all19    plain, constant 32      0.7513
    Rprog__Gnone__input_bilinear         fixed 16/24/32          0.7920

Both are ``Gnone``: no Gaussian anywhere, so **resolution is the only operator
that moves** and a difference is attributable to it.

Why resolution is the easier axis
---------------------------------
Every failure in the sigma work after the first was a step-*magnitude* error --
clipping at ``dmin``, then a misread of ``delta_ref``.  Resolution has no
magnitude: ``r`` is an integer with two transitions, so the step is simply "next
stage" and only the *timing* is free.  There is also no ``dL/dr`` to measure, so
the sensitivity machinery, the ``phi`` coordinate and the underflow floor are
all irrelevant here.  What remains is exactly what the transfer gap supplies.

The trigger
-----------
gap = L_target - L_current on the fixed probe, with target = 32x32 and current =
the resolution actually being trained at.  Fire when the gap stops shrinking
over a window (see continuation.gap_trigger).  Measurements are taken every 100
updates for a smooth signal, but the **decision is taken at the epoch boundary**
because the resolution can only change there -- so measurement, reset and change
stay aligned.

Deadlines are two-sided freedom, not one-sided
----------------------------------------------
``res_force_by = [12, 21]``: the run must be at 24 by epoch 12 and at 32 by
epoch 21.  The fixed ``Rprog`` schedule transitions at 6 and 12, so the trigger
may move **earlier or later** than the fixed table.  Allowing only "earlier"
would have prejudged the answer, and the worst case still leaves nine terminal
epochs at the target resolution.

Pipeline order is unchanged: uint8 -> /255 -> resize -> normalize, with the
channel statistics of the full original training split shared across all
resolutions, so a change in input statistics is never absorbed by refitted
constants.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.campaign_driver import log, run_job
from scripts.job_adaptive_sigma import assert_paired, build_assets

SEED, EPOCHS = 0, 30

CELLS = [
    {"id": "ADAPTRES", "cell_id": "ADAPTRES__seed0", "group": "ADAPTRES",
     "seed": SEED, "resolution": "R32", "gaussian": "Gnone",
     "operator": "none", "reduction": "input_bilinear", "mask": "all19",
     "adaptive": False,                      # sigma is NOT walked
     "adaptive_resolution": True,            # resolution IS walked
     "trigger": "gap",
     "res_stages": [16, 24, 32],
     "res_force_by": [12, 21],
     "adaptive_params": {"gap": {
         "cadence": 100, "window": 3,
         # eps must be strictly positive: with eps = 0 a perfectly FLAT gap
         # gives shrink = 0 and `0 < 0` is false, so "stopped shrinking" would
         # never fire -- only a growing gap would.  0.005 means "shrinking by
         # less than 0.5% per window counts as stopped".
         "eps": 0.005, "patience": 2, "blackout": 2,
         "min_dwell_updates": 391,          # >= one epoch between transitions
         # res_force_by is the single deadline authority, so the trigger's own
         # per-stage cap is set never to bind.
         "max_dwell_updates": 30 * 391}},
     },
]


if __name__ == "__main__":
    work = Path(os.environ.get("CAMPAIGN_WORK", "/kaggle/working"))
    assets = build_assets(work / "_assets_local")
    assert_paired(assets, work / "_pairing")
    os.environ["CAMPAIGN_ASSETS"] = str(assets)
    log("adaptive RESOLUTION, stages %s, force_by %s, trigger=transfer_gap"
        % (CELLS[0]["res_stages"], CELLS[0]["res_force_by"]))
    log("paired controls (recorded): plain 0.7513, fixed Rprog 0.7920")
    run_job(0, CELLS)
