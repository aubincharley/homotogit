"""Adaptive continuation driven by the TRANSFER GAP, seed 0.

One cell, ResNet-20 + BN, 50k/10k CIFAR-10, 30 epochs, constant resolution 32,
Gaussian at all 19 sites.  Sigma is walked adaptively; the trigger is the
marginal-value rule of :mod:`continuation.gap_trigger`:

    step when  (gap[-1-W] - gap[-1]) / |gap[-1-W]|  <  eps,

i.e. when dwelling at the current sigma no longer reduces the distance to the
target objective.  gap = L_target - L_current on the fixed 500-image probe.

Why this trigger and not the gradient norm
------------------------------------------
Measured on the fixed Gplateau arm: ||g|| RISES 1.45x over training, its
scale-invariant form ||g||.||w|| rises 1.57x, gradient cosine has no structure,
and ||dtheta||/||theta|| just tracks the cosine LR.  Every plateau test on those
fired at its own structural minimum, so hyperparameters became the schedule --
the first adaptive run collapsed sigma to the floor by epoch 5 and scored 0.7634
against 0.7791 for the fixed schedule.

The gap does have structure, with a sign change near sigma = 0.5: above it the
gap GROWS while dwelling (the model specialises to a deformation it is about to
leave), below it the gap shrinks.  So the trigger should move quickly through
the high-sigma levels and dwell at the low ones -- the opposite of the fixed
schedule's uniform three-epoch dwell.

Parameters, and the one judgement call
--------------------------------------
The stepper is theirs, unchanged in form (`SensitivityStepper`), but
the stepper keeps its own defaults.  An earlier attempt set ``delta_ref = 0.10``
on the belief that it was the step size.  It is not: the step is
``sigma/n * delta_ref * r`` with ``n = default_stages = 6``, so 0.10 produced
0.017 -- below ``dmin`` -- and every step clipped to the floor of 0.02.  Sigma
crawled 1.00 -> 0.84 over seventeen epochs and the run scored 0.7539.

Note also that ``r`` is 1.0 wherever there is no sensitivity signal, and
|dL/dsigma| is measured to be ~0 at most sites, so the per-site scaling is
largely inert and the step is effectively ``sigma/6``.  Untouched, that is a
geometric walk 1.00 -> 0.83 -> 0.69 -> ... -> 0.13 over eleven steps -- matching
the eleven fires the trigger produces on the recorded gap trace.

``min_dwell = 391`` = one epoch, so the trigger cannot fire twice in an epoch.
Deadline: linear ramp from epoch 18, exactly zero by 21, nine terminal epochs on
the exact target objective.

Paired controls (recorded campaign, seed 0)
-------------------------------------------
    R32__Gnone     plain            0.7513
    R32__Gplateau  fixed schedule   0.7791   (reproduced 0.7797 / 0.7794 / 0.7833)
    adaptive, grad-norm trigger     0.7634

Assets are regenerated in-job and re-checked against the campaign's four pinned
sha256 digests; the job aborts before training if any differs.

Note on the instrumentation: this cell evaluates two probe paths every 100
updates.  The one previous run that did that (signals-cal3) landed at 0.7833
rather than the 0.7791-0.7797 of runs without mid-epoch evaluation -- 0.39 pp,
about four times the observed spread of the other three.  No state leak was
found (evaluate restores row/resolution/bypass and recomputes q; ``value`` is
unused by ``apply_at``), so the likely cause is float non-determinism amplified
by interleaved eval-mode passes at a different batch shape.  Treat comparisons
at the 0.1 pp level with that in mind.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.campaign_driver import log, run_job
from scripts.job_adaptive_sigma import assert_paired, build_assets

SEED = 0

ADAPTIVE_PARAMS = {
    "sigma_init": 1.0,
    "sens_batch": 256,
    "ramp_from": 18,
    "zero_by": 21,
    # Stepper left at ITS OWN DEFAULTS (delta_ref 1.0, dmin 0.03, dmax 0.45,
    # kappa 3.0, default_stages 6).  A previous run overrode delta_ref to 0.10
    # believing it was the step size; it is not -- the step is
    # sigma/n * delta_ref * r, so 0.10 gave 1.0/6*0.10 = 0.017, below dmin, and
    # every step clipped to 0.02.  Untouched, the same rule gives a clean
    # geometric walk 1.00 -> 0.83 -> 0.69 -> ... -> 0.13 in eleven steps, which
    # is what the trigger actually produces.
    "stepper": {"sigma_max": 1.0},
    "gap": {"cadence": 100, "window": 3, "eps": 0.0, "patience": 2,
            "blackout": 2, "min_dwell_updates": 391,
            "max_dwell_updates": 3 * 391},
}

CELLS = [
    {"id": "ADAPTGAP", "cell_id": "ADAPTGAP2__seed0", "group": "ADAPTGAP",
     "seed": SEED, "resolution": "R32", "gaussian": "Gplateau",
     "reduction": "input_bilinear", "mask": "all19", "operator": "gaussian",
     "adaptive": True, "trigger": "gap", "adaptive_params": ADAPTIVE_PARAMS},
]


if __name__ == "__main__":
    work = Path(os.environ.get("CAMPAIGN_WORK", "/kaggle/working"))
    assets = build_assets(work / "_assets_local")
    assert_paired(assets, work / "_pairing")
    os.environ["CAMPAIGN_ASSETS"] = str(assets)
    log("cells: %s  trigger=transfer_gap" % [c["cell_id"] for c in CELLS])
    run_job(0, CELLS)
