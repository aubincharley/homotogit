r"""Transfer-gap trigger: step the homotopy when dwelling stops paying off.

Motivation
----------
Convergence-based triggers do not work in this protocol.  Measured on a fixed
``Gplateau`` run: the gradient norm *rises* 1.45x across training (BatchNorm
makes the loss scale invariant, so ``||g|| ~ 1/||w||`` and weight decay inflates
it), its scale-invariant correction ``||g||.||w||`` rises 1.57x, gradient cosine
is structureless, and ``||dtheta||/||theta||`` merely tracks the cosine learning
rate.  Every plateau test on those signals fires at its own structural minimum,
so the hyperparameters become the schedule.

The transfer gap does have structure.  With ``L_cur`` the loss under the
configuration actually being trained and ``L_tgt`` the loss at the target
endpoint (sigma = 0), both on a fixed probe,

    gap(t) = L_tgt(theta_t) - L_cur(theta_t)  >= 0,

measured on the same fixed ``Gplateau`` run, *within* a stage of constant sigma:

    sigma 1.00   1.67 -> 2.36 -> 2.97     gap GROWS
    sigma 0.85   2.77 -> 3.64 -> 4.51     gap GROWS
    sigma 0.70   3.53 -> 3.77 -> 4.94     gap GROWS
    sigma 0.50   1.34 -> 1.19 -> 1.11     gap shrinks
    sigma 0.40  0.214 ->0.179 ->0.176     gap shrinks, flattening
    sigma 0.30 0.0091 ->0.0058 ->0.0050   gap shrinks, flattening

There is a sign change around sigma = 0.5.  A growing gap means further training
at this level is making the model *better on the easy problem and worse on the
target* -- it is specialising to a deformation it is about to leave.  That is
exactly when to step.  A shrinking gap means dwelling still transfers, so stay.

So the criterion is not "has the corrector converged" but **"is dwelling here
still buying progress on the target objective"** -- a marginal-value rule, which
remains well posed even though nothing converges in this budget.

    fire  <=>  (gap[-1-W] - gap[-1]) / |gap[-1-W]|  <  eps      (not shrinking)

with ``eps = 0`` the pure sign test.  The rule is scale free in the gap, which
matters because the gap spans three orders of magnitude along the path.

Caveat carried deliberately
---------------------------
The gap conflates genuine function specialisation with **BatchNorm statistic
mismatch**: the target path is evaluated with running statistics accumulated
under sigma > 0.  Both terms are real costs of sitting at the wrong level and
both vanish as sigma -> 0, but they are not separated here.  Recalibrating BN
before the target evaluation would isolate the function term; that is a distinct
measurement and is not what this trigger uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GapTriggerConfig:
    cadence: int = 100          # updates between gap measurements
    window: int = 3             # W, in measurements
    eps: float = 0.0            # relative-shrink threshold; 0 = pure sign test
    patience: int = 2           # consecutive measurements failing to shrink
    blackout: int = 2           # measurements discarded after a step
    min_dwell_updates: int = 391      # never step twice within one epoch
    max_dwell_updates: int = 3 * 391  # deadline per stage; must be > 0


class TransferGapTracker:
    """Fires when the transfer gap stops shrinking.

    Pure and stateful: consumes scalars, returns a decision.  No model, data or
    optimizer, so it is unit testable without a training run.
    """

    def __init__(self, cfg: GapTriggerConfig | None = None):
        self.cfg = cfg or GapTriggerConfig()
        if self.cfg.max_dwell_updates <= 0:
            raise ValueError(
                "max_dwell_updates must be > 0: a marginal-value test alone "
                "does not guarantee the target endpoint is reached")
        if self.cfg.min_dwell_updates > self.cfg.max_dwell_updates:
            raise ValueError("min_dwell must not exceed max_dwell")
        self._stage_start = 0
        self.reset_stage(0)

    def reset_stage(self, update: int):
        self._stage_start = int(update)
        self._hist: list = []
        self._seen = 0
        self._stale = 0
        self.reason = None

    def observe(self, update: int, gap: float):
        """Record one gap measurement.  Call every ``cadence`` updates."""
        self._seen += 1
        if self._seen <= self.cfg.blackout:
            return                      # the step itself jumps the gap
        self._hist.append(float(gap))
        if len(self._hist) > self.cfg.window:
            prev = self._hist[-1 - self.cfg.window]
            shrink = (prev - self._hist[-1]) / max(abs(prev), 1e-30)
            self._stale = self._stale + 1 if shrink < self.cfg.eps else 0
        else:
            self._stale = 0

    def should_step(self, update: int) -> bool:
        dwell = int(update) - self._stage_start
        if dwell >= self.cfg.max_dwell_updates:
            self.reason = "deadline"
            return True
        if dwell < self.cfg.min_dwell_updates:
            return False
        if self._stale >= self.cfg.patience:
            self.reason = "gap_not_shrinking"
            return True
        return False

    def state_dict(self) -> dict:
        return {"stage_start": self._stage_start, "hist": list(self._hist),
                "seen": self._seen, "stale": self._stale, "reason": self.reason}

    def load_state_dict(self, d: dict):
        self._stage_start = d["stage_start"]; self._hist = list(d["hist"])
        self._seen = d["seen"]; self._stale = d["stale"]; self.reason = d["reason"]

    def describe(self) -> dict:
        c = self.cfg
        return {"kind": "transfer_gap_marginal_value",
                "rule": ("fire when (gap[-1-W]-gap[-1])/|gap[-1-W]| < eps, i.e. "
                         "dwelling no longer reduces the distance to the target"),
                "signal": "L_target(theta) - L_current(theta) on the fixed probe",
                "cadence": c.cadence, "window": c.window, "eps": c.eps,
                "patience": c.patience, "blackout": c.blackout,
                "min_dwell_updates": c.min_dwell_updates,
                "max_dwell_updates": c.max_dwell_updates,
                "caveat": ("the gap conflates function specialisation with BN "
                           "statistic mismatch; both vanish at sigma=0")}
