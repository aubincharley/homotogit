"""Adaptive predictor-corrector continuation over the per-site widths.

The fixed-table controller in :mod:`continuation.campaign_ops` decides the whole
path before the first update.  This module instead *walks* the path: train at a
fixed sigma until the corrector has converged, measure how sensitive the loss is
to each site's width, take a measured step, repeat.

Why the step is not a gradient descent on sigma
-----------------------------------------------
The obvious rule -- move sigma so as to decrease the loss -- does **not**
terminate.  It makes sigma a free variable in a joint minimisation
``min_{theta,sigma} L(theta, sigma)``, whose solution has no reason to sit at
sigma = 0: feature smoothing acts as a regulariser, so ``dL/dsigma`` can be
negative and drive sigma *away* from the target.  That is the failure
``GeometricSigmaSchedule`` already refuses to permit -- a schedule that only
decays does not arrive at the endpoint by itself -- and it forfeits the exact
identity endpoint that makes this a homotopy rather than a learned blur layer.

So the **direction is fixed by construction** (sigma only ever decreases) and
only the **magnitude** is adaptive.  ``|dL/dsigma_l|`` is exactly the right
quantity for the magnitude: it says how fast the objective changes along the
path, so a large sensitivity means a stiff region and a small step.

    dsigma_l = -clip( delta_ref * g_ref / (|dL/dsigma_l| + eps), dmin, dmax )

with ``g_ref`` the **median** sensitivity across currently-filtered sites.  The
normalisation is global rather than per-site on purpose: the raw magnitudes
differ across depth by orders of magnitude (a stage-1 map holds 16384 elements
against stage-3's 4096, and deep sites sit nearer the loss), so an un-normalised
rule would anneal one stage instantly and freeze another -- but normalising each
site by *its own* scale would drive every ratio to one and collapse back to a
uniform schedule, destroying the very signal being measured.  Dividing by a
common ``g_ref`` fixes the units while preserving the relative structure across
sites: a site of median sensitivity takes the reference step.

Termination
-----------
Adaptivity must not be allowed to miss the target objective.  ``K`` terminal
epochs are reserved at exactly sigma = 0, and a deadline epoch ``D`` is set: if
any site is still positive at ``D``, :meth:`SensitivityStepper.ramp` overrides
the adaptive rule with a linear ramp to zero over the remaining epochs and the
run records ``deadline_fired``.  A run that fires the deadline is partly a fixed
schedule and must be reported as such.

Reproducibility
---------------
An adaptive run is not comparable to anything unless it can be replayed, so the
controller records the **realised** ``[epochs][19]`` table.  Feeding that table
back through the ordinary fixed-table ``SiteController`` reproduces the run, and
it carries the same sha256 provenance as any other arm.
"""
from __future__ import annotations

import math
import statistics

import torch
import torch.nn.functional as F

from .campaign_ops import LEVEL_DECIMALS, N_SITES, SiteController
from .transforms.gaussian import blur_with_sigma_grad

#: Snap to the exact identity below this width.  With K=4 off-centre taps per
#: axis and two separable passes the operator differs from the identity by at
#: most ``32 * exp(-1/(2 sigma^2)) * |h|_inf``, which at sigma = 0.12 is
#: ``2.7e-14 * |h|_inf`` -- several orders below one float32 ULP.  So the snap is
#: not an approximation of the homotopy; it relabels a point the homotopy has
#: already reached in this arithmetic.  General form, should ``truncate`` or
#: ``sigma_max`` change: the floor is ``sigma < 1/sqrt(2*ln(32/eps_float32))``.
#: Recorded per site, never silent.
SIGMA_FLOOR = 0.12


class GradNormTracker:
    """Detects that the corrector has converged, by plateau rather than threshold.

    Under SGD the gradient norm does not vanish -- it settles onto a noise floor
    set by the minibatch variance and the learning rate -- so an absolute
    threshold either fires on the first step or never fires at all.  This tracks
    an EMA and declares convergence when it stops *improving* by more than a
    relative tolerance for ``patience`` consecutive updates, which needs no
    problem-specific constant.
    """

    def __init__(self, beta: float = 0.9, tol: float = 0.01, patience: int = 30,
                 min_steps: int = 40, max_steps: int = 10 ** 9):
        if not (0.0 < beta < 1.0):
            raise ValueError("beta must lie in (0,1), got %r" % (beta,))
        if tol < 0:
            raise ValueError("tol must be >= 0, got %r" % (tol,))
        if min_steps > max_steps:
            raise ValueError("min_steps must be <= max_steps")
        self.beta, self.tol = float(beta), float(tol)
        self.patience = int(patience)
        self.min_steps, self.max_steps = int(min_steps), int(max_steps)
        self.reset_stage()

    def reset_stage(self):
        self.ema = None
        self.best = None
        self.stale = 0
        self.steps = 0

    def update(self, grad_norm: float):
        g = float(grad_norm)
        self.steps += 1
        self.ema = g if self.ema is None else self.beta * self.ema + (1 - self.beta) * g
        if self.best is None or self.ema < (1.0 - self.tol) * self.best:
            self.best, self.stale = self.ema, 0
        else:
            self.stale += 1
        return self.ema

    def should_step(self) -> bool:
        if self.steps < self.min_steps:
            return False
        if self.steps >= self.max_steps:
            return True
        return self.stale >= self.patience

    def describe(self) -> dict:
        return {"kind": "grad_norm_ema_plateau", "beta": self.beta, "tol": self.tol,
                "patience": self.patience, "min_steps": self.min_steps,
                "max_steps": self.max_steps,
                "rationale": ("SGD gradient norms plateau at a noise floor, so "
                              "convergence is relative-plateau, not a threshold")}


def phi_of_sigma(sigma: float) -> float:
    """Off-centre tap mass ``phi = exp(-1/(2 sigma^2))``: the blur the discrete
    operator actually applies, and the variable in which its sensitivity is
    conditioned.

    Measured on this operator (radius 4, unit spacing), over sigma in
    [0.1, 1.0]:  ``|dL/dsigma|`` spans a condition number of ~9e17 while
    ``|dL/dphi|`` spans ~18, and ``|dL/dphi|`` is flat to 1% below sigma = 0.25.
    Note this is **not** heat time ``a = sigma^2/2`` -- that is the continuum
    answer, and it is wrong here because the support is truncated at radius 4 and
    the grid spacing is fixed at one pixel.
    """
    if sigma <= 0:
        return 0.0
    return math.exp(-1.0 / (2.0 * float(sigma) ** 2))


class SensitivityStepper:
    """Turns per-site sensitivities into a monotone step toward sigma = 0.

    Three things make this stable, and each of them is a correction to the
    obvious rule ``dsigma ~ 1/|dL/dsigma|``:

    * **Work in phi.**  ``|dL/dsigma|`` collapses super-exponentially as sigma
      falls (17 orders between sigma=0.3 and sigma=0.12 on this operator), so a
      rule that divides by it has no signal in the lower half of the path.
      :func:`phi_of_sigma` fixes the conditioning.
    * **Bound the spread with ``kappa``.**  Dividing by a sensitivity is positive
      feedback: a site that anneals faster gets a smaller gradient, hence a
      larger step, hence anneals faster still.  Left unbounded that produces a
      bang-bang front in which sites collapse one at a time in whatever order
      they started.  ``kappa`` is the only knob deciding how much per-site
      freedom the controller has, and it must be reported.
    * **Make the base step schedule-aware.**  ``base = sigma_l / stages_left``
      lands every site on zero on schedule *without consulting the gradient*, so
      the sensitivity only ever redistributes rates around a sane default.  This
      is also what keeps the tail sensible where the gradient has no authority.

    A per-site normalisation (each ``g_l`` by its own scale) would be wrong: it
    drives every ratio to one, collapses back to a uniform schedule, and destroys
    the very signal being measured.  The median is pooled across sites on purpose.
    """

    def __init__(self, delta_ref: float = 1.0, dmin: float = 0.03,
                 dmax: float = 0.45, eps: float = 1e-30, kappa: float = 3.0,
                 sigma_floor: float = SIGMA_FLOOR, sigma_max: float = 1.0,
                 default_stages: int = 6, allow_early_zero: bool = False):
        if not (0 < dmin <= dmax):
            raise ValueError("need 0 < dmin <= dmax")
        if kappa < 1.0:
            raise ValueError("kappa must be >= 1 (1 disables per-site freedom)")
        if dmin <= 10 ** (-LEVEL_DECIMALS):
            raise ValueError("dmin must exceed the table quantum or a step can "
                             "round to no change and the run stalls")
        self.delta_ref, self.dmin, self.dmax = float(delta_ref), float(dmin), float(dmax)
        self.eps, self.kappa = float(eps), float(kappa)
        self.sigma_floor, self.sigma_max = float(sigma_floor), float(sigma_max)
        self.default_stages = int(default_stages)
        # Arms must be matched on updates spent at the exact target objective:
        # an adaptive arm that reaches sigma=0 early would simply be training
        # longer on the target than its fixed controls, which alone could
        # explain any accuracy difference.  Sites are therefore held at the
        # floor until the final scheduled stage unless this is turned off.
        self.allow_early_zero = bool(allow_early_zero)
        self.last_ratios = None
        self.probe_invalid = 0

    def _phi_grads(self, row, grads, active):
        """``|dL/dphi_l|`` from the measured ``dL/dsigma_l``."""
        out = {}
        for l in active:
            g = float(grads[l])
            s = row[l]
            phi = phi_of_sigma(s)
            dphi_ds = phi / (s ** 3) if s > 0 else 0.0
            if not math.isfinite(g) or dphi_ds <= 0.0:
                out[l] = None                      # unusable; falls back to base
            else:
                out[l] = abs(g) / dphi_ds
        return out

    def step(self, row, grads, sites, stages_left=None) -> list:
        """One predictor step.  ``grads[l]`` is ``dL/dsigma_l`` (sign ignored)."""
        active = [l for l in sites if row[l] > 0.0]
        out = list(row)
        if not active:
            self.last_ratios = None
            return out
        n = max(int(stages_left if stages_left else self.default_stages), 1)
        gphi = self._phi_grads(row, grads, active)
        usable = [v for v in gphi.values() if v is not None and v > 0]
        if len(usable) < len(active):
            self.probe_invalid += 1
        gref = statistics.median(usable) if usable else 0.0
        ratios = {}
        for l in active:
            g = gphi[l]
            if gref > 0 and g is not None and g > 0:
                r = gref / (g + self.eps)
            else:
                r = 1.0                            # no signal -> schedule only
            r = min(max(r, 1.0 / self.kappa), self.kappa)
            ratios[l] = r
            d = min(max((row[l] / n) * self.delta_ref * r, self.dmin), self.dmax)
            v = row[l] - d                         # direction fixed: always down
            if v < self.sigma_floor:
                if n <= 1 or self.allow_early_zero:
                    out[l] = 0.0
                else:
                    # park at the floor -- but never above where we already are,
                    # or a site already below the floor would be pushed *up*
                    out[l] = min(round(self.sigma_floor, LEVEL_DECIMALS), row[l])
            else:
                out[l] = round(v, LEVEL_DECIMALS)
        self.last_ratios = ratios
        # monotonicity is a contract, not a hope
        for l in sites:
            assert out[l] <= row[l] + 1e-12, "sigma increased at site %d" % l
            assert math.isfinite(out[l]) and out[l] >= 0.0
        return out

    def ramp(self, row, stages_left: int, sites) -> list:
        """Forced linear descent to zero, used once the deadline has passed."""
        out = list(row)
        if stages_left <= 1:
            for l in sites:
                out[l] = 0.0
            return out
        for l in sites:
            v = row[l] - row[l] / float(stages_left)
            out[l] = 0.0 if v < self.sigma_floor else round(v, LEVEL_DECIMALS)
        return out

    def describe(self) -> dict:
        return {"kind": "sensitivity_normalised_in_phi",
                "rule": ("dsigma_l = -clip( (sigma_l/stages_left) * delta_ref * "
                         "clip(median_l|dL/dphi_l| / |dL/dphi_l|, 1/kappa, kappa), "
                         "dmin, dmax); direction fixed toward 0"),
                "variable": "phi = exp(-1/(2 sigma^2)); NOT heat time sigma^2/2",
                "delta_ref": self.delta_ref, "dmin": self.dmin, "dmax": self.dmax,
                "kappa": self.kappa, "eps": self.eps,
                "sigma_floor": self.sigma_floor,
                "default_stages": self.default_stages,
                "allow_early_zero": self.allow_early_zero,
                "exposure_note": ("sites are held at sigma_floor until the final "
                                  "scheduled stage so every arm spends the same "
                                  "number of updates at the exact target objective"),
                "probe_invalid_count": self.probe_invalid,
                "normalisation": ("global median across filtered sites, in phi -- a "
                                  "per-site normalisation would collapse to uniform; "
                                  "kappa bounds the positive feedback of dividing by "
                                  "a sensitivity that itself falls with sigma")}


class AdaptiveSiteController(SiteController):
    """A ``SiteController`` whose row is walked, not read from a table.

    The parent's level table is left as ``None``; :attr:`row` is the live state,
    and every epoch's row is appended to :attr:`realised` so the run can be
    replayed as an ordinary fixed-table arm.
    """

    def __init__(self, sigma_init, sites=None, resolution_by_epoch=None,
                 reduction="input_bilinear", tracker=None, stepper=None):
        super().__init__("gaussian", sites=sites,
                         resolution_by_epoch=resolution_by_epoch,
                         reduction=reduction)
        row = [float(v) for v in sigma_init]
        if len(row) != N_SITES:
            raise ValueError("sigma_init has %d entries, need %d" % (len(row), N_SITES))
        smax = self.gauss.sigma_max
        for l, v in enumerate(row):
            if v < 0 or v > smax + 1e-12:
                raise ValueError("sigma_init[%d]=%g outside [0, %g]" % (l, v, smax))
        self.row = row
        self.realised = []
        self.tracker = tracker or GradNormTracker()
        self.stepper = stepper or SensitivityStepper(sigma_max=smax)
        self.probe = None            # list of 19 tensors while measuring
        self.n_steps = 0
        self.deadline_fired = False
        self.snapped_at = [None] * N_SITES

    # -- the row is live, so set_epoch only records it --------------------

    def set_epoch(self, e: int):
        self.resolution = self._resolution_at(e)
        self.q = self.q_for(self.resolution)
        while len(self.realised) <= int(e):
            self.realised.append(list(self.row))
        self.realised[int(e)] = list(self.row)
        return self.value

    # -- sensitivity probe ------------------------------------------------

    def apply_at(self, site: int, h: torch.Tensor) -> torch.Tensor:
        if self.probe is None:
            return super().apply_at(site, h)
        if self.bypass_all or site not in self.sites:
            return h
        s = self.probe[site]
        if s is None:
            return h                                   # exact bypass at zero
        return blur_with_sigma_grad(h, self.q[site] * s, self.gauss.radius)

    @torch.enable_grad()
    def measure(self, model, pipe, images, labels) -> list:
        """``dL/dsigma_l`` at every filtered site, on a fixed probe batch.

        Evaluated in ``eval()`` mode with BatchNorm buffers untouched and the
        prior mode restored -- the convention every other measurement in this
        repository follows, and the one that makes successive sensitivities
        comparable.  ``torch.autograd.grad`` is used rather than ``backward``, so
        no ``.grad`` on the model is touched.
        """
        prev_mode = model.training
        prev_probe = self.probe
        model.eval()
        sig = [None] * N_SITES
        for l in self.sites:
            if self.row[l] > 0.0:
                sig[l] = torch.tensor(self.row[l], dtype=torch.float32,
                                      device=images.device, requires_grad=True)
        self.probe = sig
        try:
            leaves = [s for s in sig if s is not None]
            if not leaves:
                return [0.0] * N_SITES
            loss = F.cross_entropy(model(pipe(images, 0.0)), labels)
            grads = torch.autograd.grad(loss, leaves, allow_unused=True)
        finally:
            self.probe = prev_probe
            if prev_mode:
                model.train()
        out, i = [0.0] * N_SITES, 0
        for l in range(N_SITES):
            if sig[l] is not None:
                g = grads[i]
                out[l] = 0.0 if g is None else float(g)
                i += 1
        return out

    # -- the predictor step -----------------------------------------------

    def advance(self, grads, stages_left=None, force_ramp: bool = False) -> dict:
        before = list(self.row)
        if force_ramp:
            self.deadline_fired = True
            self.row = self.stepper.ramp(self.row, stages_left or 1, self.sites)
        else:
            self.row = self.stepper.step(self.row, grads, self.sites,
                                         stages_left=stages_left)
        self.n_steps += 1
        for l in self.sites:
            if before[l] > 0.0 and self.row[l] == 0.0 and self.snapped_at[l] is None:
                self.snapped_at[l] = self.n_steps
        return {"step": self.n_steps, "before": before, "after": list(self.row),
                "grads": [float(g) for g in grads], "forced_ramp": bool(force_ramp)}

    def all_zero(self) -> bool:
        return all(self.row[l] <= 0.0 for l in self.sites)

    def describe(self) -> dict:
        d = super().describe()
        d.update({"controller": "adaptive_predictor_corrector",
                  "predictor": "warm start only (theta carried unchanged)",
                  "trigger": self.tracker.describe(),
                  "stepper": self.stepper.describe(),
                  "realised_table": self.realised,
                  "n_predictor_steps": self.n_steps,
                  "deadline_fired": self.deadline_fired,
                  "snapped_at_step": self.snapped_at,
                  "sigma_floor": self.stepper.sigma_floor})
        return d
