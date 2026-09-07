"""lambda(t): the closed loop, and the open-loop arms it has to be told apart from.

The controller is a multiplicative integrator on the drift error, run in log-space:

    u <- u + clip(beta * (d_ema - d*(t)))          lambda = clip(exp(u))

with d*(t) = D_max * t/T. Drifting faster than the schedule tightens the anchor,
slower loosens it. In log-space because lambda spans four decades and an additive
step would be a 4%% correction at one end and a 400x one at the other.

Three things here are not decoration, they are what makes the resulting lambda
trace interpretable.

d_ema, not d. The raw d is deterministic given w, but it is not smooth in t, and
an integrator fed the raw signal produces lambda chatter. That chatter is not
merely cosmetic: the open-loop replay arm (§8.1) replays this exact trace, and if
most of its structure is high-frequency noise the ablation compares the method
against its own jitter rather than against its schedule.

u0 = log lambda_max. The run starts maximally anchored and is allowed to relax as
the drift budget opens up, rather than starting free and being reined in. The
first few hundred steps are the initial transient, where K_0 is not yet a
meaningful reference, so the loop is held shut there rather than trusted.

TWO LOOPS. The integrator above is the LEVEL, u. On top of it sits an
allocation a_g, one offset per parameter group, with

    log lambda_g = u + a_g,      sum_g a_g = 0

so the allocation can only ever redistribute the anchor between groups, never
strengthen or weaken it overall -- that remains the level's job alone, and the
level's transfer function is therefore unchanged by anything the allocation does.
The allocation runs an order of magnitude slower (Allocator, below) and is driven
by per-group tension rather than by per-group drift. With anchor_alloc off it
holds a_g = 0 and this file behaves exactly as it did before groups existed.

The monitors. A controller that has silently lost authority -- the sign of
dd/dlambda flipped, or the drift signal went flat -- keeps producing a plausible
lambda trace and a plausible loss curve. Both conditions are detected, logged, and
in the sign case acted on, because neither is visible after the fact.
"""
import json
import math

MODES = ("off", "const", "adaptive", "replay", "exp", "critical")

# Windows for the two monitors, in probes. Fixed rather than configurable: they
# are properties of the diagnostic, not knobs of the experiment.
_SIGN_WINDOW = 10
_SIGN_MIN_CORRELATION = 0.5
_SIGN_MIN_EXCITATION = 0.05      # as a fraction of the rate limit
_SIGN_CONSECUTIVE = 2
_SATURATION_WINDOW = 20
_SATURATION_DRIFT = 1e-3
_SATURATION_LAMBDA = 2.0


def bounds(cfg, total_steps, min_w0_sq=1.0):
    """[lambda_min, lambda_max] from §6's stability bracket 1/(eta T) << nu << 1/eta.

    The bracket is on the RAW coefficient nu_g = lambda_g / ||w0_g||^2, which is
    what actually multiplies (w - w0) in the update. With gradients off the pull
    is (1 - eta*nu)^k: below 1/(eta T) the anchor does not close a meaningful
    fraction of the distance to w0 over the whole run and is inert; at 1/eta it
    closes all of it in a single step. eta here is the BASE lr, since the bracket
    is about the run as a whole.

    Converting back to lambda multiplies through by ||w0_g||^2, and the bracket
    has to hold for EVERY group, so the binding one is the smallest:

        lambda_min = min_g ||w0_g||^2 / (eta T)
        lambda_max = min_g ||w0_g||^2 / eta

    min rather than max at both ends because a bracket wide enough for the
    largest group would put the smallest one past its overshoot bound, and
    overshoot is a silent oscillation while inertness is a visible clip.

    lambda_max carries a safety factor. At exactly min||w0||^2/eta the pull factor
    eta_k*lambda/||w0||^2 reaches 1.0 for the smallest group the moment warmup
    finishes, which lands its weights on w0 and trips anchor.pull's overshoot
    assertion. Set anchor_lambda_max_frac to 1.0 for §6 literally.

    min_w0_sq defaults to 1 so an un-normalised anchor (gate 5) gets the original
    bracket back unchanged.
    """
    base_lr = float(cfg["lr"])
    lo = min_w0_sq / (base_lr * max(total_steps, 1))
    hi = float(cfg["anchor_lambda_max_frac"]) * min_w0_sq / base_lr
    return lo, hi


class DriftController:
    """One per run. Produces lambda for every step and consumes d at every probe."""

    def __init__(self, cfg, total_steps, min_w0_sq=1.0):
        self.mode = cfg["anchor_mode"]
        if self.mode not in MODES:
            raise SystemExit(f"!! unknown anchor_mode {self.mode!r}: "
                             f"pick from {', '.join(MODES)}")
        self.total_steps = max(int(total_steps), 1)
        self.lambda_min, self.lambda_max = bounds(cfg, self.total_steps, min_w0_sq)
        self.beta = float(cfg["anchor_beta"])
        self.dmax = float(cfg["anchor_dmax"])
        self.rho = float(cfg["anchor_rho"])
        self.rate_limit = float(cfg["anchor_rate_limit"])
        self.hold_steps = int(cfg["anchor_hold_steps"])
        self.const_lambda = float(cfg["anchor_lambda"])
        self.exp_c = float(cfg["anchor_exp_c"])
        self.critical_frac = float(cfg["anchor_critical_frac"])

        self.u = math.log(self.lambda_max)
        self.d_ema = None
        self.clip_hits = 0
        self.frozen = False
        self.sign_flip_step = -1
        self.saturation_step = -1
        self._sign_strikes = 0
        self._probes = []            # (step, u, d_ema, target) per observation
        self._replay = (_load_replay(cfg) if self.mode == "replay" else None)

    # -- what the training loop asks for every step ------------------------

    def lam(self, step):
        """lambda for this step. Constant between probes in the adaptive mode."""
        if self.mode == "off":
            return 0.0
        if self.mode == "const":
            return self.const_lambda
        if self.mode == "exp":
            return self.lambda_max * math.exp(-self.exp_c * step / self.total_steps)
        if self.mode == "critical":
            return (self.lambda_max
                    if step < self.critical_frac * self.total_steps else 0.0)
        if self.mode == "replay":
            return _replay_at(self._replay, step)
        return math.exp(self.u)

    def target(self, step):
        """d*(t), linear from 0 to D_max. The homotopy schedule itself."""
        return self.dmax * step / self.total_steps

    # -- what the probe hands back -----------------------------------------

    def observe(self, step, d):
        """Fold one drift measurement in and return the row to log.

        Called on every arm, not just the adaptive one: the open-loop arms need
        the same d and d* columns or the comparison plots cannot be drawn.
        """
        self.d_ema = d if self.d_ema is None else \
            self.rho * self.d_ema + (1.0 - self.rho) * d

        row = {"d_ema": self.d_ema, "d_target": self.target(step),
               "controller_frozen": bool(self.frozen)}

        if self.mode != "adaptive":
            self._probes.append((step, self.u, self.d_ema, self.target(step)))
            row["lam"] = self.lam(step)
            return row

        if step < self.hold_steps:
            row["lam"] = math.exp(self.u)
            row["held"] = True
            self._probes.append((step, self.u, self.d_ema, self.target(step)))
            return row

        if not self.frozen:
            error = self.d_ema - self.target(step)
            delta = max(-self.rate_limit,
                        min(self.rate_limit, self.beta * error))
            proposed = self.u + delta
            clipped = max(math.log(self.lambda_min),
                          min(math.log(self.lambda_max), proposed))
            if clipped != proposed:
                self.clip_hits += 1
            self.u = clipped

        self._probes.append((step, self.u, self.d_ema, self.target(step)))
        self._check_sign(step)
        self._check_saturation(step)
        row["lam"] = math.exp(self.u)
        row["clip_hits"] = self.clip_hits
        slope, correlation, excitation = self._sign_fit()
        row["sign_slope"] = slope
        row["sign_corr"] = correlation
        row["sign_excitation"] = excitation
        row["controller_frozen"] = bool(self.frozen)
        return row

    # -- monitors ----------------------------------------------------------

    def _sign_fit(self):
        """(slope, |correlation|, rms excitation) for the drift response to lambda.

        The regressand is the change in the tracking ERROR, d_ema - d*, not the
        change in d_ema. That correction is not cosmetic. Near equilibrium the
        loop is doing its job, so delta u goes to zero while d_ema keeps climbing
        because d* is climbing -- and a regression of raw delta d_ema on delta u
        there is a regression of one small noise on another. Measured: a plant
        with the CORRECT sign froze at step 17600 on a fitted slope of +0.021.
        Subtracting the schedule removes the part of the movement that is known in
        advance and is nothing to do with lambda.

        Paired with a one-probe lag: the u chosen at probe k shows up at k+1.
        Physically the slope must be negative -- more anchor, less drift.

        Returns nan when lambda was not excited enough to identify anything. That
        is a real blind spot rather than a safe default: while lambda sits pinned
        against a clip, every delta u is identical, there is no variance to
        regress on, and a flipped sign there would go unseen. The clip-hit counter
        is what surfaces that case, and print_report prints it.
        """
        window = self._probes[-(_SIGN_WINDOW + 2):]
        if len(window) < 5:
            return float("nan"), 0.0, 0.0
        xs, ys = [], []
        for (_, u_prev, _, _), (_, u_now, d_now, t_now), (_, _, d_next, t_next) \
                in zip(window, window[1:], window[2:], strict=False):
            xs.append(u_now - u_prev)
            ys.append((d_next - t_next) - (d_now - t_now))

        mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
        var_x = sum((x - mean_x) ** 2 for x in xs)
        var_y = sum((y - mean_y) ** 2 for y in ys)
        excitation = (var_x / len(xs)) ** 0.5
        if excitation < _SIGN_MIN_EXCITATION * self.rate_limit or var_y <= 0.0:
            return float("nan"), 0.0, excitation
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
        return cov / var_x, abs(cov) / (var_x * var_y) ** 0.5, excitation

    def _sign_slope(self):
        return self._sign_fit()[0]

    def _check_sign(self, step):
        """Freeze only on a slope that is positive, identifiable, and persistent.

        All three conditions are needed. Positive is the claim; a correlation
        floor is what makes the slope more than an artefact of two noisy signals;
        and requiring consecutive strikes stops one unlucky window from
        converting the rest of a six-hour run into constant-lambda training,
        which is a far more expensive mistake than reacting a probe late.
        """
        slope, correlation, _ = self._sign_fit()
        if (math.isnan(slope) or slope <= 0.0
                or correlation < _SIGN_MIN_CORRELATION):
            self._sign_strikes = 0
            return
        self._sign_strikes += 1
        if self._sign_strikes < _SIGN_CONSECUTIVE or self.frozen:
            return
        self.frozen = True
        self.sign_flip_step = step
        print(f"  !! CONTROLLER FROZEN at step {step}: d responds POSITIVELY "
              f"to lambda over {_SIGN_CONSECUTIVE} consecutive windows "
              f"(slope {slope:+.4g}, |r| {correlation:.2f}). The feedback sign "
              f"has flipped, so the loop is destabilising rather than "
              f"regulating. lambda is held at {math.exp(self.u):.4g} for the "
              f"rest of the run; everything after this step is constant-lambda "
              f"training.")

    def _check_saturation(self, step):
        """Drift stopped responding while lambda kept moving: the signal is dead.

        Not an error and not something to correct -- it is a finding. But it must
        be recorded, because every step after it is constant-lambda training
        wearing a controller's clothes.
        """
        if self.saturation_step >= 0 or len(self._probes) < _SATURATION_WINDOW:
            return
        window = self._probes[-_SATURATION_WINDOW:]
        drift_span = (max(d for _, _, d, _ in window)
                      - min(d for _, _, d, _ in window))
        u_span = max(u for _, u, _, _ in window) - min(u for _, u, _, _ in window)
        if drift_span < _SATURATION_DRIFT and u_span > math.log(_SATURATION_LAMBDA):
            self.saturation_step = step
            print(f"  !! DRIFT SIGNAL SATURATED at step {step}: d_ema moved "
                  f"{drift_span:.2e} over {_SATURATION_WINDOW} probes while "
                  f"lambda moved {math.exp(u_span):.1f}x. The controller has no "
                  f"authority left; treat the rest of the run as constant lambda.")

    # -- summary and persistence -------------------------------------------

    def summary(self):
        return {"anchor_mode": self.mode,
                "lambda_final": self.lam(self.total_steps - 1),
                "lambda_min": self.lambda_min, "lambda_max": self.lambda_max,
                "d_ema_final": self.d_ema if self.d_ema is not None else float("nan"),
                "clip_hits": self.clip_hits,
                "sign_flip_step": self.sign_flip_step,
                "saturation_step": self.saturation_step}

    def state_dict(self):
        return {"u": self.u, "d_ema": self.d_ema, "clip_hits": self.clip_hits,
                "frozen": self.frozen, "sign_flip_step": self.sign_flip_step,
                "saturation_step": self.saturation_step,
                "sign_strikes": self._sign_strikes, "probes": self._probes}

    def load_state_dict(self, state):
        self.u = state["u"]
        self.d_ema = state["d_ema"]
        self.clip_hits = state["clip_hits"]
        self.frozen = state["frozen"]
        self.sign_flip_step = state["sign_flip_step"]
        self.saturation_step = state["saturation_step"]
        self._sign_strikes = state.get("sign_strikes", 0)
        self._probes = [tuple(row) for row in state["probes"]]

    def describe(self):
        if self.mode == "off":
            return "anchor: off (drift still measured and logged)"
        detail = {"const": f"lambda = {self.const_lambda:g}",
                  "adaptive": (f"beta = {self.beta:g}, D_max = {self.dmax:g}, "
                               f"rho = {self.rho:g}, hold {self.hold_steps} steps"),
                  "exp": f"lambda_max * exp(-{self.exp_c:g} t/T)",
                  "critical": f"lambda_max for t < {self.critical_frac:g} T",
                  "replay": f"{len(self._replay or [])} recorded lambda points",
                  }[self.mode]
        return (f"anchor: {self.mode}, {detail}; "
                f"clip [{self.lambda_min:.3g}, {self.lambda_max:.3g}]")


ALLOC_MODES = ("off", "adaptive")

# Floors, not knobs. tau is a ratio of two norms either of which can be exactly
# zero on the first probe (no backward yet, or w still exactly at w0), and log(0)
# would take the whole allocation to nan in one step.
_TENSION_EPS = 1e-12
_TENSION_FLOOR = 1e-30


class Allocator:
    """a_g: how the drift budget is split across groups. The slow loop.

        log lambda_g = u + a_g,        sum_g a_g = 0

    u is the level and a is the allocation. They are updated on deliberately
    different timescales -- a fires once every anchor_alloc_every probes, the
    level every probe -- and that separation IS the stability argument. The fast
    loop is entitled to treat a as constant while it converges; the slow loop is
    entitled to treat u as converged when it moves. Run them at the same rate and
    neither assumption holds: the two integrators then chase each other, and
    because they are both in log lambda the result is a lambda trace that looks
    like a controller working.

    The signal is TENSION, not per-group drift. For group g,

        tau_g = ||grad_g L|| * ||w0_g||^2 / (lambda_g * ||w_g - w0_g|| + eps)

    the ratio of the gradient force pulling the group away from w0 to the anchor
    force holding it there. tau_g > 1 means the group is anchor-limited: the data
    wants it to move and the anchor is what is stopping it, so it should be given
    more of the budget. The update is a log-space integrator on the deviation of
    log tau_g from its mean over groups, which is what keeps sum_g a_g = 0 an
    invariant rather than something to restore afterwards.

    Why tension and not d_g. The per-group drifts are measured and logged, and
    they are the right thing for the coupling gate, but they make a poor
    regressand for a diagonal controller: d_g responds to lambda_m for every m,
    and steering each d_g by its own lambda_g is exactly the diagonal-control-on-a
    -non-diagonal-plant failure the coupling gate exists to detect. Tension is
    local by construction -- both of its terms are properties of group g alone.

    The shrinkage term -gamma*a_g is what makes "no evidence" mean "uniform". A
    pure integrator on a signal with no persistent structure random-walks away
    from zero and the allocation ends up somewhere arbitrary but confident.
    """

    def __init__(self, cfg, groups, beta):
        self.mode = cfg["anchor_alloc"]
        if self.mode not in ALLOC_MODES:
            raise SystemExit(f"!! unknown anchor_alloc {self.mode!r}: "
                             f"pick from {', '.join(ALLOC_MODES)}")
        self.groups = tuple(groups)
        self.every = int(cfg["anchor_alloc_every"])
        self.beta_a = float(cfg["anchor_alloc_beta_frac"]) * float(beta)
        self.gamma = float(cfg["anchor_alloc_shrink"])
        self.clip = float(cfg["anchor_alloc_clip_decades"]) * math.log(10.0)
        self.tau_rho = float(cfg["anchor_tau_rho"])
        self.hold_steps = int(cfg["anchor_hold_steps"])

        self.a = {group: 0.0 for group in self.groups}
        self.tau = {group: None for group in self.groups}
        self.updates = 0
        self.skipped = 0
        self.clip_hits = 0
        self._probes = 0

    @property
    def active(self):
        return self.mode == "adaptive"

    def offsets(self):
        """The a_g the pull should use. None when the allocation is off, which is
        what makes anchor.lambdas fall back to one shared lambda exactly."""
        return self.a if self.active else None

    # -- the per-probe read and the per-cadence update ---------------------

    def observe(self, step, lambdas, grad_norms, delta_norms, w0_sq):
        """Fold one tension reading in; update a_g when the cadence comes round.

        Called on EVERY arm, including the ones with the allocation off, so tau_g
        is a column in every probes.jsonl and the arms can be compared on it.
        """
        row = {}
        # With no anchor there is no anchor force, so the tension is not large --
        # it is undefined. Emitting eps-divided infinities on the unanchored arm
        # would put a 1e12 column in every probes.csv and poison the EMA the
        # moment an arm switched the anchor on.
        measurable = all(lambdas.get(group, 0.0) > 0.0 for group in self.groups)
        for group in self.groups:
            if measurable:
                raw = (grad_norms.get(group, 0.0) * w0_sq[group]
                       / (lambdas[group] * delta_norms.get(group, 0.0)
                          + _TENSION_EPS))
                if not math.isfinite(raw):
                    raw = _TENSION_FLOOR
                previous = self.tau[group]
                self.tau[group] = (raw if previous is None else
                                   self.tau_rho * previous
                                   + (1.0 - self.tau_rho) * raw)
                row[f"tau_{group}"] = self.tau[group]
            row[f"a_{group}"] = self.a[group]

        self._probes += 1
        row["a_sum"] = sum(self.a.values())
        row["alloc_updates"] = self.updates
        if not self.active or not measurable or step < self.hold_steps:
            return row
        if self._probes % self.every:
            return row

        self._update()
        for group in self.groups:
            row[f"a_{group}"] = self.a[group]
        row["a_sum"] = sum(self.a.values())
        row["alloc_updates"] = self.updates
        return row

    def _update(self):
        """One slow-loop step. Integrate, shrink, re-centre, clip."""
        if any(self.tau[group] is None for group in self.groups):
            self.skipped += 1
            return
        logs = {group: math.log(max(self.tau[group], _TENSION_FLOOR))
                for group in self.groups}
        if not all(math.isfinite(value) for value in logs.values()):
            # Refusing to update is the right failure: an allocation built from a
            # nan is silently wrong for the rest of the run, and the counter says
            # it happened.
            self.skipped += 1
            return
        mean_log = sum(logs.values()) / len(self.groups)

        # High tension means anchor-limited, which wants a SMALLER lambda_g, so
        # the deviation enters with a minus.
        for group in self.groups:
            self.a[group] -= (self.beta_a * (logs[group] - mean_log)
                              + self.gamma * self.a[group])

        mean_a = sum(self.a.values()) / len(self.groups)
        for group in self.groups:
            self.a[group] -= mean_a

        for group in self.groups:
            clipped = max(-self.clip, min(self.clip, self.a[group]))
            if clipped != self.a[group]:
                self.clip_hits += 1
            self.a[group] = clipped
        self.updates += 1

    # -- reporting and persistence -----------------------------------------

    def summary(self):
        out = {"anchor_alloc": self.mode, "alloc_updates": self.updates,
               "alloc_skipped": self.skipped, "alloc_clip_hits": self.clip_hits,
               "a_sum": sum(self.a.values())}
        for group in self.groups:
            out[f"a_{group}_final"] = self.a[group]
            out[f"tau_{group}_final"] = (self.tau[group] if self.tau[group]
                                         is not None else float("nan"))
        return out

    def describe(self):
        if not self.active:
            return "allocation: off (tension still measured and logged)"
        return (f"allocation: adaptive over {len(self.groups)} groups, "
                f"beta_a = {self.beta_a:g}, gamma = {self.gamma:g}, "
                f"every {self.every} probes, |a| <= {self.clip:.3g}")

    def state_dict(self):
        return {"a": dict(self.a), "tau": dict(self.tau),
                "updates": self.updates, "skipped": self.skipped,
                "clip_hits": self.clip_hits, "probes": self._probes}

    def load_state_dict(self, state):
        if set(state["a"]) != set(self.groups):
            raise SystemExit(
                f"!! this checkpoint's allocation is over "
                f"{sorted(state['a'])} but this run's groups are "
                f"{sorted(self.groups)}. Resuming would carry an allocation "
                f"belonging to a different partition.")
        self.a = dict(state["a"])
        self.tau = dict(state["tau"])
        self.updates = state["updates"]
        self.skipped = state.get("skipped", 0)
        self.clip_hits = state.get("clip_hits", 0)
        self._probes = state["probes"]


def _load_replay(cfg):
    """The recorded (step, lambda) points from a closed-loop run's probes.jsonl.

    Read verbatim and held piecewise-constant. Never refitted to a smooth curve:
    the whole point of this arm is to replay what the loop actually did, and a fit
    would quietly turn it into the drift-matched-exponential arm.
    """
    path = cfg["anchor_replay_path"]
    seed = int(cfg["anchor_replay_seed"])
    points = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if "lam" not in row or "step" not in row:
                continue
            if row.get("seed", seed) != seed:
                continue
            points.append((int(row["step"]), float(row["lam"])))
    if not points:
        raise SystemExit(f"!! no (step, lam) rows for seed {seed} in {path}. "
                         f"Point anchor_replay_path at a probes.jsonl from an "
                         f"adaptive run and anchor_replay_seed at a seed in it.")
    points.sort()
    print(f"replay: {len(points)} lambda points from {path} (seed {seed}), "
          f"lambda {points[0][1]:.4g} -> {points[-1][1]:.4g}")
    return points


def _replay_at(points, step):
    """The last recorded lambda at or before `step`. A step function, as recorded."""
    lo, hi = 0, len(points) - 1
    if step < points[0][0]:
        return points[0][1]
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if points[mid][0] <= step:
            lo = mid
        else:
            hi = mid - 1
    return points[lo][1]
