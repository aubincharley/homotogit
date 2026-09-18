# Plan — resolution advancement triggered by probe gradient norm

Status: **specification only. Nothing implemented, nothing launched.**
Branch: `adaptative-resolution`. Baseline: `per-layer-sigma` @ `8112db0`.

---

## 0. What this replaces, and the one thing that must be preserved

Today `r(e)` is a fixed piecewise-constant function of the epoch index
(`Rprog` = 16 for six epochs, 24 for six, then 32). The proposal is to advance
`r` when the network has stopped making progress on the current objective,
measured by the **gradient norm of the current objective on the fixed train
probe**.

The invariant that must survive: the **total horizon stays fixed** at
`total_steps = 11,730` (30 epochs x 391 updates). The LR schedule is a pure
function of the global update index, so keeping `K` fixed means the LR path is
bit-identical to every existing arm. Adaptivity redistributes updates *between*
stages; it never changes the horizon. Without this, "adaptive resolution helped"
and "the LR schedule changed" are confounded, which is the exact failure the
fixed-horizon design was built to prevent.

---

## 1. Why gradient norm and not a loss difference

The rejected criterion was `|L_k - L_{k-1}| < eps`. For one SGD step,

    L(theta_{k+1}) - L(theta_k)  ~=  -alpha_k * ||g_k||^2 ,

so that criterion fires when `alpha_k ||g_k||^2 < eps` — either because the
gradient is small (converged: wanted) or because the learning rate is small
(cosine decay: not wanted). The two are indistinguishable in the signal.

The probe gradient norm removes the confound outright: `||g||` evaluated at a
fixed theta does **not** depend on `alpha_k`. It is the corrector residual, which
is what a predictor-corrector continuation actually tests before stepping the
parameter.

Two further properties of measuring it **on the fixed probe** rather than on
minibatches:

* **Zero evaluation noise.** The probe is a fixed set of 5,000 images, so the
  full-probe gradient is a deterministic function of theta. Minibatch `||g||`
  carries sampling variance that does not vanish as you converge; the probe
  version has none. (The series is still noisy because theta itself follows a
  stochastic trajectory — but that is trajectory noise, not measurement noise,
  and it is far smaller.)
* **It is already the project's evaluation convention.** The probe is a fixed
  5,000-image subset that is already evaluated at every checkpoint, so the
  pairing digests already cover it.

---

## 2. Signal definition

### 2.1 Which objective

The **current path** — the configuration the weights and BN buffers were
actually trained under:

    g_k  =  grad_theta  H(theta_k ;  r_now, sigma_now)   on the probe.

**Not** the target path. Mid-continuation, the target path measures a premature
configuration change including BN statistic mismatch (measured at -32 pp at one
checkpoint from that alone). Its gradient is contaminated for the same reason and
must not drive a controller.

Weight decay is excluded, exactly as it is excluded from the reported objective:
the signal is the gradient of the empirical risk, not of the regularised
objective.

### 2.2 The raw scalar

    gamma_raw(k)  =  || grad_theta H ||_2   over all trainable parameters,
                  =  sqrt( sum_p || g_p ||^2 ).

### 2.3 Scale-freeness — required, not cosmetic

A bare `||g||` is not comparable across stages, for two reasons:

1. The objective differs between stages (`r=16` and `r=32` are different losses),
   so the natural gradient scale differs.
2. **BatchNorm makes `||g||` weight-scale dependent.** For a BN network, scaling
   a layer's weights by `c` scales that layer's gradient by `1/c`. Weight norms
   drift during training, so `gamma_raw` drifts for reasons unrelated to
   convergence.

Therefore the criterion is built on a **stage-relative** quantity. With
`k0` the first check after the current stage began (post-blackout, §3.3):

    gamma(k)  =  EMA_beta( gamma_raw )            (smoothing, §3.1)
    rho(k)    =  gamma(k) / gamma(k0)             (dimensionless, per stage)

`rho` starts at 1 in every stage by construction, which makes a single `eps`
mean the same thing at every stage. This was the decisive defect of an absolute
threshold on loss differences and it must not be reintroduced here.

> **Amended after verification (section 10).** For the *relative-decrease* test
> of section 3.2, `gamma(k0)` cancels algebraically:
>
>     delta = (rho_prev - rho)/rho_prev
>           = (gamma_prev/g0 - gamma/g0) / (gamma_prev/g0)
>           = (gamma_prev - gamma)/gamma_prev
>
> so the criterion is **already scale-free without `rho`**. Keep `rho` as a
> reported diagnostic (it is the natural way to say how far the stage has come),
> but do not treat it as load-bearing, and do not add state for it. `rho` only
> becomes necessary if a *level* test (`rho < eps`) is added later.

---

## 3. The criterion

### 3.1 Smoothing

    gamma(k) = beta * gamma(k-1) + (1 - beta) * gamma_raw(k),   beta = 0.5 default

with `gamma(k0) = gamma_raw(k0)`. At an every-epoch cadence there are at most
~12 checks inside a stage, so heavy smoothing would eat the whole stage; `beta`
must stay small.

### 3.2 Plateau test

Fire when the relative decrease of `rho` over a window `W` falls below `eps`:

    Delta(k)  =  ( rho(k - W) - rho(k) ) / max(rho(k - W), tiny)
    trigger   =  Delta(k) < eps    for `patience` consecutive checks

Defaults to be *calibrated in Phase 0, not assumed*: `W = 2` checks,
`eps = 0.02`, `patience = 2`. Section 10.4 shows these values never fire on a
CE-like signal within 30 epochs, so they are almost certainly wrong for `||g||`
too; treat them as placeholders until Phase 0 data exists.

Rationale for a *relative-decrease* test rather than a *level* test
(`rho < eps`): a level test requires knowing the attainable floor, which is
unknown a priori and stage-dependent. Plateau detection needs no such knowledge.

### 3.3 Guards — all four are mandatory

| Guard | Rule | Why |
|---|---|---|
| **Hard deadline** | advance unconditionally at `step >= k_max(stage)` | Guarantees the terminal `r=32` stage is reached and bounds compute. Direct precedent: the geometric sigma schedule refuses to construct without an explicit terminal target stage, because a schedule that only *approaches* its endpoint never optimises the target problem. An adaptive trigger has the same failure mode and cannot even be bounded in advance. |
| **Minimum dwell** | never advance before `step >= k_min(stage)` | Blocks a spurious early trigger; also gives BN running statistics time to adapt to the new resolution. |
| **Post-switch blackout** | discard the first `B` checks after a switch; **re-seed** the EMA at the first post-blackout check and set a new `k0` | `r` changes jump the objective and desynchronise BN statistics, so `gamma_raw` spikes. Comparing across the boundary is meaningless. **`B` must be at least the spike length**, and the EMA must not carry spike values across the boundary — see the bug in section 10.2. |
| **Fixed horizon** | `K = 11,730` regardless of realised switch times | Keeps the LR path identical to every existing arm (§0). |

With `k_min` and `k_max` set to the same value the controller collapses exactly
to the current fixed `Rprog` schedule. That degenerate case is the first
regression test.

### 3.4 Cadence and cost

Probe = 5,000 images; training epoch = 50,000. One forward+backward over the
probe is therefore ~10% of one epoch of training work. At an every-epoch
cadence that is ~+10% wall time (~+47 s against the plain arm's ~472 s train
time) — acceptable, and to be **timed separately** from training, as evaluation
overhead already is.

Do not check more often than once per epoch: at 391 updates/epoch, sub-epoch
checks multiply cost without adding resolvable signal for a decision that has
only two boundaries.

**But it must be at least once per epoch.** The existing checkpoint cadence is
every *two* epochs, which is too coarse for this purpose: at that spacing the
post-switch spike is invisible (it has decayed before the next check), and with
only ~3 checks inside a six-epoch stage neither the window `W` nor the patience
count has room to operate. Phase 0 must log at every epoch.

---

## 4. Purity requirements for the measurement

The gradient evaluation must not perturb the training trajectory. Any one of
these leaking would silently break pairing:

1. Run under `eval()` mode; restore the previous mode afterwards.
2. BN **running statistics must not be updated**, and must not be recalibrated.
3. Gradients must not land in the live `.grad` buffers used by the optimizer —
   accumulate into scratch storage, or zero afterwards *before* the next
   optimizer step. Momentum buffers must be untouched.
4. No RNG draws consumed: the probe is a fixed, ordered subset with no shuffling
   and no augmentation. The sampler position must be unchanged.
5. Deterministic given `(theta, probe, r, sigma)`.

A regression test must assert that inserting the measurement leaves a short run
**bitwise identical** to the same run without it. This is the same standard the
operator placement audit already meets (max grad diff 0.0 for the post-bypass
phase).

---

## 5. Isolate one operator

Run the first controller experiments with **`Gnone`** — no internal Gaussian.

Reason: the coupling rule `sigma_l = q_l * g(e)` means a data-dependent switch
time makes the *effective sigma* trajectory data-dependent too, and that
trajectory is already non-monotone at resolution changes. With the Gaussian on,
the controller perturbs two operators at once and the result is uninterpretable.

`Rprog__Gnone__input_bilinear` (79.48 +- 0.54) is the correct comparator, not the
80.71 winner. Adding the Gaussian back is a later, separate step, and it needs a
stated decision on whether `g(e)` or the effective `sigma_l(e)` is the thing held
fixed.

---

## 6. Protocol

### Phase 0 — instrument and characterise (decisive, ~1 GPU-hour)

**The probe gradient norm has never been measured in this project.** There is no
`grad_norm` field anywhere in the recorded metrics, so `eps`, `W` and `beta`
cannot be calibrated from existing data, and it is not yet known whether the
signal has a plateau to detect at all.

So: log `gamma_raw` every epoch on **existing fixed schedules**, no controller
attached, no behaviour change:

* `R32__Gnone` (plain control)
* `Rprog__Gnone__input_bilinear` (fixed schedule, the comparator)

3 seeds each. Then report:

1. scale and dynamic range of `gamma_raw` over training;
2. its behaviour at the **known** boundaries (epochs 6 and 12) — the size and
   decay time of the post-switch spike, which sets `B`;
3. **whether a plateau exists before each boundary.** If `gamma` decreases
   steadily with no flattening, there is nothing for the trigger to fire on and
   the idea is dead — for one GPU-hour and no protocol risk.
4. within-stage variance, which sets the achievable `eps` and `W`.

> This phase must complete and be read before any controller is built. It is the
> cheap decisive test.

### Phase 1 — controller, calibration pass

Attach the controller. 3 seeds, `Gnone`, everything else at the campaign
profile. Primary outcome is **not** accuracy:

> **Does the controller pick materially different switch epochs from the fixed
> schedule, and are its choices stable across seeds?**

Three possible readings, all informative:

| Observation | Reading |
|---|---|
| Realised switches ~= epochs 6/12 on all seeds | Adaptivity buys nothing over the fixed schedule. Cheap, clean negative result — stop here. |
| Switches differ from 6/12 but agree across seeds | There is a better *fixed* schedule. Extract it and use Phase 2; the controller was a schedule-design tool. |
| Switches differ across seeds | Genuine per-trajectory adaptation. Interesting, but see §7 — it is also the hardest case to report. |

Record per seed: realised switch steps, which rule fired (plateau vs deadline),
`rho` at each switch, updates spent per stage, and the `gamma` trace.

### Phase 2 — comparison pass, paired and reportable

Take the realised switch steps from Phase 1 and **hard-code them as an ordinary
`piecewise_constant` resolution schedule**. Then run the standard paired
comparison against `Rprog__Gnone__input_bilinear` and `R32__Gnone`, with the
full existing machinery: sha256 digests of subset, probe indices, per-epoch
permutations and initial weights, verified before training.

This is the step that makes the result publishable. It converts adaptivity from
a runtime behaviour into a schedule-design tool, which is what is actually
wanted from it given there are only two boundaries to place.

---

## 7. Known costs of the adaptive runtime, stated up front

These are why Phase 2 exists, and they apply to any run where the controller is
live at comparison time:

* **Pairing degrades.** With data-dependent switch times, two seeds of the same
  configuration switch at different epochs, so "the configuration" is a random
  variable and there is no single schedule to report.
* **Compute becomes random and correlates with accuracy.** Lower `r` is cheaper,
  so a seed that lingers at `r=16` costs less *and* scores differently. "Accuracy
  at equal compute" stops being well defined — and that framing is what makes the
  progressive-resolution result interesting in the first place.
* **Hyperparameter count rises.** `eps`, `W`, `beta`, `patience`, `B`, `k_min`,
  `k_max` replace two stage-boundary integers. Phase 0 must fix most of them from
  data or the comparison is a search over the controller rather than a test of it.
* **Only two decisions exist.** With `r` in {16, 24, 32} there are exactly two
  transitions in a run, which caps what adaptivity can buy.

---

## 8. Files and signatures

Nothing below is implemented. Names are proposed so the diff is predictable.

### New: `continuation/triggers.py`

```
@dataclass
class GradNormTriggerConfig:
    eps: float = 0.02          # relative-decrease threshold on rho
    window: int = 2            # W, in checks
    beta: float = 0.5          # EMA coefficient
    patience: int = 2          # consecutive checks below eps
    blackout: int = 2          # checks discarded after a switch
    min_dwell_steps: int = 0   # k_min per stage
    max_dwell_steps: int = 0   # k_max per stage; 0 -> required, must be > 0

class GradNormPlateauTrigger:
    """Stage-relative plateau detector on a scalar signal.

    Pure and stateful, but with no dependence on the model, the data or the
    optimizer: it consumes scalars and returns a decision, so it is unit
    testable without a training run.
    """
    def __init__(self, cfg: GradNormTriggerConfig) -> None: ...
    def observe(self, step: int, gamma_raw: float) -> None: ...
    def should_advance(self, step: int) -> bool: ...
    def on_stage_change(self, step: int) -> None: ...   # resets EMA, k0, blackout
    def state_dict(self) -> dict: ...                   # must land in the checkpoint
    def load_state_dict(self, d: dict) -> None: ...
    @property
    def reason(self) -> str: ...                        # "plateau" | "deadline" | None
```

### New: `continuation/probe_grad.py`

```
def probe_grad_norm(model, probe_loader, loss_fn, *,
                    device, current_config) -> dict:
    """Full-probe gradient L2 norm of the CURRENT objective.

    Returns {"grad_norm": float, "n": int, "seconds": float, "per_group": dict}.
    Must satisfy every purity requirement in section 4: eval() mode, BN buffers
    frozen, live .grad untouched, no RNG consumed, deterministic.
    """
```

### Modified: `continuation/schedules.py`

Add a resolution schedule that consumes the reserved `state` argument that
`Schedule.parameter(self, step, state=None)` already carries — the interface was
built for this. `build_schedule` must stop raising for this specific kind while
continuing to raise for anything else adaptive.

### Modified: trainer / engine

Call site for the per-epoch measurement, the trigger update, and the stage
advance. The trigger's `state_dict` must be written into the existing
`full_state_v1` checkpoint alongside optimizer state, sampler position and RNG
state, or a resumed run will not reproduce.

### New tests: `tests/test_grad_norm_trigger.py`

1. `k_min == k_max` reproduces the fixed `Rprog` schedule exactly.
2. Deadline fires when the plateau never does (monotone decreasing signal).
3. Blackout suppresses the post-switch spike.
4. Patience suppresses a single-check dip.
5. `rho` is invariant to a global rescaling of the signal (scale-freeness).
6. `state_dict` round-trips; a resumed controller makes identical decisions.
7. Attaching the measurement leaves a short run **bitwise identical** (§4).

---

## 9. Recommendation on priority

This is a refinement of a schedule whose *shape* has not yet been shown to
matter: the plateau-vs-geometric contrast was ~0.8 pp, close to the seed spread,
against a ~0.1 pp noise floor. Q-01 (explicit RGB prefilter before the
reduction) and Q-02 (are intermediate internal stages necessary at all) are
cheaper and more decisive, and neither touches the protocol that makes the
existing numbers trustworthy.

Phase 0 alone is cheap and worth doing regardless of whether the controller is
ever built — the probe gradient norm is a useful diagnostic in its own right,
and it is currently not measured at all.


---

## 10. Verification results

The trigger of sections 3.1-3.3 was implemented standalone as a pure
scalar-in / decision-out function and tested against synthetic signals and
against **real recorded traces** from
`Rprog__Gnone__input_bilinear__all19`, 3 seeds. Nothing in `continuation/` was
touched. **16 of 16 checks pass** after two fixes, both described below.

### 10.1 Guard behaviour (synthetic) — all pass

| Property tested | Result |
|---|---|
| `max_dwell_steps = 0` is refused at construction | pass — terminal stage cannot be silently lost |
| Steady geometric decay -> **deadline** fires, plateau never does | pass |
| Decay-then-flatten -> **plateau** fires, after flattening begins | pass |
| `patience = 2` suppresses a one-check dip that `patience = 1` catches | pass |
| Decision invariant under global rescaling (x1e6, x1e-6) | pass |
| `min_dwell` blocks an early trigger | pass |
| **`min_dwell == max_dwell` reproduces the fixed schedule exactly** | pass (D = 5, 6, 7) |
| `state_dict` round-trip -> identical subsequent decisions | pass |

### 10.2 Bug found: the EMA leaked across the blackout boundary

First implementation kept the EMA warm during blackout, so spike values entered
`gamma` through `beta * gamma_prev` even though the spike checks were themselves
discarded. Effect on a 2-check spike followed by decay-then-plateau:

| `blackout` | fires at check | |
|---:|---:|---|
| 0 | 18 | spike decay read as progress |
| 1 | 18 | |
| 2 | 18 | **wrong** — should have recovered the spike-free timing |
| 3 | 15 | |

After re-seeding the EMA at the first post-blackout check, `blackout = 2` fires
at check 15 — exactly the spike-free timing (13) offset by the spike length (2),
which is now a regression test. **Requirement: `B >= ` the spike length, and the
EMA must be re-seeded, not carried.** The spike length is unknown until Phase 0
measures it.

### 10.3 The LR confound, confirmed on real data

Measured on the real per-checkpoint probe CE and logged LR, epochs 4-30
(initial transient dropped), first-half vs second-half means:

| Quantity | first/second-half ratio | reading |
|---|---:|---|
| raw `|dCE|` | **5.41x** | collapses across training |
| `|dCE|` / LR-mass | **0.83x** | flat |

The raw loss difference falls by a factor of five while the LR-normalised
version is flat to within 20%. So most of the decline in `|dCE|` is the cosine
LR decay, not convergence — which is exactly the confound that motivated
switching to gradient norm. **The rejection of the loss-difference trigger is
now empirically supported, not just argued.**

### 10.4 Decisive proxy finding: there may be no plateau to detect

Replaying the detector on real probe CE (`W = 2`, `patience = 2`), first-fire
epoch per seed. The fixed schedule switches at **epochs 6 and 12**.

| `eps` | seed 0 | seed 1 | seed 2 | spread |
|---:|---:|---:|---:|---:|
| 0.02 | — | — | — | never fires |
| 0.05 | — | — | — | never fires |
| 0.10 | — | 30 | 30 | |
| 0.20 | 8 | 8 | 8 | **0** |
| 0.30 | 8 | 8 | 8 | **0** |
| 0.50 | 8 | 8 | 8 | **0** |

Two things follow.

**There is no `eps` that behaves like a plateau detector near the boundaries.**
At any `eps` tight enough to mean "progress has stalled" (<= 0.05) the detector
never fires within the 30-epoch horizon; at `eps >= 0.20` it fires at epoch 8,
but `eps = 0.20` means "advance when CE improved by less than 20% over two
checks", which is a fast-progress criterion, not a plateau. The probe CE is
still improving briskly at epochs 6 and 12. A plateau controller on such a
signal would **always defer to the deadline** — i.e. degenerate exactly into the
fixed schedule it was meant to replace.

**Where it does fire, the timing is identical across seeds (spread 0).** That is
the "cheap negative" reading of Phase 1 arriving early: if the trigger picks the
same epoch on every seed, per-trajectory adaptation buys nothing over a fixed
schedule, and the whole protocol cost in section 7 is paid for nothing.

> [!warning] What this does and does not establish
> Probe **CE is a proxy**, not `||g||`. The gradient norm is not the loss, and
> for a BN network its dynamics differ substantially — `||g||` can flatten while
> CE still improves. So this does **not** kill the idea. What it establishes is:
> (a) the detector machinery and all four guards work; (b) the loss-difference
> trigger is confounded, measured; (c) the proposed defaults are wrong; and
> (d) **Phase 0 is mandatory and is the whole experiment.** If `||g||` turns out
> to look like CE — monotone, no plateau before the boundaries — the idea dies
> for one GPU-hour, which is the cheapest possible outcome.

### 10.5 Revised Phase 0 requirements

1. Log `gamma_raw` **every epoch**, not every two (section 3.4).
2. Report the **post-switch spike length and amplitude** at the known boundaries
   (epochs 6, 12); this sets `B`.
3. Report whether `||g||` **flattens before** epochs 6 and 12, and at what
   relative-decrease rate — this is what `eps` must be calibrated against, and
   the table in 10.4 is the null hypothesis to beat.
4. Report `||g||` alongside probe CE so the two can be compared directly; if
   they are monotonically related, this whole approach reduces to a
   loss-difference trigger with extra steps.


---

## 11. EXP-A — the cheap local go/no-go run

Single seed, CPU, **one arm, no controller**. This is Phase 0 cut down to the
smallest run that can still kill the idea.

### 11.1 Measured local cost

Hardware here is **CPU only, 14 threads** (no CUDA). Measured median
fwd+bwd+step at microbatch 32, and the projected training cost per 50k epoch:

| resolution | us/image | min per epoch (50k) |
|---:|---:|---:|
| r=16 | 3,456 | **2.88** |
| r=24 | 5,330 | **4.44** |
| r=32 | 12,284 | **10.24** |

So a faithful full 30-epoch run costs **~228 min (3.8 h)** locally. That is the
number to beat.

Note the cost is driven by **images processed**, i.e. `updates x batch`. So
subsampling the training set saves nothing unless the update count falls with
it, and reducing the update count is what changes the answer. The only free
lever is to **stop the run early**.

### 11.2 The design

> Run `Rprog__Gnone__input_bilinear__all19`, seed 0, full 50k train, and
> **stop after epoch 7**: six epochs at `r=16`, then one epoch at `r=24`.

| | |
|---|---|
| Arm | `Rprog__Gnone__input_bilinear__all19` — `Gnone`, so exactly one operator moves (section 5) |
| Seeds | 1 (seed 0) |
| Train data | full 50k, unchanged |
| Epochs run | **7** of 30 (stop after the first epoch at `r=24`) |
| Resolution boundary | epoch 6 = **update 2346**, the real index |
| **LR schedule** | spans the **full** `K = 11,730`, unchanged | 
| Controller | **none.** Pure instrumentation. |
| Signal logged | probe `||g||` of the current objective, every **half-epoch** (~196 updates), 5,000-image probe |

**The LR schedule must not be recompressed to 7 epochs.** It is a pure function
of the global update index, so leaving `K = 11,730` means epochs 0-6 see exactly
the learning rates they would see in the real 30-epoch run, and the trace is
directly comparable to the recorded ones. Recomputing the cosine over 7 epochs
would answer a different question.

Why stop at 7: the first boundary is the one that matters — it is where the
trigger would first act, and if `||g||` has not flattened there the method is
already dead. The extra epoch at `r=24` is kept only to measure the post-switch
spike, which is what sets `B`.

### 11.3 Cost

| item | cost |
|---|---:|
| training, 6 epochs @ r=16 | 17.3 min |
| training, 1 epoch @ r=24 | 4.4 min |
| probe gradient, 14 checks | 7.4 min |
| **total** | **~29 min** |

That is **13%** of a faithful full run. If it needs to be cheaper: log every
epoch instead of every half-epoch (-3.7 min), or use a 2,000-image probe
(-4.4 min), either giving ~25 min. The 17.3 min of `r=16` training is
irreducible — it is the cost of reaching the state the boundary occurs in.

### 11.4 Decision rule — fix this before looking at the trace

Let `gamma(e)` be the EMA-smoothed probe gradient norm, and define the relative
decrease over the last two epochs of stage 1:

    D = ( gamma(4) - gamma(6) ) / gamma(4)

| outcome | reading | action |
|---|---|---|
| **D < 0.05** | `||g||` has genuinely flattened before the boundary. A plateau trigger has something to fire on. | **GO** — proceed to Phase 0 proper (3 seeds, both boundaries) |
| **D > 0.10** | `||g||` is still falling briskly at the boundary. Any `eps` tight enough to mean "plateau" will not fire, the deadline always wins, and the controller degenerates into the fixed schedule. | **NO-GO** — stop |
| 0.05 - 0.10 | `eps` would have to operate inside the noise band. | marginal; needs 3 seeds to separate signal from trajectory noise |

The **null hypothesis is NO-GO**: section 10.4 already showed that on probe CE
no `eps <= 0.05` fires anywhere in 30 epochs. This run exists to try to
falsify that on the gradient norm, which is a genuinely different signal.

### 11.5 Three secondary readings, all free from the same run

1. **Spike length and amplitude** at the epoch-6 boundary, from the `r=24`
   epoch. Sets `B` (section 3.3); the bug in 10.2 means `B` must be at least the
   spike length.
2. **Rank correlation of `||g||` against probe CE** over stage 1. If
   `|rho_s| > 0.95`, the gradient norm carries no information the loss does not,
   and the approach collapses back into the LR-confounded loss-difference
   trigger of section 10.3. **This is an independent kill criterion.**
3. **Within-stage variance** of `gamma` between consecutive half-epoch checks —
   the noise floor, which lower-bounds any usable `eps`.

### 11.6 What must not drift from the recorded runs

So the trace stays comparable to the existing `Rprog__Gnone` traces:

* channel-normalisation constants estimated once on the full original train
  split and **shared unchanged across resolutions**;
* seed-0 initial weights and per-epoch permutations (digests already exist);
* probe indices unchanged;
* the reduction acts on the **float** image after the `/255` conversion, before
  channel normalisation.

### 11.7 Measurement purity — one practical note

The existing evaluation entry point is decorated `@torch.no_grad()`, so it
cannot be reused for this; a sibling function is needed. It must reproduce the
existing `eval()` / restore-previous-mode discipline, must not update BN running
statistics, and must not disturb the optimizer's gradient buffers.

Simplest way to guarantee the last point in this diagnostic run: **take the
measurement immediately after an optimizer step has completed**, when no
gradient accumulation is in flight, and zero the gradients afterwards. At a
half-epoch cadence that is always satisfiable.


---

## 12. Phase 0 executed — 2026-09-16, Kaggle 2x T4, account martingraffin

Job `scripts/job_adaptive_phase0.py`, kernel `adaptive-phase0-20260916-185116`,
outputs under `results/kaggle_outputs/adaptive-phase0-20260916-185116/`, analysis
by `scripts/analyze_adaptive_phase0.py` → `results/adaptive_phase0_analysis.{json,md}`.
16 runs, 4,394 s elapsed on two T4s (~2.4 GPU-hours), 0 failures. The shared
state was regenerated in the kernel and **all eight sha256 digests matched the
pinned campaign assets** (subset, probe, three permutations, three inits), so
Part A is paired with the 63-cell campaign. Every per-epoch measurement asserted
that parameters, BN buffers and the train/eval mode were unchanged afterwards.

Departures from sections 6 and 11: the monitor set for gradient signals is a fixed
class-balanced **2,000**-image subset of the training split (the pairing probe is
500 images, not the 5,000 assumed in §1); measurements are taken once per epoch;
Part B (schedule grid) was added because the trigger replay in §10.4 already
suggested that adaptivity over two boundaries had little to buy.

### 12.1 Replication of the six paired cells

| cell | this job | campaign | diff (pp) |
|---|---:|---:|---:|
| R32 seed 0 / 1 / 2 | 74.76 / 76.01 / 76.40 | 75.13 / 75.62 / 76.10 | −0.37 / +0.39 / +0.30 |
| Rprog seed 0 / 1 / 2 | 78.63 / 78.91 / 79.87 | 79.20 / 79.10 / 80.10 | −0.57 / −0.19 / −0.23 |

Same magnitude as the campaign's own re-runs of EXP-008 cells (max 0.91 pp). The
three Rprog differences share a sign; with the measurement verified pure, this
is GPU non-determinism until shown otherwise, but it is the noise floor against
which every Part B number below must be read.

### 12.2 The gradient-norm plateau criterion: **NO-GO**

Two versions of the corrector residual were measured on the current objective.

**Eval-mode BN (as specified in §4).** Values swing by a factor of 2–3 between
consecutive epochs with no stage structure (R32 seed 0: 4.2, 4.1, 10.0, 4.5, 8.1,
7.4, 5.5, 2.4 … 14.1 at epoch 15). The §11.4 rule gives contradictory verdicts on
the same schedule: Rprog stage r=16 D = 0.41 (NO-GO), 0.05 (marginal), 0.04 (GO)
for seeds 0/1/2. This signal is dominated by the mismatch between running
statistics and the weights, not by convergence. It is unusable as a controller
input, and the cosines below inherit some of that noise.

**Train-mode BN (the training objective's own gradient).** Flat at **1.0 ± 0.2**
from epoch 1 to epoch 20 in every run, at every resolution, then drifting to
0.83 as the cosine LR goes to zero:

    Rprog seed 0:  1.50 0.82 1.00 0.93 1.14 1.24 | 1.46 1.11 1.33 1.09 1.00 1.26 | 0.97 1.19 … 0.86
    R32   seed 0:  1.20 1.28 1.22 1.19 1.50 1.18 1.20 0.92 1.40 1.15 1.07 1.00 … 0.83

There is no within-stage decrease, hence no plateau to detect, and the level does
not distinguish r=16 from r=32. SGD at this LR sits at a stationary gradient-norm
floor; the premise of §3 — that the residual falls and flattens before a boundary
— is false for this recipe. A plateau controller would defer to the deadline in
every run, i.e. reproduce the fixed schedule at extra cost. Spearman(||g||, CE)
inside a stage is 0.4–0.6 and sometimes negative: the two signals differ, but the
extra information in ||g|| is not about stage completion.

Post-switch spikes (eval-mode): entering r=24 multiplies ||g|| by 1.6–2.7 for up
to 4 checks; the Δr ≤ 4 steps of `Rsteps4`/`Rlin12` give ×1.0–1.4 for one check.

### 12.3 Q-04 answered: how much of the target-path collapse is BatchNorm

BN statistics were rebuilt at fixed weights on the monitor set and the test set
re-evaluated on the 32×32 target path (seeds 0 / 1):

| epoch, stage | current path | target path | target after BN recal |
|---|---:|---:|---:|
| 6, end of r=16 | 56.5 / 59.2 | 30.4 / 40.6 | 43.9 / 47.9 |
| 12, end of r=24 | 68.8 / 70.5 | 65.6 / 67.2 | 68.2 / 69.8 |

At r=16 recalibration recovers about half of the 26 pp collapse; the rest is a
genuine function mismatch. At r=24 essentially the whole 3 pp collapse is BN.

### 12.4 Look-ahead at the next resolution

The BN-recalibrated CE of the *next* resolution decreases monotonically through
every stage of every run (Rprog seed 0, r=16 stage: 1.74 → 1.55 → 1.49 → 1.42 →
1.39 → 1.36 on the 24×24 problem). Coarse training never stops helping the finer
problem before the fixed boundary, so a "switch when it stops helping" rule would
also defer to the deadline. It does decelerate: the per-epoch improvement of the
next-resolution CE relative to the current-resolution CE (ratio ρ) falls from ~0.8
to 0.3–0.4 by epochs 5–6 at r=16 for seed 0 and fires a `ρ < 0.5 twice` rule at
exactly epoch 6; seeds 1 and 2 are too noisy on 2,000 images for the rule to fire
(ratios 0.2–0.8 without trend). Candidate, not a result.

### 12.5 Gradient alignment depends on the size of the resolution jump

Transfer efficiency τ = ⟨g_r, g_{r'}⟩ / ||g_{r'}||² is the fraction of a fine step's
first-order progress that a coarse step delivers on the finer objective.
(Eval-mode gradients; the epoch-3 dip to ≈0 seen in every run coincides with the
eval-mode ||g|| spike and is a BN artefact.)

| jump | τ over the stage (seed 0) |
|---|---|
| 16 → 24 (Rprog) | 0.53, 0.32, 0.15, ≈0, 0.14, 0.09, 0.07 |
| 24 → 32 (Rprog) | 0.23, 0.16, 0.43, 0.48, 0.37, 0.35 |
| Δr = 4 (Rsteps4) | 0.3–0.9 |
| Δr = 1–2 (Rlin12) | 0.6–1.0 |

Alignment is governed by the gap between consecutive resolutions far more than by
time within a stage: a 16 → 24 jump wastes ~90 % of the coarse gradient from the
finer problem's point of view by epoch 4, while Δr ≤ 2 keeps 60–100 %.

### 12.6 Schedule grid (seed 0, no filter, bilinear input reduction — exploratory)

| schedule | final test acc | test acc at epoch 13 (entering 32) | train CE / test CE at 30 |
|---|---:|---:|---|
| **Rsteps4** 16/20/24/28 × 3 ep | **79.80** | 75.2 | 0.33 / 0.59 |
| **Rlin12** one size per epoch | **79.70** | 75.6 | 0.35 / 0.59 |
| Rmixed per-update {16,24,32} | 79.16 | 67.9 | 0.44 / 0.61 |
| Rb9_15 | 79.05 | 66.7 | 0.41 / 0.61 |
| Rb6_9 / Rb6_18 | 78.96 / 78.96 | 74.7 / 67.2 | |
| Rb3_12 | 78.84 | 74.0 | |
| **Rprog (6,12)** | 78.63 | 73.3 | 0.36 / 0.62 |
| Rb9_18 | 78.60 | 66.5 | |
| Rb3_9 | 78.20 | 74.1 | |
| Rb12_18 | 77.46 | 62.3 | 0.50 / 0.64 |
| R32 | 74.76 | 72.9 | 0.24 / 0.77 |

Two readings. (i) **Boundary placement is nearly flat**: six of the seven
alternative two-boundary schedules land within ±0.45 pp of Rprog, inside the
replication noise; only a twelve-epoch r=16 stage (12,18) clearly hurts. There is
little for a two-boundary controller to win. (ii) **Granularity is not flat**: the
two fine-grained schedules beat Rprog by +1.1 / +1.2 pp on the same seed, same
hardware, same job, and arrive at the 32×32 stage 2 pp ahead. One seed; the
Rprog seed spread is ±0.6 pp, so this needs the paired three-seed comparison
before it is a result. It is, however, exactly what §12.5 predicts.

The generalisation gap (test CE − train CE) is 0.53 for the plain control and
0.14–0.36 for every progressive schedule: most of the resolution effect in this
no-augmentation recipe is a generalisation effect, not faster optimisation.

## 13. Revised proposals — how adaptive resolution could actually work

1. **Retire the plateau trigger** (sections 3 and 11). Its input is stationary. No
   controller built on `||g||` alone can beat the deadline guard.
2. **Confirm granularity first (cheap, paired, decisive).** `Rsteps4` and `Rlin12`
   with seeds 0/1/2 against `Rprog` — six new cells, ~35 min on two T4s. Then the
   same two schedules with the Gaussian plateau filter (the campaign winner is
   `Rprog + Gplateau` at 80.71); the σ-coupling `σ_l = q_l·g(e)` already handles
   arbitrary r. If the +1 pp survives three seeds, the "adaptive resolution"
   question becomes "how fine, over how many epochs", which is a one-parameter
   family, not a controller.
3. **Adapt the step size, not the boundary** (predictor-corrector logic). Every
   epoch, measure τ for candidate next sizes r+Δ, Δ ∈ {1, 2, 4, 8}, and take the
   largest Δ with τ ≥ θ (θ ≈ 0.5). This is a step-size controller, exactly as in
   numerical continuation: the predictor (coarse gradient) is accepted while it
   still explains most of the fine step. Requirements: **train-mode BN gradients**
   (eval-mode is too noisy, §12.2), a monitor set of ≥ 2,000 images, the fixed
   horizon K = 11,730 and a hard schedule floor r(e) ≥ Rlin12(e) so that r=32 is
   reached by epoch 12 at the latest. Cost of the measurement: ~5 % of an epoch.
   Phase 1 for this controller = instrument `Rlin12` and `Rsteps4` with τ at
   several Δ, then replay; no live controller until the replay shows seed-stable,
   non-trivial step choices.
4. **Look-ahead rate ratio as a secondary trigger.** ρ = ΔCE_next(recal) /
   ΔCE_current is LR-free by construction (both terms see the same LR) and did
   fall below 0.5 at epoch 6 on seed 0. To make it usable: EMA over two epochs and
   a 5,000-image monitor set (the per-epoch difference on 2,000 images is at the
   noise floor). Replay first.
5. **Mixed resolution as the BN-shock-free continuation.** `Rmixed` beat Rprog by
   +0.5 pp (seed 0) with no boundary at all and BN statistics that already cover
   32×32 during the coarse phase (its 32×32 target-path accuracy during epochs 1–6
   was 32–51 % against 25–33 % for Rprog). Its mixture weights are the natural
   adaptive knob — shift mass to finer sizes as τ falls — and it never needs a
   blackout because nothing switches. Three seeds, then a τ-driven mixture.
6. **BN recalibration at every boundary is free and should be standard.** It
   recovers half of the collapse at 16→32 and all of it at 24→32 (§12.3), costs one
   forward pass over 2,000 images, and does not touch the weights. It also removes
   the spike that forced the blackout guard in §3.3.
7. **Where adaptivity pays: STL-10.** On CIFAR T4 kernels reduced resolution
   saves no wall time (EXP-011 §9), so a controller can only redistribute updates.
   At 96×96 the coarse stages are genuinely cheaper, and the same τ / ρ signals
   turn into an accuracy-per-second controller. Port the instrumentation with the
   STL-10 protocol, not before the three-seed confirmations in item 2.

Not started here: nothing above was run with the Gaussian filter, with
augmentation, or on more than one seed for Part B. The one-seed grid selected on
an already-consulted test set is exploratory by the project's own standard.


---

## 14. Phase 1 executed — 2026-09-17, proposals 2, 3 (instrumentation) and 5

Job `scripts/job_adaptive_phase1.py` (thin launcher for `job_adaptive_phase0.py`
with `ADAPT_PHASE=1`), kernel `adaptive-phase1-20260917-075615`, outputs under
`results/kaggle_outputs/adaptive-phase1-20260917-075615/`. 15 runs, 5,653 s
elapsed on two T4s (~3.1 GPU-hours), 0 failures, all eight pinned digests
matched, every measurement verified pure. Seed 0 of the unfiltered `Rsteps4`,
`Rlin12` and `Rmixed` arms comes from the Phase 0 job (same pinned assets, other
kernel), so those three contrasts mix two jobs; the three `Rprog + Gplateau`
cells replicate the campaign at −0.12 / −0.43 / −0.14 pp.

### 14.1 Granularity confirmed without the filter, absorbed by the filter

Final test accuracy (%), seeds 0 / 1 / 2, and paired differences per seed.

| configuration | seeds | mean | train s |
|---|---|---:|---:|
| Rprog + Gplateau (campaign winner) | 80.03 / 80.14 / 81.28 | **80.48** | 667 |
| Rsteps4 + Gplateau | 79.64 / 80.31 / 81.32 | 80.42 | 670 |
| Rlin12 + Gplateau | 79.73 / 80.11 / 80.71 | 80.18 | 676 |
| **Rlin12, no filter** | 79.70 / 79.97 / 80.74 | **80.14** | 457 |
| **Rsteps4, no filter** | 79.80 / 79.54 / 80.71 | **80.02** | 463 |
| Rmixed, no filter | 79.16 / 79.66 / 80.09 | 79.64 | 462 |
| Rprog, no filter | 78.63 / 78.91 / 79.87 | 79.14 | 451 |

| paired contrast | per seed (pp) | mean |
|---|---|---:|
| Rlin12 − Rprog (no filter) | +1.07 / +1.06 / +0.87 | **+1.00** |
| Rsteps4 − Rprog (no filter) | +1.17 / +0.63 / +0.84 | **+0.88** |
| Rmixed − Rprog (no filter) | +0.53 / +0.75 / +0.22 | +0.50 |
| Rsteps4+G − Rprog+G | −0.39 / +0.17 / +0.04 | −0.06 |
| Rlin12+G − Rprog+G | −0.30 / −0.03 / −0.57 | −0.30 |
| Gplateau's own gain on Rprog / Rsteps4 / Rlin12 | | +1.35 / +0.41 / +0.05 |
| Rlin12 (no filter) − Rprog+G | −0.33 / −0.17 / −0.54 | −0.35 |

Three readings. (i) The Phase 0 one-seed result holds: **finer resolution steps
are worth about +1 pp over the 16/24/32 schedule, positive on every seed**, for
the same update budget and the same wall time. (ii) **The Gaussian filter and the
fine steps are substitutes, not complements.** The filter's gain shrinks from
+1.35 pp on Rprog to +0.41 on Rsteps4 and +0.05 on Rlin12, and adding fine steps
to the filtered winner gives nothing. Recall that under `σ_l = q_l·g(e)` the fine
ramp also changes the filter's effective schedule (σ rises smoothly with r
instead of jumping), so the two filtered ramps are not the same operator as
`Rprog + Gplateau` with a different resolution path. (iii) The unfiltered fine
ramps reach 80.0–80.1 %, within 0.35–0.47 pp of the best filtered arm, at **68 %
of its training time** (457 vs 667 s). For the accuracy-per-time frontier this
is the new candidate, ahead of `stem_max` without Gaussian (79.97 %, EXP-011).

### 14.2 Train-mode transfer efficiency is a clean signal, and it does decay with dwell time

τ_Δ = ⟨g_r, g_{r+Δ}⟩ / ||g_{r+Δ}||² with train-mode BN gradients, 2,000-image
monitor set, measured every epoch on every run (Phase 0's eval-mode cosines were
BN-contaminated; §12.5 overstated the 16→24 collapse).

`Rprog + Gplateau`, seed 0, stage r=16, epochs 0…6:

| Δ | τ over the stage |
|---|---|
| +2 | 1.00, 0.84, 0.67, 0.71, 0.55, 0.57, 0.49 |
| +4 | 0.91, 0.69, 0.61, 0.57, 0.39, 0.42, 0.38 |
| +8 (the scheduled jump) | 0.87, 0.50, 0.30, 0.27, 0.21, 0.16, 0.17 |

The network specialises to the coarse scale as it dwells: alignment with the
+8 target falls from 0.87 to 0.17 in five epochs and flattens there, while
alignment with +2 stays above 0.5. Along the fine schedules, τ for the step
actually taken stays at 0.6–0.9 throughout (`Rsteps4`, Δ=4: 0.75–0.94 at r=16,
0.74–0.91 at r=20, 0.75–0.91 at r=24, 0.72–0.90 at r=28; `Rlin12`, Δ=1–2:
0.65–1.04). So **the fixed-schedule ranking of §14.1 is the ranking by the
alignment kept at each switch.**

A parity effect appears in the Δ=1 rows: a step into an **odd** size aligns
worse than a step of 2 or 4 into an even size (from 16 at epoch 3: τ(+1→17) =
0.36, τ(+2→18) = 0.57, τ(+4→20) = 0.75; in `Rlin12`, the even→odd steps 16→17 and
20→21 have the two lowest τ, 0.54 and 0.65, while odd→even and odd→odd steps sit
at 0.86–1.04). Odd sizes change the sampling phase of the three stride-2 stages
(17→9→5 vs 18→9→5). Cheap check: the same 12-epoch ramp rounded to even sizes
(`Rlin12even`, Phase 2).

### 14.3 What the step-size replay says, and what it does not

Replaying "largest Δ with τ ≥ 0.5" on the fine-schedule traces picks Δ=8 almost
everywhere, because along those trajectories the network never specialises
enough for τ(+8) to drop. Taken literally that rule would rebuild Rprog-like
jumps, which §14.1 shows are the worse choice. The right reading of τ is
therefore the opposite of the ODE step-size heuristic: **a high τ says staying
coarse costs nothing on the finer objective, a low τ says the coarse stage is
exhausted for that target.** The controller of §13.3 is restated as: dwell while
τ(r → r+Δ) stays above θ, advance by Δ when it falls below, with Δ small enough
that the post-switch τ is high again (Δ = 4 from the data) and a schedule floor
so that 32×32 is reached by epoch 12. Whether that produces a schedule that
beats the fixed `Rsteps4` cannot be read off these traces — the decision changes
the trajectory — hence Phase 2 below.

### 14.4 Not done, and why

BN recalibration at boundaries as a *training* arm (proposal 6) was dropped:
training uses batch statistics, so rebuilding the running statistics changes no
weight and the epoch-30 model is identical. It matters for evaluating a model
mid-continuation and for measurement, which the job already does. The
look-ahead rate ratio (proposal 4) was not re-measured with a larger monitor set;
Phase 1 kept the 2,000-image set for cost. Nothing was run with augmentation or
on STL-10.

## 15. Phase 2 — live controller (2026-09-17, kernels `adaptive-phase2-20260917-093614` and `adaptive-phase2b-20260917-113110`)

Job `scripts/job_adaptive_phase2.py`. Nine unfiltered runs, seeds 0/1/2:

* `Rctrl50` / `Rctrl65`: start at r=16; after every epoch compute τ(r → r+4)
  with train-mode BN gradients on the monitor set, EMA β=0.5 re-seeded after each
  switch; advance by 4 when the EMA falls below θ = 0.50 / 0.65; floor
  r ≥ 16 + 4·(e−8) from epoch 8, so r=32 from epoch 12 at the latest; horizon and
  LR path unchanged. Realised schedules and every decision are logged.
* `Rlin12even`: 16, 18, 18, 20, 22, 22, 24, 26, 26, 28, 30, 30, then 32 — the
  parity check of §14.2.

**First launch invalid for the controller arms.** `SiteController` keeps its own
copy of the resolution table, so the controller's decisions updated the job's
schedule list but never the training path: the six `Rctrl*` networks trained at
16×16 for all 30 epochs while being *evaluated* at the decided resolution
(final 25–38 %, i.e. the r=16 target-path collapse of §12.3). The runs are
marked `INVALID.json` and excluded from the analysis; their signal traces remain
valid measurements of a network held at r=16. Fix: the controller now writes
into the site controller's table as well, and the training loop raises if the
resolution it is about to use differs from the schedule. Controller arms re-run
in `adaptive-phase2b-20260917-113110`; the three `Rlin12even` cells of the first
launch are unaffected.

### 15.1 Even-size ramp (valid, first launch)

`Rlin12even` 80.07 / 79.84 / 80.45 → **80.12 %**, against `Rlin12` 79.70 / 79.97 /
80.74 → 80.14 % (paired: +0.37 / −0.13 / −0.29). The parity effect seen in τ
(§14.2) does not carry to final accuracy; odd sizes are harmless. Fine-grained
ramps now stand at 80.0–80.1 % on three independent schedule variants.

Readings fixed in advance (plan §6, Phase 1 table): switches at the same epochs
on all seeds → a better fixed schedule, extract it; switches differing across
seeds with accuracy ≥ `Rsteps4` → genuine adaptation; controller ≤ `Rprog` or
dominated by the floor → the τ-timing idea is dead and granularity alone is the
result.

### 15.2 Controller results (re-run `adaptive-phase2b-20260917-113110`, 1,792 s, 6 runs, pinned digests matched)

**The τ trigger never fired.** At both thresholds the EMA of τ(r → r+4) stayed
between 0.62 and 0.94 through nine epochs at r=16 and above 0.71 at every later
size; every one of the 24 switches was made by the floor. All six runs therefore
realised the same schedule, 16 ×9 → 20 → 24 → 28 → 32 from epoch 12:

| run | seeds 0 / 1 / 2 | mean | vs Rprog | vs Rsteps4 | vs Rlin12 |
|---|---|---:|---:|---:|---:|
| Rctrl50 (θ = 0.50) | 78.61 / 78.88 / 80.29 | 79.26 | +0.12 | −0.76 | −0.88 |
| Rctrl65 (θ = 0.65) | 78.57 / 79.21 / 79.56 | 79.11 | −0.03 | −0.90 | −1.03 |

The two variants ran an identical schedule, so their per-seed differences
(0.04 / 0.33 / 0.73 pp) are a direct measurement of the run-to-run noise on this
hardware.

This is the third pre-registered reading of §15: the controller is dominated by
its guard and lands at Rprog's level. It also settles a sharper question than
the one asked. The floor schedule has **the same coarse-epoch mass (12), the same
step size (4) and the same terminal epoch (12) as `Rsteps4`**; the only
difference is that it spends nine epochs at 16 and one at each of 20/24/28
instead of three at each. That costs about 1 pp on every seed. So the +1 pp of
the fine ramps is not about *when* a coarse stage is exhausted — by the
alignment criterion, nine epochs at 16 never exhausted it — but about
**spreading the dwell evenly across scales**. Gradient alignment does not see
this; nothing measured in Phases 0–2 does.

### 15.3 Conclusion for "adaptive resolution"

Three trigger families were tested on real trajectories with the horizon and LR
path held fixed: plateau of the gradient norm (stationary signal, §12.2),
look-ahead improvement (monotone, never fires, §12.4), and transfer efficiency
(live controller, never fires, §15.2). None produces a data-dependent schedule
that beats a fixed one; each degenerates into its deadline guard. What did move
the result is a fixed design choice: a ramp with small steps and equal dwell,
worth +0.9 to +1.0 pp over the 16/24/32 schedule on three paired seeds, three
schedule variants (`Rsteps4`, `Rlin12`, `Rlin12even`), no filter, same wall time
— and 80.0–80.1 % at 68 % of the time of the filtered campaign winner (80.5 %).
With the Gaussian filter the ramp adds nothing.

Recommended next steps, none launched: (i) promote `Rlin12`-type ramps to the
campaign protocol as the unfiltered reference and re-run the accuracy-per-time
comparison against `stem_max` (EXP-011) within one job; (ii) test whether the
"equal dwell" gain survives standard augmentation, since §12.6 showed the
resolution effect is largely a generalisation effect; (iii) carry the ramp, not
a controller, to STL-10, where coarse epochs are genuinely cheaper.


---

## 16. STL-10 — 2026-09-17, kernel `stl10-resolution-20260917-121258` (+ controls `stl10-controls-*`)

Job `scripts/job_stl10_resolution.py`, analysis `scripts/analyze_stl10_resolution.py`
→ `results/stl10_resolution_analysis.md`. Labelled STL-10 (5,000 train / 8,000
test, 96×96, official binaries from the Kaggle dataset `yellowflag/stl10labeled2`),
no augmentation, channel statistics fitted on the training set, ResNet-20 BN,
SGD 0.005 / 0.9 / 5e-4, warmup 60 then cosine, batch 128 as 4×32 (last group of
8 weighted by count), 40 updates per epoch, **60 epochs = 2,400 updates**. This
matches the `stl10-transfer` branch's protocol, whose plain control reached
57.3 % (three seeds), and is *not* paired with it: init weights and
permutations were regenerated here from the project's named RNG streams and
their digests recorded. 15 runs, 4,674 s elapsed on two T4s, 0 failures, every
evaluation verified pure.

Arms are the CIFAR schedules with sizes ×3 and epoch boundaries ×2: `R96`;
`Rprog` 48 ×12 / 72 ×12 / 96 ×36; `Rsteps4` 48/60/72/84 ×6 then 96; `Rlin24` one
even size per epoch 48 → 94 then 96 ×36; and `Rlin24eq`, the same ramp with the
horizon extended to 72 epochs so that its *nominal compute* (61.6 units of a
96×96 epoch) matches `R96`'s 60.

### 16.1 Coarse epochs are now genuinely cheaper

Median training seconds per epoch: 48×48 2.34 s, 72×72 5.72 s, 96×96 8.90 s,
i.e. 0.26 / 0.64 / 1.00 of the full-size epoch against nominal (r/96)² =
0.25 / 0.56 / 1.00. Unlike CIFAR on the same T4 (EXP-011 §9), resolution now
buys wall time almost proportionally to pixel count.

### 16.2 Accuracy at equal updates, and at equal time

| arm | epochs | final test acc (seeds 0/1/2) | mean | vs R96 per seed | train s |
|---|---:|---|---:|---|---:|
| R96 | 60 | 57.66 / 58.76 / 56.59 | 57.67 | | 542 |
| Rprog | 60 | 58.77 / 57.99 / 59.54 | 58.77 | +1.11 / −0.77 / +2.95 | 425 |
| Rsteps4 | 60 | 59.36 / 58.41 / 58.85 | 58.88 | +1.70 / −0.35 / +2.26 | 453 |
| Rlin24 | 60 | 58.61 / 58.41 / 57.11 | 58.05 | +0.95 / −0.35 / +0.52 | 460 |
| **Rlin24eq** | 72 | 60.62 / 60.46 / 59.67 | **60.25** | **+2.96 / +1.70 / +3.09** | 582 |

Test accuracy on the 96×96 target path at matched *training time*
(interpolated on each run's own clock; `R96` needs 542 s):

| arm | 25 % of R96's time | 50 % | 75 % | 100 % | final |
|---|---:|---:|---:|---:|---:|
| R96 | 40.8 | 49.5 | 56.2 | 58.6 | 57.7 |
| Rprog | 48.7 | 56.9 | 58.7 | ended at 78 % | 58.8 |
| Rsteps4 | 47.1 | 56.2 | 58.5 | ended at 84 % | 58.9 |
| Rlin24 | 47.1 | 54.8 | 57.9 | ended at 85 % | 58.1 |
| Rlin24eq | 45.1 | 53.2 | 58.7 | 60.1 | 60.3 |

Readings. (i) At equal updates the three 60-epoch progressive arms are
+0.4 to +1.2 pp above plain on average, with seed 1 negative in each case; with
5,000 training images the seed spread of the plain control alone is 2.2 pp, so
**three seeds do not rank the progressive schedules against each other** here,
and the CIFAR "fine ramp beats Rprog" result is neither confirmed nor
contradicted at this scale. (ii) At equal *time* the picture is not ambiguous:
every progressive arm is 7–8 pp ahead at a quarter of the budget, 5–7 pp ahead
at half, and matches or exceeds plain's *final* accuracy at three quarters. The
anytime curve of progressive resolution dominates plain's throughout. (iii) The
equal-compute ramp `Rlin24eq` is +2.6 pp over plain, positive on every seed, for
7 % more wall time — but it also runs 20 % more updates on a different LR
horizon, so this contrast confounds resolution with budget. The two controls
that separate them (`R96_72`: plain on the same 72-epoch horizon; `Rprogeq`:
the 48/72/96 schedule extended at 96 to `R96`'s compute) were launched as
`stl10-controls-*`; see §16.4.

### 16.3 Same mechanism as on CIFAR

Target-path (96×96) accuracy during the coarse stages is only a few points
below the current path (e.g. `Rprog` epoch 12 at r=48: current 39.1, target
34.0, target after BN recalibration 35.7), against a 26 pp collapse at
16→32 on CIFAR: a 48×48 STL-10 image carries far more of the 96×96 content than
a 16×16 CIFAR image carries of the 32×32 one, and BN recalibration recovers a
third to a half of what is lost. The generalisation gap (test CE − train-probe
CE) is 0.18 for plain and 0.13–0.15 for the 60-epoch progressive arms; at
60 epochs and 2,400 updates none of the arms is near the CIFAR memorisation
regime (train-probe CE stays around 1.0).

### 16.4 Equal-compute controls (`stl10-controls-20260917-133339`, 6 runs, 2,271 s)

| arm | epochs | nominal units | seeds 0 / 1 / 2 | mean | train s |
|---|---:|---:|---|---:|---:|
| R96 | 60 | 60.0 | 57.66 / 58.76 / 56.59 | 57.67 | 542 |
| R96_72 (plain, longer horizon) | 72 | 72.0 | 59.27 / 60.72 / 58.56 | 59.52 | 674 |
| Rlin24eq | 72 | 61.6 | 60.62 / 60.46 / 59.67 | 60.25 | 582 |
| **Rprogeq** (48 ×12 / 72 ×12 / 96 ×50) | 74 | 59.8 | 60.27 / 60.16 / 61.01 | **60.48** | 573 |

| paired contrast | per seed (pp) | mean | time |
|---|---|---:|---|
| Rprogeq − R96 (equal nominal compute) | +2.61 / +1.40 / +4.43 | **+2.81** | 573 vs 542 s |
| Rlin24eq − R96 (equal nominal compute) | +2.96 / +1.70 / +3.09 | **+2.58** | 582 vs 542 s |
| R96_72 − R96 (plain, +20 % updates) | +1.61 / +1.96 / +1.97 | +1.85 | 674 vs 542 s |
| Rprogeq − R96_72 | +1.00 / −0.56 / +2.45 | +0.96 | 573 vs 674 s (−15 %) |
| Rlin24eq − R96_72 | +1.35 / −0.26 / +1.11 | +0.73 | 582 vs 674 s (−14 %) |
| Rprogeq − Rlin24eq | −0.35 / −0.30 / +1.34 | +0.23 | |

What separates. Plain gains +1.85 pp from 20 % more updates on its own, so
roughly two thirds of `Rlin24eq`'s +2.6 over the 60-epoch plain is budget, not
resolution. Against the *same-horizon* plain (`R96_72`) the progressive arms
keep +0.7 to +1.0 pp while using 14–15 % less wall time. Reading plain's
accuracy against its own clock (57.7 % at 542 s, 59.5 % at 674 s), the
progressive arms at 573–582 s sit about **+2.3 pp above what plain reaches in
the same time**. The two progressive extensions do not differ (+0.23, mixed
signs). Seed 1 is the negative seed in every 60- and 72-epoch contrast except
the pure budget one, which is what a 5,000-image training set does to three
seeds; the STL-10 numbers are consistent with CIFAR in sign and mechanism, and
weaker in resolution.

### 16.5 What STL-10 says about "adaptive"

The reason to go to STL-10 was that coarse epochs would become cheap enough for
a time-aware controller to matter. They did (§16.1). But the trade it would
manage is now visible without a controller: at fixed updates, resolution buys
0.4–1.2 pp and 15–22 % of the time; at fixed time, it buys about 2.3 pp; the
choice of *which* coarse schedule is inside the seed noise. A controller could
only redistribute epochs between sizes, and Phase 2 on CIFAR showed that the
signals available to it do not detect the one thing that mattered there (equal
dwell). The honest recommendation stands: a fixed coarse-to-fine schedule with
the horizon chosen for the time budget, no trigger. If a controller is still
wanted, the STL-10 setting is the right place to test one — but the comparison
must be at equal *time* against the extended fixed schedules of §16.4, with more
than three seeds.

Not done on STL-10: no Gaussian filter, no augmentation, no pairing with the
`stl10-transfer` branch (different init assets), no unlabelled data.


---

## 17. A signal for scale specialisation, and a controller that bounds it

The user's target after §15–16: a quantity that measures how *specialised to
the current scale* the network has become, and a rule that moves the scale when
that quantity says so — so that the homotopy path is no longer a hand-written
table.

### 17.1 Definition

For the current resolution r and a probe step Δ, with the BatchNorm running
statistics rebuilt at r+Δ on the monitor set (fixed weights, restored
afterwards):

    g_Δ(θ) = [ L_{r+Δ}^{BN-recal}(θ) − L_r(θ) ] / L_r(θ)

the relative loss penalty the current network pays when shown the next scale,
with the part that is only BatchNorm statistics removed. Properties wanted from
a specialisation measure: zero for a scale-invariant predictor; growing with
dwell at a fixed scale; resetting when the scale moves; larger for a larger
jump. Cost: one BN recalibration pass plus one evaluation on 2,000 images per
probe step, ~1 % of an epoch.

A **label-free companion** is logged alongside it: the BN-statistics shift
between the training path and the recalibrated statistics at r+Δ (mean over
layers of |μ' − μ|/σ and of |log(σ'²/σ²)|), for the case where a controller
must not see training labels.

### 17.2 Replay on the recorded traces (all CIFAR runs of §12–15)

Every property above holds in the data:

| trajectory | g at consecutive epochs |
|---|---|
| 9 epochs at 16, Δ=+4 (`Rctrl*`, 3 seeds) | 0.00, 0.03, 0.02, **0.06 / 0.07 / 0.07**, 0.09, 0.13, 0.14, 0.20, 0.26; then 0.06 one epoch after moving to 20 |
| 6 epochs at 16, Δ=+8 (`Rprog`, 3 seeds) | 0.02, 0.06, 0.08, 0.10, 0.15, **0.21 / 0.26 / 0.17** at the switch |
| 12 epochs at 16, Δ=+8 (`Rb12_18`) | … 0.39, 0.42, 0.47, **0.63** at the switch |
| `Rsteps4` (3 epochs per size, Δ=+4) | never above 0.07 at a switch (0.02–0.04 at 16, 0.06 at 20, 0.06–0.09 at 24, 0.04–0.11 at 28) |
| `Rlin12` (Δ=+1/+2 each epoch) | −0.02 … 0.06, no trend |

Growth is close to linear in dwell (about +0.03 per epoch at 16 for Δ=4), the
+8 probe grows about twice as fast as the +4 probe, the reset after a switch is
immediate, and the three seeds agree to ±0.01 at the point where a threshold
would act. Most importantly **g at the moment of switching ranks the fixed
schedules by final accuracy**: Rlin12 / Rsteps4 (g ≤ 0.09) 80.0–80.1 %, Rprog
(0.21) 79.1 %, the Phase 2 floor schedule (0.26) 79.2 %, Rb12_18 (0.63) 77.5 %.
The gradient-alignment signal of §14–15 saw none of this: it measures the
usefulness of the coarse gradient, not the drift of the function.

### 17.3 Control law: bounded specialisation

    every epoch:  g ← g_{Δ}(θ)   (Δ = 4, EMA β = 0.5, re-seeded after a switch)
                  if g ≥ g*:  r ← min(r + Δ, 32)
    guard:        r ≥ 16 + 4·(e − 11) from epoch 11  (32 by epoch 15 at the latest)
                  horizon K = 11,730 and LR path unchanged

Read from the replay, g* ≈ 0.06 should give about 4 epochs at 16 then ~3 at
each of 20/24/28, i.e. reproduce the Rsteps4 pattern *from the signal*. The
interesting outcomes are the ones where it does not: a systematically longer
dwell at some scales would be the adaptive answer that a fixed table cannot
give. One hyperparameter (g*) replaces the schedule table; Δ is a design
constant here and can later be chosen from the g_Δ probes (logged for Δ ∈ {2,
4, 8} on every run).

### 17.4 Phase 3 — launched 2026-09-17, kernel `adaptive-phase3-20260917-142057`

Nine unfiltered CIFAR-10 runs, seeds 0/1/2, g* ∈ {0.04, 0.06, 0.08}, paired with
everything before. Readings fixed in advance:

| observation | reading |
|---|---|
| switches at the same epochs on all seeds, accuracy ≈ Rsteps4 / Rlin12 (80.0–80.1) | the law recovers the good fixed schedule from the signal: the table is gone, g* remains |
| dwell differs across scales systematically, accuracy ≥ Rsteps4 | genuine adaptive schedule design |
| dwell differs across seeds with accuracy ≥ Rsteps4 | per-trajectory adaptation (hardest to report; needs Phase 2-style hard-coding) |
| controller dominated by the guard, or accuracy ≤ Rprog | the signal is right but the law is wrong (threshold / step); revisit with the Δ-probes |

### 17.5 Phase 3 results (2,893 s, 9 runs, 0 failures, pinned digests matched, measurements pure)

**The signal fires.** 35 of the 36 switches were made by the gap rule; one (g* =
0.08, seed 2, the last step to 32) by the guard. Realised dwell per size
(epochs at 16 / 20 / 24 / 28, then 32 from the epoch shown):

| g* | seed 0 | seed 1 | seed 2 | acc per seed | mean | vs Rprog | vs Rsteps4 | vs Rlin12 |
|---|---|---|---|---|---:|---:|---:|---:|
| 0.04 | 5/1/5/1 → 32 @12 | 4/3/2/4 → @13 | 4/3/3/1 → @11 | 79.61 / 79.85 / 80.49 | 79.98 | +0.84 | −0.04 | −0.16 |
| **0.06** | 6/2/3/2 → @13 | 5/2/6/1 → @14 | 6/2/3/1 → @12 | 79.86 / 79.83 / 80.38 | **80.02** | **+0.88** | **0.00** | −0.12 |
| 0.08 | 6/3/3/3 → @15 | 6/3/4/2 → @15 | 7/3/3/2 → @15 | 79.09 / 79.68 / 80.56 | 79.78 | +0.64 | −0.24 | −0.36 |

References on the same seeds: Rprog 79.14, Rsteps4 80.02, Rlin12 80.14 (§14.1).
All nine controller runs beat Rprog (+0.5 to +1.2 pp per seed); none beats the
fixed fine ramps. This is the first pre-registered reading of §17.4: **the law
recovers the good schedule from the signal, so the schedule table is replaced by
one threshold**, and that threshold is not fragile — 0.04 and 0.06 are
indistinguishable, 0.08 (which tolerates more specialisation and reaches 32
only at epoch 15) costs about 0.25 pp, in the direction the signal predicts.

**What the controller chose is not equal dwell.** Across all thresholds and
seeds the first scale gets the longest stay (4–7 epochs at 16) and the later
scales 1–4 epochs each: the network starts unspecialised (g = 0 at
initialisation), so the gap needs longer to reach g* at the first scale, and
resets to a non-zero value after each move. That is the "adaptive answer"
row of §17.4, with the qualification that it did not translate into accuracy
above the fixed ramps: the timing differs across seeds by one to two epochs
without visible effect on the result, consistent with §12.6's finding that the
accuracy surface is flat around the fine schedules.

**The label-free companion does not work.** The BN-statistics shift toward
r+Δ is flat over dwell (0.023–0.025 at r=16 from epoch 1 to 6, while g goes
from 0.00 to 0.10) and simply decreases with r: it measures the fixed
resolution mismatch of the statistics, not the drift of the function. A
label-free specialisation measure has to look at the weights' function, not at
BatchNorm.

### 17.6 Where this leaves "adaptive resolution"

There is now a signal that measures scale specialisation — the BN-recalibrated
relative loss penalty at the next scale — and a one-parameter law that uses it
to drive the resolution during training, with the horizon fixed. Measured on
CIFAR-10, three seeds, three thresholds: it decides 97 % of the switches itself,
reproduces the accuracy of the best hand-written schedules (+0.9 pp over the
campaign's 16/24/32 arm), chooses a non-uniform dwell no table proposed, and is
insensitive to its threshold over a factor of two. What it does not do is
exceed the fixed fine ramps; on this recipe the accuracy surface around them is
flat, so no scheduler can. The value of the controller is that the path is no
longer a hyperparameter to tune per dataset — which is precisely the claim to
test next on STL-10 (§16), where the fixed schedules could not be ranked and a
data-chosen dwell would be informative rather than merely equivalent.


---

## 18. Two-sided bounded specialisation — can the controller beat the fixed ramps?

The user's requirement after §17: not equal the fine ramps, beat them. Every
member of the ramp family, fixed or controlled, sits at 80.0–80.1 %, so timing
alone cannot do it; the controller needs a degree of freedom the ramps lack.

### 18.1 The degree of freedom: fine-scale specialisation after the ascent

The mechanism identified in §12.6 is generalisation: plain training memorises
(train CE 0.24 / test CE 0.75), the ramps halve the gap, and then partially
rebuild it during the 18 epochs at 32 (train-probe CE 0.71 → 0.36 while test
accuracy is flat from epoch 24). The specialisation signal of §17 can be read
*downward*: g(32→24) = [L_24^{recal} − L_32] / L_32 measures how much the
network at 32 has come to rely on detail that 24×24 does not carry. On the
`Rmixed` traces (the only runs that logged it through the 32 phase, three
seeds) it is ≈ 0 when 32×32 training begins (epoch 17–18), grows by ≈ 0.05 per
epoch to 0.40–0.51 by epoch 28, and saturates with the LR — while test accuracy
stops improving around epoch 24–26. Fine-scale specialisation keeps growing
after generalisation has stalled. No fixed ramp acts on it.

### 18.2 Law

    ascent:   as §17.3 (Δ = 4, g* = 0.06, floor from epoch 11)
    at 32:    g_down ← g(32→24), EMA β = 0.5, re-seeded after each reheat
              if g_down ≥ g*_down and more than `settle` = 3 epochs remain:
                  train the next epoch at 24 ("reheat"), then return to 32
    horizon and LR path unchanged; the last three epochs are always at 32

g*_down ∈ {0.15, 0.30} (about one reheat every 3–4 epochs vs one or two in
total, from the growth rate above). Fixed control for the *mechanism*:
`Rsteps4rh` = Rsteps4 with one 24×24 epoch at epochs 20, 24 and 28, so that a
gain from coarse epochs late in training can be separated from the gain, if
any, of choosing their timing from the signal.

### 18.3 Phase 4 — launched 2026-09-17, kernel `adaptive-phase4-20260917-151733`

Nine unfiltered CIFAR-10 runs, seeds 0/1/2: `Rgap2s15`, `Rgap2s30`, `Rsteps4rh`.
Comparators on the same seeds: Rlin12 80.14 (79.70 / 79.97 / 80.74), Rsteps4
80.02 (79.80 / 79.54 / 80.71), Rgap06 80.02 (79.86 / 79.83 / 80.38). Readings
fixed in advance: a paired gain ≥ +0.3 pp over Rsteps4 *and* Rlin12 on all
three seeds is the target; a gain in `Rsteps4rh` but not in the controller
means the mechanism works and the timing rule does not; a loss in both means
late coarse epochs under a small LR do not regularise and this degree of
freedom is closed on this recipe.

### 18.4 Results — first launch (`adaptive-phase4-20260917-151733`) and re-run (`adaptive-phase4b-20260917-154529`)

The six controller runs of the first launch failed before training on a
dispatch bug (the new controller kind fell through to the θ-branch and raised
`KeyError: 'theta'`); they are marked `INVALID.json` and were re-run after the
fix. The three fixed-control runs of the first launch are valid.

**The mechanism works.** `Rsteps4rh` (Rsteps4 + one 24×24 epoch at epochs 20,
24, 28): **80.32 / 80.54 / 81.22, mean 80.69 %** — +0.52 / +1.00 / +0.51 over
Rsteps4 and +0.62 / +0.57 / +0.48 over Rlin12, positive on every seed, and above
the filtered campaign winner measured in this protocol (Rprog + Gplateau 80.48,
§14.1) at 68 % of its training time. Each reheat resets g(32→24) from 0.6–0.7 to
slightly negative, lifts the train-probe CE by ≈ 0.15–0.2, and the next 32×32
epoch comes back *above* the pre-reheat test accuracy (seed 0: 79.3 → 79.8 →
80.2 → 80.4 across the three reheats). Late coarse epochs under a small LR do
regularise.

**The controller beats the fixed ramps, at the right bound.**

| arm | reheats (seeds 0/1/2) | acc per seed | mean | vs Rsteps4 | vs Rlin12 | vs Rsteps4rh |
|---|---|---|---:|---:|---:|---:|
| Rgap2s15 (g*_down 0.15, settle 3) | 5 / 4 / 5 | 79.65 / 79.47 / 80.69 | 79.94 | −0.08 | −0.20 | −0.75 |
| **Rgap2s30** (g*_down 0.30, settle 3) | 3 / 4 / 4 | 80.18 / 80.00 / 80.99 | **80.39** | **+0.37** (+0.38 / +0.46 / +0.28) | **+0.25** (+0.48 / +0.03 / +0.25) | −0.30 |

At g*_down = 0.30 the controller is above Rsteps4 on all three seeds and above
or equal to Rlin12 on all three, choosing 3–4 reheats at seed-dependent epochs
(19–26). At 0.15 it reheats every other epoch from epoch 14 and
over-regularises. The pre-registered target (≥ +0.3 pp over *both* ramps on all
seeds) is met against Rsteps4 and met on average but not on every seed against
Rlin12.

**Why the fixed control still wins by 0.3 pp.** It reheats when the gap has
reached 0.65–0.71 and keeps doing so up to epoch 28; the controller at 0.30
fires at 0.34–0.40, soon after arriving at 32, and the `settle = 3` rule
forbids the last three epochs. Both are threshold choices, so a higher bound
with `settle = 1` (g*_down ∈ {0.45, 0.60}) was launched as
`adaptive-phase4c-20260917-165354`; §18.5.

### 18.5 Higher reheat bounds (`adaptive-phase4c-20260917-165354`, 6 runs, settle = 1)

| arm | reheats (seeds 0/1/2) | acc per seed | mean | vs Rsteps4 | vs Rlin12 | vs Rsteps4rh |
|---|---|---|---:|---:|---:|---:|
| Rgap2s45 | 3 / 2 / 3 | 79.46 / 79.56 / 79.93 | 79.65 | −0.37 | −0.49 | −1.04 |
| Rgap2s60 | 2 / 1 / 2 | 79.89 / 79.62 / 80.44 | 79.98 | −0.04 | −0.16 | −0.71 |

Raising the bound did not close the gap to the fixed control; it widened it.
Across the four bounds the mean is 79.94 / **80.39** / 79.65 / 79.98 for
0.15 / 0.30 / 0.45 / 0.60 — non-monotone, with 0.45 below both neighbours.
With three seeds and a measured run-to-run noise of up to 0.7 pp on identical
schedules (§15.2), this ranking is not resolvable; the one setting that beat the
ramps on every seed may be the upper tail of that noise. What *is* resolved is
that `Rsteps4rh` — fixed ascent, three reheats at 20/24/28 — is above every fine
ramp on every seed, and above every controller variant.

Two confounds separate the controller from the fixed control: the controller
also adapts the ascent (reaching 32 at epoch 13–15 instead of 12, so fewer
32×32 epochs remain once reheats are subtracted), and its reheats cluster soon
after arrival. Phase 5 removes the first confound and doubles the seeds.

### 18.6 Phase 5 — launched 2026-09-18, kernel `adaptive-phase5-20260918-055940`

Seeds 0–5 (0–2 verified against the pinned assets, 3–5 new, paired within the
job). Arms: `Rsteps4ar` = Rsteps4 ascent + adaptive reheats at g_down ≥ 0.30,
settle 2 (seeds 0–5); `Rsteps4rh` and `Rsteps4` (seeds 3–5, completing six);
`Rgap2s30` (seeds 3–5, completing six). The question is now sharp: on six
paired seeds, does choosing the reheat epochs from the signal beat choosing
them by a fixed rule, and does either beat the plain ramp?

### 18.7 Phase 5 results (`adaptive-phase5-20260918-055940`, 15 runs, 4,499 s, 0 failures; seeds 0–2 verified against the pinned assets, 3–5 regenerated and paired within the job)

Unfiltered CIFAR-10, final test accuracy (%), seeds 0–5:

| arm | 0 | 1 | 2 | 3 | 4 | 5 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rsteps4 (fixed fine ramp) | 79.80 | 79.54 | 80.71 | 79.34 | 79.88 | 80.20 | 79.91 |
| Rsteps4rh (ramp + fixed reheats 20/24/28) | 80.32 | 80.54 | 81.22 | 79.44 | 80.28 | 80.44 | 80.37 |
| **Rsteps4ar (ramp + adaptive reheats, g_down ≥ 0.30)** | 80.79 | 79.73 | 81.28 | 80.12 | 80.81 | 80.40 | **80.52** |
| Rgap2s30 (adaptive ascent + adaptive reheats) | 80.18 | 80.00 | 80.99 | 79.54 | 80.59 | 80.16 | 80.24 |

| paired contrast | per seed (pp) | mean ± sd | positive |
|---|---|---:|---:|
| **Rsteps4ar − Rsteps4** | +0.99 / +0.19 / +0.57 / +0.78 / +0.93 / +0.20 | **+0.61 ± 0.35** | **6/6** |
| Rsteps4rh − Rsteps4 | +0.52 / +1.00 / +0.51 / +0.10 / +0.40 / +0.24 | +0.46 ± 0.31 | 6/6 |
| Rgap2s30 − Rsteps4 | +0.38 / +0.46 / +0.28 / +0.20 / +0.71 / −0.04 | +0.33 ± 0.25 | 5/6 |
| Rsteps4ar − Rsteps4rh | +0.47 / −0.81 / +0.06 / +0.68 / +0.53 / −0.04 | +0.15 ± 0.55 | 4/6 |
| Rsteps4ar − Rgap2s30 | +0.61 / −0.27 / +0.29 / +0.58 / +0.22 / +0.24 | +0.28 ± 0.32 | 5/6 |

The adaptive-reheat controller chose 4–5 reheats per run, at seed-dependent
epochs between 14 and 27, spaced 2–3 epochs apart (e.g. seed 0: 15, 18, 20, 22,
25; seed 4: 15, 17, 20, 22, 26). The paired gain over the fixed fine ramp is
+0.61 pp with six of six seeds positive (paired t ≈ 4.3). Against the
hand-placed reheat schedule it is +0.15 with four of six positive — not
resolved, i.e. the signal places the reheats at least as well as a tuned fixed
rule without being told when. The full two-sided controller is +0.33 (5/6):
adapting the ascent as well costs about 0.3 pp against the fixed Rsteps4 ascent
(§17.5: the ascent controller reaches 32 one to three epochs later).

### 18.8 Conclusion

**The adaptive controller beats the fixed fine ramps.** Not by timing the
ascent — every ascent, fixed or adaptive, sits on the same 80.0–80.1 %
plateau — but by acting after it, on the one quantity the ramps leave
uncontrolled: fine-scale specialisation during the 32×32 phase. The signal is
the BN-recalibrated relative loss penalty at a coarser scale, g(32→24); the law
is "insert one 24×24 epoch whenever it exceeds 0.30, never in the last two
epochs"; the result on six paired seeds is +0.61 pp over the best fixed ramp,
80.52 % against 79.91 %, positive on every seed, at the same update budget and
the same wall time, with no filter. It also exceeds the campaign's filtered
winner (Rprog + Gplateau, 80.48 % in this protocol) at 68 % of its training
time. The two remaining hyperparameters are the ascent table (or g* = 0.06 for
the adaptive ascent, at −0.3 pp) and one bound g*_down, whose useful range is
narrow (0.15 over-regularises, 0.45–0.60 under-regularise on three seeds).

What this does not establish: robustness of g*_down beyond CIFAR-10 / ResNet-20
/ no augmentation; behaviour with the Gaussian filter; behaviour on STL-10,
where the reheat epochs are also cheaper. Those are the next experiments, in
that order.


---

## 19. STL-10 — adaptive reheats and a joint step-size controller (launched 2026-09-18, kernel `paulinezarka/stl10-adaptive-20260918-115842`)

### 19.1 Two facts from the multi-step probes that shape the design

Replaying the logged g_Δ for Δ ∈ {2, 4, 8} along the six fixed-ascent runs of
§18.7 shows that **g is not monotone in the step**: from r = 16, the +2 step
(to 18) costs 0.10–0.14 while +4 (to 20) costs 0.03 and +8 (to 24) 0.06; from
28, +2 (to 30) costs 0.20 and +4 (to 32) 0.04. The expensive sizes are those
not divisible by 4: after the two stride-2 stages, 18 → 9 → 5 lands on a
shifted sampling grid, 20 → 10 → 5 does not. The network is specialised to the
**phase of its sampling grid** as much as to scale. Consequence: candidate
sizes must be restricted to the grid compatible with the architecture's strides
(multiples of 4 here), otherwise the signal measures an artefact. This is also
the mechanism behind the parity effect of §14.2.

Second, on the three-epoch dwells of the fixed ramp, g(+4) stays within the
noise (−0.02 to 0.06, noise ≈ ±0.02); and a step rule "largest Δ with
g_Δ ≤ 0.10" would take +8 from 16 at epoch 2–3, reconstructing the short
16 → 24 jumps that the §12.6 grid found worse. The immediate transfer penalty
does not capture everything the small step buys, so g_jump must be tight.

### 19.2 Design

Equal compute for every arm (72-epoch horizon ≈ 60 units of a 96×96 epoch, as
`R96`'s 60 epochs), six seeds, STL-10 protocol of §16.

| arm | ascent | fine phase |
|---|---|---|
| `Rsteps4_72` | fixed 48/60/72/84 ×6, 96 from epoch 24 | none (comparator) |
| `Rsteps4ar_72` | same fixed ascent | adaptive reheats at 72×72 when g(96→72) ≥ 0.30, settle 2 |
| `Rjoint_72` | **joint**: start 48; candidates r+12 / r+24 / r+48 (all multiples of 4, ≤ 96); *timing* — advance when the EMA of g toward the smallest candidate ≥ g_up = 0.05 after ≥ 2 epochs; *step* — the largest candidate with g_Δ ≤ g_jump = 0.05, else the smallest; floor r ≥ 48 + 12·(e−18), 96 by epoch 30 | same adaptive reheats |

Every probe is one BN recalibration plus one evaluation on the 1,000-image
monitor set at the candidate size, weights and buffers verified restored.
Comparators already measured on seeds 0–2 (§16.4): `R96_72` 59.52, `Rprogeq`
60.48, `Rlin24eq` 60.25.

Readings fixed in advance: (i) `Rsteps4ar_72 − Rsteps4_72` ≥ +0.5 pp on ≥ 5/6
seeds → the reheat mechanism transfers; (ii) `Rjoint_72` ≈ `Rsteps4ar_72` with
seed-stable steps → the ascent table can be dropped without cost; `Rjoint_72`
below it → adaptive ascent costs here as on CIFAR; (iii) both ≈ `Rsteps4_72` →
the STL-10 fine phase is not specialising enough at 40 updates per epoch for
reheats to matter.

### 19.3 Results (`paulinezarka/stl10-adaptive-20260918-115842`, 18 runs, 6,518 s, 0 failures)

| arm | seeds 0–5 (%) | mean | train s |
|---|---|---:|---:|
| Rsteps4_72 (fixed fine ramp, equal compute) | 60.82 / 60.52 / 60.50 / 61.81 / 61.06 / 60.90 | **60.94** | 537 |
| Rsteps4ar_72 (+ adaptive reheats) | 60.88 / 60.59 / 60.30 / 61.36 / 60.68 / 60.50 | 60.72 | 546 |
| Rjoint_72 (joint step-size controller + reheats) | 60.26 / 59.90 / 60.41 / 61.46 / 61.24 / 60.77 | 60.68 | 503 |

Paired: reheats − fixed = −0.22 ± 0.23 (2/6 positive); joint − fixed =
−0.26 ± 0.31 (1/6); joint − reheats = −0.04 ± 0.50. Earlier comparators on seeds
0–2: `R96_72` 59.52, `Rprogeq` 60.48, `Rlin24eq` 60.25 — the fixed fine ramp at
equal compute is the **best STL-10 arm so far** (60.61 on the same three seeds).

**Both controllers were dominated by their guards.** The reheat rule fired 0–1
times per run (one late reheat on seeds 3–5, each time slightly harmful); the
joint ascent never triggered on the signal and its floor produced 48 ×19 → 60 →
72 → 84 → 96 from epoch 22 on every seed. The traces say why, and it is not a
failure of the signal: g grows at the same rate **per update** as on CIFAR
(≈ 1.1–1.3 × 10⁻⁴ per update in both datasets) but STL-10 has 40 updates per
epoch instead of 391, so the fine phase reaches g(96→72) ≈ 0.2–0.3 only in the
last epochs, and g(+12) during the 48 stage stays within ±0.04 for nineteen
epochs. Thresholds calibrated on CIFAR in *epochs* do not transfer; the
quantity that transfers is the specialisation rate per update, and at 2,400
updates STL-10 simply never specialises enough for reheats to pay. This is the
third pre-registered reading of §19.2.

What this establishes and what it does not. Established: the reheat mechanism
needs a fine phase long enough (in updates) to specialise; at STL-10's budget it
does not exist, so the fixed ramp is the right answer there. Not established:
whether the joint controller would work with thresholds set per update (e.g.
g_up ≈ 0.02 and a probe every 5 epochs) — the floor took every decision, so its
step rule was never exercised. A fair test needs either a longer STL-10 horizon
(≥ 10,000 updates) or thresholds scaled by the update budget.

### 19.4 Overall status of "adaptive resolution", end of 2026-09-18

* CIFAR-10 (11,730 updates): a fixed fine ramp plus **adaptive reheats** driven
  by the specialisation signal beats the best fixed ramp by +0.61 pp on six
  paired seeds (§18.7). This is the method to keep.
* Adaptive *ascent*, in any form tested (plateau, alignment, specialisation
  threshold, joint step size): never better than the fixed ramp, sometimes
  worse by 0.3 pp; on STL-10 the thresholds did not fire at all.
* STL-10 (2,400–2,880 updates): progressive resolution buys real wall time and
  about +2.3 pp at equal time; the fixed fine ramp at equal compute (60.94 %)
  is the best arm; reheats have nothing to correct at this budget.
* One discovery worth keeping for any future controller: candidate sizes must
  be compatible with the network's stride pattern (multiples of 4 for
  ResNet-20); otherwise the signal measures a sampling-grid artefact (§19.1).
