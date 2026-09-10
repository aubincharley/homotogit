# Gradient-norm plateau trigger: calibration result

Run: `paulinezarka/gradnorm-cal-20260910-105845`, seed 0, ~10 min T4.
Arm: `R32__Gplateau__input_bilinear__all19`, instrumentation only, no controller.

**Validity check.** Final test acc **0.7797** against the recorded **0.7791** for
the same arm — 0.06 pp, inside the ~0.1 pp single-seed nondeterminism floor. The
logging did not perturb training. Pairing verified in-job: all four campaign
digests reproduced.

## Result: the signal has no plateau to detect

The global gradient norm **rises** across training rather than falling:

| epoch | 0 | 6 | 12 | 18 | 22 | 28 |
|---|---:|---:|---:|---:|---:|---:|
| sigma | 1.00 | 0.70 | 0.50 | 0.30 | 0.00 | 0.00 |
| mean \|\|g\|\| | 2.39 | 3.15 | 3.13 | 3.08 | 3.39 | 3.40 |

Overall ratio final/first = **1.43**. It oscillates in a band of roughly 2.2-5.0
throughout, and is largely **insensitive to sigma** — the schedule moves 1.00 to
0, the gradient norm does not respond.

The relative decrease of the EMA over a window is therefore typically
*negative* (the EMA usually rises), and:

| window | median delta | frac(delta < 0.02) | frac(delta < 0.05) |
|---:|---:|---:|---:|
| W=3 | -0.0097 | 89.6% | **100%** |
| W=5 | -0.0146 | 87.6% | **100%** |
| W=10 | -0.0309 | 83.3% | **100%** |

**Every** window at eps >= 0.05 counts as "plateaued". The condition is
satisfied essentially always, so it carries no information.

## Consequence: every setting degenerates to a fixed schedule

Replaying the trigger offline over cadence x tol x patience x window, the gap
between consecutive fires always equals the *structural minimum* of the setting:

| rule | setting | steps | median gap | = |
|---|---|---:|---:|---|
| best (as shipped) | cadence 1, tol .01, pat 30, min 40 | 225 | 45 | `min_steps` |
| best | cadence 50, tol .01, pat 30 | 6 | 1700 | `cadence x patience` |
| window | cadence 50, eps .02, W 3, pat 2 | 44 | 250 | `cadence x (W+pat)` |
| window | cadence 100, eps .05, W 5, pat 2 | 16 | 700 | `cadence x (W+pat)` |
| window | cadence 200, eps .10, W 5, pat 2 | 8 | 1400 | `cadence x (W+pat)` |

`eps` is nearly irrelevant: 0.02, 0.05 and 0.10 give the same step counts. The
trigger fires the instant its history buffer is long enough, every time. Whatever
values are chosen, the result is a **hand-designed schedule whose spacing is set
by the hyperparameters, not by the data** — which is what the first adaptive run
did (141 of 142 fires at exactly `min_steps`).

## Diagnosis: BatchNorm scale invariance

For a BN network the loss is invariant to the scale of each layer's weights:
`L(cw) = L(w)`, hence `grad L(cw) = grad L(w) / c`. So

    ||g||  scales as  1 / ||w||.

Weight decay shrinks `||w||` throughout training, which **mechanically inflates
`||g||`** — independently of whether optimisation is converging. The measured
1.43x rise is that effect, and it swamps whatever convergence signal exists.

This was flagged as a risk in the plan (section 2.3, "BatchNorm makes `||g||`
weight-scale dependent... weight norms drift, so the signal drifts for reasons
unrelated to convergence"). The calibration shows it does not merely add noise:
it **dominates**.

## The fix, and what it costs to test

The signal must be made scale-invariant. For a scale-invariant loss the invariant
combination is the product, not the raw norm:

    ||g|| . ||w||        (or per-layer,  sum_l ||g_l|| . ||w_l||)

equivalently the relative update size `||dw|| / ||w||`. Per-layer is likely
better than global, since only the BN-preceding layers are scale invariant.

Testing this needs **one more ~10 min run** logging `||w||` (global and per
layer) alongside `||g||`, on the same arm. Everything after that is free offline
replay against the recorded trace, exactly as here.

Do **not** re-tune `tol`/`patience`/`eps` on the raw norm: the table above shows
there is nothing there to tune.

## Follow-up: the scale-invariant fix was tested, and it fails

Run `paulinezarka/gradnorm-cal2-20260910-111547`, same arm, ~11 min.
Final test acc **0.7794** (references: 0.7791 recorded, 0.7797 first calibration)
— three independent reproductions inside the noise floor.

Logged per update: `||g||_all`, `||g||_conv`, `||w||_conv`, `||w||_all`, lr; plus
per-layer norms for all 21 conv kernels every 50 updates. Only the conv kernels
are scale invariant (each is followed by BN), so BN affines and the classifier
were kept separate rather than pooled in.

### The predicted mechanism is real but far too small

| signal | epoch 0 | epoch 28 | ratio | |
|---|---:|---:|---:|---|
| `\|\|g\|\|` raw | 2.380 | 3.446 | 1.448 | rises |
| `\|\|g\|\|_conv` | 1.930 | 3.378 | 1.750 | rises |
| `\|\|w\|\|_conv` | 36.89 | 33.15 | **0.899** | **decreases, as predicted** |
| `\|\|g\|\|_conv . \|\|w\|\|_conv` | 71.15 | 111.98 | **1.574** | **rises — worse than raw** |

Weight decay does shrink the conv weights, and the BN relation `||g|| ~ 1/||w||`
does operate. But it accounts for only `1/0.899 = 1.11x` of the `1.75x` rise in
`||g||_conv`. The residual `1.57x` is **genuine growth of the scale-invariant
gradient**. So the earlier diagnosis was directionally right and quantitatively
wrong: BN scaling is a minor term, not the cause.

The proposed fix is therefore **falsified**. Every trigger variant stays pinned
at the structural minimum gap of `cadence x (W+patience) = 700`:

| signal | eps=0.005 | eps=0.02 | eps=0.05 |
|---|---|---|---|
| `\|\|g\|\|` raw | 77% pinned | 100% | 100% |
| `\|\|g\|\|_conv` | 77% | 100% | 100% |
| `\|\|g\|\|_conv . \|\|w\|\|_conv` | 83% | 100% | 100% |

### The actual reason: nothing converges within a stage

Probe cross-entropy on this run falls **monotonically at every single epoch**,
including the last:

    ep 20  0.5252     ep 24  0.4424     ep 28  0.4150
    ep 22  0.4595     ep 26  0.4203     ep 30  0.4112

It is still decreasing at epoch 30. There is no epoch at any sigma where the
corrector has stopped making progress.

Predictor-corrector continuation assumes each subproblem is **solved to
tolerance** before the parameter is stepped. Within a 30-epoch budget on 50k
CIFAR-10 under a cosine schedule, the corrector never converges at any sigma. So
a convergence trigger has nothing to detect — the problem is not the statistic,
it is that the **premise does not hold in this compute regime**.

### What follows

* Do not tune further statistics of gradient magnitude. Four variants, three
  tolerances, four cadences: all uninformative for the same structural reason.
* A "converge then step" rule needs either a much longer per-stage budget (which
  changes the comparison being made) or replacement by a rule that is not
  convergence-based at all — e.g. a cost-benefit rule stepping when marginal
  improvement per unit compute at the current sigma falls below what the next
  sigma offers. That still needs a measurable, and may hit the same wall.
* On present evidence the fixed schedule is the right choice for this budget:
  it reproduces at 0.7791 / 0.7797 / 0.7794, while the adaptive attempt reached
  0.7634.

## Files

- trace (11,730 rows): `results/kaggle_outputs/gradnorm-cal-20260910-105845/campaign_job0_20260910-105911/Gplateau_cal__seed0/grad_norm_trace.json`
- summary with `final_test_acc`: same directory, `summary.json`

---

# Part 2 — the transfer-gap trigger

## The signal (measured, `signals-cal3`)

`gap(t) = L_target(theta_t) - L_current(theta_t)` on the fixed probe, *within*
each constant-sigma stage of the fixed `Gplateau` arm:

| sigma | gap across the stage | |
|---|---|---|
| 1.00 | 1.67 -> 2.36 -> 2.97 | **grows** |
| 0.85 | 2.77 -> 3.64 -> 4.51 | **grows** |
| 0.70 | 3.53 -> 3.77 -> 4.94 | **grows** |
| 0.50 | 1.34 -> 1.19 -> 1.11 | shrinks |
| 0.40 | 0.214 -> 0.179 -> 0.176 | shrinks |
| 0.30 | 0.0091 -> 0.0058 -> 0.0050 | shrinks |

Sign change near sigma = 0.5. Unlike every magnitude signal, this has real
structure and a natural terminus (gap = 0 at sigma = 0). Trigger:
`fire when (gap[-1-W]-gap[-1])/|gap[-1-W]| < eps`. Replayed on the recorded
trace it fires **11** times with **0/10** gaps pinned at the dwell floor,
against 141/142 pinned for the gradient-norm trigger.

The other two candidates failed: gradient cosine is structureless (0.007-0.02,
no pattern); `||dtheta||/||theta||` merely tracks the cosine learning rate.

## Results

| arm | test acc | realised trajectory |
|---|---:|---|
| plain `Gnone` | 0.7513 | — |
| gap trigger, stepper mis-set | 0.7539 | 1.00 -> 0.84 over 17 ep, then crash to 0 |
| adaptive, grad-norm trigger | 0.7634 | -> floor by epoch 5 |
| **gap trigger, corrected** | **0.7713** | 1.00 -> 0.62 over 17 ep, ramp to 0 |
| fixed `Gplateau` | **0.7791** | 1.00 -> 0.30 over 21 ep, smooth |

(`Gplateau` reproduced at 0.7797 / 0.7794, and 0.7833 with mid-epoch probe
evaluation — the gap-trigger runs carry that same instrumentation, so read
0.7713 against roughly 0.78, not 0.7791.)

## Reading: the schedule's *shape* is what works

Every adaptive variant **under-traverses** the path and then deadline-ramps to
zero, and the results order monotonically by how close the realised trajectory
comes to a smooth complete traversal. Each fix moved the number toward the fixed
schedule from below without passing it. `deadline_fired` is true in all three
adaptive runs, so by the module's own standard each is partly a fixed schedule.

The binding arithmetic: 8 gap-triggered fires x ~0.045 per step = 0.36 of
descent in 17 epochs, against the ~1.0 the path needs. Trigger pace times step
size does not cover the path in the budget, so the terminal ramp does the rest
in three epochs.

Note the per-site scaling *is* engaging here (step 0.0556 at sigma = 1 implies
`r = 1/kappa = 1/3`, i.e. that site's sensitivity was above 3x the median) —
correcting an earlier claim that `r = 1` everywhere.

## What would be a different experiment, not another tweak

Raising the fire rate or step size would close the remaining gap by making the
controller reproduce `Gplateau` more exactly, which answers nothing. The open
questions that are not reparameterisations:

* Is there any trajectory that **beats** a smooth complete traversal, or is the
  fixed schedule optimal for this budget? Three adaptive attempts all approach
  it from below.
* Does the gap trigger help where a fixed schedule cannot be hand-tuned — e.g.
  a new dataset or a longer budget where the right dwell is unknown?
* Isolate the BN term: recalibrate BN before the target evaluation, so the gap
  measures function specialisation alone.
