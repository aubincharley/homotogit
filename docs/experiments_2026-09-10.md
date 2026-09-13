# Runs of 2026-09-10 — what varied, and what each comparison licenses [AI-Generated]

Every run below is seed 0, ResNet-20 + BatchNorm, official CIFAR-10 50,000 /
10,000 split, no augmentation, effective batch 128 as 4x32 accumulation, SGD
momentum 0.9 / weight decay 5e-4, warmup 60 then cosine over the whole budget.

**Pairing.** Every job regenerates the shared assets (class-balanced subset,
fixed train probe, per-epoch permutations, initial weights) and **aborts before
training** unless they reproduce the completed campaign's four pinned sha256
digests. So each arm shares initial weights and minibatch order with the
recorded controls, and differences are paired differences. The 120-epoch job
checks `perms[:30]` instead of the whole array, because a 120-row permutation
array cannot hash to a 30-row digest; matching the first 30 rows proves the same
generator stream.

**Noise floor.** The campaign measured ~0.1 pp single-seed variation from
float32 reduction order. Two runs define no distribution, so treat it as an
order of magnitude, not a threshold. Independent reruns of the *same* fixed arm
here gave 0.7791 / 0.7797 / 0.7794 / 0.7833.

---

## Group A — adaptive sigma: does a trigger beat a fixed schedule?

**Held fixed:** 30 epochs, LR 0.005, constant resolution 32, Gaussian at all 19
main-path convolutions, seven levels 1.00 -> 0.30 available, exact identity
endpoint, nine terminal epochs at sigma = 0.

**Varied:** *only* how the transitions between sigma levels are chosen.

| arm | how transitions are chosen | test acc |
|---|---|---:|
| plain | no filtering at all | 0.7513 |
| fixed `Gplateau` | fixed table, 3 epochs per level | **0.7791** |
| adaptive, gradient-norm trigger | fire when the grad-norm EMA plateaus | 0.7634 |
| adaptive, gap trigger (stepper mis-set) | fire when the transfer gap stops shrinking | 0.7539 |
| adaptive, gap trigger (corrected) | same trigger, stepper at its defaults | 0.7713 |

**What this licenses.** That *these* controllers, at *these* settings, did not
beat the fixed table. It does **not** license "adaptive scheduling does not
work": diagnostics showed both adaptive runs were dominated by the step
*magnitude*, not the trigger — the first fired 141/142 times at exactly its
`min_steps` floor, and both realised trajectories under-traversed the path and
were finished by the deadline ramp. Group B explains why that particular
failure is costly.

---

## Group B — allocation search: is uniform dwell optimal?

**Held fixed:** everything in Group A, *and* the level set, the level order, the
21 pre-bypass epochs and the 9 terminal epochs. No controller anywhere — these
are plain per-epoch level tables.

**Varied:** *only* how the 21 pre-bypass epochs are distributed across the seven
levels.

| allocation | dwell per level | test acc | vs uniform | probe acc |
|---|---|---:|---:|---:|
| front-loaded | 6,5,4,2,2,1,1 | 0.7657 | −1.34 pp | 0.810 |
| uniform (= `Gplateau`) | 3,3,3,3,3,3,3 | 0.7791 | — | 0.882 |
| **back-loaded** | 1,1,2,2,4,5,6 | **0.7874** | **+0.83 pp** | 0.916 |

**What this licenses.** Allocation shape matters — a 2.17 pp spread, ~20x the
noise floor — and **uniform is not optimal**: dwell near the target, rush the
strongly-deformed levels. The probe accuracies give the mechanism: front-loading
underfits (0.810), because most of the budget is spent on a heavily blurred
problem that is then abandoned.

This also retro-explains Group A. Every adaptive sigma run accidentally
front-loaded, and their scores interleave on exactly that axis:

    front-loaded 0.7657  <  gap trigger corrected 0.7713  <  uniform 0.7791
                                                         <  back-loaded 0.7874

with the worst adaptive run (0.7539, 17 epochs above sigma 0.84) below even the
front-loaded table. The gap trigger's *intent* was right — it fires when the gap
grows, which happens at high sigma, i.e. "leave here quickly", which is
back-loading — but the stepper could not act on it. That is condition (8.1) of
`adaptive_continuation_maths.md`, `F * mean_step >~ path length`, failing as
written.

---

## Group C — adaptive resolution

**Held fixed:** 30 epochs, LR 0.005, **`Gnone` throughout** — no Gaussian
anywhere, so resolution is the only operator that moves. Stages 16/24/32,
bilinear antialiased reduction of the float image before channel normalisation,
normalisation constants shared across resolutions.

**Varied:** *only* when the two resolution transitions happen.

| arm | transitions at | test acc |
|---|---|---:|
| plain (constant 32) | — | 0.7513 |
| fixed `Rprog` | epochs 6, 12 | **0.7920** |
| adaptive, gap trigger | epochs **2, 4** (both on the gap condition) | 0.7729 |

**What this licenses.** The gap trigger transitions far too early on this axis,
and the reason is measurable: the gap is **0.00–0.40** for resolution against
**3–13** for sigma, and is exactly 0.0000 from epoch 4 on. Global average
pooling makes the network genuinely resolution-transferable, so a
resolution-trained network evaluated at 32x32 loses almost nothing.

That exposes a defect in the rule, not a parameter to retune: a gap that is
**small and flat** is ambiguous between "this level is exhausted" and "this
level was never far from the target". For resolution it is the second, and the
rule cannot tell them apart.

---

## Group D — long horizon: does any of this survive convergence?

**Held fixed:** 120 epochs, constant resolution 32, seven Gaussian levels where
filtered. `Gplateau` scaled by *shape*: 7 levels x 12 epochs, bypass at 84, 36
terminal epochs — preserving the per-level dwell fraction and the 30% terminal
fraction rather than absolute epoch counts.

**Varied:** two factors, fully crossed — schedule kind (plain / fixed / adaptive)
and peak LR (0.005 / 0.05).

| arm | LR | test acc | test CE | probe acc | gap |
|---|---:|---:|---:|---:|---:|
| plain | 0.005 | 0.7710 | 1.0017 | 1.0000 | 0.229 |
| fixed `Gplateau` | 0.005 | **0.8068** | 0.7897 | 1.0000 | 0.193 |
| adaptive, gap trigger | 0.005 | 0.7840 | 0.9344 | 1.0000 | 0.216 |
| plain | 0.05 | 0.8604 | 0.4950 | 1.0000 | 0.140 |
| fixed `Gplateau` | 0.05 | **0.8716** | 0.4588 | 1.0000 | 0.128 |
| adaptive, gap trigger | 0.05 | 0.8673 | 0.4664 | 1.0000 | 0.133 |

Paired differences:

| LR | fixed − plain | adaptive − plain | adaptive − fixed |
|---|---:|---:|---:|
| 0.005 | **+3.58 pp** | +1.30 pp | −2.28 pp |
| 0.05 | **+1.12 pp** | +0.69 pp | −0.43 pp |

**What this licenses.**

1. **All six arms interpolate** (probe accuracy 1.0000), so the earlier
   "nothing converges" finding was a property of the 30-epoch budget, not of the
   problem. It also **falsifies** the LR-mass prediction in section 10 of the
   maths doc, which said the 0.005 leg would fall ~6x short of converging; that
   section is corrected in place.
2. **The fixed schedule's benefit survives full convergence** — it is not an
   artifact of a truncated budget — but it *shrinks* as the optimiser gets
   stronger, +3.58 pp -> +1.12 pp.
3. Since every arm interpolates the training set, the train/test gap is pure
   generalisation, and the continuation **reduces** it in both legs
   (0.229 -> 0.193 and 0.140 -> 0.128). That is the cleanest evidence here that
   it acts as a regulariser rather than an optimisation aid.
4. The controller loses to the fixed schedule at both LRs, but by **less** at
   the higher one (−0.43 vs −2.28 pp), consistent with under-traversal costing
   proportionally less when there are more epochs to traverse in.

**What it does not license.** One seed per cell, and the two LRs differ in
several coupled ways (effective step size, implicit regularisation, time to
interpolation). "LR 0.05 is better" is not a claim this design supports beyond
these two points.

---

## Standing summary

| | best fixed | best adaptive | delta |
|---|---:|---:|---:|
| sigma, 30 epochs | 0.7874 (back-loaded) | 0.7713 | −1.61 pp |
| resolution, 30 epochs | 0.7920 | 0.7729 | −1.91 pp |
| sigma, 120 epochs, LR 0.005 | 0.8068 | 0.7840 | −2.28 pp |
| sigma, 120 epochs, LR 0.05 | 0.8716 | 0.8673 | −0.43 pp |

Eight adaptive runs, none beating its fixed counterpart. Two positives:
back-loaded allocation beats uniform, and the fixed schedule's gain survives
convergence while reducing the generalisation gap.

The open prediction: pair the gap trigger with a stepper satisfying (8.1) and it
should produce a back-loaded schedule and land at or above 0.7874. Not yet run.
