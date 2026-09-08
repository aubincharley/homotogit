# Experiment 1 — Gaussian warm starts and an intermediate continuation stage

**Status: executed in full.** 3 prefixes + 6 branched trajectories (W and P × seeds
{0,1,2}). Arm A reused unchanged from Experiment 0. Nothing remains unexecuted.
The official test set was not touched.

Raw records: `results/exp1_gaussian_warmstart/`. Experiment 0's records are
unmodified.

## Questions

1. Does a short initial phase at σ = 1 improve subsequent learning on `L_0`, at a
   fixed total update budget?
2. Does an intermediate phase at σ = 0.5 add value compared with switching
   directly to σ = 0?

## 1. Inspected configuration (unchanged from Experiment 0)

Read from `results/exp0_gaussian/level_0__seed_0/run_description.json` and the
code, and held fixed for the main comparison:

| Item | Value |
|---|---|
| Architecture | ResNet-20, GroupNorm (`channels_per_group=8`), option-A shortcuts, 269,722 params |
| Normalization | per-channel mean `[0.49118823, 0.48212275, 0.44643784]`, std `[0.2470552, 0.24351363, 0.2615779]`, fitted on the unfiltered training subset, applied **after** the transform |
| Split | 45,000 / 5,000 stratified, `split_seed=12345` (identical fingerprints) |
| Preprocessing | none beyond `uint8/255 → T_σ → normalize`; **no augmentation** |
| Optimizer | SGD, momentum 0.9, weight decay 5e-4, `nesterov=False` |
| Batch size | 128 (351 updates/epoch) |
| LR schedule | peak 0.1, **linear warmup 400 updates**, then **cosine to 0** over B = 14,040, indexed by global update. `milestones = []` |
| Normalization state | GroupNorm keeps **no running statistics**; the model has zero buffers, so nothing is recalibrated at a transformation change |
| Mixed precision | not used; no `GradScaler` exists (recorded as `amp: null`) |

### The sharp loss decrease at 10,000–12,000 updates is *not* a learning-rate change

There are no milestones and no discrete drops — the schedule is a smooth cosine.
It does, however, coincide with the cosine entering its low-LR phase:

| update | 8,000 | 10,000 | 11,000 | 12,000 | 13,000 | 14,039 |
|---|---|---|---|---|---|---|
| lr | 0.0411 | 0.0201 | 0.0118 | 0.0054 | 0.0014 | 0.0000 |

So the drop coincides with a **smooth** decay through lr ≈ 0.02 → 0.005, not with
a step change. The same feature appears at the same global updates in all three
arms of Experiment 1 (see `figures/exp1_equal_budget.png`), which is expected
since the LR schedule is indexed by global update and is never restarted.

## 2. Checkpoint inventory

Experiment 0 wrote **15 files, all `checkpoint_final.pt` at update 14,040**, all
classified `model_and_optimizer_only`: parameters + optimizer state + step, but
**no** RNG state and **no** data-order position.

* No checkpoint exists at update 1,500. Eval records every 500 updates do not
  imply checkpoints there.
* Even at 14,040 the files are not complete resumable states.

The σ = 1 prefix was therefore **rerun** for each seed, on the original
B = 14,040 LR horizon and merely stopped at 1,500 (`Trainer.fit(max_steps=…)`
bounds execution without touching `optim.total_steps`). A new `full_state_v1`
checkpoint format stores parameters, optimizer state (momentum buffers), global
update counter, sampler position, and RNG states; `amp` is explicitly `null`.

Arm A was **reused unchanged** — no incompatibility was found.

## 3. Pre-run checks (all passed)

| Check | Result |
|---|---|
| Branch checkpoint type | `full_resumable` |
| Global counter at branch | 1,500 (all seeds) |
| σ before → after branch | W: 1.0 → 0.0, P: 1.0 → 0.5 |
| LR at branch | 0.09840385594331022 = `lr_at(1500, B=14040)`, **not** the ≈0 a compressed 1,500-horizon would give |
| Momentum buffers present at branch | yes (carried, not reset) |
| Sampler position at branch | epoch 5, position 96 of 351 — **mid-epoch**, restored exactly |
| σ = 0 supplies original images | `pipeline(x, 0.0)` bitwise-equals `normalize(x/255)` |
| Resume vs uninterrupted | identical parameters, `atol=rtol=0`, branching mid-epoch (`tests/test_branching.py`) |
| Minibatch sequence vs arm A | permutation identical at update 1,500; next 2,000 index vectors identical |
| **Target-metric invariance at transitions** | **9 checks, max \|Δ\| = 0.0** |

The invariance check compares original-image metrics at *identical weights*
immediately before and after the active σ changes, for all three transitions
(the branch at 1,500 for W and P, and P's in-run switch at 3,000). Only the
active-level metrics moved; every target metric was bit-identical. Recorded in
`results/exp1_gaussian_warmstart/transition_invariance.json`.

83 tests pass, including the resume-exactness and invariance guards.

## 4. Results at the common budget B = 14,040

Mean over 3 paired seeds, sample sd in parentheses.

| Arm | target val accuracy | target val CE | target train probe CE |
|---|---|---|---|
| A — direct baseline | **0.8411** (0.0060) | **0.6755** (0.0223) | 0.0034 |
| W — warm start σ=1→0 | 0.8293 (0.0011) | 0.7113 (0.0252) | 0.0030 |
| P — continuation σ=1→0.5→0 | 0.8319 (0.0051) | 0.7116 (0.0283) | 0.0031 |

Per-seed target validation accuracy:

| seed | A | W | P |
|---|---|---|---|
| 0 | 0.8464 | 0.8286 | 0.8260 |
| 1 | 0.8422 | 0.8306 | 0.8352 |
| 2 | 0.8346 | 0.8288 | 0.8346 |

### Paired differences at B (accuracy in percentage points)

| comparison | mean | sd | per-seed | target val CE |
|---|---|---|---|---|
| W − A | **−1.17 pp** | 0.60 | −1.78, −1.16, −0.58 | +0.036 |
| P − A | **−0.91 pp** | 1.04 | −2.04, −0.70, 0.00 | +0.036 |
| P − W | **+0.26 pp** | 0.45 | −0.26, +0.46, +0.58 | +0.0003 |

W − A is negative in all three seeds. P − A is negative in two and zero in one.
P − W changes sign across seeds.

### Optimization diagnostic: first common checkpoint reaching a target-probe CE threshold

Global updates **including** warm-start pretraining; not interpolated.

| Arm | CE ≤ 0.1 (per seed) | CE ≤ 0.01 (per seed) |
|---|---|---|
| A | 10,500 / 10,000 / 10,000 | 11,500 / 11,500 / 11,500 |
| W | 10,000 / 10,000 / 10,500 | 11,500 / 11,500 / 11,500 |
| P | 10,000 / 10,000 / 10,500 | 11,500 / 11,000 / 11,500 |

All thresholds were reached by all seeds in all arms. Differences are at most one
500-update grid cell and are not consistent in sign — i.e. **no measurable
optimization speedup**, in either direction, at this resolution.

### Adaptation after the switch

Target validation accuracy, mean of 3 seeds, around the branch:

| global update | A | W | P |
|---|---|---|---|
| 1,500 (before any target training for W/P) | 0.6113 | 0.5557 | 0.5557 |
| 1,600 | — | 0.5987 | 0.5931 |
| 1,900 | — | 0.6323 | 0.6373 |
| 2,500 | 0.7076 | 0.6865 | 0.6784 |
| 3,000 | 0.7154 | 0.6957 | 0.6899 |
| 5,000 | 0.7750 | 0.7739 | 0.7605 |
| 10,000 | 0.8238 | 0.8147 | 0.8139 |
| 14,040 | 0.8411 | 0.8293 | 0.8319 |

`figures/exp1_adaptation.png` shows the same data aligned by updates *since each
arm began training on original images*. **That alignment flatters W and P**: they
reach update 0 of that axis having already spent 1,500 and 3,000 updates
respectively, while A starts from initialization. Equal numbers of target-stage
updates are not equal total computational cost. Paired claims above use common
global checkpoints only.

## 5. Answers to the three questions

**1. Does performance on original images recover quickly after switching? Yes.**
At the branch the σ=1 model scores 0.5557 versus A's 0.6113 at the same global
update — 5.6 pp behind, not catastrophic. Within 100 updates it reaches 0.5987;
within 1,000 updates of the switch it is at 0.6865. Recovery is fast and there is
no sign of a persistent penalty from having trained on blurred images.

**2. Do W or P catch up with or exceed A at equal global update count? No.**
W never matches A at any common checkpoint from 1,500 onward. P touches A once at
update 6,000 and ends below it. At B, W is 1.17 pp below A (negative in all three
seeds) and P is 0.91 pp below (negative in two seeds, tied in the third). The sd
across seeds (0.60 and 1.04 pp) is comparable to the effect, so with three seeds
this is **a consistent but small negative trend, not a firmly established
effect** — though the sign consistency for W − A is notable.

**3. Does P improve on W enough to justify the intermediate stage? No evidence
that it does.** P − W is +0.26 pp with sd 0.45 pp, and the sign flips across
seeds (−0.26, +0.46, +0.58). Target validation CE is indistinguishable
(+0.0003). Three seeds cannot resolve a difference this small, and nothing here
suggests the σ = 0.5 stage is doing useful work.

Recovery alone (question 1) does not establish a net benefit. On these
measurements the warm start neither helped nor catastrophically hurt: it cost
about one percentage point of final target validation accuracy at a matched
budget.

## 6. Cost

| item | value |
|---|---|
| Updates executed by this experiment | 79,740 (3 × 1,500 prefix + 6 × 12,540 arm) |
| Incremental active compute | **2,735 s ≈ 0.76 h** |
| — prefixes | 151 s |
| — arms | 2,584 s |
| Transformation cost | 9.1 s, **0.3 %** of incremental compute |
| Arm A incremental compute | **0** (reused from Experiment 0) |

Method cost in budget terms is equal by construction: A = 14,040, W = 1,500 +
12,540 = 14,040, P = 1,500 + 12,540 = 14,040 updates. The shared prefix is
computed once but counted in both W and P.

No machine suspension occurred during Experiment 1, so no exclusion was needed.
(One Experiment 0 run, `level_0p5__seed_1`, has a wall time contaminated by a
suspension; it is excluded from cost statements there and is unrelated here.)

## 7. Interpretation and scope

What these measurements do **not** support:

* Not that 1,500 updates is an optimal warm-start length, nor that σ = 1 is the
  right warm level. Stage lengths were pragmatic pilot choices fixed in advance.
* Not that σ > 1 cannot help — untested here.
* Not that **late** coarse checkpoints behave like this one. This experiment
  tests an **early** warm start only; whether a converged σ=1 model is a worse
  initializer after adaptation requires a separate controlled comparison.
* Not that near-perfect final training accuracy rules out optimization speedups
  at smaller budgets. The threshold diagnostic covers this budget and this
  schedule.
* A negative result applies to the tested transformation, schedules, optimizer
  and data. It does not generalize to TV budgets, other schedules, or other
  architectures.

Three paired seeds support a **preliminary** comparison. Differences of ~1 pp
with sd of 0.6–1.0 pp should not be treated as firmly established, and the
overlapping error bars for P − W in particular carry no evidence of a difference.

## 8. Reproducing

```bash
py -m pytest tests -q
py -m continuation.cli exp1        --config configs/exp1_warmstart.yaml --seeds 0
py -m continuation.cli exp1        --config configs/exp1_warmstart.yaml --seeds 1 2
py -m continuation.cli exp1-report --config configs/exp1_warmstart.yaml
```

`exp1` is resumable: completed prefixes and arms are skipped. Arm A is read
read-only from `results/exp0_gaussian/`.

Figures: `results/exp1_gaussian_warmstart/figures/exp1_equal_budget.png`,
`exp1_adaptation.png`, `exp1_paired_differences.png`.
Machine-readable: `aggregate.json`, `aggregate.csv`, `manifest.json`,
`transition_invariance.json`, per-run `metrics.jsonl` / `summary.json` /
`run_description.json` (with `lineage` naming each branch's parent checkpoint).
