# Branch `continuation_new_loss` — what was tested and what came out

For anyone (or any agent) picking this branch up. It starts from
`continuation-core-experiments` and answers one question that turned out to
change the study's headline claim.

**The short version.** The +4.5 pp curriculum gain recorded in this benchmark is
not a generalisation gain. It is a **recovery**: the curriculum recaptures part
of what a correctly configured run gets anyway. Raise the baseline — by any of
three unrelated means — and the gain goes to zero.

---

## 1. What was run

156 training runs on Kaggle T4s, 4 frozen methods × 3 seeds, always evaluated on
the **target path** (operator at identity) so all arms are compared on the same
class of network.

| grid | cells | what it varied | doc |
|---|---|---|---|
| objectives | 96 | cross-entropy, label smoothing 0.1, focal γ=2, square loss on logits (6 learning rates) | [`OBJECTIVES.md`](OBJECTIVES.md) |
| augmentation | 60 | random crop + horizontal flip, 30 and 60 epochs, lr 0.005 and 0.01 | [`AUGMENTATION.md`](AUGMENTATION.md) |

Reproduce any table:

    python tools/analyze_lossgrid.py results/kaggle_outputs
    python tools/analyze_augment.py results/kaggle_outputs --synthesis
    python tools/build_curves_data.py        # regenerates docs/curves/data.json

---

## 2. The result

Every setting, ordered by how strong the control is:

| setting | control test | control train | R | G | RG |
|---|---|---|---|---|---|
| CE, lr 0.0025 | 71.33 % | 79.40 % | +2.75 | +4.19 | +5.43 |
| focal, lr 0.005 | 72.86 % | 89.20 % | +4.88 | +4.72 | +6.38 |
| label smoothing, lr 0.005 | 74.84 % | 89.13 % | +4.19 | +4.16 | +5.35 |
| **CE, lr 0.005 — the benchmark** | **75.93 %** | 93.87 % | **+4.28** | **+4.01** | **+5.42** |
| CE, lr 0.01 | 78.21 % | 99.60 % | +4.74 | +3.72 | +5.67 |
| crop_flip, 30e, lr 0.005 | 79.78 % | 81.73 % | +0.42 | +0.86 | +1.32 |
| square, lr 0.05 | 80.15 % | 95.87 % | +2.60 | +2.24 | +3.32 |
| **no aug, 60e, lr 0.01** | **80.27 %** | **100.00 %** | **+3.25** | **+2.57** | **+4.24** |
| crop_flip, 30e, lr 0.01 | 84.70 % | 87.93 % | +0.09 | −0.96 | +0.25 |
| square, lr 0.1 | 85.38 % | 98.80 % | +0.70 | +1.13 | +1.15 |
| crop_flip, 60e, lr 0.005 | 85.51 % | 89.40 % | +0.19 | −0.09 | +0.52 |
| square, lr 0.5 | 86.20 % | 98.47 % | −0.07 | −0.65 | −0.03 |
| **crop_flip, 60e, lr 0.01** | **88.27 %** | 94.20 % | **+0.19** | **−0.44** | **+0.12** |

Three unrelated ways of moving the baseline — the learning rate, the objective,
data augmentation — produce the same erosion. Below 79 % the gain is +2.6 to
+5.7 pp; above 85 % it is zero.

### The mechanism, measured

| | control gap (train − test) | curricula |
|---|---|---|
| no augmentation, 30e | 17.93 pp | 8.72 – 11.25 pp |
| with augmentation, 30e | **1.95 pp** | 1.56 – 1.90 pp |

The curriculum was closing an overfitting gap. Augmentation closes the same gap
first, and further. Two regularisers aimed at the same thing do not stack — blur,
resolution reduction, crop and flip all say *do not rely on exact pixel
positions*. Under augmentation the curricula fit the training set **more** than
the control (82.0–82.7 % against 81.73 %): the signature the whole mechanistic
story rested on inverts.

### Three alternative explanations, each eliminated by a measurement

* **Interpolation.** The gain is intact at 79.4 % train accuracy and at 100 %.
  Not the variable.
* **Schedule dilution.** Schedules are indexed by absolute epoch, so a 60-epoch
  run leaves the operator off for 39 of 60. The `no aug, 60e` corner has exactly
  that dilution and the gain **survives** (+2.6 to +4.2). Not the variable.
* **The objective.** Label smoothing and focal reproduce cross-entropy to within
  0.1 pp. Not the shape of the loss.

---

## 3. What this does and does not touch

**Changes.** The headline number is a recovery, not a generalisation gain, and
should be stated that way. It also *explains* the earlier negative result: 25
candidate quantities were tested for predicting the gain, with a detection
ceiling of 1.00, and none did — because it was not a generalisation effect.

**Survives untouched.** Every measurement on the frozen methods. The three
dissociations ([`CROSS_STUDY.md`](CROSS_STUDY.md)) are facts about what the
operators *do*, not about whether they raise accuracy. So is the flatness result
(36/36), the BatchNorm gauge quotient, the exact frequency profile, and the
statistical guards.

**Not established.** That curricula are useless in general — one dataset, one
architecture, one augmentation recipe, 30–60 epochs. The schedules were never
retuned for a strong-baseline regime. And R's natural claim, *efficiency*, has
never been tested where it could show: at 32×32 the kernels are too small to
saturate a T4, so cutting the spatial positions by four bought **6 %** of wall
time. That question belongs at 224×224.

---

## 4. What changed in the code

| file | what |
|---|---|
| `continuation_core/losses.py` | five objectives behind `build(cfg)`; every one a per-sample mean, because `Trainer._step` rebuilds the batch gradient from microbatches of 32 |
| `continuation_core/augment.py` | pad 4 / random 32×32 crop / flip; each draw a pure function of `(seed, epoch, batch, offset)`, so a resumed run reproduces its crops |
| `continuation_core/config.py` | `LossConfig`, and augmentation fields on `DataConfig`; defaults reproduce every recorded run bit-for-bit |
| `continuation_core/grid.py` | `build_loss_grid`, `build_augment_grid`, `stretch_method` |
| `continuation_core/evaluate.py` | reports `obj` beside `ce`; `ce` is reported whatever was trained, so columns share one scale |
| `tools/kaggle_run.py` | retries transient network failures and re-checks status before calling a local timeout a failed run |
| `assets/cifar10_resnet20bn_e60/` | the pinned assets extended to 60 epochs |

`tests/` covers all of it — **89 passing**.

### Two traps that cost a run each

**A learning-rate sweep placed on the wrong scale.** The square loss divides by
`C`, so its gradient is ten times smaller and its optimum sits an order of
magnitude above cross-entropy's. The first sweep (0.0025 / 0.005 / 0.01) produced
41–57 % accuracy with a training cross-entropy of 2.0 against `ln(10) = 2.303` —
networks that had not learned. The second (0.05 / 0.1 / 0.5) was deliberately
**disjoint**, so a best value landing on an endpoint again would be visible as a
misplaced sweep rather than read as a result. It landed interior, at 0.1.

**The pinned assets hold exactly 30 per-epoch permutations.** A 60-epoch run is
refused rather than silently recycling the data order. The batch-order generator
is sequential, so a 60-epoch set *extends* the pinned one — the first 30
permutations are bit-identical, and `subset`, `train_probe` and the initial
weights were copied verbatim rather than regenerated (regenerating depends on the
torch build and would have broken comparability).

### One knob to know about

`build_augment_grid(stretch=...)` decides what a longer budget means. `False`
(what ran) keeps absolute epochs: the method as defined, then 39 epochs of bare
training. `True` multiplies the boundaries so the operator keeps its share.
**These are different experiments**; the analyzer never pools cells that disagree
about it, and each run records its choice in `config.json`.

---

## 5. Where things are

* **Results**: `results/kaggle_outputs/cc-{grid,loss,sq,aug,a60}-*` — configs,
  per-epoch metrics and summaries are versioned; checkpoints are not.
* **Reports**: `results/loss_report.json`, `results/augment_report.json`.
* **Curves**: `docs/curves/` — a standalone page, plus `data.json` rebuilt by
  `tools/build_curves_data.py`.
* **Docs**: [`OBJECTIVES.md`](OBJECTIVES.md), [`AUGMENTATION.md`](AUGMENTATION.md),
  and from the parent branch [`RESULTS_GRID.md`](RESULTS_GRID.md),
  [`CROSS_STUDY.md`](CROSS_STUDY.md), [`PROBES.md`](PROBES.md).

## 6. What I would do next

1. **Retune a schedule for the strong-baseline regime** before concluding the
   method is dead there. Everything tested used schedules chosen under the weak
   configuration.
2. **Test R's efficiency claim where resolution costs something** — a larger
   input, and "accuracy at equal compute" rather than at equal epochs.
3. **Leave the dissociations alone.** They are the part of this work that does
   not depend on the headline number, and they are measured.
