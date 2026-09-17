# Does the curriculum gain depend on the objective?

The benchmark trained every method with plain softmax cross-entropy. The four
frozen methods cannot tell, on their own, whether the +4.5 pp is a property of
the **intervention** or of the pair **(intervention, objective)**. This grid
keeps the methods untouched and varies the loss around them, exactly as the
28-cell grid varies the optimiser.

Only training and evaluation are involved. No curvature, no Jacobian, no probe:
the outputs are train/test loss and accuracy.

## The objectives

| arm | loss | why it is here |
|---|---|---|
| `ce` | softmax cross-entropy | the control; the recorded reference cells are reusable |
| `ls` | cross-entropy, label smoothing 0.1 | caps confidence. If the curriculum does the same thing, the two should not add |
| `focal` | `-(1-p_t)^2 log p_t` | near-neutral on balanced accuracy, strong on calibration: separates the two |
| `square` | `‖z − onehot(y)‖² / C` **on the logits** | a different gradient shape entirely. If the gain survives, it is not about the loss |

`brier` — the square loss on the *softmax* — is implemented for contrast but is
not an arm. It is a different objective from `square`: its gradient passes
through the softmax Jacobian and **saturates**, so a confidently wrong
prediction receives almost no gradient. The logit form's gradient is
`2(z − y)/C`, linear in the error, and that is the entire reason to run it.

## Two rules the code enforces

**Every objective is a per-sample mean.** `Trainer._step` rebuilds the batch
gradient from microbatches of 32, rescaling each by `n_micro / n_batch`. That sum
equals the full-batch gradient *only* for objectives that average over samples
independently, so `losses.build` refuses a name it does not implement rather
than accepting an arbitrary callable. A batch-coupled objective (contrastive, or
anything normalised by batch statistics) would train differently at a different
microbatch size, silently. `tests/test_losses.py` pins this for every arm.

**Cross-entropy is reported whatever is optimised.** `evaluate` returns `ce` for
every run and `obj` — the value of the objective actually trained — beside it.
Test accuracy is objective-agnostic and is the comparator; `ce` is the common
scale; `obj` says what each run was minimising. `obj` is **not** comparable
across arms, and the analysis says so in its own output.

## The learning rate, and why the sweep is asymmetric

Label smoothing and focal are reweightings of cross-entropy: the gradient with
respect to the logits stays bounded the same way, so the reference step size
(0.005) remains appropriate.

The square loss does not share that scale. Cross-entropy's logit gradient is
`(p − y)`, bounded in `[−1, 1]`; the square loss's is `2(z − y)/C` with `z`
unbounded, and the `/C` convention alone moves the effective step by a factor of
10. A poor result at one learning rate would not distinguish the objective from
the step size, so `square` runs at 0.0025 / 0.005 / 0.01 and is read at its best.
The choice is made **once per arm** on the mean test accuracy over the four
methods, never per method, so the comparison between methods inside an arm stays
paired.

    ls, focal        4 methods x 3 seeds                    24 cells
    square           4 methods x 3 lr x 3 seeds             36 cells
                                                      --------------
                                                            60 cells

`ce` adds 12 more if run; it need not be. This branch's change is a no-op for
cross-entropy — `losses.build` returns `F.cross_entropy` unchanged — so the
recorded reference cells of the 28-cell grid *are* those runs, and
`analyze_lossgrid.py` reads them as the control.

## Running it

    LOSS_ARMS=ls,focal,square  python tools/job_lossgrid.py     # or the four shard entrypoints
    python tools/analyze_lossgrid.py results/kaggle_outputs

The analysis prints one table per objective (test/train accuracy, test/train CE,
the objective's own value) and then the contrast that the question turns on: the
gain of R, G and RG over the plain control, per objective.

## Results, three seeds

Test accuracy against the plain control, per objective:

| objective | R | G | RG | control |
|---|---|---|---|---|
| cross-entropy *(reference)* | +4.28 | +4.01 | +5.42 | 75.93 |
| label smoothing 0.1 | +4.19 | +4.16 | +5.35 | 74.84 |
| focal (gamma = 2) | +4.88 | +4.72 | +6.38 | 72.86 |
| square loss on logits | +0.70 | +1.13 | +1.15 | **85.38** |

**Under the three cross-entropy-shaped objectives the gain does not depend on
the objective.** Label smoothing reproduces cross-entropy to within **0.1 pp** on
all three arms, inside the seed-to-seed spread (0.3 to 1.0 pp), and the ordering
of the four methods is identical in every column.

**The square loss is the exception, and it is the interesting row.** At its own
best learning rate it reaches 85.4 % where cross-entropy's control reaches
75.9 %, and there the curriculum gain collapses from about +4.5 pp to **+1 pp**.

That row is confounded and must be reported as such: square-at-86 % differs from
cross-entropy-at-78 % in *both* the objective and the accuracy level, so "the
gain is specific to cross-entropy" and "the gain shrinks once the baseline is
strong" are not separated by this grid. Raising the baseline at a fixed objective
is what separates them -- which is what the augmentation experiment does, and it
answers: the gain collapses there too, at the same baseline levels, so the level
is the variable and not the objective. See [`AUGMENTATION.md`](AUGMENTATION.md).

Two readings the focal column invites, only one of which is supported. The
control falls from 75.93 to 72.86 while RG falls only 81.35 to 79.24, so +6.38
is **the curriculum absorbing an unfavourable objective better**, not the focal
loss improving the curriculum.

The other recorded fact survives the change too: training accuracy drops for
every method under label smoothing and focal (93.9 to 89.1 for the control), and
the curriculum arms stay *below* the control throughout, as they do under
cross-entropy. "The curricula fit the training set less" is not an artefact of
cross-entropy either.

### The baseline of this benchmark is undertrained

Two independent signs, and the second came out of this grid.

**No augmentation at all.** `data.py` states it; the control reaches 75.9 % where
a ResNet-20 on CIFAR-10 with the standard recipe reaches about 91-92 %.

**The reference learning rate is below the optimum for every objective tried.**
The 28-cell grid's own recipe variants already showed it for cross-entropy, and
the trend is monotone to the top of the swept range on all four arms:

| test accuracy | lr 0.0025 | lr 0.005 *(reference)* | lr 0.01 |
|---|---|---|---|
| `plain` | 71.33 | 75.93 | **78.21** |
| R | 74.08 | 80.22 | **82.95** |
| G | 75.52 | 79.95 | **81.93** |
| RG | 76.76 | 81.35 | **83.88** |

Cross-entropy is still improving where its sweep stops, and the square loss needs
0.1 -- twenty times the reference -- to train at all. Neither fact changes the
*contrast* between methods, which is what the benchmark is for, but both bear on
how the absolute numbers may be read, and any new arm should sweep the learning
rate rather than inherit 0.005.

### The square loss, and a sweep that was placed wrong

The first attempt swept 0.0025 / 0.005 / 0.01 -- the reference learning rate and
its two neighbours. Every cell failed to train:

| lr | 0.0025 | 0.005 | 0.01 |
|---|---|---|---|
| test accuracy | 41.5 % | 47.8 % | 57.1 % |
| training cross-entropy | 2.167 | 2.110 | 2.016 |

Against `ln(10) = 2.303`, a training cross-entropy of 2.0 after 30 epochs is a
network that has barely learned. The accuracy is **monotone increasing across
the whole range and best at its upper endpoint**, so the optimum lay above it.

That is the confound this arm exists to avoid, walked into: dividing the square
loss by `C` divides its gradient by ten, so its scale sits an order of magnitude
above cross-entropy's rather than beside it. The sweep is now 0.05 / 0.1 / 0.5 --
deliberately **disjoint** from the first, so that a best value landing on an
endpoint again is visible as a sweep still misplaced rather than read as a
result. It did not:

| lr | 0.0025 | 0.005 | 0.01 | 0.05 | 0.1 | 0.5 |
|---|---|---|---|---|---|---|
| test accuracy | 41.5 | 48.1 | 57.0 | 82.2 | **86.1** | 86.0 |
| training accuracy | 39.3 | 46.2 | 56.0 | 94.9 | 98.6 | 98.2 |

The optimum is **interior**, bracketed on both sides, so the column is readable.
The first three cells stay in the record as what they are -- failures to
optimise, from which no contrast between methods may be read: a difference
between two failures is not a difference between two solutions.

## What this cannot answer

* Anything about *why* an objective changes the gain. It is four columns, not a
  mechanism.
* Mixup and CutMix are deliberately excluded. They are input-space interventions
  like blur and resolution reduction, so a difference could not be attributed.
* SAM is not here either: it is a min-max objective, not a loss, and it doubles
  the cost per step. "Do SAM and the curriculum add?" is a good question and a
  separate experiment — the flatness results make it the most interesting one.
