# The gain tracks the baseline, not the curriculum

The recorded benchmark trains with **no augmentation** -- `data.py` says so -- and
its control reaches 75.9 % where a ResNet-20 on CIFAR-10 with the standard recipe
reaches about 91-92 %. This grid asks whether the +4.5 pp curriculum gain is
recovering part of what random crop and horizontal flip give for free.

It is. And the same collapse appears, at the same baseline levels, from an
intervention that has nothing to do with augmentation.

## The recipe

He et al. 2016, sec. 4.2: pad four pixels of zeros on each side, take a random
32x32 crop (81 positions), flip horizontally with probability one half. 162
variants per image, a fresh draw every epoch. **Training only** -- `evaluate`
never comes through the augmenter, so the train probe measures clean images and
the train/test gap stays readable.

No operator needed adapting. Augmentation preserves the 32x32 input, and the
resolution reduction's `r` is a *feature-map* size at `blocks[2]`, which is
unchanged by anything done to the image.

Every draw is a pure function of `(run seed, epoch, batch index, microbatch
offset)`, so a resumed run reproduces the crops an uninterrupted one saw. See
[`continuation_core/augment.py`](../continuation_core/augment.py).

## The result, three seeds throughout

| | baseline test | baseline train | R | G | RG |
|---|---|---|---|---|---|
| no augmentation, lr 0.005 | 75.93 % | 93.87 % | +4.28 | +4.01 | +5.42 |
| no augmentation, lr 0.01 | 78.21 % | 99.60 % | +4.74 | +3.72 | +5.67 |
| **crop_flip, lr 0.005** | **79.78 %** | 81.73 % | **+0.42** | **+0.86** | **+1.32** |
| **crop_flip, lr 0.01** | **84.70 %** | 87.93 % | **+0.09** | **−0.96** | **+0.25** |

The gain falls from about +5 pp to about zero. At `crop_flip, lr 0.01` the
Gaussian arm is **below** the control.

### 60 epochs, and the control that settles it

Schedules are indexed by absolute epoch, so a 60-epoch run reaches its target
state at the same epoch as a 30-epoch one and trains bare for the remaining 39.
That dilution could by itself shrink the gain, which is why `none` was run at the
same budget and learning rate:

| | baseline test | baseline train | R | G | RG |
|---|---|---|---|---|---|
| **no augmentation, 60e, lr 0.01** | 80.27 % | **100.00 %** | **+3.25** | **+2.57** | **+4.24** |
| crop_flip, 60e, lr 0.005 | 85.51 % | 89.40 % | +0.19 | −0.09 | +0.52 |
| **crop_flip, 60e, lr 0.01** | **88.27 %** | 94.20 % | **+0.19** | **−0.44** | **+0.12** |

**The gain survives the dilution and dies on the augmentation.** Same budget,
same schedule, same learning rate: +3.25 / +2.57 / +4.24 without, +0.19 / −0.44 /
+0.12 with.

The training accuracies say why. Without augmentation every arm reaches
**100.00 %** — complete memorisation, a 20-point train/test gap — and that is
where the curriculum still has something to close. With augmentation training
stops at 94.2 %, the gap falls to 6 points, and there is nothing left to recover.

The augmented baseline now reaches 88.27 %, within reach of the 91–92 % the
recipe is worth, so the comparison is no longer being made in a degraded regime.

## The same curve, reached two different ways

Every setting measured on this branch, ordered by how strong the baseline is:

| setting | baseline test | baseline train | R | G | RG | n |
|---|---|---|---|---|---|---|
| CE, lr 0.0025 | 71.33 % | 79.40 % | +2.75 | +4.19 | +5.43 | 1 |
| focal, lr 0.005 | 72.86 % | 89.20 % | +4.88 | +4.72 | +6.38 | 3 |
| label smoothing, lr 0.005 | 74.84 % | 89.13 % | +4.19 | +4.16 | +5.35 | 3 |
| CE, lr 0.005 *(the benchmark)* | 75.93 % | 93.87 % | +4.28 | +4.01 | +5.42 | 3 |
| CE, lr 0.01 | 78.21 % | 99.60 % | +4.74 | +3.72 | +5.67 | 1 |
| crop_flip, lr 0.005 | 79.78 % | 81.73 % | +0.42 | +0.86 | +1.32 | 3 |
| square, lr 0.05 | 80.15 % | 95.87 % | +2.60 | +2.24 | +3.32 | 3 |
| crop_flip, lr 0.01 | 84.70 % | 87.93 % | +0.09 | −0.96 | +0.25 | 3 |
| square, lr 0.1 | 85.38 % | 98.80 % | +0.70 | +1.13 | +1.15 | 3 |
| square, lr 0.5 | 86.20 % | 98.47 % | −0.07 | −0.65 | −0.03 | 3 |

**The gain is a function of the baseline and of nothing else visible here.** It
is worth +4 to +5.7 pp below 79 %, about +1 pp between 80 and 85 %, and zero
above 85 %.

Two facts make that reading hard to avoid:

1. **Two unrelated interventions produce the same erosion.** Changing the
   objective and augmenting the data share no machinery, and both destroy the
   gain at the same baseline levels.
2. **Training accuracy does not explain it.** The gain is intact at 79.4 % train
   accuracy and intact at 99.6 %; it is gone at 81.7 %. Interpolation is not the
   variable.

## What this changes, and what it does not

**Changes.** The +4.5 pp is not a generalisation gain, it is a **recovery**: the
curriculum recaptures part of what a correctly configured run -- augmented, at a
learning rate that is not below its optimum -- obtains anyway. Once the baseline
is sound the curriculum contributes nothing measurable. The central claim of the
study has to be stated that way.

**Does not change.** Every measurement stands. Blur and resolution reduction are
still structurally different operators, the three dissociations
([`CROSS_STUDY.md`](CROSS_STUDY.md)) were measured on the frozen methods and are
unaffected, and the negative results on flatness and PAC-Bayes are untouched. It
is the *interpretation* of the headline number that moves, not the data.

## Not yet excluded

* **The schedule was never retuned for these regimes.** `Rprog` and the Gaussian
  plateaus are indexed by absolute epoch and were chosen under cross-entropy at
  lr 0.005. A curriculum designed for an augmented run at lr 0.01 is a different
  experiment, and this grid does not run it.
* **One architecture, one dataset.** As before.

## Running it

    AUG_CORNERS=crop_flip:30:0.005,crop_flip:30:0.01 python tools/job_augment.py
