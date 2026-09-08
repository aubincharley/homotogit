# exp6 -- PlainNet-18, the pilot on 5k images

The hypothesis: residual connections already convexify the loss surface
(Li et al. 2018), so the activation homotopy has nothing left to fix on a
ResNet -- which exp1 to exp4 confirmed. Strip the skips and the surface is
chaotic again, and that is where a near-linear start should earn its keep.

**It does not.** On this pilot the homotopy loses on every arm, and loses
*more* than it did on a ResNet.

PlainNet-18 (`use_residual: false`), 5 000 images, 50 epochs, seed 1, run
locally on MPS. Configs: `src/cifarbase/configs/plain_*50.yaml`.
Raw runs (untracked): `out/exp6_plainnet_5k_seed1/`.

## Result

| arm | alpha | lambda | test acc | vs base | test loss | train acc | gap |
|---|---|---|---|---|---|---|---|
| plain_baseline50 | 0 | 0 | **0.7777** | -- | 0.9875 | 0.9992 | +0.2215 |
| plain_act_linear50 | 1->0 / 25 ep | 0 | 0.7218 | **-5.59 pt** | 1.0801 | 0.9830 | +0.2612 |
| plain_act_anchor50 | 1->0 / 25 ep | 1e-4 | 0.7211 | **-5.66 pt** | 1.0824 | 0.9820 | +0.2609 |
| plain_act_staircase50 | 5 x 10 ep | 1e-4 | 0.6959 | **-8.18 pt** | 0.9342 | 0.8616 | +0.1657 |

Gradient updates to reach a level, relative to the baseline:

    50%   linear 2.24x   anchor 2.25x   staircase 4.19x
    60%          1.79x          1.70x             3.31x
    70%          1.87x          1.84x             never
    75%          never          never             never

## Readings

**The hypothesis is not supported here.** Removing the skips was supposed to
be the regime where the homotopy helps. Instead the penalty grows: -1.2 points
on a ResNet at 50k (exp4), -5.6 to -8.2 points on a PlainNet. Whatever the
homotopy costs, a harder landscape makes it cost more, not less.

**The anchor is inert once again.** linear and anchor differ by 0.07 points
(0.7218 vs 0.7211) with identical loss and train accuracy -- a third
independent confirmation, after the lambda sweep of exp2 and the ResNet arms
of exp3/exp4, that lambda=1e-4 changes nothing.

**The staircase is the only arm that behaves differently at all.** It ends
lowest on accuracy but *best* on test loss (0.9342 against the baseline's
0.9875) and far below on train accuracy (0.8616 against 0.9992). It has simply
not finished fitting: ten epochs at alpha=0 out of fifty. That is the same
"training less regularises" signature exp3 showed at 10k and exp4 refuted at
50k -- it should not be read as a win.

## Caveat that limits what this settles

**A PlainNet-18 may not be hard enough for the premise to bite.** He et al.
(2015) found plain and residual networks near-equivalent at 18 layers; the
degradation that motivates skip connections appears at 34. The baseline here
reaches 77.8% and trains without trouble, which suggests the landscape did not
become chaotic enough to be the regime the hypothesis describes.

Two runs would settle it, and neither is in yet:

  * `resnet18` at this exact format (5k, 50 epochs, seed 1). If plain and
    residual land within a couple of points, the premise fails at this depth
    and exp6 tests nothing about landscape geometry.
  * `--arch resnet34`, where the plain/residual gap is documented, and where
    the baseline would actually be in difficulty.

**And 5 000 images is the memorisation regime** (baseline gap +0.22). exp3 and
exp4 together showed that conclusions drawn there can reverse at full scale.
This is a pilot: it justifies or kills a 50k run, it does not conclude.
