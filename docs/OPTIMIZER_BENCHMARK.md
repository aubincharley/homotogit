# The continuation gain under four optimizers

The frozen benchmark measures continuation under one optimizer: SGD, momentum
0.9, lr 0.005. That is a low learning rate on a 30-epoch budget, in a regime
where the control underfits badly — 75 % without augmentation. So the gain it
reports may be an *optimization-speed* effect rather than a property of the
continuation path, and nothing in the repository could tell the two apart.

This campaign separates them. Same four frozen methods, same three seeds, same
pinned assets, same everything except the optimizer.

## Protocol

4 methods x 3 seeds x 4 optimizers = 48 arms, of which 36 were run here and 12
are the recorded `unified_selected` SGD arms, reused. Reference recipe
throughout: CIFAR-10, no augmentation, 30 epochs = 11,730 updates, effective
batch 128 in microbatches of 32, weight decay 5e-4 on every parameter, 60 warmup
updates then cosine. Only the optimizer and its learning rate differ; the
schedule shape, warmup and weight decay are held fixed, so an optimizer contrast
is not also a schedule contrast.

Learning rates were swept per optimizer on `plain`, seed 0, before the grid ran,
and frozen: adam 3e-3, adamw 2e-2, radam 3e-3, sgd 5e-3.
[LR_SWEEP.md](LR_SWEEP.md) has the grid and the selection rule. Tuning on
`plain` gives the control its best showing, so the bias runs *against*
continuation.

**Reusing the SGD arms is checked, not assumed.** Rerunning `plain` under the
reference recipe here gives 74.57 / 75.69 / 76.35 against the recorded 74.58 /
75.66 / 76.05 — two seeds to within 0.03 pt, the third to 0.30, all inside the
0.73 pt this repo documents between batches on identical assets.

## The quantity

Absolute accuracy drifts between GPU sessions, so the reported quantity is the
paired continuation gain

    D(method, opt, seed) = acc(method, opt, seed) - acc(plain, opt, seed)

taken inside one optimizer, one seed and one batch, where both terms drifted
together. The SGD row therefore differences the imported methods against the
*imported* `plain`, never against the drift-control `plain`, which came from a
different session.

## Final test accuracy

Mean over seeds 0-2, target path, epoch 30.

| method | sgd | adam | adamw | radam |
|---|---:|---:|---:|---:|
| `plain` | 75.43 | 84.13 | 84.12 | 84.38 |
| `resolution_max_b1` | 80.37 | 85.31 | **86.28** | 85.63 |
| `gaussian_postrelu` | 79.91 | 83.95 | 83.71 | 84.75 |
| `resolution_max_b1_gaussian_conv` | 81.51 | 85.28 | 86.21 | 85.26 |

## The paired gain

| method | sgd | adam | adamw | radam |
|---|---:|---:|---:|---:|
| `resolution_max_b1` | +4.94 | +1.18 | +2.16 | +1.25 |
| `gaussian_postrelu` | +4.48 | −0.18 | −0.41 | +0.37 |
| `resolution_max_b1_gaussian_conv` | +6.08 | +1.15 | +2.09 | +0.88 |

Per seed, with sign consistency:

| method | opt | mean | SD | same sign | per seed |
|---|---|---:|---:|:--:|---|
| `resolution_max_b1` | sgd | +4.94 | 0.50 | yes | +5.39, +4.40, +5.02 |
| | adam | +1.18 | 0.40 | yes | +1.20, +1.57, +0.77 |
| | adamw | +2.16 | 0.68 | yes | +2.94, +1.74, +1.80 |
| | radam | +1.25 | 0.53 | yes | +1.85, +1.06, +0.85 |
| `gaussian_postrelu` | sgd | +4.48 | 0.97 | yes | +5.48, +4.40, +3.55 |
| | adam | −0.18 | 0.61 | **no** | +0.18, +0.16, −0.88 |
| | adamw | −0.41 | 0.72 | **no** | +0.42, −0.91, −0.73 |
| | radam | +0.37 | 0.18 | yes | +0.48, +0.16, +0.47 |
| `resolution_max_b1_gaussian_conv` | sgd | +6.08 | 0.56 | yes | +6.72, +5.84, +5.68 |
| | adam | +1.15 | 0.36 | yes | +1.22, +1.46, +0.76 |
| | adamw | +2.09 | 0.62 | yes | +2.77, +1.56, +1.94 |
| | radam | +0.88 | 0.26 | yes | +1.11, +0.93, +0.60 |

## What it says

**1. The optimizer is worth more than the intervention.** `plain` goes from
75.43 to 84.38 — **+8.95 pt** for changing the optimizer, against +4 to +6 pt
for continuation. A plain ResNet-20 under any of the three adaptive optimizers
beats every continuation arm in the frozen benchmark, whose best is 81.51. The
intervention was measured against a control leaving roughly twice the
intervention's own effect on the table.

**2. The resolution reduction survives, at a quarter to a half its size.**
+4.94 under SGD becomes +1.18 / +2.16 / +1.25. Small, but the same sign on all
three seeds under all four optimizers, and larger than its own seed spread. This
is the part of the benchmark that holds up.

**3. The Gaussian blur does not survive.** +4.48 under SGD becomes −0.18 / −0.41
/ +0.37. Under Adam and AdamW the sign is not even consistent across seeds. On
its own, annealed Gaussian filtering of activations does nothing once the
optimizer is a reasonable one.

**4. The combined method's advantage was SGD-only.** Blur *on top of* the
resolution reduction is worth +1.14 pt under SGD and −0.03 / −0.07 / −0.37 under
the three adaptive optimizers. `resolution_max_b1_gaussian_conv` is the best arm
in the frozen benchmark; here it never beats `resolution_max_b1` alone. The best
arm of this campaign is `resolution_max_b1` under AdamW, at 86.28.

The coherent reading: the Gaussian part of the benchmark's effect was an
optimization-speed artifact, and an optimizer that adapts its per-parameter step
captures the same thing for free. The resolution part is not — it survives
optimizer changes, so it is doing something the optimizer does not, though four
times smaller than the SGD table suggests.

## Limits

* One architecture, one dataset, no augmentation, 30 epochs. The 120-epoch
  single-seed runs on the exploratory branch already showed the SGD gains
  shrinking with budget (plain 86.04, plateau 87.16); this is a second axis
  along which they shrink, not an independent confirmation.
* Three seeds. The differences between the three adaptive optimizers (+1.18 vs
  +2.16 vs +1.25 on `resolution_max_b1`) are of the same order as the seed
  spread and should not be ranked.
* Weight decay was held at the reference 5e-4 for every optimizer, to keep the
  Adam/AdamW contrast single-factor. For decoupled AdamW that is far below
  common practice, so the two are closer than a tuned comparison would make
  them, and neither is a tuned AdamW.
* Learning rates were tuned on `plain` only. A rate that favours the control is
  the conservative choice for this question, but it is not the rate a
  continuation arm would have picked for itself.
* The test set has been examined many times across this project.

## Reproducing

```bash
py scripts/stage_assets.py
py scripts/kaggle_run.py scripts/job_optbench_lr_sweep.py --gpu ...      # 12 cells
py scripts/kaggle_run.py scripts/job_optbench_lr_sweep_ext.py --gpu ...  #  2 cells
py scripts/kaggle_run.py scripts/job_optbench_sgd_control.py --gpu ...   #  3 cells
py scripts/kaggle_run.py scripts/job_optbench_grid_{0,1,2}.py --gpu ...  # 36 cells
py scripts/aggregate_optimizers.py
```

At most two GPU kernels run at once; see [kaggle_cli.md](kaggle_cli.md). Total
cost measured: 53 runs, about 7.5 GPU-hours. Per-run records are in
`results/optimizer_benchmark/`.
