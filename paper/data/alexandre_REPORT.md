# Does the continuation gain survive a change of optimizer?

Report for branch `continuation-core-optimizer-benchmark`.
Campaign of 2026-09-14 — 53 training runs, Kaggle 2×T4, about 7.5 GPU-hours.

---

## 1. The question

Branch `continuation-core` freezes four configurations and reports that
**continuation** — degrading the network early in training (blur, reduced
resolution) and then restoring the exact task — is worth **+4 to +6 points**
over a control with no intervention.

That number was measured under **one optimizer**: SGD, momentum 0.9, lr 0.005.
It is a deliberately low learning rate on a short 30-epoch budget, in a regime
where the control underfits badly — it tops out at 75 %, while a properly
trained ResNet-20 passes 91 %.

Which leaves a competing explanation that nothing in the repository could rule
out:

> What if the continuation gain is not a property of the continuation path at
> all, but of **optimization speed**? An adaptive optimizer, adjusting its step
> per parameter, would then be doing the same work — and the gain would vanish.

Put plainly: does continuation genuinely help, or does it compensate for an
under-tuned optimizer? That is what this campaign settles.

---

## 2. What is compared

**Four methods** — the four frozen configurations, unchanged:

| method | what it does |
|---|---|
| `plain` | control, no intervention |
| `resolution_max_b1` | reduces the activation map after block 1: 16 → 24 → 32 px |
| `gaussian_postrelu` | blurs 10 post-ReLU tensors, σ annealed from 1.0 to 0 |
| `resolution_max_b1_gaussian_conv` | both: reduction plus blur at the 19 convolution outputs |

Every method cancels its own intervention before the end — blur off at epoch 21,
resolution restored at epoch 12 — so **every arm finishes with at least nine
epochs on the exact target task** and is evaluated as the same network as the
control. That is what separates continuation from simply changing the
architecture.

**Four optimizers:**

| optimizer | what makes it interesting here |
|---|---|
| **SGD** | the reference, momentum 0.9 |
| **Adam** | adaptive, weight decay as coupled L2 |
| **AdamW** | identical to Adam **but for the decoupled-weight-decay flag** |
| **RAdam** | corrects the variance of the first updates — the exact phase where continuation intervenes |

Adam and AdamW differ by one flag, which makes the pair a single-factor
contrast, and it bites here because the recipe applies weight decay to *every*
parameter, BatchNorm and biases included.

**Three seeds** (0, 1, 2), with the same initial weights, the same data order
and the same probe as every result already on record — pinned files, verified by
sha256 before each run.

4 methods × 4 optimizers × 3 seeds = **48 arms**.

---

## 3. How

### The recipe, held fixed

Full CIFAR-10 (50,000 / 10,000), **no augmentation**, 30 epochs = 11,730
updates, effective batch 128 in microbatches of 32, weight decay 5e-4, 60 warmup
updates then cosine.

**Only the optimizer and its learning rate change.** The schedule shape, the
warmup and the weight decay are identical throughout — otherwise an optimizer
contrast would also be a schedule contrast.

### Tuning the learning rate, or the comparison means nothing

lr 0.005 is an SGD value. Adam at 0.005 underperforms badly. Comparing
optimizers at a fixed rate would measure nothing but "rate not tuned for this
optimizer".

So each optimizer was swept **before** the grid ran, on the `plain` control
alone, seed 0:

| lr | Adam | AdamW | RAdam |
|---|---:|---:|---:|
| 3e-4 | 75.37 | 73.89 | 74.09 |
| 1e-3 | 81.29 | 79.79 | 80.07 |
| 3e-3 | **83.95** | 82.98 | **84.58** |
| 1e-2 | 81.97 | 83.80 | 84.42 |
| 2e-2 | — | **84.52** | — |
| 3e-2 | — | 83.83 | — |

Selection rule, fixed **before** the numbers were read: highest accuracy on
`plain`; ties inside 0.3 pt go to the lower rate. AdamW peaked on the edge of
the grid, so the grid was extended until its optimum was bracketed — a symmetric
rule, applied to the only optimizer that triggered it.

**Chosen: Adam 3e-3, AdamW 2e-2, RAdam 3e-3.** The three land within 0.63 pt of
each other, so none goes into the campaign handicapped by a bad rate.

Tuning on `plain` gives the **control** its best showing, so the bias runs
*against* continuation — the right direction for a test of whether it holds.

### The quantity measured

Absolute accuracy drifts between GPU sessions: this repository documents up to
0.73 pt on the same arm, same seed, same pinned files (cuDNN is not
deterministic). So the comparison is not between raw accuracies but between
**paired gains**:

```
Δ(method, optimizer, seed) = acc(method) − acc(plain)
```

taken **inside** one optimizer, one seed and one session. Both terms drifted
together, so the drift cancels.

### Reusing the SGD numbers — checked, not assumed

The 12 SGD arms come from the batch already on record. To validate that reuse,
`plain` was rerun under the reference recipe:

| seed | recorded | rerun | difference |
|---|---:|---:|---:|
| 0 | 74.58 | 74.57 | −0.01 |
| 1 | 75.66 | 75.69 | +0.03 |
| 2 | 76.05 | 76.35 | +0.30 |

Two seeds reproduce to within 0.03 pt, the third to 0.30 — all well inside the
documented drift. **The reuse is sound.** These are also the first complete
training runs ever executed with the `continuation-core` code, and they land on
the expected values.

---

## 4. The optimizer benchmark

### Final test accuracy (%), mean over 3 seeds

| method | SGD | Adam | AdamW | RAdam |
|---|---:|---:|---:|---:|
| `plain` *(control)* | 75.43 | 84.13 | 84.12 | 84.38 |
| `resolution_max_b1` | 80.37 | 85.31 | **86.28** | 85.63 |
| `gaussian_postrelu` | 79.91 | 83.95 | 83.71 | 84.75 |
| `resolution_max_b1_gaussian_conv` | 81.51 | 85.28 | 86.21 | 85.26 |

### Paired continuation gain (points), method − control

| method | SGD | Adam | AdamW | RAdam |
|---|---:|---:|---:|---:|
| `resolution_max_b1` | **+4.94** | +1.18 | +2.16 | +1.25 |
| `gaussian_postrelu` | **+4.48** | −0.18 | −0.41 | +0.37 |
| `resolution_max_b1_gaussian_conv` | **+6.08** | +1.15 | +2.09 | +0.88 |

### Per seed, with sign consistency

A gain is only credible if it points the same way on all three seeds.

| method | opt | mean | SD | same sign | per seed |
|---|---|---:|---:|:--:|---|
| `resolution_max_b1` | SGD | +4.94 | 0.50 | yes | +5.39, +4.40, +5.02 |
| | Adam | +1.18 | 0.40 | yes | +1.20, +1.57, +0.77 |
| | AdamW | +2.16 | 0.68 | yes | +2.94, +1.74, +1.80 |
| | RAdam | +1.25 | 0.53 | yes | +1.85, +1.06, +0.85 |
| `gaussian_postrelu` | SGD | +4.48 | 0.97 | yes | +5.48, +4.40, +3.55 |
| | Adam | −0.18 | 0.61 | **no** | +0.18, +0.16, −0.88 |
| | AdamW | −0.41 | 0.72 | **no** | +0.42, −0.91, −0.73 |
| | RAdam | +0.37 | 0.18 | yes | +0.48, +0.16, +0.47 |
| `resolution_max_b1_gaussian_conv` | SGD | +6.08 | 0.56 | yes | +6.72, +5.84, +5.68 |
| | Adam | +1.15 | 0.36 | yes | +1.22, +1.46, +0.76 |
| | AdamW | +2.09 | 0.62 | yes | +2.77, +1.56, +1.94 |
| | RAdam | +0.88 | 0.26 | yes | +1.11, +0.93, +0.60 |

---

## 5. Conclusions

### 5.1 — The optimizer is worth more than the intervention

The `plain` control goes from **75.43 to 84.38**: **+8.95 points** for changing
the optimizer alone, against +4 to +6 points for continuation.

The direct consequence: **a ResNet-20 with no intervention whatsoever, under any
of the three adaptive optimizers, beats every continuation arm in the frozen
benchmark** — whose best is 81.51.

The intervention had therefore been measured against a control leaving roughly
twice the intervention's own effect on the table.

### 5.2 — The resolution reduction survives, four times smaller

+4.94 under SGD becomes +1.18 / +2.16 / +1.25. Small, but **real**: the same
sign on all three seeds under all four optimizers, and larger than its own
seed-to-seed spread.

This is the part of the benchmark that holds up. It does something the optimizer
does not — just four times less than the SGD table suggested.

### 5.3 — The Gaussian blur does not survive

+4.48 under SGD becomes **−0.18 / −0.41 / +0.37**. Under Adam and AdamW the sign
is not even consistent across seeds: the effect is indistinguishable from zero.

Worse, the blur no longer adds anything **on top of** the resolution reduction:

| blur's contribution beyond the resolution reduction | |
|---|---:|
| SGD | **+1.14** |
| Adam | −0.03 |
| AdamW | −0.07 |
| RAdam | −0.37 |

`resolution_max_b1_gaussian_conv`, the best arm of the frozen benchmark, **never
beats `resolution_max_b1` alone** once you leave SGD.

### 5.4 — Reading it as a whole

> The **Gaussian** part of the effect was an optimization-speed artifact: an
> optimizer that adapts its step per parameter captures the same thing for free.
>
> The **resolution** part is not: it survives the change of optimizer, so it is
> doing something the optimizer does not — but its real size is on the order of
> 1 to 2 points, not 5.

The best arm of this campaign is **`resolution_max_b1` under AdamW, at 86.28 %**,
nearly 5 points above the best arm of the frozen benchmark.

---

## 6. Limits

* **One setting.** One architecture, one dataset, no augmentation, 30 epochs.
* **Budget pushes the same way.** The 120-epoch runs on the exploratory branch
  already showed the SGD gains shrinking (control 86.04, plateau 87.16). This
  report is a second axis along which they shrink, not independent confirmation.
* **Three seeds.** The differences among the three adaptive optimizers
  (+1.18 vs +2.16 vs +1.25) are of the same order as the seed spread: **they
  must not be ranked**.
* **Weight decay held at 5e-4 everywhere**, to keep the Adam/AdamW contrast
  single-factor. That is far below common practice for decoupled AdamW, so the
  two sit closer than a tuned comparison would put them, and **neither is a
  tuned AdamW**.
* **Learning rates tuned on `plain` only.** The conservative choice for this
  question, but not the rate a continuation arm would have picked for itself.
* **The test set has been examined many times** across this project. Three seeds
  give a descriptive spread, not a significance test.

---

## 7. Reproducing

```bash
py -m pytest -q                              # 53 tests
py scripts/stage_assets.py                   # publish the pinned files
py scripts/kaggle_run.py scripts/job_optbench_lr_sweep.py     --gpu ...   # 12
py scripts/kaggle_run.py scripts/job_optbench_lr_sweep_ext.py --gpu ...   #  2
py scripts/kaggle_run.py scripts/job_optbench_sgd_control.py  --gpu ...   #  3
py scripts/kaggle_run.py scripts/job_optbench_grid_{0,1,2}.py --gpu ...   # 36
py scripts/aggregate_optimizers.py
```

Kaggle allows only **two concurrent GPU kernels**, so the slices go up two at a
time. Details in [docs/kaggle_cli.md](docs/kaggle_cli.md).

**Verification passed:** 53 unit tests; bitwise parity 44/44 against the code
that produced the original benchmark, including the check that steps the
optimizer across every transition; exact target-state bypass (|Δ| = 0) under
Adam; the reference SGD configurations field-for-field identical to before the
campaign.

**Detailed documents:**
[docs/OPTIMIZER_BENCHMARK.md](docs/OPTIMIZER_BENCHMARK.md) (campaign),
[docs/LR_SWEEP.md](docs/LR_SWEEP.md) (sweep),
[docs/METHODS.md](docs/METHODS.md) (exact definition of the four methods).
Per-run records in `results/optimizer_benchmark/`.
