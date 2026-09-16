

---

# `README.md`

# continuation-core — optimizer benchmark

The four frozen continuation methods, measured under four optimizers instead of
one. **Full report: [REPORT.md](REPORT.md).**

The parent branch `continuation-core` reports that continuation is worth +4 to
+6 points over a plain ResNet-20. It measured that under SGD at lr 0.005 — a low
rate on a 30-epoch budget, where the control tops out at 75 % against the 91 % a
properly trained ResNet-20 reaches. This branch asks whether the gain is a
property of the continuation path or of optimization speed.

## The result

Final test accuracy (%), mean over seeds 0–2, and the paired gain over `plain`:

| method | SGD | Adam | AdamW | RAdam |
|---|---:|---:|---:|---:|
| `plain` *(control)* | 75.43 | 84.13 | 84.12 | 84.38 |
| `resolution_max_b1` | 80.37 _(+4.94)_ | 85.31 _(+1.18)_ | **86.28** _(+2.16)_ | 85.63 _(+1.25)_ |
| `gaussian_postrelu` | 79.91 _(+4.48)_ | 83.95 _(−0.18)_ | 83.71 _(−0.41)_ | 84.75 _(+0.37)_ |
| `resolution_max_b1_gaussian_conv` | 81.51 _(+6.08)_ | 85.28 _(+1.15)_ | 86.21 _(+2.09)_ | 85.26 _(+0.88)_ |

Three things follow.

* **The optimizer is worth more than the intervention.** `plain` gains +8.95 pt
  from the optimizer alone, against +4 to +6 pt for continuation. A plain
  ResNet-20 under any adaptive optimizer beats every arm of the frozen benchmark.
* **The resolution reduction survives, four times smaller** — +4.94 becomes
  +1.18 / +2.16 / +1.25, same sign on all three seeds under all four optimizers.
* **The Gaussian blur does not.** +4.48 becomes −0.18 / −0.41 / +0.37, with the
  sign inconsistent across seeds under Adam and AdamW, and it adds nothing on top
  of the resolution reduction (+1.14 pt under SGD, −0.03 / −0.07 / −0.37
  otherwise).

Learning rates were swept per optimizer on the control before the grid ran and
then frozen — comparing optimizers at a fixed rate would measure nothing but a
mistuned rate. The reuse of the recorded SGD arms was validated by rerunning the
control, which reproduced to within 0.03 pt on two seeds of three.
[REPORT.md](REPORT.md) has the protocol, the per-seed numbers and the limits;
[docs/LR_SWEEP.md](docs/LR_SWEEP.md) has the sweep.

## The methods

| method | what it does | test acc under SGD, 3 seeds |
|---|---|---|
| `plain` | ResNet-20 BN, no intervention | 75.43 ± 0.76 |
| `resolution_max_b1` | max-pool the output of block 1 to 16 → 24 → 32 | 80.37 ± 0.61 |
| `gaussian_postrelu` | annealed Gaussian at the 10 post-ReLU tensors | 79.91 ± 0.27 |
| `resolution_max_b1_gaussian_conv` | block-1 reduction + annealed Gaussian at 19 conv outputs, sigma × r/32 | 81.51 ± 0.22 |

These are representatives of three families, **not** a factorial design. The
Gaussian placement and sigma scaling differ between the Gaussian-only and
combined methods. Exact schedules, sites and bypass rules:
[docs/METHODS.md](docs/METHODS.md). The exploratory history (wavelets, TV, other
reduction operators, adaptive schedules, every raw result) is on branch
`benchmark-organized`; nothing here imports it.

## Install

```bash
py -m pip install -r requirements.txt
```

Data is external: put the CIFAR-10 python batches in `data/cifar-10-batches-py/`
(or pass `--data-root`). STL-10: `data/stl10_binary/`. The pinned reference
assets are versioned in `assets/cifar10_resnet20bn/`:

```bash
py -m continuation_core verify-assets
```

## Running one arm

```bash
py -m continuation_core describe --method resolution_max_b1_gaussian_conv
```

```bash
py -m continuation_core dry-run --method resolution_max_b1_gaussian_conv --data-root data
```

```bash
py -m continuation_core train --method resolution_max_b1 --seed 0 --data-root data --out runs
```

Swap the optimizer with `--optimizer {sgd,adam,adamw,radam}`. It needs an
explicit `--lr`: nothing about SGD's 0.005 carries over to an adaptive
optimizer, so no rate is inherited. `--weight-decay` defaults to the reference
5e-4. Any optimizer other than the reference marks the run
`optimizer-variant` rather than `reference`, and the run directory carries the
optimizer and the rate, so arms never overwrite each other.

```bash
py -m continuation_core train --method resolution_max_b1 --optimizer adamw --lr 2e-2 --seed 0 --data-root data --out runs
```

```bash
py -m continuation_core train --method resolution_max_b1 --seed 0 --data-root data --out runs --resume runs/resolution_max_b1__sgd_lr0.005__seed0/rolling.pt
```

```bash
py -m continuation_core evaluate --checkpoint runs/resolution_max_b1__sgd_lr0.005__seed0/checkpoints/epoch_006.pt --state "used;next;target" --bn-policy running_stats,fixed_batch_stats --split test
```

`dry-run` builds everything, verifies asset digests, runs forward/backward at
every distinct state and checks the target-state bypass. It trains nothing.
`train` writes the tree described in
[docs/RESULTS_SCHEMA.md](docs/RESULTS_SCHEMA.md); an example is in
`docs/examples/synthetic_run/`. Resumption is mid-epoch and carries the RNG.

## Running the campaign on Kaggle

```bash
py scripts/stage_assets.py                                    # once: publish the pinned files
py scripts/kaggle_run.py scripts/job_optbench_lr_sweep.py --gpu \
    --accelerator NvidiaTeslaT4 \
    --dataset alexandrecorrard/cifar-10-batches-py \
    --dataset alexandrecorrard/continuation-core-r20bn-assets \
    --include continuation_core --include scripts --timeout-seconds 28800
py scripts/aggregate_optimizers.py
```

`kaggle_run.py` zips the package into the kernel source, pushes, polls and pulls
the output back. The campaign is split into slices that each own a deterministic
share of the cell list, so a cell belongs to exactly one kernel however the
slices are launched. Kaggle allows **two concurrent GPU kernels**; a third push
is refused. Details, launchers and per-slice costs:
[docs/kaggle_cli.md](docs/kaggle_cli.md).

## Analysis tools

Save checkpoints around transitions, then export, project and evaluate:

```bash
py -m continuation_core train --method resolution_max_b1 --seed 0 --data-root data --transition-offsets -50,-1,1,50
```

```bash
py -m continuation_core export-trajectory --runs runs/plain__sgd_lr0.005__seed0 runs/resolution_max_b1__sgd_lr0.005__seed0 --out analysis/traj.npz
```

```bash
py -m continuation_core pca-plane --trajectory analysis/traj.npz --center mean --out analysis/plane.pt
```

```bash
py -m continuation_core plane-loss --checkpoint runs/resolution_max_b1__sgd_lr0.005__seed0/checkpoints/epoch_012.pt --plane analysis/plane.pt --grid -20:20:21 --states "used;target" --bn-policies fixed_batch_stats --out analysis/plane_loss.json
```

```bash
py -m continuation_core perturb --checkpoint runs/resolution_max_b1__sgd_lr0.005__seed0/checkpoints/epoch_030.pt --epsilons 0.01,0.05,0.1 --directions 10 --states target --bn-policies running_stats,fixed_batch_stats --out analysis/perturb.json
```

* **Question 1.** Loss at identical weights under different states:
  `evaluate --state "used;next;target"` on a transition checkpoint, or any
  explicit `r=16,sigma=0.5`.
* **Question 2.** Trajectories: `export-trajectory` then `pca-plane`. The plane
  JSON reports explained variance and per-point projection residuals, so a
  misleading 2-D picture is visible.
* **Question 3.** Sensitivity: `perturb` with filter-normalised directions.

BatchNorm policy is always explicit. `running_stats` evaluates stored
checkpoints as they are. `fixed_batch_stats` uses per-batch statistics on
cloned buffers, for weights without matching running statistics. Outputs
record the path label, state, per-site sigma and BN policy. Python API:
`continuation_core.analysis.CheckpointEvaluator`, `pca_plane`, `plane_grid`,
`perturbation_sensitivity`, `to_vector` / `from_vector`.

## Transfer

Dataset, architecture, optimizer, method, budget, evaluation and checkpoint
cadence are independent config sections (`continuation_core/config.py`). The
reference preset is `continuation_core.presets.reference`. Transfer configs come
from `presets.transfer`, which requires every decision (resolution ratios,
sigma units, schedule duration, insertion mapping, optimizer lr and weight
decay) and marks the result `unvalidated`. Available adapters: `cifar10`,
`stl10`; `resnet20_bn_cifar`, `vgg11_bn` (no `block1` mapping, fails clearly);
`sgd`, `adam`, `adamw`, `radam`. Guide: [docs/EXTENDING.md](docs/EXTENDING.md).

Adam and RAdam pin `decoupled_weight_decay=False` explicitly rather than
inheriting the torch default, which can move between releases. Adam and AdamW
therefore differ by that one flag and nothing else.

## Theory

The final records keep training-probe CE and test CE on both the current and
target paths each epoch (`metrics.json`), and checkpoints hold the full state.
That is the raw material for relating higher final training CE to lower test
CE without retraining.

## Verification

[docs/VERIFICATION.md](docs/VERIFICATION.md). Bitwise parity with the code that
actually ran the unified batch (44 checks, still 44/44 on this branch), unit
tests (`py -m pytest -q`, 53 tests), and remaining limitations. The campaign's
SGD control runs are the first complete 30-epoch runs made with this code, and
they reproduce the recorded numbers to within 0.03 pt on two seeds of three —
closing the "no full training run" gap that document lists.


---

# `docs/OPTIMIZER_BENCHMARK.md`

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


---

# `docs/LR_SWEEP.md`

# Learning-rate sweep

Adam, AdamW and RAdam had never been run anywhere in this repository, so there
was no basis for a learning rate. The reference 0.005 is an SGD value and
nothing about it transfers. This sweep picks one per optimizer before the grid
runs.

**Protocol.** `plain` only, seed 0, the reference recipe with the optimizer and
learning rate swapped and nothing else: 30 epochs, no augmentation, weight decay
5e-4 on every parameter, 60 warmup updates then cosine. Pinned assets verified
on the worker. Kernel
`alexandrecorrard/job-optbench-lr-sweep-20260914-100029`, 2x T4, 12/12 cells
complete, none diverged.

**Selection rule, fixed before the results were read:** highest final test
accuracy on `plain`, seed 0; ties inside 0.3 pt go to the lower learning rate;
a diverged arm is discarded and recorded. Tuning on `plain` gives the control
its best showing, so the bias runs *against* continuation — the right direction
for a test of whether continuation survives.

## Results

Final test accuracy, target path, epoch 30.

| lr | adam | adamw | radam |
|---|---:|---:|---:|
| 3e-4 | 75.37 | 73.89 | 74.09 |
| 1e-3 | 81.29 | 79.79 | 80.07 |
| 3e-3 | **83.95** | 82.98 | **84.58** |
| 1e-2 | 81.97 | 83.80 | 84.42 |
| 2e-2 | — | **84.52** | — |
| 3e-2 | — | 83.83 | — |

Reference for scale: `plain` under SGD at lr 0.005 is 75.43 ± 0.76 over three
seeds, and 74.58 on seed 0 — the same seed and the same pinned assets as every
cell above.

## Boundary extension

AdamW is the only optimizer whose best value landed on an endpoint of the grid,
which means the grid did not bracket its optimum. Adam and RAdam both peak in
the interior. Comparing a boundary-limited AdamW against two interior-optimum
arms would put back exactly the "learning rate not tuned for this optimizer"
confound the sweep exists to remove, so the rule applied is: **extend past any
endpoint optimum**. Only AdamW qualifies, and it was extended to 2e-2 and 3e-2
(kernel `job-optbench-lr-sweep-ext-20260914-120927`, 2/2 complete). The rule is
symmetric — it would have been applied to any optimizer that hit an endpoint, in
either direction.

The extension brackets the optimum: AdamW rises to 84.52 at 2e-2 and falls back
to 83.83 at 3e-2. All three optimizers now peak in the interior of their grid,
so none is boundary-limited going into the campaign.

## Chosen

| optimizer | lr | `plain` seed 0 | why |
|---|---|---:|---|
| sgd | 5e-3 | 74.58 | the reference recipe, unchanged |
| adam | 3e-3 | 83.95 | interior maximum |
| adamw | 2e-2 | 84.52 | interior maximum after the extension |
| radam | 3e-3 | 84.58 | 84.58 against 84.42 at 1e-2 is inside 0.3 pt, so the tie rule takes the lower rate |

At their chosen rates the three adaptive optimizers land within 0.63 pt of each
other (83.95 / 84.52 / 84.58). The grid therefore compares optimizers that have
each been given their best shot, which is the condition under which a Δ
difference between them means something.

## What this already says about the benchmark

The sweep was meant to be preparation. It is not.

`plain` reaches **84.58 %** under RAdam at 3e-3. The best *continuation* arm
ever recorded on this recipe is `resolution_max_b1_gaussian_conv` at
**81.51 ± 0.22** under SGD. A plain ResNet-20, with no intervention of any
kind, beats every continuation arm in the frozen benchmark by about three
points once the optimizer is allowed to be a reasonable one.

The gap between the SGD control and the adaptive-optimizer control is **+9.2 pt**
(74.58 → 84.58, same seed, same initial weights, same data order). The entire
continuation effect the benchmark reports is +4 to +6 pt. So the intervention
was measured against a baseline that was leaving roughly twice the intervention's
own effect on the table.

This does **not** show that continuation does nothing. It shows that the
headline numbers cannot be read as "continuation is worth ~5 points", because a
change of optimizer is worth about twice that on the same control. Whether
continuation still adds anything *on top of* a well-tuned adaptive optimizer is
exactly what the grid measures, and it is now the whole question.

Two things to keep in mind when reading the grid:

* The effect could be **subsumed** — if continuation buys speed of optimization
  early, an optimizer that already adapts its per-parameter step may capture the
  same thing, and Δ collapses toward zero.
* It could also **survive**, at reduced size, if the mechanism is regularisation
  or a genuine easier-to-harder path rather than conditioning. The paired Δ is
  what separates these, and it is computed inside each optimizer so the level
  difference above cancels.

Either outcome is a result. The one reading now foreclosed is the one where the
SGD table stands on its own.
