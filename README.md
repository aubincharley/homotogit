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
