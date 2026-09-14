# continuation-core

Three frozen continuation methods, the plain control, and the shared code to
train, checkpoint, evaluate and analyse them. The exploratory history
(wavelets, TV, other reduction operators, adaptive schedules, every raw result)
is on branch `benchmark-organized`. Nothing here imports it.

| method | what it does | source run (unified batch) | test acc, 3 seeds |
|---|---|---|---|
| `plain` | ResNet-20 BN, no intervention | `plain` | 75.43 ± 0.76 |
| `resolution_max_b1` | max-pool the output of block 1 to 16 → 24 → 32 | `shrink_b1` | 80.37 ± 0.61 |
| `gaussian_postrelu` | annealed Gaussian at the 10 post-ReLU tensors | `blur_relu` | 79.91 ± 0.27 |
| `resolution_max_b1_gaussian_conv` | block-1 reduction + annealed Gaussian at 19 conv outputs, sigma × r/32 | `shrink_b1_conv` | 81.51 ± 0.22 |

These are representatives of three families, **not** a factorial design. The
Gaussian placement and sigma scaling differ between the Gaussian-only and
combined methods. Exact schedules, sites and bypass rules:
[docs/METHODS.md](docs/METHODS.md).

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

## Reference run, dry run, evaluation

```bash
py -m continuation_core describe --method resolution_max_b1_gaussian_conv
```

```bash
py -m continuation_core dry-run --method resolution_max_b1_gaussian_conv --data-root data
```

```bash
py -m continuation_core train --method resolution_max_b1_gaussian_conv --seed 0 --data-root data --out runs
```

```bash
py -m continuation_core train --method resolution_max_b1_gaussian_conv --seed 0 --data-root data --out runs --resume runs/resolution_max_b1_gaussian_conv__seed0/rolling.pt
```

```bash
py -m continuation_core evaluate --checkpoint runs/resolution_max_b1_gaussian_conv__seed0/checkpoints/epoch_006.pt --state "used;next;target" --bn-policy running_stats,fixed_batch_stats --split test
```

`dry-run` builds everything, verifies asset digests, runs forward/backward at
every distinct state and checks the target-state bypass. It trains nothing.

## STL-10

The same four methods transferred to STL-10 / 96x96: sigma x3, resolution
`(48, 72, 96)` with reference 96, 60 epochs with every schedule boundary
doubled. Every decision lives in `scripts/stl10_configs.py` and is written into
each config; the reasoning is in [docs/EXTENDING.md](docs/EXTENDING.md).
3 seeds, Kaggle T4, records under `results/stl10/`.

| method | STL-10 test | paired vs plain | CIFAR-10 test |
|---|---|---|---|
| `resolution_max_b1` | **59.93 +- 0.45** | **+2.58 +- 0.13** | 80.37 +- 0.61 |
| `gaussian_postrelu` | 58.44 +- 1.55 | +1.09 +- 1.07 | 79.91 +- 0.27 |
| `plain` | 57.35 +- 0.53 | - | 75.43 +- 0.76 |
| `resolution_max_b1_gaussian_conv` | 56.06 +- 0.44 | **-1.29 +- 0.53** | 81.51 +- 0.22 |

The paired column is the per-seed difference from `plain`; since all four
methods share an initialization and a data order within a seed, it is far
tighter than the marginal sd and is the column to read.

**The CIFAR-10 ordering does not transfer.** `resolution_max_b1` helps in every
seed. `gaussian_postrelu` helps by a margin its own seed spread swallows
(+2.15, +1.11, +0.01). And `resolution_max_b1_gaussian_conv`, the best method on
CIFAR-10, is **below the control in every seed**.

Its target-path metric says why: for all 42 epochs of active blur the weights sit
at chance (0.100-0.111) on the plain network, against 0.28 for `gaussian_postrelu`
and 0.24 for `resolution_max_b1`. Filtering all 19 convolution outputs *before*
BatchNorm makes the solution wholly blur-dependent, and the schedule then leaves
only 18 epochs to recover. That is 720 updates here against 3,519 on CIFAR-10,
because the boundaries were stretched in **epochs** while an STL-10 epoch is 40
updates against CIFAR-10's 391. It is still the fastest-improving of the four at
epoch 60, so this looks recovery-starved rather than beaten -- untested either way.

No parity reference exists for any of this.

```bash
py -m continuation_core make-assets --dataset stl10 --data-root data \
   --arch resnet20_bn_cifar --epochs 90 --init-from assets/cifar10_resnet20bn \
   --out assets/stl10_resnet20bn
py scripts/stl10_configs.py --data-root data --seeds 0
py -m continuation_core dry-run --config configs/stl10/plain__seed0.json
py -m continuation_core train   --config configs/stl10/plain__seed0.json --out runs
```

`scripts/kaggle/push_stl10.py` wraps those four steps into one Kaggle kernel per
method, cloning this repository at a pinned commit so `summary.json` still
records a git SHA.
`train` writes the tree described in
[docs/RESULTS_SCHEMA.md](docs/RESULTS_SCHEMA.md); an example is in
`docs/examples/synthetic_run/`.

## Visualization tools 

Save checkpoints around transitions, then export, project and evaluate:

```bash
py -m continuation_core train --method resolution_max_b1 --seed 0 --data-root data --transition-offsets -50,-1,1,50
```

```bash
py -m continuation_core export-trajectory --runs runs/plain__seed0 runs/resolution_max_b1__seed0 --out analysis/traj.npz
```

```bash
py -m continuation_core pca-plane --trajectory analysis/traj.npz --center mean --out analysis/plane.pt
```

```bash
py -m continuation_core plane-loss --checkpoint runs/resolution_max_b1__seed0/checkpoints/epoch_012.pt --plane analysis/plane.pt --grid -20:20:21 --states "used;target" --bn-policies fixed_batch_stats --out analysis/plane_loss.json
```

```bash
py -m continuation_core perturb --checkpoint runs/resolution_max_b1__seed0/checkpoints/epoch_030.pt --epsilons 0.01,0.05,0.1 --directions 10 --states target --bn-policies running_stats,fixed_batch_stats --out analysis/perturb.json
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
`sgd`, `adamw`. Guide: [docs/EXTENDING.md](docs/EXTENDING.md).

## Theory 

The final records keep training-probe CE and test CE on both the current and
target paths each epoch (`metrics.json`), and checkpoints hold the full state.
That is the raw material for relating higher final training CE to lower test
CE without retraining.

## Verification

[docs/VERIFICATION.md](docs/VERIFICATION.md). Bitwise parity with the code that
actually ran the unified batch (44 checks), unit tests (`py -m pytest -q`), and
remaining limitations.
