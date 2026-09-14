# continuation-core — optimizer benchmark

> **Lire d'abord : [RAPPORT.md](RAPPORT.md).** Cette branche mesure les quatre
> méthodes gelées sous quatre optimiseurs (SGD, Adam, AdamW, RAdam), et le
> résultat change la lecture du tableau ci-dessous : à optimiseur correctement
> réglé, le témoin `plain` atteint 84,4 % — au-dessus de **tous** les bras
> ci-dessous. Le gain de la réduction de résolution survit à ~1-2 points ; celui
> du flou gaussien disparaît.
>
> Les chiffres du tableau suivant restent exacts **sous SGD lr 0,005**, qui est
> la seule condition dans laquelle ils ont été mesurés.

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
