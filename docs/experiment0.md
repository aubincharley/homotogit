# Experiment 0 — independent training at fixed Gaussian levels

## Question

How does training at a *fixed* Gaussian smoothing level affect optimization
progress on that transformed problem, generalization on that transformed
problem, and transfer to the original (unfiltered) classification problem?

This is the fixed-parameter special case of the proposed continuation. **No
continuation run is performed**: no weights are reused across levels and no
transformation changes during a run.

## Protocol

| Item | Setting |
|---|---|
| Dataset | CIFAR-10, official train set split 45,000 / 5,000 (stratified, `split_seed=12345`) |
| Test set | **untouched** — reserved for final comparisons |
| Augmentation | none (no crops, flips, rotations, colour jitter) |
| Normalization | per-channel, fitted on the **training subset only**, on **unfiltered** images, identical in every run, applied **after** the transformation |
| Model | ResNet-20 with GroupNorm (`channels_per_group=8`), option-A shortcuts, 269,722 parameters |
| Optimizer | SGD, momentum 0.9, batch size 128, weight decay 5e-4 (fixed across conditions, excluded from reported CE) |
| LR schedule | 0.1 peak, 400-step linear warmup, cosine to 0, **indexed by global step** |
| Budget | identical gradient-update count in every condition (see below) |
| Grid | $\sigma \in \{0, 0.5, 1, 2, 3\}$ pixels — pre-declared |
| Seeds | 3 paired seeds (`[0, 1, 2]`); config supports 5+ |

**Pairing.** For a given seed, every level uses the same parameter
initialization *and* the same sequence of minibatch sample indices. This is
enforced structurally, not by sharing one seed: `continuation/seeding.py`
derives independent named streams, and `BatchIndexStream` draws the batch order
from its own stream, so a condition that consumed randomness differently would
still see identical batches. The Gaussian transform is deterministic and draws
no random numbers at all.

**Grid selection.** The grid was fixed before the main comparison, after the
visual/pilot phase, and the same grid is used for every reported condition. No
per-condition range was chosen and the test set was not used to choose anything.
Any adjustment made during the visual/pilot phase is recorded in
`docs/results.md`.

## Recorded at every shared checkpoint

Checkpoints are every `eval_every` gradient updates (plus step 0 and the final
step), at identical steps in every condition. All four are **eval-mode**,
no-gradient metrics on **fixed documented subsets**, and all cross-entropies are
**unregularized** (weight decay never enters the reported loss):

1. `transformed_train_probe` — CE and accuracy at the run's own level, on a fixed
   class-balanced 5,000-image probe subset of the training split
   (full-train evaluation at every checkpoint would be needlessly expensive);
2. `transformed_val` — CE and accuracy at the run's own level, full 5,000-image
   validation split;
3. `target_train_probe` — CE and accuracy on **original unfiltered** images at
   the same weights, same probe subset;
4. `target_val` — CE and accuracy on **original unfiltered** images, full
   validation split.

Separately logged and never merged with the above:

* `train_minibatch` — training-mode minibatch CE/accuracy averaged over the
  logging window. These are noisy, are measured during the update, and carry an
  explicit `note` field saying they are not eval-mode metrics.
* `transform_stats` — reconstruction MSE and retained TV ratio
  $\operatorname{TV}(T_\sigma x)/\operatorname{TV}(x)$ on a fixed 512-image
  subset, with the TV convention recorded inline and zero-TV images counted
  separately rather than assigned a ratio of 1.
* `timing` — wall time, and **transformation/preprocessing cost separately**.

At $\sigma=0$ the transformed and target objectives are the same objective, so
the trainer reuses the computation and the two sets of numbers are identical by
construction.

## Interpretation limits

These are constraints on what may be concluded, not caveats to be waived:

* Experiment 0 does **not** establish that transformed losses have wider
  attraction basins, nor that continuation will improve the final result.
* Validation performance measures generalization on a *particular transformed
  task*. It is not a direct measurement of optimization difficulty.
* Raw losses from different transformed tasks may have **different attainable
  minima**. A quick plateau is not evidence of a better optimization problem,
  and information loss can raise the best attainable classification error.
* Original-image evaluation of a model trained only on transformed images
  measures **transfer under a distribution change**. It is informative, but it
  is not a substitute for actually testing a warm start on the target loss.
* Small gradient norms alone do not imply useful convergence.
* Common optimizer settings give a *controlled primary comparison*, not proof
  that each task was optimized as well as it could be. If instability drives a
  result it is documented, and a **clearly separated, pre-declared** LR
  sensitivity check is used rather than silently tuning individual conditions.
* Equal update counts control the **optimization budget**, not total compute:
  transformation cost is reported separately, and it will matter much more for
  future iterative TV transformations than it does for a 25-tap separable blur.

The curves and summaries are therefore reported as separate quantities —
transformed-task optimization progress, transformed-task generalization, and
transfer to the target task — and are **never merged into a single "ease of
optimization" score**.

## Out of scope in this phase

Deliberately not implemented: TV solvers, compression/rate methods, feature-map
smoothing, activation or parameter-space homotopies, adaptive continuation, and
any full continuation run. Schedules for future continuation *are* implemented
and unit-tested (`continuation/schedules.py`) but are not exercised by any run
here; `build_schedule(kind="adaptive")` raises `NotImplementedError` on purpose.

## Reproducing

```bash
py -m pytest tests -q
py -m continuation.cli prepare-data --config configs/exp0_gaussian.yaml
py -m continuation.cli visualize    --config configs/exp0_gaussian.yaml
py -m continuation.cli exp0         --config configs/exp0_gaussian.yaml
py -m continuation.cli report       --config configs/exp0_gaussian.yaml
```

Each run writes `run_description.json` (full config, resolved seeds, split
fingerprint, probe subset, software versions, GPU), `metrics.jsonl`,
`transform_stats.json`, `summary.json` and a final checkpoint. `exp0` is
resumable: completed runs are skipped unless `--overwrite` is passed.
