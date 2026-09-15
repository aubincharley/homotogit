# Loss-landscape study v1: report

This study evaluated existing checkpoints only; no training was run. Commands
and raw data are described in [docs/LANDSCAPE_STUDY.md](../../docs/LANDSCAPE_STUDY.md),
and the numbers come from `figures/summary.json`. Figures and a suggested
reading order are listed in [FIGURES.md](FIGURES.md).

## 0. Scope: what could and could not be studied

**Covered.** Every figure compares **plain against resolution-only** on the
**resbench checkpoints**. The resbench batch saved checkpoints at epochs
6/12/18/30 for two cells, seeds 0–2 (24 checkpoints in total), plus the pinned
initial weights:

- `none__none__none` = plain;
- `max__D1__Rprog` = `resolution_max_b1`: adaptive max-pool pre-hook on
  `blocks[2]`, r = 16/24/32, identity at 32.

These runs match the unified configurations, as recorded in
`inputs/sources_report.json`:

- identical asset digests, so both methods share initialization and data order
  within each seed;
- the same recipe;
- recorded per-epoch resolution equal to the core schedule;
- strict checkpoint loading.

They are nevertheless **other runs** than the unified reference runs:

| method | seed 0 | seed 1 | seed 2 | unified batch (same config) |
|---|---|---|---|---|
| plain | 75.31 | 75.34 | 75.95 | 75.43 ± 0.76 |
| resolution-only | 80.37 | 80.67 | 80.75 | 80.37 ± 0.61 |

**Not available.**

- The unified runs kept no checkpoint: `unified_driver.py` deletes `rolling.pt`
  at the end of each cell.
- No batch with saved checkpoints ran `gaussian_postrelu` or
  `resolution_max_b1_gaussian_conv`.
- Gaussian-only and combined visualizations are therefore absent. They are
  **not** replaced by illustrative surfaces, inferred trajectories or other
  checkpoints.
- For the same reason, the segments plain–Gaussian, plain–combined and
  resolution-only–combined are absent.

The following were fixed in `inputs/preregistration.json` before any
evaluation: seeds, checkpoints, direction draws, grids, and the rule for
changing the range.

## 1. Conventions

- **Loss.** Mean cross-entropy (CE) without weight decay; accuracy is also
  recorded.
- **Probes.** Class-balanced sets of 1,000 training and 1,000 test images
  (100 per class). Calibration uses 2,000 training images, disjoint from the
  training probe; test data are never used for calibration. Probe values are
  not full-dataset values.
- **Recalibrated BatchNorm.** Used for every landscape, surface, segment and
  plane.
  1. Reset the running statistics.
  2. Take a cumulative average over the calibration set: fixed order, batches
     of 500, training mode, `no_grad`, under the evaluated intervention state.
  3. Evaluate in inference mode.

  No learned parameter changes, and ResNet-20 has no dropout.
- **Saved statistics.** Used only for unperturbed checkpoints. On the Kaggle
  T4s they reproduce every recorded metric of the 24 checkpoints exactly
  (maximum absolute error 0).
- **3D surfaces.** Every vertex is a measured grid point, and faces are the
  standard patches between neighbouring vertices. There is no smoothing,
  extrapolation or added point. Within a comparison, the range, camera, box
  aspect, vertical limits and colour scale are shared; train and test have
  separate scales. Centre-subtracted surfaces keep negative values.
- **Checks.** All passed locally and on the three T4s (`results/*/checks_job*.json`):
  - direction normalization (relative block-norm error below 1e-15; no
    zero-norm blocks);
  - bitwise reconstruction and interpolation endpoints;
  - PCA coordinates;
  - r = 32 bitwise identical to plain;
  - repeatable, order-independent evaluation;
  - no mutation of checkpoint files or loaded tensors.

  No non-finite value occurred.

### BatchNorm sensitivity (saved vs recalibrated, full test set)

| checkpoint (resolution-only) | evaluated state | saved statistics | recalibrated |
|---|---|---|---|
| epoch 6, seeds 0 / 1 / 2 | target r = 32 | 13.1 / 10.6 / 16.5% | 34.3 / 39.6 / 43.3% |
| epoch 6, seeds 0 / 1 / 2 | own r = 16 | 65.8 / 67.2 / 67.4% | 66.6 / 68.2 / 68.6% |
| epoch 12, seeds 0 / 1 / 2 | target r = 32 | 24.3 / 20.2 / 14.4% | 58.3 / 65.3 / 63.8% |
| epoch 12, seeds 0 / 1 / 2 | own r = 24 | 75.4 / 75.4 / 76.4% | 75.6 / 76.0 / 76.7% |

For epoch-30 checkpoints of both methods, saved and recalibrated test accuracy
differ by at most 0.18 points.

- **Statistics mismatch.** Part of the early target-path degradation comes from
  BatchNorm statistics accumulated under the reduced state. Recalibration
  raises target accuracy by 21–49 points at epochs 6 and 12.
- **Degradation remaining after recalibration.** At the target state,
  recalibrated accuracy at those checkpoints is still 11–32 points below the
  accuracy under the state the weights were trained in.
- **Consequence for the paper.** The saved-statistics target-path diagnostic in
  the paper therefore combines both effects.

## 2. Q1: does the intervention change the landscape at fixed weights?

**Setting (Analysis B).**

- Resolution-only checkpoint, seed 0, after epoch 6: weights produced by
  updates at r = 16.
- Weights, centre, directions (pair 0), subsets, axes and colour scales are
  identical across states. Only the block-1 reduction changes, and BatchNorm is
  recalibrated under each state.
- Only the resolution family could be evaluated.
- Figures: `B_surface3d_absolute`, `B_surface3d_centered`,
  `B_fixed_weights_resolution_states` (contours), `B_secondary_1d`.

**Observations.**

| state | train CE, centre | test CE, centre | mean test ΔCE over the 21×21 grid | lowest sampled test CE minus centre |
|---|---|---|---|---|
| r = 16 | 0.817 | 0.921 | 0.201 | −0.001 (adjacent vertex) |
| r = 24 | 1.065 | 1.073 | 0.192 | −0.001 (adjacent vertex) |
| r = 32 (target) | 1.847 | 1.925 | 0.095 | −0.094 at (a, b) = (0.15, −0.075) |

- **Loss level.** Absolute surfaces rise with the state. At every grid point
  the r = 32 surface lies 0.59–1.07 nats (train) and 0.64–1.04 nats (test)
  above the r = 16 surface; r = 24 lies 0.10–0.26 above r = 16.
- **Shape at r = 16 and r = 24.** Centre-subtracted surfaces are bowls of
  visually similar shape and depth. The centre lies within 0.002 of the lowest
  sampled point.
- **Shape at r = 32.** The surface is tilted: the centre is **not the lowest
  sampled point**, since grid points toward +D, −E are up to 0.10 lower. The
  mean increase over the grid is smaller (0.095). That is a finite-grid
  sensitivity statistic, and here it largely reflects the tilt at a higher
  loss level.
- **Epoch 12 (last updates at r = 24).** CE is lowest under r = 24 (train 0.56,
  test 0.69), then r = 16 (0.87 / 0.89), then r = 32 (1.16 / 1.18).
- **Control.** Plain epoch-6 weights are lowest at r = 32 (test 0.98), then
  r = 24 (1.13), then r = 16 (1.48).

**Interpretation.**

- At fixed weights, changing the intervention mostly shifts the loss **level**,
  and each set of weights has its lowest loss under the state it was trained in.
- Within the ±0.25 relative-perturbation range, the r = 16 and r = 24 slices
  look alike in shape. These slices give no indication that the reduced
  objective is smoother in parameter space around these weights.
- Mean grid increase is not a curvature measurement, and equal curvature is not
  claimed.
- These are counterfactual evaluations of one set of weights, not three trained
  models.

**Unresolved.**

- The analysis uses one seed, one direction pair and one checkpoint per phase.
- Gaussian and combined states have no checkpoints.
- Whether the r = 32 centre is a local minimum in parameter space is not
  determined: the grid only shows lower sampled points in this 2D slice.

## 3. Q2: how do the trajectories differ?

**Setting (Analysis C).**

- Plain and resolution-only, seed 0, **five checkpoints per run**: updates
  0 / 2346 / 4692 / 7038 / 11730 (epochs 0/6/12/18/30). Epoch 0 is the shared
  initialization.
- The vector holds all 269,722 learned parameters, including BN affine;
  running statistics are excluded.
- One PCA basis is fitted over the 10 points, centred at their mean.
- Figures:
  - `C_pca_trajectories`: the main view, for reading the projected paths;
  - `C_surface3d_plane`: the evaluated plane in 3D, with projected checkpoints
    drawn on the floor only;
  - `C_checkpoint_ce`: actual checkpoint CE.

**Projection quality.**

- **Variance explained.** PC1 and PC2 explain **91%** of the variance of the
  10 points (55% and 36%). With only 10 points, a high value is expected.
- **Off-plane distance.** Relative projection residuals are 0.13–0.23 for
  epochs 0, 18 and 30, but **0.64 (plain) and 0.66 (resolution-only) at epoch
  6**, and 0.30 and 0.35 at epoch 12. The epoch-6 checkpoints lie far from the
  plane, and their projected positions should be read with that in mind.

**Observations, restricted to the sampled checkpoints.**

- At epoch 6, the two projected checkpoints lie on opposite sides of PC1 = 0
  (−3.5 and +3.3). They are also on opposite sides at epochs 12, 18 and 30,
  ending at −6.8 and +7.3. Nothing is observed between checkpoints: the
  connecting lines are drawn for readability and are not optimization paths.
- Both projected sequences move in the same direction along PC2 across epochs
  6–30.
- The learning rate warms up linearly over updates 0–59, then decays by cosine,
  with no step change. The resolution changes after the epoch-6 checkpoint
  (r = 16 → 24) and after the epoch-12 checkpoint (24 → 32). No checkpoint
  exists inside a transition window.
- The target loss on the plane has two low regions, around the projections of
  the two sets of late checkpoints, separated by a higher ridge near PC1 ≈ 0.
  This describes the reconstructed in-plane points, not the full space.

Actual checkpoint CE (probe splits; `C_checkpoint_ce`):

| epochs completed | plain, recalibrated, target: train / test | res.-only, recalibrated, target: train / test | res.-only, recalibrated, own state: train / test |
|---|---|---|---|
| 0 | 2.31 / 2.31 | 2.31 / 2.31 | (no update yet) |
| 6 | 0.89 / 0.98 | 1.85 / 1.93 | r = 16: 0.82 / 0.92 |
| 12 | 0.59 / 0.74 | 1.16 / 1.18 | r = 24: 0.56 / 0.69 |
| 18 | 0.39 / 0.71 | 0.45 / 0.58 | = target |
| 30 | 0.22 / 0.73 | 0.30 / 0.54 | = target |

All entries use recalibrated BatchNorm. The saved-statistics counterparts are
plotted separately in `C_checkpoint_ce`. For resolution-only at the target
state they are 4.12 / 4.14 at epoch 6 and 2.97 / 3.00 at epoch 12; at the own
state, 0.84 / 0.95 and 0.57 / 0.71.

**Interpretation.**

- **Separation.** At every sampled checkpoint after initialization, the two
  runs occupy clearly separated positions in this projection.
- **Different objective.** Early resolution-only checkpoints were trained on a
  different objective, consistent with their high target-state loss and low
  own-state loss.
- **What the projection cannot show.** With five checkpoints and large early
  residuals, it does not show when or how the runs diverge, nor what happens
  between checkpoints.

**Unresolved.** Denser checkpoints, including windows around the transition
updates, and more seeds would be needed.

## 4. Q3: how do the final solutions differ?

### Local sensitivity (Analysis A, robustness slices)

**Setting.**

- Epoch-30 solutions, target state, recalibrated BatchNorm.
- Filter-normalized directions: the same random draws scaled by each solution's
  own filter norms. These are **matched relative perturbations, not an
  identical physical plane**.
- Range ±0.25, kept by the preregistered offset check.
- Figures: `A_surface3d_absolute`, `A_surface3d_centered`,
  `A_local_landscapes_seed0` (contours), `A_robustness_1d_slices`.

**Observations.**

- **2D surfaces (seed 0, pair 0).** Each centre is the lowest of the 441
  sampled points of its slice.
  - Centre CE: plain has the lower train CE (0.224 vs 0.305) and the higher
    test CE (0.726 vs 0.537).
  - Mean CE increase over the grid: plain 0.67 vs resolution-only 0.47 on
    train, 0.47 vs 0.38 on test.
- **1D robustness.** CE increase at |a| = 0.25, comparing the two methods along
  the same draw:
  - comparisons: 6 directions per seed × 3 seeds = 18. There is one trained
    solution per method per seed, so these are **not 18 independent training
    runs**, and the directions share each seed's two solutions;
  - training probe: the increase is smaller for resolution-only in **18 of 18**
    comparisons (per-seed means: plain 0.97 / 1.03 / 1.02, resolution-only
    0.73 / 0.70 / 0.81);
  - test probe: it is smaller in **15 of 18** (plain 0.68 / 0.72 / 0.76,
    resolution-only 0.60 / 0.58 / 0.67). The exceptions are seed 0 pair 1
    (−0.0003) and seed 2 pairs 1 and 2.
- The 2D grid row b = 0 equals the separately computed 1D slice exactly.

**Interpretation.**

- **Positive finding.** Under the tested relative filter perturbations,
  resolution-only exhibits smaller loss increases, consistently on the training
  probe and in most test-probe comparisons.
- **Held-out performance, observed separately.** Resolution-only has lower test
  CE and higher test accuracy.
- **What does not follow.** These visualizations do **not** establish that the
  reduced sensitivity causes the better held-out performance. The statistic
  depends on the normalization and on BatchNorm parameterization (Dinh et al.,
  2017), and the two solutions start from different centre losses.

### Straight segments (Analysis D)

**Setting.**

- Plain (α = 0) to resolution-only (α = 1) at epoch 30, seeds 0–2, 51 points.
- All learned parameters are interpolated; BatchNorm is recalibrated; target
  state.
- Figure: `D_interpolation`.

**Observations.**

| seed | train CE: plain / res. / max (α) | train barrier | test CE: plain / res. / max (α) | test barrier | lowest test-probe acc |
|---|---|---|---|---|---|
| 0 | 0.224 / 0.305 / 1.179 (0.50) | 0.874 | 0.726 / 0.537 / 1.283 (0.52) | 0.557 | 55.6% |
| 1 | 0.201 / 0.290 / 1.150 (0.50) | 0.860 | 0.715 / 0.536 / 1.243 (0.46) | 0.528 | 58.1% |
| 2 | 0.195 / 0.285 / 1.088 (0.52) | 0.803 | 0.746 / 0.553 / 1.181 (0.50) | 0.435 | 63.0% |

The barrier is a grid estimate: B = max over the 51 α values minus the higher
endpoint loss.

**Interpretation.**

- In every seed, the **straight segment** between the paired solutions passes
  through a region of higher loss: 0.80–0.87 nats above the higher endpoint on
  the training probe, 0.43–0.56 on the test probe.
- This concerns straight segments only. It does not establish disconnected
  basins, since a curved low-loss path may exist and was not searched for.

## 5. What the study does not show

- **Other methods.** Nothing about Gaussian-only or combined methods, which have
  no checkpoints.
- **Smoothing.** Neither convexification nor smoothing of the loss in parameter
  space by the intervention.
- **Mechanism.** No causal link between reduced perturbation sensitivity, or
  the straight-line barrier, and the held-out improvement.
- **Transitions.** No behaviour between checkpoints or inside transition
  windows.

## 6. Unresolved issues

1. **Gaussian and combined checkpoints.** New training with saved checkpoints
   would be required, which is outside this study.
2. **Transition windows.** Checkpoints at transition updates ± k are needed.
3. **Batch identity.** The resbench runs are not the runs behind the paper's
   Table `unified-reference`, and any figure used in the paper must say so.
4. **Seeds.** The 2D surfaces, Analysis B and PCA exist for seed 0 only. The 1D
   slices and segments cover seeds 0–2.
