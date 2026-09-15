# Loss-landscape study v2: report

Four frozen `continuation-core` configurations were retrained on five paired
seeds, 20 new runs in total. These are not the unified or resbench runs. Every
number below comes from `tables/` and `figures/`; commands are in
[docs/LANDSCAPE_V2.md](../../docs/LANDSCAPE_V2.md) and the figure index is
[FIGURES.md](FIGURES.md). The analysis choices were frozen before any result
existed, in `inputs/preregistration.json`.

## What reproduced and what did not

**Reproduced across all five seeds:**

- **Accuracy.** Every continuation method beat plain in every seed.
  - Full test set (mean ± SD over seeds): plain 75.65 ± 0.78, resolution-only
    80.21 ± 0.36, Gaussian-only 80.12 ± 0.49, combined 81.58 ± 0.30.
  - Paired gains per seed: resolution-only +3.4 to +5.8 points, Gaussian-only
    +3.3 to +5.6, combined +5.4 to +6.8.
- **Training loss.** All three reach a *higher* full-training-set cross-entropy
  (CE) than plain: 0.30 / 0.27 / 0.31 against 0.20.
- **Saved statistics.** Resolution-only and combined showed smaller loss
  increases than plain in 5/5 seeds, at every amplitude, on both probes.
- **Recalibrated statistics, training probe.** The same held in 5/5 seeds at
  every amplitude.
- **Straight segments.** Every one of the 20 segments had a loss barrier.
- **Probes and calibration.** Probe-based differences matched larger
  evaluation sets, and changing the calibration set from 2,000 to 10,000
  images left the sensitivity values essentially unchanged.

**Did not reproduce, or depends on the BatchNorm policy:**

- **Gaussian-only's ranking flips with the BatchNorm policy.**
  - Saved statistics: smaller increases than plain at ε ≥ 0.20 (4–5 seeds out
    of 5) and no consistent difference at ε ≤ 0.10.
  - Recalibrated statistics: *larger* increases than plain in 5/5 seeds at
    every ε ≥ 0.05, on both probes.
- **Resolution-only and combined on the test probe with recalibration.** The
  advantage is small and not uniform:
  - resolution-only: 3/5 seeds at ε = 0.025, 4/5 at 0.20, 5/5 otherwise; mean
    differences of 0.0002–0.08 nats against a plain value of 0.006–0.71;
  - combined: 4/5 seeds (seed 3 reverses at ε ≥ 0.10).
- **Validation at ε = 0.10.** On the full test set, resolution-only's
  difference is essentially zero (−0.006, 2/5 seeds below plain).

**Not resolved by this study:**

- whether any method *causes* a flatter neighbourhood in the full parameter
  space;
- whether lower sensitivity explains the accuracy gain;
- convexification.

## Terms used

- **Probe.** A fixed evaluation subset: 1,000 training or 1,000 test images,
  100 per class. It is not the full dataset. Larger checks use a
  10,000-image training subset and the full training and test sets.
- **Seed.** One training replication. The four methods of a seed share the
  initial weights, the BatchNorm buffers and the data order (identical digests
  in `checks/`). The five seeds are the replication unit; all spreads across
  seeds are descriptive SDs over n = 5.
- **Direction.** One random perturbation of the weights. It is generated from
  its own seed stream (separate from the training seed) and reused for the four
  methods of a seed. Twenty directions per network give 100 measurements per
  method, but they are *not* 100 independent training replications.
- **Relative perturbation.** Each output filter of a convolution, and each
  output row of the final linear layer, is moved by ε times its own norm (Li et
  al., 2018). Biases and BatchNorm scale/shift stay fixed. The same random
  pattern is rescaled to each network's own filter norms, so two methods receive
  matched *relative* perturbations, not the same weight-space vector.
- **Saved-statistics policy.** Evaluate the stored network with the BatchNorm
  running statistics it was saved with. This measures that stored model under
  the perturbation.
- **Recalibrated policy.** Recompute the BatchNorm running statistics from a
  fixed 2,000-image training calibration subset for every evaluated weight
  vector and state, then evaluate. This measures the network after its
  normalization has adapted; it is a different function, not "the true loss".
- **S(ε).** (L(θ+εd) + L(θ−εd))/2 − L(θ), the symmetric
  finite-perturbation sensitivity. It is not a Hessian measurement and not
  evidence of convexity.
- **Final state.** Full resolution with no filter; identical to plain
  ResNet-20.

## Q3: how do the final solutions differ in perturbation sensitivity?

### Primary measurement

Setup:

- epoch-30 checkpoints evaluated in the final state;
- 20 directions per network, both signs, ε ∈ {0.025, 0.05, 0.10, 0.20, 0.25};
- both BatchNorm policies, train and test probes;
- statistic averaged over directions within each seed first, then compared with
  plain within the seed.

Figures: `S1`, `S2`, `S3`. Tables: `sensitivity_per_seed.csv`,
`sensitivity_paired_vs_plain.csv`.

Mean over seeds of S̄(ε) (± SD across seeds) at ε = 0.10 and 0.25:

| policy, split | ε | plain | resolution-only | Gaussian-only | combined |
|---|---|---|---|---|---|
| saved, train probe | 0.10 | 0.735 ± 0.055 | 0.467 ± 0.025 | 0.711 ± 0.078 | 0.423 ± 0.047 |
| saved, train probe | 0.25 | 5.29 ± 0.36 | 3.82 ± 0.28 | 4.40 ± 0.18 | 3.53 ± 0.34 |
| saved, test probe | 0.10 | 0.628 ± 0.060 | 0.416 ± 0.027 | 0.638 ± 0.079 | 0.384 ± 0.035 |
| saved, test probe | 0.25 | 4.88 ± 0.36 | 3.64 ± 0.28 | 4.18 ± 0.19 | 3.37 ± 0.30 |
| recalibrated, train probe | 0.10 | 0.194 ± 0.015 | 0.132 ± 0.006 | 0.277 ± 0.041 | 0.126 ± 0.024 |
| recalibrated, train probe | 0.25 | 1.02 ± 0.06 | 0.77 ± 0.02 | 1.27 ± 0.10 | 0.72 ± 0.09 |
| recalibrated, test probe | 0.10 | 0.104 ± 0.010 | 0.095 ± 0.005 | 0.216 ± 0.044 | 0.091 ± 0.020 |
| recalibrated, test probe | 0.25 | 0.71 ± 0.07 | 0.63 ± 0.01 | 1.09 ± 0.11 | 0.61 ± 0.09 |

Number of seeds, out of 5, in which the method's S̄ is below plain's:

| policy, split | method | ε = 0.025 | 0.05 | 0.10 | 0.20 | 0.25 |
|---|---|---|---|---|---|---|
| saved, train probe | resolution / Gaussian / combined | 5 / 3 / 5 | 5 / 3 / 5 | 5 / 2 / 5 | 5 / 5 / 5 | 5 / 5 / 5 |
| saved, test probe | resolution / Gaussian / combined | 5 / 4 / 5 | 5 / 3 / 5 | 5 / 2 / 5 | 5 / 4 / 5 | 5 / 5 / 5 |
| recalibrated, train probe | resolution / Gaussian / combined | 5 / 1 / 5 | 5 / 0 / 5 | 5 / 0 / 5 | 5 / 0 / 5 | 5 / 0 / 5 |
| recalibrated, test probe | resolution / Gaussian / combined | 3 / 0 / 4 | 5 / 0 / 4 | 5 / 0 / 4 | 4 / 0 / 4 | 5 / 0 / 4 |

**Observations.**

1. **Scale of the policy effect.** Saved statistics give S̄ values 3–7 times
   larger than recalibrated statistics. Under saved statistics, weight
   perturbations also mis-scale the activations that BatchNorm then normalizes
   with stale statistics.
2. **Resolution-only and combined, saved statistics.** They are less sensitive
   than plain in every seed. The difference is large: −36% and −42% at
   ε = 0.10, −28% and −33% at ε = 0.25.
3. **Resolution-only and combined, recalibrated statistics.**
   - On the training probe the advantage stays consistent (−32% and −35% at
     ε = 0.10, −25% and −30% at ε = 0.25).
   - On the test probe it shrinks to −9% to −15% at ε = 0.10–0.25 and is not
     present in every seed. Combined reverses in seed 3; resolution-only is at
     0.001 in seed 0 at ε = 0.25.
4. **Gaussian-only's ranking depends on the policy.** It ties plain at small ε
   and is less sensitive at large ε under saved statistics, but is clearly
   *more* sensitive under recalibration: +43% at ε = 0.10 on the training
   probe and +108% on the test probe, in 5/5 seeds.
5. **Spread across directions.** Direction-to-direction SD within one network
   (`S3`) is 20–30% of the mean. That is larger than the seed-to-seed SD of the
   direction averages, so single-direction plots are unreliable; this is why
   averages are used.

**Interpretation.**

- **What the evidence supports.** "Smaller loss increases under relative
  filter perturbations" is a robust property of resolution-only and combined
  when the stored network is perturbed. Once BatchNorm is allowed to adapt, it
  remains robust on the training probe and becomes weak on test images.
- **Gaussian-only.** It does not share the property under recalibration, yet
  it has almost the same accuracy gain as resolution-only (+4.5 points). Within
  this study, the accuracy gain therefore does not track recalibrated
  perturbation sensitivity.
- **The link to accuracy is not established.** Lower sensitivity and higher
  accuracy co-occur for two of the three methods, and neither is shown to cause
  the other.

### Validation of probe conclusions

The prespecified validation points (directions 0 and 1, ε = 0.10 and 0.25, both
signs, all methods and seeds, both policies) were re-evaluated on the
10,000-image training subset and the full test set. Figure `V1`; table
`validation_probe_vs_large.csv`.

- **Agreement with the probes.** The larger-set values lie on the diagonal of
  the probe values. The paired difference against plain has the same sign in
  ≥ 4/5 seeds for every method, policy and amplitude.
- **Resolution-only, recalibrated, full test set.** Mean differences are
  −0.006 at ε = 0.10 (2/5 seeds below plain) and −0.058 at ε = 0.25 (3/5).
  The weak test-set effect is therefore confirmed as weak, not strengthened.
- **Gaussian-only, recalibrated.** Larger than plain on both larger sets in
  5/5 seeds (+0.13 to +0.43).
- **Calibration size.** Seed 0, 32 comparisons: switching calibration from
  2,000 to 10,000 images changes S by at most 0.005 and the centre loss by at
  most 0.002.

These checks cover the prespecified points only, not every plotted point.

### Surfaces

Figures: `A1_*`, `A2_*` (3D and contour), all seeds at 21×21 and the primary
seed 0 at 41×41 (prespecified), plus `surfaces_interactive.html`.

- **Shape.** All 24 surfaces are bowl-shaped around their centre in the two
  displayed directions.
- **Centre-subtracted rise.** It is visibly smaller for resolution-only and
  combined than for plain and Gaussian-only. This agrees with the
  recalibrated-policy training-probe numbers above.
- **Cross-check.** 480 surface points coincide with the separately computed
  1D measurements (maximum difference 0.0 CE).

These are two-direction local slices with matched relative perturbations. A
bowl-shaped slice does not establish a convex neighbourhood.

### Straight segments

Setup: 51 points, final state, recalibrated BatchNorm, the same seed at both
ends. Figure `D1`; table `interpolation_barriers.csv`.

Sampled barrier above the higher endpoint, range over the five seeds:

| segment | train probe | test probe |
|---|---|---|
| plain → resolution-only | 0.81–0.92 | 0.46–0.59 |
| plain → Gaussian-only | 1.23–1.56 | 0.81–1.10 |
| plain → combined | 0.91–1.22 | 0.53–0.84 |
| resolution-only → combined | 0.60–0.99 | 0.44–0.81 |

**Observation.** Every straight segment crosses a region of higher loss. The
plain → Gaussian-only barrier is the largest in every seed, and the barrier
between the two resolution methods is not smaller than between plain and
resolution-only.

**Interpretation.** On straight lines, the four solutions of a seed are
separated from one another. This does not establish disconnected basins: a
curved low-loss path was not searched for.

## Q1: at fixed weights, how does changing the intervention change the loss?

Setup:

- Checkpoints are taken exactly at the first and last transition of each
  schedule (selected by rule in advance). Each holds the weights produced by
  the last update of the old state.
- Weights, directions and data are identical across the states compared.
- BatchNorm is recalibrated per state; saved-statistics centre losses are also
  recorded.
- 10 directions per state.

Figures: `F1`, `B1_*`, `B2_*` (surfaces for seed 0). Table:
`fixed_weight_states.csv`. Test-probe values below are means over five seeds.

| method, checkpoint | state | centre CE, recalibrated | centre CE, saved | S̄(0.10) | S̄(0.25) |
|---|---|---|---|---|---|
| resolution, update 2346 (trained at r = 16) | r = 16 (before) | 0.924 | 0.944 | 0.052 | 0.316 |
| | r = 24 (after) | 1.064 | 1.141 | 0.055 | 0.332 |
| | r = 32 (final) | 1.639 | 4.081 | 0.059 | 0.319 |
| resolution, update 4692 (trained at r = 24) | r = 24 (before) | 0.692 | 0.704 | 0.062 | 0.400 |
| | r = 32 (after = final) | 1.133 | 3.579 | 0.102 | 0.554 |
| | r = 16 | 0.917 | 1.020 | 0.064 | 0.400 |
| Gaussian, update 1173 (trained at G = 1.0) | G = 1.0 (before) | 1.126 | 1.169 | 0.126 | 0.605 |
| | G = 0.85 (after) | 1.125 | 1.220 | 0.116 | 0.561 |
| | G = 0 (final) | 1.868 | 5.057 | 0.053 | 0.231 |
| Gaussian, update 8211 | G = 0.30 (before) | 0.580 | 0.586 | 0.208 | 1.020 |
| | G = 0 (after = final) | 0.584 | 0.592 | 0.207 | 1.014 |
| combined, update 1173 (trained at r = 16, G = 1.0) | before | 1.075 | 1.092 | 0.076 | 0.416 |
| | after (r = 16, G = 0.85) | 1.144 | 1.403 | 0.059 | 0.328 |
| | final (r = 32, G = 0) | 1.840 | 4.104 | 0.034 | 0.172 |
| combined, update 8211 | r = 32, G = 0.30 (before) | 0.543 | 0.546 | 0.081 | 0.545 |
| | r = 32, G = 0 (after = final) | 0.545 | 0.549 | 0.081 | 0.539 |

**Observations.**

1. **Early checkpoints.** Switching to the final state raises the centre loss
   by 0.4–0.7 nats with recalibration, and by about 3 nats with saved
   statistics.
   - The saved-statistics jump is mostly a BatchNorm statistics mismatch.
   - Recalibration removes most of it, but a gap of 0.4–0.7 nats remains.
2. **Late switch-off (G 0.30 → 0).** Centre loss and sensitivity are unchanged
   to within 0.005 nats for both Gaussian methods. By then the weights no
   longer depend on the filter.
3. **Sensitivity around the early centres.** It changes with the state in a
   method-specific way:
   - Gaussian and combined: S̄ around the final-state centre is 2–3 times
     *lower* than around the trained-state centre, at a much higher centre
     loss.
   - Resolution-only at update 4692: S̄ is *higher* in the final state (0.55
     against 0.40).
   - Resolution-only at update 2346: essentially flat across states.
4. **Each set of weights has its lowest centre loss under the state it was
   last trained in.**

**Interpretation.** At fixed weights, the intervention mainly shifts the loss
level, together with a normalization-statistics effect. Its effect on local
sensitivity depends on the method and the checkpoint. A lower S̄ around a
high-loss, non-trained-state centre is not evidence that the intervention
smooths the loss in parameter space. Changing the combined state changes both
components at once, so it isolates neither.

These are local sensitivities and centre ranks. They do not establish equal
curvature or local minima.

## Q2: how do the observed training trajectories differ?

Setup:

- one PCA basis per seed, fit on the matched epoch checkpoints 0–30 of the four
  runs;
- transition-window checkpoints projected afterwards;
- all learned parameters included (BatchNorm scale/shift included, running
  statistics excluded), no filter normalization.

Figures: `T1` (trajectories), `T2` (actual checkpoint loss), `T3` (projection
error), all seeds; `T4` (plane contour) and `T5` (3D plane), seed 0.

**Projection quality: a real limitation.**

- **Two components are not enough.** PC1 + PC2 explain only 57–59% of the fit
  variance in every seed. The first four components explain 93%, and ten
  explain 98.6%.
- **Distortion.** For some early resolution-only checkpoints (epochs 1–2),
  99% of their squared distance to the mean lies outside the 2D plane.
- **Ten components suffice.** With ten components, the median unexplained
  fraction is 2% and the maximum 10%.

The 2D view therefore distorts the early phase, and roughly four comparable
directions are needed.

**Observations, restricted to saved checkpoints.**

1. **Separation.** From the first saved epoch, the four runs of a seed project
   onto four distinct directions from the shared initialization. Projected
   positions move little after about epoch 21, where the checkpoints bunch
   together in `T1`. The same four-branch layout appears in all five seeds.
2. **Continuation runs, final-state loss (T2).** During the active phases, the
   final-state CE of the continuation runs is well above their current-state CE.
   - With saved statistics it jumps by several nats at checkpoints taken inside
     reduced or filtered phases.
   - After recalibration the gap is 0.4–0.7 nats.
   - The gap closes at the last transitions (update 4692 for resolution-only,
     8211 for Gaussian and combined).
3. **Plain and the plane background.** Plain's final-state loss decreases
   monotonically. On the PCA plane of seed 0, the reconstructed in-plane points
   show four low-loss regions, one around each run's late projections,
   separated by higher regions.

**Interpretation.** Within the observed checkpoints, the methods move along
different directions from the start and end in separate regions on straight
lines. The contour describes reconstructed in-plane points only, not the
off-plane checkpoints. Given the 57–59% two-component variance, the 2D picture
is a partial view.

## BatchNorm conventions and projection limitations, in short

- **Saved statistics** answer "what does the stored network do when its
  weights are perturbed?". Mismatched statistics inflate losses, especially
  away from the trained state.
- **Recalibration** answers "what does it do after normalization adapts?".
  Conclusions that hold under both policies (resolution-only and combined, most
  settings) are the robust ones. Conclusions that flip (Gaussian-only) must be
  stated per policy.
- **PCA views** show five checkpoint phases in a two-dimensional shadow of a
  space where about four directions matter. Lines join saved checkpoints; they
  are not observed paths. Surface heights are losses of in-plane
  reconstructions.

## Integrity, compute and incompleteness

- **Training.** 20/20 runs completed. Every checkpoint expected by the schedule
  was present and loadable, with update counts and states matching the
  schedule, on every account (Kaggle-side checks):
  - plain: 31 checkpoints;
  - resolution-only: 43;
  - Gaussian-only and combined: 73 each.

  In every seed, the four initialization checkpoints were identical to each
  other and to the asset, and the configs differed only in method and run name.
  Saved-statistics evaluation reproduced each run's own final metrics exactly
  (difference 0.0).
- **Checks.** All passed in every seed:
  - exact bypass of each method's final state against hook-free ResNet-20
    (inference and training mode);
  - direction normalization error ≤ 9e-16, with no zero-norm blocks;
  - bitwise reconstruction and interpolation endpoints;
  - repeatable, order-independent evaluation under both policies;
  - no mutation of files or tensors.
- **Downloads.** Every file listed in a job's manifests was compared by sha256:
  - `maxnicaise` (seed 0): 246 run files and 125 evaluation files, all present
    and matching;
  - `maxnikezz` (seed 2) and `maximemonstrenikez` (seed 3): 246 + 75 each, all
    present and matching;
  - `maxlefrr` (seeds 1 and 4): 491 run files and 149 evaluation files, all
    present and matching. The first full download broke on a network error
    (`IncompleteRead`) despite a zero exit code, leaving the combined runs of
    seeds 1 and 4 incomplete and both manifests absent. The missing files were
    fetched by pattern and verified afterwards: every hash matches, including
    the file that was being written when the connection broke.
- **Evaluation tasks.** 155/155 completed, with no deadline stops, no failures
  and no non-finite values.
- **Compute (Kaggle, 2 × T4 per account).**

| account | seeds | training | evaluation | total |
|---|---|---|---|---|
| `maxnicaise` | 0 | 21.4 min | 44.0 min | 65.6 min |
| `maxnikezz` | 2 | 21.5 min | 13.0 min | 34.7 min |
| `maximemonstrenikez` | 3 | 20.8 min | 11.6 min | 32.7 min |
| `maxlefrr` | 1, 4 | 39.8 min | 22.8 min | 63.0 min |

  Per-run training wall time on one T4, including per-epoch evaluation and
  checkpoint writes (`tables/training_runs.csv`): plain about 8.3 min,
  resolution-only 8.1, Gaussian-only 10.5, combined 12.4. Total about 3.5
  GPU-hours of training and 1.5 hours of evaluation across the four jobs.
- **Deviations from the plan.**
  - The seed 1/4 run-file download needed a retry.
  - Fixed-weight measurements used 10 directions (preregistered), not 20.
  - No other deviation.
- **Not done, by design.** Optimized connecting paths, Hessian analysis,
  method tuning. **Convexification remains unresolved.**
