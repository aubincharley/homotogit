# geometry_final: protocol (frozen before the main job)

Written 2026-09-17, after the audit and the T4 timing pilot, before the main job. No training.
No new eigenproblem, spectral density, pointwise-recalibrated Hessian, intermediate-checkpoint
Hessian, optimized path or input-Jacobian grid. Existing results are reused, not repeated.

## 1. Cohort and objective (unchanged from landscape_v3)

- 20 final checkpoints (`epoch_030.pt`) of the landscape_v2 runs: plain, `resolution_max_b1`,
  `gaussian_postrelu`, `resolution_max_b1_gaussian_conv`; training seeds 0–4; native filter-free
  final state. sha256 in `landscape_v3/protocol/expected_digests.json`, rechecked on Kaggle.
- **Primary BN policy: centre-frozen.** Recalibrate once at the unperturbed checkpoint: reset,
  cumulative average (momentum None), `calib2k` (2,000 training images, fixed order), 4 batches of
  500, training mode, no_grad. Buffers then held fixed for every loss evaluation and every HVP around
  that checkpoint. Evaluation in inference mode.
- Probes: 1,000-image class-balanced training and test probes (`subsets.npz`).
- Variables: 268,336 conv + fc weights in 698 filter/row blocks (v2 mask). Biases, BN affine
  parameters and buffers fixed. Mean CE, no weight decay. float32, TF32 off, cuDNN deterministic.
- HVP: exact double back-propagation of this objective (`landscape_v3.hessian.FrozenBNObjective`,
  validated in landscape_v3: dense reference 5e-16, repeat bitwise, symmetry ≤ 5.7e-6,
  f32/f64 ≤ 4.1e-4). Not the Hessian of the pointwise-recalibrated loss; not a train-mode minibatch
  Hessian.

## 2. Traces (new; none existed for this cohort)

40 objectives: 4 methods × 5 seeds × {train, test} probe, centre-frozen, ordinary coordinates.
One HVP per Rademacher draw z; per block q_g = z_gᵀ(Hz)_g; per draw

| quantity | aggregate | estimates |
|---|---|---|
| ordinary signed trace | Σ_g q_g | tr(H) |
| block-relative signed trace | Σ_g ‖θ_g‖² q_g | tr(AᵀHA), A_g = ‖θ_g‖ I |
| direction-covariance trace | Σ_g (‖θ_g‖²/p_g) q_g | tr(HC) = E[dᵀHd] for the filter-normalised directions |

- **Draws.** z = ±1 from raw PCG64 bits, seed `derive_seed(training_seed, "geometry_final::trace::rademacher::draw::<j>")`:
  shared by the four methods and both probes of a training seed, independent of every other stream,
  regenerable bitwise on any machine (sha256 of each z recorded).
- **Precision rule (predeclared).** Start with 64 draws per objective. For each seed × probe quartet,
  if any of its 12 traces (4 methods × 3 quantities) has Monte Carlo SEM > 5% of |estimate|,
  extend the whole quartet to 128 draws, then (if still failing) to 256, keeping earlier draws.
  256 is the cap: anything still failing is reported as unresolved at that precision. Sampling is
  never continued for any other reason.
- **Raw estimator is primary.** Per-block q_g of every draw are stored, so paired differences
  (same z across methods), their Monte Carlo SEM, and an optional control-variate estimate using
  the existing top1/top2 ordinary centre-frozen eigenvectors (`zᵀWHz − λ(zᵀWu)(uᵀz) + λ uᵀWu`,
  unbiased for any fixed u) are computed offline and reported separately.
- **Checks.** New estimator validated on a dense conv+BN+ReLU+max-pool reference
  (`checks/dense_reference.json`: block formula = dense quadratic form to 9e-14; 20,000-draw means
  within 0.7 SEM of the exact traces; buffers bitwise unchanged). On Kaggle, per objective: centre
  loss and gradient norm must reproduce the existing centre-frozen Hessian record; buffers equal the
  centre statistics, and buffers and masked weights hash identically before and after all draws.
- Monte Carlo SEM (within checkpoint) and seed-to-seed SD are reported separately. Signed traces
  are not interpreted as positive spectral mass.

## 3. Seed-0 centre-frozen random grid (new vertices only)

- Same checkpoints, same direction pair (v2 draws 0/1, filter-normalised to each model's own
  filter norms), same 41×41 grid over [−0.5, 0.5]², both probes; only the BN policy changes
  relative to the existing pointwise grid.
- 29 vertices per model (centre and the axis points at ±{0.025, 0.05, 0.1, 0.2, 0.25, 0.35, 0.5})
  already exist as centre-frozen block-A points with bitwise-identical perturbed weights (verified:
  pointwise grid axis vertices equal block-A pointwise values, max |ΔCE| = 0). They are reused and
  re-evaluated only as a reproduction check (tolerance 1e-5 CE; on failure every vertex is
  evaluated). 1,652 vertices per model are computed.
- The job also checks that the centre-frozen centre equals the pointwise centre bitwise.

## 4. Timing pilot (T4) and projection

Kaggle account `maxmonstre` only, kernel `geometry-final-pilot`, 2 × Tesla T4, torch 2.10.0+cu128:
16 HVP draws (2 objectives) and 120 centre-frozen grid points, 11/11 manifest files verified,
identity checks exact (relative difference 0.0), buffers and weights unchanged.

| operation | measured (median) |
|---|---|
| one trace draw (HVP, 1,000-image probe, f32, + block aggregation) | 0.97 s |
| one centre-frozen grid vertex (both probes) | 0.094 s |

Projection: stage 64 = 40 × 64 × 0.97 s ≈ 2,480 GPU-s; grid 6,608 × 0.094 s ≈ 620 GPU-s;
checks < 60 s → ≈ 0.87 T4 GPU-hours (≈ 30 min wall on 2 GPUs). Worst case (every quartet
extended to 256): + 7,450 GPU-s → ≈ 2.9 GPU-hours (≈ 1.6 h wall). Deadline guard 11 h; results are
flushed per draw / per vertex and resumable.

## 5. Existing measurements used without new computation

Sensitivity (block A, three policies), quadratic forms along the 20 random directions, quadratic vs
finite differences, 160 Hessian problems with verification, eigendirection cuts (frozen and
pointwise), seed-0 Hessian planes, pointwise random grids (v3 raw + v2 reused vertices), stored
eigenvectors (for radial/tangential decomposition and control variates).
