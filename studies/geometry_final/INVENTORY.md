# Reuse / missing-work inventory (written before any new computation)

Audited 2026-09-17 after `git fetch --all`. Branch tips: `visualization` = `9e9275c` (no newer
commit; raw outputs collected afterwards are uncommitted in the `visualization` worktree),
`continuation-core-experiments` = `1a60649` (four commits after the `4179584` snapshot the
paper cites). No other branch contains landscape or curvature results.

## 1. Matching criteria

A result is reused only if checkpoint (sha256), data indices (`subsets.npz` sha256), parameter mask,
BatchNorm buffers and calibration protocol, coordinates, directions (draw files by sha256, scaled to
the same checkpoint) and precision all match. Matching seed labels alone do not qualify.

## 2. landscape_v3 (`visualization@9e9275c` + collected raw outputs): the primary cohort

Verified from code (`landscape_v3/{evaluator,hessian,run}.py`, unchanged since `9e9275c`) and from the
raw records, not from filenames:

| item | executed objective / convention (verified) | status for this task |
|---|---|---|
| 20 final checkpoints | v2 runs, `epoch_030.pt`, sha256 in `expected_digests.json`; native final state (r = 32, no filter) | **reused** (already staged in `maxmonstre/landscape-v3-inputs`) |
| centre-frozen BN | `Evaluator.centre_stats`: reset, momentum None, `calib2k` in file order, batches of 500, train mode, no_grad, at unperturbed weights; buffers copied into every evaluation / HVP | primary policy; reused |
| probes | `train_probe_idx`, `test_probe_idx` (1,000 each), batches of 500 | reused |
| mask | `in_mask`: weight tensors with dim 2 or 4 → 19 conv + fc = 268,336 in 698 blocks | reused |
| HVP | double back-prop of batch-weighted mean CE, `model.eval()`, float32 | reused unchanged |
| block A sensitivity | 20 dirs × 10 ε × ± × 3 policies × 2 probes, 24,000 direction values | **reused** (no new sweep) |
| 160 Hessian problems | top1, top2, min + vectors, ‖∇L‖, residuals, 2 starts, f32/f64 RQ, float64 FD along top1/min | **reused** (no new eigenproblem) |
| quadratic forms | dᵀHd for 20 random directions, saved + centre-frozen, both probes | **reused** (Monte Carlo reference for tr(HC)) |
| quadratic vs finite differences | C(ε)=2S/ε² ÷ mean dᵀHd, 10 ε | **reused** |
| eigendirection cuts | top1/top2/min, frozen policies; top1/min also pointwise (train, centre-frozen, relative) | **reused** |
| seed-0 Hessian planes | (top1, top2), (top1, min), saved and centre-frozen, 41×41 unit grid | **reused** |
| random 41×41 seed-0 grid | **pointwise only** (1,240 v3 vertices + 441 reused v2 vertices per model) | reused for the pointwise panels |
| random grid, centre-frozen | **absent**, except 29 vertices per model that coincide with centre-frozen block-A points (centre and axis points at ±{0.025, 0.05, 0.1, 0.2, 0.25, 0.35, 0.5}); bitwise-identical weights, verified on the pointwise analogue (max ΔCE = 0 over 32 comparable pairs) | **missing: 1,652 vertices × 4 models** |
| traces | **absent** ("no trace" in the v3 report; no trace file in any raw directory) | **missing: 40 objectives** |

## 3. Idriss (`continuation-core-experiments`)

| | endpoint grid (`4179584`) | BN-policy 2×2 (`59dc69c`, results in `1a60649`) |
|---|---|---|
| checkpoints | his own 28-cell grid; `__reference__` cells, seeds 0–2, Kaggle dataset `cc-grid-final` (binaries not committed; no sha256 available) | same 12 reference checkpoints |
| relation to our cohort | **different training executions**: seed labels 0–2 coincide, weights do not | same |
| parameters | all 269,722 learnable parameters (conv, fc weight **and bias**, BN γ and β) | same |
| data | 5,000 images of the pinned training subset | first 2,000 images of the pinned training subset or of the test set, batches of 500 |
| BN | `eval()`: saved running statistics | `running_stats` (saved) **or `fixed_batch_stats` = `model.train()`: each 500-image batch normalised by its own statistics, differentiated through** |
| estimator | Hutchinson, Rademacher, 64 draws, deflated by 5 power-iteration eigenpairs (unbiased control variate) | 24 draws, deflated by 1 eigenpair (25 power iterations) |
| coordinates | ordinary, all parameters | ordinary |

New evidence since `4179584` (read in full): `docs/CROSS_STUDY.md`, `docs/RESULTS_GRID.md` §1 and §3,
`tools/job_bnpolicy.py`, `tools/analyze_bnpolicy.py`, four shard outputs (48 measurements,
2 × Tesla T4 environments), `curvature.bn_mode`. It measures tr H and λ_max for 4 methods × 3 seeds
× {saved, train-mode batch statistics} × {train, test}: all 36 method-vs-control contrasts negative.

**Compatibility verdict: not reusable for the primary cohort, and not pointwise recalibration.**
The column labelled "recalibrated" in `CROSS_STUDY.md` is the train-mode minibatch Hessian (a third
object: batch-dependent, differentiated through the batch statistics), not the pointwise-recalibrated
function of landscape_v3 (running buffers re-estimated on 2,000 calibration images, then inference).
Its checkpoints, parameter set, image sets and batch composition also differ. It is kept as an
independent, identified replication of the *direction* of the trace contrast on other checkpoints
and conventions, and its input-derivative contribution is unaffected.

## 4. Work scheduled (and nothing else)

1. Traces for the 40 centre-frozen objectives (64 draws, predeclared extension to 128/256 per
   seed × probe quartet), reusing the validated HVP.
2. The 1,652 missing centre-frozen vertices per seed-0 model; 29 reused vertices re-evaluated only as
   a reproduction check.

Not scheduled: eigenproblems, sensitivity sweeps, verification campaigns, pointwise grids, multi-seed
surfaces, spectral densities, pointwise-recalibrated Hessians, intermediate checkpoints, paths,
input-Jacobian grids, training, CBS/SDPoint.
