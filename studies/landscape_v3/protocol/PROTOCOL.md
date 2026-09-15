# landscape_v3 extension protocol

**Status.** This file was written before the main v3 evaluations, after:

- reading the v2 implementation, preregistration, manifests, raw results and
  checkpoints;
- running the local numerical pilot (`../pilot/`).

It is a prospective extension *informed by v2*, not a claim that the v2 results
were unseen. Where a Kaggle timing pilot changes a resource estimate, the change
is recorded in §8 before the main jobs start. Settings that were not changed
are frozen.

**Scope.**

- No training. Only the 20 v2 networks (4 frozen configurations × seeds 0–4)
  and their saved checkpoints are used.
- The v2 raw data and figures are preserved untouched.
- Outputs go to `studies/landscape_v3/`.
- Code lives in `landscape_v3/`.

## 1. Inputs and verification (done: `input_verification.json`)

**Checkpoints.** 150 needed checkpoint files, all verified:

- *Selection.* The checkpoint epochs are the temporal set ∪ every transition
  epoch.
- *Integrity.* Each file's sha256 equals the v2 `runs_manifest.json` entry.
- *Schema and identity.* Schema, update count (= 391 × epoch), method and
  seed are correct.
- *States.* The "used for last update" and "next update" states equal both
  the schedule and the epoch record in `metrics.json`.
- *Parameter order.* It equals `named_parameters()` of ResNet-20.

Result: 0 problems.

**Pairing.** Taken from the v2 Kaggle-side checks (`checks_seed*.json`):

- the 4 initial states of a seed are identical to each other and to the asset;
- the configs differ only in method and run name.

**Directions.**

- The 100 v2 draw files (seeds 0–4 × 20) match the v2 eval manifest.
- Local torch 2.5.1 does **not** regenerate the Kaggle torch 2.10 draws
  bitwise. The draw *files* are therefore shipped, and their sha256 is
  rechecked on Kaggle.

**Subsets.**

- `subsets.npz` matches the v2 input manifest.
- Per-array sha256 values are listed in `expected_digests.json`: train/test
  probes, calib2k, calib10k, train_large.
- `test_full` is the 10,000 CIFAR-10 test images in file order.

**Operators, placements and schedules.**

- These come unchanged from `continuation_core` presets.
- Gaussian placement differs between Gaussian-only (post-ReLU, 10 sites) and
  combined (conv outputs, 19 sites, σ = (r/32)·G). The methods are not a
  factorial decomposition.

## 2. Measurements

**Loss.**

- Mean cross-entropy (CE) without weight decay; accuracy is recorded
  separately.
- Evaluation is in inference mode, with batches of 500, cuDNN deterministic
  and TF32 disabled.
  - The local pilot showed that TF32 convolutions on an Ampere GPU shift the
    CE by up to 4e-4.
  - T4 GPUs have no TF32; the flag is set anyway.

**Perturbations.** Symmetric sensitivity

S_{m,s,k}(ε) = (L(θ+εd) + L(θ−εd))/2 − L(θ),

S̄_{m,s}(ε) = mean over k, and Δ_{m,s} = S̄_{m,s} − S̄_{plain,s}.

- Every direction, sign and centre is stored. Nothing is clamped.
- Directions are the v2 Li et al. filter-wise draws.
  - Conv output filters and fc rows are rescaled to the reference
    checkpoint's block norms.
  - Biases and BN affine parameters get zero.
  - Draws are shared by the 4 methods of a seed.
- Perturbed weights are θ (float64) + ε d, cast to float32; this is the v2
  code path. In float64 evaluations they stay float64.

**BatchNorm policies.**

| name in data | label in figures | definition |
|---|---|---|
| `saved` | saved/frozen BN | checkpoint running statistics reloaded before every evaluation |
| `recalibrated` | recalibrated at every point | v2 procedure per evaluated weight vector and state: reset; cumulative average over the fixed 2,000-image training calibration set (fixed order, batches of 500, training mode, no_grad); then inference |
| `centre_frozen` | recalibrated at centre, then frozen | the same calibration run once at the unperturbed reference weights and state; those statistics are then held fixed for every perturbed point and for the Hessian |

- Calibration never uses test images.
- Every evaluation reloads parameters and buffers, so nothing leaks between
  points.
- Pilot checks:
  - `centre_frozen` at the centre is bitwise identical to `recalibrated` at
    the centre;
  - v2 points reproduce within 6e-8 CE on another GPU.

**Reuse.** A v2 measurement is reused only if its canonical key matches:

- method, seed, checkpoint, intervention state;
- policy, calibration set, precision (float32);
- evaluation set, perturbation.

Before trusting reused points, each Kaggle job re-evaluates a sample of them
(`checks/reuse_reproduction.json`).

- **Tolerance:** 1e-5 CE.
- **On failure:** the job re-evaluates every reused point of that account.

## 3. Evaluation matrix

Counts are points: one point = one evaluation of both probes, or of both
larger sets. Full details are in `evaluation_matrix.json`.

| block | question | specification | required | reused (v2) | new |
|---|---|---|---|---|---|
| A | Q1 | epoch 30, final state; directions 0–19; ε ∈ {0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.35, 0.5}; ± ; 3 policies; train+test probes; centres | 24,060 | 8,040 | 16,020 |
| A-f64 | Q1 precision | the same, float64, directions 0–1, ε ∈ {0.005, 0.01, 0.02, 0.05}, 3 policies | 1,020 | 0 | 1,020 |
| B | Q2 | epoch 30; directions 0–19; ε ∈ {0.1, 0.25}; ± ; saved + pointwise; train_large (10k) + test_full (10k); centres on the same sets | 3,240 | 360 | 2,880 |
| C-temporal | Q3 | epochs 3, 6, 9, 12, 18, 21, 30; directions 0–9, normalised at each checkpoint; ε ∈ {0.1, 0.25}; ± ; saved + pointwise; probes. Continuation methods: produced state + final state (one when equal); plain: its own state | 18,040 | 4,440 | 13,600 |
| C-transitions | Q3 | every schedule transition (resolution-only 6, 12; Gaussian-only and combined 3, 6, 9, 12, 15, 18, 21); states before/after/final (resolution-only r = 16/24/32); directions 0–9; pointwise ε ∈ {0.025, 0.05, 0.1, 0.2, 0.25} (the v2 fixed-weight protocol) + saved ε ∈ {0.1, 0.25}; centres under both | 32,660 | 8,360 | 24,300 |
| E-g21 | figures | random pair 0/1 over [−0.5, 0.5]², 21×21, pointwise, all 20 networks | 8,820 | 2,420 | 6,400 |
| E-g41 | figures | the same, 41×41, seed 0 | 6,724 | 1,764 | 4,960 |

**Hessian (Q4).** 20 networks × 2 probes (train, test) × 2 BN policies (saved,
centre_frozen) × 2 coordinates (ordinary H, relative H_rel) = 160 eigenproblems.

- **Starts.** Each problem runs 2 deterministic Lanczos starts. The start
  seed is derived from the problem id and the start index.
- **Per network, additionally:**
  - `hchecks`: HVP repeatability, symmetry, float32 vs float64, and central
    finite differences of float64 gradients (all masked parameters, and fc
    only as a smooth control);
  - `quadform`: d_kᵀ H d_k for the 20 random directions, both probes, both
    frozen policies, plus g·d_k;
  - `hcut`: 1D cuts along the eigendirections (§6).
- **Seed 0 only:** `hplane`, two 41×41 planes per frozen policy (§6).

**States.**

- Produced state = `used_for_last_update` of the checkpoint (the state that
  produced the weights).
- Next-update state = `next_update`.
- At a boundary checkpoint the two differ. The temporal overview uses the
  produced state, and the transition block compares before (= produced),
  after (= next) and final at identical weights.
- Combined at epochs 6 and 12 changes r and G together. This is labelled a
  **joint change**.

## 4. Numerical tolerances (`numerical_protocol.json`)

**Evaluations.**

- **Precision check.** Block A-f64 re-evaluates small amplitudes in float64
  under all 3 policies.
- **Suspicious value** (reported, not removed):
  - |S_f32 − S_f64| > 1e-6 + 1e-3·|S_f64|;
  - or a non-finite value.
- Negative S values are kept and counted.

**Hessian–vector products.**

- Exact double back-propagation of the mean CE over the probe, in batches of
  500, with inference-mode BN (saved or centre-frozen statistics).
- Variables are the conv weights + fc weight: P = 268,336 in 698 blocks (the
  v2 mask). Every other parameter and buffer is constant.
- It is not the Fisher and not the Gauss–Newton matrix.
- Pilot results:
  - on a small conv+BN+ReLU+max-pool reference, the HVP matches the dense
    autograd Hessian to 5e-16;
  - on ResNet-20, repeats are bitwise identical, uᵀHv vs vᵀHu agrees to
    ≤ 3.4e-6 (relative), and float32 vs float64 HVPs agree to ≤ 8e-5.

**Lanczos.**

- Full reorthogonalisation (two Gram–Schmidt passes), float64 basis.
- The same probe, batches and deterministic operator at every iteration.
- Convergence is checked every 10 iterations after 20.
- A Ritz pair (θ, y) is declared converged if the **explicit** residual
  satisfies ‖Hy − θy‖ ≤ 1e-3 · max(|θ|, 1e-3·|θ_top1|). This scale floor
  handles values near zero.
- The budget is 400 iterations. Pilot: plain and resolution-only on the
  train probe converged in 70 iterations.
- Reported per problem: iteration count, Ritz-value and residual-estimate
  history, explicit residuals, basis orthogonality error, and the agreement
  between the 2 starts.
- An unconverged extreme Ritz value is labelled *unconverged* and never
  reported as an eigenvalue.

**Verification of the extreme vectors.** For top1 and min (start 0):

- Rayleigh quotients with fresh float32 and float64 HVPs;
- float64 finite differences C(t) = 2S(t)/t² along the direction scaled to
  r = 1, at t ∈ {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2}, under the same
  frozen-BN objective;
- the gradient component g·δ;
- the largest per-block relative displacement at t = 1, and the number of
  blocks holding 90% of r².

*Revision from the local dry run, before any main job.* The first step grid
(0.001–0.05) was replaced. The r-normalised eigendirections concentrate in few
blocks: along plain's top direction, CE rose by more than 100 nats already at
t = 0.01, so those steps were far outside the quadratic regime.

**Float64 statistics.** Float64 evaluations and float64 HVPs load *the same*
statistics as float32:

- saved: the checkpoint statistics;
- centre_frozen: the float32 centre calibration, cast to float64.

Only the arithmetic changes. In the dry run, an independent float64
recalibration changed the min-direction Rayleigh quotient (−5.5 vs −26.9),
which mixed a calibration difference into a precision check. Pointwise
recalibration in float64 necessarily recalibrates in float64.

A negative quotient that is stable across float32/float64 is **evidence of
negative curvature of the a.e. Hessian at that point**. It is not a statement
that the point is a saddle, and failure to find one is not proof of positive
semidefiniteness. The gradient norm is reported; checkpoints are not called
stationary points.

**Nonsmoothness (found in the pilot).**

- ReLU and max-pool make the loss piecewise smooth.
- Central finite differences of float64 gradients *diverge* from the HVP as
  h → 0: relative error 0.11 at h = 1e-2 and 7 at h = 1e-6. This is the
  signature of gradient jumps at activation kinks, not roundoff.
- Likewise, 2S/ε² along random directions differs from dᵀHd by 10–40% for
  ε ∈ [0.001, 0.1], identically in float32 and float64.
- These discrepancies are documented, not forced to agree. The fc-only
  control (no kinks downstream) tests the HVP itself.

## 5. Relative coordinates

- A_g = ‖θ_g‖₂ I_g per conv output filter / fc row, frozen at the reference
  checkpoint (no differentiation through the norms).
- θ(q) = θ + Aq and H_rel = AᵀHA.
- Zero-norm blocks get A_g = 0; their coordinates are masked out of the
  Krylov space and counted (v2 found none).
- A relative eigenvector q maps to weight space as δ = Aq. It is **not**
  renormalised per filter.

**RMS relative block displacement.**

r(δ) = ( (1/G) Σ_g ‖δ_g‖² / ‖θ_g‖² )^{1/2}, over the G = 698 nonzero blocks.

- Under the v2 normalisation, r(εd) = |ε|.
- For a unit q, r(Aq) = ‖q‖/√G.
- All cuts use δ / r(δ): one scalar for the whole direction, so the axis t is
  directly comparable with ε. For such a direction, δᵀHδ = G·λ_rel.

**What the random directions measure.** With d = Au, where u_g is uniform on
the unit sphere of block g (n_g entries):

E[dᵀHd] = Σ_g tr(H_rel[g,g]) / n_g.

This is a block-size-weighted trace. It is neither tr(H_rel) nor an
eigenvalue, so the random-direction average is not equated with any single
Hessian summary.

**Coordinate dependence** (Dinh et al., 2017). Eigenvalues change under
reparametrisation. H and H_rel are two stated conventions, and neither is
claimed invariant. The spectral method follows Ghorbani et al. (2019);
landscape plots follow Li et al. (2018).

## 6. Figures and plotting choices

**General conventions.**

- Seed 0 is the representative display seed.
- Quantitative claims use all five seeds: 5 seed values, mean, descriptive
  SD (n − 1), and the count of seeds with each sign.
- Directions and images are not replications.

**1D sensitivity.**

- S̄(ε) on log–log axes, with seed dots and mean ± SD.
- Δ(ε) with seed dots, a zero line and a symmetric-log scale.
- C_d(ε) = 2S/ε² per policy, with the Hessian quadratic form
  mean_k d_kᵀHd_k drawn as a reference for the two frozen policies.
- Direction spread as per-seed boxes.

**Validation.**

- Paired Δ on the probe vs on the larger set, identical points, per seed.
- A sign-retention table.

**Temporal.**

- Separate panels for centre CE, S̄(0.1), S̄(0.25) and weight norms.
- x = epoch, with dotted lines at resolution transitions and grey ticks at
  Gaussian transitions.
- Solid = produced state, dashed = final state.
- Relative normalisation is stated in the caption; the planes are not one
  physical plane.

**Transitions.** Centre CE and S̄ for before/after/final, with seed dots and a
mean bar. Combined at 6 and 12 is labelled "joint r+G change".

**Hessian.**

- λ_top1, λ_top2, λ_min per method (seed dots), for policy × probe × coordinate.
- Convergence histories.
- Finite-difference verification plots.
- **Cuts** along top1 / top2 / min (relative) and top1 / min (ordinary).
  - Grid: t = 0 and ±{0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
    0.2, 0.35, 0.5}, in r units.
  - Absolute CE on shared axes per policy/probe, with a symmetric-log t axis.
  - This grid replaced a linear 41-point grid on [−0.5, 0.5] after the dry
    run, because that grid put almost every point in the exploding regime.
- **Seed-0 planes** (top1, top2) and (top1, min) of H_rel (train probe),
  under the matching frozen policy; 3D surface, contour and the measured 1D
  cuts.
  - Unit grid u ∈ [−1, 1], 41×41.
  - Physical t_i = u·T_i, with T_i = √(2·2 nats / (G·|λ_i|)).
  - λ_i is **plain's** eigenvalue for the same seed, policy, probe and
    coordinates, so the axes are shared by the four methods. On plain, the
    quadratic model rises by 2 nats at the axis end.
  - Each axis has its own extent; captions state T_1 and T_2.
- The mapped eigendirections are not orthogonal in weight space; captions say
  so.

**Random planes.**

- The existing v2 local view over [−0.25, 0.25]² is preserved.
- A new wide view over [−0.5, 0.5]² (21×21 for all seeds, 41×41 for seed 0),
  pointwise BN.
- Absolute and centre-subtracted CE come from the same grid.

**Common rules.**

- Shared axes, camera (elev 28, azim −58), z-limits and colour scales within a
  comparison.
- Absolute CE uses a sequential colour map (viridis). Centre-subtracted and
  paired differences use a diverging RdBu map centred at 0.
- Missing or non-finite vertices are drawn as grey × markers and counted in
  the caption.
- Only measured vertices are shown: no smoothing, no fitted surfaces.
- Concise titles; protocol details go in the captions (`FIGURES.md`).
- Every figure is exported as PDF and PNG; surfaces also get a rotatable HTML
  view.
- Every rendered PNG is inspected.

## 7. Interpolation and PCA

**Interpolation.**

- The v2 straight-segment results are re-summarised from raw data, with no new
  compute.
- The v2 statement that the resolution→combined barrier is "not smaller than"
  plain→resolution is false in seeds 1 and 4 on the train probe (0.603 vs
  0.869; 0.829 vs 0.844) and is corrected.

**PCA.**

- Kept descriptive: explained variance and reconstruction errors only.
- Actual checkpoint losses are not placed on the reconstructed plane.

## 8. Resources

**Final account layout (after the user verified the phone numbers and added
`maxmonstre`).**

| account | work |
|---|---|
| maxnicaise | seed 0 + Hessian planes |
| maxlefrr | seed 1 |
| maxnikezz | seed 2 |
| maximemonstrenikez | seed 3 |
| maxlebossdu91 | seed 4 |
| maxfrrsava | 21×21 wide random planes, all seeds |
| maxmonstre | seed-0 41×41 wide plane + float64 block |

**T4 pilot (`maxlebossdu91`, 2 × T4, torch 2.10+cu128).**

- Timings:
  - float32 HVP ≈ 1.0 s;
  - float64 HVP ≈ 4.1 s;
  - Lanczos ≈ 1.1 s per iteration;
  - hchecks 79 s (2 steps);
  - quadratic forms 89 s.
- All pilot tasks completed; the 18 manifest files verified.
- The fc-only gradient finite-difference error is 2.2e-4 at h = 1e-2 and
  2.2e-6 at h = 1e-3, second-order convergence, so the HVP is correct.
- The error over all masked weights is 0.08 → 0.25 as h decreases (kinks).

**Dry-run finding, fixed before launch.**

- *The bug.* The verification step reused the float64 model after the
  float64 finite differences along `top1` had left it perturbed. The `min`
  float64 Rayleigh quotient was therefore evaluated off-centre (−26.9).
- *The fix.* The objective is now rebuilt at the centre for each vector.
- *Direct check.* Float32 vs float64 quotients for the same vector are
  −5.48126 vs −5.48128 (`pilot/diag_min_direction.json`).
- *Observation.* In the same diagnostic, float64 second differences along
  the min direction are positive at every step (1e-4 to 3e-2 in coordinate
  units). Along top1 they are within 4–5% of the quotient. Kink contributions
  therefore dominate finite perturbations along the negative-curvature
  direction of the a.e. Hessian. This is reported as such.

**Earlier Kaggle pilot result, superseded by the phone verification.**

- The `maxfrrsava` pilot ran on **CPU**: `environment.json` shows torch
  2.10.0+cpu with 0 GPUs.
  - It confirms the float64 and grid code paths.
  - Its reuse reproduction matched v2 to 1.5e-7 CE on CPU.
  - It gives no T4 timing.
- The `maxlebossdu91` pilot stayed queued for over 40 minutes.
- Both new accounts evidently lack GPU access (unverified accounts), so they
  are excluded from GPU work.
- `plan matrix --exclude-accounts maxfrrsava maxlebossdu91` moves:
  - seed 4 to the least-loaded verified account;
  - the extra work (wide random planes, float64 block) to the next one.
- The resulting assignment is recorded in `evaluation_matrix.json`
  (`assignment`).
- T4 Hessian timings therefore come from the main jobs themselves
  (`timing.json`, task `seconds`).

**Accounts as originally planned.**

| account | seed | extra work |
|---|---|---|
| maxnicaise | 0 | + Hessian planes |
| maxlefrr | 1 | |
| maxnikezz | 2 | |
| maximemonstrenikez | 3 | |
| maxlebossdu91 | 4 | |
| maxfrrsava | — | all wide random planes and the float64 block |

- Each account runs 2 × T4.
- Every account ships the checkpoints and draw files its tasks need. The
  tasks travel inside the kernel payload.

**Local pilot timings** (RTX 3050 laptop, TF32 off, per evaluation of both
1,000-image probes):

| evaluation | time |
|---|---|
| saved or centre_frozen, float32 | 0.12 s |
| pointwise, float32 | 0.25 s |
| saved or centre_frozen, float64 | 2.4 s |
| pointwise, float64 | 4.8 s |
| HVP, float32 (one probe) | 0.71 s |
| HVP, float64 (one probe) | 17 s |
| Lanczos iteration | ≈ 0.75 s |

**Estimates** (with v2 T4 evaluation timings and the HVP timing from the
Kaggle pilot, before launch): see `evaluation_matrix.json`. Wall time and GPU
time are recorded separately: `timing.json` gives the job wall time and the
sum of per-task seconds per GPU worker.
