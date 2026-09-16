# Loss landscape and Hessian: latest results (`landscape_v3`)

## Identity and scope

- **Study directory:** `studies/landscape_v3/`; code in `landscape_v3/`, branch
  `visualization`.
- **Code:** commit `9e9275c` pushed 2026-09-15 17:24. The raw results and the
  analysis were collected afterwards (uncommitted; shipped in
  `landscape_latest_details.zip`).
- **Relation to v2:** a no-training extension of `landscape_v2`.
  - Same 20 networks and checkpoints.
  - Same probes and calibration set.
  - Same random directions.
  - 33,820 matching v2 evaluations were reused by exact key.
- **Protocol:** written before the main jobs, with dated pilot revisions, in
  `protocol/PROTOCOL.md` inside the details archive.

## 1. Questions and completed matrix

The four questions:

1. Do the v2 sensitivity differences persist close to and far from the final
   solutions?
2. Does the weak test-image advantage survive a larger evaluation with all
   directions?
3. When during training do the differences emerge?
4. Do Hessian-informed directions reveal curvature the random slices miss?

| block | specification | status |
|---|---|---|
| A: amplitudes | epoch 30, final state; directions 0–19, both signs; ε ∈ {0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.35, 0.5}; 3 BN policies; train and test probes | **complete**, 24,000/24,000 direction values |
| A-f64 | directions 0–1, ε ∈ {0.005, 0.01, 0.02, 0.05}, 3 policies, float64 vs float32 | **complete**, 960 comparisons |
| B: validation | directions 0–19, ε ∈ {0.1, 0.25}, both signs; saved and every-point recalibration; 10,000-image training subset and full 10,000-image test set | **complete** |
| C-temporal | epochs 3, 6, 9, 12, 18, 21, 30; directions 0–9 renormalised per checkpoint; ε ∈ {0.1, 0.25}; saved and every-point; produced state and final state | **complete** |
| C-transitions | every schedule transition (resolution-only 2; Gaussian-only and combined 7 each), before/after/final states, v2 fixed-weight protocol plus saved ε ∈ {0.1, 0.25} | **complete** |
| D: Hessian | 20 networks × 2 probes × 2 frozen BN policies × 2 coordinate systems = 160 eigenproblems, 2 starts each, plus checks, quadratic forms, 1D cuts | **complete**, 160/160 converged |
| E: seed-0 planes | random 41×41 over [−0.5, 0.5]²; Hessian (top1, top2) and (top1, min) planes under both frozen policies | **complete** |
| E: wide random planes, all seeds | 21×21 over [−0.5, 0.5]², seeds 1–4, every-point recalibration | **complete**: 0 missing vertices in 48 surfaces; rerun on four GPU accounts (see §6) |

**Totals.**

- 606/606 tasks complete across 10 Kaggle jobs (6 main jobs and 4 wide-plane
  reruns), and 96,116 new evaluation points.
- No non-finite value. 0 missing direction points in blocks A–C; 0 missing
  surface vertices.

**Integrity.**

- Every downloaded file matched its job manifest by sha256: 286, 286, 286,
  286, 318 and 50 files for the main jobs; 10 per rerun.
- Every job's `environment.json` shows 2 × Tesla T4.
- Before trusting reused points, each job re-evaluated 6–30 of the reused v2
  points: maximum absolute CE difference 0.0.

**Compute.** 54,652 GPU-seconds (15.2 T4 GPU-hours). Job wall times were
22–104 min for the main jobs and about 2.5 min for each rerun, on 2 × T4.

## 2. Setup

- **Networks:** plain, resolution-only (`resolution_max_b1`), Gaussian-only
  (`gaussian_postrelu`), combined (`resolution_max_b1_gaussian_conv`); CIFAR-10
  ResNet-20-BN.
  - Training seeds 0–4. Runs of a seed share their initialisation, BN buffers
    and data order.
  - Seeds are the only replication unit.
- **Checkpoints:**
  - epoch 30 for blocks A, B and D;
  - epochs 3–30 for C;
  - the checkpoint taken at each transition update for the transition block.
- **State:**
  - *final* = full resolution, no filter; bitwise plain ResNet-20.
  - *produced* = the state used by the last update before the checkpoint.
  - *next* = the state of the next update.
- **Data:**
  - probes: 1,000 class-balanced training images and 1,000 class-balanced
    test images;
  - validation: `train_large` (10,000 training images containing the probe,
    disjoint from calibration) and the full test set;
  - calibration: 2,000 training images (`calib2k`), disjoint from the probe,
    fixed order, batches of 500.
- **Loss:** mean cross-entropy **without** weight decay; accuracy recorded
  separately.
- **Precision:** float32 with TF32 off and cuDNN deterministic. Float64 was
  used for the A-f64 block and for Hessian verification.

**BatchNorm policies.**

| policy | definition |
|---|---|
| saved/frozen | checkpoint running statistics, reloaded before every evaluation |
| recalibrated at every point | reset BN; cumulative average over `calib2k` (training mode, no_grad) for every evaluated weight vector and state; then inference mode |
| recalibrated at centre, then frozen | the same calibration once at the unperturbed weights and state; statistics held fixed for every perturbation and for the Hessian |

Float64 evaluations under centre-freezing reuse the float32 centre statistics,
so only the arithmetic changes.

## 3. Definitions

**Parameter subset and directions.**

- Perturbed: every convolution weight and the final linear weight,
  P = 268,336 in G = 698 blocks. Biases, BN affine parameters and BN buffers
  stay fixed.
- A direction d_k is a Li et al. filter-normalised Gaussian draw:
  d_g = ‖θ_g‖ · u_g/‖u_g‖ for each output filter or fc row g.
- The same draws are shared by the four methods of a seed. Directions for a
  checkpoint are normalised at that checkpoint.

**Scale.** r(δ) = ((1/G) Σ_g ‖δ_g‖²/‖θ_g‖²)^½, so r(εd) = ε.

**Sensitivity statistics.**

- Per direction: S_{m,s,k}(ε) = (L(θ + εd_k) + L(θ − εd_k))/2 − L(θ). Every
  direction, sign and centre is kept, and nothing is clamped.
- Per seed: S̄_{m,s}(ε) = mean over k.
  - Within-checkpoint (Monte Carlo) uncertainty = SD over directions,
    typically 30% of S̄.
  - This is **not** seed variation.
- Paired contrast: Δ_{m,s}(ε) = S̄_{m,s} − S̄_{plain,s}. Negative Δ means a
  smaller perturbation-induced increase, **not** a lower loss.
- Summaries report the 5 seed values, the mean, the SD (n − 1) and the count
  of negative Δ.
- Curvature proxy: C_d(ε) = 2S_d(ε)/ε², approximately dᵀHd only where a
  second-order expansion applies.

**Hessian.**

- **Objective:** the scalar L(w) = mean CE over one probe (no weight decay),
  with inference-mode BN and **frozen buffers**:
  - the saved checkpoint statistics, or
  - statistics recalibrated once at the centre.
- **Variables:** w = the 268,336 perturbed weights; everything else is
  constant.
- **The object:** the exact Hessian (double back-propagation), not a Fisher or
  Gauss–Newton approximation. ReLU and max-pool make it an almost-everywhere
  second derivative.
- **Two coordinate systems:**
  - ordinary H;
  - relative H_rel = AᵀHA, with A_g = ‖θ_g‖·I frozen at the checkpoint (no
    zero-norm blocks occurred).
- **What was estimated:**
  - the two largest and the smallest algebraic eigenvalues, with
    eigenvectors, by Lanczos with full reorthogonalisation, 2 deterministic
    starts;
  - ‖∇L‖;
  - dᵀHd for the 20 random directions;
  - no trace.
- **Convergence criterion:** explicit ‖Hy − θy‖ ≤ 1e-3 · max(|θ|, 1e-3·|θ_top1|).
- **Scope:** this is **not** the Hessian of the loss with every-point
  recalibration, whose statistics depend on the weights. That loss was studied
  only through finite differences (blocks A–C).

## 4. Results

### 4.1 Centre loss (epoch 30, final state; seed means)

| probe, policy | plain | resolution | Gaussian | combined |
|---|---|---|---|---|
| train CE, recalibrated | 0.192 | 0.300 | 0.269 | 0.299 |
| test CE, recalibrated | 0.741 | 0.562 | 0.577 | 0.531 |
| test accuracy, recalibrated | 0.763 | 0.806 | 0.800 | 0.820 |
| train CE, saved | 0.193 | 0.301 | 0.269 | 0.298 |
| test CE, saved | 0.745 | 0.566 | 0.579 | 0.534 |

- Centre-frozen centres equal the every-point centres bitwise.
- These predictive results are separate from sensitivity. Every continuation
  method has higher probe training CE and lower test CE than plain.

### 4.2 Q1: amplitude dependence (block A)

The table shows mean Δ, the relative change vs plain, and the number of seeds
with Δ < 0, out of 5.

| policy, probe | ε | resolution | Gaussian | combined |
|---|---|---|---|---|
| saved, train | 0.005 | −0.0006 (−35%) 5 | −0.0002 (−13%) 4 | −0.0007 (−39%) 5 |
| | 0.1 | −0.268 (−36%) 5 | −0.024 (−3%) 2 | −0.312 (−43%) 5 |
| | 0.5 | −3.19 (−15%) 5 | −4.30 (−20%) 5 | −4.63 (−21%) 5 |
| saved, test | 0.005 | −0.0004 (−28%) 5 | −0.0001 (−9%) 4 | −0.0005 (−36%) 5 |
| | 0.1 | −0.212 (−34%) 5 | +0.010 (+2%) 2 | −0.245 (−39%) 5 |
| | 0.5 | −2.94 (−14%) 5 | −4.01 (−19%) 5 | −4.32 (−20%) 5 |
| recalibrated, train | 0.005 | −0.0003 (−42%) 5 | +0.0000 (+5%) 1 | −0.0003 (−43%) 5 |
| | 0.1 | −0.061 (−31%) 5 | +0.084 (+44%) 0 | −0.068 (−35%) 5 |
| | 0.5 | −0.456 (−16%) 5 | −0.041 (−1%) 3 | −0.562 (−19%) 5 |
| recalibrated, test | 0.005 | +0.0001 (+47%) 1 | +0.0003 (+145%) 0 | +0.0000 (+12%) 1 |
| | 0.02 | +0.0000 (+3%) 3 | +0.0030 (+78%) 0 | −0.0005 (−12%) 4 |
| | 0.1 | −0.0096 (−9%) 5 | +0.111 (+107%) 0 | −0.013 (−13%) 4 |
| | 0.5 | −0.178 (−7%) 5 | +0.208 (+9%) 0 | −0.264 (−11%) 5 |
| centre-frozen, train | 0.1 | −0.268 (−36%) 5 | −0.026 (−3%) 2 | −0.315 (−43%) 5 |
| centre-frozen, test | 0.1 | −0.212 (−33%) 5 | +0.007 (+2%) 2 | −0.248 (−39%) 5 |

Every ε, policy and probe is in `A_paired_vs_plain.csv`.

**Observations.**

- **Frozen policies, resolution-only and combined.** Under saved and
  centre-frozen statistics, resolution-only and combined are less sensitive
  than plain in 5/5 seeds at **every** ε from 0.005 to 0.5 on both probes. The
  relative gap is largest near the solution (−28 to −43%) and shrinks at
  ε = 0.5 (−14 to −21%).
- **Frozen policies, Gaussian-only.** Mixed at ε = 0.02–0.1 (2–4/5); below
  plain in 5/5 seeds from ε = 0.25 (train from 0.2).
- **Centre-frozen ≈ saved.** The two agree to within a few percent at every ε.
  Recalibrating BN once at the centre barely changes the frozen-statistics
  picture; per-point recalibration does.
- **Every-point recalibration, training probe.** Resolution-only and combined
  stay below plain in 5/5 seeds at all ε. Gaussian-only is **above** plain in
  5/5 seeds for ε = 0.05–0.35 (+11 to +44%) and mixed at the smallest and
  largest ε.
- **Every-point recalibration, test probe.**
  - Resolution-only is below plain from ε = 0.05 (5/5, except 4/5 at 0.2),
    but **not near the solution**: 1/5 at ε ≤ 0.01, 3/5 at 0.02–0.025.
  - Combined is below plain in 4/5 seeds for ε = 0.02–0.35 and 5/5 at 0.5;
    near the solution it is 1/5 at ε = 0.005 and 3/5 at 0.01.
  - Gaussian-only is above plain in 5/5 seeds at every ε (+9 to +145%).
- **Near-solution values are tiny.** With every-point recalibration on the test
  probe, S̄(0.005) for plain is 2.0e-4 ± 0.6e-4 nats, so near-solution
  differences are of order 1e-4 nats.

**Numerical checks.**

- Float64 re-evaluation: no suspicious difference in 960 comparisons (maximum
  |S_f32 − S_f64| = 4.0e-7). Near-solution results are not a precision
  artefact.
- 11 of 24,000 individual S values are negative, and 1 in float64; they are
  kept.

**C_d(ε) does not stabilise.** For plain (median over seeds):

| policy | trend of C̄ as ε goes 0.005 → 0.5 |
|---|---|
| saved, train probe | rises 119 → 175 |
| every-point recalibration, train probe | falls 48 → 23 |
| every-point recalibration, test probe | 13 → 22 |

Under frozen policies, C̄(ε)/mean dᵀHd at the smallest ε is 1.17 on the
training probe and 0.80 on the test probe; it is not 1. So even at ε = 0.005
the finite-difference quantity is not the HVP quadratic form.

This agrees with the checks in §5: the loss is piecewise smooth (ReLU,
max-pool), and finite differences include kink contributions. The smallest
amplitude is not automatically the "most Hessian-like".

### 4.3 Q2: larger evaluation, all 20 directions (block B)

- **Sign retention.** Every paired difference kept its sign between the probe
  and the larger set in ≥ 4/5 seeds; 23 of 24 contrasts in 5/5.
- **Magnitudes.** They agree within ~40%.
- **Every-point recalibration, full test set:**

| method | ε = 0.1 | ε = 0.25 |
|---|---|---|
| resolution-only | −0.0133 (5/5; probe −0.0096) | −0.084 (5/5) |
| combined | −0.0105 (4/5) | −0.093 (4/5) |
| Gaussian-only | +0.111 (0/5) | +0.379 (0/5) |

- **Training subset (10k), every-point recalibration.** Resolution-only
  −0.057 / −0.250, combined −0.064 / −0.291 (5/5); Gaussian-only +0.091 /
  +0.264 (0/5).

**Answer.** The weak test-image advantage of resolution-only survives
20-direction full-test validation in 5/5 seeds, but it stays small
(≈ 10% of plain's S̄). Combined's reverses in one seed. This supersedes the
v2 two-direction validation (resolution-only −0.006, 2/5 seeds).

### 4.4 Q3: during training (block C)

Directions are renormalised at each checkpoint: this measures sensitivity to
**relative** perturbations over time, not a fixed physical plane. Masked-weight
norms are nearly identical across methods (36.3 at epoch 3 → 33.5 at epoch 30
for all four), so the relative normalisation is not hiding a norm difference.

Counts are seeds with Δ < 0 at ε = 0.25, network in its **produced** state.

| policy, probe | method | e3 | e6 | e9 | e12 | e18 | e21 | e30 |
|---|---|---|---|---|---|---|---|---|
| saved, train | resolution | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| | Gaussian | 1 | 0 | 0 | 1 | 2 | 5 | 5 |
| | combined | 5 | 5 | 3 | 3 | 5 | 5 | 5 |
| every-point, train | resolution | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| | Gaussian | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| | combined | 0 | 2 | 0 | 1 | 4 | 5 | 5 |
| every-point, test | resolution | 5 | 5 | 4 | 4 | 4 | 4 | 5 |
| | Gaussian | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| | combined | 0 | 2 | 0 | 2 | 2 | 4 | 4 |

**Observations.**

- **Resolution-only.** Less sensitive than plain from the first measured
  checkpoint (epoch 3), under both policies, while still training at r = 16.
- **Gaussian-only, saved statistics.** More sensitive than plain while the
  filter is active; it becomes less sensitive only at epochs 21–30, after
  G = 0.
- **Gaussian-only, every-point recalibration.** More sensitive than plain at
  every measured epoch, from epoch 3.
- **Combined.**
  - Saved statistics: below plain early (epochs 3, 6), mixed at epochs 9–12,
    below from 18.
  - Every-point recalibration: not below plain until epoch 18 (train) or 21
    (test), i.e. after the resolution is restored and near the end of
    filtering.
- **Final state at intermediate checkpoints.** Evaluated there instead of the
  produced state, continuation checkpoints before epoch 18 have much higher
  centre CE than plain: +0.40 to +0.96 nats with every-point recalibration,
  +2.3 to +4.9 with saved statistics. Paired sensitivities change too; for
  example, Gaussian-only is below plain at epochs 3–9 in the final state
  (every-point, 5/5). Those are sensitivities around a high-loss centre.

**Fixed-weight transitions** (`C_transitions_*.csv`, figure `C4`).

- **Lowest centre CE.** The state that produced the weights ("before") has the
  lowest centre CE in 5/5 seeds at every transition, policy and probe, with
  one exception. At the Gaussian-only epoch-3 transition (G 1 → 0.85) with
  every-point recalibration, "after" is lower than "before" in 1/5 seeds on
  the training probe and 3/5 on the test probe; the test-probe means are
  1.125 vs 1.126 nats.
- **Late switch-off (epoch 21, G 0.3 → 0).** It changes centre CE by at most
  0.004 nats.
- **Joint changes.** Combined at epochs 6 and 12 changes r and G together and
  is labelled a joint change. For example, at epoch 12 (r 24 → 32 and
  G 0.6 → 0.5) the test CE with every-point recalibration rises from 0.646 to
  0.999. Neither component is identified separately.

### 4.5 Q4: Hessian (block D)

**Convergence and verification.**

- **Lanczos.** 160/160 problems met the criterion for all three targets.
  - 50–100 iterations; basis orthogonality error ≤ 5.8e-15.
  - Explicit relative residuals ≤ 6.1e-7 (top1, top2) and ≤ 9.9e-4 (min).
  - Start-to-start agreement ≤ 7.5e-8 (top) and ≤ 1.5% (min).
  - Float32 vs float64 Rayleigh quotients ≤ 1.6e-4 relative.
- **HVP checks, 40 (each network, both policies).**
  - Repeats bitwise identical.
  - Symmetry |uᵀHv − vᵀHu| ≤ 5.7e-6 relative.
  - Float32 vs float64 ≤ 4.1e-4.
  - Central finite differences of float64 gradients converge at second order
    when only the fc weight is perturbed: 5.8e-5, 5.8e-7, 5.8e-9 at
    h = 1e-2, 1e-3, 1e-4. The HVP itself is correct.
  - With all weights perturbed the error *grows* as h shrinks (0.12, 0.33,
    1.0): gradient jumps at ReLU/max-pool kinks.
- **Dense reference.** On a small conv+BN+ReLU+max-pool network, HVPs match the
  dense Hessian to 5e-16 (local pilot).

**Spectrum.** Seed means; the ratio to plain in brackets with the number of
seeds below 1; all 5 seeds per cell.

| coordinates, policy, probe | quantity | plain | resolution | Gaussian | combined |
|---|---|---|---|---|---|
| ordinary, saved, train | λ₁ | 2382 | 1444 (0.61; 5) | 1281 (0.54; 5) | 1148 (0.48; 5) |
| ordinary, saved, test | λ₁ | 3237 | 1677 (0.52; 5) | 1500 (0.47; 5) | 1282 (0.40; 5) |
| ordinary, centre-frozen, train | λ₁ | 2354 | 1438 (0.61; 5) | 1275 (0.55; 5) | 1135 (0.48; 5) |
| relative, centre-frozen, train | λ₁ | 3775 | 2297 (0.61; 5) | 2090 (0.56; 5) | 1838 (0.49; 5) |
| relative, centre-frozen, test | λ₁ | 5194 | 2654 (0.51; 5) | 2430 (0.47; 5) | 2052 (0.40; 5) |
| ordinary, saved, train | λ_min | −5.51 | −4.08 (4) | −3.61 (5) | −2.98 (5) |
| relative, centre-frozen, test | λ_min | −15.14 | −7.47 (5) | −8.22 (5) | −7.05 (5) |
| ordinary, saved, train | ‖∇L‖ | 3.71 | 2.93 (4) | 2.78 (5) | 2.23 (5) |

For λ_min the "seeds below 1" count refers to |λ_min| relative to plain. All
160 cells are in `D_hessian_problems.csv` and `D_hessian_summary.csv`.

**Observations.**

- **Top curvature.** In every coordinate system, frozen policy and probe, all
  three continuation methods have smaller λ₁ than plain in 5/5 seeds (ratios
  0.40–0.61), with the ordering combined < Gaussian < resolution < plain.
- **Negative curvature.** The smallest eigenvalue is negative in all 160
  problems, stable in float32 and float64, and smaller in magnitude for the
  continuation methods (4–5/5).
- **Gradient norms.** 2.2–6.7 are clearly nonzero, so the checkpoints are
  **not** stationary points, and nothing here identifies local minima.
- **Gaussian-only is inverted between frozen and recalibrated measurements.**
  Its **frozen-BN** Hessian has a smaller top eigenvalue than resolution-only,
  while its **every-point-recalibrated** finite sensitivity is the largest of
  all methods. The two measure different functions, so they must not be
  merged.

**Do Hessian directions show what random slices miss?**

- **Top direction.** Along the r-normalised top eigendirection,
  δᵀHδ = G·λ_rel ≈ 698 × 3775 ≈ 2.6e6 per unit r² for plain, against a random
  mean dᵀHd of about 100–170.
  - The extreme direction is ~10⁴ times steeper at equal RMS relative
    displacement. It is concentrated in stages 2–3 (39% and 46% of r²) and
    almost absent from the fc layer (0.3%).
  - Float64 finite differences confirm this curvature: 2S/t² ÷ δᵀHδ has
    median 1.004 (range 0.93–1.06) at t = 1e-5 and 1.010 at t = 1e-4. At
    t = 1e-2 the ratio is 0.47: already strongly non-quadratic.
- **Negative curvature.**
  - Random slices essentially never show it: S < 0 in 11/24,000 values.
  - Along the min eigendirection, finite differences have the Hessian's
    (negative) sign in 140/160 problems at t = 1e-4 and 139/160 at t = 1e-3,
    with magnitudes ≈ 0.45–0.6 of the quadratic form.
  - At t = 1e-2 the second difference is positive in 160/160.
  - The a.e.-Hessian negative curvature is therefore real in a very small
    neighbourhood (t ≲ 1e-3). It is dominated by nonsmooth and higher-order
    effects beyond that, and is invisible at the amplitudes of the random
    study.

**Seed-0 planes (figures `E_hessian_plane_*`).**

- Axis extents follow the prespecified rule T_i = √(4/(G·|λ_i^plain|)):
  - T₁ = 0.0012 along top1;
  - T₂ = 0.0015 along top2;
  - T₂ = 0.032 along min.
- The top1/top2 plane rises to 4–6 nats at its corners.
- On the top1/min plane, the loss along min is flat for |t| ≲ 3e-3, then rises
  to 40–170 nats by |t| = 0.03. The rule assumed a quadratic model, which fails
  along min; the plane therefore does not display the negative-curvature
  region.
- These are representative slices (seed 0 only). The mapped directions are
  not orthogonal in weight space and differ from any random plane.

## 5. What repeats, what is mixed, what is unresolved

**Repeats in 5/5 seeds.**

- Resolution-only and combined:
  - smaller finite sensitivity than plain under saved and centre-frozen BN at
    all ε = 0.005–0.5;
  - under every-point recalibration on the training probe and the 10k training
    subset.
- Gaussian-only: larger than plain under every-point recalibration at
  ε = 0.05–0.35 (train) and at all ε (test, full test set).
- Resolution-only: less sensitive than plain from epoch 3.
- Top Hessian eigenvalue: continuation < plain, in both coordinate systems and
  under both frozen policies.
- Negative λ_min in every problem.

**Mixed or convention-dependent.**

- Gaussian-only's ranking depends on the BN policy and ε.
- Resolution-only and combined on test images with every-point recalibration:
  weak (−7 to −15%), absent or reversed at ε ≤ 0.02, combined 4/5.
- Combined during training under every-point recalibration.
- Near-solution C_d is not the HVP quadratic form.

**Unresolved.**

- Why every-point recalibration reverses Gaussian-only.
- Whether the negative curvature matters for training.
- Any causal link from sensitivity or curvature to test accuracy. Gaussian-only
  improves accuracy as much as resolution-only while being more sensitive
  under recalibration, so sensitivity does not track accuracy here.

**Proposed explanations** (untested): BN statistics shifting under
perturbation; filter-induced changes in activation scale. None is established
by these measurements.

## 6. Failures, incomplete cells, deviations

- **Wide random planes, seeds 1–4 (21×21).**
  - Originally assigned to account `maxfrrsava`, whose job had been pushed
    before GPU access was active and ran without a GPU. It was still running
    after more than 2 h against a ~20 min estimate. Its outputs are not used.
  - The same 16 tasks were rerun unchanged, one seed per GPU account
    (`landscape-v3-<account>-rerun`), with a new guard that aborts a job when
    no CUDA device is visible.
  - All completed and verified. Results are identical in definition, and the
    reused v2 points reproduced exactly.
- **Changes before any main job** (dated in `PROTOCOL.md`):
  - finite-difference steps reduced to 1e-5–1e-2 and the cut grid made
    log-spaced, because r-normalised eigendirections explode by t = 0.01;
  - centre-frozen float64 made to reuse the float32 statistics;
  - a verification bug fixed: the float64 model was left perturbed between
    vectors;
  - account layout changed after two accounts initially lacked GPU access,
    and a seventh account was added.
- **Plane-extent rule.** Executed as prespecified, but uninformative along the
  min direction (§4.5).
- **Interpolation.** No new interpolation or PCA compute.
  - 20/20 v2 straight segments have barriers.
  - The v2 claim that the resolution→combined barrier is never smaller than
    plain→resolution is false in seeds 1 and 4 (train probe) and is corrected.
  - PCA remains descriptive only: PC1+2 explain 57–59%.
- **Not done by design:**
  - Hessian of the every-point-recalibrated loss;
  - optimised connecting paths;
  - training;
  - Hessian at intermediate checkpoints.
