# Loss geometry and Hessian: final analysis and paper handoff

**Date.** 2026-09-17.
**Branch.** `geometry-final`, started from `visualization@9e9275c`; worktree
`Projet_filiere-worktrees/geometry-final`, committed locally, not pushed.
**Account.** Kaggle `maxmonstre` only, 2 × Tesla T4.
**Manuscript.** Not edited.

## 1. What existed and what was computed

The full audit is in `INVENTORY.md` (inside `geometry_reproducibility.zip`).

| item | status | how |
|---|---|---|
| 20 final checkpoints (4 methods × seeds 0–4), sensitivity (3 BN policies, 20 directions, 10 amplitudes), larger sets, temporal, transitions, interpolation, PCA | **reused** | landscape_v3 raw outputs, unchanged |
| 160 Hessian problems (top1/top2/min, vectors, gradient norms, residuals, float64 checks) | **reused** | no new eigenproblem |
| quadratic forms d_kᵀHd_k (20 directions), quadratic vs finite differences, eigendirection cuts (frozen and pointwise), seed-0 Hessian planes | **reused** | re-analysed only |
| pointwise seed-0 41×41 random grids | **reused** | 1,240 v3 + 441 v2 vertices per model; they match the paper's records to 9e-16 |
| centre-frozen seed-0 41×41 random grids | **new, missing vertices only** | 1,652 × 4 vertices computed; 29 × 4 reused block-A points re-evaluated as a check (max \|ΔCE\| = 0.0) |
| traces tr(H), tr(AᵀHA), tr(HC) for 40 centre-frozen objectives | **new** | 128 Rademacher draws each (5,120 HVPs) |
| Idriss grid traces (`4179584`) and BN-policy 2×2 (`1a60649`) | **not reusable** | other checkpoints, all 269,722 parameters, 5k/2k images, saved or train-mode batch statistics (§6) |

**Primary objective, unchanged from landscape_v3:**

- mean CE without weight decay;
- 1,000-image training and test probes;
- 268,336 conv + fc weights (698 blocks);
- centre-frozen BN: recalibrated once at the unperturbed weights on `calib2k` (2,000 training images, original order, 4 × 500), buffers then held fixed, inference mode;
- float32.

**Checks on the new work:**

- **Dense reference** (conv + BN + ReLU + max-pool, 528 parameters):
  - the block aggregation equals the dense quadratic form to 9e-14;
  - 20,000-draw means lie within 0.7 SEM of the exact traces;
  - buffers are bitwise unchanged.
- **On Kaggle, for all 40 objectives:**
  - the centre loss and gradient norm reproduce the existing centre-frozen Hessian records exactly (relative difference 0.0);
  - buffers equal the centre statistics, and buffers and masked weights hash identically before and after every stage.
- **Rademacher draws** regenerate bitwise locally (120/120 checked).
- **Centre CE:** centre-frozen equals pointwise bitwise at the centre, for all 4 models.
- **Integrity:** 153/153 manifest files verified by sha256.

**Precision rule (predeclared in `protocol/PROTOCOL.md`).**

- At 64 draws, 29 of 120 trace estimates had SEM > 5 %, and every seed × probe quartet contained at least one. All 10 quartets were extended to 128 draws.
- At 128 draws every estimate meets the rule: maximum SEM 4.8 %, range 2.7–4.8 %.
- The 256-draw cap was not needed.
- **Control variate** (existing top1/top2 eigenpairs; secondary, raw estimates remain primary):
  - SEM × 0.71 (median);
  - CV minus raw estimates: mean 0.07 raw SEM, SD 0.83.

**Measured compute.**

| stage | T4 GPU-seconds | GPU-hours |
|---|---|---|
| timing pilot (16 HVP draws, 120 vertices) | 31 | 0.01 |
| traces (5,120 draws; 1.08 s median per draw) | 5,689 | 1.58 |
| centre-frozen grid (6,608 vertices; 0.108 s median) | 739 | 0.21 |
| **total main job** (incl. worker startup and checks) | **6,463** | **1.80** |

- Wall time: 55 min on 2 × T4.
- The pre-launch projection was 0.87 GPU-h at 64 draws and 2.9 GPU-h worst case.
- For reference, landscape_v3 used 15.2 GPU-h.

## 2. Verified findings

All results use CIFAR-10, ResNet-20-BN, SGD, the final 30-epoch checkpoints and five training seeds. ± denotes the SD over seeds; Monte Carlo SEM is reported separately.

### 2.1 Leading eigenvalues and traces (centre-frozen, 1k probes)

| probe | method | λ_max(H) | tr(H) | λ_max(AᵀHA) | tr(AᵀHA) | tr(HC) |
|---|---|---|---|---|---|---|
| train | Plain | 2,354 ± 160 | 14,976 ± 621 | 3,775 ± 260 | 24,092 ± 961 | 104.7 ± 6.5 |
| train | R | 1,438 ± 107 | 9,966 ± 304 | 2,297 ± 167 | 16,002 ± 454 | 73.5 ± 1.6 |
| train | G | 1,275 ± 128 | 9,844 ± 729 | 2,090 ± 215 | 16,624 ± 1,205 | 99.1 ± 5.6 |
| train | RG | 1,135 ± 70 | 8,535 ± 513 | 1,838 ± 107 | 13,957 ± 781 | 66.3 ± 2.8 |
| test | Plain | 3,230 ± 296 | 21,408 ± 1,198 | 5,194 ± 480 | 34,463 ± 1,812 | 150.3 ± 6.1 |
| test | R | 1,664 ± 70 | 11,505 ± 365 | 2,654 ± 126 | 18,485 ± 570 | 85.3 ± 3.1 |
| test | G | 1,482 ± 104 | 11,591 ± 625 | 2,430 ± 179 | 19,580 ± 1,047 | 117.1 ± 7.6 |
| test | RG | 1,268 ± 69 | 9,662 ± 528 | 2,052 ± 107 | 15,806 ± 786 | 75.1 ± 2.1 |

Within-seed ratios to Plain (mean ± SD). In brackets: seeds below Plain by more than 2 paired Monte Carlo SEM / seeds above by more than 2 SEM.

| probe | method | λ_max(H) | tr(H) | tr(AᵀHA) | tr(HC) |
|---|---|---|---|---|---|
| train | R | 0.61 ± 0.07 | 0.67 ± 0.04 (5/0) | 0.67 ± 0.04 (5/0) | 0.71 ± 0.05 (5/0) |
| train | G | 0.55 ± 0.08 | 0.66 ± 0.07 (5/0) | 0.69 ± 0.07 (5/0) | **0.95 ± 0.11 (2/1)** |
| train | RG | 0.48 ± 0.05 | 0.57 ± 0.06 (5/0) | 0.58 ± 0.05 (5/0) | 0.64 ± 0.06 (5/0) |
| test | R | 0.52 ± 0.04 | 0.54 ± 0.03 (5/0) | 0.54 ± 0.03 (5/0) | 0.57 ± 0.01 (5/0) |
| test | G | 0.46 ± 0.06 | 0.54 ± 0.05 (5/0) | 0.57 ± 0.04 (5/0) | 0.78 ± 0.07 (5/0) |
| test | RG | 0.40 ± 0.05 | 0.45 ± 0.04 (5/0) | 0.46 ± 0.04 (5/0) | 0.50 ± 0.02 (5/0) |

- **Agreement across the two estimators of tr(HC).** The trace estimate and the existing mean over 20 sampled directions agree: z = (tr(HC) − mean d_kᵀHd_k) / combined SEM has mean 0.06 and SD 0.82 over 40 objectives, with none beyond ±2.
- **Leading two eigenvalues' share.** (λ₁ + λ₂) / tr(H) is 0.24–0.29, lower for R, G and RG than for Plain.

### 2.2 Extreme versus distribution-averaged curvature

- **Scale gap.** Per unit squared RMS relative displacement r²:
  - the steepest direction, G·λ_max(AᵀHA), is 1.3–3.6 × 10⁶;
  - our random-direction distribution sees tr(HC) = 66–150;
  - the ratio is 13,000–27,000 per checkpoint.
- **What random slices cannot show.** A 2-D slice through random directions cannot display either extreme direction.
- **Eigendirections are not rescalings.**
  - The radial share (the part of the squared relative displacement that rescales whole filters or rows) is 0.4–1.0 % for all stored extreme eigenvectors.
  - A random filter-normalised direction has 0.43 %.
- **The rankings differ.**
  - λ_max(H): RG < G < R < Plain in 10/10 seed-probe pairs.
  - tr(HC): RG < R < G < Plain in 9/10 pairs; in the remaining pair (training probe) G is above Plain.
  - The existing 20-direction mean on the training probe: G is above Plain in 3/5 seeds.

### 2.3 BN policy: same perturbations, same directions

- **Plain.** Pointwise recalibration lowers S̄ relative to centre-frozen statistics by a factor of 2.7–7.4 on the training probe and 5.4–8.5 on the test probe (median ratio 0.14–0.37 and 0.12–0.18). The factor is larger at larger ε.
- **Sign changes for G's difference from Plain.**
  - Training probe: −2 to −20 % under centre-frozen statistics, versus +5 to +44 % under pointwise recalibration for ε ≤ 0.35 (0–1 of 5 seeds below Plain).
  - Test probe: +9 to +145 % under pointwise recalibration, 0/5 seeds below Plain at every ε.
- **R and RG.** They stay below Plain under both policies on the training probe (5/5). On the test probe under pointwise recalibration they are weak or reversed at ε ≤ 0.02 (R: 1–3/5 seeds below).
- **Along each model's leading block-relative eigendirection** (training probe, median over 20 model-seed pairs, same weights):

  | t (r units) | centre-frozen rise | pointwise rise |
  |---|---|---|
  | 0.001 | 0.94 nats | 0.005 |
  | 0.01 | 29.5 | 0.45 |
  | 0.1 | 2.3 × 10⁴ | 4.4 |

  Along the min direction, pointwise recalibration also removes most of the rise: 83 % at t = 0.001, 97 % at 0.01 and more than 99.7 % from t = 0.1.

### 2.4 Local HVP versus finite differences

- **Leading eigendirection** (float64, centre-frozen, 80 problems):
  - C_fd / δᵀHδ has median 1.01 (range 0.93–1.06) for t = 1e-5 to 1e-4;
  - median 1.02 at 3e-4, and 1.10 (0.92–1.62) at 1e-3;
  - 0.48 (0.22–0.66) at 1e-2;
  - the sign agrees 80/80 everywhere.
- **Min direction:**
  - the sign agrees in 53/80, 57/80, 70/80, 72/80 and 67/80 problems for t = 1e-5 to 1e-3, with magnitude ratio ≈ 0.4–0.6;
  - the second difference is positive in 58/80 at 3e-3 and in 80/80 at 1e-2.
- **Random directions** (centre-frozen, median over seeds of C(ε) / mean dᵀHd):
  - training probe: 1.15–1.26 at ε = 0.005, and the same (1.16–1.26) at ε = 0.05;
  - test probe: 0.68–0.93 at ε = 0.005;
  - rising above 1.3–2.1 at ε ≥ 0.2;
  - the ratio does not approach 1 as ε decreases.
- **Numerics excluded.** Float32 versus float64 S differ by ≤ 4e-7 (existing A-f64). Gradient finite differences converge at second order when only the fc weight moves, and diverge as h → 0 when conv weights move (existing hchecks).

## 3. Reconciling the apparent disagreements

1. **BN policy.** The frozen-BN Hessian and pointwise-recalibrated sensitivity describe different functions.
   - Freezing statistics makes perturbations that shift channel activation statistics very costly.
   - Pointwise recalibration absorbs most of that cost, even along the leading eigendirections, which are not filter rescalings.
   - G's reversal between the two policies (§2.3) is therefore a property of the evaluated function, not a numerical conflict.
   - No Hessian of the pointwise-recalibrated function was computed, so the curvature statements in §2.1 do not transfer to it.
2. **Direction.**
   - λ_max and tr(HC) weight the spectrum completely differently: the steepest direction is ~10⁴ times steeper than the average random direction.
   - G reduces the leading eigenvalue and the ordinary trace as much as R does. The average curvature seen by filter-normalised random directions, however, is reduced much less for G: 0.95 on the training probe, not resolved in 3 of 5 seeds.
   - Hence "smaller λ_max" does not imply "smaller curvature in every direction". Random slices and quadratic forms, which sample tr(HC), rank G near Plain, while eigenvalues rank it below R.
3. **Amplitude.**
   - Along the leading eigendirection the curvature is ~10⁶ per r². Finite differences match it until the quadratic model itself predicts rises of order 1 nat (t ≈ 3e-4), then depart as higher-order terms dominate.
   - Along random directions the quadratic form is only ~10² per r². The finite-difference ratio sits at an amplitude-independent 0.7–1.3 over ε = 0.005–0.05, and is not rescued by float64.
   - The piecewise-linear ReLU/max-pool network gives an interpretation consistent with this pattern and with the h → 0 divergence of gradient finite differences. Finite differences include contributions from activation-boundary crossings that the almost-everywhere Hessian omits. Such a term is negligible next to 10⁶ but comparable to 10².
   - This is an interpretation of existing measurements, not a separately established decomposition.
   - The min-direction negative curvature is visible only for t ≲ 1e-3 and is swamped by these effects beyond.

## 4. Answers

- **Do smaller leading eigenvalues accompany smaller traces, in the same models and convention?**
  - Yes.
  - In the centre-frozen objective, R, G and RG have both smaller λ_max and smaller ordinary and block-relative traces than Plain, in 30/30 method × seed × probe comparisons.
  - Every trace difference is resolved beyond 2 Monte Carlo SEM.
  - Trace reductions (ratios 0.45–0.69) are somewhat smaller than eigenvalue reductions (0.40–0.61).
  - The R-versus-G order on traces is not stable across seeds (3/2 splits).
- **How does curvature under our random-direction distribution compare with extreme curvature?**
  - It is four orders of magnitude smaller: tr(HC) = 66–150 against G·λ_max(AᵀHA) = 1.3–3.6 × 10⁶ per r².
  - It ranks methods differently: RG < R < G < Plain versus RG < G < R < Plain.
  - The difference is driven by G: training-probe tr(HC) ratio 0.95 ± 0.11 (2 resolved below, 1 above, 2 unresolved), test 0.78 ± 0.07 (5/5).
  - R and RG are lower on every measure in 5/5 seeds.
- **What changes when BN is recalibrated at each point with directions held fixed?**
  - The evaluated function changes.
  - Sensitivity falls 3–8 fold, and the rise along the eigendirections nearly vanishes.
  - G's sign against Plain flips to "more sensitive" at most amplitudes on both probes.
  - R and RG remain less sensitive on the training probe but lose their test-probe advantage at ε ≤ 0.02.
  - The frozen-BN Hessian results cannot be read as statements about this function.
- **What do the existing finite-difference checks establish?**
  - The HVP is correct: the fc-only control converges at second order, the dense reference matches, and symmetry and precision are verified.
  - Along the leading eigendirection the local quadratic model holds only for t ≲ 3e-4 (r units).
  - Along random directions, finite differences over ε = 0.005–0.05 differ from dᵀHd by 7–32 % (method medians) in a nearly amplitude-independent way, and grow further beyond ε ≈ 0.1.
  - Negative curvature is real only in a t ≲ 1e-3 neighbourhood.
  - Local (HVP) and finite-scale (S, surfaces) geometry are therefore different quantities, and neither substitutes for the other.
- **What is consistent across all five seeds, and what is mixed?**
  - **Consistent (5/5, every probe and convention measured):**
    - λ_max below Plain for R, G and RG;
    - tr(H) and tr(AᵀHA) below Plain (resolved);
    - tr(HC) below Plain for R and RG (both probes) and for G (test probe);
    - λ_min < 0 in all 160 problems;
    - R and RG below Plain in S̄ under frozen statistics at every ε;
    - leading-eigendirection finite differences at t ≤ 3e-4.
  - **Mixed:**
    - G's tr(HC) and its 20-direction mean on the training probe;
    - G's frozen-statistics S̄ at ε = 0.02–0.1 (2–4/5);
    - pointwise test-probe sensitivity of R and RG at small ε;
    - the order of R and G on traces.
  - **No geometric quantity orders the four procedures like their test accuracy:** G gains roughly as much accuracy as R while its tr(HC) and pointwise sensitivity are near or above Plain.
  - None of these measurements establishes a causal account of generalization.

## 5. Figures (all seed-0 illustrations use pre-specified directions; no seed or pair search)

- **Main text:** `F_surfaces_centre_frozen_seed0_zoom`, the common-scale zoom |a|, |b| ≤ 0.1 of the same measured vertices. The full range is dominated by one corner of R, which reaches about 170 nats. Its companion is `F_surfaces_centre_frozen_seed0` (full range, retained).
- **Main or appendix:** `F_curvature_ratios_to_plain`, `F_hessian_vs_random_rise_seed0`, `F_sensitivity_centre_frozen_pointwise` (the regenerated sensitivity figure; it matches the paper's records to 4e-15).
- **Appendix:**
  - `F_surfaces_policy_comparison_{shared,rowwise}_scale[_test]`
  - `F_surface_axis_profiles_seed0`
  - `F_surfaces_pointwise_seed0[_zoom][_test]`
  - `F_hessian_plane_top1_top2_centre_frozen_seed0`
  - `F_curvature_scales_r_units`
  - `F_sensitivity_saved_appendix`

Standalone captions and source mappings are in `FIGURES.md` inside the figures archive.

## 6. Idriss's branch

- **New since `4179584` (at `1a60649`):**
  - a BN-policy 2×2 (48 measurements; 4 methods × 3 seeds × {saved, train-mode batch statistics} × {train, test});
  - `docs/CROSS_STUDY.md`;
  - a corrected input-side linearity table.
- **Direction agrees:** all 36 method-versus-control trace contrasts are negative, as ours are.
- **Not compatible with the primary cohort:**
  - different trained checkpoints (hashes unavailable);
  - all learnable parameters;
  - 5k/2k images;
  - ordinary coordinates only.
- **Mislabelled column.** Its "recalibrated" column is the Hessian through train-mode per-batch statistics. That is a third object, neither centre-frozen nor pointwise recalibration of running buffers. CROSS_STUDY's statement that "recalibration does not flip the curvature" should be read in that sense only.
- **Still valid:** his input-derivative contribution is unaffected and stays separately identified (`tab_geometry_studies.tex`).

## 7. Limitations

- **Evaluation scope.** 1k-image probes; float32 HVPs; final checkpoints only; one dataset, architecture and optimizer.
- **Surfaces.** One seed, one direction pair. The four panels do not show one physical plane: each model scales the shared draws by its own filter norms. The Hessian-plane eigenvectors are model-specific and not orthogonal in weight space.
- **Hessian scope.** No Hessian of the pointwise-recalibrated or train-mode function. Traces are signed, with negative eigenvalues present.
- **Kink interpretation.** The account of the random-direction finite-difference offset (§3.3) is not separately tested.
- **Selection.** The historical methods were selected with CIFAR-10 test exposure.

## 8. Attachments

1. `GEOMETRY_FINAL_REPORT.md` (this file)
2. `geometry_results.zip`: tidy tables, per-draw traces and per-block q, paired comparisons, surface and plane vertices, checks, raw job outputs with manifest
3. `geometry_figures.zip`: PDF + PNG, `FIGURES.md` captions and sources
4. `geometry_reproducibility.zip`: code, protocol, inventory, task bundle, environment, regeneration instructions, input hashes and locations
5. `geometry_paper_handoff.zip`: primary and appendix LaTeX tables, number macros, proposed insertions
