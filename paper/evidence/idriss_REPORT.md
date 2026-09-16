# Idriss's work on `continuation-core-experiments`

## Source and verification

- **Branch.** `origin/continuation-core-experiments` at `4179584`
  (2026-09-15 16:32, "Grid results: the dissociation holds, no quantity acts
  as a dial") and parent `28a91a5` (11:11), based on
  `continuation-core@ef5564c`.
- **How it was read.** Remotely via `git show`, without merging or modifying
  anything.
- **What was checked.** I read the code (`continuation_core/analysis/*.py`,
  `continuation_core/grid.py`, `tools/job_probe.py`, `tools/analyze_grid.py`),
  the configs and summaries of the 28 runs, the four probe output shards and
  `results/grid_report.json`. The method means in `docs/RESULTS_GRID.md` §1
  were recomputed from the probe files and **match to the printed precision**
  (for example tr H: plain 14,730.5, Gaussian 10,360.8, resolution 10,108.2,
  combined 8,760.8).

## What was actually done

This was **not only** input derivatives at fixed weights. The branch contains
three kinds of work.

1. **New training.** 28 runs on Kaggle T4:
   - the four frozen methods × seeds 0–2 at the reference recipe (12 cells);
   - the four methods × four one-knob recipe variants at seed 0 (16 cells):
     lr 0.0025, lr 0.01, weight decay 1e-4, weight decay 2e-3.

   That makes 20 "conditions". The variants are labelled `unvalidated`.
2. **Endpoint probes** on each epoch-30 checkpoint:
   - *function side:* input Jacobian of the logits, its frequency profile,
     and finite logit displacements;
   - *parameter side:* Hessian trace and top eigenvalues of the training loss.
3. **A statistical analysis across conditions**: the "dial test" of whether
   any probe predicts test error at equal training error.

Code for a gauge-minimal distance, PAC-Bayes quantities and a per-epoch
trajectory probe also exists. **No output files for them are on the branch.**
The numbers in `RESULTS_GRID.md` §5 (distances at epochs 6/12) and the "gauge
≤ 2–3 %" statements therefore cannot be verified from committed data. The
report itself says the per-epoch probe "was not run".

## 1. Question

Do the four procedures leave measurably different solutions, in their input
sensitivity, frequency response and loss curvature? And does any such quantity
track test error once training error is controlled?

## 2. Objects measured

Notation:

- x: an image in **[0,1] intensity units**, 3×32×32 = 3,072 inputs. The
  per-channel normalisation is **inside** the differentiated map.
- f(x) ∈ R¹⁰: the **logits**.
- θ: all learnable parameters.

All probes use `model.eval()` with the **checkpoint's BN running statistics**,
at the **target** intervention state (at epoch 30 this is the exact identity
for every method).

| object | definition | derivative variable, order | aggregation, units | data |
|---|---|---|---|---|
| **Input Jacobian** J(x) | ∂f/∂x, 10 × 3,072, exact via 10 backward passes | input, first order | ‖J‖_F and σ_max(J) per image from the 10×10 Gram matrix, **mean over images**; participation ratio (Σσ²)²/Σσ⁴. Units: logit per unit intensity. | first 1,000 images of the pinned training subset (subset order), no labels |
| **Frequency profile** | orthonormal 2-D DFT of each Jacobian row; radial energy by rounded radius k (cycles per 32 px) | — | `mean_radius` = Σ k·(energy fraction); fraction of energy at k ≥ 8. Fractions averaged over batches of 250. | same 1,000 images |
| **Amplitude curve** S(ε) | mean over images of ‖f(x + ε v₁(x)) − f(x)‖₂, with v₁(x) the unit top right singular vector of J(x) **recomputed per image**; ε ∈ {0.05, 0.1, 0.25, 0.5, 1, 2, 3, 6, 12} | finite input perturbation, **not clipped to [0,1]** | logit L2 displacement; also accuracy of the perturbed batch | same 1,000 images |
| **Linearity ratio** | S(ε) / (ε · mean σ_max) | — | 1 would mean first order holds | same |
| **Parameter Hessian** H | ∇²_θ of the **mean CE** (no weight decay) | **parameters** (all 269,722: conv, BN affine, fc weight and bias), second order | Hutchinson tr H with 64 Rademacher draws, deflated by the top 5 eigenpairs; per-block traces from the same draws; top 5 eigenvalues by power iteration (≤ 40 iterations, relative change < 1e-4) | first 5,000 images of the training subset, batches of 500 |
| **Train/test error and CE** | eval-mode evaluation | — | error rate, CE in nats | **all 50,000** training images; 10,000 test images |

These are different objects:

- **J** is an *input Jacobian of the logits* (first order).
- **No input Hessian** and no input gradient of the CE was computed.
- **H** is a *parameter* Hessian of the training loss.
- **S(ε)** is a *finite* displacement of logits along a data-dependent input
  direction.

None of them is a robustness guarantee. Because J and S are logit-based, they
avoid the direct confidence and loss-scale effects of CE-based sensitivity,
but logit scale still matters: a network with larger logit magnitudes has a
larger ‖J‖ at equal decisions. The branch documents that ‖J‖_F correlates
−0.59 with training error across conditions.

## 3. Numerical computation and verification

**Implementation.**

- Jacobian spectra use float64 Gram matrices.
- HVPs are exact double back-propagation, accumulated per batch with size
  weights.
- Directions are drawn on CPU from seeded generators.

**Validity gate.** Probed test accuracy equals the accuracy recorded at
training time in all 28 cells (maximum difference 0).

**Power iteration.** 135 of 140 top-5 eigenvalues met the 1e-4 criterion. The
five unconverged cases:

- `gaussian_postrelu__reference__seed0` #3
- `gaussian_postrelu__wd_high__seed0` #3
- `resolution_max_b1_gaussian_conv__reference__seed0` #4
- `resolution_max_b1_gaussian_conv__reference__seed2` #3
- `resolution_max_b1_gaussian_conv__wd_high__seed0` #3

The top eigenvalue converged in every cell.

**Not present.**

- No finite-difference or dense-Hessian check of the HVP.
- No Ritz residuals.
- ReLU/max-pool nonsmoothness is not discussed for H.

landscape_v3 found that finite differences disagree with the a.e. Hessian in
this network; see its report.

**Statistical guards** (author's design, `analysis/power.py`):

- partial correlations controlling training error;
- Pearson and Spearman;
- Bonferroni correction over 25 quantities;
- between/within-group decomposition (control vs curriculum cells);
- detection ceiling from measurement noise.

## 4. Results, as measured

Values are per-seed and per-cell in `idriss_per_seed.csv`.

**Reproduction of the reference runs (§0).**

- Test accuracy at seeds 0/1/2:
  - plain 74.73 / 76.51 / 76.56;
  - resolution 80.21 / 80.22 / 80.22;
  - Gaussian 79.95 / 79.98 / 79.91;
  - combined 81.30 / 81.30 / 81.45.
- Method means are within 0.5 pp of the unified-batch record.
- **The seed SDs differ strongly from that record.** The continuation methods
  have near-identical seeds (SD 0.006–0.09), while plain has SD 1.04. The
  author flags this as unexplained. Distinct seeds and initialisations were
  verified.
- This matters before any "variance" claim. Resolution-only at 80.21 / 80.22 /
  80.22 over three initialisations is unusual and should be checked by the
  owner.
- *Minor correction:* the Gaussian-only row of `RESULTS_GRID.md` §0 lists
  79.91 / 79.95 / 79.98, which is sorted, not seed order. The seed-order values
  are 79.95 / 79.98 / 79.91.

**Reference-cell means, 3 seeds (§1).**

| | tr H | ‖J‖_F | S(3) | mean radius | test error |
|---|---|---|---|---|---|
| plain | 14,730.5 | 143.2 | 12.79 | 12.08 | 0.2407 |
| resolution | 10,108.2 | 100.0 | 10.67 | 12.63 | 0.1978 |
| Gaussian | 10,360.8 | 74.8 | 11.14 | 11.01 | 0.2005 |
| combined | 8,760.8 | 78.1 | 10.41 | 11.46 | 0.1865 |

- **Lower quantities.** The continuation cells have lower tr H (−30 to −41 %),
  lower ‖J‖_F and lower S(3) than plain.
- **Frequency profile.** Gaussian-only shifts the Jacobian's frequency profile
  down (mean radius −8.8 %), while resolution-only does not (+4.6 %), at
  similar test error. The author calls this a "dissociation".
- **Linearity.** The linearity ratio is 0.32–0.47 at ε = 0.5 and 0.05–0.09 at
  ε = 3, so a first-order description does not hold at the measured scales.

**Across the 20 conditions (§4).**

- Strong pooled partial correlations with test error (‖J‖, σ_max, S@0.05 ≈
  +0.8) vanish within groups (−0.41 to +0.24, none significant).
- The author concludes that no probe predicts test error at equal training
  error inside the curriculum family, and that a +0.72 predictor from an
  earlier branch does not replicate.

## 5. Limitations and incomplete work

- **Seed replication.** Only the 12 reference cells have 3 seeds. The 16
  variant cells are single-seed, so correlations are across **conditions**,
  not seeds.
- **Probe data.** Probes use training images: the first 1,000 or 5,000 in the
  subset's stored order, which is not class-balanced by construction.
- **Scope.** Single checkpoint (epoch 30), one dataset, one architecture. The
  per-epoch trajectory probe was not run.
- **Unverifiable claims.** PAC-Bayes distance (§5) and gauge claims have no
  committed outputs.
- **Causal claims.** None are supported, as the author also states.
- **Figure.** `grid_figures.png` has French labels and no vector version.

## 6. Relation to our landscape study (not interchangeable)

| | Idriss | landscape_v2/v3 |
|---|---|---|
| runs | 28 new runs (3 seeds reference) | 20 v2 runs (5 seeds) |
| loss curvature | tr H, top-5 by power iteration; all params; 5,000 train images; checkpoint BN statistics only | extreme eigenpairs (top 2, min) by Lanczos; conv + fc weights only; 1,000-image probes; saved and centre-frozen BN; ordinary and relative coordinates |
| finite perturbations | input-space, logits, per-image top direction | weight-space, CE, filter-normalised random and Hessian directions, three BN policies |

Both report that continuation solutions have lower parameter-space
curvature/sensitivity than plain in these conventions. They measure different
objects on different runs, so the numbers must not be combined.
