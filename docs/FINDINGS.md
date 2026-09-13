# Findings, limitations and abandoned approaches

Numbers are final test accuracy (%), mean ± sample SD over seeds 0–2 unless a
seed count is given, from [BENCHMARK_TABLE.md](BENCHMARK_TABLE.md). Exploratory;
the test set has been examined many times.

## What the evidence currently supports

1. **Internal intervention helps; input-image blur does not.** Annealed
   Gaussian input blur lost 3.45 pp (GroupNorm pilots, EXP-000). Annealed
   Gaussian filtering of activations gains: in `unified`, 79.91 ± 0.27 post-ReLU
   and 78.37 ± 0.77 at conv outputs, against plain 75.43 ± 0.76.
2. **A real resolution reduction, restored later, does at least as well as
   the Gaussian alone.** Max-pool after block 1 on a 16 → 24 → 32 schedule:
   80.37 ± 0.61 (`unified`), 80.60 ± 0.20 (`resbench`), against the best
   Gaussian-only arms at 80.08 ± 0.21 (ablation `P_postblock`) and
   79.91 ± 0.27 (`unified`). That gap is within the cross-batch spread. At D1,
   the six valid reduction operators span 79.85–80.60, so the operator choice
   there shows no clear effect; the seventh (perceptual) diverged.
3. **Combining reduction and blur helps a little more, in several
   placements.** The best 3-seed arms: block-1 max-pool + blur at conv outputs
   81.51 ± 0.22 (`unified`); block-2 max-pool + receptive-field blur profile
   81.14 ± 0.22 (Idriss, other initial weights); block-1 + post-ReLU blur
   81.00 ± 0.43. These gaps are comparable to the spread between batches on
   identical assets (AUDIT A2), so their order is not established.
4. **Controls that never return to the target are worse** than their annealed
   counterparts: fixed 16x16 after block 1 at 73.13 ± 0.54, and the
   constant-sigma arms (one seed each).
5. **Blur amplitude has an interior optimum in Aubin's sigma0 sweep**
   (per-layer rho = 1): sigma0 = 0.25 / 0.5 / 1 / 1.5 / 2 give 73.81 / 76.41 /
   77.84 / 74.55 / 70.59, plain 74.13. Amplitude and effective schedule are not
   separable there.
6. **A longer budget changes the picture.** 120-epoch single-seed runs (Aubin,
   `long-conv`, high LR): plain 86.04, plateau 87.16, adaptive gap 86.73. Gains
   measured at 30 epochs should not be read as final-accuracy gains at
   convergence.

## Limitations

* One architecture, one dataset, no augmentation. The plain baseline (~75 %) is
  far below a tuned ResNet-20.
* Three seeds at most. On identical assets, plain varies by up to 0.73 pp
  between batches.
* BatchNorm interacts with every intervention: running statistics accumulate
  under the filtered or reduced state, so mid-training target-path numbers are
  confounded.
* The reduced-resolution phase did not translate into T4 wall-time savings
  (errata C-18, C-19).
* The sites and sigma scaling of the best arms differ in more than one factor.
  None of the tables is a factorial design.

## Abandoned or closed

| approach | status | evidence |
|---|---|---|
| Gaussian input blur (fixed or annealed, warm starts) | negative | EXP-000, EXP-001 |
| db2 undecimated wavelet shrinkage | closed by decision: +0.94 pp at one seed, ~61x cost | EXP-009 |
| exact TV-L2 / TV-H^-1 budget projections | previews only, never trained; ~10^4 solver iterations per image | EXP-002 |
| identity-Gaussian mixture (Gmix), early-7 masks, geometric decay | explored in `campaign_grid21`, below the retained arms | table |
| perceptual (SSIM-style) reduction at D1 | diverged in 3 / 3 seeds | `resbench_resolution_only` |
| SoftPool, MaxBlur, L2 and H^-1 constrained reductions, bilinear inside the network | trained; no advantage over adaptive max at D1 | `resbench_resolution_only` |
| data-triggered schedules (gradient norm, transfer gap), dwell allocation | single-seed runs, not better than the fixed plateau | `adaptive_continuation` |
| per-layer sigma profiles without reduction | uniform sigma not beaten | `per_layer_sigma`, ablation Q_A* |

## Carried forward

Branch `continuation-core` freezes `resolution_max_b1` (unified `shrink_b1`),
`gaussian_postrelu` (`blur_relu`), `resolution_max_b1_gaussian_conv`
(`shrink_b1_conv`) and `plain`, as representatives of the three families.
