# Figures

All included figures are vector PDFs generated from the audited records by
`py paper/tools/make_paper_assets.py`. Per-figure provenance (experiments,
configuration ids, seeds, cell ids and record files at their commits) is in
`paper/provenance/generated_assets.json`. Do not edit the PDFs by hand.

| File | Location | Content | Source records |
|---|---|---|---|
| `appendix_unified.pdf` | App. B.1, Fig. `fig:unified` | All 16 unified configurations: final accuracy and per-seed paired differences to plain | `unified_selected`, 48 runs |
| `appendix_unified_curves.pdf` | App. B.1, Fig. `fig:unified-curves` | Retained procedures: current-path test accuracy, target-path diagnostic (separate panel), training-probe CE, test CE | `metrics.json` records 0–30 of 12 runs |
| `appendix_resolution_operators.pdf` | App. B.2, Fig. `fig:resolution` | Seven operators after block 1 and at the input; perceptual divergence marked | `resbench_resolution_only` |
| `appendix_resolution_location_schedule.pdf` | App. B.2, Fig. `fig:resolution-location` | Max-pool location, schedules for three operators, fixed-resolution controls | `resbench_resolution_only` |
| `appendix_campaign.pdf` | App. B.3, Fig. `fig:campaign` | 21-configuration grid; earlier full-data runs | `campaign_grid21`, `fulldata_gaussian`, `progressive_resolution_pilot` |
| `appendix_ablation.pdf` | App. B.3, Fig. `fig:ablation` | Placement, masks, reductions, profiles, controls | `ablation_aa` at `591e125` |
| `appendix_perlayer_adaptive.pdf` | App. B.3, Fig. `fig:perlayer` | Per-layer sigma profiles; adaptive schedules (seed 0) | `per_layer_sigma`, `adaptive_continuation` |

Reserved slots, still empty (they compile as labelled boxes or pending notes):

| File | Location | Owner |
|---|---|---|
| `main_landscape.pdf` | §6, Fig. `fig:main-landscape` | Max |
| `appendix_geometry.pdf` | App. D, Additional visualizations | Max |

## Conventions

- Plain in black; retained procedures: Resolution `#0072B2` (blue), Gaussian
  `#E69F00` (orange), Combined `#7B3294` (purple); every other arm grey.
- Open circles are seeds, filled circles means over valid seeds, bars and bands
  ±1 **sample SD**. They are never called confidence intervals. One-seed arms are
  annotated and have no SD.
- Final comparisons use the native, filter-free endpoint. Controls that keep an
  intervention at inference are squares and are labelled as such.
- Learning curves label the path: current path in their own panels, the target-path
  (bypass) diagnostic in a separate panel. No dashed curves; vertical grey lines
  mark state changes and are explained in the caption. Missing records stay missing.
- Each panel draws the plain mean of its own batch only. No cross-batch offsets or
  cross-batch paired bars.
- Loss landscapes (pending) need identical coordinates and colour scales across
  states, and must state the BatchNorm policy, data subset, PCA explained variance
  and off-plane residuals.
