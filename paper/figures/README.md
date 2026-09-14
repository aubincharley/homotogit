# Figures

All figures are vector PDFs generated from the audited records by
`py paper/tools/make_paper_assets.py`. Per-figure provenance (experiments,
configuration ids, seeds, cell ids and record files at their commits) is in
`paper/provenance/generated_assets.json`. Do not edit the PDFs by hand.

## In the paper

| File | Location | Question it answers |
|---|---|---|
| `appendix_resolution.pdf` | App. B.3, `fig:resolution` | Where to reduce (max-pool location) and which operator after block 1 |
| `appendix_unified.pdf` | App. B.4, `fig:unified` | Final paired comparison of the 16 candidates, with per-seed paired differences |
| `appendix_unified_curves.pdf` | App. B.4, `fig:unified-curves` | Effect of the interventions during training: current-path accuracy, target-path diagnostic, training-probe CE (500 images), test CE |

## In the archive only (`paper/archive/exploration_archive.tex`)

| File | Content |
|---|---|
| `archive_resolution_operators.pdf` | Seven operators after block 1 and at the input |
| `archive_resolution_schedule.pdf` | Schedules for three operators; fixed-resolution controls |
| `archive_campaign.pdf` | 21-configuration grid; earlier full-data runs |
| `archive_ablation.pdf` | Ablation batch (own initial weights) |
| `archive_perlayer_adaptive.pdf` | Per-layer σ profiles; data-triggered schedules (seed 0) |

## Conventions

- Plain in black; Resolution `#0072B2`, Gaussian `#E69F00`, Combined `#7B3294`;
  other arms grey.
- Open circles are seeds, filled circles means over valid seeds, bars and bands
  ±1 sample SD (never confidence intervals). One-seed arms are annotated.
- Each panel draws the plain mean of its own batch only; no cross-batch pairing.
- The target-path panel is a diagnostic: it mixes the effect of bypassing the
  intervention with BatchNorm statistics accumulated on the current path.
