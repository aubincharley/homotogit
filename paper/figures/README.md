# Figure manifest

Every figure in the paper is generated from the tidy record tables in `data/records/` by
`tools/build_assets.py`, or, for the three historical exploration figures, from the audited
experiment index by `tools/make_legacy_assets_with_repo.py`. Both write a vector PDF and a PNG
preview. No image-generation model, hand-drawn curve or interpolated grid value is used. The compact 3D surfaces join all 41-by-41 measured vertices without fitting or smoothing.

In the newly generated diagnostic plots, Plain is dark grey, Resolution blue, Gaussian orange and Combined purple; the historical figures retain their audited palette. In diagnostic curves, markers sit on measured values, and a line only joins measured values; a band is
$\pm1$ sample SD across *training seeds*, never a confidence interval; logarithmic axes are named in
the axis label; panels that do not share a vertical scale say so in the caption.

| Asset | Source table | Reading |
|---|---|---|
| `sensitivity_summary.pdf` | `landscape_sensitivity_paired.csv` | Main figure: paired $\Delta S$ against plain, saved and pointwise-recalibrated statistics, both 1,000-image probes. Five seed-level differences per point, each already averaged over 20 directions and both signs. Panels share the horizontal scale only. |
| `sensitivity_centre_frozen.pdf` | same | The third BatchNorm convention, same definitions and layout. |
| `landscape_compact.pdf` | `landscape_random_planes.csv` | Main-text four-panel 3D surfaces: all 41-by-41 measured seed-0 vertices, own-centre CE subtraction, train probe, pointwise BN recalibration. One common camera, horizontal and vertical limits, and colour scale. Every grid vertex is used (stride 1); no fitted values. The same seed 0 as in the preceding map version is retained. |
| `random_surfaces_wide.pdf` | `landscape_random_planes.csv` | The four seed-0 models on one 41-by-41 measured plane, $[-0.5,0.5]^2$, pointwise recalibration, train probe. Heights are CE minus that model's own centre CE; common camera, colour scale and vertical limits. Retained as a larger source asset; its rendering uses a stride of two through the measured grid. An illustrative plane, not a five-seed result. |
| `interpolation_summary.pdf` | `landscape_interpolation.csv` | Four pairs of trained solutions, five seeds, 51 measured points per segment, both probes, pointwise recalibration. The horizontal axis is weight interpolation, not training time. |
| `input_diagnostics.pdf` | `input_diagnostics_frequency.csv`, `input_diagnostics_cells.csv` | The separate 28-run grid, reference group: input-Jacobian energy by spatial frequency, and finite logit displacement along each image's top singular direction, unclipped. Three seeds. |
| `appendix_resolution.pdf` | experiment index | Historical resolution-only batch: reduction location and operator, three seeds. |
| `appendix_unified.pdf` | experiment index | Historical selection batch: all 16 configurations, accuracy and per-seed paired difference. |
| `appendix_unified_curves.pdf` | experiment index | Selection-batch training records in Appendix B; the target-path panel is a diagnostic that mixes intervention removal with a statistics mismatch. |

The `archive_*.pdf` files are exhaustive historical views kept for reference and are not included in
the paper; their provenance is in `provenance/generated_assets.json`.

A colour or axis choice must never hide a sign reversal: the recalibrated panels of
`sensitivity_summary.pdf` keep positive and negative differences on one axis, and the surface figure
keeps one colour scale for all four models.
