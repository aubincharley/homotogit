# Changes in this revision

This revision keeps the prose, structure and claims of the delivered paper. It changes how tables
and figures are built and labelled, and it replaces derived exports by the original experiment
records. Full status per item: `FINAL_SOURCE_CHECKS.md`.

## Pipeline

- Added `tools/build_records.py`: reads the original run records (`summary.json`/`metrics.json` of
  every cell of the experiment index, the transfer branches, the optimizer job summaries, the
  input-probe shards) and the raw landscape evaluation files, and writes tidy tables to
  `data/records/` with `SOURCES.json` (source commits, row counts, sha256 per table).
- Rewrote `tools/build_assets.py` to read `data/records/` only. It can no longer restore the
  previous table layouts: they are not in the file any more.
- Added `tools/build_catalogue.py`, which builds the companion exploration catalogue.
- Added `tables/numbers_generated.tex`: the numbers quoted in the prose are now generated from the
  same records as the tables.

## Tables

- **Table 2** now reports final test accuracy for Plain, R, G and RG, each as mean $\pm$ sample SD
  of its own per-seed accuracies, in dataset blocks that name the training and test population, with
  model, activation, normalization, optimizer and budget visible in every row. The within-seed
  gains, their SDs and the per-setting counts of positive seeds moved to the new
  Table~\ref{tab:transfer-paired} instead of being duplicated.
- **The protocol table** (Appendix C) gained the same explicit fields, grouped by dataset.
- **The cost table** now reports measured wall time in minutes per method, mean $\pm$ sample SD,
  including the reference batch, with the evaluation frequency in the row; the within-seed ratios
  stay in the numerical export.
- **The sensitivity table** now reports $S$ itself for the four methods, grouped by BatchNorm policy
  and evaluation population, with the amplitude and units stated; the paired differences and
  negative-seed counts moved to a companion table.
- **The Hessian, endpoint and input-diagnostic tables** kept their quantities and gained
  self-contained captions: dataset, model, optimizer, checkpoint, evaluation population, replication
  count and what the uncertainty summarizes. Table 3 keeps the within-seed ratio, with the absolute
  eigenvalues in the appendix.
- Training cross-entropies are printed to a fixed number of significant digits instead of switching
  units between rows.

## Figures

- The five new figures are regenerated from the record tables, with measured points marked,
  logarithmic axes named, one colour per method throughout, and legends moved out of the data area.
- The three historical figures were regenerated from the experiment index and reproduce the supplied
  files pixel-for-pixel.

## Corrections

- **Straight-segment barriers.** The R--RG barrier is lower than plain--R on the training probe in
  three of the five seeds (1, 2 and 4), not two, and also in one seed on the test probe. The earlier
  statement named seeds 1 and 4 only.
- **CIFAR-10 5k assets.** The training subset (5,000 distinct images), the 500-image probe, its
  membership in the subset and the inherited 50,000-image normalization are now verified from the
  regenerated index arrays; the appendix states them instead of recording the size as unverified.
- Values that the prose had rounded independently (full-test differences, Jacobian radii, trace
  estimates, residuals, PCA shares, wall times) are now taken from the generated macros.

## Proposed, separable

- The complete 165-configuration catalogue ships as `catalogue/` (readable PDF and Markdown,
  machine-readable CSV, reconstruction script, change log) and Appendix F points to it. Set
  `\catalogueinpapertrue` in `main.tex` to print it inside the paper again.
