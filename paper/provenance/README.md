# Provenance

`generated_assets.json` is written by `paper/tools/make_paper_assets.py`. For each
generated figure and table it lists:

- the experiments and configuration ids used, with valid and attempted seeds;
- every cell id with its `summary.json` / `metrics.json` location, as
  `branch@commit:path`;
- extra sources (for example `continuation_core` modules for the schedule table).

The top level records the generator command, the UTC time, the manuscript `HEAD`
at generation (and whether the generator or `continuation_core` had uncommitted
changes), the benchmark branch and commit of the index, software versions, and the
plotting conventions (SD bands, pairing, record indexing).

Hand-written tables (`tables/methods.tex`, `tables/transfer.tex`) carry their
sources in comments. When transfer, theory or visualization results are added, add
their run ids, commits, seeds and commands here.
