# Continuation methods for training CNNs — working paper

AISTATS-style working paper of the project, on branch `manuscript`. It uses the
AISTATS 2026 style files with the `preprint` option: a research-project manuscript,
not a submission.

The paper presents completed work only: the method definitions and the CIFAR-10
exploration that led to the three retained methods. Unfinished parts (transfer,
theory, visualization, discussion) are marked by red notes; their plans are in
`handover/FUTURE_WORK_NOTES.md`.

## Build

* Overleaf: upload this directory as a ZIP, select `main.tex`, compiler **pdfLaTeX**.
* Locally, from `paper/`:

  ```bash
  latexmk -pdf main.tex
  ```

* The exhaustive record of the exploration (complete configuration catalogue,
  asset hashes, campaign-wide figures) is a separate document, from `paper/archive/`:

  ```bash
  latexmk -pdf exploration_archive.tex
  ```

The two official `.sty` files are unchanged. Do not edit them or compress margins,
fonts or spacing.

## Generated figures, tables and numbers

Every number in the paper and the archive comes from records:

```bash
py paper/tools/make_paper_assets.py
```

It reads `experiments/index.json` and the run records of `benchmark-organized` at
the pinned commit `8d353f2` (and the ablation at `591e125`) through `git show`, plus
the frozen presets of `continuation_core`, and asserts that they agree. It needs
`matplotlib`, `numpy` and `torch`.

| Output | Used in |
|---|---|
| `tables/numbers.tex` (all numbers quoted in the text, exploration included) | paper |
| `tables/unified_reference.tex`, `tables/schedules.tex` | paper |
| `figures/appendix_resolution.pdf`, `appendix_unified.pdf`, `appendix_unified_curves.pdf` | paper |
| `tables/asset_sets.tex`, `plain_arms.tex`, `resolution_operators.tex`, `all_exploratory.tex`, `figures/archive_*.pdf` | archive |
| `provenance/generated_assets.json` | configurations, seeds, cell ids and files behind each output |

Generated files start with `% GENERATED`; edit the generator, not the output.

## Structure

| File | Content |
|---|---|
| `sections/01`–`07` | Main text (Max's wording; red notes for unfinished sections) |
| `appendices/a_reference_protocol.tex` | Recipe, conventions, sites, schedules, operators |
| `appendices/b_exploratory_results.tex` | CIFAR-10 exploration in four decisions |
| `appendices/c_other_approaches.tex` | TV budgets vs reconstruction operators, other directions, limitations |
| `archive/exploration_archive.tex` | Exhaustive catalogue and campaign figures |
| `handover/SOURCES_AND_PENDING.md` | Verification, corrections, pending team work |
| `handover/FUTURE_WORK_NOTES.md` | Plans for transfer, theory, visualization |
