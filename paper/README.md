# Spatial continuation — working paper

AISTATS-style working paper of the project, on branch `manuscript` (created from
`continuation-core` at `ef5564c`). It uses the AISTATS 2026 style files with the
`preprint` option: this is a research-project manuscript, not a submission, and
no venue notice is printed. Authors and their order follow Max's progress report
of 10 September 2026.

The manuscript is deliberately unfinished: the transfer table, the theoretical
result and the loss visualizations are reserved for the team's ongoing work.

## Build

* Overleaf: upload the ZIP of this directory, select `main.tex`, compiler
  **pdfLaTeX** (BibTeX runs automatically).
* Locally, from `paper/`:

  ```bash
  latexmk -pdf main.tex
  ```

The two official `.sty` files are byte-identical to the supplied pack. Do not edit
them or compress margins, fonts or spacing. Citations are author–year (`natbib`,
`apalike`). `checklist_template.tex` is the original, unfilled checklist and is not
included; fill it honestly before setting `\includechecklisttrue`.

## Generated figures and tables

Everything numerical in the paper comes from records, never from hand-typed
values:

```bash
py paper/tools/make_paper_assets.py
```

Run it from anywhere in the repository. It reads, without checking anything out:

* `experiments/index.json` and the run records of `benchmark-organized` at the
  pinned commit `8d353f2`, and the ablation records on
  `continuation-gaussian-tv_exploration_1` at `591e125`, through `git show`
  (run `git fetch origin` first on a fresh clone);
* the frozen presets of `continuation_core` on this branch, for the schedule table.

It needs `matplotlib`, `numpy` and `torch` (for importing `continuation_core`).
It first asserts that the index, the raw `metrics.json` files and the preset
values in `continuation_core/methods.py` agree.

| Output | Content |
|---|---|
| `tables/numbers.tex` | Macros for every number quoted in the text (abstract included) |
| `tables/unified_reference.tex`, `final_losses.tex` | Retained procedures in the unified batch |
| `tables/resolution_operators.tex` | Resolution-only operator table |
| `tables/schedules.tex` | Executed schedules and effective sigma, from `continuation_core` |
| `tables/asset_sets.tex`, `plain_arms.tex` | Pairing evidence |
| `tables/all_exploratory.tex` | Complete audited record (longtable) |
| `figures/appendix_*.pdf` | Seven vector figures, see `figures/README.md` |
| `provenance/generated_assets.json` | Configurations, seeds, cell ids and record files behind each output |

Generated files start with `% GENERATED`; edit the generator, not the output.
Hand-written tables: `tables/methods.tex`, `tables/transfer.tex`.

## Where to edit

| File | Role | Owner |
|---|---|---|
| `sections/01_introduction.tex` | Motivation, scope, principal observation | Max / team |
| `sections/02_related_work.tex` | Literature organized around the project | Max / team |
| `sections/03_methods.tex` | Three fixed procedures and their differences | verified against `continuation-core` |
| `sections/04_experiments.tex`, `tables/transfer.tex` | Protocol and transfer table | Aubin / Alexandre |
| `sections/05_theory.tex`, `appendices/d_future_details.tex` (§ Theoretical details) | Statistical question; theorem pending | Idriss |
| `sections/06_visualization.tex`, `appendices/d_future_details.tex` (§ Additional visualizations) | Three geometric questions; analyses pending | Max |
| `sections/07_discussion.tex` | Provisional interpretation and limits | team |
| `appendices/a_reference_protocol.tex` | Recipe, sites, schedules, operators, pairing | generated tables + verified text |
| `appendices/b_exploratory_results.tex` | Exploration figures and complete table | generated |
| `appendices/c_other_approaches.tex` | TV previews vs reconstruction operators, other directions | team |
| `handover/SOURCES_AND_PENDING.md` | What was verified, corrections, pending items | — |

## Adding results safely

- Treat each table as a view of run-level data; extend the generator rather than
  copying numbers.
- Fill `tables/transfer.tex` only from completed, verified runs. Keep *not yet
  available*, *failed* and *not applicable* distinct. Never reuse the exploratory,
  test-selected CIFAR-10 results as transfer confirmation.
- Record run ids, commits, seeds and the generator command for every new figure or
  table in `provenance/`.
- Revisit abstract, introduction and discussion when transfer, theory or
  visualization results arrive, so completed and planned contributions stay
  distinguishable.
- Do not change the method definitions in `continuation_core` to fit the text.
