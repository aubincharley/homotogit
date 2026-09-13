# Audit of derived tables, labels and claims (2026-09-13)

Raw measurements (`summary.json`, `metrics.json`, logs) were not modified.
Everything below corrects **derived** material: tables, figure captions, labels
and prose. Each item names the reproducible source. Most checks are one
command:

```bash
py scripts/build_experiment_index.py && py scripts/benchmark_table.py
```

The index keeps every location (`experiments/index.json`), so each statement
can be traced to a file.

| id | earlier claim (where) | correction | source |
|---|---|---|---|
| A1 | The unified batch used its own "asset set C" and was "the only internally paired set" (`docs/BENCHMARK_TABLE.md` at 4f3bc19, `scripts/plot_all_methods.py` caption, figure 25 subtitle, `docs/RESUME_UNIFIED.md`). | The unified jobs mounted `<account>/r20bn-campaign-assets` and verified it against its manifest. That manifest carries the digests in `results/campaign_manifest_frozen.json`, the set produced by `fulldata_gaussian`. `fulldata_gaussian`, `progressive_resolution_pilot`, `campaign_grid21`, `resbench_resolution_only`, `unified_selected` and Aubin's seed-0 adaptive runs therefore share one asset set. Unified was not uniquely paired. | index → `experiments[*].groups[*].assets`; `launch.json` → `datasets` |
| A2 | Cross-batch comparisons carry a ~0.56–0.6 pp "batch offset" (gather/plot docstrings, table, resume note). | No universal offset is supported. Plain arms on **identical** assets finish at 74.58 / 75.13 / 75.31 / 75.10 % for seed 0 in four batches; the spread is up to 0.73 pp, and the epoch-2 records already differ. The cause is not established. | index → `observations[same-assets-plain-differs]` |
| A3 | The ablation batch used "a different pinned set" (implying different data). | Its subset, probe and per-epoch permutation digests equal the campaign set's. Only the initial weights and BN buffers differ. | `experiments/assets/*.json`; `observations[ablation-assets-share-data-order]` |
| A4 | Single-seed rows were shown with SD 0.00 (`gather_all_methods._mean_sd`). | SD is *n/a (1 seed)*. | `configurations[*].sd_status` |
| A5 | The three `perceptual__D1__Rprog` cells were absent from the overview table and figures (`gather_all_methods` skipped configs with `n_seeds == 0`). | Listed as **diverged, 0 / 3 valid** (non-finite CE in every seed). | `cells[*].numerically_valid`; `results/resbench_results.json → diverged_cells` |
| A6 | Every table row was "final-epoch test accuracy on the current path" (table header at 4f3bc19). | Summaries report different paths: target (`campaign`, `unified`, progressive `resbench` arms), current for fixed-resolution controls (`resbench` `fixed16` / `fixed24`), and the declared `primary_path` for the ablation. Constant-sigma arms and `R6` report current; BlurPool reports target with the BlurPool architecture kept. The table now has a `path` column and flags these controls. | `cells[*].summary_final_path`, `configurations[*].flags` |
| A7 | "Per-layer blur strength, exponent 1: 77.65 ± 0.64, 6 seeds". | Seeds 0–2 in **two launches** (`per-layer-sigma`, `per-layer-adaptive-fix`), not six seeds, and they are now separate rows. The same study also holds a 6,000-image CPU pilot and 5,000-image smoke runs, now separated by run conditions. 42 summary files are layout copies from `scripts/normalize_per_layer_outputs.py` (same numbers) and are collapsed. | `configurations[*].run_conditions`, `groups[*].duplicate_summaries_collapsed` |
| A8 | P_A1–P_A4 were blur-only profiles; Q_A1–Q_A4 were placed "after every conv". | Already corrected in 4f3bc19: P_A* carry the block-2 max-pool reduction (Rprog); Q_A* are post-ReLU without reduction. Idriss's 81.14 % arm is therefore a reduction + blur arm. | `scripts/method_labels.py`; ablation `summary.json → reduction, placement` |
| A9 | — | `EXP-012` names two different records on two branches (Idriss's anti-aliasing ablation, Aubin's per-layer sigma). Both files are kept; the index disambiguates by branch. | `observations[kb-experiment-id-collision]` |
| A10 | The merged branch was assumed to run the historical ablation/unified code. | Merging Aubin's adaptive commits changed `SiteController.__init__` to validate schedules through `q_for`. `AblationController.q_for` reads attributes set after the parent constructor, so `build_from_cell` raised `AttributeError`. The fix sets those attributes first; no value changes. The same merge made `SiteController.value` a read-only property, which broke the save/restore in `scripts/unified_driver.py::evaluate` and `scripts/verify_unified.py`; both now restore the per-site row. `verify_unified.py` passes on the merged tree. Regression test: `tests/test_unified_controllers.py`. Behavioural parity of the fixed merged tree with the executed code: `continuation-core: verification/parity_report_merged_tree.json`. | `continuation/ablation_ops.py` |
| A11 | `tests/test_gaussian.py::test_radius_must_fit_the_image` expected an error. | Radius ≥ map size is supported by explicit reflection (errata C-04). The test now checks that path. | the test |
| A12 | Unified jobs' code commit unknown. | The shipped bundles (`<group>/_repo`, git-ignored) equal commit 3829fa3 after CRLF→LF normalisation. The launcher was committed after launch, which is why the launch-time estimate is empty. | `groups[*].shipped_code_check` |
| A13 | "Single-seed GPU nondeterminism is ~±0.1 pp" and a "measured noise floor ~±0.1 pp" (`docs/HANDOVER.md` §3, §4). | That figure came from one pair of runs (errata C-12). Same-asset plain runs later differ by up to 0.73 pp (A2). Neither number is a noise floor, and nondeterminism is a candidate cause, not an established one. Annotated in place. | A2 |
| A14 | TV previews and TV-like operators were conflated. | `tv_budget_previews` trained nothing. `resbench`'s `l2` and `hminus1` are trained constrained-quadratic *resolution-reduction* layers. The table keeps them in separate sections. | index kinds |
| A15 | Aubin's `gradnorm-cal*`, `adaptive-*` manifests declare `paired_with_campaign: false`. | Their recorded subset, probe and `init_seed0` digests match the campaign set. The index records both the declaration and the content comparison, and does not override the authors' declaration. | `groups[*].assets.declared_paired_with_campaign` vs `.match` |

## Entry points moved to `archive/`

`git mv`, history intact; the repository-root path in each file was adjusted.
See [archive/README.md](../archive/README.md).
