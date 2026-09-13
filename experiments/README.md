# Experiment index

`index.json` is the machine-readable record of every experiment in this
repository and on the teammates' branches it references. It is **generated**;
do not edit it by hand.

```bash
py scripts/build_experiment_index.py      # rebuild index.json + assets/*.json
py scripts/benchmark_table.py             # regenerate docs/BENCHMARK_TABLE.md
```

The builder only reads. Records on other branches are read from pinned commits
with `git cat-file`, so nothing is copied into this tree. Run
`git fetch origin` first so those commits exist locally.

## Schema (version 1)

Top level:

| key | content |
|---|---|
| `built_from_commit` | commit of this tree when the index was built |
| `pinned_refs` | teammates' branches and the exact commits read |
| `asset_sets` | pinned-asset manifests, identified by content digests |
| `reference_recipe` | the shared CIFAR-10 / ResNet-20 training recipe |
| `experiments[]` | one entry per experiment (a batch or a pilot series) |
| `configurations[]` | one entry per arm **under one set of run conditions** |
| `cells[]` | one entry per trained run (arm x seed) |
| `observations[]` | cross-experiment facts, stated without causal claims |

`experiments[]`: `id`, `title`, `kind` (`training_benchmark`,
`training_pilot`, `preview_no_training`, `analysis_no_training`,
`infrastructure_probe`), `owner`, `knowledge_base_record`, `conditions`,
`reference_recipe` (null when the recipe differs), `code`, `timing_scope`,
`asset_sets`, `counts` (attempted per job manifests, with summary, numerically
valid, diverged, failure markers, configurations), and `groups[]`. Each group
is one output directory with `location {ref, path}`, `introduced_by_commit`,
`kaggle_kernel`, `code_commit_estimate`, `assets` (evidence and match),
`cells_attempted`, `missing_from_manifest`, `failure_markers`, and a note on
where checkpoints live.

`configurations[]`: `experiment`, `config_id`, `label`, `run_conditions`,
`seeds_attempted`, `seeds_valid`, `seeds_diverged`, `status` (`valid`,
`partially_valid`, `diverged`), `acc_mean`, `acc_sd`, `sd_status`
(why SD is null), `acc_per_seed`, `ce_mean`, `ce_sd`, `split`, `final_path`,
`asset_sets`, `wall_seconds_mean`, `flags` (controls whose final inference path
is not plain ResNet-20, budget differences), and `controller`.

`cells[]`: identity (`cell_id`, `config_id`, `seed`, `experiment`, `group`),
`status`, `numerically_valid`, `all_recorded_ce_finite`, record locations
(`summary`, `metrics`), sizes (`n_train`, `n_eval`, `split`, `epochs`,
`updates`), `final_acc`, `final_ce`, `summary_final_path`,
`last_record_acc_by_path`, `current_equals_target_at_end`, `state_at_end`,
timing (`wall_seconds`, `train_seconds`, `eval_seconds`), `peak_mem_mib`,
`device`, `asset_set`.

## Conventions

* **Final metric** = the last record of the run. No best-epoch selection.
* **Paths.** `current` = the state the last completed updates used;
  `target` = forced full resolution with every annealed operator bypassed.
  `summary_final_path` says which one the summary's final number equals.
  `current=target` means both are equal at the end.
* **Validity.** A cell is numerically valid when its final CE and every
  recorded CE are finite. Diverged cells stay in the index and are excluded
  from means.
* **SD.** Sample SD over valid seeds. With one valid seed it is `null` and
  `sd_status` says so; it is never reported as 0.
* **Pairing** is a property of asset digests, recorded per group. Two
  experiments are paired only where their digests match component by
  component.
