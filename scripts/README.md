# scripts/

Flat on purpose: recorded cells name their builders by import path
(`scripts.ablation_manifest`, `continuation.ablation_ops.build_from_cell`), so
moving files would break historical launchers. This file is the map.

## Launchers (one file per Kaggle job; thin configs)

| experiment | launchers | driver | manifest / operators |
|---|---|---|---|
| `unified_selected` | `job_unified.py`, `job_unified_{0..3}.py` | `unified_driver.py` | `unified_manifest.py`, `continuation/ablation_ops.py` |
| `resbench_resolution_only` | `job_resbench.py`, `job_resbench_{0..2}.py` | `resbench_driver.py` | `resbench_manifest.py`, `continuation/resolution_ops.py` |
| `campaign_grid21` | `job_campaign.py`, `job_campaign_{0..2}.py` | `campaign_driver.py` | `campaign_manifest.py`, `continuation/campaign_ops.py` |
| `ablation_aa` (Idriss) | on branch `continuation-gaussian-tv_exploration_1` | `campaign_driver.py` | `ablation_manifest.py`, `ablation2_manifest.py`, `ablation3_manifest.py` (ported copies here) |
| `adaptive_continuation` (Aubin) | `job_adaptive_*.py`, `job_gradnorm_calibration.py`, `job_allocation_search.py`, `job_long_convergence.py` | `campaign_driver.py` | `continuation/adaptive.py`, `continuation/gap_trigger.py` |
| `per_layer_sigma` (Aubin) | `job_per_layer_gpu.py`, `job_per_layer_gpu_smoke.py`, `study_per_layer_cpu.py` | same files | `normalize_per_layer_outputs.py` |
| pilots (`fulldata`, `progres`, `db2`, `resnet18/20bn`, GN audit) | `job_fulldata_campaign.py`, `job_progressive_resolution.py`, `job_db2_pilot.py`, `job_resnet18_gaussian.py`, `job_resnet20bn_gaussian.py`, `job_plain_study.py`, `job_gaussian_study.py`, `job_lr_diagnostic.py`, `job_lr_control_002.py`, `kaggle_pilot_continuation.py` | `continuation_driver.py`, `_study_common.py` | `continuation/transforms/*` |

Infrastructure: `kaggle_run.py` (packages, pushes, polls, downloads; see
`docs/kaggle_cli.md`), `kaggle_probe_env.py` (accelerator probe),
`stage_campaign_assets.py` (publishes the pinned asset set per account).

## Verification (CPU, no training)

`verify_unified.py`, `verify_resolution_ops.py`, `verify_campaign_ops.py`,
`verify_progressive_resolution.py`, `verify_db2_operator.py`,
`verify_grad_accumulation.py`, `audit_gaussian_placement.py`.

## Records, aggregation and tables

| script | reads | writes |
|---|---|---|
| `build_experiment_index.py` | every record, including teammates' branches through `records.py` | `experiments/index.json`, `experiments/assets/*.json` |
| `benchmark_table.py` | the index | `docs/BENCHMARK_TABLE.md` |
| `analyze_unified.py`, `analyze_resbench.py`, `analyze_campaign.py`, `aggregate_runs.py` | one batch | `results/*_summary.json`, `results/*_results.json` |
| `gather_all_methods.py` | the index + metrics | `results/all_methods.json` |

## Figures

`plot_all_methods.py` (figures 24–25, all methods), `make_presentation.py` +
`presentation_labels.py` (01–13, French), `make_presentation_ablation.py`
(14–17), `plot_resbench.py` (18–23), `plot_progressive_resolution.py`,
`plot_db2_pilot.py`, `plot_loss_landscape.py`, `loss_landscape_blur.py`
(Aubin). Plain-English labels: `method_labels.py`.

## Previews (no training)

`tv_previews.py`, `wavelet_previews.py`, `wavelet_benchmark.py`,
`wavelet_optimized_benchmark.py`, `wavelet_profile.py`, `resbench_previews.py`.

Archived entry points: [../archive/README.md](../archive/README.md).
