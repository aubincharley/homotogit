# Claim-to-source index

Every claim of `OVERNIGHT_REPORT.md`, with the file and the filter that produces it. Paths are relative to
the study directory; in `overnight_results.zip` they sit under `overnight_results/`, in
`overnight_paper_handoff.zip` the compact ones sit under `data/`.

| # | claim | source | filter |
|---|---|---|---|
| 1 | 63/63 runs completed, 62,560 updates each | `analysis/checks_new_runs.csv` | all rows, `updates_ok` |
| 2 | every integrity check passes | `analysis/checks_new_runs.csv` | all boolean columns true (63 rows) |
| 3 | all downloaded files match their digests | `analysis/download_manifest_verification.csv` | `missing = hash_mismatch = 0` |
| 4 | reused P0/P1 of the 42 historical endpoints reproduced exactly | `analysis/checks_historical_endpoints.csv` | `reuse_checks_match`, `max_abs_diff_acc_P0_P1 = 0` |
| 5 | the first launch trained nothing | `raw/t1/*/ovn/cells/*/FAILED.json`, `protocol/AMENDMENTS.md` | 63 identical guard rejections |
| 6 | 48.6 training GPU-hours, 53.0 h in total, 0.53–1.10 h per run, peak 0.53–0.83 GiB | `analysis/costs.csv` | `training_gpu_hours`, `run_wall_seconds_total`, `peak_cuda_memory_allocated_mib` |
| 7 | absolute accuracies (§2 table) | `analysis/summary_by_arm.csv` | `split = test`, `policy = P1` |
| 8 | paired gains and sign counts (§2 table) | `analysis/paired_contrasts.csv` | `split = test`, `policy = P1` |
| 9 | no arm beats Plain under SGD 160 crop+flip | `analysis/paired_contrasts.csv` | `campaign = sgd_standard_aug_160`, `contrast` ends in `- Plain`, `acc_n_positive = 0` |
| 10 | Plain reaches 91.49 % with saved statistics under SGD 160 | `analysis/summary_by_arm.csv` | `policy = P0`, `campaign = sgd_standard_aug_160`, `arm = plain` |
| 11 | SDPoint +3.99 pp under long AdamW without augmentation (3/3) | `analysis/paired_contrasts.csv` | `campaign = adamw_long_noaug_160`, `contrast = SDPoint - Plain`, `policy = P1` |
| 12 | gains shrink from 30 to 160 epochs and again with augmentation | `analysis/gain_changes_across_regimes.csv` | `policy = P1`, the three `from → to` pairs |
| 13 | G loses 7.1 pp under augmented long AdamW | `analysis/paired_contrasts.csv` | `campaign = adamw_long_aug_160`, `contrast = G - Plain` |
| 14 | the two CBS arms differ by ≤ 0.4 pp in the new regimes | `analysis/paired_contrasts.csv` | `contrast = CBS-bm - CBS-pub` |
| 15 | accuracy and CE disagree in 14 of 60 contrasts | `analysis/paired_contrasts.csv` | `policy = P1`, `split = test`: sign of `acc_diff_mean_pp` vs sign of `ce_diff_mean` |
| 16 | published CBS keeps 19 identity-valued 3×3 kernels at inference after 160 epochs | `protocol/schedules.json` | `cbs_published_schedule.native_inference_state` (σ = 0.9³¹); kernel check in `tests/test_overnight.py::test_cbs_long_schedules_integer_boundaries_and_finite_kernels` |
| 17 | without augmentation the four policies agree within ≈ 0.2 pp, except SDPoint's P0 | `analysis/bn_policy_sensitivity_per_run.csv` | campaigns `historical_*`, `adamw_long_noaug_160` |
| 18 | with augmentation P0 ≈ P3 and P1 ≈ P2, gap up to 7.3 pp | `analysis/bn_policy_sensitivity_per_run.csv` | campaigns `sgd_standard_aug_160`, `adamw_long_aug_160` |
| 19 | SDPoint vs Plain under SGD 160 flips with the policy (−1.23 pp P1, +0.37 pp P3) | `analysis/paired_contrasts.csv` | `contrast = SDPoint - Plain`, `campaign = sgd_standard_aug_160`, policies P1 and P3 |
| 20 | R − SDPoint changes sign with the policy under augmented AdamW | `analysis/r_rg_vs_sdpoint_across_policies.csv` | `campaign = adamw_long_aug_160`, `contrast = R - SDPoint` |
| 21 | RG − SDPoint is negative under every calibrated policy in the new regimes | `analysis/r_rg_vs_sdpoint_across_policies.csv` | rows `RG - SDPoint`, columns P1–P3 |
| 22 | SDPoint under P3 is the best single cell (91.90 %) | `analysis/summary_by_arm.csv` | `policy = P3`, `campaign = sgd_standard_aug_160`, `arm = sdpoint` |
| 23 | learning curves: P1 at epochs 40/80/120/160 and the per-epoch saved-buffer diagnostic | `analysis/learning_curves_p1.csv`, `analysis/learning_curves_saved_bn_per_epoch.csv` | — |
| 24 | Idriss's 60 augmentation records are complete and reconcile with his report and document | `recovered_augmentation/recovered_augmentation_summary.json` | `shard_json_all_match`, `report_json_all_match`, `doc_tables_all_match_2dp` |
| 25 | his 60-epoch no-augmentation control still gains +3.25 / +2.57 / +4.24 | `recovered_augmentation/recovered_augmentation_paired_gains.csv` | `augmentation = none`, `epochs = 60` |
| 26 | his 60-epoch asset set shares our initial weights and first 30 epochs of order | `recovered_augmentation/recovered_augmentation_summary.json` | `assets_60_epoch` |
| 27 | four of his jobs are launch-only; four variants are single-seed | `recovered_augmentation_summary.json`, `data/team` coverage table | `launch_only_jobs_without_outputs` |

Figures: `figures/fig_r_rg_vs_sdpoint_by_policy` (claims 19–21), `figures/fig_gain_vs_plain_p1` (8, 9, 13),
`figures/fig_accuracy_vs_training_cost` (6, 7), `figures/fig_learning_curves_p1_and_saved` (23).
LaTeX tables: `figures/table_absolute_test_accuracy_P1.tex` (7), `table_paired_contrasts_P1.tex` (8),
`table_r_rg_vs_sdpoint_by_policy.tex` (19–21).
