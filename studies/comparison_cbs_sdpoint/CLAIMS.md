# Claim-to-source index

All paths are relative to the handoff root. `pairs` means `results/paired_comparisons.csv`, filtered to split `test_full` unless stated otherwise.

| # | claim | source |
|---|---|---|
| 1 | 42/42 cells done; 42 distinct checkpoints at update 11,730; recorded and repeat checks pass | `results/collection_checks.json`; `results/per_run.csv` (columns `final_updates`, `recorded_check_matches`, `repeat_evaluation_identical`, `checkpoint_sha256`) |
| 2 | Shard outputs match their manifests | `raw/verify_main_<account>.json` |
| 3 | SGD Plain/R/G/RG checkpoints reused: asset hashes and recipe match | `protocol/reuse_audit.json` |
| 4 | Reused checkpoints reproduce their recorded saved-statistics test results | `raw/main/<account>/cmp/cells/sgd__*/recorded_check.json` |
| 5 | AdamW historical cells rerun (no retrievable checkpoints); reruns within mean 0.34 pp (max 0.89) of history | `protocol/reuse_audit.json` (`decision: rerun`); `results/adamw_historical_reproduction.csv` |
| 6 | Protocol frozen before main runs | `protocol/FROZEN_PROTOCOL.json` (`frozen_utc` 10:52:43Z) vs `raw/launch_main_*.json` (10:53:08Z onward) |
| 7 | maxfrrsava had no GPU | `protocol/pilot_maxfrrsava_no_gpu_environment.json` |
| 8 | Panel A accuracies and SDs (§3 first table) | `results/summary_by_arm.csv` (panel `panelA`, split `test_full`) |
| 9 | Δ vs Plain and predeclared comparisons, counts, CE differences | `pairs`, panel `panelA` |
| 10 | Published CBS without filters: 44.34 / 72.61 (recalibrated), 11.67 / 33.28 (saved) | `results/summary_by_arm.csv` (panels `panelB`, `saved_original`) |
| 11 | Saved statistics within 1 pp of Panel A except SDPoint; SDPoint 77.92 / 85.95 | `results/summary_by_arm.csv` (panel `saved_native`) |
| 12 | R − SDPoint under saved statistics: +2.46 / +0.65 | `pairs`, panel `saved_native` |
| 13 | Training-set accuracies; SDPoint AdamW test CE 0.386 vs Plain 0.953 | `results/summary_by_arm.csv` (split `train_full`; `ce_mean`) |
| 14 | Training-loop minutes, evaluation and final-evaluation minutes | `results/per_run.csv` (`train_loop_seconds`, `epoch_eval_seconds`, `final_calibration_evaluation_seconds`, `timing_scope`); `paper/tab_comparators_costs.tex` |
| 15 | Total compute 9,504 s wall per 2-GPU kernel sum; 4.8 busy GPU-h | `raw/main/<account>/cmp/job_timing.json`; `raw/main/<account>/cmp/cells/*/timing.json` |
| 16 | CBS kernel and forward equal the author code; schedule boundaries; SDPoint defects and corrections | `code/tests/test_comparators.py`; `METHOD_MAPPING.md` §3–4; `code/comparison/reference_code/` |
| 17 | Every schedule transition and σ value; SDPoint draw counts per seed | `protocol/schedules.json` |
| 18 | T4 pilot timing and memory | `protocol/FROZEN_PROTOCOL.json` (`pilot`) |
