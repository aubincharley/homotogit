# Data dictionary: `overnight_results.zip`

All experiments use CIFAR-10 (50,000 training / 10,000 test images) and ResNet-20 with BN/ReLU. The loss is cross-entropy and computation is float32. Seeds 0, 1 and 2 are the replication unit, and repeated BN evaluations of one checkpoint are **not** additional seeds.

## Shared vocabulary

| column / value | meaning |
|---|---|
| `campaign` | training recipe. `historical_sgd_30`: SGD lr 0.005, 60-update warm-up then cosine, wd 5e-4, 30 epochs, no augmentation (previous comparison; the Plain/R/G/RG endpoints are reused landscape_v2 checkpoints). `historical_adamw_30`: AdamW lr 0.02, same cosine, wd 5e-4, 30 epochs, no augmentation. `sgd_standard_aug_160`: SGD lr 0.1, momentum 0.9, wd 1e-4, lr ÷10 at epochs 80 and 120, physical batch 128, pad-4 crop + flip, 160 epochs. `adamw_long_noaug_160`: historical AdamW recipe with the cosine over 160 epochs. `adamw_long_aug_160`: the same with crop + flip |
| `arm` | `plain`, `resolution_max_b1` (R), `gaussian_postrelu` (G), `resolution_max_b1_gaussian_conv` (RG), `cbs_published_schedule` (CBS-pub), `cbs_budget_matched` (CBS-bm), `sdpoint` |
| `policy` | BN evaluation at the native inference state. `P0` = saved buffers. `P1` = **primary**: reset, then a cumulative average over the 50,000 clean training images in official order, batch 500. `P2` = P1 with batch 32. `P3` = EMA pass (momentum 0.1, no reset) over a seed-specific training order with the recipe's physical batch and augmentation (author-style adaptation of SDPoint's `validate`) |
| `split` | `train` = full clean training set (50,000); `test` = full test set (10,000) |
| `acc` | accuracy in [0, 1]; `*_pct` columns are percent |
| `ce` | mean cross-entropy in nats |
| `*_diff_*_pp` | seed-paired difference `arm_a − arm_b` in **percentage points** |
| `*_sd_pp` | sample SD (ddof = 1) of the paired differences, i.e. SD of the gain, not the difference of SDs |
| `acc_n_positive / negative / zero` | number of seeds with the sign |
| `ce_n_lower / higher` | seeds where `arm_a` has lower / higher CE than `arm_b` |

## `analysis/`

| file | content |
|---|---|
| `per_run_policy_long.csv` | one row per campaign × arm × seed × policy × split. `source` says whether the value is reused (P0/P1 of the historical campaigns), newly evaluated (their P2/P3) or from a new run. Includes checkpoint and BN-buffer digests |
| `per_run_wide.csv` | the same, one row per run, columns `P<k>_<split>_<acc|ce>` |
| `summary_by_arm.csv` | mean, SD and per-seed values per campaign × policy × split × arm |
| `paired_contrasts.csv` | every predeclared contrast (each arm − Plain, R − SDPoint, RG − SDPoint, G − CBS-pub, G − CBS-bm, RG − R, CBS-bm − CBS-pub) per campaign × policy × split, for accuracy and CE |
| `r_rg_vs_sdpoint_across_policies.csv` | R − SDPoint and RG − SDPoint test-accuracy means and sign counts under P0–P3, their range and whether the sign of the mean changes |
| `bn_policy_sensitivity_per_run.csv` | P0 − P1, P2 − P1, P3 − P1 per run (test accuracy pp, test CE) |
| `gain_changes_across_regimes.csv` | each arm's gain over Plain per seed in two recipes and the paired change of the gain. Short → long AdamW (pairing: same initial state and first-30-epoch order; schedules and horizon differ). Long AdamW no-aug → aug (differs only by augmentation). Historical SGD → standard SGD (descriptive only) |
| `learning_curves_saved_bn_per_epoch.csv` | new runs, every epoch: lr of the last update, online (augmented where applicable) training loss, scheduled state, saved-buffer accuracy/CE on the fixed 500-image training probe and the test set, on the `current` (scheduled) and `target` (intervention-free) paths |
| `learning_curves_p1.csv` | new runs: P1 train/test accuracy and CE after completed epochs 40, 80, 120 (scheduled state at that point) and 160 (endpoint) |
| `costs.csv` | per new run: GPU, training-loop seconds (excludes periodic evaluation and P1-curve diagnostics; includes checkpoint writes), periodic-evaluation, diagnostic and final-policy seconds, peak allocated GPU memory, number of kernel sessions. Two runs shared each kernel, one per T4. Energy is not inferred |
| `checks_new_runs.csv` / `checks_historical_endpoints.csv` | integrity checks: update count, checkpoint digest, frozen config and data-order digests, saved-buffer re-evaluation equals the recorded final evaluation, P0 repeat identical, learned tensors unchanged, reused P0/P1 reproduced |
| `download_manifest_verification.csv` | sha256 verification of every pulled Kaggle output file |
| `status_manifest.csv` | the 63 planned runs: regime, arm, seed, priority, frozen config hash, allocated account/GPU queue, completing account and tag, GPU type, status, attempts, output location |

## `recovered_augmentation/` (Idriss, `continuation_new_loss@badce2a`)

| file | content |
|---|---|
| `recovered_augmentation_runs.csv` | 60 run records (24 at 30 epochs, 36 at 60 epochs): SGD lr 0.005 / 0.01, wd 5e-4, crop_flip or none. Historical saved-buffer evaluation, target path. `trainprobe500_*` is the fixed 500-image probe, not the full training set |
| `recovered_augmentation_coverage.csv` | method × seed coverage per setting |
| `recovered_augmentation_paired_gains.csv` | seed-paired gains over Plain, with agreement checks against `augment_report.json` and the tables of `docs/AUGMENTATION.md` |
| `recovered_augmentation_summary.json` | provenance, reconciliation flags, launch-only jobs, the 60-epoch asset check, and what is not verifiable |
