# Comparison with Curriculum By Smoothing and SDPoint

**Date.** 2026-09-17.
**Branch.** `comparison-cbs-sdpoint`, started from `continuation-core-optimizer-benchmark@ac23aa7`; worktree
`Projet_filiere-worktrees/cbs-sdpoint`, committed locally, not pushed.
**Manuscript.** Not edited.
**Scope.** CIFAR-10 (50,000 training / 10,000 test images) with our ResNet-20 with BatchNorm and ReLU, no augmentation,
30 epochs (11,730 updates), seeds 0–2, two recipes:

- SGD, lr 0.005;
- AdamW, lr 0.02, decoupled weight decay 5e-4.

This compares specified procedures under our recipes. It is not a reproduction of either paper.

## 1. What was run, reused and rerun

| cells | decision | evidence |
|---|---|---|
| SGD × {Plain, R, G, RG} × 3 seeds (12) | **reused**: landscape_v2 final checkpoints | init/data-order/subset hashes equal the pinned assets; config differs only in paths and checkpoint cadence; training code identical (`git diff` empty); each checkpoint's saved-statistics test result reproduced its record (≤ 0.01 pp, ≤ 6e-6 nats) |
| AdamW × {Plain, R, G, RG} × 3 seeds (12) | **rerun** | the historical grid kept no checkpoint, metrics or summary on any account we control, only job summaries, so it could not be verified or recalibrated. Reruns reproduce the historical accuracies to a mean \|Δ\| of 0.34 pp (max 0.89) (`results/adamw_historical_reproduction.csv`) |
| {SGD, AdamW} × {CBS published, CBS budget-matched, SDPoint} × 3 seeds (18) | **new** | — |

**Completion.**

- 42/42 cells complete, with 42 distinct final checkpoints, all at update 11,730.
- Every native-endpoint checkpoint reproduces its run's recorded final saved-statistics test accuracy and CE.
- A saved-statistics evaluation repeated after all recalibrations is identical to the first for every endpoint, and learned tensors are hash-unchanged.
- 5 shards, 421 output files, verified against their manifests.
- No failure, no exclusion, no relaunch.

**Accounts.**

- `maxmonstre` was left to the geometry job.
- `maxfrrsava`'s pilot kernel was granted no GPU (torch 2.10.0+cpu; aborted in 5 s), so its SGD seed-2 shard moved to `maxlebossdu91`.
- Five accounts ran the matrix: maximemonstrenikez (SGD s0), maxlebossdu91 (SGD s1–2), maxnicaise, maxlefrr and maxnikezz (AdamW s0/s1/s2).

**Protocol freeze.** `protocol/FROZEN_PROTOCOL.json` was frozen at 10:52:43 UTC, before any main run. It holds the configs, schedules, reuse audit, evaluation rules, predeclared comparisons and T4 pilot, and was committed as `b188f98`.

## 2. Implementations (details in METHOD_MAPPING.md)

**CBS.**

- Author 3×3 kernel (`utils.get_gaussian_filter`, pairlab/CBS@5f62e7d), zero padding, depthwise.
- Placed between each of the 19 main-path convolutions and their BatchNorm.
- Verified bitwise against the author function and module.
- **Published schedule:** σ = 0.9^⌊e/5⌋, native inference with σ = 0.59049.
- **Budget-matched:** 22 update plateaus until u_off = 8,211, then exact bypass. This is our predetermined adaptation, not an author schedule.
- The best-test-accuracy checkpoint rule of the released solver was not used.

**SDPoint.**

- Per logical batch: p ~ U{0..9} and ratio ~ U{0.5, 0.75}.
- Adaptive average pooling after the residual addition and before the final ReLU of block p.
- **Two released-code defects corrected** (both reproduced on the author code):
  - BasicBlocks are never selected;
  - a randomly drawn 0 becomes `None`.
- Full-resolution inference instance, fixed in advance.

**Pre-training checks passed:**

- identical parameter keys, shapes and initialization;
- no RNG consumption;
- identity states reproduce Plain logits and gradients exactly;
- CBS placement, kernels and all schedule boundaries;
- SDPoint reaches all 9 blocks with both ratios, with valid shapes and finite gradients;
- χ² uniformity and independence over 11,730 updates;
- one state per logical batch across 4 microbatches;
- bitwise resumption.

The comparator tests pass locally and on Kaggle. The five Kaggle failures are unrelated tests whose files (`assets/`, `scripts/`) were not in the kernel payload.

## 3. Results

In the results tables, ± is the sample SD over 3 seeds, and Δ is the seed-paired difference: mean ± SD of the three differences, with the number of positive differences in brackets.

### Panel A (primary): native inference network, BN recalibrated on all 50,000 training images

| procedure | SGD test acc (%) | Δ vs Plain (pp) | AdamW test acc (%) | Δ vs Plain (pp) |
|---|---|---|---|---|
| Plain | 75.46 ± 0.85 | — | 84.28 ± 0.20 | — |
| R | 80.46 ± 0.51 | +5.00 ± 0.76 (3/3) | 86.65 ± 0.14 | +2.37 ± 0.29 (3/3) |
| G | 79.94 ± 0.03 | +4.48 ± 0.88 (3/3) | 83.75 ± 0.23 | −0.53 ± 0.31 (0/3) |
| RG | 81.63 ± 0.27 | +6.17 ± 0.76 (3/3) | 86.24 ± 0.36 | +1.96 ± 0.41 (3/3) |
| CBS, published schedule | 78.01 ± 0.61 | +2.55 ± 0.44 (3/3) | 80.56 ± 0.24 | −3.72 ± 0.18 (0/3) |
| CBS, budget-matched | 78.71 ± 0.90 | +3.25 ± 0.69 (3/3) | 83.23 ± 0.38 | −1.05 ± 0.52 (0/3) |
| SDPoint (full resolution) | 79.83 ± 0.34 | +4.37 ± 0.70 (3/3) | **87.57 ± 0.59** | **+3.29 ± 0.51 (3/3)** |

Predeclared comparisons (Panel A, test accuracy in pp, test CE in nats):

| comparison | SGD Δacc | SGD ΔCE | AdamW Δacc | AdamW ΔCE |
|---|---|---|---|---|
| G − CBS published | +1.93 ± 0.63 (3/3) | −0.054 | +3.20 ± 0.17 (3/3) | −0.002 |
| G − CBS budget-matched | +1.23 ± 0.91 (3/3) | −0.068 | +0.53 ± 0.23 (3/3) | −0.197 |
| R − SDPoint | +0.63 ± 0.19 (3/3) | −0.013 | **−0.92 ± 0.53 (0/3)** | +0.310 |
| RG − R | +1.17 ± 0.24 (3/3) | −0.033 | −0.41 ± 0.49 (0/3) | −0.056 |
| CBS budget-matched − published | +0.70 ± 0.36 (3/3) | +0.014 | +2.67 ± 0.40 (3/3) | +0.195 |

### Panel B: every added intervention removed, same recalibration

Panel B equals Panel A for every procedure except published-schedule CBS:

| setting | native, recalibrated | filters removed, recalibrated | filters removed, saved BN |
|---|---|---|---|
| SGD | 78.01 ± 0.61 | 44.34 ± 2.30 | 11.67 ± 1.37 |
| AdamW | 80.56 ± 0.24 | 72.61 ± 0.51 | 33.28 ± 1.56 |

Removing the final σ = 0.59 filters is not a benign transfer to the plain network. Panel B does not replace the native result.

### Saved-statistics diagnostic (native endpoint)

- **Within one percentage point of Panel A for every procedure except SDPoint.**
- **SDPoint:**
  - SGD 77.92 ± 1.17 against 79.83 recalibrated;
  - AdamW 85.95 ± 0.69 against 87.57;
  - its running statistics mix instances, as the paper's Sec. 5.1 anticipates;
  - under saved statistics, R − SDPoint is +2.46 (SGD) and +0.65 (AdamW, 2/3 positive).
- **Policy choice changes that comparison:** instance-specific statistics are essential for SDPoint.

### Training set (50,000 images, Panel A accuracy)

| setting | procedures |
|---|---|
| SGD | Plain 94.45, CBS budget-matched 94.93, G 91.60, R 90.57, RG 89.97, CBS published 87.87, SDPoint 84.21 |
| AdamW | Plain, R and CBS budget-matched 100.00; RG 99.98; G 99.62; SDPoint 96.45; CBS published 96.25 |

- **SDPoint under AdamW** fits the training set less than Plain, R or budget-matched CBS (96.45 against 100.00), yet has the lowest test CE (0.386 against 0.953 for Plain).
- The 500-image probe and the online training loss are in `results/training_curves.csv` and are not mixed with these full-set values.

### Measured cost (runs trained here, one Tesla T4, float32, one run per GPU)

Training-loop minutes, excluding per-epoch evaluation, including checkpoint writes:

| procedure | AdamW | SGD |
|---|---|---|
| Plain | 7.6 | — |
| R | 7.5 | — |
| G | 9.6 | — |
| RG | 11.3 | — |
| CBS published | 8.9 | 9.0 |
| CBS budget-matched | 8.6 | 8.7 |
| SDPoint | 7.4 | 7.6 |

- **Other costs:** per-epoch evaluations add 0.5–0.7 min. Final recalibration and full-set evaluation takes 0.23–0.24 min (0.43–0.44 min for published CBS, which needs two states).
- **SGD Plain/R/G/RG** were not retrained. Their recorded times come from another session with more checkpoint writes, so no SGD speed comparison with them is made.
- **Total compute:**
  - kernels: 9,504 s wall × 2 T4 ≈ 5.3 allocated GPU-hours, 4.8 busy;
  - pilots: 143 s.

## 4. What the results support

- **Under SGD**, every procedure beats Plain in all seeds.
  - G is ahead of both CBS arms (+1.23 and +1.93 pp) and R is ahead of full-resolution SDPoint (+0.63 pp), all 3/3.
  - RG exceeds R (+1.17, 3/3).
- **Under AdamW** the picture changes:
  - SDPoint is the best procedure (87.57 %), above R in all three seeds (+0.92 pp) with much lower test CE;
  - R and RG still beat Plain (3/3);
  - G, both CBS arms and RG − R are negative in all three seeds.
- **Gaussian arms.**
  - Our G is ahead of CBS in both recipes.
  - Under AdamW, none of the Gaussian-only procedures (G and both CBS arms) beats Plain; RG, which adds the Gaussian to the resolution schedule, does (+1.96) but trails R.
  - The budget-matched CBS schedule (unfiltered end) is better than the published one in both recipes (+0.70 and +2.67 pp).
- **Published-schedule CBS** in our 30-epoch setting keeps a strong filter at the end: removing it destroys accuracy (§3 Panel B).

## 5. Limits of interpretation

- **G's origin.** Our G is a CBS-inspired variant, not an independent invention of Gaussian curricula. These are comparisons of specified procedures under our two recipes.
- **CBS.** A weak 30-epoch CBS result on ResNet-20 does not show CBS is ineffective in general. The paper used longer schedules, other backbones and another optimizer recipe.
- **SDPoint.** Only the full-resolution instance was evaluated. Its cost–accuracy trade-off over reduced-cost instances, the paper's main use, is outside this comparison.
- **No isolation of factors.** Placement, pooling operator (max/average), schedule and stochasticity are not isolated.
- **Seeds.** Three seeds: differences of a few tenths of a point (e.g. RG − R under AdamW, −0.41 ± 0.49) do not establish a ranking.
- **Selection exposure.** Our historical procedures and the AdamW learning rate (0.02, chosen on Plain) were selected with CIFAR-10 test exposure. The comparators received no tuning, and this comparison does not undo that asymmetry.
- **BN protocol.** The shared recalibration protocol (reset, cumulative average, all training images, no augmentation, batch 500, one pass) is our implementation of instance-specific BN. It differs from the released SDPoint `validate` loop, which uses running-statistic updates over the augmented training loader.
- **Adverse results.** SDPoint outperforming our methods under AdamW, and our G underperforming Plain there, are valid results.

## 6. Contents of the handoff

| path | content |
|---|---|
| `COMPARISON_REPORT.md`, `METHOD_MAPPING.md`, `CLAIMS.md` | report, method mapping (paper vs code vs port vs adaptation), claim-to-source index |
| `protocol/` | frozen protocol, 30 configs, schedules with every transition, reuse audit, pilot environments |
| `results/` | `per_run.csv` (42 rows: every panel, full-set accuracy/CE, timing, checks, checkpoint sha256), `summary_by_arm.csv`, `paired_comparisons.csv` (with per-seed differences), `training_curves.csv`, `adamw_historical_reproduction.csv`, `collection_checks.json` |
| `figures/` | C1 accuracy (Panel A with CBS Panel B), C2 CBS native vs unfiltered, C3 measured training time: PDF + PNG + `CAPTIONS.json` |
| `paper/` | LaTeX tables: primary Panel A, predeclared paired comparisons, all policies, costs |
| `raw/` | shard outputs: per-cell config, metrics, summary, panels, recorded checks, timing, recalibrated buffers, **final checkpoints**; manifests, environments, logs, launch and verify records |
| `code/` | `comparison/` (matrix, panels, job, launcher, freeze, analysis, packaging, verbatim author code), `continuation_core/` as run, `tests/test_comparators.py` |

**Retrieval.**

- The 12 reused SGD checkpoints are in `raw/` (the `maximemonstrenikez` and `maxlebossdu91` shard trees) and in the `visualization` worktree under `studies/landscape_v2/raw/`.
- Every checkpoint's sha256 is in `results/per_run.csv`.
- Kaggle kernels `<account>/cbs-sdpoint-main-<account>` hold the original outputs.

**Regeneration.**

```bash
py -m comparison.analyze
```

```bash
py -m comparison.package --dest <folder>
```
