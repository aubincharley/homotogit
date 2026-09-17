# Longer training, augmentation and BatchNorm evaluation: 63 new runs

CIFAR-10 (50,000 train / 10,000 test), ResNet-20 with BatchNorm and ReLU, cross-entropy, float32, three
seeds per cell. Branch `overnight-long-aug-bn`, based on the audited comparison `comparison-cbs-sdpoint@6b28489`.
Protocol frozen at 2026-09-17 16:39 UTC, before any run of this study existed
(`protocol/FROZEN_PROTOCOL.json`; operational fixes listed in `protocol/AMENDMENTS.md`).

## 1. Completion

| | planned | done | notes |
|---|---|---|---|
| new 160-epoch runs | 63 | **63** | 3 regimes × 7 arms × 3 seeds, every run 62,560 updates |
| BN policies on the 42 historical 30-epoch endpoints | 42 | **42** | P0/P1 reused after exact reproduction; P2/P3 new |
| recovered augmentation records (Idriss) | — | 60 | 24 at 30 epochs, 36 at 60 epochs, all reconciled |

Every new run passes every integrity check (`analysis/checks_new_runs.csv`, 63/63 true on each column):
update count, checkpoint digest, frozen configuration digest, frozen data-order digest, finite final
training loss, learned tensors unchanged by all four calibrations, P0 identical when re-run after the other
policies, and the checkpoint's saved-statistics evaluation matching the run's own final record. All 1,189
downloaded files match their manifest digests (`analysis/download_manifest_verification.csv`).

**Unresolved cells: none.** Two failed attempts are kept and labelled: the whole first launch (tag `t1`,
16:53–16:56 UTC) was rejected cell by cell by the configuration guard, because the digest included local
paths; no update was taken and no test image was evaluated. The relaunch (tag `t2`) completed everything.
`maxfrrsava` was given no GPU for the third time and was not used.

**Cost.** 48.6 T4 GPU-hours of training loop for the 63 runs (0.53–1.10 h per run), 53.0 h including all
periodic evaluation, learning-curve diagnostics and final calibration; 4.1–6.0 h wall per account, two runs
at a time (one per T4). Peak allocated GPU memory 0.53–0.83 GiB, so the physical batch of 128 was never a
constraint. Energy is not inferred from these numbers.

## 2. Main results (primary policy P1, test accuracy %, mean ± SD over 3 seeds)

| arm | SGD 0.005, 30 ep, no aug. *(hist.)* | AdamW 0.02, 30 ep, no aug. *(hist.)* | **SGD 0.1 steps, 160 ep, crop+flip** | **AdamW 0.02, 160 ep, no aug.** | **AdamW 0.02, 160 ep, crop+flip** |
|---|---|---|---|---|---|
| Plain | 75.46 ± 0.85 | 84.28 ± 0.20 | **89.66 ± 0.17** | 84.95 ± 0.26 | 88.30 ± 0.65 |
| R | 80.46 ± 0.51 | 86.65 ± 0.14 | 88.56 ± 1.28 | 85.83 ± 0.65 | 88.46 ± 0.44 |
| G | 79.94 ± 0.03 | 83.75 ± 0.23 | 85.67 ± 1.63 | 84.74 ± 0.83 | 81.18 ± 1.87 |
| RG | 81.63 ± 0.27 | 86.24 ± 0.36 | 87.17 ± 1.13 | 86.55 ± 0.32 | 85.65 ± 0.71 |
| CBS published | 78.01 ± 0.61 | 80.56 ± 0.24 | 88.85 ± 0.31 | 83.33 ± 0.47 | 86.48 ± 0.67 |
| CBS budget-matched | 78.71 ± 0.90 | 83.23 ± 0.38 | 89.16 ± 0.33 | 83.71 ± 0.31 | 86.40 ± 0.49 |
| SDPoint | 79.83 ± 0.34 | 87.57 ± 0.59 | 88.43 ± 0.43 | **88.94 ± 0.12** | 88.18 ± 0.46 |

Seed-paired differences with sign counts are in `analysis/paired_contrasts.csv`; the P1 gains over Plain:

| contrast | SGD 160 crop+flip | AdamW 160 no aug. | AdamW 160 crop+flip |
|---|---|---|---|
| R − Plain | −1.10 ± 1.39 (0/3 +) | **+0.88 ± 0.44 (3/3)** | +0.16 ± 1.01 (2/3) |
| G − Plain | −4.00 ± 1.62 (0/3) | −0.21 ± 0.59 (1/3) | −7.12 ± 1.25 (0/3) |
| RG − Plain | −2.50 ± 1.11 (0/3) | **+1.60 ± 0.32 (3/3)** | −2.65 ± 1.33 (0/3) |
| CBS published − Plain | −0.81 ± 0.42 (0/3) | −1.62 ± 0.44 (0/3) | −1.83 ± 1.30 (0/3) |
| CBS budget-matched − Plain | −0.50 ± 0.45 (0/3) | −1.24 ± 0.48 (0/3) | −1.90 ± 1.12 (0/3) |
| SDPoint − Plain | −1.23 ± 0.28 (0/3) | **+3.99 ± 0.14 (3/3)** | −0.13 ± 0.94 (2/3) |
| R − SDPoint | +0.13 ± 1.49 (2/3) | −3.11 ± 0.55 (0/3) | +0.28 ± 0.16 (3/3) |
| RG − SDPoint | −1.26 ± 1.27 (0/3) | −2.39 ± 0.30 (0/3) | −2.52 ± 0.84 (0/3) |
| RG − R | −1.39 ± 2.32 (1/3) | +0.72 ± 0.44 (3/3) | −2.81 ± 0.69 (0/3) |
| CBS bm − CBS pub | +0.31 ± 0.06 (3/3) | +0.38 ± 0.34 (2/3) | −0.07 ± 0.18 (1/3) |

### Observations

1. **Under the standard augmented SGD recipe, no intervention beats Plain.** Plain is the best arm
   (89.66 under P1; 91.49 with its own saved statistics, in line with the ~91.25 % that He et al. report for
   ResNet-20). Every other arm is below it in 3/3 seeds under P1, including both published precedents.
2. **Under long AdamW without augmentation, SDPoint wins clearly** (+3.99 pp over Plain, 3/3 seeds), ahead of
   RG (+1.60) and R (+0.88). G is flat (−0.21).
3. **Adding augmentation to the same long AdamW recipe removes what was left**: R +0.16, SDPoint −0.13,
   RG −2.65, G −7.12.
4. **The two effects are separable.** Going from the historical 30-epoch AdamW recipe to the 160-epoch one
   *without* augmentation costs R 1.49 pp of gain (3/3 seeds) and RG 0.36 pp; adding augmentation at the same
   160-epoch budget costs R a further 0.73, RG 4.25, G 6.91 and SDPoint 4.12 pp (3/3 seeds each,
   `analysis/gain_changes_across_regimes.csv`). Duration alone is not the explanation; the schedules were
   rescaled with the budget, so the 30 → 160 comparison also changes them.
5. **G is the fragile arm** in every direction tested: it is the worst arm under both augmented recipes and
   the only one that loses more than 7 pp when augmentation is added.
6. **RG > R holds only where the gain survives** (long AdamW without augmentation, 3/3); under both augmented
   recipes R is better than RG.
7. **The two CBS arms are close to each other** (|difference| ≤ 0.4 pp) and both below Plain in all three new
   regimes; the budget-matched schedule is no longer clearly better than the published one, as it was at 30
   epochs (+2.67 pp under AdamW).
8. **Accuracy and cross-entropy do not always agree**: in 14 of the 60 contrasts the signs differ. The clearest
   case is SDPoint under long augmented AdamW: 0.13 pp *below* Plain in accuracy but 0.24 nats *better* in
   cross-entropy. "Lower accuracy" is not "worse on every metric".
9. **Cost**: R is the cheapest arm (0.53 h under SGD, because it trains at reduced resolution for 40 % of the
   budget), RG the most expensive (1.08–1.10 h, ≈ +55 % over Plain). Published-schedule CBS also keeps 19
   extra depthwise convolutions at inference; after 160 epochs its σ = 0.9³¹ and those kernels are
   numerically identity, so the extra inference cost buys nothing.

## 3. BatchNorm sensitivity

`analysis/bn_policy_sensitivity_per_run.csv`, `analysis/r_rg_vs_sdpoint_across_policies.csv`,
`figures/fig_r_rg_vs_sdpoint_by_policy.pdf`.

- **Without augmentation, all four policies agree** within ≈ 0.2 pp, with one exception: SDPoint's saved
  buffers (P0) cost it 1.6–3.2 pp, because they mix the random downsampling instances seen in training. This
  reproduces the 30-epoch finding.
- **With augmentation the policies split into two groups**, by whether the statistics are estimated on
  augmented or clean images: P0 (saved, augmented) ≈ P3 (EMA over the augmented training loader) and
  P1 ≈ P2 (cumulative over clean training images). The gap is large: +1.8 to +3.5 pp for most arms under
  SGD 160, +2.0 to +3.8 under AdamW 160, and **+7.3 pp for G**. P2 − P1 stays within 0.2 pp everywhere, so
  the calibration batch size is not the factor; the calibration *data* is.
- **Consequences for rankings.** Under the augmented recipes the primary policy P1 is the least favourable
  convention for every arm, and it is the one we fixed in advance; we report it as primary regardless.
  Two orderings depend on the policy:
  - **SDPoint vs Plain under SGD 160**: −1.23 pp under P1 but **+0.37 pp under P3** (its authors' own
    convention), and SDPoint under P3 is the single best cell measured in this study (91.90 %).
  - **R vs SDPoint under long augmented AdamW**: +0.28 pp under P1/P2 (3/3 seeds) but −0.65 pp under P3
    (0/3). The sign of R − SDPoint changes with the convention in four of the five recipes.
- **RG − SDPoint is negative under every calibrated policy in every new regime**, and positive only with
  saved statistics, which is exactly the convention that penalises SDPoint.

## 4. Recovered augmentation records (Idriss)

`recovered_augmentation/`. His branch `continuation_new_loss@badce2a` (pushed 2026-09-17 18:01 local) now
carries **all 24** 30-epoch records — the export we built earlier that day came from commit `f0daed0`, which
had only 12 — **plus 36 new 60-epoch records**, including a no-augmentation control at the same budget.
Every record reconciles with the shard JSON written by the same kernel, with `results/augment_report.json`
and with the tables of `docs/AUGMENTATION.md` to two decimals. His 60-epoch asset set shares our pinned
initial weights and its first 30 epochs of data order are identical to ours.

What his records establish: with crop+flip at 30 epochs the gains fall to +0.42 / +0.86 / +1.32 (R/G/RG) at
lr 0.005 and to +0.09 / −0.96 / +0.25 at lr 0.01; at 60 epochs with augmentation they are ≤ +0.52; and his
60-epoch *unaugmented* control at the same learning rate still shows +3.25 / +2.57 / +4.24. So in his grid
too the collapse follows the augmentation, not the budget. What remains unverifiable: his four `cc-aug60-*`
jobs have launch files only, four of his recipe variants have a single seed (his document presents one of
them beside three-seed rows), and his Kaggle accounts are not readable from ours, so nothing beyond the
committed records can be checked. These runs are a **separate campaign** from ours (different learning
rates, weight decay, horizon and augmentation stream) and are not pooled with the 63 runs above.

## 5. Interpretation, and its limits

- These results **narrow where our procedures help**: in this scope the gain exists for a short,
  unaugmented, small-learning-rate recipe, shrinks when the budget grows, and disappears — or reverses —
  once the standard augmented recipe is used. That is consistent with Idriss's grid and with the team-wide
  pattern that the gain tracks how weak the baseline is, but the present data do not identify a mechanism:
  a gain that disappears under a stronger recipe is compatible with several explanations, and we tested none
  of them here.
- **This is not a reproduction of CBS or SDPoint.** Shared backbone, our trainer, our budget, one dataset.
  A weak CBS result here says nothing about CBS in its own setting; SDPoint's strong showing under long
  AdamW is likewise specific to this scope.
- **Prior test-set exposure is unequal**: our four methods, their schedules and the AdamW learning rate were
  chosen earlier with the CIFAR-10 test set visible; the precedents were not tuned; the SGD recipe is the
  textbook one and was not tuned here either.
- Three seeds support descriptive comparisons. Differences below roughly the seed SD (0.2–1.9 pp depending
  on the cell) are not rankings. The SGD 160 cells have the largest seed spread for R, G and RG.
- The 160-epoch SGD recipe is an **adaptation** of He et al. §4.2 to our trainer, initialisation and
  normalisation (50k training images, per-channel standardisation, weight decay also on BN parameters), not
  an exact reproduction, and its comparison with AdamW is not an isolated optimizer comparison.

## 6. The four closing questions

**Which comparisons are settled within our tested scope?**
- Under the standard augmented SGD recipe at 160 epochs, **none of the six interventions beats Plain**
  (0/3 seeds each under P1; under the saved-statistics convention R and CBS budget-matched tie Plain within
  0.15 pp). Whatever the curriculum is worth, it is not worth anything here.
- Under long AdamW without augmentation, **SDPoint beats R and RG** (3/3 seeds, −3.11 and −2.39 pp), and
  R and RG beat Plain (3/3).
- **G is dominated** in all three new regimes.
- At 30 epochs, the historical result stands: our arms beat both precedents under SGD 0.005.

**Which conclusions depend on BatchNorm handling?** Absolute accuracies for every augmented run (P1 costs
1.8–7.3 pp relative to the saved/EMA conventions) and two orderings: SDPoint vs Plain under SGD 160, and
R vs SDPoint under augmented AdamW. Everything measured without augmentation is policy-independent except
SDPoint's saved-statistics penalty.

**Does a stronger recipe change our methods' value?** Yes, downwards. R keeps a small positive gain only in
the unaugmented long AdamW regime; RG keeps +1.6 there and loses elsewhere; G is negative wherever the
recipe is strong. Under the strongest recipe tested (SGD 0.1, crop+flip, 160 epochs, Plain at 89.7 / 91.5)
all arms cost accuracy.

**Is any essential information still missing?** For the questions asked, no cell is missing. Three gaps are
worth naming: the schedules were transported by budget fraction and never re-tuned for an augmented or
step-decayed recipe (re-tuning would be a separate, exploratory study); only one architecture and one
dataset; and Idriss's four launch-only 60-epoch jobs plus his single-seed variants remain unverifiable from
git.

## 7. Files

`analysis/` (tables, one row per run × policy), `figures/` (PDF + PNG + LaTeX tables + captions),
`protocol/` (frozen protocol, 63 configurations, schedules, allocation, amendments),
`recovered_augmentation/`, `raw/<tag>/<account>/` (kernel outputs; checkpoints are git-ignored and stay in
the Kaggle outputs — see `CHECKPOINTS.md`), `DATA_DICTIONARY.md`, `REPRODUCE.md`, `CLAIMS.md`.
