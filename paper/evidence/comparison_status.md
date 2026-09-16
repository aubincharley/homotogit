# Comparison with existing methods (CBS, SDPoint): status

Collected 2026-09-15 from every branch on `origin` (`aubincharley/homotogit`)
and the local worktrees.

## Status: not started

No implementation, configuration, job or result for a comparison against
Curriculum by Smoothing (CBS) or SDPoint exists on any branch or local
worktree. Searched: `git grep` for `CBS`, `SDPoint`, "curriculum by smoothing"
over all remote branches, and directory names locally.

| item | status |
|---|---|
| Frozen experiment matrix | none |
| CBS reproduction or baseline runs | **never run**. `benchmark-organized:docs/HANDOVER.md` ("CBS reference reproduction") specifies a configuration (ResNet-18, BN, physical batch 64 because gradient accumulation does not reproduce BN statistics, 200 epochs, LR 0.1 with /10 steps) and estimates 7–13 h per arm; it was not launched. |
| SDPoint implementation | none (only cited in `manuscript:paper/sections/02_related_work.tex`) |
| Result or paper-handoff directories | none exist. No location has been designated. |

## What exists and must not be presented as a comparison

- **"CBS-inspired" placement only.** The retained Gaussian methods use a
  CBS-inspired placement: feature maps after convolution, annealed. They are
  not CBS.
- **Differences recorded in the benchmark documents.** Our runs differ from
  `pairlab/CBS @ 5f62e7da` in five ways, per `HANDOVER.md` and
  `docs/research/continuation/SCIENTIFIC_APPROACH.md`:
  - CBS uses a non-separable 3×3 kernel with zero padding; ours is a
    separable 9-tap kernel with reflection padding;
  - CBS starts at σ0 = 1 and multiplies by 0.9 every 5 epochs, never reaching
    0; ours uses 3-epoch plateaus and is exactly 0 from epoch 21;
  - CBS uses ResNet-18 with projection shortcuts; ours is ResNet-20 with
    option-A shortcuts;
  - CBS uses batch 64 for 200 epochs at LR 0.1; ours uses batch 128 in
    microbatches of 32 for 30 epochs at LR 0.005;
  - normalisation and initialisation also differ.

  These documents say explicitly that it is "inspired by CBS, not an exact
  reproduction".
- **EXP-006 (ResNet-18 BN).** `benchmark-organized:docs/research/continuation/experiments/EXP-006_resnet18_bn.md`
  is a one-seed exploratory pair with an architecture close to CBS (10k
  images, 1,200 updates; +7.90 points). It is not a CBS baseline.

## Implication for the manuscript

Related work can position the methods relative to CBS and SDPoint; no
empirical comparison can be claimed.
