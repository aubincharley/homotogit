All three still `RUNNING`. **\~62 minutes elapsed** of a predicted **83 minutes**, so roughly **20 minutes to go** if the estimate holds — landing around 11:15Z, comfortably inside the 2.5 h target and well clear of the 3 h ceiling at 12:51Z.

I can't see inside a running kernel — Kaggle only exposes logs and outputs once a kernel finishes, so the per-cell progress lines the jobs are printing aren't retrievable yet. What I have is status plus the schedule prediction, nothing finer.

The three background pollers will notify me the moment each job completes, and each downloads its own outputs automatically. Nothing needs doing until then.

One thing worth flagging now: the jobs finish independently, and job 0 carries 18 cells against 17 for the others, so expect it to land last. If any job overruns the window I'll pull the rolling checkpoints and report exactly which cells are incomplete rather than dropping seeds or shortening training.

**Job 0 has landed — 21 cells, 0 failures, 6,270 s.** Jobs 1 and 2 still running at 11:41Z (1h50m elapsed).

I need to correct my earlier estimate, and flag a bug of mine.

**The jobs are running 21 cells each, not 17–18 — all 63 cells, not the 52 pending ones.** `find_reusable()` scans `results/kaggle_outputs/` to detect the 11 reusable cells, but I shipped only `continuation` and `scripts` to the kernels. Inside the kernel that directory doesn't exist, so the scan found nothing, every job planned over all 63 cells, and each took 21. The job log says it plainly: `campaign job 0: 21 of 63 pending cells (0 reused campaign-wide)`.

What this does and doesn't break:

- **The partition is still correct.** All three jobs ran the same deterministic assignment over the same 63-cell list, so the three slices are disjoint and cover all 63 exactly once. Job 0's manifest has 21 unique cells. No duplicates, no gaps.
- **Nothing is unpaired.** Every job verified the pinned assets before training (`assets all_match: True`), so all 63 cells share the same initial weights, BN buffers, permutations and probe indices.
- **The cost is redundant work**, not wrong work: \~10 GPU-hours instead of 8.24, and job 0 took 6,270 s against 5,018 s predicted.
- **There's a silver lining I won't oversell**: the 11 cells I meant to reuse are now being retrained under the identical code path, which gives a genuine independent replication of the historical numbers rather than a cross-campaign splice.

**Timing.** Launched 09:51Z. Job 0 finished \~11:35Z (1h44m). If jobs 1 and 2 track it, they land \~11:40–11:50Z, i.e. **\~1h50m elapsed** — inside the 2.5 h target (12:21Z) and well clear of the 3 h ceiling (12:51Z). So the overrun costs GPU time, not the deadline.

Early signal from job 0's top results (one seed each, not yet aggregated — treat as provisional):

| cell test acc                                      |        |
| -------------------------------------------------- | ------ |
| Rprog / Gplateau / input\_bilinear / all19 / seed2 | 0.8142 |
| Rprog / Gnone / **stem\_max** / all19 / seed2      | 0.8064 |
| Rprog / Gplateau / stem\_max / all19 / seed2       | 0.8062 |
| Rprog / **Gmix** / input\_bilinear / all19 / seed2 | 0.8034 |

All above the historical plain baseline of \~0.751, and above the 0.7993 progressive+Gaussian seed-0 result. I'm not drawing conclusions from a single job's slice — the full three-seed picture, paired differences and figures come once jobs 1 and 2 land.

Job 1 landed: 21 cells, 0 failures, 7,110 s. Job 2 still running. Writing the analysis while it finishes.

Partition confirmed disjoint and complete — the 21 missing cells are exactly job 2's slice. Writing the figures.

**Two of three jobs have landed.** Analysis and figure code is written and dry-run against the partial data.

| job account cells failures elapsed  |            |    |   |               |
| ----------------------------------- | ---------- | -- | - | ------------- |
| 0                                   | maxnicaise | 21 | 0 | 6,270 s       |
| 1                                   | maxlefrr   | 21 | 0 | 7,110 s       |
| 2                                   | maxnikezz  | 21 | — | still running |

42 of 63 cells collected, and the 21 missing are **exactly** job 2's slice — confirming the three job manifests partition the campaign disjointly and completely, despite the reuse-detection bug I reported (each job planned over all 63 cells rather than 52, so all 63 are being trained fresh).

Elapsed 2h04m at 11:55Z. Job 2 started \~40 s after job 0 and carries the same 21 cells, so on jobs 0/1 timings it should land around **12:00–12:10Z, i.e. \~2h10–2h20m** — inside the 2.5 h target (12:21Z), with the 3 h ceiling at 12:51Z as the real deadline. It has a rolling checkpoint per cell, so if it did overrun I can recover unfinished work rather than losing it.

Ready to run the moment it lands: `analyze_campaign.py` (per-seed metrics, three-seed mean and sample SD, all prespecified paired differences, manifest coverage check, cumulative GPU time) and `plot_campaign.py` (three figures — final accuracy grouped A–E with individual seeds visible, trajectories one panel per group, and accuracy-versus-measured-cost — following the plot checklist, so no 21-curve tangle and every line style in the legend).

One extra check the bug made possible: the three R32 configurations were retrained from the identical pinned initial weights, so I can report the recomputed versus historical accuracies directly as a replication. I'll report those deltas descriptively and won't turn them into a noise-floor claim.

**Campaign complete. 63/63 cells, 0 failures, elapsed 2 h 08 min** (09:51Z → 11:59Z), inside the 2.5 h target. Pushed as `f191fa6`.

## Manifest status

All 63 cells were trained fresh, including the 11 I had planned to reuse — the bug I reported: `find_reusable()` scans `results/kaggle_outputs/`, which isn't shipped into the kernels, so each job planned over all 63 and took 21. The three slices still partitioned the campaign disjointly and completely. Cost: **11.13 cumulative GPU-hours** instead of \~8.2. Benefit: one internally consistent campaign, and a genuine replication of the three historical R32 configurations.

**Replication vs. the earlier campaign** (same pinned initial weights, same permutations): plain mean Δ **+0.0021** (max |Δ| 0.0036), plateau **+0.0028** (0.0091), geometric **−0.0018** (0.0074). Same configuration, same starting state, differences of this size — worth holding in mind when reading anything below that is under \~1 pp.

Jobs: j0 6,270 s / j1 7,110 s / j2 7,431 s, all three `assets_verified: True`. Evaluation was 3.3% of wall time; peak memory 512–616 MiB.

## Leading configurations (3-seed mean ± sample SD)

| configuration acc sd CE wall s  |            |        |        |     |
| ------------------------------- | ---------- | ------ | ------ | --- |
| Rprog / Gplateau                | **0.8071** | 0.0065 | 0.5682 | 700 |
| Rprog / Gplateau / input\_max   | 0.8034     | 0.0076 | 0.5720 | 738 |
| Rprog / Gnone / stem\_max       | 0.7997     | 0.0058 | 0.5894 | 491 |
| Rprog / Ggeo                    | 0.7991     | 0.0092 | 0.5918 | 709 |
| Rprog / Gnone                   | 0.7948     | 0.0054 | 0.5989 | 492 |
| R32 / Gplateau                  | 0.7869     | 0.0068 | 0.6210 | 696 |
| R32 / Gnone                     | 0.7562     | 0.0049 | 0.7552 | 489 |
| R32 / Gplateau / early7         | 0.7524     | 0.0074 | 0.7381 | 593 |

## Paired differences

**Resolution beats filtering, and the two partly substitute.** Rprog−R32 is **+3.87 pp** (sd 0.30) unfiltered but only **+2.02 pp** with Gplateau; Rgentle−R32 is +2.95 → +1.08 pp. Symmetrically, Gplateau−Gnone is **+3.08 pp** at R32 but **+1.23 pp** at Rprog and +1.21 pp at Rgentle. Every one of these is same-sign across all three seeds. The gains are not additive: whatever the Gaussian provides at full resolution is largely already supplied by training at reduced resolution.

**Gmix loses to Gplateau** at both resolutions: −0.51 pp (sd 0.33) at R32, **−1.20 pp** (sd 0.17) at Rprog, same-sign in every seed, and it is the most expensive operator (0.067–0.071 s/update). Mixing toward identity is worse than reducing the Gaussian width — though as specified, its alpha levels were never strength-matched to the sigma levels, so this compares two schedules, not two operators at matched strength.

**early7 is clearly worse than all19**: −3.46 pp (sd 1.10) at R32, −1.44 pp (sd 0.71) at Rprog. At R32 it lands at 0.7524, essentially the unfiltered baseline — restricting the Gaussian to the stem and stage 1 removes almost the entire benefit.

**Reduction operator and location, in both conditions** — this interaction is the most interesting result:

| comparison Gnone Gplateau        |                        |                        |
| -------------------------------- | ---------------------- | ---------------------- |
| input\_max − input\_bilinear     | **+0.31** pp (sd 0.28) | **−0.37** pp (sd 0.11) |
| stem\_bilinear − input\_bilinear | −0.58 pp (sd 0.48)     | −1.42 pp (sd 0.16)     |
| stem\_max − input\_bilinear      | **+0.48** pp (sd 0.06) | −0.90 pp (sd 0.17)     |

The sign **flips** for both max-pooling variants: max-pooling helps slightly without the Gaussian and hurts with it. Reducing after the stem is worse than reducing the input in both conditions, and markedly worse when filtered. Had I run only the filtered condition I would have concluded max-pooling is simply worse.

**Order matters when filtering.** Rreverse−Rprog is −0.26 pp (sd 0.61, sign flips across seeds) unfiltered — indistinguishable — but **−1.39 pp** (sd 0.27, same-sign) with Gplateau. Same time at each resolution, coarse-first ordering; the difference only appears once the Gaussian is present.

## Cost

Timing probe (T4, 4×32 accumulation, gradients on): plain 0.032–0.035 s/update across all four reductions; Gaussian 0.057–0.065; Gmix 0.067–0.071; early7 0.043. Reduced resolution again produced **no per-update saving** — r=16 is not faster than r=32 for the plain path, consistent with the earlier finding that these convolutions don't saturate a T4. All timings here were newly measured in this campaign; none are reused across jobs.

The best accuracy-per-second cell is **Rprog / Gnone / stem\_max**: 0.7997 at 491 s, i.e. within 0.74 pp of the top configuration for 70% of its runtime.

## Caveats

This is an exploratory benchmark with prior test-set exposure. Three seeds give a descriptive sample SD, not a confidence statement, and **the leading cell is not an independently confirmed estimate**. Several differences above are smaller than the replication deltas measured on identical configurations. All comparisons use the frozen epoch-30 checkpoint with every model on the common full-resolution unfiltered path; no best-epoch selection, no post-hoc schedule tuning.

Artifacts: `results/campaign_results.json`, `results/campaign_manifest_frozen.json`, `results/campaign_verification.json`, `results/campaign_previews/`, per-cell metrics and checkpoints under `results/kaggle_outputs/campaign-j{0,1,2}-*/`.

Stopping here as instructed — STL-10 and further parameter searches are yours to decide.