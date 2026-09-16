# Paper-rewrite handoff: inventory and status

**Collected** 2026-09-15, 17:30–18:30 CEST, from `origin` =
`github.com/aubincharley/homotogit` after `git fetch --all`, plus local
worktrees. Nothing was merged, rewritten, retrained or re-run for this handoff.
Ongoing Kaggle jobs were not interrupted.

## 1. Sources

| material | branch @ commit | notes |
|---|---|---|
| Manuscript | `manuscript` @ `1bd1fe7` (2026-09-14 17:09, "aaaa"; previous "monvier" 15:06) | worktree clean; `origin/manuscript` identical |
| Frozen methods / core code | `continuation-core` @ `ef5564c` (remote tip; the local worktree `7e8fccc` differs only by a README commit) | method, operator and training code unchanged since the previous handoff |
| Landscape studies | `visualization` @ `9e9275c` (pushed 2026-09-15 17:24) + local files produced after the push | see §3 |
| Idriss | `continuation-core-experiments` @ `4179584` (2026-09-15 16:32) | read only via `git show`; not merged |
| Aubin, Alexandre | `activation-transfer` @ `2b013ef` (plus the `stl10`/`svhn`/`vgg11`/`groupnorm`/`cifar10-subset` tips), `continuation-core-optimizer-benchmark` @ `ac23aa7` | already supplied, **not resent**; no commits since that export |
| Exploration record | `benchmark-organized` @ `8d353f2` | already supplied |

**Uncommitted material included in the exports.** landscape_v3 raw outputs
downloaded after 17:24 (seeds 0–4 Hessian and evaluation jobs), their analysis
tables and figures, and the handoff files themselves. Binaries (checkpoints,
direction draws, Hessian eigenvectors) are kept local by `.gitignore`; their
sha256 values are in the shipped manifests.

## 2. Attachments

| file | content |
|---|---|
| `paper_current.zip` | LaTeX project at `manuscript@1bd1fe7`, with a PDF compiled from those exact sources (12 pages, clean build) and `BUILD_AND_VERSION.md` |
| `HANDOFF.md` | this file |
| `PROTOCOL_AUDIT.md` | implementation facts with code locations, including the probe vs full-train-CE correction |
| `landscape_latest_REPORT.md` | landscape_v3: amplitudes, validation, temporal, Hessian |
| `landscape_latest_per_seed.csv` | long-format seed-level summary: metric id, definition, units, number of directions/probes, within-checkpoint SD, paired Δ, status |
| `landscape_latest_details.zip` | tables, protocol, pilots, raw per-point JSONL, Hessian JSON, checks, manifests, code excerpts, schema README |
| `landscape_latest_figures.zip` | PDF + PNG + interactive HTML, manifest with captions; main-evidence figures flagged |
| `idriss_REPORT.md` | what `continuation-core-experiments` actually measured |
| `idriss_per_seed.csv` | one row per trained cell (28): train/test on full sets, Jacobian, amplitude curve, Hessian trace and eigenvalues, convergence |
| `idriss_details.zip` | his docs, code, 28 run records, probe shards, grid report, schema README, reconciliation |
| `idriss_figures.zip` | his single figure (PNG only), manifest |
| `comparison_status.md` | CBS/SDPoint comparison: not started |

## 3. Experiment status

| experiment | status | detail |
|---|---|---|
| Manuscript | current in repo; **Overleaf state unknown** | see §5 |
| landscape_v2 (20 runs) | complete | already supplied; one correction in §4 |
| landscape_v3 block A (amplitudes, 3 BN policies) | **complete** | 20 networks, 20 directions, 10 amplitudes |
| landscape_v3 block A-f64 (float64 precision check) | **complete** | 960 comparisons, none suspicious |
| landscape_v3 block B (validation, 20 directions) | **complete** | |
| landscape_v3 block C (temporal + all transitions) | **complete** | |
| landscape_v3 Hessian (160 eigenproblems + checks + cuts) | **complete** | 160/160 converged |
| landscape_v3 seed-0 Hessian planes | **complete** | the axis-extent rule is uninformative along the min direction |
| landscape_v3 wide random planes (21×21, all seeds) | **complete** | rerun on four GPU accounts after the original `maxfrrsava` job ran without a GPU; 0 missing vertices. The stale `maxfrrsava` kernel may still show as running and its output is unused. |
| Idriss grid (28 runs + endpoint probes) | complete | trained, probed and analysed |
| Idriss per-epoch trajectory probe | **never run** | code only (`tools/job_trajectory.py`) |
| Idriss gauge distance and PAC-Bayes displaced prior | **unavailable** | numbers are quoted in `RESULTS_GRID.md` §5, but no output files are committed |
| CBS / SDPoint comparison | **never started** | `comparison_status.md` |
| CBS reference reproduction | **never run** | specified in `benchmark-organized:docs/HANDOVER.md` |
| Theory (§5 of the paper) | not in any repository | the author's red note says the section may be removed |
| Transfer (Aubin) and optimizers (Alexandre) | complete | already supplied, unchanged; no new commits since |

## 4. Corrections to previously supplied material

1. **landscape_v2 report, straight-segment barriers.** It said the
   resolution→combined barrier is never smaller than plain→resolution. That
   is false on the train probe in seed 1 (0.603 vs 0.869) and seed 4 (0.829
   vs 0.844). The v2 `REPORT.md` now carries the correction (committed in
   `9e9275c`); data are in `tables/F_barrier_comparison_res_comb_vs_plain_res.csv`.
2. **landscape_v2 report, fixed-weight states.** It said "each set of weights
   has its lowest centre loss under the state it was last trained in". The
   v3 audit over every transition: this holds in 5/5 seeds everywhere except the Gaussian-only epoch-3 transition with every-point recalibration, where the “after” state is lower in 1/5 seeds (train probe) and 3/5 (test probe; 1.125 vs 1.126 nats). The blanket claim must be dropped or qualified (C_transitions_audit.csv).
3. **Manuscript, Appendix A "Training loss".** The paper says full-training-set
   CE was not evaluated. That holds for the unified-batch runs used in the
   paper's tables. Full-training-set CE now exists for other runs:
   - the 20 landscape_v2 runs;
   - Idriss's 28 runs.

   See `PROTOCOL_AUDIT.md` §7.
4. **Idriss `RESULTS_GRID.md` §0.** The Gaussian-only row prints per-seed
   accuracies in sorted order (79.91 / 79.95 / 79.98). In seed order they are
   79.95 / 79.98 / 79.91.
5. **landscape_v2 validation.** It used only directions 0 and 1. For
   resolution-only on the full test set with every-point recalibration it
   reported −0.006 (2/5 seeds below plain) at ε = 0.1 and −0.058 (3/5) at
   ε = 0.25. With all 20 directions (v3 block B) the values are −0.0133 (5/5)
   and −0.084 (5/5); combined is 4/5 at both ε. Use the v3 values
   (`landscape_latest_REPORT.md` §4.3).
6. **Formatting note for correction 2.** The audit table is
   `tables/C_transitions_audit.csv` inside `landscape_latest_details.zip`.

## 5. Unavailable material vs never-performed work

**Unavailable** (exists or may exist, but could not be collected):

- the author's Overleaf or local manuscript edits after 2026-09-14 17:09;
- output files behind Idriss's §5 distances and gauge percentages.

**Never performed:**

- a CBS/SDPoint comparison and a CBS reproduction;
- optimised connecting paths between solutions;
- a differentiable Hessian of the pointwise-recalibrated loss;
- Idriss's per-epoch probes;
- any causal test linking sensitivity or curvature to generalisation;
- any theory result.

## 6. Questions only the code/data owners can answer

1. **Max:** is `manuscript@1bd1fe7` the latest text, or does Overleaf hold
   newer edits? Which red-note sections stay (theory may be dropped)?
2. **Idriss:** where are the outputs behind `RESULTS_GRID.md` §5 (distances
   at epochs 6/12, the 13.6 displacement) and the gauge "≤ 2–3 %"
   measurements? And why are the reference re-runs' seed SDs 7–100× smaller
   than the unified-batch record? Resolution-only gave 80.21 / 80.22 / 80.22
   over three initialisations.
3. **Idriss:** are the 28 grid runs meant to be reported as a replication of
   the unified batch, or only as probe material?
4. **Team:** is the CBS/SDPoint comparison still planned, and where should its
   outputs go?
5. **Max:** should the landscape section use the frozen-BN Hessian results, the every-point-recalibrated finite sensitivities, or both? They rank Gaussian-only in opposite directions (see landscape_latest_REPORT.md §4.5).

