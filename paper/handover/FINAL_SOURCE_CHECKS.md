# Final source checks

Status of every displayed table, every figure and every group of numerical claims in the paper,
together with the presentation changes applied. Written 16 September 2026.

Two levels are kept apart throughout:

* **verified against original records** — recomputed from the `summary.json` / `metrics.json` of the
  runs themselves, the committed job summaries and probe shards, or the raw landscape evaluation
  files, through `tools/build_records.py`;
* **recomputed from supplied exports** — agreement between two derived files, which is not
  independent provenance.

Everything below is at the first level unless it says otherwise. Sources, row counts and a sha256
for each table are in `data/records/SOURCES.json`; the aggregate counts the generator re-derives are
in `evidence/RECOMPUTED_CHECKS.json`.

## 1. Sources read

| Material | Snapshot | How it was read |
|---|---|---|
| Historical experiment index and its 377 run records | `benchmark-organized@8d353f20` | `git show`, per cell |
| Transfer runs (96) | `activation-transfer@2b013ef`, `groupnorm-transfer@3a77427`, `vgg11-transfer@216fd86`, `svhn-transfer@51f38b0`, `stl10-transfer@3a9c34c`, `cifar10-subset@29c18ea` | `git show`, per run |
| Optimizer runs (53) | `continuation-core-optimizer-benchmark@ac23aa7` | committed job summaries |
| Input-derivative grid (28 runs) | `continuation-core-experiments@4179584` | run records + probe shards |
| Landscape study | `visualization@9e9275c` worktree, raw outputs | 96,116 v3 points, 33,820 reused v2 points, 160 Hessian records |
| Method definitions | `continuation-core@ef5564c` | code |
| Manuscript text | `manuscript@1bd1fe7` | worktree |

## 2. Tables

| Item | Status | Source and code |
|---|---|---|
| `tab:methods` (selected procedures) | verified | `continuation_core/methods.py`; sites and schedules confirmed against `controller.py` |
| `tab:transfer` (final accuracy) | **corrected presentation**, values verified | all 144 runs recomputed; `build_records.py` → `benchmark_settings.csv` → `build_assets.py` |
| `tab:transfer-paired` (within-seed gains) | **new companion**, values verified | same records; each gain is the mean of per-seed differences, SD of those differences |
| `tab:transfer-protocol` (adaptations) | **corrected presentation**, values verified | per-run `config.json` of each setting, plus the config scripts of the transfer branches |
| `tab:transfer-ce` | **corrected presentation**, values verified | same records; probe sizes read from each run's own record |
| `tab:costs` (wall time) | **corrected presentation**, values verified | per-run `timing.wall_seconds`; means and SDs computed from the runs, not from ratios |
| `tab:sensitivity-validation` ($S$) | **corrected presentation**, values verified | raw large-set evaluations; 160 seed-level values |
| `tab:sensitivity-validation-paired` ($\Delta S$) | **new companion**, values verified | same records |
| `tab:hessian` (ratio) | retained by design, caption made self-contained | 160 Hessian records; within-seed ratios |
| `tab:hessian-all` (absolute) | verified | same records |
| `tab:landscape-endpoints` | verified | v2 validation centres; reproduce each run's training-log endpoint exactly (max difference 0) |
| `tab:input-reference`, `tab:input-grid` | verified | 28 probe-shard cells; every field matches to $\le4\times10^{-12}$ |
| `tab:sites`, `tab:schedules` | verified | regenerated from `continuation_core` through `make_legacy_assets_with_repo.py`, byte-identical |
| `tab:operators`, `tab:unified-reference` | verified | regenerated from the experiment index, byte-identical |
| `tab:catalogue` / companion catalogue | verified | 165 configurations recomputed from the 377 per-run summaries; every value reproduces the index |

## 3. Figures

| Figure | Status | Source |
|---|---|---|
| `fig:sensitivity`, `fig:centre-frozen` | verified, regenerated | `landscape_sensitivity_paired.csv` from raw points; 24,000 directional values behind them |
| `fig:wide-surfaces` | verified, regenerated | 41-by-41 measured vertices per model, 0 missing |
| `fig:interpolation` | verified, regenerated | 1,020 measured points, 51 per segment |
| `fig:input` | verified, regenerated | probe shards of the 28-run grid |
| `fig:resolution`, `fig:unified`, `fig:unified-curves` | verified, regenerated from records | `make_legacy_assets_with_repo.py` reproduces the supplied PDFs pixel-for-pixel (0 differing pixels at 100 dpi) |

## 4. Numerical claims in the prose

All quoted numbers are now generated into `tables/numbers_generated.tex` (and the exploration
macros into `tables/numbers.tex`), so the text cannot drift from the tables.

| Claim group | Status |
|---|---|
| Twelve settings, R positive in 12/12 means and 36/36 seed pairs, gains 1.16--6.28 pp | verified |
| STL-10 RG $-1.29$ pp, negative in all three seeds; 5k gains $+10.62$ and $+11.78$ | verified |
| Adaptive-optimizer rows: G negative under Adam/AdamW, positive and small under RAdam | verified |
| Full-training-set CE of the landscape batch (0.197/0.303/0.273/0.312) | verified |
| GroupNorm and STL-10 counterexamples on training-probe CE | verified |
| Selection-batch wall times | verified; now stated in minutes, matching the cost table |
| $S$ and $\Delta S$, both probes, three policies, ten amplitudes | verified; 1,200 seed rows and 180 paired rows recomputed |
| Full-test differences at $\varepsilon=0.1$ and $0.25$ | verified |
| 160 Hessian problems, all converged from both starts, top-1 below plain in every seed | verified |
| Smallest eigenvalue negative in all 160; gradient norms 1.88--7.37 | verified |
| Residuals, float64 check, HVP symmetry and precision | verified |
| Gaussian epoch-3 transition exception (1 train seed, 3 test seeds) | verified against all 64 audited transition conditions |
| Straight-segment barriers | **corrected**, see §5 |
| PCA explained variance 56.6--59.0\% and 93.2--93.4\% | verified from the trajectory arrays |
| Input Jacobian radii and trace estimates; 135/140 power-iteration estimates converged | verified |
| Exploration narrative of Appendix B (all macros) | verified: the generator re-derives them from the index, and the index reproduces the raw records |
| Catalogue totals: 19 experiments, 165 configurations, 377 runs | verified |

## 5. Corrections applied

1. **Straight-segment barriers.**
   *Was:* "The barriers are positive on both probes, but the R--RG barrier is not always higher than
   plain--R. On the training probe it is lower in seeds 1 and 4."
   *Now:* "The barriers are positive on both probes, but the R--RG barrier is not always higher than
   plain--R: it is lower in 3 of the five seeds on the training probe (seeds 1, 2 and 4) and in 1 on
   the test probe (seed 1)."
   *Cause:* seed 2 (0.7951 against 0.8082 nats) and the test-probe case were omitted. The supplied
   export `F_barrier_comparison_res_comb_vs_plain_res.csv` already contained both; only the prose
   was incomplete. The same omission appears in the earlier handoff note and in the landscape
   `REPORT.md` correction, which the owner may wish to amend as well.
   *Affected:* Appendix D.5 of this paper; `provenance/CLAIM_AUDIT.md`.

2. **CIFAR-10 5k training probe.**
   *Was:* "The supplied exports do not independently establish the probe size in the two CIFAR-10 5k
   settings; their reported training CEs remain probe measurements."
   *Now:* the appendix states the verified facts: one pinned asset set for both budgets and all four
   methods, 5,000 distinct training images, a 500-image probe, every probe index inside the subset,
   and per-channel normalization inherited from the complete 50,000-image split.
   *Cause:* the index arrays were not in the delivered bundle. They regenerate bit-for-bit from the
   recorded generator, and their sha256 match the digests stored in all 24 run records
   (`subset 6a14a115…`, `train_probe 0634544f…`, `perm_seed0 936b46b8…`, 300 epochs of order).
   *Affected:* Appendix C. No table value changes: the evaluated population was labelled correctly.

3. **Lanczos iteration count.** "they required 50--100 iterations" → generated range 40--110, which
   covers both deterministic starts of all 160 problems.

4. **Rounding, now generated rather than transcribed.** The Gaussian full-test difference at
   $\varepsilon=0.1$ prints as $+0.110$ (was $+0.111$) and the float32--float64 HVP bound as
   $4.1\times10^{-4}$ (was $4.2\times10^{-4}$).

5. **Cost table.** Previously the plain mean in minutes plus mean within-seed ratios; ratios were
   not multiplied into absolute times anywhere, so no published number was wrong. The table now
   reports the measured time of each method, computed from the per-run times.

## 6. Presentation changes

* Dataset blocks with the training and test population in the heading; model, activation,
  normalization, optimizer and budget visible in every row of the accuracy, gain, CE, cost and
  protocol tables. No row depends on a distant cross-reference.
* Absolute quantities are the primary reading (accuracy, $S$, wall time); paired differences,
  ratios and negative-seed counts are kept in identified companion tables, except Table~3, whose
  within-seed Hessian ratio answers the relative-curvature question directly and whose absolute
  eigenvalues are in the appendix.
* Every caption now states dataset, model, optimizer, checkpoint, evaluation population, replication
  count, units and what the $\pm$ summarizes ("sample SD across training seeds"), and says when a
  single seed has no SD.
* Training population, training probe, test population and BatchNorm calibration set are named
  separately wherever they appear.
* Units are in the headers (percent, percentage points, nats, minutes); training cross-entropies use
  significant digits instead of switching scale between rows.
* Figures: measured points marked, logarithmic axes named, one colour per method throughout,
  legends moved out of the data area, and the caption says when panels do not share a scale.

## 7. Unresolved

1. **Newer manuscript text.** `origin/manuscript` is still `1bd1fe7`; no local edit, stash or
   downloaded file newer than that snapshot exists on this machine. Whether the author's Overleaf
   copy has newer edits cannot be determined here.
2. **Idriss's checkpoint binaries.** The 28 final checkpoints live in a Kaggle dataset
   (`cc-grid-final`) and are not committed, so their hashes cannot be checked. What *is* verified:
   28 distinct run directories with distinct metrics and summary files, correct per-seed initial
   weight and data-order digests, per-epoch trajectories that diverge from the first epoch in every
   seed pair, and probe records naming one distinct checkpoint path per cell. Nothing indicates a
   reused checkpoint or seed label; the narrow seed spread remains unexplained and is reported
   descriptively.
3. **Mixed code commits within two transfer studies.** The GroupNorm and VGG-11 runs were launched
   across three commits each, because the launcher was extended mid-campaign. The differences
   between those commits touch only the launcher and the addition of other architectures; no file
   used by these runs changed. Recorded in `data/records/benchmark_runs.csv` (`code` column).
4. **CBS / SDPoint comparison.** Still not started; no output exists on any branch as of
   16 September 2026, so no comparative claim is made.
5. **Idriss's branch has moved on.** `continuation-core-experiments` is now at `1a60649`, four
   commits past the snapshot this paper cites, adding a BatchNorm-policy curvature experiment and a
   cross-study note. Those results are not used here, and the paper continues to cite `4179584`.

## 8. Not requested and not produced

Theory; per-epoch input-derivative trajectories; an optimized mode-connecting path; gauge or
PAC-Bayes numbers; a Hessian differentiated through pointwise BatchNorm recalibration. No training,
evaluation or landscape sweep was run for this revision, and no running job was interrupted.
