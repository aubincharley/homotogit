# Evidence, corrections and pending items

## Revision of 14 September 2026 (afternoon): appendix restructured

- The appendix now tells the CIFAR-10 exploration as four decisions (Gaussian
  filtering → resolution reduction → operators/locations/schedules → final paired
  selection), followed by discarded directions and limitations. The paper went
  from 24 to 11 pages.
- Moved out of the paper into `paper/archive/exploration_archive.tex` (and its PDF):
  the complete configuration catalogue, asset-set hashes, the plain-arm
  batch-to-batch table, the resolution-operator table, and the campaign-wide
  figures (operators, schedules and controls, 21-configuration grid, ablation,
  per-layer and adaptive studies). No data or failed run was dropped.
- Removed from the paper because the work is not done: the transfer table and
  mapping appendix, the theory appendix, the reserved landscape figure. Their plans
  are in `paper/handover/FUTURE_WORK_NOTES.md`; the paper keeps red notes.
- Factual corrections:
  1. At the input, the highest mean belongs to the perceptual operator (79.98 %),
     then the Ḣ⁻¹ reconstruction (79.76 %); the earlier text named Ḣ⁻¹. The
     generator now asserts this order.
  2. The resolution-only batch is 29 targeted configurations (87 runs, 84 valid),
     not "7 operators × 5 sites × 4 schedules"; the archive table header is
     overridden accordingly.
  3. "Resolution-only" is defined as without an additional Gaussian continuation;
     smoothing built into an operator (antialiased bilinear, MaxBlur) is part of the
     operator.
  4. The table of final results and the table of final losses were merged; the
     operator table was dropped from the paper in favour of the figure.
- Every number of the exploration narrative is a macro generated from the index
  (`EXPLORATION` in the generator).


Updated 14 September 2026, when the supplied working paper was integrated into the
repository on branch `manuscript`. The supplied snapshot had been written from the
conversation and earlier reports. This version records what was checked against the
code and records, what changed as a result, and what still waits on the team.

## Evidence used, in order of precedence

1. Run records: `summary.json` / `metrics.json` of every run, and the shipped-code
   check of the unified jobs (bundle equal to commit `3829fa3` after CRLF→LF).
2. The audited index `experiments/index.json` on `benchmark-organized` at
   `8d353f2` (19 experiments, 165 configurations, 377 runs; built from `c78ae05`;
   Idriss's ablation read at `continuation-gaussian-tv_exploration_1@591e125`,
   the sigma0 sweep at `adaptative-schedule@ea186fe`), with `docs/AUDIT.md`.
3. `continuation-core` (`ef5564c`): frozen presets, controller, site maps, and the
   44/44 bitwise CPU parity report against the executed code.
4. Earlier prose, including the supplied draft.

## What was verified

| Claim in the draft | Check | Result |
|---|---|---|
| Unified three-seed numbers 75.43±0.76, 80.37±0.61, 79.91±0.27, 81.51±0.22 | index, raw `metrics.json` (last record), `continuation_core/methods.py`; asserted by the generator | confirmed |
| Gains 4.94 / 4.48 / 6.08 pp | per-seed paired differences on identical assets | confirmed; SD of paired differences now shown (0.50 / 0.97 / 0.56) |
| Placement contrasts 79.91 vs 78.37 and 81.51 vs 81.00, paired 0.51±0.24 | index cells | confirmed; 1.54±0.88 added for the first |
| "After block 1" = pre-hook on `blocks[2]` | `continuation_core/models/resnet20_bn.py` (`"block1": ("input", "blocks.2")`), `controller.attach`; for resbench `continuation/resolution_ops.py` (`D1` → `model.blocks[2]`) | confirmed, same location in both batches |
| Sigma scaled by r/32 at all 19 sites, including the 5 upstream ones | `InterventionController.site_sigma`; schedule table built by calling the controller | confirmed |
| Schedules 16/24/32 for 6/6/18 epochs; Gaussian plateaus | `RPROG`, `PLATEAU_G`; parity check 4 against every recorded state | confirmed |
| Resolution-operator table and resbench baseline 75.53±0.36 | index | confirmed |
| Location, schedule and fixed-control statements | index | confirmed; MaxBlur exception kept |
| Recipe (SGD, lr 0.005, warm-up 60, cosine, wd 5e-4, 128 = 4×32, 391 updates) | index `reference_recipe`, `presets.reference`, `unified_driver.py` | confirmed; last batch 80 = 32+32+16 and example-weighted microbatch losses added |
| Higher training loss, lower test loss | record 30 of the 12 retained runs | confirmed for the 500-image training probe and for the last-epoch running training loss; full-training CE of the final weights was never evaluated, and the text now says so |
| Asset provenance | index `asset_sets`, group digests | no "set C"; unified shares `r20bn-campaign-assets` with five experiments; ablation differs in initial weights only |
| Perceptual D1 diverged 3/3; 87 completed vs 84 valid | index counts | confirmed |
| TV previews vs L2/H^-1 reconstruction | `tv_budget_previews` (no training), `resolution_ops.py` (constrained quadratic reconstruction) | confirmed; Appendix C now states the reconstruction problem from the code |
| Author names and order | `rapport.pdf` (progress report of 10 September) | confirmed |
| Style files unchanged | SHA-256 of `aistats2026.sty`, `fancyhdr.sty` vs supplied ZIP | identical |

## Corrections made to the supplied draft

1. **Resolution-only wall times.** "Approximately 428–519 s per run at block 1,
   against 428 s for plain" became **433–519 s** for the valid block-1 operators
   against 428 s for plain; the lower bound 428 was the plain arm itself. Their scope
   (evaluation every two epochs, checkpoint writes, two runs per two-GPU kernel) is
   stated, and they are explicitly not transferred to the unified batch, whose
   evaluation runs every epoch.
2. **Training-loss observation.** "A lower training loss is not necessary for a
   better test loss" is kept, but tied to the measured quantities: a 500-image
   training probe (eval mode, running statistics) and the running loss of the last
   epoch. Section 5 quotes the numbers from records.
3. **Pairing statement (§4).** "Paired runs when their assets and protocols match"
   now reads: same asset hashes, same code and conditions, within a batch; the
   0.73 pp spread of plain arms on identical assets is cited as the reason no
   cross-batch correction is made.
4. **Test-set reuse** is stated in the abstract and introduction, not only in §4.
5. **Appendix A** was completed from the frozen presets: recipe, site table,
   schedule and effective-sigma table (generated through the controller), kernel
   support and normalization, reflection padding and the explicit 4×4 path, bypass
   identities, sigma units, transition indexing, pairing, endpoints and timing scope.
6. **Appendix C, parallel directions.** The draft described diagnostics of other
   branches (branch scaling absorbed by BatchNorm, anchor drift, ordering versus
   repeated exposure). They are not in the audited index and were not checked, so
   the text now says their results are not reported. The db2 pilot is added with its
   recorded numbers.
7. **Transfer table.** Dashes became explicit `n.y.a.` entries, with codes for
   *failed* and *not applicable*, and a mapping appendix (App. E) was added.
8. **Figures.** The five reserved slots became seven generated figures (unified
   final + paired differences; unified curves with a separate target-path panel;
   operators; location/schedule/fixed controls; campaign grid; ablation; per-layer
   and adaptive). The complete table is generated as a longtable.
9. **Numbers in the text** are macros from `tables/numbers.tex`, generated from
   records, so abstract, introduction, discussion and appendices change together.

No method definition, schedule or result was altered, and no run was launched.

## Pending — waiting on the team

### Aubin, Alexandre (transfer)
- Completed, verified runs for `tables/transfer.tex`: which dataset / architecture /
  optimizer rows are actually run (the four rows are placeholders).
- For each row: seeds, budget, optimizer settings chosen without the test set, pinned
  assets, and run ids/commits for provenance.
- STL-10 (96×96): reference resolution, reduced sides, Gaussian scale in pixels and
  schedule duration.
- A non-ResNet architecture (VGG-11 adapter exists, never run): Gaussian site rule
  (all ReLUs vs block-like outputs) and an explicitly named reduction point, or
  `n/a` for the resolution procedures.
- AdamW: learning rate, weight decay, schedule.
- Status for any failed runs (`fail k/n`), kept visible.

### Idriss (theory)
- The model, assumptions and principal result for §5, and the proof and scope for
  App. D. Until then §5 keeps the question and the pending note. No theorem, proof or
  bound was written on his behalf.

### Max (visualization)
- Checkpoints saved around the transitions of the retained procedures (the
  exploratory checkpoints are not versioned).
- `figures/main_landscape.pdf` for §6 and optional `figures/appendix_geometry.pdf`,
  each with the loss, states, BatchNorm policy, data subset, PCA explained variance
  and projection residuals recorded in `provenance/`.

### Team
- Revise abstract, introduction and discussion once transfer, theory or
  visualization results exist.
- Before any real submission: that year's style and checklist (the 2027 pack was not
  released on 13 September 2026), the checklist filled with evidence, and the
  main/supplement split.

## Bibliography: primary sources checked for the supplied draft

All entries are in `references.bib`; no unchecked citation keys were carried over.

| Key | Primary source | Role |
|---|---|---|
| hazan2016graduated | https://proceedings.mlr.press/v48/hazanb16.html | Conditional graduated-optimization guarantees |
| bengio2009curriculum | https://icml.cc/2009/papers/119.pdf | Curriculum motivation |
| sinha2020curriculum | https://proceedings.neurips.cc/paper_files/paper/2020/hash/f6a673f09493afcd8b129a0bcf1cd5bc-Abstract.html | Gaussian feature-map curriculum |
| tan2021efficientnetv2 | https://proceedings.mlr.press/v139/tan21a.html | Progressive input resolution with regularization |
| rahaman2019spectral | https://proceedings.mlr.press/v97/rahaman19a.html | Spectral bias; distinguish frequency domains |
| zhang2019shift | https://proceedings.mlr.press/v97/zhang19a.html | Filtering before subsampling |
| vasconcelos2021aliasing | https://arxiv.org/abs/2108.03489 | Aliasing, placement and generalization |
| michaeli2023aliasfree | https://arxiv.org/abs/2303.08085 | Nonlinearities and aliasing |
| neyshabur2017geometry | https://arxiv.org/abs/1705.03071 | Parameterization and implicit regularization |
| ali2019early | https://proceedings.mlr.press/v89/ali19a.html | Finite-time regression risk; no transfer theorem asserted |
| hardt2016stability | https://proceedings.mlr.press/v48/hardt16.html | Algorithmic stability |
| li2018visualizing | https://arxiv.org/abs/1712.09913 | Filter-normalized planes and trajectories |
| dinh2017sharp | https://proceedings.mlr.press/v70/dinh17b.html | Reparameterization caveat for sharpness |
| he2016resnet | https://arxiv.org/abs/1512.03385 | Reference architecture |
