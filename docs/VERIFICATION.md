# Verification report

## What was compared

**Old** = the code that actually ran the `unified_selected` batch: the Kaggle
bundle shipped with job `unified-j0-20260911-113738` (`_repo/`). Its content
equals commit `3829fa3` after CRLF→LF normalisation (checked file by file;
recorded in the benchmark index as `shipped_code_check`). The recorded runs
come from branch `benchmark-organized`.

**New** = this branch, `continuation_core`.

Script: `verification/parity.py`. Report: `verification/parity_report.json`.
CPU, torch 2.5.1, float32. **44 checks, 0 failures, 203 s.** Every check is
bitwise (`torch.equal` or exact float equality); no tolerance was needed.

| # | check | scope | result |
|---|---|---|---|
| 1 | CIFAR-10 arrays (train/test images and labels), normalisation mean and std | new pickle reader vs old torchvision loader | identical |
| 2 | pinned asset digests; manifest vs `results/campaign_manifest_frozen.json` | all files, arrays, states | identical |
| 3 | 19 `conv_out` names and order, 10 post-ReLU positions, block1 = input of `blocks.2` | | identical |
| 4 | resolution, G and effective sigma at every site, epochs 0–31 | 4 methods vs old controller | identical |
| 4 | the same vs the `current` state recorded in every metrics record of the 48 runs | 93 records per method | identical |
| 5 | pipeline output; logits, loss, all gradients and BN buffers in train mode; eval logits | every distinct state: plain {0, 29}; `resolution_max_b1` {0, 6, 12, 29}; both Gaussian methods {0, 3, …, 21, 29} | identical |
| 6 | target state vs plain ResNet-20 without hooks, and vs the old target path | 4 methods | identical |
| 7 | parameters, momentum buffers, BN buffers and learning rate after updates at (epoch, batch) (0,0) (0,1) (5,389) (5,390) (6,0) (6,1) (11,390) (12,0) (20,390) (21,0) | real CIFAR-10 batches, pinned order, microbatch accumulation, across the 5→6, 11→12 and 20→21 transitions | identical |
| 8 | `evaluate` current and target, 1,000 test images, epoch-6 state | 3 methods | identical CE sums and accuracies |
| 9 | 6 updates across the 5→6 boundary, uninterrupted vs 2 + save + fresh trainer + resume + 4 | weights, momentum, positions | identical |
| 9 | epoch-6 checkpoint stores `used = (16, 0.85)` and `next = (24, 0.70)` separately; transition-window checkpoints at updates 2345 and 2347 | | as specified |
| 10 | evaluating `used` / `next` / `target` × `running_stats` / `fixed_batch_stats` leaves weights, buffers, torch RNG, controller state, train/eval mode and the checkpoint file unchanged | | unchanged |

Also run against the fixed merged tree of `benchmark-organized`: 44 / 44,
confirming that the merge fixes there (docs/AUDIT.md A10) preserved behaviour.

Unit tests (`tests/`, 32 tests, ~2 min CPU): operator identities, the executed
schedule tables, the effective-sigma table, hook counts and ordering,
downstream grid sizes, bitwise target bypass for all methods, the VGG adapter
and its refusal of `block1`, config round-trip, AdamW requiring explicit
settings, STL-10 and CIFAR-10 parsers on synthetic files, resumption, the
checkpoint schema, evaluation side effects, parameter vectors,
filter-normalised directions, PCA plane, trajectory export, and an import
check that nothing from the benchmark (or wavelet, TV, SoftPool, MaxBlur,
perceptual or adaptive code) is loaded.

Real-data dry run (`dry-run --method resolution_max_b1_gaussian_conv`,
CPU): assets verified; finite logits and gradients at all 8 distinct states;
transitions at updates 1173, 2346, 3519, 4692, 5865, 7038, 8211; target bypass
max |Δ| = 0.

Merged benchmark tree: the same 44 checks against the fixed
`benchmark-organized` code also pass (`verification/parity_report_merged_tree.json`).

Independence: a fresh `git clone --branch continuation-core --single-branch`
into an empty directory ran `verify-assets` (all digests match) and the full
test suite (32 passed) with no benchmark code on the path.

## Not verified / limitations

* **GPU.** Parity is on CPU. The reference runs used T4 GPUs with cuDNN at its
  defaults (`deterministic=False`). GPU runs of old or new code are not
  bitwise reproducible, and same-asset plain runs have differed by up to
  0.73 pp between batches.
* **Full training.** No complete 30-epoch run of the new code was performed, by
  instruction. Update-level parity over 10 representative updates stands in for
  it.
* **Checkpoint cadence.** The benchmark saved a rolling checkpoint per epoch
  with only the CPU RNG. The new checkpoints are richer, so old rolling
  checkpoints cannot be resumed here (they can be loaded as weights).
* **Evaluation cost.** `fixed_batch_stats` results depend on the evaluation
  batch size and composition by design.
* **Transfer presets, STL-10 and VGG-11** are untrained and unvalidated.
  `make-assets` creates new, unpaired asset identities.
* **Data loader.** CIFAR-10 is read from the python pickles without
  torchvision; equality with the torchvision arrays is check 1.
