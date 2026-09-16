# Protocol audit: implementation facts for accurate writing

**Verified on 2026-09-15** against code and configurations:

- `continuation-core` (remote tip `ef5564c`, which differs from the local
  `7e8fccc` only by README), identical to `visualization@9e9275c` for these
  files;
- Aubin's `activation-transfer@2b013ef` (plus the stl10/svhn tips);
- `continuation-core-experiments@4179584`;
- `manuscript@1bd1fe7`.

Paths are relative to the repository root. "Unresolved" marks what the code
cannot settle.

## 1. The three selected procedures and the control

Source: `continuation_core/methods.py`.

| id (code) | paper name | intervention | schedule (epoch e, zero-based) |
|---|---|---|---|
| `plain` | Plain | none | — |
| `resolution_max_b1` | Resolution | adaptive **max** pooling of the tensor entering `blocks.2` | r = 16 (e 0–5), 24 (6–11), 32 = exact bypass (≥ 12): `RPROG` |
| `gaussian_postrelu` | Gaussian | Gaussian at 10 post-ReLU sites | G = 1.00 / 0.85 / 0.70 / 0.60 / 0.50 / 0.40 / 0.30 for 3 epochs each, 0 from e = 21: `PLATEAU_G` |
| `resolution_max_b1_gaussian_conv` | Combined | `RPROG` reduction + Gaussian at 19 conv outputs | `RPROG`, and σ = (r(e)/32)·G(e) at **all 19** sites |

- **Transitions.** A state holds for a whole epoch of 391 updates, so the
  first update of a new state is 391·e: 1173, 2346, 3519, 4692, 5865, 7038,
  8211 (`controller.state_for_epoch`, `train.py` header).
- **Final state.** In the final state (r = 32, G = 0) every hook returns its
  input object, so every method is bitwise plain ResNet-20. The v2 checks
  confirmed identical logits in eval and train mode.
- **Not a factorial design.** Gaussian-only and Combined use different sites,
  and Combined adds the r/32 factor (`methods.py` docstring).

## 2. Placement

Sources: `continuation_core/models/resnet20_bn.py` (`SITE_MAP`) and
`controller.py` (`attach`).

**Network.** `conv1 → bn1 → relu → blocks[0..8] → adaptive_avg_pool(1) → fc`.
Each block computes `conv1 → bn1 → relu → conv2 → bn2 → (+ shortcut) → relu`.
Option-A shortcuts: stride-2 subsampling plus zero channel padding at
`blocks.3` and `blocks.6`.

**Reduction.** Forward pre-hook on `blocks.2`: the complete output of
`blocks.1` (after its residual addition and final ReLU) is pooled before
entering block 2. Both the residual branch and the shortcut of block 2 and of
every later block see the reduced grid.

- Block indices are zero-based: "block 1" is the second block.
- There is **no upsampling**. Later layers, including the stride-2 convs,
  process the smaller map, and the global average pool restores the classifier
  input shape.
- Map sides at r = 16 / 24 / 32:
  - block 2: 16 / 24 / 32;
  - blocks 3–5: 8 / 12 / 16;
  - blocks 6–8: 4 / 6 / 8.

**Gaussian-only (`post_relu`, 10 sites).**

- Site 0 is a pre-hook on `blocks`: the stem ReLU output.
- Sites 1–9 are forward hooks on `blocks.k`: each block's output after the
  residual addition and final ReLU.
- The ReLU inside a block is **not** a site. BN never sees filtered
  activations directly; the next block's conv and its shortcut do.

**Combined (`conv_out`, 19 sites).**

- Forward hooks on `conv1` and on `blocks.k.conv1`/`conv2`.
- Each filter sits between a convolution and its BatchNorm, so BN normalises
  **filtered** activations and accumulates running statistics on them.
- Shortcuts, pooled vectors and logits are never filtered.
- Five sites (c0–c4) lie upstream of the reduction point at 32×32 but still
  use σ = (r/32)·G.

## 3. Downsampling operator

Source: `continuation_core/operators.py`.

- **Operator.** `torch.nn.functional.adaptive_max_pool2d(x, (r, r))`, whose
  windows are `[floor(i·n/r), ceil((i+1)·n/r))`.
  - 32 → 16: disjoint 2×2 windows.
  - 32 → 24: width-2 windows, with consecutive windows overlapping.
  - r = n: returns the input object.
- **Nature.** It is max-pooling, not average pooling and not a linear low-pass
  projection.

## 4. Gaussian kernel

Source: `operators.py`, `FixedSupportGaussian`.

- **Support.** Fixed for every σ: radius = ceil(4.0 · σ_max) with σ_max = 1.0,
  so radius 4 and 9 taps. Taps are exp(−d²/2σ²) for d = −4…4, computed in
  float64, normalised to sum 1, then cast to the tensor dtype.
- **Application.** Separable and depthwise (channels never mix): width pass,
  then height pass.
- **Padding.** Whole-sample reflection by 4 on each axis. When the radius is
  at least the map side (4×4 maps at r = 16), explicit index reflection
  r(i) = min(m, P − m) with m = i mod P and P = 2(n − 1) is used.
- **Bypass and range.** σ = 0 returns the input unchanged; σ > σ_max raises
  an error.
- **Units.** σ is in pixels **of the feature map at the site**. There is no
  per-site adaptation to map size.
- **Resolution scaling.** Only the Combined method scales σ, by r/32 at all
  19 sites. This makes the effective σ non-monotone: 0.500 → 0.425 → 0.525 →
  0.450 → 0.500 → 0.400 → 0.300 → 0. The manuscript's
  `tables/schedules.tex` states this correctly.

## 5. Gaussian-only vs the Gaussian component of Combined

| | Gaussian-only | Combined |
|---|---|---|
| sites | 10 post-ReLU (stem output + 9 block outputs) | 19 conv outputs, before BN |
| σ at a site | G(e) | (r(e)/32)·G(e), including the 5 full-resolution sites |
| BN sees filtered activations | no (filter after BN + ReLU) | yes |
| schedule | `PLATEAU_G` | `PLATEAU_G` × r/32 |

## 6. Inference state and BatchNorm policy, by study

| study | inference state | BN policy | source |
|---|---|---|---|
| Training records: CIFAR-10 exploration, unified batch, continuation-core runs | two paths per epoch. `current` = state of the epoch just completed; `target` = r = 32, G = 0. After epoch 30 they coincide. | eval mode with the checkpoint's own running statistics (`running_stats`). BN statistics are accumulated in training over **microbatches of 32** (batches of 128 split into 4). | `presets.reference`, `train.py` `snapshot`, `evaluate.py` |
| Aubin's transfer studies | same `current`/`target` convention, `running_stats` | STL-10 evaluates every 3rd epoch and only `current` for plain | `scripts/stl10_configs.py` |
| Alexandre's optimizer benchmark | as the reference (final record) | `running_stats` | his branch (already supplied) |
| landscape_v2 | final state for sensitivity; fixed-weight studies also use before/after/final states | `saved` = checkpoint statistics; `recalibrated` = reset BN and cumulative average over a fixed 2,000-image **training** calibration set, per evaluated point, batches of 500 | `landscape_v2/evaluator.py` |
| landscape_v3 | as v2, plus produced/next-update states at intermediate checkpoints | adds `centre_frozen`: recalibrate once at the unperturbed weights and state, then hold fixed for every perturbation and for the Hessian | `landscape_v3/evaluator.py` |
| Idriss grid probes | `target` state at epoch 30 | eval mode, checkpoint running statistics | `continuation_core/analysis/{sensitivity,curvature}.py`, `tools/job_probe.py` |

A second evaluation policy, `fixed_batch_stats` (training-mode BN on cloned
buffers), exists in `evaluate.py`. No reported number uses it.

## 7. Training cross-entropy: probe vs full training set

| measurement | images | where |
|---|---|---|
| Per-epoch "train_probe" CE and accuracy in every `metrics.json`/`summary.json` (exploration, unified batch, continuation-core, Aubin, Alexandre, Idriss's training logs) | **500** pinned training images (`train_probe` asset) | `train.py` `snapshot` |
| `train_loss_epoch` | running mean of the minibatch losses **during** the epoch, while the weights change, under the current state | `train.py` |
| landscape_v2 validation centres (`train_full`) | **all 50,000** training images | v2 `validation__*.jsonl` |
| landscape_v2/v3 probes | 1,000 class-balanced training images (`train_probe_idx`) and 1,000 test images; larger checks use a 10,000-image training subset (`train_large`) and the full test set | `studies/landscape_v2/inputs/subsets.npz` |
| Idriss's grid: `train.ce`/`err` | **all 50,000** training images (`CheckpointEvaluator(split="train")` = the pinned `subset`, which is the whole training set) | `tools/job_probe.py`, `analysis/landscape.py` |

**Correction to the manuscript.** `appendices/a_reference_protocol.tex` says
that training CE "is evaluated on the fixed probe of 500 training images, not
on the full training set, which we did not evaluate". That remains true for
the **unified-batch runs the paper's tables use**. Full-training-set CE now
exists for two other sets of runs:

- the 20 landscape_v2 runs (5 seeds): mean full-train CE 0.20 plain vs 0.30
  resolution / 0.27 Gaussian / 0.31 combined (v2 report);
- Idriss's 28 grid runs (all 50,000 images, per cell in `idriss_per_seed.csv`).

The sentence must stay scoped to the unified batch.

## 8. Material adaptations across datasets and architectures

This supplements Aubin's reports; each point is verified in his config
scripts on `activation-transfer`.

- **STL-10** (`scripts/stl10_configs.py`):
  - input 96×96;
  - resolution schedule 48/72/96 with `reference_resolution` = 96 (reference
    *ratios* 1/2, 3/4, 1);
  - Gaussian levels ×3, so σ_max = 3.0 and the fixed radius becomes 12
    (**25 taps**, not 9);
  - 60 epochs of 40 updates (2,400 updates), boundaries doubled;
  - optimizer unchanged, so the 60 warm-up updates are 1.5 epochs;
  - evaluation every 3rd epoch;
  - 5,000 training images.
- **SVHN** (`scripts/svhn_configs.py`):
  - all method settings are reference values;
  - the training set is **cut from 73,257 to 50,000** so updates per epoch and
    schedule boundaries match the reference;
  - test set of 26,032 images.
- **CIFAR-10, 5,000 images** (`scripts/cifar10_subset_configs.py`), two arms:
  - `stl_budget`: 60 epochs, STL-10's boundaries, 2,400 updates;
  - `reference_updates`: 293 epochs with boundaries scaled by 391/40, 11,720
    updates.

  Reference σ and resolution values in both.
- **VGG-11-BN** (`scripts/vgg11_configs.py`):
  - 8 post-ReLU sites and 8 conv-output sites (not 10/19);
  - reduction point `after_pool1` (native size 16, input of `features.4`),
    schedule 8/12/16 with reference 16;
  - four max-pools follow, so r < 16 drives the last convolutions to 1×1;
  - same data order and indices as the reference, new initial weights.
- **ResNet-20-GroupNorm** (`scripts/resnet20_gn_configs.py`):
  - same site map and parameter count (269,722);
  - no running statistics, so BN-policy distinctions do not apply;
  - lr 0.005 not retuned;
  - new initial weights.
- **ResNet-20-BN with GELU/SiLU** (`scripts/resnet20_act_configs.py`):
  - functional activations;
  - loads the **reference** pinned initial weights and data order directly;
  - Kaiming-ReLU gain kept.
- **Optimizers** (Alexandre): already supplied. Nothing new found.

## 9. Unresolved from code

- **CIFAR-10 5k arms.** Whether the `train_probe` asset still has 500 images
  and the training subset exactly 5,000; not re-checked here. See the asset
  manifests already supplied.
- **Idriss's §0 discrepancy.** The seed spreads of his reference re-runs do not
  match the unified-batch records, and his report calls this "unexplained". Nothing in the configs explains
  it (see `idriss_REPORT.md`).
- **Other notes.** Whether the author's Overleaf copy contains protocol
  statements beyond commit `1bd1fe7` (see `HANDOFF.md`).
