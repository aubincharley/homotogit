# Method definitions

Four frozen configurations, recovered from the code that produced the
`unified_selected` batch on branch `benchmark-organized`
(`scripts/unified_driver.py`, `continuation/ablation_ops.py`,
`continuation/campaign_ops.py`, `scripts/ablation_manifest.py`,
`scripts/ablation2_manifest.py`) and cross-checked against the per-epoch
state recorded in all 48 runs (`verification/parity_report.json`, checks 4).

| public id | source config id (unified batch) | final test accuracy, seeds 0 / 1 / 2 | mean ± SD |
|---|---|---|---|
| `plain` | `plain` | 74.58 / 75.66 / 76.05 | 75.43 ± 0.76 |
| `resolution_max_b1` | `shrink_b1` | 79.97 / 80.06 / 81.07 | 80.37 ± 0.61 |
| `gaussian_postrelu` | `blur_relu` | 80.06 / 80.06 / 79.60 | 79.91 ± 0.27 |
| `resolution_max_b1_gaussian_conv` | `shrink_b1_conv` | 81.30 / 81.50 / 81.73 | 81.51 ± 0.22 |

Asset set `r20bn-campaign-assets` (committed under
`assets/cifar10_resnet20bn/`). Final metric = epoch-30 test accuracy; at
epoch 30 every method is in its target state, so the current and target
evaluations are identical. These are exploratory results on a test set that
has been examined repeatedly. The three methods are representatives of their
families, not optimal settings.

**Not a factorial design.** `gaussian_postrelu` filters 10 post-ReLU tensors
with sigma = G(e). `resolution_max_b1_gaussian_conv` filters 19 convolution
outputs (before BatchNorm) with sigma = (r(e)/32)·G(e). The combined method is
**not** `resolution_max_b1` plus `gaussian_postrelu`, and differences between
the four rows confound placement, site count and sigma scaling.

## Reference network and forward locations

`resnet20_bn_cifar` (269,722 parameters). Module names are the ones hooks
attach to.

```
input 3x32x32
 conv1 (site c0) -> bn1 -> relu                          stem, 16 ch, 32x32
   │                                    ← post_relu p0: input of `blocks`
 blocks.0  conv1 (c1) -> bn1 -> relu -> conv2 (c2) -> bn2 -> + id -> relu   16 ch
   │                                    ← p1: output of blocks.0
 blocks.1  conv1 (c3) ... conv2 (c4) ... + id -> relu                        16 ch
   │                                    ← p2: output of blocks.1
   │                                    ← REDUCTION "block1": input of blocks.2
 blocks.2  conv1 (c5) ... conv2 (c6) ... + id -> relu                        16 ch
   │                                    ← p3
 blocks.3  conv1 stride 2 (c7) ... conv2 (c8) ... + optionA(x) -> relu      32 ch
   │                                    ← p4
 blocks.4  (c9, c10)   ← p5          blocks.5 (c11, c12)   ← p6
 blocks.6  conv1 stride 2 (c13) ... conv2 (c14) ... + optionA(x) -> relu    64 ch
   │                                    ← p7
 blocks.7  (c15, c16)  ← p8          blocks.8 (c17, c18)   ← p9
 adaptive_avg_pool2d(1) -> flatten -> fc (64 -> 10)
```

Option A shortcut: `x[:, :, ::2, ::2]` then zero-pad 8 channels on each side
(16 → 32, 32 → 64). It contains no convolution and is never filtered.

### "block 1" and the reduction

`block1` is the **complete output of `blocks[1]`**, after its residual
addition and final ReLU. The operator is a forward pre-hook on `blocks[2]`
that replaces that block's input with
`adaptive_max_pool2d(x, (r, r))`. It is applied **once**. Every later layer
genuinely processes the smaller grid, including both branches of each
residual block, since the shortcut uses the same reduced tensor.

| `r` at block1 | blocks.2 | blocks.3–5 | blocks.6–8 | pooled features |
|---|---|---|---|---|
| 16 (epochs 0–5) | 16x16 | 8x8 | 4x4 | 64 |
| 24 (epochs 6–11) | 24x24 | 12x12 | 6x6 | 64 |
| 32 (epochs 12–29) | 32x32 (hook returns its input) | 16x16 | 8x8 | 64 |

The global average pool produces 64 numbers whatever the spatial size, so the
classifier is unchanged and the same parameters serve every resolution.
Adaptive max windows are `[floor(i·n/r), ceil((i+1)·n/r))`. 32 → 16 gives
disjoint 2x2 windows; 32 → 24 gives 2-wide windows, 16 consecutive pairs of
which overlap per axis.

### `gaussian_postrelu`: the 10 sites, in order

| site | tensor | spatial at 32x32 |
|---|---|---|
| p0 | stem output after its ReLU = input of `blocks` (pre-hook on `blocks`) | 32 |
| p1–p3 | output of `blocks.0`, `blocks.1`, `blocks.2` (after `+ shortcut` and final ReLU) | 32 |
| p4–p6 | output of `blocks.3`, `blocks.4`, `blocks.5` | 16 |
| p7–p9 | output of `blocks.6`, `blocks.7`, `blocks.8` | 8 |

The ReLU **inside** each block (after `bn1`) is not a site. "After every ReLU"
means these 10 tensors, as executed; no further activation calls are filtered.
The filtered tensor is what the next block (or the pooling layer, for p9)
receives, on both its residual and shortcut paths.

### `resolution_max_b1_gaussian_conv`: the 19 sites, in order

A forward hook replaces each 3x3 convolution's output, so within a block the
order is:

```
conv1 -> GAUSSIAN -> bn1 -> relu -> conv2 -> GAUSSIAN -> bn2 -> + shortcut(x) -> relu
```

and for the stem `conv1 -> GAUSSIAN -> bn1 -> relu`. BatchNorm therefore
normalises, and accumulates running statistics on, **filtered** activations.

| site | module | spatial at r = 16 / 24 / 32 |
|---|---|---|
| c0 | `conv1` (stem) | 32 / 32 / 32 |
| c1, c2 | `blocks.0.conv1`, `blocks.0.conv2` | 32 / 32 / 32 |
| c3, c4 | `blocks.1.conv1`, `blocks.1.conv2` | 32 / 32 / 32 |
| c5, c6 | `blocks.2.conv1`, `blocks.2.conv2` | 16 / 24 / 32 |
| c7 | `blocks.3.conv1` (stride 2, already decimated) | 8 / 12 / 16 |
| c8–c12 | `blocks.3.conv2` … `blocks.5.conv2` | 8 / 12 / 16 |
| c13 | `blocks.6.conv1` (stride 2) | 4 / 6 / 8 |
| c14–c18 | `blocks.6.conv2` … `blocks.8.conv2` | 4 / 6 / 8 |

Hook registration order: reduction pre-hook first, then the 19 conv hooks.
They sit on different modules, so the order has no numerical effect here.

## Schedules (all indexed by zero-based epoch; one state per whole epoch)

With 391 updates per epoch, epoch `e` covers updates `391·e … 391·e + 390`.
A transition at epoch `e` means update `391·e` is the first to use the new
state.

**Gaussian level G(e)**, from `_PLATEAU[min(e // 3, 6)] if e < 21 else 0`:

| epochs | 0–2 | 3–5 | 6–8 | 9–11 | 12–14 | 15–17 | 18–20 | 21–29 |
|---|---|---|---|---|---|---|---|---|
| G | 1.00 | 0.85 | 0.70 | 0.60 | 0.50 | 0.40 | 0.30 | 0 |
| first update | 0 | 1173 | 2346 | 3519 | 4692 | 5865 | 7038 | 8211 |

**Resolution r(e)** at block1: 16 for epochs 0–5, 24 for 6–11, 32 from 12
(first updates 0, 2346, 4692).

**Effective per-site sigma** (pixels of the feature map at the site):

| epochs | `gaussian_postrelu` (all 10) | `resolution_max_b1_gaussian_conv` (all 19) |
|---|---|---|
| 0–2 | 1.00 | 16/32 · 1.00 = 0.500 |
| 3–5 | 0.85 | 16/32 · 0.85 = 0.425 |
| 6–8 | 0.70 | 24/32 · 0.70 = 0.525 |
| 9–11 | 0.60 | 24/32 · 0.60 = 0.450 |
| 12–14 | 0.50 | 0.500 |
| 15–17 | 0.40 | 0.400 |
| 18–20 | 0.30 | 0.300 |
| 21–29 | 0 (bypass) | 0 (bypass) |

For the combined method this effective sigma is **non-monotone**, and the
factor r/32 is applied at **all 19 sites**, including c0–c4, which run at
32x32 upstream of the reduction. That is what `SiteController.q_for` executed
for a block reduction (it only special-cases `stem` reductions). It is
preserved here rather than "corrected" to a local width ratio.

## Gaussian operator

* fixed support: `sigma_max = 1.0`, `truncate = 4.0`, radius 4, 9 taps for
  every sigma;
* taps `exp(-d²/(2σ²))`, `d = -4…4`, computed in float64, normalised to sum 1,
  cast to float32;
* separable depthwise convolution, width pass then height pass, channels never
  mixed;
* reflection padding without repeating the edge sample: PyTorch `reflect` when
  `4 < n`, otherwise the explicit index `min(m, P−m)`, `m = i mod P`,
  `P = 2(n−1)`. The explicit path is used only on 4x4 maps, i.e. sites c13–c18
  of the combined method during epochs 0–5. `gaussian_postrelu` never reaches
  it (its smallest map is 8x8).

## Bypass conditions (exact identities)

* Gaussian site: `G == 0` (or no Gaussian in the method) → the hook returns its
  input tensor object. No kernel, no padding.
* Reduction: `r` equal to the incoming size (32) → the pre-hook returns its
  input object.
* **Target state** = `(r = 32, G = 0)`. Under it every method is bitwise the
  plain ResNet-20, in train and eval mode (tests and parity check 6).

## Training recipe (reference preset, unchanged)

| | |
|---|---|
| data | CIFAR-10 python batches, all 50,000 training images, official 10,000 test images; no augmentation |
| order | `train = train.images[subset]`; epoch `e` visits `perm_seed<k>[e]`; batches are consecutive slices of 128, the last one 80 |
| probe | `train_probe` (500 indices into the subset-ordered tensor) |
| normalisation | per-channel mean/std of the 50,000 unfiltered training images (float64 → float32), checked against pinned values to 1e-7 |
| model init | pinned `init_seed<k>.pt` (parameters and BN buffers), loaded strictly |
| optimiser | SGD, lr 0.005, momentum 0.9, weight decay 5e-4 on all parameters (including BN and biases), no Nesterov |
| LR | `lr·(u+1)/60` for `u < 60`, then `0.5·lr·(1 + cos(π·(u−60)/(11730−60)))`, `u` = global update |
| batching | microbatches of 32; each microbatch's mean CE scaled by `n_micro / n_batch`; one step per 128 (80 at epoch end = 32+32+16). BN sees microbatches of 32 (or 16). |
| budget | 30 epochs = 11,730 updates |
| evaluation | before training and after every epoch: train probe and test set, current path and target path, BN running statistics, batches of 500 |
| source runs | Kaggle T4, torch 2.10.0+cu128; cuDNN flags left at their defaults |

Metrics record `k` holds the state used by epoch `k − 1` (record 0 is the state
of epoch 0, before any update).
