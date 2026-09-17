# Method mapping: seven arms on our CIFAR-10 ResNet-20 recipes

Written 2026-09-17, before any main run. The code referred to is on branch `comparison-cbs-sdpoint`
(started from `continuation-core-optimizer-benchmark@ac23aa7`).

## 0. Sources actually read

| source | version | read |
|---|---|---|
| CBS paper | arXiv:2003.01367v5 (Sinha, Garg, Larochelle) | Sec. 3 (3.1 Gaussian kernel layer, 3.2 curriculum), Sec. 4 (image classification), App. B–D |
| CBS code | github.com/pairlab/CBS **@ 5f62e7da5b290e8f62f408c6f89146f3f361cc2e** (2020-12-06), verified by `git rev-parse` | `resnet.py`, `utils.py`, `arguments.py`, `solver_cbs.py`, `solver_base.py`, `data.py`, `main.py` |
| SDPoint paper | arXiv:1801.09335v1 (Kuen et al.) | Sec. 4 (4.1–4.5), Sec. 5 (5.1 instance-specific BN), Sec. 6, Algorithm 1 |
| SDPoint code | github.com/xternalz/SDPoint **@ 0013c5dafe80780ea749198ebc42824c3ed41e6c** (2019-11-05), verified | `models/resnet.py`, `models/preresnet.py`, `main.py` (`train`, `validate`, evaluation loop) |
| our methods | `continuation_core/methods.py`, `controller.py`, `operators.py`, `models/resnet20_bn.py`, `train.py` | whole files |

Verbatim copies (MIT licences included) and sha256: `comparison/reference_code/README.md`.

## 1. Backbone, trainer and recipes (shared by all seven arms)

- **Backbone.** Our ResNet-20 (widths 16/32/64, 3 BasicBlocks per stage, option-A zero-padding
  shortcuts, BN momentum 0.1, global average pooling, linear classifier). Neither author backbone
  is used: CBS uses a ResNet-18 (width 64, 1×1-convolution shortcuts with BN, `avg_pool2d(4)`);
  SDPoint's ResNet is the ImageNet model (7×7 stride-2 stem, max-pool, bottleneck/basic blocks).
- **Trainer.** `continuation_core.train.Trainer`, unchanged in its computation: pinned initial state
  `init_seed<k>.pt`, epoch order `perm_seed<k>[e]`, batches of 128 in microbatches of 32 (loss scaled
  by microbatch share, one optimizer step per logical batch), 391 updates × 30 epochs = 11,730,
  float32 without autocast, no augmentation, per-epoch evaluation (500-image training probe and full
  test set, current and target paths, saved BN statistics).
- **SGD recipe** (`presets.reference`): lr 0.005, momentum 0.9, no Nesterov, coupled weight decay
  5e-4 on every parameter, 60 linear warm-up updates then cosine to 0 (indexed by global update).
- **AdamW recipe** (`scripts/job_optimizer_benchmark.config_for`, `CHOSEN_LR['adamw']`): identical
  except AdamW lr 0.02, betas (0.9, 0.999), eps 1e-8, decoupled weight decay 5e-4.
- **Only recorded setting changed:** checkpoint cadence (final checkpoint + `rolling.pt`); it does not
  enter the computation.

## 2. Arms

| arm | operator | insertion sites | schedule | inference state (native) | BN at training |
|---|---|---|---|---|---|
| `plain` | none | — | — | the network | batch statistics, running momentum 0.1 |
| `resolution_max_b1` (R) | adaptive **max** pooling to r×r | input of `blocks[2]` = output of the second residual block after addition and final ReLU; no immediate upsampling | r = 16 / 24 / 32 over epochs 0–5 / 6–11 / 12–29 (32 = exact bypass) | r = 32 = plain network | same |
| `gaussian_postrelu` (G) | fixed 9×9 support (radius 4), separable depthwise Gaussian, whole-sample reflection padding, exact bypass at 0 | 10 post-ReLU tensors: stem output and the complete output of each of the 9 blocks; the ReLU inside blocks is not a site | g(e) = 1, .85, .70, .60, .50, .40, .30 for three epochs each, 0 from epoch 21 | g = 0 = plain network | same |
| `resolution_max_b1_gaussian_conv` (RG) | R's pooling + the same 9×9 Gaussian | pooling as R; Gaussian at the 19 main-path convolution outputs, before BN | r(e) as R; σ = g(e)·r(e)/32 at **every** site, including those upstream of the reduction | r = 32, g = 0 = plain network | same |
| `cbs_published_schedule` | CBS 3×3 normalised Gaussian, zero padding 1, depthwise (`utils.get_gaussian_filter`) | the same 19 main-path conv outputs, between conv and BN; not shortcuts, block outputs, pooled features or logits | σ(e) = 0.9^⌊e/5⌋: 1, .9, .81, .729, .6561, .59049 for epochs 0–4 … 25–29 | **keeps the epoch-29 filter, σ = 0.59049** | same |
| `cbs_budget_matched` | identical operator and sites | U = 11,730, u_off = ⌊0.7U⌋ = 8,211, K = ⌈log 0.1 / log 0.9⌉ = 22; σ(u) = 0.9^⌊22u/8211⌋ for u < 8,211, exact bypass after | 22 positive plateaus starting at updates ⌈i·8211/22⌉ (0, 374, 747, …, 7,838), bypass from update 8,211 (epoch 21, batch 0) | bypass (unfiltered network) | same |
| `sdpoint` | adaptive **average** pooling to ⌊n·ratio⌉ (`int(round(n*ratio))`) | after the residual addition and **before the final ReLU** of residual block p (9 hook points `blocks.k.post_add`); original strides and global average pooling kept | per logical batch: p ~ U{0,…,9} (p = 0: no downsampling), ratio ~ U{0.5, 0.75}, independent; one choice for all examples and all 4 microbatches of the batch | full-resolution instance (p = 0) | same |

G and RG differ in placement, operator site count and effective σ; they are not a factorial pair, and
G is **not** an implementation of CBS. G is a CBS-inspired variant: annealed Gaussian filtering of
activations was introduced by CBS; our operator (9×9, reflection, post-ReLU sites, plateau schedule
reaching exactly 0) was selected in our own exploration.

Schedule exports: `studies/comparison_cbs_sdpoint/protocol/schedules.json` (every transition
index and σ), regenerated by `comparison/freeze.py`.

## 3. CBS: paper, released code, port

**What the paper specifies.** A Gaussian kernel layer after the output of every convolutional layer,
h = ReLU(pool(G_σ ⊛ (W ⊛ x))) (Eq. 3.1); BN "may also be used"; σ annealed towards 0; for ResNet
variants initial σ = 1, decay factor 0.9 every 5 epochs (Sec. 4, App. C). Kernel size is not stated
in the text.

**What the released code implements.**
- `utils.get_gaussian_filter(kernel_size=3, sigma, channels)`: integer grid → float32, mean 1, variance
  σ², 2-D Gaussian with the 1/(2πσ²) prefactor, divided by its sum, repeated per channel into a
  frozen depthwise `nn.Conv2d(C, C, 3, groups=C, bias=False, padding=1)` (zero padding).
- `resnet.py`: `conv1 → kernel1 → bn1 → relu` in the stem; in each `BasicBlock`,
  `conv1 → kernel1 → bn1 → relu → conv2 → kernel2 → bn2`, `+ shortcut(x)`, `relu`. Shortcut
  convolutions are not filtered.
- `get_new_kernels(epoch)`: at the start of each epoch, `std *= 0.9` when `epoch % 5 == 0 and epoch
  != 0`, and all kernels are rebuilt, i.e. σ(e) = 0.9^⌊e/5⌋ (defaults `--std 1 --std_factor 0.9
  --epoch 5 --kernel_size 3`).
- `solver_cbs.solve`: the kernels are new modules created after the optimizer, so they are never
  optimized; `test()` runs with the current kernels (filtered inference); the solver keeps the model
  with the best test accuracy (`best_model`) and saves it.
- Recipe (not ours): SGD lr 0.1, momentum 0.9, wd 5e-4, lr ÷10 at epochs 30/60/90, batch 64, 200
  epochs by default, inputs normalised with (0.5, 0.5, 0.5); `data.py` applies no augmentation.

**Disagreements and code details, resolved explicitly.**
1. Placement relative to BN: the paper's equation has no BN; the code filters between conv and BN.
   We follow the code (the requested placement).
2. `epoch_count is not 0` relies on small-integer identity; behaviourally equal to `!= 0`. No change.
3. Iterative `std *= 0.9` vs 0.9^k: kernels are bitwise identical for k = 0…5 (test
   `test_cbs_kernel_and_forward_match_the_author_code`); we use the formula.

**Bugs corrected in the port.** None needed for the operator. The best-test-accuracy checkpoint rule
is **not** copied: we evaluate the final prescribed checkpoint.

**Deliberate adaptations.** Our backbone, trainer, recipes, 30-epoch budget, pinned initialization
and data order; the kernel is applied functionally in a forward hook (no parameters, no optimizer
state, no RNG use, no module creation); the budget-matched schedule (§2) is a CBS-derived adaptation
to an update-indexed horizon with an unfiltered final 30%, fixed before any result and not tuned.

**Verified.** Kernels equal the author function bitwise for all 22 σ values used (C = 16, 32, 64);
depthwise forward equals the author `nn.Conv2d` bitwise; the 19 sites reproduce a hand-written
author-style block forward bitwise; σ = 0 reproduces plain logits and gradients exactly; all schedule
boundaries and the 22 + 1 segments; building/applying kernels leaves torch, numpy and python RNG
states unchanged.

## 4. SDPoint: paper, released code, port

**What the paper specifies.** At every training iteration draw a downsampling point p uniformly from
P = {0, 1, …, N} and a ratio r uniformly from R = {0.5, 0.75}; the same instance for every sample of
the mini-batch; p = 0 keeps the original network; for residual networks D_avg (adaptive average
pooling) is applied "right after the residual addition" of block p (Eq. 1, Algorithm 1). Global
average pooling before the classifier absorbs spatial size changes. For CIFAR, points are the
N residual blocks (WRN-d28: N = 12). Inference instances use instance-specific BN statistics
(Sec. 5.1).

**What the released code implements.**
- Blocks get consecutive `blockID`s 0…N−1; the ResNet itself has `blockID = N` (a stem candidate that
  replaces the stem max-pool by adaptive max pooling).
- `BasicBlock`/`Bottleneck.forward`: `out += residual`, then `adaptive_avg_pool2d(out,
  int(round(out.size(2) * ratio)))` if `ratio < 1`, then `relu`. `preresnet.py` pools after the
  addition (pre-activation blocks have no final ReLU).
- `stochastic_downsampling(blockID=None, ratio=None)`:
  `block_chosen = blockID is None and random.randint(-1, self.blockID) or blockID`; ratio
  `[0.5, 0.75][random.randint(0, 1)]`; the loop assigning ratios iterates only over
  `isinstance(m, Bottleneck)` modules.
- `validate` for an instance: `model.train()` forward passes over the **training loader** (with its
  RandomResizedCrop/flip augmentation) under `no_grad`, updating running statistics with the default
  momentum and without reset, then `model.eval()` evaluation. Instances are enumerated for the
  cost–accuracy plot.

**Defects found, checked, and corrected.**
1. **BasicBlocks never downsample.** Because the loop checks only `Bottleneck`, a BasicBlock model
   never downsamples any block (only the stem candidate can fire). Checked on the author
   `resnet18()`: 400 random draws, zero downsampled BasicBlocks
   (`test_author_sdpoint_selection_defects_are_real`). **Port:** all 9 of our BasicBlocks are points.
2. **A drawn 0 becomes None.** `None is None and 0 or None` evaluates to `None`; block 0 can never be
   drawn at random and its mass joins "no downsampling". Checked by forcing `randint` to return 0.
   **Port:** explicit control flow, p = integer in {0..9}, 0 = none.

**Deliberate adaptations.**
- 9 points = the 9 residual blocks of ResNet-20 (paper's CIFAR block-based definition); no stem
  candidate, because our CIFAR stem has no pooling stage (the author stem candidate belongs to the
  ImageNet model). P(p = 0) = 1/10.
- One draw per **logical** optimizer batch, shared by its four microbatches.
- Draws from a counter-based stream `derive_seed(seed, "sdpoint::update::<u>")` (PCG64): independent
  of data order (a precomputed permutation) and of all global RNGs, and identical after resumption.
- Rounding `int(round(n·r))` (Python half-to-even); all products for n ∈ {32, 16, 8}, r ∈ {0.5,
  0.75} are integers, so the convention never binds.
- Primary inference instance fixed in advance: full resolution (p = 0), never chosen on test data.
  The paper's reduced-cost inference instances and its cost–accuracy trade-off are **outside** this
  comparison.

**Verified.** Forced selection reaches all 9 blocks with both ratios (finite forward and backward,
valid residual shapes including stride-2 blocks 4 and 7); the hooked forward equals a hand-written
pool-after-addition-before-ReLU forward bitwise; p = 0 reproduces plain logits and gradients
exactly; over 11,730 updates, point and ratio frequencies and their independence pass χ² tests at
p = 0.001; the trainer applies one state to all microbatches of each update; mid-run resumption is
bitwise.

## 5. Evaluation and BatchNorm (all arms)

| panel | inference state | BN |
|---|---|---|
| **A (primary)** | native (§2) | recalibrated: all 50,000 training images, official file order, batches of 500, reset running mean/var/counters, cumulative average (momentum None), one pass, no gradients; then eval mode |
| **B** | original path: every added intervention disabled (CBS σ = 0, SDPoint p = 0, R r = 32, G/RG g = 0) | the same recalibration; identical to A for every arm except `cbs_published_schedule` |
| saved-statistics diagnostic | native, and original for `cbs_published_schedule` | checkpoint running buffers; for SDPoint a mixture over training instances, reported as a diagnostic only |

Reported: full training set (50,000) and full test set (10,000) accuracy and mean CE. Every policy
starts from the untouched checkpoint tensors; learned tensors are hashed before and after; a repeat
evaluation must reproduce the first. This shared protocol implements instance-specific BN for
SDPoint but differs from the released `validate` (reset, cumulative averaging, no augmentation,
fixed order, batch 500, one pass).

Panel B for CBS measures transfer to the unfiltered network; it does not replace the native result.

## 6. What this comparison cannot establish

Placement, pooling operator and schedule are not isolated; three seeds do not rank small
differences; a weak 30-epoch CBS result does not show CBS is ineffective in general; full-resolution
SDPoint does not evaluate its cost–accuracy trade-off; our historical methods and the AdamW learning
rate were selected with CIFAR-10 test exposure, which this comparison does not undo.
