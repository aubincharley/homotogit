# Protocol and terminology

## Reference recipe (all `training_benchmark` experiments unless the table says otherwise)

CIFAR-10, all 50,000 training images and the official 10,000 test images; no
augmentation. Model: ResNet-20 with BatchNorm and option-A shortcuts (269,722
parameters). SGD, lr 0.005, momentum 0.9, weight decay 5e-4 on all
parameters. 60 warmup updates, then cosine to 0, indexed by global update.
Effective batch 128 as microbatches of 32 weighted by example count. 30 epochs
= 11,730 updates. Per-channel normalisation fitted on the unfiltered training
images. Evaluation on the full test set and a 500-image train probe, every
epoch (`unified`) or every second epoch (`campaign`, `resbench`, ablation).

Earlier pilots used 10,000-image subsets, a 5,000-image validation split,
GroupNorm, or other budgets. They are listed separately and are not comparable.

## Terms

| term | meaning |
|---|---|
| **intervention** | anything that changes the network's computation during training: an internal Gaussian filter, a spatial resolution reduction, a mixture, a wavelet shrinkage |
| **state** | the intervention's setting for an update: resolution `r` and/or Gaussian level `G` (or alpha, or wavelet `s`) |
| **schedule** | state as a function of epoch, piecewise constant; one state per whole epoch |
| **target state** | full resolution, every annealed operator off; the network is then the plain ResNet-20 |
| **current path** | evaluation under the state the last completed updates used |
| **target path** (`bypass32`, `bypassed`) | evaluation of the same weights forced into the target state. A **diagnostic** while an arm is still filtering: BatchNorm running statistics were estimated under the filtered state (errata C-08, C-09) |
| **control** | an arm whose intervention never reaches the target state (constant sigma, fixed 16x16 / 24x24, BlurPool). Its honest final metric is its own inference path, which is not the plain network |
| **site** | a tensor an intervention acts on: `conv_out` = 19 convolution outputs before BN; `post_block` / post-ReLU = stem output + 9 block outputs (10); `post_bn` = 19 BN outputs |
| **reduction point** | where resolution changes: `input` (image, before normalisation), `stem` (input of `blocks`), `D0/D1/D2` = `block0/1/2` (input of `blocks[1/2/3]`, i.e. the complete output of blocks 0/1/2) |
| **plateau** (`Gplateau`) | G = 1.00, 0.85, 0.70, 0.60, 0.50, 0.40, 0.30 for three epochs each, then exactly 0 from epoch 21 |
| **Rprog** | r = 16 for epochs 0–5, 24 for 6–11, 32 from 12 |
| **per-site scaling q** | sigma at a site = `q · G`, `q = r / 32` for all sites after an input or block reduction (and the stem stays at 1 for stem reductions) |
| **asset set** | pinned initial weights and BN buffers, training subset, train-probe indices and per-epoch permutations, identified by sha256 digests |
| **paired** | two arms share an asset set for the same seed, so they see the same initialisation and example order. Pairing does not make runs bitwise reproducible across batches (see AUDIT A2) |
| **valid cell** | final CE and every recorded CE are finite |

## Rules applied to every result

* Final metric = last record. No best-epoch selection, no schedule tuning after
  seeing results.
* The test set has been examined repeatedly. Three seeds give a descriptive
  sample SD; one seed gives none.
* Differences between batches on the same asset set are observations. Their
  cause (GPU kernels, code revisions, library builds) is not established.
* A proposal written in an archived prompt or report is not authorisation to
  run it.
