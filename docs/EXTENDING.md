# Adding a dataset, an architecture or an optimizer

Rule for every transfer: **decisions are explicit and marked unvalidated until
trained.** The core never scales sigma, rescales a resolution schedule,
lengthens or shortens a schedule, relocates an insertion site, or copies a
learning rate from one optimizer to another.

## Dataset

1. Add a loader in `continuation_core/data.py` returning
   `Dataset(name, Split(uint8 NCHW, int64 labels), Split(...), num_classes,
   native_resolution, class_names)` and register it in `DATASETS`.
   `stl10` is an example: it reads the official binaries and converts
   column-major to row-major.
2. Generate an asset set for it, since there is no pinned one:
   ```bash
   py -m continuation_core make-assets --dataset stl10 --data-root data --arch resnet20_bn_cifar \
      --epochs 90 --init-from assets/cifar10_resnet20bn --out assets/stl10_resnet20bn
   ```
   This creates a **new** asset identity (`paired_with_reference: false`) with
   its own digests. Commit its manifest, or keep the files with their hashes.

   `--init-from` copies `init_seed<k>.pt` from an existing set (after a strict
   load into the new model) instead of drawing new weights. ResNet-20's
   parameters do not depend on the input size, so CIFAR-10's initial states load
   unchanged into an STL-10 model: two datasets then differ only in their data.
   It also removes the torch-build dependence — with it, `make-assets` touches
   no torch RNG, only numpy's PCG64, so the whole set is bit-identical anywhere.

   `--epochs` sizes `perm_seed<k>` as `[epochs, n_train]` and `Trainer` only
   requires `shape[0] >= budget.epochs`, so generate with headroom: regenerating
   later to lengthen a run changes the `.npz` digest and voids pairing with the
   runs already done.
3. Normalisation defaults to per-channel statistics fitted on the training
   images. Pin `expected_mean` / `expected_std` once they are known.

## Architecture

1. Add `continuation_core/models/<name>.py` with `build(num_classes,
   in_channels, **options)` and a `SITE_MAP`:
   ```python
   SITE_MAP = {
       "conv_out":  ["features.0", ...],                  # forward-hook outputs, in order
       "post_relu": [("output", "features.2"), ...],      # ("input"|"output", module name)
       "reduction": {"after_pool1": ("input", "features.4")},   # named points, may be {}
       "reference_resolution": 32,
   }
   ```
2. Register it in `continuation_core/models/__init__.py` with
   `"validated": False`.
3. Add a test that every name resolves and that the target state equals the
   plain model bitwise (copy `tests/test_models_and_sites.py`).

`vgg11_bn` is the non-ResNet example. It supports `conv_out` and `post_relu`
(8 sites each) and has **no** reduction points. `resolution_max_b1` on it
raises `UnsupportedInsertionError: vgg11_bn has no reduction point 'block1'`.
To study resolution on VGG, name a point explicitly and write a new method
preset. Do not reuse the `block1` label.

## Optimizer

`OptimizerConfig(name="sgd" | "adamw", lr=..., weight_decay=...)`. `lr` and
`weight_decay` have no defaults, so an AdamW config without them fails. The
warmup/cosine schedule is shared machinery, but its `warmup_updates` and
`min_lr` belong to each config. To add another optimizer, extend
`build_optimizer` in `continuation_core/optim.py` and give it its own required
fields.

## Transfer presets

`continuation_core.presets.transfer` requires, for a method that has them:

| decision | argument |
|---|---|
| resolution schedule and its reference size | `resolution_schedule`, `reference_resolution` |
| Gaussian schedule and units | `gaussian_schedule`, `gaussian_units` |
| schedule duration | `epochs` plus the schedules' own boundaries |
| insertion mapping | `insertion_mapping_note` |
| optimizer | `optimizer` (with its own lr / weight decay) |

"Keep the same numbers" is already a choice: pixel sigma versus relative sigma,
ratio versus absolute resolution, epochs versus updates. Record which one you
made. The resulting config has `validation_status: "unvalidated"`, and its
method id ends in `__transfer`.

### The STL-10 transfer, as taken

`scripts/stl10_configs.py` is the one place these decisions live, and it writes
them into every `configs/stl10/*.json`. Untrained at the time of writing.

| decision | value | why |
|---|---|---|
| resolution | `EpochSchedule((0, 12, 24), (48, 72, 96))`, `reference_resolution=96` | `block1` is the input of `blocks.2`, which runs at the **input** resolution, so the reference `(16, 24, 32)` would be a 6× reduction on a 96×96 image rather than a 2× one. These keep the reference *ratios* 1/2, 3/4, 1 |
| Gaussian | the reference levels ×3: `(3.00, 2.55, 2.10, 1.80, 1.50, 1.20, 0.90, 0)` | sigma is in pixels of the feature map at the site, and STL feature maps are 3× wider (96/48/24 against 32/16/8), so the reference sigma would blur a third as much of the image |
| duration | 60 epochs, every boundary ×2 | STL-10 has 5,000 training images, so an epoch is 40 updates against CIFAR-10's 391. The *proportion* of the run spent at each level is unchanged |
| optimizer | the reference recipe unchanged | changing the setting and the optimizer at once would confound. `warmup_updates=60` is now 1.5 epochs, not 0.15 |
| insertion | `block1`, unchanged | |

Two consequences of the σ ×3 worth knowing before you run it:

* `transfer` derives `sigma_max = max(1.0, max(schedule.values)) = 3.0` from the
  **unscaled** schedule and leaves `truncate` at 4.0, so the fixed support
  becomes `radius = 12`, **25 taps** for every sigma (CIFAR-10: radius 4, 9
  taps). That is the reference convention — support fixed by `sigma_max`, not by
  the largest sigma actually used — so the combined method, whose largest
  effective sigma is 1.575, carries about twice the kernel it needs.
* the smallest map a Gaussian site sees drops to **12×12** (`blocks.6-8` conv
  outputs at `r=48`), where `pad == n` and `reflect_pad_axis` takes its explicit
  index-gather branch. This is the exact analogue of CIFAR-10's 4×4 maps at
  `r=16`, and `tests/test_stl10_transfer.py` pins it.

Effective per-site sigma for `resolution_max_b1_gaussian_conv` is then exactly
3× the CIFAR-10 table in [METHODS.md](METHODS.md), non-monotonicity included:
1.500, 1.275, **1.575**, 1.350, 1.500, 1.200, 0.900, 0.

Evaluation is every third epoch, and `plain` evaluates only the `current` path.
One STL-10 snapshot costs about 7.3 CIFAR-10 snapshots — the test set shrinks by
a fifth while each image costs 9× — so per-epoch evaluation of both paths would
cost more than the training it measures.
