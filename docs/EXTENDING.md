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
   py -m continuation_core make-assets --dataset stl10 --data-root data --arch resnet20_bn_cifar --epochs 30 --out assets/stl10_resnet20bn
   ```
   This creates a **new** asset identity (`paired_with_reference: false`) with
   its own digests. Commit its manifest, or keep the files with their hashes.
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

| decision | argument | example (STL-10, ResNet-20) |
|---|---|---|
| resolution schedule and its reference size | `resolution_schedule`, `reference_resolution` | `EpochSchedule((0, 6, 12), (48, 72, 96))`, `96` keeps the reference **ratios** 1/2, 3/4, 1 |
| Gaussian schedule and units | `gaussian_schedule`, `gaussian_units` | `PLATEAU_G`, `"pixels of the feature map at the site"` keeps the reference **pixel** values |
| schedule duration | `epochs` plus the schedules' own boundaries | 30 epochs, same boundaries, i.e. 5,000 images × 30 / 128 = 1,200 updates, **10× fewer** than the reference |
| insertion mapping | `insertion_mapping_note` | "input of blocks.2, as in the reference" |
| optimizer | `optimizer` (with its own lr / weight decay) | must be chosen |

The example column shows that "keep the same numbers" already means choices:
pixel sigma versus relative sigma, ratio versus absolute resolution, epochs
versus updates. Record which one you made. The resulting config has
`validation_status: "unvalidated"`, and its method id ends in `__transfer`.
