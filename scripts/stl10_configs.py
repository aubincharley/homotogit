"""Write the STL-10 transfer configs for the four frozen methods.

The core never scales a schedule (`docs/EXTENDING.md`), so every decision that
turns a 32x32 method into a 96x96 one is made *here*, in caller code, and is
visible in the JSON that ends up next to every run.

Decisions, and why
------------------
``resolution``  ``(48, 72, 96)`` with ``reference_resolution = 96``.  The
                reduction point ``block1`` is the input of ``blocks.2``, which
                runs at the input resolution, so the reference ``(16, 24, 32)``
                on a 96x96 image would be a 6x reduction rather than a 2x one.
                These values keep the reference *ratios* 1/2, 3/4, 1.
``gaussian``    the reference levels x3.  Sigma is in pixels of the feature map
                at the site, and STL feature maps are 3x wider (96/48/24 against
                32/16/8), so the reference sigma would blur a third as much of
                the image.  ``presets.transfer`` then derives
                ``sigma_max = 3.0`` from these values and, with ``truncate``
                left at 4.0, a fixed radius of 12 (25 taps, against 4 and 9 on
                CIFAR-10).
``duration``    60 epochs, every schedule boundary doubled.  STL-10 has 5,000
                training images, so an epoch is 40 updates against CIFAR-10's
                391; 60 epochs is 2,400 updates.  The proportions of the run
                spent at each level are unchanged.
``optimizer``   the reference recipe, untouched.  Changing the setting and the
                optimizer at once would confound the comparison.  Note that
                ``warmup_updates = 60`` is now 1.5 epochs rather than 0.15.
``evaluation``  every third epoch.  One STL-10 snapshot costs about 7.3 CIFAR-10
                snapshots (the test set shrinks by a fifth while each image
                costs 9x), so per-epoch evaluation would cost more than the
                training it measures.  For ``plain`` the current and target
                paths are the same state, so only ``current`` is evaluated.

Nothing here is validated: every config comes out ``unvalidated`` with a method
id ending in ``__transfer``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from continuation_core.config import OptimizerConfig                  # noqa: E402
from continuation_core.methods import get_method                      # noqa: E402
from continuation_core.presets import transfer                        # noqa: E402
from continuation_core.schedules import EpochSchedule                 # noqa: E402

METHODS = ("plain", "resolution_max_b1", "gaussian_postrelu",
           "resolution_max_b1_gaussian_conv")

EPOCHS = 60
NATIVE_RESOLUTION = 96
SIGMA_SCALE = 3.0                      # 96 / 32, the feature-map width ratio
BOUNDARY_SCALE = 2                     # 60 epochs instead of 30

#: G(e) = the reference plateau x SIGMA_SCALE, boundaries x BOUNDARY_SCALE.
STL_G = EpochSchedule(tuple(b * BOUNDARY_SCALE for b in (0, 3, 6, 9, 12, 15, 18, 21)),
                      (3.00, 2.55, 2.10, 1.80, 1.50, 1.20, 0.90, 0.0))

#: r(e) at block1: half, three quarters, then native (an exact bypass).
STL_R = EpochSchedule(tuple(b * BOUNDARY_SCALE for b in (0, 6, 12)), (48, 72, 96))

OPTIMIZER = dict(name="sgd", lr=0.005, weight_decay=5e-4, momentum=0.9,
                 nesterov=False, schedule="warmup_cosine", warmup_updates=60,
                 min_lr=0.0)

GAUSSIAN_UNITS = ("pixels of the feature map at the site; the reference levels "
                  "scaled by 96/32 so the blur covers the same fraction of the "
                  "image extent as on CIFAR-10")
INSERTION_NOTE = ("input of blocks.2, as in the reference; on a 96x96 image that "
                  "site runs at 96, so reference_resolution is 96 and not 32")

#: STL-10 has no pinned statistics yet.  Filled in by --data-root, or left None,
#: in which case build_pipeline fits them without a check.
STL10_MEAN = None
STL10_STD = None


def _stats(data_root):
    """Per-channel statistics of the STL-10 training images, with a size check."""
    from continuation_core.config import DataConfig
    from continuation_core.data import channel_stats, load_dataset
    ds = load_dataset(DataConfig(name="stl10", root=str(data_root)))
    shape = tuple(ds.train.images.shape)
    if shape != (5000, 3, 96, 96):
        raise SystemExit("STL-10 train split is %s, expected (5000, 3, 96, 96); the "
                         "loader only checks that the byte count divides by 3*96*96, "
                         "so this is probably a truncated or wrong file" % (shape,))
    if tuple(ds.test.images.shape) != (8000, 3, 96, 96):
        raise SystemExit("STL-10 test split is %s, expected (8000, 3, 96, 96)"
                         % (tuple(ds.test.images.shape),))
    mean, std = channel_stats(ds.train.images)
    return [float(v) for v in mean], [float(v) for v in std]


def build(method_id: str, seed: int, *, data_root: str, assets_dir: str,
          out_dir: str, mean=None, std=None):
    base = get_method(method_id)          # the registry, not the id string, decides
    has_gaussian = base.gaussian is not None
    has_resolution = base.resolution is not None
    cfg = transfer(
        method_id, dataset="stl10", data_root=data_root, arch="resnet20_bn_cifar",
        optimizer=OptimizerConfig(**OPTIMIZER), epochs=EPOCHS,
        resolution_schedule=STL_R if has_resolution else None,
        reference_resolution=NATIVE_RESOLUTION if has_resolution else None,
        gaussian_schedule=STL_G if has_gaussian else None,
        gaussian_units=GAUSSIAN_UNITS if has_gaussian else None,
        insertion_mapping_note=INSERTION_NOTE,
        assets_dir=assets_dir, seed=seed, out_dir=out_dir)

    # transfer() copies the base description verbatim, and it still quotes the
    # CIFAR-10 numbers (16/24/32, r/32).  Say what this method actually does.
    cfg.method["description"] = base.description.replace("16 / 24 / 32", "48 / 72 / 96") \
                                                .replace("r(e)/32", "r(e)/96")

    # transfer() covers the method and the optimizer; these sections keep their
    # dataclass defaults, which are tuned for 32x32 and 50,000 images.
    cfg.data.expected_mean, cfg.data.expected_std = mean, std
    cfg.evaluation.every_epochs = 3
    if method_id == "plain":
        # current == target at every epoch, so the default ("current", "target")
        # would evaluate 8,500 images twice for the same number.
        cfg.evaluation.paths = ("current",)
    # metrics.json / summary.json / rolling.pt only: per-epoch checkpoints are
    # ~300 MB per run and are only needed for the landscape analysis.
    cfg.checkpoint.every_epoch = False
    cfg.checkpoint.keep_rolling = True
    cfg.notes.append("sigma x%g and resolution x%g from the CIFAR-10 reference; "
                     "schedule boundaries x%d for %d epochs"
                     % (SIGMA_SCALE, NATIVE_RESOLUTION / 32, BOUNDARY_SCALE, EPOCHS))
    cfg.notes.append("normalisation statistics "
                     + ("pinned from the training images" if mean else "NOT pinned"))
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default="data",
                    help="if STL-10 is readable here, pin its normalisation statistics")
    ap.add_argument("--no-stats", action="store_true",
                    help="skip the statistics, even if the data is present")
    ap.add_argument("--assets", default="assets/stl10_resnet20bn")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", default="configs/stl10")
    ap.add_argument("--run-out", default="runs")
    args = ap.parse_args(argv)

    mean, std = STL10_MEAN, STL10_STD
    if not args.no_stats and mean is None:
        try:
            mean, std = _stats(args.data_root)
            print("STL-10 statistics: mean %s std %s" % (mean, std))
        except FileNotFoundError as exc:
            print("no STL-10 under %s (%s); leaving the statistics unpinned"
                  % (args.data_root, exc))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in (int(s) for s in args.seeds.split(",")):
        for method_id in METHODS:
            cfg = build(method_id, seed, data_root=args.data_root,
                        assets_dir=args.assets, out_dir=args.run_out,
                        mean=mean, std=std)
            path = out / ("%s__seed%d.json" % (method_id, seed))
            cfg.save(path)
            print("wrote %s  (%s)" % (path, cfg.run.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
