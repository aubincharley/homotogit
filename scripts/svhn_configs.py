"""Write the SVHN transfer configs for the four frozen methods.

SVHN is 32x32, so unlike STL-10 **nothing is rescaled**: the resolution schedule,
the Gaussian levels, the placements and the optimizer are the reference values,
passed through `presets.transfer` unchanged.  That is the experiment.  STL-10
moved image size, dataset size, update budget and task at once, so it could not
say which mattered.  Here every one of those is held at the CIFAR-10 value and
only the image statistics change.

Why it is worth running
-----------------------
CIFAR-10 and STL-10 are both natural images, where a Gaussian anneal is a
coarse-to-fine prior over real scale structure.  SVHN is centred, cropped house
numbers: flat regions, sharp strokes, weak scale structure, and distractor digits
at the edges of many crops.  If the Gaussian methods help here as much as on
CIFAR-10, the mechanism is not about natural-image scale structure and is better
described as optimization smoothing.  If they help less while
``resolution_max_b1`` still helps, that separates the two families.

The one decision that is not "unchanged"
----------------------------------------
SVHN has 73,257 training images against CIFAR-10's 50,000, which at batch 128
would be 573 updates per epoch instead of 391 -- a 47% larger optimization budget
over the same 30 epochs, and the schedules are indexed by epoch.  The training
set is therefore cut to 50,000 (``--subset-size``), so updates per epoch, total
updates, and every schedule boundary in updates match the reference exactly.
Recorded here rather than inferred.  Using all 73,257 is a defensible
alternative; it just is not the same experiment.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from continuation_core.config import OptimizerConfig                  # noqa: E402
from continuation_core.methods import PLATEAU_G, RPROG, get_method    # noqa: E402
from continuation_core.presets import transfer                        # noqa: E402

METHODS = ("plain", "resolution_max_b1", "gaussian_postrelu",
           "resolution_max_b1_gaussian_conv")

EPOCHS = 30                    # the reference budget
SUBSET_SIZE = 50000            # = CIFAR-10, so 391 updates/epoch and 11,730 total
NATIVE_RESOLUTION = 32

#: the reference schedules, verbatim -- 32x32 needs no scaling
SVHN_G = PLATEAU_G
SVHN_R = RPROG

OPTIMIZER = dict(name="sgd", lr=0.005, weight_decay=5e-4, momentum=0.9,
                 nesterov=False, schedule="warmup_cosine", warmup_updates=60,
                 min_lr=0.0)

GAUSSIAN_UNITS = ("pixels of the feature map at the site; the reference levels "
                  "unchanged, since SVHN is 32x32 like CIFAR-10")
INSERTION_NOTE = ("input of blocks.2, as in the reference; SVHN is 32x32 so the "
                  "site runs at 32 and reference_resolution stays 32")

#: fitted on the 50,000-image subset by --data-root; None until then
SVHN_MEAN = None
SVHN_STD = None


def _stats(data_root):
    """Per-channel statistics of the **whole** training split.

    Not of the 50,000-image subset: ``build_pipeline`` fits on
    ``dataset.train.images`` before any subsetting (``data.py``), which is the
    documented "unfiltered training images" rule.  On CIFAR-10 the subset is the
    whole split so the two coincide; here they do not, and pinning the subset's
    statistics makes every run fail its own normalisation check.  So the
    normalisation sees 73,257 images while training sees 50,000 of them.
    """
    from continuation_core.config import DataConfig
    from continuation_core.data import channel_stats, load_dataset
    ds = load_dataset(DataConfig(name="svhn", root=str(data_root)))
    if tuple(ds.train.images.shape) != (73257, 3, 32, 32):
        raise SystemExit("SVHN train split is %s, expected (73257, 3, 32, 32)"
                         % (tuple(ds.train.images.shape),))
    if tuple(ds.test.images.shape) != (26032, 3, 32, 32):
        raise SystemExit("SVHN test split is %s, expected (26032, 3, 32, 32)"
                         % (tuple(ds.test.images.shape),))
    mean, std = channel_stats(ds.train.images)
    return [float(v) for v in mean], [float(v) for v in std]


def build(method_id: str, seed: int, *, data_root: str, assets_dir: str,
          out_dir: str, mean=None, std=None):
    base = get_method(method_id)
    cfg = transfer(
        method_id, dataset="svhn", data_root=data_root, arch="resnet20_bn_cifar",
        optimizer=OptimizerConfig(**OPTIMIZER), epochs=EPOCHS,
        resolution_schedule=SVHN_R if base.resolution else None,
        reference_resolution=NATIVE_RESOLUTION if base.resolution else None,
        gaussian_schedule=SVHN_G if base.gaussian else None,
        gaussian_units=GAUSSIAN_UNITS if base.gaussian else None,
        insertion_mapping_note=INSERTION_NOTE,
        assets_dir=assets_dir, seed=seed, out_dir=out_dir)

    cfg.data.expected_mean, cfg.data.expected_std = mean, std
    if method_id == "plain":
        # current == target at every epoch; the default would evaluate twice
        cfg.evaluation.paths = ("current",)
    cfg.checkpoint.every_epoch = False
    cfg.checkpoint.keep_rolling = True
    cfg.notes.append("schedules, placements and optimizer identical to the "
                     "CIFAR-10 reference; SVHN is 32x32 so nothing is rescaled")
    cfg.notes.append("train set cut to %d images so updates/epoch match CIFAR-10's "
                     "391 (SVHN has 73,257)" % SUBSET_SIZE)
    cfg.notes.append("normalisation fitted on all 73,257 training images (the "
                     "pipeline is built before subsetting), while training uses "
                     "%d of them" % SUBSET_SIZE)
    cfg.notes.append("normalisation statistics "
                     + ("pinned" if mean else "NOT pinned"))
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--no-stats", action="store_true")
    ap.add_argument("--assets", default="assets/svhn_resnet20bn")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", default="configs/svhn")
    ap.add_argument("--run-out", default="runs")
    args = ap.parse_args(argv)

    mean, std = SVHN_MEAN, SVHN_STD
    if not args.no_stats and mean is None:
        try:
            mean, std = _stats(args.data_root)
            print("SVHN statistics (all 73,257 training images): mean %s std %s"
                  % (mean, std))
        except FileNotFoundError as exc:
            print("no SVHN under %s (%s); leaving the statistics unpinned"
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
