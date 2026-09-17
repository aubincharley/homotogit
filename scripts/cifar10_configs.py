"""CIFAR-10 on the reference recipe, as a study module the optimizer swap can use.

The reference campaign is the benchmark's baseline, and it is the one dataset
with no optimizer arm: every recorded CIFAR-10 run uses SGD lr 0.005.  This
module exposes that recipe with the same ``build(...)`` signature as
``svhn_configs`` and ``stl10_configs``, so ``optimizer_configs`` can override the
optimizer and leave everything else alone.

Nothing here is rescaled and nothing is chosen: 32x32, all 50,000 images in the
reference's own pinned order, the reference schedules, 30 epochs, 391 updates per
epoch.  The configs it produces under SGD are the reference recipe; what makes
them ``unvalidated`` is only that they are rebuilt through ``transfer`` rather
than being the frozen ``reference`` preset.

It uses the **pinned reference asset set directly** -- same initial weights, same
data order -- because the architecture and the data are the reference's own.  No
new asset set exists and none is needed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from continuation_core.config import OptimizerConfig                       # noqa: E402
from continuation_core.methods import PLATEAU_G, RPROG, get_method         # noqa: E402
from continuation_core.presets import CIFAR10_MEAN, CIFAR10_STD, transfer  # noqa: E402

METHODS = ("plain", "resolution_max_b1", "gaussian_postrelu",
           "resolution_max_b1_gaussian_conv")

ARCH = "resnet20_bn_cifar"
EPOCHS = 30                      # 391 updates/epoch, 11,730 total
NATIVE_RESOLUTION = 32
REFERENCE_ASSETS = "assets/cifar10_resnet20bn"

#: the reference schedules, verbatim
CIFAR_G = PLATEAU_G
CIFAR_R = RPROG

#: the frozen recipe's own optimizer; optimizer_configs replaces this
OPTIMIZER = dict(name="sgd", lr=0.005, weight_decay=5e-4, momentum=0.9,
                 nesterov=False, schedule="warmup_cosine", warmup_updates=60,
                 min_lr=0.0)

INSERTION_NOTE = "input of blocks.2, exactly as in the reference"


def _stats(data_root):
    """The pinned CIFAR-10 statistics.

    Signature matches the other study modules so `optimizer_configs` can call it
    uniformly.  These are the values every reference run checked its own fit
    against to 1e-7, so they are returned rather than refitted.
    """
    return list(CIFAR10_MEAN), list(CIFAR10_STD)


def build(method_id: str, seed: int, *, data_root: str, assets_dir: str,
          out_dir: str, mean=None, std=None):
    base = get_method(method_id)
    cfg = transfer(
        method_id, dataset="cifar10", data_root=data_root, arch=ARCH,
        optimizer=OptimizerConfig(**OPTIMIZER), epochs=EPOCHS,
        resolution_schedule=CIFAR_R if base.resolution else None,
        reference_resolution=NATIVE_RESOLUTION if base.resolution else None,
        gaussian_schedule=CIFAR_G if base.gaussian else None,
        gaussian_units=("pixels of the feature map at the site; the reference "
                        "levels unchanged") if base.gaussian else None,
        insertion_mapping_note=INSERTION_NOTE,
        assets_dir=assets_dir, seed=seed, out_dir=out_dir)

    cfg.data.expected_mean = mean if mean is not None else list(CIFAR10_MEAN)
    cfg.data.expected_std = std if std is not None else list(CIFAR10_STD)
    if method_id == "plain":
        cfg.evaluation.paths = ("current",)
    cfg.checkpoint.every_epoch = False
    cfg.checkpoint.keep_rolling = True
    cfg.run.name = "%s__cifar10__seed%d" % (method_id, seed)
    cfg.notes.append("the reference recipe rebuilt through transfer so the "
                     "optimizer can be swapped; schedules, data and assets are "
                     "the reference's own")
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--assets", default=REFERENCE_ASSETS)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", default="configs/cifar10")
    ap.add_argument("--run-out", default="runs")
    ap.add_argument("--no-stats", action="store_true")
    args = ap.parse_args(argv)

    mean, std = (None, None) if args.no_stats else _stats(args.data_root)
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
