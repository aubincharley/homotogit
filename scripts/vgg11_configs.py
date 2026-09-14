"""VGG-11-BN on CIFAR-10: the same four methods on a non-residual network.

Everything about the *data and recipe* is the CIFAR-10 reference, unchanged --
all 50,000 training images, 391 updates per epoch, 30 epochs, SGD lr 0.005, and
the reference schedules, because this is still 32x32.  The architecture is the
only thing that differs, so this asks whether the four methods need residual
connections.

Same images, same order
-----------------------
``make-assets --indices-from assets/cifar10_resnet20bn`` gives these runs the
reference campaign's exact ``subset``, ``train_probe`` and ``perm_seed<k>``, so
VGG-11 sees the same images in the same order as the ResNet-20 numbers it is
compared against.  Only the initial weights are new -- they have to be, the
parameter shapes differ -- so ``--init-from`` correctly refuses.

The sites are not the same sites
--------------------------------
They cannot be.  ``gaussian_postrelu`` is 10 post-ReLU tensors on ResNet-20 and
**8** here; ``conv_out`` is 19 there and **8** here.  The operators are identical
and the placement rule is identical ("every convolution output", "every
post-ReLU tensor"), but any cross-architecture comparison is confounded by site
count.  The within-architecture comparison against this model's own control is
clean; the cross-architecture one is suggestive.

The reduction point, and a caveat recorded before the runs
----------------------------------------------------------
``block1`` is a ResNet-20 location and is not mapped here.  VGG-11 names
``after_pool1`` -- the input of ``features.4``, one convolution upstream and
seven downstream -- whose native size is **16**, half the image, because a max
pool has already run.  The schedule is therefore ``(8, 12, 16)`` against
``reference_resolution=16``: the reference ratios 1/2, 3/4, 1.

**Four max pools follow that point, where ResNet-20 has two stride-2 steps.** So
the reduction bites far harder here:

    r     features.4-6   8-13   15-20   22-27
    8         8            4      2       1
    12       12            6      3       1
    16       16            8      4       2   (native, exact bypass)

Any ``r < 16`` puts the last two convolutions at **1x1**, where both the
convolution and the Gaussian are degenerate, and no choice of point or schedule
avoids it -- VGG-11 on a 32x32 image is already spatially exhausted at 2x2.  The
target state is still bitwise the plain network and the last 18 epochs run at
native resolution, so the homotopy is intact.  But if the resolution-bearing
methods underperform here, that 1x1 tail is the first thing to suspect, and it is
a fact about this architecture on small images rather than about the method.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from continuation_core.config import OptimizerConfig                       # noqa: E402
from continuation_core.methods import PLATEAU_G, get_method                # noqa: E402
from continuation_core.presets import CIFAR10_MEAN, CIFAR10_STD, transfer  # noqa: E402
from continuation_core.schedules import EpochSchedule                      # noqa: E402

METHODS = ("plain", "resolution_max_b1", "gaussian_postrelu",
           "resolution_max_b1_gaussian_conv")

ARCH = "vgg11_bn"
EPOCHS = 30                     # the reference budget, 391 upd/epoch, 11,730 total
REDUCTION_POINT = "after_pool1"
NATIVE_RESOLUTION = 16          # at this point, for a 32x32 image

#: the reference Gaussian levels, unchanged -- this is still 32x32
VGG_G = PLATEAU_G
#: the reference *ratios* 1/2, 3/4, 1 against a native 16
VGG_R = EpochSchedule((0, 6, 12), (8, 12, 16))

OPTIMIZER = dict(name="sgd", lr=0.005, weight_decay=5e-4, momentum=0.9,
                 nesterov=False, schedule="warmup_cosine", warmup_updates=60,
                 min_lr=0.0)

GAUSSIAN_UNITS = ("pixels of the feature map at the site; the reference levels "
                  "unchanged, since VGG-11 here is 32x32 like the reference")
INSERTION_NOTE = (
    "reduction at 'after_pool1' = input of features.4, native 16 (a max pool has "
    "already halved the image); 'block1' is a ResNet-20 name and is not mapped. "
    "Four max pools follow, against ResNet-20's two stride-2 steps, so r < 16 "
    "puts features.22-27 at 1x1. Gaussian sites are 8 here, not 10 or 19.")


def build(method_id: str, seed: int, *, data_root: str, assets_dir: str, out_dir: str):
    base = get_method(method_id)
    cfg = transfer(
        method_id, dataset="cifar10", data_root=data_root, arch=ARCH,
        optimizer=OptimizerConfig(**OPTIMIZER), epochs=EPOCHS,
        resolution_schedule=VGG_R if base.resolution else None,
        reference_resolution=NATIVE_RESOLUTION if base.resolution else None,
        resolution_point=REDUCTION_POINT if base.resolution else None,
        gaussian_schedule=VGG_G if base.gaussian else None,
        gaussian_units=GAUSSIAN_UNITS if base.gaussian else None,
        insertion_mapping_note=INSERTION_NOTE,
        assets_dir=assets_dir, seed=seed, out_dir=out_dir)

    # unchanged from the reference: same data, same pipeline, same statistics
    cfg.data.expected_mean, cfg.data.expected_std = CIFAR10_MEAN, CIFAR10_STD
    if method_id == "plain":
        cfg.evaluation.paths = ("current",)
    cfg.checkpoint.every_epoch = False
    cfg.checkpoint.keep_rolling = True
    cfg.run.name = "%s__vgg11_cifar10__seed%d" % (method_id, seed)
    cfg.notes.append("architecture is the only change from the reference recipe: "
                     "same 50,000 images, same order, same optimizer, same schedules")
    cfg.notes.append("VGG-11-BN has 8 convolutions, so conv_out is 8 sites (19 on "
                     "ResNet-20) and post_relu is 8 (10 on ResNet-20)")
    if base.resolution:
        cfg.notes.append("r < 16 puts features.22-27 at 1x1; see "
                         "scripts/vgg11_configs.py for why no schedule avoids it")
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--assets", default="assets/vgg11_cifar10")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", default="configs/vgg11_cifar10")
    ap.add_argument("--run-out", default="runs")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in (int(s) for s in args.seeds.split(",")):
        for method_id in METHODS:
            cfg = build(method_id, seed, data_root=args.data_root,
                        assets_dir=args.assets, out_dir=args.run_out)
            path = out / ("%s__seed%d.json" % (method_id, seed))
            cfg.save(path)
            print("wrote %s  (%s)" % (path, cfg.run.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
