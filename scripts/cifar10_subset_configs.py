"""CIFAR-10 cut to 5,000 images: the control STL-10 is missing.

STL-10 moved three things at once against the CIFAR-10 reference -- image size
32 -> 96, training set 50,000 -> 5,000, and updates per epoch 391 -> 40 -- and
``resolution_max_b1_gaussian_conv`` went from best method to below the control.
SVHN then reproduced the CIFAR-10 ordering exactly at 32x32 with the full update
budget, so STL-10 is the outlier; but its three changes are still confounded.

This holds the image size at 32x32 and the training set at 5,000, and varies only
the schedule, in two arms:

``stl_budget``          60 epochs, STL-10's boundaries.  2,400 updates, and 720
                        of them after the Gaussian switches off -- identical to
                        STL-10.  Differs from STL-10 *only* in resolution.
``reference_updates``   293 epochs, boundaries scaled by 391/40 so every
                        transition lands on the reference's update count.
                        11,720 updates and 3,520 after G -> 0, against the
                        reference's 11,730 and 3,519.  Differs from CIFAR-10
                        *only* in how many distinct images those updates see.

Reading the result
------------------
=========================  ==========================================
both arms work             STL-10's failure was the 96x96 resolution
both arms fail             it was data scarcity; the recovery-window
                           hypothesis is wrong
stl_budget fails,          it was the recovery window, as proposed --
reference_updates works    720 updates is not enough to undo the blur
=========================  ==========================================

Sigma and the resolution schedule keep their **reference values** (1.00..0.30 and
16/24/32 with reference 32) because this is 32x32.  Only the timing differs, so
against STL-10 the intervention is the same one in relative terms.
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
from continuation_core.schedules import EpochSchedule                 # noqa: E402

METHODS = ("plain", "resolution_max_b1", "gaussian_postrelu",
           "resolution_max_b1_gaussian_conv")

SUBSET_SIZE = 5000             # = STL-10's training set
NATIVE_RESOLUTION = 32
UPDATES_PER_EPOCH = 40         # ceil(5000 / 128); the reference is 391

#: Both arms keep the reference sigma levels and resolution values -- this is
#: 32x32, so nothing needs rescaling.  Only the boundaries move.
ARMS = {
    "stl_budget": {
        "epochs": 60,
        "g_starts": (0, 6, 12, 18, 24, 30, 36, 42),      # STL-10's, x2
        "r_starts": (0, 12, 24),
        "every_epochs": 3,                                # as STL-10 recorded it
        "note": "STL-10's schedule at 32x32: 2,400 updates, 720 after G -> 0",
    },
    "reference_updates": {
        "epochs": 293,                                    # round(30 * 391/40)
        "g_starts": (0, 29, 59, 88, 117, 147, 176, 205),  # round(e * 391/40)
        "r_starts": (0, 59, 117),
        "every_epochs": 10,                               # ~30 snapshots
        "note": "boundaries on the reference's update counts: 11,720 updates, "
                "3,520 after G -> 0, against 11,730 and 3,519",
    },
}

OPTIMIZER = dict(name="sgd", lr=0.005, weight_decay=5e-4, momentum=0.9,
                 nesterov=False, schedule="warmup_cosine", warmup_updates=60,
                 min_lr=0.0)


def schedules(arm: str):
    a = ARMS[arm]
    return (EpochSchedule(a["g_starts"], PLATEAU_G.values),
            EpochSchedule(a["r_starts"], RPROG.values))


def recovery_updates(arm: str) -> int:
    """Updates left after the Gaussian reaches zero -- the quantity in question."""
    a = ARMS[arm]
    return (a["epochs"] - a["g_starts"][-1]) * UPDATES_PER_EPOCH


def build(method_id: str, seed: int, arm: str, *, data_root: str, assets_dir: str,
          out_dir: str):
    a = ARMS[arm]
    base = get_method(method_id)
    g, r = schedules(arm)
    cfg = transfer(
        method_id, dataset="cifar10", data_root=data_root, arch="resnet20_bn_cifar",
        optimizer=OptimizerConfig(**OPTIMIZER), epochs=a["epochs"],
        resolution_schedule=r if base.resolution else None,
        reference_resolution=NATIVE_RESOLUTION if base.resolution else None,
        gaussian_schedule=g if base.gaussian else None,
        gaussian_units=("pixels of the feature map at the site; the reference "
                        "levels unchanged, since this is 32x32")
                       if base.gaussian else None,
        insertion_mapping_note="input of blocks.2, exactly as in the reference",
        assets_dir=assets_dir, seed=seed, out_dir=out_dir)

    # the pinned CIFAR-10 statistics still apply: the pipeline fits on the whole
    # 50,000-image split before subsetting, exactly as the reference does
    from continuation_core.presets import CIFAR10_MEAN, CIFAR10_STD
    cfg.data.expected_mean, cfg.data.expected_std = CIFAR10_MEAN, CIFAR10_STD
    cfg.evaluation.every_epochs = a["every_epochs"]
    if method_id == "plain":
        cfg.evaluation.paths = ("current",)
    cfg.checkpoint.every_epoch = False
    cfg.checkpoint.keep_rolling = True
    cfg.run.name = "%s__cifar10_5k_%s__seed%d" % (method_id, arm, seed)
    cfg.notes.append("arm %s: %s" % (arm, a["note"]))
    cfg.notes.append("training set cut to %d of 50,000, the STL-10 size; "
                     "normalisation still fitted on all 50,000 (the pipeline is "
                     "built before subsetting), as in the reference" % SUBSET_SIZE)
    cfg.notes.append("sigma and resolution keep their reference values; only the "
                     "schedule boundaries differ between arms")
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--assets", default="assets/cifar10_5k_resnet20bn")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", default="configs/cifar10_5k")
    ap.add_argument("--run-out", default="runs")
    args = ap.parse_args(argv)

    a = ARMS[args.arm]
    print("arm %s: %d epochs x %d updates = %d updates, %d after G -> 0"
          % (args.arm, a["epochs"], UPDATES_PER_EPOCH,
             a["epochs"] * UPDATES_PER_EPOCH, recovery_updates(args.arm)))
    out = Path(args.out) / args.arm
    out.mkdir(parents=True, exist_ok=True)
    for seed in (int(s) for s in args.seeds.split(",")):
        for method_id in METHODS:
            cfg = build(method_id, seed, args.arm, data_root=args.data_root,
                        assets_dir=args.assets, out_dir=args.run_out)
            path = out / ("%s__seed%d.json" % (method_id, seed))
            cfg.save(path)
            print("wrote %s  (%s)" % (path, cfg.run.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
