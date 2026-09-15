"""ResNet-20-BN with GELU or SiLU: does the effect need the rectifier?

The tightest controlled comparison in this benchmark.  Activations here are
functional, so they carry no parameters and the state dict is identical in keys
and shapes to ``resnet20_bn``'s.  These runs therefore load
``assets/cifar10_resnet20bn`` **directly** -- the pinned initial weights *and*
the pinned data order of the original reference campaign, not a regenerated
equivalent.  No new asset set exists, and none is needed.

So between a run here and the corresponding reference run, exactly one thing
differs: which function is applied after each BatchNorm.  Same weights at step
zero, same images in the same order, same optimizer, same schedules, same sites.

What it tests
-------------
ReLU is why the Gaussian interacts with the activation the way it does.  It is
exactly zero on half its domain, so blurring *before* it mixes dead and live
units, and blurring *after* it smooths a non-negative, sparse signal.  GELU and
SiLU are smooth, non-monotone near zero and never exactly zero, so both
properties disappear.  If the methods depend on the rectifier's sparsity, the
gaps should shrink here -- most for ``gaussian_postrelu``, whose placement is
literally named after it.

Two arms, ``gelu`` and ``silu``, share everything else.  SiLU is the milder
change (monotone above its minimum, closer to ReLU for large positive input);
GELU deviates more.  Running both distinguishes "any smooth activation breaks it"
from "how far from ReLU you go matters".

Not retuned, on purpose
-----------------------
``kaiming_normal_(mode="fan_in", nonlinearity="relu")`` is kept for every
activation.  The gain is part of the frozen recipe, and the whole point is that
the initial weights are the reference's own; drawing new ones with a different
gain would give up the shared starting point for a second-order correction.
``lr 0.005`` is likewise unchanged.

The relu arm is not run: it would be ``resnet20_bn`` exactly, and a test asserts
the two are bitwise identical.  The reference table is its control.
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

ARCH = "resnet20_act_cifar"
ARMS = ("gelu", "silu")
EPOCHS = 30                     # the reference budget: 391 upd/epoch, 11,730 total
NATIVE_RESOLUTION = 32

#: the pinned reference assets, used as they are -- same init, same data order
REFERENCE_ASSETS = "assets/cifar10_resnet20bn"

#: the reference schedules, verbatim
ACT_G = PLATEAU_G
ACT_R = RPROG

OPTIMIZER = dict(name="sgd", lr=0.005, weight_decay=5e-4, momentum=0.9,
                 nesterov=False, schedule="warmup_cosine", warmup_updates=60,
                 min_lr=0.0)

INSERTION_NOTE = ("input of blocks.2, exactly as in the reference; the site map is "
                  "activation-agnostic (it names convolutions and block outputs, "
                  "never an activation), so every method attaches at the same "
                  "tensors as the ReLU reference")


def build(method_id: str, seed: int, activation: str, *, data_root: str,
          assets_dir: str, out_dir: str):
    if activation not in ARMS:
        raise SystemExit("activation must be one of %s" % (ARMS,))
    base = get_method(method_id)
    cfg = transfer(
        method_id, dataset="cifar10", data_root=data_root, arch=ARCH,
        optimizer=OptimizerConfig(**OPTIMIZER), epochs=EPOCHS,
        resolution_schedule=ACT_R if base.resolution else None,
        reference_resolution=NATIVE_RESOLUTION if base.resolution else None,
        gaussian_schedule=ACT_G if base.gaussian else None,
        gaussian_units=("pixels of the feature map at the site; the reference "
                        "levels unchanged") if base.gaussian else None,
        insertion_mapping_note=INSERTION_NOTE,
        assets_dir=assets_dir, seed=seed, out_dir=out_dir)

    cfg.model.options = {"activation": activation}
    cfg.data.expected_mean, cfg.data.expected_std = CIFAR10_MEAN, CIFAR10_STD
    if method_id == "plain":
        cfg.evaluation.paths = ("current",)
    cfg.checkpoint.every_epoch = False
    cfg.checkpoint.keep_rolling = True
    cfg.run.name = "%s__r20%s_cifar10__seed%d" % (method_id, activation, seed)
    cfg.notes.append("F.%s replaces F.relu; everything else is the reference "
                     "recipe" % activation)
    cfg.notes.append("uses the pinned reference asset set as-is: the same initial "
                     "weights and the same data order as the ResNet-20-BN campaign, "
                     "because the state dict is interchangeable")
    cfg.notes.append("initialization gain and lr are not retuned for the new "
                     "activation; that is deliberate, see the module docstring")
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--activation", required=True, choices=ARMS)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--assets", default=REFERENCE_ASSETS)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", default="configs/resnet20act_cifar10")
    ap.add_argument("--run-out", default="runs")
    args = ap.parse_args(argv)

    out = Path(args.out) / args.activation
    out.mkdir(parents=True, exist_ok=True)
    for seed in (int(s) for s in args.seeds.split(",")):
        for method_id in METHODS:
            cfg = build(method_id, seed, args.activation, data_root=args.data_root,
                        assets_dir=args.assets, out_dir=args.run_out)
            path = out / ("%s__seed%d.json" % (method_id, seed))
            cfg.save(path)
            print("wrote %s  (%s)" % (path, cfg.run.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
