"""ResNet-20 with GroupNorm on CIFAR-10: is the blur-BatchNorm coupling the mechanism?

Everything is the CIFAR-10 reference recipe, unchanged -- all 50,000 images in
the reference's own order, 391 updates per epoch, 30 epochs, SGD lr 0.005, and
the reference schedules, since this is 32x32 and the same architecture shape.
**Only the normalisation layer differs.**

The hypothesis
--------------
``resolution_max_b1_gaussian_conv`` filters all 19 convolution outputs *before*
normalisation, so BatchNorm computes its statistics on blurred activations and
accumulates its running buffers on them.  The network's normalisation is then
fitted to a distribution the schedule later removes.  That coupling is the
leading explanation for the combined method's behaviour, and on STL-10 it is what
held the target path at chance for all 42 epochs the blur was active.

GroupNorm removes that coupling and nothing else: per-sample, per-group, no batch
dependence, no running statistics.  There is no fitted state for the blur to
corrupt and nothing to re-estimate when it anneals away.  So:

============================  =========================================
if the coupling is the        the combined method loses most of its
mechanism                     advantage here; gaussian_postrelu, which
                              blurs *after* normalisation, is less
                              affected; resolution_max_b1, which does
                              not blur, is unaffected
if all three hold up          the coupling is not the mechanism, and
                              the STL-10 target-path collapse needs
                              another explanation
============================  =========================================

Same images, same order, same sites
-----------------------------------
``make-assets --indices-from assets/cifar10_resnet20bn`` reuses the reference
campaign's ``subset``, ``train_probe`` and ``perm_seed<k>``.  ``--init-from``
cannot be used and correctly refuses: GroupNorm has no ``running_mean`` /
``running_var`` / ``num_batches_tracked``, so the state dicts differ.  Parameter
*count* is identical (269,722), and ``resnet20_gn`` copies the BatchNorm site map
verbatim -- it names convolutions, block outputs and the input of ``blocks.2``,
never a normalisation layer -- so all four methods attach at exactly the same
tensors in both models.

The one thing that is not controlled
------------------------------------
``lr 0.005`` was frozen against BatchNorm.  GroupNorm is not BatchNorm and may
simply train to a different absolute accuracy under it; no attempt is made to
retune, because changing the normalisation and the optimizer together would
confound the comparison.  Read the **ordering of the four methods within this
model**, and the size of each gap relative to its own control.  A straight
comparison of GN accuracies against the BN reference table is not meaningful.
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

ARCH = "resnet20_gn_cifar"
EPOCHS = 30                      # the reference budget: 391 upd/epoch, 11,730 total
NATIVE_RESOLUTION = 32
GROUPS = 8                       # divides 16, 32 and 64 evenly

#: the reference schedules, verbatim -- nothing about the data or the geometry
#: changed, so nothing is rescaled
GN_G = PLATEAU_G
GN_R = RPROG

OPTIMIZER = dict(name="sgd", lr=0.005, weight_decay=5e-4, momentum=0.9,
                 nesterov=False, schedule="warmup_cosine", warmup_updates=60,
                 min_lr=0.0)

GAUSSIAN_UNITS = ("pixels of the feature map at the site; the reference levels "
                  "unchanged, since the geometry is the reference's")
INSERTION_NOTE = ("input of blocks.2, exactly as in the reference; resnet20_gn "
                  "copies the BatchNorm site map verbatim, so every method "
                  "attaches at the same tensors in both models")


def build(method_id: str, seed: int, *, data_root: str, assets_dir: str, out_dir: str):
    base = get_method(method_id)
    cfg = transfer(
        method_id, dataset="cifar10", data_root=data_root, arch=ARCH,
        optimizer=OptimizerConfig(**OPTIMIZER), epochs=EPOCHS,
        resolution_schedule=GN_R if base.resolution else None,
        reference_resolution=NATIVE_RESOLUTION if base.resolution else None,
        gaussian_schedule=GN_G if base.gaussian else None,
        gaussian_units=GAUSSIAN_UNITS if base.gaussian else None,
        insertion_mapping_note=INSERTION_NOTE,
        assets_dir=assets_dir, seed=seed, out_dir=out_dir)

    cfg.model.options = {"groups": GROUPS}
    cfg.data.expected_mean, cfg.data.expected_std = CIFAR10_MEAN, CIFAR10_STD
    if method_id == "plain":
        cfg.evaluation.paths = ("current",)
    cfg.checkpoint.every_epoch = False
    cfg.checkpoint.keep_rolling = True
    cfg.run.name = "%s__resnet20gn_cifar10__seed%d" % (method_id, seed)
    cfg.notes.append("GroupNorm(%d) replaces BatchNorm; everything else is the "
                     "reference recipe, including the data order" % GROUPS)
    cfg.notes.append("bn_policy is meaningless here: GroupNorm has no running "
                     "buffers and no train/eval difference, so running_stats and "
                     "fixed_batch_stats give identical numbers")
    cfg.notes.append("lr 0.005 was frozen against BatchNorm and is not retuned; "
                     "compare the four methods within this model, not against "
                     "the BatchNorm reference table")
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--assets", default="assets/resnet20gn_cifar10")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", default="configs/resnet20gn_cifar10")
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
