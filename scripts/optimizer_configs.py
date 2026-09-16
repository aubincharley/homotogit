"""SVHN and STL-10 under Adam and AdamW: does the optimizer change the picture?

Every study so far used the frozen SGD recipe: lr 0.005, weight decay 5e-4,
momentum 0.9, warmup_cosine.  This swaps the optimizer and **nothing else**.
The dataset-specific decisions -- SVHN's 50,000-image subset and reference
schedules, STL-10's sigma x3 and (48, 72, 96) resolution over 60 epochs -- come
from `scripts/svhn_configs.py` and `scripts/stl10_configs.py` unchanged, by
importing them and overriding only `cfg.optimizer`.  So any difference from the
recorded SGD tables is the optimizer's.

Settings, and what they are not
-------------------------------
    adam    lr 1e-3, weight_decay 0
    adamw   lr 1e-3, weight_decay 0.01

These are the textbook values, chosen deliberately over a calibration sweep.
They are **not** matched to the SGD recipe's difficulty.  SGD lr 0.005 is about a
twentieth of a normal CIFAR learning rate, and that is precisely why the
reference control sits undertrained at 75.4% with 25 points of headroom for the
interventions to recover.  Adam at 1e-3 is a normal learning rate, so the control
should train considerably further, and the headroom should shrink.

**Read the gaps, not the absolute numbers, and read them against this column's
own control.**  If the interventions show little or nothing here, the first
explanation to rule out is that a better-optimized control left them nothing to
recover -- the same confound SVHN's converged 93.1% baseline raised -- not that
the optimizer broke them.  Settling that would need a run whose lr puts the
control back near 75%, which is the calibration that was not done.

Adam's `weight_decay` is an L2 term added to the gradient and AdamW's is
decoupled, so the two numbers are not comparable to each other either; 0 and 0.01
are each that optimizer's own convention rather than a matched pair.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from continuation_core.config import OptimizerConfig                  # noqa: E402

import stl10_configs                                                  # noqa: E402
import svhn_configs                                                   # noqa: E402

STUDIES = {"svhn": svhn_configs, "stl10": stl10_configs}

#: warmup and the cosine shape are shared machinery and stay as the recipe has
#: them; only the optimizer, its lr and its decay change
OPTIMIZERS = {
    "adam": dict(name="adam", lr=1e-3, weight_decay=0.0,
                 betas=(0.9, 0.999), eps=1e-8,
                 schedule="warmup_cosine", warmup_updates=60, min_lr=0.0),
    "adamw": dict(name="adamw", lr=1e-3, weight_decay=0.01,
                  betas=(0.9, 0.999), eps=1e-8,
                  schedule="warmup_cosine", warmup_updates=60, min_lr=0.0),
}

METHODS = ("plain", "resolution_max_b1", "gaussian_postrelu",
           "resolution_max_b1_gaussian_conv")


def build(dataset: str, optimizer: str, method_id: str, seed: int, *,
          data_root: str, assets_dir: str, out_dir: str, stats=(None, None)):
    mod = STUDIES[dataset]
    # both study modules take pinned statistics and check them at build time; that
    # check is what caught SVHN's subset-vs-split mismatch, so keep it for both
    cfg = mod.build(method_id, seed, data_root=data_root, assets_dir=assets_dir,
                    out_dir=out_dir, mean=stats[0], std=stats[1])

    before = cfg.optimizer
    cfg.optimizer = OptimizerConfig(**OPTIMIZERS[optimizer])
    cfg.run.name = "%s__%s_%s__seed%d" % (method_id, dataset, optimizer, seed)
    cfg.notes.append(
        "optimizer swapped from %s lr %g wd %g to %s lr %g wd %g; every other "
        "decision is this dataset's study unchanged"
        % (before.name, before.lr, before.weight_decay,
           cfg.optimizer.name, cfg.optimizer.lr, cfg.optimizer.weight_decay))
    cfg.notes.append(
        "lr is the textbook value, NOT matched to the SGD recipe's difficulty: "
        "SGD lr 0.005 is ~1/20 of a normal rate and is what keeps the control "
        "undertrained. Expect less headroom here; read gaps against this "
        "column's own control, and rule out a better-optimized baseline before "
        "concluding the optimizer broke the methods.")
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=sorted(STUDIES))
    ap.add_argument("--optimizer", required=True, choices=sorted(OPTIMIZERS))
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--assets", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", default="configs/optimizer")
    ap.add_argument("--run-out", default="runs")
    ap.add_argument("--no-stats", action="store_true")
    args = ap.parse_args(argv)

    stats = (None, None)
    if not args.no_stats:
        try:
            stats = STUDIES[args.dataset]._stats(args.data_root)
            print("%s statistics: mean %s std %s" % (args.dataset, *stats))
        except FileNotFoundError as exc:
            print("no %s under %s (%s); statistics unpinned"
                  % (args.dataset, args.data_root, exc))

    o = OPTIMIZERS[args.optimizer]
    print("%s / %s: lr %g, weight_decay %g" % (args.dataset, args.optimizer,
                                               o["lr"], o["weight_decay"]))
    out = Path(args.out) / args.dataset / args.optimizer
    out.mkdir(parents=True, exist_ok=True)
    for seed in (int(s) for s in args.seeds.split(",")):
        for method_id in METHODS:
            cfg = build(args.dataset, args.optimizer, method_id, seed,
                        data_root=args.data_root, assets_dir=args.assets,
                        out_dir=args.run_out, stats=stats)
            path = out / ("%s__seed%d.json" % (method_id, seed))
            cfg.save(path)
            print("wrote %s  (%s)" % (path, cfg.run.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
