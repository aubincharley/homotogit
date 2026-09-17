"""Are the recorded gaps a short-budget artifact?

Every study so far ran one budget per dataset: CIFAR-10 and SVHN at 30 epochs
(11,730 updates), STL-10 at 60 (2,400).  This stretches the budget and **scales
every schedule boundary with it**, so the intervention occupies the same fraction
of training as before and the only thing that changes is how much training there
is.  If the gaps hold, they are not an artifact of a short run; if they close,
they were.

    cifar10   x3   30 -> 90 epochs    11,730 -> 35,190 updates   SGD
    stl10     x2   60 -> 120 epochs    2,400 ->  4,800 updates   Adam

The proportional stretch is what makes this comparable to the recorded tables.
The alternative -- leaving the boundaries fixed and letting the tail run long --
asks a different question (does `plain` eventually catch up?) and is not
comparable, because `plain` would get a budget the methods' schedules were never
designed around.  That arm was not run.

Why these two
-------------
CIFAR-10 under SGD is where the effects are largest and best characterised
(+4.48 to +6.08 pp), so it is the strongest test of "short-budget artifact".

STL-10 under Adam is the least understood result in the benchmark: the Gaussian
vanishes into noise (+1.09 -> +0.06 +- 1.68) while the resolution reduction more
than doubles (+2.58 -> +5.93).  STL-10 also has the shortest budget of any study
at 2,400 updates, so it is the most likely to be budget-limited -- and if the
Gaussian returns with more training, the split is about budget rather than about
the optimizer.

Everything else is each dataset's own study, unchanged: CIFAR-10 keeps the
reference recipe, STL-10 keeps sigma x3 and (48, 72, 96).  Only `budget.epochs`
and the schedule boundaries move.

One thing that cannot be held fixed
-----------------------------------
The pinned reference asset set holds **30 epochs** of data order
(``perm_seed<k>`` is ``(30, 50000)``), and `Trainer` refuses a budget it does not
cover.  A 90-epoch CIFAR-10 run therefore needs a new asset set: ``--init-from``
keeps the reference's *initial weights*, but the permutations are newly drawn, so
these runs do **not** share the reference's data order.  Seeds average over that,
and the base-budget arm this is compared against is the recorded reference, so it
is a real if minor asymmetry -- recorded rather than hidden.  STL-10's set was
generated with 90 epochs of headroom and covers the x2 stretch directly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from continuation_core.config import OptimizerConfig                  # noqa: E402
from continuation_core.methods import MethodSpec                      # noqa: E402
from continuation_core.schedules import EpochSchedule                 # noqa: E402

import cifar10_configs                                                # noqa: E402
import optimizer_configs                                              # noqa: E402
import stl10_configs                                                  # noqa: E402

METHODS = ("plain", "resolution_max_b1", "gaussian_postrelu",
           "resolution_max_b1_gaussian_conv")

#: dataset -> (study module, stretch factor, optimizer, base epochs, upd/epoch)
ARMS = {
    "cifar10": (cifar10_configs, 3, "sgd", 30, 391),
    "stl10": (stl10_configs, 2, "adam", 60, 40),
}


def stretch(sched: EpochSchedule, k: int) -> EpochSchedule:
    """Same values, boundaries scaled -- the intervention keeps its share of the run."""
    return EpochSchedule(tuple(s * k for s in sched.starts), sched.values)


def build(dataset: str, method_id: str, seed: int, *, data_root: str,
          assets_dir: str, out_dir: str, stats=(None, None)):
    mod, k, opt, base_epochs, _ = ARMS[dataset]
    cfg = mod.build(method_id, seed, data_root=data_root, assets_dir=assets_dir,
                    out_dir=out_dir, mean=stats[0], std=stats[1])
    if opt != "sgd":
        cfg.optimizer = OptimizerConfig(**optimizer_configs.OPTIMIZERS[opt])

    spec = cfg.method_spec()
    d = spec.to_dict()
    if spec.gaussian:
        d["gaussian"]["schedule"] = stretch(spec.gaussian.schedule, k).to_dict()
    if spec.resolution:
        d["resolution"]["schedule"] = stretch(spec.resolution.schedule, k).to_dict()
    cfg.method = MethodSpec.from_dict(d).to_dict()
    cfg.budget.epochs = base_epochs * k

    # keep the recorded evaluation cadence per dataset, scaled so the number of
    # snapshots stays comparable rather than tripling
    cfg.evaluation.every_epochs = max(1, cfg.evaluation.every_epochs * k)
    cfg.run.name = "%s__%s_x%d_%s__seed%d" % (method_id, dataset, k, opt, seed)
    cfg.notes.append(
        "budget x%d: %d -> %d epochs, with every schedule boundary scaled by the "
        "same factor so the intervention keeps its share of the run" % (
            k, base_epochs, cfg.budget.epochs))
    cfg.notes.append(
        "compare against this dataset's %s column at the base budget; the "
        "question is whether the gaps are a short-budget artifact" % opt)
    if dataset == "cifar10":
        cfg.notes.append(
            "the pinned reference asset set covers only 30 epochs of data order, "
            "so this budget needs a new one: initial weights are the reference's "
            "via --init-from, the permutations are not")
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=sorted(ARMS))
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--assets", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", default="configs/longer")
    ap.add_argument("--run-out", default="runs")
    ap.add_argument("--no-stats", action="store_true")
    args = ap.parse_args(argv)

    mod, k, opt, base, upd = ARMS[args.dataset]
    stats = (None, None)
    if not args.no_stats:
        try:
            stats = mod._stats(args.data_root)
            print("%s statistics pinned" % args.dataset)
        except FileNotFoundError as exc:
            print("no %s under %s (%s); statistics unpinned"
                  % (args.dataset, args.data_root, exc))
    print("%s x%d: %d -> %d epochs, %d -> %d updates, optimizer %s"
          % (args.dataset, k, base, base * k, base * upd, base * k * upd, opt))

    out = Path(args.out) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    for seed in (int(s) for s in args.seeds.split(",")):
        for method_id in METHODS:
            cfg = build(args.dataset, method_id, seed, data_root=args.data_root,
                        assets_dir=args.assets, out_dir=args.run_out, stats=stats)
            path = out / ("%s__seed%d.json" % (method_id, seed))
            cfg.save(path)
            print("wrote %s  (%s)" % (path, cfg.run.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
