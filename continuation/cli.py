"""Command-line entry points.

    py -m continuation.cli prepare-data  --config configs/exp0_gaussian.yaml
    py -m continuation.cli visualize     --config configs/exp0_gaussian.yaml
    py -m continuation.cli train         --config configs/exp0_gaussian.yaml --level 1.0 --seed 0
    py -m continuation.cli exp0          --config configs/exp0_gaussian.yaml
    py -m continuation.cli report        --config configs/exp0_gaussian.yaml

Any config value can be overridden on the command line, e.g.
``--set optim.total_steps=2000 --set run.device=cpu``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import deep_update, load_config, parse_override, to_dict


def _load(args):
    overrides = {}
    for item in args.set or []:
        overrides = deep_update(overrides, parse_override(item))
    return load_config(args.config, overrides)


def _out_root(cfg, args) -> Path:
    return Path(args.out or cfg.run.out_dir) / (args.name or cfg.run.name)


def cmd_prepare_data(args):
    from .data import build_dataset, save_split, split_fingerprint

    cfg = _load(args)
    bundle = build_dataset(cfg.data)
    root = _out_root(cfg, args)
    save_split(bundle, root / "split.npz")
    print(json.dumps(split_fingerprint(bundle), indent=2))
    print("split written to %s" % (root / "split.npz"))


def cmd_visualize(args):
    from .data import build_dataset
    from .transforms import build_transform
    from .viz import attenuation_figure, boundary_effect_figure, class_diverse_indices, level_grid

    cfg = _load(args)
    bundle = build_dataset(cfg.data)
    transform = build_transform(cfg.transform)
    levels = args.levels if args.levels else list(cfg.sweep.get("levels", [0.0]))
    out_dir = _out_root(cfg, args) / "figures"

    idx = class_diverse_indices(bundle.train.labels.numpy(), per_class=args.per_class,
                                seed=cfg.evaluation.probe_seed)
    sub = bundle.train.subset(idx, "viz")
    p1 = level_grid(sub.images, sub.labels.numpy(), bundle.class_names, transform, levels,
                    out_dir / "transform_levels.png")
    p2 = boundary_effect_figure(transform, levels, out_dir / "boundary_effects.png")
    p3 = attenuation_figure(transform, levels, out_dir / "attenuation.png")
    meta = {"levels": levels, "example_indices_in_train_split": idx.tolist(),
            "original_dataset_indices": bundle.train.indices[idx].tolist(),
            "labels": [bundle.class_names[int(l)] for l in sub.labels.numpy()],
            "transform": transform.describe()}
    (out_dir / "visualization.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for p in (p1, p2, p3):
        print("wrote %s" % p)


def cmd_train(args):
    from .config import from_dict
    from .data import build_dataset
    from .engine import Trainer
    from .metrics import RunLogger

    cfg = _load(args)
    raw = to_dict(cfg)
    if args.level is not None:
        raw = deep_update(raw, {"schedule": {"kind": "constant",
                                             "params": {"value": float(args.level)}}})
    if args.seed is not None:
        raw = deep_update(raw, {"run": {"seed": int(args.seed)}})
    raw.pop("sweep", None)
    cfg = from_dict(raw)

    level = cfg.schedule.params.get("value", 0.0)
    run_dir = _out_root(cfg, args) / ("single_level_%g_seed_%d" % (float(level), cfg.run.seed))
    bundle = build_dataset(cfg.data)
    logger = RunLogger(run_dir)
    try:
        summary = Trainer(cfg, bundle, run_dir, logger).fit()
    finally:
        logger.close()
    print(json.dumps(summary["final"], indent=2))


def cmd_exp0(args):
    from .experiments.exp0 import aggregate, run_experiment

    cfg = _load(args)
    root = _out_root(cfg, args)
    run_experiment(cfg, root, only_levels=args.levels, only_seeds=args.seeds,
                   overwrite=args.overwrite)
    agg = aggregate(root)
    print("\naggregate written to %s" % (root / "aggregate.json"))
    for e in agg["levels"]:
        print("  sigma=%-4g  target_val_acc=%.4f  transformed_val_acc=%.4f  (n=%d seeds)"
              % (e["level"], e["final"]["target_val.accuracy"]["mean"],
                 e["final"]["transformed_val.accuracy"]["mean"], e["n_seeds"]))


def cmd_exp1(args):
    from .experiments.exp1 import aggregate, run_experiment

    cfg = _load(args)
    root = _out_root(cfg, args)
    run_experiment(cfg, root, seeds=args.seeds, arms=args.arms)
    agg = aggregate(root, cfg.sweep["baseline_dir"], cfg.sweep["seeds"],
                    cfg.evaluation.eval_every, cfg.sweep["total_steps"])
    print("\naggregate written to %s" % (root / "aggregate.json"))
    for arm, e in agg["arms"].items():
        print("  arm %s  target_val_acc=%.4f (sd %.4f, n=%d)"
              % (arm, e["final"]["target_val.accuracy"]["mean"],
                 e["final"]["target_val.accuracy"]["std"] or 0.0, e["n_seeds"]))


def cmd_exp1_report(args):
    from .experiments.exp1 import aggregate, collect_runs
    from .plotting import plot_exp1

    cfg = _load(args)
    root = _out_root(cfg, args)
    seeds, base = cfg.sweep["seeds"], cfg.sweep["baseline_dir"]
    runs = collect_runs(root, base, seeds)
    if not runs:
        raise SystemExit("no runs found under %s" % root)
    agg = aggregate(root, base, seeds, cfg.evaluation.eval_every, cfg.sweep["total_steps"])
    for p in plot_exp1(runs, agg, root / "figures", cfg.sweep):
        print("wrote %s" % p)
    print("aggregate: %s" % (root / "aggregate.json"))


def cmd_report(args):
    from .experiments.exp0 import aggregate, collect_runs
    from .plotting import plot_curves, plot_final_vs_level, plot_transform_stats

    cfg = _load(args)
    root = _out_root(cfg, args)
    runs = collect_runs(root)
    if not runs:
        raise SystemExit("no completed runs found under %s" % root)
    agg = aggregate(root)
    figs = plot_curves(runs, root / "figures")
    figs.append(plot_final_vs_level(agg, root / "figures"))
    figs.append(plot_transform_stats(agg, root / "figures"))
    for p in figs:
        print("wrote %s" % p)
    print("aggregate: %s" % (root / "aggregate.json"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="continuation", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--config", required=True)
        sp.add_argument("--set", action="append", metavar="section.key=value")
        sp.add_argument("--out", default=None, help="override run.out_dir")
        sp.add_argument("--name", default=None, help="override run.name")
        return sp

    common(sub.add_parser("prepare-data")).set_defaults(func=cmd_prepare_data)

    sp = common(sub.add_parser("visualize"))
    sp.add_argument("--levels", type=float, nargs="*", default=None)
    sp.add_argument("--per-class", type=int, default=1)
    sp.set_defaults(func=cmd_visualize)

    sp = common(sub.add_parser("train"))
    sp.add_argument("--level", type=float, default=None)
    sp.add_argument("--seed", type=int, default=None)
    sp.set_defaults(func=cmd_train)

    sp = common(sub.add_parser("exp0"))
    sp.add_argument("--levels", type=float, nargs="*", default=None)
    sp.add_argument("--seeds", type=int, nargs="*", default=None)
    sp.add_argument("--overwrite", action="store_true")
    sp.set_defaults(func=cmd_exp0)

    sp = common(sub.add_parser("exp1"))
    sp.add_argument("--seeds", type=int, nargs="*", default=None)
    sp.add_argument("--arms", nargs="*", default=None, choices=["W", "P"])
    sp.set_defaults(func=cmd_exp1)

    common(sub.add_parser("exp1-report")).set_defaults(func=cmd_exp1_report)
    common(sub.add_parser("report")).set_defaults(func=cmd_report)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
