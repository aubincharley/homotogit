"""Experiment 0: independent training from scratch at fixed Gaussian levels.

Every condition shares architecture, batch size, optimizer settings,
learning-rate schedule and gradient-update budget.  Seeds are *paired*: for a
given seed, all levels use the same parameter initialization and the same
sequence of minibatch sample indices (see :mod:`continuation.data`).

Nothing here reuses weights across levels and nothing here changes the
transformation during a run -- that is what makes this the fixed-parameter
special case rather than a continuation experiment.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import ExperimentConfig, from_dict, to_dict, deep_update
from ..data import build_dataset, save_split, split_fingerprint
from ..diagnostics import summarize_across_seeds
from ..engine import Trainer
from ..metrics import RunLogger, read_metrics


def run_dir_name(level: float, seed: int) -> str:
    return "level_%s__seed_%d" % (("%g" % float(level)).replace(".", "p"), int(seed))


def discard_partial_run(run_dir, active_window_seconds: float = 180.0) -> list:
    """Set aside artifacts of a run that never produced a ``summary.json``.

    A run interrupted mid-flight leaves a truncated ``metrics.jsonl``.  Since
    :class:`~continuation.metrics.RunLogger` opens that file in append mode, a
    restarted run would otherwise interleave stale records with new ones and
    silently corrupt the curves.

    Two safeguards:

    * Files are **renamed**, never deleted -- an interrupted run's partial
      metrics are evidence, not garbage, and are kept as
      ``<name>.interrupted-<timestamp>``.
    * If the run's ``metrics.jsonl`` was written within ``active_window_seconds``,
      another process is very likely still training this run, so we refuse
      rather than trample it.  (On Windows the open file handle would also block
      us; on POSIX nothing would.)

    Completed runs, which have a summary, are never touched.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists() or (run_dir / "summary.json").exists():
        return []

    metrics = run_dir / "metrics.jsonl"
    if metrics.exists():
        age = time.time() - metrics.stat().st_mtime
        if age < active_window_seconds:
            raise RuntimeError(
                "%s has no summary.json but its metrics.jsonl was modified %.0f s ago; "
                "another process is probably still running this run. Wait for it to "
                "finish, or stop it, before restarting this sweep." % (run_dir, age)
            )

    stamp = time.strftime("%Y%m%dT%H%M%S")
    moved = []
    for path in sorted(run_dir.iterdir()):
        if path.is_file():
            path.rename(path.with_name("%s.interrupted-%s" % (path.name, stamp)))
            moved.append(path.name)
    return moved


def build_run_config(base: ExperimentConfig, level: float, seed: int, out_dir: Path) -> ExperimentConfig:
    raw = to_dict(base)
    raw = deep_update(raw, {
        "run": {"name": run_dir_name(level, seed), "seed": int(seed),
                "out_dir": str(out_dir)},
        "schedule": {"kind": "constant", "params": {"value": float(level)}},
    })
    raw.pop("sweep", None)
    return from_dict(raw)


def run_experiment(cfg: ExperimentConfig, out_root, only_levels=None, only_seeds=None,
                   overwrite: bool = False) -> dict:
    levels = list(cfg.sweep.get("levels", [0.0]))
    seeds = list(cfg.sweep.get("seeds", [cfg.run.seed]))
    if only_levels is not None:
        levels = [l for l in levels if any(abs(l - o) < 1e-12 for o in only_levels)]
    if only_seeds is not None:
        seeds = [s for s in seeds if s in set(only_seeds)]

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    bundle = build_dataset(cfg.data)
    save_split(bundle, out_root / "split.npz")

    manifest = {
        "experiment": "exp0_fixed_gaussian_levels",
        "levels_planned": list(cfg.sweep.get("levels", [])),
        "levels_run": levels,
        "seeds": seeds,
        "base_config": to_dict(cfg),
        "split": split_fingerprint(bundle),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs": [],
    }

    for level in levels:
        for seed in seeds:
            name = run_dir_name(level, seed)
            run_dir = out_root / name
            summary_path = run_dir / "summary.json"
            if summary_path.exists() and not overwrite:
                print("[skip] %s already complete" % name, flush=True)
                manifest["runs"].append({"name": name, "level": level, "seed": seed,
                                         "status": "reused", "dir": str(run_dir)})
                continue
            discarded = discard_partial_run(run_dir)
            if discarded:
                print("[resume] set aside incomplete artifacts from %s: %s"
                      % (name, ", ".join(discarded)), flush=True)
            run_cfg = build_run_config(cfg, level, seed, run_dir)
            print("\n=== %s (sigma=%g, seed=%d, %d updates) ==="
                  % (name, level, seed, run_cfg.optim.total_steps), flush=True)
            logger = RunLogger(run_dir)
            try:
                trainer = Trainer(run_cfg, bundle, run_dir, logger)
                summary = trainer.fit()
            finally:
                logger.close()
            manifest["runs"].append({"name": name, "level": level, "seed": seed,
                                     "status": "completed", "dir": str(run_dir),
                                     "final": summary["final"], "timing": summary["timing"]})
            with open(out_root / "manifest.json", "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2, default=str)

    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(out_root / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return manifest


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

_FINAL_KEYS = [
    ("transformed_train_probe", "ce"), ("transformed_train_probe", "accuracy"),
    ("transformed_val", "ce"), ("transformed_val", "accuracy"),
    ("target_train_probe", "ce"), ("target_train_probe", "accuracy"),
    ("target_val", "ce"), ("target_val", "accuracy"),
]


def collect_runs(out_root) -> list:
    runs = []
    for summary_path in sorted(Path(out_root).glob("*/summary.json")):
        with open(summary_path, "r", encoding="utf-8") as fh:
            summary = json.load(fh)
        run_dir = summary_path.parent
        runs.append({
            "dir": str(run_dir),
            "name": run_dir.name,
            "level": float(summary["parameter"]),
            "seed": int(summary["seed"]),
            "summary": summary,
            "metrics": read_metrics(run_dir / "metrics.jsonl"),
        })
    return runs


def aggregate(out_root) -> dict:
    runs = collect_runs(out_root)
    by_level: dict = {}
    for r in runs:
        by_level.setdefault(r["level"], []).append(r)

    levels_out = []
    for level in sorted(by_level):
        group = sorted(by_level[level], key=lambda r: r["seed"])
        entry = {
            "level": level,
            "parameter_name": group[0]["summary"]["parameter_name"],
            "n_seeds": len(group),
            "seeds": [r["seed"] for r in group],
            "is_target_endpoint": bool(group[0]["summary"]["final"]["is_target_endpoint"]),
            "final": {},
            "timing": {},
            "transform_stats": group[0]["summary"]["transform_stats"]["start"],
        }
        for block, key in _FINAL_KEYS:
            vals = [r["summary"]["final"][block][key] for r in group]
            entry["final"]["%s.%s" % (block, key)] = summarize_across_seeds(vals)
        for key in ("train_seconds", "eval_seconds", "transform_seconds_train"):
            entry["timing"][key] = summarize_across_seeds(
                [r["summary"]["timing"][key] for r in group])
        levels_out.append(entry)

    agg = {
        "experiment": "exp0_fixed_gaussian_levels",
        "n_runs": len(runs),
        "levels": levels_out,
        "uncertainty": ("mean, sample standard deviation and standard error across "
                        "paired seeds; with 3 seeds these intervals are wide"),
        "metric_definitions": {
            "transformed_train_probe": "eval-mode CE/accuracy at this run's own level, fixed class-balanced training probe subset",
            "transformed_val": "eval-mode CE/accuracy at this run's own level, full validation split",
            "target_train_probe": "eval-mode CE/accuracy on ORIGINAL unfiltered images, same weights, same probe subset",
            "target_val": "eval-mode CE/accuracy on ORIGINAL unfiltered images, full validation split",
            "note": "all cross-entropies are unregularized; weight decay is excluded",
        },
    }
    out = Path(out_root) / "aggregate.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=2, default=str)
    write_csv(agg, Path(out_root) / "aggregate.csv")
    return agg


def write_csv(agg: dict, path) -> None:
    import csv

    cols = ["level", "n_seeds"]
    metric_cols = ["%s.%s" % (b, k) for b, k in _FINAL_KEYS]
    for m in metric_cols:
        cols += [m + "_mean", m + "_std"]
    cols += ["reconstruction_mse_mean", "retained_tv_ratio_mean", "n_zero_tv_images"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for e in agg["levels"]:
            row = [e["level"], e["n_seeds"]]
            for m in metric_cols:
                s = e["final"][m]
                row += [s["mean"], s["std"]]
            ts = e["transform_stats"]
            row += [ts["reconstruction_mse"]["mean"], ts["retained_tv_ratio"]["mean"],
                    ts["n_zero_tv_images"]]
            w.writerow(row)
