"""Single entrypoint: everything a run does starts here.

Locally:   python main.py [--config NAME] [--epochs 5] [--no-augment] ...
           python main.py --help          # a flag for every config key
           python main.py --selfcheck     # the anchor's correctness gates
On Kaggle: `python build.py` inlines src/ into dist/main.py, which runs the code
           below with the package already importable. A script kernel gets no
           command line, so it uses KAGGLE_CONFIG -- edit it and re-push.

A config may declare several `arms`, and then this runs each of them in turn
against one shared copy of the data and prints them on one comparison table. That
exists because a Kaggle kernel is one file with no argv: a comparison split
across several pushes is a comparison whose arms ran different code.
"""
import os
import sys

# Local runs need src/ on the path. Inside the bundle the generated header has
# already done this and there is no src/ directory, so the block is a no-op.
_HERE = (os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals()
         else os.getcwd())
_SRC = os.path.join(_HERE, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Kaggle captures stdout through a pipe, which makes python block-buffer it: the
# log then stays empty for hours and arrives in one lump at the end. A long run
# you cannot watch is a long run you cannot abort.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:                                      # pragma: no cover
    pass

from cifarbase import __version__, wandb_setup
from cifarbase.config import arm_configs, load_config, seed_list
from cifarbase.data import load_cifar10
from cifarbase.report import (arm_dir, dump, dump_history, dump_invocation,
                              print_comparison, print_report, run_dir,
                              summarise)
from cifarbase.selfcheck import run_gates
from cifarbase.train import train_once
from cifarbase.utils.device import pick_device

# The recipe a Kaggle push runs. A script kernel is handed no argv and no
# environment, so this constant is the only way to choose one there.
KAGGLE_CONFIG = "compare"

# Which arm the comparison table states differences against. Ignored when the
# config declares no arms, or names no arm this way.
BASELINE_ARM = "baseline_wd"


def main():
    print(f"cifar10-resnet {__version__}")
    cfg = load_config(default=KAGGLE_CONFIG)
    seeds = seed_list(cfg)
    arms = arm_configs(cfg)
    print(f"config '{cfg['config']}': {cfg['arch']}, {cfg['epochs']} epochs, "
          f"seeds {seeds}" + (f", {len(arms)} arms" if arms[0][0] else ""))

    device = pick_device()

    if cfg["selfcheck"]:
        # Before any GPU quota is spent. A gate that fails here is a bug in the
        # anchor or the estimator; a gate that fails after six hours of training
        # is the same bug plus six hours.
        raise SystemExit(0 if run_gates(cfg, device) else 1)

    # A directory of its own, made before training: a new experiment never
    # overwrites the last one's numbers, and a run that crashes still leaves
    # behind exactly what was launched.
    out = run_dir(cfg)
    print(f"run dir: {out}")
    dump_invocation(cfg, out)

    # Loaded once and shared across arms and seeds. Re-reading 240 MB per arm
    # would buy nothing: the splits are deterministic and never mutated, which
    # is why an arm is forbidden from setting a key that decides them.
    train, val, test = load_cifar10(device, cfg)

    run = wandb_setup.init_run(cfg) if cfg["wandb"] else None
    finished = []
    try:
        for name, arm_cfg in arms:
            finished.append(_run_arm(name, arm_cfg, train, val, test, device,
                                     out, run))
    finally:
        # Whatever finished before the crash is on disk, and the comparison is
        # drawn over however many arms got that far. An eight-hour sweep that
        # dies on the last arm must not cost the first seven.
        if len(finished) > 1:
            print_comparison(finished, baseline=BASELINE_ARM)
        if run is not None:
            wandb_setup.finish(run)

    print(f"\nDone. Everything from this run is in {out}")
    if len(arms) > 1:
        print(f"Compare the arms with:  python analyze.py {out}")


def _run_arm(name, cfg, train, val, test, device, parent, run):
    """One arm: every seed, its own directory, its own results.json."""
    where = arm_dir(parent, name)
    if name:
        print(f"\n{'=' * 78}\nARM '{name}'  --  anchor {cfg['anchor_mode']}"
              + (f", lambda {cfg['anchor_lambda']:g}"
                 if cfg["anchor_mode"] == "const" else "")
              + f", wd {cfg['weight_decay']:g}\n{'=' * 78}")
        dump_invocation(cfg, where)

    runs, history, probes = [], [], []
    prefix = f"{name}/" if name else ""
    try:
        for seed in seed_list(cfg):
            print(f"\n--- {prefix}seed {seed} ---")
            summary, rows, probe_rows = train_once(cfg, train, val, test, device,
                                                   seed, run, out=where)
            summary["arm"] = name or cfg["config"]
            runs.append(summary)
            history.extend(rows)
            probes.extend(probe_rows)
    finally:
        # Write whatever finished before saving the arm, so a crash on seed 3
        # still leaves seeds 0-2 on disk.
        if runs:
            stats = summarise(runs)
            print_report(cfg, runs, stats, test_n=len(test))
            dump({"config": cfg, "stats": stats, "runs": runs}, where)
            dump_history(history, where)
            # A second stream, one row per probe rather than per epoch. It is
            # also what the replay arm reads back, so it is written even when the
            # run crashed partway.
            dump_history(probes, where, stem="probes")
    return {"name": name or cfg["config"], "stats": summarise(runs) if runs else {},
            "runs": runs, "dir": where}


if __name__ == "__main__":
    main()
