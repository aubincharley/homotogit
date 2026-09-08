"""Single entrypoint: everything a run does starts here.

Locally:   python main.py [--config NAME] [--epochs 5] [--no-augment] ...
           python main.py --help          # a flag for every config key
On Kaggle: `python build.py` inlines src/ into dist/main.py, which runs the code
           below with the package already importable. A script kernel gets no
           command line, so it uses KAGGLE_CONFIG -- edit it and re-push.
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
from cifarbase.config import load_config, seed_list
from cifarbase.data import load_cifar10
from cifarbase.report import (dump, dump_history, dump_invocation, print_report,
                              run_dir, summarise)
from cifarbase.train import train_once
from cifarbase.utils.device import pick_device

# The recipe a Kaggle push runs. A script kernel is handed no argv and no
# environment, so this constant is the only way to choose one there.
KAGGLE_CONFIG = "baseline"


def main():
    print(f"cifar10-resnet {__version__}")
    cfg = load_config(default=KAGGLE_CONFIG)
    seeds = seed_list(cfg)
    print(f"config '{cfg['config']}': {cfg['arch']}, {cfg['epochs']} epochs, "
          f"seeds {seeds}")

    device = pick_device()

    # A directory of its own, made before training: a new experiment never
    # overwrites the last one's numbers, and a run that crashes still leaves
    # behind exactly what was launched.
    out = run_dir(cfg)
    print(f"run dir: {out}")
    dump_invocation(cfg, out)

    # Loaded once and shared across seeds. Re-reading 240 MB per seed would buy
    # nothing: the splits are deterministic and never mutated.
    train, val, test = load_cifar10(device, cfg)

    run = wandb_setup.init_run(cfg) if cfg["wandb"] else None
    runs = []
    history = []
    try:
        for seed in seeds:
            print(f"\n--- seed {seed} ---")
            summary, rows = train_once(cfg, train, val, test, device, seed, run,
                                       out_dir=out)
            runs.append(summary)
            history.extend(rows)
    finally:
        # Write whatever finished before saving the run, so a crash on seed 3
        # still leaves seeds 0-2 on disk.
        if runs:
            stats = summarise(runs)
            print_report(cfg, runs, stats, test_n=len(test))
            dump({"config": cfg, "stats": stats, "runs": runs}, out)
            dump_history(history, out)
        if run is not None:
            wandb_setup.finish(run)

    print(f"\nDone. Everything from this run is in {out}")


if __name__ == "__main__":
    main()
