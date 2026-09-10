"""Single entrypoint: everything a run does starts here.

Locally:   python main.py                          # the study named by KAGGLE_STUDY
           python main.py --study smoke            # a named grid
           python main.py --config baseline --study ""   # one arm, every seed
           python main.py --help                   # a flag for every config key
On Kaggle: `python build.py` inlines src/ into dist/main.py, which runs the code
           below with the package already importable. A script kernel gets no
           command line, so no argv means "run the study", and the whole
           campaign is one push instead of one push per arm.

A STUDY is the unit that matters here. Comparing a curriculum against a baseline
means holding everything else identical, and the cheapest way to be sure of that
is to run every arm in one process, off one copy of the data, against one
teacher. Two pushes days apart are two experiments.
"""
import json
import os
import sys
import time

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
from cifarbase.config import _validate, load_config, load_study, seed_list
from cifarbase.data import load_cifar10
from cifarbase.report import (dump, dump_arm, dump_history, dump_invocation,
                              print_report, run_dir, study_dir, summarise)
from cifarbase.scoring import fit_teacher
from cifarbase.train import train_once
from cifarbase.utils.device import pick_device

# What a Kaggle push runs, and what `python main.py` with no arguments runs. A
# script kernel is handed no argv and no environment, so a constant is the only
# way to choose there. Set it to "" to fall back to KAGGLE_CONFIG as a single
# arm instead.
KAGGLE_STUDY = "pacing_vs_order"
KAGGLE_CONFIG = "baseline"


def run_arm(cfg, device, train, val, test, base=None, stem=None,
            order_scores=None):
    """One recipe, every seed, one run directory of its own.

    A directory made before training: a new experiment never overwrites the last
    one's numbers, and a run that crashes still leaves behind exactly what was
    launched.
    """
    label = f"{cfg['regime']}/{cfg['arm']}" if cfg["arm"] else cfg["config"]
    print(f"\n{'=' * 78}\n{label}: {cfg['arch']}, {cfg['epochs']} epochs, "
          f"seeds {seed_list(cfg)}, curriculum {cfg['curriculum']}\n{'=' * 78}")
    out = run_dir(cfg, base=base, stem=stem, link=stem is None)
    print(f"run dir: {out}")
    dump_invocation(cfg, out)
    if cfg["arm"]:
        dump_arm(cfg, out)

    run = wandb_setup.init_run(cfg) if cfg["wandb"] else None
    runs, history = [], []
    try:
        for seed in seed_list(cfg):
            print(f"\n--- seed {seed} ---")
            summary, rows = train_once(cfg, train, val, test, device, seed, run,
                                       out=out, order_scores=order_scores)
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
    return out


# --------------------------------------------------------------------------
# the grid
# --------------------------------------------------------------------------

def _arm_config(cfg, study, regime, arm):
    """base < regime < arm, then the flags the user actually typed.

    The command line wins last on purpose: `--seeds 0 --epochs 2` has to be able
    to shrink a grid to something that finishes in a minute, or nobody will ever
    check the plumbing before spending the GPU hours.
    """
    merged = dict(cfg)
    for block in (study.get("base", {}), regime, arm):
        merged.update({k: v for k, v in block.items() if k != "name"})
    merged["study"] = study["name"]
    merged["regime"] = regime["name"]
    merged["arm"] = arm["name"]
    if "seeds" in study:
        merged["seeds"] = study["seeds"]
    for key in cfg.get("cli_keys", []):
        merged[key] = cfg[key]
    return _validate(merged)


def _data_key(cfg):
    """What has to be equal for two regimes to share one copy of the data."""
    return (float(cfg["label_noise"]), int(cfg["train_subset"]),
            int(cfg["val_size"]))


def run_study(cfg, device):
    """Every arm of a grid, in one process, off one copy of the data.

    Regimes run in the order the YAML lists them, and `budget_min` cuts the tail
    rather than the middle: a grid that turns out slower than estimated loses its
    last regime, not a seed here and an arm there. What was cut is written into
    study.json, so a partial grid is still an honest one.
    """
    import torch

    study = load_study(cfg["study"])
    root = study_dir(study["name"])
    seeds = len(seed_list(_arm_config(cfg, study, study["regimes"][0],
                                      study["arms"][0])))
    planned = sum(int(_arm_config(cfg, study, r, a)["epochs"]) * seeds
                  for r in study["regimes"] for a in study["arms"])
    # The grid carries its own budget, because the budget is a property of the
    # grid; --budget-min still wins, so a session with less time left can say so.
    budget_min = float(cfg["budget_min"] if "budget_min" in cfg.get("cli_keys", [])
                       else study.get("budget_min", cfg["budget_min"]))
    budget_s = budget_min * 60.0
    started = time.time()

    print(f"study {study['name']}: {len(study['regimes'])} regimes x "
          f"{len(study['arms'])} arms x {seeds} seeds = "
          f"{len(study['regimes']) * len(study['arms']) * seeds} runs, "
          f"{planned} epoch-seeds"
          + (f", budget {budget_min:.0f} min" if budget_s else ""))
    print(f"study dir: {root}")

    manifest = {"study": study["name"], "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "seeds": seeds, "planned_epoch_seeds": planned,
                "budget_min": budget_min, "arms": [], "skipped": [],
                "teachers": {}}
    data, data_key = None, None
    teachers = {}
    done_epoch_seeds = 0

    for regime in study["regimes"]:
        rcfg = _arm_config(cfg, study, regime, study["arms"][0])
        if data is None or _data_key(rcfg) != data_key:
            print(f"\nloading data for regime {regime['name']} "
                  f"(label_noise {rcfg['label_noise']})")
            data = load_cifar10(device, rcfg)
            data_key = _data_key(rcfg)
        train, val, test = data

        # One teacher per regime, shared by every arm and every seed in it. The
        # ordering has to be a fixed property of the experiment: refitting it per
        # arm would make "curriculum" and "anti" disagree about which examples
        # are easy, and their difference would stop meaning direction.
        arm_configs = [_arm_config(cfg, study, regime, a) for a in study["arms"]]
        wants_teacher = any(a["curriculum"] in ("easy_first", "hard_first")
                            and a["scoring"] == "transfer" for a in arm_configs)
        scores = None
        if wants_teacher:
            key = data_key
            if key not in teachers:
                print()
                teachers[key] = fit_teacher(rcfg, train, test, device)
                path = os.path.join(root, f"teacher_{regime['name']}.pt")
                torch.save({"margins": teachers[key][0], **teachers[key][1]}, path)
                teachers[key][1]["fitted_for"] = regime["name"]
            # Two regimes over the same data share one ordering, which is what
            # makes them comparable to each other as well as to their baselines.
            # Recorded per regime rather than per fit, so the report can say so.
            manifest["teachers"][regime["name"]] = teachers[key][1]
            scores = teachers[key][0]

        for arm, acfg in zip(study["arms"], arm_configs, strict=True):
            cost = int(acfg["epochs"]) * len(seed_list(acfg))
            elapsed = time.time() - started
            # Project from what the grid has actually cost so far rather than
            # from an estimate made before it started.
            rate = elapsed / done_epoch_seeds if done_epoch_seeds else 0.0
            if budget_s and rate and elapsed + cost * rate > budget_s:
                print(f"\n!! budget: skipping {regime['name']}/{arm['name']} "
                      f"({cost * rate / 60:.1f} min needed, "
                      f"{(budget_s - elapsed) / 60:.1f} left)")
                manifest["skipped"].append(f"{regime['name']}__{arm['name']}")
                continue

            out = run_arm(acfg, device, train, val, test, base=root,
                          stem=f"{regime['name']}__{arm['name']}",
                          order_scores=scores)
            done_epoch_seeds += cost
            manifest["arms"].append({"regime": regime["name"], "arm": arm["name"],
                                     "dir": os.path.basename(out),
                                     "elapsed_min": (time.time() - started) / 60})
            rate = (time.time() - started) / done_epoch_seeds
            print(f"\n[study] {done_epoch_seeds}/{planned} epoch-seeds, "
                  f"{(time.time() - started) / 60:.1f} min spent, "
                  f"{(planned - done_epoch_seeds) * rate / 60:.1f} min projected")

    manifest["wall_min"] = (time.time() - started) / 60
    with open(os.path.join(root, "study.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return root


def main():
    print(f"cifar10-resnet {__version__}")
    cfg = load_config(default=KAGGLE_CONFIG)

    # A script kernel gets no argv at all, and neither does `python main.py`.
    # Both mean "run the campaign", which is the study.
    if KAGGLE_STUDY and not cfg["study"] and len(sys.argv) == 1:
        cfg["study"] = KAGGLE_STUDY

    device = pick_device()

    if cfg["study"]:
        out = run_study(cfg, device)
        print(f"\nDone. Study written to {out}")
        print(f"Next: python analyze.py {out}")
        return

    # Loaded once and shared across seeds. Re-reading 240 MB per seed would buy
    # nothing: the splits are deterministic and never mutated.
    train, val, test = load_cifar10(device, cfg)
    out = run_arm(cfg, device, train, val, test)
    print()
    print(f"Done. Run written to {out}")


if __name__ == "__main__":
    main()
