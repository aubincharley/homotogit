#!/usr/bin/env python3
"""Is the continuation worth it? Arms overlaid against gradient updates.

    python compare.py runs                       # every finished run in runs/
    python compare.py runs/A runs/B              # just these
    python compare.py runs --target 0.90         # updates to reach 90%

Reads history.jsonl only -- no torch, no GPU, no checkpoints. Instant.

The x-axis is gradient updates rather than epochs because that is what "was the
continuation more efficient" actually asks. With one batch size across the arms
the two are proportional, so the axis is a relabelling; it stops being one the
moment an arm changes batch_size, and then epochs would silently lie.

The trap this file exists to avoid:

    A homotopy run's `train_loss` at epoch e is L_{s(e)}(theta_e). While s < 1
    that is a *different objective* at every epoch, and a smaller number than
    the baseline's does not mean better -- it means an easier problem. At s=0
    the residual branches are switched off entirely, and the loss of the
    shortcut-only network is not comparable to the loss of a ResNet by any
    reading.

    So the loss panel is drawn with the ramp shaded, and it is not the panel to
    conclude from. `test_acc_at_s1` is: it evaluates every arm as the full
    ResNet, which is the one model all of them are trying to produce.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from analyze import CYCLE, MUTED, _theme


def load(directory):
    """One run's config and history, or None if it never finished."""
    results = os.path.join(directory, "results.json")
    history = os.path.join(directory, "history.jsonl")
    if not (os.path.isfile(results) and os.path.isfile(history)):
        return None
    with open(results) as fh:
        payload = json.load(fh)
    with open(history) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if not rows:
        return None

    cfg = payload["config"]
    # Updates per epoch, from the config rather than from the row count: an
    # epoch is len(train)//batch_size batches because batches() drops the last
    # partial one.
    train_n = (cfg["train_subset"] or 50000) - cfg["val_size"]
    per_epoch = max(1, train_n // cfg["batch_size"])

    seeds = sorted({r["seed"] for r in rows})
    return {"dir": directory, "cfg": cfg, "rows": rows, "seeds": seeds,
            "per_epoch": per_epoch,
            "name": os.path.basename(os.path.normpath(os.path.realpath(directory))),
            "label": f"{cfg['config']} ({cfg['s_schedule']})"}


def series(run, key):
    """(updates, mean across seeds) for one metric, epochs in order."""
    by_epoch = {}
    for row in run["rows"]:
        value = row.get(key)
        if value is not None:
            by_epoch.setdefault(row["epoch"], []).append(value)
    epochs = sorted(by_epoch)
    updates = [(e + 1) * run["per_epoch"] for e in epochs]
    means = [sum(by_epoch[e]) / len(by_epoch[e]) for e in epochs]
    return updates, means


def ramp_end_update(run):
    """The update at which s finally reaches 1, or None if it always was."""
    steps, values = series(run, "s_mean")
    if not values or min(values) >= 1.0:
        return None
    for step, value in zip(steps, values, strict=True):
        if value >= 1.0 - 1e-9:
            return step
    return None


def updates_to_reach(run, key, target):
    """First update at which `key` reaches `target`, by linear interpolation.

    Interpolated rather than snapped to the epoch grid, because an epoch is 390
    updates here and rounding to it would quantise the answer far more coarsely
    than the difference being measured.
    """
    steps, values = series(run, key)
    previous_step = previous_value = None
    for step, value in zip(steps, values, strict=True):
        if value >= target:
            if previous_step is None or value == previous_value:
                return step
            fraction = (target - previous_value) / (value - previous_value)
            return previous_step + fraction * (step - previous_step)
        previous_step, previous_value = step, value
    return None


def table(runs, targets, key):
    print(f"\nupdates de gradient pour atteindre un niveau de `{key}`")
    print(f"  (mesure comme le ResNet complet, s=1 -- la seule lecture "
          f"comparable entre bras)\n")
    header = f"{'run':<34}" + "".join(f"{t:>12.0%}" for t in targets) + f"{'final':>10}"
    print(header)
    print("-" * len(header))

    reference = None
    for run in runs:
        cells = []
        for target in targets:
            reached = updates_to_reach(run, key, target)
            cells.append(f"{reached:>12,.0f}" if reached else f"{'--':>12}")
        _, values = series(run, key)
        print(f"{run['label']:<34}" + "".join(cells) + f"{values[-1]:>10.4f}")
        if reference is None:
            reference = run

    if len(runs) > 1:
        print(f"\nrelatif a '{reference['label']}' (>1 = plus lent a y arriver)")
        print(header)
        print("-" * len(header))
        for run in runs[1:]:
            cells = []
            for target in targets:
                a = updates_to_reach(reference, key, target)
                b = updates_to_reach(run, key, target)
                cells.append(f"{b / a:>12.2f}" if (a and b) else f"{'--':>12}")
            _, values = series(run, key)
            _, base = series(reference, key)
            cells.append(f"{values[-1] - base[-1]:>+10.4f}")
            print(f"{run['label']:<34}" + "".join(cells))


def plot(runs, outdir, key):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6))
    (ax_cmp, ax_loss), (ax_raw, ax_s) = axes

    for index, run in enumerate(runs):
        colour = CYCLE[index % len(CYCLE)]
        end = ramp_end_update(run)

        for ax, metric in ((ax_cmp, key), (ax_loss, "train_loss"),
                           (ax_raw, "test_acc")):
            steps, values = series(run, metric)
            if steps:
                ax.plot(steps, values, color=colour, lw=1.7, label=run["label"])
        steps, values = series(run, "s_mean")
        ax_s.plot(steps, values, color=colour, lw=1.7, label=run["label"])

        # Where the arm stops being a different model from the baseline.
        if end is not None:
            for ax in (ax_cmp, ax_loss, ax_raw, ax_s):
                ax.axvline(end, color=colour, lw=0.9, ls=":", alpha=0.7)

    # Titles stay short: two panels side by side at this width run their
    # titles into each other, and the long-form warning belongs in the module
    # docstring, not overprinted on the figure.
    ax_cmp.set_title(f"{key}  --  read as the full ResNet  (comparable)")
    ax_cmp.set_ylabel("accuracy at s=1")
    ax_cmp.legend(loc="lower right", fontsize=7)

    ax_loss.set_title("train loss  --  NOT comparable while s < 1")
    ax_loss.set_ylabel("loss")
    ax_loss.set_yscale("log")

    ax_raw.set_title("test accuracy at each arm's own s")
    ax_raw.set_ylabel("accuracy")

    ax_s.set_title("s  (dotted line marks where it reaches 1)")
    ax_s.set_ylabel("s")
    ax_s.set_ylim(-0.05, 1.05)

    for ax in axes.flat:
        ax.set_xlabel("gradient updates")
        ax.grid(alpha=0.15, linewidth=0.5)
    fig.suptitle("continuation vs baseline, against gradient updates  --  "
                 "each arm minimises its own L_s, so only the top-left panel "
                 "compares like with like", x=0.5, y=1.02, fontsize=10)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "compare_updates.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Overlay finished runs against gradient updates.")
    parser.add_argument("folders", nargs="+",
                        help="run directories, or one folder holding them")
    parser.add_argument("--metric", default="test_acc_at_s1",
                        help="the comparable curve (default: test_acc_at_s1)")
    parser.add_argument("--target", type=float, action="append", default=None,
                        help="accuracy level for the table; repeatable")
    parser.add_argument("--out", default="runs/figures",
                        help="where compare_updates.png goes")
    args = parser.parse_args()

    found = []
    for folder in args.folders:
        run = load(folder)
        if run is not None:
            found.append(run)
            continue
        if not os.path.isdir(folder):
            raise SystemExit(f"!! not a directory: {folder}")
        for entry in sorted(os.listdir(folder)):
            path = os.path.join(folder, entry)
            if os.path.isdir(path) and not os.path.islink(path):
                run = load(path)
                if run is not None:
                    found.append(run)

    if not found:
        raise SystemExit("!! no finished run found: need results.json and "
                         "history.jsonl, which are only written when a run ends")
    # Baseline arms first, so the relative table is expressed against one.
    found.sort(key=lambda r: (r["cfg"]["s_schedule"] != "const", r["name"]))

    print(f"{len(found)} run(s):")
    for run in found:
        print(f"  {run['label']:<30} {run['cfg']['epochs']:>3} epochs x "
              f"{run['per_epoch']} updates, seeds {run['seeds']}")

    targets = args.target or [0.70, 0.85, 0.90, 0.93]
    table(found, targets, args.metric)

    import matplotlib
    matplotlib.use("Agg")
    _theme(matplotlib)
    print(f"\nwrote {plot(found, args.out, args.metric)}")


if __name__ == "__main__":
    main()
