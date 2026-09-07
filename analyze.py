#!/usr/bin/env python3
"""Plot what a finished run produced. Takes the folder the results live in.

    python3 analyze.py runs/latest                 # one run: curves + per-class
    python3 analyze.py runs                         # every run: overlaid
    python3 analyze.py out/runs/20260907-113045_baseline_resnet18

The argument is a directory. If it holds a results.json it is treated as one
run; otherwise every immediate subdirectory that holds one is loaded and the
runs are compared. Figures land in <folder>/figures/.

Deliberately NOT part of src/, so matplotlib never has to exist inside the
kernel. The kernel's job is to produce numbers; reading them is a local activity.

Curves are the mean across seeds with a min..max band, never a single seed
dressed up as the result -- a CIFAR-10 baseline moves by half a point between
seeds, which is most of the difference anyone is trying to see.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from cifarbase.report import CLASSES

# Neutral ink for train (the thing you already know), colour for the splits you
# are actually reading. Held fixed across every figure so they can be learned
# once.
TRAIN = "#52514e"
VAL = "#c9a227"
TEST = "#2a78d6"
MUTED = "#8f8e88"
CYCLE = ["#2a78d6", "#eb6834", "#1baf7a", "#9a5fd0", "#c9a227", "#52514e",
         "#d63a6a", "#3aa8b8"]


def _theme(mpl):
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 120, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlelocation": "left", "axes.titlesize": 10,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    })


def load_run(directory):
    """One run's results.json plus its history rows, or None if incomplete."""
    results = os.path.join(directory, "results.json")
    if not os.path.isfile(results):
        return None
    with open(results) as fh:
        payload = json.load(fh)

    history = []
    jsonl = os.path.join(directory, "history.jsonl")
    if os.path.isfile(jsonl):
        with open(jsonl) as fh:
            history = [json.loads(line) for line in fh if line.strip()]

    payload["history"] = history
    payload["dir"] = directory
    # realpath, so pointing at runs/latest labels the figure with the run it
    # resolves to rather than the word "latest".
    payload["name"] = os.path.basename(os.path.normpath(os.path.realpath(directory)))
    return payload


def find_runs(folder):
    """Either the folder itself is a run, or its subdirectories are."""
    single = load_run(folder)
    if single is not None:
        return [single]
    found = []
    for entry in sorted(os.listdir(folder)):
        path = os.path.join(folder, entry)
        # islink skips runs/latest, which would otherwise plot twice.
        if os.path.isdir(path) and not os.path.islink(path):
            run = load_run(path)
            if run is not None:
                found.append(run)
    return found


def band(history, key):
    """Per-epoch mean across seeds, with the min and max as the band.

    Returns empty lists when no row carries the key, which is what happens to
    val_loss and val_acc whenever val_size is 0.
    """
    per_epoch = {}
    for row in history:
        value = row.get(key)
        if value is None:
            continue
        per_epoch.setdefault(row["epoch"], []).append(value)
    epochs = sorted(per_epoch)
    if not epochs:
        return [], [], [], []
    values = [per_epoch[e] for e in epochs]
    return (epochs,
            [sum(v) / len(v) for v in values],
            [min(v) for v in values],
            [max(v) for v in values])


def _marker(epochs):
    """Short runs need markers: a line through one point draws nothing at all,
    which reads as a broken script rather than as a two-epoch smoke test. Past
    ~15 epochs the markers are just noise on top of the line."""
    return {"marker": "o", "ms": 3.5} if len(epochs) <= 15 else {}


def _draw(ax, history, key, colour, label):
    """One mean line plus its across-seed band. No-op if the key is absent."""
    epochs, mean, lo, hi = band(history, key)
    if not epochs:
        return False
    ax.plot(epochs, mean, color=colour, lw=1.6, label=label, **_marker(epochs))
    if any(top > bot for top, bot in zip(hi, lo, strict=True)):
        ax.fill_between(epochs, lo, hi, color=colour, alpha=0.15, linewidth=0)
    return True


def plot_curves(run, outdir):
    """The four panels worth having for a classification run."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    history = run["history"]
    if not history:
        return None
    cfg = run["config"]
    n_seeds = len({row["seed"] for row in history})

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    (ax_loss, ax_acc), (ax_lr, ax_gap) = axes

    _draw(ax_loss, history, "train_loss", TRAIN, "train")
    _draw(ax_loss, history, "val_loss", VAL, "val")
    _draw(ax_loss, history, "test_loss", TEST, "test")
    ax_loss.set_title("cross-entropy")
    ax_loss.set_ylabel("loss")
    ax_loss.legend()

    _draw(ax_acc, history, "train_acc_batchwise", TRAIN, "train (batchwise)")
    _draw(ax_acc, history, "val_acc", VAL, "val")
    _draw(ax_acc, history, "test_acc", TEST, "test")
    ax_acc.set_title("accuracy")
    ax_acc.set_ylabel("accuracy")
    ax_acc.legend(loc="lower right")

    _draw(ax_lr, history, "lr", MUTED, "lr")
    ax_lr.set_title(f"learning rate  ({cfg['schedule']}, "
                    f"{cfg['warmup_epochs']} warmup epochs)")
    ax_lr.set_ylabel("lr")

    # train accuracy here is the running batchwise figure, which is measured
    # under augmentation and so understates the clean train accuracy. The gap is
    # therefore a lower bound on the real one -- results.json has the clean
    # number if you need it exactly.
    epochs, train_mean, _, _ = band(history, "train_acc_batchwise")
    _, test_mean, _, _ = band(history, "test_acc")
    if epochs and test_mean:
        gap = [t - e for t, e in zip(train_mean, test_mean, strict=True)]
        ax_gap.plot(epochs, gap, color="#eb6834", lw=1.6, **_marker(epochs))
        ax_gap.axhline(0, color="#0b0b0b", lw=0.8)
    ax_gap.set_title("generalisation gap  (batchwise train - test)")
    ax_gap.set_ylabel("accuracy difference")

    for ax in (ax_lr, ax_gap):
        ax.set_xlabel("epoch")
    for ax in axes.flat:
        # Epochs are integers. Without this a two-epoch run gets ticked at 0.4.
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    acc = run["stats"].get("test_acc_selected", {})
    fig.suptitle(f"{run['name']}  --  {cfg['arch']}, {cfg['epochs']} epochs, "
                 f"{n_seeds} seed(s), test acc "
                 f"{acc.get('mean', float('nan')):.4f}",
                 x=0.5, y=1.01, fontsize=11)

    path = os.path.join(outdir, "curves.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_per_class(run, outdir):
    """Which classes the model is actually failing, worst first."""
    import matplotlib.pyplot as plt

    per_seed = [r["per_class"] for r in run["runs"] if r.get("per_class")]
    if not per_seed:
        return None

    means = [sum(col) / len(col) for col in zip(*per_seed, strict=True)]
    order = sorted(range(len(CLASSES)), key=lambda i: means[i])
    overall = run["stats"].get("test_acc_selected", {}).get("mean")

    fig, ax = plt.subplots(figsize=(6.4, 4))
    positions = range(len(order))
    ax.barh(positions, [means[i] for i in order], color=TEST, alpha=0.75,
            height=0.62)
    for y, i in zip(positions, order, strict=True):
        # Each seed as a dot, so a class that is unstable across seeds does not
        # hide behind its mean.
        ax.scatter([row[i] for row in per_seed], [y] * len(per_seed),
                   color="#0b0b0b", s=10, alpha=0.6, zorder=3)
    if overall is not None:
        ax.axvline(overall, color="#eb6834", lw=1.2,
                   label=f"overall {overall:.4f}")
        ax.legend(loc="lower right")
    ax.set_yticks(list(positions))
    ax.set_yticklabels([CLASSES[i] for i in order])
    # barh puts position 0 at the bottom, which would read best-first top-down
    # and contradict the title.
    ax.invert_yaxis()
    ax.set_xlabel("accuracy")
    ax.set_title(f"per-class accuracy, worst first  ({len(per_seed)} seed(s))")

    path = os.path.join(outdir, "per_class.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_compare(runs, outdir):
    """Every run's test curve on one pair of axes, plus the final numbers."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    usable = [r for r in runs if r["history"]]
    if len(usable) < 2:
        return None

    fig, (ax_curve, ax_final) = plt.subplots(
        1, 2, figsize=(13, 4.4),
        gridspec_kw={"width_ratios": [1.7, 1], "wspace": 0.12})

    for i, run in enumerate(usable):
        colour = CYCLE[i % len(CYCLE)]
        _draw(ax_curve, run["history"], "test_acc", colour, run["name"])
    ax_curve.set_title("test accuracy")
    ax_curve.set_xlabel("epoch")
    ax_curve.set_ylabel("accuracy")
    ax_curve.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_curve.legend(loc="lower right")

    # +-2 sem, because that is roughly the width at which two of these stop
    # overlapping by chance. A bar that crosses another is not a result.
    for i, run in enumerate(usable):
        acc = run["stats"].get("test_acc_selected", {})
        if "mean" not in acc:
            continue
        err = 2 * acc.get("sem", 0.0)
        ax_final.errorbar(acc["mean"], i, xerr=err, fmt="o",
                          color=CYCLE[i % len(CYCLE)], capsize=4, ms=6)
    ax_final.set_yticks(range(len(usable)))
    ax_final.set_yticklabels([r["name"] for r in usable], fontsize=7)
    # Run names are long. Left-side ticks grow into the curve panel and collide
    # with its legend, so they go outward on the right instead.
    ax_final.yaxis.tick_right()
    ax_final.spines["right"].set_visible(True)
    ax_final.spines["left"].set_visible(False)
    ax_final.invert_yaxis()
    ax_final.set_xlabel("test accuracy (+-2 sem across seeds)")
    ax_final.set_title("final, with the spread that matters")

    path = os.path.join(outdir, "compare.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def print_table(runs):
    print(f"{'run':<44} {'arch':<9} {'ep':>4} {'n':>3} "
          f"{'test acc':>10} {'+-2sem':>8} {'train':>8}")
    for run in runs:
        cfg = run["config"]
        acc = run["stats"].get("test_acc_selected", {})
        train = run["stats"].get("train_acc", {})
        sem = acc.get("sem")
        print(f"{run['name']:<44} {cfg['arch']:<9} {cfg['epochs']:>4} "
              f"{acc.get('n', 0):>3} {acc.get('mean', float('nan')):>10.4f} "
              f"{(2 * sem if sem else float('nan')):>8.4f} "
              f"{train.get('mean', float('nan')):>8.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot a run directory, or compare every run inside a folder.")
    parser.add_argument("folder", help="a run directory, or a folder of them")
    parser.add_argument("--out", default=None,
                        help="where the pngs go (default: <folder>/figures)")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        raise SystemExit(f"!! not a directory: {args.folder}")

    runs = find_runs(args.folder)
    if not runs:
        raise SystemExit(f"!! no results.json in {args.folder} or its "
                         f"subdirectories. Point this at a run directory, or at "
                         f"the runs/ folder holding them.")

    print_table(runs)

    import matplotlib
    matplotlib.use("Agg")           # no display on Kaggle, and none needed here
    _theme(matplotlib)

    outdir = args.out or os.path.join(args.folder, "figures")
    os.makedirs(outdir, exist_ok=True)

    made = []
    if len(runs) == 1:
        made += [plot_curves(runs[0], outdir), plot_per_class(runs[0], outdir)]
    else:
        made.append(plot_compare(runs, outdir))
        # Per-run curves too, each in its own directory, so a folder of runs
        # gives both the comparison and every individual set of curves.
        for run in runs:
            sub = os.path.join(run["dir"], "figures")
            os.makedirs(sub, exist_ok=True)
            made += [plot_curves(run, sub), plot_per_class(run, sub)]

    print()
    for path in made:
        if path:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
