#!/usr/bin/env python3
"""Plot what a finished run produced. Takes the folder the results live in.

    python3 analyze.py runs/latest                 # one run: curves + per-class
    python3 analyze.py runs                         # every run: overlaid
    python3 analyze.py out/runs/20260907-113045_pacing_vs_order

The argument is a directory. If it holds a results.json it is treated as one
run; otherwise every immediate subdirectory that holds one is loaded. When those
subdirectories carry an arm.json -- i.e. they are the arms of a study -- the
comparison becomes arm-aware: runs are grouped into regimes, differences are
taken against each regime's baseline PAIRED BY SEED, and the effect is split
into the part the pacing bought and the part the ordering bought. Figures and
report_tables.md land in <folder>/figures/.

Deliberately NOT part of src/, so matplotlib never has to exist inside the
kernel. The kernel's job is to produce numbers; reading them is a local activity.

Curves are the mean across seeds with a min..max band, never a single seed
dressed up as the result -- a CIFAR-10 baseline moves by half a point between
seeds, which is most of the difference anyone is trying to see.
"""
import argparse
import json
import math
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

# An arm keeps its colour in every figure and every regime, so the reader learns
# four colours once instead of re-reading a legend per panel. Grey is the
# reference on purpose: the three coloured arms are the ones being compared to
# it, and the eye should go to them.
ARM_ORDER = ["baseline", "curriculum", "random", "anti"]
ARM_COLOURS = {"baseline": "#52514e", "curriculum": "#2a78d6",
               "random": "#eb6834", "anti": "#d63a6a"}

# Widths and dashes for the lambda panel only, where the three paced arms are
# supposed to draw the same line. Nested rather than overlaid, so agreement
# looks like agreement instead of like a missing arm.
LAMBDA_STYLE = {"baseline": (1.6, "-"), "curriculum": (4.0, "-"),
                "random": (2.2, "--"), "anti": (1.0, ":")}

# What each difference isolates. The whole point of running four arms.
EFFECTS = [
    ("pacing", "random", "baseline",
     "the smaller pool and its repetition, with no difficulty information"),
    ("ordering", "curriculum", "random",
     "the difficulty ranking, at identical pacing"),
    ("direction", "curriculum", "anti",
     "easy-first against hard-first, same ranking"),
    ("total", "curriculum", "baseline",
     "the curriculum as usually reported = pacing + ordering"),
]


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

    # An arm is a fact about the experiment, written by the study driver. Read
    # from a file rather than parsed out of the directory name, which would
    # break the first time someone renamed a folder.
    arm = os.path.join(directory, "arm.json")
    if os.path.isfile(arm):
        with open(arm) as fh:
            payload["arm_meta"] = json.load(fh)
        payload["arm"] = payload["arm_meta"].get("arm")
        payload["regime"] = payload["arm_meta"].get("regime")

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


def _draw(ax, history, key, colour, label, style="-", band_too=True):
    """One mean line plus its across-seed band. No-op if the key is absent.

    `style` exists for the comparison figure, where colour already encodes which
    run a line belongs to and the split has to be encoded by something else.
    """
    epochs, mean, lo, hi = band(history, key)
    if not epochs:
        return False
    ax.plot(epochs, mean, color=colour, lw=1.6, ls=style, label=label,
            **(_marker(epochs) if style == "-" else {}))
    if band_too and any(top > bot for top, bot in zip(hi, lo, strict=True)):
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
    """Every run's trajectory, split by split, plus the final numbers.

    Colour is the RUN and line style is the split, which is the opposite of
    plot_curves. A comparison figure has to let you read "which run" first: a
    reader who cannot tell two arms apart at a glance will read the panel wrong,
    and no amount of legend fixes it.

    Both loss and accuracy are drawn because they answer different questions
    under label noise -- accuracy against corrupted training labels saturates
    near the clean share, while the loss keeps moving and shows what the
    optimiser is actually doing.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    usable = [r for r in runs if r["history"]]
    if len(usable) < 2:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    (ax_acc, ax_loss), (ax_lam, ax_final) = axes

    # val only exists when val_size > 0; with the reference recipe it does not,
    # and labelling the test curve "validation" would be a lie in a figure.
    has_val = any(band(r["history"], "val_acc")[0] for r in usable)
    styles = [("test_acc", "test_loss", "-", "test"),
              ("train_acc_batchwise", "train_loss", "--", "train")]
    if has_val:
        styles.insert(1, ("val_acc", "val_loss", ":", "val"))

    for i, run in enumerate(usable):
        colour = CYCLE[i % len(CYCLE)]
        for acc_key, loss_key, style, _ in styles:
            # The band is drawn for the test split only: three overlapping bands
            # per run turns the panel into a wash.
            _draw(ax_acc, run["history"], acc_key, colour, None, style,
                  band_too=(style == "-"))
            _draw(ax_loss, run["history"], loss_key, colour, None, style,
                  band_too=(style == "-"))
        _draw(ax_lam, run["history"], "lam", colour, run["name"])

    ax_acc.set_title("accuracy")
    ax_loss.set_title("cross-entropy")
    for ax, ylabel in ((ax_acc, "accuracy"), (ax_loss, "loss")):
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Two legends, because the figure encodes two things at once. Colour goes on
    # the accuracy panel and style on the loss panel, so neither has to carry
    # run x split entries.
    ax_acc.legend(handles=[Line2D([], [], color=CYCLE[i % len(CYCLE)], lw=1.6,
                                  label=r["name"])
                           for i, r in enumerate(usable)],
                  loc="lower right", fontsize=7)
    ax_loss.legend(handles=[Line2D([], [], color=MUTED, lw=1.6, ls=s, label=n)
                            for _, _, s, n in styles], loc="upper right")

    # lambda(t) is the experiment itself: a flat line at 1.0 is a run with no
    # curriculum, and everything else is the path it took. Without this panel
    # the reader has to take the config's word for what actually ran.
    ax_lam.set_title("lambda(t)  -- share of the train set visible")
    ax_lam.set_xlabel("epoch")
    ax_lam.set_ylabel("lambda")
    ax_lam.set_ylim(0, 1.05)
    ax_lam.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_lam.legend(loc="lower right", fontsize=7)

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
    # Run names are long. Left-side ticks grow into the neighbouring panel and
    # collide with its legend, so they go outward on the right instead.
    ax_final.yaxis.tick_right()
    ax_final.spines["right"].set_visible(True)
    ax_final.spines["left"].set_visible(False)
    ax_final.invert_yaxis()
    ax_final.set_xlabel("test accuracy (+-2 sem across seeds)")
    ax_final.set_title("final, with the spread that matters")
    # A single seed has no spread, and an axis auto-scaled to two bare dots
    # turns any gap into a chasm. Anchor it to a full point of accuracy so the
    # picture cannot oversell a difference the run cannot resolve.
    lo, hi = ax_final.get_xlim()
    if hi - lo < 0.02:
        mid = (hi + lo) / 2
        ax_final.set_xlim(mid - 0.01, mid + 0.01)

    fig.tight_layout()
    path = os.path.join(outdir, "compare.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _mean_sem(values):
    """Mean and standard error, or (mean, None) when one value cannot have one."""
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return None, None, 0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None, len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var / len(values)), len(values)


def _per_seed(run, key="test_acc_selected"):
    """{seed: value} from the per-seed summaries, for pairing across arms."""
    return {r["seed"]: r[key] for r in run.get("runs", []) if key in r}


def paired_delta(arm, base, key="test_acc_selected"):
    """arm - base, seed by seed.

    Pairing is not a nicety here. The arms share a seed, so they share their
    initialisation and their corrupted labels, and most of the seed-to-seed
    spread cancels in the difference. Comparing two means with their own sems
    throws that away and needs several times as many runs to see the same effect.
    """
    a, b = _per_seed(arm, key), _per_seed(base, key)
    shared = sorted(set(a) & set(b))
    return _mean_sem([a[s] - b[s] for s in shared])


def delta_curve(arm, base, key="test_acc"):
    """Per-epoch paired difference, with its across-seed band."""
    a, b = {}, {}
    for store, run in ((a, arm), (b, base)):
        for row in run["history"]:
            store[(row["seed"], row["epoch"])] = row.get(key)
    epochs = sorted({e for _, e in a} & {e for _, e in b})
    seeds = sorted({s for s, _ in a} & {s for s, _ in b})
    out = []
    for epoch in epochs:
        diffs = [a[(s, epoch)] - b[(s, epoch)] for s in seeds
                 if a.get((s, epoch)) is not None and b.get((s, epoch)) is not None]
        mean, sem, _ = _mean_sem(diffs)
        out.append((epoch, mean, sem or 0.0))
    return ([e for e, _, _ in out], [m for _, m, _ in out],
            [s for _, _, s in out])


def arm_metrics(run, baseline_final=None):
    """The four numbers worth putting in a row, per arm."""
    epochs, mean, _, _ = band(run["history"], "test_acc")
    stats = run["stats"].get("test_acc_selected", {})
    out = {"final": stats.get("mean"), "sem": stats.get("sem"),
           "n": stats.get("n", 0), "peak": None, "peak_epoch": None,
           "auc": None, "estar": None}
    if mean:
        out["peak"] = max(mean)
        out["peak_epoch"] = epochs[mean.index(out["peak"])]
        out["auc"] = sum(mean) / len(mean)
        if baseline_final:
            # Not a fixed 85%: a 30-epoch arm reaches it and an 8-epoch arm
            # never does, and a threshold no arm crosses compares nothing. 95%
            # of the regime's own baseline is a target every regime can reach.
            target = 0.95 * baseline_final
            out["estar"] = next((e for e, v in zip(epochs, mean, strict=True)
                                 if v >= target), None)
    return out


def group_arms(runs):
    """{regime: {arm: run}} for the runs that declare both."""
    grid = {}
    for run in runs:
        if run.get("arm") and run.get("regime"):
            grid.setdefault(run["regime"], {})[run["arm"]] = run
    return grid


def regime_order(folder, grid):
    """The order the study ran them in, which is the order of priority.

    Alphabetical would put clean_long -- the null control -- first, ahead of the
    two regimes that carry the argument.
    """
    manifest = os.path.join(folder, "study.json")
    if os.path.isfile(manifest):
        with open(manifest) as fh:
            seen = [a["regime"] for a in json.load(fh).get("arms", [])]
        ordered = [r for i, r in enumerate(seen) if r not in seen[:i]]
        return [r for r in ordered if r in grid] + sorted(set(grid) - set(ordered))
    return sorted(grid)


def _pct(value, digits=2):
    return "--" if value is None else f"{100 * value:.{digits}f}"


def study_tables(grid, order, folder, outdir):
    """Two tables per regime -- where the arms landed, and what the difference
    between them is attributable to -- as markdown, printed and written out."""
    lines = ["# Study tables", ""]
    manifest = os.path.join(folder, "study.json")
    if os.path.isfile(manifest):
        with open(manifest) as fh:
            info = json.load(fh)
        lines += [f"`{info.get('study')}`, {info.get('seeds')} seeds per arm, "
                  f"{info.get('wall_min', 0):.0f} min.", ""]
        if info.get("skipped"):
            lines += [f"**Cut by the budget:** {', '.join(info['skipped'])}.", ""]
        for regime, teacher in (info.get("teachers") or {}).items():
            accs = ", ".join(f"{a:.4f}" for a in teacher.get("test_acc", []))
            lines += [f"Teacher for `{regime}`: {teacher.get('folds')} folds x "
                      f"{teacher.get('epochs')} epochs, test acc {accs}, "
                      f"misclassifies {100 * teacher.get('misclassified', 0):.1f}% "
                      f"of the train set.", ""]

    for regime in order:
        arms = grid[regime]
        base = arms.get("baseline")
        base_final = (base or {}).get("stats", {}).get(
            "test_acc_selected", {}).get("mean")
        lines += [f"## {regime}", "",
                  "| arm | test acc % | +-sem | peak % (ep) | AUC % | e* |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for name in ARM_ORDER:
            run = arms.get(name)
            if run is None:
                continue
            m = arm_metrics(run, base_final)
            peak = f"{_pct(m['peak'])} ({m['peak_epoch']})"
            lines.append(f"| {name} | {_pct(m['final'])} | "
                         f"{_pct(m['sem'], 3) if m['sem'] else '--'} | {peak} | "
                         f"{_pct(m['auc'])} | "
                         f"{m['estar'] if m['estar'] is not None else '--'} |")
        lines += ["", "| effect | points | +-sem | n | what it isolates |",
                  "| --- | --- | --- | --- | --- |"]
        sems = []
        for label, left, right, what in EFFECTS:
            if left not in arms or right not in arms:
                continue
            mean, sem, n = paired_delta(arms[left], arms[right])
            if mean is None:
                continue
            sems.append(sem or 0.0)
            lines.append(f"| {label} = {left} - {right} | {100 * mean:+.2f} | "
                         f"{_pct(sem, 3) if sem else '--'} | {n} | {what} |")
        if sems:
            lines += ["", _resolution_line(sems, base_final)]
        # The repetition the pacing buys, in the units the reader can check.
        paced = arms.get("curriculum") or arms.get("random")
        if paced and (paced.get("arm_meta") or {}).get("ramp_epochs"):
            lines += ["", _exposure_line(paced)]
        lines.append("")

    text = "\n".join(lines)
    print()
    print(text)
    path = os.path.join(outdir, "report_tables.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return path


def _resolution_line(sems, baseline_final):
    """How large an effect has to be here before it is an effect.

    Two floors, and the larger one wins. The seeds give one: 2 sem of the paired
    difference, which is where an interval stops crossing zero. The test set
    gives the other, which no number of seeds gets under -- 10 000 images cannot
    resolve a difference finer than their own binomial noise. Printed next to
    the effects so nobody has to eyeball whether +0.3 means anything.
    """
    seed_floor = 2 * max(sems) if sems and max(sems) > 0 else None
    binom = None
    if baseline_final:
        binom = 2 * math.sqrt(baseline_final * (1 - baseline_final) / 10000)
    if seed_floor is None:
        return ("*One seed per arm: no interval, and no smallest detectable "
                "difference. Every number above is a single draw.*")
    floor = max(seed_floor, binom or 0.0)
    return (f"*Resolution: 2 sem of the paired difference is "
            f"{100 * seed_floor:.2f} pt and the binomial floor on 10 000 test "
            f"images is {100 * (binom or 0):.2f} pt, so an effect under "
            f"**{100 * floor:.2f} pt** has not been shown.*")


def _exposure_line(run):
    """How many times each end of the order is drawn, over the whole run.

    A fixed step budget means a pool of k gets drawn n/k times per epoch, so the
    easy end of a curriculum is seen more often than the baseline sees anything
    and the hard end is seen less. That is the confound `random` controls for,
    and it belongs in the report as a number rather than as a caveat.
    """
    from cifarbase.scoring import exposure_profile
    cfg = run["config"]
    n = int(cfg["train_subset"]) or (50000 - int(cfg["val_size"]))
    seen = exposure_profile(run["arm_meta"], n)
    return (f"Exposure under this pacing: the easy decile of the order is drawn "
            f"{seen['easy_decile']:.1f} times over the run and the hard decile "
            f"{seen['hard_decile']:.1f}, against {seen['baseline']:.1f} each for "
            f"the baseline. Same total draws, different distribution -- which is "
            f"what `random` exists to price.")


def plot_study(grid, order, outdir):
    """One row per regime: where the arms went, what they gained on the
    baseline, and the pacing that was actually applied."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if not order:
        return None
    fig, axes = plt.subplots(len(order), 3, figsize=(14, 3.5 * len(order)),
                             squeeze=False)
    for i, regime in enumerate(order):
        arms = grid[regime]
        base = arms.get("baseline")
        ax_acc, ax_delta, ax_lam = axes[i]

        for name in ARM_ORDER:
            run = arms.get(name)
            if run is None:
                continue
            colour = ARM_COLOURS[name]
            _draw(ax_acc, run["history"], "test_acc", colour, name)
            # The paced arms MUST coincide here, so they are drawn nested --
            # thick solid under thin dotted. One visible line would otherwise be
            # ambiguous between "they agree" and "only one of them ran", and
            # that distinction is what the whole decomposition rests on.
            epochs, lam, _, _ = band(run["history"], "lam")
            if epochs:
                width, style = LAMBDA_STYLE.get(name, (1.6, "-"))
                ax_lam.plot(epochs, lam, color=colour, lw=width, ls=style,
                            label=name, **_marker(epochs))
            if base is None or name == "baseline":
                continue
            epochs, mean, sem = delta_curve(run, base)
            if not epochs:
                continue
            ax_delta.plot(epochs, mean, color=colour, lw=1.6, label=name,
                          **_marker(epochs))
            if any(sem):
                ax_delta.fill_between(
                    epochs, [m - s for m, s in zip(mean, sem, strict=True)],
                    [m + s for m, s in zip(mean, sem, strict=True)],
                    color=colour, alpha=0.15, linewidth=0)

        ax_delta.axhline(0, color=MUTED, lw=1, ls=":")
        ax_acc.set_title(f"{regime}  --  test accuracy")
        ax_delta.set_title("paired difference against the baseline")
        ax_lam.set_title("lambda(t): the pool actually used  "
                         "(the three paced arms must coincide)")
        ax_lam.set_ylim(0, 1.05)
        for ax, ylabel in ((ax_acc, "accuracy"), (ax_delta, "delta accuracy"),
                           (ax_lam, "lambda")):
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylabel)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_acc.legend(loc="lower right")
        ax_delta.legend(loc="lower right")

    fig.tight_layout()
    path = os.path.join(outdir, "study.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_effects(grid, order, outdir):
    """The figure the report is built on: what the pacing bought, what the
    ordering bought, and whether either clears its own error bar."""
    import matplotlib.pyplot as plt

    rows = [(label, left, right) for label, left, right, _ in EFFECTS]
    fig, ax = plt.subplots(figsize=(9, 1.1 + 0.55 * len(order) * len(rows)))
    colours = {"pacing": ARM_COLOURS["random"],
               "ordering": ARM_COLOURS["curriculum"],
               "direction": ARM_COLOURS["anti"], "total": ARM_COLOURS["baseline"]}

    ticks, labels, y = [], [], 0.0
    for regime in order:
        arms = grid[regime]
        for label, left, right in rows:
            if left not in arms or right not in arms:
                continue
            mean, sem, n = paired_delta(arms[left], arms[right])
            if mean is None:
                continue
            # +-2 sem of the PAIRED difference: an interval that crosses zero is
            # an effect this many seeds cannot see, and saying so in the picture
            # is cheaper than saying it in a caption nobody reads.
            ax.errorbar(100 * mean, y, xerr=200 * (sem or 0.0), fmt="o",
                        color=colours[label], capsize=4, ms=6)
            ticks.append(y)
            labels.append(f"{regime}  {label}")
            y += 1
        y += 0.6

    ax.axvline(0, color=MUTED, lw=1, ls=":")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("effect on test accuracy, points (+-2 sem, paired by seed)")
    ax.set_title("what the curriculum's effect decomposes into")
    fig.tight_layout()
    path = os.path.join(outdir, "effects.png")
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

    grid = group_arms(runs)
    made = []
    if len(runs) == 1:
        made += [plot_curves(runs[0], outdir), plot_per_class(runs[0], outdir)]
    elif grid:
        # A study: the arms are named, so compare them as arms.
        order = regime_order(args.folder, grid)
        made += [study_tables(grid, order, args.folder, outdir),
                 plot_study(grid, order, outdir),
                 plot_effects(grid, order, outdir)]
        for run in runs:
            sub = os.path.join(run["dir"], "figures")
            os.makedirs(sub, exist_ok=True)
            made += [plot_curves(run, sub)]
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
