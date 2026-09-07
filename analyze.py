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

    probes = []
    probe_jsonl = os.path.join(directory, "probes.jsonl")
    if os.path.isfile(probe_jsonl):
        with open(probe_jsonl) as fh:
            probes = [json.loads(line) for line in fh if line.strip()]

    payload["history"] = history
    payload["probes"] = probes
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


def by_step(probes, key):
    """Per-step mean across seeds, with min..max as the band.

    band() above keys on epoch; the anchor series are sampled every probe_every
    STEPS, several times an epoch, and the whole point of them is the shape
    between epoch boundaries.
    """
    per_step = {}
    for row in probes:
        value = row.get(key)
        if value is None or not isinstance(value, (int, float)):
            continue
        per_step.setdefault(row["step"], []).append(value)
    steps = sorted(per_step)
    if not steps:
        return [], [], [], []
    values = [per_step[s] for s in steps]
    return (steps,
            [sum(v) / len(v) for v in values],
            [min(v) for v in values],
            [max(v) for v in values])


def _draw_steps(ax, probes, key, colour, label, dashed=False):
    steps, mean, lo, hi = by_step(probes, key)
    if not steps:
        return False
    ax.plot(steps, mean, color=colour, lw=1.6, label=label,
            ls="--" if dashed else "-")
    if any(top > bot for top, bot in zip(hi, lo, strict=True)):
        ax.fill_between(steps, lo, hi, color=colour, alpha=0.15, linewidth=0)
    return True


def spearman(xs, ys):
    """Rank correlation, without pulling in scipy for one number.

    Average ranks for ties, then Pearson on the ranks -- which is what Spearman
    is. Ties matter here: a saturated drift signal produces long stretches of
    identical d, and integer ranking would invent an ordering inside them.
    """
    if len(xs) < 3:
        return float("nan")

    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = shared
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def plot_anchor(run, outdir):
    """The homotopy coordinate, and everything needed to believe it.

    Six panels, all against STEP rather than epoch, because that is the axis the
    controller lives on. Two of them decide whether the rest means anything:

    d vs d* is whether the loop actually tracked its schedule. If it did not, no
    accuracy number from this run means what it claims to.

    The kernel scale is whether d was measuring what it is supposed to measure.
    d^2 = scale^2 - 2 a scale + 1, so a drift of 29 with an alignment of 0.27 is
    a kernel whose NORM grew 29x -- feature learning is not what was regulated.
    The alignment panel next to it carries 1 - a, the same quantity with the
    scale divided out, which is what probe_signal=alignment steers by.
    """
    import matplotlib.pyplot as plt

    probes = run.get("probes") or []
    if not probes:
        return None
    cfg = run["config"]

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10.2))
    (ax_drift, ax_lam), (ax_align, ax_dist), (ax_hess, ax_pay) = axes
    # Room for the right-hand twin axes, whose labels otherwise land on top of
    # the next panel's left label.
    fig.subplots_adjust(wspace=0.34, hspace=0.45, top=0.93)

    drawn = _draw_steps(ax_drift, probes, "d", TEST, "d (measured)")
    _draw_steps(ax_drift, probes, "d_ema", "#eb6834", "d (EMA, what the loop sees)")
    _draw_steps(ax_drift, probes, "d_target", "#0b0b0b", "d* (target)", dashed=True)
    ax_drift.set_title("the controlled quantity: drift against its schedule")
    ax_drift.set_ylabel("drift signal")
    # Guarded: a run shorter than anchor_reference_step has no drift columns at
    # all, and legend() on an empty axis is a warning rather than a legend.
    if drawn:
        ax_drift.legend(loc="upper left")
    twin_scale = ax_drift.twinx()
    if _draw_steps(twin_scale, probes, "scale_ntk", "#1baf7a", "||K_t||/||K_0||"):
        twin_scale.set_yscale("log")
        twin_scale.set_ylabel("kernel scale", color="#1baf7a")
        twin_scale.spines["right"].set_visible(True)
        twin_scale.legend(loc="lower right")

    _draw_steps(ax_lam, probes, "lam", VAL, "lambda")
    _draw_steps(ax_lam, probes, "lam_applied", MUTED, "lambda applied")
    for bound, label in (("lambda_min", "clip lo"), ("lambda_max", "clip hi")):
        value = run["stats"].get(bound, {}).get("mean")
        if value:
            ax_lam.axhline(value, color="#d63a6a", lw=0.9, ls=":")
            ax_lam.annotate(label, (0, value), fontsize=7, color="#d63a6a",
                            va="bottom")
    _, lam_mean, _, _ = by_step(probes, "lam")
    if lam_mean and min(lam_mean) > 0:
        ax_lam.set_yscale("log")
    ax_lam.set_title(f"lambda  ({cfg.get('anchor_mode', '?')})")
    ax_lam.set_ylabel("lambda")
    ax_lam.legend(loc="best")

    # Scale-free view of the same measurement, on one bounded axis.
    drawn = _draw_steps(ax_align, probes, "a_ntk", TEST, "a  (NTK)")
    _draw_steps(ax_align, probes, "a_feature", "#eb6834", "a  (features)")
    _draw_steps(ax_align, probes, "dalign_ntk", MUTED,
                "1 - a  (the scale-free signal)", dashed=True)
    ax_align.set_title("kernel alignment: geometry alone, scale divided out")
    ax_align.set_ylabel("cosine alignment with K_0")
    if drawn:
        ax_align.legend(loc="best")
    else:
        _blank(ax_align)

    _draw_steps(ax_dist, probes, "dist_rel", TEST, "||w-w0|| / ||w0||")  # always present
    twin_eta = ax_dist.twinx()
    # eta_eff on its own log axis: it spans layers whose norms differ by an order
    # of magnitude, and it is the defence against the reading that an arm simply
    # trained at a better effective learning rate.
    if _draw_steps(twin_eta, probes, "eta_eff_mean", "#1baf7a", "mean eta_eff"):
        twin_eta.set_yscale("log")
        twin_eta.set_ylabel("eta / ||w_l||^2", color="#1baf7a")
        twin_eta.spines["right"].set_visible(True)
    ax_dist.set_title("distance from the anchor, and the effective lr it implies")
    ax_dist.set_ylabel("||w-w0|| / ||w0||")
    ax_dist.legend(loc="upper left")

    if _draw_steps(ax_hess, probes, "hessian_lambda_min", "#9a5fd0", "lambda_min(H)"):
        ax_hess.axhline(0, color="#0b0b0b", lw=0.8)
        ax_hess.set_title("smallest Hessian eigenvalue  (a crossing is a branch event)")
        ax_hess.set_ylabel("eigenvalue")
    else:
        ax_hess.set_title("lambda_min(H) not logged  (hessian_every = 0)")
        _blank(ax_hess)

    # The payoff: accuracy bought per unit of drift spent. Read off the epoch
    # rows, which carry both the last probe of the epoch and the test accuracy.
    history = run.get("history") or []
    pairs = sorted((row["d"], row["test_acc"], row["epoch"]) for row in history
                   if isinstance(row.get("d"), (int, float))
                   and isinstance(row.get("test_acc"), (int, float)))
    if len(pairs) >= 2:
        ax_pay.plot([p[0] for p in pairs], [p[1] for p in pairs],
                    color=TEST, lw=1.4, marker="o", ms=3)
        ax_pay.set_xlabel("drift spent  d")
        ax_pay.set_ylabel("test accuracy")
        ax_pay.set_title("accuracy bought per unit of feature learning spent")
    else:
        ax_pay.set_title("no epoch carries both drift and accuracy yet")
        _blank(ax_pay)

    for ax in (ax_drift, ax_lam, ax_align, ax_dist, ax_hess):
        ax.set_xlabel("step")
    fig.suptitle(f"{run['name']}  --  anchor {cfg.get('anchor_mode', '?')}, "
                 f"signal {cfg.get('probe_signal', 'drift')} on the "
                 f"{cfg.get('probe_kernel', 'ntk')} kernel, "
                 f"D_max {cfg.get('anchor_dmax', 0):.3f}, "
                 f"beta {cfg.get('anchor_beta', 0):.3g}",
                 x=0.5, y=0.975, fontsize=11)

    path = os.path.join(outdir, "anchor.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_alloc(run, outdir):
    """What the allocation did, and whether it was entitled to.

    Four panels. The two on the left are the allocation itself: a_g in log space,
    which is the controller's own coordinate, and the lambda_g it produces once
    the level is folded back in. The two on the right are the evidence -- the
    tension each group reported, which is what the slow loop integrates, and the
    per-group drift, which is what it is ultimately spending.

    Drawn for the single-lambda arms too, where every a_g is flat at zero by
    construction. That is not a wasted figure: the d_g and tau_g panels are then
    the picture of how differently the groups behave under ONE lambda, which is
    the entire premise of allocating at all. If those two panels are flat as
    well, the allocation has nothing to find and the honest result is that it
    changes nothing.
    """
    import matplotlib.pyplot as plt

    probes = run.get("probes") or []
    groups = (run.get("runs") or [{}])[0].get("anchor_groups")
    if not probes or not groups:
        return None
    cfg = run["config"]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0))
    (ax_a, ax_lam), (ax_tau, ax_d) = axes
    fig.subplots_adjust(wspace=0.24, hspace=0.42, top=0.90)

    panels = ((ax_a, "a", "allocation a_g  (log lambda offset)", False),
              (ax_lam, "lam", "lambda_g", True),
              (ax_tau, "tau", "tension tau_g = |grad_g| |w0_g|^2 / "
                              "(lambda_g |w_g - w0_g|)", True),
              (ax_d, "d", "per-group drift d_g", False))
    for ax, prefix, title, log in panels:
        drawn = False
        for index, group in enumerate(groups):
            drawn |= _draw_steps(ax, probes, f"{prefix}_{group}",
                                 CYCLE[index % len(CYCLE)], group)
        ax.set_title(title)
        ax.set_xlabel("step")
        if log and drawn:
            ax.set_yscale("log")
        if drawn:
            ax.legend(ncol=2)
        else:
            _blank(ax)

    # The line a_g cannot cross, and the line it must not drift off. Both are
    # invariants of the update rather than outcomes, so a plot that violates
    # either is a bug and not a finding.
    clip = float(cfg.get("anchor_alloc_clip_decades", 1.0)) * math.log(10.0)
    for sign in (-1.0, 1.0):
        ax_a.axhline(sign * clip, color=MUTED, lw=0.8, ls=":")
    ax_a.axhline(0.0, color="#0b0b0b", lw=0.8)

    fig.suptitle(f"{run['name']}  --  allocation "
                 f"{cfg.get('anchor_alloc', 'off')}, every "
                 f"{cfg.get('anchor_alloc_every', '?')} probes, beta_a = "
                 f"{cfg.get('anchor_alloc_beta_frac', 0) * cfg.get('anchor_beta', 0):.3g}"
                 f", gamma {cfg.get('anchor_alloc_shrink', 0):g}",
                 x=0.5, y=0.975, fontsize=11)

    path = os.path.join(outdir, "alloc.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _blank(ax):
    """An empty panel that reads as "nothing to show" rather than as a plot that
    failed to draw."""
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)


def plot_drift_validation(run, outdir):
    """The cheap estimator against the real one, over a whole trajectory.

    This figure is a decision, not decoration: the penultimate-feature Gram costs
    one forward pass instead of R JVPs, and may be substituted for the NTK
    estimator only if Spearman rho over the trajectory clears 0.95. Agreement at
    one point in training is worthless -- the two disagree exactly where the
    controller is doing something -- which is why the correlation is taken over
    every probe rather than at the end.

    One trap, and the figure now names it. If both series rise monotonically with
    step -- which is what happens on a constant-lambda or unanchored arm -- then
    their rank correlation is 1 by construction, whatever the two estimators
    actually think, because Spearman only sees the ordering and time already
    orders both. Measured on a short const-lambda run: rho = 1.0000 while the two
    drifts differed by more than a factor of two (28.9 against 13.8). The
    substitution can only be justified on an arm where lambda moves enough to
    make the ordering non-trivial.
    """
    import matplotlib.pyplot as plt

    probes = [r for r in run.get("probes") or []
              if isinstance(r.get("d_ntk"), (int, float))
              and isinstance(r.get("d_feature"), (int, float))]
    if len(probes) < 5:
        return None

    ntk = [r["d_ntk"] for r in probes]
    feature = [r["d_feature"] for r in probes]
    rho = spearman(ntk, feature)

    # Both monotone in step means the ranks agree for free.
    def monotone(values):
        return (all(b >= a for a, b in zip(values, values[1:], strict=False))
                or all(b <= a for a, b in zip(values, values[1:], strict=False)))

    trivial = monotone(ntk) and monotone(feature)

    fig, (ax_scatter, ax_time) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fig.subplots_adjust(wspace=0.26, top=0.82)
    ax_scatter.scatter(ntk, feature, s=12, alpha=0.6,
                       c=[r["step"] for r in probes], cmap="viridis")
    ax_scatter.set_xlabel("d from the NTK sketch (primary)")
    ax_scatter.set_ylabel("d from the feature Gram (cheap)")
    ax_scatter.set_title("colour is step; doubling back = ranks disagree")

    _draw_steps(ax_time, probes, "d_ntk", TEST, "NTK")
    _draw_steps(ax_time, probes, "d_feature", "#eb6834", "feature Gram")
    ax_time.set_xlabel("step")
    ax_time.set_ylabel("relative drift")
    ax_time.legend()
    ax_time.set_title("both estimators over the run")

    if trivial:
        verdict = ("UNINFORMATIVE: both series are monotone in step, so rho is 1 "
                   "by construction. Read this on an arm where lambda moves.")
    else:
        verdict = (f"the feature Gram is "
                   f"{'ADMISSIBLE' if rho >= 0.95 else 'NOT admissible'} as a "
                   f"substitute (threshold 0.95)")
    fig.suptitle(f"{run['name']}  --  Spearman rho = {rho:.4f} over "
                 f"{len(probes)} probes\n{verdict}", x=0.5, y=0.99, fontsize=9)

    path = os.path.join(outdir, "drift_validation.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def check_pilot(runs):
    """Stage 1's go/no-go, and stage 2's two numbers.

    Separation. d(t) must order monotonically in lambda: more anchor, less drift.
    Checked as a rank correlation between lambda and d at several points in
    training, not just at the end, because an arm can converge to the same drift
    by a different route. A non-negative correlation means lambda has no
    authority over drift and the adaptive arm cannot work at any beta -- the
    design is dead and no amount of tuning revives it.

    The gain. g = -d(d_ema)/d(log lambda) at mid-training, by OLS over the swept
    lambdas. beta = 1/g makes the loop gain 1, i.e. one probe to correct an error.
    lambda = 0 is excluded from the fit (no logarithm) but reported: it is the
    free drift curve, and it is what D_max is a fraction of.
    """
    arms, excluded = [], []
    for run in runs:
        probes = run.get("probes") or []
        if not probes:
            continue
        # Only the ANCHORED arms may enter the lambda-vs-drift fit. An arm running
        # weight decay, or with the anchor off, has its drift set by a different
        # regulariser entirely -- letting the tuned-decay reference in as a
        # "lambda = 0" point would put a number from another experiment into the
        # separation correlation and quietly bias the gain.
        cfg = run["config"]
        if cfg.get("anchor_mode", "off") == "off" or cfg.get("weight_decay", 0):
            excluded.append((run["name"], cfg.get("anchor_mode", "off"),
                             cfg.get("weight_decay", 0)))
            continue
        # Rows written before K_0 was captured carry no drift columns at all --
        # a legitimate state, not a hole to interpolate over.
        probes = [r for r in probes if isinstance(r.get("d_ema"), (int, float))]
        if not probes:
            continue
        arms.append({
            "lam": float(run["config"].get("anchor_lambda", 0.0)),
            "name": run["name"],
            "acc": run["stats"].get("test_acc_selected", {}).get("mean", float("nan")),
            "probes": probes,
        })
    arms.sort(key=lambda a: a["lam"])
    for name, mode, decay in excluded:
        print(f"  excluded from the fit: {name} (anchor_mode={mode}, "
              f"weight_decay={decay:g}) -- its drift is not lambda's doing")
    if len(arms) < 3:
        print(f"!! only {len(arms)} pilot runs with probe rows: the separation "
              f"check needs at least 3. Run `make pilot` first.")
        return False

    def drift_at(arm, fraction):
        target = fraction * max(r["step"] for r in arm["probes"])
        row = min(arm["probes"], key=lambda r: abs(r["step"] - target))
        return row["d_ema"]

    print(f"\npilot: {len(arms)} arms")
    print("  " + "".join(f"{h:>13}" for h in
                         ["lambda", "d@25%", "d@50%", "d@75%", "d@100%", "test acc"]))
    fractions = (0.25, 0.5, 0.75, 1.0)
    for arm in arms:
        cells = [f"{arm['lam']:.4g}"] + [f"{drift_at(arm, f):.4f}" for f in fractions]
        cells.append(f"{arm['acc']:.4f}")
        print("  " + "".join(f"{c:>13}" for c in cells))

    lams = [a["lam"] for a in arms]
    print("\n  separation (rank correlation of lambda against d; want <= -0.9):")
    correlations = []
    for fraction in fractions:
        rho = spearman(lams, [drift_at(a, fraction) for a in arms])
        correlations.append(rho)
        print(f"    at {fraction:>5.0%} of training   rho = {rho:+.4f}")

    final = [drift_at(a, 1.0) for a in arms]
    span = max(final) / max(min(final), 1e-12)
    separated = all(rho <= -0.9 for rho in correlations) and span > 1.5
    print(f"    d_T spread across the sweep: {span:.2f}x  (want > 1.5x)")
    print(f"\n  -> {'SEPARATED' if separated else 'NOT SEPARATED'}")
    if not separated:
        print("     lambda does not control drift monotonically over this sweep.")
        print("     There is nothing for a controller to control. Do NOT run an")
        print("     adaptive arm: stop and report this, it is a real negative")
        print("     result about the mechanism rather than a tuning problem.")
        return False

    # g at mid-training, over the arms with a logarithm.
    fitted = [(math.log(a["lam"]), drift_at(a, 0.5)) for a in arms if a["lam"] > 0]
    if len(fitted) < 3:
        print("     (need at least 3 non-zero lambdas to fit the gain)")
        return False
    mean_x = sum(x for x, _ in fitted) / len(fitted)
    mean_y = sum(y for _, y in fitted) / len(fitted)
    slope = (sum((x - mean_x) * (y - mean_y) for x, y in fitted)
             / sum((x - mean_x) ** 2 for x, _ in fitted))
    gain = -slope
    beta = 1.0 / gain

    best = max((a for a in arms if a["lam"] > 0),
               key=lambda a: (a["acc"] if a["acc"] == a["acc"] else -1))
    dmax = drift_at(best, 1.0)
    free = next((a for a in arms if a["lam"] == 0.0), None)

    print(f"\n  gain      g = -d(d_ema)/d(log lambda) at 50% = {gain:.6g}")
    print(f"  gain      beta = 1/g = {beta:.6g}   (loop gain beta*g = 1.0, "
          f"inside the stable band 0 < beta*g < 2)")
    print(f"  budget    best fixed lambda = {best['lam']:.4g} "
          f"(test acc {best['acc']:.4f}), its d_T = {dmax:.6g}")
    if free is not None:
        print(f"  budget    free drift at lambda = 0 reached d_T = "
              f"{drift_at(free, 1.0):.6g}, so the budget above is "
              f"{dmax / max(drift_at(free, 1.0), 1e-12):.2%} of unconstrained")
    if gain <= 0:
        print("  !! the fitted gain is not positive: beta = 1/g would invert the "
              "feedback. Treat the separation verdict above as unreliable.")
        return False

    print("\n  paste into src/cifarbase/configs/anchor.yaml:")
    print(f"    anchor_beta: {beta:.6g}")
    print(f"    anchor_dmax: {dmax:.6g}")
    return True


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
    parser.add_argument("--pilot", action="store_true",
                        help="stage 1's go/no-go: does lambda control drift "
                             "monotonically? Then print beta and D_max.")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        raise SystemExit(f"!! not a directory: {args.folder}")

    runs = find_runs(args.folder)
    if not runs:
        raise SystemExit(f"!! no results.json in {args.folder} or its "
                         f"subdirectories. Point this at a run directory, or at "
                         f"the runs/ folder holding them.")

    print_table(runs)

    if args.pilot:
        raise SystemExit(0 if check_pilot(runs) else 1)

    import matplotlib
    matplotlib.use("Agg")           # no display on Kaggle, and none needed here
    _theme(matplotlib)

    outdir = args.out or os.path.join(args.folder, "figures")
    os.makedirs(outdir, exist_ok=True)

    made = []
    if len(runs) == 1:
        made += [plot_curves(runs[0], outdir), plot_per_class(runs[0], outdir),
                 plot_anchor(runs[0], outdir), plot_alloc(runs[0], outdir),
                 plot_drift_validation(runs[0], outdir)]
    else:
        made.append(plot_compare(runs, outdir))
        # Per-run curves too, each in its own directory, so a folder of runs
        # gives both the comparison and every individual set of curves.
        for run in runs:
            sub = os.path.join(run["dir"], "figures")
            os.makedirs(sub, exist_ok=True)
            made += [plot_curves(run, sub), plot_per_class(run, sub),
                     plot_anchor(run, sub), plot_alloc(run, sub),
                     plot_drift_validation(run, sub)]

    print()
    for path in made:
        if path:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
