#!/usr/bin/env python3
"""Every figure and every number in RESULTS.md, regenerated from the pulled data.

    python results.py [RUN_DIR] [--out docs/img]

RUN_DIR defaults to the CPU pilot under out-cpu/. `make pull-cpu` puts it there.

This file exists so the prose cannot drift from the data. It ends by printing a
NUMBERS block containing every scalar RESULTS.md quotes; a number in the markdown
that this script does not print is either stale or wrong. Editing a claim in the
document therefore means either finding it in this output or adding it here.

Top level rather than under src/, for the same reason analyze.py is: it imports
matplotlib, and build.py bundles src/ and nothing else, so neither ever reaches
the Kaggle kernel.

The theme, the palette and the inks are analyze.py's, so these figures sit next
to the ones `make plot` produces without looking like they came from elsewhere.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from analyze import CYCLE, MUTED, TEST, TRAIN, _theme, find_runs, spearman

DEFAULT_RUN = "out-cpu/runs/20260907-144502_cpupilot_resnet18"
# Set by main(), read by print_numbers, so the kernel log can be found relative to
# whichever run directory was actually plotted.
args_run_dir = [DEFAULT_RUN]
DEFAULT_OUT = "docs/img"

# The unanchored reference. Excluded from every lambda-ordered panel, because its
# drift is weight decay's doing and not lambda's -- the same exclusion
# analyze.check_pilot makes before fitting the gain.
BASELINE = "baseline_wd"

# Fractions of training the separation is checked at. check_pilot uses these, and
# figure 2b has to agree with `make pilot-check` or one of them is lying.
FRACTIONS = (0.25, 0.50, 0.75, 1.00)
SEPARATION_THRESHOLD = -0.9

GROUPS = ("stem", "stage1", "stage2", "stage3", "stage4", "head", "bn_affine")


# ---------------------------------------------------------------------------
# loading


def load_arms(run_dir):
    """The pilot's arms, sorted by lambda, with the baseline held out separately."""
    runs = find_runs(run_dir)
    if not runs:
        raise SystemExit(
            f"!! no results.json under {run_dir}. Run `make pull-cpu` first, or "
            f"pass the run directory as an argument.")
    arms, baseline = [], None
    for run in runs:
        cfg = run["config"]
        probes = [r for r in run.get("probes") or []
                  if isinstance(r.get("d_ntk"), (int, float))]
        entry = {
            "name": run["name"],
            "cfg": cfg,
            "summary": run["runs"][0],
            "probes": probes,
            "lam": float(cfg.get("anchor_lambda", 0.0)),
            "anchored": cfg.get("anchor_mode", "off") != "off"
                        and not cfg.get("weight_decay", 0.0),
        }
        (arms if entry["anchored"] else [])
        if entry["anchored"]:
            arms.append(entry)
        elif baseline is None:
            baseline = entry
    arms.sort(key=lambda a: a["lam"])
    if len(arms) < 3:
        raise SystemExit(f"!! only {len(arms)} anchored arms in {run_dir}: the "
                         f"figures need the whole ladder")
    return arms, baseline


def at(arm, fraction, key):
    """The probe nearest `fraction` of the way through, by step.

    Nearest rather than interpolated: every value in these figures is something
    that was actually measured, so a reader can find it in probes.csv.
    """
    target = fraction * max(r["step"] for r in arm["probes"])
    row = min(arm["probes"], key=lambda r: abs(r["step"] - target))
    return row.get(key)


def final(arm, key):
    return arm["probes"][-1].get(key)


def label(arm):
    return "lambda = 0" if arm["lam"] == 0 else f"lambda = {arm['lam']:g}"


# ---------------------------------------------------------------------------
# figures


def fig_pull_authority(arms, out):
    """The anchor's grip on the weights, and how unevenly it is distributed.

    This is the figure the rest of the document leans on. The normalised penalty
    divides by ||w0_g||^2, which makes lambda_g dimensionless -- but the quantity
    that decides how fast a group actually relaxes toward w0 is the per-step pull
    fraction eta*lambda_g/||w0_g||^2, and dividing by a norm that spans two
    decades makes THAT span two decades too. Un-normalised it would be eta*lambda
    for every group alike.
    """
    import matplotlib.pyplot as plt

    reference = arms[len(arms) // 2]
    lam, lr = reference["lam"] or 1.0, float(reference["cfg"]["lr"])
    w0_sq = reference["summary"]["anchor_w0_sq"]
    factors = [lr * lam / w0_sq[g] for g in GROUPS]
    spread = max(w0_sq[g] for g in GROUPS) / min(w0_sq[g] for g in GROUPS)

    fig, (ax_factor, ax_dist) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    fig.subplots_adjust(wspace=0.30, top=0.84, bottom=0.22)

    positions = range(len(GROUPS))
    ax_factor.bar(positions, factors,
                  color=[CYCLE[i % len(CYCLE)] for i in positions])
    ax_factor.set_yscale("log")
    ax_factor.set_xticks(list(positions))
    ax_factor.set_xticklabels(GROUPS, rotation=35, ha="right")
    ax_factor.set_ylabel(f"per-step pull fraction at lambda = {lam:g}")
    ax_factor.set_title("(a) the same lambda is a different anchor in every group")
    ax_factor.axhline(lr * lam, color=MUTED, lw=1.0, ls="--")
    ax_factor.text(-0.4, lr * lam * 1.25,
                   "un-normalised, this would be eta*lambda for every group alike",
                   va="bottom", ha="left", fontsize=7.5, color=MUTED)
    ax_factor.annotate(f"{spread:.0f}x spread\nin ||w0_g||^2",
                       xy=(0.03, 0.06), xycoords="axes fraction", fontsize=8.5,
                       color="#0b0b0b")

    for index, group in enumerate(GROUPS):
        xs = [a["lam"] for a in arms]
        ys = [final(a, f"dist_rel_{group}") for a in arms]
        # lambda = 0 has no place on a log axis; it is drawn as the leftmost tick
        # and labelled, rather than dropped, because it is the free-drift point.
        ax_dist.plot(range(len(xs)), ys, marker="o", ms=4, lw=1.6,
                     color=CYCLE[index % len(CYCLE)], label=group)
    ax_dist.set_yscale("log")
    ax_dist.set_xticks(range(len(arms)))
    ax_dist.set_xticklabels([("0" if a["lam"] == 0 else f"{a['lam']:g}")
                             for a in arms])
    ax_dist.set_xlabel("lambda")
    ax_dist.set_ylabel("final ||w_g - w0_g|| / ||w0_g||")
    ax_dist.set_title("(b) and moves each group by a different amount")
    ax_dist.legend(ncol=2, fontsize=7.5)

    fig.suptitle("The anchor has authority over the weights -- unequally by depth",
                 x=0.5, y=0.965, fontsize=11)
    return _save(fig, out, "pull_authority.png")


def fig_no_separation(arms, out):
    """Stage 1's go/no-go. d(t) must order monotonically in lambda; it does not."""
    import matplotlib.pyplot as plt

    fig, (ax_time, ax_rho) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    fig.subplots_adjust(wspace=0.26, top=0.84)

    for index, arm in enumerate(arms):
        steps = [r["step"] for r in arm["probes"]]
        values = [r["d_ema"] for r in arm["probes"]]
        ax_time.plot(steps, values, lw=1.7, color=CYCLE[index % len(CYCLE)],
                     label=label(arm))
    ax_time.set_xlabel("step")
    ax_time.set_ylabel("d (EMA) -- the signal the controller would steer by")
    ax_time.set_title("(a) the trajectories cross instead of ordering")
    ax_time.legend(fontsize=8)

    rhos = [spearman([a["lam"] for a in arms],
                     [at(a, f, "d_ema") for a in arms]) for f in FRACTIONS]
    xs = [100 * f for f in FRACTIONS]
    ax_rho.plot(xs, rhos, marker="o", ms=6, lw=1.8, color=TEST)
    ax_rho.axhline(SEPARATION_THRESHOLD, color="#d63a6a", lw=1.2, ls="--")
    ax_rho.axhspan(-1.05, SEPARATION_THRESHOLD, color="#1baf7a", alpha=0.10,
                   linewidth=0)
    ax_rho.axhline(0.0, color=MUTED, lw=0.8)
    ax_rho.text(xs[0], SEPARATION_THRESHOLD - 0.06,
                "  accept below here (rho <= -0.9)", fontsize=8, color="#d63a6a",
                va="top")
    ax_rho.text(xs[0], 0.04, "  positive: MORE anchor, MORE drift", fontsize=8,
                color="#0b0b0b", va="bottom")
    for x, rho in zip(xs, rhos, strict=True):
        ax_rho.annotate(f"{rho:+.2f}", (x, rho), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8)
    ax_rho.set_ylim(-1.05, 1.05)
    ax_rho.set_xlabel("% of training")
    ax_rho.set_ylabel("Spearman rho of lambda against d")
    ax_rho.set_title("(b) and the correlation has the wrong sign throughout")

    fig.suptitle("NOT SEPARATED: lambda does not control the drift signal",
                 x=0.5, y=0.965, fontsize=11)
    return _save(fig, out, "no_separation.png"), rhos


def fig_scale_not_geometry(arms, out):
    """Why d fails: it is reporting the kernel's norm, not its geometry.

    The three logged quantities satisfy d^2 = scale^2 - 2 a scale + 1, so
    d >= scale - 1 with equality at a = 1. Plotting every probe against that
    bound shows how little of d is anything but norm growth.
    """
    import matplotlib.pyplot as plt

    fig, (ax_bound, ax_align) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    fig.subplots_adjust(wspace=0.26, top=0.84)

    for index, arm in enumerate(arms):
        ax_bound.scatter([r["scale_ntk"] for r in arm["probes"]],
                         [r["d_ntk"] for r in arm["probes"]],
                         s=16, alpha=0.75, color=CYCLE[index % len(CYCLE)],
                         label=label(arm), linewidths=0)
    top = max(r["scale_ntk"] for a in arms for r in a["probes"]) * 1.05
    ax_bound.plot([1, top], [0, top - 1], color="#0b0b0b", lw=1.2, ls="--")
    ax_bound.text(top * 0.55, top * 0.42, "d = scale - 1\n(pure norm growth,\n"
                  "zero geometry)", fontsize=8, color="#0b0b0b")
    ax_bound.set_xlabel("scale = ||K_t||_F / ||K_0||_F")
    ax_bound.set_ylabel("d = ||K_t - K_0||_F / ||K_0||_F")
    ax_bound.set_title("(a) every probe sits on the pure-norm-growth bound")
    ax_bound.legend(fontsize=7.5, loc="upper left")

    for index, arm in enumerate(arms):
        ax_align.plot([r["step"] for r in arm["probes"]],
                      [1.0 - r["a_ntk"] for r in arm["probes"]],
                      lw=1.7, color=CYCLE[index % len(CYCLE)], label=label(arm))
    ax_align.set_xlabel("step")
    ax_align.set_ylabel("1 - a: the scale-free geometry term")
    ax_align.set_title("(b) the scale-free signal does not order by lambda either")
    ax_align.legend(fontsize=8)

    fig.suptitle("d is ~99% kernel norm. There is almost no geometry in it to steer",
                 x=0.5, y=0.965, fontsize=11)
    return _save(fig, out, "scale_not_geometry.png")


def fig_reference_confound(arms, out):
    """The reference K_0 is taken on each arm's own trajectory, so it moves with
    lambda -- and the feature Gram makes that visible."""
    import matplotlib.pyplot as plt

    fig, (ax_feat, ax_ntk) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    fig.subplots_adjust(wspace=0.28, top=0.80)

    xs = range(len(arms))
    ticks = [("0" if a["lam"] == 0 else f"{a['lam']:g}") for a in arms]
    for ax, keys, title in (
            (ax_feat, ("d_feature", "scale_feature"),
             "(a) feature Gram: monotone in lambda, and backwards"),
            (ax_ntk, ("d_ntk", "scale_ntk"),
             "(b) NTK: no ordering at all")):
        # scale drawn wide underneath and d dashed on top. On the NTK panel the
        # two coincide to within 0.1 -- which is figure 3's point, not a plotting
        # fault -- and overplotting them identically reads as a missing series.
        for key, colour, width, style in ((keys[1], "#eb6834", 4.5, "-"),
                                          (keys[0], TEST, 1.8, "--")):
            ax.plot(xs, [final(a, key) for a in arms], marker="o", ms=5,
                    lw=width, ls=style, color=colour, alpha=0.9, label=key)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(ticks)
        ax.set_xlabel("lambda")
        ax.set_title(title)
        ax.legend(fontsize=8)
    ax_feat.set_ylabel("value at the final probe")
    ax_ntk.set_ylabel("value at the final probe")
    ax_ntk.annotate("the two coincide:\nd = scale - 1 (fig 3)", xy=(0.03, 0.86),
                    xycoords="axes fraction", fontsize=8, color="#0b0b0b")

    fig.suptitle("Each arm takes K_0 at step 100 of ITS OWN trajectory.\nA strong "
                 "anchor holds that point near w0, where the kernel is degenerate "
                 "-- so the references are not comparable.",
                 x=0.5, y=0.975, fontsize=10)
    return _save(fig, out, "reference_confound.png")


def fig_depth_redistribution(arms, out):
    """Where the drift and the displacement go, group by group, as lambda rises."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    fig.subplots_adjust(wspace=0.34, top=0.80, bottom=0.20)

    for ax, prefix, title, fmt in (
            (axes[0], "d", "(a) per-group drift d_g", "{:.1f}"),
            (axes[1], "dist_rel", "(b) per-group displacement", "{:.3f}")):
        matrix = [[final(a, f"{prefix}_{g}") for g in GROUPS] for a in arms]
        # Row-normalised, so the colour shows WHERE each arm's drift sits rather
        # than how big that arm's total happens to be -- which is the claim.
        shaded = [[v / max(row) for v in row] for row in matrix]
        ax.imshow(shaded, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels(GROUPS, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([("lam 0" if a["lam"] == 0 else f"lam {a['lam']:g}")
                            for a in arms], fontsize=8)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, fmt.format(value), ha="center", va="center",
                        fontsize=7.5,
                        color="#0b0b0b" if shaded[i][j] < 0.6 else "#fcfcfb")
        ax.set_title(title + "   (shading is row-relative)")

    fig.suptitle("The anchor moves drift around the network rather than removing "
                 "it", x=0.5, y=0.955, fontsize=11)
    return _save(fig, out, "depth_redistribution.png")


def fig_accuracy(arms, baseline, out):
    """What it cost. One seed, so the figure says so."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    fig.subplots_adjust(top=0.84, bottom=0.14)

    xs = range(len(arms))
    accs = [a["summary"]["test_acc_final"] for a in arms]
    gaps = [a["summary"]["gen_gap"] for a in arms]

    ax.plot(xs, accs, marker="o", ms=6, lw=1.9, color=TEST, label="test accuracy")
    if baseline is not None:
        ax.axhline(baseline["summary"]["test_acc_final"], color=TRAIN, lw=1.3,
                   ls="--")
        ax.text(len(arms) - 1, baseline["summary"]["test_acc_final"],
                f"  weight decay 5e-4: {baseline['summary']['test_acc_final']:.4f}",
                fontsize=8, color=TRAIN, va="bottom", ha="right")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([("0" if a["lam"] == 0 else f"{a['lam']:g}") for a in arms])
    ax.set_xlabel("lambda")
    ax.set_ylabel("test accuracy")

    twin = ax.twinx()
    twin.plot(xs, gaps, marker="s", ms=5, lw=1.5, color="#eb6834",
              label="generalisation gap")
    twin.set_ylabel("train acc - test acc", color="#eb6834")
    twin.spines["right"].set_visible(True)

    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels, fontsize=8, loc="lower left")

    fig.suptitle("The anchor regularises -- and costs accuracy at this scale\n"
                 "1 seed, resnet18 w=16, 10k images, 18 epochs: read the ranking, "
                 "not the third decimal", x=0.5, y=0.975, fontsize=10)
    return _save(fig, out, "accuracy.png")


def fig_coupling(dumps, out):
    """The two plants the allocation could be judged on, side by side.

    selfcheck's H7 writes one JSON per grouping. The protocol states the coupling
    criterion on the DRIFT plant G = -d(d_l)/d(log lambda_m), but the slow loop
    integrates log tau, so the plant it actually closes around is
    T = -d(log tau_l)/d(log lambda_m). Both are drawn, because the verdict differs
    between them and the difference is the finding.

    Colour is the off-diagonal-to-diagonal ratio per row, which is what diagonal
    dominance is about; the printed number is that ratio, and the sign of the
    diagonal is called out separately because a plant with the wrong sign is a
    different failure from a merely crowded one.
    """
    import matplotlib.pyplot as plt

    if not dumps:
        return None
    fig, axes = plt.subplots(len(dumps), 2,
                             figsize=(11.0, 3.1 * len(dumps) + 1.0),
                             squeeze=False)
    fig.subplots_adjust(hspace=0.55, wspace=0.30, top=1 - 0.42 / len(dumps),
                        bottom=0.10)

    for row_index, dump in enumerate(dumps):
        groups = dump["groups"]
        for col_index, (key, name) in enumerate(
                (("G", "drift plant G  (the protocol's criterion)"),
                 ("T_tension", "tension plant T  (what the loop closes around)"))):
            ax = axes[row_index][col_index]
            matrix = dump[key]
            ratios, positive, signs = [], [], 0
            for target in groups:
                diagonal = matrix[target][target]
                off = sum(abs(matrix[m][target]) for m in groups if m != target)
                ratios.append(off / max(abs(diagonal), 1e-300))
                positive.append(diagonal > 0)
                signs += diagonal > 0
            # Dominance alone is not "this row is fine". A row whose diagonal has
            # the WRONG sign -- tightening its own anchor pushes its own signal the
            # wrong way -- is a worse failure than a merely crowded one, and
            # colouring it green because its ratio is small would say the
            # opposite. Wrong-sign bars are hatched and outlined whatever their
            # ratio.
            bars = ax.bar(range(len(groups)), ratios,
                          color=["#1baf7a" if (r < 1.0 and ok) else "#d63a6a"
                                 for r, ok in zip(ratios, positive, strict=True)])
            for bar, ok in zip(bars, positive, strict=True):
                if not ok:
                    bar.set_hatch("///")
                    bar.set_edgecolor("#0b0b0b")
                    bar.set_linewidth(0.8)
            ax.axhline(1.0, color="#0b0b0b", lw=1.1, ls="--")
            ax.set_yscale("log")
            # Headroom for the tallest bar's two-line annotation, which otherwise
            # collides with the panel title.
            ax.set_ylim(top=max(ratios) * 3.0)
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(groups, rotation=35, ha="right", fontsize=7.5)
            ax.set_ylabel("off / diag")
            ax.set_title(f"{dump['grouping']}: {name}\n"
                         f"diagonals with the physical sign: {signs}/{len(groups)}"
                         f"   worst row {max(ratios):.2f}", fontsize=8.5)
            for i, (ratio, ok) in enumerate(zip(ratios, positive, strict=True)):
                ax.annotate(f"{ratio:.1f}" + ("" if ok else "\nwrong sign"),
                            (i, ratio), ha="center", va="bottom", fontsize=7,
                            textcoords="offset points", xytext=(0, 2))

    fig.suptitle("Coupling: is a diagonal allocation justified?\n"
                 "green = dominant AND the diagonal has the physical sign; "
                 "hatched = wrong sign, which no ratio redeems", x=0.5,
                 y=1 - 0.10 / len(dumps), fontsize=10)
    return _save(fig, out, "coupling.png")


# The allocation's own constants. Duplicated here rather than imported from
# CONFIG because this file must run against a dump taken with any settings, and
# reading today's config would silently re-label yesterday's measurement.
ALLOC_GAMMA = 0.01                  # anchor_alloc_shrink
ALLOC_CLIP = math.log(10.0)         # anchor_alloc_clip_decades * log 10
SHIPPED_BETA_A = 0.2                # anchor_alloc_beta_frac, times a beta of 1


def tension_matrix(dump):
    """T[l][m] = -d(log tau_l)/d(log lambda_m), as a nested list.

    The dump is keyed {perturbed column m: {observed row l: value}}, which is the
    transpose of how the matrix is written down. Getting that backwards analyses
    the wrong operator and nothing complains.
    """
    groups = dump["groups"]
    return [[dump["T_tension"][m][l] for m in groups] for l in groups]


def _projector(size):
    """P = I - (1/L) 11^T, the zero-sum subspace the allocation lives in."""
    import numpy as np
    return np.eye(size) - np.ones((size, size)) / size


def spectral_radius(dump, beta_a, sign=-1.0, gamma=ALLOC_GAMMA):
    """rho of the slow loop's iteration matrix, restricted to sum(a) = 0.

    Linearising controller.Allocator._update around the measured plant, using
    log tau(a) = log tau_0 - T a:

        a <- a + sign * beta_a * P(log tau) - gamma a
           = [(1 - gamma) I - sign * beta_a * P T] a + const

    sign = -1 is the shipped rule. The loop converges iff rho < 1.
    """
    import numpy as np
    matrix = np.array(tension_matrix(dump))
    size = matrix.shape[0]
    proj = _projector(size)
    iteration = (1.0 - gamma) * np.eye(size) - sign * beta_a * proj @ matrix
    basis = np.linalg.qr(proj)[0][:, :size - 1]
    return float(max(abs(np.linalg.eigvals(basis.T @ iteration @ basis))))


def simulate_alloc(dump, beta_a, sign=-1.0, steps=2000, kick=0.01):
    """controller.Allocator._update, verbatim, against the measured plant.

    Returns |a|_max after each update. The clip is applied exactly where the real
    loop applies it, so a saturating run reads as saturated rather than as a
    number that grew without bound.
    """
    import numpy as np
    matrix = np.array(tension_matrix(dump))
    groups = dump["groups"]
    log_tau0 = np.array([math.log(max(dump["baseline_tau"][g], 1e-300))
                         for g in groups])
    alloc = np.zeros(len(groups))
    alloc[0] = kick
    alloc -= alloc.mean()
    trace = []
    for _ in range(steps):
        deviation = log_tau0 - matrix @ alloc
        deviation -= deviation.mean()
        alloc = alloc + sign * beta_a * deviation - ALLOC_GAMMA * alloc
        alloc -= alloc.mean()
        alloc = np.clip(alloc, -ALLOC_CLIP, ALLOC_CLIP)
        trace.append(float(np.abs(alloc).max()))
    return trace


def fig_stability(dumps, out):
    """Does the allocation converge at all, at any gain, with the shipped sign?

    The coupling gate asks whether the plant is diagonal. This asks the question
    that actually decides whether the loop can be run: given that plant, does the
    update rule converge? It is linear algebra on data already collected -- no
    training -- and it is why cutting beta_a is not the remedy the protocol takes
    it for.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not dumps:
        return None
    fig, (ax_rho, ax_sim) = plt.subplots(1, 2, figsize=(12.0, 4.7))
    fig.subplots_adjust(wspace=0.26, top=0.79, bottom=0.14)

    betas = np.logspace(-4, 0.5, 90)
    for index, dump in enumerate(dumps):
        ax_rho.plot(betas, [spectral_radius(dump, b) for b in betas], lw=1.8,
                    color=CYCLE[index % len(CYCLE)],
                    label=f"{dump['grouping']} ({len(dump['groups'])} groups)")
    ax_rho.axhline(1.0, color="#d63a6a", lw=1.3, ls="--")
    for beta, tag in ((SHIPPED_BETA_A, "shipped\n0.2"),
                      (SHIPPED_BETA_A / 5, "cut 5x\n0.04")):
        ax_rho.axvline(beta, color=MUTED, lw=1.0, ls=":")
        ax_rho.annotate(tag, (beta, 0.9875), fontsize=7.5, ha="center",
                        color=MUTED)
    ax_rho.text(1.2e-4, 1.003, "unstable above here", fontsize=8, color="#d63a6a")
    ax_rho.set_xscale("log")
    ax_rho.set_ylim(0.985, 1.06)
    ax_rho.set_xlabel("beta_a")
    ax_rho.set_ylabel("spectral radius of the slow loop")
    ax_rho.set_title("(a) the shipped sign is unstable at every usable gain")
    ax_rho.legend(fontsize=8, loc="upper left")

    reference = dumps[0]
    for index, beta in enumerate((SHIPPED_BETA_A, SHIPPED_BETA_A / 5,
                                  SHIPPED_BETA_A / 25)):
        ax_sim.plot(simulate_alloc(reference, beta), lw=1.7,
                    color=CYCLE[index % len(CYCLE)],
                    label=f"shipped sign, beta_a = {beta:g}")
    ax_sim.plot(simulate_alloc(reference, SHIPPED_BETA_A, sign=+1.0), lw=1.8,
                ls="--", color="#0b0b0b",
                label=f"sign FLIPPED, beta_a = {SHIPPED_BETA_A:g}")
    ax_sim.axhline(ALLOC_CLIP, color="#d63a6a", lw=1.2, ls="--")
    ax_sim.text(300, ALLOC_CLIP * 1.03, "the |a| <= log 10 clip", fontsize=8,
                color="#d63a6a", ha="right")
    ax_sim.set_xscale("log")
    ax_sim.set_xlabel("slow-loop updates")
    ax_sim.set_ylabel("|a|max")
    ax_sim.set_title(f"(b) {reference['grouping']}: a 0.01 kick, and what becomes "
                     f"of it")
    ax_sim.legend(fontsize=8, loc="upper left")

    fig.suptitle("Closed-loop stability of the allocation, from the measured "
                 "tension plant.\nCutting beta_a delays saturation; it cannot "
                 "prevent it, because the feedback sign is positive.",
                 x=0.5, y=0.965, fontsize=10)
    return _save(fig, out, "stability.png")


def print_stability(dumps):
    """The gain at which the allocation stops converging, and with which sign."""
    import numpy as np

    if not dumps:
        return
    print("\n" + "=" * 78)
    print("STABILITY  --  the slow loop against the measured tension plant")
    print("=" * 78)
    print(f"\niteration matrix (1-gamma) I - sign*beta_a*P T on sum(a)=0; "
          f"gamma = {ALLOC_GAMMA}, clip = +-{ALLOC_CLIP:.4f}")
    for dump in dumps:
        matrix = np.array(tension_matrix(dump))
        size = matrix.shape[0]
        proj = _projector(size)
        basis = np.linalg.qr(proj)[0][:, :size - 1]
        eig = sorted(np.linalg.eigvals(basis.T @ (proj @ matrix) @ basis).real)
        biggest = max(eig)
        print(f"\ngrouping {dump['grouping']!r}:")
        print("  eigenvalues of P T on the zero-sum subspace: "
              + ", ".join(f"{v:+.3f}" for v in eig))
        print(f"  all strictly positive: {all(v > 0 for v in eig)}"
              f"  -> the shipped sign is POSITIVE feedback")
        print(f"  shipped sign converges only for beta_a < gamma/lam_max = "
              f"{ALLOC_GAMMA / biggest:.5f}")
        print(f"  flipped sign converges for beta_a < (2-gamma)/lam_max = "
              f"{(2 - ALLOC_GAMMA) / biggest:.3f}")
        row = "  rho: "
        for beta in (SHIPPED_BETA_A, SHIPPED_BETA_A / 5, SHIPPED_BETA_A / 25):
            row += f"beta_a={beta:<7.4g}{spectral_radius(dump, beta):.4f}   "
        print(row)

    reference = dumps[0]
    print(f"\nsimulated on {reference['grouping']!r}, from a 0.01 kick, "
          f"2000 updates:")
    for sign, name in ((-1.0, "shipped"), (+1.0, "flipped")):
        for beta in (SHIPPED_BETA_A, SHIPPED_BETA_A / 5, SHIPPED_BETA_A / 25):
            trace = simulate_alloc(reference, beta, sign=sign)
            if trace[-1] >= ALLOC_CLIP - 1e-6:
                first = next(i for i, v in enumerate(trace)
                             if v >= ALLOC_CLIP - 1e-6)
                verdict = f"PINNED at the clip after {first + 1} updates"
            else:
                verdict = f"converged, |a|max = {trace[-1]:.3f}"
            print(f"  {name:<8} beta_a={beta:<7.4g} {verdict}")
    print("=" * 78)


def load_coupling(groupings=("stage", "coarse", "trunk")):
    """H7's dumps, for whichever groupings have been measured."""
    import tempfile
    found = []
    for grouping in groupings:
        path = os.path.join(tempfile.gettempdir(), f"coupling_G_{grouping}.json")
        if os.path.isfile(path):
            with open(path) as fh:
                found.append(json.load(fh))
    return found


def _save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------


def kernel_wall_seconds(run_dir):
    """The Kaggle kernel's total wall clock, from the log it ships beside runs/.

    Not the same as the sum of the arms' training time: the kernel also downloads
    CIFAR-10, evaluates, and syncs W&B. Read from the log rather than remembered,
    so the document cannot quote a number nobody can check.
    """
    root = run_dir
    for _ in range(3):
        root = os.path.dirname(os.path.normpath(root))
        for name in os.listdir(root) if os.path.isdir(root) else []:
            if name.endswith(".log"):
                last = 0.0
                with open(os.path.join(root, name)) as fh:
                    for line in fh:
                        marker = '"time":'
                        if marker in line:
                            for piece in line.split(marker)[1:]:
                                try:
                                    last = max(last, float(piece.split(",")[0]))
                                except ValueError:
                                    pass
                return last
    return float("nan")


def print_numbers(arms, baseline, rhos):
    """Every scalar RESULTS.md quotes. If it is not here, it is not checkable."""
    reference = arms[len(arms) // 2]
    w0_sq = reference["summary"]["anchor_w0_sq"]
    lr = float(reference["cfg"]["lr"])
    cfg = reference["cfg"]
    steps = (cfg["train_subset"] // cfg["batch_size"]) * cfg["epochs"]

    print("\n" + "=" * 78)
    print("NUMBERS  --  every scalar RESULTS.md is allowed to quote")
    print("=" * 78)

    print(f"\nsetup: resnet18 w={cfg['width']}, "
          f"{reference['summary']['params']:,} params, {cfg['train_subset']} "
          f"images, {cfg['epochs']} epochs, batch {cfg['batch_size']} -> {steps} "
          f"steps, lr {lr}, {len(arms)}+1 arms, {len(cfg['seeds'].split(','))} seed")
    wall = sum(a["summary"]["wall_s"] for a in arms)
    wall += baseline["summary"]["wall_s"] if baseline else 0.0
    kernel = kernel_wall_seconds(args_run_dir[0])
    print(f"wall clock: {wall:.0f} s of training over all arms "
          f"({wall / 60:.0f} min); {kernel:.0f} s of kernel wall clock "
          f"({kernel / 60:.0f} min) including the download, evals and W&B sync")
    lo = min(w0_sq.values()) / (lr * steps)
    hi = 0.5 * min(w0_sq.values()) / lr
    print(f"lambda bracket at this scale: [{lo:.4g}, {hi:.4g}]")

    print("\nper-group anchor norms and pull fraction "
          f"(at lambda = {reference['lam']:g}):")
    for group in GROUPS:
        print(f"  {group:<10} ||w0_g||^2 = {w0_sq[group]:>9.3f}   "
              f"eta*lambda/||w0_g||^2 = {lr * reference['lam'] / w0_sq[group]:.3e}")
    print(f"  spread in ||w0_g||^2, and therefore in the pull fraction: "
          f"{max(w0_sq.values()) / min(w0_sq.values()):.0f}x")

    # The sketch Gram is not additive over the partition -- MM^T carries cross
    # terms the blocks' Grams do not -- so this need not come to 1, in either
    # direction. Printed because RESULTS.md quotes it as evidence for that.
    sums = []
    for arm in arms:
        row = arm["probes"][-1]
        sums.append(sum(v for k, v in row.items() if k.startswith("share_")))
    print(f"\nsum over groups of share_g = ||K_0^(g)||_F / ||K_0||_F, final probe: "
          f"{min(sums):.4f} .. {max(sums):.4f} across the ladder "
          f"(would be exactly 1 if the sketch Gram were additive; it is not)")

    print("\nseparation (Spearman rho of lambda against d_ema; accept <= -0.9):")
    for fraction, rho in zip(FRACTIONS, rhos, strict=True):
        print(f"  at {100 * fraction:>5.0f}% of training   rho = {rho:+.4f}")
    finals = [final(a, "d_ema") for a in arms]
    print(f"  d_T spread across the ladder: "
          f"{max(finals) / max(min(finals), 1e-300):.2f}x")

    print("\nper arm, at the final probe:")
    header = (f"  {'arm':<12}{'lambda':>7}{'test':>8}{'train':>8}{'gap':>8}"
              f"{'d_ntk':>8}{'scale':>8}{'a_ntk':>8}{'d_feat':>8}{'sc_feat':>9}"
              f"{'dist_rel':>10}")
    print(header)
    rows = ([baseline] if baseline else []) + arms
    for arm in rows:
        summary = arm["summary"]
        lam = "--" if not arm["anchored"] else f"{arm['lam']:g}"
        print(f"  {arm['name']:<12}{lam:>7}"
              f"{summary['test_acc_final']:>8.4f}{summary['train_acc']:>8.4f}"
              f"{summary['gen_gap']:>+8.4f}{final(arm, 'd_ntk'):>8.2f}"
              f"{final(arm, 'scale_ntk'):>8.2f}{final(arm, 'a_ntk'):>8.3f}"
              f"{final(arm, 'd_feature'):>8.2f}{final(arm, 'scale_feature'):>9.2f}"
              f"{final(arm, 'dist_rel'):>10.4f}")

    for prefix, title, fmt in (("dist_rel", "per-group displacement", "{:>10.4f}"),
                               ("d", "per-group drift d_g", "{:>10.3f}")):
        print(f"\n{title} at the final probe:")
        print(f"  {'arm':<12}" + "".join(f"{g[:9]:>10}" for g in GROUPS))
        for arm in arms:
            print(f"  {arm['name']:<12}"
                  + "".join(fmt.format(final(arm, f"{prefix}_{g}"))
                            for g in GROUPS))
    print("=" * 78)


def print_coupling(dumps):
    """H7's numbers, so RESULTS.md's coupling section is checkable too."""
    if not dumps:
        print("\n(no coupling dumps found -- run selfcheck's H7 to produce them)")
        return
    print("\n" + "=" * 78)
    print("COUPLING  --  selfcheck H7, one row per grouping")
    print("=" * 78)
    for dump in dumps:
        groups = dump["groups"]
        print(f"\ngrouping {dump['grouping']!r}: {len(groups)} groups, "
              f"{dump['steps']} steps, +-{dump['delta']} in log lambda, "
              f"base pull fraction {dump['base_fraction']:g}")
        for key, name in (("G", "drift  G"), ("PGP", "proj   PGP"),
                          ("T_tension", "tension T")):
            matrix = dump[key]
            ratios, signs = [], 0
            for target in groups:
                diagonal = matrix[target][target]
                off = sum(abs(matrix[m][target]) for m in groups if m != target)
                ratios.append(off / max(abs(diagonal), 1e-300))
                signs += diagonal > 0
            dominant = sum(1 for r in ratios if r < 1.0)
            print(f"  {name:<11} off/diag {min(ratios):>7.2f} .. {max(ratios):>7.2f}"
                  f"   rows dominant {dominant}/{len(groups)}"
                  f"   correct-sign diagonals {signs}/{len(groups)}")
        print("  baseline tau_l: "
              + "  ".join(f"{g}={dump['baseline_tau'][g]:.3g}" for g in groups))
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate every figure and number in RESULTS.md.")
    parser.add_argument("run_dir", nargs="?", default=DEFAULT_RUN,
                        help=f"the pilot run directory (default: {DEFAULT_RUN})")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"where the pngs go (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    args_run_dir[0] = args.run_dir
    arms, baseline = load_arms(args.run_dir)

    import matplotlib
    matplotlib.use("Agg")
    _theme(matplotlib)
    os.makedirs(args.out, exist_ok=True)

    made = [fig_pull_authority(arms, args.out)]
    separation, rhos = fig_no_separation(arms, args.out)
    made.append(separation)
    made += [fig_scale_not_geometry(arms, args.out),
             fig_reference_confound(arms, args.out),
             fig_depth_redistribution(arms, args.out),
             fig_accuracy(arms, baseline, args.out)]
    dumps = load_coupling()
    for figure in (fig_coupling(dumps, args.out), fig_stability(dumps, args.out)):
        if figure:
            made.append(figure)
    print_coupling(dumps)
    print_stability(dumps)

    print_numbers(arms, baseline, rhos)
    print()
    for path in made:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
