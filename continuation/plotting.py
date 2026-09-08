"""Plots for Experiment 0.

Quantities are kept separate on purpose: transformed-task optimization progress,
transformed-task generalization, and transfer to the target task are different
things and are never merged into a single "ease of optimization" score.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CURVES = [
    ("transformed_train_probe", "ce",
     "Transformed training CE (eval mode, fixed probe subset)", "cross-entropy"),
    ("target_train_probe", "ce",
     "Target training CE on original images (same weights)", "cross-entropy"),
    ("transformed_val", "ce", "Transformed validation CE", "cross-entropy"),
    ("transformed_val", "accuracy", "Transformed validation accuracy", "accuracy"),
    ("target_val", "ce", "Target validation CE on original images", "cross-entropy"),
    ("target_val", "accuracy", "Target validation accuracy on original images", "accuracy"),
]


def _eval_series(runs):
    """(level -> steps, {block.key: [n_seeds, n_steps] array})."""
    per_level = defaultdict(list)
    for r in runs:
        recs = [m for m in r["metrics"] if m.get("record") == "eval"]
        recs.sort(key=lambda m: m["step"])
        if recs:
            per_level[r["level"]].append(recs)
    out = {}
    for level, seed_recs in per_level.items():
        n = min(len(s) for s in seed_recs)
        steps = np.array([m["step"] for m in seed_recs[0][:n]])
        series = {}
        for block, key, _, _ in CURVES:
            series["%s.%s" % (block, key)] = np.array(
                [[m[block][key] for m in s[:n]] for s in seed_recs])
        out[level] = (steps, series)
    return out


def _minibatch_series(runs):
    per_level = defaultdict(list)
    for r in runs:
        recs = [m for m in r["metrics"] if m.get("record") == "train_minibatch"]
        recs.sort(key=lambda m: m["step"])
        if recs:
            per_level[r["level"]].append(recs)
    out = {}
    for level, seed_recs in per_level.items():
        n = min(len(s) for s in seed_recs)
        steps = np.array([m["step"] for m in seed_recs[0][:n]])
        vals = np.array([[m["minibatch_transformed_ce_mean"] for m in s[:n]] for s in seed_recs])
        out[level] = (steps, vals)
    return out


def _band(ax, x, y, label, color):
    mean = y.mean(axis=0)
    ax.plot(x, mean, lw=1.6, label=label, color=color)
    if y.shape[0] > 1:
        sd = y.std(axis=0, ddof=1)
        ax.fill_between(x, mean - sd, mean + sd, alpha=0.18, color=color, linewidth=0)


def plot_curves(runs, out_dir, param_name: str = "sigma") -> list:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _eval_series(runs)
    levels = sorted(data)
    colors = plt.cm.viridis(np.linspace(0, 0.9, max(len(levels), 1)))
    written = []

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 12.0))
    for ax, (block, key, title, ylabel) in zip(axes.flatten(), CURVES):
        for c, level in enumerate(levels):
            steps, series = data[level]
            _band(ax, steps, series["%s.%s" % (block, key)],
                  "%s=%g" % (param_name, level), colors[c])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("gradient updates")
        ax.set_ylabel(ylabel)
        if key == "ce":
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Experiment 0: fixed Gaussian levels (mean +/- 1 sd over paired seeds)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = out_dir / "exp0_curves.png"
    fig.savefig(p, dpi=150); plt.close(fig); written.append(p)

    mb = _minibatch_series(runs)
    if mb:
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        for c, level in enumerate(sorted(mb)):
            steps, vals = mb[level]
            _band(ax, steps, vals, "%s=%g" % (param_name, level), colors[c])
        ax.set_title("Training-mode minibatch CE (logged during training, NOT an eval metric)",
                     fontsize=10)
        ax.set_xlabel("gradient updates"); ax.set_ylabel("cross-entropy")
        ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        fig.tight_layout()
        p = out_dir / "exp0_minibatch_ce.png"
        fig.savefig(p, dpi=150); plt.close(fig); written.append(p)
    return written


def plot_final_vs_level(agg: dict, out_dir) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = [e["level"] for e in agg["levels"]]
    param = agg["levels"][0]["parameter_name"] if agg["levels"] else "sigma"

    panels = [
        ("transformed_val.accuracy", "Transformed validation accuracy", "accuracy"),
        ("target_val.accuracy", "Target validation accuracy (original images)", "accuracy"),
        ("transformed_train_probe.ce", "Transformed training CE (probe)", "cross-entropy"),
        ("target_train_probe.ce", "Target training CE (original images)", "cross-entropy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    for ax, (key, title, ylabel) in zip(axes.flatten(), panels):
        means = [e["final"][key]["mean"] for e in agg["levels"]]
        errs = [(e["final"][key]["std"] or 0.0) for e in agg["levels"]]
        ax.errorbar(levels, means, yerr=errs, marker="o", capsize=3, lw=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("%s (pixels)" % param); ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    fig.suptitle("Experiment 0 final metrics at a common update budget "
                 "(mean +/- 1 sd over paired seeds)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = out_dir / "exp0_final_vs_level.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def plot_transform_stats(agg: dict, out_dir) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = [e["level"] for e in agg["levels"]]
    mse = [e["transform_stats"]["reconstruction_mse"]["mean"] for e in agg["levels"]]
    tvr = [e["transform_stats"]["retained_tv_ratio"]["mean"] for e in agg["levels"]]
    param = agg["levels"][0]["parameter_name"] if agg["levels"] else "sigma"

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].plot(levels, mse, "o-")
    axes[0].set_xlabel("%s (pixels)" % param)
    axes[0].set_ylabel("mean per-image MSE, [0,1] units")
    axes[0].set_title("Reconstruction MSE on the fixed image subset", fontsize=10)
    axes[1].plot(levels, tvr, "o-")
    axes[1].set_xlabel("%s (pixels)" % param)
    axes[1].set_ylabel("mean TV(T x) / TV(x)")
    axes[1].set_title("Retained TV ratio (isotropic, forward differences)", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Transformation statistics -- calibration only; equal MSE or TV does not "
                 "mean equal label information", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = out_dir / "exp0_transform_stats.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


# --------------------------------------------------------------------------
# Experiment 1: warm starts and an intermediate continuation stage
# --------------------------------------------------------------------------

ARM_STYLE = {
    "A": {"color": "#333333", "label": "A  direct baseline (sigma=0 throughout)"},
    "W": {"color": "#1f77b4", "label": "W  warm start (sigma=1 -> 0)"},
    "P": {"color": "#d62728", "label": "P  continuation (sigma=1 -> 0.5 -> 0)"},
}


def _arm_series(runs, block, key, common_only=None):
    """arm -> (steps, [n_seeds, n_steps]) from checkpoint records only."""
    out = {}
    for arm in ("A", "W", "P"):
        rs = [r for r in runs if r["arm"] == arm]
        if not rs:
            continue
        per_seed = []
        for r in rs:
            by_step = {}
            for m in r["metrics"]:
                if m.get("record") != "eval" or m.get("phase") == "post_transition":
                    continue
                by_step[m["step"]] = m[block][key]
            per_seed.append(by_step)
        steps = sorted(set.intersection(*[set(d) for d in per_seed]))
        if common_only is not None:
            steps = [s for s in steps if s in set(common_only)]
        if not steps:
            continue
        out[arm] = (np.array(steps), np.array([[d[s] for s in steps] for d in per_seed]))
    return out


def _mark_stages(ax, sweep):
    warm = int(sweep["warm_steps"])
    mid = int(sweep["mid_steps"])
    for x, label in ((warm, "switch 1"), (warm + mid, "switch 2")):
        ax.axvline(x, color="gray", ls="--", lw=0.9, alpha=0.8)
        ax.annotate(label, xy=(x, 1.005), xycoords=("data", "axes fraction"),
                    fontsize=7, ha="center", color="gray")


def plot_exp1(runs, agg, out_dir, sweep) -> list:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    common = agg.get("common_checkpoints")

    panels = [
        ("target_train_probe", "ce", "Target training CE (original images, probe subset)", True),
        ("target_val", "ce", "Target validation CE (original images)", True),
        ("target_val", "accuracy", "Target validation accuracy (original images)", False),
        ("target_train_probe", "accuracy", "Target training accuracy (probe subset)", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6))
    for ax, (block, key, title, logy) in zip(axes.flatten(), panels):
        data = _arm_series(runs, block, key, common)
        for arm, (steps, vals) in data.items():
            _band(ax, steps, vals, ARM_STYLE[arm]["label"], ARM_STYLE[arm]["color"])
        _mark_stages(ax, sweep)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("global gradient updates")
        ax.set_ylabel("cross-entropy" if key == "ce" else "accuracy")
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Experiment 1: equal total budget B=%d (mean +/- 1 sd over 3 paired seeds)"
                 % int(sweep["total_steps"]), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = out_dir / "exp1_equal_budget.png"
    fig.savefig(p, dpi=150); plt.close(fig); written.append(p)

    # Adaptation after the switch to original images (dense records included).
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    warm, mid = int(sweep["warm_steps"]), int(sweep["mid_steps"])
    switch = {"A": 0, "W": warm, "P": warm + mid}
    for ax, (block, key, title) in zip(
            axes, [("target_val", "accuracy", "Target validation accuracy"),
                   ("target_train_probe", "ce", "Target training CE (probe)")]):
        for arm in ("A", "W", "P"):
            rs = [r for r in runs if r["arm"] == arm]
            if not rs:
                continue
            per_seed = []
            for r in rs:
                by_step = {}
                for m in r["metrics"]:
                    if m.get("record") != "eval":
                        continue
                    # at a transition keep the post-switch record
                    if m["step"] == switch[arm] and m.get("phase") == "pre_transition":
                        continue
                    if switch[arm] <= m["step"] <= switch[arm] + 2000:
                        by_step[m["step"] - switch[arm]] = m[block][key]
                per_seed.append(by_step)
            steps = sorted(set.intersection(*[set(d) for d in per_seed]))
            if not steps:
                continue
            vals = np.array([[d[s] for s in steps] for d in per_seed])
            _band(ax, np.array(steps), vals, ARM_STYLE[arm]["label"], ARM_STYLE[arm]["color"])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("updates since that arm began training on original images")
        ax.set_ylabel("accuracy" if key == "accuracy" else "cross-entropy")
        if key == "ce":
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Adaptation after the switch to sigma=0 -- NOTE: arms reach update 0 of this "
                 "axis with different pretraining budgets (A:0, W:%d, P:%d)" % (warm, warm + mid),
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = out_dir / "exp1_adaptation.png"
    fig.savefig(p, dpi=150); plt.close(fig); written.append(p)

    # Paired per-seed differences at B.
    pairs = agg.get("paired_differences", {})
    if pairs:
        keys = ["target_val.accuracy", "target_train_probe.accuracy"]
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        names = list(pairs)
        width = 0.35
        for j, k in enumerate(keys):
            xs = np.arange(len(names)) + (j - 0.5) * width
            means = [pairs[n][k]["mean"] for n in names]
            errs = [pairs[n][k]["std"] or 0.0 for n in names]
            ax.bar(xs, means, width, yerr=errs, capsize=4, label=k)
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels([n.replace("_minus_", " - ") for n in names])
        ax.set_ylabel("difference at B (percentage points)")
        ax.set_title("Paired differences at the common budget (mean +/- 1 sd, 3 seeds)",
                     fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = out_dir / "exp1_paired_differences.png"
        fig.savefig(p, dpi=150); plt.close(fig); written.append(p)
    return written
