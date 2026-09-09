"""Campaign figures: one panel per experimental group, never 21 curves on one axis.

Three outputs:

* ``campaign_final.png``   final accuracy per configuration, grouped A-E, with
  the three individual seeds shown as dots beside the mean.
* ``campaign_curves.png``  test-accuracy trajectories, one panel per group.
  Current-path curves (the configuration actually trained under) are solid and
  primary; target-path diagnostics (32x32, all filters and stem reductions
  bypassed) are faint dashed and drawn only where they differ.
* ``campaign_cost.png``    final accuracy against measured wall time.

Curves are read with a fallback so a configuration whose operator is inactive
still draws: for an unfiltered arm the current path is a real curve, not a gap.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "campaign_results.json"
FIGS = ROOT / "results"
GROUPS = {"A": "A - main grid (resolution x Gaussian)",
          "B": "B - identity mixture (Gmix)",
          "C": "C - insertion ablation (early7)",
          "D": "D - reduction operator and location",
          "E": "E - order control (Rreverse)"}
COLORS = {"Gnone": "#7a7a7a", "Gplateau": "#1f77b4", "Ggeo": "#2ca02c",
          "Gmix": "#d62728"}
MARK = {"R32": "o", "Rprog": "^", "Rgentle": "s", "Rreverse": "D"}


def short(cid):
    res, g, red, mask = cid.split("__")
    bits = [res, g]
    if red != "input_bilinear":
        bits.append(red)
    if mask != "all19":
        bits.append(mask)
    return " / ".join(bits)


def load():
    return json.loads(RESULTS.read_text())


def fig_final(res):
    cfgs = res["configurations"]
    fig, axes = plt.subplots(1, 5, figsize=(19, 5.4),
                             gridspec_kw={"width_ratios": [9, 2, 2, 6, 2]})
    for ax, (g, title) in zip(axes, GROUPS.items()):
        items = sorted([c for c in cfgs.values() if c["group"] == g],
                       key=lambda c: -c["acc_mean"])
        ys = range(len(items))
        for i, c in enumerate(items):
            col = COLORS.get(c["gaussian"], "#555555")
            ax.scatter(c["acc_per_seed"], [i] * len(c["acc_per_seed"]),
                       color=col, alpha=0.45, s=22, zorder=2)
            ax.scatter([c["acc_mean"]], [i], color=col, s=95, zorder=3,
                       marker=MARK.get(c["resolution"], "o"),
                       edgecolor="white", linewidth=0.8)
            if c["acc_sd"]:
                ax.plot([c["acc_mean"] - c["acc_sd"], c["acc_mean"] + c["acc_sd"]],
                        [i, i], color=col, lw=1.6, alpha=0.8, zorder=1)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([short(c["id"]) for c in items], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("final test accuracy")
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(0.73, 0.83)
    handles = [Line2D([], [], color=v, marker="o", ls="", label=k)
               for k, v in COLORS.items()]
    handles += [Line2D([], [], color="0.3", marker=v, ls="", label=k)
                for k, v in MARK.items()]
    handles += [Line2D([], [], color="0.3", lw=1.6, label="+/- 1 sample SD (3 seeds)"),
                Line2D([], [], color="0.3", marker="o", ls="", alpha=0.45,
                       label="individual seeds")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("21-configuration CIFAR-10 campaign -- final epoch-30 accuracy, "
                 "3 seeds, all models on the common full-resolution unfiltered path\n"
                 "exploratory benchmark with prior test-set exposure: the leading "
                 "cell is not an independently confirmed estimate", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    fig.savefig(FIGS / "campaign_final.png", dpi=140, bbox_inches="tight")
    print("wrote", FIGS / "campaign_final.png")


def curves_for(path):
    m = json.loads((ROOT / path / "metrics.json").read_text())
    ep = [r["epoch"] for r in m]
    cur = [r.get("test_acc_current") for r in m]
    tgt = [r.get("test_acc_target") for r in m]
    return ep, cur, tgt


def fig_curves(res):
    cfgs = res["configurations"]
    runs = {}
    for job in res["jobs"]:
        study = ROOT / job["study"]
        for d in study.iterdir():
            if d.is_dir() and (d / "metrics.json").is_file():
                runs[d.name] = d.relative_to(ROOT)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.6), sharey=True)
    for ax, (g, title) in zip(axes, GROUPS.items()):
        items = sorted([c for c in cfgs.values() if c["group"] == g],
                       key=lambda c: -c["acc_mean"])
        for c in items:
            cid = "%s__seed0" % c["id"]
            if cid not in runs:
                cid = "%s__seed%d" % (c["id"], c["seeds"][0])
            if cid not in runs:
                continue
            ep, cur, tgt = curves_for(runs[cid])
            col = COLORS.get(c["gaussian"], "#555555")
            ax.plot(ep, cur, "-", color=col, lw=1.8,
                    marker=MARK.get(c["resolution"], "o"), ms=3.5, markevery=3,
                    label=short(c["id"]))
            xs = [e for e, a, b in zip(ep, cur, tgt)
                  if a is not None and b is not None and abs(a - b) > 1e-12]
            ys = [b for a, b in zip(cur, tgt)
                  if a is not None and b is not None and abs(a - b) > 1e-12]
            if xs:
                ax.plot(xs, ys, "--", color=col, lw=0.7, alpha=0.4)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("completed epoch")
        ax.grid(alpha=0.25)
        ax.axvline(21, color="0.6", lw=0.8, ls=":")
        ax.set_ylim(0.05, 0.85)
        ax.legend(fontsize=6.5, loc="lower right", frameon=False)
    axes[0].set_ylabel("test accuracy")
    handles = [Line2D([], [], color="0.3", lw=1.8,
                      label="solid = current path (configuration trained under)"),
               Line2D([], [], color="0.3", lw=0.7, ls="--", alpha=0.6,
                      label="dashed = target path (32x32, all filters/reductions "
                            "bypassed), only where it differs")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Test-accuracy trajectories, seed 0 shown per configuration "
                 "(one panel per group)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(FIGS / "campaign_curves.png", dpi=140, bbox_inches="tight")
    print("wrote", FIGS / "campaign_curves.png")


def fig_cost(res):
    cfgs = res["configurations"]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for c in cfgs.values():
        col = COLORS.get(c["gaussian"], "#555555")
        ax.scatter(c["wall_mean"], c["acc_mean"], color=col, s=90,
                   marker=MARK.get(c["resolution"], "o"), edgecolor="white",
                   linewidth=0.8, zorder=3)
        ax.annotate(short(c["id"]), (c["wall_mean"], c["acc_mean"]),
                    fontsize=6.2, xytext=(5, 3), textcoords="offset points",
                    color="0.35")
    ax.set_xlabel("measured mean wall time per run (s), including evaluation")
    ax.set_ylabel("final test accuracy (3-seed mean)")
    ax.grid(alpha=0.3)
    handles = [Line2D([], [], color=v, marker="o", ls="", label=k)
               for k, v in COLORS.items()]
    handles += [Line2D([], [], color="0.3", marker=v, ls="", label=k)
                for k, v in MARK.items()]
    ax.legend(handles=handles, fontsize=8, ncol=2, loc="lower right", frameon=False)
    ax.set_title("Accuracy versus measured cost\n"
                 "all timings newly measured in this campaign on Tesla T4s "
                 "(no cross-job reuse)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIGS / "campaign_cost.png", dpi=140, bbox_inches="tight")
    print("wrote", FIGS / "campaign_cost.png")


if __name__ == "__main__":
    r = load()
    fig_final(r)
    fig_curves(r)
    fig_cost(r)
