"""Two overview figures, in English, covering every method we have trained.

Figure A -- final accuracy of all 30-epoch configurations, one row each, colour
by method family and marker by batch.  The six 120-epoch runs are drawn in their
own short panel because four times the budget is not a method effect.

Figure B -- test accuracy against epoch for the same set: every arm is drawn,
the leaders are highlighted and labelled, and a zoom panel covers the final third
so the ordering is readable.

Honest about what is being overlaid: ``campaign``, ``resbench`` and ``adaptive``
share pinned initial weights and are mutually paired; ``ablation`` used a
different pinned set, and its shared plain baseline lands 0.56 pp lower, so its
arms carry a batch offset of about that size.  Marker shape encodes the batch so
the reader can see which is which.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from scripts.gather_all_methods import FAMILY, main as gather

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "presentation"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "legend.fontsize": 9, "figure.facecolor": "white"})

FAM_COLOR = {
    "plain": "#000000", "gauss": "#1f77b4", "res": "#2ca02c",
    "res_gauss": "#d62728", "profile": "#17becf", "operator": "#ff7f0e",
    "adaptive": "#9467bd", "constant": "#8c564b", "mix": "#e377c2",
    "long": "#7f7f7f",
}
BATCH_MARKER = {"campaign": "o", "resbench": "^", "ablation": "s",
                "adaptive": "D", "long120": "*"}
BATCH_NOTE = ("Marker = batch.  campaign / resbench / adaptive share pinned "
              "initial weights and are mutually paired; the ablation batch used a "
              "different pinned set (its plain baseline is 0.56 pp lower), so its "
              "arms carry a batch offset of roughly that size.")
SD_NOTE = ("Bars / bands = +/- 1 sample SD over the seeds available "
           "(descriptive, not confidence intervals); arms with one seed have none.")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / ("%s.%s" % (name, ext)), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / (name + ".png"))


def curve(rec, field="test_acc_current", fallback="test_acc_target"):
    """Seed-mean curve on the epochs every available seed reports."""
    per = rec["curves"]
    if not per:
        return None
    tabs = []
    for m in per.values():
        t = {}
        for r in m:
            v = r.get(field)
            if v is None:
                v = r.get(fallback)
            if v is not None:
                t[r["epoch"]] = v
        tabs.append(t)
    ep = sorted(set.intersection(*[set(t) for t in tabs]))
    if not ep:
        return None
    return ep, [100 * st.mean([t[e] for t in tabs]) for e in ep]


# --------------------------------------------------------------------------

def fig_final(rows):
    short = {k: v for k, v in rows.items() if v["batch"] != "long120"}
    long_ = {k: v for k, v in rows.items() if v["batch"] == "long120"}
    order = sorted(short, key=lambda k: short[k]["acc_mean"])
    n = len(order)
    fig, axes = plt.subplots(
        2, 1, figsize=(13.5, 0.235 * n + 4.2),
        gridspec_kw={"height_ratios": [n, max(len(long_), 3)], "hspace": 0.09})
    ax, axl = axes

    base = next((v["acc_mean"] for v in short.values()
                 if v["label"] == "Plain baseline"), None)
    if base:
        ax.axvline(base * 100, color="0.2", lw=1.5, ls="--", zorder=1)
        ax.text(base * 100, n * 0.45, "  plain baseline %.2f %%" % (base * 100),
                fontsize=9, color="0.2", va="bottom", rotation=90,
                ha="right", backgroundcolor="white")
    for i, k in enumerate(order):
        v = short[k]
        col = FAM_COLOR[v["family"]]
        if v["n_seeds"] > 1:
            s = v["acc_sd"] * 100
            ax.plot([v["acc_mean"] * 100 - s, v["acc_mean"] * 100 + s], [i, i],
                    color=col, lw=1.5, alpha=0.8, zorder=2)
            for a in v["acc_per_seed"]:
                ax.scatter([a * 100], [i], color=col, alpha=0.35, s=16, zorder=2)
        ax.scatter([v["acc_mean"] * 100], [i], color=col, s=72,
                   marker=BATCH_MARKER[v["batch"]], edgecolor="white",
                   linewidth=0.7, zorder=3)
        if v["n_seeds"] == 1:
            ax.text(v["acc_mean"] * 100 + 0.09, i, "1 seed", fontsize=6,
                    color="0.6", va="center")
    ax.set_yticks(range(n))
    ax.set_yticklabels([short[k]["label"] for k in order], fontsize=7.4)
    ax.set_ylim(-1, n)
    ax.set_xlabel("final test accuracy (%) — epoch 30")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Every method we have trained — final accuracy, 30-epoch budget\n"
                 "%d configurations across four batches, CIFAR-10, ResNet-20 + BN"
                 % n, fontsize=12)

    lo = sorted(long_, key=lambda k: long_[k]["acc_mean"])
    for i, k in enumerate(lo):
        v = long_[k]
        axl.scatter([v["acc_mean"] * 100], [i], color=FAM_COLOR["long"], s=85,
                    marker="*", edgecolor="white", linewidth=0.7, zorder=3)
    axl.set_yticks(range(len(lo)))
    axl.set_yticklabels([long_[k]["label"] for k in lo], fontsize=7.4)
    axl.set_ylim(-0.8, max(len(lo), 3) - 0.2)
    axl.set_xlabel("final test accuracy (%) — 120 epochs")
    axl.grid(axis="x", alpha=0.3)
    axl.set_title("Separate budget: 120 epochs — four times the training, "
                  "not a method effect. Do not read against the panel above.",
                  fontsize=10, color="0.3")

    handles = [Line2D([], [], color=c, marker="o", ls="", label=FAMILY[f])
               for f, c in FAM_COLOR.items() if f != "long"]
    handles += [Line2D([], [], color="0.35", marker=m, ls="", label=b)
                for b, m in BATCH_MARKER.items()]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8,
              ncol=2)
    fig.text(0.5, 0.012, BATCH_NOTE + "  " + SD_NOTE, ha="center", fontsize=8,
             color="0.35", wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "24_all_methods_final_accuracy")


def fig_curves(rows, n_top=8):
    short = {k: v for k, v in rows.items() if v["batch"] != "long120"}
    long_ = {k: v for k, v in rows.items() if v["batch"] == "long120"}
    top = sorted(short, key=lambda k: -short[k]["acc_mean"])[:n_top]
    plain = [k for k, v in short.items() if v["family"] == "plain"]

    fig = plt.figure(figsize=(16.5, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.30, wspace=0.16)
    ax = fig.add_subplot(gs[0, :])
    az = fig.add_subplot(gs[1, 0])
    al = fig.add_subplot(gs[1, 1])

    for k, v in short.items():
        c = curve(v)
        if not c or k in top or k in plain:
            continue
        ax.plot(c[0], c[1], "-", color="0.78", lw=0.7, alpha=0.75, zorder=1)
        az.plot(c[0], c[1], "-", color="0.80", lw=0.7, alpha=0.8, zorder=1)
    for k in plain:
        c = curve(short[k])
        if c:
            for a in (ax, az):
                a.plot(c[0], c[1], "-", color="#000000", lw=2.4,
                       label=short[k]["label"], zorder=4)
    cyc = ["#d62728", "#17becf", "#2ca02c", "#ff7f0e", "#9467bd", "#1f77b4",
           "#8c564b", "#e377c2"]
    for i, k in enumerate(top):
        c = curve(short[k])
        if not c:
            continue
        v = short[k]
        lab = "%s  (%.2f %%)" % (v["label"], v["acc_mean"] * 100)
        for a in (ax, az):
            a.plot(c[0], c[1], "-", color=cyc[i % len(cyc)], lw=2.0, label=lab,
                   zorder=3)

    for a in (ax, az):
        for x in (6, 12):
            a.axvline(x, color="0.88", lw=0.8, ls="--", zorder=0)
        a.axvline(21, color="0.7", lw=0.8, ls=":", zorder=0)
        a.set_xlabel("completed epoch")
        a.set_ylabel("test accuracy (%)")
        a.grid(alpha=0.25)
    ax.set_xlim(0, 30)
    ax.set_ylim(20, 84)
    ax.set_title("All %d configurations, 30-epoch budget — grey = the other arms, "
                 "coloured = the %d best, black = plain baselines"
                 % (len(short), n_top), fontsize=11)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    az.set_xlim(18, 30)
    az.set_ylim(74.5, 82)
    az.set_title("Zoom on the final third — who actually wins", fontsize=10.5)

    for k, v in long_.items():
        c = curve(v)
        if c:
            al.plot(c[0], c[1], "-", lw=1.6, label="%s (%.2f %%)"
                    % (v["label"], v["acc_mean"] * 100))
    al.set_xlabel("completed epoch")
    al.set_ylabel("test accuracy (%)")
    al.grid(alpha=0.25)
    al.set_title("Separate 120-epoch budget — different schedule, not comparable\n"
                 "to the panels on the left", fontsize=10.5, color="0.3")
    al.legend(loc="lower right", frameon=False, fontsize=7.5)

    fig.suptitle("Test accuracy against epoch — every method, CIFAR-10, "
                 "ResNet-20 + BatchNorm", fontsize=13)
    fig.text(0.5, 0.005, BATCH_NOTE + "  Curves are the current training path, "
             "seed-averaged.", ha="center", fontsize=8, color="0.35", wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    save(fig, "25_all_methods_accuracy_vs_epoch")


if __name__ == "__main__":
    rows = gather()
    fig_final(rows)
    fig_curves(rows)
