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
arms share its data order but not its initial weights (docs/AUDIT.md A1-A3).  Marker shape encodes the batch so
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
from scripts.method_labels import REPRESENTATIVE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "presentation"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "legend.fontsize": 9, "figure.facecolor": "white"})

FAM_COLOR = {
    "plain": "#000000", "gauss": "#1f77b4", "res": "#2ca02c",
    "res_gauss": "#d62728", "profile": "#17becf", "operator": "#ff7f0e",
    "adaptive": "#9467bd", "constant": "#8c564b", "mix": "#e377c2",
    "control": "#7f7f7f",
}
BATCH_MARKER = {"campaign": "o", "resbench": "^", "ablation": "s",
                "adaptive": "D", "long120": "*", "unified": "P"}
BATCH_NOTE = ("Marker = batch.  campaign, resbench and unified (and Aubin's seed-0 "
              "runs) share the pinned asset set r20bn-campaign-assets; the ablation "
              "batch shares its data order but not its initial weights.  Plain runs on "
              "identical assets still finish up to 0.73 pp apart across batches (cause "
              "not established), so cross-batch gaps of that size are not interpretable.")
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
    order = sorted(rows, key=lambda k: (rows[k]["acc_mean"] is not None,
                                        rows[k]["acc_mean"] or 0.0))
    n = len(order)
    fig, ax = plt.subplots(figsize=(14.5, 0.235 * n + 2.6))

    base = next((v["acc_mean"] for v in rows.values()
                 if v["label"].startswith("Baseline: no blur, full 32x32 throughout")
                 and v["batch"] == "unified"), None)
    if base:
        ax.axvline(base * 100, color="0.2", lw=1.5, ls="--", zorder=1)
        ax.text(base * 100, n * 0.5, " baseline %.2f %% " % (base * 100),
                fontsize=9, color="0.2", va="center", rotation=90, ha="center",
                backgroundcolor="white")
    for i, k in enumerate(order):
        v = rows[k]
        col = FAM_COLOR[v["family"]]
        if v["acc_mean"] is None:
            ax.text(0.01, i, "diverged: non-finite loss in %d / %d seeds (no accuracy)"
                    % (v["n_attempted"], v["n_attempted"]), transform=ax.get_yaxis_transform(),
                    fontsize=7, color="#b00000", va="center")
            continue
        if v["n_seeds"] > 1:
            sd = v["acc_sd"] * 100
            ax.plot([v["acc_mean"] * 100 - sd, v["acc_mean"] * 100 + sd], [i, i],
                    color=col, lw=1.5, alpha=0.8, zorder=2)
            for a in v["acc_per_seed"]:
                ax.scatter([a * 100], [i], color=col, alpha=0.35, s=16, zorder=2)
        ax.scatter([v["acc_mean"] * 100], [i], color=col, s=74,
                   marker=BATCH_MARKER[v["batch"]], edgecolor="white",
                   linewidth=0.7, zorder=3)
        if v["n_seeds"] == 1:
            ax.text(v["acc_mean"] * 100 + 0.09, i, "1 seed, no SD", fontsize=6,
                    color="0.6", va="center")
    ax.set_yticks(range(n))
    ax.set_yticklabels([rows[k]["label"] for k in order], fontsize=7.6)
    ax.set_ylim(-1, n)
    ax.set_xlabel("final test accuracy (%) after 30 epochs")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Every method we have trained — final accuracy\n"
                 "%d configurations, CIFAR-10, ResNet-20 + BatchNorm, "
                 "30 epochs, no augmentation" % n, fontsize=12)
    handles = [Line2D([], [], color=c, marker="o", ls="", label=FAMILY[f])
               for f, c in FAM_COLOR.items() if f in FAMILY]
    handles += [Line2D([], [], color="0.35", marker=m, ls="", label=b)
                for b, m in BATCH_MARKER.items() if b != "long120"]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8, ncol=2)
    fig.text(0.5, 0.004, BATCH_NOTE + "  " + SD_NOTE, ha="center", fontsize=8,
             color="0.35", wrap=True)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, "24_all_methods_final_accuracy")


def fig_curves(rows):
    keys = [k for k in REPRESENTATIVE if k in rows]
    missing = [k for k in REPRESENTATIVE if k not in rows]
    if missing:
        print("representative keys not found:", missing)
    keys.sort(key=lambda k: -rows[k]["acc_mean"])

    fig, (ax, az) = plt.subplots(1, 2, figsize=(16, 6.4),
                                 gridspec_kw={"width_ratios": [1.35, 1]})
    cyc = ["#d62728", "#17becf", "#2ca02c", "#ff7f0e", "#9467bd", "#1f77b4",
           "#8c564b", "#e377c2", "#bcbd22", "#7f7f7f", "#c49c94"]
    ci = 0
    for k in keys:
        v = rows[k]
        c = curve(v) if v["acc_mean"] is not None else None
        if not c:
            continue
        plain = v["family"] == "plain"
        col = "#000000" if plain else cyc[ci % len(cyc)]
        if not plain:
            ci += 1
        lab = "%s  —  %.2f %%" % (v["label"], v["acc_mean"] * 100)
        for a in (ax, az):
            a.plot(c[0], c[1], "-", color=col, lw=2.6 if plain else 1.9,
                   label=lab, zorder=4 if plain else 3)
    for a in (ax, az):
        for x in (6, 12):
            a.axvline(x, color="0.88", lw=0.8, ls="--", zorder=0)
        a.axvline(21, color="0.72", lw=0.8, ls=":", zorder=0)
        a.set_xlabel("completed epoch")
        a.set_ylabel("test accuracy (%)")
        a.grid(alpha=0.25)
    ax.set_xlim(0, 30)
    ax.set_ylim(20, 84)
    ax.set_title("Full training run", fontsize=11)
    az.set_xlim(18, 30)
    az.set_ylim(74.5, 82)
    az.set_title("Zoom on the last third — where the ordering settles", fontsize=11)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("Test accuracy against epoch — one representative run per idea\n"
                 "CIFAR-10, ResNet-20 + BatchNorm, 30 epochs, seed-averaged; eight curves from the unified batch, four from other batches",
                 fontsize=12.5)
    fig.text(0.5, 0.005,
             "Dashed lines mark the resolution changes (epochs 6 and 12); the "
             "dotted line marks where the blur switches off (epoch 21).  " + BATCH_NOTE.replace("Marker = batch.  ", ""),
             ha="center", fontsize=8, color="0.35", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    save(fig, "25_representative_accuracy_vs_epoch")


if __name__ == "__main__":
    rows = gather()
    fig_final(rows)
    fig_curves(rows)
