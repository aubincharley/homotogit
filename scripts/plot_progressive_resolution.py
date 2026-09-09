"""Progressive-resolution figure: two new arms against the campaign's seed-0 controls.

All four runs share verified initial weights, BN buffers, training-probe indices
and per-epoch permutations.

Two evaluation paths are plotted:

* **current** (solid, primary) -- the resolution and sigma of the most recently
  completed update, i.e. the configuration the weights and BN buffers were
  actually trained under.  When an arm has no active filter and runs at 32x32
  (the plain arm throughout, every arm from epoch 21), this *is* the target
  configuration, so the recorded bypassed metric is the current-path value.
  An earlier version of this script read only ``*_filtered``, which is ``None``
  whenever no filter is active -- that dropped the plain curve entirely and cut
  the Gaussian curve off at epoch 21.
* **target** (thin dashed, diagnostic) -- original 32x32, filters bypassed,
  drawn only where it differs from the current path.  Its large excursions are
  BatchNorm mismatch, not predictor quality, so the CE panels are scaled to the
  solid curves and the dashed ones run off the top; the accuracy panel shows
  them in full.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

NEW = Path("results/kaggle_outputs/progres-r20bn-20260909-075846/"
           "progres_r20bn_20260909-075919")
CAMP = Path("results/kaggle_outputs/fulldata-r20bn-20260908-161221/"
            "fulldata_r20bn_20260908-161234")
RUNS = [("plain", CAMP / "plain_seed0", "#7a7a7a", "o"),
        ("Gaussian, 3-epoch steps", CAMP / "plateau_seed0", "#1f77b4", "s"),
        ("progressive resolution", NEW / "progres_seed0", "#2ca02c", "^"),
        ("progressive res. + Gaussian", NEW / "progres_gauss_seed0", "#d62728", "D")]

# title, current key, target key, xlim, ylim, show dashed
PANELS = [
    ("test accuracy", "test_acc_filtered", "test_acc", (0, 30), (0.05, 0.85), True),
    ("test accuracy -- final third (same data, zoomed)",
     "test_acc_filtered", "test_acc", (12, 30), (0.66, 0.82), False),
    ("test cross-entropy", "test_ce_filtered", "test_ce", (0, 30), (0.45, 3.0), True),
    ("training-probe cross-entropy", "train_probe_ce_filtered",
     "train_probe_ce_bypassed", (0, 30), (0.0, 3.0), True),
]
OUT = Path("results/progressive_resolution.png")


def paths(metrics, cur_key, tgt_key):
    """(epochs, current, target).  Current falls back to the bypassed metric
    whenever no filter was active -- there the two paths coincide by definition."""
    ep = [r["epoch"] for r in metrics]
    tgt = [r.get(tgt_key) for r in metrics]
    cur = [r.get(cur_key) if r.get(cur_key) is not None else r.get(tgt_key)
           for r in metrics]
    return ep, cur, tgt


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4))
    ax = axes.ravel()

    for i, (title, cur_key, tgt_key, xlim, ylim, dashed) in enumerate(PANELS):
        a = ax[i]
        if xlim[0] == 0:                     # resolution bands
            a.axvspan(0, 6, color="#efefef", zorder=0, lw=0)
            a.axvspan(6, 12, color="#f7f7f7", zorder=0, lw=0)
        elif xlim[0] < 12:
            a.axvspan(xlim[0], 12, color="#f7f7f7", zorder=0, lw=0)
        for label, d, col, mk in RUNS:
            m = json.loads((d / "metrics.json").read_text())
            ep, cur, tgt = paths(m, cur_key, tgt_key)
            a.plot(ep, cur, "-", color=col, lw=2.2, marker=mk, ms=4.5,
                   markevery=2, label=label, zorder=3)
            if dashed:
                xs = [e for e, c, t in zip(ep, cur, tgt)
                      if c is not None and t is not None and abs(c - t) > 1e-12]
                ys = [t for c, t in zip(cur, tgt)
                      if c is not None and t is not None and abs(c - t) > 1e-12]
                if xs:
                    a.plot(xs, ys, "--", color=col, lw=0.9, alpha=0.55, zorder=2)
        a.set_title(title, fontsize=11)
        a.set_xlabel("completed epoch")
        a.set_xlim(*xlim)
        a.set_ylim(*ylim)
        a.grid(alpha=0.25)
        a.axvline(21, color="0.6", lw=0.9, ls=":", zorder=1)

    for x, t in ((3, "r=16"), (9, "r=24"), (22, "r=32")):
        ax[0].text(x, 0.085, t, ha="center", fontsize=8.5, color="#555555")
    ax[0].text(21.5, 0.30, "filters off\nfrom epoch 21", fontsize=8, color="0.45")
    for j in (2, 3):
        ax[j].text(0.985, 0.95, "dashed diagnostics run off-scale",
                   transform=ax[j].transAxes, ha="right", va="top",
                   fontsize=7.5, color="0.5", style="italic")
    ax[1].text(0.02, 0.96, "final values are the frozen epoch-30 comparison",
               transform=ax[1].transAxes, ha="left", va="top",
               fontsize=7.5, color="0.5", style="italic")

    handles = [Line2D([], [], color=c, lw=2.2, marker=mk, ms=4.5, label=l)
               for l, _, c, mk in RUNS]
    handles += [Line2D([], [], color="0.3", lw=2.2,
                       label="solid = current path (configuration actually trained under)"),
                Line2D([], [], color="0.3", lw=0.9, ls="--",
                       label="dashed = target path (32x32, filters bypassed), only "
                             "where it differs")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Progressive resolution -- CIFAR ResNet-20 + BatchNorm, seed 0, "
                 "full CIFAR-10, 30 epochs\n"
                 "all four runs share verified initial weights, BN buffers, probe "
                 "indices and per-epoch permutations", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
