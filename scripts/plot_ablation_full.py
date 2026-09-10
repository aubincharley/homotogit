"""Figures for both waves of the anti-aliasing ablation (18 cells, 1 seed).

Two PNGs under ``results/``:

* ``ablation_full_curves.png``     -- trajectories, one panel per question
* ``ablation_full_synthesis.png``  -- the shrinkage, cost/gain, what annealing
                                      buys, and the final ranking

Same plotting rules as ``scripts/plot_ablation.py`` (``docs/HANDOVER.md`` §4):
the **current path** is primary and solid, the target path appears only in the
panel that exists to show why it must not be read, axes are bounded to the
primary curves, and end labels are decluttered *before* the axis is bounded --
a label nudged past the top is silently lost, which cost three of them once.

Colour comes from the validated categorical palette (dataviz), light mode.
Each panel answers one question with at most four series, so slots are assigned
within a panel; the plain control keeps slot 1 everywhere.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.plot_ablation import (INK, INK2, INK3, SLOT, SURFACE, declutter,
                                   scatter_labels, style_axis)

L = {
    "C_plain": "32 constant, rien",
    "C_plateau": "32 + flou ancien",
    "P_postblock": "32 + flou NOUVEL",
    "R1": "16>32 image, rien",
    "R2": "16>32 image + flou ancien",
    "R3": "16>32 image + flou NOUVEL",
    "R4": "16>32 apres stem, rien",
    "R5": "16>32 apres stem + flou NOUVEL",
    "R6": "32 + flou NOUVEL, sigma constant",
    "K_const050": "32 + flou ancien, sigma constant",
}
MAIN = ["C_plain", "C_plateau", "P_postblock", "R1", "R2", "R3", "R4", "R5", "R6"]

# One colour per arm, fixed across every panel: an entity must never change
# colour between views.  Eight validated categorical slots (dataviz palette,
# light mode) -- all six hard gates pass on the adjacent pairlist.
CMAP = {"C_plain": "#2a78d6", "C_plateau": "#eb6834", "P_postblock": "#1baf7a",
        "R1": "#eda100", "R2": "#e87ba4", "R3": "#008300",
        "R4": "#4a3aa7", "R5": "#e34948",
        # R6 is the only non-homotopy of the main set: a neutral ink marks it as
        # a different kind of object rather than a ninth categorical identity
        "R6": "#6b6a66",
        "K_const050": "#eb6834"}

PANELS = [
    ("La resolution : l'effet dominant",
     ["C_plain", "R1", "R4"],
     "aucun flou nulle part ; seul le lieu de la reduction change"),
    ("Le flou a resolution constante",
     ["C_plain", "C_plateau", "P_postblock"],
     "vague 1 : passer le flou apres le ReLU vaut +1.7 point"),
    ("Le meme flou, avec la resolution",
     ["R1", "R2", "R3"],
     "vague 2 : le meme ecart tombe a +1.0 point, et l'ancien endroit n'apporte plus rien"),
    ("Reduire apres le stem plutot que sur l'image",
     ["R1", "R3", "R4", "R5"],
     "sans flou la reduction post-stem gagne ; avec flou elle perd"),
]


def load():
    arms, summ = {}, {}
    for f in glob.glob(str(ROOT / "results/kaggle_outputs/abl*-j*/**/metrics.json"),
                       recursive=True):
        arms[Path(f).parent.name.split("__seed")[0]] = json.loads(Path(f).read_text())
    summ = json.loads((ROOT / "results" / "ablation_aa_results.json").read_text())["arms"]
    return arms, summ


def color_for(cid, ids=None):
    return CMAP[cid]


def curves(arms, dest):
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0), facecolor=SURFACE)
    for ax, (title, ids, sub) in zip(axes.ravel(), PANELS):
        style_axis(ax)
        lo, hi, ends = 1.0, 0.0, []
        for cid in ids:
            m = arms[cid]
            c = color_for(cid, ids)
            e = [r["epoch"] for r in m]
            cur = [r["test_acc_current"] for r in m]
            lo, hi = min(lo, min(cur[1:])), max(hi, max(cur))
            ax.plot(e, cur, color=c, lw=2.0, zorder=3, solid_capstyle="round")
            ends.append((cur[-1], L[cid], c))
        span = hi - lo
        placed = declutter(ends, span * 0.05)
        ylo = min([lo] + [y for y, _, _ in placed]) - span * 0.06
        yhi = max([hi] + [y for y, _, _ in placed]) + span * 0.06
        for y, lab, c in placed:
            ax.annotate(" " + lab, (30.4, y), color=c, fontsize=8.5, va="center",
                        ha="left", fontweight="bold", zorder=4)
        ax.axvline(21, color=INK3, lw=0.8, ls=":", zorder=1)
        ax.annotate("extinction e21", (20.5, ylo + span * 0.02), color=INK3,
                    fontsize=7, rotation=90, va="bottom", ha="right")
        ax.set_ylim(ylo, yhi); ax.set_xlim(1.4, 45)
        ax.set_title(title, color=INK, fontsize=11.5, fontweight="bold",
                     loc="left", pad=24)
        ax.annotate(sub, xy=(0, 1.018), xycoords="axes fraction", color=INK2,
                    fontsize=8.5, va="bottom")
        ax.set_xlabel("epoque", color=INK2, fontsize=9)
        ax.set_ylabel("accuracy test (chemin courant)", color=INK2, fontsize=9)
    fig.suptitle("Ablation complete - ResNet-20 BN, CIFAR-10 50k/10k, 30 epoques, 1 graine",
                 color=INK, fontsize=13.5, fontweight="bold", y=0.988)
    fig.text(0.5, 0.958, "18 bras, tous partis des memes poids initiaux epingles, donc "
             "directement comparables entre eux. Une seule graine : les ecarts sous 1 point "
             "ne sont pas exploitables.", color=INK2, fontsize=9, ha="center")
    fig.tight_layout(rect=[0, 0.005, 1, 0.945])
    fig.savefig(dest, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def synthesis(summ, dest):
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), facecolor=SURFACE)
    A = lambda k: summ[k]["acc"]
    base = A("C_plain")

    def head(ax, t, s):
        ax.set_title(t, color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=24)
        ax.annotate(s, xy=(0, 1.018), xycoords="axes fraction", color=INK2,
                    fontsize=8.5, va="bottom")

    # --- E : le retrecissement de l'apport du flou --------------------------
    ax = axes[0, 0]; style_axis(ax)
    groups = ["ancien endroit\n(avant le ReLU)", "NOUVEL endroit\n(apres le ReLU)"]
    at32 = [(A("C_plateau") - base) * 100, (A("P_postblock") - base) * 100]
    atpr = [(A("R2") - A("R1")) * 100, (A("R3") - A("R1")) * 100]
    x = [0, 1]; w = 0.34
    # two conditions of one measure -> a sequential pair, deliberately NOT the
    # categorical slots, which already stand for individual arms in this figure
    ax.bar([i - w/2 - .012 for i in x], at32, w, color="#12447e", zorder=3,
           label="a 32x32 constant")
    ax.bar([i + w/2 + .012 for i in x], atpr, w, color="#8fb8e8", zorder=3,
           label="ajoute a la resolution progressive")
    for i, (a, b) in enumerate(zip(at32, atpr)):
        ax.annotate("+%.2f" % a, (i - w/2 - .012, a), ha="center", va="bottom",
                    color=INK, fontsize=9, fontweight="bold",
                    textcoords="offset points", xytext=(0, 3))
        ax.annotate("+%.2f" % b, (i + w/2 + .012, b), ha="center", va="bottom",
                    color=INK, fontsize=9, fontweight="bold",
                    textcoords="offset points", xytext=(0, 3))
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9, color=INK2)
    ax.set_ylim(0, max(at32) * 1.18)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    head(ax, "Ce que le flou apporte, selon qu'il y a deja la resolution",
         "la resolution mange l'essentiel du benefice du flou")
    ax.set_ylabel("gain d'accuracy (points)", color=INK2, fontsize=9)

    # --- F : cout contre gain ----------------------------------------------
    ax = axes[0, 1]; style_axis(ax)
    wp = summ["C_plain"]["wall_s"]
    pts = []
    for k in MAIN:
        if k == "C_plain":
            continue
        c = CMAP[k]
        xx, yy = summ[k]["wall_s"] / wp, (A(k) - base) * 100
        ax.plot(xx, yy, "o", ms=9, color=c, mec=SURFACE, mew=2, zorder=3)
        pts.append((xx, yy, L[k], c))
    ax.set_xlim(0.96, 1.78)
    ax.set_ylim(min(t[1] for t in pts) - .4, max(t[1] for t in pts) + .5)
    scatter_labels(ax, pts, (max(t[1] for t in pts) - min(t[1] for t in pts)) * .06)
    head(ax, "Cout contre gain", "temps total rapporte au temoin")
    ax.set_xlabel("temps du run / temps du temoin", color=INK2, fontsize=9)
    ax.set_ylabel("gain d'accuracy (points)", color=INK2, fontsize=9)

    # --- G : ce que l'annelage achete --------------------------------------
    ax = axes[1, 0]; style_axis(ax)
    show = ["C_plateau", "P_postblock", "R3", "R4", "K_const050", "R6"]
    ys = list(range(len(show)))[::-1]
    for y, k in zip(ys, show):
        cur, tgt = summ[k]["acc_current"], summ[k]["acc_target"]
        homo = summ[k]["primary_path"] == "target"
        c = CMAP[k]
        ax.plot([tgt, cur], [y, y], color=c, lw=2.5, zorder=2, alpha=.45)
        ax.plot(cur, y, "o", ms=10, color=c, mec=SURFACE, mew=2, zorder=3)
        ax.plot(tgt, y, "o", ms=7, color=SURFACE, mec=c, mew=2, zorder=3)
        # the two markers coincide for a homotopy, which reads as a missing
        # marker unless the zero is stated
        ax.annotate("  %s" % ("0 pt" if homo else "-%.0f pts" % ((cur - tgt) * 100)),
                    (cur, y), va="center", color=c if not homo else INK2,
                    fontsize=8.5, fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels([L[k] + ("" if summ[k]["primary_path"] == "target"
                                else "  [non homotopie]") for k in show],
                       fontsize=8.5, color=INK2)
    ax.set_xlim(0.15, 0.95); ax.set_ylim(-0.7, len(show) - 0.3)
    ax.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", ms=9, color=INK2, label="avec son filtre"),
        plt.Line2D([], [], marker="o", ls="", ms=7, mfc=SURFACE, mec=INK2, mew=2,
                   color=INK2, label="filtre retire")],
        frameon=False, fontsize=8.5, labelcolor=INK2, loc="center left")
    head(ax, "Ce que l'annelage achete : l'architecture cible",
         "les bras a sigma constant s'effondrent des qu'on retire leur filtre")
    ax.set_xlabel("accuracy test", color=INK2, fontsize=9)

    # --- H : classement -----------------------------------------------------
    ax = axes[1, 1]; style_axis(ax)
    order = sorted(MAIN, key=A)
    ys = list(range(len(order)))
    for y, k in zip(ys, order):
        c = CMAP[k]
        ax.barh(y, (A(k) - 0.74) * 100, color=c, height=.6, zorder=3)
        ax.annotate("  %.4f" % A(k), ((A(k) - 0.74) * 100, y), va="center",
                    color=INK, fontsize=8.5, fontweight="bold")
    ax.set_yticks(ys); ax.set_yticklabels([L[k] for k in order], fontsize=8.5, color=INK2)
    ax.set_xlim(0, (max(A(k) for k in MAIN) - 0.74) * 100 * 1.22)
    ax.set_xticks([]); ax.grid(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    head(ax, "Classement des 9 configurations", "accuracy test finale, chemin primaire")

    fig.suptitle("Ablation complete - synthese", color=INK, fontsize=13.5,
                 fontweight="bold", y=0.988)
    fig.tight_layout(rect=[0, 0.005, 1, 0.955])
    fig.savefig(dest, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    arms, summ = load()
    d1 = ROOT / "results" / "ablation_full_curves.png"
    d2 = ROOT / "results" / "ablation_full_synthesis.png"
    curves(arms, d1); synthesis(summ, d2)
    print("ecrit ->", d1); print("ecrit ->", d2)


if __name__ == "__main__":
    main()
