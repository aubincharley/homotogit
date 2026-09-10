"""Figures for all three waves of the ablation (24 cells, 1 seed).

* ``ablation_all_grid.png``    -- the depth x blur grid: the wave-3 result
* ``ablation_all_ranking.png`` -- every arm, ranked, with its cost beside it

Plotting rules from ``docs/HANDOVER.md`` section 4: the **current path** is the
primary series, axes are bounded to it, end labels are decluttered *before* the
axis is bounded, and the rendered PNG is read back rather than trusted.

Colour encodes the **family** of intervention, on a four-slot palette validated
across *all* pairs (not just adjacent ones) because in a ranked chart any two
families can end up side by side.  The constant-sigma arms -- the ones that never
reach the target endpoint, and so are not continuations -- carry a hatch as a
secondary encoding rather than a fifth hue: no five-hue set cleared the
all-pairs colour-vision gate.
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
from scripts.plot_ablation import (INK, INK2, INK3, SURFACE, declutter,
                                   scatter_labels, style_axis)

FAM = {"temoin": "#2a78d6", "flou": "#eb6834",
       "resolution": "#1baf7a", "resolution+flou": "#4a3aa7"}

ARMS = {
    "C_plain":            ("temoin",          "temoin : rien du tout"),
    "C_plateau":          ("flou",            "32 : flou avant ReLU, 19 sites"),
    "P_postbn":           ("flou",            "32 : flou apres BN, 19 sites"),
    "P_postblock":        ("flou",            "32 : flou apres ReLU, 10 sites"),
    "M_predown":          ("flou",            "32 : flou 2 sites pre-decimation"),
    "M_nodown":           ("flou",            "32 : flou 17 sites hors decimation"),
    "K_const030":         ("flou",            "32 : flou constant 0.30"),
    "K_const050":         ("flou",            "32 : flou constant 0.50"),
    "K_const080":         ("flou",            "32 : flou constant 0.80"),
    "K_const100":         ("flou",            "32 : flou constant 1.00"),
    "B_blurpool":         ("flou",            "32 : BlurPool fixe, 2 sites"),
    "B_blurpool_plateau": ("flou",            "32 : BlurPool + flou avant ReLU"),
    "R6":                 ("flou",            "32 : flou apres ReLU, constant 0.50"),
    "R1":                 ("resolution",      "reduction sur l'image"),
    "R4":                 ("resolution",      "reduction apres le stem"),
    "D0":                 ("resolution",      "reduction apres blocks[0]"),
    "D1":                 ("resolution",      "reduction apres blocks[1]"),
    "D2":                 ("resolution",      "reduction apres blocks[2]"),
    "R2":                 ("resolution+flou", "image + flou avant ReLU"),
    "R3":                 ("resolution+flou", "image + flou apres ReLU"),
    "R5":                 ("resolution+flou", "stem + flou apres ReLU"),
    "D0G":                ("resolution+flou", "blocks[0] + flou apres ReLU"),
    "D1G":                ("resolution+flou", "blocks[1] + flou apres ReLU"),
    "D2G":                ("resolution+flou", "blocks[2] + flou apres ReLU"),
}

DEPTH = [("sur l'image", "R1", "R3"), ("apres le stem", "R4", "R5"),
         ("apres blocks[0]", "D0", "D0G"), ("apres blocks[1]", "D1", "D1G"),
         ("apres blocks[2]", "D2", "D2G")]


def load():
    curves = {}
    for f in glob.glob(str(ROOT / "results/kaggle_outputs/abl*-j*/**/metrics.json"),
                       recursive=True):
        curves[Path(f).parent.name.split("__seed")[0]] = json.loads(Path(f).read_text())
    summ = json.loads((ROOT / "results/ablation_aa_results.json").read_text())["arms"]
    return curves, summ


def grid_figure(curves, S, dest):
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), facecolor=SURFACE)
    A = lambda k: S[k]["acc"]
    base = A("C_plain")

    def head(ax, t, s):
        ax.set_title(t, color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=24)
        ax.annotate(s, xy=(0, 1.018), xycoords="axes fraction", color=INK2,
                    fontsize=8.5, va="bottom")

    # --- A1 : gain contre profondeur ---------------------------------------
    ax = axes[0, 0]; style_axis(ax)
    xs = list(range(len(DEPTH)))
    g_no = [(A(n) - base) * 100 for _, n, _ in DEPTH]
    g_bl = [(A(g) - base) * 100 for _, _, g in DEPTH]
    ax.plot(xs, g_no, color=FAM["resolution"], lw=2.2, marker="o", ms=9,
            mec=SURFACE, mew=2, zorder=3, label="sans flou")
    ax.plot(xs, g_bl, color=FAM["resolution+flou"], lw=2.2, marker="o", ms=9,
            mec=SURFACE, mew=2, zorder=3, label="avec flou apres le ReLU")
    # place each label away from the other series, or the two collide wherever
    # the lines cross -- which they do three times
    for x, a, b in zip(xs, g_no, g_bl):
        ax.annotate("%.2f" % a, (x, a), textcoords="offset points",
                    xytext=(0, 11 if a >= b else -17), ha="center",
                    color=FAM["resolution"], fontsize=8.5, fontweight="bold")
        ax.annotate("%.2f" % b, (x, b), textcoords="offset points",
                    xytext=(0, 11 if b > a else -17), ha="center",
                    color=FAM["resolution+flou"], fontsize=8.5, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([d[0].replace("apres ", "apres\n") for d in DEPTH],
                       fontsize=8.5, color=INK2)
    ax.set_ylim(min(g_no + g_bl) - .55, max(g_no + g_bl) + .55)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    head(ax, "Profondeur de la reduction : un plateau, pas un pic",
         "les 4 positions internes tiennent dans 0.5 point, sous le seuil exploitable a une graine")
    ax.set_ylabel("gain d'accuracy sur le temoin (points)", color=INK2, fontsize=9)

    # --- A2 / A3 : trajectoires --------------------------------------------
    # depth is an ORDERED variable, so it gets a sequential ramp -- one hue,
    # light to dark -- not a categorical set and not a multi-hue map
    for ax, col, ramp, title, sub in (
            (axes[0, 1], 1, plt.cm.Greens, "Serie de profondeur, sans flou",
             "reduire a l'interieur bat reduire sur l'image ; teinte claire = peu profond"),
            (axes[1, 0], 2, plt.cm.Purples, "Serie de profondeur, avec flou apres le ReLU",
             "le flou ne reordonne pas les profondeurs ; teinte claire = peu profond")):
        style_axis(ax)
        lo, hi, ends = 1.0, 0.0, []
        for i, d in enumerate(DEPTH):
            cid = d[col]
            m = curves[cid]
            c = ramp(0.38 + 0.55 * i / (len(DEPTH) - 1))
            e = [r["epoch"] for r in m]
            cur = [r["test_acc_current"] for r in m]
            lo, hi = min(lo, min(cur[1:])), max(hi, max(cur))
            ax.plot(e, cur, color=c, lw=2.0, zorder=3)
            ends.append((cur[-1], d[0], c))
        span = hi - lo
        placed = declutter(ends, span * 0.05)
        ylo = min([lo] + [y for y, _, _ in placed]) - span * .06
        yhi = max([hi] + [y for y, _, _ in placed]) + span * .06
        for y, lab, c in placed:
            ax.annotate(" " + lab, (30.4, y), color=c, fontsize=8.5, va="center",
                        fontweight="bold")
        ax.set_ylim(ylo, yhi); ax.set_xlim(1.4, 46)
        head(ax, title, sub)
        ax.set_xlabel("epoque", color=INK2, fontsize=9)
        ax.set_ylabel("accuracy test", color=INK2, fontsize=9)

    # --- A4 : cout contre gain, les 24 + frontiere de Pareto ---------------
    ax = axes[1, 1]; style_axis(ax)
    wb = S["C_plain"]["wall_s"]
    pts = []
    for k in ARMS:
        fam, _ = ARMS[k]
        x, y = S[k]["wall_s"] / wb, (A(k) - base) * 100
        hatch = S[k]["primary_path"] == "current"
        ax.plot(x, y, "o", ms=8, color=FAM[fam], mec=SURFACE, mew=1.6,
                alpha=.45 if hatch else 1.0, zorder=3)
        pts.append((x, y, k))
    front = sorted([(S[k]["wall_s"] / wb, (A(k) - base) * 100) for k in ARMS])
    fx, fy, best = [], [], -99
    for x, y in front:
        if y > best:
            fx.append(x); fy.append(y); best = y
    ax.step(fx, fy, where="post", color=INK3, lw=1.2, ls="--", zorder=2)
    ax.set_xlim(0.95, 1.90); ax.set_ylim(-4.6, 6.6)
    named = [(S[k]["wall_s"] / wb, (A(k) - base) * 100, k, INK)
             for k in ("D1", "D2G", "R3", "C_plateau", "P_postblock")]
    scatter_labels(ax, named, 0.55)
    head(ax, "Cout contre gain, les 24 bras",
         "pointille = frontiere de Pareto ; translucide = non-homotopie")
    ax.set_xlabel("temps du run / temps du temoin", color=INK2, fontsize=9)
    ax.set_ylabel("gain d'accuracy (points)", color=INK2, fontsize=9)

    fig.suptitle("Ablation, vague 3 : ou placer l'unique reduction de resolution",
                 color=INK, fontsize=13.5, fontweight="bold", y=.988)
    fig.tight_layout(rect=[0, .005, 1, .955])
    fig.savefig(dest, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def ranking_figure(S, dest):
    order = sorted(ARMS, key=lambda k: S[k]["acc"])
    n = len(order)
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 11.0), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [2.5, 1]})
    base, wb = S["C_plain"]["acc"], S["C_plain"]["wall_s"]

    ax = axes[0]; style_axis(ax)
    for i, k in enumerate(order):
        fam, lab = ARMS[k]
        cst = S[k]["primary_path"] == "current"
        ax.barh(i, (S[k]["acc"] - .70) * 100, color=FAM[fam], height=.72, zorder=3,
                hatch="///" if cst else None, edgecolor=SURFACE, linewidth=.8)
        ax.annotate("  %.4f   %+5.2f pt" % (S[k]["acc"], (S[k]["acc"] - base) * 100),
                    ((S[k]["acc"] - .70) * 100, i), va="center", color=INK,
                    fontsize=8.5, fontweight="bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels([ARMS[k][1] for k in order], fontsize=8.5, color=INK2)
    ax.set_xlim(0, (max(S[k]["acc"] for k in order) - .70) * 100 * 1.30)
    ax.set_xticks([]); ax.grid(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.set_ylim(-.8, n - .2)
    ax.set_title("Les 24 bras, classes par accuracy", color=INK, fontsize=12.5,
                 fontweight="bold", loc="left", pad=26)
    ax.annotate("une graine : les ecarts sous 1 point ne sont pas exploitables. "
                "Hachures = sigma constant, donc pas une continuation : le reseau final "
                "garde ses filtres.", xy=(0, 1.012), xycoords="axes fraction",
                color=INK2, fontsize=8.5, va="bottom")
    ax.legend(handles=[plt.Line2D([], [], marker="s", ls="", ms=10, color=v, label=k)
                       for k, v in FAM.items()],
              frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")

    ax = axes[1]; style_axis(ax)
    for i, k in enumerate(order):
        fam, _ = ARMS[k]
        ax.barh(i, S[k]["wall_s"] / wb, color=FAM[fam], height=.72, zorder=3,
                alpha=.85)
        ax.annotate("  %.2fx" % (S[k]["wall_s"] / wb), (S[k]["wall_s"] / wb, i),
                    va="center", color=INK, fontsize=8, fontweight="bold")
    ax.axvline(1.0, color=INK3, lw=1.0, ls="--", zorder=4)
    ax.set_yticks(range(n)); ax.set_yticklabels([])
    ax.set_xlim(0, 2.05); ax.set_ylim(-.8, n - .2)
    ax.set_title("Cout", color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=26)
    ax.annotate("temps du run rapporte au temoin", xy=(0, 1.012),
                xycoords="axes fraction", color=INK2, fontsize=8.5, va="bottom")
    ax.set_xlabel("x temoin", color=INK2, fontsize=9)

    fig.suptitle("Ablation complete - 24 bras, memes poids initiaux, ResNet-20 BN, "
                 "CIFAR-10 50k/10k, 30 epoques, 1 graine",
                 color=INK, fontsize=12.5, fontweight="bold", y=.985)
    fig.tight_layout(rect=[0, .005, 1, .955])
    fig.savefig(dest, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    curves, S = load()
    d1 = ROOT / "results" / "ablation_all_grid.png"
    d2 = ROOT / "results" / "ablation_all_ranking.png"
    grid_figure(curves, S, d1); ranking_figure(S, d2)
    print("ecrit ->", d1); print("ecrit ->", d2)


if __name__ == "__main__":
    main()
