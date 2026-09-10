"""What three seeds show: raw spread, the common seed effect, and paired deltas.

``results/ablation_seeds.png``

Four arms were re-run at seeds 0/1/2 (``D0``, ``D1``, ``D2``, ``D2G``).  The
figure exists to answer one question honestly: **a single-seed ranking of these
arms was not a result**, and the reason is visible rather than asserted.

The story runs across the four panels:

1. each arm varies by up to 0.8 point between seeds -- wider than every gap the
   one-seed ranking was built on;
2. but the seeds move the arms **together**: they share pinned initial weights
   and per-epoch permutations, so seed 2 lifts all four;
3. so the *paired* differences are far tighter than the raw spread, and three of
   them keep their sign on every seed;
4. shown one comparison at a time for the pair that matters.

Colour is one validated categorical slot per arm, stable across panels; the
palette clears the colour-vision gate on **all** pairs, not just adjacent ones,
because any two arms sit side by side somewhere in this figure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_ablation import collect, compare      # noqa: E402
from scripts.plot_ablation import INK, INK2, INK3, SURFACE, style_axis  # noqa: E402

ARMS = ["D2G", "D1", "D0", "D2"]
C = {"D2G": "#4a3aa7", "D1": "#1baf7a", "D0": "#2a78d6", "D2": "#eb6834"}
L = {"D2G": "blocks[2] + flou", "D1": "blocks[1], sans flou",
     "D0": "blocks[0], sans flou", "D2": "blocks[2], sans flou"}
PAIRS = [("D2G", "D1"), ("D2G", "D0"), ("D2G", "D2"),
         ("D1", "D0"), ("D1", "D2"), ("D0", "D2")]


def main():
    a = collect()
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0), facecolor=SURFACE)

    def head(ax, t, s):
        ax.set_title(t, color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=24)
        ax.annotate(s, xy=(0, 1.018), xycoords="axes fraction", color=INK2,
                    fontsize=8.5, va="bottom")

    # --- 1 : valeurs par graine --------------------------------------------
    ax = axes[0, 0]; style_axis(ax)
    for i, k in enumerate(ARMS):
        ys = [v * 100 for v in a[k]["acc_per_seed"]]
        y = len(ARMS) - 1 - i
        ax.plot([min(ys), max(ys)], [y, y], color=C[k], lw=2.0, alpha=.35, zorder=2)
        # two seeds can land on the same value (D2 gives 0.7999 and 0.8000):
        # the markers coincide and one label hides under the other, which reads
        # as a missing seed.  Alternate the label side when that happens.
        placed = []
        for s, v in zip(a[k]["seeds"], ys):
            ax.plot(v, y, "o", ms=8, color=C[k], mec=SURFACE, mew=1.6, zorder=3)
            up = not any(abs(v - w) < 0.06 for w in placed)
            ax.annotate(str(s), (v, y), textcoords="offset points",
                        xytext=(0, 9 if up else -15), ha="center",
                        color=INK3, fontsize=7)
            placed.append(v)
        ax.plot(a[k]["acc"] * 100, y, "|", ms=22, mew=2.5, color=C[k], zorder=4)
        ax.annotate("  etendue %.2f pt" % (max(ys) - min(ys)), (max(ys), y),
                    color=INK2, fontsize=8, va="center")
    ax.set_yticks(range(len(ARMS)))
    ax.set_yticklabels([L[k] for k in ARMS[::-1]], fontsize=9, color=INK2)
    ax.set_xlim(79.6, 82.0); ax.set_ylim(-.6, len(ARMS) - .4)
    head(ax, "1. Chaque bras varie beaucoup d'une graine a l'autre",
         "les chiffres au-dessus des points sont les numeros de graine ; la barre est la moyenne")
    ax.set_xlabel("accuracy test (%)", color=INK2, fontsize=9)

    # --- 2 : les graines bougent ensemble ----------------------------------
    ax = axes[0, 1]; style_axis(ax)
    seeds = a["D1"]["seeds"]
    for k in ARMS:
        ys = [v * 100 for v in a[k]["acc_per_seed"]]
        ax.plot(seeds, ys, color=C[k], lw=2.2, marker="o", ms=8, mec=SURFACE,
                mew=1.6, zorder=3)
        ax.annotate("  " + L[k], (seeds[-1], ys[-1]), color=C[k], fontsize=8.5,
                    va="center", fontweight="bold")
    ax.set_xticks(seeds); ax.set_xlim(-.15, 2.9)
    head(ax, "2. Mais elles les bougent TOUS dans le meme sens",
         "poids initiaux et permutations sont partages : la graine 2 souleve les quatre bras")
    ax.set_xlabel("graine", color=INK2, fontsize=9)
    ax.set_ylabel("accuracy test (%)", color=INK2, fontsize=9)

    # --- 3 : differences appariees -----------------------------------------
    ax = axes[1, 0]; style_axis(ax)
    for i, (x, y) in enumerate(PAIRS):
        c = compare(a, x, y)
        yy = len(PAIRS) - 1 - i
        same = c["all_same_sign"]
        col = C[x]
        ax.plot([min(c["d_per_seed"]), max(c["d_per_seed"])], [yy, yy],
                color=col, lw=2.0, alpha=.35, zorder=2)
        for v in c["d_per_seed"]:
            ax.plot(v, yy, "o", ms=8, color=col, mec=SURFACE, mew=1.6, zorder=3)
        ax.plot(c["d_acc_pp"], yy, "|", ms=22, mew=2.5, color=col, zorder=4)
        ax.annotate("  %+.2f%s" % (c["d_acc_pp"], "" if same else "  signe variable"),
                    (max(c["d_per_seed"]), yy), color=INK if same else INK3,
                    fontsize=8.5, va="center", fontweight="bold" if same else "normal")
    ax.axvline(0, color=INK3, lw=1.2, ls="--", zorder=1)
    ax.set_yticks(range(len(PAIRS)))
    ax.set_yticklabels(["%s - %s" % p for p in PAIRS[::-1]], fontsize=9, color=INK2)
    ax.set_xlim(-.35, 1.55); ax.set_ylim(-.6, len(PAIRS) - .4)
    head(ax, "3. Les differences appariees sont bien plus serrees",
         "sd non appariee 0.36 pt contre 0.26 pt sur les differences ; pointille = zero")
    ax.set_xlabel("ecart d'accuracy, par graine (points)", color=INK2, fontsize=9)

    # --- 4 : la comparaison qui decide -------------------------------------
    ax = axes[1, 1]; style_axis(ax)
    for i, s in enumerate(seeds):
        va = a["D2G"]["acc_per_seed"][i] * 100
        vb = a["D1"]["acc_per_seed"][i] * 100
        ax.plot([i, i], [vb, va], color=INK3, lw=1.6, zorder=2)
        ax.plot(i, vb, "o", ms=11, color=C["D1"], mec=SURFACE, mew=2, zorder=3)
        ax.plot(i, va, "o", ms=11, color=C["D2G"], mec=SURFACE, mew=2, zorder=3)
        ax.annotate("+%.2f" % (va - vb), (i, (va + vb) / 2),
                    textcoords="offset points", xytext=(10, 0), color=INK,
                    fontsize=9, fontweight="bold", va="center")
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(["graine %d" % s for s in seeds], fontsize=9, color=INK2)
    ax.set_xlim(-.4, 2.5)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=10, color=C[k],
                                  label=L[k]) for k in ("D2G", "D1")],
              frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    head(ax, "4. D2G bat D1 sur les trois graines",
         "a une graine l'ecart valait 0.12 pt et je l'avais classe comme du bruit ; il ne l'est pas")
    ax.set_ylabel("accuracy test (%)", color=INK2, fontsize=9)

    fig.suptitle("Trois graines sur les quatre meilleurs bras - ResNet-20 BN, "
                 "CIFAR-10 50k/10k, 30 epoques", color=INK, fontsize=13,
                 fontweight="bold", y=.988)
    fig.text(.5, .957, "trois executions restent un petit echantillon et le jeu de test "
             "a deja ete expose : ceci decrit une dispersion, ce n'est pas une affirmation "
             "de confiance.", color=INK2, fontsize=9, ha="center")
    fig.tight_layout(rect=[0, .005, 1, .945])
    dest = ROOT / "results" / "ablation_seeds.png"
    fig.savefig(dest, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("ecrit ->", dest)


if __name__ == "__main__":
    main()
