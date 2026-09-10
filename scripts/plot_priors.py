"""Wave 5: does a depth prior on sigma beat a flat one?  It does not.

``results/ablation_priors.png``

Four priors, two architectures, three seeds each, against a uniform-sigma
control.  Seven of the eight paired comparisons are negative with the same sign
on every seed; the eighth changes sign.  The figure is built to show that the
failure is **ordered** -- the further a profile departs from flat, the more it
loses -- rather than to leave the reader hunting for a winner.

The uniform arm is drawn in ink rather than in a fifth categorical hue: it is the
reference the others are measured against, not a peer.  The four profile hues
clear the colour-vision gate on *all* pairs.

Paired per-seed deltas are the primary evidence, not the means: wave 4 showed an
unpaired seed spread of 0.36 point against 0.26 on paired differences, so a
comparison of means alone would understate what the seeds actually agree on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_ablation import collect, compare            # noqa: E402
from scripts.ablation5_manifest import PROFILES, H               # noqa: E402
from scripts.plot_ablation import (INK, INK2, INK3, SURFACE, declutter,  # noqa: E402
                                   style_axis)

C = {"A2": "#1baf7a", "A1": "#2a78d6", "A3": "#eb6834", "A4": "#4a3aa7"}
LBL = {"A2": "A2  ~ racine(H)", "A1": "A1  ~ H",
       "A3": "A3  ~ 1/H", "A4": "A4  ~ champ receptif"}
ORDER = ["A2", "A1", "A3", "A4"]


def main():
    a = collect()
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2), facecolor=SURFACE)

    def head(ax, t, s):
        ax.set_title(t, color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=24)
        ax.annotate(s, xy=(0, 1.018), xycoords="axes fraction", color=INK2,
                    fontsize=8.5, va="bottom")

    # --- 1 : les profils ----------------------------------------------------
    ax = axes[0, 0]; style_axis(ax)
    xs = list(range(10))
    ax.plot(xs, [1.0] * 10, color=INK, lw=2.4, marker="o", ms=7, mec=SURFACE,
            mew=1.4, zorder=4)
    for k in ORDER:
        ax.plot(xs, PROFILES[k][0], color=C[k], lw=2.0, marker="o", ms=6,
                mec=SURFACE, mew=1.2, zorder=3)
    # uniforme, A3 and A4 all end at 1.0, so their end labels stack
    ends = [(1.0, "uniforme (temoin)", INK)] + [(PROFILES[k][0][-1], k, C[k])
                                                for k in ORDER]
    for y, lab, c in declutter(ends, 0.085):
        ax.annotate("  " + lab, (9, y), color=c, fontsize=8.5,
                    fontweight="bold", va="center")
    for b, lab in ((3.5, "cartes 32"), (6.5, "16"), (9.4, "8")):
        ax.axvline(b, color=INK3, lw=.8, ls=":", zorder=1) if b < 9 else None
    ax.annotate("cartes 32x32", (1.5, 1.26), color=INK3, fontsize=8, ha="center")
    ax.annotate("16x16", (5, 1.26), color=INK3, fontsize=8, ha="center")
    ax.annotate("8x8", (8, 1.26), color=INK3, fontsize=8, ha="center")
    ax.set_xticks(xs); ax.set_xlim(-.3, 11.4); ax.set_ylim(0, 1.34)
    head(ax, "Les quatre a priori testes",
         "multiplicateur applique a sigma, position par position ; normalise sur le maximum")
    ax.set_xlabel("position du flou (0 = apres le stem, 9 = sortie du dernier bloc)",
                  color=INK2, fontsize=9)
    ax.set_ylabel("multiplicateur sur sigma", color=INK2, fontsize=9)

    # --- 2 et 3 : ecarts apparies ------------------------------------------
    for ax, (fam, ctrl, title, sub) in zip(
            (axes[0, 1], axes[1, 0]),
            (("Q", "P_postblock", "Sans reduction (32x32 constant)",
              "l'a priori seul : les quatre perdent, sur les trois graines"),
             ("P", "D2G", "Avec la reduction apres blocks[2]",
              "le verdict ne change pas ; seul A4 devient indecidable"))):
        style_axis(ax)
        for i, k in enumerate(ORDER):
            c = compare(a, "%s_%s" % (fam, k), ctrl)
            y = len(ORDER) - 1 - i
            ax.plot([min(c["d_per_seed"]), max(c["d_per_seed"])], [y, y],
                    color=C[k], lw=2.0, alpha=.35, zorder=2)
            for v in c["d_per_seed"]:
                ax.plot(v, y, "o", ms=8, color=C[k], mec=SURFACE, mew=1.6, zorder=3)
            ax.plot(c["d_acc_pp"], y, "|", ms=22, mew=2.5, color=C[k], zorder=4)
            same = c["all_same_sign"]
            ax.annotate("  %+.2f%s" % (c["d_acc_pp"], "" if same else "   signe variable"),
                        (max(c["d_per_seed"]), y), color=INK if same else INK3,
                        fontsize=8.5, va="center",
                        fontweight="bold" if same else "normal")
        ax.axvline(0, color=INK, lw=1.4, ls="--", zorder=1)
        ax.annotate("uniforme", (0, len(ORDER) - .35), color=INK, fontsize=8.5,
                    fontweight="bold", ha="center")
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels([LBL[k] for k in ORDER[::-1]], fontsize=9, color=INK2)
        ax.set_xlim(-4.2, 1.5); ax.set_ylim(-.6, len(ORDER) - .15)
        head(ax, title, sub)
        ax.set_xlabel("ecart au temoin uniforme, par graine (points)",
                      color=INK2, fontsize=9)

    # --- 4 : les deux familles s'accordent ---------------------------------
    ax = axes[1, 1]; style_axis(ax)
    lim = [-3.6, 1.0]
    ax.plot(lim, lim, color=INK3, lw=1.0, ls="--", zorder=1)
    ax.axhline(0, color=INK, lw=1.0, alpha=.5, zorder=1)
    ax.axvline(0, color=INK, lw=1.0, alpha=.5, zorder=1)
    for k in ORDER:
        q = compare(a, "Q_%s" % k, "P_postblock")["d_acc_pp"]
        p = compare(a, "P_%s" % k, "D2G")["d_acc_pp"]
        ax.plot(q, p, "o", ms=11, color=C[k], mec=SURFACE, mew=2, zorder=3)
        ax.annotate("  " + k, (q, p), color=C[k], fontsize=9, fontweight="bold",
                    va="center")
    ax.plot(0, 0, "o", ms=11, color=INK, mec=SURFACE, mew=2, zorder=3)
    ax.annotate("  uniforme", (0, 0), color=INK, fontsize=9, fontweight="bold",
                va="center")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    head(ax, "Les deux architectures donnent le meme verdict",
         "chaque a priori perd des deux cotes ; la reduction n'absorbe pas l'echec")
    ax.set_xlabel("ecart au temoin, SANS reduction (points)", color=INK2, fontsize=9)
    ax.set_ylabel("ecart au temoin, AVEC reduction (points)", color=INK2, fontsize=9)

    fig.suptitle("Un a priori sur sigma selon la profondeur - aucun ne bat le profil plat",
                 color=INK, fontsize=13.5, fontweight="bold", y=.988)
    fig.text(.5, .957, "ResNet-20 BN, CIFAR-10 50k/10k, 30 epoques, 3 graines par bras, "
             "poids initiaux partages. Sept comparaisons sur huit sont negatives avec le "
             "meme signe sur les trois graines.", color=INK2, fontsize=9, ha="center")
    fig.tight_layout(rect=[0, .005, 1, .945])
    dest = ROOT / "results" / "ablation_priors.png"
    fig.savefig(dest, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("ecrit ->", dest)


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------

def curves_figure():
    """Test-accuracy trajectories for wave 5, mean over seeds with a spread band.

    ``results/ablation_priors_curves.png``

    The band is the **min-max across the three seeds**, not a standard error:
    with three runs a spread is the honest object, and wave 4 showed the seeds
    move the arms together, so a band drawn per arm overstates how separated two
    arms really are.  The paired per-seed deltas in ``ablation_priors.png``
    remain the evidence; these curves show *when* the profiles separate.
    """
    import glob
    import json

    runs = {}
    for f in glob.glob(str(ROOT / "results/kaggle_outputs/abl*-j*/**/metrics.json"),
                       recursive=True):
        cid = Path(f).parent.name.split("__seed")[0]
        runs.setdefault(cid, []).append(json.loads(Path(f).read_text()))

    def band(cid):
        ms = runs[cid]
        ep = [r["epoch"] for r in ms[0]]
        per = [[r["test_acc_current"] * 100 for r in m] for m in ms]
        mean = [sum(c[i] for c in per) / len(per) for i in range(len(ep))]
        lo = [min(c[i] for c in per) for i in range(len(ep))]
        hi = [max(c[i] for c in per) for i in range(len(ep))]
        return ep, mean, lo, hi

    FAM = (("Q", "P_postblock", "Sans reduction (32x32 constant)"),
           ("P", "D2G", "Avec la reduction apres blocks[2]"))
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0), facecolor=SURFACE)

    for col, (fam, ctrl, title) in enumerate(FAM):
        series = [(ctrl, "uniforme (temoin)", INK)] + \
                 [("%s_%s" % (fam, k), LBL[k], C[k]) for k in ORDER]
        for row, (lo_e, hi_e, zoom) in enumerate(((1.4, 34, False), (17.5, 34, True))):
            ax = axes[row][col]; style_axis(ax)
            ends, ylo, yhi = [], 100.0, 0.0
            for cid, lab, c in series:
                ep, mean, lo, hi = band(cid)
                keep = [i for i, e in enumerate(ep) if e >= lo_e]
                ax.fill_between([ep[i] for i in keep], [lo[i] for i in keep],
                                [hi[i] for i in keep], color=c, alpha=.13,
                                linewidth=0, zorder=2)
                ax.plot([ep[i] for i in keep], [mean[i] for i in keep], color=c,
                        lw=2.2 if cid == ctrl else 1.9, zorder=3)
                ylo = min(ylo, min(lo[i] for i in keep))
                yhi = max(yhi, max(hi[i] for i in keep))
                ends.append((mean[-1], lab if not zoom else lab.split()[0], c))
            span = yhi - ylo
            placed = declutter(ends, span * .055)
            for y, lab, c in placed:
                ax.annotate("  " + lab, (30.4, y), color=c, fontsize=8.5,
                            va="center", fontweight="bold", zorder=4)
            ax.set_ylim(min([ylo] + [y for y, _, _ in placed]) - span * .07,
                        max([yhi] + [y for y, _, _ in placed]) + span * .07)
            ax.set_xlim(lo_e, 40 if zoom else 44)
            ax.axvline(21, color=INK3, lw=.8, ls=":", zorder=1)
            ax.set_title(title if not zoom else title + "  -- zoom fin de course",
                         color=INK, fontsize=11.5, fontweight="bold", loc="left",
                         pad=24)
            ax.annotate("bande = etendue des 3 graines ; trait = moyenne"
                        if not zoom else "les ecarts finaux, a l'echelle",
                        xy=(0, 1.018), xycoords="axes fraction", color=INK2,
                        fontsize=8.5, va="bottom")
            ax.set_xlabel("epoque", color=INK2, fontsize=9)
            ax.set_ylabel("accuracy test (%)", color=INK2, fontsize=9)

    fig.suptitle("A priori sur sigma - trajectoires d'accuracy test, 3 graines par bras",
                 color=INK, fontsize=13.5, fontweight="bold", y=.988)
    fig.tight_layout(rect=[0, .005, 1, .955])
    dest = ROOT / "results" / "ablation_priors_curves.png"
    fig.savefig(dest, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("ecrit ->", dest)
