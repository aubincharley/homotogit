"""Figures for the resolution-only benchmark, in the campaign's French style.

One figure per question -- operator, depth, path -- plus a compact
accuracy-versus-cost view.  At most four primary curves share a panel; the plain
control is repeated in every panel as a thin grey reference rather than counted
as a primary curve.

Curves are the **current path**: the resolution the most recently completed
training update actually used, in eval mode, with no BatchNorm recalibration.
Forced-bypass diagnostics live in their own figure and never overlay these.

Arms whose loss went non-finite are drawn nowhere: a diverged run is not a
measurement of its operator.  They are named in the console output and in
``resbench_results.json``.
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

from continuation.resolution_ops import LOCATIONS, O_REF, OPERATORS
from scripts.analyze_resbench import JOBS, OUTPUTS, PLAIN, finite
from scripts.resbench_manifest import PATH_OPERATORS, PATH_SET, SEEDS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "resbench_results.json"
OUT = ROOT / "results" / "presentation"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "legend.fontsize": 9, "figure.facecolor": "white"})

OP_FR = {"bilinear": "Bilinéaire antialiasé", "max": "Max adaptatif (référence)",
         "maxblur": "MaxBlur", "softpool": "SoftPool",
         "l2": "Reconstruction L²", "hminus1": "Reconstruction Ḣ⁻¹",
         "perceptual": "Perceptuel"}
OP_COLOR = {"bilinear": "#1f77b4", "max": "#d62728", "maxblur": "#ff7f0e",
            "softpool": "#9467bd", "l2": "#2ca02c", "hminus1": "#17becf",
            "perceptual": "#8c564b"}
LOC_FR = {"input": "Image d'entrée", "stem": "Après le stem",
          "D0": "Après le bloc 0", "D1": "Après le bloc 1",
          "D2": "Après le bloc 2"}
LOC_COLOR = {"input": "#1f77b4", "stem": "#ff7f0e", "D0": "#2ca02c",
             "D1": "#d62728", "D2": "#9467bd"}
PATH_FR = {"Rprog": "Rprog 16→24→32", "Rgentle": "Rgentle 24→32",
           "Rreverse": "Rreverse 24→16→32", "Rlate": "Rlate (plus long en réduit)",
           "fixed16": "16 fixe (inférence réduite)",
           "fixed24": "24 fixe (inférence réduite)"}
PATH_COLOR = {"Rprog": "#d62728", "Rgentle": "#1f77b4", "Rreverse": "#ff7f0e",
              "Rlate": "#2ca02c"}
PLAIN_FR = "Témoin — aucune réduction"
SD_NOTE = ("Bandes / barres = ± 1 écart-type d'échantillon sur 3 graines "
           "(descriptif, ce ne sont pas des intervalles de confiance).")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / ("%s.%s" % (name, ext)), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / (name + ".png"))


def load_curves():
    curves = {}
    for job in JOBS:
        for study in (OUTPUTS / job).glob("resbench_job*"):
            for cell in study.iterdir():
                mf, sf = cell / "metrics.json", cell / "summary.json"
                if not (mf.is_file() and sf.is_file()):
                    continue
                s = json.loads(sf.read_text(encoding="utf-8"))
                if not finite(s):
                    continue
                cid, seed = s["cell_id"].rsplit("__seed", 1)
                curves.setdefault(cid, {})[int(seed)] = json.loads(
                    mf.read_text(encoding="utf-8"))
    return curves


def series(curves, cid, field="test_acc_current"):
    per = curves.get(cid)
    if not per:
        return None
    tab = {s: {r["epoch"]: r.get(field) for r in m} for s, m in per.items()}
    ep = sorted(set.intersection(*[set(t) for t in tab.values()]))
    ep = [e for e in ep if all(t[e] is not None for t in tab.values())]
    mean = [st.mean([tab[s][e] for s in tab]) for e in ep]
    sd = [st.stdev([tab[s][e] for s in tab]) if len(tab) > 1 else 0.0 for e in ep]
    return ep, mean, sd, len(tab)


def draw(ax, curves, cid, color, label, lw=2.0, thin=False):
    r = series(curves, cid)
    if r is None:
        return
    ep, mean, sd, n = r
    y = np.array(mean) * 100
    s = np.array(sd) * 100
    if not thin:
        ax.fill_between(ep, y - s, y + s, color=color, alpha=0.15, lw=0)
    ax.plot(ep, y, "-", color=color, lw=lw,
            label=label + ("" if n == 3 else " (%d graines)" % n),
            zorder=2 if thin else 3)


def transitions(ax, xs=(6, 12)):
    for x in xs:
        ax.axvline(x, color="0.82", lw=0.8, ls="--", zorder=0)


def finish(ax, ylim=(38, 84)):
    ax.set_xlabel("époques complétées")
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)


# --------------------------------------------------------------------------

def fig_operators(res, curves):
    cfg = res["configurations"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), sharey=True)
    groups = [["max", "maxblur", "bilinear"], ["softpool", "l2", "hminus1",
                                               "perceptual"]]
    for ax, ops in zip(axes, groups):
        draw(ax, curves, PLAIN, "0.45", PLAIN_FR, lw=1.6, thin=True)
        for op in ops:
            cid = "%s__D1__Rprog" % op
            if cid in cfg and cfg[cid]["n_seeds"]:
                draw(ax, curves, cid, OP_COLOR[op], OP_FR[op])
        transitions(ax)
        finish(ax)
    axes[0].set_ylabel("précision de test (%)")
    axes[0].set_title("Opérateurs de type maximum / interpolation", fontsize=10.5)
    axes[1].set_title("Opérateurs pondérés et reconstructions", fontsize=10.5)
    fig.suptitle("Quel opérateur de réduction ? — réduction unique après le bloc 1, "
                 "chemin Rprog\n" + SD_NOTE, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "18_resbench_operateurs")

    # final accuracy, input vs D1
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    base = cfg[PLAIN]["acc_mean"] * 100
    ax.axvline(base, color="0.15", lw=1.5, ls="--")
    ax.text(base, -0.7, "  témoin %.2f %%" % base, fontsize=9, color="0.15")
    ops = [o for o in OPERATORS]
    for i, op in enumerate(ops):
        for loc, marker, off in (("input", "s", -0.16), ("D1", "o", 0.16)):
            c = cfg.get("%s__%s__Rprog" % (op, loc))
            if not c or not c["n_seeds"]:
                ax.annotate("perte non finie — exclu", (0.015, i + off),
                            xycoords=("axes fraction", "data"), fontsize=8,
                            color="#b22222", va="center", style="italic")
                continue
            m, s = c["acc_mean"] * 100, c["acc_sd"] * 100
            ax.plot([m - s, m + s], [i + off] * 2, color=OP_COLOR[op], lw=1.6,
                    alpha=0.8)
            ax.scatter([m], [i + off], color=OP_COLOR[op], marker=marker, s=85,
                       edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(range(len(ops)))
    ax.set_yticklabels([OP_FR[o] for o in ops])
    ax.invert_yaxis()
    ax.set_xlabel("précision de test finale (%)")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(handles=[Line2D([], [], color="0.3", marker="s", ls="",
                              label="sur l'image d'entrée"),
                       Line2D([], [], color="0.3", marker="o", ls="",
                              label="après le bloc 1 (D1)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2,
              frameon=False)
    ax.set_title("Opérateur × emplacement — précision finale, chemin Rprog\n"
                 "moyenne ± 1 écart-type sur 3 graines", fontsize=11)
    fig.tight_layout()
    save(fig, "19_resbench_operateur_x_emplacement")


def fig_depth(res, curves):
    cfg = res["configurations"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), sharey=True)
    for ax, locs in zip(axes, [["input", "stem", "D0"], ["D1", "D2"]]):
        draw(ax, curves, PLAIN, "0.45", PLAIN_FR, lw=1.6, thin=True)
        for loc in locs:
            cid = "%s__%s__Rprog" % (O_REF, loc)
            if cid in cfg and cfg[cid]["n_seeds"]:
                draw(ax, curves, cid, LOC_COLOR[loc], LOC_FR[loc])
        transitions(ax)
        finish(ax)
    axes[0].set_ylabel("précision de test (%)")
    axes[0].set_title("Réduction précoce", fontsize=10.5)
    axes[1].set_title("Réduction plus profonde", fontsize=10.5)
    fig.suptitle("Où insérer la réduction ? — Max adaptatif, chemin Rprog\n"
                 + SD_NOTE, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "20_resbench_profondeur")


def fig_paths(res, curves):
    cfg = res["configurations"]
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4), sharey=True)
    for ax, op in zip(axes, PATH_OPERATORS):
        draw(ax, curves, PLAIN, "0.45", PLAIN_FR, lw=1.6, thin=True)
        for path in PATH_SET:
            cid = "%s__D1__%s" % (op, path)
            if cid in cfg and cfg[cid]["n_seeds"]:
                draw(ax, curves, cid, PATH_COLOR[path], PATH_FR[path], lw=1.9)
        transitions(ax)
        finish(ax)
        ax.set_title(OP_FR[op], fontsize=10.5)
    axes[0].set_ylabel("précision de test (%)")
    fig.suptitle("Comment faire évoluer la résolution ? — réduction après le bloc 1\n"
                 "Rreverse est un contrôle d'ordre : même temps passé à chaque "
                 "résolution que Rprog, ordre inversé", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    save(fig, "21_resbench_chemins")


def fig_cost(res):
    cfg = res["configurations"]
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    base = cfg[PLAIN]
    ax.axhline(base["acc_mean"] * 100, color="0.15", lw=1.2, ls="--")
    ax.axvline(base["wall_mean"], color="0.15", lw=1.2, ls="--")
    for cid, c in cfg.items():
        if not c["n_seeds"]:
            continue
        fixed = c["path"].startswith("fixed")
        col = ("#000000" if cid == PLAIN
               else OP_COLOR.get(c["operator"], "0.5"))
        ax.scatter([c["wall_mean"]], [c["acc_mean"] * 100], color=col,
                   marker="D" if cid == PLAIN else ("^" if fixed else "o"),
                   s=95 if cid == PLAIN else 62, alpha=0.95 if fixed else 0.85,
                   edgecolor="white", linewidth=0.7, zorder=3)
    lead = sorted((c for c in cfg.values() if c["n_seeds"]),
                  key=lambda c: -c["acc_mean"])[:5]
    for c in lead:
        ax.annotate("%s / %s" % (OP_FR[c["operator"]].split(" (")[0],
                                 LOC_FR.get(c["location"], c["location"])),
                    (c["wall_mean"], c["acc_mean"] * 100), fontsize=7.5,
                    xytext=(6, -3), textcoords="offset points", color="0.25")
    ax.set_xlabel("temps mesuré par exécution (s), entraînement + évaluation")
    ax.set_ylabel("précision de test finale (%)")
    ax.grid(alpha=0.3)
    handles = [Line2D([], [], color=OP_COLOR[o], marker="o", ls="", label=OP_FR[o])
               for o in OPERATORS]
    handles += [Line2D([], [], color="k", marker="D", ls="", label=PLAIN_FR),
                Line2D([], [], color="0.3", marker="^", ls="",
                       label="résolution fixe (inférence réduite)")]
    ax.legend(handles=handles, fontsize=7.5, loc="lower right", frameon=False,
              ncol=2)
    ax.set_title("Précision en fonction du coût mesuré\n"
                 "les deux bras à résolution fixe terminent sur une inférence "
                 "réduite : ce n'est pas le même point de comparaison",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "22_resbench_precision_vs_cout")


def fig_bypass(res, curves):
    """Forced-bypass diagnostics, kept away from every primary figure."""
    cfg = res["configurations"]
    picks = [c for c in ["%s__D1__Rprog" % O_REF, "%s__input__Rprog" % O_REF,
                         "hminus1__D1__Rprog", "%s__D1__fixed16" % O_REF]
             if c in cfg and cfg[c]["n_seeds"]]
    fig, axes = plt.subplots(1, len(picks), figsize=(4.6 * len(picks), 4.8),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cid in zip(axes, picks):
        c = cfg[cid]
        for field, col, lab in (("test_acc_current", "#1f77b4",
                                 "Configuration courante"),
                                ("test_acc_target", "#d62728",
                                 "Forcé en 32×32, opérateur désactivé")):
            r = series(curves, cid, field)
            if r is None:
                continue
            ep, mean, sd, _ = r
            y = np.array(mean) * 100
            s = np.array(sd) * 100
            ax.fill_between(ep, y - s, y + s, color=col, alpha=0.13, lw=0)
            ax.plot(ep, y, color=col, lw=2.0, label=lab)
        transitions(ax)
        ax.set_title("%s / %s / %s" % (OP_FR[c["operator"]].split(" (")[0],
                                       LOC_FR.get(c["location"], c["location"]),
                                       PATH_FR.get(c["path"], c["path"])),
                     fontsize=9.5)
        finish(ax, ylim=(20, 84))
    axes[0].set_ylabel("précision de test (%)")
    fig.suptitle("Annexe — deux évaluations du MÊME état entraîné\n"
                 "l'écart mêle un changement de résolution d'entrée et les "
                 "statistiques de BatchNorm ; il ne mesure pas la qualité du "
                 "prédicteur courant", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    save(fig, "23_resbench_annexe_bypass")


def main():
    res = json.loads(RESULTS.read_text(encoding="utf-8"))
    curves = load_curves()
    print("configurations with curves: %d" % len(curves))
    if res.get("diverged_cells"):
        print("excluded (diverged):", json.dumps(res["diverged_cells"]))
    fig_operators(res, curves)
    fig_depth(res, curves)
    fig_paths(res, curves)
    fig_cost(res)
    fig_bypass(res, curves)


if __name__ == "__main__":
    main()
