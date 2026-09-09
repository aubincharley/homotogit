"""Rebuild the CIFAR-10 campaign presentation from the 63 completed runs.

Reads the raw per-epoch metrics for all three seeds, cross-checks the final
aggregates against ``results/campaign_results.json``, and writes readable French
figures (PNG + PDF + SVG) plus the configuration mapping table into
``results/presentation/``.  No training, no new model evaluation: every number
comes from files already on disk, and the original results are left untouched.

Main figures use CURRENT-PATH metrics only -- the configuration actually used by
the most recently completed training update.  Bypass ("target path") evaluations
are confined to the diagnostic appendix.
"""
from __future__ import annotations

import csv
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

from scripts.campaign_manifest import build_configs
from scripts.presentation_labels import (BASELINE, FAMILY_FR, GAUSS_COLOR,
                                         GAUSS_FR, MAIN, MAIN_COLOR, MAIN_LABEL,
                                         RED_COLOR, RED_FR, RES_COLOR, RES_FR,
                                         SCHEDULE_NOTE, SD_NOTE, label_for,
                                         mapping_rows)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "campaign_results.json"
OUT = ROOT / "results" / "presentation"
SEEDS = (0, 1, 2)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                     "axes.labelsize": 10, "legend.fontsize": 9,
                     "figure.facecolor": "white"})


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load():
    res = json.loads(RESULTS.read_text())
    curves, finals = {}, {}
    for job in res["jobs"]:
        study = ROOT / job["study"]
        for d in sorted(p for p in study.iterdir() if p.is_dir()):
            mf, sf = d / "metrics.json", d / "summary.json"
            if not (mf.is_file() and sf.is_file()):
                continue
            s = json.loads(sf.read_text())
            cid, seed = s["cell_id"].rsplit("__seed", 1)
            curves[(cid, int(seed))] = json.loads(mf.read_text())
            finals[(cid, int(seed))] = s
    return res, curves, finals


def series(curves, cid, field):
    """Mean and sample SD across seeds on the actual recorded epochs.

    Epoch coordinates are taken as recorded; nothing is smoothed, shifted or
    interpolated.  A checkpoint is plotted only where all three seeds report it,
    and the coverage is returned so gaps are visible rather than silent.
    """
    per_seed = {}
    for s in SEEDS:
        m = curves.get((cid, s))
        if m is None:
            continue
        per_seed[s] = {r["epoch"]: r.get(field) for r in m}
    if not per_seed:
        return [], [], [], {}
    epochs = sorted(set.intersection(*[set(v) for v in per_seed.values()]))
    epochs = [e for e in epochs
              if all(per_seed[s].get(e) is not None for s in per_seed)]
    mean = [st.mean([per_seed[s][e] for s in per_seed]) for e in epochs]
    sd = [st.stdev([per_seed[s][e] for s in per_seed]) if len(per_seed) > 1 else 0.0
          for e in epochs]
    coverage = {e: len(per_seed) for e in epochs}
    return epochs, mean, sd, coverage


def final_stats(finals, cid):
    accs = [finals[(cid, s)]["final_test_acc"] for s in SEEDS if (cid, s) in finals]
    ces = [finals[(cid, s)]["final_test_ce"] for s in SEEDS if (cid, s) in finals]
    walls = [finals[(cid, s)]["wall_seconds"] for s in SEEDS if (cid, s) in finals]
    trains = [finals[(cid, s)]["train_seconds"] for s in SEEDS if (cid, s) in finals]
    pce = [finals[(cid, s)]["final_train_probe_ce"] for s in SEEDS
           if (cid, s) in finals]
    return {"acc": accs, "acc_mean": st.mean(accs),
            "acc_sd": st.stdev(accs) if len(accs) > 1 else 0.0,
            "ce_mean": st.mean(ces), "ce_sd": st.stdev(ces) if len(ces) > 1 else 0.0,
            "probe_ce_mean": st.mean(pce),
            "wall_mean": st.mean(walls), "train_mean": st.mean(trains),
            "n_seeds": len(accs)}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / ("%s.%s" % (name, ext)), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / (name + ".png"))


def band(ax, cid, curves, field, color, label, lw=2.0, scale=1.0, ls="-"):
    e, m, sd, _cov = series(curves, cid, field)
    if not e:
        return
    m = np.array(m) * scale
    sd = np.array(sd) * scale
    ax.fill_between(e, m - sd, m + sd, color=color, alpha=0.16, lw=0)
    ax.plot(e, m, ls, color=color, lw=lw, label=label, zorder=3)


def transitions(ax, ymin=None):
    for x in (6, 12):
        ax.axvline(x, color="0.75", lw=0.8, ls="--", zorder=0)
    ax.axvline(21, color="0.55", lw=0.9, ls=":", zorder=0)


# --------------------------------------------------------------------------
# 1. main overview
# --------------------------------------------------------------------------

def fig_overview(curves, finals):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for cid, lab, col, lw, ls in MAIN:
        band(axes[0], cid, curves, "test_acc_current", col, lab, lw=lw, scale=100)
        band(axes[1], cid, curves, "test_acc_current", col, lab, lw=lw, scale=100)
    axes[0].set_title("Précision de test — trajectoire complète")
    axes[0].set_ylim(8, 85)
    axes[1].set_title("Précision de test — phase finale (zoom)")
    axes[1].set_xlim(12, 30)
    axes[1].set_ylim(68, 82)
    for ax in axes:
        transitions(ax)
        ax.set_xlabel("époques complétées")
        ax.set_ylabel("précision de test (%)")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="lower right", frameon=False)
    fig.suptitle("Vue d'ensemble — chemin courant, moyenne sur 3 graines\n"
                 + SCHEDULE_NOTE, fontsize=11)
    fig.text(0.5, -0.03, SD_NOTE, ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "01_vue_ensemble_precision")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    for cid, lab, col, lw, ls in MAIN:
        band(axes[0], cid, curves, "test_ce_current", col, lab, lw=lw)
        band(axes[1], cid, curves, "train_probe_ce_current", col, lab, lw=lw)
    axes[0].set_title("Entropie croisée de test")
    axes[0].set_ylabel("entropie croisée (test)")
    axes[0].set_ylim(0.5, 2.4)
    axes[1].set_title("Entropie croisée sur la sonde d'entraînement (échelle log)")
    axes[1].set_ylabel("entropie croisée (sonde d'entraînement)")
    axes[1].set_yscale("log")
    for ax in axes:
        transitions(ax)
        ax.set_xlabel("époques complétées")
        ax.grid(alpha=0.25, which="both")
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle("Vue d'ensemble — pertes, chemin courant, moyenne sur 3 graines",
                 fontsize=11)
    fig.text(0.5, -0.03, SD_NOTE, ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "02_vue_ensemble_pertes")


# --------------------------------------------------------------------------
# 2. final results overview
# --------------------------------------------------------------------------

def fig_final_overview(finals, configs):
    stats = {cid: final_stats(finals, cid) for cid in configs}
    order = sorted(stats, key=lambda c: stats[c]["acc_mean"])
    base = stats[BASELINE]["acc_mean"] * 100

    fig, ax = plt.subplots(figsize=(13, 11))
    ax.axvline(base, color="0.15", lw=1.6, ls="--", zorder=1)
    ax.text(base, len(order) - 0.3, "  témoin %.2f %%" % base, fontsize=9,
            color="0.15", va="top")
    for i, cid in enumerate(order):
        s = stats[cid]
        grp = configs[cid]["group"]
        col = "#000000" if cid == BASELINE else GAUSS_COLOR[configs[cid]["gaussian"]]
        ax.scatter([a * 100 for a in s["acc"]], [i] * len(s["acc"]),
                   color=col, alpha=0.40, s=26, zorder=2)
        ax.plot([s["acc_mean"] * 100 - s["acc_sd"] * 100,
                 s["acc_mean"] * 100 + s["acc_sd"] * 100], [i, i],
                color=col, lw=1.8, alpha=0.85, zorder=2)
        ax.scatter([s["acc_mean"] * 100], [i], color=col, s=110, zorder=4,
                   marker="o" if cid != BASELINE else "D",
                   edgecolor="white", linewidth=1.0)
        ax.text(1.012, i, FAMILY_FR[grp].split(" — ")[0], fontsize=7.5,
                color="0.45", va="center", transform=ax.get_yaxis_transform(),
                clip_on=False)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label_for(c) for c in order], fontsize=9)
    ax.set_xlim(74.5, 82.4)
    ax.set_xlabel("précision de test finale (%) — époque 30, chemin commun 32×32 sans filtre")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Résultats finaux des 21 configurations\n"
                 "ordre par précision moyenne (descriptif) ; "
                 "3 graines individuelles + moyenne", fontsize=11.5)
    handles = [Line2D([], [], color=v, marker="o", ls="", label=GAUSS_FR[k])
               for k, v in GAUSS_COLOR.items()]
    handles += [Line2D([], [], color="k", marker="D", ls="", label="témoin"),
                Line2D([], [], color="0.3", lw=1.8, label="± 1 écart-type (3 graines)"),
                Line2D([], [], color="0.3", marker="o", ls="", alpha=0.4,
                       label="graines individuelles")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06),
              ncol=4, frameon=False, fontsize=8.5)
    fig.text(0.5, -0.085, SD_NOTE + "  Référence : témoin en tirets.",
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout()
    save(fig, "03_resultats_finaux")
    return stats


# --------------------------------------------------------------------------
# 3. focused questions
# --------------------------------------------------------------------------

def fig_q1(curves, finals):
    RES = ["R32", "Rprog", "Rgentle"]
    GS = ["Gnone", "Gplateau", "Ggeo"]
    grid = np.zeros((3, 3))
    sdg = np.zeros((3, 3))
    for i, r in enumerate(RES):
        for j, g in enumerate(GS):
            s = final_stats(finals, "%s__%s__input_bilinear__all19" % (r, g))
            grid[i, j], sdg[i, j] = s["acc_mean"] * 100, s["acc_sd"] * 100

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    im = ax.imshow(grid, cmap="viridis")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, "%.2f %%\n± %.2f" % (grid[i, j], sdg[i, j]),
                    ha="center", va="center", fontsize=10.5,
                    color="white" if grid[i, j] < grid.max() - 1.5 else "black")
    ax.set_xticks(range(3))
    ax.set_xticklabels([GAUSS_FR[g] for g in GS], fontsize=9)
    ax.set_yticks(range(3))
    ax.set_yticklabels([RES_FR[r] for r in RES], fontsize=9)
    ax.set_title("Que changent la résolution et le calendrier gaussien ?\n"
                 "précision finale (%), moyenne ± écart-type sur 3 graines\n"
                 "case en haut à gauche = témoin", fontsize=11)
    fig.colorbar(im, ax=ax, label="précision finale (%)")
    fig.tight_layout()
    save(fig, "04_q1_grille_resolution_gaussian")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), sharey=True)
    for ax, r in zip(axes, RES):
        for g in GS:
            cid = "%s__%s__input_bilinear__all19" % (r, g)
            is_base = cid == BASELINE
            band(ax, cid, curves, "test_acc_current",
                 "#000000" if is_base else GAUSS_COLOR[g],
                 ("Témoin — sans filtre" if is_base else GAUSS_FR[g]),
                 lw=3.0 if is_base else 2.0, scale=100)
        ax.set_title(RES_FR[r], fontsize=10.5)
        ax.set_xlabel("époques complétées")
        ax.set_xlim(0, 30)
        ax.set_ylim(20, 84)
        ax.grid(alpha=0.25)
        transitions(ax)
        ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    axes[0].set_ylabel("précision de test (%)")
    fig.suptitle("Que changent la résolution et le calendrier gaussien ? — "
                 "trajectoires, un panneau par résolution\n" + SCHEDULE_NOTE,
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "05_q1_trajectoires_par_resolution")


def fig_q2(curves):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, r in zip(axes, ("R32", "Rprog")):
        for g in ("Gnone", "Gplateau", "Gmix"):
            cid = "%s__%s__input_bilinear__all19" % (r, g)
            is_base = cid == BASELINE
            band(ax, cid, curves, "test_acc_current",
                 "#000000" if is_base else GAUSS_COLOR[g],
                 ("Témoin — sans filtre" if is_base else GAUSS_FR[g]),
                 lw=3.0 if is_base else 2.0, scale=100)
        ax.set_title(RES_FR[r], fontsize=10.5)
        ax.set_xlabel("époques complétées")
        ax.set_ylim(20, 84)
        ax.grid(alpha=0.25)
        transitions(ax)
        ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    axes[0].set_ylabel("précision de test (%)")
    fig.suptitle("Comment réintroduire les détails ?\n"
                 "à chaque résolution : sans filtre, paliers de sigma, "
                 "mélange identité–Gaussian", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "06_q2_reintroduire_details")


def fig_q3(curves):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, r in zip(axes, ("R32", "Rprog")):
        specs = [("%s__Gnone__input_bilinear__all19" % r, "sans filtre", "#7a7a7a"),
                 ("%s__Gplateau__input_bilinear__all19" % r,
                  "Gaussian par paliers — 19 insertions", "#1f77b4"),
                 ("%s__Gplateau__input_bilinear__early7" % r,
                  "Gaussian — 7 insertions", "#ff7f0e")]
        for cid, lab, col in specs:
            is_base = cid == BASELINE
            band(ax, cid, curves, "test_acc_current",
                 "#000000" if is_base else col,
                 ("Témoin — sans filtre" if is_base else lab),
                 lw=3.0 if is_base else 2.0, scale=100)
        ax.set_title(RES_FR[r], fontsize=10.5)
        ax.set_xlabel("époques complétées")
        ax.set_ylim(20, 84)
        ax.grid(alpha=0.25)
        transitions(ax)
        ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    axes[0].set_ylabel("précision de test (%)")
    fig.suptitle("Faut-il filtrer toutes les couches ?\n"
                 "7 insertions = tronc + étage 1 ; 19 insertions = toutes les "
                 "convolutions 3×3 du chemin principal", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "07_q3_nombre_insertions")


def fig_q4(curves, finals):
    REDS = ["input_bilinear", "input_max", "stem_bilinear", "stem_max"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, g in zip(axes, ("Gnone", "Gplateau")):
        for red in REDS:
            cid = "Rprog__%s__%s__all19" % (g, red)
            band(ax, cid, curves, "test_acc_current", RED_COLOR[red],
                 RED_FR[red], lw=2.0, scale=100)
        ax.set_title("Résolution 16→24→32 — %s" % GAUSS_FR[g], fontsize=10.5)
        ax.set_xlabel("époques complétées")
        ax.set_ylim(20, 84)
        ax.grid(alpha=0.25)
        transitions(ax)
        ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    axes[0].set_ylabel("précision de test (%)")
    fig.suptitle("Où et comment réduire la résolution ?\n"
                 "quatre chemins de réduction, à résolution progressive ; "
                 "« bilinéaire sur l'image » est la référence commune", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "08_q4_reduction_trajectoires")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, g in zip(axes, ("Gnone", "Gplateau")):
        grid = np.zeros((2, 2))
        sdg = np.zeros((2, 2))
        for i, loc in enumerate(("input", "stem")):
            for j, op in enumerate(("bilinear", "max")):
                s = final_stats(finals, "Rprog__%s__%s_%s__all19" % (g, loc, op))
                grid[i, j], sdg[i, j] = s["acc_mean"] * 100, s["acc_sd"] * 100
        im = ax.imshow(grid, cmap="viridis")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, "%.2f %%\n± %.2f" % (grid[i, j], sdg[i, j]),
                        ha="center", va="center", fontsize=11,
                        color="white" if grid[i, j] < grid.max() - 0.6 else "black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["bilinéaire", "max"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["sur l'image", "après la 1re convolution"], fontsize=9)
        ax.set_title(GAUSS_FR[g], fontsize=10.5)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Où et comment réduire la résolution ? — précision finale (%)\n"
                 "opérateur (colonnes) × emplacement (lignes), résolution 16→24→32",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, "09_q4_reduction_tableau")


def fig_q5(curves):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, g in zip(axes, ("Gnone", "Gplateau")):
        for res in ("R32", "Rprog", "Rreverse"):
            cid = "%s__%s__input_bilinear__all19" % (res, g)
            is_base = cid == BASELINE
            band(ax, cid, curves, "test_acc_current",
                 "#000000" if is_base else RES_COLOR[res],
                 ("Témoin — 32 constant, sans filtre" if is_base
                  else RES_FR[res]),
                 lw=3.0 if is_base else 2.0, scale=100)
        ax.set_title(GAUSS_FR[g], fontsize=10.5)
        ax.set_xlabel("époques complétées")
        ax.set_ylim(20, 84)
        ax.grid(alpha=0.25)
        transitions(ax)
        ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    axes[0].set_ylabel("précision de test (%)")
    fig.suptitle("L'ordre des résolutions compte-t-il ?\n"
                 "16→24→32 contre 24→16→32 (même temps passé à chaque "
                 "résolution), avec le témoin à résolution constante\n"
                 "avec filtrage, la comparaison change aussi la trajectoire "
                 "conjointe résolution/sigma : sigma est mis à l'échelle par la "
                 "résolution", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "10_q5_ordre_resolutions")


# --------------------------------------------------------------------------
# 4. paired differences and cost
# --------------------------------------------------------------------------

PAIRS = [
    ("Résolution seule vs témoin",
     "Rprog__Gnone__input_bilinear__all19", BASELINE),
    ("Gaussian seul vs témoin",
     "R32__Gplateau__input_bilinear__all19", BASELINE),
    ("Ajout du Gaussian à la résolution progressive",
     "Rprog__Gplateau__input_bilinear__all19", "Rprog__Gnone__input_bilinear__all19"),
    ("Max après 1re conv. vs bilinéaire image (sans filtre)",
     "Rprog__Gnone__stem_max__all19", "Rprog__Gnone__input_bilinear__all19"),
    ("Mélange identité–Gaussian vs paliers (rés. progressive)",
     "Rprog__Gmix__input_bilinear__all19", "Rprog__Gplateau__input_bilinear__all19"),
    ("7 insertions vs 19 (rés. progressive)",
     "Rprog__Gplateau__input_bilinear__early7",
     "Rprog__Gplateau__input_bilinear__all19"),
    ("Ordre inversé vs 16→24→32 (sans filtre)",
     "Rreverse__Gnone__input_bilinear__all19", "Rprog__Gnone__input_bilinear__all19"),
    ("Ordre inversé vs 16→24→32 (Gaussian)",
     "Rreverse__Gplateau__input_bilinear__all19",
     "Rprog__Gplateau__input_bilinear__all19"),
]


def fig_paired(finals):
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.axvline(0, color="0.25", lw=1.4, zorder=1)
    rows = []
    for i, (lab, a, b) in enumerate(PAIRS):
        d = [100 * (finals[(a, s)]["final_test_acc"]
                    - finals[(b, s)]["final_test_acc"]) for s in SEEDS]
        mean = st.mean(d)
        col = "#2ca02c" if mean > 0 else "#d62728"
        ax.scatter(d, [i] * 3, color=col, alpha=0.45, s=34, zorder=2)
        ax.scatter([mean], [i], color=col, s=130, marker="|", linewidth=3, zorder=3)
        ax.plot([min(d), max(d)], [i, i], color=col, lw=1.2, alpha=0.5, zorder=2)
        rows.append({"comparaison": lab, "diff_par_graine_pp": [round(x, 2) for x in d],
                     "moyenne_pp": round(mean, 2),
                     "etendue_pp": round(max(d) - min(d), 2),
                     "meme_signe": all(x > 0 for x in d) or all(x < 0 for x in d)})
    ax.set_yticks(range(len(PAIRS)))
    ax.set_yticklabels([p[0] for p in PAIRS], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("différence de précision finale (points de pourcentage)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Effets appariés — différences calculées graine par graine\n"
                 "3 points = 3 graines appariées ; barre verticale = moyenne",
                 fontsize=11)
    fig.text(0.5, -0.02,
             "Les différences sont appariées au sein de chaque graine ; "
             "l'étendue montrée est celle des différences elles-mêmes, "
             "jamais déduite des écarts-types marginaux.",
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout()
    save(fig, "11_effets_apparies")
    return rows


def fig_cost(finals, stats, configs):
    fig, ax = plt.subplots(figsize=(11, 6.6))
    highlight = {c for c, _l, _col, _w, _s in MAIN}
    highlight |= {"Rprog__Gplateau__input_max__all19",
                  "Rprog__Gmix__input_bilinear__all19",
                  "Rprog__Gplateau__input_bilinear__early7"}
    for cid, s in stats.items():
        big = cid in highlight
        col = MAIN_COLOR.get(cid, GAUSS_COLOR[configs[cid]["gaussian"]])
        ax.scatter([s["wall_mean"]], [s["acc_mean"] * 100],
                   color=col if big else "0.72", s=110 if big else 34,
                   zorder=3 if big else 2, edgecolor="white" if big else "none",
                   linewidth=0.9)
        if big:
            ax.annotate(MAIN_LABEL.get(cid, label_for(cid)),
                        (s["wall_mean"], s["acc_mean"] * 100), fontsize=8,
                        xytext=(7, -3), textcoords="offset points", color="0.2")
    ax.axhline(stats[BASELINE]["acc_mean"] * 100, color="0.15", lw=1.2, ls="--")
    ax.set_xlabel("temps mesuré par exécution (s) — entraînement + évaluation, "
                  "moyenne sur 3 graines")
    ax.set_ylabel("précision de test finale (%)")
    ax.grid(alpha=0.3)
    ax.set_title("Précision en fonction du coût mesuré\n"
                 "toutes les durées proviennent de cette campagne sur Tesla T4 ; "
                 "compromis montré directement, sans score composite", fontsize=11)
    fig.text(0.5, -0.02, "Les écarts de temps sont descriptifs (une exécution par "
                         "graine, machines partagées).", ha="center",
             fontsize=8.5, color="0.35")
    fig.tight_layout()
    save(fig, "12_precision_vs_cout")


# --------------------------------------------------------------------------
# 5. diagnostic appendix
# --------------------------------------------------------------------------

def fig_appendix(curves, configs):
    cids = [c for c, _l, _col, _w, _s in MAIN] + [
        "Rprog__Gmix__input_bilinear__all19",
        "Rprog__Gplateau__stem_max__all19"]
    n = len(cids)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.4 * rows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    checks = []
    for ax, cid in zip(axes, cids):
        band(ax, cid, curves, "test_acc_current", "#1f77b4",
             "Configuration courante", lw=2.2, scale=100)
        band(ax, cid, curves, "test_acc_target", "#d62728",
             "Évaluation forcée en 32×32,\nfiltres et réduction interne désactivés",
             lw=1.6, scale=100, ls="-")
        e, mc, _s, _c = series(curves, cid, "test_acc_current")
        _e, mt, _s2, _c2 = series(curves, cid, "test_acc_target")
        agree = abs(mc[-1] - mt[-1]) if mc and mt else None
        checks.append({"configuration": label_for(cid), "id": cid,
                       "ecart_final_pp": None if agree is None else round(100 * agree, 6)})
        ax.set_title(label_for(cid), fontsize=9.5)
        ax.set_xlabel("époques complétées")
        ax.grid(alpha=0.25)
        transitions(ax)
        ax.set_ylim(8, 85)
        ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].set_ylabel("précision de test (%)")
    fig.suptitle("Annexe diagnostique — deux évaluations du MÊME état entraîné\n"
                 "ce ne sont pas deux modèles entraînés séparément ; l'écart "
                 "combine un changement de fonction et, souvent, un changement de "
                 "résolution d'entrée, pas seulement les statistiques de "
                 "BatchNorm", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "13_annexe_diagnostique")
    return checks


# --------------------------------------------------------------------------

def main():
    res, curves, finals = load()
    configs = {c["id"]: c for c in build_configs()}
    OUT.mkdir(parents=True, exist_ok=True)

    # cross-check against the recorded aggregates
    mismatches = []
    for cid in configs:
        s = final_stats(finals, cid)
        ref = res["configurations"][cid]
        if abs(s["acc_mean"] - ref["acc_mean"]) > 1e-9 or s["n_seeds"] != 3:
            mismatches.append({"id": cid, "recomputed": s["acc_mean"],
                               "recorded": ref["acc_mean"], "n_seeds": s["n_seeds"]})
    print("cross-check vs campaign_results.json: %d mismatch(es)" % len(mismatches))

    # seed coverage at every plotted checkpoint
    coverage_issues = []
    for cid in configs:
        e, _m, _sd, cov = series(curves, cid, "test_acc_current")
        if not e or any(v != 3 for v in cov.values()):
            coverage_issues.append({"id": cid, "epochs": len(e)})
    print("checkpoints with fewer than 3 seeds: %d configuration(s)"
          % len(coverage_issues))

    fig_overview(curves, finals)
    stats = fig_final_overview(finals, configs)
    fig_q1(curves, finals)
    fig_q2(curves)
    fig_q3(curves)
    fig_q4(curves, finals)
    fig_q5(curves)
    paired_rows = fig_paired(finals)
    fig_cost(finals, stats, configs)
    appendix = fig_appendix(curves, configs)

    # mapping table
    rows = mapping_rows(configs)
    with (OUT / "mapping_configurations.csv").open("w", newline="",
                                                   encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = ["| id de configuration | famille | nom lisible | résolution | filtre | détail | réduction | insertions |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| `%s` | %s | %s | %s | %s | %s | %s | %s |" % tuple(r.values()))
    (OUT / "mapping_configurations.md").write_text("\n".join(md), encoding="utf-8")

    # companion results table
    table = []
    for cid in sorted(stats, key=lambda c: -stats[c]["acc_mean"]):
        s = stats[cid]
        table.append({"nom": label_for(cid), "id": cid,
                      "famille": FAMILY_FR[configs[cid]["group"]],
                      "precision_finale_pct": round(100 * s["acc_mean"], 2),
                      "ecart_type_pct": round(100 * s["acc_sd"], 2),
                      "precision_par_graine_pct": [round(100 * a, 2) for a in s["acc"]],
                      "ce_test": round(s["ce_mean"], 4),
                      "ce_sonde_entrainement": round(s["probe_ce_mean"], 4),
                      "temps_s": round(s["wall_mean"]),
                      "temps_entrainement_s": round(s["train_mean"])})
    with (OUT / "tableau_resultats.csv").open("w", newline="",
                                              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    (OUT / "controles.json").write_text(json.dumps(
        {"cross_check_mismatches": mismatches,
         "coverage_issues": coverage_issues,
         "paired_differences": paired_rows,
         "appendix_final_agreement_pp": appendix,
         "n_runs": len(finals), "seeds": list(SEEDS)}, indent=2,
        ensure_ascii=False), encoding="utf-8")
    print("runs used: %d" % len(finals))
    for a in appendix:
        print("  appendix final agreement %-46s %s pp"
              % (a["id"][:46], a["ecart_final_pp"]))


if __name__ == "__main__":
    main()
