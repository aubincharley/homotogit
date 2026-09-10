"""Figures for the anti-aliasing ablation batch, in the style of the campaign set.

Source: branch ``continuation-gaussian-tv_exploration_1``, commit ``efe24dc``
(Idriss El khamlichi, 10 September 2026).  Extract its data with::

    git archive efe24dc results/kaggle_outputs results/ablation_aa_results.json \
        | tar -x -C <ABL_ROOT>

**Why these arms are not drawn on the campaign's axes.**  The ablation batch was
trained from *different pinned initial weights* -- its own JSON says so -- and it
shows: the campaign's best configuration re-runs at 79.15 % here against 80.71 %
there, and the shared baseline at 75.06 % against 75.62 %.  Absolute levels are
therefore not comparable across batches.  What *is* comparable is each arm's
distance to **its own batch's baseline**, so that is the figure that puts both
batches on one axis.

One seed for every ablation arm, against three for the campaign: signs and
magnitudes only, no significance.  Twelve of the twenty-four arms have per-epoch
metrics committed; the R and D families are final-value only, and are omitted
from the trajectory figure rather than drawn from interpolated points.
"""
from __future__ import annotations

import json
import os
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
from scripts.presentation_labels import BASELINE, SD_NOTE, label_for

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "presentation"
ABL_ROOT = Path(os.environ.get(
    "ABL_ROOT",
    Path.home() / "AppData/Local/Temp/claude/C--Users-mnica-Documents-Projet-filiere"
                  "/1e7c0fdf-8f5a-466c-b47e-badfce660f23/scratchpad/ablation_data"))
ABL_JSON = ABL_ROOT / "results" / "ablation_aa_results.json"
SOURCE_COMMIT = "efe24dc"
ABL_BASE = "C_plain"

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "legend.fontsize": 9, "figure.facecolor": "white"})

# arm -> (French label, family, colour)
FAM = {
    "C":  ("Références du lot",              "#000000"),
    "P":  ("Emplacement du filtre",          "#1f77b4"),
    "M":  ("Sites filtrés",                  "#ff7f0e"),
    "K":  ("Sigma constant (architecture)",  "#9467bd"),
    "B":  ("BlurPool fixe",                  "#8c564b"),
    "R":  ("Résolution × emplacement",       "#2ca02c"),
    "D":  ("Réduction unique interne",       "#d62728"),
}
ARMS = {
    "C_plain":            ("Témoin du lot — aucun filtre", "C"),
    "C_plateau":          ("Gaussian par paliers (référence campagne, rejouée)", "C"),
    "P_postbn":           ("Filtre après BatchNorm — 19 sites", "P"),
    "P_postblock":        ("Filtre après ReLU — 10 positions", "P"),
    "M_predown":          ("Filtre aux 2 sites avant décimation", "M"),
    "M_nodown":           ("Filtre aux 17 autres sites", "M"),
    "K_const030":         ("Sigma constant 0,30", "K"),
    "K_const050":         ("Sigma constant 0,50", "K"),
    "K_const080":         ("Sigma constant 0,80", "K"),
    "K_const100":         ("Sigma constant 1,00", "K"),
    "B_blurpool":         ("BlurPool fixe (σ 0,5) avant les 2 décimations", "B"),
    "B_blurpool_plateau": ("BlurPool + Gaussian par paliers", "B"),
    "R1":                 ("Résolution progressive seule", "R"),
    "R2":                 ("Meilleure config. campagne, rejouée", "R"),
    "R3":                 ("Résolution progressive + filtre après ReLU", "R"),
    "R4":                 ("Réduction après le stem, sans filtre", "R"),
    "R5":                 ("Les deux interventions côté post-ReLU", "R"),
    "R6":                 ("Sigma constant, placement après ReLU", "R"),
    "D0":                 ("Réduction unique après bloc 0, sans flou", "D"),
    "D1":                 ("Réduction unique après bloc 1, sans flou", "D"),
    "D2":                 ("Réduction unique après bloc 2, sans flou", "D"),
    "D0G":                ("Réduction unique après bloc 0 + Gaussian", "D"),
    "D1G":                ("Réduction unique après bloc 1 + Gaussian", "D"),
    "D2G":                ("Réduction unique après bloc 2 + Gaussian", "D"),
}
TRAJ_PANELS = [
    ("Emplacement du filtre",
     ["C_plain", "C_plateau", "P_postbn", "P_postblock"]),
    ("Quels sites portent l'effet ?",
     ["C_plain", "C_plateau", "M_predown", "M_nodown"]),
    ("Sigma constant : filtre ou recuit ?",
     ["C_plain", "C_plateau", "K_const030", "K_const050", "K_const100"]),
    ("BlurPool fixe et redondance",
     ["C_plain", "C_plateau", "B_blurpool", "B_blurpool_plateau"]),
]
NOTE_BATCH = ("Lot d'ablation : 1 graine, poids initiaux différents de la "
              "campagne — les niveaux absolus ne sont pas comparables d'un lot "
              "à l'autre, seuls les écarts au témoin du même lot le sont.")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / ("%s.%s" % (name, ext)), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / (name + ".png"))


def load_ablation():
    d = json.loads(ABL_JSON.read_text(encoding="utf-8"))
    return d["arms"], d["comparisons"], d["caveats"]


def load_ablation_curves():
    """Per-epoch metrics for the arms whose raw outputs were committed."""
    curves = {}
    base = ABL_ROOT / "results" / "kaggle_outputs"
    if not base.is_dir():
        return curves
    for mf in base.glob("abl-j*/ablation_job*/*__seed0/metrics.json"):
        arm = mf.parent.name.replace("__seed0", "")
        curves[arm] = json.loads(mf.read_text(encoding="utf-8"))
    return curves


def campaign_finals():
    res = json.loads((ROOT / "results" / "campaign_results.json").read_text())
    return {cid: c for cid, c in res["configurations"].items()}


# --------------------------------------------------------------------------

def fig_ablation_final(arms):
    base = arms[ABL_BASE]["acc"] * 100
    order = sorted(arms, key=lambda k: arms[k]["acc"])
    fig, ax = plt.subplots(figsize=(12.5, 9.5))
    ax.axvline(base, color="0.15", lw=1.6, ls="--", zorder=1)
    ax.text(base, len(order) - 0.4, "  témoin du lot %.2f %%" % base,
            fontsize=9, color="0.15", va="top")
    for i, k in enumerate(order):
        v = arms[k]
        lab, fam = ARMS[k]
        col = "#000000" if k == ABL_BASE else FAM[fam][1]
        ax.scatter([v["acc"] * 100], [i], color=col, s=105, zorder=3,
                   marker="D" if k == ABL_BASE else "o",
                   edgecolor="white", linewidth=0.9)
        if v["primary_path"] == "current":
            ax.scatter([v["acc"] * 100], [i], facecolor="none", edgecolor=col,
                       s=280, lw=1.1, zorder=2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([ARMS[k][0] for k in order], fontsize=9)
    ax.set_xlabel("précision de test finale (%) — époque 30, 1 graine")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Ablation anticrénelage — 24 bras, précision finale\n"
                 "commit %s, lot séparé de la campagne" % SOURCE_COMMIT,
                 fontsize=11.5)
    handles = [Line2D([], [], color=c, marker="o", ls="", label=n)
               for n, c in FAM.values()]
    handles.append(Line2D([], [], color="0.35", marker="o", ls="", mfc="none",
                          ms=13, label="lu sur le chemin courant (filtre actif)"))
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5)
    fig.text(0.5, -0.02, NOTE_BATCH, ha="center", fontsize=8.5, color="0.35",
             wrap=True)
    fig.tight_layout()
    save(fig, "14_ablation_precision_finale")


def fig_delta_combined(arms, cfgs):
    """The one cross-batch-comparable view: distance to each batch's own baseline."""
    camp_base = cfgs[BASELINE]["acc_mean"]
    camp = []
    for cid, c in cfgs.items():
        if cid == BASELINE:
            continue
        per_seed = [100 * (a - camp_base) for a in c["acc_per_seed"]]
        camp.append((label_for(cid), 100 * (c["acc_mean"] - camp_base),
                     st.stdev(per_seed) if len(per_seed) > 1 else 0.0))
    abl_base = arms[ABL_BASE]["acc"]
    abl = [(ARMS[k][0], 100 * (v["acc"] - abl_base), None)
           for k, v in arms.items() if k != ABL_BASE]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 9.2))
    for ax, (data, title, col) in zip(axes, [
            (camp, "Campagne — 21 configurations, 3 graines", "#1f77b4"),
            (abl, "Ablation — 24 bras, 1 graine", "#d62728")]):
        data = sorted(data, key=lambda t: t[1])
        ax.axvline(0, color="0.2", lw=1.5, zorder=1)
        for i, (lab, d, sd) in enumerate(data):
            c = "#2ca02c" if d > 0 else "#d62728"
            if sd:
                ax.plot([d - sd, d + sd], [i, i], color=c, lw=1.6, alpha=0.75,
                        zorder=2)
            ax.scatter([d], [i], color=c, s=80, zorder=3, edgecolor="white",
                       linewidth=0.8)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels([t[0] for t in data], fontsize=8)
        ax.set_xlabel("écart au témoin du même lot (points de pourcentage)")
        ax.grid(axis="x", alpha=0.3)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlim(-5.5, 6.6)
    fig.suptitle("Écart au témoin — la seule lecture comparable entre les deux lots\n"
                 "à gauche barres = ± 1 écart-type sur 3 graines ; à droite une "
                 "seule graine, donc aucune barre", fontsize=11.5)
    fig.text(0.5, -0.015, NOTE_BATCH, ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "15_ecart_au_temoin_deux_lots")


def fig_ablation_curves(curves, arms):
    have = [p for p in TRAJ_PANELS if all(a in curves for a in p[1])]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.4), sharey=True)
    axes = axes.ravel()
    # One distinct colour per arm inside a panel: a family colour would make the
    # two blues (or three purples) of a panel indistinguishable.
    cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
    for ax, (title, keys) in zip(axes, have):
        n = 0
        for k in keys:
            m = curves[k]
            ep = [r["epoch"] for r in m]
            # Current path for every arm, as in the campaign figures: it is the
            # configuration actually trained under.  The target path during
            # continuation is the premature-ablation diagnostic and belongs in
            # the appendix, not here.
            y = [100 * r["test_acc_current"] for r in m]
            lab, _fam = ARMS[k]
            if k == ABL_BASE:
                ax.plot(ep, y, "-", color="#000000", lw=2.8, label=lab, zorder=4)
            elif k == "C_plateau":
                ax.plot(ep, y, "--", color="#000000", lw=1.8, label=lab, zorder=4)
            else:
                ax.plot(ep, y, "-", color=cycle[n % len(cycle)], lw=1.9,
                        label=lab, zorder=3)
                n += 1
        for x in (6, 12):
            ax.axvline(x, color="0.85", lw=0.7, ls="--", zorder=0)
        ax.axvline(21, color="0.55", lw=0.9, ls=":", zorder=0)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("époques complétées")
        ax.grid(alpha=0.25)
        ax.set_ylim(38, 84)
        ax.legend(loc="lower right", frameon=False, fontsize=8)
    axes[0].set_ylabel("précision de test (%)")
    axes[2].set_ylabel("précision de test (%)")
    fig.suptitle("Ablation anticrénelage — trajectoires (12 bras dont les métriques "
                 "par époque sont versionnées)\n"
                 "les bras à sigma constant sont lus sur le chemin courant : ce ne "
                 "sont pas des continuations", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "16_ablation_trajectoires")


def fig_hypothesis(arms):
    """The decisive contrast: is the Gaussian anti-aliasing, or a smoothness prior?"""
    base = arms[ABL_BASE]["acc"]
    groups = [
        ("Sites filtrés", [("2 sites avant décimation", "M_predown"),
                           ("17 autres sites", "M_nodown"),
                           ("les 19 sites (référence)", "C_plateau")]),
        ("Emplacement", [("après convolution (campagne)", "C_plateau"),
                         ("après BatchNorm", "P_postbn"),
                         ("après ReLU (correct pour l'anticrénelage)", "P_postblock")]),
        ("Filtre fixe vs recuit", [("sigma constant 0,50", "K_const050"),
                                   ("sigma constant 1,00", "K_const100"),
                                   ("paliers recuits", "C_plateau")]),
        ("BlurPool", [("BlurPool seul", "B_blurpool"),
                      ("BlurPool + paliers", "B_blurpool_plateau"),
                      ("paliers seuls", "C_plateau")]),
    ]
    # NOT sharey: each panel has its own three categories, and a shared
    # y axis would silently paint the last panel's labels on all four.
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.8))
    for ax, (title, items) in zip(axes, groups):
        labs = [t[0] for t in items]
        vals = [100 * (arms[k]["acc"] - base) for _l, k in items]
        cols = ["#2ca02c" if v > 0 else "#d62728" for v in vals]
        ax.barh(range(len(vals)), vals, color=cols, alpha=0.85, height=0.6)
        for i, v in enumerate(vals):
            ax.text(v + (0.08 if v >= 0 else -0.08), i, "%+.2f" % v,
                    va="center", ha="left" if v >= 0 else "right", fontsize=9)
        ax.set_yticks(range(len(labs)))
        ax.set_yticklabels(labs, fontsize=8.5)
        ax.axvline(0, color="0.2", lw=1.3)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("écart au témoin (pts)")
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlim(-6.8, 6.4)
    fig.suptitle("Le filtre interne agit-il comme un anticrénelage ?\n"
                 "sous l'hypothèse anticrénelage, les 2 sites avant décimation "
                 "devraient porter l'essentiel de l'effet — ils en portent le moins",
                 fontsize=11.5)
    fig.text(0.5, -0.04, NOTE_BATCH, ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, "17_hypothese_anticrenelage")


def main():
    if not ABL_JSON.is_file():
        raise SystemExit("ablation data not found at %s — see the module "
                         "docstring for the git archive command" % ABL_ROOT)
    arms, comparisons, caveats = load_ablation()
    curves = load_ablation_curves()
    cfgs = campaign_finals()
    print("ablation arms: %d, with trajectories: %d, campaign configs: %d"
          % (len(arms), len(curves), len(cfgs)))
    missing = [k for k in arms if k not in ARMS]
    if missing:
        raise SystemExit("unlabelled arms: %s" % missing)

    fig_ablation_final(arms)
    fig_delta_combined(arms, cfgs)
    fig_ablation_curves(curves, arms)
    fig_hypothesis(arms)

    summary = {
        "source_commit": SOURCE_COMMIT,
        "source_branch": "continuation-gaussian-tv_exploration_1",
        "n_arms": len(arms), "n_with_trajectories": len(curves),
        "seeds": 1, "batch_baseline_acc": arms[ABL_BASE]["acc"],
        "campaign_baseline_acc": cfgs[BASELINE]["acc_mean"],
        "caveats": caveats,
        "delta_vs_own_baseline_pp": {
            k: round(100 * (v["acc"] - arms[ABL_BASE]["acc"]), 2)
            for k, v in sorted(arms.items(), key=lambda kv: -kv[1]["acc"])},
    }
    (OUT / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT / "ablation_summary.json")


if __name__ == "__main__":
    main()
