"""Figures for the grid: the contrasts, the dissociation, and the missing linear regime.

    py tools/plot_grid.py [--report results/grid_report.json]

One figure per question rather than one per quantity, so a panel answers
something.  Reference cells carry seed error bars; recipe variants are drawn
without, since they have one seed each and a bar would imply a precision they do
not have.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from continuation_core.analysis.power import pearson  # noqa: E402

LABEL = {"plain": "temoin",
         "resolution_max_b1": "resolution",
         "gaussian_postrelu": "flou",
         "resolution_max_b1_gaussian_conv": "resolution + flou"}
COLOR = {"plain": "#c0392b", "resolution_max_b1": "#16a085",
         "gaussian_postrelu": "#8e44ad",
         "resolution_max_b1_gaussian_conv": "#d68910"}
ORDER = list(LABEL)


def load(path):
    return json.load(open(path))


def _ref(conds):
    return {c["method"]: c for c in conds if c["variant"] == "reference"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default=str(ROOT / "results/grid_report.json"))
    ap.add_argument("--out", default=str(ROOT / "results/grid_figures.png"))
    args = ap.parse_args()
    rep = load(args.report)
    conds, ref = rep["conditions"], _ref(rep["conditions"])

    fig, ax = plt.subplots(2, 4, figsize=(25.5, 10.4))
    fig.suptitle("Ce qu'un curriculum laisse sur la solution -- %d cellules, %d conditions"
                 % (rep["n_cells"], len(conds)), fontsize=13, y=0.98)

    # A -- flatness contrast
    a = ax[0, 0]
    for i, m in enumerate(ORDER):
        if m not in ref:
            continue
        c = ref[m]
        a.errorbar([i], [c["trace_H"]], yerr=[c.get("trace_H__sd") or 0.0],
                   fmt="o", ms=10, capsize=4, color=COLOR[m])
    a.set_xticks(range(len(ORDER)))
    a.set_xticklabels([LABEL[m] for m in ORDER], fontsize=9, rotation=12)
    a.set_ylabel("tr H")
    a.set_title("A. courbure totale a la solution\nbarres = dispersion de graine",
                fontsize=10.5, loc="left")
    a.grid(alpha=.3, axis="y")

    # A bis -- where the curvature sits
    a = ax[0, 1]
    blocks = [k[len("block_"):] for k in ref["plain"] if k.startswith("block_")
              and not k.endswith("__sd")]
    blocks = ["stem"] + sorted([b for b in blocks if b.startswith("block")],
                               key=lambda s: int(s[5:])) + ["fc"]
    blocks = [b for b in blocks if "block_" + b in ref["plain"]]
    for m in ORDER:
        if m not in ref:
            continue
        a.plot(range(len(blocks)), [ref[m]["block_" + b] for b in blocks], "o-",
               color=COLOR[m], label=LABEL[m], lw=2.0 if m == "plain" else 1.5)
    a.set_xticks(range(len(blocks)))
    a.set_xticklabels([b.replace("block", "b") for b in blocks], fontsize=8, rotation=45)
    a.set_ylabel("trace portee par le bloc")
    a.set_title("A bis. ou la courbure disparait", fontsize=10.5, loc="left")
    a.legend(fontsize=8)
    a.grid(alpha=.3)

    # C -- the dissociation
    a = ax[0, 2]
    for m in ORDER:
        if m not in ref:
            continue
        a.plot([ref[m]["mean_radius"]], [ref[m]["test_error"]], "o", ms=13,
               color=COLOR[m], label=LABEL[m])
    a.set_xlabel("rayon frequentiel moyen du Jacobien")
    a.set_ylabel("erreur de test")
    a.set_title("C. la dissociation\nle flou deplace le spectre, la resolution non --\n"
                "les deux generalisent mieux que le temoin", fontsize=10.5, loc="left")
    a.legend(fontsize=8)
    a.grid(alpha=.3)

    # D -- the shared magnitude
    a = ax[1, 0]
    for i, m in enumerate(ORDER):
        if m not in ref:
            continue
        a.errorbar([i], [ref[m]["jac_frobenius"]],
                   yerr=[ref[m].get("jac_frobenius__sd") or 0.0],
                   fmt="o", ms=10, capsize=4, color=COLOR[m])
    a.set_xticks(range(len(ORDER)))
    a.set_xticklabels([LABEL[m] for m in ORDER], fontsize=9, rotation=12)
    a.set_ylabel("||J||_F")
    rj = pearson([c["jac_frobenius"] for c in conds], [c["train_error"] for c in conds])
    a.set_title("D. la signature partagee par les deux interventions\n"
                "a lire a cote de l'erreur d'entrainement :\n"
                "||J||_F correle %+.2f avec elle sur cette grille" % rj,
                fontsize=10.5, loc="left")
    a.grid(alpha=.3, axis="y")

    # E -- no linear regime
    a = ax[1, 1]
    eps = sorted({float(k[4:]) for k in ref["plain"] if k.startswith("lin@")
                  and not k.endswith("__sd")})
    for m in ORDER:
        if m not in ref:
            continue
        y = [ref[m].get("lin@%g" % e) for e in eps]
        if any(v is None for v in y):
            continue
        a.semilogx(eps, y, "o-", color=COLOR[m], label=LABEL[m], lw=1.8)
    a.axhline(1.0, color="k", ls="--", lw=1.2)
    a.text(eps[0], 1.03, "regime lineaire", fontsize=8)
    a.set_xlabel("amplitude")
    a.set_ylabel("S(eps) / (eps * sigma_max)")
    a.set_title("E. aucun regime lineaire accessible\nle premier ordre decrit une limite "
                "que la fonction n'occupe jamais", fontsize=10.5, loc="left")
    a.legend(fontsize=8)
    a.grid(alpha=.3, which="both")

    # D bis -- the amplitude curve itself, the quantity the grid was sized for
    a = ax[1, 2]
    eps_all = sorted({float(k[2:]) for k in ref["plain"] if k.startswith("S@")
                      and not k.endswith("__sd")})
    for m in ORDER:
        if m not in ref:
            continue
        a.semilogx(eps_all, [ref[m]["S@%g" % e] for e in eps_all], "o-",
                   color=COLOR[m], label=LABEL[m], lw=2.0 if m == "plain" else 1.5)
    a.set_xlabel("amplitude de la poussee")
    a.set_ylabel("|| f(x + eps v1) - f(x) ||")
    a.set_title("D bis. la courbe d'amplitude (resp_top)\nla fenetre informative etait "
                "eps dans [2,3] ;\nelle s'effondre au-dela", fontsize=10.5, loc="left")
    a.legend(fontsize=8)
    a.grid(alpha=.3, which="both")

    # F -- the dial test
    a = ax[1, 3]
    rows = [r for r in rep["dial_test"]["rows"] if "skipped" not in r]
    # keep one amplitude per family, or the panel is unreadable
    keep = ("trace_H", "lambda_max", "top_share", "jac_frobenius", "jac_spectral",
            "mean_radius", "frac_above_k8", "S@0.05", "S@0.5", "S@3", "S@12", "lin@3")
    rows = [r for r in rows if r["field"] in keep]
    names = [r["field"] for r in rows]
    x = range(len(names))
    a.barh([i - 0.26 for i in x], [r["pearson"] for r in rows], 0.25,
           color="#bdc3c7", label="brute")
    a.barh([i for i in x], [r["partial"] for r in rows], 0.25,
           color="#2c3e50", label="partielle (err_train controlee)")
    a.barh([i + 0.26 for i in x], [(r["groups"]["within_pooled"] or 0.0) for r in rows],
           0.25, color="#c0392b", label="dans les groupes")
    ceil = rep["detection_ceiling"]["ceiling"]
    for s in (-1, 1):
        a.axvline(s * ceil, color="#c0392b", ls=":", lw=1.5)
    a.text(ceil, len(names) - 0.5, " plafond %.2f" % ceil, fontsize=8, color="#c0392b")
    a.set_yticks(list(x))
    a.set_yticklabels(names, fontsize=8)
    a.axvline(0, color="k", lw=.8)
    a.set_xlabel("correlation avec l'erreur de test")
    n_art = sum(1 for r in rows if r["groups"]["grouping_artefact"])
    a.set_title("F. le test du curseur, %d conditions\n%d des %d quantites sont des "
                "artefacts de groupe :\nla barre noire s'effondre sur la rouge"
                % (rep["dial_test"]["n_conditions"], n_art, len(rows)),
                fontsize=10.5, loc="left")
    a.legend(fontsize=8)
    a.grid(alpha=.3, axis="x")

    ax[0, 3].axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=145)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
