"""Curvature across the BatchNorm policy x split square, and what it settles.

    py tools/analyze_bnpolicy.py

Two choices in the endpoint probe were load-bearing and untested: the stored
BatchNorm statistics, and the training split.  A companion loss-landscape study
on the same four methods reports that the first reverses the ranking of the
Gaussian method (43-108 % *more* sensitive once recalibrated, 5 seeds out of 5)
and that the second turns the resolution method's advantage into -0.006 nats on
the full test set.

This reads the 2x2 and answers three questions in order: does the curvature
contrast survive recalibration, does it survive the test split, and does the
policy change the *amount*.

Reads only; writes nothing.
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

METHODS = ("plain", "gaussian_postrelu", "resolution_max_b1",
           "resolution_max_b1_gaussian_conv")
LABEL = {"plain": "temoin", "gaussian_postrelu": "flou",
         "resolution_max_b1": "resolution",
         "resolution_max_b1_gaussian_conv": "res+flou"}
POLICIES = ("running_stats", "fixed_batch_stats")
SPLITS = ("train", "test")
CELL = {("running_stats", "train"): "gele/train", ("running_stats", "test"): "gele/test",
        ("fixed_batch_stats", "train"): "recal/train",
        ("fixed_batch_stats", "test"): "recal/test"}


def load():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results/kaggle_outputs/cc-bnpol-s*/bnpolicy_shard*.json"))):
        rows += json.load(open(f))["rows"]
    return rows


def group(rows, m, p, s, key="trace_H"):
    v = [r[key] for r in rows if r["method"] == m and r["bn_policy"] == p and r["split"] == s]
    return v


def main():
    rows = load()
    if not rows:
        raise SystemExit("no shards under results/kaggle_outputs/cc-bnpol-s*/")
    print("%d mesures = %d methodes x %d graines x %d politiques x %d decoupes\n"
          % (len(rows), len(METHODS), len(rows) // 16, len(POLICIES), len(SPLITS)))

    print("=== tr H, moyenne sur les graines (+- ecart-type) ===")
    print("%-12s %s" % ("", " ".join("%20s" % CELL[(p, s)] for p in POLICIES for s in SPLITS)))
    base = {}
    for m in METHODS:
        cells = []
        for p in POLICIES:
            for s in SPLITS:
                v = group(rows, m, p, s)
                if m == "plain":
                    base[(p, s)] = sum(v) / len(v)
                cells.append("%11.1f +-%-6.1f" % (sum(v) / len(v),
                                                  st.stdev(v) if len(v) > 1 else 0.0))
        print("%-12s %s" % (LABEL[m], " ".join(cells)))

    print("\n=== variation contre le temoin, et nombre de graines du bon cote ===")
    print("%-12s %s" % ("", " ".join("%20s" % CELL[(p, s)] for p in POLICIES for s in SPLITS)))
    n_below = 0
    n_total = 0
    for m in METHODS:
        if m == "plain":
            continue
        cells = []
        for p in POLICIES:
            for s in SPLITS:
                v = group(rows, m, p, s)
                b = group(rows, "plain", p, s)
                below = sum(1 for x, y in zip(sorted(v), sorted(b)) if x < y)
                n_below += below
                n_total += len(v)
                cells.append("%12.1f%%  %d/%d" % (100 * (sum(v) / len(v) / base[(p, s)] - 1),
                                                  below, len(v)))
        print("%-12s %s" % (LABEL[m], " ".join(cells)))
    print("\n   le contraste tient dans %d cas sur %d" % (n_below, n_total))

    print("\n=== 1) la recalibration inverse-t-elle le classement ? ===")
    for m in METHODS:
        if m == "plain":
            continue
        signs = []
        for p in POLICIES:
            for s in SPLITS:
                d = sum(group(rows, m, p, s)) / 3 - base[(p, s)]
                signs.append(d < 0)
        print("   %-12s plus plat que le temoin dans %d des 4 cases%s"
              % (LABEL[m], sum(signs), "" if all(signs) else "   <-- INVERSION"))

    print("\n=== 2) l'effet est-il une propriete du train ? ===")
    for m in METHODS:
        if m == "plain":
            continue
        for p in POLICIES:
            tr = 100 * (sum(group(rows, m, p, "train")) / 3 / base[(p, "train")] - 1)
            te = 100 * (sum(group(rows, m, p, "test")) / 3 / base[(p, "test")] - 1)
            print("   %-12s %-18s train %+6.1f%%   test %+6.1f%%   %s"
                  % (LABEL[m], p, tr, te,
                     "plus fort sur le test" if te < tr else "plus faible sur le test"))

    print("\n=== 3) de combien la politique change-t-elle l'amplitude ? ===")
    for m in METHODS:
        for s in SPLITS:
            a = sum(group(rows, m, "running_stats", s)) / 3
            b = sum(group(rows, m, "fixed_batch_stats", s)) / 3
            print("   %-12s %-6s gele / recalibre = %.1f x" % (LABEL[m], s, a / b))
    print("\n   une grande part de la politique gelee est un desaccord de statistiques,")
    print("   pas de la geometrie ; l'effet du curriculum y survit mais retrecit.")


if __name__ == "__main__":
    main()
