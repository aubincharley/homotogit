"""Read the probed grid, apply the guards, and report the contrasts and the dial test.

    py tools/analyze_grid.py [--probes DIR] [--out report.json]

Structure follows the six questions the grid was built to answer:

  A  is the solution flatter?                      curvature, against the control
  B  does a PAC-Bayes bound explain it?            distance, KL, factor needed
  C  is it a low-frequency bias?                   frequency profile, blur vs resolution
  D  what do the two interventions share?          Jacobian magnitude
  E  does a first-order description ever hold?     linearity ratio
  F  does any of it predict generalisation?        the dial test, over conditions

The guards run before any of it: the validity gate decides which cells are
readable at all, the seed spread gives the noise floor, and the detection ceiling
is printed beside every coefficient so a number at the ceiling is not read as a
failure.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from continuation_core.analysis import power  # noqa: E402

CONTROL = "plain"


def load(pattern):
    cells = {}
    for f in sorted(glob.glob(pattern, recursive=True)):
        cells.update(json.load(open(f))["cells"])
    return cells


def split_name(cell: str):
    method, variant, seed = cell.rsplit("__", 2)
    return method, variant, int(seed.replace("seed", ""))


def condition_of(cell: str) -> str:
    m, v, _ = split_name(cell)
    return "%s__%s" % (m, v)


def flatten(rec: dict) -> dict:
    s, c = rec["sensitivity"], rec["curvature"]
    j, f = s["jacobian"], s["frequency"]
    curve = {p["epsilon"]: p["displacement"] for p in s["amplitude_curve"]}
    lin = {p["epsilon"]: p["ratio"] for p in s["linearity_ratio"]}
    out = {"train_error": rec["train"]["err"], "test_error": rec["test"]["err"],
           "gap": rec["gap_err"],
           "trace_H": c["trace"]["mean"], "trace_H_sem": c["trace"]["sem"],
           "lambda_max": c["top_eigenvalues"][0]["eigenvalue"],
           "top_share": c["top_share_of_trace"],
           "jac_frobenius": j["frobenius_mean"], "jac_spectral": j["spectral_mean"],
           "jac_participation": j["participation_mean"],
           "mean_radius": f["mean_radius"],
           "frac_above_k8": f["fraction_above"]["8"]}
    for e in curve:                      # the whole amplitude curve
        out["S@%g" % e] = curve[e]
        out["lin@%g" % e] = lin[e]
    for b, v in c["per_block"].items():
        out["block_%s" % b] = v["mean"]
    return out


def by_condition(rows: dict) -> list:
    groups = {}
    for cell, r in rows.items():
        groups.setdefault(condition_of(cell), []).append(r)
    out = []
    for cond, g in sorted(groups.items()):
        m, v = cond.split("__")
        rec = {"condition": cond, "method": m, "variant": v, "n_seeds": len(g),
               "group": "control" if m == CONTROL else "curriculum"}
        for k in g[0]:
            vals = [x[k] for x in g if x.get(k) is not None]
            if vals:
                rec[k] = sum(vals) / len(vals)
                rec[k + "__sd"] = st.stdev(vals) if len(vals) > 1 else None
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", default=str(ROOT / "results/kaggle_outputs/cc-probe-*/probe_shard*.json"))
    ap.add_argument("--out", default=str(ROOT / "results/grid_report.json"))
    args = ap.parse_args()

    cells = load(args.probes)
    if not cells:
        raise SystemExit("no probe shards matched %s" % args.probes)
    rows = {k: flatten(v) for k, v in cells.items()}
    print("%d cellules, %d conditions\n" % (len(rows), len({condition_of(k) for k in rows})))

    # -- guard 1: noise floor from the reference seeds ----------------------
    ref = {k: v for k, v in rows.items() if "__reference__" in k}
    floors = {}
    for field in ("trace_H", "jac_frobenius", "S@3", "test_error", "gap"):
        per_method = {}
        for k, v in ref.items():
            per_method.setdefault(split_name(k)[0], []).append(v[field])
        sds = [st.stdev(g) for g in per_method.values() if len(g) > 1]
        floors[field] = {"seed_sd_mean": sum(sds) / len(sds) if sds else None,
                         "per_method_sd": {m: (st.stdev(g) if len(g) > 1 else None)
                                           for m, g in per_method.items()}}
    print("=== garde-fou : plancher de bruit (dispersion de graine, cellules de reference) ===")
    for f, v in floors.items():
        print("   %-16s ecart-type moyen %s" % (f, ("%.4g" % v["seed_sd_mean"]) if v["seed_sd_mean"] else "-"))

    # -- A to E: contrasts against the control ------------------------------
    conds = by_condition(rows)
    base = next(c for c in conds if c["method"] == CONTROL and c["variant"] == "reference")
    print("\n=== A-E : contraste contre le temoin (cellules de reference) ===")
    hdr = ("trace_H", "jac_frobenius", "S@3", "mean_radius", "frac_above_k8", "test_error", "gap")
    print("   %-34s %s" % ("condition", " ".join("%13s" % h for h in hdr)))
    for c in conds:
        if c["variant"] != "reference":
            continue
        print("   %-34s %s" % (c["condition"], " ".join("%13.4f" % c[h] for h in hdr)))
    print("\n   variation relative au temoin :")
    for c in conds:
        if c["variant"] != "reference" or c["method"] == CONTROL:
            continue
        print("   %-34s %s" % (c["condition"],
                               " ".join("%12.1f%%" % (100 * (c[h] / base[h] - 1)) for h in hdr)))

    # -- E: is there ever a linear regime? ----------------------------------
    print("\n=== E : rapport S(eps) / (eps * sigma_max)  --  1 = regime lineaire ===")
    for c in conds:
        if c["variant"] != "reference":
            continue
        print("   %-34s %s" % (c["condition"],
                               " ".join("%s=%.3f" % (e, c.get("lin@%s" % e, float("nan")))
                                        for e in ("0.5", "1", "3"))))

    # -- D bis: the amplitude curve itself ----------------------------------
    eps_all = sorted({float(k[2:]) for k in conds[0] if k.startswith("S@")
                      and not k.endswith("__sd")})
    print("\n=== D bis : la courbe d'amplitude S(eps), cellules de reference ===")
    print("   %-34s %s" % ("condition", " ".join("%8g" % e for e in eps_all)))
    for c in conds:
        if c["variant"] != "reference":
            continue
        print("   %-34s %s" % (c["condition"],
                               " ".join("%8.2f" % c["S@%g" % e] for e in eps_all)))

    # -- F: the dial test, over conditions ----------------------------------
    gaps = [c["gap"] for c in conds]
    ceiling = power.detection_ceiling(st.stdev(gaps), floors["gap"]["seed_sd_mean"] or 0.0)
    eps_present = sorted({float(k[2:]) for k in conds[0] if k.startswith("S@")
                          and not k.endswith("__sd")})
    fields = ["trace_H", "lambda_max", "top_share", "jac_frobenius", "jac_spectral",
              "mean_radius", "frac_above_k8"]
    # every amplitude, because the exploratory curve peaked at 3 and collapsed by
    # 12 -- testing one point would hide both the optimum and the collapse
    fields += ["S@%g" % e for e in eps_present]
    fields += ["lin@%g" % e for e in eps_present]
    res = power.correlate(conds, fields, n_tested=len(fields), group_key="group")
    print("\n=== F : le test du curseur, %d conditions ===" % res["n_conditions"])
    print("   plafond de detection impose par le bruit de la cible : |r| <= %.2f" % ceiling["ceiling"])
    print("   seuil corrige pour %d quantites testees : p < %.4f\n" % (len(fields), res["alpha_corrected"]))
    print("   %-18s %9s %9s %10s %9s %11s %11s %10s"
          % ("", "pearson", "rang", "partielle", "p", "temoin", "curriculum", "dans-grp"))
    for r in res["rows"]:
        if "skipped" in r:
            continue
        w = r["groups"]["within"]
        def fmt(k):
            v = w.get(k, {}).get("partial")
            return "%+11.3f" % v if v is not None else "%11s" % ("n=%d" % w.get(k, {}).get("n", 0))
        flag = "  <--" if r["passes_corrected"] else (
            "  artefact" if r["groups"]["grouping_artefact"] else "")
        wp = r["groups"].get("within_pooled")
        print("   %-18s %+9.3f %+9.3f %+10.3f %9.4f %s %s %s%s"
              % (r["field"], r["pearson"], r["spearman"], r["partial"], r["p_partial"],
                 fmt("control"), fmt("curriculum"),
                 ("%+10.3f" % wp) if wp is not None else "%10s" % "-", flag))

    report = {"n_cells": len(rows), "noise_floor": floors, "conditions": conds,
              "detection_ceiling": ceiling, "dial_test": res}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
