"""Aggregate landscape_v3 raw results (+ reused v2 points) into tables, figures and a summary.

    py -m landscape_v3.analyze [--skip-surfaces]

Seeds are the replication unit: every comparison reports the five seed values,
their mean, descriptive SD and sign counts.  Missing tasks, failures, deadline
stops and non-finite values are listed in ``tables/summary.json``.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import ahess, asens, asurf
from . import common as C
from .acommon import TAB, Data, write_csv


def interpolation_and_pca(summary):
    rows = []
    segs = (("plain", "resolution_max_b1"), ("plain", "gaussian_postrelu"), ("plain", "resolution_max_b1_gaussian_conv"),
            ("resolution_max_b1", "resolution_max_b1_gaussian_conv"))
    for s in C.SEEDS:
        ev = C.V2_RAW / C.V2_ACCOUNT_OF_SEED[s] / "v2" / "eval"
        for a, b in segs:
            pts = [json.loads(l) for l in (ev / ("interp__%s_%s__seed%d.jsonl" % (C.SHORT[a], C.SHORT[b], s))).read_text().splitlines()]
            pts = sorted({p["key"]: p for p in pts}.values(), key=lambda p: p["alpha"])
            rec = {"A": a, "B": b, "seed": s, "n_points": len(pts)}
            for sp in C.PROBES:
                ce = np.array([p["splits"][sp]["ce"] for p in pts])
                rec[sp + "_barrier"] = float(ce.max() - max(ce[0], ce[-1]))
                rec[sp + "_argmax_alpha"] = float(pts[int(ce.argmax())]["alpha"])
                rec[sp + "_n_interior_above_both_ends"] = int((ce[1:-1] > max(ce[0], ce[-1])).sum())
            rows.append(rec)
    write_csv("F_interpolation_barriers", rows)
    cmp_ = []
    for s in C.SEEDS:
        for sp in C.PROBES:
            pr = [r for r in rows if r["seed"] == s and r["A"] == "plain" and r["B"] == "resolution_max_b1"][0][sp + "_barrier"]
            rc = [r for r in rows if r["seed"] == s and r["A"] == "resolution_max_b1"][0][sp + "_barrier"]
            cmp_.append({"seed": s, "split": sp, "barrier_plain_to_resolution": pr, "barrier_resolution_to_combined": rc,
                         "resolution_to_combined_smaller": rc < pr})
    write_csv("F_barrier_comparison_res_comb_vs_plain_res", cmp_)
    pca = []
    for s in C.SEEDS:
        z = np.load(C.V2_RAW / C.V2_ACCOUNT_OF_SEED[s] / "v2" / "pca" / ("pca_seed%d.npz" % s), allow_pickle=True)
        evr = z["explained_variance_ratio"]
        d2 = np.maximum(z["dist_to_mean"] ** 2, 1e-30)
        rec = {"seed": s, "evr_pc1": float(evr[0]), "evr_pc1_2": float(evr[:2].sum()), "evr_pc1_4": float(evr[:4].sum()),
               "evr_pc1_10": float(evr[:10].sum())}
        for k in (2, 5, 10):
            f = z["resid%d" % k] ** 2 / d2
            rec["median_unexplained_frac_%dpc" % k] = float(np.median(f))
            rec["max_unexplained_frac_%dpc" % k] = float(f.max())
        pca.append(rec)
    write_csv("F_pca_projection_quality", pca)
    summary["F_interpolation"] = {"segments": rows, "res_comb_vs_plain_res": cmp_,
                                  "n_all_segments_with_barrier": int(sum(r["train_probe_barrier"] > 0 and r["test_probe_barrier"] > 0 for r in rows))}
    summary["F_pca"] = pca


def inventory(D, summary):
    counts, bad = D.status_counts()
    nonfinite = [k for k, r in D.rows.items() if any(not v.get("finite", True) for v in r["splits"].values())]
    by_acc = {}
    for d in D.dirs:
        t = json.loads((d / "timing.json").read_text()) if (d / "timing.json").exists() else {}
        env = json.loads((d / "environment.json").read_text()) if (d / "environment.json").exists() else {}
        rr = json.loads((d / "checks" / "reuse_reproduction.json").read_text()) if (d / "checks" / "reuse_reproduction.json").exists() else {}
        ic = json.loads((d / "checks" / "inputs_check.json").read_text()) if (d / "checks" / "inputs_check.json").exists() else {}
        ver = C.STUDY / "raw" / ("verify_%s.json" % d.parent.name)
        by_acc[d.parent.name] = {"timing": t, "gpus": env.get("gpus"), "torch": env.get("torch"),
                                 "reuse_reproduction": {k: rr.get(k) for k in ("n", "max_abs_diff_ce", "valid")},
                                 "inputs_all_ok": ic.get("all_ok"),
                                 "download_verification": json.loads(ver.read_text()) if ver.exists() else None}
    summary["inventory"] = {"task_status_counts": counts, "not_complete": bad, "n_points_v3": D.n_v3,
                            "n_points_v2_reused_in_analysis": D.n_v2_used, "non_finite_points": nonfinite[:200],
                            "n_non_finite": len(nonfinite), "parse_issues": D.issues[:50], "accounts": by_acc}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-surfaces", action="store_true")
    a = ap.parse_args()
    D = Data()
    summary = {}
    inventory(D, summary)
    quad, _ = ahess.quadforms(D, summary)
    Sbar = asens.amplitudes(D, summary, quad)
    import csv
    per_seed = [dict(r, amplitude=float(r["amplitude"]), seed=int(r["seed"]), C_mean=float(r["C_mean"]))
                for r in csv.DictReader(open(TAB / "A_sensitivity_per_seed.csv"))]
    ahess.quad_vs_fd(per_seed, quad, summary)
    asens.validation(D, summary, Sbar)
    asens.temporal(D, summary)
    asens.transitions(D, summary)
    ahess.spectra(D, summary)
    ahess.hchecks(D, summary)
    ahess.layer_energy(D, summary)
    ahess.cuts(D, summary, Sbar)
    interpolation_and_pca(summary)
    if not a.skip_surfaces:
        asurf.random_planes(D, summary)
        asurf.hessian_planes(D, summary)
        summary["interactive"] = asurf.interactive(D)
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary["inventory"]["task_status_counts"]), "v3 points", D.n_v3, "v2 points used", D.n_v2_used)


if __name__ == "__main__":
    main()
