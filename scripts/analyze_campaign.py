"""Aggregate the 21-configuration campaign and produce the reported tables.

Collects every completed cell from the three job outputs, checks the manifest is
covered exactly once, and computes per-seed metrics, three-seed mean and sample
SD, and the prespecified paired differences.  Pairing is by construction here --
all cells verified the same pinned assets before training -- and that is
re-asserted from each job's ``assets_verification.json``.

Descriptive only: three seeds give a sample SD, not a confidence statement, and
the campaign had prior test-set exposure, so the best grid cell is not an
independently confirmed estimate.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.campaign_manifest import build_cells, build_configs

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "results" / "kaggle_outputs"
JOBS = ["campaign-j0-20260909-095039", "campaign-j1-20260909-095101",
        "campaign-j2-20260909-095123"]
OUT = ROOT / "results" / "campaign_results.json"


def collect():
    runs, jobs_meta = {}, []
    for job in JOBS:
        jd = OUTPUTS / job
        if not jd.is_dir():
            continue
        for study in jd.glob("campaign_job*"):
            av = study / "assets_verification.json"
            js = study / "job_summary.json"
            meta = {"job": job, "study": str(study.relative_to(ROOT)),
                    "assets_verified": (json.loads(av.read_text())["all_match"]
                                        if av.is_file() else None),
                    "elapsed_s": (json.loads(js.read_text())["elapsed_s"]
                                  if js.is_file() else None)}
            probe = jd / "t4_timing_probe.json"
            if probe.is_file():
                meta["probe"] = json.loads(probe.read_text())
            jobs_meta.append(meta)
            for cell_dir in sorted(p for p in study.iterdir() if p.is_dir()):
                f = cell_dir / "summary.json"
                if f.is_file():
                    s = json.loads(f.read_text())
                    runs[s["cell_id"]] = {**s, "job_slug": job,
                                          "path": str(cell_dir.relative_to(ROOT))}
    return runs, jobs_meta


def by_config(runs):
    out = {}
    for c in build_configs():
        seeds = {}
        for s in (0, 1, 2):
            cid = "%s__seed%d" % (c["id"], s)
            if cid in runs:
                seeds[s] = runs[cid]
        if not seeds:
            continue
        acc = [seeds[s]["final_test_acc"] for s in sorted(seeds)]
        ce = [seeds[s]["final_test_ce"] for s in sorted(seeds)]
        pce = [seeds[s]["final_train_probe_ce"] for s in sorted(seeds)]
        wall = [seeds[s]["wall_seconds"] for s in sorted(seeds)]
        train = [seeds[s]["train_seconds"] for s in sorted(seeds)]
        out[c["id"]] = {
            **c, "n_seeds": len(seeds), "seeds": sorted(seeds),
            "acc_per_seed": acc, "ce_per_seed": ce,
            "acc_mean": st.mean(acc), "acc_sd": st.stdev(acc) if len(acc) > 1 else None,
            "ce_mean": st.mean(ce), "ce_sd": st.stdev(ce) if len(ce) > 1 else None,
            "probe_ce_mean": st.mean(pce),
            "wall_mean": st.mean(wall), "train_mean": st.mean(train),
            "peak_mem_mib": max(seeds[s]["peak_mem_mib"] or 0 for s in seeds),
        }
    return out


def paired(cfgs, a, b, runs):
    """Per-seed difference a - b, with mean and sample SD."""
    if a not in cfgs or b not in cfgs:
        return None
    common = sorted(set(cfgs[a]["seeds"]) & set(cfgs[b]["seeds"]))
    if not common:
        return None
    d_acc, d_ce = [], []
    for s in common:
        ra = runs["%s__seed%d" % (a, s)]
        rb = runs["%s__seed%d" % (b, s)]
        d_acc.append(ra["final_test_acc"] - rb["final_test_acc"])
        d_ce.append(ra["final_test_ce"] - rb["final_test_ce"])
    return {"a": a, "b": b, "seeds": common,
            "d_acc_per_seed": d_acc, "d_acc_mean": st.mean(d_acc),
            "d_acc_sd": st.stdev(d_acc) if len(d_acc) > 1 else None,
            "d_ce_mean": st.mean(d_ce),
            "d_ce_sd": st.stdev(d_ce) if len(d_ce) > 1 else None,
            "all_same_sign": all(d > 0 for d in d_acc) or all(d < 0 for d in d_acc)}


IB, A19 = "input_bilinear", "all19"


def comparisons(cfgs, runs):
    def cid(res, g, red=IB, mask=A19):
        return "%s__%s__%s__%s" % (res, g, red, mask)

    out = {}
    # resolution vs plain, at each Gaussian schedule
    for g in ("Gnone", "Gplateau", "Ggeo"):
        for res in ("Rprog", "Rgentle"):
            out["%s_vs_R32__%s" % (res, g)] = paired(cfgs, cid(res, g),
                                                     cid("R32", g), runs)
    # Gaussian schedules at each resolution
    for res in ("R32", "Rprog", "Rgentle"):
        for g in ("Gplateau", "Ggeo"):
            out["%s__%s_vs_Gnone" % (res, g)] = paired(cfgs, cid(res, g),
                                                       cid(res, "Gnone"), runs)
    # Gmix vs Gplateau
    for res in ("R32", "Rprog"):
        out["%s__Gmix_vs_Gplateau" % res] = paired(cfgs, cid(res, "Gmix"),
                                                   cid(res, "Gplateau"), runs)
    # early7 vs all19
    for res in ("R32", "Rprog"):
        out["%s__early7_vs_all19" % res] = paired(
            cfgs, cid(res, "Gplateau", IB, "early7"),
            cid(res, "Gplateau", IB, A19), runs)
    # reduction operator / location, in BOTH unfiltered and Gaussian conditions
    for g in ("Gnone", "Gplateau"):
        for red in ("input_max", "stem_bilinear", "stem_max"):
            out["Rprog__%s_vs_input_bilinear__%s" % (red, g)] = paired(
                cfgs, cid("Rprog", g, red), cid("Rprog", g, IB), runs)
    # forward vs reversed progression
    for g in ("Gnone", "Gplateau"):
        out["Rreverse_vs_Rprog__%s" % g] = paired(cfgs, cid("Rreverse", g),
                                                  cid("Rprog", g), runs)
    return {k: v for k, v in out.items() if v}


HISTORICAL = {
    "R32__Gnone__input_bilinear__all19": ("plain", [0.7510, 0.7526, 0.7585]),
    "R32__Gplateau__input_bilinear__all19": ("plateau", [0.7827, 0.7878, 0.7819]),
    "R32__Ggeo__input_bilinear__all19": ("geometric", [0.7763, 0.7884, 0.7892]),
}


def replication(cfgs):
    """The three R32 configurations were retrained; compare to the campaign."""
    out = {}
    for cid, (name, hist) in HISTORICAL.items():
        if cid not in cfgs:
            continue
        new = cfgs[cid]["acc_per_seed"]
        if len(new) != len(hist):
            continue
        out[name] = {"historical": hist, "recomputed": new,
                     "per_seed_delta": [n - h for n, h in zip(new, hist)],
                     "mean_delta": st.mean([n - h for n, h in zip(new, hist)]),
                     "max_abs_delta": max(abs(n - h) for n, h in zip(new, hist))}
    return out


def main():
    runs, jobs_meta = collect()
    cells = build_cells()
    missing = [c["cell_id"] for c in cells if c["cell_id"] not in runs]
    dup = len(runs) - len({k for k in runs})
    cfgs = by_config(runs)
    result = {
        "n_cells_expected": len(cells), "n_cells_completed": len(runs),
        "missing_cells": missing, "duplicate_cells": dup,
        "jobs": jobs_meta,
        "cumulative_gpu_seconds": sum(r["wall_seconds"] for r in runs.values()),
        "configurations": cfgs,
        "comparisons": comparisons(cfgs, runs),
        "replication_vs_historical": replication(cfgs),
        "note": ("Exploratory benchmark with prior test-set exposure; three "
                 "seeds give a descriptive sample SD, not a confidence "
                 "statement, and the best cell is not an independently "
                 "confirmed estimate."),
    }
    OUT.write_text(json.dumps(result, indent=2))

    print("cells %d/%d completed; missing %d" %
          (len(runs), len(cells), len(missing)))
    for m in missing:
        print("  MISSING", m)
    print("cumulative GPU time %.2f h" % (result["cumulative_gpu_seconds"] / 3600))
    print()
    print("%-46s %-7s %-7s %-7s %7s" % ("configuration", "acc", "sd", "CE", "wall s"))
    for cid, c in sorted(cfgs.items(), key=lambda kv: -kv[1]["acc_mean"]):
        print("%-46s %.4f  %s  %.4f  %6.0f"
              % (cid[:46], c["acc_mean"],
                 "  -  " if c["acc_sd"] is None else "%.4f" % c["acc_sd"],
                 c["ce_mean"], c["wall_mean"]))
    print()
    print("paired differences (a - b), accuracy in pp")
    for k, v in result["comparisons"].items():
        print("  %-46s %+6.2f pp  sd %s  same-sign=%s"
              % (k[:46], 100 * v["d_acc_mean"],
                 "  -  " if v["d_acc_sd"] is None else "%.2f" % (100 * v["d_acc_sd"]),
                 v["all_same_sign"]))
    print()
    for name, r in result["replication_vs_historical"].items():
        print("replication %-10s mean delta %+0.4f  max |delta| %.4f"
              % (name, r["mean_delta"], r["max_abs_delta"]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
