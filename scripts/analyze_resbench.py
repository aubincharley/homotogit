"""Aggregate the resolution-only benchmark and compute its prespecified contrasts.

Reads every completed cell from the three job outputs, checks the frozen
manifest is covered exactly once, and writes ``results/resbench_results.json``.

All arms share pinned initial weights, BN buffers, probe indices and per-epoch
permutations within a seed, so every difference below is computed **per seed and
then averaged** -- never as a difference of marginal means.

The fixed-resolution controls end at a reduced-resolution inference path, unlike
every progressive arm, which ends at 32x32 with the operator bypassed.  They are
kept separate in the tables for that reason.
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from continuation.resolution_ops import LOCATIONS, O_REF, OPERATORS
from scripts.resbench_manifest import PATH_OPERATORS, PATH_SET, SEEDS, build_configs

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "results" / "kaggle_outputs"
JOBS = ["resbench-j0-20260910-214422", "resbench-j1-20260910-214450",
        "resbench-j2-20260910-214517"]
OUT = ROOT / "results" / "resbench_results.json"
PLAIN = "none__none__none"
FIXED = ("%s__D1__fixed16" % O_REF, "%s__D1__fixed24" % O_REF)


def collect():
    runs, jobs_meta = {}, []
    for job in JOBS:
        jd = OUTPUTS / job
        if not jd.is_dir():
            continue
        for study in sorted(jd.glob("resbench_job*")):
            js = study / "job_summary.json"
            meta = {"job": job, "study": str(study.relative_to(ROOT))}
            if js.is_file():
                d = json.loads(js.read_text(encoding="utf-8"))
                meta["elapsed_s"] = d.get("elapsed_s")
                meta["failed_or_incomplete"] = [
                    c.get("cell_id", c.get("id")) for c in
                    d.get("failed_or_incomplete", [])]
            av = study / "assets_verification.json"
            if av.is_file():
                meta["assets_verified"] = json.loads(
                    av.read_text(encoding="utf-8")).get("all_match")
            probe = jd / "t4_timing_probe.json"
            if probe.is_file():
                meta["probe"] = json.loads(probe.read_text(encoding="utf-8"))
            jobs_meta.append(meta)
            for cell in sorted(p for p in study.iterdir() if p.is_dir()):
                f = cell / "summary.json"
                if f.is_file():
                    s = json.loads(f.read_text(encoding="utf-8"))
                    runs[s["cell_id"]] = {**s, "job_slug": job,
                                          "path_on_disk": str(cell.relative_to(ROOT))}
    return runs, jobs_meta


def finite(run) -> bool:
    """A cell that produced a non-finite loss did not train; it is not a result."""
    for k in ("final_test_acc", "final_test_ce", "final_train_probe_ce"):
        v = run.get(k)
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return False
    return True


def by_config(runs):
    out = {}
    for c in build_configs():
        allseeds = {s: runs["%s__seed%d" % (c["id"], s)] for s in SEEDS
                    if "%s__seed%d" % (c["id"], s) in runs}
        seeds = {s: r for s, r in allseeds.items() if finite(r)}
        diverged = sorted(s for s in allseeds if s not in seeds)
        if not seeds:
            if allseeds:
                out[c["id"]] = {
                    **{k: c[k] for k in ("id", "operator", "location", "path",
                                         "groups", "priority")},
                    "n_seeds": 0, "seeds": [], "diverged_seeds": diverged,
                    "status": "diverged (non-finite loss in every seed)",
                    "acc_per_seed": [allseeds[s]["final_test_acc"] for s in diverged]}
            continue
        acc = [seeds[s]["final_test_acc"] for s in sorted(seeds)]
        ce = [seeds[s]["final_test_ce"] for s in sorted(seeds)]
        wall = [seeds[s]["wall_seconds"] for s in sorted(seeds)]
        mem = [seeds[s]["peak_mem_mib"] for s in sorted(seeds)]
        pce = [seeds[s]["final_train_probe_ce"] for s in sorted(seeds)]
        sd = (lambda v: st.stdev(v) if len(v) > 1 else 0.0)
        out[c["id"]] = {
            **{k: c[k] for k in ("id", "operator", "location", "path",
                                 "groups", "priority")},
            "n_seeds": len(seeds), "seeds": sorted(seeds),
            "diverged_seeds": diverged,
            "status": "ok" if not diverged else "partial (some seeds diverged)",
            "acc_per_seed": acc, "acc_mean": st.mean(acc), "acc_sd": sd(acc),
            "ce_per_seed": ce, "ce_mean": st.mean(ce), "ce_sd": sd(ce),
            "probe_ce_mean": st.mean(pce),
            "wall_mean": st.mean(wall), "wall_sd": sd(wall),
            "mem_mean": st.mean(mem), "mem_max": max(mem),
            "endpoint": seeds[sorted(seeds)[0]].get("endpoint"),
        }
    return out


def paired(runs, a, b):
    """Per-seed a - b, in percentage points."""
    common = sorted(s for s in SEEDS
                    if "%s__seed%d" % (a, s) in runs and "%s__seed%d" % (b, s) in runs
                    and finite(runs["%s__seed%d" % (a, s)])
                    and finite(runs["%s__seed%d" % (b, s)]))
    if not common:
        return None
    d = [100 * (runs["%s__seed%d" % (a, s)]["final_test_acc"]
                - runs["%s__seed%d" % (b, s)]["final_test_acc"]) for s in common]
    dce = [runs["%s__seed%d" % (a, s)]["final_test_ce"]
           - runs["%s__seed%d" % (b, s)]["final_test_ce"] for s in common]
    return {"a": a, "b": b, "seeds": common, "n": len(common),
            "d_acc_pp_per_seed": [round(v, 3) for v in d],
            "d_acc_pp": st.mean(d),
            "d_acc_sd": st.stdev(d) if len(d) > 1 else 0.0,
            "d_ce": st.mean(dce),
            "all_same_sign": all(v > 0 for v in d) or all(v < 0 for v in d)}


def contrasts(runs, cfgs):
    def cid(op, loc, path):
        return "%s__%s__%s" % (op, loc, path)

    out = {}
    have = set(cfgs)

    def add(key, a, b):
        if a in have and b in have:
            r = paired(runs, a, b)
            if r:
                out[key] = r

    # 1. everything vs plain
    for k in cfgs:
        if k != PLAIN:
            add("vs_plain/%s" % k, k, PLAIN)
    # 2. each operator vs O_ref at the same location and path
    for loc in ("input", "D1"):
        for op in OPERATORS:
            if op != O_REF:
                add("vs_oref/%s@%s" % (op, loc), cid(op, loc, "Rprog"),
                    cid(O_REF, loc, "Rprog"))
    for op in PATH_OPERATORS:
        for path in PATH_SET:
            if op != O_REF:
                add("vs_oref/%s@D1@%s" % (op, path), cid(op, "D1", path),
                    cid(O_REF, "D1", path))
    # 3. input vs D1 for each operator
    for op in OPERATORS:
        add("input_vs_D1/%s" % op, cid(op, "input", "Rprog"),
            cid(op, "D1", "Rprog"))
    # 4. the five depths under O_ref / Rprog
    for loc in LOCATIONS:
        if loc != "D1":
            add("depth/%s_vs_D1" % loc, cid(O_REF, loc, "Rprog"),
                cid(O_REF, "D1", "Rprog"))
    # 5. the four paths, at fixed operator and location
    for op in PATH_OPERATORS:
        for path in PATH_SET:
            if path != "Rprog":
                add("path/%s@%s_vs_Rprog" % (op, path), cid(op, "D1", path),
                    cid(op, "D1", "Rprog"))
    # 6. named operator questions
    add("maxblur_vs_max@D1", cid("maxblur", "D1", "Rprog"), cid("max", "D1", "Rprog"))
    add("maxblur_vs_max@input", cid("maxblur", "input", "Rprog"),
        cid("max", "input", "Rprog"))
    add("hminus1_vs_l2@D1", cid("hminus1", "D1", "Rprog"), cid("l2", "D1", "Rprog"))
    add("hminus1_vs_l2@input", cid("hminus1", "input", "Rprog"),
        cid("l2", "input", "Rprog"))
    # 7. progressive vs fixed native D1 (different final inference configuration)
    for f in FIXED:
        add("progressive_vs_fixed/%s" % f, cid(O_REF, "D1", "Rprog"), f)
    return out


def main():
    runs, jobs_meta = collect()
    cfgs = build_configs()
    expected = {"%s__seed%d" % (c["id"], s) for c in cfgs for s in SEEDS}
    missing = sorted(expected - set(runs))
    extra = sorted(set(runs) - expected)
    stats = by_config(runs)
    diverged = {cid: c["diverged_seeds"] for cid, c in stats.items()
                if c.get("diverged_seeds")}

    result = {
        "n_configurations_expected": len(cfgs),
        "n_configurations_present": len(stats),
        "n_cells_expected": len(expected), "n_cells_completed": len(runs),
        "missing_cells": missing, "unexpected_cells": extra,
        "diverged_cells": diverged,
        "jobs": jobs_meta,
        "cumulative_gpu_seconds": sum(r["wall_seconds"] for r in runs.values()),
        "o_ref": O_REF,
        "configurations": stats,
        "contrasts": contrasts(runs, stats),
        "notes": [
            "All internal Gaussian filters disabled; these arms vary resolution only.",
            "Differences are computed per seed against the paired arm, then averaged.",
            "Progressive arms end at 32x32 with the operator bypassed; the two fixed "
            "controls end on their reduced-resolution inference path and are not a "
            "schedule-only comparison.",
            "Three seeds give a descriptive sample SD, not a significance statement; "
            "the test set has prior exposure.",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("cells %d/%d, configurations %d/%d, missing %d"
          % (len(runs), len(expected), len(stats), len(cfgs), len(missing)))
    for m in missing[:10]:
        print("   MISSING", m)
    for cid, ds in diverged.items():
        print("   DIVERGED %s seeds %s (non-finite loss; excluded from aggregates)"
              % (cid, ds))
    print("cumulative GPU time %.2f h" % (result["cumulative_gpu_seconds"] / 3600))
    print()
    print("%-34s %-7s %-7s %-7s %-7s %s" % ("configuration", "acc%", "sd", "CE",
                                            "wall s", "endpoint"))
    for k, c in sorted(stats.items(), key=lambda kv: -kv[1].get("acc_mean", -1)):
        if c["n_seeds"] == 0:
            print("%-34s %s" % (k, c["status"]))
            continue
        print("%-34s %-7.2f %-7.2f %-7.4f %-7.0f %s%s"
              % (k, 100 * c["acc_mean"], 100 * c["acc_sd"], c["ce_mean"],
                 c["wall_mean"], (c["endpoint"] or "")[:26],
                 "  [n=%d]" % c["n_seeds"] if c["n_seeds"] != 3 else ""))
    print()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
