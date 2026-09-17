"""Deal the 63 runs to GPU queues from measured pilot costs (operational, not a scientific choice).

    py -m overnight.allocate --pilot <pilot ovn dir> --accounts acc:gpus:offset_hours ...

Cells are taken in the frozen priority order (regime priority, then seed, then arm) and each goes
to the GPU queue with the smallest projected finishing time, so every GPU works through
priority 1 before priority 2 and so on.  Costs come only from pilot timings (no score is read).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from . import matrix as MX

#: per-run hours not measured by update timing: 3 intermediate P1 evaluations, the final four
#: policies and checkpoint writes; replaced by pilot policy timings when available
FIXED_OVERHEAD_H = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--accounts", nargs="+", required=True, help="account:n_gpu:start_offset_hours")
    a = ap.parse_args()
    cost, src = {}, {}
    for p in Path(a.pilot).glob("pilot/*/pilot_timing.json"):
        r = json.loads(p.read_text())
        regime, arm, _ = r["run"].split("__")
        pol = sum(r["policy_seconds_train_scoring_only"].values()) if r["policy_seconds_train_scoring_only"] else None
        cost[(regime, arm)] = r["projected_train_hours_62560"] + r["projected_periodic_eval_hours_161"] + FIXED_OVERHEAD_H
        src[(regime, arm)] = r
    # the pilot timed updates 40-240, i.e. R/RG in their cheap 16x16 phase: bound them conservatively
    for regime in MX.REGIMES:
        plain = cost[(regime, "plain")]
        cost[(regime, "resolution_max_b1")] = max(cost[(regime, "resolution_max_b1")], plain)
        cost[(regime, "resolution_max_b1_gaussian_conv")] = 1.3 * max(cost[(regime, "resolution_max_b1_gaussian_conv")],
                                                                     cost[(regime, "gaussian_postrelu")])
    missing = [(c["regime"], c["arm"]) for c in MX.cells() if (c["regime"], c["arm"]) not in cost]
    if missing:
        raise SystemExit("pilot has no timing for %s" % sorted(set(missing)))
    gpus = []
    for spec in a.accounts:
        acc, n, off = spec.split(":")
        for g in range(int(n)):
            gpus.append({"account": acc, "gpu": g, "load_h": float(off), "cells": []})
    order = sorted(MX.cells(), key=lambda c: (c["priority"], c["seed"], MX.ARMS.index(c["arm"])))
    for c in order:
        q = min(gpus, key=lambda x: (x["load_h"], x["account"], x["gpu"]))
        q["cells"].append(c["run"])
        q["load_h"] += cost[(c["regime"], c["arm"])]
    out = {"created_utc": datetime.now(timezone.utc).isoformat(), "rule": __doc__.strip(),
           "cost_hours_per_run": {"%s__%s" % k: v for k, v in sorted(cost.items())},
           "total_projected_gpu_hours": sum(cost[(c["regime"], c["arm"])] for c in MX.cells()),
           "accounts": {}}
    for q in gpus:
        acc = out["accounts"].setdefault(q["account"], {"queues": [], "projected_hours": []})
        acc["queues"].append(q["cells"])
        acc["projected_hours"].append(q["load_h"])
    path = MX.STUDY / "protocol" / "ALLOCATION.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {"projected_hours": [round(h, 2) for h in v["projected_hours"]], "n": [len(x) for x in v["queues"]]}
                      for k, v in out["accounts"].items()}, indent=1), "total", round(out["total_projected_gpu_hours"], 1))


if __name__ == "__main__":
    main()
