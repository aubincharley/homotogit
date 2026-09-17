"""Build the task bundle of the missing work only, from the existing landscape_v3 outputs.

    py -m geometry_final.plan

Writes ``protocol/tasks.json`` and ``protocol/reuse_inventory.json``:

* trace objectives: 4 methods x 5 seeds x 2 probes under centre-frozen BN (none exist);
* seed-0 centre-frozen random grid vertices that do not already exist.  Existing
  centre-frozen block-A points coincide with grid vertices on the axes
  (r1:k:eps == r2:0:1 with the other coefficient 0, bitwise the same weights), so
  they are reused and re-evaluated on Kaggle only as a reproduction check;
* identity checks: the centre loss and gradient norm of every objective must
  reproduce the existing centre-frozen ordinary Hessian record.
"""
from __future__ import annotations

import csv
import glob
import json

from landscape_v3 import common as V3

from . import common as C


def _v3_points():
    pts = {}
    for f in glob.glob(str(C.V3_RAW / "*" / "v3" / "eval" / "*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "splits" in r and "key" in r:
                pts.setdefault(r["key"], {"splits": r["splits"], "file": f})
    return pts


def main():
    pts = _v3_points()
    probs = {r["problem"]: r for r in csv.DictReader(open(C.V3_TABLES / "D_hessian_problems.csv"))}
    objectives, identity = [], {}
    for s in C.SEEDS:
        for probe in C.PROBES:
            for m in C.METHODS:
                oid = C.objective_id(m, s, probe)
                pr = probs[V3.hess_problem(m, s, probe, C.CFROZEN, "ordinary")]
                identity[oid] = {"problem": pr["problem"], "loss": float(pr["loss"]),
                                 "grad_norm": float(pr["grad_norm"]), "n_params": int(pr["n_params"]),
                                 "n_blocks": int(pr["n_nonzero_blocks"])}
                objectives.append({"id": oid, "m": m, "s": s, "probe": probe})
    axis = list(C.GRID_AXIS)
    grid, reuse, inventory = [], [], {}
    for m in C.METHODS:
        tgt = V3.target(m)
        todo, have = [], []
        for a in axis:
            for b in axis:
                if a == 0 and b == 0:
                    spec = "c"
                elif b == 0:
                    spec = V3.spec_r1(0, a)
                elif a == 0:
                    spec = V3.spec_r1(1, b)
                else:
                    spec = None
                key = None if spec is None else V3.pkey(m, 0, C.FINAL, tgt, C.CFROZEN, "probes", spec)
                if key and key in pts:
                    have.append({"a": a, "b": b, "source_key": key,
                                 "expected": {sp: pts[key]["splits"][sp]["ce"] for sp in C.PROBES},
                                 "source_file": pts[key]["file"].split("studies")[-1]})
                else:
                    todo.append([a, b])
        # the grid key itself must not already exist under centre-frozen BN
        clash = [ab for ab in todo if C.grid_key(m, *ab) in pts]
        assert not clash, clash
        pw = sum(1 for a in axis for b in axis
                 if V3.pkey(m, 0, C.FINAL, tgt, C.POINTWISE, "probes", V3.spec_r2(0, 1, a, b)) in pts)
        inventory[m] = {"grid_vertices": len(axis) ** 2, "reused_centre_frozen_axis_points": len(have),
                        "to_compute": len(todo), "pointwise_vertices_in_v3_raw": pw,
                        "pointwise_vertices_reused_from_v2": len(axis) ** 2 - pw}
        grid.append({"id": "cfgrid__%s__seed0" % C.SHORT[m], "m": m, "points": todo})
        reuse.append({"m": m, "points": have})
    bundle = {"objectives": objectives, "identity": identity, "grid": grid, "reuse_checks": reuse,
              "constants": {"trace_stages": C.TRACE_STAGES, "trace_sem_rel": C.TRACE_SEM_REL,
                            "trace_stream": C.TRACE_STREAM, "grid_axis": axis, "grid_pair": C.GRID_PAIR,
                            "reuse_tolerance_ce": 1e-5, "identity_tolerance_rel": 1e-6}}
    C.PROTO.mkdir(parents=True, exist_ok=True)
    (C.PROTO / "tasks.json").write_text(json.dumps(bundle, indent=1))
    (C.PROTO / "reuse_inventory.json").write_text(json.dumps(inventory, indent=1))
    print(json.dumps(inventory, indent=1))
    print("objectives", len(objectives), "grid points", sum(len(g["points"]) for g in grid),
          "reuse checks", sum(len(r["points"]) for r in reuse), "tasks sha", C.sha_json(bundle))


if __name__ == "__main__":
    main()
