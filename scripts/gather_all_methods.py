"""Collect every comparable trained arm into one table, from the experiment index.

Reads ``experiments/index.json`` and, for curves, the metrics files it points at
(working tree or teammates' pinned commits through ``scripts/records.py``).
Nothing is read from anyone's scratch directory.

Included: configurations trained under the reference conditions (50,000 train /
10,000 test, 30 epochs) with at least one attempted seed. Diverged
configurations are kept with ``status = "diverged"`` so figures can show them.

Batches are kept distinguishable. ``campaign``, ``resbench``, ``unified`` and
Aubin's seed-0 runs share the pinned asset set ``r20bn-campaign-assets``; the
ablation batch shares its data order but not its initial weights. Same-asset
plain runs still differ across batches (docs/AUDIT.md A2).

Writes ``results/all_methods.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.records import Location, Reader

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "experiments" / "index.json"
OUT = ROOT / "results" / "all_methods.json"

FAMILY = {
    "plain": "Baseline (no intervention)",
    "gauss": "Internal Gaussian",
    "res": "Resolution reduction",
    "res_gauss": "Resolution + Gaussian",
    "profile": "Per-layer sigma profile",
    "operator": "Reduction operator / placement",
    "adaptive": "Adaptive (data-driven) schedule",
    "constant": "Constant filter (architecture)",
    "mix": "Identity-Gaussian mixture",
    "control": "Fixed-resolution control",
}

BATCH = {"campaign_grid21": "campaign", "resbench_resolution_only": "resbench",
         "ablation_aa": "ablation", "adaptive_continuation": "adaptive",
         "per_layer_sigma": "adaptive", "unified_selected": "unified"}


def _unified_family(ctrl: dict) -> str:
    blur = ctrl.get("operator") == "gaussian"
    red = set(ctrl.get("resolution_by_epoch") or [32]) != {32}
    if not blur and not red:
        return "plain"
    if blur and not red:
        return "gauss"
    if red and not blur:
        return "res"
    return "profile" if ctrl.get("sigma_profile") else "res_gauss"


def main():
    ix = json.loads(INDEX.read_text(encoding="utf-8"))
    cells_by_cfg = {}
    for c in ix["cells"]:
        cells_by_cfg.setdefault((c["experiment"], c["config_id"], c["epochs"],
                                 c["n_train"], c["n_eval"]), []).append(c)
    rows = {}
    reader = Reader()
    for cfg in ix["configurations"]:
        batch = BATCH.get(cfg["experiment"])
        rc = cfg["run_conditions"]
        if batch is None or (rc["epochs"], rc["n_train"], rc["n_eval"]) != (30, 50000, 10000):
            continue
        fam = cfg["family"]
        if cfg["experiment"] == "unified_selected":
            fam = _unified_family(cfg["controller"] or {})
        fam = fam or "adaptive"
        key = "%s/%s" % (batch, cfg["config_id"])
        if any(g for g in rc["groups"] if "launch" in cfg["label"]):
            key += "@" + rc["groups"][0].rsplit("/", 1)[-1]
        curves = {}
        for c in cells_by_cfg.get((cfg["experiment"], cfg["config_id"], rc["epochs"],
                                   rc["n_train"], rc["n_eval"]), []):
            if c["group"] not in rc["groups"] or not c["metrics"]:
                continue
            try:
                curves[c["seed"]] = reader.read_json(Location(c["metrics"]["path"],
                                                              c["metrics"]["ref"]))
            except Exception:
                continue
        valid = [v for s, v in cfg["acc_per_seed"].items()
                 if int(s) in cfg["seeds_valid"] and v is not None]
        rows[key] = {"key": key, "label": cfg["label"], "family": fam, "batch": batch,
                     "status": cfg["status"],
                     "acc_mean": cfg["acc_mean"], "acc_sd": cfg["acc_sd"] or 0.0,
                     "sd_status": cfg["sd_status"],
                     "acc_per_seed": sorted(valid), "n_seeds": cfg["n_valid"],
                     "n_attempted": len(cfg["seeds_attempted"]),
                     "epochs": 30, "wall_mean": cfg["wall_seconds_mean"],
                     "final_path": cfg["final_path"], "flags": cfg["flags"],
                     "curves": curves}
    reader.close()
    payload = {"source": "experiments/index.json @ %s" % ix["built_from_commit"][:10],
               "n_configurations": len(rows), "families": FAMILY,
               "configurations": {k: {kk: vv for kk, vv in v.items() if kk != "curves"}
                                  for k, v in rows.items()}}
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    print("configurations: %d (diverged %d)" % (len(rows), sum(r["status"] == "diverged"
                                                                  for r in rows.values())))
    print("by batch:", dict(Counter(r["batch"] for r in rows.values())))
    print("with curves:", sum(1 for r in rows.values() if r["curves"]))
    return rows


if __name__ == "__main__":
    main()
