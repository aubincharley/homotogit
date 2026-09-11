"""Aggregate the unified batch: per-seed finals, mean, sample SD, paired deltas.

Every arm in this batch shares one pinned asset set, so the difference against
``plain`` is genuinely paired seed-by-seed.  Curves are the **current** path,
the convention used everywhere else in this repo.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.unified_manifest import build_configs

ROOT = Path("results/kaggle_outputs")
OUT = Path("results/unified_summary.json")


def load():
    cells = {}
    for s in ROOT.glob("unified-j*/*/*/summary.json"):
        d = json.loads(s.read_text())
        m = json.loads((s.parent / "metrics.json").read_text())
        cells[d["cell_id"]] = {"summary": d, "metrics": m}
    return cells


def main():
    cells = load()
    cfgs = {c["id"]: c for c in build_configs()}
    by_id = defaultdict(dict)
    for cid, c in cells.items():
        by_id[c["summary"]["id"]][c["summary"]["seed"]] = c

    rows = {}
    for cid, cfg in cfgs.items():
        seeds = by_id.get(cid, {})
        acc = {s: v["summary"]["final_test_acc"] for s, v in sorted(seeds.items())}
        if not acc:
            continue
        vals = list(acc.values())
        rows[cid] = {
            "label": cfg["label"], "group": cfg["group"],
            "n_seeds": len(vals),
            "per_seed": acc,
            "mean": st.mean(vals),
            "sd": st.stdev(vals) if len(vals) > 1 else 0.0,
            "final_train_probe_ce": st.mean(
                [v["summary"]["final_train_probe_ce"] for v in seeds.values()]),
            "epoch_test_acc_current": [
                st.mean([seeds[s]["metrics"][e]["test_acc_current"] for s in acc])
                for e in range(len(next(iter(seeds.values()))["metrics"]))],
            "epoch_train_loss": [
                (None if seeds[min(acc)]["metrics"][e]["train_loss_epoch"] is None
                 else st.mean([seeds[s]["metrics"][e]["train_loss_epoch"] for s in acc]))
                for e in range(len(next(iter(seeds.values()))["metrics"]))],
        }

    base = by_id["plain"]
    for cid, r in rows.items():
        if cid == "plain":
            r["delta_vs_plain"] = 0.0
            r["delta_sd"] = 0.0
            continue
        common = sorted(set(r["per_seed"]) & set(base))
        d = [r["per_seed"][s] - base[s]["summary"]["final_test_acc"] for s in common]
        r["delta_vs_plain"] = st.mean(d)
        r["delta_sd"] = st.stdev(d) if len(d) > 1 else 0.0
        r["delta_n"] = len(d)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))

    order = sorted(rows, key=lambda k: -rows[k]["mean"])
    print("%-22s %-5s %6s %6s %8s %s" % ("id", "grp", "mean%", "sd", "d_plain", "label"))
    for cid in order:
        r = rows[cid]
        print("%-22s %-5s %6.2f %6.2f %+8.2f %s"
              % (cid, r["group"], 100 * r["mean"], 100 * r["sd"],
                 100 * r["delta_vs_plain"], r["label"][:52]))
    print("\n%d configurations, %d cells" % (len(rows), len(cells)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
