"""Aggregate the optimizer campaign into the paired continuation gain.

    py scripts/aggregate_optimizers.py

Reads every `job_summary.json` under `results/kaggle_outputs/*/optbench_*/`,
adds the imported SGD arms, and writes `results/optimizer_benchmark/` :
`runs.csv`, `delta.json`, and the tables that go into
`docs/OPTIMIZER_BENCHMARK.md`.

The quantity
------------
Absolute accuracy drifts between GPU sessions: the same `plain` arm, same seed,
same pinned assets, has spanned 0.73 pt across batches. So the reported quantity
is the paired continuation gain

    D(method, opt, seed) = acc(method, opt, seed) - acc(plain, opt, seed)

computed **inside** one optimizer and one seed, where both terms drifted
together and the drift cancels.

That pairing is also why the SGD row must use the *imported* `plain`, not the
drift-control `plain` run here: the imported methods and the imported `plain`
come from one batch, and differencing across batches is exactly what the pairing
exists to avoid. The drift control's job is to say whether the import is sound
at all, and it is reported separately.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUT = REPO_ROOT / "results" / "optimizer_benchmark"
KAGGLE = REPO_ROOT / "results" / "kaggle_outputs"
METHODS = ("plain", "resolution_max_b1", "gaussian_postrelu",
           "resolution_max_b1_gaussian_conv")
OPTIMIZERS = ("sgd", "adam", "adamw", "radam")


def imported_sgd() -> list:
    """The unified_selected arms, read from the frozen method definitions."""
    from continuation_core.methods import METHODS as SPECS
    rows = []
    for mid, spec in SPECS.items():
        src = spec.source
        for seed, acc in src["final_test_acc_per_seed"].items():
            rows.append({"method": mid, "optimizer": "sgd", "lr": 0.005,
                         "seed": int(seed), "acc": acc * 100.0,
                         "imported": True, "batch": src["experiment"],
                         "source_config_id": src["config_id"],
                         "source_branch": src["branch"],
                         "asset_set": src["asset_set"]})
    return rows


def measured() -> list:
    rows = []
    for js in sorted(KAGGLE.glob("*/optbench_*/job_summary.json")):
        batch = json.loads(js.read_text(encoding="utf-8"))
        for r in batch["rows"]:
            if r["status"] not in ("complete", "already_complete"):
                continue
            if "final_test_acc" not in r:
                continue
            rows.append({"method": r["method"], "optimizer": r["optimizer"],
                         "lr": r["lr"], "seed": r["seed"],
                         "acc": r["final_test_acc"] * 100.0, "imported": False,
                         "batch": batch["batch"], "kernel": js.parent.parent.name,
                         "wall_seconds": r.get("wall_seconds")})
    return rows


def paired_delta(rows: list) -> dict:
    """D = method - plain, inside one optimizer, one seed and one batch."""
    by = {}
    for r in rows:
        by.setdefault((r["optimizer"], r["batch"], r["seed"]), {})[r["method"]] = r["acc"]
    out = {}
    for (opt, batch, seed), accs in by.items():
        if "plain" not in accs:
            continue
        for method, acc in accs.items():
            if method == "plain":
                continue
            out.setdefault((opt, method), []).append(
                {"seed": seed, "batch": batch, "delta": acc - accs["plain"],
                 "acc": acc, "plain": accs["plain"]})
    return out


def summarise(values: list) -> dict:
    d = [v["delta"] for v in values]
    return {"n": len(d), "mean": statistics.mean(d),
            "sd": statistics.stdev(d) if len(d) > 1 else None,
            "per_seed": sorted(values, key=lambda v: v["seed"]),
            "same_sign": all(x > 0 for x in d) or all(x < 0 for x in d)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = imported_sgd() + measured()

    with (OUT / "runs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)

    deltas = {"%s|%s" % k: summarise(v) for k, v in paired_delta(rows).items()}
    (OUT / "delta.json").write_text(json.dumps(deltas, indent=2, default=str),
                                    encoding="utf-8")

    print("runs: %d (%d imported, %d measured)"
          % (len(rows), sum(r["imported"] for r in rows),
             sum(not r["imported"] for r in rows)))
    print()

    # The accuracy table is campaign arms only.  The sweep ran `plain` at four
    # learning rates and the drift control ran it again in a later session;
    # averaging those together would report a number belonging to no arm.
    campaign = [r for r in rows if r["batch"] in ("grid", "unified_selected")]
    drift = [r for r in rows if r["batch"] == "sgd_control"]

    print("Final test accuracy, mean over seeds (campaign arms only)")
    print("%-34s %s" % ("method", "".join("%12s" % o for o in OPTIMIZERS)))
    acc = {}
    for r in campaign:
        acc.setdefault((r["method"], r["optimizer"]), []).append(r["acc"])
    for m in METHODS:
        cells = []
        for o in OPTIMIZERS:
            v = acc.get((m, o))
            cells.append("%12s" % ("%.2f" % statistics.mean(v) if v else "-"))
        print("%-34s %s" % (m, "".join(cells)))
    print()

    print("Paired continuation gain, mean over seeds (method - plain)")
    print("%-34s %s" % ("method", "".join("%12s" % o for o in OPTIMIZERS)))
    for m in METHODS[1:]:
        cells = []
        for o in OPTIMIZERS:
            d = deltas.get("%s|%s" % (o, m))
            cells.append("%12s" % ("%+.2f" % d["mean"] if d else "-"))
        print("%-34s %s" % (m, "".join(cells)))
    print()

    if drift:
        imported = {r["seed"]: r["acc"] for r in rows
                    if r["imported"] and r["method"] == "plain"}
        print("SGD drift control: does the import still reproduce?")
        for r in sorted(drift, key=lambda r: r["seed"]):
            o = imported.get(r["seed"])
            print("  seed %d  rerun %.2f  imported %.2f  %+.2f pt"
                  % (r["seed"], r["acc"], o, r["acc"] - o))
        print()
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
