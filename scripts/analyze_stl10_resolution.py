"""Offline analysis of the STL-10 progressive-resolution job.

Reads ``results/kaggle_outputs/stl10-resolution-*/stl10_resolution_*/*/{summary,metrics}.json``
and reports: final accuracy per arm and seed with paired contrasts against R96,
measured training seconds per arm (the time question), seconds per epoch by
resolution, accuracy at matched *time* (interpolated on the cumulative training
clock), and the BN-recalibrated target path during the coarse stages.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(slug):
    runs = {}
    for p in glob.glob(str(ROOT / "results" / "kaggle_outputs" / slug / "stl10_resolution_*" / "*" / "summary.json")):
        s = json.load(open(p))
        m = json.load(open(Path(p).parent / "metrics.json"))
        runs[s["label"]] = {"summary": s, "metrics": m}
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="stl10-resolution-*")
    ap.add_argument("--out", default=str(ROOT / "results" / "stl10_resolution_analysis.md"))
    a = ap.parse_args()
    runs = load(a.slug)
    by_arm = {}
    for label, r in runs.items():
        by_arm.setdefault(r["summary"]["schedule"], {})[r["summary"]["seed"]] = r
    seeds = sorted({s for arm in by_arm.values() for s in arm})
    L = ["# STL-10 progressive resolution — analysis\n", "loaded %d runs" % len(runs), ""]

    L.append("## Final test accuracy (%), train seconds, nominal compute units\n")
    L.append("| arm | epochs | acc per seed | mean | vs R96 per seed | mean diff | train s | s / nominal unit |")
    L.append("|---|---:|---|---:|---|---:|---:|---:|")
    ref = by_arm.get("R96", {})
    ARMS = [a for a in ("R96", "Rprog", "Rsteps4", "Rlin24", "Rlin24eq", "R96_72", "Rprogeq") if a in by_arm]
    for arm in ARMS:
        cells = by_arm[arm]
        accs = [cells[s]["summary"]["final_test_acc"] * 100 for s in seeds if s in cells]
        diffs = [cells[s]["summary"]["final_test_acc"] * 100 - ref[s]["summary"]["final_test_acc"] * 100
                 for s in seeds if s in cells and s in ref]
        ts = [cells[s]["summary"]["train_seconds"] for s in seeds if s in cells]
        units = cells[seeds[0]]["summary"]["compute_units_nominal"]
        L.append("| %s | %d | %s | %.2f | %s | %+.2f | %.0f | %.1f |" % (
            arm, cells[seeds[0]]["summary"]["epochs"], " / ".join("%.2f" % v for v in accs), np.mean(accs),
            " / ".join("%+.2f" % v for v in diffs) if diffs else "-", np.mean(diffs) if diffs else 0.0,
            np.mean(ts), np.mean(ts) / units))

    L.append("\n## Seconds per training epoch by resolution (median over epochs and seeds)\n")
    L.append("| resolution | s / epoch | relative to 96 | nominal (r/96)^2 |\n|---:|---:|---:|---:|")
    per_res = {}
    for r in runs.values():
        for res, sec in zip(r["summary"]["resolution_by_epoch"], r["summary"]["epoch_seconds"]):
            per_res.setdefault(res, []).append(sec)
    s96 = np.median(per_res.get(96, [np.nan]))
    for res in sorted(per_res):
        med = np.median(per_res[res])
        L.append("| %d | %.2f | %.2f | %.2f |" % (res, med, med / s96, (res / 96.0) ** 2))

    L.append("\n## Test accuracy (target path, %) at matched training time — interpolated on each run's clock\n")
    if ref:
        t_ref = np.mean([ref[s]["summary"]["train_seconds"] for s in ref])
        marks = [0.25, 0.5, 0.75, 1.0]
        L.append("| arm | " + " | ".join("t = %.0f%% of R96 (%.0f s)" % (m * 100, m * t_ref) for m in marks) + " | final |")
        L.append("|---|" + "---:|" * (len(marks) + 1))
        for arm in ARMS:
            vals = []
            for m in marks:
                per_seed = []
                for s, r in by_arm[arm].items():
                    t = np.array([rec["train_seconds_cumulative"] for rec in r["metrics"]])
                    acc = np.array([rec["test_acc_target"] for rec in r["metrics"]]) * 100
                    tt = m * t_ref
                    per_seed.append(float(np.interp(tt, t, acc)) if tt <= t[-1] else np.nan)
                vals.append(np.nanmean(per_seed))
            fin = np.mean([r["summary"]["final_test_acc"] * 100 for r in by_arm[arm].values()])
            L.append("| %s | %s | %.2f |" % (arm, " | ".join("%.2f" % v if not np.isnan(v) else "ended" for v in vals), fin))

    L.append("\n## Target path (96x96) during coarse stages, seed 0: current / target / target after BN recal (%)\n")
    L.append("| arm | epoch | r | current | target | target recal |\n|---|---:|---:|---:|---:|---:|")
    for arm in ("Rprog", "Rsteps4", "Rlin24"):
        if arm in by_arm and 0 in by_arm[arm]:
            for rec in by_arm[arm][0]["metrics"]:
                if rec["resolution_current"] != 96 and rec["test_acc_target_bnrecal"] is not None:
                    L.append("| %s | %d | %d | %.1f | %.1f | %.1f |" % (
                        arm, rec["epoch"], rec["resolution_current"], 100 * rec["test_acc_current"],
                        100 * rec["test_acc_target"], 100 * rec["test_acc_target_bnrecal"]))

    L.append("\n## Generalisation gap at the end (test CE − train-probe CE), mean over seeds\n")
    for arm in ARMS:
        if arm in by_arm:
            g = np.mean([r["summary"]["final_test_ce"] - r["summary"]["final_train_probe_ce"] for r in by_arm[arm].values()])
            L.append("- %s: %.3f" % (arm, g))
    txt = "\n".join(L) + "\n"
    Path(a.out).write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
