"""Recover and reconcile Idriss's augmentation records (branch continuation_new_loss).

    py -m overnight.recover_aug <extracted tree of continuation_new_loss@badce2a> <out dir>

Reads every ``cc-aug-*`` (30 epochs) and ``cc-a60-*`` (60 epochs) run summary, cross-checks it with
the shard JSON written by the same kernel, checks the design (method x seed x corner coverage),
recomputes seed-paired gains over Plain, compares them with ``results/augment_report.json`` and the
tables of ``docs/AUGMENTATION.md``, and checks the 60-epoch asset set against the pinned one.
Nothing is retrained.  All metrics are the historical per-epoch evaluation (saved BN statistics,
target path, 500-image training probe, full test set).
"""
from __future__ import annotations

import glob
import hashlib
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

import numpy as np

METHODS = ("plain", "resolution_max_b1", "gaussian_postrelu", "resolution_max_b1_gaussian_conv")
SHORT = {"resolution_max_b1": "R", "gaussian_postrelu": "G", "resolution_max_b1_gaussian_conv": "RG"}
COMMIT = "badce2a"

#: AUGMENTATION.md tables as committed (baseline test %, baseline train %, R, G, RG gains in pp)
DOC = {("crop_flip", 30, 0.005): (79.78, 81.73, 0.42, 0.86, 1.32),
       ("crop_flip", 30, 0.01): (84.70, 87.93, 0.09, -0.96, 0.25),
       ("none", 60, 0.01): (80.27, 100.00, 3.25, 2.57, 4.24),
       ("crop_flip", 60, 0.005): (85.51, 89.40, 0.19, -0.09, 0.52),
       ("crop_flip", 60, 0.01): (88.27, 94.20, 0.19, -0.44, 0.12)}


def main(src: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rows, problems = [], []
    for p in sorted(glob.glob(str(src / "results/kaggle_outputs/cc-a*/runs/*/summary.json"))):
        p = Path(p)
        job = p.parts[-4]
        if not (job.startswith("cc-aug-") or job.startswith("cc-a60-")):
            continue
        s = json.loads(p.read_text())
        cfg = json.loads(p.with_name("config.json").read_text())
        method, aug, ep, lr, seed = p.parent.name.split("__")
        f = s["final"]
        rows.append({"source_commit": COMMIT, "kaggle_job": job, "record": p.relative_to(src).as_posix(),
                     "method": method, "augmentation": aug, "epochs": int(ep[1:]), "lr": float(lr[2:]), "seed": int(seed[4:]),
                     "optimizer": cfg["optimizer"]["name"], "weight_decay": cfg["optimizer"]["weight_decay"],
                     "schedule": cfg["optimizer"]["schedule"], "loss": cfg.get("loss", {}).get("name"),
                     "method_schedule_stretched": False, "updates": s["updates"], "summary_epochs": s["epochs"],
                     "test_acc": f["target"]["test"]["acc"], "test_ce": f["target"]["test"]["ce"],
                     "trainprobe500_acc": f["target"]["train_probe"]["acc"], "trainprobe500_ce": f["target"]["train_probe"]["ce"],
                     "bn_eval": "saved running statistics (historical per-epoch evaluation)", "path": "target",
                     "assets_dir": (cfg.get("assets") or {}).get("dir")})
        if s["epochs"] != int(ep[1:]) or cfg["data"]["augmentation"] != aug or abs(cfg["optimizer"]["lr"] - float(lr[2:])) > 1e-12:
            problems.append("config/name mismatch: %s" % p.parent.name)
    # shard JSON cross-check
    shard_cells = {}
    for sp in glob.glob(str(src / "results/kaggle_outputs/cc-a*/augment_shard*.json")):
        job = Path(sp).parts[-2]
        for c in json.loads(Path(sp).read_text())["cells"]:
            shard_cells[(job, c["cell"])] = c["final"]["target"]["test"]["acc"]
    for r in rows:
        key = (r["kaggle_job"], "__".join([r["method"], r["augmentation"], "e%d" % r["epochs"], "lr%g" % r["lr"], "seed%d" % r["seed"]]))
        v = shard_cells.get(key)
        r["shard_json_test_acc"] = v
        r["shard_json_matches_summary"] = v is not None and abs(v - r["test_acc"]) < 1e-12
    import csv
    with open(out / "recovered_augmentation_runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # coverage and paired gains
    corners = sorted({(r["augmentation"], r["epochs"], r["lr"]) for r in rows})
    cov, gains = [], []
    rep = {tuple(e["corner"][:3]): e for e in json.loads((src / "results/augment_report.json").read_text())}
    for c in corners:
        rs = [r for r in rows if (r["augmentation"], r["epochs"], r["lr"]) == c]
        by = {(r["method"], r["seed"]): r for r in rs}
        dup = len(rs) - len(by)
        cov.append({"augmentation": c[0], "epochs": c[1], "lr": c[2], "records": len(rs), "duplicates": dup,
                    **{"%s_seeds" % (SHORT.get(m, m)): ",".join(str(s) for s in (0, 1, 2) if (m, s) in by) for m in METHODS},
                    "complete_4x3": all((m, s) in by for m in METHODS for s in (0, 1, 2))})
        base = [100 * by[("plain", s)]["test_acc"] for s in (0, 1, 2) if ("plain", s) in by]
        base_tr = [100 * by[("plain", s)]["trainprobe500_acc"] for s in (0, 1, 2) if ("plain", s) in by]
        for m in METHODS[1:]:
            seeds = [s for s in (0, 1, 2) if (m, s) in by and ("plain", s) in by]
            d = [100 * (by[(m, s)]["test_acc"] - by[("plain", s)]["test_acc"]) for s in seeds]
            dce = [by[(m, s)]["test_ce"] - by[("plain", s)]["test_ce"] for s in seeds]
            e = rep.get(c)
            doc = DOC.get(c)
            g = {"augmentation": c[0], "epochs": c[1], "lr": c[2], "arm": SHORT[m], "n_pairs": len(d), "seeds": ",".join(map(str, seeds)),
                 "plain_test_acc_mean_pct": st.fmean(base) if base else None, "plain_trainprobe500_acc_mean_pct": st.fmean(base_tr) if base_tr else None,
                 "arm_test_acc_mean_pct": st.fmean(100 * by[(m, s)]["test_acc"] for s in seeds) if seeds else None,
                 **{"diff_seed%d_pp" % s: (100 * (by[(m, s)]["test_acc"] - by[("plain", s)]["test_acc"]) if s in seeds else None) for s in (0, 1, 2)},
                 "diff_mean_pp": st.fmean(d) if d else None, "diff_sd_pp": st.stdev(d) if len(d) > 1 else None,
                 "n_positive": sum(x > 0 for x in d), "n_negative": sum(x < 0 for x in d),
                 "test_ce_diff_mean": st.fmean(dce) if dce else None,
                 "report_json_gain_pp": e["gains"].get(SHORT[m]) if e else None,
                 "report_json_matches": (e is not None and abs(e["gains"][SHORT[m]] - st.fmean(d)) < 1e-9) if d else None,
                 "doc_table_gain_pp": doc[2 + ("R", "G", "RG").index(SHORT[m])] if doc else None,
                 "doc_table_matches_2dp": (doc is not None and round(st.fmean(d), 2) == doc[2 + ("R", "G", "RG").index(SHORT[m])]) if (d and doc) else None,
                 "doc_baseline_matches_2dp": (doc is not None and round(st.fmean(base), 2) == doc[0] and round(st.fmean(base_tr), 2) == doc[1]) if (base and doc) else None}
            gains.append(g)
    for name, tab in (("recovered_augmentation_coverage.csv", cov), ("recovered_augmentation_paired_gains.csv", gains)):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(tab[0]))
            w.writeheader()
            w.writerows(tab)
    # launch-only jobs
    launch_only = sorted(Path(j).name for j in glob.glob(str(src / "results/kaggle_outputs/cc-aug60-*"))
                         if not glob.glob(str(Path(j) / "runs/*/summary.json")))
    # 60-epoch asset set vs pinned set
    assets = {}
    try:
        repo = Path(__file__).resolve().parents[1]
        pinned = np.load(repo / "assets/cifar10_resnet20bn/shared_indices.npz")
        raw = subprocess.run(["git", "show", "%s:assets/cifar10_resnet20bn_e60/shared_indices.npz" % COMMIT], capture_output=True,
                             cwd=str(repo.parents[0] / "overnight")).stdout
        import io
        e60 = np.load(io.BytesIO(raw))
        for s in (0, 1, 2):
            a, b = pinned["perm_seed%d" % s], e60["perm_seed%d" % s]
            assets["perm_seed%d" % s] = {"e60_shape": list(b.shape), "first_30_epochs_equal_pinned": bool(np.array_equal(a, b[:30]))}
        man = json.loads(subprocess.run(["git", "show", "%s:assets/cifar10_resnet20bn_e60/assets_manifest.json" % COMMIT], capture_output=True,
                                        text=True, cwd=str(repo)).stdout)
        pin = json.loads((repo / "assets/cifar10_resnet20bn/assets_manifest.json").read_text())
        assets["init_states_equal_pinned"] = man["states"] == pin["states"]
        assets["subset_and_probe_equal"] = all(man["arrays"][k] == pin["arrays"][k] for k in ("subset", "train_probe"))
        assets["generator_note"] = man.get("generated_by") or man.get("note") or "not stated in manifest"
    except Exception as exc:                                    # reported, not fatal
        assets["error"] = repr(exc)
    summary = {"source": "origin/continuation_new_loss @ %s (pushed 2026-09-17 18:01 +0200)" % COMMIT,
               "previous_export": "analyst_bundles/all_results_2026-09-17 was built from f0daed0, which held only the 12 records of shards s1/s3; "
                                  "shards s0/s2 and the 60-epoch runs were committed later in badce2a",
               "records": len(rows), "by_epochs": {str(e): sum(r["epochs"] == e for r in rows) for e in (30, 60)},
               "shard_json_all_match": all(r["shard_json_matches_summary"] for r in rows),
               "corners": cov, "launch_only_jobs_without_outputs": launch_only,
               "report_json_all_match": all(g["report_json_matches"] for g in gains if g["report_json_matches"] is not None),
               "doc_tables_all_match_2dp": all(g["doc_table_matches_2dp"] for g in gains if g["doc_table_matches_2dp"] is not None),
               "doc_baselines_all_match_2dp": all(g["doc_baseline_matches_2dp"] for g in gains if g["doc_baseline_matches_2dp"] is not None),
               "assets_60_epoch": assets, "problems": problems,
               "not_verifiable": ["Kaggle outputs of idrisselk / idrisselkhamlichi accounts (not accessible from our accounts); checkpoints of these runs are not in git",
                                  "cc-aug60-* jobs: launch.json and plan only; no run records anywhere accessible"]}
    (out / "recovered_augmentation_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("records", "by_epochs", "shard_json_all_match", "report_json_all_match",
                                              "doc_tables_all_match_2dp", "doc_baselines_all_match_2dp", "launch_only_jobs_without_outputs",
                                              "assets_60_epoch", "problems")}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
