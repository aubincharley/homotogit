"""Compact results bundle for an external analyst (<= 20 files).

    py -m landscape_v2.bundle

Consolidates ``studies/landscape_v2/raw`` into a few long CSV tables in
``studies/landscape_v2/analyst_bundle/``.  Every row is a measured value; no
aggregation except the tables whose name says so.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from . import common as C

RAW = C.STUDY / "raw"
OUT = C.STUDY / "analyst_bundle"
SPLITS = ("train_probe", "test_probe", "train_large", "test_full", "train_full")


def eval_rows(pattern):
    for acc in sorted(RAW.glob("*/v2")):
        for p in sorted((acc / "eval").glob(pattern)):
            tid = p.stem
            by_key = {}
            # fixed1d re-evaluated each centre once per direction (same values,
            # checked); keep one row per evaluated point
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    by_key.setdefault(r["key"], r)
            for r in by_key.values():
                yield tid, r


def split_cols(r, splits=SPLITS):
    out = {}
    for s in splits:
        v = r.get("splits", {}).get(s)
        out[s + "_ce"] = v["ce"] if v else ""
        out[s + "_acc"] = v["acc"] if v else ""
    return out


def write(name, rows):
    rows = list(rows)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def parse_id(tid):
    parts = tid.split("__")
    long = {v: k for k, v in C.SHORT.items()}
    return parts, long


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    counts = {}
    long = {v: k for k, v in C.SHORT.items()}

    # 1. primary sensitivity, one row per evaluation (centres included, amplitude 0)
    rows = []
    for tid, r in eval_rows("sens1d__*.jsonl"):
        if r["key"].startswith("norm|"):
            continue
        _, m, s = tid.split("__")
        rows.append({"method": long[m], "seed": int(s[4:]), "policy": r["policy"], "direction": r["k"],
                     "amplitude": r["amp"], "sign": r["sign"],
                     **split_cols(r, ("train_probe", "test_probe"))})
    counts["sensitivity_evaluations.csv"] = write("sensitivity_evaluations.csv", rows)

    # 2. validation on larger sets + calibration-size check
    rows = []
    for tid, r in eval_rows("validation__*.jsonl"):
        _, m, s = tid.split("__")
        rows.append({"check": "larger_sets", "method": long[m], "seed": int(s[4:]), "policy": r["policy"],
                     "calibration": r.get("calibration") or "", "direction": r["k"], "amplitude": r["amp"],
                     "sign": r["sign"], **split_cols(r)})
    for tid, r in eval_rows("calibsens__*.jsonl"):
        _, m, s = tid.split("__")
        rows.append({"check": "calibration_10k", "method": long[m], "seed": int(s[4:]), "policy": "recalibrated",
                     "calibration": "calib10k", "direction": r["k"], "amplitude": r["amp"], "sign": r["sign"],
                     **split_cols(r)})
    counts["validation_evaluations.csv"] = write("validation_evaluations.csv", rows)

    # 3. interpolation curves
    rows = []
    for tid, r in eval_rows("interp__*.jsonl"):
        _, pair, s = tid.split("__")
        a, b = pair.split("_")
        rows.append({"A": long[a], "B": long[b], "seed": int(s[4:]), "alpha": r["alpha"],
                     **split_cols(r, ("train_probe", "test_probe"))})
    counts["interpolation_curves.csv"] = write("interpolation_curves.csv", rows)

    # 4. fixed-weight comparisons (1-D, per direction)
    rows = []
    for tid, r in eval_rows("fixed1d__*.jsonl"):
        _, m, s, u = tid.split("__")
        rows.append({"method": long[m], "seed": int(s[4:]), "checkpoint_update": int(u[1:]), "state_label": r["label"],
                     "resolution": r["state"]["resolution"], "sigma_G": r["state"]["sigma"], "policy": r["policy"],
                     "direction": r["k"], "amplitude": r["amp"], "sign": r["sign"],
                     **split_cols(r, ("train_probe", "test_probe"))})
    counts["fixed_weight_evaluations.csv"] = write("fixed_weight_evaluations.csv", rows)

    # 5. surfaces (final solutions and fixed weights), one row per grid point
    rows = []
    for tid, r in eval_rows("surface__*.jsonl"):
        _, m, s, g = tid.split("__")
        rows.append({"kind": "final", "method": long[m], "seed": int(s[4:]), "grid": int(g[1:]), "checkpoint_update": C.TOTAL_UPDATES,
                     "state_label": "final", "a_dir0": r["a"], "b_dir1": r["b"],
                     **split_cols(r, ("train_probe", "test_probe"))})
    for tid, r in eval_rows("fixedsurf__*.jsonl"):
        _, m, s, u, lab = tid.split("__")
        rows.append({"kind": "fixed_weight", "method": long[m], "seed": int(s[4:]), "grid": 21,
                     "checkpoint_update": int(u[1:]), "state_label": r["label"], "a_dir0": r["a"], "b_dir1": r["b"],
                     **split_cols(r, ("train_probe", "test_probe"))})
    counts["surfaces.csv"] = write("surfaces.csv", rows)

    # 6. actual checkpoint losses along training
    rows = []
    for tid, r in eval_rows("traj__*.jsonl"):
        _, m, s = tid.split("__")
        rows.append({"method": long[m], "seed": int(s[4:]), "checkpoint": r["file"], "update": r["global_update"],
                     "reason": r["reason"], "state_label": r["label"], "resolution": r["state"]["resolution"],
                     "sigma_G": r["state"]["sigma"], "policy": r["policy"],
                     **split_cols(r, ("train_probe", "test_probe"))})
    counts["checkpoint_losses.csv"] = write("checkpoint_losses.csv", rows)

    # 7. training records (recorded during training, saved BN statistics, pinned 500-image probe + full test)
    rows = []
    for acc in sorted(RAW.glob("*/v2")):
        for rd in sorted((acc / "runs").glob("*__seed*")):
            m, s = rd.name.split("__seed")
            for rec in json.loads((rd / "metrics.json").read_text()):
                su = rec.get("state_used") or {}
                st = su.get("state") or {}
                row = {"method": m, "seed": int(s), "epoch": rec["epoch"], "update": rec["update"],
                       "lr_last_update": rec["lr_last_update"], "train_loss_epoch": rec.get("train_loss_epoch"),
                       "state_resolution": st.get("resolution"), "state_sigma_G": st.get("sigma")}
                for path in ("current", "target"):
                    for sp, name in (("train_probe", "train_probe500"), ("test", "test_full")):
                        v = rec["eval"][path][sp]
                        row["%s_%s_ce" % (path, name)], row["%s_%s_acc" % (path, name)] = v["ce"], v["acc"]
                rows.append(row)
    counts["training_curves.csv"] = write("training_curves.csv", rows)

    # 8. PCA: per-checkpoint projections and variance
    rows, var = [], []
    for p in sorted(RAW.glob("*/v2/pca/pca_seed*.npz")):
        s = int(p.stem[8:])
        z = np.load(p, allow_pickle=True)
        for i in range(len(z["files"])):
            row = {"seed": s, "method": str(z["method"][i]), "checkpoint": str(z["files"][i]), "update": int(z["update"][i]),
                   "used_in_fit": bool(z["fit"][i]), "dist_to_mean": float(z["dist_to_mean"][i]),
                   "resid_2pc": float(z["resid2"][i]), "resid_5pc": float(z["resid5"][i]), "resid_10pc": float(z["resid10"][i])}
            row.update({"pc%d" % (k + 1): float(z["coords"][i, k]) for k in range(z["coords"].shape[1])})
            rows.append(row)
        var.append({"seed": s, **{"evr_pc%d" % (k + 1): float(v) for k, v in enumerate(z["explained_variance_ratio"][:10])}})
    counts["pca_checkpoints.csv"] = write("pca_checkpoints.csv", rows)
    counts["pca_variance.csv"] = write("pca_variance.csv", var)
    rows = []
    for tid, r in eval_rows("pcaplane__*.jsonl"):
        rows.append({"seed": int(tid.split("seed")[1]), "pc1": r["x"], "pc2": r["y"],
                     **split_cols(r, ("train_probe", "test_probe"))})
    counts["pca_plane_grid.csv"] = write("pca_plane_grid.csv", rows)

    # 9. integrity and compute, one JSON
    integ = {}
    for acc in sorted(RAW.glob("*/v2")):
        a = acc.parent.name
        integ[a] = {"seeds": C.ACCOUNT_SEEDS.get(a),
                    "timing_seconds": json.loads((acc / "timing.json").read_text()),
                    "download_verification": json.loads((RAW / ("verify_%s.json" % a)).read_text()),
                    "checks": {p.stem: json.loads(p.read_text())["summary"] for p in sorted((acc / "checks").glob("*.json"))},
                    "pairing": {p.stem: json.loads(p.read_text()) for p in sorted((acc / "runs").glob("pairing_seed*.json"))},
                    "task_status": {json.loads(p.read_text())["task"]["id"]: json.loads(p.read_text())["status"]
                                    for p in sorted((acc / "eval").glob("*.meta.json"))},
                    "runs": {rd.name: {k: json.loads((rd / "summary.json").read_text())[k] for k in ("final", "timing")}
                             for rd in sorted((acc / "runs").glob("*__seed*"))}}
    (OUT / "integrity.json").write_text(json.dumps(integ, indent=1))

    for name in ("REPORT.md",):
        shutil.copy2(C.STUDY / name, OUT / name)
    shutil.copy2(C.STUDY / "inputs" / "preregistration.json", OUT / "preregistration.json")
    for fig in ("S2_paired_differences.png", "D1_interpolation_segments.png"):
        shutil.copy2(C.STUDY / "figures" / fig, OUT / fig)

    readme = """# landscape_v2 results bundle

Four frozen methods x 5 paired training seeds (0-4), CIFAR-10 / ResNet-20, 30 epochs.
Choices fixed beforehand: preregistration.json. Interpretation: REPORT.md.

Common columns
- method: plain | resolution_max_b1 | gaussian_postrelu | resolution_max_b1_gaussian_conv
- seed: training seed (replication unit, n=5). direction: random-direction index within a seed
  (matched across methods of the same seed; not independent replications).
- policy: saved (checkpoint BatchNorm statistics) | recalibrated (BN statistics re-estimated on a fixed
  2,000-image train calibration set; calibration_10k = 10,000 images).
- *_ce / *_acc: mean cross-entropy (no weight decay) / accuracy on
  train_probe, test_probe (1,000 class-balanced images each), train_large (10,000 train),
  test_full (10,000), train_full (50,000). Empty = not evaluated for that row.
- amplitude, sign: relative filter-wise perturbation theta + sign*amplitude*d (Li et al.); amplitude 0 = centre.
  Symmetric sensitivity S = (L(+) + L(-))/2 - L(centre), computed within (method, seed, policy, direction).
- final state = full resolution, no filter (identical to plain ResNet-20).

Files (%s)
- sensitivity_evaluations.csv   primary: epoch-30 solutions, 20 directions, 5 amplitudes, both signs, both policies, final state
- validation_evaluations.csv    prespecified points re-evaluated on larger sets (check=larger_sets) and with 10k calibration (check=calibration_10k, seed 0)
- interpolation_curves.csv      straight segments between final solutions of a seed, 51 alphas (A at 0, B at 1), recalibrated
- fixed_weight_evaluations.csv  checkpoints at each schedule's first/last transition update, evaluated under several intervention states (resolution, sigma_G); 10 directions, recalibrated perturbations + centres under both policies
- surfaces.csv                  2-D grids along directions 0 (a) and 1 (b), recalibrated; kind=final (all seeds 21x21, seed 0 also 41x41) or fixed_weight (seed 0, 21x21)
- checkpoint_losses.csv         every saved checkpoint (epochs + transition windows) under final and current state, both policies
- training_curves.csv           records written during training (saved BN; pinned 500-image train probe, full test set; current and target state)
- pca_checkpoints.csv           per-seed shared PCA (fit on epoch checkpoints of the 4 methods): PC1-10 coordinates, residual norms after 2/5/10 PCs
- pca_variance.csv              explained-variance ratios per seed
- pca_plane_grid.csv            seed 0: final-state loss of points reconstructed in the PC1-PC2 plane (not the checkpoints' losses)
- integrity.json                per account: timings, download hash verification, checks, pairing digests, task statuses, run summaries
- S2_paired_differences.png, D1_interpolation_segments.png  two key figures
""" % ", ".join("%s: %d rows" % (k, v) for k, v in counts.items())
    (OUT / "README.md").write_text(readme)
    files = sorted(p.name for p in OUT.iterdir())
    print(len(files), "files"); print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
