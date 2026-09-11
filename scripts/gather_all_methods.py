"""Collect every trained arm from every batch into one table.

Five sources, deliberately kept distinguishable because they are **not** one
experiment:

===========  ==========================================  ==================
batch        what                                        shared assets
===========  ==========================================  ==================
campaign     21 configurations, 3 seeds                  ours (pinned)
resbench     resolution-only, 28 configurations, 3 seeds ours (same pinned)
ablation     Idriss's operator/placement batch, 32 arms  his own pinned set
adaptive     Aubin's triggered-schedule arms, 30 epochs  ours (verified)
long120      Aubin's 120-epoch arms                      ours, LONGER BUDGET
===========  ==========================================  ==================

``campaign``, ``resbench`` and ``adaptive`` reproduce the same initial weights,
so they are mutually paired.  ``ablation`` used a different pinned set -- its
shared baseline lands 0.56 pp lower -- so its absolute level carries a batch
offset.  ``long120`` trains four times longer and is not comparable on accuracy
to anything else here; it is carried so it can be shown on an epoch axis, where
the longer budget is visible rather than hidden.

Writes ``results/all_methods.json``.  Reads only; no training.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.method_labels import (ABLATION_LABEL, ADAPTIVE_LABEL,
                                   campaign_label, resbench_label)

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get(
    "SCRATCH_ROOT",
    Path.home() / "AppData/Local/Temp/claude/C--Users-mnica-Documents-Projet-filiere"
                  "/1e7c0fdf-8f5a-466c-b47e-badfce660f23/scratchpad"))
ABL = SCRATCH / "ablation_data" / "results"
AUBIN = SCRATCH / "aubin" / "results" / "kaggle_outputs"
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
    "long": "120-epoch budget",
}


def _mean_sd(v):
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)


def _add(rows, key, label, family, batch, accs, epochs, wall, curves=None,
         n_seeds=None, note=None):
    if not accs:
        return
    m, s = _mean_sd(accs)
    rows[key] = {"key": key, "label": label, "family": family, "batch": batch,
                 "acc_mean": m, "acc_sd": s, "acc_per_seed": sorted(accs),
                 "n_seeds": n_seeds or len(accs), "epochs": epochs,
                 "wall_mean": wall, "curves": curves or {}, "note": note}


# --------------------------------------------------------------------------

def from_campaign(rows):
    f = ROOT / "results" / "campaign_results.json"
    if not f.is_file():
        return
    res = json.loads(f.read_text(encoding="utf-8"))
    studies = [ROOT / j["study"] for j in res["jobs"]]
    for cid, c in res["configurations"].items():
        lab, fam = campaign_label(cid)
        curves = {}
        for s in c["seeds"]:
            for st_dir in studies:
                p = st_dir / ("%s__seed%d" % (cid, s)) / "metrics.json"
                if p.is_file():
                    curves[s] = json.loads(p.read_text(encoding="utf-8"))
                    break
        _add(rows, "campaign/" + cid, lab, fam, "campaign", c["acc_per_seed"],
             30, c["wall_mean"], curves)


def from_resbench(rows):
    f = ROOT / "results" / "resbench_results.json"
    if not f.is_file():
        return
    res = json.loads(f.read_text(encoding="utf-8"))
    base = ROOT / "results" / "kaggle_outputs"
    for cid, c in res["configurations"].items():
        if not c.get("n_seeds"):
            continue
        lab, fam = resbench_label(cid)
        curves = {}
        for s in c.get("seeds", []):
            for p in base.glob("resbench-j*/resbench_job*/%s__seed%d/metrics.json"
                               % (cid, s)):
                curves[s] = json.loads(p.read_text(encoding="utf-8"))
                break
        _add(rows, "resbench/" + cid, lab, fam, "resbench", c["acc_per_seed"],
             30, c["wall_mean"], curves)


ABL_LABEL = {
    "C_plain": ("Plain baseline (ablation batch)", "plain"),
    "C_plateau": ("Gaussian plateau, conv output", "gauss"),
    "P_postbn": ("Gaussian after BatchNorm", "gauss"),
    "P_postblock": ("Gaussian after ReLU (10 sites)", "gauss"),
    "M_predown": ("Gaussian at 2 pre-decimation sites", "gauss"),
    "M_nodown": ("Gaussian at the other 17 sites", "gauss"),
    "K_const030": ("Constant sigma 0.30", "constant"),
    "K_const050": ("Constant sigma 0.50", "constant"),
    "K_const080": ("Constant sigma 0.80", "constant"),
    "K_const100": ("Constant sigma 1.00", "constant"),
    "B_blurpool": ("BlurPool (fixed, pre-decimation)", "constant"),
    "B_blurpool_plateau": ("BlurPool + Gaussian plateau", "res_gauss"),
    "R1": ("Progressive resolution only", "res"),
    "R2": ("Campaign best, replayed", "res_gauss"),
    "R3": ("Progressive resolution + Gaussian after ReLU", "res_gauss"),
    "R4": ("Reduction after stem, no filter", "res"),
    "R5": ("Both interventions post-ReLU", "res_gauss"),
    "R6": ("Constant sigma, post-ReLU placement", "constant"),
    "D0": ("Single reduction after block 0", "res"),
    "D1": ("Single reduction after block 1", "res"),
    "D2": ("Single reduction after block 2", "res"),
    "D0G": ("Reduction after block 0 + Gaussian", "res_gauss"),
    "D1G": ("Reduction after block 1 + Gaussian", "res_gauss"),
    "D2G": ("Reduction after block 2 + Gaussian", "res_gauss"),
    "P_A1": ("Profile: sigma ~ map size (post-ReLU)", "profile"),
    "P_A2": ("Profile: sigma ~ sqrt(map size) (post-ReLU)", "profile"),
    "P_A3": ("Profile: sigma rising with depth (post-ReLU)", "profile"),
    "P_A4": ("Profile: sigma ~ receptive field (post-ReLU)", "profile"),
    "Q_A1": ("Profile: sigma ~ map size (conv out)", "profile"),
    "Q_A2": ("Profile: sigma ~ sqrt(map size) (conv out)", "profile"),
    "Q_A3": ("Profile: sigma rising with depth (conv out)", "profile"),
    "Q_A4": ("Profile: sigma ~ receptive field (conv out)", "profile"),
}


def from_ablation(rows):
    f = ABL / "ablation_aa_results.json"
    if not f.is_file():
        return
    arms = json.loads(f.read_text(encoding="utf-8"))["arms"]
    for k, v in arms.items():
        lab, fam = ABLATION_LABEL.get(k, (k, "operator"))
        curves = {}
        for p in (ABL / "kaggle_outputs").glob("abl*-j*/*job*/%s__seed*/metrics.json" % k):
            seed = int(p.parent.name.rsplit("__seed", 1)[1])
            curves[seed] = json.loads(p.read_text(encoding="utf-8"))
        _add(rows, "ablation/" + k, lab, fam, "ablation",
             v.get("acc_per_seed") or [v["acc"]], 30, v.get("wall_s"), curves,
             n_seeds=v.get("n_seeds", 1),
             note=("read on the current path (never reaches the target)"
                   if v.get("primary_path") == "current" else None))


AUBIN_LABEL = {
    "ADAPTSIG": ("Adaptive sigma, gradient-norm trigger", "adaptive"),
    "ADAPTGAP": ("Adaptive sigma, transfer-gap trigger (mis-set)", "adaptive"),
    "ADAPTGAP2": ("Adaptive sigma, transfer-gap trigger", "adaptive"),
    "ADAPTRES": ("Adaptive resolution, transfer-gap trigger", "adaptive"),
    "alloc_back": ("Allocation search (back-loaded dwell)", "adaptive"),
    "alloc_front": ("Allocation search (front-loaded dwell)", "adaptive"),
    "alloc_even": ("Allocation search (even dwell)", "adaptive"),
    "Gplateau_cal": ("Gaussian plateau, instrumented rerun", "gauss"),
    "Gplateau_cal2": ("Gaussian plateau, instrumented rerun 2", "gauss"),
    "Gplateau_cal3": ("Gaussian plateau, instrumented rerun 3", "gauss"),
    "plain_hi": ("Plain, 120 epochs, high LR", "long"),
    "plain_lo": ("Plain, 120 epochs, low LR", "long"),
    "gplateau_hi": ("Gaussian plateau, 120 epochs, high LR", "long"),
    "gplateau_lo": ("Gaussian plateau, 120 epochs, low LR", "long"),
    "adaptgap_hi": ("Adaptive gap, 120 epochs, high LR", "long"),
    "adaptgap_lo": ("Adaptive gap, 120 epochs, low LR", "long"),
    "ADAPT": ("Adaptive sigma, first trigger version", "adaptive"),
    "rho0.5": ("Per-layer sigma, rho = 0.5", "profile"),
    "rho1": ("Per-layer sigma, rho = 1", "profile"),
    "rho2": ("Per-layer sigma, rho = 2", "profile"),
    "adaptive": ("Per-layer sigma, adaptive rho", "profile"),
    "plain": ("Plain baseline (per-layer batch)", "plain"),
}


def from_aubin(rows):
    if not AUBIN.is_dir():
        return
    buckets = {}
    for f in AUBIN.glob("*/*/*/summary.json"):
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        acc = s.get("final_test_acc")
        if acc is None:
            continue
        cell = s.get("cell_id") or s.get("label") or f.parent.name
        arm, _, seed = str(cell).rpartition("__seed")
        arm = arm or str(cell).rsplit("_seed", 1)[0]
        seed = int(seed) if seed.isdigit() else 0
        job = f.parts[len(AUBIN.parts)]
        if job.startswith(("fulldata-", "progres-", "campaign-", "resbench-",
                           "pls-smoke", "probe-")):
            continue                      # already covered by their own batch
        ep = s.get("epochs", 30)
        key = (arm, ep)
        b = buckets.setdefault(key, {"accs": [], "wall": [], "curves": {},
                                     "job": job.split("-2026")[0]})
        b["accs"].append(acc)
        b["wall"].append(s.get("wall_seconds") or 0)
        m = f.parent / "metrics.json"
        if m.is_file():
            b["curves"][seed] = json.loads(m.read_text(encoding="utf-8"))
    for (arm, ep), b in buckets.items():
        if ep and ep > 30:
            continue          # 120-epoch runs: four times the budget, excluded
        lab, fam = ADAPTIVE_LABEL.get(arm, (None, "adaptive"))
        if lab is None:
            lab = "%s (%s)" % (arm.replace("_", " "), b.get("job", "Aubin batch"))
        _add(rows, "adaptive/%s@%d" % (arm, ep), lab, fam, "adaptive",
             b["accs"], ep, st.mean(b["wall"]) if b["wall"] else None, b["curves"])


def _unified_family(cfg):
    blur = cfg["operator"] == "gaussian"
    red = cfg["resolution"] != "R32"
    if not blur and not red:
        return "plain"
    if blur and not red:
        return "gauss"
    if red and not blur:
        return "res"
    return "profile" if cfg.get("sigma_profile") else "res_gauss"


def from_unified(rows):
    """The one batch where every arm shares assets AND is evaluated every epoch."""
    f = ROOT / "results" / "unified_summary.json"
    if not f.is_file():
        return
    from scripts.unified_manifest import build_configs
    cfgs = {c["id"]: c for c in build_configs()}
    res = json.loads(f.read_text(encoding="utf-8"))
    base = ROOT / "results" / "kaggle_outputs"
    for cid, r in res.items():
        curves = {}
        for s in r["per_seed"]:
            for p in base.glob("unified-j*/unified_job*/%s__seed%s/metrics.json"
                               % (cid, s)):
                curves[int(s)] = json.loads(p.read_text(encoding="utf-8"))
                break
        lab = r["label"] + (" (unified batch)" if cid == "plain" else "")
        _add(rows, "unified/" + cid, lab, _unified_family(cfgs[cid]),
             "unified", list(r["per_seed"].values()), 30, None, curves)


def main():
    rows = {}
    from_campaign(rows)
    from_resbench(rows)
    from_ablation(rows)
    from_aubin(rows)
    from_unified(rows)
    payload = {"n_configurations": len(rows),
               "batches": sorted({r["batch"] for r in rows.values()}),
               "families": FAMILY,
               "configurations": {k: {kk: vv for kk, vv in v.items() if kk != "curves"}
                                  for k, v in rows.items()}}
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    from collections import Counter
    print("configurations: %d" % len(rows))
    print("by batch:", dict(Counter(r["batch"] for r in rows.values())))
    print("by family:", dict(Counter(r["family"] for r in rows.values())))
    print("with curves:", sum(1 for r in rows.values() if r["curves"]))
    print("wrote", OUT)
    return rows


if __name__ == "__main__":
    main()
