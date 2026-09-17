"""Freeze configurations, schedules, reuse audit, evaluation rules and the run matrix.

    py -m comparison.freeze [--pilot-dir studies/comparison_cbs_sdpoint/raw/pilot/<account>/cmp]

Written before any new test result is read.  Records the new comparison protocol; it is not a
retrospective preregistration of the historical exploration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from continuation_core.controller import InterventionController
from continuation_core.methods import get_method
from continuation_core.models import site_map

from . import matrix as MX

PROTO = MX.STUDY / "protocol"
UPE = 391


def sha_json(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()


def controller(arm, seed=0):
    c = InterventionController(get_method(arm), site_map("resnet20_bn_cifar"), "resnet20_bn_cifar")
    return c.configure(UPE, MX.TOTAL_UPDATES, seed)


def schedules():
    out = {"updates_per_epoch": UPE, "total_updates": MX.TOTAL_UPDATES}
    for arm in ("cbs_published_schedule", "cbs_budget_matched"):
        spec = get_method(arm).cbs
        out[arm] = {"rule": spec.schedule, "segments": spec.table(UPE, MX.TOTAL_UPDATES),
                    "native_inference_sigma": controller(arm).native_state().sigma}
    b = get_method("cbs_budget_matched").cbs
    out["cbs_budget_matched"].update({"u_off": int(np.floor(0.7 * MX.TOTAL_UPDATES)), "K": b.n_plateaus()})
    for arm in MX.HISTORICAL[1:]:
        c = controller(arm)
        out[arm] = [{"epoch": e, **{k: v for k, v in c.state_for_epoch(e).to_dict().items() if k != "label"},
                     "per_site_sigma": (c.set_state(c.state_for_epoch(e)) and c.per_site_sigma())}
                    for e in range(30)]
    sd = {}
    for seed in MX.SEEDS:
        c = controller("sdpoint", seed)
        draws = [c.sdpoint_draw(u) for u in range(MX.TOTAL_UPDATES)]
        pts = np.bincount([p for p, _ in draws], minlength=10).tolist()
        sd["seed%d" % seed] = {"point_counts_p0_to_p9": pts,
                               "ratio_0.5_count": sum(r == 0.5 for p, r in draws if p > 0),
                               "ratio_0.75_count": sum(r == 0.75 for p, r in draws if p > 0),
                               "first_20": draws[:20], "sha256_all_draws": sha_json(draws)}
    out["sdpoint"] = {"stream": get_method("sdpoint").sdpoint.stream, "draws": sd}
    return out


def reuse_audit():
    rows = {}
    pinned = json.loads((MX.ROOT / "assets/cifar10_resnet20bn/assets_manifest.json").read_text())
    v2 = MX.ROOT.parent / "visualization" / "studies" / "landscape_v2" / "raw"
    for seed in MX.SEEDS:
        acc = MX.V2_ACCOUNT[seed]
        rm = json.loads((v2 / acc / "v2" / "runs_manifest.json").read_text())
        for arm in MX.HISTORICAL:
            d = v2 / acc / "v2" / "runs" / ("%s__seed%d" % (arm, seed))
            cfg, summ = json.loads((d / "config.json").read_text()), json.loads((d / "summary.json").read_text())
            ours = MX.config("sgd", arm, seed, "<data>", "<assets>", "<out>").to_dict()
            diff = sorted("%s.%s" % (sec, k) for sec in cfg for k in (cfg[sec] if isinstance(cfg[sec], dict) else {"": 0})
                          if isinstance(cfg[sec], dict) and cfg[sec].get(k) != ours[sec].get(k))
            prov = summ["provenance"]
            assets_ok = (prov["asset_states"]["init_seed%d" % seed] == pinned["states"]["init_seed%d" % seed]
                         and all(prov["asset_arrays"][k] == pinned["arrays"][k]["sha256"]
                                 for k in ("subset", "train_probe", "perm_seed%d" % seed)))
            rows["sgd__%s__seed%d" % (arm, seed)] = {
                "decision": "reuse", "source": "visualization worktree studies/landscape_v2/raw/%s/v2/runs/%s__seed%d" % (acc, arm, seed),
                "checkpoint": "checkpoints/epoch_030.pt (schema landscape_v2.analysis_checkpoint/1)",
                "checkpoint_sha256": rm["runs/%s__seed%d/checkpoints/epoch_030.pt" % (arm, seed)]["sha256"],
                "asset_hashes_match_pinned": assets_ok, "updates": summ["updates"], "torch": prov["torch"],
                "config_fields_differing_from_this_study": diff,
                "allowed_differences": "checkpoint cadence, run name/out_dir, data root, asset dir path, notes, validation_status",
                "recorded_final_target_test_acc_saved_bn": summ["final"]["target"]["test"]["acc"],
                "code": "continuation_core identical to continuation-core@ef5564c for training (git diff empty)"}
    hist = list(csv.DictReader(open(MX.ROOT / "results/optimizer_benchmark/runs.csv")))
    for r in hist:
        if r["optimizer"] == "adamw" and r["batch"] == "grid":
            rows["adamw__%s__seed%s" % (r["method"], r["seed"])] = {
                "decision": "rerun", "reason": "no checkpoint, metrics.json or summary.json retrievable: the grid ran on "
                "account alexandrecorrard with per-epoch checkpoints disabled; only job summaries are versioned, so the "
                "endpoint cannot be verified or recalibrated",
                "historical_final_test_acc_saved_bn_percent": float(r["acc"]), "historical_kernel": r["kernel"],
                "historical_wall_seconds": float(r["wall_seconds"])}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir")
    a = ap.parse_args()
    PROTO.mkdir(parents=True, exist_ok=True)
    (PROTO / "configs").mkdir(exist_ok=True)
    cfgs = {}
    for c in MX.cells():
        if c["action"] == "reuse_landscape_v2_checkpoint":
            continue
        d = MX.config(c["setting"], c["arm"], c["seed"], "/kaggle/input/<cifar10-python>", "<inputs>/assets", "/kaggle/working/cmp/cells").to_dict()
        (PROTO / "configs" / (c["run"] + ".json")).write_text(json.dumps(d, indent=1))
        cfgs[c["run"]] = sha_json(d)
    sch = schedules()
    (PROTO / "schedules.json").write_text(json.dumps(sch, indent=1))
    audit = reuse_audit()
    (PROTO / "reuse_audit.json").write_text(json.dumps(audit, indent=1))
    pilot = None
    if a.pilot_dir:
        p = Path(a.pilot_dir)
        pilot = {"environment": json.loads((p / "environment.json").read_text()),
                 "cells": {q.parent.name: json.loads(q.read_text()) for q in sorted(p.glob("pilot/*/pilot_timing.json"))},
                 "job_timing": json.loads((p / "job_timing.json").read_text()),
                 "tests_tail": (p / "pilot_tests.txt").read_text()[-600:]}
    frozen = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "statement": "Frozen before any new test result was read. Records the new comparison protocol; not a "
                     "retrospective preregistration of the historical exploration, whose methods and AdamW learning "
                     "rate had CIFAR-10 test exposure.",
        "matrix": MX.cells(), "shards": {k: {"setting": v[0], "seeds": list(v[1])} for k, v in MX.SHARDS.items()},
        "reserved_account": MX.RESERVED_ACCOUNT, "no_gpu_accounts": list(MX.NO_GPU_ACCOUNTS),
        "config_sha256": cfgs, "schedules_sha256": sha_json(sch), "reuse_audit_sha256": sha_json(audit),
        "evaluation": {
            "checkpoint": "final prescribed checkpoint after update 11,730; no best-epoch selection",
            "panel_A_primary": "native inference state; BN recalibrated on all 50,000 training images, official order, batch 500, reset, cumulative average, one pass, no gradients; eval mode",
            "panel_B": "all added interventions disabled; same recalibration; differs from A only for cbs_published_schedule",
            "saved_statistics_diagnostic": "checkpoint BN buffers at the native endpoint, and the unfiltered endpoint for cbs_published_schedule; SDPoint value is a diagnostic",
            "metrics": "full test set (10,000) and full training set (50,000) accuracy and mean CE",
            "sdpoint_inference_instance": "full resolution, fixed in advance",
            "checkpoint_identity_check": "saved-statistics test acc/CE (batch 500) of the native endpoint must match the run's recorded final record within 2e-4 acc and 1e-4 CE",
        },
        "analysis": {
            "per_setting_per_panel": ["individual seeds", "mean and sample SD over 3 seeds",
                                      "seed-paired accuracy differences (pp) with SD of differences",
                                      "CE differences in nats", "training costs with measurement scope"],
            "predeclared_comparisons": ["every method vs plain", "gaussian_postrelu vs cbs_published_schedule",
                                        "gaussian_postrelu vs cbs_budget_matched", "resolution_max_b1 vs sdpoint",
                                        "resolution_max_b1_gaussian_conv vs resolution_max_b1",
                                        "cbs_budget_matched vs cbs_published_schedule"],
            "exclusions": "none: finite low-accuracy runs are kept; relaunches after interruptions resume the same cell and are not new seeds",
            "no_tuning": "no hyperparameter, schedule, instance or checkpoint is chosen from results",
        },
        "pilot": pilot,
    }
    (PROTO / "FROZEN_PROTOCOL.json").write_text(json.dumps(frozen, indent=1, default=str))
    print("frozen", frozen["frozen_utc"], "configs", len(cfgs), "audit", len(audit))


if __name__ == "__main__":
    main()
