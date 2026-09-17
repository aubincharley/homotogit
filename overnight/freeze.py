"""Freeze the follow-up protocol before any new training or evaluation result exists.

    py -m overnight.freeze

Writes ``studies/overnight_long_aug_bn/protocol/``: ``FROZEN_PROTOCOL.json`` (matrix, recipes,
schedules, BN policies, analysis plan, test-exposure statement), ``configs/<run>.json`` (the 63
exact configurations), ``schedules.json`` (transition updates and per-site effective scales of
every arm) and ``overnight/frozen_hashes.json`` (config, data-order and calibration-order digests
that every Kaggle job checks before training).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone

import numpy as np

from continuation_core.controller import InterventionController
from continuation_core.models import site_map
from continuation_core.optim import lr_at

from . import matrix as MX
from .bnpolicies import P3_AUG_STREAM, P3_ORDER_STREAM, POLICIES, p3_draws, p3_order

PROTO = MX.STUDY / "protocol"


def schedules() -> dict:
    out = {}
    for arm in MX.ARMS:
        c = InterventionController(MX.method_spec(arm), site_map("resnet20_bn_cifar"), "resnet20_bn_cifar")
        c.configure(MX.UPE, MX.U, 0)
        if arm == "sdpoint":
            out[arm] = {"kind": "stochastic per logical batch", "point": "U{0..9}", "ratio": "U{0.5, 0.75}",
                        "stream": MX.method_spec(arm).sdpoint.stream, "inference": c.native_state().to_dict(),
                        "transitions": "none (independent draw per update)"}
            continue
        segs, prev = [], None
        for u in range(MX.U):
            st = c.state_for_update(u)
            key = (st.resolution, st.sigma)
            if key != prev:
                c.set_state(st)
                segs.append({"first_update": u, "epoch": u // MX.UPE, "update_fraction": u / MX.U,
                             "resolution": st.resolution, "scheduled_level": st.sigma,
                             "per_site_effective_sigma": sorted(set(c.per_site_sigma())),
                             "n_sites": len(c.per_site_sigma())})
                prev = key
        c.set_state(c.native_state())
        out[arm] = {"segments": segs, "transition_updates": [s["first_update"] for s in segs[1:]],
                    "native_inference_state": c.native_state().to_dict(),
                    "native_per_site_sigma": sorted(set(c.per_site_sigma()))}
    out["_lr"] = {r: {"update_0": lr_at(0, MX.optimizer(r), MX.U), "update_59": lr_at(59, MX.optimizer(r), MX.U),
                      "update_31279": lr_at(31279, MX.optimizer(r), MX.U), "update_31280": lr_at(31280, MX.optimizer(r), MX.U),
                      "update_46920": lr_at(46920, MX.optimizer(r), MX.U), "update_62559": lr_at(MX.U - 1, MX.optimizer(r), MX.U)}
                  for r in MX.REGIMES}
    return out


def main():
    PROTO.mkdir(parents=True, exist_ok=True)
    (PROTO / "configs").mkdir(exist_ok=True)
    z = np.load(MX.ROOT / "assets/cifar10_resnet20bn/shared_indices.npz")
    perm_sha = {str(s): MX.sha_array(MX.extended_perms(z["perm_seed%d" % s], s)) for s in MX.SEEDS}
    p3 = {str(s): {"order_sha256": hashlib.sha256(p3_order(s, 50000).tobytes()).hexdigest(),
                   "augmentation_draws_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in p3_draws(s, 50000).items()}}
          for s in MX.SEEDS}
    cfg_sha = {}
    for c in MX.cells():
        cfg = MX.config(c["regime"], c["arm"], c["seed"])
        cfg_sha[c["run"]] = MX.config_sha(cfg)
        (PROTO / "configs" / (c["run"] + ".json")).write_text(json.dumps(cfg.to_dict(), indent=1))
    (MX.ROOT / "overnight" / "frozen_hashes.json").write_text(json.dumps(
        {"config_sha256": cfg_sha, "extended_perms_sha256": perm_sha, "p3": p3}, indent=1))
    sched = schedules()
    (PROTO / "schedules.json").write_text(json.dumps(sched, indent=1))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=MX.ROOT, capture_output=True, text=True).stdout.strip()
    proto = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "code_base_commit": commit,
        "statement": ("Prospective rules for this follow-up, written before any of its training runs or new BN evaluations "
                      "produced a result. Not a retrospective preregistration of the project: our four methods, their "
                      "schedules and the AdamW learning rate 0.02 were selected earlier with CIFAR-10 test-set exposure; "
                      "the 42-run 30-epoch comparison (branch comparison-cbs-sdpoint @ 6b28489) and its P0/P1 results "
                      "were already seen. No tuning grid; no recipe change after a poor finite run."),
        "base_study": {"branch": "comparison-cbs-sdpoint", "commit": "6b28489", "cells": 42,
                       "reused_results": ["saved_native (P0)", "panelA_recalibrated_native (P1)"],
                       "reuse_rule": "reused only if the checkpoint sha256, the native state and the recomputed P0/P1 match (|dacc| <= 2e-4, |dCE| <= 1e-4)"},
        "dataset": "CIFAR-10, full 50,000 training / official 10,000 test", "architecture": "ResNet-20 BN/ReLU (resnet20_bn_cifar)",
        "loss": "cross-entropy", "precision": "float32, no autocast, TF32 off, no compilation",
        "initialisation": "pinned assets/cifar10_resnet20bn/init_seed<k>.pt (parameters and BN buffers), shared by all arms of a seed",
        "normalisation": {"rule": "per-channel mean/std of the 50,000 unfiltered training images (float64, population std)",
                          "mean": MX.config("adamw_long_noaug_160", "plain", 0).data.expected_mean,
                          "std": MX.config("adamw_long_noaug_160", "plain", 0).data.expected_std},
        "data_order": {"epochs_0_29": "pinned perm_seed<k> (unchanged)", "epochs_30_159": "PCG64(derive_seed(k, '%s' %% e)).permutation(50000)" % MX.PERM_EXTENSION_STREAM,
                       "sha256_of_160x50000_int32": perm_sha, "batches": "consecutive slices of 128; 391 updates per epoch, last batch 80"},
        "augmentation": {"kind": "pad4_crop32_hflip", "definition": "zero padding of 4 px per side on uint8 images, uniform 32x32 crop, horizontal flip p=1/2, before normalisation; training images only",
                         "stream": "PCG64(derive_seed(k, 'augment::pad4_crop32_hflip::epoch::%d' % e)): dx, dy, flip arrays over example indices",
                         "pairing": "same crop/flip for the same image in the same epoch across arms and batch conventions; independent of SDPoint's stream and all global RNGs"},
        "regimes": {r: {"priority": MX.PRIORITY[r], "optimizer": MX.optimizer(r).__dict__, "augmentation": MX.AUGMENTATION[r],
                        "epochs": MX.EPOCHS, "updates": MX.U, "logical_batch": 128, "physical_batch": MX.microbatch(r),
                        "note": ("adaptation of He et al. (2016) Sec. 4.2 to our trainer, initialisation and normalisation; not an exact reproduction"
                                 if r.startswith("sgd") else "historical AdamW recipe with the cosine extended to 62,560 updates")}
                    for r in MX.REGIMES},
        "arms": {a: MX.method_spec(a).to_dict() for a in MX.ARMS},
        "schedule_rules": {"R": "r=16 for u<0.2U, 24 for 0.2U<=u<0.4U, else 32 (bypass): updates 12,512 and 25,024 (epochs 32, 64)",
                           "G": "g=[1,.85,.70,.60,.50,.40,.30] on consecutive tenths of u/U, bypass for the last 30% (from epoch 112)",
                           "RG": "same r and g; sigma_j = g*r/32 at all 19 conv outputs (upward jumps at resolution transitions kept)",
                           "CBS_published": "sigma(e)=0.9**floor(e/5) by actual epoch; native inference keeps sigma 0.9**31",
                           "CBS_budget_matched": "u_off=floor(0.7U)=43,792 (integer arithmetic), K=22, sigma=0.9**floor(K*u/u_off) for u<u_off, then bypass",
                           "SDPoint": "unchanged stochastic procedure for all 160 epochs; inference at point 0"},
        "matrix": MX.cells(), "n_runs": len(MX.cells()),
        "execution_order": "priority 1 (sgd_standard_aug_160) before 2 (adamw_long_noaug_160) before 3 (adamw_long_aug_160); within a regime complete seven-arm seeds; never reallocated by preliminary scores",
        "checkpoints": {"at_completed_epochs": list(MX.CHECKPOINT_EPOCHS), "rolling": "every completed epoch and at the kernel time budget"},
        "periodic_evaluation": "after every epoch (and epoch 0): saved BN statistics, current and target paths, 500-image fixed training probe and full test set (historical diagnostic)",
        "p1_learning_curve": {"epochs": list(MX.P1_CURVE_EPOCHS), "state": "scheduled state of the last update of the epoch (SDPoint: full-resolution instance); 160 = endpoint P1",
                              "isolation": "separate model and controller, all RNG states restored, time recorded as diagnostic"},
        "bn_policies": {"P0_saved_native": "checkpoint buffers, native state (diagnostic for SDPoint)",
                        "P1_cumulative_clean_500": "PRIMARY. reset, momentum None, 50,000 clean train images in official order, batch 500, one train()-mode pass, then eval()",
                        "P2_cumulative_clean_32": "P1 with batch 32 (1,563 batches incl. a final 16); cumulative per-batch average, not a pooled dataset variance",
                        "P3_ema_training_loader": "author-style adaptation of SDPoint validate(): no reset, saved buffers as start, momentum 0.1, one train()-mode pass over all 50,000 train images in a frozen seed order (%s), batch = physical training batch (128 new SGD; 32 AdamW and historical SGD/AdamW), crop/flip from %s only for regimes trained with it" % (P3_ORDER_STREAM, P3_AUG_STREAM),
                        "p3_digests": p3, "native_state": "Plain/R/G/RG/CBS budget-matched: original path; SDPoint: point 0; CBS published: final filter",
                        "rules": "fresh restoration per policy; learned-tensor hash unchanged; P0 re-run identical after calibrations; train-set labels unused; eval() after calibration; no test-batch statistics",
                        "order": list(POLICIES)},
        "analysis_plan": {"metrics": "accuracy and mean CE on full clean train (50,000) and test (10,000) for every final checkpoint x policy",
                          "paired_contrasts": ["each arm - plain", "R - SDPoint", "RG - SDPoint", "G - CBS published", "G - CBS budget-matched",
                                               "RG - R", "CBS budget-matched - CBS published"],
                          "per_contrast": "individual seed differences, mean, sample SD of the paired differences, sign counts; pp for accuracy",
                          "cross_regime": "short AdamW 30e (historical) -> long AdamW no-aug -> long AdamW aug, gains vs plain per seed; pairing across campaigns only where initial state and data order are shared (all use init_seed<k> and perm_seed<k> epochs 0-29); schedules and horizons differ",
                          "primary_policy": "P1, fixed before results", "replication_unit": "training seed (3); repeated BN evaluations are not seeds"},
    }
    (PROTO / "FROZEN_PROTOCOL.json").write_text(json.dumps(proto, indent=1, default=str))
    print("frozen", proto["frozen_utc"], "runs", proto["n_runs"], "perm sha", perm_sha)


if __name__ == "__main__":
    main()
