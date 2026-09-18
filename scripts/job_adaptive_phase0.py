"""Adaptive resolution, Phase 0: instrument fixed schedules, probe the boundary grid.

Nothing adaptive runs *live* here -- every schedule is fixed and known before
training, so the horizon (30 x 391 = 11,730 updates) and the LR path are
identical to every campaign arm.  What is new is what gets **measured** at the
end of every epoch, at the current weights, without perturbing training:

* monitor-set CE/acc and the full gradient norm of the current objective (the
  corrector residual of docs/adaptive_resolution_plan.md), with eval-mode and
  with train-mode BatchNorm;
* the same gradient at the *next* scheduled resolution and at the 32x32 target,
  and the cosine between the current gradient and each of them -- how useful
  the coarse surrogate still is for the finer problem;
* CE/acc at the next resolution and at the target **after recalibrating
  BatchNorm statistics at fixed weights** (restored bitwise afterwards): the
  BN-free look-ahead, and the answer to Q-04 of OPEN_QUESTIONS;
* test CE/acc on the current path and on the target path, the 500-image probe
  CE, per-group weight norms, the epoch's mean training loss.

Part A -- the plan's two comparators, ``R32__Gnone`` and ``Rprog__Gnone``,
seeds 0/1/2, paired with the 63-cell campaign: the shared state is regenerated
in the kernel and its sha256 digests are checked against the pinned campaign
assets before any update runs.  The final accuracies are therefore also a
replication of six campaign cells.

Part B -- schedule design, seed 0, no filter, bilinear input reduction: seven
alternative placements of the two boundaries, a per-epoch linear ramp 16->32
over twelve epochs, four steps 16/20/24/28, and a per-update *mixed*
resolution schedule whose low-resolution mass matches Rprog.  Each adds switch
events to the spike / plateau characterisation for free.

Set ``ADAPT_SMOKE=1`` for a tiny CPU smoke test (200 images, 3 epochs; the
pairing check is reported but not enforced).

``ADAPT_PHASE=1`` selects the follow-up job (proposals of plan section 13):
seeds 1/2 of ``Rsteps4``, ``Rlin12`` and ``Rmixed``; ``Rprog``, ``Rsteps4`` and
``Rlin12`` with the campaign's Gaussian plateau filter, seeds 0/1/2; and, on every
run, the step-size controller's Phase 1 signal -- transfer efficiency
tau_Delta = <g_r, g_{r+Delta}> / ||g_{r+Delta}||^2 for Delta in {1, 2, 4, 8}, with
**train-mode** BatchNorm gradients -- so the controller can be replayed offline.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from continuation.campaign_ops import SiteController, attach_sites
from continuation.config import DataConfig, ModelConfig, OptimConfig
from continuation.data import build_dataset, fixed_subset_indices
from continuation.models import build_model
from continuation.optim import build_optimizer, lr_at, set_lr
from continuation.pipeline import ChannelNormalizer, InputPipeline
from continuation.probe_signals import (BNState, ce_acc, cosine, full_gradient,
                                        group_norms, param_groups,
                                        parameter_checksum, parameters_equal,
                                        recalibrate_bn, weight_norms)
from continuation.seeding import numpy_generator
from continuation.transforms.gaussian import GaussianSmoothing
from scripts import continuation_driver as D
from scripts.campaign_manifest import GAUSSIAN, OPERATOR

SMOKE = os.environ.get("ADAPT_SMOKE") == "1"
PHASE = int(os.environ.get("ADAPT_PHASE", "0"))
WORK = Path(os.environ.get("STUDY_OUT", "/kaggle/working"))
T0 = time.perf_counter()

EPOCHS = 3 if SMOKE else 30
PROTOCOL = {"epochs": EPOCHS, "warmup": 60, "effective_batch": 128, "microbatch": 32,
            "lr": 0.005, "momentum": 0.9, "weight_decay": 5e-4}
SHARED_CFG = {"name": "adaptive_phase0", "arch": "resnet20_bn_cifar",
              "subset_seed": 0, "per_class": 20 if SMOKE else 5000,
              "n_subset": 200 if SMOKE else 50000, "num_val": 0,
              "seeds": [int(v) for v in os.environ.get("ADAPT_SEEDS", "0,1,2").split(",")],
              "probe_seed": 0, "train_probe": 500,
              "epochs": EPOCHS}
# sha256 of the campaign's pinned assets (results/kaggle_outputs/campaign-j0-*/
# .../assets_verification.json).  Regenerated state must reproduce them exactly.
PINNED = {
    "subset": "33236cc6bd19fa6b89e06d441d3fcd8eb37dc8540f6a4f2b627b20af10894a41",
    "train_probe": "a33645e0e4f024d8c47c0c5bdd7fbc7ea59a976090a815c58ba029908f2c86bc",
    "perm_seed0": "5df32d0d922962e72fb4dd718a5eaad13f3209f75b8363aff64e57b89b4bb22f",
    "perm_seed1": "b3c1460f166f3e3afb821c9b5cebc30a1915ce47b3a2502b85921041e8b58cb0",
    "perm_seed2": "995dcd74c2f86f26d89d170041b8996b37e7dc881ebe9db103ddf2fc76f3e5bb",
    "init_seed0": "6e652c49e5139e785887b9cb6b08ce5b0dee734bde6384550be6a8a3641e2c1b",
    "init_seed1": "f6fb22087956ae3e39398efa42ab92580bbd66d86a8886a468e13bb4fc28ef5e",
    "init_seed2": "fb19c69f3b02419133c852d831622cba0adc539775e578f826fdb0944ab815db",
}
EXPECTED_STATS = {"mean": [0.4913996756076813, 0.4821584224700928, 0.44653090834617615],
                  "std": [0.24703222513198853, 0.24348512291908264, 0.26158782839775085]}
# measurement-only subset of the training split; class balanced; never trained on
# differently from the rest -- it is *part of* the training set, like the probe
MONITOR = {"size": 100 if SMOKE else 2000, "seed": 0, "stream": "adaptive_monitor"}
RES = (16, 24, 32)
TAU_DELTAS = (1, 2, 4, 8)        # candidate step sizes for the step-size controller


def log(msg):
    print("[%7.1fs] %s" % (time.perf_counter() - T0, msg), flush=True)


# --------------------------------------------------------------------------
# schedules
# --------------------------------------------------------------------------

def steps(s1, s2, E=EPOCHS):
    """16 for ``s1`` epochs, 24 until epoch ``s2``, then 32."""
    if not 0 <= s1 <= s2 <= E:
        raise ValueError((s1, s2, E))
    return [16] * s1 + [24] * (s2 - s1) + [32] * (E - s2)


if SMOKE:
    SCHEDULES = {"R32": [32] * 3, "Rprog": [16, 24, 32], "Rlin12": [16, 24, 32],
                 "Rsteps4": [20, 28, 32]}
    MIX_PROBS = [(0.6, 0.3, 0.1), (0.25, 0.5, 0.25), (0.0, 0.0, 1.0)]
else:
    SCHEDULES = {
        "R32": [32] * 30,
        "Rprog": steps(6, 12),                  # the campaign schedule
        "Rb3_9": steps(3, 9), "Rb3_12": steps(3, 12),
        "Rb6_9": steps(6, 9), "Rb6_18": steps(6, 18),
        "Rb9_15": steps(9, 15), "Rb9_18": steps(9, 18),
        "Rb12_18": steps(12, 18),
        # one new resolution every epoch: 16,17,19,20,21,23,24,25,27,28,29,31, then 32
        "Rlin12": [16 + int(round(16.0 * e / 12.0)) for e in range(12)] + [32] * 18,
        "Rsteps4": [16] * 3 + [20] * 3 + [24] * 3 + [28] * 3 + [32] * 18,
        # the same 12-epoch ramp rounded to even sizes (stride-2 phase kept): 16,16,18,20,20,22,24,24,26,28,28,30
        "Rlin12even": [2 * int(round((16 + 16.0 * e / 12.0) / 2.0)) for e in range(12)] + [32] * 18,
        # Rsteps4 with one 24x24 "reheat" epoch at epochs 19, 23, 27 (0-based): fixed
        # control for the two-sided controller of phase 4
        "Rsteps4rh": [16] * 3 + [20] * 3 + [24] * 3 + [28] * 3
                     + [24 if e in (19, 23, 27) else 32 for e in range(12, 30)],
    }
    # per-update draw over (16, 24, 32); low-resolution mass 12.3 epoch-equivalents
    # against Rprog's 12; pure 32 from epoch 18 as in Rprog's final 18 epochs? no --
    # Rprog is pure 32 from epoch 12; the mixture keeps some coarse batches to 18.
    MIX_PROBS = ([(0.6, 0.3, 0.1)] * 6 + [(0.25, 0.5, 0.25)] * 6
                 + [(0.1, 0.3, 0.6)] * 6 + [(0.0, 0.0, 1.0)] * 12)


def _run(sched, gauss, seed, part, mixed=False):
    return {"label": "%s__%s__input_bilinear__seed%d" % (sched, gauss, seed),
            "part": part, "schedule": sched, "arm": sched, "gaussian": gauss, "seed": seed,
            "mixed": mixed}


def build_runs():
    runs = []
    if PHASE == 0:
        for sched in ("R32", "Rprog"):                           # Part A
            for s in (0, 1, 2):
                runs.append(_run(sched, "Gnone", s, "A"))
        for sched in SCHEDULES:                                   # Part B
            if sched not in ("R32", "Rprog"):
                runs.append(_run(sched, "Gnone", 0, "B"))
        runs.append(_run("Rmixed", "Gnone", 0, "B", mixed=True))
    elif PHASE == 2:
        # live step-size controller (plan section 13.3), two thresholds, plus the even ramp
        for theta in (0.5, 0.65):
            for sd in (0, 1, 2):
                r = _run("Rctrl%02d" % int(theta * 100), "Gnone", sd, "D")
                r["controller"] = {"theta": theta, "delta": 4, "ema_beta": 0.5,
                                   "floor": "r >= 16 + 4*(e-8) for e >= 8; 32 from epoch 12"}
                runs.append(r)
        if os.environ.get("ADAPT_CONTROLLER_ONLY") != "1":
            for sd in (0, 1, 2):
                runs.append(_run("Rlin12even", "Gnone", sd, "D"))
    elif PHASE == 3:
        # bounded-specialisation controller (plan section 17): advance by delta when the
        # relative scale-transfer gap g = (CE_{r+delta}^{BN-recal} - CE_r) / CE_r exceeds g*
        for gstar in (0.04, 0.06, 0.08):
            for sd in (0, 1, 2):
                r = _run("Rgap%02d" % int(round(gstar * 100)), "Gnone", sd, "E")
                r["controller"] = {"kind": "gap", "gstar": gstar, "delta": 4, "ema_beta": 0.5,
                                   "floor_start": 11,
                                   "floor": "r >= 16 + 4*(e-11) for e >= 11; 32 from epoch 15"}
                runs.append(r)
    elif PHASE == 4:
        # two-sided bounded specialisation: phase-3 ascent (g* = 0.06), then at 32x32 one
        # reheat epoch at 24 whenever the downward gap g(32 -> 24) exceeds down_gstar
        sweep = ((0.45, 1), (0.60, 1)) if os.environ.get("ADAPT_SWEEP") == "high" else ((0.15, 3), (0.30, 3))
        for dg, settle in sweep:
            for sd in (0, 1, 2):
                r = _run("Rgap2s%02d" % int(round(dg * 100)), "Gnone", sd, "F")
                r["controller"] = {"kind": "gap2s", "gstar": 0.06, "delta": 4, "ema_beta": 0.5,
                                   "floor_start": 11, "down_gstar": dg, "reheat_r": 24,
                                   "settle": settle,
                                   "floor": "ascent only: r >= 16 + 4*(e-11) for e >= 11"}
                runs.append(r)
        if os.environ.get("ADAPT_CONTROLLER_ONLY") != "1":       # Rsteps4rh done in adaptive-phase4-20260917-151733
            for sd in (0, 1, 2):
                runs.append(_run("Rsteps4rh", "Gnone", sd, "F"))
    elif PHASE == 5:
        # separate the reheat controller from the ascent controller, six seeds for the
        # three arms that matter (fixed reheat, adaptive reheat on a fixed ascent, ramp)
        new_seeds = (3, 4, 5)
        for sd in new_seeds:
            runs.append(_run("Rsteps4rh", "Gnone", sd, "G"))
        for sd in (0, 1, 2) + new_seeds:
            r = _run("Rsteps4ar", "Gnone", sd, "G")
            r["schedule"] = "Rsteps4"
            r["controller"] = {"kind": "gap2s", "ascent": "fixed", "ema_beta": 0.5,
                               "down_gstar": 0.30, "reheat_r": 24, "settle": 2,
                               "delta": 4, "gstar": None, "floor_start": None}
            runs.append(r)
        for sd in new_seeds:
            runs.append(_run("Rsteps4", "Gnone", sd, "G"))
        for sd in new_seeds:
            r = _run("Rgap2s30", "Gnone", sd, "G")
            r["controller"] = {"kind": "gap2s", "gstar": 0.06, "delta": 4, "ema_beta": 0.5,
                               "floor_start": 11, "down_gstar": 0.30, "reheat_r": 24, "settle": 3}
            runs.append(r)
    else:
        # Gaussian arms first (longest); Rprog + Gplateau replicates a campaign cell
        for sched in ("Rprog", "Rsteps4", "Rlin12"):
            for s in (0, 1, 2):
                runs.append(_run(sched, "Gplateau", s, "C"))
        for sched in ("Rsteps4", "Rlin12"):
            for s in (1, 2):
                runs.append(_run(sched, "Gnone", s, "C"))
        for s in (1, 2):
            runs.append(_run("Rmixed", "Gnone", s, "C", mixed=True))
    if SMOKE:
        runs = [r for r in runs if r["seed"] == 0][:3]
    return runs


# --------------------------------------------------------------------------
# shared state and pairing
# --------------------------------------------------------------------------

def _sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _sha_state(p):
    sd = torch.load(p, map_location="cpu", weights_only=True)
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(np.ascontiguousarray(sd[k].cpu().numpy()).tobytes())
    return h.hexdigest()


def verify_pairing(run_dir: Path) -> dict:
    z = np.load(run_dir / "shared_indices.npz")
    got = {k: _sha_array(z[k]) for k in ("subset", "train_probe", "perm_seed0",
                                         "perm_seed1", "perm_seed2")}
    for s in (0, 1, 2):
        got["init_seed%d" % s] = _sha_state(run_dir / ("init_seed%d.pt" % s))
    matches = {k: got[k] == PINNED[k] for k in PINNED}
    result = {"expected": PINNED, "observed": got, "matches": matches,
              "all_match": all(matches.values()), "enforced": not SMOKE}
    (run_dir / "pairing_verification.json").write_text(json.dumps(result, indent=2))
    log("pairing verification: %s" % json.dumps(matches))
    if not result["all_match"] and not SMOKE:
        raise RuntimeError("regenerated shared state does not reproduce the "
                           "campaign's pinned assets: %s" % matches)
    return result


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------

def bn_shift(saved):
    """Label-free scale-specialisation proxy: how far the BatchNorm statistics estimated
    at another resolution sit from the training path's running statistics.

    Averaged over BN layers: mean of |mu' - mu| / sigma and of |log(var'/var)|, taken
    *before* ``saved.restore()`` so ``m`` holds the recalibrated values."""
    import math
    dm, dv, n = 0.0, 0.0, 0
    with torch.no_grad():
        for m, (rm, rv, _nb, _mom) in zip(saved.mods, saved.saved):
            sd = (rv + m.eps).sqrt()
            dm += float(((m.running_mean - rm).abs() / sd).mean())
            dv += float(((m.running_var + m.eps) / (rv + m.eps)).log().abs().mean())
            n += 1
    return {"mean_shift": dm / max(n, 1), "logvar_shift": dv / max(n, 1)}


def next_distinct(sched, done):
    cur = sched[done - 1] if done > 0 else sched[0]
    for r in sched[done:]:
        if r != cur:
            return r
    return None


def train_run(spec, shared_dir: Path, gpu: int, out_dir: Path, worker: int):
    dev = torch.device("cuda:%d" % gpu if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    run_dir = out_dir / spec["label"]
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = int(spec["seed"])

    z = np.load(shared_dir / "shared_indices.npz")
    subset, tp_idx, perms = z["subset"], z["train_probe"], z["perm_seed%d" % seed]

    root, _dl = D.data_source()
    bundle = build_dataset(DataConfig(root=root, download=False, num_val=0)).to(dev)
    if not SMOKE:
        ok = (np.allclose([float(v) for v in bundle.mean], EXPECTED_STATS["mean"], atol=1e-7)
              and np.allclose([float(v) for v in bundle.std], EXPECTED_STATS["std"], atol=1e-7))
        if not ok:
            raise RuntimeError("normalization statistics differ from the pinned values")

    tr_imgs = bundle.train.images[torch.as_tensor(subset, device=dev)]
    tr_lbls = bundle.train.labels[torch.as_tensor(subset, device=dev)]
    n = int(subset.size)
    tp = torch.as_tensor(tp_idx, device=dev)
    probe_imgs, probe_lbls = tr_imgs[tp], tr_lbls[tp]
    test_imgs, test_lbls = bundle.test.images, bundle.test.labels
    mon_idx = fixed_subset_indices(n, MONITOR["size"], MONITOR["seed"], MONITOR["stream"],
                                   labels=tr_lbls.cpu().numpy())
    mon = torch.as_tensor(mon_idx, device=dev)
    mon_imgs, mon_lbls = tr_imgs[mon], tr_lbls[mon]

    pipe = InputPipeline(GaussianSmoothing(sigma_max=1.0),
                         ChannelNormalizer(bundle.mean, bundle.std).to(dev))
    model = build_model(ModelConfig(arch=SHARED_CFG["arch"]), 10, seed=seed).to(dev)
    model.load_state_dict(torch.load(shared_dir / ("init_seed%d.pt" % seed),
                                     map_location=dev, weights_only=True))
    model.train()
    groups = param_groups(model)

    mixed = bool(spec["mixed"])
    controller = spec.get("controller")
    if controller and controller.get("ascent") == "fixed":
        sched = list(SCHEDULES[spec["schedule"]])      # fixed ascent, adaptive reheats only
    elif controller:
        sched = [16] * EPOCHS          # realised schedule, filled in as decisions are taken
    else:
        sched = [32] * EPOCHS if mixed else list(SCHEDULES[spec["schedule"]])
    ctl_state = {"ema": None, "log": [], "reached_32": False, "ema_down": None, "reheats": 0}

    def floor_resolution(e):
        start = (controller or {}).get("floor_start", 8)
        return min(32, max(16, 16 + 4 * (e - start)))

    def controller_decide(done, sig):
        """After epoch ``done`` (1-based), choose the resolution for epoch ``done``.

        Advance by ``delta`` when the EMA of tau(r -> r + delta) -- measured with
        train-mode BN gradients right after the epoch -- drops below theta; the
        floor guarantees 32x32 from epoch 12 whatever happens.  The EMA is
        re-seeded after every switch (post-switch blackout of one check).
        """
        if done >= EPOCHS:
            return
        r_cur = sched[done - 1]
        entry = {"after_epoch": done, "r": r_cur, "reason": None}
        two_sided = controller.get("kind") == "gap2s"
        if r_cur == 32:
            ctl_state["reached_32"] = True
        if two_sided and ctl_state["reached_32"]:
            # descent side: bound the specialisation to the fine scale
            if r_cur == 32:
                cur, nxt = sig["current"], sig.get("next", {})
                gap = None
                if "monitor_ce_bnrecal" in nxt and cur["monitor_ce"] > 0:
                    gap = (nxt["monitor_ce_bnrecal"] - cur["monitor_ce"]) / cur["monitor_ce"]
                    b = controller["ema_beta"]
                    ctl_state["ema_down"] = (gap if ctl_state["ema_down"] is None
                                             else b * ctl_state["ema_down"] + (1 - b) * gap)
                entry["gap_down"] = gap
                entry["ema_down"] = ctl_state["ema_down"]
                can = done < EPOCHS - controller["settle"]
                if (can and ctl_state["ema_down"] is not None
                        and ctl_state["ema_down"] >= controller["down_gstar"]):
                    sched[done] = controller["reheat_r"]
                    ctrl.resolution_by_epoch[done] = controller["reheat_r"]
                    ctl_state["ema_down"] = None
                    ctl_state["reheats"] += 1
                    entry["reason"] = "reheat"
                entry["r_next_epoch"] = sched[done]
            else:
                entry["r_next_epoch"] = sched[done]          # back to 32 after a reheat epoch
            ctl_state["log"].append(entry)
            return
        if r_cur < 32 and controller.get("ascent") == "fixed":
            entry["r_next_epoch"] = sched[done]
            ctl_state["log"].append(entry)
            return
        if r_cur < 32:
            kind = controller.get("kind", "tau")
            b = controller["ema_beta"]
            r_new = r_cur
            if kind in ("gap", "gap2s"):
                cur, nxt = sig["current"], sig.get("next", {})
                gap = None
                if "monitor_ce_bnrecal" in nxt and cur["monitor_ce"] > 0:
                    gap = (nxt["monitor_ce_bnrecal"] - cur["monitor_ce"]) / cur["monitor_ce"]
                    ctl_state["ema"] = gap if ctl_state["ema"] is None else b * ctl_state["ema"] + (1 - b) * gap
                entry["gap"] = gap
                entry["ema"] = ctl_state["ema"]
                if ctl_state["ema"] is not None and ctl_state["ema"] >= controller["gstar"]:
                    r_new, entry["reason"] = min(r_cur + controller["delta"], 32), "gap_above_gstar"
            else:
                cand = sig.get("tau_train", {}).get(str(min(r_cur + controller["delta"], 32)))
                tau = None if cand is None else cand["tau"]
                if tau is not None:
                    ctl_state["ema"] = tau if ctl_state["ema"] is None else b * ctl_state["ema"] + (1 - b) * tau
                entry["tau"] = tau
                entry["ema"] = ctl_state["ema"]
                if ctl_state["ema"] is not None and ctl_state["ema"] < controller["theta"]:
                    r_new, entry["reason"] = min(r_cur + controller["delta"], 32), "tau_below_theta"
            if floor_resolution(done) > r_new:
                r_new, entry["reason"] = floor_resolution(done), "floor"
            if r_new != r_cur:
                ctl_state["ema"] = None
                for k in range(done, EPOCHS):
                    sched[k] = r_new
                    # SiteController keeps its *own copy* of the schedule: update it too,
                    # or the decision never reaches the training path (bug found in the
                    # first Phase 2 launch, adaptive-phase2-20260917-093614).
                    ctrl.resolution_by_epoch[k] = r_new
            entry["r_next_epoch"] = r_new
        ctl_state["log"].append(entry)
    gauss = spec.get("gaussian", "Gnone")
    levels = GAUSSIAN[gauss]
    if levels is not None and len(levels) != EPOCHS:      # smoke: truncate the table
        levels = list(levels[:EPOCHS])
    ctrl = SiteController(OPERATOR[gauss], levels=levels, sites=None,
                          resolution_by_epoch=sched, reduction="input_bilinear")
    handles = attach_sites(model, ctrl)                # identity hooks; same path as the campaign
    mix_rng = numpy_generator(seed, "resolution_mix") if mixed else None

    B, mb = PROTOCOL["effective_batch"], PROTOCOL["microbatch"]
    per_epoch = (n + B - 1) // B
    total_updates = EPOCHS * per_epoch
    ocfg = OptimConfig(lr=PROTOCOL["lr"], momentum=PROTOCOL["momentum"],
                       weight_decay=PROTOCOL["weight_decay"], batch_size=B,
                       total_steps=total_updates, lr_schedule="cosine",
                       warmup_steps=PROTOCOL["warmup"], min_lr=0.0)
    opt = build_optimizer(model, ocfg)

    metrics, gstep = [], 0
    eval_seconds = [0.0]
    t_start = time.perf_counter()

    def forward_at(r, level):
        """Forward at an explicit (resolution, filter level); level None/0 bypasses."""
        def fwd(x):
            ctrl.bypass_all = False
            ctrl.set_state(level, r)
            if level is None or float(level) == 0.0:
                ctrl.bypass_all = True
            return model(pipe(x, 0.0, res=r))
        return fwd

    def snapshot(done, epoch_train_loss=None, res_counts=None):
        t0 = time.perf_counter()
        ref_params = parameter_checksum(model)
        bn_ref = BNState(model)
        prev_state = (ctrl.value, ctrl.resolution, ctrl.bypass_all)
        last = max(done - 1, 0)
        r_cur = 32 if mixed else sched[last]
        r_next = None if mixed else next_distinct(sched, done)
        if controller and r_cur < 32:
            r_next = min(r_cur + controller["delta"], 32)
        if controller and controller.get("kind") == "gap2s" and ctl_state["reached_32"]:
            r_next = controller["reheat_r"] if r_cur == 32 else 32
        cur_level = ctrl.levels[last] if ctrl.levels else None
        filtered = cur_level is not None and float(cur_level) > 0.0

        # roles: current path (trained-under), next scheduled resolution with the
        # same filter level, and the unfiltered 32x32 target path
        roles = [("current", r_cur, cur_level)]
        if r_next is not None:
            roles.append(("next", r_next, cur_level))
        if mixed:
            roles += [("r16", 16, None), ("r24", 24, None)]
        elif r_cur != 32 or filtered:
            roles.append(("target", 32, 0.0))

        sig, grads = {}, {}
        for role, r, lvl in roles:
            fwd = forward_at(r, lvl)
            ce, acc = ce_acc(model, fwd, mon_imgs, mon_lbls)
            g = full_gradient(model, fwd, mon_imgs, mon_lbls)
            grads[role] = g
            entry = {"resolution": r, "level": lvl, "monitor_ce": ce, "monitor_acc": acc,
                     "grad_norm": group_norms(g, groups)}
            if role == "current":
                gt = full_gradient(model, fwd, mon_imgs, mon_lbls, train_mode_bn=True)
                grads["current_train"] = gt
                entry["grad_norm_trainbn"] = group_norms(gt, groups)
                entry["cos_evalbn_trainbn"] = cosine(g, gt)
            else:
                saved = BNState(model)
                recalibrate_bn(model, fwd, mon_imgs)
                rce, racc = ce_acc(model, fwd, mon_imgs, mon_lbls)
                entry["monitor_ce_bnrecal"] = rce
                entry["monitor_acc_bnrecal"] = racc
                entry["bn_shift"] = bn_shift(saved)
                if role == "target":
                    tce, tacc = ce_acc(model, fwd, test_imgs, test_lbls)
                    entry["test_ce_bnrecal"] = tce
                    entry["test_acc_bnrecal"] = tacc
                saved.restore()
                entry["cos_with_current"] = cosine(grads["current"], g)
            sig[role] = entry

        # scale-transfer gap at several candidate steps (label-using, BN part removed)
        if not mixed and r_cur < 32 and not (controller and ctl_state.get("reached_32")):
            gp = {}
            for d in (2, 4, 8):
                rc = r_cur + d
                if rc > 32:
                    continue
                fwd = forward_at(rc, cur_level)
                saved = BNState(model)
                recalibrate_bn(model, fwd, mon_imgs)
                rce, racc = ce_acc(model, fwd, mon_imgs, mon_lbls)
                shift = bn_shift(saved)
                saved.restore()
                gp[str(rc)] = {"delta": d, "monitor_ce_bnrecal": rce, "monitor_acc_bnrecal": racc,
                               "gap_rel": (rce - sig["current"]["monitor_ce"]) / sig["current"]["monitor_ce"],
                               "bn_shift": shift}
            sig["gap_probe"] = gp

        # step-size controller signal: train-mode gradients at r + Delta
        if not mixed and r_cur < 32 and not (controller and ctl_state.get("reached_32")):
            gt = grads["current_train"]
            tau = {}
            for d in TAU_DELTAS:
                rc = r_cur + d
                if rc > 32:
                    continue
                gc = full_gradient(model, forward_at(rc, cur_level), mon_imgs, mon_lbls,
                                   train_mode_bn=True)
                nc = float(gc.norm())
                tau[str(rc)] = {"delta": d, "cos": cosine(gt, gc), "grad_norm": nc,
                                "tau": float(torch.dot(gt, gc)) / (nc * nc) if nc > 0 else None}
                del gc
            sig["tau_train"] = tau
        del grads

        pce, pacc = ce_acc(model, forward_at(r_cur, cur_level), probe_imgs, probe_lbls)
        tce_c, tacc_c = ce_acc(model, forward_at(r_cur, cur_level), test_imgs, test_lbls)
        tce_t, tacc_t = ce_acc(model, forward_at(32, 0.0), test_imgs, test_lbls)
        ctrl.value, ctrl.resolution, ctrl.bypass_all = prev_state
        ctrl.q = ctrl.q_for(ctrl.resolution)

        rec = {"epoch": done, "update": gstep,
               "lr": lr_at(min(max(gstep - 1, 0), total_updates - 1), ocfg),
               "resolution_current": r_cur, "resolution_next": r_next,
               "level_current": cur_level, "operator": ctrl.operator,
               "mixed": mixed, "mixed_counts": res_counts,
               "train_loss_epoch_mean": epoch_train_loss,
               "train_probe_ce_current": pce, "train_probe_acc_current": pacc,
               "test_ce_current": tce_c, "test_acc_current": tacc_c,
               "test_ce_target": tce_t, "test_acc_target": tacc_t,
               "signals": sig, "weight_norms": weight_norms(model),
               "purity": {"params_unchanged": parameters_equal(model, ref_params),
                          "bn_restored": not bn_ref.differs(),
                          "mode_train": model.training},
               "elapsed_s": time.perf_counter() - t_start,
               "peak_mem_mib": (torch.cuda.max_memory_allocated(dev) / 1024 ** 2
                                if dev.type == "cuda" else None)}
        if not (rec["purity"]["params_unchanged"] and rec["purity"]["bn_restored"]
                and rec["purity"]["mode_train"]):
            raise RuntimeError("measurement perturbed the training state: %s" % rec["purity"])
        metrics.append(rec)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
        eval_seconds[0] += time.perf_counter() - t0
        cur, nxt = sig["current"], sig.get("next", {})
        tau_s = "-"
        if "tau_train" in sig and sig["tau_train"]:
            tau_s = " ".join("%s:%.2f" % (k, v["tau"]) for k, v in sig["tau_train"].items()
                             if v["tau"] is not None)
        gap_s = "-"
        if "gap_probe" in sig and sig["gap_probe"]:
            gap_s = " ".join("%s:%.3f" % (k, v["gap_rel"]) for k, v in sig["gap_probe"].items())
        log("%-40s ep=%2d/%d r=%-2s->%-4s lvl=%-5s | test cur=%.4f tgt=%.4f | "
            "||g||tr=%.3f cos(next)=%s tau=%s gap=%s | %.1fs"
            % (spec["label"][:40], done, EPOCHS, r_cur, r_next,
               "-" if cur_level is None else "%.2f" % cur_level, tacc_c, tacc_t,
               cur["grad_norm_trainbn"]["total"],
               "-" if "cos_with_current" not in nxt else "%.2f" % nxt["cos_with_current"],
               tau_s, gap_s, time.perf_counter() - t0))
        return sig

    snapshot(0)
    for e in range(EPOCHS):
        ctrl.set_epoch(e)
        res_in = ctrl.input_resolution()
        if not mixed and int(res_in) != int(sched[e]):
            raise RuntimeError("training resolution %s differs from the schedule %s at epoch %d"
                               % (res_in, sched[e], e))
        perm = perms[e]
        loss_sum, loss_n = 0.0, 0
        counts = {str(r): 0 for r in RES} if mixed else None
        for s0 in range(0, n, B):
            batch = perm[s0:s0 + B]
            total = int(batch.size)
            idx = torch.as_tensor(batch, dtype=torch.long, device=dev)
            if mixed:
                res_in = int(mix_rng.choice(RES, p=MIX_PROBS[e]))
                counts[str(res_in)] += 1
            set_lr(opt, lr_at(gstep, ocfg))
            opt.zero_grad(set_to_none=True)
            for a in range(0, total, mb):
                sl = idx[a:a + mb]
                n_m = int(sl.numel())
                loss = F.cross_entropy(model(pipe(tr_imgs[sl], 0.0, res=res_in)),
                                       tr_lbls[sl])
                (loss * (n_m / total)).backward()
                loss_sum += float(loss) * n_m
                loss_n += n_m
            opt.step()
            gstep += 1
        done = e + 1
        sig = snapshot(done, epoch_train_loss=loss_sum / max(loss_n, 1), res_counts=counts)
        if controller:
            controller_decide(done, sig)
            metrics[-1]["controller"] = dict(ctl_state["log"][-1])
            (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
        ctrl.set_epoch(e)

    last = metrics[-1]
    wall = time.perf_counter() - t_start
    summary = {**spec, "worker": worker, "gpu_index": gpu,
               "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
               "epochs": EPOCHS, "updates": gstep, "updates_per_epoch": per_epoch,
               "n_train": n, "n_test": int(test_imgs.shape[0]),
               **{k: PROTOCOL[k] for k in ("warmup", "effective_batch", "microbatch")},
               "resolution_by_epoch": None if mixed else sched,
               "controller": controller, "controller_log": ctl_state["log"] if controller else None,
               "reheats": ctl_state["reheats"] if controller else None,
               "mix_probs": MIX_PROBS if mixed else None,
               "reduction": "input_bilinear (align_corners=False, antialias=True; "
                            "exact bypass at 32)",
               "operator": ctrl.operator, "gaussian": gauss,
               "levels": levels, "controller": ctrl.describe(),
               "monitor": {**MONITOR, "n": int(mon_idx.size),
                           "sha256": _sha_array(mon_idx)},
               "normalization": {"mean": [float(v) for v in bundle.mean],
                                 "std": [float(v) for v in bundle.std]},
               "final_test_acc": last["test_acc_target"],
               "final_test_ce": last["test_ce_target"],
               "final_train_probe_ce": last["train_probe_ce_current"],
               "wall_seconds": wall, "eval_seconds": eval_seconds[0],
               "train_seconds": wall - eval_seconds[0],
               "peak_mem_mib": (torch.cuda.max_memory_allocated(dev) / 1024 ** 2
                                if dev.type == "cuda" else None)}
    for h in handles:
        h.remove()
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log("%-40s DONE acc=%.4f ce=%.4f  %.0fs (%.0fs eval)"
        % (spec["label"][:40], summary["final_test_acc"], summary["final_test_ce"],
           wall, eval_seconds[0]))
    return summary


# --------------------------------------------------------------------------
# workers
# --------------------------------------------------------------------------

def _worker(queue, specs_by_label, shared_dir, out_dir, worker, gpu, counter, total):
    while True:
        try:
            label = queue.get_nowait()
        except Exception:
            return
        spec = specs_by_label[label]
        try:
            train_run(spec, Path(shared_dir), gpu, Path(out_dir), worker)
        except Exception:
            tb = traceback.format_exc()
            fail = Path(out_dir) / label
            fail.mkdir(parents=True, exist_ok=True)
            (fail / "FAILED.json").write_text(json.dumps(
                {"spec": spec, "worker": worker, "traceback": tb}, indent=2))
            log("FAILED %s\n%s" % (label, tb))
        with counter.get_lock():
            counter.value += 1
            k = counter.value
        el = time.perf_counter() - T0
        log("progress: %d/%d complete, elapsed %.0fs, ETA %.0fs"
            % (k, total, el, el / max(k, 1) * (total - k)))


def main():
    out_dir = WORK / ("adaptive_phase%d_%s" % (PHASE, time.strftime("%Y%m%d-%H%M%S")))
    out_dir.mkdir(parents=True, exist_ok=True)
    ngpu = torch.cuda.device_count()
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "torch": torch.__version__, "cuda": torch.version.cuda, "n_gpu": ngpu,
           "gpus": [torch.cuda.get_device_name(i) for i in range(ngpu)], "smoke": SMOKE}
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2))
    log("environment: %s" % env)
    if ngpu == 0 and not SMOKE:
        (out_dir / "ABORTED.json").write_text(json.dumps(
            {"reason": "no accelerator granted", "environment": env}))
        raise SystemExit("no accelerator granted")

    D.build_shared(SHARED_CFG, out_dir)
    verify_pairing(out_dir)

    runs = build_runs()
    (out_dir / "config.json").write_text(json.dumps(
        {"phase": PHASE, "protocol": PROTOCOL, "shared": SHARED_CFG, "schedules": SCHEDULES,
         "gaussian_levels": GAUSSIAN, "tau_deltas": TAU_DELTAS,
         "mix_probs": MIX_PROBS, "monitor": MONITOR, "runs": runs}, indent=2))
    log("%d runs: %s" % (len(runs), [r["label"] for r in runs]))

    by_label = {r["label"]: r for r in runs}
    if ngpu >= 2:
        ctx = mp.get_context("spawn")
        queue, counter = ctx.Queue(), ctx.Value("i", 0)
        for r in runs:
            queue.put(r["label"])
        procs = [ctx.Process(target=_worker, args=(queue, by_label, str(out_dir),
                                                   str(out_dir), w, w, counter, len(runs)))
                 for w in range(2)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
    else:
        for i, r in enumerate(runs):
            try:
                train_run(r, out_dir, 0, out_dir, 0)
            except Exception:
                tb = traceback.format_exc()
                (out_dir / r["label"]).mkdir(parents=True, exist_ok=True)
                (out_dir / r["label"] / "FAILED.json").write_text(json.dumps(
                    {"spec": r, "traceback": tb}, indent=2))
                log("FAILED %s\n%s" % (r["label"], tb))

    summaries, failures = [], []
    for r in runs:
        d = out_dir / r["label"]
        if (d / "summary.json").is_file():
            summaries.append(json.loads((d / "summary.json").read_text()))
        else:
            failures.append(r["label"])
    (out_dir / "study_summary.json").write_text(json.dumps(
        {"completed": summaries, "failed_or_incomplete": failures,
         "elapsed_s": time.perf_counter() - T0}, indent=2))
    for s in summaries:
        log("SUMMARY %-44s acc=%.4f ce=%.4f  %.0fs (%.0fs eval)"
            % (s["label"], s["final_test_acc"], s["final_test_ce"],
               s["wall_seconds"], s["eval_seconds"]))
    log("finished: %d complete, %d failed, %.0fs" % (len(summaries), len(failures),
                                                    time.perf_counter() - T0))


if __name__ == "__main__":
    main()
