"""Progressive resolution on full CIFAR-10, ResNet-20 + BatchNorm, seed 0.

Two new arms, compared against the completed full-data campaign
(``fulldata-r20bn-20260908-161221``) whose ``plain_seed0`` and ``plateau_seed0``
runs are reused as controls:

* ``progres_seed0``          -- progressive resolution, no internal Gaussian
* ``progres_gauss_seed0``    -- progressive resolution + internal Gaussian

Protocol is the campaign's, unchanged: official 50,000 / 10,000 split, 30 epochs
x 391 updates = 11,730 updates, effective batch 128 as four microbatches of 32
keeping the final group of 80 and weighting each microbatch by its actual example
count, SGD peak LR 0.005 / momentum 0.9 / wd 5e-4 / no Nesterov, 60-update warmup
then cosine over the whole budget, no augmentation, normalization statistics
computed once from the full original training split.

Resolution schedule (e = zero-based epoch):  16 for e<6, 24 for 6<=e<12,
32 for 12<=e<30.  Each resolution is built from the original float image by
bilinear resize (``align_corners=False``, ``antialias=True``) before network
normalization; at r=32 the original tensor is passed through with no resize
operation.  The same parameters are used at every resolution -- no weight
interpolation, no extra blocks, no upsampling back to 32, no prediction blending.

Effective sigma rescales the campaign's plateau schedule by r(e)/32:

    epochs 0-2   r=16  ref 1.00  ->  0.500
    epochs 3-5   r=16  ref 0.85  ->  0.425
    epochs 6-8   r=24  ref 0.70  ->  0.525
    epochs 9-11  r=24  ref 0.60  ->  0.450
    epochs 12-14 r=32  ref 0.50  ->  0.500
    epochs 15-17 r=32  ref 0.40  ->  0.400
    epochs 18-20 r=32  ref 0.30  ->  0.300
    epochs 21-29 r=32  ref 0.00  ->  0.000

The rises at the two resolution transitions are intentional: the convention
approximately preserves relative blur width and makes no claim of exact
equivalence between discrete filters at different resolutions.  The resize
antialiasing and the internal Gaussian are separate operations; no extra blur is
applied to the input.  Both arms end with nine complete epochs at the exact
target configuration -- original 32x32 inputs, filters bypassed.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F

from scripts import continuation_driver as D
from scripts.continuation_driver import run_study

EPOCHS, BYPASS_FROM = 30, 21
PLATEAU = [1.00, 0.85, 0.70, 0.60, 0.50, 0.40, 0.30]


def resolution_at(e):
    return 16 if e < 6 else (24 if e < 12 else 32)


RES_BY_EPOCH = [resolution_at(e) for e in range(EPOCHS)]
REFERENCE_SIGMA = [PLATEAU[min(e // 3, len(PLATEAU) - 1)] if e < BYPASS_FROM else 0.0
                   for e in range(EPOCHS)]
SIGMA_BY_EPOCH = [RES_BY_EPOCH[e] / 32.0 * REFERENCE_SIGMA[e] for e in range(EPOCHS)]

# sha256 of the completed campaign's shared artifacts.  The runs are paired only
# if these are reproduced exactly; a matching seed number proves nothing.
CAMPAIGN = "fulldata-r20bn-20260908-161221"
EXPECTED_HASHES = {
    "subset": "33236cc6bd19fa6b89e06d441d3fcd8eb37dc8540f6a4f2b627b20af10894a41",
    "train_probe": "a33645e0e4f024d8c47c0c5bdd7fbc7ea59a976090a815c58ba029908f2c86bc",
    "perm_seed0": "5df32d0d922962e72fb4dd718a5eaad13f3209f75b8363aff64e57b89b4bb22f",
    "init_seed0": "6e652c49e5139e785887b9cb6b08ce5b0dee734bde6384550be6a8a3641e2c1b",
}

CFG = {
    "name": "progres_r20bn",
    "arch": "resnet20_bn_cifar",
    "subset_seed": 0, "per_class": 5000, "n_subset": 50000,
    "num_val": 0,
    "eval_split": "test",
    "seeds": [0], "probe_seed": 0, "train_probe": 500,
    "epochs": EPOCHS, "warmup": 60,
    "effective_batch": 128, "microbatch": 32,
    "eval_every_epochs": 2, "eval_extra_epochs": [6, 12, BYPASS_FROM],
    "checkpoint_epochs": [6, 12, BYPASS_FROM, EPOCHS],
    "cont_end": 1,
    "runs": [
        {"label": "progres_seed0", "kind": "none", "seed": 0, "lr": 0.005,
         "resolution_by_epoch": RES_BY_EPOCH},
        {"label": "progres_gauss_seed0", "kind": "gaussian", "seed": 0, "lr": 0.005,
         "resolution_by_epoch": RES_BY_EPOCH, "sigma_by_epoch": SIGMA_BY_EPOCH},
    ],
}


def verify_pairing(run_dir):
    """Hard-fail unless the regenerated shared state matches the campaign's."""
    def sha_arr(a):
        return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

    z = np.load(Path(run_dir) / "shared_indices.npz")
    got = {k: sha_arr(z[k]) for k in ("subset", "train_probe", "perm_seed0")}
    sd = torch.load(Path(run_dir) / "init_seed0.pt", map_location="cpu",
                    weights_only=True)
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(np.ascontiguousarray(sd[k].cpu().numpy()).tobytes())
    got["init_seed0"] = h.hexdigest()

    result = {"reference_campaign": CAMPAIGN, "expected": EXPECTED_HASHES,
              "observed": got,
              "matches": {k: got[k] == EXPECTED_HASHES[k] for k in EXPECTED_HASHES},
              "n_init_tensors": len(sd)}
    result["all_match"] = all(result["matches"].values())
    (Path(run_dir) / "pairing_verification.json").write_text(json.dumps(result, indent=2))
    D.log("pairing verification: %s" % json.dumps(result["matches"]))
    if not result["all_match"]:
        raise RuntimeError(
            "shared state does not reproduce the campaign's; the new arms would "
            "not be paired with plain_seed0 / plateau_seed0: %s" % result["matches"])
    return result


def timing_probe(cfg, reps=6):
    """Per-update cost at representative scheduled configurations, on this GPU.

    One update = four microbatches of 32 (forward + backward) plus the optimizer
    step.  Random data; nothing here touches the dataset, the saved initial
    weights or the sampler -- ``build_model`` reseeds the global torch RNG
    immediately before every construction, so the runs start from fresh state
    regardless of this probe.  Bounded to roughly two minutes.
    """
    if not torch.cuda.is_available():
        return {"skipped": "no accelerator"}
    from continuation.config import ModelConfig, OptimConfig
    from continuation.models import build_model
    from continuation.optim import build_optimizer

    budget_s, t_probe = 120.0, time.perf_counter()
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    model = build_model(ModelConfig(arch=cfg["arch"]), 10, seed=999).to(dev).train()
    ctrl = D.Controller("gaussian")
    handles = D.attach(model, ctrl)
    opt = build_optimizer(model, OptimConfig(lr=0.005, momentum=0.9, weight_decay=5e-4,
                                             batch_size=128, total_steps=11730,
                                             lr_schedule="cosine", warmup_steps=60,
                                             min_lr=0.0))
    mb, accum = cfg["microbatch"], cfg["effective_batch"] // cfg["microbatch"]
    x32 = torch.rand(accum * mb, 3, 32, 32, device=dev)
    y = torch.randint(0, 10, (accum * mb,), device=dev)

    cases = [("r16_sigma0.500", 16, 0.500), ("r24_sigma0.525", 24, 0.525),
             ("r32_sigma0.500", 32, 0.500), ("r16_nofilter", 16, 0.0),
             ("r24_nofilter", 24, 0.0), ("r32_nofilter", 32, 0.0)]
    out = {}
    for name, r, sig in cases:
        if time.perf_counter() - t_probe > budget_s:
            out[name] = {"skipped": "probe time budget reached"}
            continue
        ctrl.value = sig
        ctrl.bypass_all = (sig == 0.0)
        torch.cuda.reset_peak_memory_stats(dev)
        for i in range(reps + 2):
            if i == 2:
                torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            for a in range(accum):
                sl = slice(a * mb, (a + 1) * mb)
                xm = D.resize_to(x32[sl], r)
                (F.cross_entropy(model(xm), y[sl]) * (mb / (accum * mb))).backward()
            opt.step()
        torch.cuda.synchronize(dev)
        out[name] = {"resolution": r, "sigma": sig,
                     "seconds_per_update": (time.perf_counter() - t0) / reps,
                     "peak_mem_mib": torch.cuda.max_memory_allocated(dev) / 1024 ** 2}
    ctrl.bypass_all = False

    for h in handles:
        h.remove()
    del model, opt, x32, y, ctrl
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)

    def sec(name):
        v = out.get(name, {})
        return v.get("seconds_per_update")

    per_epoch = 391
    est = {}
    if all(sec(n) for n in ("r16_nofilter", "r24_nofilter", "r32_nofilter")):
        est["progres_train_seconds"] = per_epoch * (
            6 * sec("r16_nofilter") + 6 * sec("r24_nofilter") + 18 * sec("r32_nofilter"))
        est["plain_equivalent_train_seconds"] = per_epoch * 30 * sec("r32_nofilter")
    if all(sec(n) for n in ("r16_sigma0.500", "r24_sigma0.525",
                            "r32_sigma0.500", "r32_nofilter")):
        est["progres_gauss_train_seconds"] = per_epoch * (
            6 * sec("r16_sigma0.500") + 6 * sec("r24_sigma0.525")
            + 9 * sec("r32_sigma0.500") + 9 * sec("r32_nofilter"))
    est["note"] = ("training time only; excludes evaluation snapshots, data "
                   "loading and checkpoint writes.  Estimates, not measurements.")
    out["estimate"] = est
    out["probe_wall_seconds"] = time.perf_counter() - t_probe
    return out


if __name__ == "__main__":
    probe = timing_probe(CFG)
    D.log("timing probe: %s" % json.dumps(probe))
    D.WORK.mkdir(parents=True, exist_ok=True)
    (D.WORK / "t4_timing_probe.json").write_text(json.dumps(probe, indent=2))

    D.VERIFY_SHARED = verify_pairing        # called by run_study after build_shared
    run_study(CFG)
