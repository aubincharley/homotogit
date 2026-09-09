"""db2 wavelet continuation pilot on ResNet-20 + BatchNorm, seed 0.

Same frozen configuration as the completed Gaussian pilot
(``resnet20bn-gaussian-20260908-154226``): 10,000 training images, the disjoint
5,000-image validation split, 2,400 optimizer updates, effective batch 128 as
four physical microbatches of 32, SGD peak LR 0.005 / momentum 0.9 / wd 5e-4 /
no Nesterov, 60-update warmup then cosine over 2,400, no augmentation, official
test set untouched.

Two runs:

* ``plain_r20bn_seed0`` -- a **fresh paired control**.  The pilot's shared
  artifacts (1.9 MB) exceed the kernel-source cap, so they cannot be shipped and
  pinned; regenerating them here and re-running plain makes the db2 pairing exact
  by construction *and* tests whether the old Gaussian run is exactly paired: if
  this control reproduces the old plain metrics, the environment regenerates
  identical initial weights, BN buffers, subset, probe and minibatch order.
* ``db2_r20bn_seed0`` -- db2 shrinkage at the same 19 main-path 3x3 convolution
  outputs, before BatchNorm.

Continuation schedule on the zero-based update index k (s, not sigma):

    0-249 0.00 | 250-499 0.25 | 500-749 0.50 | 750-999 0.70
    1000-1249 0.85 | 1250-1499 0.95 | 1500-1699 0.99 | 1700-2399 1.00

1,700 filtered updates, then 700 with the operator bypassed exactly at s=1.
Held flat inside each plateau, fixed across an update's microbatches and shared
by all insertion points.  Optimizer, BatchNorm and LR state are never reset.
These levels are a predefined exploratory schedule, not a calibration to sigma.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import torch
import torch.nn.functional as F

from scripts._study_common import BASE
from scripts import continuation_driver as D
from scripts.continuation_driver import run_study

S_PIECEWISE = [[0, 0.00], [250, 0.25], [500, 0.50], [750, 0.70],
               [1000, 0.85], [1250, 0.95], [1500, 0.99], [1700, 1.00]]

CFG = {**BASE,
       "name": "db2_pilot_r20bn",
       "arch": "resnet20_bn_cifar",
       "seeds": [0],
       "updates": 2400,
       "warmup": 60,
       "cont_end": 1700,                 # informational; the piecewise table rules
       "sigma_piecewise": S_PIECEWISE,   # for kind="db2" the table lists s
       "eval_every": 200,
       "checkpoint_updates": [600, 1200, 1700, 2400],
       "runs": [{"label": "plain_r20bn_seed0", "kind": "none", "seed": 0, "lr": 0.005},
                {"label": "db2_r20bn_seed0", "kind": "db2", "seed": 0, "lr": 0.005}]}


def timing_probe(cfg, reps=12):
    """Measure real per-update cost on this accelerator, filtered and bypassed.

    One update = four microbatches of 32 (forward + backward) plus the optimizer
    step, exactly as in training.  Random data is used; nothing here touches the
    dataset, the saved initial weights or the RNG streams the runs draw from
    (``build_model`` reseeds the global torch RNG immediately before every
    construction), so training states are fresh regardless of this probe.
    """
    if not torch.cuda.is_available():
        return {"skipped": "no accelerator"}
    from continuation.config import ModelConfig, OptimConfig
    from continuation.models import build_model
    from continuation.optim import build_optimizer

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    model = build_model(ModelConfig(arch=cfg["arch"]), 10, seed=999).to(dev).train()
    ctrl = D.Controller("db2", cont_end=cfg["cont_end"], piecewise=cfg["sigma_piecewise"])
    handles = D.attach(model, ctrl)
    opt = build_optimizer(model, OptimConfig(lr=0.005, momentum=0.9, weight_decay=5e-4,
                                             batch_size=128, total_steps=cfg["updates"],
                                             lr_schedule="cosine",
                                             warmup_steps=cfg["warmup"], min_lr=0.0))
    mb, accum = cfg["microbatch"], cfg["effective_batch"] // cfg["microbatch"]
    x = torch.randn(accum * mb, 3, 32, 32, device=dev)
    y = torch.randint(0, 10, (accum * mb,), device=dev)

    out = {}
    for name, s in (("filtered_s0.00", 0.00), ("filtered_s0.99", 0.99),
                    ("bypassed_s1.00", 1.00)):
        ctrl.value = s
        torch.cuda.reset_peak_memory_stats(dev)
        for r in range(reps + 2):                      # two warm-up updates
            if r == 2:
                torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            for a in range(accum):
                sl = slice(a * mb, (a + 1) * mb)
                (F.cross_entropy(model(x[sl]), y[sl]) * (mb / (accum * mb))).backward()
            opt.step()
        torch.cuda.synchronize(dev)
        out[name] = {"s": s, "seconds_per_update": (time.perf_counter() - t0) / reps,
                     "peak_mem_mib": torch.cuda.max_memory_allocated(dev) / 1024 ** 2}

    for h in handles:
        h.remove()
    del model, opt, x, y, ctrl
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)

    filt = out["filtered_s0.00"]["seconds_per_update"]
    byp = out["bypassed_s1.00"]["seconds_per_update"]
    out["estimate"] = {
        "filtered_updates": 1700, "bypassed_updates": 700,
        "db2_run_seconds": 1700 * filt + 700 * byp,
        "plain_run_seconds": 2400 * byp,
        "note": "training time only; excludes the 13 evaluation snapshots, "
                "data loading and checkpoint writes",
    }
    return out


if __name__ == "__main__":
    probe = timing_probe(CFG)
    D.log("timing probe: %s" % json.dumps(probe))
    D.WORK.mkdir(parents=True, exist_ok=True)
    (D.WORK / "t4_timing_probe.json").write_text(json.dumps(probe, indent=2))
    run_study(CFG)
