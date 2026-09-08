"""Architecture-and-initialization comparison: plain vs Gaussian on CIFAR ResNet-18.

Same protocol as the ResNet-20 study: 10,000 train / 5,000 val images, seed 0,
1,200 updates, effective batch 128 as 4 physical microbatches of 32, SGD 0.9 /
wd 5e-4 / peak LR 0.005 / 60-update warmup / cosine to zero, no augmentation.

Only the model changes: reference-style CIFAR ResNet-18 (BatchNorm, learned
1x1+BN projection shortcuts, fan_in conv init, N(0,0.01^2) classifier), giving
17 Gaussian insertions.  The Gaussian operator and schedule are unchanged
(9-tap, radius 4, reflect, sigma_k = max(1 - k/600, 0)).

BatchNorm sees the physical microbatch of 32 -- accumulation does NOT make its
statistics equivalent to a physical batch of 128, and this differs from the
reference's physical batch of 64.  It is identical between the two arms.

Runs a bounded timing probe first and only launches the full pair if the
projection fits the agreed budget.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from scripts._study_common import BASE
from scripts.continuation_driver import Controller, attach, run_study, log, WORK
from continuation.config import ModelConfig, OptimConfig
from continuation.models import build_model, count_parameters
from continuation.optim import build_optimizer, set_lr

ARCH = "resnet18_bn_cifar"
BUDGET_S = 5 * 60          # both arms, in parallel, one per T4
PROBE_S = 2 * 60

CFG = {"name": "resnet18_gaussian", **BASE, "arch": ARCH, "seeds": [0],
       "runs": [{"label": "plain_r18_seed0", "kind": "none", "seed": 0, "lr": 0.005},
                {"label": "gaussian_r18_seed0", "kind": "gaussian", "seed": 0, "lr": 0.005}]}


def probe(kind, updates=12, warm=4):
    """Measure ms/update at microbatch 32 x 4 accumulation on this GPU."""
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    m = build_model(ModelConfig(arch=ARCH), 10, seed=0).to(dev).train()
    o = build_optimizer(m, OptimConfig(lr=0.005, momentum=0.9, weight_decay=5e-4))
    set_lr(o, 0.005)
    ctrl = Controller(kind, cont_end=CFG["cont_end"]); ctrl.set_update(0)
    hs = attach(m, ctrl)
    x = torch.rand(CFG["microbatch"], 3, 32, 32, device=dev)
    y = torch.randint(0, 10, (CFG["microbatch"],), device=dev)
    accum = CFG["effective_batch"] // CFG["microbatch"]

    def step():
        o.zero_grad(set_to_none=True)
        for _ in range(accum):
            (F.cross_entropy(m(x), y) / accum).backward()
        o.step()

    for _ in range(warm):
        step()
    torch.cuda.synchronize(dev); torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    for _ in range(updates):
        step()
    torch.cuda.synchronize(dev)
    ms = (time.perf_counter() - t0) / updates * 1000
    mem = torch.cuda.max_memory_allocated(dev) / 1024 ** 2
    n_hooks = len(hs)
    for h in hs:
        h.remove()
    del m, o, x, y, ctrl                       # restore fresh state after timing
    torch.cuda.empty_cache()
    return ms, mem, n_hooks


def main():
    if not torch.cuda.is_available():
        log("no accelerator; aborting"); return
    t0 = time.perf_counter()
    params = count_parameters(build_model(ModelConfig(arch=ARCH), 10, seed=0))
    res = {}
    for kind in ("none", "gaussian"):
        ms, mem, nh = probe(kind)
        res[kind] = {"ms_per_update": ms, "peak_mib": mem, "n_insertions": nh,
                     "projected_train_min": ms * CFG["updates"] / 60000}
        log("probe %-8s %7.1f ms/update  peak %6.0f MiB  hooks %d  -> %.2f min for %d updates"
            % (kind, ms, mem, nh, res[kind]["projected_train_min"], CFG["updates"]))
    # evaluation: 7 checkpoints x (probe 500 + val 5000) forward passes
    eval_min = 7 * 5500 / (CFG["microbatch"] * 4) * (res["gaussian"]["ms_per_update"] / 4) / 60000
    slowest = max(r["projected_train_min"] for r in res.values()) + eval_min
    res["projected_total_min_parallel"] = slowest
    res["budget_min"] = BUDGET_S / 60
    res["params"] = params
    res["probe_seconds"] = time.perf_counter() - t0
    log("projected wall (parallel, incl. eval) %.2f min vs budget %.1f min"
        % (slowest, BUDGET_S / 60))

    out = WORK / "resnet18_timing_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))

    if slowest * 60 <= BUDGET_S:
        log("within budget -> launching both arms, one per GPU")
        run_study(CFG)
    else:
        log("OVER budget -> not launching; configuration saved")
        (WORK / "DEFERRED.json").write_text(json.dumps(
            {"deferred": True, "reason": "projected %.2f min > %.1f min budget"
             % (slowest, BUDGET_S / 60), "config": CFG, "probe": res}, indent=2))


if __name__ == "__main__":
    main()
