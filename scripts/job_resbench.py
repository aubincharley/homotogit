"""One resolution-benchmark job: timing probe, frozen selection, then its slice.

The work assignment and the budget selection are computed from **fixed**
estimates, identically in all three jobs, so every job derives the same manifest
and the three slices partition the grid exactly once.  The on-GPU probe is
measured and reported but deliberately never feeds back into the assignment --
each job probes its own machine, and a probe-driven split would let the jobs
disagree about who owns which cell.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from continuation.resolution_ops import O_REF
from scripts.campaign_driver import find_assets
from scripts.resbench_driver import log, run_job
from scripts.resbench_manifest import build_cells, build_configs, select

#: seconds per run, per operator, at 30 epochs.  Anchored on the campaign's
#: measured 489 s plain / 447-455 s for the historical D-arms; the heavier
#: operators carry a margin.  Used for scheduling only.
EST = {"none": 500.0, "max": 470.0, "bilinear": 480.0, "maxblur": 520.0,
       "softpool": 560.0, "l2": 520.0, "hminus1": 520.0, "perceptual": 620.0}
BUDGET_SECONDS = 2.6 * 3600


def per_cell_seconds(c):
    return EST.get(c["operator"], 550.0)


def plan(n_jobs=3, workers_per_job=2):
    cfgs = build_configs()
    chosen, deferred, total = select(cfgs, BUDGET_SECONDS, per_cell_seconds,
                                     workers=n_jobs * workers_per_job)
    cells = build_cells(chosen)
    for c in cells:
        c["est_seconds"] = per_cell_seconds(c)
    # longest-first, spreading a configuration's three seeds across the jobs
    load = [0.0] * n_jobs
    jobs = [[] for _ in range(n_jobs)]
    by_cfg = {}
    for c in cells:
        by_cfg.setdefault(c["id"], []).append(c)
    order = sorted(by_cfg.items(), key=lambda kv: -sum(x["est_seconds"] for x in kv[1]))
    for i, (_cid, group) in enumerate(order):
        for j, cell in enumerate(sorted(group, key=lambda c: c["seed"])):
            pref = [(i + j + k) % n_jobs for k in range(n_jobs)]
            t = min(pref, key=lambda p: load[p])
            cell["job"] = t
            jobs[t].append(cell)
            load[t] += cell["est_seconds"]
    for j in range(n_jobs):
        jobs[j].sort(key=lambda c: -c["est_seconds"])
    return {"configs": cfgs, "chosen": chosen, "deferred": deferred,
            "jobs": jobs, "load": load, "total_gpu_seconds": total,
            "predicted_elapsed": max(load) / 1.0}


def timing_probe(reps=4, budget=150.0):
    """Complete optimizer updates at representative operator/location/r."""
    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available():
        return {"skipped": "no accelerator"}
    from continuation.config import ModelConfig, OptimConfig
    from continuation.models import build_model
    from continuation.optim import build_optimizer
    from continuation.resolution_ops import (OPERATORS, ResolutionController,
                                             attach_resolution, reduce_with)

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    x32 = torch.rand(128, 3, 32, 32, device=dev)
    y = torch.randint(0, 10, (128,), device=dev)
    t0all, out = time.perf_counter(), {}
    cases = [(op, loc, r) for op in OPERATORS for loc in ("input", "D1")
             for r in (16, 24)] + [("none", "none", 32)]
    for op, loc, r in cases:
        name = "%s@%s_r%d" % (op, loc, r)
        if time.perf_counter() - t0all > budget:
            out[name] = {"skipped": "probe budget reached"}
            continue
        model = build_model(ModelConfig(arch="resnet20_bn_cifar"), 10,
                            seed=999).to(dev).train()
        ctrl = ResolutionController(op, loc, "Rprog")
        ctrl.set_state(r)
        handles = attach_resolution(model, ctrl)
        opt = build_optimizer(model, OptimConfig(
            lr=0.005, momentum=0.9, weight_decay=5e-4, batch_size=128,
            total_steps=11730, lr_schedule="cosine", warmup_steps=60, min_lr=0.0))
        xb = (reduce_with(x32, r, op) if loc == "input" and op != "none" else x32)
        torch.cuda.reset_peak_memory_stats(dev)
        for i in range(reps + 2):
            if i == 2:
                torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            for a in range(4):
                sl = slice(a * 32, (a + 1) * 32)
                (F.cross_entropy(model(xb[sl]), y[sl]) * 0.25).backward()
            opt.step()
        torch.cuda.synchronize(dev)
        spu = (time.perf_counter() - t0) / reps
        out[name] = {"seconds_per_update": spu,
                     "projected_run_seconds": spu * 11730,
                     "peak_mem_mib": torch.cuda.max_memory_allocated(dev) / 1024 ** 2}
        for h in handles:
            h.remove()
        del model, opt
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    out["probe_wall_seconds"] = time.perf_counter() - t0all
    return out


if __name__ == "__main__":
    job = int(os.environ.get("RESBENCH_JOB", "0"))
    assets = find_assets()
    probe = timing_probe()
    log("timing probe: %s" % json.dumps(probe))
    from scripts.resbench_driver import WORK
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "t4_timing_probe.json").write_text(json.dumps(probe, indent=2))

    p = plan()
    mine = p["jobs"][job]
    log("resbench job %d: %d cells (grid %d configs, %d selected, %d deferred)"
        % (job, len(mine), len(p["configs"]), len(p["chosen"]), len(p["deferred"])))
    log("predicted %.0f s for this job over 2 workers -> %.0f s elapsed"
        % (p["load"][job], p["load"][job] / 2))
    run_job(job, mine, extra={"selected": [c["id"] for c in p["chosen"]],
                              "deferred": [c["id"] for c in p["deferred"]],
                              "estimates": EST, "probe": probe,
                              "o_ref": O_REF})
