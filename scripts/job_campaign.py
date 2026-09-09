"""One campaign job: the cells assigned to this execution environment.

The canonical manifest is built here from ``scripts/campaign_manifest.py`` and
the pinned asset hashes, so all three jobs derive the identical 21-configuration
/ 63-cell manifest and then select their own slice by index.  ``CAMPAIGN_JOB``
(0/1/2) picks the slice; the assignment itself is deterministic, so job 1 sees
the same cells whichever machine runs it.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.campaign_driver import find_assets, run_job, log
from scripts.campaign_manifest import assign, build_cells, find_reusable


def canonical_plan(assets_dir: Path, probe=None):
    man = json.loads((assets_dir / "assets_manifest.json").read_text())
    ref = {k: v["sha256"] for k, v in man["arrays"].items()}
    ref.update(man["states"])
    reusable = find_reusable(ref)
    reusable.pop("_rejected", None)
    cells = build_cells()
    pending = [c for c in cells if c["cell_id"] not in reusable]
    plan = assign(pending, n_jobs=3, workers_per_job=2, probe=probe)
    return cells, reusable, pending, plan


def timing_probe(reps=4):
    """Measure per-update cost for each execution family on this T4.

    Forward, backward and the optimizer step, with the exact 4x32 accumulation.
    Gradients stay enabled for the stem paths.  Bounded to about two minutes.

    Used for **reporting and ETA only** -- deliberately *not* fed back into the
    work assignment, because each job probes its own machine and a probe-driven
    assignment could make the three jobs disagree about who owns which cell.
    """
    import time

    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        return {"skipped": "no accelerator"}
    from continuation.campaign_ops import (EARLY7, N_SITES, SiteController,
                                           attach_sites)
    from continuation.config import ModelConfig, OptimConfig
    from continuation.models import build_model
    from continuation.optim import build_optimizer

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    x32 = torch.rand(128, 3, 32, 32, device=dev)
    y = torch.randint(0, 10, (128,), device=dev)
    budget, t0all, out = 120.0, time.perf_counter(), {}

    cases = []
    for red in ("input_bilinear", "input_max", "stem_bilinear", "stem_max"):
        cases.append(("%s__plain" % red, red, "none", None, 16, None))
        cases.append(("%s__filtered" % red, red, "gaussian", 1.0, 16, None))
        cases.append(("%s__plain_r32" % red, red, "none", None, 32, None))
        cases.append(("%s__filtered_r32" % red, red, "gaussian", 1.0, 32, None))
    cases.append(("gmix__r16", "input_bilinear", "gmix", 1.0, 16, None))
    cases.append(("gmix__r32", "input_bilinear", "gmix", 1.0, 32, None))
    cases.append(("early7__r32", "input_bilinear", "gaussian", 1.0, 32, EARLY7))

    for name, red, op, lvl, r, sites in cases:
        if time.perf_counter() - t0all > budget:
            out[name] = {"skipped": "probe budget reached"}
            continue
        model = build_model(ModelConfig(arch="resnet20_bn_cifar"), 10,
                            seed=999).to(dev).train()
        ctrl = SiteController(op, levels=None if lvl is None else [lvl] * 30,
                              sites=sites, resolution_by_epoch=[r] * 30,
                              reduction=red)
        handles = attach_sites(model, ctrl)
        ctrl.set_epoch(0)
        opt = build_optimizer(model, OptimConfig(
            lr=0.005, momentum=0.9, weight_decay=5e-4, batch_size=128,
            total_steps=11730, lr_schedule="cosine", warmup_steps=60, min_lr=0.0))
        res_in = ctrl.input_resolution()
        from continuation.campaign_ops import reduce_spatial
        xb = x32 if res_in is None else reduce_spatial(x32, res_in, red)
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
        out[name] = {"seconds_per_update": (time.perf_counter() - t0) / reps,
                     "peak_mem_mib": torch.cuda.max_memory_allocated(dev) / 1024 ** 2,
                     "resolution": r, "operator": op,
                     "n_sites": len(ctrl.sites)}
        for h in handles:
            h.remove()
        del model, opt
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    out["probe_wall_seconds"] = time.perf_counter() - t0all
    return out


if __name__ == "__main__":
    job = int(os.environ.get("CAMPAIGN_JOB", "0"))
    assets = find_assets()

    probe = timing_probe()
    log("timing probe: %s" % json.dumps(probe))
    from scripts.campaign_driver import WORK
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "t4_timing_probe.json").write_text(json.dumps(probe, indent=2))

    _cells, reusable, pending, plan = canonical_plan(assets)
    mine = plan["jobs"][job]
    log("campaign job %d: %d of %d pending cells (%d reused campaign-wide)"
        % (job, len(mine), len(pending), len(reusable)))
    log("predicted job seconds %.0f over 2 workers -> %.0f elapsed"
        % (plan["predicted_job_seconds"][job],
           plan["predicted_elapsed_seconds"][job]))
    run_job(job, mine)
