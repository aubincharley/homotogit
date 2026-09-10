"""Anti-aliasing ablation, wave 2 -- the slice of the 6-cell manifest for this job.

Runs on four Kaggle environments (two accounts x two kernels), two T4s each.
``ABLATION2_JOB`` (0-3) picks the slice; the partition is deterministic.

Before a single update runs, this job checks -- and aborts on failure:

* the sha256 of every pinned asset (shared with wave 1, so the 18 cells pair);
* the ablation operators themselves (``job_ablation.verify_ablation_ops``);
* **that each reduction actually reduces**.  This is the specific silent failure
  this wave introduces: a ``stem_*`` cell whose reduction hook is missing trains
  happily at 32x32 and reports itself as a reduced-resolution run.  The check
  traces real tensor shapes through both reduction paths and requires the stem
  output to be 16x16 when 16 is requested.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts.ablation2_manifest import assign, build_cells, summary
from scripts.campaign_driver import (WORK, find_assets, log, run_job,
                                     verify_assets)
from scripts.job_ablation import verify_ablation_ops


def verify_reductions(out_dir: Path) -> dict:
    """Trace real shapes through every reduction path this wave uses."""
    from continuation.ablation_ops import AblationController, attach_ablation
    from continuation.config import ModelConfig
    from continuation.models import build_model
    from continuation.pipeline import resize_unit_float

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(ModelConfig(arch="resnet20_bn_cifar"), 10, seed=999).to(dev)
    model.eval()
    res = {}

    for reduction in ("input_bilinear", "stem_max"):
        for r in (16, 24, 32):
            ctrl = AblationController(
                operator="gaussian", levels=[0.5] * 30,
                sites=tuple(range(10)), resolution_by_epoch=[r] * 30,
                reduction=reduction, placement="post_block", n_positions=10)
            ctrl.set_epoch(0)
            handles = attach_ablation(model, ctrl)

            seen = {}
            h = model.blocks.register_forward_pre_hook(
                lambda _m, args: seen.__setitem__("into_blocks",
                                                  tuple(args[0].shape[-2:])))
            # the driver resizes the input itself, via ctrl.input_resolution()
            res_in = ctrl.input_resolution()
            x = resize_unit_float(torch.rand(4, 3, 32, 32, device=dev),
                                  res_in if res_in else 32)
            with torch.no_grad():
                out = model(x)
            h.remove()
            for hh in handles:
                hh.remove()

            key = "%s@r%d" % (reduction, r)
            res[key] = {
                "input_resolution": res_in,
                "network_input": tuple(x.shape[-2:]),
                "into_blocks": seen.get("into_blocks"),
                "q_per_position": round(ctrl.q[0], 4),
                "logits": tuple(out.shape),
                "expected_into_blocks": (r, r),
                "ok": seen.get("into_blocks") == (r, r) and out.shape[-1] == 10,
            }

    res["all_ok"] = all(v["ok"] for k, v in res.items() if k != "all_ok")
    (out_dir / "reduction_verification.json").write_text(json.dumps(res, indent=2))
    log("reduction verification: %s" % ("OK" if res["all_ok"] else res))
    if not res["all_ok"]:
        raise SystemExit(
            "a reduction did not reduce -- a stem_* arm would have trained at "
            "32x32 while reporting a reduced resolution: %s" % res)
    return res


if __name__ == "__main__":
    job = int(os.environ.get("ABLATION2_JOB", "0"))
    n_jobs = int(os.environ.get("ABLATION2_N_JOBS", "4"))

    WORK.mkdir(parents=True, exist_ok=True)
    plan = summary()
    (WORK / "ablation2_plan.json").write_text(json.dumps(plan, indent=2))
    log("wave-2 plan: %s" % json.dumps(plan))

    assets = find_assets()
    verify_assets(assets, WORK)
    verify_ablation_ops(WORK)
    verify_reductions(WORK)

    mine = assign(build_cells(), n_jobs)[job]
    log("wave-2 job %d/%d: %d cells (%s)"
        % (job, n_jobs, len(mine), [c["cell_id"] for c in mine]))
    run_job(job, mine, prefix="ablation2_job")
