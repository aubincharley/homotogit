"""Anti-aliasing ablation, wave 2 -- the slice of the 6-cell manifest for this job.

Runs on four Kaggle environments (two accounts x two kernels), two T4s each.
``ABLATION4_JOB`` (0-3) picks the slice; the partition is deterministic.

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

from scripts.ablation4_manifest import assign, build_cells, summary
from scripts.campaign_driver import (WORK, find_assets, log, run_job,
                                     verify_assets)
from scripts.job_ablation import verify_ablation_ops


def verify_reductions(out_dir: Path) -> dict:
    """Trace real shapes: each depth must reduce exactly where it claims to.

    This wave's specific silent failure is a reduction registered on the wrong
    block -- the run would train happily and report a depth it never used.  The
    check pins, for every depth, which block outputs are still 32x32 and which
    are already reduced, plus the per-position sigma rescaling that follows.
    """
    from continuation.ablation_ops import (INTERNAL_REDUCTIONS,
                                           AblationController, attach_ablation)
    from continuation.config import ModelConfig
    from continuation.models import build_model
    from continuation.pipeline import resize_unit_float

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(ModelConfig(arch="resnet20_bn_cifar"), 10, seed=999).to(dev)
    model.eval()
    res, R = {}, 16

    # after blocks[k] the map must still be 32 up to and including k, then 16
    expected = {"block0_max": [32, 16, 16], "block1_max": [32, 32, 16],
                "block2_max": [32, 32, 32]}

    for reduction, exp in expected.items():
        ctrl = AblationController(
            operator="gaussian", levels=[0.5] * 30, sites=tuple(range(10)),
            resolution_by_epoch=[R] * 30, reduction=reduction,
            placement="post_block", n_positions=10)
        ctrl.set_epoch(0)
        handles = attach_ablation(model, ctrl)
        seen = {}
        hk = [model.blocks[b].register_forward_hook(
                  lambda _m, _i, o, bb=b: seen.__setitem__(bb, int(o.shape[-1])))
              for b in range(4)]
        r_in = ctrl.input_resolution()
        x = resize_unit_float(torch.rand(4, 3, 32, 32, device=dev),
                              r_in if r_in else 32)
        with torch.no_grad():
            out = model(x)
        for h in hk + handles:
            h.remove()

        got = [seen[b] for b in range(3)]
        first = INTERNAL_REDUCTIONS[reduction.split("_")[0]]["first_reduced"]
        q_ok = (all(abs(v - 1.0) < 1e-9 for v in ctrl.q[:first])
                and all(abs(v - R / 32.0) < 1e-9 for v in ctrl.q[first:10]))
        res[reduction] = {
            "network_input": int(x.shape[-1]),
            "block_outputs_0_1_2": got, "expected": exp,
            "into_blocks3": seen.get(3),
            "first_reduced_position": first,
            "q": [round(v, 3) for v in ctrl.q[:10]],
            "q_consistent": q_ok,
            "ok": got == exp and int(x.shape[-1]) == 32 and q_ok
                  and out.shape[-1] == 10,
        }

    res["all_ok"] = all(v["ok"] for k, v in res.items() if k != "all_ok")
    (out_dir / "reduction_verification.json").write_text(json.dumps(res, indent=2))
    log("depth-reduction verification: %s" % ("OK" if res["all_ok"] else res))
    if not res["all_ok"]:
        raise SystemExit(
            "a reduction did not happen where the cell claims: %s" % res)
    return res


if __name__ == "__main__":
    job = int(os.environ.get("ABLATION4_JOB", "0"))
    n_jobs = int(os.environ.get("ABLATION4_N_JOBS", "4"))

    WORK.mkdir(parents=True, exist_ok=True)
    plan = summary()
    (WORK / "ablation4_plan.json").write_text(json.dumps(plan, indent=2))
    log("wave-4 plan: %s" % json.dumps(plan))

    assets = find_assets()
    verify_assets(assets, WORK)
    verify_ablation_ops(WORK)
    verify_reductions(WORK)

    mine = assign(build_cells(), n_jobs)[job]
    log("wave-4 job %d/%d: %d cells (%s)"
        % (job, n_jobs, len(mine), [c["cell_id"] for c in mine]))
    run_job(job, mine, prefix="ablation4_job")
