"""Anti-aliasing ablation -- the slice of the 36-cell manifest assigned to this job.

Two execution environments (two Kaggle accounts), two T4s each, four GPUs total.
The manifest is built identically in both jobs and each selects its own slice by
index, so the assignment is deterministic and independent of which machine runs
which job.  ``ABLATION_JOB`` (0/1) picks the slice.

Before a single update runs this job:

* re-checks the sha256 of every pinned asset (``campaign_driver.verify_assets``);
* verifies the ablation operators themselves -- where each placement attaches,
  that the two decimating blocks are where the module claims, that ``sigma = 0``
  is a bitwise identity through the whole network, and that BlurPool survives
  ``bypass_all``.  A failure aborts before training rather than producing
  results whose operator is unverified.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from scripts.ablation_manifest import assign, build_cells, summary
from scripts.campaign_driver import (WORK, find_assets, log, run_job,
                                     verify_assets)


def verify_ablation_ops(out_dir: Path) -> dict:
    """Check every claim the ablation's operators make, on this worker."""
    from continuation.ablation_ops import (DECIMATING_BLOCKS, MASKS, NODOWN,
                                           PREDOWN, AblationController,
                                           attach_ablation)
    from continuation.campaign_ops import N_SITES
    from continuation.config import ModelConfig
    from continuation.models import build_model

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    res = {}

    model = build_model(ModelConfig(arch="resnet20_bn_cifar"), 10, seed=999).to(dev)
    model.eval()

    convs = [m for m in model.modules()
             if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3)]
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    res["n_conv_3x3"] = len(convs)
    res["n_batchnorm"] = len(bns)
    res["strided_conv_sites"] = [i for i, c in enumerate(convs) if c.stride == (2, 2)]
    res["n_blocks"] = len(model.blocks)
    res["decimating_blocks_declared"] = list(DECIMATING_BLOCKS)
    res["shortcut_subsampling_blocks"] = [
        i for i, b in enumerate(model.blocks)
        if type(b.shortcut).__name__ == "_PadShortcut"]

    # the two decimating blocks must be exactly those holding a strided conv,
    # and each must also carry a subsampling shortcut on the SAME input tensor
    res["decimation_consistent"] = (
        sorted(res["shortcut_subsampling_blocks"]) == sorted(DECIMATING_BLOCKS)
        and all(model.blocks[b].conv1.stride == (2, 2) for b in DECIMATING_BLOCKS))

    # masks partition all19
    res["masks_partition_all19"] = (
        sorted(set(PREDOWN) | set(NODOWN)) == list(range(N_SITES))
        and not (set(PREDOWN) & set(NODOWN))
        and len(MASKS["all19"]) == N_SITES)

    x = torch.rand(8, 3, 32, 32, device=dev)
    with torch.no_grad():
        ref = model(x).clone()

    # sigma = 0 must be a bitwise identity for every placement
    identity = {}
    for placement in ("conv_out", "post_bn", "post_block"):
        n_pos = 10 if placement == "post_block" else N_SITES
        ctrl = AblationController(operator="gaussian", levels=[0.0] * 30,
                                  sites=tuple(range(n_pos)),
                                  resolution_by_epoch=[32] * 30,
                                  placement=placement, n_positions=n_pos)
        ctrl.set_epoch(0)
        handles = attach_ablation(model, ctrl)
        with torch.no_grad():
            got = model(x)
        identity[placement] = {
            "n_positions": n_pos,
            "max_abs_diff": float((got - ref).abs().max()),
            "bitwise_identical": bool(torch.equal(got, ref))}
        for h in handles:
            h.remove()
    res["sigma0_is_identity"] = identity

    # a nonzero sigma must actually change the output at every placement
    active = {}
    for placement in ("conv_out", "post_bn", "post_block"):
        n_pos = 10 if placement == "post_block" else N_SITES
        ctrl = AblationController(operator="gaussian", levels=[0.5] * 30,
                                  sites=tuple(range(n_pos)),
                                  resolution_by_epoch=[32] * 30,
                                  placement=placement, n_positions=n_pos)
        ctrl.set_epoch(0)
        handles = attach_ablation(model, ctrl)
        with torch.no_grad():
            got = model(x)
        active[placement] = {"max_abs_diff": float((got - ref).abs().max()),
                             "changes_output": not torch.equal(got, ref)}
        for h in handles:
            h.remove()
    res["sigma_positive_changes_output"] = active

    # BlurPool: must act, and must NOT be disabled by bypass_all
    ctrl = AblationController(operator="none", levels=None,
                              resolution_by_epoch=[32] * 30, blurpool_sigma=0.5)
    ctrl.set_epoch(0)
    handles = attach_ablation(model, ctrl)
    with torch.no_grad():
        bp_on = model(x).clone()
        ctrl.bypass_all = True
        bp_bypassed = model(x).clone()
    for h in handles:
        h.remove()
    res["blurpool"] = {
        "changes_output": not torch.equal(bp_on, ref),
        "max_abs_diff_vs_plain": float((bp_on - ref).abs().max()),
        "survives_bypass_all": bool(torch.equal(bp_on, bp_bypassed)),
        "note": "BlurPool is architecture: bypass_all must not disable it"}

    # gradients must flow through every placement
    grads = {}
    for placement in ("conv_out", "post_bn", "post_block"):
        n_pos = 10 if placement == "post_block" else N_SITES
        ctrl = AblationController(operator="gaussian", levels=[0.5] * 30,
                                  sites=tuple(range(n_pos)),
                                  resolution_by_epoch=[32] * 30,
                                  placement=placement, n_positions=n_pos)
        ctrl.set_epoch(0)
        handles = attach_ablation(model, ctrl)
        model.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        gn = sum(float(p.grad.norm()) for p in model.parameters() if p.grad is not None)
        finite = all(torch.isfinite(p.grad).all() for p in model.parameters()
                     if p.grad is not None)
        grads[placement] = {"total_grad_norm": gn, "all_finite": bool(finite),
                            "nonzero": gn > 0}
        model.zero_grad(set_to_none=True)
        for h in handles:
            h.remove()
    res["gradients"] = grads

    res["all_ok"] = bool(
        res["n_conv_3x3"] == N_SITES
        and res["n_batchnorm"] == N_SITES
        and res["strided_conv_sites"] == [7, 13]
        and res["decimation_consistent"]
        and res["masks_partition_all19"]
        and all(v["bitwise_identical"] for v in identity.values())
        and all(v["changes_output"] for v in active.values())
        and res["blurpool"]["changes_output"]
        and res["blurpool"]["survives_bypass_all"]
        and all(g["all_finite"] and g["nonzero"] for g in grads.values()))

    (out_dir / "ablation_ops_verification.json").write_text(json.dumps(res, indent=2))
    log("ablation operator verification: %s" % ("OK" if res["all_ok"] else res))
    if not res["all_ok"]:
        raise SystemExit("ablation operator verification failed: %s" % res)
    return res


if __name__ == "__main__":
    job = int(os.environ.get("ABLATION_JOB", "0"))
    n_jobs = int(os.environ.get("ABLATION_N_JOBS", "2"))

    WORK.mkdir(parents=True, exist_ok=True)
    plan = summary()
    (WORK / "ablation_plan.json").write_text(json.dumps(plan, indent=2))
    log("ablation plan: %s" % json.dumps(plan))

    assets = find_assets()
    verify_assets(assets, WORK)
    verify_ablation_ops(WORK)

    mine = assign(build_cells(), n_jobs)[job]
    log("ablation job %d/%d: %d cells, est %.0f s over 2 workers"
        % (job, n_jobs, len(mine), sum(c["est_seconds"] for c in mine)))
    for c in mine:
        log("  queued %-28s est %4ds" % (c["cell_id"], c["est_seconds"]))
    run_job(job, mine, prefix="ablation_job")
