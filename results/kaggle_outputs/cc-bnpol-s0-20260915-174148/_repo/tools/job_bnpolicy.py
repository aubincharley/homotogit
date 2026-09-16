"""Curvature of the twelve reference checkpoints, under both BatchNorm policies
and both splits.

``BNPOL_SHARD`` picks the slice.

Why
---
Our curvature was measured under one policy (stored statistics) and on training
images only.  A companion loss-landscape study on the same four methods reports
two things that make both choices load-bearing:

* finite-amplitude sensitivity is three to seven times larger under stored
  statistics than under recalibration, and the **ranking of the Gaussian method
  reverses** between them -- less sensitive than the control when frozen,
  43-108 % more sensitive once recalibrated, in 5 seeds out of 5;
* the resolution method's advantage, clear on a training probe, shrinks to
  -0.006 nats on the full test set with 2/5 seeds below the control.

So "curriculum solutions are flatter" may be a property of one policy, of the
training split, or of both.  This job measures the same quantity across the 2x2
and lets the answer decide.

Settings are deliberately lighter than the endpoint probe: 24 Hutchinson draws
and one eigenvalue at 25 iterations give about 2.7 % precision on the trace,
against a contrast of 30 % -- and cost 49 Hessian-vector products per point
instead of 264.
"""
import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

POLICIES = ("running_stats", "fixed_batch_stats")
SPLITS = ("train", "test")


def find(pattern, what):
    hits = sorted(glob.glob("/kaggle/input/**/" + pattern, recursive=True))
    if not hits:
        raise SystemExit("%s not found; attach the dataset" % what)
    return str(Path(hits[0]).parent)


def reference_checkpoints(root):
    out = {}
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir() or "__reference__" not in d.name:
            continue
        cks = sorted(d.glob("epoch_*.pt")) or sorted((d / "checkpoints").glob("epoch_*.pt"))
        if cks:
            out[d.name] = str(cks[-1])
    return out


def main(ckpt_root, assets_dir, out_dir, shard, device="cuda"):
    from continuation_core import assets as assets_mod
    from continuation_core.analysis import CheckpointEvaluator, curvature

    pinned = {s: assets_mod.load(assets_dir, s, verify_first=True) for s in (0, 1, 2)}
    cks = reference_checkpoints(ckpt_root)
    si, sn = (int(v) for v in shard.split("/"))
    todo = [(c, p, s) for c in sorted(cks) for p in POLICIES for s in SPLITS][si::sn]
    print("shard %s: %d of %d measurements" % (shard, len(todo), 4 * len(cks)), flush=True)

    t0, rows = time.perf_counter(), []
    for cell, policy, split in todo:
        seed = int(cell.rsplit("seed", 1)[-1])
        ev = CheckpointEvaluator(cks[cell], device=device, split=split,
                                 batch_size=500, assets=pinned.get(seed))
        c = curvature.probe(ev, n_images=2000, batch_size=500, draws=24,
                            top_k=1, power_iters=25, seed=seed, bn_policy=policy)
        rows.append({"cell": cell, "method": cell.split("__")[0], "seed": seed,
                     "bn_policy": policy, "split": split,
                     "trace_H": c["trace"]["mean"], "trace_H_sem": c["trace"]["sem"],
                     "lambda_max": c["top_eigenvalues"][0]["eigenvalue"],
                     "per_block": {k: v["mean"] for k, v in c["per_block"].items()}})
        r = rows[-1]
        print("  %-38s %-18s %-6s trH=%10.1f lam=%9.1f  %5.0fs"
              % (cell, policy, split, r["trace_H"], r["lambda_max"],
                 time.perf_counter() - t0), flush=True)
        del ev

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / ("bnpolicy_shard%d.json" % si)).write_text(
        json.dumps({"shard": shard, "policies": list(POLICIES),
                    "splits": list(SPLITS), "rows": rows}, indent=2))
    print("done in %.0f s" % (time.perf_counter() - t0))


if __name__ == "__main__":
    shard = os.environ.get("BNPOL_SHARD", "0/4")
    find("cifar-10-batches-py", "CIFAR-10")
    assets = find("assets_manifest.json", "pinned assets")
    ckpts = find("grid_checkpoints_manifest.json", "staged grid checkpoints")
    out = Path(os.environ.get("STUDY_OUT", "/kaggle/working"))
    print("assets=%s\ncheckpoints=%s\nshard=%s" % (assets, ckpts, shard), flush=True)
    main(ckpts, assets, str(out), shard)
