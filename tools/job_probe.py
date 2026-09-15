"""Probe the finished checkpoints of the grid: gap, curvature, gauge, sensitivity.

``PROBE_SHARD`` picks the slice, set inside the shard entrypoint.

Order matters.  The **exact gap** comes first and is computed on the whole pinned
training subset, not on the 500-image probe: with the probe, the gap carries 1.3
points of sampling noise against a real spread of 0.4, which caps any correlation
at 0.25 and makes the analysis unable to answer either way.  The **validity
gate** comes second: a cell is admitted only if its probed test error reproduces
the one the trainer recorded, which catches a network being evaluated as a
different function.  Only then are the expensive probes run.
"""
import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def find(pattern, what):
    hits = sorted(glob.glob("/kaggle/input/**/" + pattern, recursive=True))
    if not hits:
        raise SystemExit("%s not found; attach the dataset" % what)
    return str(Path(hits[0]).parent)


def find_checkpoints(runs_dir):
    """Final checkpoint of every run, keyed by the run directory name."""
    out = {}
    for d in sorted(Path(runs_dir).iterdir()):
        cks = sorted((d / "checkpoints").glob("checkpoint_epoch_*.pt")) if d.is_dir() else []
        if cks:
            out[d.name] = str(cks[-1])
    return out


def main(runs_dir, data_root, out_dir, shard, device="cuda"):
    from continuation_core.analysis import CheckpointEvaluator, curvature, sensitivity

    si, sn = (int(v) for v in shard.split("/"))
    cks = find_checkpoints(runs_dir)
    keys = sorted(cks)[si::sn]
    print("shard %s: %d of %d checkpoints" % (shard, len(keys), len(cks)), flush=True)

    t0 = time.perf_counter()
    results = {}
    for name in keys:
        path = cks[name]
        # -- exact gap, whole pinned training subset ------------------------
        tr = CheckpointEvaluator(path, device=device, split="train", batch_size=1000)
        te = CheckpointEvaluator(path, device=device, split="test", batch_size=1000)
        r_tr, r_te = tr.loss(state="target"), te.loss(state="target")
        rec = {"cell": name, "checkpoint": path,
               "train": {"ce": r_tr["ce"], "err": 1.0 - r_tr["acc"], "n": int(tr.images.shape[0])},
               "test": {"ce": r_te["ce"], "err": 1.0 - r_te["acc"], "n": int(te.images.shape[0])},
               "gap_err": (1.0 - r_te["acc"]) - (1.0 - r_tr["acc"])}
        del te

        # -- probes, on training images only, so they stay predictors -------
        rec["sensitivity"] = sensitivity.probe(tr, n_images=1000, batch_size=250)
        rec["curvature"] = curvature.probe(tr, n_images=5000, batch_size=500,
                                           draws=64, top_k=5, power_iters=40,
                                           seed=int(tr.cfg.run.seed))
        results[name] = rec
        s, c = rec["sensitivity"]["jacobian"], rec["curvature"]
        print("  %-52s gap=%.4f |J|=%7.2f trH=%9.1f  %5.0fs"
              % (name, rec["gap_err"], s["frobenius_mean"], c["trace"]["mean"],
                 time.perf_counter() - t0), flush=True)
        del tr

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / ("probe_shard%d.json" % si)).write_text(
        json.dumps({"shard": shard, "cells": results}, indent=2))
    print("done in %.0f s" % (time.perf_counter() - t0))


if __name__ == "__main__":
    shard = os.environ.get("PROBE_SHARD", "0/4")
    data_root = find("cifar-10-batches-py", "CIFAR-10")
    runs = find("grid_plan.json", "trained grid runs")
    out = Path(os.environ.get("STUDY_OUT", "/kaggle/working"))
    print("data=%s\nruns=%s\nshard=%s" % (data_root, runs, shard), flush=True)
    main(str(Path(runs) / "runs"), data_root, str(out), shard)
