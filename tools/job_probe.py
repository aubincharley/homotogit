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


def find_checkpoints(root):
    """Last checkpoint of every cell, keyed by cell name.

    Accepts either the staged dataset layout (``<cell>/epoch_NNN.pt``) or a live
    run tree (``<cell>/checkpoints/epoch_NNN.pt``), so the same job runs against
    a published dataset and against local output.
    """
    out = {}
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir():
            continue
        cks = sorted(d.glob("epoch_*.pt")) or sorted((d / "checkpoints").glob("epoch_*.pt"))
        if cks:
            out[d.name] = str(cks[-1])
    return out


def main(runs_dir, data_root, out_dir, shard, device="cuda", assets_dir=None):
    from continuation_core import assets as assets_mod
    from continuation_core.analysis import CheckpointEvaluator, curvature, sensitivity

    # The checkpoint records the asset directory as it existed on the *training*
    # worker.  Loading them here explicitly, with digests verified, decouples the
    # probe from that path and fails loudly on a mismatched asset set rather than
    # pairing runs against different initial weights.
    pinned = {s: assets_mod.load(assets_dir, s, verify_first=True) for s in (0, 1, 2)} \
        if assets_dir else {}

    si, sn = (int(v) for v in shard.split("/"))
    cks = find_checkpoints(runs_dir)
    keys = sorted(cks)[si::sn]
    print("shard %s: %d of %d checkpoints" % (shard, len(keys), len(cks)), flush=True)

    t0 = time.perf_counter()
    results = {}
    for name in keys:
        path = cks[name]
        # -- exact gap, whole pinned training subset ------------------------
        seed = int(name.rsplit("seed", 1)[-1])
        tr = CheckpointEvaluator(path, device=device, split="train", batch_size=1000,
                                 assets=pinned.get(seed))
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
    assets = find("assets_manifest.json", "pinned assets")
    ckpts = find("grid_checkpoints_manifest.json", "staged grid checkpoints")
    out = Path(os.environ.get("STUDY_OUT", "/kaggle/working"))
    print("data=%s\nassets=%s\ncheckpoints=%s\nshard=%s"
          % (data_root, assets, ckpts, shard), flush=True)
    # the checkpoint carries the config it was trained with, including the asset
    # directory as it existed on the training worker; point it at this worker's
    main(ckpts, data_root, str(out), shard, assets_dir=assets)
