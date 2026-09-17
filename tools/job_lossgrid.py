"""Train one slice of the objective grid on a Kaggle worker.

Same shape as ``job_grid.py``: ``LOSS_SHARD`` picks the slice and is set inside
the shard entrypoint, never exported by the caller, because the launcher ships a
script and not a shell.

``LOSS_ARMS`` selects which objectives to run, comma-separated.  The default
drops ``ce``: this branch's change is a no-op for cross-entropy, so the twelve
recorded reference cells remain valid as the control and need not be paid for
again.  Pass ``LOSS_ARMS=ce,ls,focal,square`` to make a launch self-contained.
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


if __name__ == "__main__":
    shard = os.environ.get("LOSS_SHARD", "0/4")
    si, sn = (int(v) for v in shard.split("/"))
    arms = tuple(a for a in os.environ.get("LOSS_ARMS", "ls,focal,square").split(",") if a)
    data_root = find("cifar-10-batches-py", "CIFAR-10")
    assets = find("assets_manifest.json", "pinned assets")
    out = Path(os.environ.get("STUDY_OUT", "/kaggle/working"))
    out.mkdir(parents=True, exist_ok=True)
    print("data=%s\nassets=%s\nshard=%s\narms=%s" % (data_root, assets, shard, arms),
          flush=True)

    from continuation_core.grid import build_loss_grid, loss_summary
    from continuation_core.train import Trainer

    (out / "loss_plan.json").write_text(json.dumps(loss_summary(arms=arms), indent=2))
    cells = build_loss_grid(arms=arms, data_root=data_root, assets_dir=assets,
                            out_dir=str(out / "runs"), device="cuda")[si::sn]
    print("shard %d/%d: %d cells" % (si, sn, len(cells)), flush=True)

    t0 = time.perf_counter()
    done = []
    for name, cfg in cells:
        tr = Trainer(cfg, device="cuda")
        res = tr.run()
        done.append({"cell": name, "loss": cfg.loss.name, "lr": cfg.optimizer.lr,
                     "final": res["final"], "validation_status": cfg.validation_status})
        print("  %-56s %s  %5.0fs" % (name, json.dumps(res["final"]),
                                      time.perf_counter() - t0), flush=True)
    (out / ("loss_shard%d.json" % si)).write_text(
        json.dumps({"shard": shard, "arms": list(arms), "cells": done}, indent=2))
    print("done in %.0f s" % (time.perf_counter() - t0))
