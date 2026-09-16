"""Train one slice of the 28-cell grid on a Kaggle worker.

``GRID_SHARD`` picks the slice and is set inside the shard entrypoint, never
exported by the caller: the launcher ships a script, not a shell, so a locally
exported variable never reaches the worker.

Datasets are discovered by content rather than by assumed path.  The asset
manifest is verified before a single update runs -- ``continuation_core.assets``
does it on load -- so a mismatched asset set aborts instead of silently pairing
runs against different initial weights.
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
    shard = os.environ.get("GRID_SHARD", "0/4")
    si, sn = (int(v) for v in shard.split("/"))
    data_root = find("cifar-10-batches-py", "CIFAR-10")
    assets = find("assets_manifest.json", "pinned assets")
    out = Path(os.environ.get("STUDY_OUT", "/kaggle/working"))
    out.mkdir(parents=True, exist_ok=True)
    print("data=%s\nassets=%s\nshard=%s" % (data_root, assets, shard), flush=True)

    from continuation_core.grid import build_grid, summary
    from continuation_core.train import Trainer

    (out / "grid_plan.json").write_text(json.dumps(summary(), indent=2))
    cells = build_grid(data_root=data_root, assets_dir=assets,
                       out_dir=str(out / "runs"), device="cuda")[si::sn]
    print("shard %d/%d: %d cells" % (si, sn, len(cells)), flush=True)

    t0 = time.perf_counter()
    done = []
    for name, cfg in cells:
        tr = Trainer(cfg, device="cuda")
        res = tr.run()
        final = res["final"]
        done.append({"cell": name, "final": final,
                     "validation_status": cfg.validation_status})
        print("  %-52s %s  %5.0fs" % (name, json.dumps(final), time.perf_counter() - t0),
              flush=True)
    (out / ("grid_shard%d.json" % si)).write_text(
        json.dumps({"shard": shard, "cells": done}, indent=2))
    print("done in %.0f s" % (time.perf_counter() - t0))
