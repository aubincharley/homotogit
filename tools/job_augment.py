"""Train one slice of the augmentation grid on a Kaggle worker.

``AUG_SHARD`` picks the slice and ``AUG_CORNERS`` the corners, both set inside
the shard entrypoint and never exported by the caller -- the launcher ships a
script, not a shell.

A corner is ``aug:epochs:lr``.  The default is the 30-epoch pair at the two
learning rates, because 0.005 is demonstrably below the optimum and the point of
this grid is to reach a baseline strong enough for the question to mean anything.
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


def parse_corners(spec):
    out = []
    for item in spec.split(","):
        if not item.strip():
            continue
        aug, epochs, lr = item.split(":")
        out.append((aug, int(epochs), float(lr)))
    return tuple(out)


if __name__ == "__main__":
    shard = os.environ.get("AUG_SHARD", "0/4")
    si, sn = (int(v) for v in shard.split("/"))
    corners = parse_corners(os.environ.get(
        "AUG_CORNERS", "crop_flip:30:0.005,crop_flip:30:0.01"))
    stretch = os.environ.get("AUG_STRETCH", "0") == "1"
    data_root = find("cifar-10-batches-py", "CIFAR-10")
    assets = find("assets_manifest.json", "pinned assets")
    out = Path(os.environ.get("STUDY_OUT", "/kaggle/working"))
    out.mkdir(parents=True, exist_ok=True)
    print("data=%s\nassets=%s\nshard=%s\ncorners=%s\nstretch=%s"
          % (data_root, assets, shard, corners, stretch), flush=True)

    from continuation_core.grid import build_augment_grid, augment_summary
    from continuation_core.train import Trainer

    (out / "augment_plan.json").write_text(
        json.dumps(augment_summary(corners=corners, stretch=stretch), indent=2))
    cells = build_augment_grid(corners=corners, stretch=stretch, data_root=data_root,
                               assets_dir=assets, out_dir=str(out / "runs"),
                               device="cuda")[si::sn]
    print("shard %d/%d: %d cells" % (si, sn, len(cells)), flush=True)

    t0 = time.perf_counter()
    done = []
    for name, cfg in cells:
        tr = Trainer(cfg, device="cuda")
        res = tr.run()
        done.append({"cell": name, "augmentation": cfg.data.augmentation,
                     "epochs": cfg.budget.epochs, "lr": cfg.optimizer.lr,
                     "stretched": stretch,
                     "final": res["final"], "validation_status": cfg.validation_status})
        print("  %-60s %s  %5.0fs" % (name, json.dumps(res["final"]),
                                      time.perf_counter() - t0), flush=True)
    (out / ("augment_shard%d.json" % si)).write_text(
        json.dumps({"shard": shard, "corners": [list(c) for c in corners],
                    "stretched": stretch, "cells": done}, indent=2))
    print("done in %.0f s" % (time.perf_counter() - t0))
