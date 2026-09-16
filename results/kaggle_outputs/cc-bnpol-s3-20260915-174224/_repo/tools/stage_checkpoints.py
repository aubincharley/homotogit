"""Strip and publish grid checkpoints as Kaggle datasets.

A checkpoint is 2.25 MB, of which the model state is 1.08 MB; the optimiser
state, the RNG state and the metric history are not read by any probe.  Stripping
them halves the transfer and keeps exactly what ``CheckpointEvaluator`` needs:
``model_state``, ``config``, ``intervention``, ``global_update`` and ``schema``.

Two sets, because they have very different sizes and the first is wanted
immediately:

``final``  the last epoch of all 28 cells            ~31 MB
``traj``   every epoch of the 12 reference cells     ~400 MB

The directory layout of the runs is preserved, so the probe jobs find each
checkpoint under its own cell name and can recover the cell from the path.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
KEEP = ("schema", "model_state", "config", "intervention", "global_update",
        "epochs_completed")


def run_dirs():
    for shard in sorted((ROOT / "results/kaggle_outputs").glob("cc-grid-s*/runs")):
        for d in sorted(shard.iterdir()):
            if d.is_dir() and (d / "checkpoints").is_dir():
                yield d


def stage(which: str, dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    pat = re.compile(r"epoch_(\d+)\.pt$")
    man, n, total, unreadable = {}, 0, 0, []
    for d in run_dirs():
        cks = sorted((d / "checkpoints").glob("epoch_*.pt"))
        if not cks:
            continue
        if which == "final":
            cks = cks[-1:]
        elif "__reference__" not in d.name:
            continue
        out = dest / d.name
        out.mkdir(exist_ok=True)
        for f in cks:
            try:
                obj = torch.load(f, map_location="cpu", weights_only=False)
            except Exception as exc:
                # a download can truncate a file; skipping it silently would put a
                # hole in the trajectory that nothing downstream would notice
                unreadable.append({"path": str(f), "error": type(exc).__name__,
                                   "bytes": f.stat().st_size})
                continue
            slim = {k: obj[k] for k in KEEP if k in obj}
            p = out / f.name
            torch.save(slim, p)
            man[str(p.relative_to(dest))] = {"bytes": p.stat().st_size,
                                             "epoch": int(pat.search(f.name).group(1))}
            n += 1
            total += p.stat().st_size
    (dest / "grid_checkpoints_manifest.json").write_text(json.dumps(
        {"which": which, "n": n, "kept_keys": list(KEEP), "files": man,
         "unreadable": unreadable}, indent=2))
    print("staged %d checkpoints (%.0f MB) into %s" % (n, total / 1e6, dest))
    if unreadable:
        print("UNREADABLE and skipped (%d):" % len(unreadable))
        for u in unreadable:
            print("   %s  %s  %d bytes" % (u["path"], u["error"], u["bytes"]))
    return man


def publish(dest: Path, slug: str, account: str) -> str:
    cfg = Path.home() / ".kaggle-accounts" / account
    env = {**os.environ, "KAGGLE_CONFIG_DIR": str(cfg)}
    user = json.loads((cfg / "kaggle.json").read_text())["username"]
    ref = "%s/%s" % (user, slug)
    (dest / "dataset-metadata.json").write_text(json.dumps(
        {"title": slug, "id": ref, "licenses": [{"name": "unknown"}]}, indent=2))

    def sh(cmd):
        print("$", " ".join(cmd), flush=True)
        return subprocess.run(cmd, text=True, capture_output=True, env=env)

    listing = sh(["kaggle", "datasets", "list", "--mine", "-s", slug])
    cmd = (["kaggle", "datasets", "version", "-p", str(dest), "-m", "update"]
           if ref in (listing.stdout or "") else
           ["kaggle", "datasets", "create", "-p", str(dest)])
    r = sh(cmd + ["--dir-mode", "zip"])
    print((r.stdout or "") + (r.stderr or ""))
    (dest / "dataset-metadata.json").unlink(missing_ok=True)
    if r.returncode != 0:
        raise SystemExit("publishing to %s failed" % ref)
    return ref


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", choices=("final", "traj"), required=True)
    ap.add_argument("--accounts", nargs="*", default=[])
    a = ap.parse_args()
    slug = "cc-grid-%s" % a.which
    dest = ROOT / "kaggle_jobs" / ("_%s" % slug)
    stage(a.which, dest)
    for acc in a.accounts:
        print("published:", publish(dest, slug, acc))
