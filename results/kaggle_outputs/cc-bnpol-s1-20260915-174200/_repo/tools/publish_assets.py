"""Publish the pinned asset set as a Kaggle dataset, once per account.

The assets are versioned in the repository (15 MB: three initial weight files and
the shared index file), which is too large for the base64 payload the kernel
launcher embeds.  They therefore travel as an attached dataset instead, with the
manifest digests intact so ``continuation_core.assets.load`` can verify them on
the worker exactly as it does locally.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "assets" / "cifar10_resnet20bn"
SLUG = "cc-assets-r20bn"


def publish(account: str) -> str:
    cfg = Path.home() / ".kaggle-accounts" / account
    env = {**os.environ, "KAGGLE_CONFIG_DIR": str(cfg)}
    user = json.loads((cfg / "kaggle.json").read_text())["username"]
    ref = "%s/%s" % (user, SLUG)
    (SRC / "dataset-metadata.json").write_text(json.dumps(
        {"title": "continuation-core pinned assets (r20bn)", "id": ref,
         "licenses": [{"name": "unknown"}]}, indent=2))

    def run(cmd):
        print("$", " ".join(cmd), flush=True)
        return subprocess.run(cmd, text=True, capture_output=True, env=env)

    listing = run(["kaggle", "datasets", "list", "--mine", "-s", SLUG])
    if ref in (listing.stdout or ""):
        r = run(["kaggle", "datasets", "version", "-p", str(SRC), "-m", "assets",
                 "--dir-mode", "zip"])
    else:
        r = run(["kaggle", "datasets", "create", "-p", str(SRC), "--dir-mode", "zip"])
    print((r.stdout or "") + (r.stderr or ""))
    (SRC / "dataset-metadata.json").unlink(missing_ok=True)
    if r.returncode != 0:
        raise SystemExit("publishing to %s failed" % ref)
    return ref


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accounts", nargs="+", required=True)
    for a in ap.parse_args().accounts:
        print("published:", publish(a))
