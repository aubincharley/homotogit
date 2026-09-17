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
TITLE = "continuation-core pinned assets (r20bn)"


def publish(account: str, src: Path = None, slug: str = None,
            title: str = None) -> str:
    src, slug = Path(src or SRC), slug or SLUG
    title = title or TITLE
    cfg = Path.home() / ".kaggle-accounts" / account
    env = {**os.environ, "KAGGLE_CONFIG_DIR": str(cfg)}
    user = json.loads((cfg / "kaggle.json").read_text())["username"]
    ref = "%s/%s" % (user, slug)
    (src / "dataset-metadata.json").write_text(json.dumps(
        {"title": title, "id": ref, "licenses": [{"name": "unknown"}]}, indent=2))

    def run(cmd):
        print("$", " ".join(cmd), flush=True)
        return subprocess.run(cmd, text=True, capture_output=True, env=env)

    listing = run(["kaggle", "datasets", "list", "--mine", "-s", slug])
    if ref in (listing.stdout or ""):
        r = run(["kaggle", "datasets", "version", "-p", str(src), "-m", "assets",
                 "--dir-mode", "zip"])
    else:
        r = run(["kaggle", "datasets", "create", "-p", str(src), "--dir-mode", "zip"])
    print((r.stdout or "") + (r.stderr or ""))
    (src / "dataset-metadata.json").unlink(missing_ok=True)
    if r.returncode != 0:
        raise SystemExit("publishing to %s failed" % ref)
    return ref


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accounts", nargs="+", required=True)
    ap.add_argument("--src", default=None, help="asset directory (default: the pinned one)")
    ap.add_argument("--slug", default=None, help="dataset slug (default: cc-assets-r20bn)")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    for acct in a.accounts:
        print("published:", publish(acct, src=a.src, slug=a.slug, title=a.title))
