"""Publish the pinned asset set as a private Kaggle dataset.

The assets are 15 MB -- ``shared_indices.npz`` alone is 12 MB -- so they cannot
ride inside the kernel source, which ``kaggle_run.py`` caps at 900 kB.  They go
up once as a dataset and every job mounts them.

Why they must be *these* files and not regenerated on the worker: the initial
weights depend on the exact torch build that produced them, so a fresh
``make-assets`` on Kaggle would give different numbers and silently break the
pairing with the recorded SGD runs this benchmark reuses.  The digests in
``assets_manifest.json`` are what makes that guarantee checkable, and
``job_optimizer_benchmark.py`` re-checks them on the worker before training.

    py scripts/stage_assets.py                  # create, or push a new version
    py scripts/stage_assets.py --dry-run        # stage and report only
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kaggle_run import _kaggle_username, _run, select_account  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "assets" / "cifar10_resnet20bn"
STAGE = REPO_ROOT / "kaggle_jobs" / "_assets"
SLUG = "continuation-core-r20bn-assets"
TITLE = "continuation-core r20bn pinned assets"


def stage(dry_run: bool = False, account=None) -> str:
    select_account(account)
    if not ASSETS.is_dir():
        raise SystemExit("no asset directory at %s" % ASSETS)

    # raises AssetMismatch if anything on disk drifted from the manifest, so a
    # corrupted copy is never what gets published
    from continuation_core import assets as assets_mod
    report = assets_mod.verify(ASSETS)
    print("verified %d digests locally" % len(report["checks"]))

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    for f in sorted(ASSETS.iterdir()):
        if f.is_file():
            shutil.copy2(f, STAGE / f.name)

    owner = _kaggle_username()
    ref = "%s/%s" % (owner, SLUG)
    (STAGE / "dataset-metadata.json").write_text(json.dumps(
        {"title": TITLE, "id": ref,
         "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")

    total = sum(f.stat().st_size for f in STAGE.iterdir() if f.is_file())
    print("staged %d files (%.1f MB) in %s" % (
        len(list(STAGE.iterdir())), total / 1e6, STAGE))
    print("dataset ref: %s" % ref)
    print("pass this to the job launcher as --dataset %s" % ref)
    if dry_run:
        print("dry run - nothing pushed")
        return ref

    exists = _run(["kaggle", "datasets", "status", ref]).returncode == 0
    if exists:
        cmd = ["kaggle", "datasets", "version", "-p", str(STAGE),
               "-m", "refresh pinned assets", "--dir-mode", "zip"]
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(STAGE), "--dir-mode", "zip"]
    res = _run(cmd)
    print(res.stdout or res.stderr)
    if res.returncode != 0:
        raise RuntimeError("kaggle datasets %s failed"
                           % ("version" if exists else "create"))
    return ref


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--account", default=None)
    args = p.parse_args()
    stage(dry_run=args.dry_run, account=args.account)


if __name__ == "__main__":
    main()
