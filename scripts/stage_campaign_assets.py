"""Pin the campaign's shared assets and publish them as a Kaggle dataset.

The 21-configuration campaign must start from *exactly* the state the completed
full-data campaign used, otherwise the reused cells are not paired with the new
ones.  Those assets cannot be regenerated locally -- initial weights depend on
the torch build, and this machine runs 2.5.1 against Kaggle's 2.10.0 -- so the
completed campaign's own artifacts are the source of truth:

    init_seed0.pt / init_seed1.pt / init_seed2.pt   initial weights + BN buffers
    shared_indices.npz                              subset, train probe, per-epoch
                                                    permutations for seeds 0/1/2

They total ~15 MB, far past the 900 kB kernel-source cap, so they ship as a
Kaggle dataset attached to every job rather than embedded in notebook source.
A private copy is uploaded per account (a private dataset is not visible to the
other accounts).  ``assets_manifest.json`` carries the sha256 of every file and
array; each job re-checks them before training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "results" / "kaggle_outputs" / "fulldata-r20bn-20260908-161221"
          / "fulldata_r20bn_20260908-161234")
STAGE = ROOT / "kaggle_jobs" / "_assets"
SLUG = "r20bn-campaign-assets"
FILES = ("init_seed0.pt", "init_seed1.pt", "init_seed2.pt", "shared_indices.npz")


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_array(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha_state(p: Path) -> str:
    sd = torch.load(p, map_location="cpu", weights_only=True)
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(np.ascontiguousarray(sd[k].cpu().numpy()).tobytes())
    return h.hexdigest()


def build_manifest() -> dict:
    """Content hashes of the pinned assets (file-level and array/state-level)."""
    man = {"source_job": "fulldata-r20bn-20260908-161221",
           "source_study": SOURCE.name, "files": {}, "arrays": {}, "states": {}}
    for name in FILES:
        p = SOURCE / name
        if not p.is_file():
            raise SystemExit("missing pinned asset: %s" % p)
        man["files"][name] = {"sha256": sha_file(p), "bytes": p.stat().st_size}
    z = np.load(SOURCE / "shared_indices.npz")
    for k in z.files:
        man["arrays"][k] = {"sha256": sha_array(z[k]), "shape": list(z[k].shape),
                            "dtype": str(z[k].dtype)}
    for name in FILES[:3]:
        man["states"][name.replace(".pt", "")] = sha_state(SOURCE / name)
    return man


def stage() -> dict:
    STAGE.mkdir(parents=True, exist_ok=True)
    man = build_manifest()
    for name in FILES:
        shutil.copy2(SOURCE / name, STAGE / name)
    (STAGE / "assets_manifest.json").write_text(json.dumps(man, indent=2))
    total = sum(v["bytes"] for v in man["files"].values())
    print("staged %d files (%.1f MB) into %s" % (len(FILES), total / 1e6, STAGE))
    for k, v in man["states"].items():
        print("  %-12s %s" % (k, v[:16]))
    for k, v in man["arrays"].items():
        print("  %-12s %s %s" % (k, v["sha256"][:16], v["shape"]))
    return man


def publish(account: str, man: dict, public: bool = False):
    """Create or version the dataset under one account."""
    import os
    cfg = Path.home() / ".kaggle-accounts" / account
    if not (cfg / "kaggle.json").is_file():
        raise SystemExit("no credentials for %r" % account)
    env = {**os.environ, "KAGGLE_CONFIG_DIR": str(cfg)}
    username = json.loads((cfg / "kaggle.json").read_text())["username"]
    ref = "%s/%s" % (username, SLUG)

    meta = {"title": "r20bn campaign assets", "id": ref,
            "licenses": [{"name": "unknown"}]}
    (STAGE / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    def run(cmd):
        print("$ %s" % " ".join(cmd), flush=True)
        return subprocess.run(cmd, text=True, capture_output=True, env=env)

    exists = run(["kaggle", "datasets", "status", ref]).returncode == 0
    if exists:
        r = run(["kaggle", "datasets", "version", "-p", str(STAGE),
                 "-m", "pinned campaign assets", "--dir-mode", "zip"])
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(STAGE), "--dir-mode", "zip"]
        if not public:
            cmd.append("--public") if public else None
        r = run(cmd)
    print(r.stdout or r.stderr)
    if r.returncode != 0:
        raise SystemExit("publishing to %s failed" % ref)
    return ref


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accounts", nargs="*", default=[])
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()
    man = stage()
    refs = [publish(a, man, args.public) for a in args.accounts]
    if refs:
        (STAGE / "dataset_refs.json").write_text(json.dumps(refs, indent=2))
        print("dataset refs:", refs)


if __name__ == "__main__":
    main()
