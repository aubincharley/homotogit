"""Pin the **anti-aliasing ablation**'s shared assets and publish them per account.

Why this exists instead of reusing ``stage_campaign_assets.py``
--------------------------------------------------------------
That script *copies* the completed full-data campaign's artifacts
(``init_seed*.pt``, ``shared_indices.npz``).  Those artifacts are gitignored and
are not present in this checkout, and the Kaggle dataset that carried them lives
on the accounts that ran the original campaign -- not on the accounts configured
here.  They are therefore **unrecoverable**, and the ablation cannot be paired
with the historical 63-cell grid by shared initial weights.

The honest consequence, applied here: generate a *fresh* pinned asset set and
run the ablation's **own controls** (plain and Gaussian-plateau) inside the same
batch.  Every arm of the ablation is then paired with every other by construction
-- identical initial weights, identical BN buffers, identical per-epoch
permutations, identical probe indices -- which is the comparison the ablation
actually needs.  Nothing here may be compared cell-for-cell against the numbers
in ``results/campaign_results.json``; see errata C-24/C-25.

Determinism note
----------------
``subset`` and the per-epoch permutations come from NumPy's PCG64 streams and are
platform- and version-independent.  The initial weights come from torch's RNG and
therefore *do* depend on the torch build -- which is precisely why they are
generated once, here, and shipped as a pinned dataset rather than regenerated on
each worker.  Reproducing them is not required; being identical across every arm
is, and that is what pinning guarantees.

Usage
-----
    py scripts/stage_ablation_assets.py --accounts idrisselkhamlichi idrisselk
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from continuation.config import DataConfig, ModelConfig       # noqa: E402
from continuation.data import build_dataset, fixed_subset_indices  # noqa: E402
from continuation.models import build_model, count_parameters     # noqa: E402
from continuation.seeding import numpy_generator                  # noqa: E402

STAGE = ROOT / "kaggle_jobs" / "_ablation_assets"
SLUG = "r20bn-ablation-assets"

# Frozen generation parameters.  These mirror the full-data campaign's protocol
# so the ablation runs the same recipe -- but the resulting weights are new.
SPEC = {
    "arch": "resnet20_bn_cifar",
    "dataset": "cifar10",
    "num_val": 0,                # full 50,000 official train split
    "subset_seed": 0,
    "per_class": 5000,
    "n_subset": 50000,
    "epochs": 30,
    "seeds": (0, 1, 2),
    "probe_seed": 0,
    "train_probe": 500,
}


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_array(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha_state(p: Path) -> str:
    """Digest of a state_dict, key-sorted -- the same rule campaign_driver uses."""
    sd = torch.load(p, map_location="cpu", weights_only=True)
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(np.ascontiguousarray(sd[k].cpu().numpy()).tobytes())
    return h.hexdigest()


def stage(data_root: str) -> dict:
    STAGE.mkdir(parents=True, exist_ok=True)
    bundle = build_dataset(DataConfig(root=data_root, download=False,
                                      num_val=SPEC["num_val"]))
    labels = bundle.train.labels.numpy()

    rng = np.random.default_rng(SPEC["subset_seed"])
    picks = [rng.permutation(np.flatnonzero(labels == c))[:SPEC["per_class"]]
             for c in range(bundle.num_classes)]
    subset = np.sort(np.concatenate(picks))
    assert subset.size == SPEC["n_subset"], (subset.size, SPEC["n_subset"])
    counts = np.bincount(labels[subset], minlength=bundle.num_classes).tolist()
    assert counts == [SPEC["per_class"]] * bundle.num_classes, counts

    train_probe = fixed_subset_indices(SPEC["n_subset"], SPEC["train_probe"],
                                       SPEC["probe_seed"], "study_train_probe",
                                       labels=labels[subset])

    perms = {}
    for seed in SPEC["seeds"]:
        model = build_model(ModelConfig(arch=SPEC["arch"]), bundle.num_classes,
                            seed=seed)
        torch.save(model.state_dict(), STAGE / ("init_seed%d.pt" % seed))
        g = numpy_generator(seed, "batch")
        perms[seed] = np.stack([g.permutation(SPEC["n_subset"]).astype(np.int32)
                                for _ in range(SPEC["epochs"])])

    np.savez_compressed(STAGE / "shared_indices.npz", subset=subset,
                        train_probe=train_probe,
                        **{"perm_seed%d" % s: perms[s] for s in SPEC["seeds"]})

    z = np.load(STAGE / "shared_indices.npz")
    man = {
        "purpose": "pinned shared state for the anti-aliasing ablation",
        "generated_by": "scripts/stage_ablation_assets.py",
        "spec": SPEC,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "params": count_parameters(build_model(ModelConfig(arch=SPEC["arch"]),
                                               bundle.num_classes, 0))["total"],
        "normalization": {"mean": [float(v) for v in bundle.mean],
                          "std": [float(v) for v in bundle.std]},
        "not_paired_with": ("results/campaign_results.json -- different initial "
                            "weights; the ablation carries its own controls"),
        "arrays": {k: {"sha256": sha_array(z[k]), "shape": list(z[k].shape),
                       "dtype": str(z[k].dtype)} for k in z.files},
        "states": {("init_seed%d" % s): sha_state(STAGE / ("init_seed%d.pt" % s))
                   for s in SPEC["seeds"]},
        "files": {},
    }
    for f in sorted(STAGE.glob("*")):
        if f.name in ("assets_manifest.json", "dataset-metadata.json"):
            continue
        man["files"][f.name] = {"sha256": sha_file(f), "bytes": f.stat().st_size}
    (STAGE / "assets_manifest.json").write_text(json.dumps(man, indent=2))

    total = sum(v["bytes"] for v in man["files"].values())
    print("staged %d files (%.1f MB) into %s" % (len(man["files"]), total / 1e6, STAGE))
    for k, v in man["states"].items():
        print("  %-12s %s" % (k, v[:16]))
    for k, v in man["arrays"].items():
        print("  %-12s %s %s" % (k, v["sha256"][:16], v["shape"]))
    return man


def publish(account: str, public: bool = False) -> str:
    cfg = Path.home() / ".kaggle-accounts" / account
    if not (cfg / "kaggle.json").is_file():
        raise SystemExit("no credentials for %r at %s" % (account, cfg))
    env = {**os.environ, "KAGGLE_CONFIG_DIR": str(cfg)}
    username = json.loads((cfg / "kaggle.json").read_text())["username"]
    ref = "%s/%s" % (username, SLUG)

    (STAGE / "dataset-metadata.json").write_text(json.dumps(
        {"title": "r20bn ablation assets", "id": ref,
         "licenses": [{"name": "unknown"}]}, indent=2))

    def run(cmd):
        print("$ %s" % " ".join(cmd), flush=True)
        return subprocess.run(cmd, text=True, capture_output=True, env=env)

    listing = run(["kaggle", "datasets", "list", "--mine", "-s", SLUG])
    exists = ref in (listing.stdout or "")
    if exists:
        r = run(["kaggle", "datasets", "version", "-p", str(STAGE),
                 "-m", "pinned ablation assets", "--dir-mode", "zip"])
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(STAGE), "--dir-mode", "zip"]
        if public:
            cmd.append("--public")
        r = run(cmd)
    print((r.stdout or "") + (r.stderr or ""))
    if r.returncode != 0:
        raise SystemExit("publishing to %s failed" % ref)
    return ref


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accounts", nargs="*", default=[],
                    help="account names under ~/.kaggle-accounts")
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--stage-only", action="store_true")
    args = ap.parse_args()

    stage(args.data_root)
    if args.stage_only:
        return
    refs = [publish(a, args.public) for a in args.accounts]
    if refs:
        (STAGE / "dataset_refs.json").write_text(json.dumps(refs, indent=2))
        print("dataset refs:", refs)


if __name__ == "__main__":
    main()
