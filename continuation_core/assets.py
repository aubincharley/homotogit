"""Pinned shared state: initial weights, BN buffers, probe indices, data order.

Layout of an asset directory::

    assets_manifest.json     digests + provenance
    init_seed<k>.pt          model state_dict (parameters AND BatchNorm buffers)
    shared_indices.npz       subset [N] int64, train_probe [P] int64,
                             perm_seed<k> [epochs, N] int32

Semantics (as executed by the benchmark):

* ``subset`` indexes the official training split; the training tensor is
  ``train.images[subset]``;
* ``train_probe`` indexes that *subset-ordered* tensor;
* ``perm_seed<k>[e]`` is the example order of epoch ``e`` over the same tensor;
  batches are consecutive slices of 128, the last one shorter.

Digests: ``sha256`` of each file, of each array's contiguous bytes, and of each
state dict (keys sorted, name bytes then tensor bytes).  Initial weights depend
on the torch build, so they are never regenerated for the reference preset.
``make_assets(..., init_from=<dir>)`` copies them from an existing set instead of
drawing new ones, which both removes that dependence and lets two datasets share
one initialization (ResNet-20's parameters do not depend on the input size).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .seeding import derive_seed


class AssetMismatch(RuntimeError):
    pass


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_array(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha_state(state: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(np.ascontiguousarray(state[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def verify(asset_dir) -> dict:
    """Recompute every digest; raises :class:`AssetMismatch` on any difference."""
    d = Path(asset_dir)
    man = json.loads((d / "assets_manifest.json").read_text(encoding="utf-8"))
    checks = {}
    for name, info in man.get("files", {}).items():
        checks["file:" + name] = sha_file(d / name) == info["sha256"]
    z = np.load(d / "shared_indices.npz")
    for name, info in man["arrays"].items():
        checks["array:" + name] = sha_array(z[name]) == info["sha256"]
    for name, digest in man["states"].items():
        sd = torch.load(d / (name + ".pt"), map_location="cpu", weights_only=True)
        checks["state:" + name] = sha_state(sd) == digest
    bad = sorted(k for k, ok in checks.items() if not ok)
    if bad:
        raise AssetMismatch("pinned assets in %s fail verification: %s" % (d, bad))
    return {"dir": str(d), "checks": checks, "all_match": True}


def load(asset_dir, seed: int, verify_first: bool = True) -> dict:
    d = Path(asset_dir)
    report = verify(d) if verify_first else None
    z = np.load(d / "shared_indices.npz")
    key = "perm_seed%d" % int(seed)
    if key not in z.files:
        raise AssetMismatch("no data order for seed %d in %s (have %s)"
                            % (seed, d, sorted(f for f in z.files if f.startswith("perm"))))
    return {"subset": z["subset"], "train_probe": z["train_probe"], "perms": z[key],
            "init_state": torch.load(d / ("init_seed%d.pt" % int(seed)),
                                     map_location="cpu", weights_only=True),
            "manifest": json.loads((d / "assets_manifest.json").read_text(encoding="utf-8")),
            "verification": report}


def make_assets(out_dir, model_builder, n_train: int, epochs: int, seeds=(0, 1, 2),
                probe_size: int = 500, provenance: dict | None = None,
                init_from=None) -> dict:
    """Generate a NEW asset set for a dataset/model without a pinned one.

    Deterministic given ``(n_train, epochs, seeds, probe_size)`` and the torch
    build.  The result is a new asset identity: it is not paired with the
    CIFAR-10 reference set and must be recorded as such.

    ``init_from`` takes the initial states from an existing asset directory
    instead of drawing them, after checking that each loads strictly into
    ``model_builder()``.  Only the index arrays are then new, and since those come
    from numpy's PCG64 rather than torch, the whole set is bit-identical on any
    machine.  It is still a new asset identity: ``paired_with_reference`` stays
    ``False``.
    """
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    src = None if init_from is None else Path(init_from)
    rng = np.random.default_rng(derive_seed(0, "subset"))
    arrays = {"subset": rng.permutation(n_train).astype(np.int64)}
    arrays["train_probe"] = np.sort(np.random.default_rng(derive_seed(0, "probe"))
                                    .permutation(n_train)[:probe_size]).astype(np.int64)
    states = {}
    for s in seeds:
        g = np.random.default_rng(derive_seed(s, "batch"))
        arrays["perm_seed%d" % s] = np.stack([g.permutation(n_train)
                                              for _ in range(epochs)]).astype(np.int32)
        if src is None:
            torch.manual_seed(derive_seed(s, "init"))
            sd = model_builder().state_dict()
        else:
            sd = torch.load(src / ("init_seed%d.pt" % s), map_location="cpu",
                            weights_only=True)
            model_builder().load_state_dict(sd, strict=True)
        torch.save(sd, d / ("init_seed%d.pt" % s))
        states["init_seed%d" % s] = sha_state(sd)
    np.savez(d / "shared_indices.npz", **arrays)
    man = {"generated_by": "continuation_core.assets.make_assets",
           "torch": torch.__version__,
           "provenance": {**(provenance or {}),
                          "init_from": None if src is None else str(src),
                          "probe_size": int(probe_size)},
           "paired_with_reference": False,
           "arrays": {k: {"sha256": sha_array(v), "shape": list(v.shape),
                          "dtype": str(v.dtype)} for k, v in arrays.items()},
           "states": states,
           "files": {p.name: {"sha256": sha_file(p), "bytes": p.stat().st_size}
                     for p in sorted(d.glob("*")) if p.suffix in (".pt", ".npz")}}
    (d / "assets_manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    return man
