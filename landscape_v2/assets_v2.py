"""Five-seed asset set for v2.

    py -m landscape_v2.assets_v2

Seeds 0-2: initial states and data orders copied from the pinned reference
assets (digests re-checked).  Seeds 3-4: generated here, reproducibly, with the
same subset (identity order) and the same 500-image train probe:

* data order ``perm_seed<k>[e] = default_rng(derive_seed(k, "landscape_v2::batch")).permutation(50000)``
  drawn for e = 0..29 in sequence;
* initial state: ``torch.manual_seed(derive_seed(k, "landscape_v2::init"))`` then
  the ResNet-20 constructor (Kaiming-normal convs, BN weight 1 / bias 0, fc
  N(0, 0.01)); parameters and BN buffers saved.

Seeds 3-4 are a new asset identity (generated with this machine's torch build);
they are not previously pinned.
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import torch

from continuation_core import assets as A
from continuation_core.models import build_model
from continuation_core.seeding import derive_seed

from .common import EPOCHS, PINNED_ASSETS, STUDY

OUT = STUDY / "inputs" / "assets"


def main():
    if (OUT / "assets_manifest.json").exists():
        print(A.verify(OUT)["all_match"])
        return
    OUT.mkdir(parents=True, exist_ok=True)
    A.verify(PINNED_ASSETS)
    z = np.load(PINNED_ASSETS / "shared_indices.npz")
    arrays = {"subset": z["subset"], "train_probe": z["train_probe"]}
    for s in (0, 1, 2):
        arrays["perm_seed%d" % s] = z["perm_seed%d" % s]
        shutil.copy2(PINNED_ASSETS / ("init_seed%d.pt" % s), OUT / ("init_seed%d.pt" % s))
    n = int(z["subset"].size)
    for s in (3, 4):
        g = np.random.default_rng(derive_seed(s, "landscape_v2::batch"))
        arrays["perm_seed%d" % s] = np.stack([g.permutation(n) for _ in range(EPOCHS)]).astype(np.int32)
        torch.manual_seed(derive_seed(s, "landscape_v2::init"))
        torch.save(build_model("resnet20_bn_cifar", 10).state_dict(), OUT / ("init_seed%d.pt" % s))
    np.savez(OUT / "shared_indices.npz", **arrays)
    pinned = json.loads((PINNED_ASSETS / "assets_manifest.json").read_text())
    states = {}
    for s in range(5):
        sd = torch.load(OUT / ("init_seed%d.pt" % s), map_location="cpu", weights_only=True)
        states["init_seed%d" % s] = A.sha_state(sd)
    man = {"generated_by": "landscape_v2.assets_v2", "torch": torch.__version__,
           "seeds": {"0-2": "copied from assets/cifar10_resnet20bn (pinned reference)",
                     "3-4": "generated: see module docstring"},
           "arrays": {k: {"sha256": A.sha_array(v), "shape": list(v.shape), "dtype": str(v.dtype)}
                      for k, v in arrays.items()},
           "states": states,
           "files": {p.name: {"sha256": A.sha_file(p), "bytes": p.stat().st_size}
                     for p in sorted(OUT.glob("*")) if p.suffix in (".pt", ".npz")}}
    for s in (0, 1, 2):
        assert states["init_seed%d" % s] == pinned["states"]["init_seed%d" % s]
        assert man["arrays"]["perm_seed%d" % s]["sha256"] == pinned["arrays"]["perm_seed%d" % s]["sha256"]
    for k in ("subset", "train_probe"):
        assert man["arrays"][k]["sha256"] == pinned["arrays"][k]["sha256"]
    (OUT / "assets_manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print("verified:", A.verify(OUT)["all_match"], OUT)


if __name__ == "__main__":
    main()
