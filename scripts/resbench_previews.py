"""Input previews at 16/24/32 and an internal activation diagnostic.

No rescaling of any training tensor: the diagnostic records how each operator
changes the mean, the contrast and the output range of a frozen feature map, and
reports those numbers rather than normalising them away.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from continuation.config import DataConfig, ModelConfig
from continuation.data import build_dataset
from continuation.models import build_model
from continuation.resolution_ops import OPERATORS, reduce_with

IDX = [14069, 16264, 16752, 22158, 23268, 23977, 27811, 30987, 35936, 43986]
OUT = Path("results/resbench_previews")
FR = {"bilinear": "Bilinéaire (AA)", "max": "Max adaptatif (O_ref)",
      "maxblur": "MaxBlur", "softpool": "SoftPool", "l2": "Reconstruction L²",
      "hminus1": "Reconstruction H⁻¹", "perceptual": "Perceptuel"}


def stats(t):
    return {"mean": float(t.mean()), "std": float(t.std()),
            "min": float(t.min()), "max": float(t.max())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = build_dataset(DataConfig(root="data", download=False, num_val=0))
    x = bundle.train.images[torch.as_tensor(IDX)].to(torch.float32) / 255.0

    rows = [("original 32", x)]
    report = {"input": {"original": stats(x)}, "activation": {}}
    for r in (24, 16):
        for op in OPERATORS:
            y = reduce_with(x, r, op)
            rows.append(("%s %d" % (FR[op], r), y))
            report["input"].setdefault(op, {})["r%d" % r] = stats(y)

    fig, ax = plt.subplots(len(rows), 10, figsize=(14, 1.45 * len(rows)))
    for i, (name, t) in enumerate(rows):
        for j in range(10):
            a = ax[i][j]
            a.imshow(t[j].permute(1, 2, 0).clamp(0, 1).numpy(),
                     interpolation="nearest")
            a.set_xticks([])
            a.set_yticks([])
            if j == 0:
                a.set_ylabel(name, fontsize=6.5, rotation=0, ha="right",
                             va="center")
    fig.suptitle("Opérateurs de réduction sur les images d'entrée (affichage "
                 "borné à [0,1] ; les tenseurs d'entraînement ne sont pas "
                 "rognés)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "input_operators.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # internal diagnostic on one frozen feature map, identical for every operator
    m = build_model(ModelConfig(arch="resnet20_bn_cifar"), 10, seed=0).eval()
    feats = {}
    h = m.blocks[1].register_forward_hook(
        lambda _m, inp, _o: feats.setdefault("d1", inp[0].detach()))
    with torch.no_grad():
        m(x)
    h.remove()
    f = feats["d1"]
    report["activation"]["site"] = "D1 (input of blocks[2] == output of blocks[1])"
    report["activation"]["shape"] = list(f.shape)
    report["activation"]["frozen_input"] = stats(f)
    for op in OPERATORS:
        report["activation"].setdefault(op, {})
        for r in (16, 24):
            y = reduce_with(f, r, op)
            s = stats(y)
            s["contrast_ratio_vs_input"] = s["std"] / max(float(f.std()), 1e-12)
            s["mean_shift"] = s["mean"] - float(f.mean())
            report["activation"][op]["r%d" % r] = s

    (OUT / "diagnostics.json").write_text(json.dumps(report, indent=2),
                                          encoding="utf-8")
    print("wrote", OUT / "input_operators.png")
    print("%-11s %-28s %-28s" % ("operator", "activation r=16 (mean/std/rng)",
                                 "r=24"))
    for op in OPERATORS:
        a, b = report["activation"][op]["r16"], report["activation"][op]["r24"]
        print("%-11s %6.3f/%5.3f [%5.2f,%5.2f]   %6.3f/%5.3f [%5.2f,%5.2f]"
              % (op, a["mean"], a["std"], a["min"], a["max"],
                 b["mean"], b["std"], b["min"], b["max"]))
    fi = report["activation"]["frozen_input"]
    print("%-11s %6.3f/%5.3f [%5.2f,%5.2f]  (unreduced reference)"
          % ("input", fi["mean"], fi["std"], fi["min"], fi["max"]))


if __name__ == "__main__":
    main()
