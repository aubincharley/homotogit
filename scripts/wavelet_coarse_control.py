r"""Inexpensive visual control: coarse-only reconstruction ``T_coarse = P W*(a_2, 0)``.

For the same ten images and four wavelets, this drops every detail band instead
of shrinking it.  It is a *control* for reading the previews -- it is not a
replacement for ``T_s`` and nothing else uses it.

Recorded at the existing preview levels:

    delta_s = ||T_s(x) - T_coarse(x)||_2 / ||x - mean(x)||_2

which measures how much the retained details still contribute.  Constant images
have a zero denominator and are reported as ``null`` and counted.

    py scripts/wavelet_coarse_control.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from continuation.config import DataConfig  # noqa: E402
from continuation.data import build_dataset  # noqa: E402
from continuation.pipeline import InputPipeline  # noqa: E402
from continuation.transforms.wavelet import (  # noqa: E402
    WAVELETS,
    crop,
    mirror_extend,
    swt2_analysis,
    swt2_synthesis,
    wavelet_shrink,
)
from continuation.viz import level_grid  # noqa: E402

LEVELS = [1.0, 0.9, 0.75, 0.5, 0.25, 0.0]


def coarse_only(x, wavelet, levels=2):
    """``P W* (a_2, 0)``: keep the coarsest approximation, zero every detail."""
    h, w = x.shape[-2], x.shape[-1]
    a, details = swt2_analysis(mirror_extend(x), wavelet, levels)
    zeros = [{o: torch.zeros_like(d[o]) for o in d} for d in details]
    return crop(swt2_synthesis(a, zeros, wavelet), h, w)


class _CoarseControl:
    """Minimal transform-like shim so the existing grid helper can render it."""

    parameter_name = "control"

    def __init__(self, wavelet):
        self.wavelet = wavelet

    def is_target(self, eta):
        return False

    def __call__(self, x, eta, meta=None):
        return coarse_only(x, self.wavelet)


def main(out_dir=None):
    torch.set_num_threads(6)
    out_dir = Path(out_dir or ROOT / "results" / "wavelet_previews")
    out_dir.mkdir(parents=True, exist_ok=True)
    viz = json.load(open(ROOT / "results" / "exp0_gaussian" / "figures"
                         / "visualization.json", encoding="utf-8"))
    bundle = build_dataset(DataConfig())
    idx = np.asarray(viz["example_indices_in_train_split"])
    sub = bundle.train.subset(idx, "coarse_control")
    x = InputPipeline.to_unit_float(sub.images).to(torch.float64)
    labels = sub.labels.numpy()

    denom = (x - x.mean(dim=(2, 3), keepdim=True)).flatten(1).norm(dim=1)
    valid = denom > 0
    report = {"classes": [bundle.class_names[int(l)] for l in labels],
              "s_values": LEVELS,
              "delta_definition": "||T_s(x) - T_coarse(x)||_2 / ||x - mean(x)||_2",
              "n_constant_images": int((~valid).sum()),
              "display": "fixed limits [0,1], no contrast adjustment",
              "wavelets": {}}

    for wv in WAVELETS:
        tc = coarse_only(x, wv)
        entry = {"delta_by_s": {}}
        for s in LEVELS:
            ts = wavelet_shrink(x, s, wv, bypass_identity=(s == 1.0))
            num = (ts - tc).flatten(1).norm(dim=1)
            d = torch.full_like(num, float("nan"))
            d[valid] = num[valid] / denom[valid]
            entry["delta_by_s"]["%g" % s] = {
                "per_image": [None if not v else round(float(val), 6)
                              for val, v in zip(d, valid)],
                "mean": float(d[valid].mean()) if valid.any() else None,
            }
        # coarse-only vs the original, for reference
        num0 = (tc - x).flatten(1).norm(dim=1)
        entry["coarse_vs_original_rel"] = float((num0[valid] / denom[valid]).mean())
        path = level_grid(sub.images, labels, bundle.class_names,
                          _CoarseControl(wv), [0.0],
                          out_dir / ("coarse_control_%s.png" % wv),
                          title="Coarse-only control T_coarse = P W*(a_2, 0)  [%s]" % wv)
        entry["figure"] = str(path)
        report["wavelets"][wv] = entry
        print("%-6s coarse-vs-original %.4f | delta_s: %s"
              % (wv, entry["coarse_vs_original_rel"],
                 "  ".join("s=%s %.4f" % (k, v["mean"])
                           for k, v in entry["delta_by_s"].items())))

    with open(out_dir / "wavelet_coarse_control.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nwrote %s" % (out_dir / "wavelet_coarse_control.json"))


if __name__ == "__main__":
    main()
