r"""Wavelet-shrinkage preview grids (CPU).  No training is launched.

Reuses the exact ten images (one per class) of the Gaussian and TV previews,
taken before network normalization, and writes one grid per wavelet with
``s in {1, 0.9, 0.75, 0.5, 0.25, 0}``.

Display uses fixed limits ``[0,1]`` with no contrast adjustment.  Raw outputs are
preserved: clamping happens only inside the figure, and every value that falls
outside ``[0,1]`` is reported in the diagnostics JSON.

    py scripts/wavelet_previews.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from continuation.config import DataConfig, TransformConfig  # noqa: E402
from continuation.data import build_dataset  # noqa: E402
from continuation.diagnostics import (  # noqa: E402
    retained_contrast,
    retained_tv_ratio,
    total_variation,
)
from continuation.pipeline import InputPipeline  # noqa: E402
from continuation.transforms import build_transform  # noqa: E402
from continuation.transforms.wavelet import WAVELETS  # noqa: E402
from continuation.viz import level_grid  # noqa: E402

LEVELS = [1.0, 0.9, 0.75, 0.5, 0.25, 0.0]


def main(out_dir=None):
    torch.set_num_threads(6)
    out_dir = Path(out_dir or ROOT / "results" / "wavelet_previews")
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_json = ROOT / "results" / "exp0_gaussian" / "figures" / "visualization.json"

    bundle = build_dataset(DataConfig())
    idx = np.asarray(json.load(open(viz_json, encoding="utf-8"))
                     ["example_indices_in_train_split"])
    sub = bundle.train.subset(idx, "wavelet_viz")
    x = InputPipeline.to_unit_float(sub.images).to(torch.float64)
    labels = sub.labels.numpy()
    classes = [bundle.class_names[int(l)] for l in labels]
    print("reusing %d images: %s" % (len(idx), classes))

    report = {
        "images": {"indices_in_train_split": idx.tolist(),
                   "original_dataset_indices": bundle.train.indices[idx].tolist(),
                   "classes": classes,
                   "tv_original": [round(float(v), 6) for v in total_variation(x)]},
        "s_values": LEVELS,
        "display": "fixed limits [0,1], no contrast adjustment; clamp is display-only",
        "contrast_definition": "rho = ||z - mean(z)|| / ||x - mean(x)||, spatial per channel",
        "device": "cpu",
        "wavelets": {},
    }

    for wv in WAVELETS:
        T = build_transform(TransformConfig(family="wavelet",
                                            params={"wavelet": wv, "levels": 2}))
        print("\n=== %s ===" % wv)
        t0 = time.perf_counter()
        per_s = []
        for s in LEVELS:
            res = T.apply(x, s)
            z = res.images
            tv_ratio, tv_valid = retained_tv_ratio(z, x)
            rho, rho_valid = retained_contrast(z, x, center="own")
            mean_drift = (z.mean(dim=(2, 3)) - x.mean(dim=(2, 3))).abs()
            zf = {k: [round(float(v), 6) for v in val.flatten()]
                  for k, val in res.info["zero_fraction"].items()}
            entry = {
                "s": s,
                "tv_ratio": [round(float(v), 6) for v in tv_ratio],
                "mean_tv_ratio": float(tv_ratio[tv_valid].mean()) if tv_valid.any() else None,
                "n_undefined_tv_ratio": int((~tv_valid).sum()),
                "retained_contrast_rho": [round(float(v), 6) for v in rho],
                "mean_retained_contrast": float(rho[rho_valid].mean()) if rho_valid.any() else None,
                "n_undefined_contrast": int((~rho_valid).sum()),
                "channel_mean_drift_max": float(mean_drift.max()),
                "channel_mean_drift_mean": float(mean_drift.mean()),
                "zero_fraction_by_band": {k: float(np.mean(v)) for k, v in zf.items()},
                "zero_fraction_by_band_per_image": zf,
                "out_of_unit_range": res.info["out_of_unit_range"],
            }
            per_s.append(entry)
            print("  s=%-4g  TV ratio %.4f  rho %.4f  mean drift %.2e  "
                  "zeroed %.3f  range [%.3f, %.3f]"
                  % (s, entry["mean_tv_ratio"], entry["mean_retained_contrast"],
                     entry["channel_mean_drift_max"],
                     float(np.mean(list(entry["zero_fraction_by_band"].values()))),
                     entry["out_of_unit_range"]["min"], entry["out_of_unit_range"]["max"]))

        path = level_grid(sub.images, labels, bundle.class_names, T, LEVELS,
                          out_dir / ("transform_levels_wavelet_%s.png" % wv),
                          title="Wavelet shrinkage (%s, 2-level undecimated): "
                                "T_s with lambda = 4(1-s) 2^(1-j) RMS" % wv)
        print("  wrote %s" % path)
        report["wavelets"][wv] = {"config": T.describe(), "figure": str(path),
                                  "wall_seconds": round(time.perf_counter() - t0, 2),
                                  "per_s": per_s}

    with open(out_dir / "wavelet_preview_diagnostics.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nwrote %s" % (out_dir / "wavelet_preview_diagnostics.json"))
    return report


if __name__ == "__main__":
    main()
