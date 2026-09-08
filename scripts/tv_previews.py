r"""Generate TV-budget preview grids on CPU.  Previews only -- no training.

Reuses the *same ten images* (one per class) as the Gaussian
``transform_levels.png``, read from the recorded
``results/exp0_gaussian/figures/visualization.json`` so the selection is
identical rather than merely regenerated.

For each method (``tv_l2``, ``tv_hminus1``) one grid is written with columns
``t in {1, 0.9, 0.75, 0.5, 0.25, 0}``, a fixed display range of ``[0,1]`` and no
per-image contrast adjustment, plus a JSON of per-image diagnostics:
achieved TV ratio, retained contrast ``rho``, fidelity value, budget/box/mean
residuals and convergence flags.

    py scripts/tv_previews.py
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
from continuation.viz import level_grid  # noqa: E402

BUDGETS = [1.0, 0.9, 0.75, 0.5, 0.25, 0.0]
METHODS = {
    "tv_l2": ("transform_levels_tv_l2.png",
              "TV-$L^2$ budget: closest reconstruction in $L^2$ with TV(z) <= t TV(x)"),
    "tv_hminus1": ("transform_levels_tv_hminus1.png",
                   r"TV-$\dot H^{-1}$ budget: closest reconstruction in $\dot H^{-1}$"
                   " with TV(z) <= t TV(x)"),
}


def main(out_dir=None, viz_json=None):
    torch.set_num_threads(6)                       # CPU only; previews are small
    out_dir = Path(out_dir or ROOT / "results" / "tv_previews")
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_json = Path(viz_json or ROOT / "results" / "exp0_gaussian" / "figures"
                    / "visualization.json")

    bundle = build_dataset(DataConfig())           # CPU tensors
    with open(viz_json, "r", encoding="utf-8") as fh:
        viz = json.load(fh)
    idx = np.asarray(viz["example_indices_in_train_split"])
    sub = bundle.train.subset(idx, "tv_viz")
    x = InputPipeline.to_unit_float(sub.images).to(torch.float64)
    labels = sub.labels.numpy()
    print("reusing %d images from %s" % (len(idx), viz_json.name))
    print("classes:", [bundle.class_names[int(l)] for l in labels])

    tv_x = total_variation(x)
    report = {
        "images": {
            "indices_in_train_split": idx.tolist(),
            "original_dataset_indices": bundle.train.indices[idx].tolist(),
            "classes": [bundle.class_names[int(l)] for l in labels],
            "tv_original": [round(float(v), 6) for v in tv_x],
            "n_zero_tv_images": int((tv_x <= 0).sum()),
        },
        "budgets": BUDGETS,
        "display": "fixed range [0,1], no per-image contrast adjustment",
        "device": "cpu",
        "methods": {},
    }

    for family, (fname, title) in METHODS.items():
        T = build_transform(TransformConfig(family=family,
                                           params={"max_iters": 40000, "tol": 1e-7}))
        print("\n=== %s ===" % family)
        t0 = time.perf_counter()
        per_budget = []
        for t in BUDGETS:
            res = T.apply(x, t)
            z = res.images
            ratio, tv_valid = retained_tv_ratio(z, x)
            rho, rho_valid = retained_contrast(z, x)
            infos = res.info.get("per_image", [])
            entry = {
                "t": t,
                "achieved_tv_ratio": [round(float(v), 6) for v in ratio],
                "retained_contrast_rho": [round(float(v), 6) for v in rho],
                "n_undefined_tv_ratio": int((~tv_valid).sum()),
                "n_undefined_contrast": int((~rho_valid).sum()),
                "mean_achieved_tv_ratio": float(ratio[tv_valid].mean()) if tv_valid.any() else None,
                "mean_retained_contrast": float(rho[rho_valid].mean()) if rho_valid.any() else None,
                "solver_used": bool(res.info.get("solver_used", True)) if "solver_used" in res.info
                else bool(infos and infos[0].get("solver_used")),
                "all_converged": res.info.get("all_converged"),
                "max_tv_budget_rel_violation": res.info.get("max_tv_budget_rel_violation"),
                "max_box_violation": res.info.get("max_box_violation"),
                "max_mean_abs_drift": res.info.get("max_mean_abs_drift"),
                "iterations": [i.get("iterations") for i in infos],
                "fidelity_value": [round(float(i.get("fidelity_value", float("nan"))), 8)
                                   for i in infos] if infos else None,
                "solve_seconds": [i.get("solve_seconds") for i in infos],
            }
            per_budget.append(entry)
            print("  t=%-4g  mean TV ratio %-8s  mean rho %-8s  converged=%s  max budget viol %s"
                  % (t,
                     "%.4f" % entry["mean_achieved_tv_ratio"] if entry["mean_achieved_tv_ratio"] is not None else "n/a",
                     "%.4f" % entry["mean_retained_contrast"] if entry["mean_retained_contrast"] is not None else "n/a",
                     entry["all_converged"],
                     ("%.2e" % entry["max_tv_budget_rel_violation"])
                     if entry["max_tv_budget_rel_violation"] is not None else "n/a"))

        path = level_grid(sub.images, labels, bundle.class_names, T, BUDGETS,
                          out_dir / fname, title=title)
        print("  wrote %s" % path)
        report["methods"][family] = {
            "config": T.describe(),
            "figure": str(path),
            "wall_seconds": round(time.perf_counter() - t0, 2),
            "per_budget": per_budget,
        }

    with open(out_dir / "tv_preview_diagnostics.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nwrote %s" % (out_dir / "tv_preview_diagnostics.json"))
    return report


if __name__ == "__main__":
    main()
