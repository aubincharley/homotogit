"""Fix every choice before any landscape is evaluated (local, CPU).

    py -m landscape_study.prepare --data-root ../../Projet_filiere/data

Writes ``studies/landscape_v1/inputs/``:

``preregistration.json``  seeds, checkpoints, draws, grids, BN convention, and
                          the rule for changing the coordinate range; written
                          before any evaluation
``sources_report.json``   provenance checks of the resbench runs
``subsets.npz``           class-balanced probes and calibration indices, plus
                          the pinned 500-image training probe
``draws.pt``              underlying random tensors V for direction pairs 0-4
``pca_plane.pt``          shared PCA plane of the seed-0 trajectories
``weights/``              stripped checkpoints (git-ignored; hashed in the
                          manifest and published as a private Kaggle dataset)
``manifest.json``         sha256 of every file above
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch

from continuation_core.analysis.params import to_vector
from continuation_core.analysis.landscape import pca_plane
from continuation_core.config import OptimizerConfig
from continuation_core.data import load_dataset
from continuation_core.config import DataConfig
from continuation_core.models import build_model
from continuation_core.optim import lr_at

from . import sources
from .directions import draw_underlying

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "studies" / "landscape_v1" / "inputs"
CORE_ASSETS = ROOT / "assets" / "cifar10_resnet20bn"

SUBSET_SEED = 20260914
N_PROBE_PER_CLASS = 100
N_CALIB_PER_CLASS = 200
DRAW_SEEDS = {k: (1000 + 2 * k, 1001 + 2 * k) for k in range(5)}
PCA_EPOCHS = (0, 6, 12, 18, 30)

PREREGISTRATION = {
    "study": "landscape_v1",
    "scope": {
        "methods_with_checkpoints": ["plain", "resolution_max_b1"],
        "source_batch": "resbench (checkpoint_ep06/12/18/30); different runs of the same "
                        "configurations as the unified reference batch",
        "missing": sources.MISSING,
    },
    "loss": "mean cross-entropy without weight decay; accuracy also recorded",
    "bn_convention": {
        "main": "recalibrated: reset running stats, cumulative average (momentum None) over "
                "the fixed calibration subset in fixed order, batches of 500, training mode "
                "under no_grad and under the evaluated intervention state; then inference "
                "mode on the probes",
        "reproduction": "saved_stats: checkpoint running statistics, inference mode",
    },
    "subsets": {"seed": SUBSET_SEED, "train_probe_per_class": N_PROBE_PER_CLASS,
                "test_probe_per_class": N_PROBE_PER_CLASS,
                "calibration_per_class": N_CALIB_PER_CLASS,
                "calibration_disjoint_from_train_probe": True,
                "calibration_split": "official training split only"},
    "directions": {
        "construction": "Li et al. 2018 filter-wise normalisation; mask = conv weights "
                        "(4-D) and fc.weight rows; biases and BN affine fixed; zero-norm "
                        "blocks get zero; no orthogonalisation",
        "draw_seeds": {str(k): v for k, v in DRAW_SEEDS.items()},
        "prespecified_2d_pair": 0,
        "robustness_pairs": [0, 1, 2, 3, 4],
    },
    "analysis_A": {"centers": "epoch-30 checkpoints, seed 0 (prespecified)",
                   "state": "target (full resolution, no filter)",
                   "grid": {"lo": -0.25, "hi": 0.25, "n": 21},
                   "robustness": "1-D slices along D of pairs 0-4 and E of pair 0, seeds 0/1/2, "
                                 "same range, 21 points",
                   "range_rule": "keep [-0.25, 0.25] unless the local offset check at "
                                 "+-{0.05,0.1,0.25,0.5} along D (seed 0) gives a non-finite "
                                 "loss or a test-probe CE increase above 5 nats at +-0.25, or "
                                 "below 0.01 nats at +-0.25, for either method; any change "
                                 "applies to all methods and is documented"},
    "analysis_B": {"family": "resolution_max_b1 (only family with checkpoints)",
                   "checkpoint": "seed 0, epoch 6 = last update at r=16 (epochs 0-5); chosen "
                                 "from the schedule as the first saved checkpoint produced "
                                 "entirely under an active reduction",
                   "states": [16, 24, 32], "grid": "same as A, pair 0 scaled at this checkpoint",
                   "secondary": "1-D along D: seed 0 epoch 12 (last update at r=24) under "
                                "16/24/32; control: plain seed 0 epoch 6 with the reduction "
                                "hook under 16/24/32",
                   "gaussian_and_combined": "not evaluable: no checkpoints"},
    "analysis_C": {"runs": "plain and resolution_max_b1, seed 0",
                   "points": "initialisation (update 0) and epochs 6/12/18/30 "
                             "(updates 0/2346/4692/7038/11730), identical steps for both runs",
                   "vector": "all learned parameters incl. BN affine; running statistics "
                             "excluded (the random-direction mask additionally excludes BN "
                             "affine and biases)",
                   "basis": "top-2 principal directions of the 10 pooled points centred at "
                            "their mean; not filter-normalised",
                   "contour": "target objective, recalibrated BN, 25x25 grid over the "
                              "projected range with 20% margin"},
    "analysis_D": {"pairs": "plain vs resolution_max_b1, epoch 30, seeds 0/1/2",
                   "alphas": 51, "interpolated": "all learned parameters incl. BN affine; "
                                                 "running statistics recalibrated",
                   "state": "target",
                   "barrier": "max_alpha L - max(L(0), L(1)) on the 51-point grid"},
    "not_done": ["optimised connecting curves", "Hessian analysis", "training of any kind"],
}


def balanced(labels: np.ndarray, rng, counts) -> list:
    out = [[] for _ in counts]
    for c in range(10):
        idx = rng.permutation(np.flatnonzero(labels == c))
        start = 0
        for k, n in enumerate(counts):
            out[k].append(idx[start:start + n])
            start += n
    return [np.sort(np.concatenate(o)).astype(np.int64) for o in out]


def strip(method, seed, epoch, init_states) -> dict:
    states = sources.states_at(method, epoch)
    if epoch == 0:
        return {"model_state": init_states[seed], "recorded_metrics": None,
                "source": {"file": "assets/cifar10_resnet20bn/init_seed%d.pt" % seed,
                           "sha256": sources.sha_file(CORE_ASSETS / ("init_seed%d.pt" % seed))},
                "method": method, "seed": seed, "epochs_completed": 0, "global_update": 0,
                "states": states}
    src = sources.load_resbench_checkpoint(method, seed, epoch)
    ck = src["ckpt"]
    rec = [m for m in ck["metrics"] if m["epoch"] == epoch]
    return {"model_state": ck["model_state"], "recorded_metrics": rec[-1] if rec else None,
            "source": {"file": str(src["path"]), "sha256": sources.sha_file(src["path"])},
            "method": method, "seed": seed, "epochs_completed": epoch,
            "global_update": int(ck["update"]), "states": states}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()
    if (OUT / "preregistration.json").exists():
        raise SystemExit("%s already exists; inputs are fixed once" % (OUT / "preregistration.json"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "weights").mkdir(exist_ok=True)

    report = sources.verify_sources(CORE_ASSETS)
    (OUT / "sources_report.json").write_text(json.dumps(report, indent=2))
    assert all(j["all_match"] and j["same_files_as_core_assets"] for j in report["jobs"].values())
    assert all(r["recorded_states_match_core_schedule"] for r in report["runs"].values())

    ds = load_dataset(DataConfig(root=args.data_root))
    z = np.load(CORE_ASSETS / "shared_indices.npz")
    assert (z["subset"] == np.arange(50000)).all(), "subset must be the identity order"
    rng = np.random.default_rng(SUBSET_SEED)
    tr_probe, calib = balanced(ds.train.labels.numpy(), rng, (N_PROBE_PER_CLASS, N_CALIB_PER_CLASS))
    (te_probe,) = balanced(ds.test.labels.numpy(), rng, (N_PROBE_PER_CLASS,))
    assert not set(tr_probe) & set(calib)
    np.savez(OUT / "subsets.npz", train_probe_idx=tr_probe, calibration_train_idx=calib,
             test_probe_idx=te_probe, pinned_train_probe_500=z["train_probe"])

    template = build_model("resnet20_bn_cifar", 10)
    names = [n for n, _ in template.named_parameters()]
    shapes = {n: p for n, p in template.named_parameters()}
    draws = {k: {"D": draw_underlying(shapes, names, sd), "E": draw_underlying(shapes, names, se),
                 "seeds": (sd, se)} for k, (sd, se) in DRAW_SEEDS.items()}
    torch.save(draws, OUT / "draws.pt")

    init_states = {s: torch.load(CORE_ASSETS / ("init_seed%d.pt" % s), map_location="cpu",
                                 weights_only=True) for s in sources.SEEDS}
    weights = {}
    for method in sources.AVAILABLE:
        for seed in sources.SEEDS:
            for ep in (0,) + sources.CKPT_EPOCHS:
                w = strip(method, seed, ep, init_states)
                name = "%s__seed%d__ep%02d.pt" % (method, seed, ep)
                torch.save({"schema": "landscape_study.weights/1", **w}, OUT / "weights" / name)
                weights[name] = w

    ocfg = OptimizerConfig(name="sgd", lr=0.005, weight_decay=5e-4, warmup_updates=60)
    rows, meta = [], []
    for method in sources.AVAILABLE:
        for ep in PCA_EPOCHS:
            w = weights["%s__seed0__ep%02d.pt" % (method, ep)]
            rows.append(to_vector(w["model_state"], names).numpy())
            u = w["global_update"]
            meta.append({"method": method, "seed": 0, "epochs_completed": ep, "global_update": u,
                         "lr_last_update": None if u == 0 else lr_at(u - 1, ocfg, sources.TOTAL_UPDATES),
                         "states": w["states"],
                         "file": "%s__seed0__ep%02d.pt" % (method, ep)})
    plane = pca_plane(np.stack(rows), center="mean")
    co = np.asarray(plane["coords"])
    lo, hi = co.min(0), co.max(0)
    span = hi - lo
    grid = {"x": np.linspace(lo[0] - 0.2 * span[0], hi[0] + 0.2 * span[0], 25).tolist(),
            "y": np.linspace(lo[1] - 0.2 * span[1], hi[1] + 0.2 * span[1], 25).tolist()}
    torch.save({"origin": plane["origin"], "d1": plane["d1"], "d2": plane["d2"], "names": names,
                "points": meta, "explained_variance_ratio": plane["explained_variance_ratio"],
                "plane_variance_ratio": plane["plane_variance_ratio"], "coords": plane["coords"],
                "residual_norm": plane["residual_norm"],
                "relative_residual": plane["relative_residual"], "grid": grid},
               OUT / "pca_plane.pt")

    prereg = dict(PREREGISTRATION, written_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  weights=sorted(weights), pca_grid_shape=[25, 25])
    (OUT / "preregistration.json").write_text(json.dumps(prereg, indent=2))
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps(
        {p.relative_to(OUT).as_posix(): sources.sha_file(p) for p in files}, indent=2))
    print("prepared %d files in %s" % (len(files), OUT))
    print("PCA explained variance (first 4):",
          [round(v, 4) for v in plane["explained_variance_ratio"][:4]],
          "plane:", round(plane["plane_variance_ratio"], 4))


if __name__ == "__main__":
    main()
