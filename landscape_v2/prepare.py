"""Freeze every v2 analysis choice before any v2 result exists (local, CPU).

    py -m landscape_v2.prepare

Writes ``studies/landscape_v2/inputs/``: ``subsets.npz``, ``preregistration.json``,
``manifest.json`` (sha256 of every input file, assets included).  Refuses to
overwrite an existing preregistration.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np

from continuation_core.seeding import derive_seed

from . import common as C
from .assets_v2 import OUT as ASSETS
from landscape_study.sources import sha_file

OUT = C.STUDY / "inputs"


def build_subsets():
    v1 = dict(np.load(C.V1_SUBSETS))
    import pickle
    from pathlib import Path
    labels = []
    root = Path("C:/Users/mnica/Documents/Projet_filiere/data/cifar-10-batches-py")
    for i in range(1, 6):
        with open(root / ("data_batch_%d" % i), "rb") as fh:
            labels.append(np.asarray(pickle.load(fh, encoding="latin1")["labels"]))
    y = np.concatenate(labels)
    probe, calib = set(v1["train_probe_idx"].tolist()), set(v1["calibration_train_idx"].tolist())
    rng = np.random.default_rng(derive_seed(0, "landscape_v2::subsets"))
    calib_large, train_large = [], []
    for c in range(10):
        idx = np.flatnonzero(y == c)
        cal_c = sorted(i for i in idx if i in calib)
        pro_c = sorted(i for i in idx if i in probe)
        rest = rng.permutation([i for i in idx if i not in calib and i not in probe])
        calib_large += cal_c + list(rest[:1000 - len(cal_c)])
        train_large += pro_c + list(rest[1000 - len(cal_c):1000 - len(cal_c) + 1000 - len(pro_c)])
    calib_large, train_large = np.sort(calib_large).astype(np.int64), np.sort(train_large).astype(np.int64)
    assert len(calib_large) == 10000 and len(train_large) == 10000
    assert calib <= set(calib_large.tolist()) and probe <= set(train_large.tolist())
    assert not set(calib_large.tolist()) & set(train_large.tolist())
    np.savez(OUT / "subsets.npz", train_probe_idx=v1["train_probe_idx"],
             calibration_train_idx=v1["calibration_train_idx"], test_probe_idx=v1["test_probe_idx"],
             calibration_large_train_idx=calib_large, train_large_idx=train_large,
             pinned_train_probe_500=np.load(ASSETS / "shared_indices.npz")["train_probe"])


def main():
    if (OUT / "preregistration.json").exists():
        raise SystemExit("preregistration already written")
    OUT.mkdir(parents=True, exist_ok=True)
    build_subsets()
    prereg = {
        "study": "landscape_v2", "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "methods": list(C.METHODS), "training_seeds": list(C.SEEDS),
        "recipe": "continuation_core.presets.reference unchanged (SGD lr 0.005 momentum 0.9 wd 5e-4, "
                  "warm-up 60 then cosine, batch 128 in microbatches of 32, 30 epochs, no augmentation, "
                  "pinned/generated init and data order); only extra checkpoint writes",
        "assets": "studies/landscape_v2/inputs/assets (seeds 0-2 pinned copies, 3-4 generated)",
        "checkpoints": {"epochs": "0..30 (0 = initialization)",
                        "transition_offsets": list(C.TRANSITION_OFFSETS),
                        "transitions": {m: C.transitions(m) for m in C.METHODS}},
        "account_seeds": {k: list(v) for k, v in C.ACCOUNT_SEEDS.items()},
        "loss": "mean cross-entropy without weight decay; accuracy separate",
        "subsets": {"train_probe_idx": "v1, 1,000 train (100/class)",
                    "test_probe_idx": "v1, 1,000 test (100/class)",
                    "calibration_train_idx": "v1, 2,000 train (200/class), disjoint from train probe",
                    "calibration_large_train_idx": "10,000 train (1,000/class) superset of the 2k "
                                                   "calibration, disjoint from train probe",
                    "train_large_idx": "10,000 train (1,000/class) superset of the train probe, "
                                       "disjoint from both calibration sets",
                    "full": "train_full = 50,000 train, test_full = 10,000 test"},
        "bn_policies": {"saved": "checkpoint running statistics, reloaded before each evaluation",
                        "recalibrated": "reset; cumulative average over calib2k, fixed order, batches "
                                        "of 500, training mode, no_grad, under the evaluated state; "
                                        "inference-mode evaluation"},
        "directions": {"construction": "Li et al. filter-wise: conv output-channel blocks and fc rows; "
                                       "biases and BN affine fixed; zero-norm blocks -> zero; normalised "
                                       "once at the reference checkpoint; no orthogonalisation",
                       "count": C.N_DIRECTIONS,
                       "seed_rule": "derive_seed(training_seed, 'landscape_v2::direction::k'), torch "
                                    "CPU generator, float32 underlying draws shared by the four methods "
                                    "of a seed"},
        "primary_sensitivity": {"center": "epoch_030", "state": "final target (full resolution, no "
                                                                "filter)", "amplitudes": list(C.AMPLITUDES),
                                "signs": [1, -1], "policies": ["saved", "recalibrated"],
                                "splits": ["train_probe", "test_probe"],
                                "statistic": "S = (L(+e) + L(-e))/2 - L(0), averaged over 20 "
                                             "directions within each seed, then paired vs plain "
                                             "per seed"},
        "validation": dict(C.VALIDATION, splits=["train_large", "test_full"], policies=["saved", "recalibrated"],
                           centers=["train_full", "test_full"]),
        "calibration_sensitivity": dict(C.CALIB_SENSITIVITY, calibration="calib10k", splits=["train_probe", "test_probe"]),
        "surfaces": {"directions": list(C.PRIMARY_PAIR), "range": [-0.25, 0.25],
                     "grid_all_seeds": C.SURFACE_GRID_ALL, "grid_primary_seed": C.SURFACE_GRID_PRIMARY,
                     "primary_display": {"seed": C.PRIMARY_SEED, "pair": list(C.PRIMARY_PAIR)},
                     "policy": "recalibrated", "state": "final target"},
        "fixed_weight": {"rule": "first and last transition of each schedule; checkpoint taken at the "
                                 "transition update (weights after the last update of the old state)",
                         "selected": {m: [{"transition": t, "states": C.fixed_weight_states(m, t)}
                                          for t in C.fixed_weight_transitions(m)]
                                      for m in C.METHODS if m != "plain"},
                         "directions_1d": list(range(C.FIXED_WEIGHT_DIRECTIONS)),
                         "amplitudes": list(C.AMPLITUDES), "policy": "recalibrated per state "
                         "(saved-statistics centre losses also recorded)",
                         "surfaces": "primary seed, 21x21, directions 0/1"},
        "pca": {"vector": "all learned parameters incl. BN affine; running statistics excluded",
                "fit_points": "epoch checkpoints 0..30 of the four methods (matched, evenly spaced)",
                "projected_afterwards": "transition-window checkpoints",
                "components_saved": 10, "normalisation": "none (no filter-wise normalisation)",
                "background": "primary seed, 31x31 over projected range +20%, target state, recalibrated"},
        "interpolation": {"segments": [["plain", "resolution_max_b1"], ["plain", "gaussian_postrelu"],
                                       ["plain", "resolution_max_b1_gaussian_conv"],
                                       ["resolution_max_b1", "resolution_max_b1_gaussian_conv"]],
                          "points": 51, "state": "final target", "policy": "recalibrated",
                          "barrier": "max_alpha L - max(L(0), L(1)) on the grid"},
        "priorities": ["training + preservation", "primary 1-D both policies", "validation + "
                       "interpolation", "fixed-weight 1-D + trajectory diagnostics", "surfaces"],
        "not_done": ["optimised connecting paths", "Hessian analysis", "method tuning"],
    }
    (OUT / "preregistration.json").write_text(json.dumps(prereg, indent=2, default=str))
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({p.relative_to(OUT).as_posix(): sha_file(p) for p in files}, indent=2))
    print("preregistered", len(files), "files")


if __name__ == "__main__":
    main()
