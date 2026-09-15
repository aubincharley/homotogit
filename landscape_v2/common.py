"""Frozen constants of the v2 study.  Everything here is fixed before any v2
result exists (see ``studies/landscape_v2/inputs/preregistration.json``)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from continuation_core import presets
from continuation_core.controller import InterventionController
from continuation_core.methods import get_method
from continuation_core.models import site_map
from continuation_core.seeding import derive_seed

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies" / "landscape_v2"
PINNED_ASSETS = ROOT / "assets" / "cifar10_resnet20bn"
V1_SUBSETS = ROOT / "studies" / "landscape_v1" / "inputs" / "subsets.npz"

METHODS = ("plain", "resolution_max_b1", "gaussian_postrelu", "resolution_max_b1_gaussian_conv")
SHORT = {"plain": "plain", "resolution_max_b1": "resolution", "gaussian_postrelu": "gaussian",
         "resolution_max_b1_gaussian_conv": "combined"}
SEEDS = (0, 1, 2, 3, 4)
UPDATES_PER_EPOCH = 391
EPOCHS = 30
TOTAL_UPDATES = UPDATES_PER_EPOCH * EPOCHS
TRANSITION_OFFSETS = (-50, -10, -1, 1, 10, 50)   # offset 0 = the epoch checkpoint

AMPLITUDES = (0.025, 0.05, 0.10, 0.20, 0.25)
N_DIRECTIONS = 20
FIXED_WEIGHT_DIRECTIONS = 10          # directions 0-9 for the fixed-weight 1-D measurements
PRIMARY_SEED = 0
PRIMARY_PAIR = (0, 1)                 # directions used as the (a, b) axes of surfaces
VALIDATION = {"directions": (0, 1), "amplitudes": (0.10, 0.25), "signs": (1, -1)}
CALIB_SENSITIVITY = {"seeds": (0,), "directions": (0, 1), "amplitudes": (0.10, 0.25), "signs": (1, -1)}
SURFACE_GRID_ALL = 21
SURFACE_GRID_PRIMARY = 41
PCA_GRID = 31

ACCOUNT_SEEDS = {"maxnicaise": (0,), "maxlefrr": (1, 4), "maxnikezz": (2,),
                 "maximemonstrenikez": (3,)}


def direction_seed(train_seed: int, k: int) -> int:
    """Direction-generation seed; a named stream distinct from every training seed."""
    return derive_seed(int(train_seed), "landscape_v2::direction::%d" % int(k))


def controller(method: str) -> InterventionController:
    return InterventionController(get_method(method), site_map("resnet20_bn_cifar"),
                                  "resnet20_bn_cifar")


def state_dict_of(state):
    return None if state is None else {"resolution": state.resolution, "sigma": state.sigma}


def transitions(method: str) -> list:
    """First update of every new intervention state, from the executed schedule."""
    c = controller(method)
    out = []
    for e in range(1, EPOCHS):
        a, b = c.state_for_epoch(e - 1), c.state_for_epoch(e)
        if (a.resolution, a.sigma) != (b.resolution, b.sigma):
            out.append({"update": e * UPDATES_PER_EPOCH, "epoch": e,
                        "before": state_dict_of(a), "after": state_dict_of(b)})
    return out


def fixed_weight_transitions(method: str) -> list:
    """Rule fixed in advance: the first and the last transition of the schedule."""
    t = transitions(method)
    return [] if not t else ([t[0], t[-1]] if len(t) > 1 else [t[0]])


def fixed_weight_states(method: str, tr: dict) -> list:
    """States compared at the checkpoint taken exactly at the transition update."""
    c = controller(method)
    final = state_dict_of(c.target_state())
    out = [("before", tr["before"]), ("after", tr["after"])]
    if method == "resolution_max_b1":
        for r in (16, 24, 32):
            s = {"resolution": r, "sigma": None}
            if s not in (tr["before"], tr["after"]):
                out.append(("r=%d" % r, s))
    elif tr["after"] != final:
        out.append(("final", final))
    return out


def config(method: str, seed: int, data_root: str, assets_dir: str, out_dir: str):
    cfg = presets.reference(method, seed, data_root=data_root, assets_dir=assets_dir,
                            out_dir=out_dir, device="auto")
    cfg.checkpoint.transition_offsets = TRANSITION_OFFSETS
    cfg.notes.append("landscape_v2: reference recipe unchanged; extra transition-window "
                     "checkpoints only; new runs, not the unified or resbench runs")
    return cfg


def sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
