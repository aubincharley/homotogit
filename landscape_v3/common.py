"""Constants and point identities of the v3 extension (no training; v2 checkpoints only).

Every evaluated quantity has one canonical key (``pkey``).  The planner, the
Kaggle job and the analysis all build keys with this function, which is how a
v2 measurement is recognised as reusable: same method, seed, checkpoint,
intervention state, BatchNorm policy, precision, evaluation set and
perturbation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from continuation_core.seeding import derive_seed
from landscape_v2 import common as V2

ROOT = V2.ROOT
STUDY = ROOT / "studies" / "landscape_v3"
V2_STUDY = V2.STUDY
V2_RAW = V2_STUDY / "raw"
V2_ACCOUNT_OF_SEED = {0: "maxnicaise", 1: "maxlefrr", 2: "maxnikezz", 3: "maximemonstrenikez", 4: "maxlefrr"}

METHODS, SHORT, SEEDS = V2.METHODS, V2.SHORT, V2.SEEDS
LONG = {v: k for k, v in SHORT.items()}
UPE = V2.UPDATES_PER_EPOCH
FINAL = "epoch_030.pt"
PRIMARY_SEED = 0

# BatchNorm policies.  "saved" and "recalibrated" keep their v2 names (so v2 points match).
SAVED, POINTWISE, CFROZEN = "saved", "recalibrated", "centre_frozen"
POLICIES = (SAVED, POINTWISE, CFROZEN)
POLICY_LABEL = {SAVED: "saved/frozen BN", POINTWISE: "recalibrated at every point",
                CFROZEN: "recalibrated at centre, then frozen"}

# 1. final-solution amplitudes
AMPLITUDES = (0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.35, 0.5)
N_DIRECTIONS = 20
F64_CHECK = {"directions": (0, 1), "amplitudes": (0.005, 0.01, 0.02, 0.05)}
# 2. larger-set validation
VALIDATION_AMPS = (0.1, 0.25)
# 3. training-time
TEMPORAL_EPOCHS = (3, 6, 9, 12, 18, 21, 30)
TEMPORAL_DIRS = tuple(range(10))
TEMPORAL_AMPS = (0.1, 0.25)
TRANSITION_DIRS = tuple(range(10))
TRANSITION_AMPS_RECAL = V2.AMPLITUDES          # v2 fixed-weight protocol
TRANSITION_AMPS_SAVED = (0.1, 0.25)            # v3 addition
# 4. Hessian
HESS_PROBES = ("train_probe", "test_probe")
HESS_POLICIES = (SAVED, CFROZEN)
HESS_COORDS = ("ordinary", "relative")
HESS_STARTS = (0, 1)
# Eigendirections scaled to r = 1 concentrate their displacement in few blocks (local pilot: the
# top direction raises CE by >100 nats at t = 0.01), so cuts use a symmetric log-spaced grid and
# finite differences use small steps.
_CUT_POS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5)
CUT_T = tuple(sorted([-t for t in _CUT_POS] + [0.0] + list(_CUT_POS)))
FD_T = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
# Hessian planes: unit grid u in [-1, 1]; physical t_i = u * T_i with T_i = sqrt(2 * PLANE_RISE /
# (G * |lambda_i|)) taken from PLAIN (same seed, policy, probe, coordinates), shared by all methods.
HESS_PLANE_U = tuple(np.round(np.linspace(-1.0, 1.0, 41), 10).tolist())
PLANE_RISE = 2.0
# 5. random planes
WIDE21 = tuple(np.round(np.linspace(-0.5, 0.5, 21), 10).tolist())
WIDE41 = tuple(np.round(np.linspace(-0.5, 0.5, 41), 10).tolist())

PROBES = ("train_probe", "test_probe")
LARGE = ("train_large", "test_full")
SPLITSETS = {"probes": PROBES, "large": LARGE}

ACCOUNTS = ("maxnicaise", "maxlefrr", "maxnikezz", "maximemonstrenikez", "maxlebossdu91", "maxfrrsava", "maxmonstre")


def fnum(x) -> str:
    s = "%.6g" % round(float(x), 10)
    return "0" if s in ("-0", "0") else s


def stag(state) -> str:
    if state is None:
        return "r-_G-"
    r, g = state.get("resolution"), state.get("sigma")
    return "r%s_G%s" % ("-" if r is None else int(r), "-" if g is None else fnum(g))


def pkey(m, s, ckpt, state, policy, splitset, spec, prec="f32") -> str:
    return "|".join([SHORT.get(m, m), "s%d" % int(s), str(ckpt).replace(".pt", ""), stag(state),
                     policy, prec, splitset, spec])


def spec_r1(k, signed_amp):
    return "r1:%d:%s" % (int(k), fnum(signed_amp))


def spec_r2(k0, k1, a, b):
    return "r2:%d:%d:%s:%s" % (int(k0), int(k1), fnum(a), fnum(b))


def spec_h1(problem, vec, t):
    return "h1:%s:%s:%s" % (problem, vec, fnum(t))


def spec_h2(problem, v1, v2, a, b):
    return "h2:%s:%s:%s:%s:%s" % (problem, v1, v2, fnum(a), fnum(b))


def target(m):
    return V2.state_dict_of(V2.controller(m).target_state())


def epoch_file(e):
    return "epoch_%03d.pt" % int(e)


def direction_seed(s, k):
    return V2.direction_seed(s, k)


def lanczos_seed(problem: str, start: int) -> int:
    h = int(hashlib.sha256(problem.encode()).hexdigest()[:8], 16)
    return derive_seed(h % 100000, "landscape_v3::lanczos::start::%d" % int(start))


def hess_problem(m, s, probe, policy, coords) -> str:
    return "%s_s%d_%s_%s_%s" % (SHORT[m], s, probe.replace("_probe", ""), policy, coords[:3])


def sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rname(m, s):
    return "%s__seed%d" % (m, s)


def v2_run_dir(m, s) -> Path:
    return V2_RAW / V2_ACCOUNT_OF_SEED[s] / "v2" / "runs" / rname(m, s)
