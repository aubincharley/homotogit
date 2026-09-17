"""Constants of the geometry-final completion (no training; landscape_v2 final checkpoints only).

Everything that fixes *what* is computed lives here and is frozen before the
timing pilot:

* the primary objective (centre-frozen BatchNorm, inference mode, mean CE without
  weight decay, the v2 mask of 268,336 conv + fc weights in 698 blocks, the
  1,000-image training and test probes), inherited unchanged from landscape_v3;
* the trace estimator, its Rademacher stream and its precision rule;
* the seed-0 centre-frozen random grid (41 x 41 over [-0.5, 0.5]^2, directions 0/1).

Existing landscape_v3 raw outputs are read from ``V3_ROOT`` (the ``visualization``
worktree by default; override with the environment variable ``GEOMETRY_V3_ROOT``).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from continuation_core.seeding import derive_seed
from landscape_v3 import common as V3

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies" / "geometry_final"
PROTO = STUDY / "protocol"
RAW = STUDY / "raw"
V3_ROOT = Path(os.environ.get("GEOMETRY_V3_ROOT", str(ROOT.parent / "visualization")))
V3_STUDY = V3_ROOT / "studies" / "landscape_v3"
V3_RAW = V3_STUDY / "raw"
V3_TABLES = V3_STUDY / "tables"

METHODS, SHORT, SEEDS, FINAL = V3.METHODS, V3.SHORT, V3.SEEDS, V3.FINAL
CFROZEN, POINTWISE, SAVED = V3.CFROZEN, V3.POINTWISE, V3.SAVED
PROBES = V3.PROBES

# The one account this task may use; the Kaggle input dataset it already owns
# (landscape-v3-inputs: all 20 final checkpoints, all direction draws, subsets.npz,
# expected_digests.json) is reused unchanged.
ACCOUNT = "maxmonstre"
INPUT_DATASET = "maxmonstre/landscape-v3-inputs"
KERNEL_SLUG = "geometry-final"

# ---- trace ------------------------------------------------------------------------
TRACE_STAGES = (64, 128, 256)          # cumulative draw counts; earlier draws are kept
TRACE_SEM_REL = 0.05                   # target: SEM <= 5% of |estimate|, every reported trace
TRACE_KINDS = ("ordinary", "relative", "covariance")
TRACE_STREAM = "geometry_final::trace::rademacher::draw::%d"

# ---- grid -------------------------------------------------------------------------
GRID_AXIS = V3.WIDE41                  # 41 values, [-0.5, 0.5], step 0.025
GRID_PAIR = (0, 1)
GRID_SEED = 0


def rademacher_seed(train_seed: int, draw: int) -> int:
    """Shared by the four methods and both probes of a training seed; a stream
    distinct from training, direction and Lanczos seeds."""
    return derive_seed(int(train_seed), TRACE_STREAM % int(draw))


def rademacher(train_seed: int, draw: int, n: int) -> np.ndarray:
    """Version-stable +-1 vector (int8): raw PCG64 output bits, little-endian.

    ``PCG64.random_raw`` is a fixed bit stream (unlike torch.randn across torch
    versions), so the draws regenerate bitwise on any machine."""
    raw = np.random.PCG64(rademacher_seed(train_seed, draw)).random_raw((n + 63) // 64).astype("<u8")
    bits = np.unpackbits(raw.view(np.uint8), bitorder="little")[:n]
    return (bits.astype(np.int8) * 2 - 1)


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def objective_id(m, s, probe) -> str:
    return "%s_s%d_%s_centre_frozen" % (SHORT[m], int(s), probe.replace("_probe", ""))


def trace_key(m, s, probe, draw) -> str:
    return "trace|%s|s%d|%s|centre_frozen|f32|%d" % (SHORT[m], int(s), probe, int(draw))


def grid_key(m, a, b) -> str:
    return V3.pkey(m, GRID_SEED, FINAL, V3.target(m), CFROZEN, "probes", V3.spec_r2(0, 1, a, b))
