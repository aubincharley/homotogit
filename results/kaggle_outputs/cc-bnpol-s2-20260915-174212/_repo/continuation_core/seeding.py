"""Named, platform-independent seed derivation (same rule as the benchmark)."""
from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def derive_seed(master_seed: int, stream: str) -> int:
    payload = ("%d::%s" % (int(master_seed), stream)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (2 ** 31 - 1)


def rng_state() -> dict:
    """Every RNG a run might touch, for checkpoints."""
    return {"torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(), "python": random.getstate()}


def set_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
