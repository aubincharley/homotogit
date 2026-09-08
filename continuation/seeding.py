"""Seed derivation and independent random streams.

A single shared seed is *not* sufficient for paired comparisons: two conditions
that consume random numbers differently would then diverge in their minibatch
order.  We therefore derive one named, independent stream per concern and never
let a transformation or an evaluation draw from the training streams.

Streams used in this codebase:
  ``init``   -- model parameter initialization           (from ``run.seed``)
  ``batch``  -- the minibatch index sequence             (from ``run.seed``)
  ``split``  -- the train/val split                      (from ``data.split_seed``)
  ``probe``  -- fixed evaluation/probe subsets           (from ``eval.probe_seed``)
"""
from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch


def derive_seed(master_seed: int, stream: str) -> int:
    """Deterministic, platform-independent sub-seed for a named stream."""
    payload = f"{int(master_seed)}::{stream}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def numpy_generator(master_seed: int, stream: str) -> np.random.Generator:
    return np.random.default_rng(derive_seed(master_seed, stream))


def torch_generator(master_seed: int, stream: str, device: str = "cpu") -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(derive_seed(master_seed, stream))
    return g


def seed_global(master_seed: int, stream: str = "global") -> int:
    """Seed the global RNGs (used only for library calls we do not control)."""
    seed = derive_seed(master_seed, stream)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def configure_determinism(deterministic: bool = True) -> dict[str, object]:
    """Best-effort deterministic cuDNN/cuBLAS configuration.

    Returns the settings actually applied so they can be logged.  Full bitwise
    determinism across machines is not claimed.
    """
    info: dict[str, object] = {"requested": deterministic}
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            info["use_deterministic_algorithms"] = "True(warn_only)"
        except Exception as exc:  # pragma: no cover - torch build dependent
            info["use_deterministic_algorithms"] = f"unavailable: {exc}"
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    info["cudnn.benchmark"] = torch.backends.cudnn.benchmark
    info["cudnn.deterministic"] = torch.backends.cudnn.deterministic
    return info
