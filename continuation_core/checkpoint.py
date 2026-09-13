"""Checkpoint format ``continuation_core.checkpoint/1``.

A checkpoint is written **after** a completed optimizer update and contains
everything needed to resume bit-for-bit on the same device type, and to
evaluate the same weights under the state that produced them and under the
state that comes next:

``model_state``        parameters and BatchNorm buffers (``state_dict``)
``optimizer_state``    optimizer ``state_dict`` (momentum buffers, AdamW moments)
``lr_schedule``        kind, total updates, the update index the schedule will
                       be evaluated at next, and that learning rate
``rng``                torch CPU / CUDA, numpy and python RNG states
``data_position``      epoch, next batch index within the epoch, updates per
                       epoch, seed, and the running training-loss accumulators
``global_update``      number of completed optimizer updates
``epochs_completed``   ``global_update / updates_per_epoch``
``config``             the full ``ExperimentConfig``
``intervention``       ``used_for_last_update`` (state + per-site sigma of the
                       update that produced these weights; ``None`` at update 0)
                       and ``next_update`` (state + per-site sigma the next
                       update will use), stored separately
``metrics``            evaluation records so far
``reason``             ``epoch_end`` | ``transition_window`` | ``every_updates``
                       | ``rolling``
``provenance``         asset manifest digests, torch version, code version
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

SCHEMA = "continuation_core.checkpoint/1"


def save_checkpoint(path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"schema": SCHEMA, **payload}, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if ckpt.get("schema") != SCHEMA:
        raise ValueError("%s is not a %s checkpoint (schema=%r)"
                         % (path, SCHEMA, ckpt.get("schema")))
    return ckpt


def list_checkpoints(run_dir) -> list:
    """All checkpoints of a run, sorted by global update then name."""
    out = []
    for p in sorted(Path(run_dir).glob("checkpoints/*.pt")):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        out.append((int(ck["global_update"]), p.name, p))
    return [p for _, _, p in sorted(out)]
