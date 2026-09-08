"""Run logging: machine-readable records, environment capture, checkpoints.

Two kinds of quantity are logged and are never merged:

``train_minibatch``
    Losses/accuracies measured **on the training minibatch, in training mode**,
    averaged over the last logging window.  These are noisy and depend on the
    current weights during the update.

``eval``
    Losses/accuracies computed **in evaluation mode, without gradients, on fixed
    documented subsets** at shared checkpoints.

Every eval record names the objective it belongs to explicitly:
``transformed_*`` uses the run's own transformation level, ``target_*`` uses the
family's target endpoint (unfiltered images for the Gaussian family).
"""
from __future__ import annotations

import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def capture_environment() -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    try:
        import torchvision
        env["torchvision"] = torchvision.__version__
    except Exception:
        env["torchvision"] = None
    try:
        import numpy
        env["numpy"] = numpy.__version__
    except Exception:
        env["numpy"] = None
    if torch.cuda.is_available():
        try:
            env["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            env["gpu_total_memory_mb"] = round(props.total_memory / 1024 ** 2)
            env["gpu_capability"] = "%d.%d" % (props.major, props.minor)
        except Exception:
            pass
    return env


class RunLogger:
    """Writes ``config.json``, ``env.json``, ``metrics.jsonl`` and ``summary.json``."""

    def __init__(self, out_dir, quiet: bool = False):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.dir / "metrics.jsonl"
        self._fh = open(self.metrics_path, "a", encoding="utf-8")
        self.quiet = quiet
        self.t0 = time.perf_counter()

    def write_json(self, name: str, payload) -> Path:
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return path

    def log(self, record: dict) -> None:
        record = dict(record)
        record.setdefault("wall_seconds", round(time.perf_counter() - self.t0, 4))
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    def console(self, message: str) -> None:
        if not self.quiet:
            print(message, flush=True)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def read_metrics(path):
    """Read a ``metrics.jsonl`` file into a list of records."""
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_checkpoint(path, model, optimizer, step: int, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": int(step),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "extra": extra or {},
    }, path)


def save_full_checkpoint(path, model, optimizer, step: int, batch_stream=None,
                         lineage: dict | None = None, config: dict | None = None,
                         extra: dict | None = None) -> dict:
    """A **complete, resumable** training state, suitable for branching.

    Beyond parameters and optimizer state this stores the global update counter,
    the data-order (sampler) position, and the RNG states, so a branch continues
    the *identical* minibatch sequence an uninterrupted run would have seen.

    The learning-rate schedule needs no state of its own: it is a pure function
    of the global update counter and ``optim.total_steps`` (see
    :func:`continuation.optim.lr_at`), which is precisely why it cannot be
    accidentally restarted at a branch.

    No AMP/GradScaler state is stored because this codebase trains in full
    float32 and never constructs a scaler; ``amp`` is recorded as ``None`` so the
    absence is explicit rather than implied.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "full_state_v1",
        "step": int(step),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "batch_stream": None if batch_stream is None else batch_stream.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": (torch.cuda.get_rng_state_all()
                                 if torch.cuda.is_available() else None),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "amp": None,
        "lineage": lineage or {},
        "config": config or {},
        "extra": extra or {},
    }
    torch.save(payload, path)
    return {"path": str(path), "step": int(step), "format": "full_state_v1"}


def checkpoint_kind(path) -> str:
    """Classify a checkpoint file: complete resumable state vs. model-only."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") == "full_state_v1" and ck.get("batch_stream") is not None:
        return "full_resumable"
    if "optimizer_state" in ck and "model_state" in ck:
        return "model_and_optimizer_only"
    return "model_only"
