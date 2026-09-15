"""Training with light analysis checkpoints (numerics identical to
``continuation_core.train.Trainer``; only what is written to disk differs).

Files per run ``<out>/<method>__seed<k>/``:

* ``checkpoints/epoch_000.pt``          initialization (update 0)
* ``checkpoints/epoch_NNN.pt``          after every completed epoch
* ``checkpoints/update_NNNNNN.pt``      transition windows (offsets -50,-10,-1,+1,+10,+50)
* ``rolling.pt``                        full resumable state (optimizer, LR position,
                                        RNG, data position), rewritten every epoch and
                                        kept at the end
* ``metrics.json``, ``summary.json``, ``config.json``, ``environment.json``

A light checkpoint holds ``model_state`` (all parameters **and** buffers,
including BatchNorm running statistics), ``global_update``, the state used by
the preceding update and the state scheduled for the next update, the LR of the
last update, the reason, and the resolved-config digest.  Checkpoints are never
deleted.  A checkpoint taken at a transition update ``t`` holds the weights after
update ``t - 1``: evaluating it before/after the change adds no training update.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from continuation_core import assets as A
from continuation_core.optim import lr_at
from continuation_core.train import Trainer

from .common import METHODS, sha_json

SCHEMA = "landscape_v2.analysis_checkpoint/1"


class LightTrainer(Trainer):
    def save(self, reason: str, name: str | None = None) -> Path:
        if reason == "rolling":
            return super().save(reason, name)
        u = self.global_update
        used = self._state_of_update(u - 1) if u > 0 else None
        nxt = self._state_of_update(u) if u < self.total_updates else None
        payload = {
            "schema": SCHEMA,
            "model_state": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
            "method": self.method.id, "seed": int(self.cfg.run.seed),
            "global_update": u, "updates_per_epoch": self.updates_per_epoch,
            "intervention": {"used_for_last_update": self._state_record(used),
                             "next_update": self._state_record(nxt)},
            "lr_last_update": (lr_at(u - 1, self.cfg.optimizer, self.total_updates) if u > 0 else None),
            "reason": reason, "config_sha256": sha_json(self.cfg.to_dict()),
            "provenance": self.provenance(),
        }
        path = self.out_dir / "checkpoints" / (name or "update_%06d.pt" % u)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return path

    def resume(self, path) -> None:
        """As ``Trainer.resume`` but loads on CPU: the core loader maps to the GPU,
        which moves the RNG byte tensors there and makes ``set_rng_state`` fail."""
        from continuation_core.checkpoint import load_checkpoint
        from continuation_core.seeding import set_rng_state
        ck = load_checkpoint(path, map_location="cpu")
        if ck["config"] != self.cfg.to_dict():
            raise ValueError("checkpoint config differs from this trainer's config")
        self.model.load_state_dict(ck["model_state"])
        self.optimizer.load_state_dict(ck["optimizer_state"])
        pos = ck["data_position"]
        self.epoch, self.batch_index = int(pos["epoch"]), int(pos["batch_index"])
        self.run_loss, self.run_n = float(pos["run_loss"]), int(pos["run_n"])
        self.global_update = int(ck["global_update"])
        self.metrics = list(ck["metrics"])
        self.eval_seconds = float(ck.get("eval_seconds", 0.0))
        self.wall_offset = float(ck.get("wall_seconds", 0.0))
        set_rng_state(ck["rng"])
        self.log("resumed at update %d (epoch %d, batch %d)" % (self.global_update, self.epoch, self.batch_index))

    def run(self, max_updates=None):
        init = self.out_dir / "checkpoints" / "epoch_000.pt"
        if self.global_update == 0 and not init.exists():
            self.save("initialization", name="epoch_000.pt")
        return super().run(max_updates=max_updates)


def pairing_record(seed: int, data_root: str, assets_dir: str, out_root: str) -> dict:
    """Digests showing that the four runs of one seed share every random input."""
    from .common import config
    a = A.load(assets_dir, seed, verify_first=True)
    cfgs = {m: config(m, seed, data_root, assets_dir, out_root).to_dict() for m in METHODS}
    base = {k: v for k, v in cfgs["plain"].items() if k not in ("method", "run")}
    same = all({k: v for k, v in c.items() if k not in ("method", "run")} == base for c in cfgs.values())
    return {"seed": seed, "init_state_sha256": A.sha_state(a["init_state"]),
            "data_order_sha256": A.sha_array(a["perms"]),
            "subset_sha256": A.sha_array(a["subset"]), "train_probe_sha256": A.sha_array(a["train_probe"]),
            "configs_identical_except_method_and_run_name": same,
            "config_sha256": {m: sha_json(c) for m, c in cfgs.items()},
            "stochastic_inputs": "data order and initial state only; no augmentation, no dropout; "
                                 "Gaussian and pooling operators are deterministic and draw no "
                                 "random numbers, so no random stream depends on the intervention",
            "gpu_nondeterminism": "cuDNN defaults (deterministic=False) as in the reference recipe; "
                                  "runs are not bitwise reproducible across processes"}


def _train_one(method, seed, data_root, assets_dir, out_root, device, log):
    from .common import config
    cfg = config(method, seed, data_root, assets_dir, out_root)
    out = Path(out_root) / ("%s__seed%d" % (method, seed))
    if (out / "summary.json").exists():
        log("skip %s seed %d (complete)" % (method, seed))
        return
    t0 = time.perf_counter()
    tr = LightTrainer(cfg, device=device, out_dir=out, log=log)
    if (out / "rolling.pt").exists():
        tr.resume(out / "rolling.pt")
    tr.run()
    log("DONE %s seed %d in %.0fs" % (method, seed, time.perf_counter() - t0))


def _worker(q, gpu, data_root, assets_dir, out_root, logfile):
    def log(msg):
        line = "[%s gpu%d] %s" % (time.strftime("%H:%M:%S"), gpu, msg)
        print(line, flush=True)
        with open(logfile, "a") as fh:
            fh.write(line + "\n")
    torch.cuda.set_device(gpu)
    while True:
        try:
            method, seed = q.get_nowait()
        except Exception:
            return
        try:
            _train_one(method, seed, data_root, assets_dir, out_root, "cuda:%d" % gpu, log)
        except Exception:
            log("FAILED %s seed %d\n%s" % (method, seed, traceback.format_exc()))


def train_all(seeds, data_root, assets_dir, out_root):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        (out_root / ("pairing_seed%d.json" % s)).write_text(
            json.dumps(pairing_record(s, data_root, assets_dir, str(out_root)), indent=2))
    order = [(m, s) for s in seeds for m in reversed(METHODS)]   # longest methods first
    n = max(torch.cuda.device_count(), 1)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    for item in order:
        q.put(item)
    procs = [ctx.Process(target=_worker, args=(q, g, data_root, assets_dir, str(out_root),
                                                str(out_root / "train_log.txt"))) for g in range(n)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
