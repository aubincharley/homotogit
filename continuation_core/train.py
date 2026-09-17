"""The one training loop shared by every method.

Reproduces ``scripts/unified_driver.py`` (benchmark branch) update for update:

* epoch ``e`` uses ``controller.state_for_epoch(e)`` for all of its updates;
* the example order of epoch ``e`` is ``perm_seed<k>[e]`` over
  ``train.images[subset]``; batches are consecutive slices of 128, the last
  one 80;
* each update: set the learning rate for this global update, zero gradients,
  then for each microbatch of 32 (last one possibly shorter) compute the mean
  cross-entropy, scale it by ``n_micro / n_batch`` and backpropagate; one
  optimizer step;
* after each epoch (and once before training), evaluate the train probe and
  the test set on the **current** path (the state the epoch just used) and the
  **target** path (full resolution, Gaussian off), BN in ``running_stats``.

Additions over the benchmark driver: mid-epoch resumption, full RNG capture,
checkpoints around intervention transitions, and explicit state records.
"""
from __future__ import annotations

import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import assets as assets_mod
from .augment import EpochAugmenter
from .checkpoint import load_checkpoint, save_checkpoint
from .config import ExperimentConfig
from .controller import InterventionController, InterventionState
from .data import build_pipeline, load_dataset
from .evaluate import evaluate
from .models import build_model, is_validated, site_map
from .optim import build_optimizer, lr_at, optimizer_tag, set_lr
from .seeding import rng_state, set_rng_state

RESULTS_SCHEMA = "continuation_core.results/1"


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class Trainer:
    def __init__(self, config: ExperimentConfig, *, dataset=None, loaded_assets=None,
                 device=None, out_dir=None, log=print):
        self.cfg = config
        self.log = log
        self.device = resolve_device(device or config.run.device)
        torch.backends.cudnn.deterministic = bool(config.run.cudnn_deterministic)
        torch.backends.cudnn.benchmark = bool(config.run.cudnn_benchmark)
        seed = int(config.run.seed)
        name = config.run.name or "%s__seed%d" % (config.method_spec().id, seed)
        self.out_dir = Path(out_dir or Path(config.run.out_dir) / name)

        self.dataset = dataset or load_dataset(config.data)
        self.pipeline = build_pipeline(self.dataset, config.data).to(self.device)
        if loaded_assets is None:
            if not config.assets.dir:
                raise ValueError("no asset directory: pinned initial state and data "
                                 "order are required (see `make-assets`)")
            loaded_assets = assets_mod.load(config.assets.dir, seed,
                                            verify_first=config.assets.verify)
        self.assets = loaded_assets

        dev = self.device
        subset = torch.as_tensor(self.assets["subset"], dtype=torch.long, device=dev)
        train = self.dataset.train.to(dev)
        self.train_images = train.images[subset]
        self.train_labels = train.labels[subset]
        probe = torch.as_tensor(self.assets["train_probe"], device=dev)
        self.probe_images, self.probe_labels = self.train_images[probe], self.train_labels[probe]
        test = self.dataset.test.to(dev)
        self.test_images, self.test_labels = test.images, test.labels
        self.perms = self.assets["perms"]

        arch = config.model.arch
        self.model = build_model(arch, self.dataset.num_classes,
                                 int(self.train_images.shape[1]), **config.model.options)
        self.model.load_state_dict(self.assets["init_state"], strict=True)
        self.model.to(dev).train()
        self.method = config.method_spec()
        self.controller = InterventionController(self.method, site_map(arch), arch)
        self.handles = self.controller.attach(self.model)

        b = config.budget
        self.n_train = int(self.train_images.shape[0])
        self.batch, self.micro = int(b.effective_batch), int(b.microbatch)
        self.updates_per_epoch = (self.n_train + self.batch - 1) // self.batch
        self.total_updates = int(b.epochs) * self.updates_per_epoch
        self.controller.configure(self.updates_per_epoch, self.total_updates, seed)
        if self.perms.shape[0] < b.epochs or self.perms.shape[1] != self.n_train:
            raise ValueError("data order %s does not cover %d epochs of %d examples"
                             % (self.perms.shape, b.epochs, self.n_train))
        self.optimizer = build_optimizer(self.model.parameters(), config.optimizer)
        self.augmenter = EpochAugmenter(config.data.augmentation, seed, self.n_train, dev,
                                        config.data.augmentation_stream)
        #: called as ``fn(trainer, done_epochs)`` after each epoch's snapshot; its time is
        #: recorded in ``diagnostic_seconds`` and excluded from ``train_seconds``
        self.epoch_callbacks: list = []
        self.diagnostic_seconds = 0.0

        self.global_update = 0
        self.epoch = 0
        self.batch_index = 0
        self.run_loss, self.run_n = 0.0, 0
        self.metrics: list = []
        self.eval_seconds = 0.0
        self.wall_offset = 0.0
        self._t0 = None
        self.transition_updates = self._transition_updates()
        self.extra_checkpoints = self._extra_checkpoint_updates()

    # -- schedule bookkeeping ---------------------------------------------

    def _transition_updates(self) -> list:
        out = []
        if self.method.sdpoint:
            return out                      # a new random instance at every update
        if self.controller.update_level or self.method.cbs:
            prev = None
            for u in range(self.total_updates):
                st = self.controller.state_for_update(u)
                if prev is not None and (st.resolution, st.sigma) != (prev.resolution, prev.sigma):
                    out.append(u)
                prev = st
            return out
        for e in range(1, int(self.cfg.budget.epochs)):
            if self.controller.state_for_epoch(e) != self.controller.state_for_epoch(e - 1):
                # compare resolution/sigma only; labels always differ
                a, b = self.controller.state_for_epoch(e), self.controller.state_for_epoch(e - 1)
                if (a.resolution, a.sigma) != (b.resolution, b.sigma):
                    out.append(e * self.updates_per_epoch)
        return out

    def _extra_checkpoint_updates(self) -> dict:
        c = self.cfg.checkpoint
        extra = {}
        for t in self.transition_updates:
            for off in c.transition_offsets:
                u = t + int(off)
                if 0 < u <= self.total_updates:
                    extra.setdefault(u, "transition_window")
        if c.every_updates:
            for u in range(int(c.every_updates), self.total_updates + 1, int(c.every_updates)):
                extra.setdefault(u, "every_updates")
        return extra

    def _state_of_update(self, u: int) -> InterventionState:
        return self.controller.state_for_update(u, self.updates_per_epoch)

    def _set_epoch_state(self, e: int) -> None:
        if self.controller.update_level:
            self.controller.set_state(self.controller.state_for_update(e * self.updates_per_epoch))
        else:
            self.controller.set_epoch(e)

    def _state_record(self, state):
        if state is None:
            return None
        prev = self.controller.state
        self.controller.set_state(state)
        sig = self.controller.per_site_sigma()
        self.controller.set_state(prev)
        return {"state": state.to_dict(), "per_site_sigma": sig}

    # -- checkpoints -------------------------------------------------------

    def checkpoint_payload(self, reason: str) -> dict:
        u = self.global_update
        used = self._state_of_update(u - 1) if u > 0 else None
        nxt = self._state_of_update(u) if u < self.total_updates else None
        return {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "lr_schedule": {"kind": self.cfg.optimizer.schedule,
                            "total_updates": self.total_updates, "next_update": u,
                            "lr_next_update": (lr_at(u, self.cfg.optimizer, self.total_updates)
                                               if u < self.total_updates else None)},
            "rng": rng_state(),
            "data_position": {"epoch": self.epoch, "batch_index": self.batch_index,
                              "updates_per_epoch": self.updates_per_epoch,
                              "seed": int(self.cfg.run.seed),
                              "run_loss": self.run_loss, "run_n": self.run_n},
            "global_update": u, "epochs_completed": u / self.updates_per_epoch,
            "config": self.cfg.to_dict(),
            "intervention": {"used_for_last_update": self._state_record(used),
                             "next_update": self._state_record(nxt)},
            "metrics": self.metrics, "eval_seconds": self.eval_seconds,
            "diagnostic_seconds": self.diagnostic_seconds,
            "wall_seconds": self._wall(),
            "reason": reason,
            "provenance": self.provenance(),
        }

    def save(self, reason: str, name: str | None = None) -> Path:
        name = name or "update_%06d.pt" % self.global_update
        folder = self.out_dir if reason == "rolling" else self.out_dir / "checkpoints"
        return save_checkpoint(folder / name, self.checkpoint_payload(reason))

    def resume(self, path) -> None:
        ck = load_checkpoint(path, map_location=self.device)
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
        self.diagnostic_seconds = float(ck.get("diagnostic_seconds", 0.0))
        self.wall_offset = float(ck.get("wall_seconds", 0.0))
        set_rng_state(ck["rng"])
        self.log("resumed at update %d (epoch %d, batch %d)"
                 % (self.global_update, self.epoch, self.batch_index))

    # -- evaluation --------------------------------------------------------

    def snapshot(self, done_epochs: int, train_loss) -> dict:
        t0 = time.perf_counter()
        ecfg = self.cfg.evaluation
        current = self.controller.eval_current_state(done_epochs)
        states = {"current": current, "target": self.controller.target_state()}
        splits = {"train_probe": (self.probe_images, self.probe_labels),
                  "test": (self.test_images, self.test_labels)}
        ev = {}
        for path in ecfg.paths:
            ev[path] = {}
            for split in ecfg.splits:
                x, y = splits[split]
                r = evaluate(self.model, self.controller, self.pipeline, x, y,
                             states[path], bn_policy=ecfg.bn_policy,
                             batch_size=ecfg.batch_size, split=split)
                ev[path][split] = {"ce": r["ce"], "acc": r["acc"], "n": r["n"]}
        u = self.global_update
        rec = {"epoch": done_epochs, "update": u,
               "lr_last_update": lr_at(min(max(u - 1, 0), self.total_updates - 1),
                                       self.cfg.optimizer, self.total_updates),
               "state_used": self._state_record(current) if done_epochs > 0 else None,
               "state_next": (self._state_record(self._state_of_update(done_epochs * self.updates_per_epoch))
                              if done_epochs < self.cfg.budget.epochs else None),
               "train_loss_epoch": train_loss, "bn_policy": ecfg.bn_policy,
               "eval": ev, "elapsed_seconds": self._wall()}
        self.metrics.append(rec)
        self.eval_seconds += time.perf_counter() - t0
        self._write_json("metrics.json", self.metrics)
        return rec

    # -- training ----------------------------------------------------------

    def _wall(self) -> float:
        return self.wall_offset + (0.0 if self._t0 is None else time.perf_counter() - self._t0)

    def train_update(self) -> None:
        e, dev = self.epoch, self.device
        if self.controller.update_level:
            # one state per logical batch, shared by all of its microbatches
            self.controller.set_state(self._state_of_update(self.global_update))
        elif self.batch_index == 0 or self.controller.state is None:
            self.controller.set_epoch(e)
        s0 = self.batch_index * self.batch
        batch = self.perms[e][s0:s0 + self.batch]
        total = int(batch.size)
        idx = torch.as_tensor(batch, dtype=torch.long, device=dev)
        set_lr(self.optimizer, lr_at(self.global_update, self.cfg.optimizer, self.total_updates))
        self.optimizer.zero_grad(set_to_none=True)
        for a in range(0, total, self.micro):
            sl = idx[a:a + self.micro]
            n_m = int(sl.numel())
            x = self.augmenter(self.train_images[sl], sl, e)
            loss = F.cross_entropy(self.model(self.pipeline(x)), self.train_labels[sl])
            (loss * (n_m / total)).backward()
            self.run_loss += float(loss.detach()) * n_m
            self.run_n += n_m
        self.optimizer.step()
        self.global_update += 1
        self.batch_index += 1

    def run(self, max_updates: int | None = None, deadline: float | None = None) -> dict | None:
        """Train to the end of the budget (or stop after ``max_updates`` more).

        ``deadline`` (``time.time()`` seconds): checked before every update; when passed, the
        exact position is saved to ``rolling.pt`` and ``None`` is returned."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._write_json("config.json", self.cfg.to_dict())
        self._write_json("environment.json", self.environment())
        if self.assets.get("verification"):
            self._write_json("assets_verification.json", self.assets["verification"])
        self._t0 = time.perf_counter()
        stop_at = None if max_updates is None else self.global_update + int(max_updates)
        epochs = int(self.cfg.budget.epochs)
        if self.global_update == 0 and self.cfg.evaluation.at_epoch_zero and not self.metrics:
            self._set_epoch_state(0)
            self.snapshot(0, None)
        while self.epoch < epochs:
            self._set_epoch_state(self.epoch)
            while self.batch_index < self.updates_per_epoch:
                if stop_at is not None and self.global_update >= stop_at:
                    return None
                if deadline is not None and time.time() >= deadline:
                    self.save("rolling", name="rolling.pt")
                    self.log("deadline reached at update %d; rolling checkpoint saved" % self.global_update)
                    return None
                self.train_update()
                reason = self.extra_checkpoints.get(self.global_update)
                if reason and self.batch_index < self.updates_per_epoch:
                    self.save(reason)
            done = self.epoch + 1
            loss = self.run_loss / max(self.run_n, 1)
            if done % int(self.cfg.evaluation.every_epochs) == 0 or done == epochs:
                self.snapshot(done, loss)
            self._set_epoch_state(self.epoch)
            self.epoch, self.batch_index = done, 0
            self.run_loss, self.run_n = 0.0, 0
            for fn in self.epoch_callbacks:
                t_cb = time.perf_counter()
                fn(self, done)
                self.diagnostic_seconds += time.perf_counter() - t_cb
            if done in tuple(int(x) for x in self.cfg.checkpoint.at_epochs):
                self.save("at_epochs", name="epoch_%03d.pt" % done)
            elif self.cfg.checkpoint.every_epoch or self.global_update in self.extra_checkpoints:
                self.save(self.extra_checkpoints.get(self.global_update, "epoch_end")
                          if not self.cfg.checkpoint.every_epoch else "epoch_end",
                          name="epoch_%03d.pt" % done)
            if self.cfg.checkpoint.keep_rolling:
                self.save("rolling", name="rolling.pt")
            self.log("epoch %d/%d  update %d  loss %.4f  test acc current %.4f target %.4f"
                     % (done, epochs, self.global_update, loss,
                        self.metrics[-1]["eval"].get("current", {}).get("test", {}).get("acc", float("nan")),
                        self.metrics[-1]["eval"].get("target", {}).get("test", {}).get("acc", float("nan"))))
        return self.finish()

    def finish(self) -> dict:
        last = self.metrics[-1]
        wall = self._wall()
        final_state = self._state_of_update(self.total_updates - 1)
        summary = {
            "schema": RESULTS_SCHEMA,
            "method": self.method.id, "seed": int(self.cfg.run.seed),
            "optimizer": self.optimizer_record(),
            "validation_status": self.cfg.validation_status,
            "dataset": self.dataset.name, "arch": self.cfg.model.arch,
            "arch_validated": is_validated(self.cfg.model.arch),
            "epochs": int(self.cfg.budget.epochs), "updates": self.global_update,
            "updates_per_epoch": self.updates_per_epoch,
            "n_train": self.n_train, "n_test": int(self.test_images.shape[0]),
            "augmentation": self.augmenter.describe(),
            "final": {path: {split: dict(v) for split, v in d.items()}
                      for path, d in last["eval"].items()},
            "final_state_used": self._state_record(final_state),
            "final_state_is_target": (final_state.resolution, final_state.sigma)
                                     == (self.controller.target_state().resolution,
                                         self.controller.target_state().sigma),
            "transition_updates": self.transition_updates,
            "native_inference_state": self.controller.native_state().to_dict(),
            "schedule_segments": (self.method.cbs.table(self.updates_per_epoch, self.total_updates)
                                  if self.method.cbs else None),
            "timing": {"wall_seconds": wall, "eval_seconds": self.eval_seconds,
                       "diagnostic_seconds": self.diagnostic_seconds,
                       "train_seconds": wall - self.eval_seconds - self.diagnostic_seconds,
                       "scope": "from the start of run() (after data and model "
                                "construction) to the last evaluation, including "
                                "checkpoint writes; resumed runs add the wall time "
                                "stored in the checkpoint"},
            "method_definition": self.method.to_dict(),
            "provenance": self.provenance(),
        }
        self._write_json("summary.json", summary)
        return summary

    # -- records -----------------------------------------------------------

    def optimizer_record(self) -> dict:
        """Optimizer identity, in ``summary.json`` so one file per run is enough
        to aggregate a benchmark that varies the optimizer."""
        o = self.cfg.optimizer
        rec = {"tag": optimizer_tag(o), "name": o.name, "lr": o.lr,
               "weight_decay": o.weight_decay, "schedule": o.schedule,
               "warmup_updates": o.warmup_updates, "min_lr": o.min_lr}
        if o.name == "sgd":
            rec.update(momentum=o.momentum, nesterov=o.nesterov)
        else:
            rec.update(betas=list(o.betas), eps=o.eps,
                       weight_decay_coupling="decoupled" if o.name == "adamw"
                                             else "coupled_l2")
        return rec

    def provenance(self) -> dict:
        man = self.assets.get("manifest") or {}
        return {"assets_dir": self.cfg.assets.dir, "asset_states": man.get("states"),
                "asset_arrays": {k: v.get("sha256") for k, v in man.get("arrays", {}).items()},
                "torch": torch.__version__, "code": code_version()}

    def environment(self) -> dict:
        dev = self.device
        return {"python": sys.version.split()[0], "platform": platform.platform(),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "device": str(dev),
                "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "numpy": np.__version__}

    def _write_json(self, name: str, obj) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / name).write_text(json.dumps(obj, indent=2, allow_nan=True) + "\n",
                                         encoding="utf-8")


def code_version() -> str | None:
    import subprocess
    try:
        root = Path(__file__).resolve().parents[1]
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None
