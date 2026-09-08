"""Trainer / evaluator.

The trainer can train or evaluate the classifier at any transformation
parameter without rebuilding the experiment: the transformation family, the
schedule and the optimizer are injected separately.  Experiment 0 drives it with
a :class:`~continuation.schedules.ConstantSchedule`, which is the fixed-level
special case of a continuation run.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import optim as optim_mod
from .config import ExperimentConfig, to_dict
from .data import DatasetBundle, BatchIndexStream, fixed_subset_indices
from .diagnostics import transform_statistics
from .metrics import (RunLogger, capture_environment, save_checkpoint,
                      save_full_checkpoint)
from .models import build_model, count_parameters
from .pipeline import ChannelNormalizer, InputPipeline
from .schedules import build_schedule
from .seeding import configure_determinism, derive_seed
from .transforms import build_transform


@dataclass
class EvalResult:
    ce: float
    accuracy: float
    n: int
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {"ce": self.ce, "accuracy": self.accuracy, "n": self.n}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def evaluate(model, images_uint8: torch.Tensor, labels: torch.Tensor,
             pipeline: InputPipeline, eta: float, batch_size: int = 500) -> EvalResult:
    """Evaluation-mode cross-entropy and accuracy on a fixed subset.

    Unregularized cross-entropy only: weight decay is never folded into this
    number.
    """
    was_training = model.training
    model.eval()
    t0 = time.perf_counter()
    total_ce, total_correct, n = 0.0, 0, int(images_uint8.shape[0])
    for start in range(0, n, batch_size):
        xb = images_uint8[start:start + batch_size]
        yb = labels[start:start + batch_size]
        logits = model(pipeline(xb, eta))
        ce = F.cross_entropy(logits, yb, reduction="sum")
        total_ce += float(ce)
        total_correct += int((logits.argmax(dim=1) == yb).sum())
    if was_training:
        model.train()
    return EvalResult(total_ce / n, total_correct / n, n, time.perf_counter() - t0)


@dataclass
class TrainerState:
    step: int = 0
    epoch: int = 0
    stage: int | None = None
    history: list = field(default_factory=list)


class Trainer:
    """Fixed- or scheduled-parameter training of the classifier."""

    def __init__(self, cfg: ExperimentConfig, bundle: DatasetBundle,
                 out_dir, logger: RunLogger | None = None, resume_from=None):
        self.cfg = cfg
        self.device = resolve_device(cfg.run.device)
        self.determinism = configure_determinism(cfg.run.deterministic)
        self.bundle = bundle.to(self.device)
        self.out_dir = Path(out_dir)
        self.logger = logger or RunLogger(self.out_dir)

        self.transform = build_transform(cfg.transform)
        self.schedule = build_schedule(cfg.schedule, cfg.optim.total_steps)
        self.normalizer = ChannelNormalizer(self.bundle.mean, self.bundle.std).to(self.device)
        self.pipeline = InputPipeline(self.transform, self.normalizer,
                                      sync_timing=(self.device.type == "cuda"))
        self.eval_pipeline = InputPipeline(self.transform, self.normalizer, sync_timing=False)

        self.model = build_model(cfg.model, self.bundle.num_classes, cfg.run.seed).to(self.device)
        self.optimizer = optim_mod.build_optimizer(self.model, cfg.optim)
        self.batches = BatchIndexStream(len(self.bundle.train), cfg.optim.batch_size,
                                        cfg.run.seed, stream="batch")
        self.state = TrainerState()

        # Fixed evaluation subsets, drawn from an independent probe stream.
        ev = cfg.evaluation
        train_labels = self.bundle.train.labels.cpu().numpy()
        self.train_probe_idx = fixed_subset_indices(
            len(self.bundle.train), ev.train_probe_size, ev.probe_seed,
            "train_probe", labels=train_labels)
        self.stats_idx = fixed_subset_indices(
            len(self.bundle.train), ev.transform_stats_size, ev.probe_seed,
            "transform_stats", labels=train_labels)
        self.train_probe = self.bundle.train.subset(self.train_probe_idx, "train_probe")
        self.stats_subset = self.bundle.train.subset(self.stats_idx, "transform_stats")

        self.target_eta = float(self.transform.target_parameter)
        self.train_seconds = 0.0
        self.eval_seconds = 0.0
        self.start_step = 0
        self.lineage = {"parent_checkpoint": None, "branched_at_step": None}
        # Explicit, recorded policy. "carry" keeps SGD momentum buffers across a
        # transformation change; "reset" drops them. They are different
        # procedures and the choice is never implicit.
        self.momentum_policy = str(
            (cfg.schedule.params or {}).get("momentum_at_stage_boundary", "carry"))
        if self.momentum_policy not in ("carry", "reset"):
            raise ValueError("momentum_at_stage_boundary must be 'carry' or 'reset', got %r"
                             % self.momentum_policy)
        if resume_from is not None:
            self.resume(resume_from)

    # -- branching ----------------------------------------------------------

    def resume(self, path) -> dict:
        """Restore a complete training state so training continues identically.

        Restores parameters, optimizer state (momentum buffers included), the
        global update counter and the data-order position.  The learning-rate
        schedule is a pure function of the global counter, so it resumes on the
        original horizon automatically -- it is never restarted or rescaled.
        """
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if ck.get("format") != "full_state_v1":
            raise ValueError(
                "%s is a %r checkpoint; branching requires a complete "
                "'full_state_v1' state (parameters, optimizer, global step, "
                "data order, RNG)." % (path, ck.get("format", "legacy"))
            )
        self.model.load_state_dict(ck["model_state"])
        self.optimizer.load_state_dict(ck["optimizer_state"])
        if ck.get("batch_stream") is not None:
            self.batches.load_state_dict(ck["batch_stream"])
        if ck.get("torch_rng_state") is not None:
            torch.set_rng_state(ck["torch_rng_state"].cpu()
                                if hasattr(ck["torch_rng_state"], "cpu") else ck["torch_rng_state"])
        if ck.get("torch_cuda_rng_state") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all([t.cpu() for t in ck["torch_cuda_rng_state"]])
            except Exception:
                pass
        self.start_step = int(ck["step"])
        self.state.epoch = self.batches.epoch
        self.lineage = {"parent_checkpoint": str(path),
                        "branched_at_step": self.start_step,
                        "parent_lineage": ck.get("lineage", {})}
        return self.lineage

    # -- description / provenance -------------------------------------------

    def describe(self) -> dict:
        return {
            "config": to_dict(self.cfg),
            "device": str(self.device),
            "determinism": self.determinism,
            "model": {"arch": self.cfg.model.arch, **count_parameters(self.model)},
            "optimizer": optim_mod.describe(self.cfg.optim),
            "schedule": self.schedule.describe(),
            "pipeline": self.pipeline.describe(),
            "target_parameter": self.target_eta,
            "seeds": {
                "run_seed": self.cfg.run.seed,
                "init_stream": derive_seed(self.cfg.run.seed, "init"),
                "batch_stream": derive_seed(self.cfg.run.seed, "batch"),
                "split_seed": self.cfg.data.split_seed,
                "probe_seed": self.cfg.evaluation.probe_seed,
            },
            "eval_subsets": {
                "train_probe_size": int(self.train_probe_idx.size),
                "train_probe_idx_first10": self.train_probe_idx[:10].tolist(),
                "val_size": len(self.bundle.val),
                "transform_stats_size": int(self.stats_idx.size),
                "note": ("probe indices are positions within the training split, "
                         "which is itself pinned by data.split_seed"),
            },
            "batches_per_epoch": self.batches.batches_per_epoch,
            "start_step": self.start_step,
            "lineage": self.lineage,
            "momentum_at_stage_boundary": self.momentum_policy,
            "normalization_state": ("GroupNorm: no running statistics, no buffers, "
                                    "no recalibration at a transformation change"),
            "environment": capture_environment(),
        }

    # -- evaluation ---------------------------------------------------------

    def evaluate_at(self, eta: float, which: str = "val", batch_size: int | None = None):
        """Evaluate at an arbitrary transformation parameter.

        ``which`` is one of ``train_probe``, ``val``, ``test``, ``train_full``.
        """
        bs = batch_size or self.cfg.evaluation.eval_batch_size
        source = {
            "train_probe": self.train_probe,
            "val": self.bundle.val,
            "test": self.bundle.test,
            "train_full": self.bundle.train,
        }[which]
        return evaluate(self.model, source.images, source.labels, self.eval_pipeline, eta, bs)

    def transform_stats(self, eta: float) -> dict:
        x = InputPipeline.to_unit_float(self.stats_subset.images)
        return transform_statistics(self.transform, eta, x)

    def checkpoint_metrics(self, step: int, eta: float, phase: str = "checkpoint") -> dict:
        """The four metric families required at every shared checkpoint.

        ``eta`` is the *active* transformation level being reported.  Target
        metrics are always computed at the family's target endpoint and must not
        depend on ``eta`` -- that invariance is asserted at every transition.
        """
        t0 = time.perf_counter()
        tr_t = self.evaluate_at(eta, "train_probe")
        va_t = self.evaluate_at(eta, "val")
        is_target = self.transform.is_target(eta)
        # At the target endpoint the two objectives coincide; reuse rather than
        # recompute so the numbers are identical by construction.
        tr_0 = tr_t if is_target else self.evaluate_at(self.target_eta, "train_probe")
        va_0 = va_t if is_target else self.evaluate_at(self.target_eta, "val")
        self.eval_seconds += time.perf_counter() - t0
        return {
            "record": "eval",
            "step": int(step),
            "epoch": self.state.epoch,
            "parameter_name": self.transform.parameter_name,
            "parameter": float(eta),
            "is_target_endpoint": bool(is_target),
            "phase": phase,
            "stage": self.schedule.stage_index(min(step, self.cfg.optim.total_steps - 1)),
            "lr": optim_mod.lr_at(min(step, self.cfg.optim.total_steps - 1), self.cfg.optim),
            "transformed_train_probe": tr_t.as_dict(),
            "transformed_val": va_t.as_dict(),
            "target_train_probe": tr_0.as_dict(),
            "target_val": va_0.as_dict(),
            "eval_seconds": round(time.perf_counter() - t0, 4),
        }

    # -- training -----------------------------------------------------------

    def train_step(self, step: int) -> dict:
        cfg = self.cfg
        eta = self.schedule.parameter(step)
        stage = self.schedule.stage_index(step)
        if (self.state.stage is not None and stage is not None and stage != self.state.stage
                and self.momentum_policy == "reset"):
            optim_mod.reset_momentum(self.optimizer)
        self.state.stage = stage

        lr = optim_mod.lr_at(step, cfg.optim)
        optim_mod.set_lr(self.optimizer, lr)

        idx = self.batches.next_indices()
        t = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        xb = self.bundle.train.images[t]
        yb = self.bundle.train.labels[t]

        self.model.train()
        inputs = self.pipeline(xb, eta)
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, yb)   # unregularized CE; weight decay is in the optimizer
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.optim.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.optim.grad_clip)
        self.optimizer.step()

        with torch.no_grad():
            acc = float((logits.argmax(dim=1) == yb).float().mean())
        self.state.epoch = self.batches.epoch
        return {"loss": float(loss.detach()), "acc": acc, "lr": lr, "eta": eta}

    def transition_steps(self) -> list:
        """Global updates at which the active transformation level changes."""
        total = self.cfg.optim.total_steps
        out = []
        prev = self.schedule.parameter(0)
        for st in range(1, total):
            cur = self.schedule.parameter(st)
            if cur != prev:
                out.append(st)
                prev = cur
        return out

    def eval_step_grid(self, end_step: int) -> set:
        """Common checkpoints + stage boundaries + dense post-transition points.

        The common ``eval_every`` grid is what paired cross-arm claims are made
        on.  The dense points after each transition are descriptive adaptation
        curves only.
        """
        ev = self.cfg.evaluation
        steps = set()
        if ev.eval_every:
            steps |= set(range(0, end_step + 1, ev.eval_every))
        steps.add(end_step)
        steps.add(self.start_step)
        for b in self.transition_steps():
            if b > end_step:
                continue
            steps.add(b)
            if ev.dense_every:
                for k in range(0, ev.dense_window + 1, ev.dense_every):
                    if b + k <= end_step:
                        steps.add(b + k)
        return {st for st in steps if self.start_step <= st <= end_step}

    def fit(self, max_steps=None) -> dict:
        """Train from ``start_step`` to ``min(total_steps, max_steps)``.

        ``max_steps`` stops execution early **without** changing the
        learning-rate horizon: the LR is always ``lr_at(global_step,
        optim.total_steps)``.  This is what lets a 1,500-update prefix run on the
        original 14,040-update schedule instead of a compressed one.
        """
        cfg = self.cfg
        ev = cfg.evaluation
        end_step = int(min(cfg.optim.total_steps, max_steps or cfg.optim.total_steps))
        self.logger.write_json("run_description.json", self.describe())

        eta0 = self.schedule.parameter(self.start_step)
        stats = {"start": self.transform_stats(eta0)}
        if not self.transform.is_target(eta0):
            stats["target"] = self.transform_stats(self.target_eta)
        self.logger.write_json("transform_stats.json", stats)
        self.logger.log({"record": "transform_stats", "step": self.start_step, **stats["start"]})

        eval_steps = self.eval_step_grid(end_step)
        transitions = set(self.transition_steps())

        if ev.eval_at_step_zero or self.start_step > 0:
            rec = self.checkpoint_metrics(self.start_step, eta0, phase="branch_start")
            rec["note"] = ("before any gradient update on this objective"
                           if self.start_step else "before any gradient update")
            self.logger.log(rec)
            self._print_eval(rec)

        window = {"loss": 0.0, "acc": 0.0, "n": 0}
        last_eval = None
        t_train0 = time.perf_counter()
        for step in range(self.start_step, end_step):
            out = self.train_step(step)
            window["loss"] += out["loss"]
            window["acc"] += out["acc"]
            window["n"] += 1
            done = step + 1

            if cfg.run.log_every and done % cfg.run.log_every == 0 and window["n"]:
                self.logger.log({
                    "record": "train_minibatch",
                    "step": done, "epoch": self.state.epoch,
                    "parameter": out["eta"], "lr": out["lr"],
                    "stage": self.schedule.stage_index(step),
                    "window": window["n"],
                    "minibatch_transformed_ce_mean": window["loss"] / window["n"],
                    "minibatch_transformed_acc_mean": window["acc"] / window["n"],
                    "note": "training-mode minibatch averages; not an eval-mode metric",
                })
                window = {"loss": 0.0, "acc": 0.0, "n": 0}

            if done in eval_steps:
                eta_out = self.schedule.parameter(step)
                rec = self.checkpoint_metrics(done, eta_out, phase="checkpoint")
                # At a transition, re-evaluate the *same weights* under the
                # incoming level before any update on it.  Target metrics must be
                # identical across the pair; only active-level metrics may change.
                # Logged as evidence, not merely asserted.  The label is set
                # before logging, since a record is serialized when logged.
                is_transition = done in transitions and done < end_step
                if is_transition:
                    rec["phase"] = "pre_transition"
                self.logger.log(rec)
                self._print_eval(rec)
                last_eval = rec

                if is_transition:
                    eta_in = self.schedule.parameter(done)
                    rec_in = self.checkpoint_metrics(done, eta_in, phase="post_transition")
                    self.logger.log(rec_in)
                    self._assert_target_invariance(rec, rec_in)
                    save_full_checkpoint(
                        self.out_dir / ("checkpoint_step%06d.pt" % done), self.model,
                        self.optimizer, done, self.batches,
                        lineage=self.lineage, config=to_dict(cfg),
                        extra={"boundary": True, "eta_out": eta_out, "eta_in": eta_in})

            if ev.checkpoint_every and done % ev.checkpoint_every == 0:
                save_full_checkpoint(self.out_dir / ("checkpoint_step%06d.pt" % done),
                                     self.model, self.optimizer, done, self.batches,
                                     lineage=self.lineage, config=to_dict(cfg))
        self.train_seconds = time.perf_counter() - t_train0

        final_ckpt = None
        if ev.save_final_checkpoint:
            if end_step < cfg.optim.total_steps:
                name = "checkpoint_step%06d.pt" % end_step
            else:
                name = "checkpoint_final.pt"
            final_ckpt = save_full_checkpoint(
                self.out_dir / name, self.model, self.optimizer, end_step, self.batches,
                lineage=self.lineage, config=to_dict(cfg),
                extra={"parameter": self.schedule.parameter(max(end_step - 1, 0))})

        if last_eval is not None and last_eval["step"] == end_step:
            final = dict(last_eval)
        else:
            final = self.checkpoint_metrics(end_step, self.schedule.parameter(max(end_step - 1, 0)))
        final["record"] = "final"
        summary = {
            "run_name": cfg.run.name,
            "seed": cfg.run.seed,
            "family": self.transform.name,
            "parameter_name": self.transform.parameter_name,
            "parameter": float(eta0),
            "schedule": self.schedule.describe(),
            "start_step": self.start_step,
            "end_step": end_step,
            "total_steps": cfg.optim.total_steps,
            "updates_executed": end_step - self.start_step,
            "transitions": sorted(transitions),
            "momentum_at_stage_boundary": self.momentum_policy,
            "lineage": self.lineage,
            "final_checkpoint": final_ckpt,
            "batch_size": cfg.optim.batch_size,
            "epochs_equivalent": round(end_step / self.batches.batches_per_epoch, 3),
            "final": {k: v for k, v in final.items() if k != "record"},
            "timing": {
                "train_seconds": round(self.train_seconds, 3),
                "eval_seconds": round(self.eval_seconds, 3),
                "transform_seconds_train": round(self.pipeline.transform_seconds, 3),
                "transform_seconds_eval": round(self.eval_pipeline.transform_seconds, 3),
                "note": ("equal update counts control the optimization budget, not the "
                         "total computational cost"),
            },
            "transform_stats": stats,
        }
        self.logger.log({"record": "final", **summary["final"]})
        self.logger.write_json("summary.json", summary)
        return summary

    @staticmethod
    def _assert_target_invariance(pre, post, tol: float = 1e-9) -> None:
        """Target metrics must not depend on the currently active training level."""
        for block in ("target_train_probe", "target_val"):
            for key in ("ce", "accuracy"):
                a, b = pre[block][key], post[block][key]
                if abs(a - b) > tol:
                    raise AssertionError(
                        "target metric %s.%s changed across a transition at step %s "
                        "(%.12g vs %.12g); original-image evaluation is coupled to the "
                        "active training transform" % (block, key, pre["step"], a, b)
                    )

    def _print_eval(self, rec: dict) -> None:
        self.logger.console(
            "step %6d | %s=%.3g | transformed val CE %.4f acc %.4f | target val CE %.4f acc %.4f"
            % (rec["step"], rec["parameter_name"], rec["parameter"],
               rec["transformed_val"]["ce"], rec["transformed_val"]["accuracy"],
               rec["target_val"]["ce"], rec["target_val"]["accuracy"])
        )
