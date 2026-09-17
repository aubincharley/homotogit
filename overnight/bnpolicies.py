"""Four BatchNorm evaluation policies for a final (or intermediate) set of weights.

Every policy starts from a fresh load of the untouched tensors into a separate model with its own
controller, set to one fixed inference state.  Learned weights and BN affine parameters are never
updated (hash-checked after each policy); calibration uses training images only, no labels, no
gradients.  Evaluation after calibration is in ``eval()`` mode on the full clean training set
(50,000) and the full test set (10,000), consecutive batches of 500.

P0 ``saved_native``          the checkpoint's own running buffers, no calibration.
P1 ``cumulative_clean_500``  ``reset_running_stats()``, ``momentum=None``, one ``train()``-mode
                             forward pass over the 50,000 clean training images in official file
                             order (0..49,999), batches of 500 (exactly 100 batches).
P2 ``cumulative_clean_32``   P1 with batches of 32 (1,563 batches: 1,562 of 32 and a final one of 16).
P3 ``ema_training_loader``   SDPoint author ``validate()`` adapted: no reset, the checkpoint's own
                             buffers as the starting point, ``momentum=0.1``, one pass in
                             ``train()`` mode over all 50,000 training images in a frozen
                             seed-specific order, batches of the regime's physical training batch
                             (final partial batch kept), crop/flip from an independent frozen
                             stream only for regimes trained with it.

``momentum=None`` in torch: after batch t the buffer is the running mean of the t per-batch
statistics with equal weight per batch (a final short batch counts as a full one); the variance
buffer averages the unbiased per-batch variances.  It is not the pooled variance of the dataset.
"""
from __future__ import annotations

import hashlib
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from continuation_core.augment import apply_crop_flip, crop_flip_draws
from continuation_core.config import ExperimentConfig
from continuation_core.controller import InterventionController
from continuation_core.data import build_pipeline
from continuation_core.models import build_model, site_map
from continuation_core.seeding import derive_seed

POLICIES = ("P0_saved_native", "P1_cumulative_clean_500", "P2_cumulative_clean_32", "P3_ema_training_loader")
P3_ORDER_STREAM = "bn_p3::calibration_order"
P3_AUG_STREAM = "bn_p3::pad4_crop32_hflip::pass::%d"
EVAL_BATCH = 500


def sha_tensors(tensors: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(tensors):
        h.update(k.encode())
        h.update(np.ascontiguousarray(tensors[k].detach().float().cpu().numpy()).tobytes())
    return h.hexdigest()


def learned(state: dict) -> dict:
    return {k: v for k, v in state.items() if "running" not in k and "num_batches" not in k}


def bn_buffers(state: dict) -> dict:
    return {k: v for k, v in state.items() if "running" in k or "num_batches" in k}


def p3_order(seed: int, n: int) -> np.ndarray:
    return np.random.Generator(np.random.PCG64(derive_seed(int(seed), P3_ORDER_STREAM))).permutation(n).astype(np.int64)


def p3_draws(seed: int, n: int) -> dict:
    return crop_flip_draws(int(seed), 0, n, P3_AUG_STREAM)


class _RngGuard:
    """Every global RNG (torch CPU/CUDA, numpy, python) restored on exit."""

    def __enter__(self):
        self.s = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                  np.random.get_state(), random.getstate())
        return self

    def __exit__(self, *exc):
        torch.set_rng_state(self.s[0])
        if self.s[1] is not None:
            torch.cuda.set_rng_state_all(self.s[1])
        np.random.set_state(self.s[2])
        random.setstate(self.s[3])


class PolicyEvaluator:
    def __init__(self, cfg: ExperimentConfig, model_state: dict, dataset, device, updates_per_epoch: int,
                 state=None, physical_batch: int | None = None, augmentation: str | None = None):
        self.cfg = cfg
        self.device = torch.device(device)
        self.method = cfg.method_spec()
        self.seed = int(cfg.run.seed)
        self.state_cpu = {k: v.detach().to("cpu").clone() for k, v in model_state.items()}
        self.learned_sha = sha_tensors(learned(self.state_cpu))
        self.saved_buffers_sha = sha_tensors(bn_buffers(self.state_cpu))
        with _RngGuard():
            self.model = build_model(cfg.model.arch, dataset.num_classes).to(self.device)
        self.ctrl = InterventionController(self.method, site_map(cfg.model.arch), cfg.model.arch)
        self.ctrl.configure(int(updates_per_epoch), int(updates_per_epoch) * int(cfg.budget.epochs), self.seed)
        self.ctrl.attach(self.model)
        self.state = state if state is not None else self.ctrl.native_state()
        self.pipeline = build_pipeline(dataset, cfg.data).to(self.device)
        tr, te = dataset.train.to(self.device), dataset.test.to(self.device)
        self.train_x, self.train_y, self.test_x, self.test_y = tr.images, tr.labels, te.images, te.labels
        self.n = int(self.train_x.shape[0])
        self.physical_batch = int(physical_batch or cfg.budget.microbatch)
        self.augmentation = augmentation if augmentation is not None else getattr(cfg.data, "augmentation", "none")

    # -- helpers -----------------------------------------------------------------------------
    def _bns(self):
        return [m for m in self.model.modules() if isinstance(m, torch.nn.BatchNorm2d)]

    def _reload(self):
        self.model.load_state_dict({k: v.to(self.device) for k, v in self.state_cpu.items()}, strict=True)
        for bn in self._bns():
            bn.momentum = 0.1
        self.model.eval()

    @torch.no_grad()
    def _score(self, x, y) -> dict:
        ce, correct = 0.0, 0
        for i in range(0, int(x.shape[0]), EVAL_BATCH):
            logits = self.model(self.pipeline(x[i:i + EVAL_BATCH]))
            yy = y[i:i + EVAL_BATCH]
            ce += float(F.cross_entropy(logits.float(), yy, reduction="sum"))
            correct += int((logits.argmax(1) == yy).sum())
        n = int(x.shape[0])
        return {"n": n, "ce": ce / n, "acc": correct / n}

    @torch.no_grad()
    def _calibrate(self, policy: str) -> dict:
        n = self.n
        if policy in ("P1_cumulative_clean_500", "P2_cumulative_clean_32"):
            bs = 500 if policy.startswith("P1") else 32
            for bn in self._bns():
                bn.reset_running_stats()
                bn.momentum = None
            self.model.train()
            n_batches = 0
            for i in range(0, n, bs):
                self.model(self.pipeline(self.train_x[i:i + bs]))
                n_batches += 1
            order = np.arange(n, dtype=np.int64)
            return {"dataset": "CIFAR-10 training set (50,000), clean", "order": "official file order 0..N-1",
                    "order_sha256": hashlib.sha256(order.tobytes()).hexdigest(), "augmentation": "none",
                    "batch_size": bs, "n_batches": n_batches, "final_batch": n - (n_batches - 1) * bs,
                    "remainder": "kept; weighted as one batch in the cumulative average",
                    "reset": True, "momentum": None, "passes": 1, "mode": "train() forward, no_grad",
                    "labels_used": False}
        if policy == "P3_ema_training_loader":
            bs = self.physical_batch
            order = p3_order(self.seed, n)
            idx_all = torch.as_tensor(order, device=self.device)
            aug = self.augmentation != "none"
            if aug:
                d = {k: torch.as_tensor(v, device=self.device) for k, v in p3_draws(self.seed, n).items()}
            for bn in self._bns():
                bn.momentum = 0.1
            self.model.train()
            n_batches = 0
            for i in range(0, n, bs):
                idx = idx_all[i:i + bs]
                x = self.train_x[idx]
                if aug:
                    x = apply_crop_flip(x, d["dx"][idx], d["dy"][idx], d["flip"][idx])
                self.model(self.pipeline(x))
                n_batches += 1
            return {"dataset": "CIFAR-10 training set (50,000)", "order": "seed-specific frozen permutation (%s)" % P3_ORDER_STREAM,
                    "order_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
                    "augmentation": ("pad4_crop32_hflip, draws %s keyed by official index" % (P3_AUG_STREAM % 0)) if aug else "none",
                    "augmentation_draws_sha256": (hashlib.sha256(json.dumps({k: hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest()
                                                                             for k, v in d.items()}, sort_keys=True).encode()).hexdigest()
                                                  if aug else None),
                    "batch_size": bs, "n_batches": n_batches, "final_batch": n - (n_batches - 1) * bs,
                    "remainder": "kept", "reset": False, "start": "checkpoint's own saved buffers",
                    "momentum": 0.1, "passes": 1, "mode": "train() forward, no_grad", "labels_used": False,
                    "note": "author-style adaptation of xternalz/SDPoint@0013c5d main.py validate(); not the released ImageNet pipeline"}
        raise ValueError(policy)

    # -- policies ----------------------------------------------------------------------------
    def run(self, policy: str, splits=("train_full", "test_full")) -> dict:
        t0 = time.perf_counter()
        with _RngGuard():
            self._reload()
            self.ctrl.set_state(self.state)
            rec = {"policy": policy, "state": self.state.to_dict(), "per_site_sigma": self.ctrl.per_site_sigma()}
            if policy == "P0_saved_native":
                rec["calibration"] = None
            else:
                rec["calibration"] = self._calibrate(policy)
            self.model.eval()
            rec["buffers_sha256"] = sha_tensors(bn_buffers(self.model.state_dict()))
            if "train_full" in splits:
                rec["train_full"] = self._score(self.train_x, self.train_y)
            if "test_full" in splits:
                rec["test_full"] = self._score(self.test_x, self.test_y)
            if sha_tensors(learned(self.model.state_dict())) != self.learned_sha:
                raise RuntimeError("learned tensors changed under %s" % policy)
            rec["learned_tensors_unchanged"] = True
            self._reload()
        rec["seconds"] = time.perf_counter() - t0
        return rec

    def all_policies(self, policies=POLICIES) -> dict:
        out = {"state": self.state.to_dict(), "seed": self.seed, "method": self.method.id,
               "learned_sha256": self.learned_sha, "saved_buffers_sha256": self.saved_buffers_sha,
               "physical_batch_for_P3": self.physical_batch, "augmentation_for_P3": self.augmentation,
               "policies": {}}
        for p in policies:
            out["policies"][p] = self.run(p)
        if "P0_saved_native" in policies:
            again = self.run("P0_saved_native")
            first = out["policies"]["P0_saved_native"]
            out["P0_repeat_identical"] = (again["train_full"] == first["train_full"] and again["test_full"] == first["test_full"]
                                          and again["buffers_sha256"] == first["buffers_sha256"] == self.saved_buffers_sha)
        return out
