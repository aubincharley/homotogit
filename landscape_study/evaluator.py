"""Loss evaluation with an explicit BatchNorm convention.

``recalibrated`` (main landscapes)
    1. copy the requested learned parameters into a private working model;
    2. reset every BatchNorm running mean to 0, variance to 1, counter to 0, and
       set momentum to ``None`` (cumulative average);
    3. set the requested intervention state;
    4. forward the fixed calibration subset (training images only) in training
       mode, in the same fixed order and batches every time, under
       ``torch.no_grad`` -- no learned parameter changes, the network has no
       dropout;
    5. evaluate in inference mode on the fixed probes.

``saved_stats``
    load parameters **and** the checkpoint's own running statistics, evaluate in
    inference mode.  Used only for unperturbed checkpoints, to reproduce the
    recorded metrics.

Loss = mean cross-entropy, no weight-decay term.  Accuracy is recorded too.
The source checkpoint tensors are only read (copied into the working model).
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import torch
import torch.nn.functional as F

from continuation_core.config import DataConfig
from continuation_core.controller import InterventionState
from continuation_core.data import build_pipeline
from continuation_core.models import build_model
from continuation_core.presets import CIFAR10_MEAN, CIFAR10_STD

from .sources import controller


def weights_sha(params: dict, names: list) -> str:
    h = hashlib.sha256()
    for n in names:
        h.update(np.ascontiguousarray(params[n].detach().to("cpu", torch.float32).numpy()).tobytes())
    return h.hexdigest()


def as_state(d) -> InterventionState | None:
    if d is None or isinstance(d, InterventionState):
        return d
    return InterventionState(d.get("resolution"), d.get("sigma"), d.get("label", "custom"))


class StudyEvaluator:
    def __init__(self, dataset, subsets: dict, pinned_train_probe, device, batch_size: int = 500):
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.pipeline = build_pipeline(dataset, DataConfig(expected_mean=CIFAR10_MEAN,
                                                           expected_std=CIFAR10_STD)).to(self.device)
        tr, te = dataset.train, dataset.test

        def take(split, idx):
            i = torch.as_tensor(np.asarray(idx), dtype=torch.long)
            return split.images[i].to(self.device), split.labels[i].to(self.device)

        # calibration is drawn from the training split only, by construction
        self.calibration = take(tr, subsets["calibration_train_idx"])
        self.splits = {
            "train_probe": take(tr, subsets["train_probe_idx"]),
            "test_probe": take(te, subsets["test_probe_idx"]),
            # historical splits, for reproducing recorded metrics
            "pinned_train_probe_500": take(tr, pinned_train_probe),
            "test_full": (te.images.to(self.device), te.labels.to(self.device)),
        }
        self._models = {}

    def model(self, method: str):
        if method not in self._models:
            m = build_model("resnet20_bn_cifar", 10).to(self.device)
            c = controller(method)
            c.attach(m)
            self.names = [n for n, p in m.named_parameters()]
            self._models[method] = (m, c)
        return self._models[method]

    @torch.no_grad()
    def _run(self, model, x, y):
        ce, correct = 0.0, 0
        for i in range(0, x.shape[0], self.batch_size):
            logits = model(self.pipeline(x[i:i + self.batch_size]))
            yy = y[i:i + self.batch_size]
            ce += float(F.cross_entropy(logits.float(), yy, reduction="sum"))
            correct += int((logits.argmax(1) == yy).sum())
        n = int(x.shape[0])
        return {"n": n, "ce": ce / n, "acc": correct / n,
                "finite": bool(np.isfinite(ce))}

    @torch.no_grad()
    def recalibrated(self, method: str, params: dict, state, splits=("train_probe", "test_probe")):
        t0 = time.perf_counter()
        model, ctrl = self.model(method)
        for n, p in model.named_parameters():
            p.copy_(params[n].to(self.device, p.dtype))
        bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        for bn in bns:
            bn.reset_running_stats()
            bn.momentum = None
        st = as_state(state)
        ctrl.set_state(st)
        model.train()
        x, _ = self.calibration
        for i in range(0, x.shape[0], self.batch_size):
            model(self.pipeline(x[i:i + self.batch_size]))
        model.eval()
        out = {s: self._run(model, *self.splits[s]) for s in splits}
        for bn in bns:
            bn.momentum = 0.1
        return {"bn_policy": "recalibrated", "state": None if st is None else st.to_dict(),
                "per_site_sigma": ctrl.per_site_sigma(), "splits": out,
                "calibration": {"n": int(x.shape[0]), "batch_size": self.batch_size,
                                "momentum": None, "mode": "train, no_grad"},
                "weights_sha256": weights_sha(params, self.names),
                "seconds": time.perf_counter() - t0}

    @torch.no_grad()
    def saved_stats(self, method: str, model_state: dict, state,
                    splits=("pinned_train_probe_500", "test_full", "train_probe", "test_probe")):
        t0 = time.perf_counter()
        model, ctrl = self.model(method)
        model.load_state_dict({k: v.to(self.device) for k, v in model_state.items()}, strict=True)
        st = as_state(state)
        ctrl.set_state(st)
        model.eval()
        out = {s: self._run(model, *self.splits[s]) for s in splits}
        params = {n: model_state[n] for n in self.names}
        return {"bn_policy": "saved_stats", "state": None if st is None else st.to_dict(),
                "per_site_sigma": ctrl.per_site_sigma(), "splits": out,
                "weights_sha256": weights_sha(params, self.names),
                "seconds": time.perf_counter() - t0}
