"""Loss evaluation under two explicit BatchNorm policies.

``saved``        load the checkpoint's parameters **and** running statistics,
                 override learned parameters if a perturbed vector is given,
                 set the intervention state, evaluate in inference mode.  The
                 running statistics are reloaded from the checkpoint before
                 every evaluation.  = the stored model under the perturbation.
``recalibrated`` copy the learned parameters, reset every BN running mean/var
                 and counter, set momentum None (cumulative average), set the
                 intervention state, forward a fixed training calibration subset
                 in fixed order and batches under ``no_grad`` in training mode,
                 then evaluate in inference mode.  No learned parameter changes;
                 ResNet-20 has no dropout.  = the model after its normalisation
                 statistics adapted; a different evaluated function.

Loss = mean cross-entropy (no weight decay); accuracy recorded.  Probes are
evaluation subsets, never the full dataset unless named ``*_full``.
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

from .common import controller


def weights_sha(params: dict, names) -> str:
    h = hashlib.sha256()
    for n in names:
        h.update(np.ascontiguousarray(params[n].detach().to("cpu", torch.float32).numpy()).tobytes())
    return h.hexdigest()


def as_state(d):
    if d is None or isinstance(d, InterventionState):
        return d
    return InterventionState(d.get("resolution"), d.get("sigma"), d.get("label", "custom"))


class Evaluator:
    def __init__(self, dataset, subsets: dict, pinned_train_probe, device, batch_size=500):
        self.device = torch.device(device)
        self.bs = int(batch_size)
        self.pipeline = build_pipeline(dataset, DataConfig(expected_mean=CIFAR10_MEAN,
                                                           expected_std=CIFAR10_STD)).to(self.device)
        tr, te = dataset.train, dataset.test

        def take(split, idx):
            i = torch.as_tensor(np.asarray(idx), dtype=torch.long)
            return split.images[i].to(self.device), split.labels[i].to(self.device)

        self.calib = {"calib2k": take(tr, subsets["calibration_train_idx"]),
                      "calib10k": take(tr, subsets["calibration_large_train_idx"])}
        self.splits = {"train_probe": take(tr, subsets["train_probe_idx"]),
                       "test_probe": take(te, subsets["test_probe_idx"]),
                       "train_large": take(tr, subsets["train_large_idx"]),
                       "test_full": (te.images.to(self.device), te.labels.to(self.device)),
                       "train_full": (tr.images.to(self.device), tr.labels.to(self.device)),
                       "pinned_train_probe_500": take(tr, pinned_train_probe)}
        self._models = {}
        self.names = None

    def model(self, method):
        if method not in self._models:
            m = build_model("resnet20_bn_cifar", 10).to(self.device)
            c = controller(method)
            c.attach(m)
            self.names = [n for n, _ in m.named_parameters()]
            self._models[method] = (m, c)
        return self._models[method]

    @torch.no_grad()
    def _run(self, model, x, y):
        ce, correct = 0.0, 0
        for i in range(0, x.shape[0], self.bs):
            logits = model(self.pipeline(x[i:i + self.bs]))
            yy = y[i:i + self.bs]
            ce += float(F.cross_entropy(logits.float(), yy, reduction="sum"))
            correct += int((logits.argmax(1) == yy).sum())
        n = int(x.shape[0])
        return {"n": n, "ce": ce / n, "acc": correct / n, "finite": bool(np.isfinite(ce))}

    @torch.no_grad()
    def evaluate(self, method, model_state, state, policy, params=None,
                 splits=("train_probe", "test_probe"), calib="calib2k"):
        t0 = time.perf_counter()
        model, ctrl = self.model(method)
        if policy == "saved":
            model.load_state_dict({k: v.to(self.device) for k, v in model_state.items()}, strict=True)
            if params is not None:
                for n, p in model.named_parameters():
                    p.copy_(params[n].to(self.device, p.dtype))
        elif policy == "recalibrated":
            src = params if params is not None else model_state
            for n, p in model.named_parameters():
                p.copy_(src[n].to(self.device, p.dtype))
        else:
            raise ValueError(policy)
        bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        st = as_state(state)
        ctrl.set_state(st)
        if policy == "recalibrated":
            for bn in bns:
                bn.reset_running_stats()
                bn.momentum = None
            model.train()
            x, _ = self.calib[calib]
            for i in range(0, x.shape[0], self.bs):
                model(self.pipeline(x[i:i + self.bs]))
        model.eval()
        out = {s: self._run(model, *self.splits[s]) for s in splits}
        for bn in bns:
            bn.momentum = 0.1
        used = params if params is not None else {n: model_state[n] for n in self.names}
        return {"policy": policy, "calibration": calib if policy == "recalibrated" else None,
                "state": None if st is None else {"resolution": st.resolution, "sigma": st.sigma},
                "per_site_sigma": ctrl.per_site_sigma(), "splits": out,
                "weights_sha256": weights_sha(used, self.names),
                "seconds": time.perf_counter() - t0}
