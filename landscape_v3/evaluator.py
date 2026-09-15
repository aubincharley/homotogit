"""Loss evaluation under three BatchNorm policies, in float32 or float64.

``saved``          checkpoint running statistics, reloaded before every evaluation
                   (identical code path and numerics to landscape_v2).
``recalibrated``   pointwise: reset, cumulative average over calib2k (fixed order,
                   batches of 500, training mode, no_grad) under the evaluated
                   weights and state, then inference mode (identical to v2).
``centre_frozen``  recalibrate once at the reference checkpoint weights and state
                   (the ``recalibrated`` procedure with the unperturbed weights),
                   store those statistics, then evaluate every perturbed weight
                   vector in inference mode with them held fixed.

float64 evaluation uses a separate double-precision copy of the model, the input
pipeline output cast to float64, and float64 BatchNorm buffers: ``saved`` casts
the checkpoint statistics, ``centre_frozen`` casts the float32 centre statistics
(the policy's statistics are those of the standard calibration), and
``recalibrated`` runs its per-point calibration in float64.
The loss is mean cross-entropy without weight decay; accuracy is recorded.
Nothing is written back to checkpoints; the model's buffers are overwritten
before every evaluation so no state leaks between points.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from landscape_v2.evaluator import Evaluator as V2Evaluator, as_state, weights_sha

from . import common as C


class Evaluator(V2Evaluator):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._models64 = {}
        self._cstats = {}

    def model64(self, method):
        if method not in self._models64:
            from continuation_core.models import build_model
            m = build_model("resnet20_bn_cifar", 10).to(self.device).double()
            c = C.V2.controller(method)
            c.attach(m)
            self._models64[method] = (m, c)
        return self._models64[method]

    @staticmethod
    def bns(model):
        return [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]

    @torch.no_grad()
    def _run_prec(self, model, x, y, dtype):
        ce, correct = 0.0, 0
        for i in range(0, x.shape[0], self.bs):
            logits = model(self.pipeline(x[i:i + self.bs]).to(dtype))
            yy = y[i:i + self.bs]
            ce += float(F.cross_entropy(logits.to(dtype if dtype == torch.float64 else torch.float32),
                                        yy, reduction="sum"))
            correct += int((logits.argmax(1) == yy).sum())
        n = int(x.shape[0])
        return {"n": n, "ce": ce / n, "acc": correct / n, "finite": bool(ce == ce and abs(ce) != float("inf"))}

    @torch.no_grad()
    def _calibrate(self, model, calib, dtype):
        for bn in self.bns(model):
            bn.reset_running_stats()
            bn.momentum = None
        model.train()
        x, _ = self.calib[calib]
        for i in range(0, x.shape[0], self.bs):
            model(self.pipeline(x[i:i + self.bs]).to(dtype))
        model.eval()

    @torch.no_grad()
    def centre_stats(self, cache_key, method, model_state, state, prec="f32", calib="calib2k"):
        """BN buffers recalibrated at the unperturbed weights (cached per key).

        Always computed with the standard float32 procedure; float64 evaluations and
        float64 HVPs load these float32 statistics cast to float64, so a float64
        re-evaluation changes only the arithmetic, not the statistics (``prec`` is
        accepted for call compatibility and ignored)."""
        key = (cache_key, C.stag(state), calib)
        if key not in self._cstats:
            if len(self._cstats) > 32:
                self._cstats.clear()
            dtype = torch.float32
            model, ctrl = self.model(method)
            for n, p in model.named_parameters():
                p.copy_(model_state[n].to(self.device, dtype))
            ctrl.set_state(as_state(state))
            self._calibrate(model, calib, dtype)
            self._cstats[key] = {k: v.detach().clone() for k, v in model.state_dict().items()
                                 if "running" in k or "num_batches" in k}
            for bn in self.bns(model):
                bn.momentum = 0.1
        return self._cstats[key]

    @torch.no_grad()
    def evaluate3(self, method, model_state, state, policy, params=None, splits=C.PROBES,
                  prec="f32", cache_key=None, calib="calib2k"):
        if prec == "f32" and policy in (C.SAVED, C.POINTWISE):
            return self.evaluate(method, model_state, state, policy, params=params, splits=splits, calib=calib)
        t0 = time.perf_counter()
        dtype = torch.float64 if prec == "f64" else torch.float32
        model, ctrl = self.model64(method) if prec == "f64" else self.model(method)
        src = params if params is not None else model_state
        st = as_state(state)
        if policy == C.SAVED:
            model.load_state_dict({k: v.to(self.device) for k, v in model_state.items()}, strict=True)
            model.to(dtype)
            for n, p in model.named_parameters():
                p.copy_(src[n].to(self.device, dtype))
            ctrl.set_state(st)
            model.eval()
        elif policy == C.POINTWISE:
            for n, p in model.named_parameters():
                p.copy_(src[n].to(self.device, dtype))
            ctrl.set_state(st)
            self._calibrate(model, calib, dtype)
        elif policy == C.CFROZEN:
            stats = self.centre_stats(cache_key, method, model_state, state, prec, calib)
            model.load_state_dict({k: v.to(self.device) for k, v in model_state.items()}, strict=True)
            model.to(dtype)
            for k, v in stats.items():
                model.get_buffer(k).copy_(v)
            for n, p in model.named_parameters():
                p.copy_(src[n].to(self.device, dtype))
            ctrl.set_state(st)
            model.eval()
        else:
            raise ValueError(policy)
        out = {s: self._run_prec(model, *self.splits[s], dtype) for s in splits}
        for bn in self.bns(model):
            bn.momentum = 0.1
        used = params if params is not None else {n: model_state[n] for n in self.names}
        return {"policy": policy, "precision": prec, "calibration": None if policy == C.SAVED else calib,
                "state": None if st is None else {"resolution": st.resolution, "sigma": st.sigma},
                "per_site_sigma": ctrl.per_site_sigma(), "splits": out,
                "weights_sha256": weights_sha(used, self.names), "seconds": time.perf_counter() - t0}
