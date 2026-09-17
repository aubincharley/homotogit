"""Final-endpoint evaluation: saved statistics, Panel A (native + recalibration), Panel B
(original path + recalibration).

Every evaluation starts from the untouched checkpoint tensors.  Recalibration protocol,
identical for every arm:

* all 50,000 CIFAR-10 training images in official file order (indices 0..49,999), no
  augmentation, the reference input pipeline, batches of 500;
* every BatchNorm: ``reset_running_stats()``, ``momentum=None`` (cumulative average);
* one pass in training mode under ``torch.no_grad()`` with the stated inference state set;
  no gradients, no optimizer, learned weights and BN affine parameters untouched (verified by
  hash); nothing else in ResNet-20 depends on the mode;
* then evaluation in eval mode.  Test images and labels are never used for calibration.

Reported per state and policy: full training set (50,000) and full test set (10,000)
accuracy and mean CE.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import torch
import torch.nn.functional as F

from continuation_core.checkpoint import load_checkpoint
from continuation_core.config import ExperimentConfig
from continuation_core.controller import InterventionController
from continuation_core.data import build_pipeline
from continuation_core.models import build_model, site_map

CAL_BATCH = 500


def _sha(tensors: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(tensors):
        h.update(k.encode())
        h.update(np.ascontiguousarray(tensors[k].detach().float().cpu().numpy()).tobytes())
    return h.hexdigest()


def _bns(model):
    return [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]


@torch.no_grad()
def _run(model, pipeline, x, y, bs=CAL_BATCH):
    ce, correct = 0.0, 0
    for i in range(0, x.shape[0], bs):
        logits = model(pipeline(x[i:i + bs]))
        yy = y[i:i + bs]
        ce += float(F.cross_entropy(logits.float(), yy, reduction="sum"))
        correct += int((logits.argmax(1) == yy).sum())
    n = int(x.shape[0])
    return {"n": n, "ce": ce / n, "acc": correct / n}


class Endpoint:
    def __init__(self, ckpt_path, dataset, device, config_path=None):
        self.path = str(ckpt_path)
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if raw.get("schema") == "landscape_v2.analysis_checkpoint/1":
            # slim analysis checkpoint of a reused landscape_v2 run: tensors + identity only;
            # the run's own config.json supplies the recipe (its hash is checked by the caller)
            import json
            cfg = json.loads(open(config_path, encoding="utf-8").read())
            raw = {**raw, "config": cfg, "data_position": {"updates_per_epoch": raw["updates_per_epoch"]}}
        else:
            raw = load_checkpoint(ckpt_path, map_location="cpu")
        self.ck = raw
        self.cfg = ExperimentConfig.from_dict(self.ck["config"])
        self.method = self.cfg.method_spec()
        self.device = torch.device(device)
        self.pipeline = build_pipeline(dataset, self.cfg.data).to(self.device)
        tr, te = dataset.train.to(self.device), dataset.test.to(self.device)
        self.train_x, self.train_y, self.test_x, self.test_y = tr.images, tr.labels, te.images, te.labels
        self.model = build_model(self.cfg.model.arch, dataset.num_classes).to(self.device)
        self.ctrl = InterventionController(self.method, site_map(self.cfg.model.arch), self.cfg.model.arch)
        upe = int(self.ck["data_position"]["updates_per_epoch"])
        self.ctrl.configure(upe, upe * int(self.cfg.budget.epochs), int(self.cfg.run.seed))
        self.ctrl.attach(self.model)
        self.learned_sha = _sha({k: v for k, v in self.ck["model_state"].items()
                                 if "running" not in k and "num_batches" not in k})
        self.calibration_order_sha = hashlib.sha256(np.arange(int(self.train_x.shape[0]), dtype=np.int64).tobytes()).hexdigest()

    def _reload(self):
        self.model.load_state_dict({k: v.to(self.device) for k, v in self.ck["model_state"].items()}, strict=True)
        for bn in _bns(self.model):
            bn.momentum = 0.1

    def _check_learned(self):
        now = _sha({k: v for k, v in self.model.state_dict().items() if "running" not in k and "num_batches" not in k})
        if now != self.learned_sha:
            raise RuntimeError("learned tensors changed during evaluation of %s" % self.path)

    @torch.no_grad()
    def evaluate(self, state, policy: str) -> dict:
        t0 = time.perf_counter()
        self._reload()
        self.ctrl.set_state(state)
        rec = {"state": state.to_dict(), "policy": policy, "per_site_sigma": self.ctrl.per_site_sigma()}
        if policy == "recalibrated":
            for bn in _bns(self.model):
                bn.reset_running_stats()
                bn.momentum = None
            self.model.train()
            for i in range(0, int(self.train_x.shape[0]), CAL_BATCH):
                self.model(self.pipeline(self.train_x[i:i + CAL_BATCH]))
            rec["calibration"] = {"images": int(self.train_x.shape[0]), "order": "official file order 0..N-1",
                                  "order_sha256": self.calibration_order_sha, "batch_size": CAL_BATCH,
                                  "momentum": None, "passes": 1}
            rec["buffers"] = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                              if "running" in k or "num_batches" in k}
            rec["buffers_sha256"] = _sha(rec["buffers"])
        elif policy != "saved":
            raise ValueError(policy)
        self.model.eval()
        rec["train_full"] = _run(self.model, self.pipeline, self.train_x, self.train_y)
        rec["test_full"] = _run(self.model, self.pipeline, self.test_x, self.test_y)
        self._check_learned()
        rec["learned_tensors_unchanged"] = True
        self._reload()                                         # leave the model as loaded
        rec["seconds"] = time.perf_counter() - t0
        return rec

    def all_panels(self) -> dict:
        native, original = self.ctrl.native_state(), self.ctrl.target_state()
        same = (native.resolution, native.sigma, native.sd_point) == (original.resolution, original.sigma, original.sd_point)
        out = {"checkpoint": self.path, "method": self.method.id, "seed": int(self.cfg.run.seed),
               "optimizer": self.cfg.optimizer.name, "global_update": int(self.ck["global_update"]),
               "native_equals_original": bool(same), "learned_sha256": self.learned_sha}
        out["saved_native"] = self.evaluate(native, "saved")
        out["panelA_recalibrated_native"] = self.evaluate(native, "recalibrated")
        if same:
            out["saved_original"] = {"same_as": "saved_native"}
            out["panelB_recalibrated_original"] = {"same_as": "panelA_recalibrated_native"}
        else:
            out["saved_original"] = self.evaluate(original, "saved")
            out["panelB_recalibrated_original"] = self.evaluate(original, "recalibrated")
        # a fresh evaluation after all others must reproduce the first (no contamination)
        again = self.evaluate(native, "saved")
        out["repeat_saved_native_identical"] = (again["test_full"] == out["saved_native"]["test_full"]
                                                and again["train_full"] == out["saved_native"]["train_full"])
        return out

    def recorded_check(self, summary: dict) -> dict:
        """Saved-statistics test accuracy/CE (batch 500) against the run's own final record."""
        from continuation_core.evaluate import evaluate
        self._reload()
        path = "current" if self.method.cbs or self.method.sdpoint else "target"
        state = self.ctrl.native_state()
        r = evaluate(self.model, self.ctrl, self.pipeline, self.test_x, self.test_y, state,
                     bn_policy="running_stats", batch_size=500, split="test")
        rec = summary["final"][path]["test"]
        return {"recorded_path": path, "recorded_acc": rec["acc"], "recorded_ce": rec["ce"],
                "recomputed_acc": r["acc"], "recomputed_ce": r["ce"],
                "abs_diff_acc": abs(r["acc"] - rec["acc"]), "abs_diff_ce": abs(r["ce"] - rec["ce"]),
                "matches": abs(r["acc"] - rec["acc"]) <= 2e-4 and abs(r["ce"] - rec["ce"]) <= 1e-4}
