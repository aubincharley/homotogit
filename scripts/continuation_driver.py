r"""Shared driver for the plain / Gaussian / db2 continuation studies.

One implementation; the phases differ only by configuration.  Job scripts build a
list of run specs and call :func:`run_study`.  Nothing here is specific to a
phase, so the LR diagnostic, the plain study, the Gaussian study and the
prepared db2 study all execute identical code.

Filtering (when enabled) acts at the 19 validated insertion points: after each
spatial 3x3 convolution, before normalization/activation; shortcuts untouched.
Network inputs are never filtered and no augmentation is used.

Continuation schedule, with ``cont_end`` the update at which filtering stops:

    r_k = max(1 - k/cont_end, 0)
    gaussian: sigma_k = r_k              (exact bypass once r_k == 0)
    db2:      s_k = 1 - (1 - s0) r_k     (exact bypass once s_k == 1)

The parameter is fixed across all microbatches of an optimizer update.
Optimizer state and the LR schedule are never reset at the transition.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from continuation.config import DataConfig, ModelConfig, OptimConfig
from continuation.data import build_dataset, BatchIndexStream, fixed_subset_indices
from continuation.seeding import numpy_generator
from continuation.models import build_model, count_parameters
from continuation.optim import build_optimizer, lr_at, set_lr
from continuation.pipeline import (ChannelNormalizer, InputPipeline,
                                   resize_unit_float)
from continuation.transforms.gaussian import GaussianSmoothing
from continuation.transforms.wavelet import wavelet_shrink

WORK = Path(os.environ.get("STUDY_OUT", "/kaggle/working"))
# optional job-supplied hook, run after build_shared; raise to abort the study
VERIFY_SHARED = None
T0 = time.perf_counter()


def log(msg):
    print("[%7.1fs] %s" % (time.perf_counter() - T0, msg), flush=True)


def data_source():
    """CIFAR-10 from an attached Kaggle dataset (kernels have no internet).

    ``download=False`` makes torchvision verify the official md5 of every batch
    file, so a tampered copy raises rather than loading silently.
    """
    import glob
    override = os.environ.get("STUDY_DATA")
    if override and (Path(override) / "cifar-10-batches-py").exists():
        return override, False
    for hit in sorted(glob.glob("/kaggle/input/**/cifar-10-batches-py", recursive=True)):
        return str(Path(hit).parent), False
    return str(WORK / "data"), True


# --------------------------------------------------------------------------

class Controller:
    """Applies the scheduled filter at each insertion point."""

    def __init__(self, kind, s0=1.0, cont_end=1, piecewise=None, epoch_sigmas=None):
        self.kind, self.s0, self.cont_end = kind, float(s0), int(cont_end)
        # piecewise: [[k_start, sigma], ...] ascending.  Sigma is held constant
        # within each plateau -- no interpolation, no extra decay.
        self.piecewise = ([(int(a), float(b)) for a, b in piecewise]
                          if piecewise else None)
        # epoch_sigmas[e] is the sigma held constant throughout epoch e
        self.epoch_sigmas = [float(v) for v in epoch_sigmas] if epoch_sigmas else None
        self.value = None
        self.bypass_all = False
        self.gauss = GaussianSmoothing(sigma_max=1.0, truncate=4.0) if kind == "gaussian" else None

    def sigma_at(self, k):
        """Value of the continuation parameter during update ``k``.

        With a piecewise table the value is held flat inside each plateau; with
        no table the legacy linear ramp is used.  The table is used for both
        kinds -- for ``gaussian`` it lists sigma, for ``db2`` it lists ``s``.
        """
        if self.piecewise is None:
            return max(1.0 - k / float(self.cont_end), 0.0)
        v = self.piecewise[0][1]
        for start, sig in self.piecewise:
            if k >= start:
                v = sig
            else:
                break
        return v

    def is_active(self, value):
        """True when the operator is not the exact identity at ``value``."""
        if self.kind == "none" or value is None:
            return False
        if self.kind == "gaussian":
            return float(value) > 0.0
        return float(value) < 1.0            # db2: s == 1 bypasses exactly

    def set_epoch(self, e):
        """Sigma is fixed for the whole epoch (set before the epoch starts)."""
        if self.kind != "gaussian" or self.epoch_sigmas is None:
            self.value = None if self.kind == "none" else self.value
            return self.value
        e = max(0, min(int(e), len(self.epoch_sigmas) - 1))
        self.value = self.epoch_sigmas[e]
        return self.value

    def set_update(self, k):
        if self.kind == "gaussian":
            self.value = self.sigma_at(k)
        elif self.kind == "db2":
            if self.piecewise is not None:
                self.value = self.sigma_at(k)          # table lists s directly
            else:
                r = max(1.0 - k / float(self.cont_end), 0.0)
                self.value = 1.0 - (1.0 - self.s0) * r
        else:
            self.value = None
        return self.value

    def __call__(self, t):
        if self.bypass_all or self.kind == "none":
            return t
        if self.kind == "gaussian":
            return self.gauss(t, self.value)          # sigma == 0 -> exact identity
        return wavelet_shrink(t, self.value, "db2", impl="fast", bypass_identity=True)

    def describe(self):
        d = {"kind": self.kind, "s0": self.s0, "cont_end": self.cont_end,
             "piecewise": self.piecewise, "epoch_sigmas": self.epoch_sigmas}
        if self.kind == "db2":
            d["wavelet"] = {"family": "db2", "levels": 2, "impl": "fast",
                            "bypass_identity": True,
                            "threshold": "lambda_{j,o} = 4 (1-s) 2^(1-j) nu_{j,o}",
                            "nu": "sqrt(mean(d^2) + 1e-12), per sample/channel/band, "
                                  "from unthresholded coefficients, not detached"}
        if self.gauss:
            d["gaussian_kernel"] = {"kernel_size": self.gauss.kernel_size,
                                    "radius": self.gauss.radius,
                                    "sigma_max": self.gauss.sigma_max}
        return d


def attach(model, controller):
    return [m.register_forward_hook(lambda _m, _i, out, c=controller: c(out))
            for m in model.modules()
            if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3)]


# --------------------------------------------------------------------------

def build_shared(cfg, out_dir):
    """Subset (superset-compatible), per-seed init weights and batch orders."""
    root, dl = data_source()
    log("CIFAR-10 root=%s download=%s num_val=%s" % (root, dl, cfg.get("num_val", 5000)))
    bundle = build_dataset(DataConfig(root=root, download=dl,
                                      num_val=cfg.get("num_val", 5000)))
    labels = bundle.train.labels.numpy()

    # Per class, take the first `per_class` of a permutation drawn from a fixed
    # seed-0 stream.  Taking 1000 yields a superset of the pilot's 500.
    rng = np.random.default_rng(cfg["subset_seed"])
    picks = [rng.permutation(np.flatnonzero(labels == c))[:cfg["per_class"]]
             for c in range(bundle.num_classes)]
    subset = np.sort(np.concatenate(picks))
    assert subset.size == cfg["n_subset"], (subset.size, cfg["n_subset"])
    counts = np.bincount(labels[subset], minlength=bundle.num_classes).tolist()
    assert counts == [cfg["per_class"]] * bundle.num_classes, counts

    inits, orders = {}, {}
    for seed in cfg["seeds"]:
        m = build_model(ModelConfig(arch=cfg.get("arch", "resnet20_gn")),
                        bundle.num_classes, seed=seed)
        torch.save(m.state_dict(), out_dir / ("init_seed%d.pt" % seed))
        inits[seed] = True
        if cfg.get("epochs"):
            # one permutation per epoch; batches (including a partial final one)
            # are sliced from it at train time
            g = numpy_generator(seed, "batch")
            orders[seed] = np.stack([g.permutation(cfg["n_subset"]).astype(np.int32)
                                     for _ in range(int(cfg["epochs"]))])
        else:
            stream = BatchIndexStream(cfg["n_subset"], cfg["effective_batch"], seed=seed)
            orders[seed] = np.stack([stream.next_indices() for _ in range(cfg["updates"])])

    train_probe = fixed_subset_indices(cfg["n_subset"], cfg["train_probe"],
                                       cfg["probe_seed"], "study_train_probe",
                                       labels=labels[subset])
    key = "perm_seed%d" if cfg.get("epochs") else "order_seed%d"
    np.savez_compressed(out_dir / "shared_indices.npz", subset=subset,
                        train_probe=train_probe,
                        **{key % s: orders[s] for s in cfg["seeds"]})
    log("subset %d (%d/class), probe %d, seeds %s, params %s"
        % (subset.size, cfg["per_class"], train_probe.size, cfg["seeds"],
           count_parameters(build_model(ModelConfig(arch=cfg.get("arch", "resnet20_gn")),
                                        10, 0))["total"]))
    return bundle


# Progressive resolution lives in :class:`InputPipeline`, applied to the float
# [0,1] image before normalization; ``resize_unit_float`` is re-exported here so
# the checks and the timing probe use exactly the training path.
resize_to = resize_unit_float


@torch.no_grad()
def evaluate(model, controller, images, labels, pipe, bypass, batch=500, res=None):
    was, mode = controller.bypass_all, model.training
    controller.bypass_all = bypass
    model.eval()
    tot, correct, n = 0.0, 0, images.shape[0]
    for i in range(0, n, batch):
        logits = model(pipe(images[i:i + batch], 0.0, res=res))
        tot += float(F.cross_entropy(logits, labels[i:i + batch], reduction="sum"))
        correct += int((logits.argmax(1) == labels[i:i + batch]).sum())
    if mode:
        model.train()
    controller.bypass_all = was
    return tot / n, correct / n


def train_run(spec, cfg, gpu_index, out_dir):
    dev = torch.device("cuda:%d" % gpu_index if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    label = spec["label"]
    run_dir = out_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)

    sh = np.load(out_dir / "shared_indices.npz")
    subset, tp_idx = sh["subset"], sh["train_probe"]
    order = sh["order_seed%d" % spec["seed"]]

    root, _ = data_source()
    bundle = build_dataset(DataConfig(root=root, download=False)).to(dev)
    tr_imgs = bundle.train.images[torch.as_tensor(subset, device=dev)]
    tr_lbls = bundle.train.labels[torch.as_tensor(subset, device=dev)]
    tp = torch.as_tensor(tp_idx, device=dev)
    probe_imgs, probe_lbls = tr_imgs[tp], tr_lbls[tp]
    val_imgs, val_lbls = bundle.val.images, bundle.val.labels      # all 5,000

    pipe = InputPipeline(GaussianSmoothing(sigma_max=1.0),
                         ChannelNormalizer(bundle.mean, bundle.std).to(dev))
    model = build_model(ModelConfig(arch=cfg.get("arch", "resnet20_gn")),
                        bundle.num_classes, seed=spec["seed"]).to(dev)
    model.load_state_dict(torch.load(out_dir / ("init_seed%d.pt" % spec["seed"]),
                                     map_location=dev, weights_only=True))
    model.train()
    ctrl = Controller(spec["kind"], spec.get("s0", 1.0), cfg["cont_end"],
                      piecewise=cfg.get("sigma_piecewise"))
    handles = attach(model, ctrl)

    ocfg = OptimConfig(lr=spec["lr"], momentum=0.9, weight_decay=5e-4,
                       batch_size=cfg["effective_batch"], total_steps=cfg["updates"],
                       lr_schedule="cosine", warmup_steps=cfg["warmup"], min_lr=0.0)
    opt = build_optimizer(model, ocfg)
    accum = cfg["effective_batch"] // cfg["microbatch"]
    ckpt_updates = set(cfg.get("checkpoint_updates", []))
    metrics, t_start = [], time.perf_counter()

    def snapshot(k):
        # Active-filter metrics use the sigma of the most recently *completed*
        # update -- the setting the current weights were trained under.
        sigma_eval = ctrl.set_update(max(k - 1, 0))
        active = ctrl.is_active(sigma_eval)
        cur, cur_acc = (evaluate(model, ctrl, probe_imgs, probe_lbls, pipe, bypass=False)
                        if active else (None, None))
        byp, byp_acc = evaluate(model, ctrl, probe_imgs, probe_lbls, pipe, bypass=True)
        vce, vacc = evaluate(model, ctrl, val_imgs, val_lbls, pipe, bypass=True)
        vce_f, vacc_f = (evaluate(model, ctrl, val_imgs, val_lbls, pipe, bypass=False)
                         if active else (None, None))
        rec = {"update": k, "lr": lr_at(min(k, cfg["updates"] - 1), ocfg),
               "param": ctrl.value, "sigma_eval": sigma_eval,
               "filters_active_at_eval": active,
               "elapsed_s": time.perf_counter() - t_start,
               "train_probe_ce_filtered": cur, "train_probe_acc_filtered": cur_acc,
               "train_probe_ce_bypassed": byp, "train_probe_acc_bypassed": byp_acc,
               "val_ce_filtered": vce_f, "val_acc_filtered": vacc_f,
               "val_ce": vce, "val_acc": vacc}
        metrics.append(rec)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        log("%-22s k=%4d lr=%.5f sig=%-5s trCE(f)=%-8s trCE(b)=%.4f valCE=%.4f valAcc=%.4f"
            % (label, k, rec["lr"],
               "-" if sigma_eval is None else "%.2f" % sigma_eval,
               "-" if cur is None else "%.4f" % cur, byp, vce, vacc))

    ctrl.set_update(0)
    snapshot(0)
    for k in range(cfg["updates"]):
        ctrl.set_update(k)                       # fixed across this update's microbatches
        set_lr(opt, lr_at(k, ocfg))
        opt.zero_grad(set_to_none=True)          # zeroed once per complete batch
        idx = torch.as_tensor(order[k], device=dev)
        total = int(idx.numel())
        for a in range(accum):
            sl = idx[a * cfg["microbatch"]:(a + 1) * cfg["microbatch"]]
            n_m = int(sl.numel())
            if n_m == 0:
                continue
            # weight by actual example count: g = sum_m (n_m / B) grad CE_m
            loss = F.cross_entropy(model(pipe(tr_imgs[sl], 0.0)), tr_lbls[sl])
            (loss * (n_m / total)).backward()
        opt.step()                               # one step per complete batch
        done = k + 1
        if done % cfg["eval_every"] == 0 or done in ckpt_updates:
            snapshot(done)
            ctrl.set_update(done)
        if done in ckpt_updates:
            torch.save({"model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict(), "update": done,
                        "spec": spec}, run_dir / ("checkpoint_u%04d.pt" % done))

    summary = {**{k: spec[k] for k in spec},
               "updates": cfg["updates"], "warmup": cfg["warmup"],
               "effective_batch": cfg["effective_batch"],
               "microbatch": cfg["microbatch"], "accumulation": accum,
               "controller": ctrl.describe(),
               "final_train_probe_ce_bypassed": metrics[-1]["train_probe_ce_bypassed"],
               "final_train_probe_acc_bypassed": metrics[-1]["train_probe_acc_bypassed"],
               "final_val_ce": metrics[-1]["val_ce"],
               "final_val_acc": metrics[-1]["val_acc"],
               "n_val": int(val_imgs.shape[0]),
               "wall_seconds": time.perf_counter() - t_start,
               "peak_mem_mib": (torch.cuda.max_memory_allocated(dev) / 1024 ** 2
                                if dev.type == "cuda" else None),
               "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"}
    for h in handles:
        h.remove()
    torch.save({"model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
                "update": cfg["updates"], "spec": spec}, run_dir / "checkpoint_final.pt")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log("%-22s DONE valAcc=%.4f valCE=%.4f trCE(b)=%.4f  %.0fs  peak %.0f MiB"
        % (label, summary["final_val_acc"], summary["final_val_ce"],
           summary["final_train_probe_ce_bypassed"], summary["wall_seconds"],
           summary["peak_mem_mib"] or 0))
    return summary



def train_run_epochs(spec, cfg, gpu_index, out_dir):
    """Epoch-based training on the full split, sigma held constant per epoch.

    Keeps partial accumulation groups: the final batch of an epoch may be
    smaller than the effective batch, and every microbatch is weighted by its
    actual example count, ``g = sum_m (n_m / B) grad CE_m``.
    """
    dev = torch.device("cuda:%d" % gpu_index if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    label = spec["label"]
    run_dir = out_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)

    sh = np.load(out_dir / "shared_indices.npz")
    subset, tp_idx = sh["subset"], sh["train_probe"]
    perms = sh["perm_seed%d" % spec["seed"]]                 # [epochs, n]

    root, _ = data_source()
    bundle = build_dataset(DataConfig(root=root, download=False,
                                      num_val=cfg.get("num_val", 5000))).to(dev)
    tr_imgs = bundle.train.images[torch.as_tensor(subset, device=dev)]
    tr_lbls = bundle.train.labels[torch.as_tensor(subset, device=dev)]
    tp = torch.as_tensor(tp_idx, device=dev)
    probe_imgs, probe_lbls = tr_imgs[tp], tr_lbls[tp]
    held = bundle.test if cfg.get("eval_split", "val") == "test" else bundle.val
    held_imgs, held_lbls = held.images, held.labels

    pipe = InputPipeline(GaussianSmoothing(sigma_max=1.0),
                         ChannelNormalizer(bundle.mean, bundle.std).to(dev))
    model = build_model(ModelConfig(arch=cfg.get("arch", "resnet20_gn")),
                        bundle.num_classes, seed=spec["seed"]).to(dev)
    model.load_state_dict(torch.load(out_dir / ("init_seed%d.pt" % spec["seed"]),
                                     map_location=dev, weights_only=True))
    model.train()
    ctrl = Controller(spec["kind"], epoch_sigmas=spec.get("sigma_by_epoch"))
    handles = attach(model, ctrl)

    epochs = int(cfg["epochs"])
    n = int(subset.size)
    B, mb = cfg["effective_batch"], cfg["microbatch"]
    per_epoch = (n + B - 1) // B                            # keeps the partial batch
    total_updates = epochs * per_epoch
    ocfg = OptimConfig(lr=spec["lr"], momentum=0.9, weight_decay=5e-4,
                       batch_size=B, total_steps=total_updates,
                       lr_schedule="cosine", warmup_steps=cfg["warmup"], min_lr=0.0)
    opt = build_optimizer(model, ocfg)

    eval_epochs = sorted(set(list(range(0, epochs + 1, cfg.get("eval_every_epochs", 2)))
                             + list(cfg.get("eval_extra_epochs", [])) + [epochs]))
    ckpt_epochs = set(cfg.get("checkpoint_epochs", [epochs]))
    # per-epoch input resolution; None means "no progressive resolution"
    res_by_epoch = spec.get("resolution_by_epoch")
    if res_by_epoch is not None:
        res_by_epoch = [int(v) for v in res_by_epoch]
        if len(res_by_epoch) != epochs:
            raise ValueError("resolution_by_epoch has %d entries, need %d"
                             % (len(res_by_epoch), epochs))
    metrics, t_start, gstep = [], time.perf_counter(), 0
    eval_overhead = [0.0]                    # cumulative seconds spent in snapshots

    def snapshot(done_epochs):
        """Two explicitly named evaluation paths.

        *current* (primary): the resolution and effective sigma used by the most
        recently **completed** training update -- the configuration the weights
        and BatchNorm buffers were actually trained under.
        *target* (diagnostic): the original 32x32 input with the internal filter
        bypassed exactly.  While continuation is running this measures a
        premature change of input/operator, BatchNorm mismatch included; it is
        not the quality of the current predictor.
        """
        last = max(done_epochs - 1, 0)
        sigma_eval = ctrl.epoch_sigmas[last] if ctrl.epoch_sigmas else None
        res_eval = res_by_epoch[last] if res_by_epoch else None
        nxt = min(done_epochs, epochs - 1)
        prev = ctrl.value
        ctrl.value = sigma_eval
        active = ctrl.is_active(sigma_eval)
        cur, cur_acc = evaluate(model, ctrl, probe_imgs, probe_lbls, pipe,
                                bypass=not active, res=res_eval)
        tce_f, tacc_f = evaluate(model, ctrl, held_imgs, held_lbls, pipe,
                                 bypass=not active, res=res_eval)
        byp, byp_acc = evaluate(model, ctrl, probe_imgs, probe_lbls, pipe,
                                bypass=True, res=None)
        tce, tacc = evaluate(model, ctrl, held_imgs, held_lbls, pipe,
                             bypass=True, res=None)
        ctrl.value = prev
        rec = {"epoch": done_epochs, "update": gstep,
               "lr": lr_at(min(max(gstep - 1, 0), total_updates - 1), ocfg),
               "sigma_eval": sigma_eval, "filters_active_at_eval": active,
               "resolution_eval": res_eval,
               "eval_current_path": {"resolution": res_eval if res_eval else 32,
                                     "sigma": sigma_eval,
                                     "filter_bypassed": not active},
               "eval_target_path": {"resolution": 32, "sigma": 0.0,
                                    "filter_bypassed": True},
               "next_sigma": (ctrl.epoch_sigmas[nxt] if ctrl.epoch_sigmas else None),
               "next_resolution": (res_by_epoch[nxt] if res_by_epoch else None),
               "elapsed_s": time.perf_counter() - t_start,
               "eval_overhead_s": eval_overhead[0],
               "train_probe_ce_current": cur, "train_probe_acc_current": cur_acc,
               "train_probe_ce_target": byp, "train_probe_acc_target": byp_acc,
               "test_ce_current": tce_f, "test_acc_current": tacc_f,
               "test_ce_target": tce, "test_acc_target": tacc,
               "train_probe_ce_filtered": cur, "train_probe_acc_filtered": cur_acc,
               "train_probe_ce_bypassed": byp, "train_probe_acc_bypassed": byp_acc,
               "test_ce_filtered": tce_f, "test_acc_filtered": tacc_f,
               "test_ce": tce, "test_acc": tacc}
        metrics.append(rec)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        log("%-26s ep=%2d/%d r=%-3s sig=%-5s | current CE=%.4f acc=%.4f | "
            "target CE=%.4f acc=%.4f  %.0fs"
            % (label, done_epochs, epochs,
               "-" if res_eval is None else str(res_eval),
               "-" if sigma_eval is None else "%.3f" % sigma_eval,
               tce_f, tacc_f, tce, tacc, rec["elapsed_s"]))

    def timed_snapshot(n):
        t0 = time.perf_counter()
        snapshot(n)
        eval_overhead[0] += time.perf_counter() - t0

    timed_snapshot(0)
    for e in range(epochs):
        ctrl.set_epoch(e)                        # sigma fixed for the whole epoch
        res_e = res_by_epoch[e] if res_by_epoch else None
        perm = perms[e]
        for start in range(0, n, B):
            batch = perm[start:start + B]
            total = int(batch.size)
            idx = torch.as_tensor(batch, dtype=torch.long, device=dev)
            set_lr(opt, lr_at(gstep, ocfg))
            opt.zero_grad(set_to_none=True)
            for a in range(0, total, mb):
                sl = idx[a:a + mb]
                n_m = int(sl.numel())
                # uint8 -> float[0,1] -> resize to r(e) -> normalize
                loss = F.cross_entropy(model(pipe(tr_imgs[sl], 0.0, res=res_e)),
                                       tr_lbls[sl])
                (loss * (n_m / total)).backward()
            opt.step()
            gstep += 1
        done = e + 1
        if done in eval_epochs:
            timed_snapshot(done)
            ctrl.set_epoch(e)                    # restore the training sigma
        if done in ckpt_epochs:
            torch.save({"model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict(), "epoch": done,
                        "update": gstep, "spec": spec},
                       run_dir / ("checkpoint_ep%02d.pt" % done))

    last = metrics[-1]
    summary = {**{k: spec[k] for k in spec
                  if k not in ("sigma_by_epoch", "resolution_by_epoch")},
               "sigma_by_epoch": spec.get("sigma_by_epoch"),
               "resolution_by_epoch": res_by_epoch,
               "resize": ("bilinear, align_corners=False, antialias=True; "
                          "r=32 returns the original tensor with no resize op"),
               "eval_overhead_s": eval_overhead[0],
               "epochs": epochs, "updates": gstep, "updates_per_epoch": per_epoch,
               "warmup": cfg["warmup"], "effective_batch": B, "microbatch": mb,
               "controller": ctrl.describe(),
               "final_train_probe_ce_bypassed": last["train_probe_ce_bypassed"],
               "final_test_ce": last["test_ce"], "final_test_acc": last["test_acc"],
               "n_test": int(held_imgs.shape[0]), "n_train": n,
               "wall_seconds": time.perf_counter() - t_start,
               "peak_mem_mib": (torch.cuda.max_memory_allocated(dev) / 1024 ** 2
                                if dev.type == "cuda" else None),
               "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"}
    for h in handles:
        h.remove()
    torch.save({"model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
                "epoch": epochs, "update": gstep, "spec": spec},
               run_dir / "checkpoint_final.pt")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log("%-26s DONE testAcc=%.4f testCE=%.4f  %.0fs total (%.0fs eval)  peak %.0f MiB"
        % (label, summary["final_test_acc"], summary["final_test_ce"],
           summary["wall_seconds"], summary["eval_overhead_s"],
           summary["peak_mem_mib"] or 0))
    return summary


def _worker(specs, cfg, gpu_index, out_dir):
    runner = train_run_epochs if cfg.get("epochs") else train_run
    for spec in specs:
        runner(spec, cfg, gpu_index, out_dir)


def run_study(cfg):
    run_dir = WORK / ("%s_%s" % (cfg["name"], time.strftime("%Y%m%d-%H%M%S")))
    run_dir.mkdir(parents=True, exist_ok=True)
    ngpu = torch.cuda.device_count()
    prov = {"python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "n_gpu": ngpu,
            "gpus": [torch.cuda.get_device_name(i) for i in range(ngpu)],
            "vram_mib": [round(torch.cuda.get_device_properties(i).total_memory / 1024 ** 2)
                         for i in range(ngpu)]}
    (run_dir / "environment.json").write_text(json.dumps(prov, indent=2))
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    log("environment: %s" % prov)
    if ngpu == 0:
        (run_dir / "ABORTED.json").write_text(json.dumps(
            {"aborted": True, "reason": "no accelerator granted", "environment": prov}))
        log("aborted: no accelerator granted")
        return run_dir

    build_shared(cfg, run_dir)
    if VERIFY_SHARED is not None:
        VERIFY_SHARED(run_dir)          # job-supplied pairing check; may raise
    specs = cfg["runs"]
    queues = [specs[i::max(ngpu, 1)] for i in range(max(ngpu, 1))]
    log("dispatching %d runs over %d GPU(s): %s"
        % (len(specs), ngpu, [[s["label"] for s in q] for q in queues]))
    if ngpu >= 2 and len(specs) > 1:
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=_worker, args=(q, cfg, i, run_dir))
                 for i, q in enumerate(queues) if q]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
    else:
        _worker(specs, cfg, 0, run_dir)

    summaries = []
    for spec in specs:
        f = run_dir / spec["label"] / "summary.json"
        if f.exists():
            summaries.append(json.loads(f.read_text()))
    (run_dir / "study_summary.json").write_text(json.dumps(summaries, indent=2))
    for s in summaries:
        acc = s.get("final_test_acc", s.get("final_val_acc"))
        ce = s.get("final_test_ce", s.get("final_val_ce"))
        log("SUMMARY %-26s trCE(b)=%.4f CE=%.4f acc=%.4f"
            % (s["label"], s["final_train_probe_ce_bypassed"], ce, acc))
    log("run directory: %s" % run_dir)
    return run_dir
