"""Worker-queue driver for the resolution-only benchmark.

Same shape as ``campaign_driver``: pinned assets verified by sha256 before any
update, two worker processes per job (one per T4) pulling from one longest-first
queue, per-cell rolling checkpoints, and a failure in one cell never stopping
the rest.  What differs is the operator: exactly one resolution reduction site,
no internal Gaussian anywhere.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from continuation.config import DataConfig, ModelConfig, OptimConfig
from continuation.data import build_dataset
from continuation.models import build_model
from continuation.optim import build_optimizer, lr_at, set_lr
from continuation.pipeline import ChannelNormalizer, InputPipeline
from continuation.resolution_ops import (PATHS, ResolutionController,
                                         attach_resolution, reduce_with)
from continuation.transforms.gaussian import GaussianSmoothing
from scripts.campaign_driver import (EXPECTED_STATS, PROTOCOL, data_source,
                                     find_assets, log, verify_assets)

WORK = Path(os.environ.get("STUDY_OUT", "/kaggle/working"))
T0 = time.perf_counter()
CKPT_EPOCHS = (6, 12, 18, 30)


def _reducer_for(ctrl):
    """Input-site reduction as a callable for :class:`InputPipeline`."""
    return (lambda x, r: reduce_with(x, r, ctrl.operator))


@torch.no_grad()
def evaluate(model, ctrl, images, labels, pipe, resolution, batch=500):
    """Evaluate at an explicit resolution.  BatchNorm statistics untouched."""
    prev_mode, prev_res, prev_bypass = model.training, ctrl.resolution, ctrl.bypass_all
    ctrl.bypass_all = False
    ctrl.set_state(resolution)
    model.eval()
    res_in, red = ctrl.input_resolution(), _reducer_for(ctrl)
    tot, correct, n = 0.0, 0, images.shape[0]
    for i in range(0, n, batch):
        logits = model(pipe(images[i:i + batch], 0.0, res=res_in, reducer=red))
        tot += float(F.cross_entropy(logits, labels[i:i + batch], reduction="sum"))
        correct += int((logits.argmax(1) == labels[i:i + batch]).sum())
    ctrl.resolution, ctrl.bypass_all = prev_res, prev_bypass
    if prev_mode:
        model.train()
    return tot / n, correct / n


def train_cell(cell, assets: Path, gpu: int, out_dir: Path, job: int, worker: int):
    dev = torch.device("cuda:%d" % gpu if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    run_dir = out_dir / cell["cell_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cell["seed"])

    z = np.load(assets / "shared_indices.npz")
    subset, tp_idx, perms = z["subset"], z["train_probe"], z["perm_seed%d" % seed]
    bundle = build_dataset(DataConfig(root=data_source(), download=False,
                                      num_val=0)).to(dev)
    if not (np.allclose([float(v) for v in bundle.mean], EXPECTED_STATS["mean"], atol=1e-7)
            and np.allclose([float(v) for v in bundle.std], EXPECTED_STATS["std"], atol=1e-7)):
        raise RuntimeError("normalization statistics differ from the pinned values")

    tr_imgs = bundle.train.images[torch.as_tensor(subset, device=dev)]
    tr_lbls = bundle.train.labels[torch.as_tensor(subset, device=dev)]
    tp = torch.as_tensor(tp_idx, device=dev)
    probe_imgs, probe_lbls = tr_imgs[tp], tr_lbls[tp]
    test_imgs, test_lbls = bundle.test.images, bundle.test.labels
    pipe = InputPipeline(GaussianSmoothing(sigma_max=1.0),
                         ChannelNormalizer(bundle.mean, bundle.std).to(dev))

    model = build_model(ModelConfig(arch="resnet20_bn_cifar"), 10, seed=seed).to(dev)
    model.load_state_dict(torch.load(assets / ("init_seed%d.pt" % seed),
                                     map_location=dev, weights_only=True))
    model.train()
    ctrl = ResolutionController(cell["operator"], cell["location"], cell["path"])
    handles = attach_resolution(model, ctrl)
    reducer = _reducer_for(ctrl)

    epochs = PROTOCOL["epochs"]
    n, B, mb = int(subset.size), PROTOCOL["effective_batch"], PROTOCOL["microbatch"]
    per_epoch = (n + B - 1) // B
    total_updates = epochs * per_epoch
    ocfg = OptimConfig(lr=PROTOCOL["lr"], momentum=PROTOCOL["momentum"],
                       weight_decay=PROTOCOL["weight_decay"], batch_size=B,
                       total_steps=total_updates, lr_schedule="cosine",
                       warmup_steps=PROTOCOL["warmup"], min_lr=0.0)
    opt = build_optimizer(model, ocfg)

    # evaluation cadence: every two epochs, plus every path transition
    table = ctrl.by_epoch
    transitions = [e for e in range(1, epochs) if table[e] != table[e - 1]]
    eval_epochs = sorted(set(list(range(0, epochs + 1, 2)) + transitions
                             + [t + 1 for t in transitions] + [epochs]))
    metrics, gstep, start_epoch, eval_seconds = [], 0, 0, [0.0]
    t_start = time.perf_counter()

    rolling = run_dir / "rolling.pt"
    if rolling.is_file():
        try:
            st = torch.load(rolling, map_location=dev, weights_only=False)
            model.load_state_dict(st["model_state"])
            opt.load_state_dict(st["optimizer_state"])
            gstep, start_epoch = st["update"], st["epoch"]
            metrics, eval_seconds = st.get("metrics", []), [st.get("eval_seconds", 0.0)]
            torch.set_rng_state(st["cpu_rng"])
            log("%s resuming after epoch %d" % (cell["cell_id"], start_epoch))
        except Exception as exc:
            log("%s rolling checkpoint unusable (%r); starting fresh"
                % (cell["cell_id"], exc))

    fixed = cell["path"].startswith("fixed")

    def snapshot(done_epochs):
        t0 = time.perf_counter()
        last = max(done_epochs - 1, 0)
        cur = table[last]
        pce, pacc = evaluate(model, ctrl, probe_imgs, probe_lbls, pipe, cur)
        tce, tacc = evaluate(model, ctrl, test_imgs, test_lbls, pipe, cur)
        # forced-bypass diagnostic: the plain 32x32 path, kept out of the
        # primary curves.  For a fixed control it is not that arm's endpoint.
        gp, gpa = evaluate(model, ctrl, probe_imgs, probe_lbls, pipe, 32)
        gt, gta = evaluate(model, ctrl, test_imgs, test_lbls, pipe, 32)
        rec = {"epoch": done_epochs, "update": gstep,
               "lr": lr_at(min(max(gstep - 1, 0), total_updates - 1), ocfg),
               "current": {"resolution": cur, "operator": cell["operator"],
                           "location": cell["location"], "path": cell["path"]},
               "next_scheduled": {"resolution": table[min(done_epochs, epochs - 1)]},
               "train_probe_ce_current": pce, "train_probe_acc_current": pacc,
               "test_ce_current": tce, "test_acc_current": tacc,
               "train_probe_ce_bypass32": gp, "train_probe_acc_bypass32": gpa,
               "test_ce_bypass32": gt, "test_acc_bypass32": gta,
               "elapsed_s": time.perf_counter() - t_start,
               "peak_mem_mib": (torch.cuda.max_memory_allocated(dev) / 1024 ** 2
                                if dev.type == "cuda" else None)}
        metrics.append(rec)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        eval_seconds[0] += time.perf_counter() - t0
        log("%-38s ep=%2d/%d r=%-2d | current acc=%.4f ce=%.4f | 32x32 acc=%.4f"
            % (cell["cell_id"][:38], done_epochs, epochs, cur, tacc, tce, gta))

    if start_epoch == 0:
        snapshot(0)
    for e in range(start_epoch, epochs):
        ctrl.set_epoch(e)
        res_in = ctrl.input_resolution()
        perm = perms[e]
        for s0 in range(0, n, B):
            batch = perm[s0:s0 + B]
            total = int(batch.size)
            idx = torch.as_tensor(batch, dtype=torch.long, device=dev)
            set_lr(opt, lr_at(gstep, ocfg))
            opt.zero_grad(set_to_none=True)
            for a in range(0, total, mb):
                sl = idx[a:a + mb]
                n_m = int(sl.numel())
                loss = F.cross_entropy(
                    model(pipe(tr_imgs[sl], 0.0, res=res_in, reducer=reducer)),
                    tr_lbls[sl])
                (loss * (n_m / total)).backward()
            opt.step()
            gstep += 1
        done = e + 1
        if done in eval_epochs:
            snapshot(done)
            ctrl.set_epoch(e)
        state = {"model_state": model.state_dict(),
                 "optimizer_state": opt.state_dict(), "epoch": done,
                 "update": gstep, "metrics": metrics,
                 "eval_seconds": eval_seconds[0], "cell": cell,
                 "cpu_rng": torch.get_rng_state()}
        torch.save(state, rolling)
        if done in CKPT_EPOCHS:
            torch.save(state, run_dir / ("checkpoint_ep%02d.pt" % done))

    last = metrics[-1]
    wall = time.perf_counter() - t_start
    # Primary endpoint: progressive arms at full resolution with the operator
    # bypassed; fixed controls on their own reduced-resolution path.
    final_acc = last["test_acc_current"] if fixed else last["test_acc_bypass32"]
    final_ce = last["test_ce_current"] if fixed else last["test_ce_bypass32"]
    summary = {**cell, "job": job, "worker": worker, "gpu_index": gpu,
               "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
               "epochs": epochs, "updates": gstep, "updates_per_epoch": per_epoch,
               "n_train": n, "n_test": int(test_imgs.shape[0]),
               "controller": ctrl.describe(),
               "endpoint": ("fixed reduced-resolution path" if fixed
                            else "32x32, operator bypassed"),
               "final_test_acc": final_acc, "final_test_ce": final_ce,
               "final_train_probe_ce": (last["train_probe_ce_current"] if fixed
                                        else last["train_probe_ce_bypass32"]),
               "wall_seconds": wall, "eval_seconds": eval_seconds[0],
               "train_seconds": wall - eval_seconds[0],
               "peak_mem_mib": (torch.cuda.max_memory_allocated(dev) / 1024 ** 2
                                if dev.type == "cuda" else None)}
    for h in handles:
        h.remove()
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if rolling.is_file():
        rolling.unlink()
    log("%-38s DONE acc=%.4f ce=%.4f  %.0fs (%.0fs eval)"
        % (cell["cell_id"][:38], final_acc, final_ce, wall, eval_seconds[0]))
    return summary


def _worker(queue, by_id, assets, out_dir, job, worker, gpu, counter, total):
    while True:
        try:
            cid = queue.get_nowait()
        except Exception:
            return
        try:
            train_cell(by_id[cid], Path(assets), gpu, Path(out_dir), job, worker)
        except Exception:
            tb = traceback.format_exc()
            d = Path(out_dir) / cid
            d.mkdir(parents=True, exist_ok=True)
            (d / "FAILED.json").write_text(json.dumps(
                {"cell": by_id[cid], "job": job, "worker": worker,
                 "traceback": tb}, indent=2))
            log("FAILED %s\n%s" % (cid, tb))
        with counter.get_lock():
            counter.value += 1
            k = counter.value
        el = time.perf_counter() - T0
        log("progress: %d/%d done, %d pending, elapsed %.0fs, ETA %.0fs"
            % (k, total, total - k, el, el / max(k, 1) * (total - k)))


def run_job(job_index: int, cells: list, extra: dict | None = None):
    out_dir = WORK / ("resbench_job%d_%s" % (job_index, time.strftime("%Y%m%d-%H%M%S")))
    out_dir.mkdir(parents=True, exist_ok=True)
    ngpu = torch.cuda.device_count()
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "torch": torch.__version__, "cuda": torch.version.cuda, "n_gpu": ngpu,
           "gpus": [torch.cuda.get_device_name(i) for i in range(ngpu)]}
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2))
    (out_dir / "job_manifest.json").write_text(json.dumps(
        {"job": job_index, "n_cells": len(cells), "cells": cells,
         "paths": PATHS, **(extra or {})}, indent=2))
    log("job %d environment: %s" % (job_index, env))
    if ngpu == 0:
        (out_dir / "ABORTED.json").write_text(json.dumps(
            {"reason": "no accelerator granted", "environment": env}))
        raise SystemExit("no accelerator granted")

    assets = find_assets()
    verify_assets(assets, out_dir)

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    for c in cells:
        q.put(c["cell_id"])
    counter = ctx.Value("i", 0)
    by_id = {c["cell_id"]: c for c in cells}
    n_workers = min(ngpu, 2)
    log("job %d: %d cells over %d worker(s)" % (job_index, len(cells), n_workers))
    procs = [ctx.Process(target=_worker,
                         args=(q, by_id, str(assets), str(out_dir), job_index,
                               w, w, counter, len(cells)))
             for w in range(n_workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    done, failed = [], []
    for c in cells:
        d = out_dir / c["cell_id"]
        if (d / "summary.json").is_file():
            done.append(json.loads((d / "summary.json").read_text()))
        elif (d / "FAILED.json").is_file():
            failed.append(json.loads((d / "FAILED.json").read_text())["cell"])
        else:
            failed.append({**c, "reason": "incomplete"})
    (out_dir / "job_summary.json").write_text(json.dumps(
        {"job": job_index, "completed": done, "failed_or_incomplete": failed,
         "elapsed_s": time.perf_counter() - T0}, indent=2))
    log("job %d finished: %d complete, %d failed/incomplete, %.0fs"
        % (job_index, len(done), len(failed), time.perf_counter() - T0))
    return out_dir
