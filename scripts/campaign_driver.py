"""Worker-queue driver for the 21-configuration CIFAR-10 campaign.

One job runs on one Kaggle environment with two T4s.  Two independent worker
processes -- one per GPU -- pull from a single shared queue of pending runs,
ordered longest-first, so neither GPU idles while the other finishes a long run.
There is no distributed training of an individual model.

Shared state (initial weights and BN buffers, per-epoch permutations, subset and
probe indices) is **pinned**: it comes from an attached Kaggle dataset, never
regenerated, and every hash is re-checked here before a single update runs.  A
mismatch aborts the job rather than producing unpaired results.

Every run writes its configuration, seed, job and worker identity into its own
summary, keeps a rolling resumable checkpoint plus fixed checkpoints after
completed epochs 6/12/21/30, and records training time, evaluation time and peak
memory separately.  A failure in one run is recorded and does not stop the rest.
"""
from __future__ import annotations

import hashlib
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

from continuation.campaign_ops import (EARLY7, N_SITES, SiteController,
                                       attach_sites, reduce_spatial)
from continuation.config import DataConfig, ModelConfig, OptimConfig
from continuation.data import build_dataset
from continuation.models import build_model
from continuation.optim import build_optimizer, lr_at, set_lr
from continuation.pipeline import ChannelNormalizer, InputPipeline

WORK = Path(os.environ.get("STUDY_OUT", "/kaggle/working"))
T0 = time.perf_counter()

EXPECTED_STATS = {"mean": [0.4913996756076813, 0.4821584224700928,
                           0.44653090834617615],
                  "std": [0.24703222513198853, 0.24348512291908264,
                          0.26158782839775085]}
PROTOCOL = {"epochs": 30, "warmup": 60, "effective_batch": 128, "microbatch": 32,
            "lr": 0.005, "momentum": 0.9, "weight_decay": 5e-4}


def log(msg):
    print("[%7.1fs] %s" % (time.perf_counter() - T0, msg), flush=True)


# --------------------------------------------------------------------------
# pinned assets
# --------------------------------------------------------------------------

def find_assets() -> Path:
    override = os.environ.get("CAMPAIGN_ASSETS")
    if override and (Path(override) / "assets_manifest.json").is_file():
        return Path(override)
    import glob
    for hit in sorted(glob.glob("/kaggle/input/**/assets_manifest.json",
                                recursive=True)):
        return Path(hit).parent
    raise SystemExit("pinned campaign assets not found; attach the dataset")


def _sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _sha_state(p):
    sd = torch.load(p, map_location="cpu", weights_only=True)
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(np.ascontiguousarray(sd[k].cpu().numpy()).tobytes())
    return h.hexdigest()


def verify_assets(assets: Path, out_dir: Path) -> dict:
    man = json.loads((assets / "assets_manifest.json").read_text())
    z = np.load(assets / "shared_indices.npz")
    checks = {}
    for k, v in man["arrays"].items():
        checks["array:" + k] = _sha_array(z[k]) == v["sha256"]
    for k, v in man["states"].items():
        checks["state:" + k] = _sha_state(assets / (k + ".pt")) == v
    result = {"assets_dir": str(assets), "manifest": man, "checks": checks,
              "all_match": all(checks.values())}
    (out_dir / "assets_verification.json").write_text(json.dumps(result, indent=2))
    log("pinned-asset verification: %s" % ("OK" if result["all_match"] else checks))
    if not result["all_match"]:
        raise SystemExit("pinned assets failed verification: %s" % checks)
    return result


def data_source():
    import glob
    override = os.environ.get("STUDY_DATA")
    if override and (Path(override) / "cifar-10-batches-py").exists():
        return override
    for hit in sorted(glob.glob("/kaggle/input/**/cifar-10-batches-py",
                                recursive=True)):
        return str(Path(hit).parent)
    raise SystemExit("CIFAR-10 not found; attach the dataset")


# --------------------------------------------------------------------------
# one training run
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, ctrl, images, labels, pipe, level, resolution, batch=500):
    """Evaluate at an explicit (level, resolution).  BN statistics untouched."""
    prev_mode = model.training
    prev = (ctrl.value, ctrl.resolution, ctrl.bypass_all)
    ctrl.bypass_all = False
    ctrl.set_state(level, resolution)
    if level is None or float(level or 0.0) == 0.0:
        ctrl.bypass_all = True
    model.eval()
    res_in, tot, correct = ctrl.input_resolution(), 0.0, 0
    n = images.shape[0]
    for i in range(0, n, batch):
        logits = model(pipe(images[i:i + batch], 0.0, res=res_in))
        tot += float(F.cross_entropy(logits, labels[i:i + batch], reduction="sum"))
        correct += int((logits.argmax(1) == labels[i:i + batch]).sum())
    ctrl.value, ctrl.resolution, ctrl.bypass_all = prev
    ctrl.q = ctrl.q_for(ctrl.resolution)
    if prev_mode:
        model.train()
    return tot / n, correct / n


def build_controller(model, cell):
    """Cell -> ``(controller, hook handles)``.

    A cell may name its own builder in ``cell["controller_builder"]`` as a
    ``module.function`` path; the anti-aliasing ablation uses this to install its
    own placements, masks and BlurPool without this driver knowing anything about
    them.  Dispatching on a **cell field** rather than on a module-level global is
    deliberate: workers are spawned, so a global set in the parent would not
    survive into the child, whereas the cell dict is pickled onto the queue.

    Cells with no builder take the original campaign path, unchanged.
    """
    builder = cell.get("controller_builder")
    if builder:
        import importlib
        mod_name, fn_name = builder.rsplit(".", 1)
        return getattr(importlib.import_module(mod_name), fn_name)(model, cell)

    from scripts.campaign_manifest import GAUSSIAN, RESOLUTIONS
    sites = EARLY7 if cell["mask"] == "early7" else tuple(range(N_SITES))
    ctrl = SiteController(operator=cell["operator"],
                          levels=GAUSSIAN[cell["gaussian"]], sites=sites,
                          resolution_by_epoch=RESOLUTIONS[cell["resolution"]],
                          reduction=cell["reduction"])
    return ctrl, attach_sites(model, ctrl)


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
    stats_ok = (np.allclose([float(v) for v in bundle.mean], EXPECTED_STATS["mean"],
                            atol=1e-7)
                and np.allclose([float(v) for v in bundle.std],
                                EXPECTED_STATS["std"], atol=1e-7))
    if not stats_ok:
        raise RuntimeError("normalization statistics differ from the pinned "
                           "full-training-set values")

    tr_imgs = bundle.train.images[torch.as_tensor(subset, device=dev)]
    tr_lbls = bundle.train.labels[torch.as_tensor(subset, device=dev)]
    tp = torch.as_tensor(tp_idx, device=dev)
    probe_imgs, probe_lbls = tr_imgs[tp], tr_lbls[tp]
    test_imgs, test_lbls = bundle.test.images, bundle.test.labels

    from continuation.transforms.gaussian import GaussianSmoothing
    pipe = InputPipeline(GaussianSmoothing(sigma_max=1.0),
                         ChannelNormalizer(bundle.mean, bundle.std).to(dev))

    model = build_model(ModelConfig(arch="resnet20_bn_cifar"), 10, seed=seed).to(dev)
    model.load_state_dict(torch.load(assets / ("init_seed%d.pt" % seed),
                                     map_location=dev, weights_only=True))
    model.train()

    ctrl, handles = build_controller(model, cell)

    epochs = PROTOCOL["epochs"]
    n, B, mb = int(subset.size), PROTOCOL["effective_batch"], PROTOCOL["microbatch"]
    per_epoch = (n + B - 1) // B
    total_updates = epochs * per_epoch
    ocfg = OptimConfig(lr=PROTOCOL["lr"], momentum=PROTOCOL["momentum"],
                       weight_decay=PROTOCOL["weight_decay"], batch_size=B,
                       total_steps=total_updates, lr_schedule="cosine",
                       warmup_steps=PROTOCOL["warmup"], min_lr=0.0)
    opt = build_optimizer(model, ocfg)

    eval_epochs = sorted(set(list(range(0, epochs + 1, 2)) + [21, epochs]))
    ckpt_epochs = {6, 12, 21, epochs}
    metrics, gstep, start_epoch = [], 0, 0
    eval_seconds = [0.0]
    t_start = time.perf_counter()

    # resume from the rolling checkpoint rather than duplicating work
    rolling = run_dir / "rolling.pt"
    if rolling.is_file():
        try:
            st = torch.load(rolling, map_location=dev, weights_only=False)
            model.load_state_dict(st["model_state"])
            opt.load_state_dict(st["optimizer_state"])
            gstep, start_epoch = st["update"], st["epoch"]
            metrics = st.get("metrics", [])
            eval_seconds = [st.get("eval_seconds", 0.0)]
            torch.set_rng_state(st["cpu_rng"])
            log("%s resuming after epoch %d" % (cell["cell_id"], start_epoch))
        except Exception as exc:                       # corrupt -> start clean
            log("%s rolling checkpoint unusable (%r); starting fresh"
                % (cell["cell_id"], exc))

    def snapshot(done_epochs):
        t0 = time.perf_counter()
        last = max(done_epochs - 1, 0)
        nxt = min(done_epochs, epochs - 1)
        cur_level = ctrl.levels[last] if ctrl.levels else None
        cur_res = ctrl.resolution_by_epoch[last]
        pce, pacc = evaluate(model, ctrl, probe_imgs, probe_lbls, pipe,
                             cur_level, cur_res)
        tce, tacc = evaluate(model, ctrl, test_imgs, test_lbls, pipe,
                             cur_level, cur_res)
        # target path: original 32x32, every filter and stem reduction bypassed
        gpce, gpacc = evaluate(model, ctrl, probe_imgs, probe_lbls, pipe, 0.0, 32)
        gtce, gtacc = evaluate(model, ctrl, test_imgs, test_lbls, pipe, 0.0, 32)
        rec = {"epoch": done_epochs, "update": gstep,
               "lr": lr_at(min(max(gstep - 1, 0), total_updates - 1), ocfg),
               "current": {"level": cur_level, "resolution": cur_res,
                           "operator": cell["operator"], "mask": cell["mask"],
                           "reduction": cell["reduction"],
                           "n_active_sites": len(ctrl.sites) if cur_level else 0},
               "next_scheduled": {"level": ctrl.levels[nxt] if ctrl.levels else None,
                                  "resolution": ctrl.resolution_by_epoch[nxt]},
               "train_probe_ce_current": pce, "train_probe_acc_current": pacc,
               "test_ce_current": tce, "test_acc_current": tacc,
               "train_probe_ce_target": gpce, "train_probe_acc_target": gpacc,
               "test_ce_target": gtce, "test_acc_target": gtacc,
               "elapsed_s": time.perf_counter() - t_start,
               "peak_mem_mib": (torch.cuda.max_memory_allocated(dev) / 1024 ** 2
                                if dev.type == "cuda" else None)}
        metrics.append(rec)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        eval_seconds[0] += time.perf_counter() - t0
        log("%-46s ep=%2d/%d r=%-2s lvl=%-5s | cur acc=%.4f ce=%.4f | tgt acc=%.4f"
            % (cell["cell_id"][:46], done_epochs, epochs, cur_res,
               "-" if cur_level is None else "%.2f" % cur_level,
               tacc, tce, gtacc))

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
                loss = F.cross_entropy(model(pipe(tr_imgs[sl], 0.0, res=res_in)),
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
        if done in ckpt_epochs:
            torch.save(state, run_dir / ("checkpoint_ep%02d.pt" % done))

    last = metrics[-1]
    wall = time.perf_counter() - t_start
    summary = {**cell, "job": job, "worker": worker, "gpu_index": gpu,
               "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
               "epochs": epochs, "updates": gstep, "updates_per_epoch": per_epoch,
               "n_train": n, "n_test": int(test_imgs.shape[0]),
               **{k: PROTOCOL[k] for k in ("warmup", "effective_batch", "microbatch")},
               "controller": ctrl.describe(),
               "normalization": {"mean": [float(v) for v in bundle.mean],
                                 "std": [float(v) for v in bundle.std]},
               "final_test_acc": last["test_acc_target"],
               "final_test_ce": last["test_ce_target"],
               # An arm whose level never reaches the target endpoint (a constant
               # sigma) is not evaluated honestly by the target path: that path
               # measures a premature ablation, BN running-statistic mismatch
               # included.  Both paths are therefore always recorded and the cell
               # declares which one to read.
               "final_test_acc_current": last["test_acc_current"],
               "final_test_ce_current": last["test_ce_current"],
               "final_train_probe_acc_current": last["train_probe_acc_current"],
               "final_train_probe_ce_current": last["train_probe_ce_current"],
               "primary_path": cell.get("primary_path", "target"),
               "final_test_acc_primary": (
                   last["test_acc_current"]
                   if cell.get("primary_path") == "current"
                   else last["test_acc_target"]),
               "final_train_probe_acc": last["train_probe_acc_target"],
               "final_train_probe_ce": last["train_probe_ce_target"],
               "wall_seconds": wall, "eval_seconds": eval_seconds[0],
               "train_seconds": wall - eval_seconds[0],
               "peak_mem_mib": (torch.cuda.max_memory_allocated(dev) / 1024 ** 2
                                if dev.type == "cuda" else None)}
    # ---- training-free diagnostics 0a / 0b, on the final weights -----------
    # Opt-in per cell, so campaign cells are unaffected.  Measured with the
    # network in the configuration its primary_path names, and with the hooks
    # still attached -- a BlurPool arm must keep its architectural prefilter.
    if cell.get("diagnostics"):
        from continuation.ablation_ops import aliasing_energy, shift_consistency
        primary = cell.get("primary_path", "target")
        prev = (ctrl.value, ctrl.resolution, ctrl.bypass_all)
        if primary == "current" and ctrl.levels:
            ctrl.set_state(ctrl.levels[epochs - 1], 32)
            ctrl.bypass_all = False
        else:
            ctrl.set_state(0.0, 32)
            ctrl.bypass_all = True
        res_diag = ctrl.input_resolution()
        t_diag = time.perf_counter()
        diag = {
            "measured_in": {"path": primary, "level": ctrl.value,
                            "resolution": 32, "filters_bypassed": ctrl.bypass_all},
            "shift_consistency": shift_consistency(
                model, pipe, test_imgs, test_lbls, res=res_diag),
            "aliasing_energy": aliasing_energy(
                model, pipe, test_imgs, res=res_diag),
        }
        ctrl.value, ctrl.resolution, ctrl.bypass_all = prev
        ctrl.q = ctrl.q_for(ctrl.resolution)
        diag["diagnostic_seconds"] = time.perf_counter() - t_diag
        (run_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2))
        summary["diagnostics"] = diag
        log("%-46s diag: shift-consistency=%.4f  alias-energy=%s"
            % (cell["cell_id"][:46],
               diag["shift_consistency"]["consistency_mean"],
               {k: (None if v["alias_energy_fraction"] is None
                    else round(v["alias_energy_fraction"], 4))
                for k, v in diag["aliasing_energy"]["measured_at"].items()}))

    for h in handles:
        h.remove()
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if rolling.is_file():
        rolling.unlink()
    log("%-46s DONE acc=%.4f ce=%.4f  %.0fs (%.0fs eval)"
        % (cell["cell_id"][:46], summary["final_test_acc"],
           summary["final_test_ce"], wall, eval_seconds[0]))
    return summary


# --------------------------------------------------------------------------
# worker queue
# --------------------------------------------------------------------------

def _worker(queue, cells_by_id, assets, out_dir, job, worker, gpu, done_counter,
            total):
    while True:
        try:
            cell_id = queue.get_nowait()
        except Exception:
            return
        cell = cells_by_id[cell_id]
        try:
            train_cell(cell, Path(assets), gpu, Path(out_dir), job, worker)
        except Exception:
            tb = traceback.format_exc()
            fail = Path(out_dir) / cell_id
            fail.mkdir(parents=True, exist_ok=True)
            (fail / "FAILED.json").write_text(json.dumps(
                {"cell": cell, "job": job, "worker": worker, "traceback": tb},
                indent=2))
            log("FAILED %s\n%s" % (cell_id, tb))
        with done_counter.get_lock():
            done_counter.value += 1
            k = done_counter.value
        el = time.perf_counter() - T0
        log("progress: %d/%d complete, %d pending, elapsed %.0fs, ETA %.0fs"
            % (k, total, total - k, el, el / max(k, 1) * (total - k)))


def run_job(job_index: int, cells: list, prefix: str = "campaign_job"):
    """Run ``cells`` on this environment's GPUs.  ``prefix`` names the output
    directory so a different study is identifiable without reading its cells."""
    out_dir = WORK / ("%s%d_%s" % (prefix, job_index,
                                   time.strftime("%Y%m%d-%H%M%S")))
    out_dir.mkdir(parents=True, exist_ok=True)
    ngpu = torch.cuda.device_count()
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "torch": torch.__version__, "cuda": torch.version.cuda, "n_gpu": ngpu,
           "gpus": [torch.cuda.get_device_name(i) for i in range(ngpu)]}
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2))
    (out_dir / "job_manifest.json").write_text(json.dumps(
        {"job": job_index, "n_cells": len(cells), "cells": cells}, indent=2))
    log("job %d environment: %s" % (job_index, env))
    if ngpu == 0:
        (out_dir / "ABORTED.json").write_text(json.dumps(
            {"reason": "no accelerator granted", "environment": env}))
        raise SystemExit("no accelerator granted")

    assets = find_assets()
    verify_assets(assets, out_dir)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    for c in cells:                       # already ordered longest-first
        queue.put(c["cell_id"])
    counter = ctx.Value("i", 0)
    cells_by_id = {c["cell_id"]: c for c in cells}
    n_workers = min(ngpu, 2)
    log("job %d: %d cells over %d worker(s)" % (job_index, len(cells), n_workers))

    procs = [ctx.Process(target=_worker,
                         args=(queue, cells_by_id, str(assets), str(out_dir),
                               job_index, w, w, counter, len(cells)))
             for w in range(n_workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    summaries, failures = [], []
    for c in cells:
        d = out_dir / c["cell_id"]
        if (d / "summary.json").is_file():
            summaries.append(json.loads((d / "summary.json").read_text()))
        elif (d / "FAILED.json").is_file():
            failures.append(json.loads((d / "FAILED.json").read_text())["cell"])
        else:
            failures.append({**c, "reason": "incomplete"})
    (out_dir / "job_summary.json").write_text(json.dumps(
        {"job": job_index, "completed": summaries, "failed_or_incomplete": failures,
         "elapsed_s": time.perf_counter() - T0}, indent=2))
    log("job %d finished: %d complete, %d failed/incomplete, %.0fs"
        % (job_index, len(summaries), len(failures), time.perf_counter() - T0))
    return out_dir
