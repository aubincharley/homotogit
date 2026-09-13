"""Driver for the unified batch: every promising method, one set of pinned assets.

Adapted from ``scripts/resbench_driver.py``.  Two differences that matter:

* the controller comes from ``continuation.ablation_ops.build_from_cell``, the
  code Idriss's arms actually ran, so a replay here differs from his numbers by
  the batch and nothing else;
* evaluation happens **after every epoch**, not every second epoch, and each
  epoch also records the mean training loss over its own minibatches -- which is
  free, and is what makes a loss-versus-epoch plot possible.

Each snapshot stores two paths: the *current* one (the resolution and blur the
last completed update actually used) and a forced 32x32 no-blur *diagnostic*.
Only the current path belongs in primary figures.
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

from continuation.ablation_ops import build_from_cell
from continuation.config import DataConfig, ModelConfig, OptimConfig
from continuation.data import build_dataset
from continuation.models import build_model
from continuation.optim import build_optimizer, lr_at, set_lr
from continuation.pipeline import ChannelNormalizer, InputPipeline
from continuation.transforms.gaussian import GaussianSmoothing

WORK = Path(os.environ.get("STUDY_OUT", "/kaggle/working"))
T0 = time.perf_counter()
PROTOCOL = {"epochs": 30, "warmup": 60, "effective_batch": 128, "microbatch": 32,
            "lr": 0.005, "momentum": 0.9, "weight_decay": 5e-4}
EXPECTED_STATS = {"mean": [0.4913996756076813, 0.4821584224700928,
                           0.44653090834617615],
                  "std": [0.24703222513198853, 0.24348512291908264,
                          0.26158782839775085]}


def log(msg):
    print("[%7.1fs] %s" % (time.perf_counter() - T0, msg), flush=True)


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
    checks = {"array:" + k: _sha_array(z[k]) == v["sha256"]
              for k, v in man["arrays"].items()}
    checks.update({"state:" + k: _sha_state(assets / (k + ".pt")) == v
                   for k, v in man["states"].items()})
    res = {"assets_dir": str(assets), "checks": checks,
           "all_match": all(checks.values())}
    (out_dir / "assets_verification.json").write_text(json.dumps(res, indent=2))
    log("pinned-asset verification: %s" % ("OK" if res["all_match"] else checks))
    if not res["all_match"]:
        raise SystemExit("pinned assets failed verification")
    return res


def data_source():
    import glob
    override = os.environ.get("STUDY_DATA")
    if override and (Path(override) / "cifar-10-batches-py").exists():
        return override
    for hit in sorted(glob.glob("/kaggle/input/**/cifar-10-batches-py",
                                recursive=True)):
        return str(Path(hit).parent)
    raise SystemExit("CIFAR-10 not found")


@torch.no_grad()
def evaluate(model, ctrl, images, labels, pipe, epoch_index, force_target=False,
             batch=500):
    """Evaluate at the configuration of ``epoch_index``, or on the plain path."""
    mode = model.training
    # since the adaptive merge the level is a per-site row and `value` is a
    # read-only property; save whichever representation the controller has
    prev = (list(ctrl.row) if getattr(ctrl, "row", None) is not None else None,
            ctrl.value, ctrl.resolution, ctrl.bypass_all)
    if force_target:
        ctrl.bypass_all = True
        ctrl.set_state(0.0, 32)
    else:
        ctrl.bypass_all = False
        ctrl.set_epoch(epoch_index)
    model.eval()
    res_in = ctrl.input_resolution()
    tot, correct, n = 0.0, 0, images.shape[0]
    for i in range(0, n, batch):
        logits = model(pipe(images[i:i + batch], 0.0, res=res_in))
        tot += float(F.cross_entropy(logits, labels[i:i + batch], reduction="sum"))
        correct += int((logits.argmax(1) == labels[i:i + batch]).sum())
    if hasattr(ctrl, "row"):
        ctrl.row = prev[0]
    else:
        ctrl.value = prev[1]
    ctrl.resolution, ctrl.bypass_all = prev[2], prev[3]
    ctrl.q = ctrl.q_for(ctrl.resolution)
    if mode:
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
    if not (np.allclose([float(v) for v in bundle.mean], EXPECTED_STATS["mean"],
                        atol=1e-7)
            and np.allclose([float(v) for v in bundle.std],
                            EXPECTED_STATS["std"], atol=1e-7)):
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
    ctrl, handles = build_from_cell(model, cell)

    epochs = PROTOCOL["epochs"]
    n, B, mb = int(subset.size), PROTOCOL["effective_batch"], PROTOCOL["microbatch"]
    per_epoch = (n + B - 1) // B
    total_updates = epochs * per_epoch
    ocfg = OptimConfig(lr=PROTOCOL["lr"], momentum=PROTOCOL["momentum"],
                       weight_decay=PROTOCOL["weight_decay"], batch_size=B,
                       total_steps=total_updates, lr_schedule="cosine",
                       warmup_steps=PROTOCOL["warmup"], min_lr=0.0)
    opt = build_optimizer(model, ocfg)

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
            log("%s rolling checkpoint unusable (%r)" % (cell["cell_id"], exc))

    def snapshot(done_epochs, train_loss=None):
        t0 = time.perf_counter()
        last = max(done_epochs - 1, 0)
        pce, pacc = evaluate(model, ctrl, probe_imgs, probe_lbls, pipe, last)
        tce, tacc = evaluate(model, ctrl, test_imgs, test_lbls, pipe, last)
        gp, gpa = evaluate(model, ctrl, probe_imgs, probe_lbls, pipe, last, True)
        gt, gta = evaluate(model, ctrl, test_imgs, test_lbls, pipe, last, True)
        ctrl.set_epoch(last)
        rec = {"epoch": done_epochs, "update": gstep,
               "lr": lr_at(min(max(gstep - 1, 0), total_updates - 1), ocfg),
               "current": {"resolution": ctrl.resolution, "sigma": ctrl.value,
                           "id": cell["id"]},
               "train_loss_epoch": train_loss,
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
        if done_epochs % 5 == 0 or done_epochs == epochs:
            log("%-34s ep=%2d/%d | test acc=%.4f ce=%.4f | plain-path acc=%.4f"
                % (cell["cell_id"][:34], done_epochs, epochs, tacc, tce, gta))

    if start_epoch == 0:
        snapshot(0)
    for e in range(start_epoch, epochs):
        ctrl.set_epoch(e)
        res_in = ctrl.input_resolution()
        perm = perms[e]
        run_loss, run_n = 0.0, 0
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
                run_loss += float(loss.detach()) * n_m
                run_n += n_m
            opt.step()
            gstep += 1
        snapshot(e + 1, run_loss / max(run_n, 1))
        ctrl.set_epoch(e)
        torch.save({"model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(), "epoch": e + 1,
                    "update": gstep, "metrics": metrics,
                    "eval_seconds": eval_seconds[0], "cell": cell,
                    "cpu_rng": torch.get_rng_state()}, rolling)

    last = metrics[-1]
    wall = time.perf_counter() - t_start
    summary = {**{k: v for k, v in cell.items() if k != "sigma_profile"},
               "sigma_profile": cell.get("sigma_profile"),
               "job": job, "worker": worker, "gpu_index": gpu,
               "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
               "epochs": epochs, "updates": gstep, "updates_per_epoch": per_epoch,
               "n_train": n, "n_test": int(test_imgs.shape[0]),
               "endpoint": "32x32, all operators bypassed",
               "final_test_acc": last["test_acc_bypass32"],
               "final_test_ce": last["test_ce_bypass32"],
               "final_train_probe_ce": last["train_probe_ce_bypass32"],
               "controller": ctrl.describe(),
               "wall_seconds": wall, "eval_seconds": eval_seconds[0],
               "train_seconds": wall - eval_seconds[0],
               "peak_mem_mib": (torch.cuda.max_memory_allocated(dev) / 1024 ** 2
                                if dev.type == "cuda" else None)}
    for h in handles:
        h.remove()
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if rolling.is_file():
        rolling.unlink()
    log("%-34s DONE acc=%.4f ce=%.4f  %.0fs (%.0fs eval)"
        % (cell["cell_id"][:34], summary["final_test_acc"],
           summary["final_test_ce"], wall, eval_seconds[0]))
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
                {"cell_id": cid, "job": job, "traceback": tb}, indent=2))
            log("FAILED %s\n%s" % (cid, tb))
        with counter.get_lock():
            counter.value += 1
            k = counter.value
        el = time.perf_counter() - T0
        log("progress: %d/%d done, %d left, %.0fs elapsed, ETA %.0fs"
            % (k, total, total - k, el, el / max(k, 1) * (total - k)))


def run_job(job_index: int, cells: list):
    out_dir = WORK / ("unified_job%d_%s" % (job_index,
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
        (out_dir / "ABORTED.json").write_text(json.dumps({"reason": "no accelerator"}))
        raise SystemExit("no accelerator granted")

    assets = find_assets()
    verify_assets(assets, out_dir)

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    for c in cells:
        q.put(c["cell_id"])
    counter = ctx.Value("i", 0)
    by_id = {c["cell_id"]: c for c in cells}
    nw = min(ngpu, 2)
    log("job %d: %d cells over %d worker(s)" % (job_index, len(cells), nw))
    procs = [ctx.Process(target=_worker,
                         args=(q, by_id, str(assets), str(out_dir), job_index,
                               w, w, counter, len(cells)))
             for w in range(nw)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    done, failed = [], []
    for c in cells:
        d = out_dir / c["cell_id"]
        if (d / "summary.json").is_file():
            done.append(json.loads((d / "summary.json").read_text()))
        else:
            failed.append(c["cell_id"])
    (out_dir / "job_summary.json").write_text(json.dumps(
        {"job": job_index, "completed": done, "failed_or_incomplete": failed,
         "elapsed_s": time.perf_counter() - T0}, indent=2))
    log("job %d finished: %d done, %d failed/incomplete, %.0fs"
        % (job_index, len(done), len(failed), time.perf_counter() - T0))
    return out_dir
