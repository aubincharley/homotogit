"""Progressive input resolution on STL-10 (96x96): does the fine equal-dwell ramp
transfer, and does coarse training now buy wall time?

CIFAR-10 result being transposed (docs/adaptive_resolution_plan.md §14-15): a
fixed ramp with small steps and equal dwell per size beats the 16/24/32
schedule by ~1 pp at equal update budget and equal time; no adaptive trigger
beat a fixed schedule.  On CIFAR the coarse epochs saved no time on a T4 (tiny
tensors).  STL-10 images are 96x96, nine times the pixels, so coarse epochs
should be genuinely cheaper -- which is where "accuracy at equal time" becomes a
real question.

Protocol (aligned with the stl10-transfer branch so the numbers are comparable
in spirit, not paired with it): labelled train 5,000 / test 8,000, no
augmentation, channel statistics fitted on the unfiltered training set,
ResNet-20 BN (GAP handles 96x96), SGD lr 0.005 / momentum 0.9 / wd 5e-4, warmup
60 updates then cosine over the whole horizon, effective batch 128 as four
microbatches of 32 (the last group of each epoch holds 8 images and is weighted
by count), 40 updates per epoch.  Horizon 60 epochs = 2,400 updates unless the
arm says otherwise.  Per-seed init weights and per-epoch permutations are
regenerated in the kernel from the project's named RNG streams and their
sha256 recorded.

Arms (sizes x3, epoch boundaries x2 relative to the CIFAR campaign):

  R96        96 throughout                                   (control)
  Rprog      48 x12, 72 x12, 96 x36                            (Rprog x3, x2)
  Rsteps4    48/60/72/84 x6 each, 96 x36                       (Rsteps4 analogue)
  Rlin24     one even size per epoch 48 -> 94 over 24 epochs, 96 x36   (Rlin12 analogue)
  Rlin24eq   same ramp, horizon 72 epochs (96 x48): built to cost about the
             same *compute* as R96's 60 epochs -- the equal-time comparison

Every arm evaluates every 3 epochs: probe / test on the current path, test on
the 96x96 target path, and the target path after BatchNorm recalibration at
fixed weights (restored afterwards).  Training seconds are measured per epoch
with CUDA synchronisation, separately from evaluation.
"""
from __future__ import annotations

import glob
import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from continuation.config import ModelConfig, OptimConfig
from continuation.data import channel_stats, fixed_subset_indices
from continuation.models import build_model
from continuation.optim import build_optimizer, lr_at, set_lr
from continuation.pipeline import ChannelNormalizer, InputPipeline
from continuation.probe_signals import (BNState, ce_acc, parameter_checksum,
                                        parameters_equal, recalibrate_bn)
from continuation.seeding import numpy_generator
from continuation.transforms.gaussian import GaussianSmoothing

WORK = Path(os.environ.get("STUDY_OUT", "/kaggle/working"))
T0 = time.perf_counter()
SIDE = 96
PROTOCOL = {"warmup": 60, "effective_batch": 128, "microbatch": 32,
            "lr": 0.005, "momentum": 0.9, "weight_decay": 5e-4}
EVAL_EVERY = 3
PROBE = {"size": 500, "seed": 0, "stream": "study_train_probe"}
MONITOR = {"size": 1000, "seed": 0, "stream": "adaptive_monitor"}
SEEDS = (0, 1, 2)


def log(msg):
    print("[%7.1fs] %s" % (time.perf_counter() - T0, msg), flush=True)


# --------------------------------------------------------------------------
# schedules
# --------------------------------------------------------------------------

def ramp_even(start, end, n):
    """``n`` sizes from ``start`` toward ``end`` (excluded), rounded to even."""
    return [2 * int(round((start + (end - start) * e / float(n)) / 2.0)) for e in range(n)]


SCHEDULES = {
    "R96":      [96] * 60,
    "Rprog":    [48] * 12 + [72] * 12 + [96] * 36,
    "Rsteps4":  [48] * 6 + [60] * 6 + [72] * 6 + [84] * 6 + [96] * 36,
    "Rlin24":   ramp_even(48, 96, 24) + [96] * 36,
    "Rlin24eq": ramp_even(48, 96, 24) + [96] * 48,
    # controls for the equal-compute comparison (second launch, STL10_PHASE=2):
    # plain with the same 72-epoch horizon as Rlin24eq, and Rprog extended at 96
    # until its nominal compute matches R96's 60 epochs (45.8 + 14 = 59.8 units)
    "R96_72":   [96] * 72,
    "Rprogeq":  [48] * 12 + [72] * 12 + [96] * 50,
}
PHASE = int(os.environ.get("STL10_PHASE", "1"))
ARMS = {1: ("R96", "Rprog", "Rsteps4", "Rlin24", "Rlin24eq"), 2: ("R96_72", "Rprogeq")}[PHASE]


def compute_units(sched):
    """Nominal convolution work relative to one epoch at 96x96."""
    return sum((r / float(SIDE)) ** 2 for r in sched)


def build_runs():
    return [{"label": "%s__stl10__seed%d" % (name, s), "schedule": name, "seed": s}
            for name in ARMS for s in SEEDS]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def find_stl10() -> Path:
    override = os.environ.get("STL10_DIR")
    if override and (Path(override) / "train_X.bin").is_file():
        return Path(override)
    for hit in sorted(glob.glob("/kaggle/input/**/train_X.bin", recursive=True)):
        return Path(hit).parent
    raise SystemExit("STL-10 binaries not found; attach yellowflag/stl10labeled2")


def load_stl10(base: Path):
    """Official binaries: column-major 96x96x3 -> [N,3,96,96] row-major; labels 1..10 -> 0..9."""
    def images(name):
        a = np.fromfile(base / name, dtype=np.uint8)
        if a.size % (3 * SIDE * SIDE):
            raise ValueError("%s has %d bytes, not a multiple of 3*96*96" % (name, a.size))
        return np.ascontiguousarray(a.reshape(-1, 3, SIDE, SIDE).transpose(0, 1, 3, 2))

    def labels(name):
        y = np.fromfile(base / name, dtype=np.uint8).astype(np.int64) - 1
        if y.min() < 0 or y.max() > 9:
            raise ValueError("%s: labels outside 1..10" % name)
        return y

    tr_x, tr_y = images("train_X.bin"), labels("train_y.bin")
    te_x, te_y = images("test_X.bin"), labels("test_y.bin")
    if len(tr_x) != len(tr_y) or len(te_x) != len(te_y):
        raise ValueError("image/label counts differ")
    return (torch.from_numpy(tr_x), torch.from_numpy(tr_y),
            torch.from_numpy(te_x), torch.from_numpy(te_y))


def _sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _sha_state(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(np.ascontiguousarray(sd[k].cpu().numpy()).tobytes())
    return h.hexdigest()


def build_shared(out_dir: Path, n_train: int, labels: np.ndarray, max_epochs: int) -> dict:
    """Per-seed init weights and per-epoch permutations; probe and monitor indices."""
    rec = {"n_train": int(n_train), "max_epochs": max_epochs, "arrays": {}, "states": {}}
    arrays = {}
    for s in SEEDS:
        m = build_model(ModelConfig(arch="resnet20_bn_cifar"), 10, seed=s)
        torch.save(m.state_dict(), out_dir / ("init_seed%d.pt" % s))
        rec["states"]["init_seed%d" % s] = _sha_state(m.state_dict())
        g = numpy_generator(s, "batch")
        arrays["perm_seed%d" % s] = np.stack([g.permutation(n_train).astype(np.int32)
                                              for _ in range(max_epochs)])
    arrays["train_probe"] = fixed_subset_indices(n_train, PROBE["size"], PROBE["seed"],
                                                 PROBE["stream"], labels=labels)
    arrays["monitor"] = fixed_subset_indices(n_train, MONITOR["size"], MONITOR["seed"],
                                             MONITOR["stream"], labels=labels)
    np.savez_compressed(out_dir / "shared_indices.npz", **arrays)
    rec["arrays"] = {k: {"sha256": _sha_array(v), "shape": list(v.shape)} for k, v in arrays.items()}
    (out_dir / "shared_manifest.json").write_text(json.dumps(rec, indent=2))
    log("shared state written: %s" % json.dumps({k: v["sha256"][:12] for k, v in rec["arrays"].items()}))
    return rec


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------

def train_run(spec, shared_dir: Path, gpu: int, out_dir: Path, worker: int):
    dev = torch.device("cuda:%d" % gpu if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    run_dir = out_dir / spec["label"]
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = int(spec["seed"])
    sched = list(SCHEDULES[spec["schedule"]])
    epochs = len(sched)

    tr_x, tr_y, te_x, te_y = load_stl10(find_stl10())
    mean, std = channel_stats(tr_x)
    tr_imgs, tr_lbls = tr_x.to(dev), tr_y.to(dev)
    test_imgs, test_lbls = te_x.to(dev), te_y.to(dev)
    n = int(tr_imgs.shape[0])
    z = np.load(shared_dir / "shared_indices.npz")
    perms = z["perm_seed%d" % seed]
    probe = torch.as_tensor(z["train_probe"], device=dev)
    mon = torch.as_tensor(z["monitor"], device=dev)
    probe_imgs, probe_lbls = tr_imgs[probe], tr_lbls[probe]
    mon_imgs, mon_lbls = tr_imgs[mon], tr_lbls[mon]

    pipe = InputPipeline(GaussianSmoothing(sigma_max=1.0),
                         ChannelNormalizer(mean, std).to(dev))
    model = build_model(ModelConfig(arch="resnet20_bn_cifar"), 10, seed=seed).to(dev)
    model.load_state_dict(torch.load(shared_dir / ("init_seed%d.pt" % seed),
                                     map_location=dev, weights_only=True))
    model.train()

    B, mb = PROTOCOL["effective_batch"], PROTOCOL["microbatch"]
    per_epoch = (n + B - 1) // B
    total_updates = epochs * per_epoch
    ocfg = OptimConfig(lr=PROTOCOL["lr"], momentum=PROTOCOL["momentum"],
                       weight_decay=PROTOCOL["weight_decay"], batch_size=B,
                       total_steps=total_updates, lr_schedule="cosine",
                       warmup_steps=PROTOCOL["warmup"], min_lr=0.0)
    opt = build_optimizer(model, ocfg)

    metrics, gstep = [], 0
    train_seconds = [0.0]
    eval_seconds = [0.0]
    epoch_seconds = []
    t_start = time.perf_counter()

    def fwd_at(r):
        return lambda x: model(pipe(x, 0.0, res=r))

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    def snapshot(done, epoch_train_loss=None):
        sync()
        t0 = time.perf_counter()
        ref = parameter_checksum(model)
        bn_ref = BNState(model)
        last = max(done - 1, 0)
        r_cur = sched[last]
        pce, pacc = ce_acc(model, fwd_at(r_cur), probe_imgs, probe_lbls)
        tce_c, tacc_c = ce_acc(model, fwd_at(r_cur), test_imgs, test_lbls)
        if r_cur != SIDE:
            tce_t, tacc_t = ce_acc(model, fwd_at(SIDE), test_imgs, test_lbls)
            saved = BNState(model)
            recalibrate_bn(model, fwd_at(SIDE), mon_imgs, batch=250)
            tce_r, tacc_r = ce_acc(model, fwd_at(SIDE), test_imgs, test_lbls)
            saved.restore()
        else:
            tce_t, tacc_t, tce_r, tacc_r = tce_c, tacc_c, None, None
        sync()
        rec = {"epoch": done, "update": gstep,
               "lr": lr_at(min(max(gstep - 1, 0), total_updates - 1), ocfg),
               "resolution_current": r_cur,
               "train_loss_epoch_mean": epoch_train_loss,
               "train_probe_ce_current": pce, "train_probe_acc_current": pacc,
               "test_ce_current": tce_c, "test_acc_current": tacc_c,
               "test_ce_target": tce_t, "test_acc_target": tacc_t,
               "test_ce_target_bnrecal": tce_r, "test_acc_target_bnrecal": tacc_r,
               "train_seconds_cumulative": train_seconds[0],
               "epoch_seconds": list(epoch_seconds),
               "purity": {"params_unchanged": parameters_equal(model, ref),
                          "bn_restored": not bn_ref.differs(), "mode_train": model.training},
               "elapsed_s": time.perf_counter() - t_start,
               "peak_mem_mib": (torch.cuda.max_memory_allocated(dev) / 1024 ** 2
                                if dev.type == "cuda" else None)}
        if not all(rec["purity"].values()):
            raise RuntimeError("evaluation perturbed the training state: %s" % rec["purity"])
        metrics.append(rec)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
        eval_seconds[0] += time.perf_counter() - t0
        log("%-24s ep=%2d/%d r=%-2d | test cur=%.4f tgt=%.4f recal=%s | train %.0fs (last epoch %.1fs)"
            % (spec["label"], done, epochs, r_cur, tacc_c, tacc_t,
               "-" if tacc_r is None else "%.4f" % tacc_r, train_seconds[0],
               epoch_seconds[-1] if epoch_seconds else 0.0))

    snapshot(0)
    eval_epochs = set(range(EVAL_EVERY, epochs + 1, EVAL_EVERY)) | {epochs}
    for e in range(epochs):
        r = sched[e]
        perm = perms[e]
        sync()
        t_ep = time.perf_counter()
        loss_sum, loss_n = 0.0, 0
        for s0 in range(0, n, B):
            batch = perm[s0:s0 + B]
            total = int(batch.size)
            idx = torch.as_tensor(batch, dtype=torch.long, device=dev)
            set_lr(opt, lr_at(gstep, ocfg))
            opt.zero_grad(set_to_none=True)
            for a in range(0, total, mb):
                sl = idx[a:a + mb]
                n_m = int(sl.numel())
                loss = F.cross_entropy(model(pipe(tr_imgs[sl], 0.0, res=r)), tr_lbls[sl])
                (loss * (n_m / total)).backward()
                loss_sum += float(loss) * n_m
                loss_n += n_m
            opt.step()
            gstep += 1
        sync()
        dt = time.perf_counter() - t_ep
        train_seconds[0] += dt
        epoch_seconds.append(round(dt, 3))
        done = e + 1
        if done in eval_epochs:
            snapshot(done, epoch_train_loss=loss_sum / max(loss_n, 1))

    last = metrics[-1]
    wall = time.perf_counter() - t_start
    summary = {**spec, "worker": worker, "gpu_index": gpu,
               "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
               "dataset": "stl10 labelled", "side": SIDE,
               "epochs": epochs, "updates": gstep, "updates_per_epoch": per_epoch,
               "n_train": n, "n_test": int(test_imgs.shape[0]), **PROTOCOL,
               "resolution_by_epoch": sched, "compute_units_nominal": compute_units(sched),
               "reduction": "bilinear on the float image, align_corners=False, antialias=True; "
                            "exact bypass at 96",
               "normalization": {"mean": [float(v) for v in mean], "std": [float(v) for v in std]},
               "final_test_acc": last["test_acc_target"], "final_test_ce": last["test_ce_target"],
               "final_train_probe_ce": last["train_probe_ce_current"],
               "wall_seconds": wall, "train_seconds": train_seconds[0],
               "eval_seconds": eval_seconds[0], "epoch_seconds": epoch_seconds,
               "peak_mem_mib": (torch.cuda.max_memory_allocated(dev) / 1024 ** 2
                                if dev.type == "cuda" else None)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log("%-24s DONE acc=%.4f ce=%.4f  train %.0fs  eval %.0fs" % (
        spec["label"], summary["final_test_acc"], summary["final_test_ce"],
        train_seconds[0], eval_seconds[0]))
    return summary


# --------------------------------------------------------------------------
# workers
# --------------------------------------------------------------------------

def _worker(queue, by_label, shared_dir, out_dir, worker, gpu, counter, total):
    while True:
        try:
            label = queue.get_nowait()
        except Exception:
            return
        try:
            train_run(by_label[label], Path(shared_dir), gpu, Path(out_dir), worker)
        except Exception:
            tb = traceback.format_exc()
            d = Path(out_dir) / label
            d.mkdir(parents=True, exist_ok=True)
            (d / "FAILED.json").write_text(json.dumps({"spec": by_label[label], "traceback": tb}, indent=2))
            log("FAILED %s\n%s" % (label, tb))
        with counter.get_lock():
            counter.value += 1
            k = counter.value
        el = time.perf_counter() - T0
        log("progress: %d/%d complete, elapsed %.0fs, ETA %.0fs" % (k, total, el, el / max(k, 1) * (total - k)))


def main():
    out_dir = WORK / ("stl10_resolution_%s" % time.strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ngpu = torch.cuda.device_count()
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "torch": torch.__version__, "cuda": torch.version.cuda, "n_gpu": ngpu,
           "gpus": [torch.cuda.get_device_name(i) for i in range(ngpu)]}
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2))
    log("environment: %s" % env)
    if ngpu == 0:
        (out_dir / "ABORTED.json").write_text(json.dumps({"reason": "no accelerator granted", "environment": env}))
        raise SystemExit("no accelerator granted")

    base = find_stl10()
    tr_x, tr_y, te_x, te_y = load_stl10(base)
    log("STL-10 from %s: train %s test %s" % (base, tuple(tr_x.shape), tuple(te_x.shape)))
    max_epochs = max(len(s) for s in SCHEDULES.values())
    build_shared(out_dir, int(tr_x.shape[0]), tr_y.numpy(), max_epochs)
    del tr_x, te_x

    runs = build_runs()
    # longest first: R96 and the 72-epoch arm
    runs.sort(key=lambda r: (-compute_units(SCHEDULES[r["schedule"]]), r["seed"]))
    (out_dir / "config.json").write_text(json.dumps(
        {"phase": PHASE, "arms": ARMS, "protocol": PROTOCOL, "schedules": SCHEDULES,
         "compute_units_nominal": {k: compute_units(v) for k, v in SCHEDULES.items()},
         "eval_every": EVAL_EVERY, "probe": PROBE, "monitor": MONITOR, "runs": runs}, indent=2))
    log("%d runs: %s" % (len(runs), [r["label"] for r in runs]))

    by_label = {r["label"]: r for r in runs}
    ctx = mp.get_context("spawn")
    queue, counter = ctx.Queue(), ctx.Value("i", 0)
    for r in runs:
        queue.put(r["label"])
    procs = [ctx.Process(target=_worker, args=(queue, by_label, str(out_dir), str(out_dir),
                                               w, w, counter, len(runs)))
             for w in range(min(ngpu, 2))]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    summaries, failures = [], []
    for r in runs:
        f = out_dir / r["label"] / "summary.json"
        (summaries if f.is_file() else failures).append(json.loads(f.read_text()) if f.is_file() else r["label"])
    (out_dir / "study_summary.json").write_text(json.dumps(
        {"completed": summaries, "failed_or_incomplete": failures, "elapsed_s": time.perf_counter() - T0}, indent=2))
    for s in summaries:
        log("SUMMARY %-24s acc=%.4f ce=%.4f  train %.0fs  units %.1f" % (
            s["label"], s["final_test_acc"], s["final_test_ce"], s["train_seconds"], s["compute_units_nominal"]))
    log("finished: %d complete, %d failed, %.0fs" % (len(summaries), len(failures), time.perf_counter() - T0))


if __name__ == "__main__":
    main()
