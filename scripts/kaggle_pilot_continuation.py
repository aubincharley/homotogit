r"""CIFAR-10 pilot: plain vs Gaussian continuation vs db2 wavelet continuation.

One-seed exploratory pilot.  Exactly three training runs, 600 optimizer updates
each, effective batch 128, sharing one subset, one initialization and one
minibatch ordering.  Runs on Kaggle via ``scripts/kaggle_run.py``.

Continuation acts at the 19 validated insertion points (after each spatial 3x3
convolution, before normalization/activation; shortcuts untouched).  Network
inputs stay unfiltered.

    r_k = max(1 - k/300, 0),  sigma_k = r_k,  s_k = 1 - (1 - s0) r_k

so updates 300-599 use exact identity bypasses.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from continuation.config import DataConfig, ModelConfig, OptimConfig
from continuation.data import build_dataset, BatchIndexStream, fixed_subset_indices
from continuation.models import build_model, count_parameters
from continuation.optim import build_optimizer, lr_at, set_lr
from continuation.pipeline import ChannelNormalizer, InputPipeline
from continuation.transforms.gaussian import GaussianSmoothing
from continuation.transforms.wavelet import wavelet_shrink

SEED = 0
PER_CLASS = 500
N_SUBSET = 5000
UPDATES = 600
CONT_END = 300
EFFECTIVE_BATCH = 128
MICROBATCH = 32
FALLBACK_MICROBATCH = 16
WARMUP = 30
EVAL_EVERY = 50
TRAIN_PROBE = 200
VAL_PROBE = 500
CALIB_IMAGES = 32
S_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 0.9)
WORKER_LIMIT_S = 40 * 60
WORK = Path(os.environ.get("PILOT_OUT", "/kaggle/working"))


def data_source():
    """Locate CIFAR-10 in an attached Kaggle dataset; kernels have no internet.

    The mount path is discovered rather than assumed (Kaggle nests it as
    ``/kaggle/input/datasets/<owner>/<slug>/``).  With ``download=False``
    torchvision verifies the official md5 of every batch file, so a tampered
    copy raises instead of being loaded silently.
    """
    import glob as _glob
    for hit in sorted(_glob.glob("/kaggle/input/**/cifar-10-batches-py",
                                 recursive=True)):
        return str(Path(hit).parent), False
    return str(WORK / "data"), True


def log(msg):
    print("[%7.1fs] %s" % (time.perf_counter() - T0, msg), flush=True)


T0 = time.perf_counter()


# --------------------------------------------------------------------------
# Filtering controller: one object per run, mutated once per optimizer update
# --------------------------------------------------------------------------

class Controller:
    """Applies the scheduled filter at each insertion point."""

    def __init__(self, kind: str, s0: float = 1.0):
        self.kind = kind                     # none | gaussian | db2
        self.s0 = float(s0)
        self.value = None                    # sigma or s for the current update
        self.bypass_all = False              # forced identity during evaluation
        self.gauss = GaussianSmoothing(sigma_max=1.0, truncate=4.0) if kind == "gaussian" else None

    def set_update(self, k: int):
        r = max(1.0 - k / float(CONT_END), 0.0)
        if self.kind == "gaussian":
            self.value = r                                  # sigma_k = r_k
        elif self.kind == "db2":
            self.value = 1.0 - (1.0 - self.s0) * r          # s_k
        else:
            self.value = None
        return self.value

    def __call__(self, t):
        if self.bypass_all or self.kind == "none":
            return t
        if self.kind == "gaussian":
            return self.gauss(t, self.value)                 # sigma=0 -> exact identity
        return wavelet_shrink(t, self.value, "db2", impl="fast", bypass_identity=True)

    def describe(self):
        return {"kind": self.kind, "s0": self.s0,
                "gaussian_kernel": None if not self.gauss else
                {"kernel_size": self.gauss.kernel_size, "radius": self.gauss.radius,
                 "sigma_max": self.gauss.sigma_max, "truncate": self.gauss.truncate}}


def attach(model, controller):
    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
            handles.append(m.register_forward_hook(
                lambda _m, _i, out, c=controller: c(out)))
    return handles


# --------------------------------------------------------------------------
# Shared setup
# --------------------------------------------------------------------------

def build_shared(out_dir: Path):
    root, dl = data_source()
    cfg_data = DataConfig(root=root, download=dl)
    log("CIFAR-10 root=%s download=%s" % (root, dl))
    bundle = build_dataset(cfg_data)
    labels = bundle.train.labels.numpy()

    rng = np.random.default_rng(SEED)
    picks = []
    for c in range(bundle.num_classes):
        idx = np.flatnonzero(labels == c)
        picks.append(rng.permutation(idx)[:PER_CLASS])
    subset = np.sort(np.concatenate(picks))
    assert subset.size == N_SUBSET
    assert np.bincount(labels[subset], minlength=10).tolist() == [PER_CLASS] * 10

    model = build_model(ModelConfig(), bundle.num_classes, seed=SEED)
    init_state = {k: v.clone() for k, v in model.state_dict().items()}
    torch.save(init_state, out_dir / "init_weights.pt")

    stream = BatchIndexStream(N_SUBSET, EFFECTIVE_BATCH, seed=SEED)
    order = np.stack([stream.next_indices() for _ in range(UPDATES)])

    train_probe = fixed_subset_indices(N_SUBSET, TRAIN_PROBE, SEED, "pilot_train_probe",
                                       labels=labels[subset])
    val_probe = fixed_subset_indices(len(bundle.val), VAL_PROBE, SEED, "pilot_val_probe",
                                     labels=bundle.val.labels.numpy())
    np.savez_compressed(out_dir / "shared_indices.npz", subset=subset, order=order,
                        train_probe=train_probe, val_probe=val_probe)
    return bundle, subset, order, train_probe, val_probe, init_state


# --------------------------------------------------------------------------
# s0 calibration (no training, no RNG or weight mutation)
# --------------------------------------------------------------------------

@torch.no_grad()
def calibrate(bundle, subset, init_state, device, out_dir: Path):
    """Pick s0 by matching median relative perturbation to Gaussian sigma=1."""
    rng_state = torch.get_rng_state()
    model = build_model(ModelConfig(), bundle.num_classes, seed=SEED).to(device).eval()
    model.load_state_dict(init_state)

    captured = []
    handles = [m.register_forward_hook(lambda _m, _i, o: captured.append(o.detach()))
               for m in model.modules()
               if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3)]

    img_idx = fixed_subset_indices(subset.size, CALIB_IMAGES, SEED, "pilot_calib",
                                   labels=bundle.train.labels.numpy()[subset])
    sub = bundle.train.subset(subset[img_idx], "calib").to(device)
    norm = ChannelNormalizer(bundle.mean, bundle.std).to(device)
    pipe = InputPipeline(GaussianSmoothing(sigma_max=1.0), norm)
    model(pipe(sub.images, 0.0))                       # filters disabled
    for h in handles:
        h.remove()

    def rel(t, h):
        mean = h.mean(dim=(-2, -1), keepdim=True)
        num = (t - h).flatten(1).norm(dim=1)
        den = (h - mean).flatten(1).norm(dim=1) + 1e-12
        return (num / den)

    gauss = GaussianSmoothing(sigma_max=1.0, truncate=4.0)
    per_layer = {"gaussian_sigma1": [], **{"db2_s%g" % s: [] for s in S_CANDIDATES}}
    for h in captured:
        per_layer["gaussian_sigma1"].append(float(rel(gauss(h, 1.0), h).median()))
        for s in S_CANDIDATES:
            t = wavelet_shrink(h, s, "db2", impl="fast", bypass_identity=True)
            per_layer["db2_s%g" % s].append(float(rel(t, h).median()))

    target = float(np.median(per_layer["gaussian_sigma1"]))
    meds = {s: float(np.median(per_layer["db2_s%g" % s])) for s in S_CANDIDATES}
    s0 = min(meds, key=lambda s: abs(meds[s] - target))
    report = {"n_layers": len(captured), "n_images": int(CALIB_IMAGES),
              "gaussian_sigma1_median": target,
              "db2_medians": {str(k): v for k, v in meds.items()},
              "selected_s0": s0,
              "definition": "d = ||T(h)-h||_F / (||h - mean_spatial(h)||_F + 1e-12), "
                            "per image and channel; median over the 19 layers",
              "caveat": ("this only approximately matches initial perturbation "
                         "strength; it is not a claim of equivalent filtering"),
              "per_layer": per_layer}
    (out_dir / "calibration.json").write_text(json.dumps(report, indent=2))
    del captured, sub, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    torch.set_rng_state(rng_state)                     # leave training RNG untouched
    log("calibration: gaussian median %.4f -> s0=%g (db2 median %.4f)"
        % (target, s0, meds[s0]))
    return s0, report


# --------------------------------------------------------------------------
# Training worker
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, controller, images, labels, pipe, bypass, batch=250):
    was = controller.bypass_all
    controller.bypass_all = bypass
    model.eval()
    tot_ce, correct, n = 0.0, 0, images.shape[0]
    for i in range(0, n, batch):
        xb, yb = images[i:i + batch], labels[i:i + batch]
        logits = model(pipe(xb, 0.0))
        tot_ce += float(F.cross_entropy(logits, yb, reduction="sum"))
        correct += int((logits.argmax(1) == yb).sum())
    model.train()
    controller.bypass_all = was
    return tot_ce / n, correct / n


def train_one(variant, s0, gpu_index, out_dir, shared_path, microbatch=MICROBATCH):
    dev = torch.device("cuda:%d" % gpu_index if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    run_dir = out_dir / variant
    run_dir.mkdir(parents=True, exist_ok=True)

    sh = np.load(shared_path / "shared_indices.npz")
    subset, order = sh["subset"], sh["order"]
    tp_idx, vp_idx = sh["train_probe"], sh["val_probe"]

    root, _ = data_source()
    cfg_data = DataConfig(root=root, download=False)
    bundle = build_dataset(cfg_data).to(dev)
    train_imgs = bundle.train.images[torch.as_tensor(subset, device=dev)]
    train_lbls = bundle.train.labels[torch.as_tensor(subset, device=dev)]
    tp_i = torch.as_tensor(tp_idx, device=dev)
    vp_i = torch.as_tensor(vp_idx, device=dev)
    probe_imgs, probe_lbls = train_imgs[tp_i], train_lbls[tp_i]
    val_imgs, val_lbls = bundle.val.images[vp_i], bundle.val.labels[vp_i]

    norm = ChannelNormalizer(bundle.mean, bundle.std).to(dev)
    pipe = InputPipeline(GaussianSmoothing(sigma_max=1.0), norm)   # inputs unfiltered

    model = build_model(ModelConfig(), bundle.num_classes, seed=SEED).to(dev)
    model.load_state_dict(torch.load(shared_path / "init_weights.pt",
                                 map_location=dev, weights_only=True))
    model.train()
    controller = Controller("none" if variant == "plain" else
                            ("gaussian" if variant == "gaussian" else "db2"), s0)
    handles = attach(model, controller)

    ocfg = OptimConfig(lr=0.1, momentum=0.9, weight_decay=5e-4, batch_size=EFFECTIVE_BATCH,
                       total_steps=UPDATES, lr_schedule="cosine", warmup_steps=WARMUP,
                       min_lr=0.0)
    opt = build_optimizer(model, ocfg)
    accum = EFFECTIVE_BATCH // microbatch

    metrics, deadline, incomplete = [], time.perf_counter() + WORKER_LIMIT_S, False
    t_start = time.perf_counter()

    def snapshot(k):
        cur_ce, _ = evaluate(model, controller, probe_imgs, probe_lbls, pipe, bypass=False)
        byp_ce, _ = evaluate(model, controller, probe_imgs, probe_lbls, pipe, bypass=True)
        v_ce, v_acc = evaluate(model, controller, val_imgs, val_lbls, pipe, bypass=True)
        rec = {"update": k, "lr": lr_at(min(k, UPDATES - 1), ocfg),
               "param": controller.value, "elapsed_s": time.perf_counter() - t_start,
               "train_probe_ce_filtered": cur_ce, "train_probe_ce_bypassed": byp_ce,
               "val_probe_ce": v_ce, "val_probe_acc": v_acc}
        metrics.append(rec)
        log("%-8s k=%3d lr=%.4f param=%s trainCE(f)=%.4f trainCE(b)=%.4f "
            "valCE=%.4f valAcc=%.4f  %.1fs"
            % (variant, k, rec["lr"],
               "-" if rec["param"] is None else "%.3f" % rec["param"],
               cur_ce, byp_ce, v_ce, v_acc, rec["elapsed_s"]))
        with open(run_dir / "metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)

    controller.set_update(0)
    snapshot(0)
    for k in range(UPDATES):
        if time.perf_counter() > deadline:
            incomplete = True
            log("%s: 40-minute limit reached at update %d" % (variant, k))
            break
        controller.set_update(k)                    # fixed across this update's microbatches
        set_lr(opt, lr_at(k, ocfg))
        opt.zero_grad(set_to_none=True)
        idx = torch.as_tensor(order[k], device=dev)
        for a in range(accum):
            sl = idx[a * microbatch:(a + 1) * microbatch]
            loss = F.cross_entropy(model(pipe(train_imgs[sl], 0.0)), train_lbls[sl]) / accum
            loss.backward()
            del loss
        opt.step()
        done = k + 1
        if done % EVAL_EVERY == 0:
            controller.set_update(done)
            snapshot(done)
            controller.set_update(done)

    final = {}
    ce, acc = evaluate(model, controller, bundle.val.images, bundle.val.labels,
                       pipe, bypass=True)
    final = {"variant": variant, "updates_completed": UPDATES if not incomplete else k,
             "incomplete": incomplete, "microbatch": microbatch,
             "accumulation": accum, "s0": s0,
             "controller": controller.describe(),
             "full_val_ce_unfiltered": ce, "full_val_acc_unfiltered": acc,
             "n_val": int(bundle.val.images.shape[0]),
             "wall_seconds": time.perf_counter() - t_start,
             "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"}
    for h in handles:
        h.remove()
    torch.save({"model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
                "update": final["updates_completed"], "variant": variant, "s0": s0},
               run_dir / "checkpoint_final.pt")
    with open(run_dir / "summary.json", "w") as fh:
        json.dump(final, fh, indent=2)
    log("%s DONE: val acc %.4f (unfiltered, %d images) incomplete=%s"
        % (variant, acc, final["n_val"], incomplete))
    return final


def worker(variants, s0, gpu_index, out_dir, shared_path, microbatch):
    for v in variants:
        try:
            train_one(v, s0, gpu_index, out_dir, shared_path, microbatch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log("%s OOM at microbatch %d -> single fallback to %d"
                % (v, microbatch, FALLBACK_MICROBATCH))
            train_one(v, s0, gpu_index, out_dir, shared_path, FALLBACK_MICROBATCH)


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

def make_figure(out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"plain": "#333333", "gaussian": "#1f77b4", "db2": "#d62728"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for v, c in colors.items():
        f = out_dir / v / "metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        k = [r["update"] for r in m]
        axes[0].plot(k, [r["train_probe_ce_bypassed"] for r in m], color=c, label=v)
        axes[1].plot(k, [r["val_probe_ce"] for r in m], color=c, label=v)
        axes[2].plot(k, [r["val_probe_acc"] for r in m], color=c, label=v)
    for ax, t, yl in zip(axes,
                         ["Training-probe CE (filters bypassed)",
                          "Validation CE (unfiltered)",
                          "Validation accuracy (unfiltered)"],
                         ["cross-entropy", "cross-entropy", "accuracy"]):
        ax.axvline(CONT_END, color="gray", ls="--", lw=1)
        ax.annotate("continuation ends", xy=(CONT_END, 1.01),
                    xycoords=("data", "axes fraction"), fontsize=7,
                    ha="center", color="gray")
        ax.set_xlabel("optimizer update")
        ax.set_ylabel(yl)
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("CIFAR-10 pilot (5,000 images, 600 updates, one seed)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = out_dir / "pilot_comparison.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------

def prelim_checks(out_dir: Path):
    """Reuse the existing wavelet/gaussian correctness tests (subset, time-capped)."""
    res = {"ran": False}
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_wavelet.py", "-q", "-x",
             "-k", "roundtrip or identity or fast_matches or constant"],
            capture_output=True, text=True, timeout=110,
            cwd=str(Path(__file__).resolve().parents[1]))   # the shipped repo root
        res = {"ran": True, "returncode": r.returncode,
               "tail": (r.stdout or r.stderr).strip().splitlines()[-3:]}
    except Exception as exc:
        res = {"ran": False, "error": repr(exc)}
    (out_dir / "prelim_checks.json").write_text(json.dumps(res, indent=2))
    log("preliminary checks: %s" % res)
    return res


def main():
    run_dir = WORK / ("pilot_continuation_seed%d_%s"
                      % (SEED, time.strftime("%Y%m%d-%H%M%S")))
    run_dir.mkdir(parents=True, exist_ok=True)
    ngpu = torch.cuda.device_count()
    prov = {"python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "n_gpu": ngpu,
            "gpus": [torch.cuda.get_device_name(i) for i in range(ngpu)],
            "vram_mib": [round(torch.cuda.get_device_properties(i).total_memory / 1024 ** 2)
                         for i in range(ngpu)]}
    try:
        import torchvision
        prov["torchvision"] = torchvision.__version__
    except Exception:
        pass
    (run_dir / "environment.json").write_text(json.dumps(prov, indent=2))
    log("environment: %s" % prov)

    if ngpu == 0:
        # Fail fast rather than spending a 40-minute slot per variant on CPU:
        # this pilot is only meaningful on an accelerator.
        msg = ("aborted: no accelerator was granted (torch=%s, cuda=%s). "
               "Kaggle accepted enable_gpu but scheduled a CPU worker, which "
               "usually means the weekly GPU quota is exhausted. Re-run the same "
               "command when quota is available." % (torch.__version__,
                                                     torch.cuda.is_available()))
        (run_dir / "ABORTED.json").write_text(
            json.dumps({"aborted": True, "reason": msg, "environment": prov}, indent=2))
        log(msg)
        return

    prelim_checks(run_dir)

    bundle, subset, order, tp, vp, init_state = build_shared(run_dir)
    log("subset %d (500/class), order %s, params %s"
        % (subset.size, order.shape, count_parameters(build_model(ModelConfig(), 10, SEED))))

    dev0 = torch.device("cuda:0" if ngpu else "cpu")
    s0, _ = calibrate(bundle, subset, init_state, dev0, run_dir)

    del bundle, init_state
    if ngpu:
        torch.cuda.empty_cache()

    cfg = {"seed": SEED, "n_subset": N_SUBSET, "per_class": PER_CLASS,
           "updates": UPDATES, "continuation_end": CONT_END,
           "effective_batch": EFFECTIVE_BATCH, "microbatch": MICROBATCH,
           "warmup": WARMUP, "eval_every": EVAL_EVERY, "s0": s0,
           "schedule": "r_k = max(1-k/300,0); sigma_k = r_k; s_k = 1-(1-s0) r_k",
           "insertion_points": "19 spatial 3x3 conv outputs, before GN/ReLU",
           "worker_limit_s": WORKER_LIMIT_S, "n_gpu": ngpu}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    if ngpu >= 2:
        log("two GPUs: worker0 = [gaussian, plain] on cuda:0, worker1 = [db2] on cuda:1")
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=worker, args=(["gaussian", "plain"], s0, 0, run_dir,
                                                  run_dir, MICROBATCH)),
                 ctx.Process(target=worker, args=(["db2"], s0, 1, run_dir, run_dir,
                                                  MICROBATCH))]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
    else:
        log("only %d GPU detected: running the three variants sequentially on cuda:0 "
            "(deviation from the two-worker plan, recorded)" % ngpu)
        cfg["deviation"] = "single GPU available; variants run sequentially"
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        worker(["gaussian", "plain", "db2"], s0, 0, run_dir, run_dir, MICROBATCH)

    fig = make_figure(run_dir)
    summary = []
    for v in ("plain", "gaussian", "db2"):
        f = run_dir / v / "summary.json"
        if f.exists():
            summary.append(json.loads(f.read_text()))
    (run_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    log("figure: %s" % fig)
    for r in summary:
        log("SUMMARY %-8s updates=%d incomplete=%s val_acc=%.4f val_ce=%.4f"
            % (r["variant"], r["updates_completed"], r["incomplete"],
               r["full_val_acc_unfiltered"], r["full_val_ce_unfiltered"]))
    log("run directory: %s" % run_dir)


if __name__ == "__main__":
    main()
