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
from continuation.adaptive import (AdaptiveSiteController, GradNormTracker,
                                   SensitivityStepper)
from continuation.gap_trigger import GapTriggerConfig, TransferGapTracker
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
    # Save the per-site row: with a level table the controller's state is a
    # 19-vector and ``value`` is a derived scalar, so restoring ``value`` would
    # leak this evaluation's uniform level into training.
    prev = (None if ctrl.row is None else list(ctrl.row),
            ctrl.resolution, ctrl.bypass_all)
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
    ctrl.row, ctrl.resolution, ctrl.bypass_all = prev
    ctrl.q = ctrl.q_for(ctrl.resolution)
    if prev_mode:
        model.train()
    return tot / n, correct / n


@torch.no_grad()
def eval_paths(model, ctrl, images, labels, pipe, batch=500):
    """(current, target) probe CE at the controller's actual per-site row.

    ``evaluate`` above takes a *uniform* level, which cannot represent an
    adaptive 19-vector, so the current path is measured with the row left in
    place.  Mode, row, resolution, bypass flag and q are all restored, and
    BatchNorm buffers are never updated.
    """
    prev_mode = model.training
    prev = (None if ctrl.row is None else list(ctrl.row),
            ctrl.resolution, ctrl.bypass_all)
    model.eval()

    def _ce():
        res_in, tot, n = ctrl.input_resolution(), 0.0, images.shape[0]
        for i in range(0, n, batch):
            logits = model(pipe(images[i:i + batch], 0.0, res=res_in))
            tot += float(F.cross_entropy(logits, labels[i:i + batch],
                                         reduction="sum"))
        return tot / n

    ctrl.bypass_all = False
    cur = _ce()                                   # row as trained
    ctrl.set_state(0.0, 32)
    ctrl.bypass_all = True
    tgt = _ce()                                   # exact target endpoint
    ctrl.row, ctrl.resolution, ctrl.bypass_all = prev
    ctrl.q = ctrl.q_for(ctrl.resolution)
    if prev_mode:
        model.train()
    return cur, tgt


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

    from scripts.campaign_manifest import GAUSSIAN, RESOLUTIONS
    sites = EARLY7 if cell["mask"] == "early7" else tuple(range(N_SITES))
    # An explicit table overrides the manifest lookup: the manifest's tables are
    # 30 entries long and a different horizon needs its own.
    levels_tab = cell.get("gaussian_table", GAUSSIAN[cell["gaussian"]])
    res_tab = cell.get("resolution_table", RESOLUTIONS[cell["resolution"]])
    adaptive = bool(cell.get("adaptive"))
    if adaptive:
        # The row is walked, not read from a table: sigma starts uniform and each
        # predictor step is scaled by the measured per-site sensitivity.
        a = cell.get("adaptive_params", {})
        ctrl = AdaptiveSiteController(
            sigma_init=[float(a.get("sigma_init", 1.0))] * N_SITES,
            sites=sites,
            resolution_by_epoch=res_tab,
            reduction=cell["reduction"],
            tracker=GradNormTracker(**a.get("tracker", {})),
            stepper=SensitivityStepper(**a.get("stepper", {})))
    else:
        ctrl = SiteController(operator=cell["operator"],
                              levels=levels_tab, sites=sites,
                              resolution_by_epoch=res_tab,
                              reduction=cell["reduction"])
    handles = attach_sites(model, ctrl)
    # Fixed, deterministic batch for the sensitivity probe: the same examples at
    # every predictor step, so successive dL/dsigma values are comparable.
    n_sens = int(cell.get("adaptive_params", {}).get("sens_batch", 256))
    sens_imgs, sens_lbls = probe_imgs[:n_sens], probe_lbls[:n_sens]
    ramp_from = int(cell.get("adaptive_params", {}).get("ramp_from", 18))
    zero_by = int(cell.get("adaptive_params", {}).get("zero_by", 21))
    predictor_log = []
    trigger_kind = cell.get("trigger", "grad_norm")
    adaptive_res = bool(cell.get("adaptive_resolution"))
    res_stages = [int(v) for v in cell.get("res_stages", [16, 24, 32])]
    # latest epoch by which transition i must have happened; the trigger may
    # move earlier OR later than a fixed table, but never past these.
    res_force_by = [int(v) for v in cell.get("res_force_by", [12, 21])]
    res_idx, res_log, res_realised = [0], [], []
    gap_tracker = (TransferGapTracker(GapTriggerConfig(
        **cell.get("adaptive_params", {}).get("gap", {})))
        if trigger_kind == "gap" and (adaptive or adaptive_res) else None)
    gap_log = []
    log_grad_norm = bool(cell.get("log_grad_norm"))
    # (update, epoch, ||g||_all, lr, ||g||_conv, ||w||_conv, ||w||_all)
    grad_trace = []
    per_layer_trace = []                  # (update, epoch, [||g_l||], [||w_l||])
    gap_trace = []                        # (update, epoch, cur_ce, tgt_ce, cur/tgt acc)
    prev_flat_g = prev_gn = prev_theta = None
    conv_params = [q for q in model.parameters() if q.dim() == 4]
    conv_names = [n for n, q in model.named_parameters() if q.dim() == 4]

    epochs = int(cell.get("epochs", PROTOCOL["epochs"]))
    n, B, mb = int(subset.size), PROTOCOL["effective_batch"], PROTOCOL["microbatch"]
    per_epoch = (n + B - 1) // B
    total_updates = epochs * per_epoch
    ocfg = OptimConfig(lr=float(cell.get("lr", PROTOCOL["lr"])),
                       momentum=PROTOCOL["momentum"],
                       weight_decay=PROTOCOL["weight_decay"], batch_size=B,
                       total_steps=total_updates, lr_schedule="cosine",
                       warmup_steps=PROTOCOL["warmup"], min_lr=0.0)
    opt = build_optimizer(model, ocfg)

    ev_every = int(cell.get("eval_every", 2))
    extra = list(cell.get("eval_extra", [21]))
    eval_epochs = sorted(set(list(range(0, epochs + 1, ev_every)) + extra + [epochs]))
    ckpt_epochs = set(cell.get("checkpoint_epochs", [6, 12, 21, epochs]))
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
    def apply_res(ep):
        """Override the table-driven resolution with the controller's stage.

        Deadlines are enforced here so a stalled trigger can never prevent the
        run from reaching the target resolution.
        """
        if not adaptive_res:
            return
        while (res_idx[0] < len(res_stages) - 1
               and ep >= res_force_by[res_idx[0]]):
            res_idx[0] += 1
            res_log.append({"epoch": ep, "to": res_stages[res_idx[0]],
                            "trigger": "deadline"})
        ctrl.resolution = res_stages[res_idx[0]]
        ctrl.q = ctrl.q_for(ctrl.resolution)
        while len(res_realised) <= int(ep):
            res_realised.append(None)
        res_realised[int(ep)] = res_stages[res_idx[0]]

    for e in range(start_epoch, epochs):
        ctrl.set_epoch(e)
        apply_res(e)
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
            if adaptive or log_grad_norm:
                # Corrector residual at theta_k: .grad still holds the
                # accumulated training gradient (step does not clear it).
                gn = torch.sqrt(sum((q.grad.detach() ** 2).sum()
                                    for q in model.parameters()
                                    if q.grad is not None)).item()
            if log_grad_norm:
                # Instrumentation only -- no controller, no behaviour change.
                # For a BatchNorm net the loss is scale invariant per layer, so
                # ||g|| ~ 1/||w|| and the raw norm tracks weight decay rather
                # than convergence.  The scale-invariant combination is the
                # PRODUCT ||g||.||w||, so both factors are recorded.  Conv
                # kernels (4-d weights) are the scale-invariant tensors; BN
                # affines (1-d) and the classifier (2-d) are not, and are kept
                # separate rather than pooled in.
                gc = torch.sqrt(sum((q.grad.detach() ** 2).sum()
                                    for q in conv_params
                                    if q.grad is not None)).item()
                wc = torch.sqrt(sum((q.detach() ** 2).sum()
                                    for q in conv_params)).item()
                wa = torch.sqrt(sum((q.detach() ** 2).sum()
                                    for q in model.parameters())).item()
                grad_trace.append((gstep, e, round(gn, 8),
                                   round(lr_at(gstep, ocfg), 10),
                                   round(gc, 8), round(wc, 8), round(wa, 8)))
                # --- scale-invariant progress signals -------------------
                # Gradient cosine and relative update size are invariant to the
                # weight scale, so neither inherits the ||g|| ~ 1/||w|| drift
                # that made every magnitude signal useless.
                fg = torch.cat([q.grad.detach().reshape(-1)
                                for q in model.parameters() if q.grad is not None])
                fgn = fg.norm()
                cos = (float((prev_flat_g @ fg) / (prev_gn * fgn + 1e-30))
                       if prev_flat_g is not None else float("nan"))
                prev_flat_g, prev_gn = fg, fgn
                with torch.no_grad():
                    th = torch.cat([q.detach().reshape(-1)
                                    for q in model.parameters()])
                    rel = (float((th - prev_theta).norm() / (th.norm() + 1e-30))
                           if prev_theta is not None else float("nan"))
                    prev_theta = th
                grad_trace[-1] = grad_trace[-1] + (round(cos, 8), round(rel, 10))
                # --- transfer gap: is more training HERE still closing the
                # distance to the TARGET objective?  evaluate() restores mode
                # and controller state and never updates BN buffers.
                if gstep % 100 == 0:
                    lvl = ctrl.levels[e] if ctrl.levels else None
                    cce, cacc = evaluate(model, ctrl, probe_imgs, probe_lbls,
                                         pipe, lvl, ctrl.resolution)
                    tce, tacc = evaluate(model, ctrl, probe_imgs, probe_lbls,
                                         pipe, 0.0, 32)
                    gap_trace.append((gstep, e, round(cce, 6), round(tce, 6),
                                      round(cacc, 6), round(tacc, 6)))
                if gstep % 50 == 0:
                    per_layer_trace.append(
                        (gstep, e,
                         [round(float(q.grad.detach().norm()), 8)
                          if q.grad is not None else 0.0 for q in conv_params],
                         [round(float(q.detach().norm()), 8) for q in conv_params]))
            if gap_tracker is not None and adaptive_res:
                # Measure only; the decision is taken at the epoch boundary,
                # because the resolution itself can only change there.
                if gstep % gap_tracker.cfg.cadence == 0:
                    cce, tce = eval_paths(model, ctrl, probe_imgs, probe_lbls, pipe)
                    gap_log.append((gstep, e, round(cce, 6), round(tce, 6),
                                    round(tce - cce, 6)))
                    gap_tracker.observe(gstep, tce - cce)
            elif gap_tracker is not None:
                if gstep % gap_tracker.cfg.cadence == 0:
                    cce, tce = eval_paths(model, ctrl, probe_imgs, probe_lbls, pipe)
                    gap = tce - cce
                    gap_log.append((gstep, e, round(cce, 6), round(tce, 6),
                                    round(gap, 6)))
                    gap_tracker.observe(gstep, gap)
                if gap_tracker.should_step(gstep) and not ctrl.all_zero():
                    grads = ctrl.measure(model, pipe, sens_imgs, sens_lbls)
                    rec = ctrl.advance(grads)
                    rec.update({"epoch": e, "update": gstep,
                                "step_size": max(rec["before"]) - max(rec["after"]),
                                "gap": gap_log[-1][4] if gap_log else None,
                                "trigger": gap_tracker.reason})
                    predictor_log.append(rec)
                    gap_tracker.reset_stage(gstep)
            elif adaptive:
                ctrl.tracker.update(gn)
                if ctrl.tracker.should_step() and not ctrl.all_zero():
                    grads = ctrl.measure(model, pipe, sens_imgs, sens_lbls)
                    rec = ctrl.advance(grads)
                    rec.update({"epoch": e, "update": gstep, "grad_norm": gn,
                                "ema": ctrl.tracker.ema, "trigger": "plateau"})
                    predictor_log.append(rec)
                    ctrl.tracker.reset_stage()
            gstep += 1
        if adaptive_res and gap_tracker is not None:
            if (gap_tracker.should_step(gstep)
                    and res_idx[0] < len(res_stages) - 1):
                res_idx[0] += 1
                res_log.append({"epoch": e + 1, "update": gstep,
                                "to": res_stages[res_idx[0]],
                                "gap": gap_log[-1][4] if gap_log else None,
                                "trigger": gap_tracker.reason})
                gap_tracker.reset_stage(gstep)
        if adaptive and (e + 1) >= ramp_from and not ctrl.all_zero():
            if gap_tracker is not None:
                gap_tracker.reset_stage(gstep)
            # Deadline: override the adaptive rule with a linear ramp so every
            # site is exactly zero by ``zero_by``, leaving the terminal epochs on
            # the exact target objective.  A run that reaches here is partly a
            # fixed schedule and records ``deadline_fired``.
            grads = ctrl.measure(model, pipe, sens_imgs, sens_lbls)
            rec = ctrl.advance(grads, stages_left=max(1, zero_by - (e + 1)),
                               force_ramp=True)
            rec.update({"epoch": e, "update": gstep, "trigger": "deadline_ramp"})
            predictor_log.append(rec)
            ctrl.tracker.reset_stage()
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

    if log_grad_norm:
        (run_dir / "grad_norm_trace.json").write_text(json.dumps(
            {"columns": ["update", "epoch", "grad_norm", "lr",
                         "g_conv", "w_conv", "w_all", "grad_cosine",
                         "rel_update"],
             "note": ("BN makes the loss scale invariant per layer, so ||g|| ~ "
                      "1/||w||; the scale-invariant signal is g_conv * w_conv"),
             "conv_params": conv_names, "n_conv": len(conv_params),
             "sigma_by_epoch": levels_tab,
             "n": len(grad_trace), "rows": grad_trace}))
        (run_dir / "per_layer_trace.json").write_text(json.dumps(
            {"columns": ["update", "epoch", "grad_norms", "weight_norms"],
             "conv_params": conv_names, "cadence": 50,
             "n": len(per_layer_trace), "rows": per_layer_trace}))
        (run_dir / "transfer_gap_trace.json").write_text(json.dumps(
            {"columns": ["update", "epoch", "cur_ce", "tgt_ce",
                         "cur_acc", "tgt_acc"],
             "note": ("cur = configuration actually trained under; tgt = sigma 0 "
                      "at 32x32. gap = tgt_ce - cur_ce. The trigger question is "
                      "d(gap)/dt at fixed sigma, not the gap level."),
             "cadence": 100, "probe_n": int(probe_imgs.shape[0]),
             "sigma_by_epoch": levels_tab,
             "n": len(gap_trace), "rows": gap_trace}))
        log("wrote grad_norm_trace (%d), per_layer_trace (%d), transfer_gap (%d)"
            % (len(grad_trace), len(per_layer_trace), len(gap_trace)))

    last = metrics[-1]
    wall = time.perf_counter() - t_start
    summary = {**cell, "job": job, "worker": worker, "gpu_index": gpu,
               "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
               "epochs": epochs, "lr": float(cell.get("lr", PROTOCOL["lr"])),
               "updates": gstep, "updates_per_epoch": per_epoch,
               "n_train": n, "n_test": int(test_imgs.shape[0]),
               **{k: PROTOCOL[k] for k in ("warmup", "effective_batch", "microbatch")},
               "controller": ctrl.describe(),
               "adaptive": adaptive, "trigger_kind": trigger_kind,
               "gap_trigger": None if gap_tracker is None else gap_tracker.describe(),
               "gap_log": gap_log,
               "adaptive_resolution": adaptive_res,
               "res_stages": res_stages, "res_force_by": res_force_by,
               "res_log": res_log, "res_realised": res_realised,
               "predictor_log": predictor_log,
               "normalization": {"mean": [float(v) for v in bundle.mean],
                                 "std": [float(v) for v in bundle.std]},
               "final_test_acc": last["test_acc_target"],
               "final_test_ce": last["test_ce_target"],
               "final_train_probe_acc": last["train_probe_acc_target"],
               "final_train_probe_ce": last["train_probe_ce_target"],
               "wall_seconds": wall, "eval_seconds": eval_seconds[0],
               "train_seconds": wall - eval_seconds[0],
               "peak_mem_mib": (torch.cuda.max_memory_allocated(dev) / 1024 ** 2
                                if dev.type == "cuda" else None)}
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


def run_job(job_index: int, cells: list):
    out_dir = WORK / ("campaign_job%d_%s" % (job_index,
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
