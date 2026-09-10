"""Per-layer sigma study.  Runs on CPU (laptop pilot) or a CUDA device.

Isolates the per-site coefficient ``c_l`` from every other axis: constant input
resolution 32 (so ``q_l == 1`` everywhere and the effective width is exactly
``c_l * G(e)``), no augmentation, no progressive resolution, one seed.

Four arms share initial weights, BN buffers, the training subset, the test
subset and every per-epoch permutation.  Those are hashed once and each arm
re-checks the digests before its first update, so an unpaired comparison aborts
rather than producing numbers.

    plain     no filtering at all
    rho1      uniform sigma at all 19 sites -- the current CBS-style schedule
    rho0.5    c = (1, 1/2, 1/4) across stem+stage1 / stage2 / stage3
              == width_l / 32, the constant-physical-scale profile
    rho2      c = (1/4, 1/2, 1) -- blur deep, not shallow

One seed cannot separate these; the GPU campaign's measured single-seed noise
floor is ~0.1 pp on final accuracy and this run is smaller and shorter, so its
own floor is wider and unmeasured here.  Read it as a mechanism check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from continuation.adaptive import (AdaptiveSiteController, GradNormTracker,
                                   SensitivityStepper)
from continuation.campaign_ops import (N_SITES, SiteController, attach_sites,
                                       per_stage_coeffs)
from continuation.config import DataConfig, ModelConfig, OptimConfig
from continuation.data import build_dataset
from continuation.models import build_model
from continuation.optim import build_optimizer, lr_at, set_lr
from continuation.pipeline import ChannelNormalizer, InputPipeline
from continuation.transforms.gaussian import GaussianSmoothing

T0 = time.perf_counter()

# the campaign's 7-block plateau, resampled onto the filtered epochs
PLATEAU = [1.00, 0.85, 0.70, 0.60, 0.50, 0.40, 0.30]


def log(msg):
    print("[%7.1fs] %s" % (time.perf_counter() - T0, msg), flush=True)


def sha(obj) -> str:
    if isinstance(obj, torch.Tensor):
        b = obj.detach().cpu().numpy().tobytes()
    elif isinstance(obj, np.ndarray):
        b = np.ascontiguousarray(obj).tobytes()
    else:
        b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(b).hexdigest()


def plateau_schedule(epochs: int, filtered_epochs: int) -> list:
    """Resample the plateau onto ``filtered_epochs``, then hold exactly 0."""
    out = []
    for e in range(epochs):
        if e >= filtered_epochs:
            out.append(0.0)
        else:
            out.append(PLATEAU[min(e * len(PLATEAU) // filtered_epochs,
                                   len(PLATEAU) - 1)])
    return out


def stratified_subset(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Equal per class, deterministic given the seed."""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per = n // len(classes)
    picks = []
    for c in classes:
        idx = np.flatnonzero(labels == c)
        picks.append(rng.choice(idx, size=per, replace=False))
    out = np.sort(np.concatenate(picks))
    return out


def grad_global_norm(model) -> float:
    """L2 norm of the full parameter gradient, read between backward and step."""
    tot = 0.0
    for p in model.parameters():
        if p.grad is not None:
            tot += float(p.grad.detach().pow(2).sum())
    return tot ** 0.5


@torch.no_grad()
def evaluate(model, ctrl, pipe, images, labels, row, bypass, batch=500):
    """Evaluate at an explicit per-site row.  BN statistics untouched."""
    prev_mode = model.training
    prev = (None if ctrl.row is None else list(ctrl.row),
            ctrl.resolution, ctrl.bypass_all)
    ctrl.bypass_all = bypass
    ctrl.set_state_row(row, ctrl.resolution)
    model.eval()
    tot, correct, n = 0.0, 0, int(images.shape[0])
    for i in range(0, n, batch):
        logits = model(pipe(images[i:i + batch], 0.0))
        tot += float(F.cross_entropy(logits, labels[i:i + batch], reduction="sum"))
        correct += int((logits.argmax(1) == labels[i:i + batch]).sum())
    ctrl.row, ctrl.resolution, ctrl.bypass_all = prev
    ctrl.q = ctrl.q_for(ctrl.resolution)
    if prev_mode:
        model.train()
    return tot / n, correct / n


def train_arm(arm, shared, cfg, out_dir: Path) -> dict:
    name = arm["name"]
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- pairing: refuse to train on anything but the shared assets ----
    for k, want in shared["digests"].items():
        got = sha(shared["assets"][k])
        if got != want:
            raise SystemExit("pairing broken for %r in arm %s: %s != %s"
                             % (k, name, got, want))

    tr_imgs, tr_lbls = shared["tr_imgs"], shared["tr_lbls"]
    te_imgs, te_lbls = shared["te_imgs"], shared["te_lbls"]
    probe_imgs, probe_lbls = shared["probe_imgs"], shared["probe_lbls"]
    perms = shared["assets"]["perms"]

    dev = shared["device"]
    model = build_model(ModelConfig(arch="resnet20_bn_cifar"), 10, seed=cfg.seed)
    model.load_state_dict(shared["init_state"])
    model = model.to(dev)
    model.train()

    adaptive = arm.get("adaptive", False)
    if adaptive:
        # Start uniform: the controller must DISCOVER a depth profile, not be
        # handed one.  sigma_init = 1.0 everywhere matches rho1's start and
        # rho2's peak, so no arm is favoured by its starting point.
        ctrl = AdaptiveSiteController(
            [cfg.sigma_init] * N_SITES,
            tracker=GradNormTracker(beta=cfg.ema_beta, tol=cfg.plateau_tol,
                                    patience=cfg.patience,
                                    min_steps=cfg.min_stage_steps),
            stepper=SensitivityStepper(delta_ref=cfg.delta_ref,
                                       dmin=cfg.dmin, dmax=cfg.dmax))
    else:
        ctrl = SiteController(
            arm["operator"],
            levels=arm.get("levels"),
            site_coeffs=arm.get("site_coeffs"),
            profile=arm.get("profile"),
            # every real training run must end on the exact target objective
            require_terminal_identity=(0 if arm["operator"] == "none"
                                       else cfg.epochs - cfg.filtered_epochs),
        )
    handles = attach_sites(model, ctrl)

    pipe = InputPipeline(GaussianSmoothing(sigma_max=1.0),
                         ChannelNormalizer(shared["mean"], shared["std"]))
    ocfg = OptimConfig(lr=cfg.lr, momentum=0.9, weight_decay=5e-4,
                       batch_size=cfg.batch, total_steps=shared["total_updates"],
                       lr_schedule="cosine", warmup_steps=cfg.warmup, min_lr=0.0)
    opt = build_optimizer(model, ocfg)

    metrics, gstep, train_seconds = [], 0, 0.0

    def snapshot(done_epochs):
        last = max(done_epochs - 1, 0)
        if adaptive:
            row = list(ctrl.realised[last]) if last < len(ctrl.realised) else list(ctrl.row)
        else:
            row = None if ctrl.L is None else list(ctrl.L[last])
        active = ctrl.L is not None and any(v > 0 for v in row)
        # current path: the configuration the weights and BN buffers were
        # actually trained under.  Primary.
        cce, cacc = evaluate(model, ctrl, pipe, te_imgs, te_lbls,
                             row, bypass=not active, batch=cfg.eval_batch)
        # target path: filters bypassed.  While filtering is on this measures a
        # premature configuration change including BN mismatch -- diagnostic only.
        tce, tacc = evaluate(model, ctrl, pipe, te_imgs, te_lbls, None, bypass=True,
                             batch=cfg.eval_batch)
        pce, pacc = evaluate(model, ctrl, pipe, tr_imgs[:2000], tr_lbls[:2000],
                             None, bypass=True, batch=cfg.eval_batch)
        metrics.append({"epoch": done_epochs, "update": gstep,
                        "row": row, "filters_active": bool(active),
                        "test_ce_current": cce, "test_acc_current": cacc,
                        "test_ce_target": tce, "test_acc_target": tacc,
                        "train_probe_ce_target": pce,
                        "train_probe_acc_target": pacc})
        log("%-8s ep=%2d/%d lvl=%-24s | cur acc=%.4f ce=%.4f | tgt acc=%.4f"
            % (name, done_epochs, cfg.epochs,
               "-" if row is None else "%.2f/%.2f/%.2f" % (row[0], row[7], row[13]),
               cacc, cce, tacc))

    snapshot(0)
    n = int(tr_imgs.shape[0])
    steps_log = []
    deadline = cfg.epochs - cfg.terminal_epochs - cfg.ramp_stages
    for e in range(cfg.epochs):
        if adaptive and e > 0 and not ctrl.all_zero():
            # Stage boundaries are quantised to epochs so the row is constant
            # within an epoch and the realised table replays exactly.
            forced = e >= deadline
            if forced or ctrl.tracker.should_step():
                g = ctrl.measure(model, pipe, probe_imgs, probe_lbls)
                # +1 so stages_left hits 1 at the FIRST bypass epoch: the fixed
                # arms filter epochs 0..(E-K-1) inclusive, and an off-by-one here
                # cost the adaptive arm three filtered epochs last run.
                rec = ctrl.advance(
                    g, stages_left=max(cfg.epochs - cfg.terminal_epochs - e + 1, 1),
                    force_ramp=forced)
                rec["epoch"] = e
                rec["corrector_steps"] = ctrl.tracker.steps
                rec["grad_norm_ema"] = ctrl.tracker.ema
                steps_log.append(rec)
                log("%-8s sigma step %d at ep%d%s -> %.2f/%.2f/%.2f"
                    % (name, rec["step"], e, " (FORCED RAMP)" if forced else "",
                       ctrl.row[0], ctrl.row[7], ctrl.row[13]))
                ctrl.tracker.reset_stage()
        ctrl.set_epoch(e)
        t0 = time.perf_counter()
        perm = perms[e]
        for s0 in range(0, n, cfg.batch):
            idx = torch.as_tensor(perm[s0:s0 + cfg.batch], dtype=torch.long,
                                  device=dev)
            set_lr(opt, lr_at(gstep, ocfg))
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(pipe(tr_imgs[idx], 0.0)), tr_lbls[idx]).backward()
            if adaptive:
                ctrl.tracker.update(grad_global_norm(model))
            opt.step()
            gstep += 1
        train_seconds += time.perf_counter() - t0
        snapshot(e + 1)
        ctrl.set_epoch(e)

    for h in handles:
        h.remove()
    last = metrics[-1]
    summary = {
        "arm": name, "seed": cfg.seed, "operator": arm["operator"],
        "profile": arm.get("profile"),
        "n_train": n, "n_test": int(te_imgs.shape[0]),
        "epochs": cfg.epochs, "updates": gstep, "batch": cfg.batch,
        "lr": cfg.lr, "warmup": cfg.warmup, "probe_size": cfg.probe_size,
        "final_test_acc": last["test_acc_target"],
        "final_test_ce": last["test_ce_target"],
        "final_train_probe_acc": last["train_probe_acc_target"],
        "final_train_probe_ce": last["train_probe_ce_target"],
        "device": str(dev),
        "train_seconds": round(train_seconds, 1),
        "seconds_per_update": round(train_seconds / max(gstep, 1), 4),
        "controller": ctrl.describe(),
        "shared_digests": shared["digests"],
    }
    if not adaptive and ctrl.L is not None:
        per_epoch_updates = (int(tr_imgs.shape[0]) + cfg.batch - 1) // cfg.batch
        summary["filtered_updates"] = per_epoch_updates * sum(
            1 for r in ctrl.L if any(v > 0 for v in r))
    if adaptive:
        # The homotopy contract, checked on the realised path rather than assumed.
        assert ctrl.all_zero(), "adaptive arm did not reach sigma = 0"
        for e in range(cfg.epochs - cfg.terminal_epochs, cfg.epochs):
            assert all(v == 0.0 for v in ctrl.realised[e]), \
                "terminal epoch %d is not the exact target objective" % e
        # updates spent with any filter active, so the arms can be checked for
        # the curriculum-vs-plain confound rather than assumed free of it
        rows = ctrl.realised
        per_epoch_updates = (int(tr_imgs.shape[0]) + cfg.batch - 1) // cfg.batch
        summary["filtered_updates"] = per_epoch_updates * sum(
            1 for r in rows if any(v > 0 for v in r))
        summary["filtered_updates_per_site"] = [
            per_epoch_updates * sum(1 for r in rows if r[l] > 0)
            for l in range(N_SITES)]
        summary["sigma_steps"] = steps_log
        summary["deadline_fired"] = ctrl.deadline_fired
        summary["realised_table"] = ctrl.realised
        (run_dir / "realised_table.json").write_text(json.dumps(ctrl.realised, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def resolve_data_root(explicit=None, work="/kaggle/working") -> str:
    """Return a directory containing ``cifar-10-batches-py/``.

    Three layouts are accepted, because a Kaggle dataset built from a flat file
    list mounts the batch files directly rather than inside the directory name
    torchvision expects.  In that case the expected layout is rebuilt by
    symlinking into a writable directory -- the batch files themselves are never
    copied or rewritten, so ``download=False`` still verifies the official md5s
    against the uploaded bytes.
    """
    import glob
    if explicit:
        return explicit
    if (Path("data") / "cifar-10-batches-py").exists():
        return "data"
    for hit in sorted(glob.glob("/kaggle/input/**/cifar-10-batches-py",
                                recursive=True)):
        return str(Path(hit).parent)
    for hit in sorted(glob.glob("/kaggle/input/**/data_batch_1", recursive=True)):
        mount = Path(hit).parent
        root = Path(work) / "cifar_root"
        (root / "cifar-10-batches-py").parent.mkdir(parents=True, exist_ok=True)
        link = root / "cifar-10-batches-py"
        if not link.exists():
            link.symlink_to(mount, target_is_directory=True)
        return str(root)
    raise SystemExit("CIFAR-10 not found; attach the dataset or populate data/")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--filtered-epochs", type=int, default=9)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    # Batch order only.  Init weights, subset and test set stay pinned to --seed,
    # so a second run at a different --order-seed measures run-to-run spread at
    # this exact scale and nothing else.
    ap.add_argument("--order-seed", type=int, default=None)
    ap.add_argument("--terminal-epochs", type=int, default=3)
    ap.add_argument("--ramp-stages", type=int, default=2)
    ap.add_argument("--sigma-init", type=float, default=1.0)
    # delta_ref multiplies the schedule-aware base step (row/stages_left).
    # It is NOT the pre-phi loss budget; 0.15 here silently clipped every
    # site to dmin and emitted a uniform ramp.  dmin must stay well below
    # base/kappa or the sensitivity range cannot be expressed at all.
    ap.add_argument("--delta-ref", type=float, default=1.0)
    ap.add_argument("--dmin", type=float, default=0.005)
    ap.add_argument("--dmax", type=float, default=0.40)
    ap.add_argument("--ema-beta", type=float, default=0.9)
    ap.add_argument("--plateau-tol", type=float, default=0.005)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--min-stage-steps", type=int, default=40)
    ap.add_argument("--probe-size", type=int, default=512)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--eval-batch", type=int, default=500)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="results/per_layer_cpu")
    ap.add_argument("--arms", default="plain,rho1,rho0.5,rho2")
    cfg = ap.parse_args(argv)
    if cfg.device is None:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.order_seed is None:
        cfg.order_seed = cfg.seed
    if cfg.threads:
        torch.set_num_threads(cfg.threads)

    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)

    root = resolve_data_root(cfg.data_root)
    log("loading CIFAR-10 from %s (download=False, official md5s verified)" % root)
    bundle = build_dataset(DataConfig(root=root, download=False, num_val=0))

    tr_labels = bundle.train.labels.numpy()
    subset = stratified_subset(tr_labels, cfg.n_train, seed=cfg.seed)
    test_idx = np.sort(np.random.default_rng(cfg.seed + 1).choice(
        len(bundle.test), size=cfg.n_test, replace=False))

    tr = bundle.train.subset(subset, "train_subset")
    te = bundle.test.subset(test_idx, "test_subset")

    n = len(tr)
    per_epoch = (n + cfg.batch - 1) // cfg.batch
    total_updates = per_epoch * cfg.epochs
    rng = np.random.default_rng(1000 + cfg.order_seed)
    perms = np.stack([rng.permutation(n) for _ in range(cfg.epochs)])

    # fixed sensitivity probe, drawn once and reused for every measurement so
    # successive sensitivities are comparable rather than confounded with
    # minibatch noise
    probe_idx = np.sort(np.random.default_rng(2000 + cfg.seed).choice(
        n, size=min(cfg.probe_size, n), replace=False))

    # normalization fitted on the unfiltered training subset, shared by all arms
    x = tr.images.to(torch.float32) / 255.0
    mean = x.mean(dim=(0, 2, 3))
    std = x.std(dim=(0, 2, 3))
    del x

    init_model = build_model(ModelConfig(arch="resnet20_bn_cifar"), 10, seed=cfg.seed)
    init_state = init_model.state_dict()
    init_blob = torch.cat([v.flatten().float() for v in init_state.values()])

    assets = {"subset": subset, "test_idx": test_idx, "perms": perms,
              "probe_idx": probe_idx, "init": init_blob}
    digests = {k: sha(v) for k, v in assets.items()}
    log("shared assets pinned: " + ", ".join("%s=%s" % (k, v[:12])
                                             for k, v in digests.items()))

    levels = plateau_schedule(cfg.epochs, cfg.filtered_epochs)
    log("schedule G(e) = %s" % levels)

    catalogue = {
        "plain": {"name": "plain", "operator": "none"},
        "rho1": {"name": "rho1", "operator": "gaussian", "levels": levels,
                 "site_coeffs": per_stage_coeffs(1.0),
                 "profile": ["per_stage", {"rho": 1.0}]},
        "rho0.5": {"name": "rho0.5", "operator": "gaussian", "levels": levels,
                   "site_coeffs": per_stage_coeffs(0.5),
                   "profile": ["per_stage", {"rho": 0.5}]},
        "rho2": {"name": "rho2", "operator": "gaussian", "levels": levels,
                 "site_coeffs": per_stage_coeffs(2.0),
                 "profile": ["per_stage", {"rho": 2.0}]},
        "adaptive": {"name": "adaptive", "operator": "gaussian", "adaptive": True,
                     "profile": ["adaptive_predictor_corrector", {}]},
    }

    dev = torch.device(cfg.device)
    probe_t = torch.as_tensor(probe_idx, dtype=torch.long)
    tr_imgs_d, tr_lbls_d = tr.images.to(dev), tr.labels.to(dev)
    te_imgs_d, te_lbls_d = te.images.to(dev), te.labels.to(dev)
    shared = {"assets": assets, "digests": digests, "init_state": init_state,
              "device": dev,
              "probe_imgs": tr_imgs_d[probe_t.to(dev)],
              "probe_lbls": tr_lbls_d[probe_t.to(dev)],
              "tr_imgs": tr_imgs_d, "tr_lbls": tr_lbls_d,
              "te_imgs": te_imgs_d, "te_lbls": te_lbls_d,
              "mean": mean.to(dev), "std": std.to(dev),
              "total_updates": total_updates}

    log("device=%s | n_train=%d n_test=%d epochs=%d updates=%d batch=%d lr=%g"
        % (cfg.device, n, len(te), cfg.epochs, total_updates, cfg.batch, cfg.lr))
    if dev.type == "cuda":
        log("gpu=%s" % torch.cuda.get_device_name(dev))
    log("seed=%d order_seed=%d probe=%d | adaptive: deadline=ep%d terminal=%d"
        % (cfg.seed, cfg.order_seed, len(probe_idx),
           cfg.epochs - cfg.terminal_epochs - cfg.ramp_stages, cfg.terminal_epochs))

    summaries = []
    for key in cfg.arms.split(","):
        arm = catalogue[key.strip()]
        log("=== arm %s ===" % arm["name"])
        summaries.append(train_arm(arm, shared, cfg, out_dir))

    (out_dir / "summaries.json").write_text(json.dumps(summaries, indent=2))

    base = next((s for s in summaries if s["arm"] == "plain"), None)
    log("")
    log("%-8s %8s %8s %9s %8s" % ("arm", "acc", "ce", "vs plain", "s/upd"))
    for s in summaries:
        delta = ("%+.2f pp" % (100 * (s["final_test_acc"] - base["final_test_acc"]))
                 if base else "-")
        log("%-8s %8.4f %8.4f %9s %8.3f"
            % (s["arm"], s["final_test_acc"], s["final_test_ce"], delta,
               s["seconds_per_update"]))
    log("wrote %s" % (out_dir / "summaries.json"))


if __name__ == "__main__":
    main()
