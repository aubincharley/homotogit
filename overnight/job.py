"""One Kaggle kernel of the follow-up: ``pilot``, ``train`` (a fixed queue per GPU) or ``evaluate_old``.

    python -m overnight.job train --queues '[["run", ...], ["run", ...]]' --out /kaggle/working/ovn [--hours 11]
    python -m overnight.job evaluate_old --out /kaggle/working/ovn
    python -m overnight.job pilot --out /kaggle/working/ovn

Order: CUDA guard -> environment -> input digests (assets, frozen data-order extension, frozen
config hashes) -> one spawned worker per GPU, one run at a time per GPU -> manifest -> COMPLETE.json.

Per training cell ``cells/<run>/``: the Trainer's run tree (config, environment, metrics with the
per-epoch saved-statistics probe/test curve on the current and target paths, summary,
checkpoints/epoch_{030,060,120,160}.pt), ``p1_curve.json`` (P1 after completed epochs 40/80/120 at
the scheduled state, 160 from the endpoint), ``endpoint_policies.json`` (P0-P3 at the native
inference state of epoch_160.pt), ``timing.json``, ``DONE.json``.  A cell with ``DONE.json`` is
skipped; a cell with ``rolling.pt`` resumes exactly.  The job stops every worker at ``--hours``
(update boundary, rolling checkpoint) and still writes its manifest; ``--resume`` copies an earlier
output tree first.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import matrix as MX

PRECISION = "float32 (no autocast), TF32 off, cudnn.benchmark False, cudnn.deterministic False, no compilation"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_dir(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return Path(hits[0]).parent if hits else None


def log(path, gpu, msg):
    line = "[%s %s gpu%s] %s" % (time.strftime("%H:%M:%S"), now()[:10], gpu, msg)
    print(line, flush=True)
    with open(path, "a") as fh:
        fh.write(line + "\n")


def frozen() -> dict:
    p = Path(__file__).with_name("frozen_hashes.json")
    return json.loads(p.read_text()) if p.exists() else {}


def load_assets(ctx, seeds):
    from continuation_core import assets as assets_mod
    out, fz = {}, frozen()
    for s in seeds:
        a = assets_mod.load(ctx["assets"], s, verify_first=False)
        ext = MX.extended_perms(a["perms"], s)
        got = MX.sha_array(ext)
        want = fz.get("extended_perms_sha256", {}).get(str(s))
        if want is not None and got != want:
            raise RuntimeError("extended data order of seed %d differs from the frozen digest: %s != %s" % (s, got, want))
        a["perms"] = ext
        a["extended_perms_sha256"] = got
        out[s] = a
    return out


# ---- training cells ------------------------------------------------------------------------

def _p1_callback(cell_dir, device):
    from .bnpolicies import PolicyEvaluator
    path = cell_dir / "p1_curve.json"

    def cb(tr, done):
        if done not in MX.P1_CURVE_EPOCHS or done == MX.EPOCHS:
            return
        curve = json.loads(path.read_text()) if path.exists() else []
        if any(r["epoch"] == done for r in curve):
            return
        st = tr.controller.eval_current_state(done)
        t0 = time.perf_counter()
        ev = PolicyEvaluator(tr.cfg, tr.model.state_dict(), tr.dataset, device, tr.updates_per_epoch, state=st)
        rec = ev.run("P1_cumulative_clean_500")
        rec.update({"epoch": done, "update": tr.global_update, "evaluated_state": "scheduled state of the last update of epoch %d" % (done - 1)
                    if not tr.method.sdpoint else "fixed full-resolution instance",
                    "wall_seconds_including_setup": time.perf_counter() - t0})
        curve.append(rec)
        path.write_text(json.dumps(sorted(curve, key=lambda r: r["epoch"]), indent=1))
    return cb


def run_train_cell(cell, ctx, gpu, log_path, deadline):
    from continuation_core.train import Trainer
    from .bnpolicies import PolicyEvaluator
    out = Path(ctx["out"]) / "cells" / cell["run"]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE.json").exists():
        log(log_path, gpu, "skip (done) %s" % cell["run"])
        return "done"
    if time.time() >= deadline:
        return "not_started"
    device = "cuda:%d" % gpu
    t_cell = time.perf_counter()
    try:
        cfg = MX.config(cell["regime"], cell["arm"], cell["seed"], ctx["data_root"], ctx["assets"], str(out.parent))
        want = frozen().get("config_sha256", {}).get(cell["run"])
        if want is not None and MX.config_sha(cfg) != want:
            raise RuntimeError("config hash differs from the frozen protocol")
        tr = Trainer(cfg, dataset=ctx["dataset"], loaded_assets=ctx["assets_loaded"][cell["seed"]], device=device,
                     out_dir=out, log=lambda m: log(log_path, gpu, "%s %s" % (cell["run"], m)))
        tr.epoch_callbacks.append(_p1_callback(out, device))
        resumed_from = None
        if (out / "rolling.pt").exists():
            tr.resume(out / "rolling.pt")
            resumed_from = tr.global_update
        torch.cuda.reset_peak_memory_stats(gpu)
        session = {"kernel_start_utc": ctx["start_utc"], "resumed_from_update": resumed_from, "session_start_utc": now()}
        summary = tr.run(deadline=deadline)
        session["peak_cuda_memory_allocated_mib"] = torch.cuda.max_memory_allocated(gpu) / 2 ** 20
        session["session_end_utc"] = now()
        sessions_path = out / "sessions.json"
        sessions = json.loads(sessions_path.read_text()) if sessions_path.exists() else []
        sessions.append(session)
        sessions_path.write_text(json.dumps(sessions, indent=1))
        if summary is None:
            (out / "INCOMPLETE.json").write_text(json.dumps({"run": cell["run"], "utc": now(), "stopped_at_update": tr.global_update,
                                                              "reason": "kernel time budget"}, indent=1))
            log(log_path, gpu, "stopped %s at update %d (time budget)" % (cell["run"], tr.global_update))
            return "incomplete"
        if int(summary["updates"]) != MX.U:
            raise RuntimeError("final update count %s" % summary["updates"])
        ck = out / "checkpoints" / "epoch_160.pt"
        t_ev = time.perf_counter()
        ckd = torch.load(ck, map_location="cpu", weights_only=False)
        ev = PolicyEvaluator(cfg, ckd["model_state"], ctx["dataset"], device, tr.updates_per_epoch)
        pol = ev.all_policies()
        pol["checkpoint"] = "checkpoints/epoch_160.pt"
        pol["checkpoint_sha256"] = sha_file(ck)
        rec = summary["final"]["current"]["test"]
        p0 = pol["policies"]["P0_saved_native"]["test_full"]
        pol["recorded_check"] = {"recorded_current_test": rec, "P0_test_full": p0,
                                 "abs_diff_acc": abs(rec["acc"] - p0["acc"]), "abs_diff_ce": abs(rec["ce"] - p0["ce"]),
                                 "matches": abs(rec["acc"] - p0["acc"]) <= 2e-4 and abs(rec["ce"] - p0["ce"]) <= 1e-4,
                                 "note": "current path at epoch 160 is the native inference state for every arm"}
        (out / "endpoint_policies.json").write_text(json.dumps(pol, indent=1))
        eval_final = time.perf_counter() - t_ev
        curve_path = out / "p1_curve.json"
        curve = json.loads(curve_path.read_text()) if curve_path.exists() else []
        p1 = dict(pol["policies"]["P1_cumulative_clean_500"])
        p1.update({"epoch": 160, "update": MX.U, "evaluated_state": "native inference state (endpoint P1 reused)"})
        curve = [r for r in curve if r["epoch"] != 160] + [p1]
        curve_path.write_text(json.dumps(sorted(curve, key=lambda r: r["epoch"]), indent=1))
        timing = {"run": cell["run"], "gpu_index": gpu, "gpu": torch.cuda.get_device_name(gpu),
                  "concurrency": "one run at a time on this GPU; %d GPU worker(s) in the kernel, each running its own queue" % ctx["n_gpu"],
                  "precision": PRECISION, "training": summary["timing"],
                  "periodic_saved_bn_evaluation_seconds": summary["timing"]["eval_seconds"],
                  "p1_curve_diagnostic_seconds": summary["timing"]["diagnostic_seconds"],
                  "final_policy_evaluation_seconds": eval_final,
                  "final_policy_seconds_each": {k: v["seconds"] for k, v in pol["policies"].items()},
                  "sessions": sessions, "cell_wall_seconds_this_session": time.perf_counter() - t_cell}
        (out / "timing.json").write_text(json.dumps(timing, indent=1))
        (out / "DONE.json").write_text(json.dumps({"run": cell["run"], "finished_utc": now(), "checkpoint_sha256": pol["checkpoint_sha256"],
                                                   "recorded_check_matches": pol["recorded_check"]["matches"],
                                                   "P0_repeat_identical": pol["P0_repeat_identical"],
                                                   "extended_perms_sha256": ctx["assets_loaded"][cell["seed"]]["extended_perms_sha256"],
                                                   "config_sha256": MX.config_sha(cfg)}, indent=1))
        for f in ("rolling.pt", "INCOMPLETE.json"):
            if (out / f).exists():
                (out / f).unlink()
        log(log_path, gpu, "done %s (%.0f s this session)" % (cell["run"], time.perf_counter() - t_cell))
        return "done"
    except Exception:
        (out / "FAILED.json").write_text(json.dumps({"run": cell["run"], "utc": now(), "traceback": traceback.format_exc()}, indent=1))
        log(log_path, gpu, "FAILED %s\n%s" % (cell["run"], traceback.format_exc()))
        return "failed"


def _setup_worker(gpu, ctx_args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(gpu)
    from continuation_core.config import DataConfig
    from continuation_core.data import load_dataset
    ctx = dict(ctx_args)
    ctx["dataset"] = load_dataset(DataConfig(name="cifar10", root=ctx["data_root"]))
    return ctx


def _train_worker(gpu, queue, ctx_args, log_path, deadline):
    ctx = _setup_worker(gpu, ctx_args)
    ctx["assets_loaded"] = load_assets(ctx, sorted({c["seed"] for c in queue}))
    status = {}
    for cell in queue:
        status[cell["run"]] = run_train_cell(cell, ctx, gpu, log_path, deadline)
    Path(ctx["out"], "worker_status_gpu%d.json" % gpu).write_text(json.dumps(status, indent=1))


# ---- old 30-epoch endpoints -----------------------------------------------------------------

def _old_worker(gpu, runs, ctx_args, log_path):
    from comparison.panels import Endpoint
    from .bnpolicies import PolicyEvaluator
    ctx = _setup_worker(gpu, ctx_args)
    inputs = Path(ctx["inputs"])
    man = json.loads((inputs / "old_endpoints_manifest.json").read_text())
    for run in runs:
        info = man["endpoints"][run]
        out = Path(ctx["out"]) / "old" / run
        out.mkdir(parents=True, exist_ok=True)
        if (out / "DONE.json").exists():
            continue
        try:
            t0 = time.perf_counter()
            src = inputs / "old" / run
            ck = src / info["checkpoint_file"]
            got = sha_file(ck)
            if got != info["checkpoint_sha256"]:
                raise RuntimeError("checkpoint digest mismatch %s != %s" % (got, info["checkpoint_sha256"]))
            ep = Endpoint(ck, ctx["dataset"], "cpu", config_path=(src / "config.json") if (src / "config.json").exists() else None)
            upe = int(ep.ck["data_position"]["updates_per_epoch"])
            pe = PolicyEvaluator(ep.cfg, ep.ck["model_state"], ctx["dataset"], "cuda:%d" % gpu, upe,
                                 physical_batch=int(ep.cfg.budget.microbatch), augmentation="none")
            pol = pe.all_policies()
            old = json.loads((src / "panels.json").read_text())
            cmp = {}
            for new_k, old_k in (("P0_saved_native", "saved_native"), ("P1_cumulative_clean_500", "panelA_recalibrated_native")):
                for split in ("train_full", "test_full"):
                    a, b = pol["policies"][new_k][split], old[old_k][split]
                    cmp["%s/%s" % (new_k, split)] = {"reused": b, "recomputed": a, "abs_diff_acc": abs(a["acc"] - b["acc"]),
                                                     "abs_diff_ce": abs(a["ce"] - b["ce"]),
                                                     "matches": abs(a["acc"] - b["acc"]) <= 2e-4 and abs(a["ce"] - b["ce"]) <= 1e-4}
            pol.update({"run": run, "checkpoint_sha256": got, "old_panels_checkpoint_sha256": old.get("checkpoint_sha256"),
                        "checkpoint_identity_matches_old_panels": got == old.get("checkpoint_sha256"),
                        "old_panels_state": old["panelA_recalibrated_native"]["state"],
                        "native_state_matches_old_panels": old["panelA_recalibrated_native"]["state"] == pol["state"],
                        "reuse_check": cmp, "all_reuse_checks_match": all(v["matches"] for v in cmp.values()),
                        "physical_batch_source": "historical microbatch (budget.microbatch=%d)" % int(ep.cfg.budget.microbatch),
                        "seconds": time.perf_counter() - t0, "gpu": torch.cuda.get_device_name(gpu)})
            (out / "endpoint_policies.json").write_text(json.dumps(pol, indent=1))
            (out / "DONE.json").write_text(json.dumps({"run": run, "utc": now(), "all_reuse_checks_match": pol["all_reuse_checks_match"]}))
            log(log_path, gpu, "old %s done %.0fs reuse-match %s" % (run, pol["seconds"], pol["all_reuse_checks_match"]))
        except Exception:
            (out / "FAILED.json").write_text(json.dumps({"run": run, "traceback": traceback.format_exc()}, indent=1))
            log(log_path, gpu, "FAILED old %s\n%s" % (run, traceback.format_exc()))


# ---- pilot ----------------------------------------------------------------------------------

def _pilot_worker(gpu, cells, ctx_args, log_path, pilot_updates):
    from continuation_core.train import Trainer
    from .bnpolicies import PolicyEvaluator
    ctx = _setup_worker(gpu, ctx_args)
    ctx["assets_loaded"] = load_assets(ctx, [0])
    ds = ctx["dataset"]
    for cell in cells:
        out = Path(ctx["out"]) / "pilot" / cell["run"]
        out.mkdir(parents=True, exist_ok=True)
        try:
            cfg = MX.config(cell["regime"], cell["arm"], 0, ctx["data_root"], ctx["assets"], str(out.parent))
            cfg.evaluation.at_epoch_zero = False          # no evaluation of test images in the pilot
            cfg.checkpoint.keep_rolling = False
            cfg.checkpoint.at_epochs = ()
            tr = Trainer(cfg, dataset=ds, loaded_assets=ctx["assets_loaded"][0], device="cuda:%d" % gpu, out_dir=out, log=lambda *_: None)
            torch.cuda.reset_peak_memory_stats(gpu)
            tr.run(max_updates=40)                       # warm-up
            torch.cuda.synchronize(gpu)
            t0 = time.perf_counter()
            tr.run(max_updates=pilot_updates)
            torch.cuda.synchronize(gpu)
            sec = (time.perf_counter() - t0) / pilot_updates
            finite = bool(np.isfinite(tr.run_loss / max(tr.run_n, 1)))
            # proxy of one per-epoch saved-statistics snapshot: 2 paths x (500 + 10,000) images, timed on
            # training images so that no test image is evaluated during the pilot
            from continuation_core.evaluate import evaluate
            x, y = tr.train_images[:10500], tr.train_labels[:10500]
            t1 = time.perf_counter()
            for st in (tr.controller.eval_current_state(1), tr.controller.target_state()):
                evaluate(tr.model, tr.controller, tr.pipeline, x, y, st, batch_size=500)
            torch.cuda.synchronize(gpu)
            snap = time.perf_counter() - t1
            pol = {}
            if cell["arm"] in ("plain", "resolution_max_b1_gaussian_conv"):
                pe = PolicyEvaluator(cfg, tr.model.state_dict(), ds, "cuda:%d" % gpu, tr.updates_per_epoch)
                for p in ("P1_cumulative_clean_500", "P2_cumulative_clean_32", "P3_ema_training_loader"):
                    pol[p] = pe.run(p, splits=("train_full",))["seconds"]
            ck = tr.save("rolling", name="pilot_state.pt")
            rec = {"run": cell["run"], "gpu": torch.cuda.get_device_name(gpu), "measured_updates": pilot_updates,
                   "seconds_per_update": sec, "projected_train_hours_62560": sec * MX.U / 3600,
                   "snapshot_proxy_seconds": snap, "projected_periodic_eval_hours_161": snap * 161 / 3600,
                   "policy_seconds_train_scoring_only": pol,
                   "peak_cuda_memory_allocated_mib": torch.cuda.max_memory_allocated(gpu) / 2 ** 20,
                   "checkpoint_bytes": ck.stat().st_size, "finite_loss": finite,
                   "physical_batch": int(cfg.budget.microbatch), "augmentation": cfg.data.augmentation}
            ck.unlink()
            (out / "pilot_timing.json").write_text(json.dumps(rec, indent=1))
            log(log_path, gpu, "pilot %s %.4f s/update -> %.2f h, peak %.0f MiB" % (cell["run"], sec, rec["projected_train_hours_62560"], rec["peak_cuda_memory_allocated_mib"]))
        except Exception:
            (out / "FAILED.json").write_text(json.dumps({"traceback": traceback.format_exc()}))
            log(log_path, gpu, "FAILED pilot %s\n%s" % (cell["run"], traceback.format_exc()))
    _resume_check(gpu, ctx, ds)


def _resume_check_worker(gpu, ctx_args):
    ctx = _setup_worker(gpu, ctx_args)
    ctx["assets_loaded"] = load_assets(ctx, [0])
    _resume_check(gpu, ctx, ctx["dataset"])


def _resume_check(gpu, ctx, ds):
    """Exact resumption on GPU (reported, not assumed bitwise: cuDNN kernels are not forced deterministic)."""
    from continuation_core.train import Trainer
    try:
        cfg = MX.config("adamw_long_aug_160", "sdpoint", 0, ctx["data_root"], ctx["assets"], str(Path(ctx["out"]) / "pilot_resume"))
        cfg.evaluation.at_epoch_zero = False
        cfg.checkpoint.keep_rolling = False
        cfg.checkpoint.at_epochs = ()
        a = Trainer(cfg, dataset=ds, loaded_assets=ctx["assets_loaded"][0], device="cuda:%d" % gpu, out_dir=Path(ctx["out"]) / "pilot_resume" / "a", log=lambda *_: None)
        a.run(max_updates=60)
        b = Trainer(cfg, dataset=ds, loaded_assets=ctx["assets_loaded"][0], device="cuda:%d" % gpu, out_dir=Path(ctx["out"]) / "pilot_resume" / "b", log=lambda *_: None)
        b.run(max_updates=25)
        ck = b.save("rolling", name="rolling.pt")
        c = Trainer(cfg, dataset=ds, loaded_assets=ctx["assets_loaded"][0], device="cuda:%d" % gpu, out_dir=Path(ctx["out"]) / "pilot_resume" / "b", log=lambda *_: None)
        c.resume(ck)
        c.run(max_updates=35)
        diffs = [float((x.float() - y.float()).abs().max()) for x, y in zip(a.model.state_dict().values(), c.model.state_dict().values())]
        rec = {"max_abs_diff_state": max(diffs), "bitwise_equal": max(diffs) == 0.0, "updates": [60, "25+35"]}
        ck.unlink()
    except Exception:
        rec = {"error": traceback.format_exc()}
    Path(ctx["out"], "pilot_resume_gpu%d.json" % gpu).write_text(json.dumps(rec, indent=1))


# ---- main -----------------------------------------------------------------------------------

def manifest(root: Path):
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name not in ("MANIFEST.json", "COMPLETE.json"):
            files[p.relative_to(root).as_posix()] = {"sha256": sha_file(p), "bytes": p.stat().st_size}
    (root / "MANIFEST.json").write_text(json.dumps(files, indent=1))
    return len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("pilot", "train", "evaluate_old"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--queues", default="[]")
    ap.add_argument("--hours", type=float, default=11.0)
    ap.add_argument("--pilot-updates", type=int, default=200)
    ap.add_argument("--resume")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()
    t_start = time.time()
    deadline = t_start + a.hours * 3600
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "job_log.txt"
    n_gpu = torch.cuda.device_count()
    env = {"utc": now(), "python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(), "numpy": np.__version__, "n_gpu": n_gpu,
           "gpus": [torch.cuda.get_device_name(i) for i in range(n_gpu)], "mode": a.mode, "hours": a.hours,
           "nvidia_smi": subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout if shutil.which("nvidia-smi") else None}
    (out / "environment.json").write_text(json.dumps(env, indent=1))
    if n_gpu == 0:
        (out / "NO_GPU_ABORTED.json").write_text(json.dumps({"utc": now(), "reason": "no CUDA device"}))
        manifest(out)
        raise SystemExit("no GPU granted; aborting before any work")
    if a.resume and Path(a.resume).exists():
        for p in Path(a.resume).rglob("*"):
            dst = out / p.relative_to(a.resume)
            if p.is_file() and not dst.exists() and p.name not in ("MANIFEST.json", "COMPLETE.json"):
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
    inputs = Path(os.environ["OVN_INPUTS"]) if os.environ.get("OVN_INPUTS") else find_dir("/kaggle/input/**/overnight_inputs_manifest.json")
    data_root = os.environ.get("OVN_DATA") or str(find_dir("/kaggle/input/**/cifar-10-batches-py/batches.meta").parent)
    assets = str(inputs / "assets")
    from continuation_core import assets as assets_mod
    ver = assets_mod.verify(assets)
    man = json.loads((inputs / "overnight_inputs_manifest.json").read_text())
    bad = [rel for rel, info in man["files"].items() if sha_file(inputs / rel) != info["sha256"]]
    (out / "inputs_check.json").write_text(json.dumps({"assets_all_match": ver["all_match"], "input_files": len(man["files"]),
                                                      "mismatched": bad, "frozen_hashes_present": bool(frozen())}, indent=1))
    if bad:
        raise SystemExit("input digest mismatch: %s" % bad)
    ctx = {"out": str(out), "data_root": data_root, "assets": assets, "inputs": str(inputs), "n_gpu": n_gpu,
           "start_utc": env["utc"]}
    ctxm = mp.get_context("spawn")
    status = {}
    if a.mode == "pilot":
        repo = Path(__file__).resolve().parents[1]
        if not (repo / "assets" / "cifar10_resnet20bn").exists():
            shutil.copytree(assets, repo / "assets" / "cifar10_resnet20bn")
        if not a.skip_tests:
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_overnight.py", "tests/test_comparators.py",
                                "tests/test_operators_and_methods.py", "tests/test_training_checkpoints_analysis.py"],
                               capture_output=True, text=True, cwd=str(repo))
            (out / "pilot_tests.txt").write_text(r.stdout[-30000:] + "\n" + r.stderr[-5000:])
            log(log_path, "-", "tests: %s" % (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "no output"))
        cells = [c for c in MX.cells() if c["seed"] == 0]
        queues = [cells[i::n_gpu] for i in range(n_gpu)]
        procs = [ctxm.Process(target=_pilot_worker, args=(g, q, ctx, str(log_path), a.pilot_updates)) for g, q in enumerate(queues)]
    elif a.mode == "train":
        by_run = {c["run"]: c for c in MX.cells()}
        queues = [[by_run[r] for r in q] for q in json.loads(a.queues)]
        (out / "queues.json").write_text(json.dumps([[c["run"] for c in q] for q in queues], indent=1))
        if not Path(out, "pilot_resume_gpu0.json").exists():
            chk = ctxm.Process(target=_resume_check_worker, args=(0, ctx))       # ~1 min, before training
            chk.start()
            chk.join()
        procs = [ctxm.Process(target=_train_worker, args=(g, q, ctx, str(log_path), deadline))
                 for g, q in enumerate(queues[:n_gpu]) if q]
        for q in queues[n_gpu:]:
            for c in q:
                status[c["run"]] = "not_started (fewer GPUs than queues)"
    else:
        runs = sorted(json.loads((inputs / "old_endpoints_manifest.json").read_text())["endpoints"])
        queues = [runs[i::n_gpu] for i in range(n_gpu)]
        procs = [ctxm.Process(target=_old_worker, args=(g, q, ctx, str(log_path))) for g, q in enumerate(queues)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    if a.mode == "train":
        for q in queues:
            for c in q:
                d = out / "cells" / c["run"]
                status.setdefault(c["run"], "done" if (d / "DONE.json").exists() else "failed" if (d / "FAILED.json").exists()
                                  else "incomplete" if (d / "INCOMPLETE.json").exists() or (d / "rolling.pt").exists() else "not_started")
    elif a.mode == "evaluate_old":
        for q in queues:
            for r in q:
                d = out / "old" / r
                status[r] = "done" if (d / "DONE.json").exists() else "failed"
    n = manifest(out)
    (out / "COMPLETE.json").write_text(json.dumps({"utc": now(), "mode": a.mode, "cells": status,
                                                   "all_done": all(v == "done" for v in status.values()) if status else None,
                                                   "manifest_files": n, "total_seconds": time.time() - t_start}, indent=1))
    print("job finished", json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
