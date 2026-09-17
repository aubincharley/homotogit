"""One Kaggle shard (one setting x seed): train the new cells, evaluate every endpoint.

    python -m comparison.job --setting sgd --seed 0 --out /kaggle/working/cmp [--pilot]

Order: CUDA guard -> environment -> asset and input digests -> (pilot: tests + timing) ->
one isolated worker per GPU, cells run sequentially on each GPU -> manifest -> COMPLETE.json.

Per cell ``cells/<run>/``: the Trainer's run tree (config, environment, metrics, summary,
checkpoints/epoch_030.pt), ``recorded_check.json``, ``panels.json``, ``recalibrated_buffers.pt``,
``timing.json`` and ``DONE.json``.  A cell with ``DONE.json`` is skipped; a cell with
``rolling.pt`` resumes.  ``--resume`` copies an earlier output tree first (a relaunch after a
session limit).  ``rolling.pt`` is deleted once the cell is done: the final checkpoint and
the records are what is kept.
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

import torch

from . import matrix as MX

#: T4 seconds for a full 30-epoch run, used only to balance the two GPU queues: plain, CBS and
#: SDPoint from the T4 pilot (maxlebossdu91, 2026-09-17: 0.0378 / 0.0460 / 0.0377 s per update
#: plus epoch evaluations); R, G, RG from the unified-batch wall times.
COST = {"plain": 473, "resolution_max_b1": 499, "gaussian_postrelu": 623,
        "resolution_max_b1_gaussian_conv": 733, "cbs_published_schedule": 601,
        "cbs_budget_matched": 606, "sdpoint": 473}


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
    line = "[%s gpu%s] %s" % (time.strftime("%H:%M:%S"), gpu, msg)
    print(line, flush=True)
    with open(path, "a") as fh:
        fh.write(line + "\n")


def _strip(rec):
    buffers = {}
    for k, v in list(rec.items()):
        if isinstance(v, dict) and "buffers" in v:
            buffers[k] = v.pop("buffers")
    return rec, buffers


def evaluate_endpoint(ckpt, dataset, device, out_dir, summary, config_path=None):
    from .panels import Endpoint
    t0 = time.perf_counter()
    ep = Endpoint(ckpt, dataset, device, config_path=config_path)
    chk = ep.recorded_check(summary)
    (out_dir / "recorded_check.json").write_text(json.dumps(chk, indent=1))
    panels, buffers = _strip(ep.all_panels())
    panels["checkpoint_sha256"] = sha_file(ckpt)
    panels["recorded_check"] = chk
    (out_dir / "panels.json").write_text(json.dumps(panels, indent=1))
    torch.save(buffers, out_dir / "recalibrated_buffers.pt")
    return time.perf_counter() - t0, chk, panels


def run_cell(cell, ctx, gpu, log_path):
    from continuation_core.train import Trainer
    out = Path(ctx["out"]) / "cells" / cell["run"]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE.json").exists():
        log(log_path, gpu, "skip (done) %s" % cell["run"])
        return
    device = "cuda:%d" % gpu
    t_cell = time.perf_counter()
    timing = {"run": cell["run"], "gpu_index": gpu, "gpu": torch.cuda.get_device_name(gpu),
              "concurrency": "one run at a time on this GPU; %d GPU worker(s) in the kernel" % ctx["n_gpu"],
              "precision": "float32 (no autocast), TF32 off, cudnn.benchmark False, cudnn.deterministic False"}
    try:
        dataset = ctx["dataset"]
        if cell["action"] == "reuse_landscape_v2_checkpoint":
            src = Path(ctx["inputs"]) / "reused" / cell["run"]
            man = json.loads((Path(ctx["inputs"]) / "inputs_manifest.json").read_text())
            ck = src / "epoch_030.pt"
            got = sha_file(ck)
            want = man["reused"][cell["run"]]["checkpoint_sha256"]
            if got != want:
                raise RuntimeError("reused checkpoint digest mismatch: %s != %s" % (got, want))
            summary = json.loads((src / "summary.json").read_text())
            timing["training"] = {"source": "landscape_v2 run (not retrained)", "recorded_timing": summary.get("timing"),
                                  "scope_note": "includes per-epoch and transition-window checkpoint writes of that study"}
            sec, chk, panels = evaluate_endpoint(ck, dataset, device, out, summary, config_path=src / "config.json")
            for f in ("config.json", "summary.json", "metrics.json"):
                if (src / f).exists():
                    shutil.copy2(src / f, out / f)
        else:
            cfg = MX.config(cell["setting"], cell["arm"], cell["seed"], ctx["data_root"], ctx["assets"], str(out.parent))
            tr = Trainer(cfg, dataset=dataset, loaded_assets=ctx["assets_loaded"][cell["seed"]], device=device,
                         out_dir=out, log=lambda m: log(log_path, gpu, "%s %s" % (cell["run"], m)))
            if (out / "rolling.pt").exists():
                tr.resume(out / "rolling.pt")
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(gpu)
            summary = tr.run()
            timing["training"] = {**summary["timing"], "peak_cuda_memory_mib":
                                  torch.cuda.max_memory_allocated(gpu) / 2 ** 20}
            ck = out / "checkpoints" / "epoch_030.pt"
            sec, chk, panels = evaluate_endpoint(ck, dataset, device, out, summary)
            if int(summary["updates"]) != MX.TOTAL_UPDATES:
                raise RuntimeError("final update count %s" % summary["updates"])
        timing["final_calibration_and_evaluation_seconds"] = sec
        timing["cell_wall_seconds"] = time.perf_counter() - t_cell
        (out / "timing.json").write_text(json.dumps(timing, indent=1))
        done = {"run": cell["run"], "action": cell["action"], "finished_utc": now(),
                "recorded_check_matches": chk["matches"],
                "panelA_test_acc": panels["panelA_recalibrated_native"]["test_full"]["acc"],
                "checkpoint_sha256": panels["checkpoint_sha256"]}
        (out / "DONE.json").write_text(json.dumps(done, indent=1))
        if (out / "rolling.pt").exists():
            (out / "rolling.pt").unlink()
        log(log_path, gpu, "done %s  panelA test %.4f  recorded-check %s  %.0fs"
            % (cell["run"], done["panelA_test_acc"], chk["matches"], timing["cell_wall_seconds"]))
    except Exception:
        (out / "FAILED.json").write_text(json.dumps({"run": cell["run"], "utc": now(),
                                                     "traceback": traceback.format_exc()}, indent=1))
        log(log_path, gpu, "FAILED %s\n%s" % (cell["run"], traceback.format_exc()))


def _worker(gpu, queue, ctx_args, log_path):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(gpu)
    from continuation_core import assets as assets_mod
    from continuation_core.config import DataConfig
    from continuation_core.data import load_dataset
    ctx = dict(ctx_args)
    ctx["dataset"] = load_dataset(DataConfig(name="cifar10", root=ctx["data_root"]))
    ctx["assets_loaded"] = {s: assets_mod.load(ctx["assets"], s, verify_first=False) for s in {c["seed"] for c in queue}}
    for cell in queue:
        if ctx.get("pilot"):
            pilot_cell(cell, ctx, gpu, log_path)
        else:
            run_cell(cell, ctx, gpu, log_path)


def pilot_cell(cell, ctx, gpu, log_path):
    """Timing and memory of ``PILOT_UPDATES`` updates (plus one epoch-0 evaluation)."""
    from continuation_core.train import Trainer
    out = Path(ctx["out"]) / "pilot" / cell["run"]
    out.mkdir(parents=True, exist_ok=True)
    cfg = MX.config(cell["setting"], cell["arm"], cell["seed"], ctx["data_root"], ctx["assets"], str(out.parent))
    cfg.checkpoint.keep_rolling = False
    tr = Trainer(cfg, dataset=ctx["dataset"], loaded_assets=ctx["assets_loaded"][cell["seed"]],
                 device="cuda:%d" % gpu, out_dir=out, log=lambda *_: None)
    torch.cuda.reset_peak_memory_stats(gpu)
    t0 = time.perf_counter()
    tr.run(max_updates=ctx["pilot_updates"])
    torch.cuda.synchronize(gpu)
    wall = time.perf_counter() - t0
    train = wall - tr.eval_seconds
    rec = {"run": cell["run"], "updates": tr.global_update, "wall_seconds": wall, "eval_seconds_epoch0": tr.eval_seconds,
           "train_seconds": train, "seconds_per_update": train / max(tr.global_update, 1),
           "projected_train_seconds_11730_updates": train / max(tr.global_update, 1) * MX.TOTAL_UPDATES,
           "projected_eval_seconds_31_snapshots": tr.eval_seconds * 31,
           "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(gpu) / 2 ** 20,
           "gpu": torch.cuda.get_device_name(gpu), "finite_last_loss": tr.run_loss == tr.run_loss}
    tr.save("rolling", name="pilot_state.pt")
    rec["checkpoint_bytes"] = (out / "pilot_state.pt").stat().st_size
    (out / "pilot_state.pt").unlink()
    (out / "pilot_timing.json").write_text(json.dumps(rec, indent=1))
    log(log_path, gpu, "pilot %s %.4f s/update, projected %.0f s + eval %.0f s, peak %.0f MiB"
        % (cell["run"], rec["seconds_per_update"], rec["projected_train_seconds_11730_updates"],
           rec["projected_eval_seconds_31_snapshots"], rec["peak_cuda_memory_mib"]))


def deal(cells, n_gpu):
    queues, load = [[] for _ in range(n_gpu)], [0.0] * n_gpu
    for c in sorted(cells, key=lambda c: -(60 if c["action"] == "reuse_landscape_v2_checkpoint" else COST[c["arm"]])):
        k = load.index(min(load))
        queues[k].append(c)
        load[k] += 60 if c["action"] == "reuse_landscape_v2_checkpoint" else COST[c["arm"]]
    return queues


def manifest(root: Path):
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name not in ("MANIFEST.json", "COMPLETE.json"):
            files[p.relative_to(root).as_posix()] = {"sha256": sha_file(p), "bytes": p.stat().st_size}
    (root / "MANIFEST.json").write_text(json.dumps(files, indent=1))
    return len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True, choices=MX.SETTINGS)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--pilot-updates", type=int, default=300)
    ap.add_argument("--resume")
    a = ap.parse_args()
    t_start = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "job_log.txt"
    n_gpu = torch.cuda.device_count()
    env = {"utc": now(), "python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(), "n_gpu": n_gpu,
           "gpus": [torch.cuda.get_device_name(i) for i in range(n_gpu)], "setting": a.setting, "seeds": a.seeds,
           "pilot": a.pilot}
    (out / "environment.json").write_text(json.dumps(env, indent=1))
    if n_gpu == 0:
        (out / "NO_GPU_ABORTED.json").write_text(json.dumps({"utc": now(), "reason": "no CUDA device"}))
        raise SystemExit("no GPU granted; aborting before any work")
    if a.resume and Path(a.resume).exists():
        for p in Path(a.resume).rglob("*"):
            dst = out / p.relative_to(a.resume)
            if p.is_file() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
    inputs = Path(os.environ["CMP_INPUTS"]) if os.environ.get("CMP_INPUTS") else find_dir("/kaggle/input/**/inputs_manifest.json")
    data_root = os.environ.get("CMP_DATA") or str(find_dir("/kaggle/input/**/cifar-10-batches-py/batches.meta").parent)
    assets = str(inputs / "assets")
    from continuation_core import assets as assets_mod
    ver = assets_mod.verify(assets)
    man = json.loads((inputs / "inputs_manifest.json").read_text())
    bad = [rel for rel, info in man["files"].items() if sha_file(inputs / rel) != info["sha256"]]
    (out / "inputs_check.json").write_text(json.dumps({"assets_all_match": ver["all_match"], "input_files": len(man["files"]),
                                                      "mismatched": bad}, indent=1))
    if bad:
        raise SystemExit("input digest mismatch: %s" % bad)
    cells = MX.shard_cells(a.setting, a.seeds)
    ctx = {"out": str(out), "data_root": data_root, "assets": assets, "inputs": str(inputs), "n_gpu": n_gpu,
           "pilot": a.pilot, "pilot_updates": a.pilot_updates}
    timing = {"start_utc": now()}
    if a.pilot:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_comparators.py", "tests/test_operators_and_methods.py",
                            "tests/test_optimizers.py", "tests/test_training_checkpoints_analysis.py"],
                           capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
        (out / "pilot_tests.txt").write_text(r.stdout[-20000:] + "\n" + r.stderr[-5000:])
        log(log_path, "-", "tests: %s" % r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "no output")
        cells = [c for c in cells if c["seed"] == a.seeds[0]
                 and c["arm"] in ("plain", "cbs_published_schedule", "cbs_budget_matched", "sdpoint")]
        cells = [dict(c, action="pilot") for c in cells]
    queues = deal(cells, n_gpu)
    (out / "queues.json").write_text(json.dumps([[c["run"] for c in q] for q in queues], indent=1))
    ctxm = mp.get_context("spawn")
    procs = [ctxm.Process(target=_worker, args=(g, q, ctx, str(log_path))) for g, q in enumerate(queues) if q]
    t0 = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    timing["workers_wall_seconds"] = time.time() - t0
    if a.pilot and cells and man.get("reused"):
        # one reused endpoint evaluated on GPU 0 for the evaluation-cost estimate
        from continuation_core.config import DataConfig
        from continuation_core.data import load_dataset
        run = sorted(man["reused"])[0]
        src = inputs / "reused" / run
        dst = out / "pilot" / ("panels__" + run)
        dst.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(DataConfig(name="cifar10", root=data_root))
        sec, chk, _ = evaluate_endpoint(src / "epoch_030.pt", ds, "cuda:0", dst,
                                        json.loads((src / "summary.json").read_text()), config_path=src / "config.json")
        timing["pilot_panel_evaluation_seconds"] = sec
        log(log_path, 0, "pilot panels %s: %.1f s, recorded check %s" % (run, sec, chk["matches"]))
    timing["total_seconds"] = time.time() - t_start
    (out / "job_timing.json").write_text(json.dumps(timing, indent=1))
    status = {}
    for c in cells:
        d = out / "cells" / c["run"]
        status[c["run"]] = ("pilot" if a.pilot else "done" if (d / "DONE.json").exists()
                            else "failed" if (d / "FAILED.json").exists() else "incomplete")
    n = manifest(out)
    (out / "COMPLETE.json").write_text(json.dumps({"utc": now(), "setting": a.setting, "seeds": a.seeds, "pilot": a.pilot,
                                                   "cells": status, "all_done": all(v in ("done", "pilot") for v in status.values()),
                                                   "manifest_files": n}, indent=1))
    print("job finished", json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
