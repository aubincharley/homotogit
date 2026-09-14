"""Kaggle job: the four frozen methods under four optimizers.

Launch (one kernel per slice; see docs/kaggle_cli.md)::

    py scripts/kaggle_run.py scripts/job_optimizer_benchmark.py --gpu \\
        --accelerator NvidiaTeslaT4 \\
        --dataset pankrzysiu/cifar10-python \\
        --dataset <owner>/continuation-core-r20bn-assets \\
        --include continuation_core --include scripts \\
        --timeout-seconds 28800

Environment read on the worker:

======================  ====================================================
``OPT_BENCH_BATCH``     ``lr_sweep`` | ``sgd_control`` | ``grid``
``OPT_BENCH_JOB``       this kernel's index (default 0)
``OPT_BENCH_NJOBS``     how many kernels share the batch (default 1)
``STUDY_DATA``          local override for the CIFAR-10 parent directory
``CORE_ASSETS``         local override for the pinned asset directory
======================  ====================================================

The cell list is built once, sorted deterministically and sliced by
``index % n_jobs``, so a cell belongs to exactly one kernel however the slices
are launched or relaunched.  Within a kernel the cells are dealt longest-first
across the granted GPUs, because the methods differ by up to 45% in cost.

Nothing here re-implements training: it builds a reference config with the
optimizer swapped and hands it to ``continuation_core.train.Trainer``, which
already verifies the pinned assets, writes the documented result tree, and can
resume mid-epoch from ``rolling.pt`` if a session is cut short.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("runs")

METHODS = ("plain", "resolution_max_b1", "gaussian_postrelu",
           "resolution_max_b1_gaussian_conv")
SEEDS = (0, 1, 2)
NEW_OPTIMIZERS = ("adam", "adamw", "radam")
SWEEP_LRS = (3e-4, 1e-3, 3e-3, 1e-2)
REFERENCE_LR = 0.005
WEIGHT_DECAY = 5e-4

#: Measured on T4 in the unified batch; used only to balance the two GPUs.
COST_SECONDS = {"plain": 505, "resolution_max_b1": 499,
                "gaussian_postrelu": 623, "resolution_max_b1_gaussian_conv": 733}

#: Filled in from docs/LR_SWEEP.md once phase 2 has been read out.  Kept in
#: code, and committed, so the grid's learning rates are auditable rather than
#: passed loosely at launch time.
CHOSEN_LR: dict = {}


# --------------------------------------------------------------------------
# the cell manifest
# --------------------------------------------------------------------------

def build_cells(batch: str) -> list:
    """Every cell of one batch, in a deterministic order."""
    cells = []
    if batch == "lr_sweep":
        for name in NEW_OPTIMIZERS:
            for lr in SWEEP_LRS:
                cells.append({"method": "plain", "optimizer": name, "lr": lr, "seed": 0})
    elif batch == "sgd_control":
        for seed in SEEDS:
            cells.append({"method": "plain", "optimizer": "sgd",
                          "lr": REFERENCE_LR, "seed": seed})
    elif batch == "grid":
        missing = [o for o in NEW_OPTIMIZERS if o not in CHOSEN_LR]
        if missing:
            raise SystemExit(
                "CHOSEN_LR has no entry for %s: run the lr_sweep batch, read it out "
                "into docs/LR_SWEEP.md, then fill CHOSEN_LR in this file before "
                "launching the grid" % ", ".join(missing))
        for name in NEW_OPTIMIZERS:
            for method in METHODS:
                for seed in SEEDS:
                    cells.append({"method": method, "optimizer": name,
                                  "lr": CHOSEN_LR[name], "seed": seed})
    else:
        raise SystemExit("unknown batch %r (lr_sweep | sgd_control | grid)" % batch)
    cells.sort(key=lambda c: (c["optimizer"], c["method"], c["seed"], c["lr"]))
    for i, c in enumerate(cells):
        c["index"] = i
    return cells


def slice_for(cells: list, job: int, n_jobs: int) -> list:
    return [c for c in cells if c["index"] % max(n_jobs, 1) == job]


def deal_across_gpus(cells: list, n_gpu: int) -> list:
    """Longest-first onto the least-loaded GPU: the methods differ by up to 45%."""
    queues = [[] for _ in range(max(n_gpu, 1))]
    load = [0.0] * len(queues)
    for c in sorted(cells, key=lambda c: -COST_SECONDS.get(c["method"], 600)):
        k = load.index(min(load))
        queues[k].append(c)
        load[k] += COST_SECONDS.get(c["method"], 600)
    return queues


# --------------------------------------------------------------------------
# locating the mounted inputs
# --------------------------------------------------------------------------

def data_source() -> str:
    """Parent directory of ``cifar-10-batches-py``.

    Kaggle inserts a path segment of its own under ``/kaggle/input`` (the real
    mount has been ``/kaggle/input/datasets/<owner>/<slug>/``), so this globs
    rather than assuming the layout.
    """
    override = os.environ.get("STUDY_DATA")
    if override and (Path(override) / "cifar-10-batches-py").is_dir():
        return override
    for hit in sorted(glob.glob("/kaggle/input/**/cifar-10-batches-py", recursive=True)):
        return str(Path(hit).parent)
    raise SystemExit("CIFAR-10 not found: attach a dataset holding "
                     "cifar-10-batches-py, or set STUDY_DATA")


def assets_source() -> str:
    override = os.environ.get("CORE_ASSETS")
    if override and (Path(override) / "assets_manifest.json").is_file():
        return override
    for hit in sorted(glob.glob("/kaggle/input/**/assets_manifest.json", recursive=True)):
        return str(Path(hit).parent)
    raise SystemExit("pinned assets not found: attach the asset dataset "
                     "(scripts/stage_assets.py), or set CORE_ASSETS")


# --------------------------------------------------------------------------
# running one cell
# --------------------------------------------------------------------------

def config_for(cell: dict, data_root: str, assets_dir: str, out_dir: str):
    from continuation_core.config import OptimizerConfig
    from continuation_core.presets import reference, reference_optimizer
    ref = reference_optimizer()
    if cell["optimizer"] == "sgd":
        opt = OptimizerConfig(name="sgd", lr=cell["lr"], weight_decay=WEIGHT_DECAY,
                              momentum=ref.momentum, nesterov=ref.nesterov,
                              schedule=ref.schedule, warmup_updates=ref.warmup_updates,
                              min_lr=ref.min_lr)
    else:
        opt = OptimizerConfig(name=cell["optimizer"], lr=cell["lr"],
                              weight_decay=WEIGHT_DECAY, schedule=ref.schedule,
                              warmup_updates=ref.warmup_updates, min_lr=ref.min_lr)
    return reference(cell["method"], cell["seed"], data_root=data_root,
                     assets_dir=assets_dir, out_dir=out_dir, optimizer=opt)


def train_cell(cell: dict, *, data_root: str, assets_dir: str, out_dir: str,
               device: str, dataset=None) -> dict:
    from continuation_core.train import Trainer
    cfg = config_for(cell, data_root, assets_dir, out_dir)
    run_dir = Path(out_dir) / cfg.run.name
    started = datetime.now(timezone.utc).isoformat()

    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        print("skip (already complete): %s" % cfg.run.name, flush=True)
        return {**cell, "run": cfg.run.name, "status": "already_complete"}

    trainer = Trainer(cfg, device=device, dataset=dataset)
    rolling = run_dir / "rolling.pt"
    if rolling.is_file():
        # a previous kernel was cut off by the session limit
        trainer.resume(rolling)
    out = trainer.run()
    return {**cell, "run": cfg.run.name, "status": "complete",
            "started_utc": started,
            "final_test_acc": out["final"]["target"]["test"]["acc"],
            "wall_seconds": out["timing"]["wall_seconds"]}


def _worker(gpu_index: int, cells: list, kwargs: dict, results_path: str) -> None:
    import torch
    device = "cuda:%d" % gpu_index if torch.cuda.is_available() else "cpu"
    rows = []
    dataset = None
    for cell in cells:
        try:
            if dataset is None:
                from continuation_core.data import load_dataset
                from continuation_core.config import DataConfig
                dataset = load_dataset(DataConfig(name="cifar10",
                                                  root=kwargs["data_root"]))
            rows.append(train_cell(cell, device=device, dataset=dataset, **kwargs))
        except Exception:
            traceback.print_exc()
            rows.append({**cell, "status": "failed", "error": traceback.format_exc()})
        Path(results_path).write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    import torch

    batch = os.environ.get("OPT_BENCH_BATCH", "lr_sweep")
    job = int(os.environ.get("OPT_BENCH_JOB", "0"))
    n_jobs = int(os.environ.get("OPT_BENCH_NJOBS", "1"))

    out_dir = WORK / ("optbench_%s" % batch)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        (out_dir / "ABORTED.json").write_text(json.dumps(
            {"reason": "no GPU granted", "batch": batch, "job": job,
             "utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
        raise SystemExit("no GPU was granted; refusing to burn a session on CPU")

    data_root = data_source()
    assets_dir = assets_source()

    # fail fast, before any training, rather than per run: these digests are
    # what ties this campaign to the recorded SGD runs it reuses
    from continuation_core import assets as assets_mod
    verification = assets_mod.verify(assets_dir)
    print("assets verified: %d digests at %s"
          % (len(verification["checks"]), assets_dir), flush=True)

    cells = build_cells(batch)
    mine = slice_for(cells, job, n_jobs)
    queues = deal_across_gpus(mine, n_gpu)

    manifest = {"batch": batch, "job": job, "n_jobs": n_jobs, "n_gpu": n_gpu,
                "cells_total": len(cells), "cells_here": len(mine),
                "data_root": data_root, "assets_dir": assets_dir,
                "assets_verified": True,
                "queues": [[c["index"] for c in q] for q in queues],
                "cells": mine,
                "utc": datetime.now(timezone.utc).isoformat()}
    (out_dir / "job_manifest.json").write_text(json.dumps(manifest, indent=2),
                                               encoding="utf-8")
    print("batch %s: %d/%d cells on %d GPU(s)"
          % (batch, len(mine), len(cells), n_gpu), flush=True)

    kwargs = {"data_root": data_root, "assets_dir": assets_dir,
              "out_dir": str(out_dir)}
    parts = [str(out_dir / ("worker_%d.json" % i)) for i in range(len(queues))]

    if n_gpu >= 2 and len(queues) > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=_worker, args=(i, queues[i], kwargs, parts[i]))
                 for i in range(len(queues)) if queues[i]]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
    else:
        _worker(0, mine, kwargs, parts[0])

    rows = []
    for part in parts:
        if Path(part).is_file():
            rows.extend(json.loads(Path(part).read_text(encoding="utf-8")))
    done = [r for r in rows if r["status"] in ("complete", "already_complete")]
    (out_dir / "job_summary.json").write_text(json.dumps(
        {"batch": batch, "job": job, "completed": len(done),
         "attempted": len(mine), "rows": rows,
         "utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    print("done: %d/%d cells" % (len(done), len(mine)), flush=True)


if __name__ == "__main__":
    sys.exit(main())
