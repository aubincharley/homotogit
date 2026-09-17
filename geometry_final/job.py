"""Kaggle job of geometry_final: centre-frozen traces and the seed-0 centre-frozen grid.

    python -m geometry_final.job --tasks tasks.json --out /kaggle/working/gf [--pilot]

Order: CUDA guard -> input digests -> reuse reproduction (existing centre-frozen
points) -> trace stage 64 and the grid on every GPU -> precision rule -> trace
stages 128 / 256 for failing seed x probe quartets -> manifest -> COMPLETE.json.

Every result is one flushed JSONL line keyed by a canonical key, so a job
restarted on the same ``--out`` (or with ``--resume`` pointing at a previous
output) skips everything already written.  Workers are isolated processes, one
per GPU.  Model buffers and masked weights are hashed before and after every
trace objective.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import base64
import glob
import hashlib
import json
import multiprocessing as mp
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from landscape_v2.run import perturb
from landscape_v3 import common as V3
from landscape_v3.hessian import make_objective
from landscape_v3.run import Ctx, manifest, setup_torch

from . import common as C
from .trace import aggregates, block_ids_and_weights, per_block_quadratic, summary, weights


def _log(path, gpu, msg):
    line = "[%s gpu%s] %s" % (time.strftime("%H:%M:%S"), gpu, msg)
    print(line, flush=True)
    with open(path, "a") as fh:
        fh.write(line + "\n")


def _done_keys(path):
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                keys.add(json.loads(line)["key"])
            except Exception:
                pass            # a torn last line from an interrupted write is ignored and redone
    return keys


def _tensor_sha(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(np.ascontiguousarray(t.detach().to("cpu").numpy()).tobytes())
    return h.hexdigest()


def _buffers(model):
    return [b for n, b in model.named_buffers() if "running" in n or "num_batches" in n]


# ---- trace -------------------------------------------------------------------------

def trace_task(task, ctx, out, deadline, gpu, log_path):
    m, s, probe, j0, j1 = task["m"], task["s"], task["probe"], task["draw_from"], task["draw_to"]
    path = out / "trace" / (task["id"] + ".jsonl")
    done = _done_keys(path)
    todo = [j for j in range(j0, j1) if C.trace_key(m, s, probe, j) not in done]
    meta_path = out / "trace" / ("%s__%d_%d.meta.json" % (task["id"], j0, j1))
    t_start = time.time()
    meta = {"task": task, "resumed": (j1 - j0) - len(todo), "status": "running", "gpu": gpu}
    if not todo:
        meta.update(status="complete", seconds=0.0)
        meta_path.write_text(json.dumps(meta, indent=1))
        return
    try:
        st = ctx.ckpt(m, s, C.FINAL)["model_state"]
        obj = make_objective(ctx.ev, m, st, V3.target(m), C.CFROZEN, probe, "f32", cache_key=(m, s, C.FINAL))
        model = obj.model
        centre_stats = ctx.ev.centre_stats((m, s, C.FINAL), m, st, V3.target(m))
        bid, norm2, sizes = block_ids_and_weights(obj.names, obj.shapes, st)
        w = weights(norm2, sizes)
        G = len(norm2)
        # identity with the existing centre-frozen ordinary Hessian problem
        L0, g = obj.loss_grad()
        ident = task["identity"]
        before = {"buffers_sha": _tensor_sha(_buffers(model)),
                  "centre_stats_sha": _tensor_sha([centre_stats[k] for k in sorted(centre_stats)]),
                  "buffers_equal_centre_stats": all(torch.equal(model.get_buffer(k), v) for k, v in centre_stats.items()),
                  "masked_weights_sha": _tensor_sha([p for _, p in obj.params]),
                  "model_training_flag": bool(model.training)}
        meta["identity"] = {"loss": L0, "grad_norm": float(g.norm()), "expected": ident,
                            "loss_rel_diff": abs(L0 - ident["loss"]) / abs(ident["loss"]),
                            "grad_norm_rel_diff": abs(float(g.norm()) - ident["grad_norm"]) / ident["grad_norm"],
                            "n_params": obj.n, "n_blocks": G}
        blocks_file = out / "trace" / ("blocks__%s__seed%d.npz" % (C.SHORT[m], s))
        if not blocks_file.exists():
            np.savez(blocks_file, norm2=norm2, sizes=sizes)
        fh = path.open("a")
        try:
            for j in todo:
                if time.time() > deadline:
                    raise TimeoutError
                t0 = time.time()
                z_np = C.rademacher(s, j, obj.n)
                z = torch.from_numpy(z_np.astype(np.float64)).to(ctx.device)
                hz = obj.hvp(z)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                q = per_block_quadratic(z, hz, bid, G)
                rec = {"key": C.trace_key(m, s, probe, j), "objective": task["id"], "m": m, "s": s, "probe": probe,
                       "draw": j, "rademacher_seed": C.rademacher_seed(s, j), "z_sha256": C.sha_bytes(z_np.tobytes()),
                       "q_blocks_f64_b64": base64.b64encode(q.astype("<f8").tobytes()).decode("ascii"),
                       "zHz_direct": float(z @ hz), "seconds": time.time() - t0}
                rec.update({"trace_" + k: v for k, v in aggregates(q, w).items()})
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
        finally:
            fh.close()
        after = {"buffers_sha": _tensor_sha(_buffers(model)),
                 "masked_weights_sha": _tensor_sha([p for _, p in obj.params]),
                 "model_training_flag": bool(model.training)}
        meta["state_check"] = {"before": before, "after": after,
                               "buffers_unchanged": before["buffers_sha"] == after["buffers_sha"],
                               "weights_unchanged": before["masked_weights_sha"] == after["masked_weights_sha"]}
        meta["status"] = "complete"
    except TimeoutError:
        meta["status"] = "deadline"
    except Exception:
        meta["status"] = "failed"
        meta["traceback"] = traceback.format_exc()
    meta["seconds"] = time.time() - t_start
    meta["n_new_draws"] = len(todo)
    meta_path.write_text(json.dumps(meta, indent=1, default=str))
    _log(log_path, gpu, "%-44s draws %d-%d %-8s %6.0fs" % (task["id"], j0, j1, meta["status"], meta["seconds"]))


# ---- grid --------------------------------------------------------------------------

def grid_task(task, ctx, out, deadline, gpu, log_path):
    m = task["m"]
    path = out / "grid" / (task["id"] + ".jsonl")
    done = _done_keys(path)
    t_start = time.time()
    meta = {"task": {k: v for k, v in task.items() if k != "points"}, "n_points": len(task["points"]),
            "resumed": len(done), "status": "running", "gpu": gpu}
    try:
        st = ctx.ckpt(m, C.GRID_SEED, C.FINAL)["model_state"]
        c64 = ctx.center64(st)
        tgt = V3.target(m)
        da, rep_a = ctx.direction(m, C.GRID_SEED, C.FINAL, C.GRID_PAIR[0])
        db, rep_b = ctx.direction(m, C.GRID_SEED, C.FINAL, C.GRID_PAIR[1])
        meta["direction_reports"] = [rep_a, rep_b]
        fh = path.open("a")
        try:
            for a, b in task["points"]:
                key = C.grid_key(m, a, b)
                if key in done:
                    continue
                if time.time() > deadline:
                    raise TimeoutError
                p = perturb(c64, [da, db], [a, b], ctx.names)
                r = ctx.ev.evaluate3(m, st, tgt, C.CFROZEN, params=p, splits=C.PROBES, cache_key=(m, C.GRID_SEED, C.FINAL))
                fh.write(json.dumps(dict(r, a=a, b=b, key=key)) + "\n")
                fh.flush()
        finally:
            fh.close()
        meta["status"] = "complete"
    except TimeoutError:
        meta["status"] = "deadline"
    except Exception:
        meta["status"] = "failed"
        meta["traceback"] = traceback.format_exc()
    meta["seconds"] = time.time() - t_start
    (out / "grid" / (task["id"] + ".meta.json")).write_text(json.dumps(meta, indent=1, default=str))
    _log(log_path, gpu, "%-44s %-8s %6.0fs" % (task["id"], meta["status"], meta["seconds"]))


def reuse_check(bundle, ctx, out):
    """Re-evaluate the existing centre-frozen points that stand in for grid vertices,
    and the pointwise centre, which must equal the centre-frozen centre."""
    rows, worst = [], 0.0
    for grp in bundle["reuse_checks"]:
        m = grp["m"]
        st = ctx.ckpt(m, C.GRID_SEED, C.FINAL)["model_state"]
        c64 = ctx.center64(st)
        tgt = V3.target(m)
        da, _ = ctx.direction(m, C.GRID_SEED, C.FINAL, 0)
        db, _ = ctx.direction(m, C.GRID_SEED, C.FINAL, 1)
        for pt in grp["points"]:
            p = perturb(c64, [da, db], [pt["a"], pt["b"]], ctx.names)
            r = ctx.ev.evaluate3(m, st, tgt, C.CFROZEN, params=p, splits=C.PROBES, cache_key=(m, C.GRID_SEED, C.FINAL))
            diff = max(abs(r["splits"][sp]["ce"] - pt["expected"][sp]) for sp in C.PROBES)
            worst = max(worst, diff)
            rows.append({"m": m, "a": pt["a"], "b": pt["b"], "source_key": pt["source_key"], "abs_diff_ce": diff,
                         "grid_key": C.grid_key(m, pt["a"], pt["b"]),
                         "value": {sp: r["splits"][sp]["ce"] for sp in C.PROBES}})
        pw = ctx.ev.evaluate3(m, st, tgt, C.POINTWISE, splits=C.PROBES)
        cf = ctx.ev.evaluate3(m, st, tgt, C.CFROZEN, splits=C.PROBES, cache_key=(m, C.GRID_SEED, C.FINAL))
        rows.append({"m": m, "centre_pointwise": {sp: pw["splits"][sp]["ce"] for sp in C.PROBES},
                     "centre_centre_frozen": {sp: cf["splits"][sp]["ce"] for sp in C.PROBES},
                     "centre_bitwise_equal": all(pw["splits"][sp]["ce"] == cf["splits"][sp]["ce"] for sp in C.PROBES)})
    tol = bundle["constants"]["reuse_tolerance_ce"]
    res = {"n": len(rows), "max_abs_diff_ce": worst, "tolerance": tol, "valid": worst <= tol,
           "centres_bitwise_equal": all(r.get("centre_bitwise_equal", True) for r in rows), "rows": rows}
    (out / "checks" / "reuse_reproduction.json").write_text(json.dumps(res, indent=1))
    return res


# ---- workers and stages -------------------------------------------------------------

def _worker(q, gpu, inputs, data_root, out, deadline, log_path):
    setup_torch()
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
    dev = "cuda:%d" % gpu if torch.cuda.is_available() else "cpu"
    ctx = Ctx(inputs, data_root, dev)
    while True:
        try:
            task = q.get_nowait()
        except Exception:
            return
        (trace_task if task["kind"] == "trace" else grid_task)(task, ctx, Path(out), deadline, gpu, log_path)


def run_stage(tasks, n_gpu, inputs, data_root, out, deadline):
    ctxm = mp.get_context("spawn")
    q = ctxm.Queue()
    for t in tasks:
        q.put(t)
    procs = [ctxm.Process(target=_worker, args=(q, g, str(inputs), data_root, str(out), deadline,
                                                 str(out / "job_log.txt"))) for g in range(n_gpu)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    return [p.exitcode for p in procs]


def read_trace(out):
    per = {}
    for f in (out / "trace").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            per.setdefault(r["objective"], {})[r["draw"]] = r
    return per


def precision_status(bundle, out, n_have):
    """Rule, per seed x probe quartet: every one of the 12 traces (4 methods x 3 kinds)
    must have SEM <= 5% |estimate| using all draws so far; otherwise extend the quartet."""
    per = read_trace(out)
    rel = bundle["constants"]["trace_sem_rel"]
    status = {}
    for s in C.SEEDS:
        for probe in C.PROBES:
            ok, detail = True, {}
            for m in C.METHODS:
                oid = C.objective_id(m, s, probe)
                draws = per.get(oid, {})
                vals = [draws[j] for j in range(n_have) if j in draws]
                if len(vals) < n_have:
                    ok = False
                    detail[oid] = {"missing_draws": n_have - len(vals)}
                    continue
                d = {}
                for k in C.TRACE_KINDS:
                    sm = summary([v["trace_" + k] for v in vals])
                    d[k] = sm
                    ok &= sm["sem_rel"] <= rel
                detail[oid] = d
            status["s%d_%s" % (s, probe)] = {"meets_rule": bool(ok), "n_draws": n_have, "detail": detail}
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--inputs")
    ap.add_argument("--data-root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume")
    ap.add_argument("--deadline-hours", type=float, default=11.0)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--allow-cpu", dest="require_gpu", action="store_false")
    ap.add_argument("--gpus", type=int)
    a = ap.parse_args()
    t_start = time.time()
    deadline = t_start + a.deadline_hours * 3600
    out = Path(a.out)
    for sd in ("trace", "grid", "checks"):
        (out / sd).mkdir(parents=True, exist_ok=True)
    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "n_gpu": torch.cuda.device_count(),
           "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
           "numpy": np.__version__, "pilot": a.pilot, "tf32": False, "cudnn_deterministic": True}
    (out / "environment.json").write_text(json.dumps(env, indent=2))
    print(env, flush=True)
    if a.require_gpu and torch.cuda.device_count() == 0:
        (out / "NO_GPU_ABORTED.txt").write_text("no CUDA device visible; job aborted before any evaluation\n")
        raise SystemExit(3)
    inputs = Path(a.inputs) if a.inputs else Path(sorted(glob.glob("/kaggle/input/**/expected_digests.json", recursive=True))[0]).parent
    data_root = a.data_root or str(Path(sorted(glob.glob("/kaggle/input/**/cifar-10-batches-py/batches.meta", recursive=True))[0]).parent.parent)
    if a.resume and Path(a.resume).exists():
        for sd in ("trace", "grid"):
            for f in (Path(a.resume) / sd).glob("*.jsonl"):
                if not (out / sd / f.name).exists():
                    shutil.copy2(f, out / sd / f.name)
    bundle = json.loads(Path(a.tasks).read_text())
    timing = {"start": t_start}

    # inputs
    t0 = time.time()
    dig = json.loads((inputs / "expected_digests.json").read_text())
    need = ["%s__seed%d__%s" % (m, s, C.FINAL) for m in C.METHODS for s in C.SEEDS]
    chk = {"checkpoints": {f: V3.sha_file(inputs / f) == dig["checkpoints"][f] for f in need},
           "directions": {f: V3.sha_file(inputs / f) == dig["direction_files"][f] for f in ("seed0_dir00.pt", "seed0_dir01.pt")},
           "subsets": V3.sha_file(inputs / "subsets.npz") == dig["subsets_file"],
           "expected_digests_sha256": V3.sha_file(inputs / "expected_digests.json")}
    chk["all_ok"] = all(chk["checkpoints"].values()) and all(chk["directions"].values()) and chk["subsets"]
    (out / "checks" / "inputs_check.json").write_text(json.dumps(chk, indent=1))
    timing["input_check_seconds"] = time.time() - t0
    if not chk["all_ok"]:
        raise SystemExit("input digest mismatch; nothing evaluated")
    n_gpu = a.gpus or max(torch.cuda.device_count(), 1)

    if a.pilot:
        # timing pilot of the missing work only: HVP draws on one objective per GPU and grid points
        tasks = []
        for g, (m, probe) in enumerate((("plain", "train_probe"), ("resolution_max_b1_gaussian_conv", "test_probe"))[:n_gpu]):
            oid = C.objective_id(m, 0, probe)
            tasks.append({"kind": "trace", "id": oid, "m": m, "s": 0, "probe": probe, "draw_from": 0, "draw_to": 8,
                          "identity": bundle["identity"][oid]})
            gt = next(x for x in bundle["grid"] if x["m"] == m)
            tasks.append({"kind": "grid", "id": gt["id"], "m": m, "points": gt["points"][:60]})
        t0 = time.time()
        run_stage(tasks, n_gpu, inputs, data_root, out, deadline)
        timing["pilot_wall_seconds"] = time.time() - t0
    else:
        setup_torch()
        ctx = Ctx(inputs, data_root, "cuda:0" if torch.cuda.is_available() else "cpu")
        t0 = time.time()
        rr = reuse_check(bundle, ctx, out)
        timing["reuse_check_seconds"] = time.time() - t0
        del ctx
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("reuse valid", rr["valid"], rr["max_abs_diff_ce"], "centres equal", rr["centres_bitwise_equal"], flush=True)
        grid_points = [dict(g, kind="grid") for g in bundle["grid"]]
        if not rr["valid"]:
            # the reused axis points did not reproduce: evaluate every vertex instead
            axis = bundle["constants"]["grid_axis"]
            grid_points = [dict(g, kind="grid", points=[[x, y] for x in axis for y in axis]) for g in bundle["grid"]]
        stages, prev, status_log = bundle["constants"]["trace_stages"], 0, []
        extend = [(s, probe) for s in C.SEEDS for probe in C.PROBES]
        for i, n in enumerate(stages):
            if not extend:
                break
            tasks = [{"kind": "trace", "id": C.objective_id(m, s, probe), "m": m, "s": s, "probe": probe,
                      "draw_from": prev, "draw_to": n, "identity": bundle["identity"][C.objective_id(m, s, probe)]}
                     for s, probe in extend for m in C.METHODS]
            # long tasks first; the grid shares the first stage
            if i == 0:
                tasks = tasks[:len(tasks) // 2] + grid_points + tasks[len(tasks) // 2:]
            t0 = time.time()
            codes = run_stage(tasks, n_gpu, inputs, data_root, out, deadline)
            timing["stage_%d_wall_seconds" % n] = time.time() - t0
            st = precision_status(bundle, out, n)
            status_log.append({"n_draws": n, "worker_exit_codes": codes,
                               "quartets": {k: v["meets_rule"] for k, v in st.items()}})
            (out / "checks" / ("precision_after_%d.json" % n)).write_text(json.dumps(st, indent=1))
            extend = [(s, probe) for s in C.SEEDS for probe in C.PROBES if not st["s%d_%s" % (s, probe)]["meets_rule"]]
            prev = n
            if time.time() > deadline:
                break
        (out / "checks" / "precision_log.json").write_text(json.dumps(status_log, indent=1))
    # timing summary from records
    per_hvp, per_pt = [], []
    for f in (out / "trace").glob("*.jsonl"):
        per_hvp += [json.loads(l)["seconds"] for l in f.read_text().splitlines() if l.strip()]
    for f in (out / "grid").glob("*.jsonl"):
        per_pt += [json.loads(l)["seconds"] for l in f.read_text().splitlines() if l.strip()]
    metas = [json.loads(p.read_text()) for p in list((out / "trace").glob("*.meta.json")) + list((out / "grid").glob("*.meta.json"))]
    timing.update({"n_gpus": n_gpu, "total_seconds": time.time() - t_start,
                   "gpu_seconds_sum_of_tasks": sum(mt.get("seconds", 0.0) for mt in metas),
                   "hvp_draw_seconds": summary(per_hvp) if len(per_hvp) > 1 else per_hvp,
                   "hvp_draw_seconds_median": float(np.median(per_hvp)) if per_hvp else None,
                   "grid_point_seconds_median": float(np.median(per_pt)) if per_pt else None,
                   "n_hvp_draws": len(per_hvp), "n_grid_points": len(per_pt),
                   "task_status": {k: sum(1 for mt in metas if mt["status"] == k) for k in {mt["status"] for mt in metas}}})
    (out / "timing.json").write_text(json.dumps(timing, indent=2))
    n_files = manifest(out, "manifest.json", ["trace", "grid", "checks"])
    complete = {"n_manifest_files": n_files, "task_status": timing["task_status"],
                "n_hvp_draws": len(per_hvp), "n_grid_points": len(per_pt), "pilot": a.pilot,
                "all_tasks_complete": all(mt["status"] == "complete" for mt in metas),
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "COMPLETE.json").write_text(json.dumps(complete, indent=1))
    print("job done", json.dumps(complete), flush=True)


if __name__ == "__main__":
    main()
