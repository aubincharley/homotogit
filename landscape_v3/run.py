"""One Kaggle job of landscape_v3 (evaluation only; no training).

    python -m landscape_v3.run --tasks tasks.json --out /kaggle/working/v3 [--pilot]

Order: input digests -> reuse reproduction -> task queue on every GPU.  Each task
writes ``eval/<id>.jsonl`` (one flushed line per point, resumable by key; key =
``common.pkey``) and ``eval/<id>.meta.json`` (status complete / deadline /
failed / blocked, seconds on its GPU).  Hessian tasks write ``hessian/<problem>.json``
and ``hessian/<problem>_vectors.pt``.  ``eval_manifest.json`` holds the sha256 of
every produced file.  Non-finite values are written as they are.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import glob
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from continuation_core.config import DataConfig
from continuation_core.data import load_dataset
from landscape_study.directions import normalization_report, scale_to
from landscape_v2.run import perturb

from . import common as C
from .evaluator import Evaluator
from .hessian import block_index, block_scales, lanczos, make_objective, rms_relative

PROTOCOL = {"lanczos_m_max": 400, "lanczos_check_every": 10, "lanczos_tol": 1e-3, "lanczos_floor": 1e-2,
            "reuse_tolerance_ce": 1e-5, "hess_fd_t": list(C.FD_T)}


def setup_torch():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


class Ctx:
    def __init__(self, inputs, data_root, device):
        self.inputs = Path(inputs)
        subsets = dict(np.load(self.inputs / "subsets.npz"))
        self.ev = Evaluator(load_dataset(DataConfig(root=data_root)), subsets, subsets["pinned_train_probe_500"], device)
        for m in C.METHODS:
            self.ev.model(m)
        self.names = self.ev.names
        self.device = torch.device(device)
        self._ck, self._dir = {}, {}

    def ckpt(self, m, s, name):
        key = (m, s, name)
        if key not in self._ck:
            if len(self._ck) > 16:
                self._ck.clear()
                self._dir.clear()
            self._ck[key] = torch.load(self.inputs / ("%s__%s" % (C.rname(m, s), name)), map_location="cpu",
                                       weights_only=False)
        return self._ck[key]

    def center64(self, st):
        return {n: st[n].to(self.device, torch.float64) for n in self.names}

    def direction(self, m, s, name, k):
        key = (m, s, name, k)
        if key not in self._dir:
            st = self.ckpt(m, s, name)["model_state"]
            draw = torch.load(self.inputs / ("seed%d_dir%02d.pt" % (s, k)), map_location="cpu", weights_only=True)
            cpu = {n: st[n] for n in self.names}
            d, _ = scale_to(cpu, draw, self.names)
            rep = normalization_report(cpu, d, self.names)
            rep.pop("masked_tensors")
            self._dir[key] = ({n: t.to(self.device) for n, t in d.items()}, rep)
        return self._dir[key]


def vec_to_dict(obj, names, vec):
    parts = dict(zip(obj.names, obj._split(vec)))
    return {n: (parts[n] if n in parts else None) for n in names}


def add_dict(c64, names, pairs, dtype):
    out = {}
    for n in names:
        t = c64[n]
        for coef, dd in pairs:
            if dd[n] is not None:
                t = t + coef * dd[n]
        out[n] = t.to(dtype)
    return out


# ----------------------------------------------------------------------------------

def run_task(task, ctx, out, deadline, reuse_valid, pilot):
    path = out / "eval" / (task["id"] + ".jsonl")
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                done.add(json.loads(line)["key"])
            except Exception:
                pass
    t_start = time.time()
    meta = {"task": {k: v for k, v in task.items() if k not in ("points", "reused_points")},
            "n_points": len(task.get("points", [])), "n_reused_points": len(task.get("reused_points", [])),
            "config_sha256": C.sha_json(task), "started": t_start, "resumed_points": len(done), "status": "running"}
    fh = path.open("a")
    ev, names = ctx.ev, ctx.names

    def emit(key, rec):
        rec["key"] = key
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        done.add(key)

    def guard():
        if time.time() > deadline:
            raise TimeoutError

    kind = task["kind"]
    try:
        if kind in ("pts", "grid"):
            m, s, name, state, prec = task["m"], task["s"], task["ckpt"], task["state"], task["prec"]
            splits = C.SPLITSETS[task["splitset"]]
            ck = ctx.ckpt(m, s, name)
            st = ck["model_state"]
            c64 = ctx.center64(st)
            dtype = torch.float64 if prec == "f64" else torch.float32
            pts = list(task["points"]) + ([] if reuse_valid else list(task["reused_points"]))
            if kind == "pts":
                pts.sort(key=lambda p: (p[1], p[2], p[0]))
                cache = (None, None)
                for pol, k, amp in pts:
                    guard()
                    spec = "c" if k < 0 else C.spec_r1(k, amp)
                    key = C.pkey(m, s, name, state, pol, task["splitset"], spec, prec)
                    if key in done:
                        continue
                    params = None
                    if k >= 0:
                        if cache[0] != (k, amp):
                            d, rep = ctx.direction(m, s, name, k)
                            if prec == "f32":
                                p = perturb(c64, [d], [amp], names)
                            else:
                                p = {n: c64[n] + amp * d[n] for n in names}
                            cache = ((k, amp), p)
                        params = cache[1]
                    r = ev.evaluate3(m, st, state, pol, params=params, splits=splits, prec=prec, cache_key=(m, s, name))
                    emit(key, dict(r, k=None if k < 0 else k, amp=amp, ckpt=name, global_update=ck["global_update"],
                                   intervention_record=ck["intervention"] if k < 0 else None))
            else:
                da, _ = ctx.direction(m, s, name, task["pair"][0])
                db, _ = ctx.direction(m, s, name, task["pair"][1])
                for pol, a, b in pts:
                    guard()
                    key = C.pkey(m, s, name, state, pol, task["splitset"], C.spec_r2(0, 1, a, b), prec)
                    if key in done:
                        continue
                    p = perturb(c64, [da, db], [a, b], names)
                    emit(key, dict(ev.evaluate3(m, st, state, pol, params=p, splits=splits, prec=prec,
                                                cache_key=(m, s, name)), a=a, b=b))
        elif kind == "hchecks":
            hchecks(task, ctx, out, emit, done, guard, pilot)
        elif kind == "hessian":
            hessian_task(task, ctx, out, guard, pilot)
        elif kind == "quadform":
            quadform(task, ctx, emit, done, guard)
        elif kind in ("hcut", "hplane"):
            hcut_plane(task, ctx, out, emit, done, guard, deadline, pilot)
        meta["status"] = "complete"
    except TimeoutError:
        meta["status"] = "deadline"
    except FileNotFoundError as exc:
        meta["status"] = "blocked"
        meta["traceback"] = repr(exc)
    except Exception:
        meta["status"] = "failed"
        meta["traceback"] = traceback.format_exc()
    finally:
        fh.close()
    meta["seconds"] = time.time() - t_start
    (out / "eval" / (task["id"] + ".meta.json")).write_text(json.dumps(meta, indent=2, default=str))
    return meta["status"], meta["seconds"]


# -- Hessian ------------------------------------------------------------------------

def hchecks(task, ctx, out, emit, done, guard, pilot):
    m, s = task["m"], task["s"]
    ev = ctx.ev
    st = ctx.ckpt(m, s, C.FINAL)["model_state"]
    tgt = C.target(m)
    hs = (1e-2, 1e-3) if pilot else (1e-2, 1e-3, 1e-4)
    for pol in C.HESS_POLICIES:
        key = "hchecks|%s" % pol
        if key in done:
            continue
        guard()
        r = {"m": m, "s": s, "policy": pol, "probe": "train_probe"}
        obj = make_objective(ev, m, st, tgt, pol, "train_probe", "f32", cache_key=(m, s, C.FINAL))
        g = torch.Generator().manual_seed(C.lanczos_seed("hchecks_%s_%d" % (m, s), 7))
        u = torch.randn(obj.n, generator=g, dtype=torch.float64).to(ctx.device)
        v = torch.randn(obj.n, generator=g, dtype=torch.float64).to(ctx.device)
        t0 = time.time()
        hv = obj.hvp(v)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        r["hvp_seconds_f32"] = time.time() - t0
        hv2 = obj.hvp(v)
        r["hvp_repeat_bitwise"] = bool(torch.equal(hv, hv2))
        r["hvp_repeat_max_abs_diff"] = float((hv - hv2).abs().max())
        hu = obj.hvp(u)
        r["symmetry_rel"] = float(abs(u @ hv - v @ hu) / max(abs(u @ hv), 1e-30))
        r["loss_f32"], gr = obj.loss_grad()
        r["grad_norm_f32"] = float(gr.norm())
        obj64 = make_objective(ev, m, st, tgt, pol, "train_probe", "f64", cache_key=(m, s, C.FINAL))
        t0 = time.time()
        hv64 = obj64.hvp(v)
        r["hvp_seconds_f64"] = time.time() - t0
        r["hvp_f32_vs_f64_rel"] = float((hv - hv64).norm() / hv64.norm())
        w0 = obj64.vector()
        off = sum(obj64.sizes[:-1])
        for lab, vv in (("all_masked", v.clone()), ("fc_only", torch.cat([torch.zeros(off, dtype=torch.float64,
                                                                                      device=ctx.device), v[off:]]))):
            sub = w0 if lab == "all_masked" else w0[off:]
            vn = vv / vv.norm() * sub.norm()
            ref = obj64.hvp(vn)
            errs = {}
            for h in hs:
                guard()
                obj64.set_vector(w0 + h * vn)
                _, gp = obj64.loss_grad()
                obj64.set_vector(w0 - h * vn)
                _, gm = obj64.loss_grad()
                errs[C.fnum(h)] = float(((gp - gm) / (2 * h) - ref).norm() / ref.norm())
            obj64.set_vector(w0)
            r["fd_gradient_vs_hvp_rel_err_%s" % lab] = errs
        obj.set_vector(obj.vector())
        emit(key, r)
        del obj, obj64


def problem_objective(ctx, task, prec="f32"):
    m, s = task["m"], task["s"]
    st = ctx.ckpt(m, s, C.FINAL)["model_state"]
    return make_objective(ctx.ev, m, st, C.target(m), task["policy"], task["probe"], prec, cache_key=(m, s, C.FINAL)), st


def hessian_task(task, ctx, out, guard, pilot):
    pid, coords = task["problem"], task["coords"]
    hd = out / "hessian"
    hd.mkdir(exist_ok=True)
    if (hd / (pid + ".json")).exists():
        return
    obj, st = problem_objective(ctx, task)
    a, info = block_scales(obj, st)
    rel = coords == "relative"
    mask = (a > 0).to(torch.float64) if rel else None
    op = (lambda q: a * obj.hvp(a * q)) if rel else obj.hvp
    L0, g = obj.loss_grad()
    res = {"problem": pid, "task": task, "loss": L0, "grad_norm": float(g.norm()),
           "grad_norm_relative_coords": float((a * g).norm()), "block_info": info, "n_params": obj.n,
           "protocol": PROTOCOL, "starts": {}}
    m_max = 40 if pilot else PROTOCOL["lanczos_m_max"]
    vecs = None
    for start in (C.HESS_STARTS[:1] if pilot else C.HESS_STARTS):
        guard()
        gen = torch.Generator().manual_seed(C.lanczos_seed(pid, start))
        v0 = torch.randn(obj.n, generator=gen, dtype=torch.float64)
        calls0 = obj.hvp_calls
        r = lanczos(op, obj.n, v0, m_max=m_max, check_every=PROTOCOL["lanczos_check_every"],
                    tol=PROTOCOL["lanczos_tol"], floor=PROTOCOL["lanczos_floor"], device=ctx.device, mask=mask)
        r["hvp_calls"] = obj.hvp_calls - calls0
        r["start_seed"] = C.lanczos_seed(pid, start)
        if start == 0:
            vecs = {k: v.clone() for k, v in r["vectors"].items()}
        r.pop("vectors")
        res["starts"][str(start)] = r
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if "1" in res["starts"]:
        r0, r1 = res["starts"]["0"], res["starts"]["1"]
        res["start_agreement_rel"] = {k: abs(r0["values"][k] - r1["values"][k]) / max(abs(r0["values"][k]), 1e-12)
                                      for k in r0["values"]}
    # verification of start-0 extreme vectors: Rayleigh quotients (f32, f64) and float64 finite differences
    bid, nb = block_index(obj)
    w0 = obj.vector()
    ver = {}
    m, s = task["m"], task["s"]
    c64 = ctx.center64(st)
    for vk in ("top1", "min"):
        guard()
        # rebuilt for every vector: the float64 evaluations below share this model and leave it perturbed
        obj64 = make_objective(ctx.ev, m, st, C.target(m), task["policy"], task["probe"], "f64", cache_key=(m, s, C.FINAL))
        obj = make_objective(ctx.ev, m, st, C.target(m), task["policy"], task["probe"], "f32", cache_key=(m, s, C.FINAL))
        y = vecs[vk]
        delta = a * y if rel else y
        r_delta, G = rms_relative(delta, w0, bid, nb)
        dr = delta / r_delta
        dn = torch.zeros(nb, dtype=torch.float64).index_add_(0, bid, dr.cpu() ** 2)
        tn = torch.zeros(nb, dtype=torch.float64).index_add_(0, bid, w0.cpu() ** 2)
        blk = torch.sqrt(dn[tn > 0] / tn[tn > 0])
        quad32 = float(dr @ obj.hvp(dr))
        quad64 = float(dr @ obj64.hvp(dr))
        dd = vec_to_dict(obj, ctx.names, dr)
        base = ctx.ev.evaluate3(m, st, C.target(m), task["policy"], splits=(task["probe"],), prec="f64",
                                cache_key=(m, s, C.FINAL))["splits"][task["probe"]]["ce"]
        fd = {}
        for t in ([0.01] if pilot else PROTOCOL["hess_fd_t"]):
            ls = []
            for sg in (1, -1):
                p = add_dict(c64, ctx.names, [(sg * t, dd)], torch.float64)
                ls.append(ctx.ev.evaluate3(m, st, C.target(m), task["policy"], params=p, splits=(task["probe"],),
                                           prec="f64", cache_key=(m, s, C.FINAL))["splits"][task["probe"]]["ce"])
            S = 0.5 * sum(ls) - base
            fd[C.fnum(t)] = {"S": S, "C_fd": 2 * S / t ** 2, "L_plus": ls[0], "L_minus": ls[1]}
        ver[vk] = {"ritz_value": res["starts"]["0"]["values"][vk], "r_of_mapped_direction": r_delta, "G": G,
                   "max_block_relative_displacement_at_t1": float(blk.max()),
                   "n_blocks_holding_90pct_of_r2": int((torch.cumsum(torch.sort(blk ** 2, descending=True).values, 0)
                                                        < 0.9 * (blk ** 2).sum()).sum()) + 1,
                   "quadratic_f32_r_normalised": quad32, "quadratic_f64_r_normalised": quad64,
                   "rayleigh_f32_coordinates": quad32 * r_delta ** 2, "rayleigh_f64_coordinates": quad64 * r_delta ** 2,
                   "gradient_dot_direction_r_normalised": float(g @ dr), "L0_f64": base, "finite_differences": fd}
    res["verification"] = ver
    del obj64
    torch.save({k: v.to(torch.float32).cpu() for k, v in vecs.items()}, hd / (pid + "_vectors.pt"))
    (hd / (pid + ".json")).write_text(json.dumps(res, indent=1, default=str))


def quadform(task, ctx, emit, done, guard):
    m, s = task["m"], task["s"]
    st = ctx.ckpt(m, s, C.FINAL)["model_state"]
    for probe in C.HESS_PROBES:
        for pol in C.HESS_POLICIES:
            obj = make_objective(ctx.ev, m, st, C.target(m), pol, probe, "f32", cache_key=(m, s, C.FINAL))
            L0, g = obj.loss_grad()
            for k in range(C.N_DIRECTIONS):
                key = "quad|%s|%s|%d" % (probe, pol, k)
                if key in done:
                    continue
                guard()
                d, _ = ctx.direction(m, s, C.FINAL, k)
                dv = torch.cat([d[n].reshape(-1) for n in obj.names]).to(ctx.device, torch.float64)
                emit(key, {"m": m, "s": s, "probe": probe, "policy": pol, "k": k, "dHd": float(dv @ obj.hvp(dv)),
                           "g_dot_d": float(g @ dv), "L0": L0, "grad_norm": float(g.norm())})


def hcut_plane(task, ctx, out, emit, done, guard, deadline, pilot):
    m, s = task["m"], task["s"]
    hd = out / "hessian"
    if task["kind"] == "hcut":
        pids = task.get("pids") or [C.hess_problem(m, s, pr, pol, co) for pr in C.HESS_PROBES
                                    for pol in C.HESS_POLICIES for co in C.HESS_COORDS]
    else:
        pids = [task["problem"]]
    avail = set(task.get("available_problems", []))
    need = set(pids) | ({C.hess_problem("plain", s, "train_probe", task["policy"], "relative")} if task["kind"] == "hplane" else set())
    if not need <= avail:
        raise FileNotFoundError("Hessian problems not scheduled in this job: %s" % sorted(need - avail))
    while True:
        guard()
        missing = [p for p in pids if not (hd / (p + "_vectors.pt")).exists()]
        if not missing:
            break
        failed = [p for p in missing if (out / "eval" / ("hessian__%s.meta.json" % p)).exists()]
        if failed and len(failed) == len(missing):
            raise FileNotFoundError("Hessian vectors missing (task not complete): %s" % failed)
        time.sleep(20)
    st = ctx.ckpt(m, s, C.FINAL)["model_state"]
    c64 = ctx.center64(st)
    tgt = C.target(m)
    obj = make_objective(ctx.ev, m, st, tgt, C.SAVED, "train_probe", "f32", cache_key=(m, s, C.FINAL))
    a, _ = block_scales(obj, st)
    bid, nb = block_index(obj)
    w0 = obj.vector()
    del obj

    def mapped(pid, vk):
        vecs = torch.load(hd / (pid + "_vectors.pt"), map_location="cpu", weights_only=True)
        y = vecs[vk].to(ctx.device, torch.float64)
        delta = a * y if pid.endswith("_rel") else y
        r_delta, G = rms_relative(delta, w0, bid, nb)
        tmp = make_objective(ctx.ev, m, st, tgt, C.SAVED, "train_probe", "f32", cache_key=(m, s, C.FINAL))
        dd = vec_to_dict(tmp, ctx.names, delta / r_delta)
        return dd, r_delta, G

    ts = list(C.CUT_T)
    if pilot:
        ts = ts[::10]
    if task["kind"] == "hcut":
        for pid in pids:
            pol = C.SAVED if "_saved_" in pid else C.CFROZEN
            vks = ("top1", "top2", "min") if pid.endswith("_rel") else ("top1", "min")
            evals = [pol] + ([C.POINTWISE] if pid.endswith("train_centre_frozen_rel") else [])
            for vk in vks:
                dd, r_delta, G = mapped(pid, vk)
                for t in ts:
                    for ep in evals:
                        if ep == C.POINTWISE and vk == "top2":
                            continue
                        key = C.pkey(m, s, C.FINAL, tgt, ep, "probes", C.spec_h1(pid, vk, t))
                        if key in done:
                            continue
                        guard()
                        p = add_dict(c64, ctx.names, [(t, dd)], torch.float32)
                        emit(key, dict(ctx.ev.evaluate3(m, st, tgt, ep, params=p, splits=C.PROBES,
                                                        cache_key=(m, s, C.FINAL)),
                                       problem=pid, vec=vk, t=t, r_of_raw_direction=r_delta, G=G))
    else:
        pid, (v1, v2), pol = task["problem"], task["pair"], task["policy"]
        plain_pid = C.hess_problem("plain", s, "train_probe", pol, "relative")
        while not (hd / (plain_pid + ".json")).exists():
            guard()
            if (out / "eval" / ("hessian__%s.meta.json" % plain_pid)).exists():
                raise FileNotFoundError("plain Hessian (plane extents) missing: %s" % plain_pid)
            time.sleep(20)
        ph = json.loads((hd / (plain_pid + ".json")).read_text())
        Gp = ph["block_info"]["n_nonzero_blocks"]
        T = {v: float(np.sqrt(2 * C.PLANE_RISE / (Gp * abs(ph["starts"]["0"]["values"][v])))) for v in (v1, v2)}
        d1, r1, G = mapped(pid, v1)
        d2, r2, _ = mapped(pid, v2)
        grid = list(C.HESS_PLANE_U)[::20] if pilot else list(C.HESS_PLANE_U)
        for a_ in grid:
            for b_ in grid:
                key = C.pkey(m, s, C.FINAL, tgt, pol, "probes", C.spec_h2(pid, v1, v2, a_, b_))
                if key in done:
                    continue
                guard()
                p = add_dict(c64, ctx.names, [(a_ * T[v1], d1), (b_ * T[v2], d2)], torch.float32)
                emit(key, dict(ctx.ev.evaluate3(m, st, tgt, pol, params=p, splits=C.PROBES, cache_key=(m, s, C.FINAL)),
                               problem=pid, v1=v1, v2=v2, u1=a_, u2=b_, t1=a_ * T[v1], t2=b_ * T[v2],
                               T1=T[v1], T2=T[v2], extent_rule="T = sqrt(2*%g/(G*|lambda_plain|)), %s" % (C.PLANE_RISE, plain_pid)))


# ----------------------------------------------------------------------------------

def _worker(q, gpu, inputs, data_root, out, deadline, log_path, reuse_valid, pilot):
    def log(msg):
        line = "[%s gpu%d] %s" % (time.strftime("%H:%M:%S"), gpu, msg)
        print(line, flush=True)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")
    setup_torch()
    dev = "cuda:%d" % gpu if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
    ctx = Ctx(inputs, data_root, dev)
    while True:
        try:
            task = q.get_nowait()
        except Exception:
            return
        status, sec = run_task(task, ctx, Path(out), deadline, reuse_valid, pilot)
        log("%-70s %-9s %7.0fs" % (task["id"][:70], status, sec))


def find(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return Path(hits[0]).parent if hits else None


def check_inputs(inputs, tasks, out):
    dig = json.loads((inputs / "expected_digests.json").read_text())
    rep = {"checkpoints": {}, "directions": {}, "blocked_files": []}
    needed = set()
    for t in tasks:
        if "ckpt" in t:
            needed.add("%s__%s" % (C.rname(t["m"], t["s"]), t["ckpt"]))
        if t["kind"] in ("hchecks", "hessian", "quadform", "hcut", "hplane"):
            needed.add("%s__%s" % (C.rname(t["m"], t["s"]), C.FINAL))
    for f in sorted(needed):
        p = inputs / f
        ok = p.exists() and C.sha_file(p) == dig["checkpoints"].get(f)
        rep["checkpoints"][f] = ok
        if not ok:
            rep["blocked_files"].append(f)
    for p in sorted(inputs.glob("seed*_dir*.pt")):
        rep["directions"][p.name] = C.sha_file(p) == dig["direction_files"].get(p.name)
    rep["subsets_ok"] = C.sha_file(inputs / "subsets.npz") == dig["subsets_file"]
    rep["all_ok"] = not rep["blocked_files"] and all(rep["directions"].values()) and rep["subsets_ok"]
    (out / "checks").mkdir(parents=True, exist_ok=True)
    (out / "checks" / "inputs_check.json").write_text(json.dumps(rep, indent=1))
    return rep


def reuse_repro(inputs, data_root, samples, out):
    setup_torch()
    ctx = Ctx(inputs, data_root, "cuda:0" if torch.cuda.is_available() else "cpu")
    rows, worst = [], 0.0
    for smp in samples:
        m, s, name, state = smp["m"], smp["s"], smp["ckpt"], smp["state"]
        st = ctx.ckpt(m, s, name)["model_state"]
        c64 = ctx.center64(st)
        splits = C.SPLITSETS[smp["splitset"]]
        if smp["kind"] == "pts":
            pol, k, amp = smp["point"]
            params = None if k < 0 else perturb(c64, [ctx.direction(m, s, name, k)[0]], [amp], ctx.names)
        else:
            pol, a, b = smp["point"]
            params = perturb(c64, [ctx.direction(m, s, name, 0)[0], ctx.direction(m, s, name, 1)[0]], [a, b], ctx.names)
        r = ctx.ev.evaluate3(m, st, state, pol, params=params, splits=splits)
        diff = max(abs(r["splits"][sp]["ce"] - smp["expected"][sp]["ce"]) for sp in splits)
        worst = max(worst, diff)
        rows.append({"key": smp["key"], "abs_diff_ce": diff})
    res = {"n": len(rows), "max_abs_diff_ce": worst, "tolerance": PROTOCOL["reuse_tolerance_ce"],
           "valid": worst <= PROTOCOL["reuse_tolerance_ce"], "rows": rows}
    (out / "checks" / "reuse_reproduction.json").write_text(json.dumps(res, indent=1))
    del ctx
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return res


def manifest(root: Path, name: str, subdirs):
    files = {}
    for sd in subdirs:
        if not (root / sd).exists():
            continue
        for p in sorted((root / sd).rglob("*")):
            if p.is_file():
                files[p.relative_to(root).as_posix()] = {"sha256": C.sha_file(p), "bytes": p.stat().st_size}
    (root / name).write_text(json.dumps(files, indent=1))
    return len(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--inputs")
    ap.add_argument("--data-root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline-hours", type=float, default=11.3)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    t_start = time.time()
    deadline = t_start + a.deadline_hours * 3600
    inputs = Path(a.inputs) if a.inputs else find("/kaggle/input/**/expected_digests.json")
    data_root = a.data_root or str(find("/kaggle/input/**/cifar-10-batches-py/batches.meta").parent)
    out = Path(a.out)
    (out / "eval").mkdir(parents=True, exist_ok=True)
    bundle = json.loads(Path(a.tasks).read_text())
    tasks, samples = bundle["tasks"], bundle["reuse_samples"]
    PROTOCOL.update(bundle.get("protocol", {}))
    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "n_gpu": torch.cuda.device_count(),
           "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
           "inputs": str(inputs), "data_root": data_root, "pilot": a.pilot, "n_tasks": len(tasks),
           "protocol": PROTOCOL, "tf32": False, "cudnn_deterministic": True}
    (out / "environment.json").write_text(json.dumps(env, indent=2))
    print(env, flush=True)
    timing = {"start": t_start}
    t0 = time.time()
    chk = check_inputs(inputs, tasks, out)
    blocked = set(chk["blocked_files"])
    timing["input_check_seconds"] = time.time() - t0
    t0 = time.time()
    rr = reuse_repro(inputs, data_root, samples, out) if samples else {"valid": True, "n": 0}
    timing["reuse_reproduction_seconds"] = time.time() - t0
    print("inputs ok", chk["all_ok"], "reuse valid", rr["valid"], rr.get("max_abs_diff_ce"), flush=True)
    runnable = []
    scheduled = [t["problem"] for t in tasks if t["kind"] == "hessian"]
    for t in tasks:
        if t["kind"] in ("hcut", "hplane"):
            t["available_problems"] = scheduled
        f = "%s__%s" % (C.rname(t["m"], t["s"]), t.get("ckpt", C.FINAL))
        if f in blocked:
            (out / "eval" / (t["id"] + ".meta.json")).write_text(json.dumps(
                {"task": {k: v for k, v in t.items() if k not in ("points", "reused_points")},
                 "status": "blocked", "reason": "input %s missing or digest mismatch" % f}, indent=1))
        else:
            runnable.append(t)
    t0 = time.time()
    ctxm = mp.get_context("spawn")
    q = ctxm.Queue()
    for t in runnable:
        q.put(t)
    n = max(torch.cuda.device_count(), 1)
    procs = [ctxm.Process(target=_worker, args=(q, g, str(inputs), data_root, str(out), deadline,
                                                 str(out / "eval_log.txt"), rr["valid"], a.pilot)) for g in range(n)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    timing["eval_wall_seconds"] = time.time() - t0
    gpu_sec = 0.0
    by_kind = {}
    for mp_ in (out / "eval").glob("*.meta.json"):
        mt = json.loads(mp_.read_text())
        sec = mt.get("seconds", 0.0)
        gpu_sec += sec
        kk = mt["task"]["kind"]
        by_kind[kk] = by_kind.get(kk, 0.0) + sec
    timing["gpu_seconds_sum_of_tasks"] = gpu_sec
    timing["gpu_seconds_by_kind"] = by_kind
    timing["n_gpus"] = n
    timing["total_seconds"] = time.time() - t_start
    (out / "timing.json").write_text(json.dumps(timing, indent=2))
    manifest(out, "eval_manifest.json", ["eval", "hessian", "checks"])
    print("job done", timing, flush=True)


if __name__ == "__main__":
    main()
