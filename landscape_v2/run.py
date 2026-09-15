"""One Kaggle job: train the seeds assigned to an account, then evaluate them.

    python -m landscape_v2.run --stage all --seeds 1 4 --out /kaggle/working/v2 --deadline-hours 11.2
    py -m landscape_v2.run --stage all --seeds 3 --pilot --inputs studies/landscape_v2/inputs --data-root DATA --out OUT

Evaluation tasks run in priority order on every visible GPU.  Each task writes
``eval/<task>.jsonl`` (one flushed line per point, resumable by key) and
``eval/<task>.meta.json`` with its status: ``complete``, ``deadline`` (stopped
by the time guard), or ``failed`` (traceback).  Non-finite values are written
as they are.  ``runs_manifest.json`` and ``eval_manifest.json`` hold the sha256
of every produced file.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from continuation_core.analysis.params import from_vector
from continuation_core.config import DataConfig
from continuation_core.data import load_dataset
from landscape_study.directions import in_mask, normalization_report, scale_to

from . import common as C
from .evaluator import Evaluator

PLAIN_TARGET = {"resolution": None, "sigma": None}


def target_of(method):
    return C.state_dict_of(C.controller(method).target_state())


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rname(m, s):
    return "%s__seed%d" % (m, s)


# --------------------------------------------------------------------------
# tasks

def build_tasks(seeds, primary, pilot=False):
    final = "epoch_003.pt" if pilot else "epoch_030.pt"
    dirs = list(range(2 if pilot else C.N_DIRECTIONS))
    amps = [0.10] if pilot else list(C.AMPLITUDES)
    g21 = np.linspace(-0.25, 0.25, 3 if pilot else C.SURFACE_GRID_ALL).tolist()
    g41 = np.linspace(-0.25, 0.25, C.SURFACE_GRID_PRIMARY).tolist()
    T = []
    for s in seeds:
        for m in C.METHODS:
            T.append(dict(prio=1, id="sens1d__%s__seed%d" % (C.SHORT[m], s), kind="sens1d", m=m, s=s,
                          center=final, dirs=dirs, amps=amps))
    for s in seeds:
        for m in C.METHODS:
            T.append(dict(prio=2, id="validation__%s__seed%d" % (C.SHORT[m], s), kind="validation",
                          m=m, s=s, center=final))
            if s in C.CALIB_SENSITIVITY["seeds"] or pilot:
                T.append(dict(prio=2, id="calibsens__%s__seed%d" % (C.SHORT[m], s), kind="calibsens",
                              m=m, s=s, center=final))
        for a, b in (("plain", "resolution_max_b1"), ("plain", "gaussian_postrelu"),
                     ("plain", "resolution_max_b1_gaussian_conv"),
                     ("resolution_max_b1", "resolution_max_b1_gaussian_conv")):
            T.append(dict(prio=2, id="interp__%s_%s__seed%d" % (C.SHORT[a], C.SHORT[b], s), kind="interp",
                          A=a, B=b, s=s, center=final,
                          alphas=np.linspace(0, 1, 3 if pilot else 51).tolist()))
    for s in seeds:
        for m in C.METHODS[1:]:
            for tr in C.fixed_weight_transitions(m):
                T.append(dict(prio=3, id="fixed1d__%s__seed%d__u%06d" % (C.SHORT[m], s, tr["update"]),
                              kind="fixed1d", m=m, s=s, tr=tr, ckpt="epoch_%03d.pt" % (tr["update"] // C.UPDATES_PER_EPOCH),
                              dirs=list(range(2 if pilot else C.FIXED_WEIGHT_DIRECTIONS)), amps=amps))
        for m in C.METHODS:
            T.append(dict(prio=3, id="traj__%s__seed%d" % (C.SHORT[m], s), kind="traj", m=m, s=s,
                          limit=4 if pilot else None))
    for s in seeds:
        if s == primary:
            T.append(dict(prio=4, id="pcaplane__seed%d" % s, kind="pcaplane", s=s, n=3 if pilot else C.PCA_GRID))
        for m in C.METHODS:
            T.append(dict(prio=4, id="surface__%s__seed%d__g%d" % (C.SHORT[m], s, len(g21)), kind="surface",
                          m=m, s=s, center=final, grid=g21))
    if not pilot:
        for m in C.METHODS:
            T.append(dict(prio=5, id="surface__%s__seed%d__g41" % (C.SHORT[m], primary), kind="surface",
                          m=m, s=primary, center=final, grid=g41))
    for m in C.METHODS[1:]:
        for tr in C.fixed_weight_transitions(m):
            for lab, st in C.fixed_weight_states(m, tr):
                T.append(dict(prio=5, id="fixedsurf__%s__seed%d__u%06d__%s" % (C.SHORT[m], primary, tr["update"],
                                                                                lab.replace("=", "")),
                              kind="fixedsurf", m=m, s=primary, ckpt="epoch_%03d.pt" % (tr["update"] // C.UPDATES_PER_EPOCH),
                              label=lab, state=st, grid=g21))
    T = [t for t in T if t["s"] in seeds]
    return sorted(T, key=lambda t: t["prio"])


class Ctx:
    def __init__(self, inputs, runs, dirs_dir, data_root, device):
        self.inputs, self.runs, self.dirs_dir = Path(inputs), Path(runs), Path(dirs_dir)
        subsets = dict(np.load(self.inputs / "subsets.npz"))
        self.ev = Evaluator(load_dataset(DataConfig(root=data_root)), subsets,
                            subsets["pinned_train_probe_500"], device)
        for m in C.METHODS:
            self.ev.model(m)
        self.names = self.ev.names
        self.device = torch.device(device)
        self._ck, self._draws = {}, {}

    def ckpt(self, m, s, name):
        key = (m, s, name)
        if key not in self._ck:
            if len(self._ck) > 64:
                self._ck.clear()
            self._ck[key] = torch.load(self.runs / rname(m, s) / "checkpoints" / name,
                                       map_location="cpu", weights_only=False)
        return self._ck[key]

    def draw(self, s, k):
        if (s, k) not in self._draws:
            self._draws[(s, k)] = torch.load(self.dirs_dir / ("seed%d_dir%02d.pt" % (s, k)),
                                             map_location="cpu", weights_only=True)
        return self._draws[(s, k)]

    def center64(self, st):
        return {n: st[n].to(self.device, torch.float64) for n in self.names}

    def direction(self, st, s, k):
        cpu = {n: st[n] for n in self.names}
        d, _ = scale_to(cpu, self.draw(s, k), self.names)
        rep = normalization_report(cpu, d, self.names)
        rep.pop("masked_tensors")
        return {n: t.to(self.device) for n, t in d.items()}, rep


def perturb(c64, dirs, coefs, names):
    return {n: (c64[n] + sum(a * d[n] for a, d in zip(coefs, dirs))).to(torch.float32) for n in names}


def run_task(task, ctx, out, deadline):
    path = out / (task["id"] + ".jsonl")
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                done.add(json.loads(line)["key"])
            except Exception:
                pass
    meta = {"task": task, "config_sha256": C.sha_json(task), "started": time.time(),
            "resumed_points": len(done), "status": "running"}
    fh = path.open("a")
    ev, names = ctx.ev, ctx.names

    def emit(key, rec):
        if time.time() > deadline:
            raise TimeoutError
        rec["key"] = key
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

    def todo(key):
        if time.time() > deadline:
            raise TimeoutError
        return key not in done

    kind = task["kind"]
    try:
        if kind in ("sens1d", "validation", "calibsens", "surface"):
            m, s = task["m"], task["s"]
            ck = ctx.ckpt(m, s, task["center"])
            st, c64, tgt = ck["model_state"], ctx.center64(ck["model_state"]), target_of(m)
            if kind == "sens1d":
                for pol in ("saved", "recalibrated"):
                    if todo("%s|center" % pol):
                        emit("%s|center" % pol, dict(ev.evaluate(m, st, tgt, pol), k=None, amp=0.0, sign=0))
                for k in task["dirs"]:
                    d, rep = ctx.direction(st, s, k)
                    if todo("norm|%d" % k):
                        emit("norm|%d" % k, {"k": k, "direction_seed": C.direction_seed(s, k), "report": rep})
                    for amp in task["amps"]:
                        for sign in (1, -1):
                            p = None
                            for pol in ("saved", "recalibrated"):
                                key = "%s|%d|%g|%d" % (pol, k, amp, sign)
                                if not todo(key):
                                    continue
                                p = p or perturb(c64, [d], [sign * amp], names)
                                emit(key, dict(ev.evaluate(m, st, tgt, pol, params=p), k=k, amp=amp, sign=sign))
            elif kind == "validation":
                for pol in ("saved", "recalibrated"):
                    if todo("%s|center" % pol):
                        emit("%s|center" % pol, dict(ev.evaluate(m, st, tgt, pol, splits=(
                            "train_full", "test_full", "train_large", "pinned_train_probe_500")), k=None, amp=0.0, sign=0))
                for k in C.VALIDATION["directions"]:
                    d, _ = ctx.direction(st, s, k)
                    for amp in C.VALIDATION["amplitudes"]:
                        for sign in C.VALIDATION["signs"]:
                            p = perturb(c64, [d], [sign * amp], names)
                            for pol in ("saved", "recalibrated"):
                                key = "%s|%d|%g|%d" % (pol, k, amp, sign)
                                if todo(key):
                                    emit(key, dict(ev.evaluate(m, st, tgt, pol, params=p,
                                                               splits=("train_large", "test_full")), k=k, amp=amp, sign=sign))
            elif kind == "calibsens":
                if todo("center"):
                    emit("center", dict(ev.evaluate(m, st, tgt, "recalibrated", calib="calib10k"), k=None, amp=0.0, sign=0))
                for k in C.CALIB_SENSITIVITY["directions"]:
                    d, _ = ctx.direction(st, s, k)
                    for amp in C.CALIB_SENSITIVITY["amplitudes"]:
                        for sign in C.CALIB_SENSITIVITY["signs"]:
                            key = "%d|%g|%d" % (k, amp, sign)
                            if todo(key):
                                emit(key, dict(ev.evaluate(m, st, tgt, "recalibrated", calib="calib10k",
                                                           params=perturb(c64, [d], [sign * amp], names)), k=k, amp=amp, sign=sign))
            else:  # surface
                da, _ = ctx.direction(st, s, C.PRIMARY_PAIR[0])
                db, _ = ctx.direction(st, s, C.PRIMARY_PAIR[1])
                g = task["grid"]
                for i, a in enumerate(g):
                    for j, b in enumerate(g):
                        key = "%d|%d" % (i, j)
                        if todo(key):
                            emit(key, dict(ev.evaluate(m, st, tgt, "recalibrated",
                                                       params=perturb(c64, [da, db], [a, b], names)), i=i, j=j, a=a, b=b))
        elif kind == "interp":
            s = task["s"]
            A = ctx.center64(ctx.ckpt(task["A"], s, task["center"])["model_state"])
            B = ctx.center64(ctx.ckpt(task["B"], s, task["center"])["model_state"])
            for i, al in enumerate(task["alphas"]):
                if todo(str(i)):
                    p = {n: ((1.0 - al) * A[n] + al * B[n]).to(torch.float32) for n in names}
                    emit(str(i), dict(ev.evaluate("plain", None, PLAIN_TARGET, "recalibrated", params=p), i=i, alpha=al))
        elif kind in ("fixed1d", "fixedsurf"):
            m, s = task["m"], task["s"]
            ck = ctx.ckpt(m, s, task["ckpt"])
            st, c64 = ck["model_state"], ctx.center64(ck["model_state"])
            if kind == "fixed1d":
                states = C.fixed_weight_states(m, task["tr"])
                for k in task["dirs"]:
                    d, rep = ctx.direction(st, s, k)
                    for lab, sd in states:
                        for pol in ("saved", "recalibrated"):
                            key = "%s|%s|center" % (lab, pol)
                            if todo(key):
                                emit(key, dict(ev.evaluate(m, st, sd, pol), label=lab, k=None, amp=0.0, sign=0))
                        for amp in task["amps"]:
                            for sign in (1, -1):
                                key = "%s|recalibrated|%d|%g|%d" % (lab, k, amp, sign)
                                if todo(key):
                                    emit(key, dict(ev.evaluate(m, st, sd, "recalibrated",
                                                               params=perturb(c64, [d], [sign * amp], names)),
                                                   label=lab, k=k, amp=amp, sign=sign))
            else:
                da, _ = ctx.direction(st, s, C.PRIMARY_PAIR[0])
                db, _ = ctx.direction(st, s, C.PRIMARY_PAIR[1])
                for i, a in enumerate(task["grid"]):
                    for j, b in enumerate(task["grid"]):
                        key = "%d|%d" % (i, j)
                        if todo(key):
                            emit(key, dict(ev.evaluate(m, st, task["state"], "recalibrated",
                                                       params=perturb(c64, [da, db], [a, b], names)),
                                           i=i, j=j, a=a, b=b, label=task["label"]))
        elif kind == "traj":
            m, s = task["m"], task["s"]
            files = sorted(p.name for p in (ctx.runs / rname(m, s) / "checkpoints").glob("*.pt"))
            files = sorted(files, key=lambda f: torch.load(ctx.runs / rname(m, s) / "checkpoints" / f,
                                                            map_location="cpu", weights_only=False)["global_update"])
            if task.get("limit"):
                files = files[:task["limit"]]
            tgt = target_of(m)
            for f in files:
                ck = ctx.ckpt(m, s, f)
                iv = ck["intervention"]
                cur = (iv["used_for_last_update"] or iv["next_update"])["state"]
                cur = {"resolution": cur["resolution"], "sigma": cur["sigma"]}
                states = [("final", tgt)] + ([("current", cur)] if cur != tgt else [])
                for lab, sd in states:
                    for pol in ("saved", "recalibrated"):
                        key = "%s|%s|%s" % (f, lab, pol)
                        if todo(key):
                            emit(key, dict(ev.evaluate(m, ck["model_state"], sd, pol), file=f, label=lab,
                                           global_update=ck["global_update"], reason=ck["reason"],
                                           current_state=cur, current_equals_final=cur == tgt))
        elif kind == "pcaplane":
            s = task["s"]
            z = np.load(out.parent / "pca" / ("pca_seed%d.npz" % s), allow_pickle=True)
            mean = torch.as_tensor(z["mean"], dtype=torch.float64, device=ctx.device)
            v1 = torch.as_tensor(z["components"][0], dtype=torch.float64, device=ctx.device)
            v2 = torch.as_tensor(z["components"][1], dtype=torch.float64, device=ctx.device)
            co = z["coords"][:, :2]
            lo, hi = co.min(0), co.max(0)
            span = hi - lo
            xs = np.linspace(lo[0] - 0.2 * span[0], hi[0] + 0.2 * span[0], task["n"])
            ys = np.linspace(lo[1] - 0.2 * span[1], hi[1] + 0.2 * span[1], task["n"])
            tmpl = dict(ctx.ev.model("plain")[0].named_parameters())
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    key = "%d|%d" % (i, j)
                    if todo(key):
                        p = from_vector(mean + float(x) * v1 + float(y) * v2, tmpl, names, device=ctx.device)
                        emit(key, dict(ev.evaluate("plain", None, PLAIN_TARGET, "recalibrated", params=p),
                                       i=i, j=j, x=float(x), y=float(y)))
        meta["status"] = "complete"
    except TimeoutError:
        meta["status"] = "deadline"
    except Exception:
        meta["status"] = "failed"
        meta["traceback"] = traceback.format_exc()
    finally:
        fh.close()
    meta["seconds"] = time.time() - meta["started"]
    (out / (task["id"] + ".meta.json")).write_text(json.dumps(meta, indent=2, default=str))
    return meta["status"]


# --------------------------------------------------------------------------
# per-seed preparation (main process)

def write_directions(runs, dirs_dir, seeds, n):
    dirs_dir.mkdir(parents=True, exist_ok=True)
    ck = torch.load(Path(runs) / rname("plain", seeds[0]) / "checkpoints" / "epoch_000.pt",
                    map_location="cpu", weights_only=False)["model_state"]
    from continuation_core.models import build_model
    names = [k for k, _ in build_model("resnet20_bn_cifar", 10).named_parameters()]
    for s in seeds:
        for k in range(n):
            p = dirs_dir / ("seed%d_dir%02d.pt" % (s, k))
            if p.exists():
                continue
            g = torch.Generator().manual_seed(C.direction_seed(s, k))
            torch.save({nm: torch.randn(ck[nm].shape, generator=g, dtype=torch.float32)
                        for nm in names if in_mask(nm, ck[nm])}, p)


def pca_seed(runs, s, out_dir):
    from continuation_core.models import build_model
    names = [k for k, _ in build_model("resnet20_bn_cifar", 10).named_parameters()]
    recs = []
    for m in C.METHODS:
        for p in sorted((Path(runs) / rname(m, s) / "checkpoints").glob("*.pt")):
            ck = torch.load(p, map_location="cpu", weights_only=False)
            vec = torch.cat([ck["model_state"][n].reshape(-1).to(torch.float64) for n in names])
            fit = ck["reason"] in ("epoch_end", "initialization") and p.name.startswith("epoch_")
            recs.append((m, p.name, int(ck["global_update"]), ck["reason"], fit, vec, ck["intervention"]))
    X = torch.stack([r[5] for r in recs])
    fit = torch.tensor([r[4] for r in recs])
    mean = X[fit].mean(0)
    Cm = X[fit] - mean
    G = Cm @ Cm.T
    lam, U = torch.linalg.eigh(G)
    order = torch.argsort(lam, descending=True)
    lam, U = lam[order].clamp_min(0), U[:, order]
    k = min(10, int(fit.sum()) - 1)
    comps = (Cm.T @ U[:, :k]) / lam[:k].sqrt()
    comps = comps.T                                            # (k, P), orthonormal rows
    D = X - mean
    coords = D @ comps.T
    dist = D.norm(dim=1)
    resid = {}
    for kk in (2, 5, 10):
        kk = min(kk, k)
        resid[kk] = (D - coords[:, :kk] @ comps[:kk]).norm(dim=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / ("pca_seed%d.npz" % s), mean=mean.numpy(), components=comps.numpy(),
             explained_variance_ratio=(lam / lam.sum()).numpy(), coords=coords.numpy(),
             dist_to_mean=dist.numpy(), resid2=resid[2].numpy(), resid5=resid[min(5, k)].numpy(),
             resid10=resid[min(10, k)].numpy(), fit=fit.numpy(),
             method=np.array([r[0] for r in recs]), files=np.array([r[1] for r in recs]),
             update=np.array([r[2] for r in recs]), reason=np.array([r[3] for r in recs]),
             intervention=np.array([json.dumps(r[6], default=str) for r in recs]))
    ortho = float((comps @ comps.T - torch.eye(k, dtype=torch.float64)).abs().max())
    return {"seed": s, "n_fit": int(fit.sum()), "n_total": len(recs), "orthonormality_error": ortho,
            "evr_top5": (lam / lam.sum())[:5].tolist()}


# --------------------------------------------------------------------------

def _worker(q, gpu, inputs, runs, dirs_dir, data_root, out, deadline, log_path):
    def log(msg):
        line = "[%s gpu%d] %s" % (time.strftime("%H:%M:%S"), gpu, msg)
        print(line, flush=True)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")
    dev = "cuda:%d" % gpu if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ctx = Ctx(inputs, runs, dirs_dir, data_root, dev)
    while True:
        try:
            task = q.get_nowait()
        except Exception:
            return
        t0 = time.time()
        status = run_task(task, ctx, Path(out), deadline)
        log("%-55s %-9s %6.0fs" % (task["id"], status, time.time() - t0))


def find(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return Path(hits[0]).parent if hits else None


def manifest(root: Path, name: str, subdirs):
    files = {}
    for sd in subdirs:
        for p in sorted((root / sd).rglob("*")):
            if p.is_file():
                files[p.relative_to(root).as_posix()] = {"sha256": sha_file(p), "bytes": p.stat().st_size}
    (root / name).write_text(json.dumps(files, indent=1))
    return len(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("train", "eval", "all"), default="all")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--inputs")
    ap.add_argument("--data-root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline-hours", type=float, default=11.2)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--primary", type=int, default=C.PRIMARY_SEED)
    a = ap.parse_args()
    t_start = time.time()
    deadline = t_start + a.deadline_hours * 3600
    inputs = Path(a.inputs) if a.inputs else find("/kaggle/input/**/preregistration.json")
    data_root = a.data_root or str(find("/kaggle/input/**/cifar-10-batches-py/batches.meta").parent)
    assets_dir = inputs / "assets" if (inputs / "assets").is_dir() else inputs
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    timing = {"start": t_start}
    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "n_gpu": torch.cuda.device_count(),
           "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
           "inputs": str(inputs), "data_root": data_root, "seeds": a.seeds, "pilot": a.pilot}
    (out / "environment.json").write_text(json.dumps(env, indent=2))
    print(env, flush=True)
    from continuation_core import assets as A
    A.verify(assets_dir)
    runs = out / "runs"
    if a.stage in ("train", "all"):
        from .train import train_all
        if a.pilot:
            raise SystemExit("pilot training is run separately (short budget)")
        t0 = time.time()
        train_all(a.seeds, data_root, str(assets_dir), str(runs))
        timing["train_seconds"] = time.time() - t0
        manifest(out, "runs_manifest.json", ["runs"])
    if a.stage in ("eval", "all"):
        t0 = time.time()
        dirs_dir = out / "directions"
        write_directions(runs, dirs_dir, a.seeds, 2 if a.pilot else C.N_DIRECTIONS)
        pca = [pca_seed(runs, s, out / "pca") for s in a.seeds]
        (out / "pca" / "pca_summary.json").write_text(json.dumps(pca, indent=2))
        timing["directions_pca_seconds"] = time.time() - t0
        from .checks import run_checks
        t0 = time.time()
        ctx = Ctx(inputs, runs, dirs_dir, data_root, "cuda:0" if torch.cuda.is_available() else "cpu")
        for s in a.seeds:
            run_checks(ctx, s, out / "checks", pilot=a.pilot)
        del ctx
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        timing["checks_seconds"] = time.time() - t0
        t0 = time.time()
        ev_out = out / "eval"
        ev_out.mkdir(parents=True, exist_ok=True)
        tasks = build_tasks(a.seeds, a.primary if a.primary in a.seeds or not a.pilot else a.seeds[0], a.pilot)
        (out / "tasks.json").write_text(json.dumps(tasks, indent=1, default=str))
        ctxm = mp.get_context("spawn")
        q = ctxm.Queue()
        for t in tasks:
            q.put(t)
        n = max(torch.cuda.device_count(), 1)
        procs = [ctxm.Process(target=_worker, args=(q, g, str(inputs), str(runs), str(dirs_dir), data_root,
                                                     str(ev_out), deadline, str(out / "eval_log.txt")))
                 for g in range(n)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        timing["eval_seconds"] = time.time() - t0
        manifest(out, "eval_manifest.json", ["eval", "pca", "checks", "directions"])
    timing["total_seconds"] = time.time() - t_start
    (out / "timing.json").write_text(json.dumps(timing, indent=2))
    print("job done", timing, flush=True)


if __name__ == "__main__":
    main()
