"""Resumable evaluation jobs.

    py -m landscape_study.run --job pilot --inputs studies/landscape_v1/inputs --data-root DATA --out OUT
    py -m landscape_study.run --job 0|1|2 ...      (the Kaggle split)
    py -m landscape_study.run --job offsets ...    (range check before the grids)

Each task writes ``OUT/<task>.jsonl`` one point per line, flushed immediately,
and skips points whose key is already present, so an interrupted job resumes.
``OUT/<task>.meta.json`` holds the task spec, a configuration hash and timings.
Non-finite values are written as they are.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import glob
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from continuation_core.data import load_dataset
from continuation_core.config import DataConfig

from .directions import scale_to
from .evaluator import StudyEvaluator

R = {"r16": {"resolution": 16, "sigma": None, "label": "r=16"},
     "r24": {"resolution": 24, "sigma": None, "label": "r=24"},
     "r32": {"resolution": 32, "sigma": None, "label": "r=32 (target)"}}
TARGET = {"plain": {"resolution": None, "sigma": None, "label": "target"},
          "resolution_max_b1": {"resolution": 32, "sigma": None, "label": "target"}}
METHODS = ("plain", "resolution_max_b1")
GRID21 = np.linspace(-0.25, 0.25, 21).tolist()


def w(method, seed, ep):
    return "%s__seed%d__ep%02d.pt" % (method, seed, ep)


def build_tasks(job: str, plane=None) -> list:
    tasks = []

    def grid(tid, model, center, pair, state, a, b):
        tasks.append({"id": tid, "kind": "grid", "model": model, "center": center, "pair": pair,
                      "state": state, "a": a, "b": b})

    def line(tid, model, center, pair, axis, state, a):
        tasks.append({"id": tid, "kind": "line", "model": model, "center": center, "pair": pair,
                      "axis": axis, "state": state, "a": a})

    if job == "offsets":
        for m in METHODS:
            line("offsets__%s" % m, m, w(m, 0, 30), 0, "D", TARGET[m],
                 [-0.5, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 0.5])
    if job in ("0", "all"):
        for m in METHODS:
            grid("A2d__%s__seed0" % m, m, w(m, 0, 30), 0, TARGET[m], GRID21, GRID21)
    if job in ("1", "all"):
        for s in ("r16", "r24", "r32"):
            grid("B2d__resolution_max_b1__seed0__ep06__%s" % s, "resolution_max_b1",
                 w("resolution_max_b1", 0, 6), 0, R[s], GRID21, GRID21)
        for s in ("r16", "r24", "r32"):
            line("B1d__resolution_max_b1__seed0__ep12__%s" % s, "resolution_max_b1",
                 w("resolution_max_b1", 0, 12), 0, "D", R[s], GRID21)
            line("B1d__plainweights_with_hook__seed0__ep06__%s" % s, "resolution_max_b1",
                 w("plain", 0, 6), 0, "D", R[s], GRID21)
    if job in ("2", "all"):
        for seed in (0, 1, 2):
            for m in METHODS:
                for pair in range(5):
                    line("A1d__%s__seed%d__pair%d__D" % (m, seed, pair), m, w(m, seed, 30),
                         pair, "D", TARGET[m], GRID21)
                line("A1d__%s__seed%d__pair0__E" % (m, seed), m, w(m, seed, 30), 0, "E",
                     TARGET[m], GRID21)
        tasks.append({"id": "C_checkpoints", "kind": "checkpoints",
                      "files": [p["file"] for p in plane["points"]] if plane else []})
        tasks.append({"id": "C_plane", "kind": "plane", "model": "plain",
                      "state": TARGET["plain"]})
        for seed in (0, 1, 2):
            tasks.append({"id": "D_interp__plain_vs_resolution_max_b1__seed%d" % seed,
                          "kind": "interp", "A": w("plain", seed, 30),
                          "B": w("resolution_max_b1", seed, 30),
                          "alphas": np.linspace(0.0, 1.0, 51).tolist(),
                          "model": "plain", "state": TARGET["plain"]})
        tasks.append({"id": "S_saved_vs_recalibrated", "kind": "checkpoints",
                      "files": [w(m, s, e) for m in METHODS for s in (0, 1, 2)
                                for e in (6, 12, 18, 30)]})
    if job == "pilot":
        small = [-0.25, 0.0, 0.25]
        grid("pilot_grid", "resolution_max_b1", w("resolution_max_b1", 0, 30), 0,
             TARGET["resolution_max_b1"], small, small)
        line("pilot_line_r16", "resolution_max_b1", w("resolution_max_b1", 0, 6), 0, "D",
             R["r16"], small)
        tasks.append({"id": "pilot_interp", "kind": "interp", "A": w("plain", 0, 30),
                      "B": w("resolution_max_b1", 0, 30), "alphas": [0.0, 0.5, 1.0],
                      "model": "plain", "state": TARGET["plain"]})
        tasks.append({"id": "pilot_checkpoints", "kind": "checkpoints",
                      "files": [w("resolution_max_b1", 0, 6), w("resolution_max_b1", 0, 30)]})
        tasks.append({"id": "pilot_plane", "kind": "plane", "model": "plain",
                      "state": TARGET["plain"], "subsample": 12})
    return tasks


class Inputs:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.subsets = dict(np.load(self.root / "subsets.npz"))
        self.draws = torch.load(self.root / "draws.pt", map_location="cpu", weights_only=False)
        self.plane = torch.load(self.root / "pca_plane.pt", map_location="cpu", weights_only=False)
        self.weights_dir = self.root / "weights"
        if not self.weights_dir.is_dir() and (self.root / "weights.zip").is_file():
            # the Kaggle CLI uploads sub-folders as zip archives; extract a private copy
            import zipfile
            dest = Path(os.environ.get("LANDSCAPE_WEIGHTS_CACHE", "/kaggle/working/_weights"))
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(self.root / "weights.zip") as zf:
                zf.extractall(dest)
            hits = list(dest.rglob("plain__seed0__ep30.pt"))
            self.weights_dir = hits[0].parent
        self._w = {}

    def _file(self, rel):
        return self.weights_dir / rel.split("/", 1)[1] if rel.startswith("weights/") else self.root / rel

    def verify(self) -> dict:
        from .sources import sha_file
        bad = [k for k, v in self.manifest.items() if sha_file(self._file(k)) != v]
        if bad:
            raise RuntimeError("inputs differ from manifest: %s" % bad)
        return {"files_verified": len(self.manifest), "weights_dir": str(self.weights_dir)}

    def weights(self, name) -> dict:
        if name not in self._w:
            self._w[name] = torch.load(self.weights_dir / name, map_location="cpu",
                                       weights_only=False)
        return self._w[name]


def params64(model_state, names, device):
    return {n: model_state[n].to(device, torch.float64) for n in names}


def cast(p64):
    return {n: t.to(torch.float32) for n, t in p64.items()}


def run_task(task, ev: StudyEvaluator, inp: Inputs, out: Path, device):
    path = out / (task["id"] + ".jsonl")
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["key"])
    spec = json.dumps(task, sort_keys=True)
    meta = {"task": task, "config_sha256": hashlib.sha256(
        (spec + json.dumps(inp.manifest, sort_keys=True)).encode()).hexdigest(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "resumed_points": len(done)}
    t0 = time.perf_counter()
    ev.model(task.get("model", "plain"))
    names = ev.names
    fh = path.open("a")

    def emit(key, rec):
        rec["key"] = key
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

    kind = task["kind"]
    if kind in ("grid", "line"):
        c = inp.weights(task["center"])
        c64 = params64(c["model_state"], names, device)
        center_cpu = {n: c["model_state"][n] for n in names}
        draw = inp.draws[task["pair"]]
        axes = ("D", "E") if kind == "grid" else (task["axis"],)
        dirs = {ax: {n: t.to(device) for n, t in scale_to(center_cpu, draw[ax], names)[0].items()}
                for ax in axes}
        pts = ([(i, j, a, b) for i, a in enumerate(task["a"]) for j, b in enumerate(task["b"])]
               if kind == "grid" else [(i, None, a, None) for i, a in enumerate(task["a"])])
        for i, j, a, b in pts:
            key = "%d|%s" % (i, j)
            if key in done:
                continue
            p = {n: c64[n] + a * dirs[axes[0]][n] + (0.0 if b is None else b * dirs["E"][n])
                 for n in names}
            r = ev.recalibrated(task["model"], cast(p), task["state"])
            r.update(i=i, j=j, a=a, b=b, center=task["center"], pair=task["pair"])
            emit(key, r)
    elif kind == "interp":
        A = params64(inp.weights(task["A"])["model_state"], names, device)
        B = params64(inp.weights(task["B"])["model_state"], names, device)
        for i, al in enumerate(task["alphas"]):
            key = str(i)
            if key in done:
                continue
            p = {n: (1.0 - al) * A[n] + al * B[n] for n in names}
            r = ev.recalibrated(task["model"], cast(p), task["state"])
            r.update(i=i, alpha=al, A=task["A"], B=task["B"])
            emit(key, r)
    elif kind == "plane":
        pl = inp.plane
        origin = pl["origin"].to(device)
        d1, d2 = pl["d1"].to(device), pl["d2"].to(device)
        xs, ys = pl["grid"]["x"], pl["grid"]["y"]
        step = int(task.get("subsample", 1))
        from continuation_core.analysis.params import from_vector
        tmpl = dict(ev.model(task["model"])[0].named_parameters())
        for i in range(0, len(xs), step):
            for j in range(0, len(ys), step):
                key = "%d|%d" % (i, j)
                if key in done:
                    continue
                vec = origin + xs[i] * d1 + ys[j] * d2
                p = from_vector(vec, tmpl, names, device=device)
                r = ev.recalibrated(task["model"], p, task["state"])
                r.update(i=i, j=j, x=xs[i], y=ys[j])
                emit(key, r)
    elif kind == "checkpoints":
        for f in task["files"]:
            wt = inp.weights(f)
            st = wt["states"]
            model = wt["method"]
            params = {n: wt["model_state"][n] for n in names}
            used = st["used_for_last_update"]
            differs = used is not None and ((used["resolution"], used["sigma"])
                                            != (st["target"]["resolution"], st["target"]["sigma"]))
            evals = [("saved_stats", "target", st["target"])]
            if differs:
                evals.append(("saved_stats", "used", used))
            evals += [("recalibrated", "target", st["target"])]
            if differs:
                evals.append(("recalibrated", "used", used))
            for pol, lab, s in evals:
                key = "%s|%s|%s" % (f, pol, lab)
                if key in done:
                    continue
                s = dict(s, label=lab)
                if pol == "saved_stats":
                    r = ev.saved_stats(model, wt["model_state"], s)
                else:
                    r = ev.recalibrated(model, params, s,
                                        splits=("train_probe", "test_probe",
                                                "pinned_train_probe_500", "test_full"))
                r.update(file=f, method=model, seed=wt["seed"],
                         epochs_completed=wt["epochs_completed"],
                         global_update=wt["global_update"], state_label=lab,
                         recorded_metrics=wt["recorded_metrics"])
                emit(key, r)
    fh.close()
    meta["seconds"] = time.perf_counter() - t0
    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / (task["id"] + ".meta.json")).write_text(json.dumps(meta, indent=2))
    print("%-60s %6.1fs" % (task["id"], meta["seconds"]), flush=True)


def find_first(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return Path(hits[0]).parent if hits else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", required=True)
    ap.add_argument("--inputs", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-checks", action="store_true")
    args = ap.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    inputs = Path(args.inputs) if args.inputs else find_first("/kaggle/input/**/preregistration.json")
    data_root = (args.data_root or str(find_first("/kaggle/input/**/cifar-10-batches-py/batches.meta").parent))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": args.device,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "cudnn_deterministic": True, "inputs": str(inputs), "data_root": data_root}
    (out / ("environment_job%s.json" % args.job)).write_text(json.dumps(env, indent=2))
    print(env, flush=True)

    inp = Inputs(inputs)
    inp.verify()
    ds = load_dataset(DataConfig(root=data_root))
    ev = StudyEvaluator(ds, inp.subsets, inp.subsets["pinned_train_probe_500"], args.device)
    if not args.skip_checks:
        from .checks import run_checks
        run_checks(ev, inp, out / ("checks_job%s.json" % args.job), args.device)
    for task in build_tasks(args.job, inp.plane):
        run_task(task, ev, inp, out, args.device)
    print("job %s done" % args.job, flush=True)


if __name__ == "__main__":
    main()
