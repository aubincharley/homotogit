"""Local numerical and timing pilot (RTX 3050 laptop; Kaggle timing pilot separate).

    py -m landscape_v3.pilot dense|real|lanczos|evalcost [--out studies/landscape_v3/pilot]

dense    small reference network: HVP vs dense autograd Hessian, symmetry,
         Lanczos extremes vs dense eigh
real     seed-0 final checkpoints: HVP repeatability, symmetry u.Hv vs v.Hu,
         HVP vs central finite differences of gradients (float64), float32 vs
         float64 HVP, quadratic prediction d.Hd vs 2S/eps^2 (frozen BN), timing
lanczos  convergence histories for the two extreme ends (ordinary + relative)
evalcost timing of each evaluation policy and v2-point reproduction on this GPU
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from continuation_core.config import DataConfig
from continuation_core.data import load_dataset

from . import common as C
from .evaluator import Evaluator
from .hessian import FrozenBNObjective, block_scales, lanczos, make_objective

DATA = "C:/Users/mnica/Documents/Projet_filiere/data"
OUT = C.STUDY / "pilot"


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def ctx(device="cuda"):
    subsets = dict(np.load(C.V2_STUDY / "inputs" / "subsets.npz"))
    ev = Evaluator(load_dataset(DataConfig(root=DATA)), subsets, subsets["pinned_train_probe_500"], device)
    for m in C.METHODS:
        ev.model(m)
    return ev


def ckpt(m, s, name=C.FINAL):
    return torch.load(C.v2_run_dir(m, s) / "checkpoints" / name, map_location="cpu", weights_only=False)


def direction(ev, st, s, k):
    from landscape_study.directions import scale_to
    d = torch.load(C.V2_RAW / C.V2_ACCOUNT_OF_SEED[s] / "v2" / "directions" / ("seed%d_dir%02d.pt" % (s, k)),
                   map_location="cpu", weights_only=True)
    out, _ = scale_to({n: st[n] for n in ev.names}, d, ev.names)
    return out


# ----------------------------------------------------------------------------------

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 10)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.max_pool2d(h, 2)
        h = F.relu(self.bn2(self.conv2(h)))
        return self.fc(F.adaptive_avg_pool2d(h, 1).flatten(1))


def dense(out):
    torch.manual_seed(0)
    net = Tiny().double()
    with torch.no_grad():
        for bn in (net.bn1, net.bn2):
            bn.running_mean.uniform_(-0.2, 0.2)
            bn.running_var.uniform_(0.5, 1.5)
            bn.weight.uniform_(0.5, 1.5)
            bn.bias.uniform_(-0.2, 0.2)
    x = torch.randint(0, 256, (24, 3, 8, 8), dtype=torch.uint8)
    y = torch.randint(0, 10, (24,))
    obj = FrozenBNObjective(net, lambda t: t.double() / 255.0 - 0.5, x, y, batch_size=10, dtype=torch.float64)
    w0 = obj.vector()

    def L(wflat):
        ps = obj._split(wflat)
        bufs = {k: v for k, v in net.named_buffers()}
        pars = {n: p for n, p in net.named_parameters()}
        for (n, _), t in zip(obj.params, ps):
            pars[n] = t
        tot = 0.0
        for i in range(0, obj.N, obj.bs):
            xb = x[i:i + obj.bs].double() / 255.0 - 0.5
            logits = torch.func.functional_call(net, {**pars, **bufs}, (xb,))
            tot = tot + F.cross_entropy(logits, y[i:i + obj.bs]) * (xb.shape[0] / obj.N)
        return tot
    H = torch.autograd.functional.hessian(L, w0)
    g = torch.Generator().manual_seed(1)
    errs = []
    for _ in range(5):
        v = torch.randn(obj.n, generator=g, dtype=torch.float64)
        hv = obj.hvp(v)
        errs.append(float((hv - H @ v).norm() / (H @ v).norm()))
    sym = float((H - H.T).abs().max() / H.abs().max())
    ev_ = torch.linalg.eigvalsh(0.5 * (H + H.T))
    lz = lanczos(obj.hvp, obj.n, torch.randn(obj.n, generator=g, dtype=torch.float64), m_max=obj.n, tol=1e-8)
    res = {"n_params": obj.n, "hvp_vs_dense_rel_err": errs, "dense_symmetry_rel": sym,
           "dense_top2": ev_[-2:].tolist()[::-1], "dense_min": float(ev_[0]),
           "lanczos_top1": lz["values"]["top1"], "lanczos_top2": lz["values"]["top2"], "lanczos_min": lz["values"]["min"],
           "lanczos_iterations": lz["iterations"], "lanczos_rel_resid": {k: v["rel_resid"] for k, v in lz["residuals"].items()},
           "n_negative_dense": int((ev_ < 0).sum())}
    log(json.dumps(res, indent=1))
    (out / "pilot_dense.json").write_text(json.dumps(res, indent=2))


def real(out, ev):
    s = 0
    res = {}
    for m in ("plain", "gaussian_postrelu"):
        ck = ckpt(m, s)
        st, tgt = ck["model_state"], C.target(m)
        for pol in (C.SAVED, C.CFROZEN):
            r = {}
            obj = make_objective(ev, m, st, tgt, pol, "train_probe", "f32", cache_key=(m, s, C.FINAL))
            g = torch.Generator().manual_seed(2)
            u = torch.randn(obj.n, generator=g, dtype=torch.float64).to(ev.device)
            v = torch.randn(obj.n, generator=g, dtype=torch.float64).to(ev.device)
            torch.cuda.synchronize(); t0 = time.time()
            hv = obj.hvp(v)
            torch.cuda.synchronize(); r["hvp_seconds_f32"] = time.time() - t0
            hv2 = obj.hvp(v)
            r["hvp_repeat_bitwise"] = bool(torch.equal(hv, hv2))
            r["hvp_repeat_max_abs"] = float((hv - hv2).abs().max())
            hu = obj.hvp(u)
            r["symmetry_uHv_vs_vHu_rel"] = float(abs(u @ hv - v @ hu) / abs(u @ hv))
            L0, g0 = obj.loss_grad()
            r["loss_f32"], r["grad_norm_f32"] = L0, float(g0.norm())
            # float64
            obj64 = make_objective(ev, m, st, tgt, pol, "train_probe", "f64", cache_key=(m, s, C.FINAL))
            torch.cuda.synchronize(); t0 = time.time()
            hv64 = obj64.hvp(v)
            torch.cuda.synchronize(); r["hvp_seconds_f64"] = time.time() - t0
            r["hvp_f32_vs_f64_rel"] = float((hv - hv64).norm() / hv64.norm())
            w0 = obj64.vector()
            vn = v / v.norm() * w0.norm()          # unit relative to ||w||
            fd = {}
            for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
                obj64.set_vector(w0 + h * vn)
                _, gp = obj64.loss_grad()
                obj64.set_vector(w0 - h * vn)
                _, gm = obj64.loss_grad()
                est = (gp - gm) / (2 * h)
                ref = hv64 * (w0.norm() / v.norm())
                fd["%g" % h] = float((est - ref).norm() / ref.norm())
            obj64.set_vector(w0)
            r["fd_grad_vs_hvp_rel_err_by_h"] = fd
            # smooth control: perturb only fc.weight (no ReLU/max-pool after it)
            fcmask = torch.zeros(obj64.n, dtype=torch.float64, device=ev.device)
            off = sum(obj64.sizes[:-1])
            fcmask[off:] = 1.0
            vf = v * fcmask
            vf = vf / vf.norm() * w0[off:].norm()
            hvf = obj64.hvp(vf)
            fdf = {}
            for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
                obj64.set_vector(w0 + h * vf)
                _, gp = obj64.loss_grad()
                obj64.set_vector(w0 - h * vf)
                _, gm = obj64.loss_grad()
                fdf["%g" % h] = float(((gp - gm) / (2 * h) - hvf).norm() / hvf.norm())
            obj64.set_vector(w0)
            r["fd_grad_vs_hvp_rel_err_by_h_fc_only"] = fdf
            # quadratic prediction along random direction 0 vs finite perturbations
            d = direction(ev, st, s, 0)
            dv = torch.cat([d[n].reshape(-1) for n in obj.names]).to(ev.device, torch.float64)
            q32 = float(dv @ obj.hvp(dv))
            q64 = float(dv @ obj64.hvp(dv))
            c64 = {n: st[n].to(torch.float64) for n in ev.names}
            fdq = {}
            for prec in ("f32", "f64"):
                l0 = ev.evaluate3(m, st, tgt, pol, splits=("train_probe",), prec=prec, cache_key=(m, s, C.FINAL))["splits"]["train_probe"]["ce"]
                for eps in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1):
                    ls = []
                    for sg in (1, -1):
                        p = {n: (c64[n] + sg * eps * d[n]).to(torch.float32 if prec == "f32" else torch.float64) for n in ev.names}
                        ls.append(ev.evaluate3(m, st, tgt, pol, params=p, splits=("train_probe",), prec=prec,
                                               cache_key=(m, s, C.FINAL))["splits"]["train_probe"]["ce"])
                    fdq["%s_eps%g" % (prec, eps)] = 2 * (0.5 * sum(ls) - l0) / eps ** 2
            r["dHd_f32"], r["dHd_f64"], r["C_d_finite"] = q32, q64, fdq
            res["%s_%s" % (m, pol)] = r
            log("%s %s %s" % (m, pol, json.dumps(r)))
            del obj, obj64
    (out / "pilot_real.json").write_text(json.dumps(res, indent=2))


def lanczos_pilot(out, ev, m_max):
    s, res = 0, {}
    for m in ("plain", "resolution_max_b1"):
        ck = ckpt(m, s)
        st, tgt = ck["model_state"], C.target(m)
        for coords in ("ordinary", "relative"):
            pol = C.CFROZEN
            obj = make_objective(ev, m, st, tgt, pol, "train_probe", "f32", cache_key=(m, s, C.FINAL))
            a, info = block_scales(obj, st)
            op = obj.hvp if coords == "ordinary" else (lambda q, a=a, obj=obj: a * obj.hvp(a * q))
            mask = None if coords == "ordinary" else (a > 0).to(torch.float64)
            pid = C.hess_problem(m, s, "train_probe", pol, coords)
            g = torch.Generator().manual_seed(C.lanczos_seed(pid, 0))
            v0 = torch.randn(obj.n, generator=g, dtype=torch.float64)
            r = lanczos(op, obj.n, v0, m_max=m_max, device=ev.device, mask=mask, log=log, check_every=10, tol=1e-3)
            r.pop("vectors")
            r["block_info"] = {k: v for k, v in info.items() if k != "layers"}
            res[pid] = r
            log("%s: %s it, conv=%s, %.0fs" % (pid, r["iterations"], r["converged_explicit"], r["seconds"]))
            (out / "pilot_lanczos.json").write_text(json.dumps(res, indent=2))


def evalcost(out, ev):
    s, m = 0, "resolution_max_b1"
    st, tgt = ckpt(m, s)["model_state"], C.target(m)
    rows = [json.loads(l) for l in (C.V2_RAW / "maxnicaise/v2/eval/sens1d__resolution__seed0.jsonl").read_text().splitlines()]
    rows = {r["key"]: r for r in rows}
    c64 = {n: st[n].to(torch.float64) for n in ev.names}
    d = direction(ev, st, s, 3)
    p = {n: (c64[n] + 0.1 * d[n]).to(torch.float32) for n in ev.names}
    res = {}
    for pol in (C.SAVED, C.POINTWISE, C.CFROZEN):
        for prec in ("f32", "f64"):
            ev.evaluate3(m, st, tgt, pol, params=p, prec=prec, cache_key=(m, s, C.FINAL))
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(5):
                r = ev.evaluate3(m, st, tgt, pol, params=p, prec=prec, cache_key=(m, s, C.FINAL))
            torch.cuda.synchronize()
            res["%s_%s_seconds" % (pol, prec)] = (time.time() - t0) / 5
            res["%s_%s_ce" % (pol, prec)] = r["splits"]["test_probe"]["ce"]
    for pol in (C.SAVED, C.POINTWISE):
        v2 = rows["%s|3|0.1|1" % pol]["splits"]
        mine = ev.evaluate3(m, st, tgt, pol, params=p)["splits"]
        res["v2_repro_%s_abs_diff" % pol] = max(abs(v2[sp]["ce"] - mine[sp]["ce"]) for sp in C.PROBES)
    c = ev.evaluate3(m, st, tgt, C.CFROZEN, cache_key=(m, s, C.FINAL))["splits"]
    cp = ev.evaluate3(m, st, tgt, C.POINTWISE)["splits"]
    res["centre_frozen_centre_equals_pointwise_centre"] = c == cp
    log(json.dumps(res, indent=1))
    (out / "pilot_evalcost.json").write_text(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("dense", "real", "lanczos", "evalcost"))
    ap.add_argument("--m-max", type=int, default=300)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False          # Ampere default is TF32 convolutions
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.what == "dense":
        dense(OUT)
        return
    ev = ctx()
    {"real": lambda: real(OUT, ev), "lanczos": lambda: lanczos_pilot(OUT, ev, a.m_max),
     "evalcost": lambda: evalcost(OUT, ev)}[a.what]()


if __name__ == "__main__":
    main()
