"""Verify the v2 inputs, resolve the v3 evaluation matrix, subtract reusable v2
points, and write per-account task files (local, CPU).

    py -m landscape_v3.plan verify        # checkpoints, manifests, states, pairing, digests
    py -m landscape_v3.plan matrix        # evaluation matrix + tasks + estimates

Outputs in ``studies/landscape_v3/protocol/``:
``input_verification.json``, ``evaluation_matrix.json`` (counts per block, new vs
reused, per-account estimates), ``tasks_<account>.json``, ``expected_digests.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict

import numpy as np
import torch

from continuation_core.models import build_model

from . import common as C
from . import v2points

PROTO = C.STUDY / "protocol"
V2 = C.V2

# account -> what it runs
SEED_ACCOUNT = {0: "maxnicaise", 1: "maxlefrr", 2: "maxnikezz", 3: "maximemonstrenikez", 4: "maxlebossdu91"}
EXTRA_ACCOUNT = "maxfrrsava"          # wide random planes 21x21 (all seeds)
EXTRA2_ACCOUNT = "maxmonstre"         # seed-0 41x41 wide plane + float64 block

# T4 seconds per evaluation (v2 timings; HVP from the Kaggle timing pilot when present)
COST = {"probe_frozen": 0.10, "probe_pointwise": 0.21, "large_frozen": 0.9, "large_pointwise": 1.0,
        "probe_f64_frozen": 2.4, "probe_f64_pointwise": 4.8, "hvp": 0.75, "f64_eval_one_probe": 1.3,
        "hvp_f64": 17.5, "hchecks": 340.0}


def names():
    return [n for n, _ in build_model("resnet20_bn_cifar", 10).named_parameters()]


def tensor_digest(d: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(k.encode())
        h.update(np.ascontiguousarray(d[k].to(torch.float32).numpy()).tobytes())
    return h.hexdigest()


def needed_epochs(m):
    e = set(C.TEMPORAL_EPOCHS)
    for t in V2.transitions(m):
        e.add(t["update"] // C.UPE)
    return sorted(e)


# ----------------------------------------------------------------------------------

def verify():
    PROTO.mkdir(parents=True, exist_ok=True)
    nm = names()
    rep = {"checkpoints": {}, "problems": []}
    digests = {"checkpoints": {}, "directions": {}, "subsets": {}}
    for s in C.SEEDS:
        acc = C.V2_ACCOUNT_OF_SEED[s]
        man = json.loads((C.V2_RAW / acc / "v2" / "runs_manifest.json").read_text())
        chk = json.loads((C.V2_RAW / acc / "v2" / "checks" / ("checks_seed%d.json" % s)).read_text())
        rep["pairing_seed%d" % s] = {"v2_checks_summary": chk["summary"],
                                     "init_sha_by_method": chk["pairing"]["init_sha_by_method"]}
        if not (chk["summary"]["pairing_ok"] and chk["summary"]["inventory_ok"]):
            rep["problems"].append("seed %d: v2 pairing/inventory check not ok" % s)
        for m in C.METHODS:
            metrics = {r["update"]: r for r in json.loads((C.v2_run_dir(m, s) / "metrics.json").read_text())}
            ctrl = V2.controller(m)
            for e in needed_epochs(m):
                f = C.epoch_file(e)
                rel = "runs/%s/checkpoints/%s" % (C.rname(m, s), f)
                p = C.v2_run_dir(m, s) / "checkpoints" / f
                r = {"exists": p.exists()}
                if p.exists():
                    sha = C.sha_file(p)
                    r["sha256"] = sha
                    r["manifest_match"] = man.get(rel, {}).get("sha256") == sha
                    ck = torch.load(p, map_location="cpu", weights_only=False)
                    u = int(ck["global_update"])
                    used = ck["intervention"]["used_for_last_update"]
                    nxt = ck["intervention"]["next_update"]
                    su = V2.state_dict_of(ctrl.state_for_epoch(e - 1))
                    sn = None if u >= V2.TOTAL_UPDATES else V2.state_dict_of(ctrl.state_for_epoch(e))
                    gu = {"resolution": used["state"]["resolution"], "sigma": used["state"]["sigma"]}
                    gn = None if nxt is None else {"resolution": nxt["state"]["resolution"], "sigma": nxt["state"]["sigma"]}
                    mr = metrics.get(u)
                    mu = None if mr is None or mr["state_used"] is None else {
                        "resolution": mr["state_used"]["state"]["resolution"], "sigma": mr["state_used"]["state"]["sigma"]}
                    mn = None if mr is None or mr["state_next"] is None else {
                        "resolution": mr["state_next"]["state"]["resolution"], "sigma": mr["state_next"]["state"]["sigma"]}
                    order = [k for k in ck["model_state"] if k in nm]
                    r.update({"schema_ok": ck["schema"] == "landscape_v2.analysis_checkpoint/1",
                              "update": u, "update_ok": u == e * C.UPE, "method_ok": ck["method"] == m,
                              "seed_ok": ck["seed"] == s, "state_used": gu, "state_next": gn,
                              "state_used_matches_schedule": gu == su, "state_next_matches_schedule": gn == sn,
                              "state_used_matches_metrics": mu == gu, "state_next_matches_metrics": mn == gn,
                              "parameter_order_ok": order == nm,
                              "config_sha256": ck["config_sha256"]})
                    digests["checkpoints"]["%s__%s" % (C.rname(m, s), f)] = sha
                ok = r.get("exists") and all(r.get(k) for k in (
                    "manifest_match", "schema_ok", "update_ok", "method_ok", "seed_ok", "state_used_matches_schedule",
                    "state_next_matches_schedule", "state_used_matches_metrics", "state_next_matches_metrics",
                    "parameter_order_ok"))
                r["ok"] = bool(ok)
                if not ok:
                    rep["problems"].append("%s %s: %s" % (C.rname(m, s), f, r))
                rep["checkpoints"]["%s/%s" % (C.rname(m, s), f)] = r
        evman = json.loads((C.V2_RAW / acc / "v2" / "eval_manifest.json").read_text())
        for k in range(C.N_DIRECTIONS):
            rel = "directions/seed%d_dir%02d.pt" % (s, k)
            p = C.V2_RAW / acc / "v2" / rel
            ok = evman.get(rel, {}).get("sha256") == C.sha_file(p)
            d = torch.load(p, map_location="cpu", weights_only=True)
            g = torch.Generator().manual_seed(C.direction_seed(s, k))
            regen = {n: torch.randn(d[n].shape, generator=g, dtype=torch.float32) for n in nm if n in d}
            digests["directions"]["seed%d_dir%02d" % (s, k)] = tensor_digest(d)
            digests["direction_files"] = digests.get("direction_files", {})
            digests["direction_files"]["seed%d_dir%02d.pt" % (s, k)] = C.sha_file(p)
            same = tensor_digest(regen) == tensor_digest(d)
            rep.setdefault("direction_local_regeneration_identical", {})["seed%d_dir%02d" % (s, k)] = same
            if not ok:
                rep["problems"].append("direction %s: file sha does not match the v2 eval manifest" % rel)
        rep["directions_seed%d" % s] = ("20 files match the v2 eval manifest; they are shipped as files (local "
                                        "torch %s does not regenerate the Kaggle torch 2.10 draws bitwise, "
                                        "so regeneration is not used)" % torch.__version__)
    sub = dict(np.load(C.V2_STUDY / "inputs" / "subsets.npz"))
    v2man = json.loads((C.V2_STUDY / "inputs" / "manifest.json").read_text())
    rep["subsets_file_matches_v2_manifest"] = v2man["subsets.npz"] == C.sha_file(C.V2_STUDY / "inputs" / "subsets.npz")
    for k, v in sub.items():
        digests["subsets"][k] = hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
    digests["subsets_file"] = C.sha_file(C.V2_STUDY / "inputs" / "subsets.npz")
    digests["test_full"] = "all 10,000 CIFAR-10 test images in file order (pankrzysiu/cifar10-python test_batch)"
    rep["n_checkpoints_checked"] = len(rep["checkpoints"])
    rep["all_ok"] = not rep["problems"] and rep["subsets_file_matches_v2_manifest"]
    (PROTO / "input_verification.json").write_text(json.dumps(rep, indent=1, default=str))
    (PROTO / "expected_digests.json").write_text(json.dumps(digests, indent=1))
    print("checked", rep["n_checkpoints_checked"], "checkpoints; problems:", len(rep["problems"]), "all_ok", rep["all_ok"])
    for p in rep["problems"][:20]:
        print("  ", p)


# ----------------------------------------------------------------------------------

class Req:
    """Required points grouped by (m, s, ckpt, state, splitset, prec)."""

    def __init__(self):
        self.groups = {}
        self.blocks = defaultdict(set)

    def add(self, block, m, s, ckpt, state, splitset, prec, pts, grid=False):
        gk = (m, s, ckpt, C.stag(state), splitset, prec, "grid" if grid else "r1")
        g = self.groups.setdefault(gk, {"m": m, "s": s, "ckpt": ckpt, "state": state, "splitset": splitset,
                                        "prec": prec, "grid": grid, "points": set()})
        for p in pts:
            g["points"].add(p)
            self.blocks[(gk, p)].add(block)

    @staticmethod
    def key(g, p):
        if g["grid"]:
            pol, a, b = p
            spec = C.spec_r2(0, 1, a, b)
        else:
            pol, k, amp = p
            spec = "c" if k < 0 else C.spec_r1(k, amp)
        return C.pkey(g["m"], g["s"], g["ckpt"], g["state"], pol, g["splitset"], spec, g["prec"])


def pts1d(policies, dirs, amps):
    out = [(pol, -1, 0.0) for pol in policies]
    out += [(pol, k, round(sg * a, 10)) for pol in policies for k in dirs for a in amps for sg in (1, -1)]
    return out


def produced_state(m, s, e):
    ck = torch.load(C.v2_run_dir(m, s) / "checkpoints" / C.epoch_file(e), map_location="cpu", weights_only=False)
    u = ck["intervention"]["used_for_last_update"]["state"]
    n = ck["intervention"]["next_update"]
    return ({"resolution": u["resolution"], "sigma": u["sigma"]},
            None if n is None else {"resolution": n["state"]["resolution"], "sigma": n["state"]["sigma"]})


def build_requirements():
    R = Req()
    for s in C.SEEDS:
        for m in C.METHODS:
            tgt = C.target(m)
            R.add("A_amplitudes", m, s, C.FINAL, tgt, "probes", "f32",
                  pts1d(C.POLICIES, range(C.N_DIRECTIONS), C.AMPLITUDES))
            R.add("A_float64_check", m, s, C.FINAL, tgt, "probes", "f64",
                  pts1d(C.POLICIES, C.F64_CHECK["directions"], C.F64_CHECK["amplitudes"]))
            R.add("B_validation", m, s, C.FINAL, tgt, "large", "f32",
                  pts1d((C.SAVED, C.POINTWISE), range(C.N_DIRECTIONS), C.VALIDATION_AMPS))
            for e in C.TEMPORAL_EPOCHS:
                used, _ = produced_state(m, s, e)
                states = [tgt] if m == "plain" else ([used] + ([tgt] if used != tgt else []))
                for st in states:
                    R.add("C_temporal", m, s, C.epoch_file(e), st, "probes", "f32",
                          pts1d((C.SAVED, C.POINTWISE), C.TEMPORAL_DIRS, C.TEMPORAL_AMPS))
            if m != "plain":
                for tr in V2.transitions(m):
                    ck = C.epoch_file(tr["update"] // C.UPE)
                    for lab, st in V2.fixed_weight_states(m, tr):
                        R.add("C_transitions", m, s, ck, st, "probes", "f32",
                              pts1d((C.POINTWISE,), C.TRANSITION_DIRS, C.TRANSITION_AMPS_RECAL)
                              + pts1d((C.SAVED,), C.TRANSITION_DIRS, C.TRANSITION_AMPS_SAVED))
            R.add("E_wide_plane_g21", m, s, C.FINAL, tgt, "probes", "f32",
                  [(C.POINTWISE, a, b) for a in C.WIDE21 for b in C.WIDE21], grid=True)
            if s == C.PRIMARY_SEED:
                R.add("E_wide_plane_g41_seed0", m, s, C.FINAL, tgt, "probes", "f32",
                      [(C.POINTWISE, a, b) for a in C.WIDE41 for b in C.WIDE41], grid=True)
    return R


def hessian_tasks(s):
    T = []
    for m in C.METHODS:
        T.append({"kind": "hchecks", "id": "hchecks__%s__seed%d" % (C.SHORT[m], s), "m": m, "s": s, "prio": 0})
        for probe in C.HESS_PROBES:
            for pol in C.HESS_POLICIES:
                for coords in C.HESS_COORDS:
                    pid = C.hess_problem(m, s, probe, pol, coords)
                    T.append({"kind": "hessian", "id": "hessian__" + pid, "problem": pid, "m": m, "s": s,
                              "probe": probe, "policy": pol, "coords": coords, "prio": 1})
        T.append({"kind": "quadform", "id": "quadform__%s__seed%d" % (C.SHORT[m], s), "m": m, "s": s, "prio": 2})
        T.append({"kind": "hcut", "id": "hcut__%s__seed%d" % (C.SHORT[m], s), "m": m, "s": s, "prio": 5})
        if s == C.PRIMARY_SEED:
            for pol in C.HESS_POLICIES:
                pid = C.hess_problem(m, s, "train_probe", pol, "relative")
                for pair in (("top1", "top2"), ("top1", "min")):
                    T.append({"kind": "hplane", "id": "hplane__%s__%s_%s" % (pid, *pair), "m": m, "s": s,
                              "problem": pid, "pair": list(pair), "policy": pol, "prio": 6})
    return T


def reuse_samples(tasks, v2, per_task=2, max_tasks=15):
    """Reused v2 points re-evaluated by the job before trusting the rest."""
    withr = [t for t in tasks if t.get("reused_points")]
    step = max(1, len(withr) // max_tasks)
    out = []
    for t in withr[::step][:max_tasks]:
        rp = t["reused_points"]
        for p in (rp[0], rp[len(rp) // 2])[:per_task]:
            if t["kind"] == "grid":
                spec = C.spec_r2(0, 1, p[1], p[2])
            else:
                spec = "c" if p[1] < 0 else C.spec_r1(p[1], p[2])
            key = C.pkey(t["m"], t["s"], t["ckpt"], t["state"], p[0], t["splitset"], spec, t["prec"])
            out.append({"key": key, "kind": t["kind"], "m": t["m"], "s": t["s"], "ckpt": t["ckpt"], "state": t["state"],
                        "splitset": t["splitset"], "point": p, "expected": v2[key]["splits"], "source": v2[key]["source"]})
    return out


def pilot_tasks(tasks):
    """Small timing/numerics pilot drawn from an account's own tasks."""
    out, seen = [], set()
    for t in tasks:
        tag = (t["kind"], t.get("splitset"), t.get("prec"), t.get("ckpt") == C.FINAL)
        if tag in seen:
            continue
        seen.add(tag)
        u = dict(t, id="pilot__" + t["id"])
        if t["kind"] in ("pts", "grid"):
            byp = defaultdict(list)
            for p in t["points"]:
                byp[p[0]].append(p)
            u["points"] = [p for v in byp.values() for p in v[:6]]
            u["reused_points"] = []
        if t["kind"] == "hessian" and not (t["coords"] == "relative" and t["policy"] == C.CFROZEN
                                           and t["probe"] == "train_probe"):
            seen.discard(tag)
            continue
        if t["kind"] == "hcut":
            u["pids"] = [C.hess_problem(t["m"], t["s"], "train_probe", C.CFROZEN, "relative")]
        if t["kind"] in ("hcut", "hplane", "quadform", "hchecks", "hessian") and t["m"] != C.METHODS[0]:
            seen.discard(tag)
            continue
        if t["kind"] == "hplane" and t["policy"] != C.CFROZEN:
            seen.discard(tag)
            continue
        out.append(u)
    return out


def cost_of(g, npts):
    big = g["splitset"] == "large"
    out = 0.0
    for p in npts:
        pol = p[0]
        if g["prec"] == "f64":
            out += COST["probe_f64_pointwise" if pol == C.POINTWISE else "probe_f64_frozen"]
        elif big:
            out += COST["large_pointwise" if pol == C.POINTWISE else "large_frozen"]
        else:
            out += COST["probe_pointwise" if pol == C.POINTWISE else "probe_frozen"]
    return out


def hessian_cost(m_iter):
    # per problem: 2 Lanczos starts + explicit residuals; verification of top1/min: 2 f32 + 2 f64 HVPs and
    # 13 float64 one-probe evaluations each.  hchecks (float64 gradients) measured at ~170 s per network.
    per_problem = 2 * m_iter * COST["hvp"] + 2 * 3 * COST["hvp"] + 2 * (COST["hvp"] + COST["hvp_f64"]) \
        + 2 * 13 * COST["f64_eval_one_probe"]
    per_net = 8 * per_problem + 4 * 20 * COST["hvp"] + COST["hchecks"]
    cuts = (2 * 2 * 5 * 40 + 2 * 2 * 40) * COST["probe_frozen"] + 2 * 40 * COST["probe_pointwise"]
    plane = 2 * 2 * 1681 * COST["probe_frozen"]
    return per_net, cuts, plane


def reassign(excluded):
    """Move the seeds / extra work of accounts without GPU access to the remaining accounts."""
    global EXTRA_ACCOUNT
    ok = [a for a in C.ACCOUNTS if a not in excluded]
    if not ok:
        raise SystemExit("no account left")
    load = {a: 0 for a in ok}
    for s, a in list(SEED_ACCOUNT.items()):
        if a in load:
            load[a] += 1
    for s, a in list(SEED_ACCOUNT.items()):
        if a in excluded:
            b = min(load, key=lambda x: (load[x], ok.index(x)))
            SEED_ACCOUNT[s] = b
            load[b] += 1
    if EXTRA_ACCOUNT in excluded:
        EXTRA_ACCOUNT = min(load, key=lambda x: (load[x], ok.index(x)))
    return {"seed_account": dict(SEED_ACCOUNT), "extra_account": EXTRA_ACCOUNT, "excluded": sorted(excluded)}


def matrix(m_iter, excluded=()):
    assignment = reassign(set(excluded)) if excluded else {"seed_account": dict(SEED_ACCOUNT),
                                                          "extra_account": EXTRA_ACCOUNT, "excluded": []}
    for p in PROTO.glob("bundle_*.json"):
        p.unlink()
    for p in PROTO.glob("pilot_*.json"):
        if p.name != "pilot_costs.json":
            p.unlink()
    if (PROTO / "pilot_costs.json").exists():
        COST.update(json.loads((PROTO / "pilot_costs.json").read_text()))
    R = build_requirements()
    v2 = v2points.load()
    blocks = defaultdict(lambda: {"required": 0, "reused_v2": 0, "new": 0})
    tasks = defaultdict(list)
    est = defaultdict(float)
    total_new, total_req = 0, 0
    for gk, g in sorted(R.groups.items(), key=lambda kv: str(kv[0])):
        new, reused = [], []
        for p in sorted(g["points"], key=str):
            k = R.key(g, p)
            hit = k in v2
            for b in R.blocks[(gk, p)]:
                blocks[b]["required"] += 1
                blocks[b]["reused_v2" if hit else "new"] += 1
            (reused if hit else new).append(list(p))
        total_req += len(g["points"])
        total_new += len(new)
        grid = g["grid"]
        if g["prec"] == "f64" or (grid and len(g["points"]) > len(C.WIDE21) ** 2):
            acc = EXTRA2_ACCOUNT
        elif grid:
            acc = EXTRA_ACCOUNT
        else:
            acc = SEED_ACCOUNT[g["s"]]
        tid = "%s__%s__seed%d__%s__%s__%s__%s" % ("grid" if grid else "pts", C.SHORT[g["m"]], g["s"],
                                                 g["ckpt"].replace(".pt", ""), C.stag(g["state"]), g["splitset"], g["prec"])
        prio = 3 if grid else (2 if g["ckpt"] == C.FINAL else 4)
        tasks[acc].append({"kind": "grid" if grid else "pts", "id": tid, "m": g["m"], "s": g["s"], "ckpt": g["ckpt"],
                           "state": g["state"], "splitset": g["splitset"], "prec": g["prec"], "pair": [0, 1],
                           "points": new, "reused_points": reused, "prio": prio,
                           "est_seconds": cost_of(g, new)})
        est[acc] += cost_of(g, new)
    per_net, cuts, plane = hessian_cost(m_iter)
    for s in C.SEEDS:
        acc = SEED_ACCOUNT[s]
        tasks[acc] += hessian_tasks(s)
        est[acc] += 4 * (per_net + cuts) + (4 * plane if s == C.PRIMARY_SEED else 0)
    proto = json.loads((PROTO / "numerical_protocol.json").read_text()) if (PROTO / "numerical_protocol.json").exists() else {}
    for acc in tasks:
        tasks[acc].sort(key=lambda t: (t["prio"], -t.get("est_seconds", 0)))
        samples = reuse_samples(tasks[acc], v2)
        (PROTO / ("bundle_%s.json" % acc)).write_text(json.dumps(
            {"account": acc, "tasks": tasks[acc], "reuse_samples": samples, "protocol": proto}, default=str))
        (PROTO / ("pilot_%s.json" % acc)).write_text(json.dumps(
            {"account": acc, "tasks": pilot_tasks(tasks[acc]), "reuse_samples": samples[:6], "protocol": proto},
            default=str))
    out = {"assignment": assignment, "blocks": dict(blocks), "total_required_points": total_req, "total_new_points": total_new,
           "total_reused_from_v2": total_req - total_new,
           "hessian": {"problems": 20 * 8, "starts_per_problem": len(C.HESS_STARTS), "assumed_iterations": m_iter,
                       "per_network_gpu_seconds": per_net, "cuts_per_network_gpu_seconds": cuts,
                       "seed0_planes_per_network_gpu_seconds": plane},
           "gpu_seconds_estimate_by_account": dict(est),
           "wall_hours_estimate_by_account_2xT4": {a: v / 2 / 3600 for a, v in est.items()},
           "cost_constants_seconds_per_evaluation": COST,
           "n_tasks_by_account": {a: len(t) for a, t in tasks.items()}}
    (PROTO / "evaluation_matrix.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "cost_constants_seconds_per_evaluation"}, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("verify", "matrix"))
    ap.add_argument("--lanczos-iterations", type=int, default=300)
    ap.add_argument("--exclude-accounts", nargs="*", default=[])
    a = ap.parse_args()
    PROTO.mkdir(parents=True, exist_ok=True)
    verify() if a.action == "verify" else matrix(a.lanczos_iterations, a.exclude_accounts)


if __name__ == "__main__":
    main()
