"""Per-seed verification, written to ``checks/checks_seed<k>.json``.

inventory     every expected checkpoint exists, loads, has the schema, its update
              count matches its name, and its used/next states match the schedule
pairing       the four initialisation checkpoints are bitwise identical and equal
              the asset initial state; the four configs differ only in method/name
reproduction  saved-statistics evaluation of the final checkpoint vs the run's own
              last metrics record (pinned 500-image probe, full test set)
bypass        each continuation model at its target state gives bitwise the logits
              of the hook-free plain network (inference and training mode)
directions    filter-wise normalisation error, zero-norm blocks
reconstruct   centre + 0 * direction == checkpoint parameters (bitwise)
endpoints     alpha 0 / 1 give the exact endpoint parameters
repeat        repeated and reverse-order evaluations identical (both BN policies)
non_mutation  checkpoint files and loaded tensors unchanged
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from continuation_core import assets as A

from . import common as C
from .evaluator import weights_sha


def _sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_checks(ctx, s, out: Path, pilot=False):
    out.mkdir(parents=True, exist_ok=True)
    res, names, ev = {"seed": s}, ctx.names, ctx.ev
    final = "epoch_003.pt" if pilot else "epoch_030.pt"
    last_epoch = 3 if pilot else C.EPOCHS

    inv = {}
    for m in C.METHODS:
        d = ctx.runs / ("%s__seed%d" % (m, s)) / "checkpoints"
        ctrl = C.controller(m)
        expected = {"epoch_%03d.pt" % e for e in range(last_epoch + 1)}
        if not pilot:
            for t in C.transitions(m):
                for o in C.TRANSITION_OFFSETS:
                    u = t["update"] + o
                    if 0 < u <= C.TOTAL_UPDATES and u % C.UPDATES_PER_EPOCH:
                        expected.add("update_%06d.pt" % u)
        present = {p.name for p in d.glob("*.pt")} if d.is_dir() else set()
        bad = []
        for f in sorted(present):
            try:
                ck = torch.load(d / f, map_location="cpu", weights_only=False)
                u = int(ck["global_update"])
                want = int(f[6:9]) * C.UPDATES_PER_EPOCH if f.startswith("epoch_") else int(f[7:13])
                used = ck["intervention"]["used_for_last_update"]
                nxt = ck["intervention"]["next_update"]
                su = None if u == 0 else C.state_dict_of(ctrl.state_for_epoch((u - 1) // C.UPDATES_PER_EPOCH))
                sn = None if u >= C.TOTAL_UPDATES else C.state_dict_of(ctrl.state_for_epoch(u // C.UPDATES_PER_EPOCH))
                gu = None if used is None else {"resolution": used["state"]["resolution"], "sigma": used["state"]["sigma"]}
                gn = None if nxt is None else {"resolution": nxt["state"]["resolution"], "sigma": nxt["state"]["sigma"]}
                ok = (ck.get("schema") == "landscape_v2.analysis_checkpoint/1" and u == want
                      and gu == su and gn == sn
                      and set(ck["model_state"]) == set(ctx.ev.model("plain")[0].state_dict()))
                if not ok:
                    bad.append({"file": f, "update": u, "want": want, "used": gu, "sched_used": su,
                                "next": gn, "sched_next": sn, "n_keys": len(ck["model_state"])})
            except Exception as exc:
                bad.append({"file": f, "error": repr(exc)})
        run_dir = d.parent
        inv[m] = {"n_present": len(present), "n_expected": len(expected),
                  "missing": sorted(expected - present), "invalid": bad,
                  "summary_json": (run_dir / "summary.json").exists(),
                  "rolling_pt": (run_dir / "rolling.pt").exists()}
    res["inventory"] = inv

    init = {m: ctx.ckpt(m, s, "epoch_000.pt")["model_state"] for m in C.METHODS}
    asset_sha = A.sha_state(torch.load(ctx.inputs / ("assets/init_seed%d.pt" % s) if (ctx.inputs / "assets").is_dir()
                                       else ctx.inputs / ("init_seed%d.pt" % s), map_location="cpu", weights_only=True))
    res["pairing"] = {"init_sha_by_method": {m: A.sha_state(v) for m, v in init.items()},
                      "asset_init_sha": asset_sha,
                      "all_equal_asset": all(A.sha_state(v) == asset_sha for v in init.values())}
    cfg_sha = {m: json.loads((ctx.runs / ("%s__seed%d" % (m, s)) / "config.json").read_text()) for m in C.METHODS}
    base = {k: v for k, v in cfg_sha["plain"].items() if k not in ("method", "run")}
    res["pairing"]["configs_identical_except_method_and_run"] = all(
        {k: v for k, v in c.items() if k not in ("method", "run")} == base for c in cfg_sha.values())

    rep = {}
    for m in C.METHODS:
        ck = ctx.ckpt(m, s, final)
        metrics = json.loads((ctx.runs / ("%s__seed%d" % (m, s)) / "metrics.json").read_text())
        rec = [r for r in metrics if r["update"] == ck["global_update"]]
        tgt = C.state_dict_of(C.controller(m).target_state())
        r = ev.evaluate(m, ck["model_state"], tgt, "saved", splits=("pinned_train_probe_500", "test_full"))
        if rec:
            e = rec[-1]["eval"]["target"]
            rep[m] = {"train_probe_ce_abs_diff": abs(r["splits"]["pinned_train_probe_500"]["ce"] - e["train_probe"]["ce"]),
                      "test_ce_abs_diff": abs(r["splits"]["test_full"]["ce"] - e["test"]["ce"]),
                      "test_acc_abs_diff": abs(r["splits"]["test_full"]["acc"] - e["test"]["acc"]),
                      "recorded_test_acc": e["test"]["acc"]}
        else:
            rep[m] = {"missing_record": True}
    res["reproduction"] = rep

    x = ev.pipeline(ev.splits["test_probe"][0][:500])
    byp = {}
    mp_, _ = ev.model("plain")
    for m in C.METHODS[1:]:
        st = ctx.ckpt(m, s, final)["model_state"]
        mm, cc = ev.model(m)
        with torch.no_grad():
            flags = {}
            for mode in ("eval", "train"):
                mm.load_state_dict({k: v.to(ctx.device) for k, v in st.items()})
                mp_.load_state_dict({k: v.to(ctx.device) for k, v in st.items()})
                cc.set_state(C.controller(m).target_state())
                mm.train(mode == "train"); mp_.train(mode == "train")
                flags[mode] = bool(torch.equal(mm(x), mp_(x)))
            first = C.transitions(m)[0]["before"]
            from .evaluator import as_state
            cc.set_state(as_state(first))
            mm.load_state_dict({k: v.to(ctx.device) for k, v in st.items()}); mm.eval(); mp_.eval()
            flags["active_state_differs"] = not torch.equal(mm(x), mp_(x))
        byp[m] = flags
    res["bypass"] = byp

    dirs, recon = {}, {}
    for m in C.METHODS:
        st = ctx.ckpt(m, s, final)["model_state"]
        worst, zero = 0.0, 0
        for k in range(2 if pilot else C.N_DIRECTIONS):
            _, r = ctx.direction(st, s, k)
            worst, zero = max(worst, r["max_abs_relative_norm_error"]), zero + r["n_zero_norm_blocks"]
        dirs[m] = {"max_abs_relative_norm_error": worst, "zero_norm_blocks": zero}
        d0, _ = ctx.direction(st, s, 0)
        c64 = ctx.center64(st)
        rebuilt = {n: (c64[n] + 0.0 * d0[n]).to(torch.float32).cpu() for n in names}
        recon[m] = all(torch.equal(rebuilt[n], st[n]) for n in names)
    res["directions"], res["reconstruction_bitwise"] = dirs, recon

    ends = {}
    for a, b in (("plain", "resolution_max_b1"), ("plain", "gaussian_postrelu"),
                 ("plain", "resolution_max_b1_gaussian_conv"), ("resolution_max_b1", "resolution_max_b1_gaussian_conv")):
        A_, B_ = ctx.ckpt(a, s, final)["model_state"], ctx.ckpt(b, s, final)["model_state"]
        e0 = all(torch.equal(((1.0 - 0.0) * A_[n].double() + 0.0 * B_[n].double()).float(), A_[n]) for n in names)
        e1 = all(torch.equal(((1.0 - 1.0) * A_[n].double() + 1.0 * B_[n].double()).float(), B_[n]) for n in names)
        ends["%s|%s" % (a, b)] = e0 and e1
    res["interpolation_endpoints_bitwise"] = ends

    m = "resolution_max_b1_gaussian_conv"
    st = ctx.ckpt(m, s, final)["model_state"]
    files = [ctx.runs / ("%s__seed%d" % (mm, s)) / "checkpoints" / final for mm in C.METHODS]
    before = {str(p): _sha(p) for p in files}
    tens_before = weights_sha(st, names)
    d0, _ = ctx.direction(st, s, 0)
    d1, _ = ctx.direction(st, s, 1)
    c64 = ctx.center64(st)
    pts = [(0.1, -0.05), (-0.25, 0.2), (0.0, 0.0)]
    tgt = C.state_dict_of(C.controller(m).target_state())
    rp = {}
    for pol in ("saved", "recalibrated"):
        def at(a, b):
            p = {n: (c64[n] + a * d0[n] + b * d1[n]).float() for n in names}
            return ev.evaluate(m, st, tgt, pol, params=p)["splits"]
        fwd = [at(*p) for p in pts]
        rev = [at(*p) for p in reversed(pts)][::-1]
        rp[pol] = {"repeat_identical": at(*pts[0]) == fwd[0], "order_independent": fwd == rev}
    res["repeatability"] = rp
    res["non_mutation"] = {"files_unchanged": all(_sha(p) == h for p, h in before.items()),
                           "tensors_unchanged": weights_sha(st, names) == tens_before}

    res["summary"] = {
        "inventory_ok": all(not v["missing"] and not v["invalid"] and (pilot or v["summary_json"]) for v in inv.values()),
        "pairing_ok": res["pairing"]["all_equal_asset"] and res["pairing"]["configs_identical_except_method_and_run"],
        "reproduction_max_test_ce_abs_diff": max((v.get("test_ce_abs_diff", float("inf")) for v in rep.values())),
        "bypass_ok": all(v["eval"] and v["train"] and v["active_state_differs"] for v in byp.values()),
        "directions_max_rel_error": max(v["max_abs_relative_norm_error"] for v in dirs.values()),
        "reconstruction_ok": all(recon.values()), "endpoints_ok": all(ends.values()),
        "repeatability_ok": all(v["repeat_identical"] and v["order_independent"] for v in rp.values()),
        "non_mutation_ok": all(res["non_mutation"].values())}
    (out / ("checks_seed%d.json" % s)).write_text(json.dumps(res, indent=2, default=str))
    print("checks seed", s, res["summary"], flush=True)
    return res
