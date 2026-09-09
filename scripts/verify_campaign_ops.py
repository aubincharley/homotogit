"""Focused checks for the 21-configuration campaign, run before dispatch.

Covers: manifest uniqueness; schedule boundaries; exact reduction and filter
bypasses; spatial shapes on every reduction path; forward/backward finiteness
for the new stem, max-pool, Gmix and early7 paths; 19-vs-7 active sites and the
per-site sigma scaling (including the full-resolution stem under stem
reduction); recovery of the ordinary inference path in the final phase; and that
evaluation touches neither BN statistics nor the training RNG.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from continuation.campaign_ops import (EARLY7, N_SITES, SiteController,
                                       attach_sites, describe_adaptive,
                                       reduce_spatial)
from continuation.config import ModelConfig
from continuation.models import build_model
from scripts.campaign_manifest import (GAUSSIAN, RESOLUTIONS, build_cells,
                                       build_configs)

OUT = Path("results/campaign_verification.json")
REPORT = {}


def model():
    return build_model(ModelConfig(arch="resnet20_bn_cifar"), 10, seed=0)


def check_manifest():
    cfgs, cells = build_configs(), build_cells()
    r = {"n_configurations": len(cfgs),
         "unique_configurations": len({c["id"] for c in cfgs}),
         "n_cells": len(cells), "unique_cells": len({c["cell_id"] for c in cells}),
         "per_group": {g: len([c for c in cfgs if c["group"] == g])
                       for g in "ABCDE"}}
    r["ok"] = (r["n_configurations"] == r["unique_configurations"] == 21
               and r["n_cells"] == r["unique_cells"] == 63)
    return r


def check_schedules():
    r = {"resolution_lengths": {k: len(v) for k, v in RESOLUTIONS.items()},
         "boundaries": {k: {"e5": v[5], "e6": v[6], "e11": v[11], "e12": v[12],
                            "e29": v[29]} for k, v in RESOLUTIONS.items()}}
    g = GAUSSIAN["Gplateau"]
    r["gplateau"] = {"e0": g[0], "e2": g[2], "e3": g[3], "e20": g[20],
                     "e21": g[21], "e29": g[29]}
    r["ggeo"] = {"e0": GAUSSIAN["Ggeo"][0], "e20": round(GAUSSIAN["Ggeo"][20], 6),
                 "e21": GAUSSIAN["Ggeo"][21], "e29": GAUSSIAN["Ggeo"][29]}
    r["gmix_alpha"] = {"e0": GAUSSIAN["Gmix"][0], "e20": GAUSSIAN["Gmix"][20],
                       "e21": GAUSSIAN["Gmix"][21]}
    r["all_zero_from_21"] = all(
        GAUSSIAN[k][e] == 0.0 for k in ("Gplateau", "Ggeo", "Gmix")
        for e in range(21, 30))
    r["all_r32_from_12"] = all(v[e] == 32 for v in RESOLUTIONS.values()
                               for e in range(12, 30))
    r["rreverse_preserves_time"] = (
        sorted(RESOLUTIONS["Rreverse"][:12]) == sorted(RESOLUTIONS["Rprog"][:12]))
    r["ok"] = r["all_zero_from_21"] and r["all_r32_from_12"] and r["rreverse_preserves_time"]
    return r


def check_adaptive():
    return {"32->16": describe_adaptive(32, 16), "32->24": describe_adaptive(32, 24)}


def check_shapes_and_finiteness():
    out = {}
    for reduction in ("input_bilinear", "input_max", "stem_bilinear", "stem_max"):
        for r in (16, 24, 32):
            m = model().train()
            ctrl = SiteController("gaussian", levels=[0.5] * 30,
                                  resolution_by_epoch=[r] * 30, reduction=reduction)
            h = attach_sites(m, ctrl)
            ctrl.set_epoch(0)
            res_in = ctrl.input_resolution()
            x = torch.rand(4, 3, 32 if res_in is None else res_in,
                           32 if res_in is None else res_in, requires_grad=True)
            sizes = []
            probes = [b.register_forward_hook(
                lambda _m, _i, o: sizes.append(int(o.shape[-1])))
                for b in m.blocks]
            y = m(x)
            y.pow(2).sum().backward()
            for p in probes:
                p.remove()
            out["%s_r%d" % (reduction, r)] = {
                "input_shape": list(x.shape),
                "stage_widths": [sizes[0], sizes[3], sizes[6]],
                "logits_shape": list(y.shape),
                "out_finite": bool(torch.isfinite(y).all()),
                "grad_finite": bool(torch.isfinite(x.grad).all()),
                "grad_norm": float(x.grad.norm()),
                "q_stem": ctrl.q[0], "q_deep": ctrl.q[-1],
            }
            for k in h:
                k.remove()
    return out


def check_bypass_exactness():
    """r=32 and level=0 must reproduce the ordinary ResNet-20 bitwise."""
    out = {}
    x = torch.rand(4, 3, 32, 32)
    m = model().eval()
    with torch.no_grad():
        reference = m(x)
    for reduction in ("input_bilinear", "input_max", "stem_bilinear", "stem_max"):
        for op, lvl in (("gaussian", 0.0), ("gmix", 0.0), ("none", None)):
            mm = model().eval()
            ctrl = SiteController(op, levels=None if lvl is None else [lvl] * 30,
                                  resolution_by_epoch=[32] * 30, reduction=reduction)
            h = attach_sites(mm, ctrl)
            ctrl.set_epoch(29)
            with torch.no_grad():
                y = mm(x if ctrl.input_resolution() is None
                       else reduce_spatial(x, ctrl.input_resolution(), reduction))
            out["%s_%s" % (reduction, op)] = {
                "max_abs_diff_vs_ordinary": float((y - reference).abs().max()),
                "bitwise_identical": bool((y - reference).abs().max() == 0)}
            for k in h:
                k.remove()
    # reduce_spatial itself must be an identity object at r == current size
    t = torch.rand(2, 3, 32, 32)
    out["reduce_identity_object"] = {
        red: reduce_spatial(t, 32, red) is t
        for red in ("input_bilinear", "input_max", "stem_bilinear", "stem_max")}
    return out


def check_sites_and_scaling():
    out = {}
    for mask, sites in (("all19", tuple(range(N_SITES))), ("early7", EARLY7)):
        m = model()
        calls = []
        ctrl = SiteController("gaussian", levels=[1.0] * 30, sites=sites,
                              resolution_by_epoch=[16] * 30,
                              reduction="input_bilinear")
        orig = ctrl.apply_at

        def spy(site, h, _o=orig):
            before = h
            after = _o(site, h)
            calls.append((site, bool(after is not before)))
            return after

        ctrl.apply_at = spy
        hs = attach_sites(m, ctrl)
        ctrl.set_epoch(0)
        m(torch.rand(2, 3, 16, 16))
        for k in hs:
            k.remove()
        active = [s for s, changed in calls if changed]
        out[mask] = {"sites_visited": len(calls), "sites_active": len(active),
                     "active_indices": active,
                     "expected_active": len(sites),
                     "ok": len(calls) == N_SITES and active == list(sites)}
    # per-site q under each reduction location
    for reduction, expect_stem in (("input_bilinear", 0.5), ("stem_bilinear", 1.0)):
        c = SiteController("gaussian", levels=[1.0] * 30,
                           resolution_by_epoch=[16] * 30, reduction=reduction)
        c.set_epoch(0)
        out["q_" + reduction] = {"stem": c.q[0], "deep": c.q[-1],
                                 "stem_expected": expect_stem,
                                 "sigma_stem": c.q[0] * 1.0,
                                 "sigma_deep": c.q[-1] * 1.0}
    c = SiteController("gaussian", levels=[1.0] * 30,
                       resolution_by_epoch=[32] * 30, reduction="input_bilinear")
    c.set_epoch(0)
    out["q_R32"] = {"all_one": all(v == 1.0 for v in c.q)}
    return out


def check_gmix():
    """alpha=1 must equal the Gaussian; alpha=0 must return h unevaluated."""
    x = torch.rand(2, 16, 32, 32)
    gmix = SiteController("gmix", levels=[1.0] * 30,
                          resolution_by_epoch=[32] * 30)
    gmix.set_epoch(0)
    gauss = SiteController("gaussian", levels=[1.0] * 30,
                           resolution_by_epoch=[32] * 30)
    gauss.set_epoch(0)
    a1 = gmix.apply_at(0, x)
    g1 = gauss.apply_at(0, x)
    gmix.set_state(0.0, 32)
    a0 = gmix.apply_at(0, x)
    gmix.set_state(0.5, 32)
    xi = x.clone().requires_grad_(True)
    a5 = gmix.apply_at(0, xi)
    a5.pow(2).sum().backward()
    return {"alpha1_matches_gaussian_max_abs": float((a1 - g1).abs().max()),
            "alpha1_bitwise": bool((a1 - g1).abs().max() == 0),
            "alpha0_returns_input_object": a0 is x,
            "alpha05_grad_finite": bool(torch.isfinite(xi.grad).all()),
            "alpha05_differs_from_input": float((a5 - x).abs().max()) > 0}


def check_eval_side_effects():
    """Evaluation must not move BN statistics or the training RNG."""
    m = model().train()
    ctrl = SiteController("gaussian", levels=[0.5] * 30,
                          resolution_by_epoch=[16] * 30, reduction="input_bilinear")
    h = attach_sites(m, ctrl)
    ctrl.set_epoch(0)
    m(torch.rand(8, 3, 16, 16))                      # populate running stats
    bn = {k: v.clone() for k, v in m.state_dict().items()
          if "running_" in k or "num_batches" in k}
    # evaluation batches are fixed tensors held by the driver (probe / test
    # splits), so they are drawn here *before* the RNG state is captured --
    # the eval loop itself must consume no randomness at all.
    eval_batches = [torch.rand(8, 3, 32, 32) for _ in range(3)]
    torch.manual_seed(1234)
    rng_before = torch.get_rng_state().clone()
    was_training = m.training
    m.eval()
    with torch.no_grad():
        for b in eval_batches:
            m(b)
    m.train(was_training)
    after = {k: v for k, v in m.state_dict().items()
             if "running_" in k or "num_batches" in k}
    for k in h:
        k.remove()
    return {"bn_unchanged": all(torch.equal(bn[k], after[k]) for k in bn),
            "rng_unchanged": bool(torch.equal(rng_before, torch.get_rng_state())),
            "mode_restored": m.training == was_training,
            "n_bn_buffers": len(bn)}


def main():
    REPORT["manifest"] = check_manifest()
    REPORT["schedules"] = check_schedules()
    REPORT["adaptive_pool_windows"] = check_adaptive()
    REPORT["shapes_and_finiteness"] = check_shapes_and_finiteness()
    REPORT["bypass_exactness"] = check_bypass_exactness()
    REPORT["sites_and_scaling"] = check_sites_and_scaling()
    REPORT["gmix"] = check_gmix()
    REPORT["eval_side_effects"] = check_eval_side_effects()

    ok = (REPORT["manifest"]["ok"] and REPORT["schedules"]["ok"]
          and all(v["bitwise_identical"] for k, v in REPORT["bypass_exactness"].items()
                  if isinstance(v, dict) and "bitwise_identical" in v)
          and all(v["out_finite"] and v["grad_finite"]
                  for v in REPORT["shapes_and_finiteness"].values())
          and REPORT["sites_and_scaling"]["all19"]["ok"]
          and REPORT["sites_and_scaling"]["early7"]["ok"]
          and REPORT["gmix"]["alpha1_bitwise"]
          and REPORT["gmix"]["alpha0_returns_input_object"]
          and REPORT["eval_side_effects"]["bn_unchanged"]
          and REPORT["eval_side_effects"]["rng_unchanged"])
    REPORT["all_ok"] = ok

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(REPORT, indent=2, default=str))

    m = REPORT["manifest"]
    print("manifest      %d configs / %d cells  ok=%s" %
          (m["n_configurations"], m["n_cells"], m["ok"]))
    print("schedules     ok=%s  (bypass from e21, r=32 from e12, Rreverse balanced)"
          % REPORT["schedules"]["ok"])
    for k, v in REPORT["adaptive_pool_windows"].items():
        print("adaptive %-7s widths=%s overlapping_pairs=%d disjoint_uniform=%s"
              % (k, v["distinct_widths"], v["n_overlapping_pairs"],
                 v["disjoint_uniform"]))
    for k, v in sorted(REPORT["shapes_and_finiteness"].items()):
        print("  %-22s in=%s stages=%s q_stem=%.3f q_deep=%.3f finite=%s"
              % (k, v["input_shape"][-1], v["stage_widths"], v["q_stem"],
                 v["q_deep"], v["out_finite"] and v["grad_finite"]))
    print("bypass exact: %s" % all(
        v["bitwise_identical"] for v in REPORT["bypass_exactness"].values()
        if isinstance(v, dict) and "bitwise_identical" in v))
    print("sites all19=%s early7=%s" % (REPORT["sites_and_scaling"]["all19"]["ok"],
                                        REPORT["sites_and_scaling"]["early7"]["ok"]))
    print("gmix %s" % json.dumps(REPORT["gmix"]))
    print("eval side effects %s" % json.dumps(REPORT["eval_side_effects"]))
    print("ALL OK: %s   ->  %s" % (ok, OUT))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
