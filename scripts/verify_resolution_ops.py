"""Checks for the resolution-only benchmark, run before any accuracy is read.

Covers: the five insertion sites and the shapes they produce; exact r=32 bypass;
constant-map behaviour; finite forward and backward including SoftPool's
temperature and the perceptual adaptive coefficients; the quadratic solution
maps against direct constrained solves; the perceptual fast path against a naive
per-patch reference; MaxBlur-without-blur against adaptive max; and every path
boundary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from continuation.config import ModelConfig
from continuation.models import build_model
from continuation.resolution_ops import (BLOCK_TARGET, LOCATIONS, O_REF,
                                         OPERATORS, PATHS,
                                         ResolutionController,
                                         attach_resolution, op_max, op_maxblur,
                                         perceptual_factor2,
                                         perceptual_reference, reduce_with,
                                         laplacian_pinv, solution_map, window_starts,
                                         window_widths, _laplacian,
                                         _upsample_matrix)

OUT = Path("results/resbench_verification.json")
R = (16, 24)


def model():
    return build_model(ModelConfig(arch="resnet20_bn_cifar"), 10, seed=0)


def check_sites():
    """Where each hook fires, and what the network then sees."""
    out = {}
    for loc in ("stem", "D0", "D1", "D2"):
        m = model().eval()
        ctrl = ResolutionController(O_REF, loc, "Rprog")
        ctrl.set_epoch(0)                                    # r = 16
        seen = []
        probes = [b.register_forward_hook(
            lambda _m, inp, o, i=i: seen.append((i, int(inp[0].shape[-1]),
                                                 int(o.shape[-1]))))
            for i, b in enumerate(m.blocks)]
        h = attach_resolution(m, ctrl)
        with torch.no_grad():
            y = m(torch.rand(2, 3, 32, 32))
        for p in probes + h:
            p.remove()
        first_reduced = next((i for i, si, _so in seen if si != 32), None)
        out[loc] = {"target_block": BLOCK_TARGET[loc],
                    "block_input_sizes": [si for _i, si, _o in seen],
                    "first_block_seeing_reduced_input": first_reduced,
                    "logits": list(y.shape)}
    # the input site reduces the image itself
    ctrl = ResolutionController(O_REF, "input", "Rprog")
    ctrl.set_epoch(0)
    out["input"] = {"input_resolution": ctrl.input_resolution(),
                    "internal_hooks": len(attach_resolution(model(), ctrl))}
    return out


def check_operators():
    out = {}
    g = torch.Generator().manual_seed(0)
    for op in OPERATORS:
        rec = {}
        for r in R:
            x = torch.rand(2, 16, 32, 32, generator=g, requires_grad=True)
            y = reduce_with(x, r, op)
            y.pow(2).sum().backward()
            rec["r%d" % r] = {
                "shape": list(y.shape),
                "finite": bool(torch.isfinite(y).all()),
                "grad_finite": bool(torch.isfinite(x.grad).all()),
                "grad_norm": float(x.grad.norm()),
                "grad_nonzero": float(x.grad.abs().sum()) > 0,
            }
        # exact bypass: the same tensor object comes back at r == H
        t = torch.rand(2, 16, 32, 32, generator=g)
        rec["bypass_is_same_object"] = reduce_with(t, 32, op) is t
        # constant map -> constant output, finite gradient
        c = torch.full((2, 16, 32, 32), 0.37, requires_grad=True)
        yc = reduce_with(c, 16, op)
        yc.sum().backward()
        rec["constant_map"] = {
            "output_std": float(yc.std()),
            "max_abs_dev_from_0.37": float((yc - 0.37).abs().max()),
            "grad_finite": bool(torch.isfinite(c.grad).all()),
        }
        out[op] = rec
    return out


def check_maxblur_reduces_to_max():
    g = torch.Generator().manual_seed(1)
    x = torch.rand(3, 8, 32, 32, generator=g)
    out = {}
    for r in R:
        d = float((op_maxblur(x, r, blur=False) - op_max(x, r)).abs().max())
        out["r%d" % r] = {"max_abs_diff_vs_adaptive_max": d, "identical": d == 0.0,
                          "window_widths": sorted(set(window_widths(32, r))),
                          "starts_head": window_starts(32, r)[:6]}
    return out


def check_quadratic():
    """Solution maps against a direct constrained solve on random data."""
    out = {}
    for kind in ("l2", "hminus1"):
        rec = {}
        for r in R:
            n = 32
            U = _upsample_matrix(n, r)
            Q = (torch.eye(n * n, dtype=torch.float64) if kind == "l2"
                 else laplacian_pinv(n))
            M = solution_map(n, r, kind, dtype=torch.float64)
            g = torch.Generator().manual_seed(7)
            h = torch.rand(n * n, dtype=torch.float64, generator=g)
            z = M @ h
            # direct KKT solve for this single h
            m = r * r
            one = torch.ones(n * n, 1, dtype=torch.float64)
            K = torch.zeros(m + 1, m + 1, dtype=torch.float64)
            K[:m, :m] = U.t() @ Q @ U
            K[:m, m:] = U.t() @ one
            K[m:, :m] = (U.t() @ one).t()
            rhs = torch.zeros(m + 1, dtype=torch.float64)
            rhs[:m] = U.t() @ Q @ h
            rhs[m] = one.t() @ h
            z_direct = torch.linalg.solve(K, rhs)[:m]
            resid = float((U @ z - h).sum())               # the mean constraint
            KK = K[:m, :m]
            # For hminus1 the normal matrix is singular BY CONSTRUCTION: L^+
            # annihilates constants and U maps the constant coefficient vector
            # to a constant image, so U^T L^+ U has exactly one null direction.
            # The mean constraint is what removes it, so the KKT matrix is the
            # object whose conditioning matters.
            rec["r%d" % r] = {
                "max_abs_diff_map_vs_direct": float((z - z_direct).abs().max()),
                "mean_constraint_residual": resid,
                "kkt_rank": int(torch.linalg.matrix_rank(K)),
                "kkt_size": m + 1,
                "kkt_cond": float(torch.linalg.cond(K)),
                "normal_matrix_cond": float(torch.linalg.cond(KK)),
                "normal_matrix_rank": int(torch.linalg.matrix_rank(KK)),
            }
        out[kind] = rec
    # H^-1 uses the pseudoinverse: L^+ must annihilate constants
    L = _laplacian(32)
    Lp = laplacian_pinv(32)
    ones = torch.ones(32 * 32, dtype=torch.float64)
    out["laplacian"] = {"L_times_ones_max": float((L @ ones).abs().max()),
                        "Lpinv_times_ones_max": float((Lp @ ones).abs().max()),
                        "symmetric": bool(torch.allclose(L, L.t()))}
    return out


def check_perceptual():
    g = torch.Generator().manual_seed(3)
    x = torch.rand(2, 4, 32, 32, generator=g)
    # Compare the algebra in float64: at float32 the convolution and the loop
    # accumulate differently and agree only to ~2e-5, which would say nothing
    # about whether the two formulations are the same operator.
    xd = x.double()
    fast = perceptual_factor2(xd)
    ref = perceptual_reference(xd)
    xi = x.clone().requires_grad_(True)
    perceptual_factor2(xi).pow(2).sum().backward()
    flat = torch.full((2, 4, 32, 32), 0.5, requires_grad=True)
    yf = perceptual_factor2(flat)
    yf.sum().backward()
    out = {
        "fast_vs_naive_max_abs": float((fast - ref).abs().max()),
        "agree": bool(torch.allclose(fast, ref, atol=1e-10)),
        "float32_fast_vs_naive_max_abs": float((perceptual_factor2(x) - perceptual_reference(x)).abs().max()),
        "shape_32_to_16": list(fast.shape),
        "shape_32_to_24": list(reduce_with(x, 24, "perceptual").shape),
        "grad_finite": bool(torch.isfinite(xi.grad).all()),
        "grad_nonzero": float(xi.grad.abs().sum()) > 0,
        "flat_input_output_max_dev": float((yf - 0.5).abs().max()),
        "flat_input_grad_finite": bool(torch.isfinite(flat.grad).all()),
    }
    return out


def check_softpool_temperature():
    """tau must be differentiated, not detached: scaling h must change the mix."""
    g = torch.Generator().manual_seed(5)
    x = torch.rand(1, 3, 32, 32, generator=g)
    xi = x.clone().requires_grad_(True)
    y = reduce_with(xi, 16, "softpool")
    y.pow(2).sum().backward()
    live = xi.grad.clone()
    # recompute with tau detached, for contrast
    from continuation.resolution_ops import window_starts as ws
    xj = x.clone().requires_grad_(True)
    idx = torch.as_tensor(ws(32, 16), dtype=torch.long)
    rows = torch.stack([xj.index_select(-2, idx + o) for o in (0, 1)], dim=-3)
    win = torch.stack([rows.index_select(-1, idx + o) for o in (0, 1)], dim=-1)
    win = win.reshape(1, 3, 2, 16, 16, 2).permute(0, 1, 3, 4, 2, 5).reshape(1, 3, 16, 16, 4)
    mean = xj.mean(dim=(-2, -1), keepdim=True)
    tau = (0.5 * torch.sqrt(((xj - mean) ** 2).mean(dim=(-2, -1), keepdim=True)
                            + 1e-12)).detach().unsqueeze(-1)
    z = win / tau
    z = z - z.amax(dim=-1, keepdim=True)
    (torch.softmax(z, -1) * win).sum(-1).pow(2).sum().backward()
    detached = xj.grad.clone()
    # a constant map must come back constant despite tau -> 0.5*sqrt(eps)
    c = torch.full((1, 3, 32, 32), -0.8, requires_grad=True)
    yc = reduce_with(c, 16, "softpool")
    yc.sum().backward()
    return {"grad_norm_live_tau": float(live.norm()),
            "grad_norm_detached_tau": float(detached.norm()),
            "max_abs_difference": float((live - detached).abs().max()),
            "tau_is_detached_in_implementation":
                bool((live - detached).abs().max() == 0),
            "constant_map_output_max_dev": float((yc + 0.8).abs().max()),
            "constant_map_grad_finite": bool(torch.isfinite(c.grad).all())}


def check_paths():
    out = {}
    for name, table in PATHS.items():
        out[name] = {"len": len(table), "e0": table[0], "e5": table[5],
                     "e6": table[6], "e8": table[8], "e9": table[9],
                     "e11": table[11], "e12": table[12], "e17": table[17],
                     "e18": table[18], "e29": table[29]}
    prog, rev = PATHS["Rprog"], PATHS["Rreverse"]
    out["checks"] = {
        "rprog_16_6_24_6_then_32": prog[:6] == [16] * 6 and prog[6:12] == [24] * 6
        and set(prog[12:]) == {32},
        "rlate_9_9_12": (PATHS["Rlate"][:9] == [16] * 9
                         and PATHS["Rlate"][9:18] == [24] * 9
                         and set(PATHS["Rlate"][18:]) == {32}),
        "rreverse_same_exposure_as_rprog": sorted(rev) == sorted(prog),
        "fixed16_never_32": set(PATHS["fixed16"]) == {16},
        "fixed24_never_32": set(PATHS["fixed24"]) == {24},
        "all_progressive_end_at_32": all(
            PATHS[p][29] == 32 for p in ("Rprog", "Rgentle", "Rreverse", "Rlate")),
    }
    return out


def check_bypass_bitwise():
    """A progressive arm at r=32 must reproduce the plain network bitwise."""
    x = torch.rand(4, 3, 32, 32)
    ref = model().eval()
    with torch.no_grad():
        y0 = ref(x)
    out = {}
    for op in OPERATORS:
        for loc in ("stem", "D0", "D1", "D2"):
            m = model().eval()
            ctrl = ResolutionController(op, loc, "Rprog")
            ctrl.set_epoch(29)                                # r = 32
            h = attach_resolution(m, ctrl)
            with torch.no_grad():
                y = m(x)
            for p in h:
                p.remove()
            d = float((y - y0).abs().max())
            out["%s@%s" % (op, loc)] = {"max_abs_diff": d, "bitwise": d == 0.0}
    return out


def main():
    report = {
        "o_ref": O_REF,
        "locations": list(LOCATIONS),
        "block_targets": BLOCK_TARGET,
        "sites": check_sites(),
        "operators": check_operators(),
        "maxblur_reduces_to_max": check_maxblur_reduces_to_max(),
        "quadratic": check_quadratic(),
        "perceptual": check_perceptual(),
        "softpool_temperature": check_softpool_temperature(),
        "paths": check_paths(),
        "bypass_bitwise": check_bypass_bitwise(),
    }
    ok = (all(v["bitwise"] for v in report["bypass_bitwise"].values())
          and all(v["identical"] for v in report["maxblur_reduces_to_max"].values())
          and report["perceptual"]["agree"]
          and report["perceptual"]["grad_finite"]
          and not report["softpool_temperature"]["tau_is_detached_in_implementation"]
          and all(report["operators"][o]["bypass_is_same_object"] for o in OPERATORS)
          and all(report["operators"][o]["r%d" % r]["grad_finite"]
                  for o in OPERATORS for r in R)
          and all(v for v in report["paths"]["checks"].values())
          and all(report["quadratic"][k]["r%d" % r]["max_abs_diff_map_vs_direct"] < 1e-8
                  for k in ("l2", "hminus1") for r in R))
    report["all_ok"] = ok
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))

    print("O_ref = %s" % O_REF)
    for loc, v in report["sites"].items():
        print("  site %-6s %s" % (loc, json.dumps(v)))
    print("bypass bitwise at r=32 (all op x loc): %s"
          % all(v["bitwise"] for v in report["bypass_bitwise"].values()))
    print("maxblur without blur == adaptive max: %s"
          % {k: v["identical"] for k, v in report["maxblur_reduces_to_max"].items()})
    for k in ("l2", "hminus1"):
        for r in R:
            q = report["quadratic"][k]["r%d" % r]
            print("  %-8s r=%d  |map-direct|=%.2e  mean-resid=%.2e  kkt_cond=%.2e  rank %d/%d"
                  % (k, r, q["max_abs_diff_map_vs_direct"],
                     abs(q["mean_constraint_residual"]), q["kkt_cond"],
                     q["kkt_rank"], q["kkt_size"]))
    print("perceptual fast vs naive: %.2e (agree=%s)"
          % (report["perceptual"]["fast_vs_naive_max_abs"],
             report["perceptual"]["agree"]))
    print("softpool tau differentiated: %s"
          % (not report["softpool_temperature"]["tau_is_detached_in_implementation"]))
    print("paths: %s" % json.dumps(report["paths"]["checks"]))
    print("ALL OK: %s  -> %s" % (ok, OUT))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
