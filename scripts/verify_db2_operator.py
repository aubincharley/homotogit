"""Focused checks on the db2 operator at the ResNet-20 activation shapes.

Reuses the existing implementation unchanged; nothing here is a new transform.
Checks, at the three stage shapes (16,32,32) / (32,16,16) / (64,8,8):

1. reconstruction ``T_1 = I`` with ``bypass_identity=False`` (round-trip exercised);
2. adjointness of synthesis w.r.t. analysis, ``<W z, g> == <z, W* g>``;
3. shape preservation at every scheduled ``s``;
4. finite forward and backward, and gradients that actually flow;
5. a zero-detail input (spatially constant per channel) for RMS stability;
6. that the RMS-dependent threshold is **differentiable**, i.e. not detached.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import torch

from continuation.transforms.wavelet import (mirror_extend, swt2_analysis,
                                             swt2_synthesis, wavelet_shrink)

SHAPES = [(2, 16, 32, 32), (2, 32, 16, 16), (2, 64, 8, 8)]
SCHEDULE_S = [0.00, 0.25, 0.50, 0.70, 0.85, 0.95, 0.99, 1.00]
OUT = Path("results/db2_operator_verification.json")


def main():
    g = torch.Generator().manual_seed(0)
    report = {"wavelet": "db2", "levels": 2, "eps": 1e-12, "shapes": {}}

    for shape in SHAPES:
        key = "x".join(str(v) for v in shape[1:])
        r = {}
        x = torch.randn(*shape, generator=g)

        # 1. reconstruction at s=1 with the bypass disabled
        y = wavelet_shrink(x, 1.0, "db2", bypass_identity=False)
        r["recon_s1_max_abs_err"] = float((y - x).abs().max())
        r["recon_s1_rel_err"] = float((y - x).norm() / x.norm())
        r["bypass_is_exact_identity"] = bool(
            (wavelet_shrink(x, 1.0, "db2", bypass_identity=True) - x).abs().max() == 0)

        # 2. adjointness on the extended domain
        z = mirror_extend(x)
        a, det = swt2_analysis(z, "db2", 2)
        ga = torch.randn_like(a)
        gd = [{k: torch.randn_like(v) for k, v in d.items()} for d in det]
        lhs = float((a * ga).sum() + sum((det[j][k] * gd[j][k]).sum()
                                         for j in range(2) for k in gd[j]))
        rhs = float((z * swt2_synthesis(ga, gd, "db2")).sum())
        r["adjoint_lhs"], r["adjoint_rhs"] = lhs, rhs
        r["adjoint_rel_err"] = abs(lhs - rhs) / max(abs(lhs), 1e-30)

        # 3/4. shape, finiteness, gradient flow at every scheduled s
        per_s = {}
        for s in SCHEDULE_S:
            xi = x.clone().requires_grad_(True)
            out = wavelet_shrink(xi, s, "db2", bypass_identity=(s == 1.0))
            out.pow(2).sum().backward()
            per_s["%.2f" % s] = {
                "shape_ok": tuple(out.shape) == tuple(x.shape),
                "out_finite": bool(torch.isfinite(out).all()),
                "grad_finite": bool(torch.isfinite(xi.grad).all()),
                "grad_norm": float(xi.grad.norm()),
                "rel_change_from_input": float((out - x).norm() / x.norm()),
            }
        r["by_s"] = per_s

        # 5. zero-detail input: constant per channel -> all detail bands are 0
        c = torch.arange(shape[1], dtype=torch.float32).view(1, -1, 1, 1)
        flat = c.expand(shape).clone().requires_grad_(True)
        o = wavelet_shrink(flat, 0.0, "db2", bypass_identity=False)
        o.pow(2).sum().backward()
        r["zero_detail"] = {
            "out_finite": bool(torch.isfinite(o).all()),
            "grad_finite": bool(torch.isfinite(flat.grad).all()),
            "max_abs_err_vs_input": float((o - flat.detach()).abs().max()),
            "grad_norm": float(flat.grad.norm()),
        }

        report["shapes"][key] = r

    # 6. thresholds differentiable through nu: perturbing a band's magnitude
    #    changes lambda, so a detached RMS would give a different gradient.
    #    Compare against an explicitly detached-nu recomputation.
    x = torch.randn(1, 8, 16, 16, generator=g)
    xi = x.clone().requires_grad_(True)
    wavelet_shrink(xi, 0.5, "db2", bypass_identity=False).pow(2).sum().backward()
    live = xi.grad.clone()

    from continuation.transforms import wavelet as W
    orig = W.torch.sqrt
    xj = x.clone().requires_grad_(True)
    z = mirror_extend(xj)
    a, det = swt2_analysis(z, "db2", 2)
    sh = []
    for j, d in enumerate(det, start=1):
        nu = {k: torch.sqrt((v * v).mean(dim=(-2, -1), keepdim=True) + 1e-12).detach()
              for k, v in d.items()}
        sh.append({k: torch.sign(v) * torch.clamp(
            v.abs() - 4.0 * (1 - 0.5) * 2.0 ** (1 - j) * nu[k], min=0.0)
            for k, v in d.items()})
    y = swt2_synthesis(a, sh, "db2")[..., :16, :16]
    y.pow(2).sum().backward()
    detached = xj.grad.clone()
    report["threshold_differentiable"] = {
        "grad_norm_live_nu": float(live.norm()),
        "grad_norm_detached_nu": float(detached.norm()),
        "max_abs_difference": float((live - detached).abs().max()),
        "rms_is_detached_in_implementation": bool((live - detached).abs().max() == 0),
    }
    report["coarse_only_claim"] = {
        "status": "unverified",
        "note": "the '~6% from coarse-only reconstruction at s=0' figure has no "
                "recorded measurement or normalization in the repository; it is "
                "not used by training and is left unverified here.",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
