"""Focused checks for the progressive-resolution arms.

Covers the three scheduled input resolutions on the verified ResNet-20 +
BatchNorm model:

1. stage feature-map sizes at 16 / 24 / 32, and shortcut outputs matching their
   main path (option-A shortcuts must line up for the residual addition);
2. a genuinely global pool: a 64-dimensional classifier input at every
   resolution, from the same parameters;
3. finite forward and backward at each resolution, with and without the
   internal Gaussian;
4. exact identity of the target input path: ``r = 32`` returns the original
   tensor object, no resize operation;
5. exact Gaussian bypass at ``sigma = 0``, bitwise, including at the 4x4
   stage-3 maps produced by 16x16 inputs (the radius-4 explicit reflected
   padding case);
6. every boundary of the frozen resolution and effective-sigma schedules.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from continuation.config import ModelConfig
from continuation.models import build_model, count_parameters
from scripts import continuation_driver as D
from scripts.job_progressive_resolution import (RES_BY_EPOCH, SIGMA_BY_EPOCH,
                                                REFERENCE_SIGMA, resolution_at)

OUT = Path("results/progressive_resolution_verification.json")
EXPECTED = {16: (16, 8, 4), 24: (24, 12, 6), 32: (32, 16, 8)}


def stage_sizes(model, x):
    """Spatial size of each stage's output, plus every shortcut output."""
    sizes, shortcuts = [], []
    hooks = []
    blocks = list(model.blocks)
    for i, b in enumerate(blocks):
        hooks.append(b.register_forward_hook(
            lambda _m, _i, out, i=i: sizes.append((i, int(out.shape[-1]),
                                                   int(out.shape[1])))))
        hooks.append(b.shortcut.register_forward_hook(
            lambda _m, _i, out, i=i: shortcuts.append((i, int(out.shape[-1]),
                                                       int(out.shape[1])))))
    pooled = {}
    h = model.fc.register_forward_hook(
        lambda _m, inp, _o: pooled.update(dim=int(inp[0].shape[1])))
    with torch.no_grad():
        model(x)
    for k in hooks + [h]:
        k.remove()
    return sizes, shortcuts, pooled["dim"]


def main():
    torch.manual_seed(0)
    report = {"expected_stage_sizes": {str(k): list(v) for k, v in EXPECTED.items()},
              "resolutions": {}}

    model = build_model(ModelConfig(arch="resnet20_bn_cifar"), 10, seed=0)
    report["params"] = count_parameters(model)

    for r in (16, 24, 32):
        x = torch.rand(4, 3, r, r)
        sizes, shortcuts, pooled = stage_sizes(model, x)
        # blocks 0-2 stage 1, 3-5 stage 2, 6-8 stage 3
        got = (sizes[0][1], sizes[3][1], sizes[6][1])
        # each shortcut output must match its block's output for the addition
        sc_ok = all(s[1] == z[1] and s[2] == z[2] for s, z in zip(shortcuts, sizes))
        rr = {"stage_sizes": list(got), "matches_expected": got == EXPECTED[r],
              "pooled_dim": pooled, "pooled_is_64": pooled == 64,
              "shortcut_shapes_match_main_path": sc_ok,
              "n_shortcut_outputs": len(shortcuts)}

        # resize identity at the target resolution
        orig = torch.rand(4, 3, 32, 32)
        rr["resize_is_identity_object_at_32"] = D.resize_to(orig, 32) is orig
        rr["resize_shape"] = list(D.resize_to(orig, r).shape)

        # forward/backward with and without the internal Gaussian
        for sig, name in ((0.5, "gaussian_sigma0.5"), (0.0, "gaussian_bypass")):
            m = build_model(ModelConfig(arch="resnet20_bn_cifar"), 10, seed=0).train()
            ctrl = D.Controller("gaussian")
            ctrl.value = sig
            handles = D.attach(m, ctrl)
            xi = torch.rand(4, 3, r, r, requires_grad=True)
            out = m(xi)
            out.pow(2).sum().backward()
            rr[name] = {
                "n_hooks": len(handles),
                "out_shape": list(out.shape),
                "out_finite": bool(torch.isfinite(out).all()),
                "grad_finite": bool(torch.isfinite(xi.grad).all()),
                "grad_norm": float(xi.grad.norm()),
            }
            for h in handles:
                h.remove()

        # exact Gaussian bypass at sigma = 0, bitwise, through the whole network
        m = build_model(ModelConfig(arch="resnet20_bn_cifar"), 10, seed=0).eval()
        ctrl = D.Controller("gaussian")
        handles = D.attach(m, ctrl)
        xz = torch.rand(4, 3, r, r)
        with torch.no_grad():
            ctrl.value = 0.0
            a = m(xz)
            ctrl.bypass_all = True
            b = m(xz)
        rr["sigma0_bypass_max_abs_diff"] = float((a - b).abs().max())
        rr["sigma0_is_exact_identity"] = bool((a - b).abs().max() == 0)
        for h in handles:
            h.remove()

        # the smallest stage-3 map at this resolution, filtered directly
        smallest = EXPECTED[r][2]
        g = D.GaussianSmoothing(sigma_max=1.0, truncate=4.0)
        t = torch.rand(2, 64, smallest, smallest)
        y = g(t, 1.0)
        rr["stage3_map"] = {
            "size": smallest, "kernel_size": g.kernel_size, "radius": g.radius,
            "filter_shape_preserved": list(y.shape) == list(t.shape),
            "finite": bool(torch.isfinite(y).all()),
            "sigma0_bitwise_identity": bool((g(t, 0.0) - t).abs().max() == 0),
        }
        report["resolutions"][str(r)] = rr

    # schedule boundaries
    sched = []
    for e in range(30):
        sched.append({"epoch": e, "resolution": RES_BY_EPOCH[e],
                      "reference_sigma": REFERENCE_SIGMA[e],
                      "effective_sigma": SIGMA_BY_EPOCH[e]})
    report["schedule"] = sched
    report["schedule_checks"] = {
        "resolution_matches_formula": all(RES_BY_EPOCH[e] == resolution_at(e)
                                          for e in range(30)),
        "boundaries": {"e5": RES_BY_EPOCH[5], "e6": RES_BY_EPOCH[6],
                       "e11": RES_BY_EPOCH[11], "e12": RES_BY_EPOCH[12]},
        "effective_equals_scaled_reference": all(
            abs(SIGMA_BY_EPOCH[e] - RES_BY_EPOCH[e] / 32.0 * REFERENCE_SIGMA[e]) < 1e-12
            for e in range(30)),
        "final_nine_epochs_at_target": all(
            RES_BY_EPOCH[e] == 32 and SIGMA_BY_EPOCH[e] == 0.0 for e in range(21, 30)),
        "effective_sigma_is_not_monotone": any(
            SIGMA_BY_EPOCH[e + 1] > SIGMA_BY_EPOCH[e] for e in range(29)),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))

    for r, v in report["resolutions"].items():
        print("r=%-3s stages=%-12s expected=%-5s pooled=%d(64:%s) shortcuts_ok=%s "
              "sigma0_exact=%s stage3=%dx%d ok=%s"
              % (r, v["stage_sizes"], v["matches_expected"], v["pooled_dim"],
                 v["pooled_is_64"], v["shortcut_shapes_match_main_path"],
                 v["sigma0_is_exact_identity"], v["stage3_map"]["size"],
                 v["stage3_map"]["size"], v["stage3_map"]["filter_shape_preserved"]))
    print("schedule:", json.dumps(report["schedule_checks"]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
