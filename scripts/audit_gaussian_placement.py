r"""Audit the executed forward/backward pass and the Gaussian insertion points.

Traces a real forward pass (functional ops and residual additions included),
verifies the 19 insertions land exactly where intended, checks the filter's
properties, confirms gradients flow through it, and checks that the k>=600
bypass reproduces the plain model bitwise in both outputs and gradients.

    py scripts/audit_gaussian_placement.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from continuation.config import ModelConfig  # noqa: E402
from continuation.models import build_model, count_parameters  # noqa: E402
from continuation.models.resnet_gn import BasicBlock, _PadShortcut  # noqa: E402
from continuation.transforms.gaussian import GaussianSmoothing  # noqa: E402
from scripts.continuation_driver import Controller, attach  # noqa: E402

OUT = {}


def named(model):
    return {id(m): n for n, m in model.named_modules()}


def audit_structure(model):
    names = named(model)
    convs = [(names[id(m)], m) for m in model.modules() if isinstance(m, nn.Conv2d)]
    gns = [(names[id(m)], m) for m in model.modules() if isinstance(m, nn.GroupNorm)]
    lin = [(names[id(m)], m) for m in model.modules() if isinstance(m, nn.Linear)]
    conv3 = [(n, m) for n, m in convs if m.kernel_size == (3, 3)]
    blocks = [m for m in model.modules() if isinstance(m, BasicBlock)]
    return {
        "trainable_params": count_parameters(model)["trainable"],
        "n_conv": len(convs), "n_conv3x3": len(conv3),
        "n_groupnorm": len(gns), "n_linear": len(lin),
        "n_basicblocks": len(blocks),
        "conv3x3_names": [n for n, _ in conv3],
        "non_3x3_convs": [n for n, m in convs if m.kernel_size != (3, 3)],
        "groupnorm_channels_per_group": sorted({m.num_channels // m.num_groups
                                                for _, m in gns}),
        "shortcut_types": sorted({type(b.shortcut).__name__ for b in blocks}),
        "linear": {n: [m.in_features, m.out_features] for n, m in lin},
        "buffers": [n for n, _ in model.named_buffers()],
    }


def trace(model, controller, x):
    """Record executed module order, shapes, and where filtering fired."""
    names = named(model)
    events = []

    def pre(mod, inp):
        events.append({"event": "enter", "module": names[id(mod)],
                       "type": type(mod).__name__,
                       "in_shape": list(inp[0].shape) if torch.is_tensor(inp[0]) else None})

    def post(mod, inp, out):
        events.append({"event": "exit", "module": names[id(mod)],
                       "type": type(mod).__name__,
                       "out_shape": list(out.shape) if torch.is_tensor(out) else None})

    handles = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.GroupNorm, nn.Linear, _PadShortcut, BasicBlock)):
            handles.append(m.register_forward_pre_hook(pre))
            handles.append(m.register_forward_hook(post))

    fired = []
    orig = controller.__call__

    def spy(t):
        fired.append({"shape": list(t.shape), "param": controller.value,
                      "bypass_all": controller.bypass_all})
        return orig(t)

    controller_call = controller.__class__.__call__
    controller.__class__.__call__ = lambda self, t: spy(t) if self is controller else controller_call(self, t)
    try:
        y = model(x)
    finally:
        controller.__class__.__call__ = controller_call
        for h in handles:
            h.remove()
    return events, fired, y


def main():
    torch.manual_seed(0)
    model = build_model(ModelConfig(), 10, seed=0).eval()
    OUT["structure"] = audit_structure(model)

    ctrl = Controller("gaussian", cont_end=600)
    ctrl.set_update(0)                              # sigma = 1.0
    handles = attach(model, ctrl)
    x = torch.rand(2, 3, 32, 32)
    events, fired, y = trace(model, ctrl, x)
    for h in handles:
        h.remove()

    # Compact ordered trace: conv -> filter -> norm -> activation structure
    seq = []
    for e in events:
        if e["event"] == "exit" and e["type"] in ("Conv2d", "GroupNorm", "Linear", "_PadShortcut"):
            seq.append("%s[%s]%s" % (e["module"], e["type"][:4],
                                     "x".join(str(v) for v in e["out_shape"][1:])))
    OUT["executed_sequence"] = seq
    OUT["n_filter_calls"] = len(fired)
    OUT["filter_call_shapes"] = [f["shape"][1:] for f in fired]
    OUT["logits_shape"] = list(y.shape)

    # --- filter properties ---
    g = GaussianSmoothing(sigma_max=1.0, truncate=4.0)
    k = g.kernel(1.0, torch.float64)
    t = torch.rand(2, 7, 16, 16, dtype=torch.float64)     # 7 channels: not 3, not RGB
    ft = g(t, 1.0)
    delta = torch.zeros(1, 3, 16, 16, dtype=torch.float64)
    delta[0, 1, 8, 8] = 1.0
    fd = g(delta, 1.0)
    OUT["filter"] = {
        "kernel_size": g.kernel_size, "radius": g.radius, "padding": g.padding,
        "taps_sum": float(k.sum()), "taps_symmetric": bool(torch.allclose(k, k.flip(0))),
        "separable_depthwise_groups": "groups=C (per channel)",
        "preserves_shape": list(ft.shape) == list(t.shape),
        "channelwise_no_leak": [float(fd[0, 0].abs().max()), float(fd[0, 2].abs().max())],
        "mass_conserved": float(fd[0, 1].sum()),
        "identity_at_sigma0": bool(g(t, 0.0) is t),
    }

    # --- gradients flow through the filter to earlier weights ---
    m2 = build_model(ModelConfig(), 10, seed=0).train()
    c2 = Controller("gaussian", cont_end=600)
    c2.set_update(0)
    hs = attach(m2, c2)
    xx = torch.rand(4, 3, 32, 32)
    F.cross_entropy(m2(xx), torch.randint(0, 10, (4,))).backward()
    stem_grad = float(m2.conv1.weight.grad.abs().max())
    last_grad = float(m2.blocks[-1].conv2.weight.grad.abs().max())
    for h in hs:
        h.remove()
    OUT["gradient_flow"] = {"stem_conv_grad_absmax": stem_grad,
                            "last_block_conv_grad_absmax": last_grad,
                            "all_convs_have_grads": all(
                                m.weight.grad is not None for m in m2.modules()
                                if isinstance(m, nn.Conv2d))}

    # --- schedule ---
    c3 = Controller("gaussian", cont_end=600)
    sched = {k_: c3.set_update(k_) for k_ in (0, 1, 300, 599, 600, 601, 1199)}
    OUT["schedule"] = {"sigma_at": sched,
                       "formula": "sigma_k = max(1 - k/600, 0), k = 0..1199",
                       "identical_across_insertion_points":
                           "single Controller instance shared by all 19 hooks",
                       "fixed_within_accumulated_batch":
                           "set_update(k) called once per optimizer update, before the "
                           "microbatch loop (continuation_driver.train_run)"}

    # --- k >= 600 bypass must reproduce the plain model exactly ---
    ma = build_model(ModelConfig(), 10, seed=0).train()
    mb = build_model(ModelConfig(), 10, seed=0).train()
    mb.load_state_dict(ma.state_dict())
    ca = Controller("none", cont_end=600)
    cb = Controller("gaussian", cont_end=600)
    cb.set_update(600)                                # sigma = 0 -> exact bypass
    ha, hb = attach(ma, ca), attach(mb, cb)
    xs = torch.rand(4, 3, 32, 32)
    ys = torch.randint(0, 10, (4,))
    oa, ob = ma(xs), mb(xs)
    F.cross_entropy(oa, ys).backward()
    F.cross_entropy(ob, ys).backward()
    gmax = max(float((pa.grad - pb.grad).abs().max())
               for pa, pb in zip(ma.parameters(), mb.parameters()))
    for h in ha + hb:
        h.remove()
    OUT["bypass_equivalence"] = {
        "sigma_at_600": cb.value,
        "logits_bitwise_identical": bool(torch.equal(oa, ob)),
        "max_abs_logit_diff": float((oa - ob).abs().max()),
        "max_abs_grad_diff": gmax}

    p = ROOT / "results" / "gaussian_placement_audit.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(OUT, indent=2))

    s = OUT["structure"]
    print("params %d | conv %d (3x3: %d, other: %s) | GN %d @ %s ch/group | linear %s"
          % (s["trainable_params"], s["n_conv"], s["n_conv3x3"], s["non_3x3_convs"],
             s["n_groupnorm"], s["groupnorm_channels_per_group"], s["linear"]))
    print("blocks %d | shortcuts %s | buffers %s"
          % (s["n_basicblocks"], s["shortcut_types"], s["buffers"] or "none"))
    print("filter calls: %d  shapes: %s"
          % (OUT["n_filter_calls"],
             sorted({tuple(v) for v in OUT["filter_call_shapes"]})))
    print("filter: %s" % json.dumps(OUT["filter"]))
    print("schedule: %s" % sched)
    print("bypass: %s" % json.dumps(OUT["bypass_equivalence"]))
    print("grad flow: %s" % json.dumps(OUT["gradient_flow"]))
    print("wrote %s" % p)


if __name__ == "__main__":
    main()
