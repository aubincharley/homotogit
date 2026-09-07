"""The drift estimator: how far the empirical NTK has moved from its value at w0.

    d_t = ||K_t - K_0||_F / ||K_0||_F          fed to the controller
    a_t = <K_t, K_0>_F / (||K_t|| ||K_0||)     logged alongside

K is estimated by sketching: R random tangents v_r, one forward-mode JVP each,
and K_hat = M M^T / R with M's columns the flattened J v_r. The estimator's
whole viability rests on one rule.

COMMON RANDOM NUMBERS. The v_r are drawn from a dedicated, fixed seed and
regenerated identically at every probe, in every seed, on every arm. With fresh
tangents each probe, ||K_hat_t - K_hat_0||_F carries estimator noise of order
R^-1/2 ||K||_F -- about 35% at R=8, which is far larger than the early drift the
controller is trying to steer by, and the loop would spend its authority chasing
sampling noise. With the tangents frozen, that noise is common to both terms and
very largely cancels, so the difference tracks true drift. Regenerating from a
seed rather than storing them is exact and costs no memory: R=8 tangents over
ResNet-18 would be ~350 MB.

Nothing here materialises K. At m=256 and 10 classes K is 2560x2560, and every
quantity above follows from the RxR blocks instead:

    ||AA^T - BB^T||_F^2 = ||A^T A||_F^2 - 2 ||A^T B||_F^2 + ||B^T B||_F^2
    <AA^T, BB^T>_F      = ||A^T B||_F^2

The block algebra runs in float64, and that is not fussiness. At w = w0 the true
d is exactly zero and the expression above is a difference of nearly equal
positive numbers; in float32 the cancellation leaves a residual around 1e-7,
i.e. a spurious d of ~3e-4, which is the same size as the drift over the first
few hundred steps.
"""
import contextlib

import torch
from torch.func import functional_call, jvp
from torch.nn.modules.batchnorm import _BatchNorm

from cifarbase.anchor import CONTAINER

# Fixed, and deliberately not derived from the run seed: every arm and every seed
# is measured along the same tangent directions. That is common random numbers,
# and it is a large variance reduction for free.
TANGENT_SEED = 20260907


@contextlib.contextmanager
def probe_context(model, deterministic=False):
    """Eval mode, BatchNorm buffers provably untouched, previous mode restored.

    eval() is the right choice and it does mean the probe measures a slightly
    different function than training sees: BatchNorm normalises by running
    statistics rather than by the batch. That is intended -- it makes K_t a
    deterministic function of w alone, which is what the controller needs -- but
    it has to be consistent, so every probe goes through here.

    The running statistics are snapshotted and checked on the way out. A probe
    that quietly updated them would corrupt training and every later probe at
    once, and nothing else in the run would look wrong.
    """
    was_training = model.training
    bns = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    momenta = [m.momentum for m in bns]
    snapshot = [(m.running_mean.detach().clone() if m.running_mean is not None else None,
                 m.running_var.detach().clone() if m.running_var is not None else None)
                for m in bns]
    cudnn = torch.backends.cudnn
    was_deterministic, was_benchmark = cudnn.deterministic, cudnn.benchmark

    model.eval()
    for module in bns:
        module.momentum = 0.0            # belt and braces: eval() alone suffices
    if deterministic:
        cudnn.deterministic, cudnn.benchmark = True, False
    try:
        yield
    finally:
        if deterministic:
            cudnn.deterministic, cudnn.benchmark = was_deterministic, was_benchmark
        for module, momentum in zip(bns, momenta, strict=True):
            module.momentum = momentum
        model.train(was_training)
        for module, (mean, var) in zip(bns, snapshot, strict=True):
            for live, kept, what in ((module.running_mean, mean, "running_mean"),
                                     (module.running_var, var, "running_var")):
                if kept is not None and not torch.equal(live, kept):
                    raise SystemExit(
                        f"!! a probe moved BatchNorm {what}. The probe must be a "
                        f"read-only measurement; every later probe and the rest of "
                        f"training are now measuring a polluted model.")


def trainable(model):
    """The parameters the kernel is taken with respect to, in a fixed order.

    Sorted by name so the tangent stream is a property of the architecture and
    not of module registration order -- otherwise a refactor that reorders
    modules would silently change every d_t in the project.
    """
    return {name: param for name, param in sorted(model.named_parameters())
            if param.requires_grad and not name.startswith(CONTAINER + ".")}


def tangents(params, count, seed=TANGENT_SEED):
    """`count` Rademacher tangent dicts, regenerated identically on every call.

    A generator yielding one at a time: the JVP consumes them one by one, so only
    one full-size tangent is ever resident. Re-seeded here, so the r-th tangent
    is the same tensor at every probe for the life of the project.
    """
    device = next(iter(params.values())).device
    generator = torch.Generator(device=device).manual_seed(int(seed))

    def stream():
        for _ in range(count):
            yield {name: ((torch.rand(param.shape, generator=generator,
                                      device=device, dtype=param.dtype) < 0.5)
                          .to(param.dtype) * 2 - 1)
                   for name, param in params.items()}
    return stream()


@torch.no_grad()
def ntk_sketch(model, probe_x, count, seed=TANGENT_SEED):
    """M with columns J v_r, flattened over (example, class). Shape (m*C, R).

    Call inside probe_context. no_grad applies to the outer graph only; jvp
    builds its own forward-mode dual computation and is unaffected.
    """
    params = trainable(model)
    buffers = {name: buf for name, buf in model.named_buffers()
               if not name.startswith(CONTAINER + ".")}

    def forward(overrides):
        return functional_call(model, {**overrides, **buffers}, (probe_x,))

    columns = []
    for tangent in tangents(params, count, seed):
        _, jv = jvp(forward, (params,), (tangent,))
        columns.append(jv.reshape(-1).double())
    return torch.stack(columns, 1)


@torch.no_grad()
def feature_sketch(model, probe_x):
    """Penultimate features as (m, F). The cheap path: one forward pass.

    K = Z Z^T is the feature Gram, and it goes through exactly the same block
    algebra as the NTK sketch, so drift() below serves both. It is logged from
    the first run alongside the NTK number so the correlation between the two is
    a byproduct rather than a separate experiment -- but it must not REPLACE the
    NTK estimator until Spearman rho over a whole trajectory clears 0.95. A high
    correlation at one point in training says nothing; the two disagree most
    exactly where the controller is doing something.
    """
    return model.forward_features(probe_x).reshape(probe_x.shape[0], -1).double()


def drift(current, reference):
    """Three numbers for two sketches, from their RxR blocks. Never forms K.

        d      ||K_t - K_0||_F / ||K_0||_F      the relative Frobenius distance
        a      <K_t, K_0>_F / (||K_t|| ||K_0||) the cosine alignment
        scale  ||K_t||_F / ||K_0||_F            how much bigger the kernel got

    All three, because d alone cannot be read. They satisfy

        d^2 = scale^2 - 2 a scale + 1

    so a d of 29 is a kernel whose NORM grew ~29x, whatever its geometry did.
    Measured on a short run here: d = 28.9 with a = 0.27, i.e. scale = 29.2 --
    the drift signal was almost entirely the kernel getting bigger. That is the
    regime the protocol's own rule anticipates: log both, feed d, and if they
    diverge sharply the scale is what is moving and a is the honest number.
    `scale` is returned so the divergence is a column rather than something to
    infer from the other two.
    """
    a_block = (current.T @ current).double()
    b_block = (reference.T @ reference).double()
    cross = (current.T @ reference).double()

    norm_a = float(a_block.norm())
    norm_b = float(b_block.norm())
    inner = float(cross.pow(2).sum())
    # Clamped because at w == w0 this is a difference of equal numbers and float64
    # rounding can land a few ulps below zero.
    gap = max(norm_a ** 2 - 2.0 * inner + norm_b ** 2, 0.0) ** 0.5
    return {"d": gap / max(norm_b, 1e-300),
            "a": inner / max(norm_a * norm_b, 1e-300),
            "scale": norm_a / max(norm_b, 1e-300)}


def measure(model, probe_x, cfg, reference=None):
    """One full probe: both estimators, and both drifts when a reference is given.

    Returns the sketches too, because step 0 needs them as the reference for the
    rest of the run.
    """
    with probe_context(model, bool(cfg["probe_deterministic"])):
        ntk = ntk_sketch(model, probe_x, int(cfg["probe_tangents"]))
        feature = feature_sketch(model, probe_x)
    sketches = {"ntk": ntk, "feature": feature}
    if reference is None:
        return sketches, {}
    out = {}
    for name, sketch in sketches.items():
        parts = drift(sketch, reference[name])
        out[f"d_{name}"] = parts["d"]
        out[f"a_{name}"] = parts["a"]
        out[f"scale_{name}"] = parts["scale"]
        # 1 - a: the same "how much has changed" orientation as d, rising from 0,
        # but invariant to the kernel's overall scale and bounded in [0, 2]. The
        # controller law needs no change to steer by it.
        out[f"dalign_{name}"] = 1.0 - parts["a"]
    return sketches, out


def selected(row, kernel, signal="drift"):
    """The one number the controller steers by.

    `drift` is the protocol's choice and the default. `alignment` steers by
    1 - a instead, which is the same quantity with the kernel's scale divided
    out -- worth reaching for when the logged scale column shows that d is mostly
    reporting norm growth. Whichever is chosen, all four columns are still logged,
    so the decision stays auditable after the run.
    """
    prefix = "dalign" if signal == "alignment" else "d"
    return row[f"{prefix}_{kernel}"]
