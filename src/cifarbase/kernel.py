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

PER GROUP. The true K decomposes exactly over any partition of the parameters,
K = sum_g K^(g), and masking each tangent to one group estimates the blocks. Two
things about that decomposition are easy to state wrongly.

The sketch MATRIX is additive: J v = sum_g J_g v, so summing the masked M_g
reproduces M to floating point. The sketch GRAM is NOT. K_hat = M M^T expands to
sum_g sum_h (J_g v)(J_h v)^T, and the g != h cross terms are zero in expectation
-- the tangent's blocks are independent -- but not zero at R = 8. So

    sum_g K_hat^(g)  !=  K_hat,        and  sum_g share_g  need not reach 1

even though the underlying quantities do satisfy sum_g K^(g) = K. Measured here:
the shares of ||K_0||_F summed to 0.87 over all eight groups. That is the cross
terms, not a bug, and it is why the global sketch is taken directly rather than
recovered by summing the blocks.

The per-group drifts are all divided by the GLOBAL ||K_0||_F, never by each
block's own norm -- a group holding 1% of the kernel would otherwise report the
same d as the whole network for a change a hundred times smaller.

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

from cifarbase.anchor import CONTAINER, all_groups

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
def ntk_sketch_by_group(model, probe_x, group_of, count, seed=TANGENT_SEED,
                        grouping="stage"):
    """{group: M_g}, with M_g's columns J_g v_r. Shape (m*C, R) each.

    The Gram decomposes exactly over a partition of the parameters,

        K = sum_g K^(g),   K^(g)_ij = <grad_{w_g} f_i, grad_{w_g} f_j>

    and each M_g estimates its block, because a JVP is linear in the tangent:
    masking v_r to group g and leaving the rest at zero gives J_g v_r.

    sum_g M_g == M to floating point, but do NOT use that to save the global
    pass -- see the module docstring. The Gram built from the sum carries cross
    terms the sum of the blocks' Grams does not, and the float32 discrepancy is
    the size of the spurious d near w0.

    Cost is L*R JVPs on top of the global sketch's R. That is the price of the
    decomposition and there is no way around it -- a masked tangent still needs a
    full forward-mode pass. probe_group_every is the knob for how often to pay it.

    Not vmapped over the L*R tangents. Stacking them is L*R*p floats -- 2.5 GB at
    ResNet-18, L=8, R=8 -- which does not fit next to the dataset, w and w0 on a
    T4. The batching that matters is already there: each JVP is one pass over all
    m probe examples at once.

    Same frozen tangents as ntk_sketch, from the same seed, masked. Common random
    numbers hold per group exactly as they do globally.
    """
    params = trainable(model)
    buffers = {name: buf for name, buf in model.named_buffers()
               if not name.startswith(CONTAINER + ".")}
    missing = sorted(set(params) - set(group_of))
    if missing:
        raise ValueError(f"no anchor group for {', '.join(missing)}: the kernel "
                         f"decomposition and the pull disagree about the partition")

    def forward(overrides):
        return functional_call(model, {**overrides, **buffers}, (probe_x,))

    # Ordered by the partition rather than by the dict, so the column order of
    # every probes.csv in the project is a property of the grouping and not of
    # parameter registration order.
    seen = set(group_of.values())
    present = [group for group in all_groups(grouping) if group in seen]
    columns = {group: [] for group in present}
    zeros = {name: torch.zeros_like(param) for name, param in params.items()}
    for tangent in tangents(params, count, seed):
        for group in present:
            masked = {name: (tangent[name] if group_of[name] == group
                             else zeros[name])
                      for name in params}
            _, jv = jvp(forward, (params,), (masked,))
            columns[group].append(jv.reshape(-1).double())
    return {group: torch.stack(cols, 1) for group, cols in columns.items()}


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


def drift(current, reference, denominator=None):
    """Three numbers for two sketches, from their RxR blocks. Never forms K.

        d      ||K_t - K_0||_F / ||K_0||_F      the relative Frobenius distance
        a      <K_t, K_0>_F / (||K_t|| ||K_0||) the cosine alignment
        scale  ||K_t||_F / ||K_0||_F            how much bigger the kernel got
        norm   ||K_0||_F                        so a caller can reuse it below

    `denominator` overrides ||K_0||_F in d and scale. The per-group drifts use it
    to divide by the GLOBAL ||K_0||_F rather than by each block's own norm, which
    is what makes d_1..d_L commensurable -- a group holding 1% of the kernel would
    otherwise report a drift as large as the whole network's for a change 100x
    smaller, and the coupling matrix built from them would be meaningless.

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
    scale_by = max(norm_b if denominator is None else denominator, 1e-300)
    return {"d": gap / scale_by,
            "a": inner / max(norm_a * norm_b, 1e-300),
            "scale": norm_a / scale_by,
            "norm": norm_b}


def measure(model, probe_x, cfg, reference=None, group_of=None):
    """One full probe: both estimators, and both drifts when a reference is given.

    Returns the sketches too, because step 0 needs them as the reference for the
    rest of the run.

    With `group_of` the NTK is taken group by group and the global sketch is their
    sum, so the decomposition costs L*R JVPs and the global number costs nothing
    extra. Without it this is the original R-JVP path, unchanged -- that is what
    makes probe_group_every a pure cost knob rather than a change of measurement.
    """
    with probe_context(model, bool(cfg["probe_deterministic"])):
        # The global sketch is ALWAYS taken directly, never recovered as the sum
        # of the masked ones. Summing would be exact in real arithmetic and is
        # accurate to about 1e-7 in float32 -- but that is the same size as the
        # spurious d near w0 that the float64 block algebra exists to avoid, and
        # it would make the number the controller steers by depend on whether the
        # per-group sketch happened to fire on this probe. probe_group_every has
        # to be a logging knob, not a measurement one. The extra R JVPs cost ~12%
        # of a group probe, which those probes only pay once every
        # probe_group_every of them.
        ntk = ntk_sketch(model, probe_x, int(cfg["probe_tangents"]))
        ntk_groups = None
        if group_of is not None:
            ntk_groups = ntk_sketch_by_group(
                model, probe_x, group_of, int(cfg["probe_tangents"]),
                grouping=cfg.get("anchor_grouping", "stage"))
        feature = feature_sketch(model, probe_x)
    sketches = {"ntk": ntk, "feature": feature}
    if ntk_groups is not None:
        sketches["ntk_groups"] = ntk_groups
    if reference is None:
        return sketches, {}
    out = {}
    for name in ("ntk", "feature"):
        parts = drift(sketches[name], reference[name])
        out[f"d_{name}"] = parts["d"]
        out[f"a_{name}"] = parts["a"]
        out[f"scale_{name}"] = parts["scale"]
        # 1 - a: the same "how much has changed" orientation as d, rising from 0,
        # but invariant to the kernel's overall scale and bounded in [0, 2]. The
        # controller law needs no change to steer by it.
        out[f"dalign_{name}"] = 1.0 - parts["a"]
        if name == "ntk":
            global_norm = parts["norm"]

    # Per-group drift, divided by the GLOBAL ||K_0||_F so the L numbers are on one
    # scale and can be read against each other. They do not sum to d, in either
    # direction: the blocks' Grams leave out K_hat's cross terms, and the blocks'
    # displacements are not orthogonal. d_g answers "how much did THIS block of
    # the kernel move, as a fraction of the whole kernel's size", and that is all.
    groups = sketches.get("ntk_groups")
    if groups is not None and "ntk_groups" in reference:
        for group, sketch in groups.items():
            if group not in reference["ntk_groups"]:
                continue
            parts = drift(sketch, reference["ntk_groups"][group],
                          denominator=global_norm)
            out[f"d_{group}"] = parts["d"]
            out[f"a_{group}"] = parts["a"]
            out[f"scale_{group}"] = parts["scale"]
            # The group's share of ||K_0||: without it a small d_g cannot be told
            # apart from a group that simply contributes almost nothing to K.
            out[f"share_{group}"] = parts["norm"] / max(global_norm, 1e-300)
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
