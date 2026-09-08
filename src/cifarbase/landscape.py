"""Measuring the loss surface in weight space. Numbers only -- no plotting.

Everything here answers one shape of question: *what does L look like near these
weights*, where "near" is a direction someone chose for a reason. The choosing is
the interesting part, so the directions are separated from the sweeping:

    random_direction   two filter-normalised random axes -- the Li et al. 2018
                       picture, for "is this minimum flat?"
    plane_from_points  the affine plane through three checkpoints -- for "did the
                       continuation path stay in one basin?"
    pca_plane          the top-2 directions the trajectory actually moved in

and then surface_2d sweeps whichever plane it is handed. surface_s_alpha is the
one axis that is not a weight direction at all: it sweeps the homotopy parameter
itself, so the picture shows the landscape *deforming* as s rises rather than a
single frozen slice of it.

Three traps, all of which produce a plausible-looking figure that means nothing:

  * BatchNorm running statistics are not parameters. They are measurements of the
    activations the weights produce, so a convex combination of two of them
    describes no network at all. Interpolating between two independently trained
    solutions without recompute_bn() manufactures a loss barrier that is an
    artefact of stale statistics, not geometry. Within one basin (a surface
    around a single checkpoint) the stored statistics are the convention and are
    what makes surfaces comparable, so recompute is a per-call decision.

  * Under BatchNorm the loss is invariant to the scale of every conv weight, so
    an un-normalised random direction plots the parameter scale rather than the
    curvature. Filter normalisation is what removes that.

  * A Hessian measured in train mode differs on every call, because BN's
    statistics depend on the very batch being differentiated. Everything here
    that touches curvature forces eval mode and a fixed batch.
"""
import contextlib
import json
import os

import torch
import torch.nn.functional as F

from cifarbase.activation import activation_sites
from cifarbase.homotopy import residual_blocks

EPS = 1e-10


# --------------------------------------------------------------------------
# weights as one flat vector
# --------------------------------------------------------------------------

def get_weights(model):
    """theta as one flat detached vector. Parameters only, never buffers."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_weights(model, vector):
    """Write a flat vector back into the model's parameters, in place."""
    expected = sum(p.numel() for p in model.parameters())
    if vector.numel() != expected:
        raise ValueError(f"vector has {vector.numel()} entries, model has "
                         f"{expected} parameters")
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(vector[offset:offset + n].view_as(p))
            offset += n


@contextlib.contextmanager
def at_weights(model, vector):
    """Evaluate at some other point in weight space, then put theta back."""
    saved = get_weights(model).clone()
    set_weights(model, vector)
    try:
        yield model
    finally:
        set_weights(model, saved)


# --------------------------------------------------------------------------
# directions
# --------------------------------------------------------------------------

def random_direction(model, seed, filter_norm=True, include_1d=False):
    """A random weight-space direction, filter-normalised (Li et al. 2018).

    Each filter's random direction is rescaled to the norm of the filter it
    perturbs, so a step of alpha means "move every filter by the same *fraction*
    of its own size" rather than "move every filter by the same number of units".
    Without it the plot is dominated by whichever layer happens to have the
    largest weights, and under BatchNorm that scale is arbitrary anyway.

    1-D parameters -- BN gammas and betas, biases -- are left at zero by default.
    They are a rounding error of the parameter count, and moving gamma changes
    the loss through BN's scale invariance rather than through the curvature the
    plot is meant to show.

    The generator is on CPU so the same seed gives the same direction whether the
    run happened on a GPU or not.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    parts = []
    for p in model.parameters():
        if p.dim() <= 1 and not include_1d:
            parts.append(torch.zeros(p.numel()))
            continue
        d = torch.randn(p.shape, generator=generator)
        if filter_norm and p.dim() >= 2:
            flat_d = d.reshape(d.shape[0], -1)
            flat_p = p.detach().reshape(p.shape[0], -1).cpu()
            scale = flat_p.norm(dim=1, keepdim=True) / (flat_d.norm(dim=1, keepdim=True) + EPS)
            d = (flat_d * scale).reshape(p.shape)
        parts.append(d.reshape(-1))
    device = next(model.parameters()).device
    return torch.cat(parts).to(device)


def plane_from_points(origin, a, b):
    """An orthonormal basis for the affine plane through three weight vectors.

    Gram-Schmidt on (a - origin, b - origin). The first axis therefore points
    exactly at `a`, which is what makes the resulting figure readable: with
    origin = theta*(0) and a = theta*(1), the x-axis *is* the continuation's net
    displacement, and everything else is measured against it.
    """
    u = a - origin
    u_norm = u.norm()
    if u_norm < EPS:
        raise ValueError("origin and a are the same point: no plane through them")
    e1 = u / u_norm
    v = b - origin
    scale = v.norm()
    v = v - (v @ e1) * e1
    v_norm = v.norm()
    # Relative, not absolute: the component of b left over after projecting out
    # e1 is computed by cancellation, so on collinear points it lands at the
    # float epsilon of |b - origin| rather than at zero. An absolute threshold
    # here passes collinear inputs through and returns a basis made of noise.
    if v_norm < 1e-6 * max(float(scale), EPS):
        raise ValueError("the three points are collinear: no plane through them")
    return e1, v / v_norm


def pca_plane(vectors, origin=None):
    """The top-2 directions a trajectory actually moved in, and their share.

    Returns (e1, e2, explained_variance_ratios). SGD on an 11M-parameter model
    moves in a tiny number of directions that carry most of the displacement, so
    a plane chosen this way shows where training spent its time; a random plane
    almost surely shows a smooth bowl and nothing else.

    Routed through the n x n Gram matrix because n is a dozen checkpoints and d
    is 11 million.
    """
    if len(vectors) < 3:
        raise ValueError(f"need at least 3 checkpoints for a PCA plane, got "
                         f"{len(vectors)}")
    origin = vectors[-1] if origin is None else origin
    matrix = torch.stack([(v - origin).float() for v in vectors])
    gram = (matrix @ matrix.T).double()
    values, basis = torch.linalg.eigh(gram)
    order = torch.argsort(values, descending=True)
    total = float(values.clamp(min=0).sum()) + EPS

    axes = []
    for index in order[:2]:
        direction = matrix.T @ basis[:, index].float()
        axes.append(direction / (direction.norm() + EPS))
    explained = [float(values[i].clamp(min=0)) / total for i in order[:2]]
    return axes[0], axes[1], explained


def project(vector, origin, e1, e2):
    """Coordinates of a weight vector in a plane, for overlaying a trajectory."""
    delta = vector - origin
    return float(delta @ e1), float(delta @ e2)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

@torch.no_grad()
def eval_batches(split, batch_size, max_images):
    """Materialise the evaluation subset once, already cropped and normalised.

    A surface is hundreds of sweeps over the *same* images. Reading them through
    split.chunks() every time re-runs the centre crop, the uint8->float cast and
    the normalisation on each grid point -- for a 21x21 grid that is 441 copies
    of identical work, and on a small evaluation subset it costs more than the
    forward passes do. Holding 5k images as float32 is 61 MB.

    A fixed prefix rather than a sample: every point on a surface must see
    identical data, or the differences between them are partly noise.
    """
    out, seen = [], 0
    for x, y in split.chunks(batch_size):
        out.append((x, y))
        seen += y.numel()
        if max_images and seen >= max_images:
            break
    return out


@torch.no_grad()
def _measure(model, batches):
    """Mean cross-entropy and accuracy over prepared batches."""
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in batches:
        logits = model(x)
        total_loss += float(F.cross_entropy(logits, y, reduction="sum"))
        correct += int((logits.argmax(1) == y).sum())
        seen += y.numel()
    return total_loss / seen, correct / seen, seen


def evaluate_at(model, split, weights=None, s=None, gate=None,
                batch_size=1000, max_images=5000):
    """One (loss, acc) reading at a chosen point of (theta, s) space."""
    was_training = model.training
    model.eval()
    try:
        with contextlib.ExitStack() as stack:
            if weights is not None:
                stack.enter_context(at_weights(model, weights))
            if s is not None and gate is not None:
                stack.enter_context(gate.at(s))
            loss, acc, _ = _measure(model, eval_batches(split, batch_size,
                                                        max_images))
    finally:
        if was_training:
            model.train()
    return loss, acc


@torch.no_grad()
def recompute_bn(model, train_split, batch_size=256, n_batches=64, seed=0):
    """Re-estimate every BatchNorm's running statistics at the current weights.

    Mandatory before evaluating a point interpolated between two independently
    trained solutions: the stored statistics describe neither endpoint's
    activations, and the loss barrier they produce is an artefact that has been
    mistaken for a real one in the literature more than once.

    momentum=None makes BatchNorm keep a cumulative average over exactly the
    batches shown here, so the result does not depend on the order they arrive
    in or on how many were seen before.
    """
    norms = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    if not norms:
        return 0
    saved = [m.momentum for m in norms]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None

    was_training = model.training
    model.train()
    generator = torch.Generator(device=train_split.device).manual_seed(seed)
    seen = 0
    for index, (x, _) in enumerate(train_split.batches(batch_size, generator,
                                                       augment=False)):
        if index >= n_batches:
            break
        model(x)
        seen += x.shape[0]

    for m, momentum in zip(norms, saved, strict=True):
        m.momentum = momentum
    model.train(was_training)
    return seen


# --------------------------------------------------------------------------
# sweeps
# --------------------------------------------------------------------------

def surface_2d(model, split, center, e1, e2, xs, ys, s=None, gate=None,
               batch_size=1000, max_images=5000, on_row=None):
    """L over a grid of theta = center + x*e1 + y*e2. Restores theta afterwards.

    Rows are computed outermost so a long sweep can stream partial results
    through `on_row` and be watched, or resumed, rather than producing nothing
    for twenty minutes and then possibly crashing.
    """
    saved = get_weights(model).clone()
    scratch = torch.empty_like(saved)
    batches = eval_batches(split, batch_size, max_images)
    loss_grid, acc_grid = [], []
    was_training = model.training
    model.eval()
    try:
        for y_value in ys:
            loss_row, acc_row = [], []
            for x_value in xs:
                torch.add(center, e1, alpha=float(x_value), out=scratch)
                scratch.add_(e2, alpha=float(y_value))
                set_weights(model, scratch)
                with (gate.at(s) if (s is not None and gate is not None)
                      else contextlib.nullcontext()):
                    loss, acc, _ = _measure(model, batches)
                loss_row.append(loss)
                acc_row.append(acc)
            loss_grid.append(loss_row)
            acc_grid.append(acc_row)
            if on_row is not None:
                on_row(len(loss_grid), len(ys), loss_row)
    finally:
        set_weights(model, saved)
        model.train(was_training)
    return {"kind": "surface_2d", "xs": list(map(float, xs)),
            "ys": list(map(float, ys)), "loss": loss_grid, "acc": acc_grid,
            "s": s}


def surface_s_alpha(model, split, center, direction, s_values, alphas, gate,
                    batch_size=1000, max_images=5000, on_row=None):
    """L_s(theta + alpha*d) over the (s, alpha) plane.

    The one figure that is specific to this project rather than borrowed. The
    x-axis is not a weight direction: it is the homotopy parameter, so the map
    shows how the landscape along a fixed weight direction *changes shape* as the
    residual branches switch on -- a valley deepening, drifting sideways, or
    splitting. A conventional loss-surface plot cannot show that, because it has
    no axis for it.
    """
    saved = get_weights(model).clone()
    scratch = torch.empty_like(saved)
    batches = eval_batches(split, batch_size, max_images)
    loss_grid, acc_grid = [], []
    was_training = model.training
    model.eval()
    try:
        for alpha in alphas:
            torch.add(center, direction, alpha=float(alpha), out=scratch)
            set_weights(model, scratch)
            loss_row, acc_row = [], []
            for s in s_values:
                with gate.at(float(s)):
                    loss, acc, _ = _measure(model, batches)
                loss_row.append(loss)
                acc_row.append(acc)
            loss_grid.append(loss_row)
            acc_grid.append(acc_row)
            if on_row is not None:
                on_row(len(loss_grid), len(alphas), loss_row)
    finally:
        set_weights(model, saved)
        model.train(was_training)
    return {"kind": "surface_s_alpha", "s_values": list(map(float, s_values)),
            "alphas": list(map(float, alphas)), "loss": loss_grid, "acc": acc_grid}


def loss_vs_s(model, split, s_values, gate, batch_size=1000, max_images=5000):
    """The homotopy slice at fixed theta: L_s(theta_t) as s sweeps 0 to 1.

    Cheap, and the first thing to look at. It says whether the weights currently
    held are good only at the s they were trained at, or across the whole path --
    which is the difference between following theta*(s) and merely riding it.
    """
    batches = eval_batches(split, batch_size, max_images)
    losses, accs = [], []
    was_training = model.training
    model.eval()
    for s in s_values:
        with gate.at(float(s)):
            loss, acc, _ = _measure(model, batches)
        losses.append(loss)
        accs.append(acc)
    model.train(was_training)
    return {"kind": "loss_vs_s", "s_values": list(map(float, s_values)),
            "loss": losses, "acc": accs}


def interpolate(model, split, w_a, w_b, n=13, extend=0.0, s=None, gate=None,
                train_split=None, bn_batches=64, batch_size=1000,
                max_images=5000):
    """L along the segment from w_a to w_b, optionally past both ends.

    Pass `train_split` to recompute BatchNorm at every point. Between two
    independently trained solutions that is not optional -- see recompute_bn.
    Between two points of one trajectory it is a choice, and leaving it off is
    the convention.
    """
    saved = get_weights(model).clone()
    scratch = torch.empty_like(saved)
    batches = eval_batches(split, batch_size, max_images)
    span = 1.0 + 2.0 * extend
    alphas = [-extend + span * i / (n - 1) for i in range(n)]
    losses, accs = [], []
    was_training = model.training
    try:
        for alpha in alphas:
            torch.lerp(w_a, w_b, float(alpha), out=scratch)
            set_weights(model, scratch)
            if train_split is not None:
                recompute_bn(model, train_split, n_batches=bn_batches)
            model.eval()
            with (gate.at(s) if (s is not None and gate is not None)
                  else contextlib.nullcontext()):
                loss, acc, _ = _measure(model, batches)
            losses.append(loss)
            accs.append(acc)
    finally:
        set_weights(model, saved)
        model.train(was_training)

    endpoints = max(losses[0], losses[-1])
    return {"kind": "interpolate", "alphas": alphas, "loss": losses, "acc": accs,
            "bn_recomputed": train_split is not None,
            # The number the figure exists to produce: how much higher the worst
            # point on the segment sits than the worse of the two endpoints.
            # Near zero means the two solutions are linearly connected, i.e. one
            # basin; a positive barrier means they are not.
            "barrier": max(losses) - endpoints}


# --------------------------------------------------------------------------
# per-block diagnostics
# --------------------------------------------------------------------------

@torch.no_grad()
def residual_ratio(model, x, gate=None):
    """rms(s*F(h)) / rms(shortcut(h)) for every block, on one fixed batch.

    The cheapest honest measure of effective depth: how loud is each residual
    branch relative to the signal it is correcting. It is also the test for
    whether s is doing anything at all -- because F ends in bn2, the network can
    absorb a small s by growing gamma, and if it does, this ratio is unchanged
    between a gated run and the baseline at the same point in training even
    though s differs. That would mean the homotopy was reparametrised away.

    Measured with the gate held at 1 so the raw branch magnitude is what is read;
    s is multiplied back in afterwards, from the values the gate actually held.
    """
    blocks = residual_blocks(model)
    s_values = gate.get() if gate is not None else [1.0] * len(blocks)
    captured = {}

    def capture(index):
        def hook(module, inputs):
            captured[index] = inputs[0].detach()
        return hook

    handles = [b.register_forward_pre_hook(capture(i))
               for i, b in enumerate(blocks)]
    was_training = model.training
    model.eval()
    rows = []
    try:
        with (gate.at(1.0) if gate is not None else contextlib.nullcontext()):
            model(x)
            for index, block in enumerate(blocks):
                h = captured[index]
                # block.act1, never F.relu: under the activation homotopy the
                # branch's nonlinearity is a LeakyReLU of slope alpha, and a
                # hardcoded relu here would measure a network that is not
                # running while producing an entirely plausible number.
                branch = block.bn2(block.conv2(
                    block.act1(block.bn1(block.conv1(h)))))
                # A PlainNet block has no shortcut attribute at all -- it is
                # removed rather than disabled -- so there is nothing to divide
                # by and the ratio is undefined, not zero. Reported as nan so a
                # plot shows a hole rather than a plausible flat line, and the
                # branch magnitude is still measured, which is the part that
                # still means something without a skip.
                skip = (block.shortcut(h) if getattr(block, "use_residual", True)
                        else None)
                # rms rather than a norm so blocks with different channel counts
                # and resolutions are on the same scale.
                f_rms = float(branch.pow(2).mean().sqrt())
                skip_rms = float(skip.pow(2).mean().sqrt()) if skip is not None else 0.0
                ratio = (s_values[index] * f_rms / (skip_rms + EPS)
                         if skip is not None else float("nan"))
                rows.append({"block": index, "s": s_values[index],
                             "f_rms": f_rms, "skip_rms": skip_rms,
                             "ratio": ratio})
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return {"kind": "residual_ratio", "blocks": rows}


def grad_norm_per_block(model, x, y, gate=None):
    """Gradient norm reaching each block's residual branch, on one batch.

    At s=0 this is exactly zero -- dL/dtheta_F is proportional to s -- which is
    the fact that decides whether an s_min of 0 means "start from the shallow
    network" or "waste the first epochs with a frozen branch". Worth measuring
    rather than believing.
    """
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    try:
        F.cross_entropy(model(x), y).backward()
        rows = []
        for index, block in enumerate(residual_blocks(model)):
            total = 0.0
            for child in (block.conv1, block.bn1, block.conv2, block.bn2):
                for p in child.parameters():
                    if p.grad is not None:
                        total += float(p.grad.pow(2).sum())
            rows.append({"block": index,
                         "s": (gate.get()[index] if gate is not None else 1.0),
                         "grad_norm": total ** 0.5})
        # The whole-model gradient norm, which is the only convergence signal
        # available without a second pass: a continuation phase that has settled
        # has a small one, a phase that ran out of epochs does not. Reported
        # alongside the per-block numbers so "did this phase converge?" can be
        # answered from the log rather than assumed from the epoch count.
        overall = sum(float(p.grad.pow(2).sum()) for p in model.parameters()
                      if p.grad is not None) ** 0.5
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    return {"kind": "grad_norm_per_block", "blocks": rows, "total": overall}


@torch.no_grad()
def activation_stats(model, x, which="all"):
    """Per activation site: how far from linear this nonlinearity actually is.

        linear_gap = rms(phi_alpha(h) - h) / rms(h)
                   = (1 - alpha) * rms(relu(-h)) / rms(h)

    the exact analogue of residual_ratio, and it answers the same question: is
    the homotopy deforming the network, or has the network routed around it?

    alpha cannot be scaled away -- phi_alpha is positively homogeneous -- but it
    can be *shifted* away: BatchNorm need only push beta positive until nearly
    every pre-activation is positive, and phi_alpha is then the identity for
    every alpha. linear_gap and neg_frac are how that shows up. A run whose
    linear_gap stays flat while alpha falls is a baseline wearing a costume,
    and finding that out after sixty epochs costs a day per arm.

    Also reported:
      neg_frac    P(h < 0) -- the shift, measured directly
      dead_frac   channels whose pre-activation never rises above 0 on this
                  batch; at alpha>0 they still pass gradient, at alpha=0 they
                  are dead, so this is the cost of arriving at ReLU
      rms         activation scale, which is where a Kaiming gain calibrated
                  for ReLU shows up if alpha=1 has thrown it off

    The statistics are taken *inside* the pre-forward hook, not from a stashed
    tensor. Activation is inplace by default, so a tensor measured after the
    forward has already had its negatives erased and every one of these numbers
    would read zero.
    """
    sites = activation_sites(model, which)
    rows = {}

    def capture(index, stage, block):
        def hook(module, inputs):
            h = inputs[0]
            rms = float(h.pow(2).mean().sqrt())
            negative = float(F.relu(-h).pow(2).mean().sqrt())
            channels = tuple(d for d in range(h.dim()) if d != 1)
            rows[index] = {
                "site": index, "stage": stage, "block": block,
                "alpha": float(module.alpha), "rms": rms,
                "neg_frac": float((h < 0).to(h.dtype).mean()),
                "linear_gap": (1.0 - float(module.alpha)) * negative / (rms + EPS),
                "dead_frac": float((h.amax(dim=channels) <= 0).to(h.dtype).mean()),
            }
        return hook

    handles = [site.register_forward_pre_hook(capture(index, stage, block))
               for index, (site, stage, block) in enumerate(sites)]
    was_training = model.training
    model.eval()
    try:
        model(x)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return {"kind": "activation_stats",
            "sites": [rows[index] for index in range(len(sites))]}


def loss_vs_alpha(model, split, alphas, gate, batch_size=1000, max_images=5000):
    """The activation homotopy slice at fixed theta: L_alpha(theta_t), 1 -> 0.

    Cheap, and the first thing to look at. It says whether the weights being
    held are good only at the alpha they were trained at, or across the whole
    path -- the difference between following theta*(alpha) and merely riding it.

    No BatchNorm recompute: this is one set of weights read at several alphas,
    so the stored statistics are the convention that makes the curve internally
    comparable. The alpha=0 *readout* in train.py is a different measurement
    and does recompute -- see the note there.
    """
    batches = eval_batches(split, batch_size, max_images)
    losses, accs = [], []
    was_training = model.training
    model.eval()
    previous = gate.get()
    try:
        for alpha in alphas:
            gate.set(float(alpha))
            loss, acc, _ = _measure(model, batches)
            losses.append(loss)
            accs.append(acc)
    finally:
        gate.set(previous)
        model.train(was_training)
    return {"kind": "loss_vs_alpha", "alphas": list(map(float, alphas)),
            "loss": losses, "acc": accs}


@torch.no_grad()
def functional_distance(model, w_a, w_b, split, alpha=None, gate=None,
                        batch_size=1000, max_images=5000):
    """How differently two weight vectors classify, rather than how far apart
    they are.

    The continuity measure a solution branch actually needs. Two points on
    theta*(alpha) can be far apart in parameter space and compute nearly the
    same function -- under BatchNorm the loss is invariant to the scale of
    every conv weight, so ||theta_a - theta_b|| is partly measuring a gauge
    choice. Disagreement and symmetric KL are not.

    Both endpoints are evaluated with their own stored BatchNorm statistics,
    which is correct here precisely because neither is an interpolated point:
    they are two trained solutions, each with statistics that describe it.

    `alpha` may be one value for both endpoints, or a pair -- one each. The
    pair is what a branch measurement wants: theta*(alpha_k) is the network it
    is at alpha_k, and comparing it to theta*(alpha_{k+1}) read at the *same*
    alpha would measure a network that was never trained.
    """
    if alpha is None or isinstance(alpha, (int, float)):
        alphas = (alpha, alpha)
    else:
        alphas = tuple(alpha)
        if len(alphas) != 2:
            raise ValueError(f"alpha must be a scalar or a pair, got {alpha!r}")

    saved = get_weights(model).clone()
    batches = eval_batches(split, batch_size, max_images)
    was_training = model.training
    model.eval()
    try:
        outputs = []
        for weights, value in zip((w_a, w_b), alphas, strict=True):
            set_weights(model, weights)
            with (gate.at(value) if (gate is not None and value is not None)
                  else contextlib.nullcontext()):
                outputs.append(torch.cat([model(x) for x, _ in batches]))
    finally:
        set_weights(model, saved)
        model.train(was_training)

    log_a = F.log_softmax(outputs[0], dim=1)
    log_b = F.log_softmax(outputs[1], dim=1)
    sym_kl = float(((log_a.exp() - log_b.exp()) * (log_a - log_b)).sum(1).mean())
    disagreement = float((outputs[0].argmax(1) != outputs[1].argmax(1))
                         .to(log_a.dtype).mean())
    return {"kind": "functional_distance", "disagreement": disagreement,
            "sym_kl": sym_kl, "n": int(outputs[0].shape[0]),
            "alpha": list(alphas),
            "weight_distance": float((w_a - w_b).norm()),
            "weight_distance_rel": float((w_a - w_b).norm() / (w_a.norm() + EPS))}


# --------------------------------------------------------------------------
# curvature
# --------------------------------------------------------------------------

def _hvp(model, params, x, y, vector):
    """One Hessian-vector product of the batch loss."""
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat = torch.cat([g.reshape(-1) for g in grads])
    product = torch.autograd.grad(flat @ vector, params)
    return torch.cat([p.reshape(-1) for p in product]).detach()


def top_hessian_eigs(model, x, y, k=2, iters=40, tol=1e-3, seed=0, gate=None,
                     s=None):
    """The k largest-magnitude Hessian eigenvalues, by power iteration.

    eval mode and one fixed batch, both mandatory: in train mode the loss depends
    on the batch statistics of the batch being differentiated, and the resulting
    "Hessian" changes on every call.

    Power iteration converges to the largest eigenvalue *in magnitude*. At a
    minimum that is the top positive curvature, which is the sharpness people
    mean; away from one it may be a negative direction, so the sign is reported
    rather than assumed. Subsequent eigenvalues come from deflation, which
    compounds the error of the earlier ones -- k beyond 2 or 3 is not worth
    trusting from this routine.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    size = sum(p.numel() for p in params)
    # dtype as well as device: torch.randn defaults to float32, and a float32
    # probe against float64 parameters fails inside the dot product rather than
    # silently, but only once the sweep is already running.
    device, dtype = params[0].device, params[0].dtype
    generator = torch.Generator(device="cpu").manual_seed(seed)

    was_training = model.training
    model.eval()
    found, vectors = [], []
    try:
        with (gate.at(s) if (gate is not None and s is not None)
              else contextlib.nullcontext()):
            for _ in range(k):
                v = torch.randn(size, generator=generator).to(device, dtype)
                v /= v.norm()
                value, converged = 0.0, False
                for _ in range(iters):
                    w = _hvp(model, params, x, y, v)
                    for previous_value, previous_vector in zip(found, vectors,
                                                               strict=True):
                        w -= previous_value * (previous_vector @ v) * previous_vector
                    new_value = float(v @ w)
                    norm = w.norm()
                    if float(norm) < EPS:
                        value = new_value
                        converged = True
                        break
                    v = w / norm
                    if abs(new_value - value) <= tol * max(abs(new_value), EPS):
                        value = new_value
                        converged = True
                        break
                    value = new_value
                found.append(value)
                vectors.append(v)
                if not converged:
                    break
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    return {"kind": "hessian_eigs", "eigenvalues": found, "s": s,
            "n_requested": k}


def hutchinson_trace(model, x, y, samples=16, seed=0, gate=None, s=None):
    """Hessian trace by Hutchinson's estimator, with its own standard error.

    The mean curvature, where top_hessian_eigs gives the worst. Reported with a
    sem because a 16-sample Rademacher estimate on an 11M-dimensional Hessian is
    noisy, and a trace quoted without one invites reading a difference that is
    not there.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    size = sum(p.numel() for p in params)
    device, dtype = params[0].device, params[0].dtype
    generator = torch.Generator(device="cpu").manual_seed(seed)

    was_training = model.training
    model.eval()
    estimates = []
    try:
        with (gate.at(s) if (gate is not None and s is not None)
              else contextlib.nullcontext()):
            for _ in range(samples):
                v = (torch.randint(0, 2, (size,), generator=generator) * 2 - 1
                     ).to(device, dtype)
                estimates.append(float(v @ _hvp(model, params, x, y, v)))
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    mean = sum(estimates) / len(estimates)
    if len(estimates) > 1:
        variance = sum((e - mean) ** 2 for e in estimates) / (len(estimates) - 1)
        sem = (variance / len(estimates)) ** 0.5
    else:
        sem = float("nan")
    return {"kind": "hessian_trace", "trace": mean, "sem": sem,
            "samples": len(estimates), "s": s}


# --------------------------------------------------------------------------
# checkpoints and records
# --------------------------------------------------------------------------

def checkpoint_dir(directory):
    path = os.path.join(directory, "checkpoints")
    os.makedirs(path, exist_ok=True)
    return path


def save_checkpoint(model, directory, tag, meta=None, half=True):
    """A full state_dict plus what it was, next to the run that produced it.

    The buffers go too: evaluating a checkpoint needs its BatchNorm statistics,
    and re-deriving them is only correct when the weights are a point on a
    segment rather than a trained solution.

    Conv weights are stored as fp16 by default -- 22 MB instead of 45 MB, and a
    relative error of 1e-3 on a weight of order 1e-2 moves the loss far below
    anything these figures resolve. 1-D tensors stay fp32: BN running variances
    span several orders of magnitude and are cheap to keep exact.
    """
    state = {}
    for key, value in model.state_dict().items():
        if half and value.is_floating_point() and value.dim() >= 2:
            state[key] = value.detach().to("cpu", torch.float16)
        else:
            state[key] = value.detach().to("cpu")
    path = os.path.join(checkpoint_dir(directory), f"{tag}.pt")
    torch.save({"state": state, "meta": dict(meta or {}), "half": bool(half)},
               path)
    return path


def load_checkpoint(path, model=None, device=None):
    """Read a checkpoint; load it into `model` if one is given."""
    payload = torch.load(path, map_location=device or "cpu", weights_only=True)
    state = payload["state"]
    if model is not None:
        reference = model.state_dict()
        restored = {k: v.to(device=reference[k].device, dtype=reference[k].dtype)
                    for k, v in state.items()}
        model.load_state_dict(restored)
    return payload


def list_checkpoints(directory):
    """Every checkpoint of a run, ordered by the epoch recorded in its meta."""
    path = os.path.join(directory, "checkpoints")
    if not os.path.isdir(path):
        return []
    found = []
    for name in sorted(os.listdir(path)):
        if not name.endswith(".pt"):
            continue
        full = os.path.join(path, name)
        meta = torch.load(full, map_location="cpu", weights_only=True)["meta"]
        found.append({"path": full, "tag": name[:-3], **meta})
    return sorted(found, key=lambda row: (row.get("epoch", -1), row["tag"]))


def record(directory, payload, filename="landscape.jsonl"):
    """Append one measurement. JSONL so a sweep that dies keeps what it had."""
    path = os.path.join(directory, filename)
    with open(path, "a") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")
    return path


def read_records(directory, filename="landscape.jsonl"):
    path = os.path.join(directory, filename)
    if not os.path.isfile(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]
