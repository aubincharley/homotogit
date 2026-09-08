"""The residual homotopy: h_out = shortcut(x) + s * F(x), with s driven from 0 to 1.

Three things live here and nothing else:

  ResidualGate   the mechanism -- how s gets applied to a model
  s_at           the policy    -- what s should be at a given point in training
  alpha_at_cfg   the same policy, read in the coordinate the *activation*
                 homotopy needs (see activation.py for its mechanism)

`model.py` is untouched by both. The gate installs a forward hook on each block's
`bn2`, whose output is exactly F(x) before the residual add, and returns a scaled
copy. That is the whole implementation: the baseline stays the baseline, a gated
model and an ungated one share a state_dict, and removing the gate returns the
model to bit-identical baseline behaviour rather than to something that merely
sets s=1.

Two consequences of the algebra are worth knowing before reading any result:

  * s multiplies the residual branch and never the shortcut. At s=0 the network
    is not the identity -- it is the shortcut-only network: the 3x3 stem, the
    three 1x1 projections at the stage transitions, global pool, and the head.
    A real, trainable, shallow network. Gating the shortcut as well would send
    the output to zero and leave nothing to train.

  * F ends in bn2, so s*BN(.) is the same function as BN with gamma -> s*gamma.
    For any s > 0 the representable function class is *identical* to the one at
    s=1: this homotopy changes the parametrisation, not the capacity. What
    breaks the exact equivalence is weight decay (which penalises gamma, not
    s*gamma), momentum, and the fact that dL/dtheta_F is proportional to s, so s
    also acts as a branch-local learning-rate multiplier. residual_parameters()
    exists to build the control arm that separates those two readings.

At s=0 exactly, dL/dtheta_F is zero: the residual convolutions receive no
gradient at all and are frozen, while their BatchNorm running statistics keep
drifting because the forward pass still runs. That is a defensible regime, but
it has to be a choice -- hence s_min.
"""
import contextlib
import math

from cifarbase.model import BasicBlock

SCHEDULES = ("const", "linear", "cosine", "staircase", "sequential")


# --------------------------------------------------------------------------
# the mechanism
# --------------------------------------------------------------------------

def residual_blocks(model):
    """Every BasicBlock, in forward order.

    `modules()` walks registration order, which for ResNet is conv1, bn1,
    layer1..layer4, fc -- the same order the forward pass visits them. That is a
    fact about model.py rather than a guarantee from torch, so test_block_order
    pins it down with hooks instead of trusting it.
    """
    return [m for m in model.modules() if isinstance(m, BasicBlock)]


def residual_parameters(model):
    """The parameters of the gated branches only: conv1/bn1/conv2/bn2 per block.

    Not the shortcut projections -- s does not touch those, so a control arm that
    applies the s profile to a learning rate has to leave them out too, or it is
    not the control it claims to be.
    """
    out = []
    for block in residual_blocks(model):
        for child in (block.conv1, block.bn1, block.conv2, block.bn2):
            out.extend(child.parameters())
    return out


def other_parameters(model):
    """Everything residual_parameters() left behind: stem, shortcuts, head."""
    gated = {id(p) for p in residual_parameters(model)}
    return [p for p in model.parameters() if id(p) not in gated]


class ResidualGate:
    """Applies s to a model's residual branches, one float per block.

    Attach once and call `set` as often as you like; the hooks read the current
    values on every forward, so there is no per-step re-installation cost. Use it
    as a context manager when the gate should not outlive the measurement.
    """

    def __init__(self, model):
        self.blocks = residual_blocks(model)
        if not self.blocks:
            raise ValueError("no BasicBlock in this model: nothing to gate")
        self.values = [1.0] * len(self.blocks)
        self._handles = [block.bn2.register_forward_hook(self._hook(index))
                         for index, block in enumerate(self.blocks)]

    def _hook(self, index):
        def hook(module, inputs, output):
            s = self.values[index]
            # Returning `output` untouched rather than `output * 1.0` is what
            # makes a gate at s=1 bit-identical to no gate at all, which is the
            # only reason an accuracy measured here is comparable to a baseline
            # number.
            return output if s == 1.0 else output * s
        return hook

    def __len__(self):
        return len(self.blocks)

    def set(self, s):
        """s is a scalar applied to every block, or one value per block."""
        if isinstance(s, (int, float)):
            values = [float(s)] * len(self.blocks)
        else:
            values = [float(v) for v in s]
            if len(values) != len(self.blocks):
                raise ValueError(f"expected {len(self.blocks)} values for "
                                 f"{len(self.blocks)} blocks, got {len(values)}")
        self.values[:] = values
        return list(values)

    def get(self):
        return list(self.values)

    @contextlib.contextmanager
    def at(self, s):
        """Temporarily hold s, then restore it.

        This is how the two accuracies get measured: the model as it currently
        runs, and the same weights read out at s=1. Without the restore, an
        evaluation would silently change the next training step.
        """
        previous = self.get()
        self.set(s)
        try:
            yield self
        finally:
            self.values[:] = previous

    def close(self):
        """Remove the hooks. The model is then byte-for-byte the baseline again."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# --------------------------------------------------------------------------
# the policy
# --------------------------------------------------------------------------

def ramp_progress(progress, ramp_start, ramp_end):
    """Training progress in [0,1] -> position along the ramp, clamped to [0,1].

    Everything downstream is written in terms of this one coordinate, so the
    invariant that matters -- s is exactly s_max once progress reaches ramp_end --
    only has to hold in one place.
    """
    if ramp_end <= ramp_start:
        return 1.0
    return min(1.0, max(0.0, (progress - ramp_start) / (ramp_end - ramp_start)))


def s_at(progress, n_blocks, schedule="const", s_min=0.0, s_max=1.0,
         ramp_start=0.0, ramp_end=0.5, stairs=0):
    """s for every block at this point in training.

    `progress` is the fraction of total steps elapsed, not the epoch: computing
    it per step keeps the schedule from depending on how many batches an epoch
    happens to contain, the same reason lr_at in train.py works that way.

    Every schedule returns exactly s_max for progress >= ramp_end. That is not a
    detail: a run whose s never reaches 1 has trained a different model than the
    baseline, and comparing their accuracies compares nothing. _validate rejects
    ramp_end >= 1 so that a non-empty tail of training always happens at s_max.
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule {schedule!r}: "
                         f"pick one of {', '.join(SCHEDULES)}")
    if schedule == "const":
        return [s_max] * n_blocks

    u = ramp_progress(progress, ramp_start, ramp_end)
    span = s_max - s_min

    if schedule == "linear":
        return [s_min + span * u] * n_blocks

    if schedule == "cosine":
        # Zero derivative at both ends, so the optimiser is never handed a
        # discontinuity in ds/dt at the moment the ramp starts or stops.
        return [s_min + span * 0.5 * (1.0 - math.cos(math.pi * u))] * n_blocks

    if schedule == "staircase":
        # `stairs` plateaus over the ramp, the last of which sits at s_max for a
        # full plateau before the post-ramp tail even begins.
        #
        # Holding s still is necessary for continuation but not sufficient: a
        # plateau is only a step along theta*(s) if theta settles during it, and
        # nothing here enforces that. See phase_bounds -- s_lr_restart gives each
        # plateau its own decay so it can, and grad_norm_total in the log is how
        # you find out whether it did.
        if stairs < 1:
            raise ValueError("staircase needs stairs >= 1")
        if stairs == 1:
            return [s_max] * n_blocks
        # The epsilon is not cosmetic. u = (progress - start) / (end - start)
        # lands a float hair below k/stairs at an exact plateau boundary, so
        # int() returns k-1 and the level steps one epoch late -- while
        # phase_bounds, which computes the same edges analytically, has already
        # restarted the learning rate. On a four-epoch plateau that is a
        # quarter of it spent at the old alpha under the new phase's lr.
        index = min(int(u * stairs + 1e-9), stairs - 1)
        return [s_min + span * index / (stairs - 1)] * n_blocks

    # sequential: depth continuation. Block k ramps over the k-th of n_blocks
    # equal windows, so the network grows deeper from the input outward instead
    # of getting uniformly louder.
    return [s_min + span * min(1.0, max(0.0, u * n_blocks - k))
            for k in range(n_blocks)]


def s_at_cfg(progress, n_blocks, cfg):
    """s_at driven from a resolved config dict."""
    return s_at(progress, n_blocks,
                schedule=cfg["s_schedule"], s_min=cfg["s_min"], s_max=cfg["s_max"],
                ramp_start=cfg["s_ramp_start"], ramp_end=cfg["s_ramp_end"],
                stairs=cfg["s_stairs"])


def alpha_at_cfg(progress, n_groups, cfg):
    """alpha per group for the activation homotopy, from a resolved config.

    alpha falls 1 -> 0 while s rises 0 -> 1, so rather than write a second
    policy this reads the schedule in

        tau = 1 - alpha

    the "nonlinearity level", which rises 0 -> 1 exactly as s does. Every
    schedule, ramp window, staircase and stagger above therefore applies to
    alpha unchanged, along with the tests that pin them down. alpha is what
    configs and logs speak in; tau never leaves this function.

    The tau endpoints are inverted as well as the values -- a_start is where
    alpha begins, so it is tau's *minimum* -- which is why this cannot be a
    prefix argument to s_at_cfg.
    """
    if cfg["a_schedule"] == "none":
        return [0.0] * n_groups
    tau = s_at(progress, n_groups, schedule=cfg["a_schedule"],
               s_min=1.0 - cfg["a_start"], s_max=1.0 - cfg["a_end"],
               ramp_start=cfg["a_ramp_start"], ramp_end=cfg["a_ramp_end"],
               stairs=cfg["a_stairs"])
    return [1.0 - t for t in tau]


# --------------------------------------------------------------------------
# continuation phases
# --------------------------------------------------------------------------

def _staircase_axes(cfg):
    """The homotopy axes asking for phases of their own, as
    (ramp_start, ramp_end, stairs, value_at).

    `.get` rather than `[...]`: a cfg predating the activation keys, or a
    hand-built one in a test, must still resolve to the residual axis alone.
    """
    axes = []
    if cfg["s_schedule"] == "staircase" and cfg.get("s_lr_restart"):
        axes.append((cfg["s_ramp_start"], cfg["s_ramp_end"], cfg["s_stairs"],
                     lambda progress: s_at_cfg(progress, 1, cfg)[0]))
    if cfg.get("a_schedule") == "staircase" and cfg.get("a_lr_restart"):
        axes.append((cfg["a_ramp_start"], cfg["a_ramp_end"], cfg["a_stairs"],
                     lambda progress: alpha_at_cfg(progress, 1, cfg)[0]))
    return axes


def phase_bounds(cfg):
    """Where the continuation changes problem, as fractions of training.

    Continuation is not "restart from the previous weights" -- theta carries over
    in every schedule here, because there is only ever one run and nothing is
    re-initialised. It is "converge at this s, *then* step". The converging is
    what makes the run track theta*(s), the actual branch of solutions, instead
    of drifting behind an objective that keeps moving.

    Only `staircase` holds s still long enough for that to mean anything, and
    even it cannot deliver it under a single global cosine: the early plateaus
    then run at high lr and never settle, the late ones at a learning rate too
    small to move. So each plateau becomes its own optimisation problem, with its
    own warmup and its own decay, warm-started from where the last one stopped.

    Either homotopy can ask for phases, and the boundaries are the union of
    what each asks for: two staircases running at once change problem at every
    edge either of them steps over. In practice the pilot runs one axis at a
    time and the union is that axis's own edges.

    Returns () when the run is a single problem, which leaves lr_at untouched and
    every existing arm bit-identical to before this function existed.
    """
    axes = _staircase_axes(cfg)
    if not axes:
        return ()
    candidates = {0.0, 1.0}
    for start, end, stairs, _ in axes:
        candidates.update((float(start), float(end)))
        for index in range(stairs + 1):
            candidates.add(start + (end - start) * index / stairs)
    ordered = sorted(e for e in candidates if 0.0 <= e <= 1.0)

    # A phase is a maximal interval on which s does not change, so an edge only
    # counts if s actually differs across it. ramp_end is the case that matters:
    # the last plateau already sits at s_max, so the tail after ramp_end is the
    # same problem continued, and splitting it off would restart the learning
    # rate in the middle of one optimisation for no reason.
    kept = [ordered[0]]
    for edge in ordered[1:-1]:
        if any(abs(value(edge - 1e-9) - value(edge + 1e-9)) > 1e-12
               for *_, value in axes):
            kept.append(edge)
    kept.append(ordered[-1])
    return tuple(kept)


def phase_at(progress, bounds):
    """(phase index, progress within that phase) for a point in training."""
    clamped = min(1.0, max(0.0, progress))
    if len(bounds) < 2:
        return 0, clamped
    last = len(bounds) - 2
    for index in range(last + 1):
        low, high = bounds[index], bounds[index + 1]
        # `index == last` catches progress == 1.0, which is below no upper edge.
        if clamped < high or index == last:
            local = (clamped - low) / max(high - low, 1e-12)
            return index, min(1.0, max(0.0, local))
    return last, 1.0
