"""The anchor: w0, the parameter partition, and the decoupled pull toward it.

The penalty is per group and normalised by the anchor norm,

    F(w) = L(w) + sum_g (lambda_g / 2) * ||w_g - w0_g||^2 / ||w0_g||^2

and it is NEVER added to the loss. It is applied straight to the update, after
the optimiser has already stepped:

    w_g <- w_g - eta_k * grad  -  eta_k * lambda_g * (w_g - w0_g) / ||w0_g||^2

Two reasons it has to be decoupled rather than folded into the loss. With an
adaptive optimiser, a penalty routed through the loss also goes through the
preconditioner, so the realised pull becomes lambda / sqrt(v_hat) -- a different,
unknown, per-coordinate penalty than the one written down. And with momentum the
penalty would enter the velocity buffer and keep acting for several steps after
lambda changed, which makes the controller's transfer function untraceable.

Scaling by the SCHEDULED eta_k, not the base lr, matters just as much: the pull is
an exponential relaxation with timescale 1/(eta*lambda), so holding lambda fixed
while eta decays by 100x over a cosine schedule silently weakens the anchor by
100x. Anything measured against lambda would then be measuring the lr schedule.

The 1/||w0_g||^2 is what makes lambda_g dimensionless and therefore comparable
ACROSS DEPTH. Without it the same numeric lambda is a hundred times stronger on
the stem than on stage 4, purely because the two groups hold different numbers of
parameters at different scales -- and the allocation the slow loop learns would be
mostly a readout of that, not of anything about the layers.

Groups. L = 7. The partition is about which parameters the anchor is allowed to
touch; it deliberately does NOT change weight decay, which stays uniform at
cfg["weight_decay"] across every group so the unanchored baseline is bit-for-bit
the recipe it always was.

    stem              conv1.weight. The 3x3 CIFAR stem.
    stage1..stage4    every conv weight in layer1..layer4, shortcut projections
                      included. The function class knob -- what the anchor is for.
    head              fc.weight. Own lambda: a linear readout on top of frozen
                      features is the lambda -> infinity endpoint, so the head is
                      exactly the part that should be allowed to move when
                      everything else cannot.
    bn_affine         BatchNorm gamma and beta, all of them, as ONE group. They
                      carry a tiny fraction of the norm while setting the
                      effective learning rate of everything downstream, so
                      folding them into the stages would let them dominate those
                      groups' ||w - w0|| per unit of actual function change.
    bias              every bias. NOT anchored. Negligible norm, and including
                      them only adds noise to ||w - w0||.

Deliberately not per-tensor: a diagonal controller on 62 groups is a diagonal
controller on a plant whose off-diagonal coupling is nowhere near small, and the
coupling gate in selfcheck.py is what says so quantitatively.

BatchNorm's running_mean and running_var are buffers, not parameters. They are
never anchored and never touched here; anchoring a running statistic is a bug,
not a variant.
"""
import math

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

# The container module holding w0 is registered under this name, so w0 rides
# along in model.state_dict() and is checkpointed and device-moved for free. The
# prefix is also what train.py filters on to keep w0 out of the best-val snapshot.
CONTAINER = "_anchor"

STAGES = ("stage1", "stage2", "stage3", "stage4")
FINE = ("stem",) + STAGES + ("head", "bn_affine")

# How the fine partition is merged before the allocation sees it. The coupling
# gate is what chooses between these: a diagonal controller needs a plant whose
# off-diagonal coupling is small, and merging two groups that move together
# removes a coupling rather than hiding it.
#
# Ordered coarse-to-fine so a failure has somewhere to retreat to. `stage` is the
# finest the gate is even asked about -- per tensor is not on the menu, because 62
# groups of a plant this coupled is not a control problem, it is a random walk.
COARSENINGS = {
    "stage": {g: g for g in FINE},
    # Merges the four stages into two by resolution: 32x32/16x16 against 8x8/4x4.
    "coarse": {"stem": "early", "stage1": "early", "stage2": "early",
               "stage3": "late", "stage4": "late",
               "head": "head", "bn_affine": "bn_affine"},
    # The whole convolutional trunk on one lambda. Three groups is the coarsest
    # partition that still says anything the four-role scheme did not.
    "trunk": {"stem": "trunk", "stage1": "trunk", "stage2": "trunk",
              "stage3": "trunk", "stage4": "trunk",
              "head": "head", "bn_affine": "bn_affine"},
}

_ORDER = {"stage": FINE,
          "coarse": ("early", "late", "head", "bn_affine"),
          "trunk": ("trunk", "head", "bn_affine")}


def anchored_groups(grouping="stage"):
    """The anchored group names under `grouping`, in a fixed order."""
    if grouping not in COARSENINGS:
        raise SystemExit(f"!! unknown anchor_grouping {grouping!r}: "
                         f"pick from {', '.join(COARSENINGS)}")
    return _ORDER[grouping]


def all_groups(grouping="stage"):
    return anchored_groups(grouping) + ("bias",)


# The default partition, and what the module-level constants still mean for code
# that has no config to hand (the gates, mostly).
ANCHORED_GROUPS = anchored_groups("stage")
GROUPS = all_groups("stage")

# Floor on ||w0_g||^2, so a group that somehow initialises at zero cannot turn the
# pull into a division by nothing. Any real group is many orders above this.
_W0_SQ_FLOOR = 1e-12


def _mangle(name):
    """A parameter name as a legal buffer name: register_buffer rejects dots."""
    return name.replace(".", "|")


class _W0(nn.Module):
    """Buffer bag. A module rather than a dict so state_dict() carries w0."""


class Anchor:
    """w0, the group partition, and the pull. One per training run.

    Built for EVERY arm, including the unanchored baseline: ||w - w0||, the
    per-layer weight norms and eta_eff are logged on every arm, because without
    them a difference between arms cannot be told apart from a difference in
    effective learning rate.
    """

    def __init__(self, model, cfg):
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[cfg["anchor_w0_dtype"]]
        self.model = model
        self.normalize = bool(cfg["anchor_normalize"])
        self.grouping = cfg.get("anchor_grouping", "stage")
        self.anchored = anchored_groups(self.grouping)
        self.all = all_groups(self.grouping)
        # Static multipliers, kept OUTSIDE the a_g mechanism so they cannot fight
        # the sum a_g = 0 constraint. Both default to 1.0, i.e. inert.
        self.ratios = {g: 1.0 for g in self.anchored}
        if "bn_affine" in self.ratios:
            self.ratios["bn_affine"] = float(cfg["anchor_ratio_bn"])
        if "head" in self.ratios:
            self.ratios["head"] = float(cfg["anchor_ratio_head"])
        self.groups = {g: {"names": [], "params": [], "w0": []} for g in self.all}

        bn_ids = _bn_param_ids(model)

        bag = _W0()
        for name, param in model.named_parameters():
            if not param.requires_grad or name.startswith(CONTAINER + "."):
                continue
            group = _group_of(name, param, bn_ids, self.grouping)
            entry = self.groups[group]
            entry["names"].append(name)
            entry["params"].append(param)
            if group == "bias":
                entry["w0"].append(None)
                continue
            w0 = param.detach().clone().to(dtype)
            bag.register_buffer(_mangle(name), w0)
            entry["w0"].append(w0)

        model.add_module(CONTAINER, bag)
        self._assert_partition(model)

        # ||w0_g||^2, the normaliser. Captured once, at step 0, from w0 and never
        # from the live weights: a divisor that moved with training would make
        # lambda_g mean a different thing at every step and the controller would
        # be integrating against a shifting definition.
        self.w0_sq_true = {g: max(sum(float(w0.float().pow(2).sum())
                                      for w0 in self.groups[g]["w0"]), _W0_SQ_FLOOR)
                           for g in self.anchored}
        # The divisor actually applied. Gate 5 needs the un-normalised penalty,
        # which is exactly this dict held at one.
        self.w0_sq = (dict(self.w0_sq_true) if self.normalize
                      else {g: 1.0 for g in self.anchored})
        self.min_w0_sq = min(self.w0_sq.values())

        # The layers eta_eff is reported for: the >1-d weights, whose norm sets
        # the effective lr of everything behind their BatchNorm.
        self.layer_names = [n for n, p in model.named_parameters()
                            if p.requires_grad and p.dim() > 1
                            and not n.startswith(CONTAINER + ".")]
        self.layer_params = [dict(model.named_parameters())[n]
                             for n in self.layer_names]
        self._w0_norm = self._norm([w0 for g in self.anchored
                                    for w0 in self.groups[g]["w0"]])

    # -- the pull ----------------------------------------------------------

    @torch.no_grad()
    def pull(self, lambdas, lr_now):
        """w_g -= lr_now * lambda_g * (w_g - w0_g) / ||w0_g||^2, after the step.

        The factor lr_now*lambda_g/||w0_g||^2 is the per-step fraction of the
        distance to w0 that is closed. At exactly 1.0 the weights land on w0 and
        the anchor has stopped being a penalty and become an assignment; above 1.0
        it overshoots and oscillates. Either is a silent disaster -- the loss curve
        looks plausible -- so it is an assertion rather than a clamp, and it is
        checked per group because the groups no longer share one factor.
        """
        for group in self.anchored:
            lam = lambdas.get(group, 0.0)
            if lam <= 0.0:
                continue
            factor = lr_now * lam / self.w0_sq[group]
            if not factor < 1.0:
                raise SystemExit(
                    f"!! anchor overshoot: lr*lambda/||w0||^2 = {lr_now:.6g} * "
                    f"{lam:.6g} / {self.w0_sq[group]:.6g} = {factor:.6g} >= 1 on "
                    f"group {group!r}. The pull would land on or past w0. Lower "
                    f"anchor_lambda_max_frac, or the lr.")
            entry = self.groups[group]
            for param, w0 in zip(entry["params"], entry["w0"], strict=True):
                # add_ promotes a bf16 w0 up to the parameter's dtype; the
                # reverse would not be allowed, which is why w0 is never fp32
                # while the parameters are something narrower.
                param.mul_(1.0 - factor).add_(w0, alpha=factor)

    def lambdas(self, level, alloc=None):
        """log lambda_g = u + a_g, then the static ratio. The controller output.

        `level` is the fast loop's u; `alloc` the slow loop's a_g (None means the
        allocation is off, i.e. a_g = 0 and every group shares one lambda).
        """
        out = {}
        for group in self.anchored:
            offset = 0.0 if alloc is None else float(alloc.get(group, 0.0))
            out[group] = math.exp(level + offset) * self.ratios[group]
        return out

    # -- what the slow loop reads ------------------------------------------

    @torch.no_grad()
    def grad_norms(self):
        """||grad_g L|| per anchored group, from the live .grad tensors.

        Read after backward and after grad clip, before optimizer.step() -- the
        same gradient the step is about to apply, which is what the tension is
        defined against. A group whose grads are all None (it happens under
        set_to_none before the first backward) reports 0.0 rather than raising.
        """
        out = {}
        for group in self.anchored:
            total = 0.0
            for param in self.groups[group]["params"]:
                if param.grad is not None:
                    total += float(param.grad.float().pow(2).sum())
            out[group] = total ** 0.5
        return out

    @torch.no_grad()
    def delta_norms(self):
        """||w_g - w0_g|| per anchored group. The other half of the tension."""
        return {group: sum(float((param - w0).float().pow(2).sum())
                           for param, w0 in zip(self.groups[group]["params"],
                                                self.groups[group]["w0"],
                                                strict=True)) ** 0.5
                for group in self.anchored}

    # -- what gets logged --------------------------------------------------

    @torch.no_grad()
    def distance(self, lr_now):
        """||w - w0||, per group and overall, plus the per-layer eta_eff trace.

        eta_eff,l = eta / ||w_l||^2 is the quantity that actually sets how fast a
        scale-invariant layer rotates. Every arm ends at a different ||w||, so
        without this trace a win cannot be distinguished from an accidentally
        better learning-rate schedule -- and that confound is unrecoverable after
        the fact, which is why it is logged even on the arms with no anchor.
        """
        out = {}
        total = 0.0
        for group in self.anchored:
            entry = self.groups[group]
            # Accumulated tensor by tensor rather than as a list of differences:
            # the differences are another full copy of the parameters, and this
            # runs inside the probe on a card already holding the dataset, w and
            # w0.
            group_total = 0.0
            for param, w0 in zip(entry["params"], entry["w0"], strict=True):
                group_total += float((param - w0).float().pow(2).sum())
            out[f"dist_{group}"] = group_total ** 0.5
            # Relative to the group's own anchor norm: the only per-group
            # displacement number that is comparable across depth.
            out[f"dist_rel_{group}"] = (group_total / self.w0_sq_true[group]) ** 0.5
            total += group_total
        out["dist"] = total ** 0.5
        out["dist_rel"] = out["dist"] / max(self._w0_norm, 1e-12)

        norms = [float(p.norm()) for p in self.layer_params]
        eta_eff = [lr_now / max(n * n, 1e-24) for n in norms]
        out["w_norms"] = norms
        out["eta_eff"] = eta_eff
        out["w_norm_total"] = self._norm(self.layer_params)
        out["eta_eff_min"] = min(eta_eff)
        out["eta_eff_max"] = max(eta_eff)
        out["eta_eff_mean"] = sum(eta_eff) / len(eta_eff)
        return out

    def describe(self):
        counts = {g: sum(p.numel() for p in self.groups[g]["params"])
                  for g in self.all}
        parts = []
        for group in self.all:
            if group == "bias":
                parts.append(f"{group} {counts[group]:,} (free)")
                continue
            ratio = ("" if self.ratios[group] == 1.0
                     else f" x{self.ratios[group]:g}")
            parts.append(f"{group} {counts[group]:,}"
                         f" |w0|^2={self.w0_sq_true[group]:.3g}{ratio}")
        norm = "on" if self.normalize else "OFF"
        return (f"anchor groups (normalise {norm}): " + ", ".join(parts)
                + f"  (||w0|| = {self._w0_norm:.3f})")

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _norm(tensors):
        total = sum(float(t.float().pow(2).sum()) for t in tensors)
        return total ** 0.5

    def _assert_partition(self, model):
        """Every trainable parameter in exactly one group, and every anchored one
        paired with a w0 of the same shape.

        A parameter silently missing from the partition would just never be
        anchored, and a parameter counted twice would be pulled twice per step.
        Neither shows up anywhere except in the final accuracy.
        """
        seen = [id(p) for g in self.all for p in self.groups[g]["params"]]
        want = {id(p) for n, p in model.named_parameters()
                if p.requires_grad and not n.startswith(CONTAINER + ".")}
        if len(seen) != len(set(seen)):
            raise ValueError("a parameter landed in more than one anchor group")
        if set(seen) != want:
            raise ValueError(f"anchor groups cover {len(set(seen))} of {len(want)} "
                             f"trainable parameters")
        empty = [g for g in self.anchored if not self.groups[g]["params"]]
        if empty:
            raise ValueError(
                f"anchor groups {', '.join(empty)} are empty. The allocation "
                f"takes a mean over groups and a missing one silently changes "
                f"what uniform means; fix the partition rather than the mean.")
        self.assert_shapes()

    def assert_shapes(self):
        """w0 still lines up with w. Re-run after a resume: a checkpoint from a
        different width or arch would otherwise pull garbage into the weights."""
        for group in self.anchored:
            entry = self.groups[group]
            for name, param, w0 in zip(entry["names"], entry["params"],
                                       entry["w0"], strict=True):
                if param.shape != w0.shape:
                    raise ValueError(f"anchor shape mismatch on {name}: "
                                     f"w is {tuple(param.shape)}, w0 is "
                                     f"{tuple(w0.shape)}")


def _bn_param_ids(model):
    ids = set()
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            ids.update(id(p) for p in (module.weight, module.bias) if p is not None)
    return ids


def _group_of(name, param, bn_ids, grouping="stage"):
    # BatchNorm FIRST. Its beta is named "...bn1.bias", so testing the bias suffix
    # ahead of BatchNorm membership silently drops every beta out of the anchor --
    # which leaves the parameters that set each block's output scale free to drift
    # while ||w - w0|| reports almost nothing, because it is not measuring them.
    # Gate H1 exists to catch exactly this and did.
    merge = COARSENINGS[grouping]
    if id(param) in bn_ids:
        return merge["bn_affine"]
    if name.endswith("bias"):
        return "bias"
    if name.startswith("fc."):
        return merge["head"]
    if name.startswith("conv1."):
        return merge["stem"]
    for index, stage in enumerate(STAGES, start=1):
        if name.startswith(f"layer{index}."):
            return merge[f"stage{index}"]
    # Not a fallback bucket: an unrecognised parameter must stop the run. Silently
    # sweeping it into a stage would put it on somebody else's lambda.
    raise ValueError(
        f"parameter {name!r} matches no anchor group. The partition is written "
        f"for the ResNet in model.py (conv1 / layer1..4 / fc / BatchNorm); a new "
        f"module needs a deliberate group, not a default one.")


def group_of_names(model, grouping="stage"):
    """{parameter name: group}, for the per-group kernel sketch.

    Built from the same _group_of the anchor uses, so the drift decomposition and
    the pull can never disagree about what a group is.
    """
    bn_ids = _bn_param_ids(model)
    return {name: _group_of(name, param, bn_ids, grouping)
            for name, param in model.named_parameters()
            if param.requires_grad and not name.startswith(CONTAINER + ".")}


def build_param_groups(model, cfg):
    """One optimiser group per anchor group, all at the same weight_decay.

    The split exists so the anchor can give each group its own lambda. It must not
    change decay: putting BatchNorm and biases in a no-decay group is a popular
    and defensible recipe change, but it IS a recipe change, and it would move
    the baseline number this branch is measured against.
    """
    decay = float(cfg["weight_decay"])
    grouping = cfg.get("anchor_grouping", "stage")
    bn_ids = _bn_param_ids(model)

    buckets = {g: [] for g in all_groups(grouping)}
    for name, param in model.named_parameters():
        if not param.requires_grad or name.startswith(CONTAINER + "."):
            continue
        buckets[_group_of(name, param, bn_ids, grouping)].append(param)
    return [{"params": buckets[g], "weight_decay": decay, "group": g}
            for g in all_groups(grouping) if buckets[g]]


def strip_w0(state_dict):
    """A state_dict without the w0 buffers.

    train.py snapshots the best-val weights on the CPU every time validation
    improves. w0 is another full copy of the parameters, so leaving it in would
    double that snapshot for nothing -- w0 is a constant and is already in the
    checkpoint.
    """
    return {k: v for k, v in state_dict.items()
            if not k.startswith(CONTAINER + ".")}
