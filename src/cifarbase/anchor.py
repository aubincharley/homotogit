"""The anchor: w0, the parameter partition, and the decoupled pull toward it.

The penalty is lambda(t)/2 * ||w - w0||^2, and it is NEVER added to the loss. It
is applied straight to the update, after the optimiser has already stepped:

    w <- w - eta_k * grad  -  eta_k * lambda_k * (w - w0)

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

Roles. The partition is about which parameters the anchor is allowed to touch;
it deliberately does NOT change weight decay, which stays uniform at
cfg["weight_decay"] across every group so the unanchored baseline is bit-for-bit
the recipe it always was.

    weights   conv weights, and every linear weight but the head. The function
              class knob -- the thing the anchor is actually for.
    bn        BatchNorm gamma and beta. Anchored, but on their own lambda: they
              carry a tiny fraction of the norm while setting the effective
              learning rate of everything downstream, so one shared lambda would
              be dominated by them per unit of ||w - w0||.
    head      fc.weight. Own lambda: a linear readout on top of frozen features
              is the lambda -> infinity endpoint, so the head is exactly the part
              that should be allowed to move when everything else cannot.
    bias      every bias. NOT anchored. Negligible norm, and including them only
              adds noise to ||w - w0||.

BatchNorm's running_mean and running_var are buffers, not parameters. They are
never anchored and never touched here; anchoring a running statistic is a bug,
not a variant.
"""
import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

# The container module holding w0 is registered under this name, so w0 rides
# along in model.state_dict() and is checkpointed and device-moved for free. The
# prefix is also what train.py filters on to keep w0 out of the best-val snapshot.
CONTAINER = "_anchor"

ANCHORED_ROLES = ("weights", "bn", "head")
ROLES = ANCHORED_ROLES + ("bias",)


def _mangle(name):
    """A parameter name as a legal buffer name: register_buffer rejects dots."""
    return name.replace(".", "|")


class _W0(nn.Module):
    """Buffer bag. A module rather than a dict so state_dict() carries w0."""


class Anchor:
    """w0, the role partition, and the pull. One per training run.

    Built for EVERY arm, including the unanchored baseline: ||w - w0||, the
    per-layer weight norms and eta_eff are logged on every arm, because without
    them a difference between arms cannot be told apart from a difference in
    effective learning rate.
    """

    def __init__(self, model, cfg):
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[cfg["anchor_w0_dtype"]]
        self.model = model
        self.ratios = {"weights": 1.0,
                       "bn": float(cfg["anchor_ratio_bn"]),
                       "head": float(cfg["anchor_ratio_head"]),
                       "bias": 0.0}
        self.roles = {role: {"names": [], "params": [], "w0": []} for role in ROLES}

        bn_ids = set()
        for module in model.modules():
            if isinstance(module, _BatchNorm):
                bn_ids.update(id(p) for p in (module.weight, module.bias)
                              if p is not None)

        bag = _W0()
        for name, param in model.named_parameters():
            if not param.requires_grad or name.startswith(CONTAINER + "."):
                continue
            role = _role_of(name, param, bn_ids)
            entry = self.roles[role]
            entry["names"].append(name)
            entry["params"].append(param)
            if role == "bias":
                entry["w0"].append(None)
                continue
            w0 = param.detach().clone().to(dtype)
            bag.register_buffer(_mangle(name), w0)
            entry["w0"].append(w0)

        model.add_module(CONTAINER, bag)
        self._assert_partition(model)

        # The layers eta_eff is reported for: the >1-d weights, whose norm sets
        # the effective lr of everything behind their BatchNorm.
        self.layer_names = [n for n, p in model.named_parameters()
                            if p.requires_grad and p.dim() > 1
                            and not n.startswith(CONTAINER + ".")]
        self.layer_params = [dict(model.named_parameters())[n]
                             for n in self.layer_names]
        self._w0_norm = self._norm([w0 for role in ANCHORED_ROLES
                                   for w0 in self.roles[role]["w0"]])

    # -- the pull ----------------------------------------------------------

    @torch.no_grad()
    def pull(self, lambdas, lr_now):
        """w -= lr_now * lambda_role * (w - w0), in place, after optimizer.step().

        The factor lr_now*lambda is the per-step fraction of the distance to w0
        that is closed. At exactly 1.0 the weights land on w0 and the anchor has
        stopped being a penalty and become an assignment; above 1.0 it overshoots
        and oscillates. Either is a silent disaster -- the loss curve looks
        plausible -- so it is an assertion rather than a clamp.
        """
        for role in ANCHORED_ROLES:
            lam = lambdas.get(role, 0.0)
            if lam <= 0.0:
                continue
            factor = lr_now * lam
            if not factor < 1.0:
                raise SystemExit(
                    f"!! anchor overshoot: lr*lambda = {lr_now:.6g} * {lam:.6g} "
                    f"= {factor:.6g} >= 1 on role {role!r}. The pull would land "
                    f"on or past w0. Lower anchor_lambda_max_frac, or the lr.")
            entry = self.roles[role]
            for param, w0 in zip(entry["params"], entry["w0"], strict=True):
                # add_ promotes a bf16 w0 up to the parameter's dtype; the
                # reverse would not be allowed, which is why w0 is never fp32
                # while the parameters are something narrower.
                param.mul_(1.0 - factor).add_(w0, alpha=factor)

    def lambdas(self, lam):
        """The scalar controller output fanned out over the roles."""
        return {role: lam * self.ratios[role] for role in ANCHORED_ROLES}

    # -- what gets logged --------------------------------------------------

    @torch.no_grad()
    def distance(self, lr_now):
        """||w - w0||, per role and overall, plus the per-layer eta_eff trace.

        eta_eff,l = eta / ||w_l||^2 is the quantity that actually sets how fast a
        scale-invariant layer rotates. Every arm ends at a different ||w||, so
        without this trace a win cannot be distinguished from an accidentally
        better learning-rate schedule -- and that confound is unrecoverable after
        the fact, which is why it is logged even on the arms with no anchor.
        """
        out = {}
        total = 0.0
        for role in ANCHORED_ROLES:
            entry = self.roles[role]
            # Accumulated tensor by tensor rather than as a list of differences:
            # the differences are another full copy of the parameters, and this
            # runs inside the probe on a card already holding the dataset, w and
            # w0.
            role_total = 0.0
            for param, w0 in zip(entry["params"], entry["w0"], strict=True):
                role_total += float((param - w0).float().pow(2).sum())
            out[f"dist_{role}"] = role_total ** 0.5
            total += role_total
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
        counts = {role: sum(p.numel() for p in self.roles[role]["params"])
                  for role in ROLES}
        return ("anchor roles: "
                + ", ".join(f"{role} {counts[role]:,}"
                            + ("" if role == "bias" else f" x{self.ratios[role]:g}")
                            for role in ROLES)
                + f"  (||w0|| = {self._w0_norm:.3f})")

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _norm(tensors):
        total = sum(float(t.float().pow(2).sum()) for t in tensors)
        return total ** 0.5

    def _assert_partition(self, model):
        """Every trainable parameter in exactly one role, and every anchored one
        paired with a w0 of the same shape.

        A parameter silently missing from the partition would just never be
        anchored, and a parameter counted twice would be pulled twice per step.
        Neither shows up anywhere except in the final accuracy.
        """
        seen = [id(p) for role in ROLES for p in self.roles[role]["params"]]
        want = {id(p) for n, p in model.named_parameters()
                if p.requires_grad and not n.startswith(CONTAINER + ".")}
        if len(seen) != len(set(seen)):
            raise ValueError("a parameter landed in more than one anchor role")
        if set(seen) != want:
            raise ValueError(f"anchor roles cover {len(set(seen))} of {len(want)} "
                             f"trainable parameters")
        self.assert_shapes()

    def assert_shapes(self):
        """w0 still lines up with w. Re-run after a resume: a checkpoint from a
        different width or arch would otherwise pull garbage into the weights."""
        for role in ANCHORED_ROLES:
            entry = self.roles[role]
            for name, param, w0 in zip(entry["names"], entry["params"],
                                       entry["w0"], strict=True):
                if param.shape != w0.shape:
                    raise ValueError(f"anchor shape mismatch on {name}: "
                                     f"w is {tuple(param.shape)}, w0 is "
                                     f"{tuple(w0.shape)}")


def _role_of(name, param, bn_ids):
    # BatchNorm FIRST. Its beta is named "...bn1.bias", so testing the bias suffix
    # ahead of BatchNorm membership silently drops every beta out of the anchor --
    # which leaves the parameters that set each block's output scale free to drift
    # while ||w - w0|| reports almost nothing, because it is not measuring them.
    # Gate H1 exists to catch exactly this and did.
    if id(param) in bn_ids:
        return "bn"
    if name.endswith("bias"):
        return "bias"
    if name.startswith("fc."):
        return "head"
    return "weights"


def build_param_groups(model, cfg):
    """One group per role, all at the same weight_decay.

    The split exists so the anchor can give each role its own lambda. It must not
    change decay: putting BatchNorm and biases in a no-decay group is a popular
    and defensible recipe change, but it IS a recipe change, and it would move
    the baseline number this branch is measured against.
    """
    decay = float(cfg["weight_decay"])
    bn_ids = set()
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            bn_ids.update(id(p) for p in (module.weight, module.bias)
                          if p is not None)

    buckets = {role: [] for role in ROLES}
    for name, param in model.named_parameters():
        if not param.requires_grad or name.startswith(CONTAINER + "."):
            continue
        buckets[_role_of(name, param, bn_ids)].append(param)
    return [{"params": buckets[role], "weight_decay": decay, "role": role}
            for role in ROLES if buckets[role]]


def strip_w0(state_dict):
    """A state_dict without the w0 buffers.

    train.py snapshots the best-val weights on the CPU every time validation
    improves. w0 is another full copy of the parameters, so leaving it in would
    double that snapshot for nothing -- w0 is a constant and is already in the
    checkpoint.
    """
    return {k: v for k, v in state_dict.items()
            if not k.startswith(CONTAINER + ".")}
