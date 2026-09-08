"""The activation homotopy: phi_alpha(x) = max(x, alpha*x), alpha driven 1 -> 0.

Two things live here:

  ActivationGate    the mechanism -- which sites hold which alpha
  site_groups       the scope     -- how sites are bundled into schedulable units

The policy is not duplicated. Writing tau = 1 - alpha turns a quantity that
falls 1 -> 0 into one that rises 0 -> 1, which is the shape homotopy.s_at
already implements and tests; homotopy.alpha_at_cfg does the change of
coordinate and nothing else.

There are no hooks here, unlike ResidualGate. `Activation.forward` reads
`self.alpha` on every call, so writing the attribute *is* the mechanism. That
also means a gate is not needed for a baseline run: with no gate the alphas
stay at 0.0 and the model is the ReLU ResNet it always was.

Two facts about this homotopy, and they are not the same as s's:

  * alpha cannot be reparametrised away. phi_alpha is positively homogeneous --
    phi_alpha(c*x) = c*phi_alpha(x) for c > 0 -- so no rescaling of any weight
    or of a BN gamma changes the mix of x and |x| it applies. s could always be
    absorbed into bn2's gamma; alpha cannot be absorbed into anything.

    The escape that remains is *shift*, not scale: BatchNorm can push beta
    positive until nearly every pre-activation is positive, and phi_alpha is
    then the identity whatever alpha says. landscape.activation_stats measures
    exactly that, and a run where it stays flat while alpha falls is a baseline
    wearing a costume.

  * alpha = 1 makes the whole network affine, which is a much stronger
    degeneracy than s = 0 was. s = 0 left a real shallow ConvNet; alpha = 1
    leaves a linear classifier, whose optimum on CIFAR-10 is around 40%, and
    whose Hessian has a large null space by construction (a deep linear network
    is invariant under W1 -> W1*G, W2 -> G^-1*W2). A continuation started
    strictly at alpha = 1 starts on a singular point of the branch it is trying
    to follow. a_start below 1 is how an arm avoids that, and is why the knob
    exists.
"""
import contextlib

from cifarbase.model import Activation

SCOPES = ("global", "stage", "block", "site")
SITES = ("all", "act1", "act2")


# --------------------------------------------------------------------------
# where the activations are
# --------------------------------------------------------------------------

def activation_sites(model, which="all"):
    """Every Activation in forward order, as (module, stage, block).

    stage is 0 for the stem and 1..4 for layer1..layer4; block is a running
    index over BasicBlocks, -1 for the stem. Those two are what `scope` groups
    by, so they are carried here rather than re-derived from module names.

    `which` selects act1 (inside the residual branch), act2 (on the main path,
    after the add) or both. It matters because they linearise different things:
    act1 alone makes F affine and leaves the block nonlinear, act2 alone leaves
    F nonlinear, and only "all" makes the network itself affine at alpha = 1.

    Walked explicitly rather than through modules() so that forward order is
    constructed rather than assumed -- test_sites_are_every_activation_in_
    forward_order checks the result against hooks either way.
    """
    if which not in SITES:
        raise ValueError(f"unknown a_sites {which!r}: pick one of {', '.join(SITES)}")
    sites = []
    if which == "all":
        sites.append((model.act0, 0, -1))
    block_index = 0
    stages = (model.layer1, model.layer2, model.layer3, model.layer4)
    for stage_index, stage in enumerate(stages, start=1):
        for block in stage:
            if which in ("all", "act1"):
                sites.append((block.act1, stage_index, block_index))
            if which in ("all", "act2"):
                sites.append((block.act2, stage_index, block_index))
            block_index += 1
    return sites


def _dense_rank(keys):
    """Distinct keys -> 0, 1, 2, ... in sorted order.

    Dense rather than "use the key as the index" because a subset of the sites
    may not start at zero: with which="act1" there is no stem, so the stages
    present are 1..4 and a raw stage index would leave group 0 empty. The
    schedule vector has one entry per group and would then never reach it.
    """
    order = {key: index for index, key in enumerate(sorted(set(keys)))}
    return [order[key] for key in keys]


def site_groups(sites, scope="global", reverse=False):
    """One group index per site: which sites move together, and in what order.

        global   every site shares one alpha
        stage    the stem and the four stages become nonlinear as units
        block    one group per BasicBlock, the stem its own
        site     every activation is independent

    `reverse` flips the ordering, turning a bottom-up continuation (input
    first) into a top-down one (head first). It only changes anything under a
    schedule that staggers groups -- `sequential` -- since every other schedule
    hands all groups the same value.
    """
    if scope not in SCOPES:
        raise ValueError(f"unknown a_scope {scope!r}: pick one of {', '.join(SCOPES)}")
    if scope == "global":
        groups = [0] * len(sites)
    elif scope == "stage":
        groups = _dense_rank([stage for _, stage, _ in sites])
    elif scope == "block":
        groups = _dense_rank([block for _, _, block in sites])
    else:
        groups = list(range(len(sites)))
    if reverse:
        top = max(groups)
        groups = [top - g for g in groups]
    return groups


# --------------------------------------------------------------------------
# the mechanism
# --------------------------------------------------------------------------

class ActivationGate:
    """Holds one alpha per group and writes it through to the sites.

    Attach once and call `set` as often as you like: there is no per-step cost
    beyond assigning a float to each site. Use it as a context manager when the
    gate should not outlive the measurement.
    """

    def __init__(self, model, which="all", scope="global", reverse=False):
        self.sites = activation_sites(model, which)
        if not self.sites:
            raise ValueError(f"no activation site matches a_sites={which!r}")
        self.groups = site_groups(self.sites, scope, reverse)
        self.n_groups = max(self.groups) + 1
        self.values = [0.0] * self.n_groups

    def __len__(self):
        return self.n_groups

    def set(self, alpha):
        """alpha is a scalar applied to every group, or one value per group."""
        if isinstance(alpha, (int, float)):
            values = [float(alpha)] * self.n_groups
        else:
            values = [float(v) for v in alpha]
            if len(values) != self.n_groups:
                raise ValueError(f"expected {self.n_groups} values for "
                                 f"{self.n_groups} groups, got {len(values)}")
        self.values[:] = values
        for (site, _, _), group in zip(self.sites, self.groups, strict=True):
            site.alpha = values[group]
        return list(values)

    def get(self):
        return list(self.values)

    def site_alphas(self):
        """The per-site vector, for a diagnostic that reports one row per site."""
        return [site.alpha for site, _, _ in self.sites]

    @contextlib.contextmanager
    def at(self, alpha):
        """Temporarily hold alpha, then restore it.

        This is how the alpha=0 readout gets measured: the model as it
        currently runs, and the ReLU network the same weights define. Without
        the restore, an evaluation would silently change the next training step.
        """
        previous = self.get()
        self.set(alpha)
        try:
            yield self
        finally:
            self.set(previous)

    def close(self):
        """Every site back to alpha = 0.0 -- exactly ReLU, not nearly ReLU.

        The model is then byte-for-byte the baseline again, which is the only
        reason an accuracy measured after this is comparable to a baseline
        number.
        """
        self.set(0.0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def build_activation_gate(model, cfg):
    """The gate this config asks for, or None when the homotopy is off.

    None rather than a gate parked at alpha=0: with no gate nothing writes to a
    module attribute at all, and a baseline run is exactly the run it was
    before this file existed.
    """
    if cfg["a_schedule"] == "none":
        return None
    return ActivationGate(model, which=cfg["a_sites"], scope=cfg["a_scope"],
                          reverse=cfg["a_reverse"])
